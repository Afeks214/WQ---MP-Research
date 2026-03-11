from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    raise RuntimeError("pandas is required") from exc

from multiprocessing.shared_memory import SharedMemory

from weightiz.shared.io.hpc_market_profile_parity import compute_market_profile_features
from weightiz.module1.core import EngineConfig, build_session_clock_vectorized


@dataclass(frozen=True)
class SharedArrayMeta:
    name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass
class SharedRegistry:
    arrays: dict[str, SharedArrayMeta]


@dataclass
class MarketDataSharedHandle:
    symbols: tuple[str, ...]
    ts_ns: np.ndarray
    timezone: str
    dataset_hash: str
    registry: SharedRegistry
    local_arrays: dict[str, np.ndarray]
    dq_report: list[dict[str, Any]]


class _SharedBufferOwner:
    def __init__(self, shm: SharedMemory, arr: np.ndarray):
        self.shm = shm
        self.arr = arr


_OWNED: dict[str, _SharedBufferOwner] = {}


def _require_cols(df: pd.DataFrame, symbol: str) -> tuple[str, str, str, str, str, str]:
    cols = {str(c).strip().lower(): str(c) for c in df.columns}
    req = {
        "timestamp": ("timestamp", "ts", "datetime", "time", "date"),
        "open": ("open", "o"),
        "high": ("high", "h"),
        "low": ("low", "l"),
        "close": ("close", "c"),
        "volume": ("volume", "vol", "v"),
    }
    resolved: dict[str, str] = {}
    for key, options in req.items():
        found = None
        for o in options:
            if o in cols:
                found = cols[o]
                break
        if found is None:
            raise RuntimeError(f"Missing required column {key!r} for symbol={symbol}")
        resolved[key] = found
    return (
        resolved["timestamp"],
        resolved["open"],
        resolved["high"],
        resolved["low"],
        resolved["close"],
        resolved["volume"],
    )


def _load_symbol_frame(path: Path, symbol: str, start: Any, end: Any) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        raw = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        raw = pd.read_csv(path)
    else:
        raise RuntimeError(f"Unsupported data format for {path}")

    cols = {str(c).strip().lower(): str(c) for c in raw.columns}
    o_c = cols.get("open", cols.get("o"))
    h_c = cols.get("high", cols.get("h"))
    l_c = cols.get("low", cols.get("l"))
    c_c = cols.get("close", cols.get("c"))
    v_c = cols.get("volume", cols.get("vol", cols.get("v")))
    if o_c is None or h_c is None or l_c is None or c_c is None or v_c is None:
        raise RuntimeError(f"Missing OHLCV columns for symbol={symbol}")

    ts_c = None
    for cand in ("timestamp", "ts", "datetime", "time", "date"):
        if cand in cols:
            ts_c = cols[cand]
            break

    if ts_c is not None:
        ts = pd.to_datetime(raw[ts_c], utc=True, errors="coerce")
    elif isinstance(raw.index, pd.DatetimeIndex):
        ts = pd.to_datetime(raw.index, utc=True, errors="coerce")
    else:
        raise RuntimeError(f"Missing required column 'timestamp' for symbol={symbol}")
    keep = np.asarray(pd.notna(ts), dtype=bool)
    if start is not None:
        start_ts = pd.to_datetime(start, utc=True)
        keep &= np.asarray(ts >= start_ts, dtype=bool)
    if end is not None:
        end_ts = pd.to_datetime(end, utc=True)
        keep &= np.asarray(ts <= end_ts, dtype=bool)

    if not bool(np.any(keep)):
        raise RuntimeError(f"No data rows after filtering for symbol={symbol}, path={path}")

    ts_kept = pd.DatetimeIndex(ts[keep]).floor("min")
    out = pd.DataFrame(
        {
            "timestamp": ts_kept,
            "open": pd.to_numeric(raw.loc[keep, o_c], errors="coerce"),
            "high": pd.to_numeric(raw.loc[keep, h_c], errors="coerce"),
            "low": pd.to_numeric(raw.loc[keep, l_c], errors="coerce"),
            "close": pd.to_numeric(raw.loc[keep, c_c], errors="coerce"),
            "volume": pd.to_numeric(raw.loc[keep, v_c], errors="coerce"),
        }
    )
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp", kind="mergesort")
    out = out.drop_duplicates(subset=["timestamp"], keep="last")
    out = out.set_index("timestamp")
    return out


def _resolve_symbol_paths(config: Any, project_root: Path) -> dict[str, Path]:
    root = Path(config.data.root)
    if not root.is_absolute():
        root = (project_root / root).resolve()
    fmt = str(config.data.format)

    paths: dict[str, Path] = {}
    for sym in [s.strip().upper() for s in config.symbols]:
        mapped = config.data.path_by_symbol.get(sym)
        if mapped is None:
            candidate = (root / f"{sym}.{fmt}").resolve()
        else:
            p0 = Path(mapped)
            candidate = p0 if p0.is_absolute() else (root / p0).resolve()
        if not candidate.exists():
            raise RuntimeError(f"Missing market file for symbol={sym}: {candidate}")
        paths[sym] = candidate
    return paths


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _forward_fill_close(close_col: np.ndarray, valid_col: np.ndarray) -> np.ndarray:
    out = np.full(close_col.shape[0], np.nan, dtype=np.float64)
    last = np.nan
    for i in range(close_col.shape[0]):
        if bool(valid_col[i]) and np.isfinite(close_col[i]):
            last = float(close_col[i])
            out[i] = last
        else:
            out[i] = last
    return out


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    if window <= 1:
        return arr.astype(np.float64, copy=True)
    csum = np.cumsum(np.where(np.isfinite(arr), arr, 0.0), dtype=np.float64)
    ccount = np.cumsum(np.isfinite(arr).astype(np.int64), dtype=np.int64)
    for i in range(arr.shape[0]):
        j = i - window + 1
        if j <= 0:
            s = csum[i]
            n = ccount[i]
        else:
            s = csum[i] - csum[j - 1]
            n = ccount[i] - ccount[j - 1]
        if n > 0:
            out[i] = float(s / n)
    return out


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    mean = _rolling_mean(arr, window)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    for i in range(arr.shape[0]):
        j = max(0, i - window + 1)
        chunk = arr[j : i + 1]
        chunk = chunk[np.isfinite(chunk)]
        if chunk.size >= 2:
            out[i] = float(np.std(chunk, ddof=0))
    return out


def _rolling_median(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    for i in range(arr.shape[0]):
        j = max(0, i - window + 1)
        chunk = arr[j : i + 1]
        chunk = chunk[np.isfinite(chunk)]
        if chunk.size > 0:
            out[i] = float(np.median(chunk))
    return out


def _compute_weightiz_features(
    ts_ns: np.ndarray,
    symbols: tuple[str, ...],
    open_px: np.ndarray,
    high_px: np.ndarray,
    low_px: np.ndarray,
    close_px: np.ndarray,
    volume: np.ndarray,
    bar_valid: np.ndarray,
    windows: list[int],
) -> dict[str, np.ndarray]:
    T, A = close_px.shape
    eps = 1e-12

    tr = np.zeros((T, A), dtype=np.float64)
    for a in range(A):
        prev_close = np.nan
        for t in range(T):
            h = high_px[t, a]
            l = low_px[t, a]
            c = close_px[t, a]
            if not np.isfinite(h) or not np.isfinite(l) or not np.isfinite(c):
                tr[t, a] = np.nan
                continue
            if np.isfinite(prev_close):
                tr[t, a] = float(max(h - l, abs(h - prev_close), abs(l - prev_close)))
            else:
                tr[t, a] = float(max(h - l, 0.0))
            prev_close = c

    atr = np.full((T, A), np.nan, dtype=np.float64)
    for a in range(A):
        atr[:, a] = _rolling_mean(tr[:, a], 14)

    out: dict[str, np.ndarray] = {"atr": np.where(np.isfinite(atr), atr, 0.0)}
    ts = pd.to_datetime(ts_ns, utc=True)
    valid = bar_valid.astype(bool)

    for W in sorted(set(int(w) for w in windows)):
        D = np.zeros((T, A), dtype=np.float64)
        A_aff = np.zeros((T, A), dtype=np.float64)
        delta_eff = np.zeros((T, A), dtype=np.float64)
        s_break = np.zeros((T, A), dtype=np.float64)
        s_reject = np.zeros((T, A), dtype=np.float64)
        rvol = np.ones((T, A), dtype=np.float64)
        poc = np.full((T, A), np.nan, dtype=np.float64)
        vah = np.full((T, A), np.nan, dtype=np.float64)
        val = np.full((T, A), np.nan, dtype=np.float64)

        for a in range(A):
            sym = str(symbols[a])
            v = volume[:, a]
            med_v = _rolling_median(v, max(5, min(390, W)))
            rv = v / np.maximum(med_v, eps)
            rv = np.where(np.isfinite(rv), rv, 1.0)
            rvol[:, a] = rv

            df_sym = pd.DataFrame(
                {
                    "timestamp": ts,
                    "open": open_px[:, a],
                    "high": high_px[:, a],
                    "low": low_px[:, a],
                    "close": close_px[:, a],
                    "volume": volume[:, a],
                    "symbol": sym,
                }
            )
            feat = compute_market_profile_features(df_sym, window=int(W))
            if feat.empty:
                continue

            feat_ts = pd.to_datetime(feat["timestamp"], utc=True).astype("int64").to_numpy(dtype=np.int64)
            pos = np.searchsorted(ts_ns, feat_ts)
            ok = (pos >= 0) & (pos < T) & (ts_ns[pos] == feat_ts)
            pos = pos[ok]
            if pos.size == 0:
                continue
            f = feat.iloc[np.where(ok)[0]]
            D[pos, a] = f["D"].to_numpy(dtype=np.float64)
            A_aff[pos, a] = f["A"].to_numpy(dtype=np.float64)
            delta_eff[pos, a] = f["DeltaEff"].to_numpy(dtype=np.float64)
            s_break[pos, a] = f["Sbreak"].to_numpy(dtype=np.float64)
            s_reject[pos, a] = f["Sreject"].to_numpy(dtype=np.float64)
            poc[pos, a] = f["POC"].to_numpy(dtype=np.float64)
            vah[pos, a] = f["VAH"].to_numpy(dtype=np.float64)
            val[pos, a] = f["VAL"].to_numpy(dtype=np.float64)

        for arr in (D, A_aff, delta_eff, s_break, s_reject, rvol):
            arr[~valid] = 0.0
        for arr in (poc, vah, val):
            arr[~valid] = np.nan

        out[f"D_W{W}"] = D
        out[f"A_W{W}"] = A_aff
        out[f"DELTA_EFF_W{W}"] = delta_eff
        out[f"S_BREAK_W{W}"] = s_break
        out[f"S_REJECT_W{W}"] = s_reject
        out[f"RVOL_W{W}"] = rvol
        out[f"POC_W{W}"] = poc
        out[f"VAH_W{W}"] = vah
        out[f"VAL_W{W}"] = val

    return out


def build_master_union_index(indices: list[np.ndarray]) -> np.ndarray:
    if len(indices) == 0:
        raise RuntimeError("No timestamps provided for union index")
    merged = np.unique(np.concatenate(indices, axis=0).astype(np.int64, copy=False))
    merged.sort(kind="mergesort")
    if merged.size == 0:
        raise RuntimeError("Union timeline is empty")
    return merged


def build_bar_valid_mask(open_px: np.ndarray, high_px: np.ndarray, low_px: np.ndarray, close_px: np.ndarray) -> np.ndarray:
    valid = np.isfinite(open_px) & np.isfinite(high_px) & np.isfinite(low_px) & np.isfinite(close_px)
    return valid.astype(bool, copy=False)


def write_shared_buffers(arrays: dict[str, np.ndarray]) -> SharedRegistry:
    metas: dict[str, SharedArrayMeta] = {}
    for key in sorted(arrays.keys()):
        arr = np.ascontiguousarray(arrays[key])
        shm = SharedMemory(create=True, size=int(arr.nbytes))
        view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        view[...] = arr
        view.flags.writeable = False
        _OWNED[key] = _SharedBufferOwner(shm=shm, arr=view)
        metas[key] = SharedArrayMeta(name=shm.name, shape=tuple(int(x) for x in arr.shape), dtype=str(arr.dtype))
    return SharedRegistry(arrays=metas)


def attach_shared_buffers(registry: SharedRegistry) -> tuple[dict[str, np.ndarray], dict[str, SharedMemory]]:
    arrays: dict[str, np.ndarray] = {}
    handles: dict[str, SharedMemory] = {}
    for key in sorted(registry.arrays.keys()):
        meta = registry.arrays[key]
        shm = SharedMemory(name=meta.name, create=False)
        arr = np.ndarray(meta.shape, dtype=np.dtype(meta.dtype), buffer=shm.buf)
        arr.flags.writeable = False
        arrays[key] = arr
        handles[key] = shm
    return arrays, handles


def close_attached_handles(handles: dict[str, SharedMemory]) -> None:
    for k in sorted(handles.keys()):
        try:
            handles[k].close()
        except Exception:
            pass


def cleanup_shared_buffers(registry: SharedRegistry) -> None:
    for key in sorted(registry.arrays.keys()):
        owner = _OWNED.get(key)
        if owner is None:
            continue
        try:
            owner.shm.close()
        except Exception:
            pass
        try:
            owner.shm.unlink()
        except Exception:
            pass
    _OWNED.clear()


def load_and_align_market_data_once(config: Any, project_root: Path, windows: list[int]) -> MarketDataSharedHandle:
    symbol_paths = _resolve_symbol_paths(config, project_root)
    symbols = tuple(sorted(symbol_paths.keys()))
    dataset_hash_acc = hashlib.sha256()
    for p in sorted(symbol_paths.values()):
        dataset_hash_acc.update(sha256_file(p).encode())
    dataset_hash = dataset_hash_acc.hexdigest()

    by_symbol: dict[str, pd.DataFrame] = {}
    idx_parts: list[np.ndarray] = []
    for sym in symbols:
        frame = _load_symbol_frame(
            path=symbol_paths[sym],
            symbol=sym,
            start=config.data.start,
            end=config.data.end,
        )
        by_symbol[sym] = frame
        idx_parts.append(pd.DatetimeIndex(frame.index).asi8.astype(np.int64))

    ts_ns = build_master_union_index(idx_parts)
    T = int(ts_ns.shape[0])
    A = int(len(symbols))

    open_px = np.full((T, A), np.nan, dtype=np.float64)
    high_px = np.full((T, A), np.nan, dtype=np.float64)
    low_px = np.full((T, A), np.nan, dtype=np.float64)
    close_px = np.full((T, A), np.nan, dtype=np.float64)
    volume = np.zeros((T, A), dtype=np.float64)

    dq_report: list[dict[str, Any]] = []
    for a, sym in enumerate(symbols):
        f = by_symbol[sym]
        idx = pd.DatetimeIndex(f.index).asi8.astype(np.int64)
        pos = np.searchsorted(ts_ns, idx)
        if np.any(pos >= T):
            raise RuntimeError(f"Index mapping overflow for symbol={sym}")

        open_vals = pd.to_numeric(f["open"], errors="coerce").to_numpy(dtype=np.float64)
        high_vals = pd.to_numeric(f["high"], errors="coerce").to_numpy(dtype=np.float64)
        low_vals = pd.to_numeric(f["low"], errors="coerce").to_numpy(dtype=np.float64)
        close_vals = pd.to_numeric(f["close"], errors="coerce").to_numpy(dtype=np.float64)
        vol_vals = pd.to_numeric(f["volume"], errors="coerce").to_numpy(dtype=np.float64)

        open_px[pos, a] = open_vals
        high_px[pos, a] = high_vals
        low_px[pos, a] = low_vals
        close_px[pos, a] = close_vals
        volume[pos, a] = np.where(np.isfinite(vol_vals), np.maximum(vol_vals, 0.0), 0.0)

    bar_valid = build_bar_valid_mask(open_px, high_px, low_px, close_px)

    last_valid_close = np.full((T, A), np.nan, dtype=np.float64)
    for a in range(A):
        last_valid_close[:, a] = _forward_fill_close(close_px[:, a], bar_valid[:, a])

    # invalid bars cannot trade new orders; mark prices with carry-forward for valuation only
    for a in range(A):
        missing = ~bar_valid[:, a]
        close_px[missing, a] = last_valid_close[missing, a]
        open_px[missing, a] = last_valid_close[missing, a]
        high_px[missing, a] = last_valid_close[missing, a]
        low_px[missing, a] = last_valid_close[missing, a]
        volume[missing, a] = 0.0

        dq_report.append(
            {
                "symbol": sym if (sym := symbols[a]) else "",
                "rows": int(T),
                "valid_rows": int(np.sum(bar_valid[:, a])),
                "invalid_rows": int(np.sum(~bar_valid[:, a])),
            }
        )

    tick_size = np.full(A, float(config.engine.tick_size_default), dtype=np.float64)
    eng_cfg = EngineConfig(
        T=T,
        A=A,
        tick_size=tick_size,
        warmup_minutes=int(config.engine.warmup_minutes),
        flat_time_minute=int(config.engine.flat_time_minute),
        gap_reset_minutes=float(config.engine.gap_reset_minutes),
        timezone=str(config.harness.timezone),
    )
    clock = build_session_clock_vectorized(ts_ns=ts_ns, cfg=eng_cfg, tz_name=str(config.harness.timezone))

    feat = _compute_weightiz_features(
        ts_ns=ts_ns,
        symbols=symbols,
        open_px=open_px,
        high_px=high_px,
        low_px=low_px,
        close_px=close_px,
        volume=volume,
        bar_valid=bar_valid,
        windows=windows,
    )

    arrays: dict[str, np.ndarray] = {
        "ts_ns": ts_ns.astype(np.int64, copy=False),
        "open": open_px.astype(np.float64, copy=False),
        "high": high_px.astype(np.float64, copy=False),
        "low": low_px.astype(np.float64, copy=False),
        "close": close_px.astype(np.float64, copy=False),
        "volume": volume.astype(np.float64, copy=False),
        "bar_valid": bar_valid.astype(np.bool_, copy=False),
        "last_valid_close": last_valid_close.astype(np.float64, copy=False),
        "minute_of_day": np.asarray(clock["minute_of_day"], dtype=np.int16),
        "session_id": np.asarray(clock["session_id"], dtype=np.int64),
        "gap_min": np.asarray(clock["gap_min"], dtype=np.float64),
        "phase": np.asarray(clock["phase"], dtype=np.int8),
        "reset_flag": np.asarray(clock["reset_flag"], dtype=np.int8),
    }
    arrays.update(feat)

    registry = write_shared_buffers(arrays)

    return MarketDataSharedHandle(
        symbols=symbols,
        ts_ns=ts_ns,
        timezone=str(config.harness.timezone),
        dataset_hash=dataset_hash,
        registry=registry,
        local_arrays=arrays,
        dq_report=dq_report,
    )
