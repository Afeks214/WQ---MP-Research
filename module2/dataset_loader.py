from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import hashlib

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetForensics:
    timestamp_source: str
    symbol_source: str
    ohlcv_columns: dict[str, str]
    rows: int
    symbols: tuple[str, ...]
    monotonic_time: bool
    missing_counts: dict[str, int]
    session_resets: int


@dataclass(frozen=True)
class LoadedDataset:
    ts_ns: np.ndarray
    symbols: tuple[str, ...]
    open_at: np.ndarray
    high_at: np.ndarray
    low_at: np.ndarray
    close_at: np.ndarray
    volume_at: np.ndarray
    reset_flag_t: np.ndarray
    forensics: DatasetForensics


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _infer_timestamp_series(df: pd.DataFrame) -> tuple[pd.Series, str]:
    cols = {str(c).strip().lower(): str(c) for c in df.columns}
    for cand in ("timestamp", "ts", "datetime", "time", "date"):
        if cand in cols:
            return pd.to_datetime(df[cols[cand]], utc=True, errors="coerce"), cols[cand]
    if isinstance(df.index, pd.DatetimeIndex):
        return pd.to_datetime(df.index, utc=True, errors="coerce").to_series(index=df.index), "index"
    raise RuntimeError("Could not infer timestamp column")


def _resolve_ohlcv_columns(df: pd.DataFrame) -> dict[str, str]:
    cols = {str(c).strip().lower(): str(c) for c in df.columns}
    req = {
        "open": ("open", "o"),
        "high": ("high", "h"),
        "low": ("low", "l"),
        "close": ("close", "c"),
        "volume": ("volume", "vol", "v"),
    }
    out: dict[str, str] = {}
    for key, candidates in req.items():
        hit = None
        for c in candidates:
            if c in cols:
                hit = cols[c]
                break
        if hit is None:
            raise RuntimeError(f"Missing OHLCV column {key!r}")
        out[key] = hit
    return out


def _read_path(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise RuntimeError(f"Unsupported market data format: {path}")


def _load_symbol_frame(path: Path, symbol: str) -> tuple[pd.DataFrame, DatasetForensics]:
    raw = _read_path(path)
    ts, ts_source = _infer_timestamp_series(raw)
    ohlcv = _resolve_ohlcv_columns(raw)

    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "open": pd.to_numeric(raw[ohlcv["open"]], errors="coerce"),
            "high": pd.to_numeric(raw[ohlcv["high"]], errors="coerce"),
            "low": pd.to_numeric(raw[ohlcv["low"]], errors="coerce"),
            "close": pd.to_numeric(raw[ohlcv["close"]], errors="coerce"),
            "volume": pd.to_numeric(raw[ohlcv["volume"]], errors="coerce"),
        }
    )
    frame = frame.dropna(subset=["timestamp"]).copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.floor("min")
    frame = frame.sort_values("timestamp", kind="mergesort")
    frame = frame.drop_duplicates(subset=["timestamp"], keep="last")
    frame["symbol"] = symbol

    missing = {
        "open": int(frame["open"].isna().sum()),
        "high": int(frame["high"].isna().sum()),
        "low": int(frame["low"].isna().sum()),
        "close": int(frame["close"].isna().sum()),
        "volume": int(frame["volume"].isna().sum()),
    }

    ts_ns = frame["timestamp"].view("int64").to_numpy(dtype=np.int64)
    monotonic = bool(ts_ns.size <= 1 or np.all(np.diff(ts_ns) > 0))
    gaps = np.zeros(ts_ns.shape[0], dtype=np.float64)
    if ts_ns.size > 1:
        gaps[1:] = (ts_ns[1:] - ts_ns[:-1]) / 60_000_000_000.0
    resets = int(np.count_nonzero(gaps > 5.0))

    forensic = DatasetForensics(
        timestamp_source=ts_source,
        symbol_source=str(path),
        ohlcv_columns=ohlcv,
        rows=int(frame.shape[0]),
        symbols=(symbol,),
        monotonic_time=monotonic,
        missing_counts=missing,
        session_resets=resets,
    )
    return frame, forensic


def _resolve_symbol_paths(dataset_path: Path, symbols: Sequence[str] | None) -> dict[str, Path]:
    if dataset_path.is_file():
        sym = str(dataset_path.stem).upper() if symbols is None else str(symbols[0]).upper()
        return {sym: dataset_path}

    if not dataset_path.is_dir():
        raise RuntimeError(f"Dataset path does not exist: {dataset_path}")

    out: dict[str, Path] = {}
    if symbols is None:
        for child in sorted(dataset_path.iterdir()):
            if child.is_dir():
                p = child / "part-2026-1Min.parquet"
                if p.exists():
                    out[child.name.upper()] = p
            elif child.suffix.lower() in {".parquet", ".csv"}:
                out[child.stem.upper()] = child
    else:
        for s in symbols:
            sym = str(s).upper()
            direct = dataset_path / f"{sym}.parquet"
            nested = dataset_path / sym / "part-2026-1Min.parquet"
            if direct.exists():
                out[sym] = direct
            elif nested.exists():
                out[sym] = nested
            else:
                raise RuntimeError(f"Missing dataset file for symbol={sym}")

    if not out:
        raise RuntimeError(f"No symbol files found under {dataset_path}")
    return {k: out[k] for k in sorted(out.keys())}


def load_market_dataset(
    dataset_path: str | Path,
    *,
    symbols: Sequence[str] | None = None,
    gap_reset_minutes: float = 5.0,
) -> LoadedDataset:
    root = Path(dataset_path)
    sym_paths = _resolve_symbol_paths(root, symbols)

    frames: list[pd.DataFrame] = []
    forensics: list[DatasetForensics] = []
    for sym, path in sym_paths.items():
        frame, forensic = _load_symbol_frame(path, sym)
        frames.append(frame)
        forensics.append(forensic)

    all_ts = pd.Index(sorted(set().union(*[set(f["timestamp"]) for f in frames])))
    if all_ts.empty:
        raise RuntimeError("No timestamps found in dataset")

    ts_ns = all_ts.view("int64").to_numpy(dtype=np.int64)
    if ts_ns.size > 1 and np.any(np.diff(ts_ns) <= 0):
        raise RuntimeError("Time axis must be strictly monotonic")

    A = len(frames)
    T = int(all_ts.shape[0])
    open_at = np.full((A, T), np.nan, dtype=np.float64)
    high_at = np.full((A, T), np.nan, dtype=np.float64)
    low_at = np.full((A, T), np.nan, dtype=np.float64)
    close_at = np.full((A, T), np.nan, dtype=np.float64)
    volume_at = np.full((A, T), np.nan, dtype=np.float64)

    symbols_out: list[str] = []
    for a, frame in enumerate(frames):
        sym = str(frame["symbol"].iloc[0]).upper()
        symbols_out.append(sym)
        idx = np.searchsorted(ts_ns, frame["timestamp"].view("int64").to_numpy(dtype=np.int64))
        open_at[a, idx] = frame["open"].to_numpy(dtype=np.float64)
        high_at[a, idx] = frame["high"].to_numpy(dtype=np.float64)
        low_at[a, idx] = frame["low"].to_numpy(dtype=np.float64)
        close_at[a, idx] = frame["close"].to_numpy(dtype=np.float64)
        volume_at[a, idx] = frame["volume"].to_numpy(dtype=np.float64)

    gaps = np.zeros(T, dtype=np.float64)
    if T > 1:
        gaps[1:] = (ts_ns[1:] - ts_ns[:-1]) / 60_000_000_000.0
    reset = (gaps > float(gap_reset_minutes)).astype(np.int8)
    reset[0] = np.int8(1)

    missing_total = {
        "open": int(np.isnan(open_at).sum()),
        "high": int(np.isnan(high_at).sum()),
        "low": int(np.isnan(low_at).sum()),
        "close": int(np.isnan(close_at).sum()),
        "volume": int(np.isnan(volume_at).sum()),
    }

    merged_forensics = DatasetForensics(
        timestamp_source=forensics[0].timestamp_source,
        symbol_source=",".join(sorted(str(p) for p in sym_paths.values())),
        ohlcv_columns=forensics[0].ohlcv_columns,
        rows=int(sum(f.rows for f in forensics)),
        symbols=tuple(symbols_out),
        monotonic_time=bool(T <= 1 or np.all(np.diff(ts_ns) > 0)),
        missing_counts=missing_total,
        session_resets=int(np.count_nonzero(reset)),
    )

    return LoadedDataset(
        ts_ns=ts_ns,
        symbols=tuple(symbols_out),
        open_at=open_at,
        high_at=high_at,
        low_at=low_at,
        close_at=close_at,
        volume_at=volume_at,
        reset_flag_t=reset,
        forensics=merged_forensics,
    )
