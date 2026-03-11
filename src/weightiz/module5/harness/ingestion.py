from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np


def find_col(df: Any, candidates: tuple[str, ...], name: str) -> str:
    cols = {str(c).strip().lower(): str(c) for c in df.columns}
    for cand in candidates:
        if cand in cols:
            return cols[cand]
    raise RuntimeError(f"Missing required column '{name}' in input file")


def load_asset_frame(path: str, tz_name: str, *, require_pandas_fn: Callable[[], Any]) -> Any:
    pdx = require_pandas_fn()
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"Data path does not exist: {path}")

    if p.suffix.lower() == ".parquet":
        df = pdx.read_parquet(p)
    else:
        df = pdx.read_csv(p)

    ts_col = None
    cols = {str(c).strip().lower(): str(c) for c in df.columns}
    for cand in ("timestamp", "ts", "datetime", "date", "time"):
        if cand in cols:
            ts_col = cols[cand]
            break
    o_col = find_col(df, ("open", "o"), "open")
    h_col = find_col(df, ("high", "h"), "high")
    l_col = find_col(df, ("low", "l"), "low")
    c_col = find_col(df, ("close", "c"), "close")
    v_col = find_col(df, ("volume", "vol", "v"), "volume")

    if ts_col is not None:
        ts_raw = pdx.to_datetime(df[ts_col], utc=True, errors="coerce")
    elif isinstance(df.index, pdx.DatetimeIndex):
        ts_raw = pdx.to_datetime(df.index, utc=True, errors="coerce")
    else:
        raise RuntimeError(f"Missing required column 'timestamp' in input file and index is not DatetimeIndex: {path}")
    ts_idx = pdx.DatetimeIndex(ts_raw)
    keep = np.asarray(ts_idx.notna(), dtype=bool)
    if not np.any(keep):
        raise RuntimeError(f"No parseable timestamps in {path}")

    out = pdx.DataFrame(
        {
            "timestamp": ts_idx[keep].floor("min"),
            "open": pdx.to_numeric(df.loc[keep, o_col], errors="coerce"),
            "high": pdx.to_numeric(df.loc[keep, h_col], errors="coerce"),
            "low": pdx.to_numeric(df.loc[keep, l_col], errors="coerce"),
            "close": pdx.to_numeric(df.loc[keep, c_col], errors="coerce"),
            "volume": pdx.to_numeric(df.loc[keep, v_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    out = out.drop_duplicates(subset=["timestamp"], keep="last")
    out = out.set_index("timestamp")
    return out


def validate_utc_minute_index(idx: Any, label: str, *, require_pandas_fn: Callable[[], Any]) -> Any:
    pdx = require_pandas_fn()
    if not isinstance(idx, pdx.DatetimeIndex):
        raise RuntimeError(f"{label} index must be pandas.DatetimeIndex, got {type(idx)!r}")
    if idx.tz is None:
        raise RuntimeError(f"{label} index must be timezone-aware")
    idx_utc = idx.tz_convert("UTC")

    if bool(idx_utc.has_duplicates):
        dup = idx_utc[idx_utc.duplicated(keep=False)][0]
        raise RuntimeError(f"{label} index has duplicate timestamp after UTC conversion: {dup!s}")

    ts_ns = idx_utc.asi8.astype(np.int64)
    if ts_ns.size > 1:
        d = np.diff(ts_ns)
        bad = np.flatnonzero(d <= 0)
        if bad.size > 0:
            i = int(bad[0])
            raise RuntimeError(
                f"{label} index must be strictly increasing in UTC: "
                f"t[{i}]={idx_utc[i]!s}, t[{i + 1}]={idx_utc[i + 1]!s}"
            )

    bad_minute = (
        (idx_utc.second.to_numpy(dtype=np.int64) != 0)
        | (idx_utc.microsecond.to_numpy(dtype=np.int64) != 0)
        | (idx_utc.nanosecond.to_numpy(dtype=np.int64) != 0)
    )
    if np.any(bad_minute):
        i = int(np.flatnonzero(bad_minute)[0])
        raise RuntimeError(
            f"{label} index must be aligned to UTC minute grid "
            f"(second/microsecond/nanosecond all zero); offending timestamp={idx_utc[i]!s}"
        )

    return idx_utc


def build_clock_override_from_utc(ts_ns_utc: np.ndarray, cfg: Any, tz_name: str, *, phase_enum: Any) -> dict[str, np.ndarray]:
    T = int(ts_ns_utc.shape[0])
    if T <= 0:
        raise RuntimeError("Cannot build clock override for empty timestamps")

    tz_local = ZoneInfo(str(tz_name))
    tz_utc = ZoneInfo("UTC")

    minute_of_day = np.empty(T, dtype=np.int16)
    day_ord = np.empty(T, dtype=np.int64)
    for i in range(T):
        ns = int(ts_ns_utc[i])
        sec = ns // 1_000_000_000
        nsec = ns % 1_000_000_000
        dtu = datetime.fromtimestamp(sec, tz=tz_utc).replace(microsecond=int(nsec // 1000))
        dtl = dtu.astimezone(tz_local)
        minute_of_day[i] = np.int16(int(dtl.hour) * 60 + int(dtl.minute))
        day_ord[i] = np.int64(int(dtl.date().toordinal()))

    tod = (minute_of_day.astype(np.int32) - int(cfg.rth_open_minute)).astype(np.int16)
    session_change = np.empty(T, dtype=bool)
    session_change[0] = True
    session_change[1:] = day_ord[1:] != day_ord[:-1]
    session_id = np.cumsum(session_change, dtype=np.int64) - 1

    gap_min = np.zeros(T, dtype=np.float64)
    gap_min[1:] = (ts_ns_utc[1:] - ts_ns_utc[:-1]) / float(60 * 1_000_000_000)
    reset_flag = ((gap_min > float(cfg.gap_reset_minutes)) | session_change).astype(np.int8)
    reset_flag = np.where(reset_flag != 0, np.int8(1), np.int8(0)).astype(np.int8)
    reset_flag[0] = np.int8(1)
    gap_min[0] = 0.0

    valid_reset = np.isin(reset_flag, np.asarray([0, 1], dtype=np.int8))
    if not np.all(valid_reset):
        i = int(np.flatnonzero(~valid_reset)[0])
        raise RuntimeError(
            f"Clock override reset_flag must be binary {{0,1}}: t={i}, value={int(reset_flag[i])}"
        )
    if np.any(gap_min < 0.0):
        i = int(np.flatnonzero(gap_min < 0.0)[0])
        raise RuntimeError(
            f"Clock override gap_min must be non-negative: t={i}, value={float(gap_min[i]):.6f}"
        )

    phase = np.full(T, np.int8(phase_enum.WARMUP), dtype=np.int8)
    is_live = (tod >= int(cfg.warmup_minutes)) & (minute_of_day < int(cfg.flat_time_minute))
    phase[is_live] = np.int8(phase_enum.LIVE)
    is_select = (minute_of_day == int(cfg.flat_time_minute)) & (tod >= int(cfg.warmup_minutes))
    phase[is_select] = np.int8(phase_enum.OVERNIGHT_SELECT)
    is_flat = minute_of_day > int(cfg.flat_time_minute)
    phase[is_flat] = np.int8(phase_enum.FLATTEN)

    return {
        "minute_of_day": minute_of_day,
        "tod": tod,
        "session_id": session_id,
        "gap_min": gap_min,
        "reset_flag": reset_flag,
        "phase": phase,
    }


def ingest_master_aligned(
    data_paths: list[str],
    symbols: list[str],
    engine_cfg: Any,
    harness_cfg: Any,
    data_loader_func: Callable[[str, str], Any] | None = None,
    *,
    require_pandas_fn: Callable[[], Any],
    load_asset_frame_fn: Callable[[str, str], Any],
    validate_utc_minute_index_fn: Callable[[Any, str], Any],
    compute_bar_valid_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    build_clock_override_from_utc_fn: Callable[[np.ndarray, Any, str], dict[str, np.ndarray]],
    replace_fn: Callable[..., Any],
    preallocate_state_fn: Callable[..., Any],
    validate_loaded_market_slice_fn: Callable[..., Any],
    validate_state_hard_fn: Callable[..., Any],
    dq_validate_fn: Callable[..., Any],
    dq_apply_fn: Callable[..., Any],
    dq_accept: str,
    dq_degrade: str,
    dq_reject: str,
) -> tuple[Any, np.ndarray, list[str], np.ndarray, dict[str, Any], np.ndarray, dict[str, Any]]:
    if len(data_paths) != len(symbols):
        raise RuntimeError("data_paths and symbols lengths must match")

    pdx = require_pandas_fn()
    loader = data_loader_func if data_loader_func is not None else load_asset_frame_fn

    raw_frames: list[Any] = []
    dq_day_reports: list[dict[str, Any]] = []
    dq_bar_flags_rows: list[dict[str, Any]] = []
    for a, p in enumerate(data_paths):
        fr = loader(p, harness_cfg.timezone)
        dq_reports = dq_validate_fn(
            df=fr,
            symbol=str(symbols[a]),
            tz_name=str(harness_cfg.timezone),
            session_open_minute=int(engine_cfg.rth_open_minute),
            session_close_minute=int(engine_cfg.flat_time_minute),
            timeframe_min=None,
        )
        fr_repaired, dq_reports_out, dq_bar_flags = dq_apply_fn(
            df=fr,
            reports=dq_reports,
            tz_name=str(harness_cfg.timezone),
        )

        dq_day_reports.extend([r.to_row() for r in dq_reports_out])
        if dq_bar_flags.shape[0] > 0:
            dq_bar_flags_rows.extend(dq_bar_flags.to_dict(orient="records"))

        idx_utc = validate_utc_minute_index_fn(fr_repaired.index, f"asset={symbols[a]} path={p}")
        fr2 = fr_repaired.copy()
        fr2.index = idx_utc
        raw_frames.append(fr2)

    master_idx = raw_frames[0].index
    for fr in raw_frames[1:]:
        master_idx = master_idx.union(fr.index)
    master_idx = master_idx.sort_values()
    master_idx = validate_utc_minute_index_fn(master_idx, "master_idx (after union+sort)")

    T = int(master_idx.shape[0])
    A0 = int(len(symbols))
    open_ta = np.full((T, A0), np.nan, dtype=np.float64)
    high_ta = np.full((T, A0), np.nan, dtype=np.float64)
    low_ta = np.full((T, A0), np.nan, dtype=np.float64)
    close_ta = np.full((T, A0), np.nan, dtype=np.float64)
    vol_ta = np.full((T, A0), np.nan, dtype=np.float64)
    dqs_ta = np.full((T, A0), 1.0, dtype=np.float64)

    for a, fr in enumerate(raw_frames):
        re = fr.reindex(master_idx)
        open_ta[:, a] = re["open"].to_numpy(dtype=np.float64)
        high_ta[:, a] = re["high"].to_numpy(dtype=np.float64)
        low_ta[:, a] = re["low"].to_numpy(dtype=np.float64)
        close_ta[:, a] = re["close"].to_numpy(dtype=np.float64)
        vol_ta[:, a] = re["volume"].to_numpy(dtype=np.float64)
        if "dqs_day" in re.columns:
            dqs_col = pdx.to_numeric(re["dqs_day"], errors="coerce").to_numpy(dtype=np.float64)
            dqs_ta[:, a] = np.where(np.isfinite(dqs_col), dqs_col, 1.0)

    bar_valid_ta = compute_bar_valid_fn(open_ta, high_ta, low_ta, close_ta, vol_ta)
    coverage = np.mean(bar_valid_ta, axis=0)
    keep_assets = coverage >= float(harness_cfg.min_asset_coverage)

    if np.sum(keep_assets) < 2:
        raise RuntimeError(
            f"Coverage filter removed too many assets: kept={int(np.sum(keep_assets))}, required>=2"
        )

    keep_idx = np.where(keep_assets)[0]
    keep_symbols = [symbols[i] for i in keep_idx.tolist()]

    tick = np.asarray(engine_cfg.tick_size, dtype=np.float64)
    if tick.shape != (A0,):
        raise RuntimeError(
            f"engine_cfg.tick_size shape mismatch: got {tick.shape}, expected {(A0,)}"
        )

    open_keep = open_ta[:, keep_idx]
    high_keep = high_ta[:, keep_idx]
    low_keep = low_ta[:, keep_idx]
    close_keep = close_ta[:, keep_idx]
    vol_keep = vol_ta[:, keep_idx]
    dqs_keep = dqs_ta[:, keep_idx]
    bar_keep = bar_valid_ta[:, keep_idx]
    tick_keep = tick[keep_idx]

    ts_ns = master_idx.asi8.astype(np.int64)
    if ts_ns.size > 1 and np.any(np.diff(ts_ns) <= 0):
        i = int(np.flatnonzero(np.diff(ts_ns) <= 0)[0])
        raise RuntimeError(
            f"master_idx -> ts_ns must be strictly increasing int64: "
            f"i={i}, ts_ns[i]={int(ts_ns[i])}, ts_ns[i+1]={int(ts_ns[i + 1])}"
        )

    cfg = replace_fn(engine_cfg, T=T, A=int(keep_idx.size), tick_size=tick_keep.copy())
    clk_override = build_clock_override_from_utc_fn(ts_ns, cfg, harness_cfg.timezone)
    state = preallocate_state_fn(
        ts_ns=ts_ns,
        cfg=cfg,
        symbols=tuple(keep_symbols),
        clock_override=clk_override,
    )

    state.open_px[:, :] = open_keep
    state.high_px[:, :] = high_keep
    state.low_px[:, :] = low_keep
    state.close_px[:, :] = close_keep
    state.volume[:, :] = vol_keep
    state.bar_valid[:, :] = bar_keep
    state.dqs_day_ta = dqs_keep.copy()

    atr0 = np.maximum(4.0 * tick_keep[None, :], 1e-12)
    state.rvol[:, :] = np.where(bar_keep, 1.0, np.nan)
    state.atr_floor[:, :] = np.where(bar_keep, atr0, np.nan)

    validate_loaded_market_slice_fn(state, 0, state.cfg.T)
    validate_state_hard_fn(state)

    ingest_meta = {
        "master_rows": T,
        "assets_input": A0,
        "assets_kept": int(keep_idx.size),
        "coverage": coverage.tolist(),
        "symbols_kept": keep_symbols,
        "dq_counts": {
            "accept": int(sum(1 for r in dq_day_reports if str(r.get("decision")) == dq_accept)),
            "degrade": int(sum(1 for r in dq_day_reports if str(r.get("decision")) == dq_degrade)),
            "reject": int(sum(1 for r in dq_day_reports if str(r.get("decision")) == dq_reject)),
            "n_days_total": int(len(dq_day_reports)),
        },
    }

    dq_bundle = {
        "day_reports": dq_day_reports,
        "bar_flags_rows": dq_bar_flags_rows,
    }

    return state, keep_idx, keep_symbols, master_idx.asi8.astype(np.int64), ingest_meta, tick_keep, dq_bundle
