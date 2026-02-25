from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np


CANONICAL_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


def _require_pandas() -> Any:
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError("pandas is required. Install with: pip install pandas") from exc
    return pd


def parse_hhmm(text: str) -> int:
    s = str(text).strip()
    parts = s.split(":")
    if len(parts) != 2:
        raise RuntimeError(f"Invalid HH:MM time string: {text!r}")
    hh = int(parts[0])
    mm = int(parts[1])
    if hh < 0 or hh > 23 or mm < 0 or mm > 59:
        raise RuntimeError(f"Invalid HH:MM range: {text!r}")
    return hh * 60 + mm


def _pick_col(df: Any, candidates: Sequence[str], label: str) -> str:
    cols = {str(c).strip().lower(): str(c) for c in df.columns}
    for c in candidates:
        key = str(c).strip().lower()
        if key in cols:
            return cols[key]
    raise RuntimeError(f"Missing required field '{label}' in bars payload")


def _require_exchange_calendars() -> Any:
    try:
        import exchange_calendars as xcals  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "calendar_mode='nyse' requires exchange_calendars. "
            "Install with: pip install exchange-calendars"
        ) from exc
    return xcals


def _effective_rth_close_minute(rth_close: str, rth_close_inclusive: bool) -> int:
    close_min = parse_hhmm(rth_close)
    if bool(rth_close_inclusive):
        return close_min
    return close_min - 1


def _build_nyse_expected_minutes(
    session_dates: Sequence[str],
    timezone: str,
    rth_open_minute: int,
    rth_last_minute: int,
) -> Dict[str, int]:
    windows = _build_nyse_expected_windows(
        session_dates=session_dates,
        timezone=timezone,
        rth_open_minute=rth_open_minute,
        rth_last_minute=rth_last_minute,
    )
    out: Dict[str, int] = {}
    for session_key, window in windows.items():
        start_min, end_min = window
        out[session_key] = int(max(end_min - start_min + 1, 0))
    return out


def _build_nyse_expected_windows(
    session_dates: Sequence[str],
    timezone: str,
    rth_open_minute: int,
    rth_last_minute: int,
) -> Dict[str, Tuple[int, int]]:
    pd = _require_pandas()
    xcals = _require_exchange_calendars()

    if not session_dates:
        return {}

    cal = xcals.get_calendar("XNYS")
    start = pd.Timestamp(min(session_dates))
    end = pd.Timestamp(max(session_dates))
    sessions = cal.sessions_in_range(start, end)

    out: Dict[str, Tuple[int, int]] = {}
    for session in sessions:
        open_utc = pd.Timestamp(cal.session_open(session))
        close_utc = pd.Timestamp(cal.session_close(session))
        if open_utc.tzinfo is None:
            open_utc = open_utc.tz_localize("UTC")
        if close_utc.tzinfo is None:
            close_utc = close_utc.tz_localize("UTC")

        open_local = open_utc.tz_convert(str(timezone))
        close_local = close_utc.tz_convert(str(timezone))
        sched_open = int(open_local.hour * 60 + open_local.minute)
        sched_last = int(close_local.hour * 60 + close_local.minute) - 1

        start_min = max(int(rth_open_minute), int(sched_open))
        end_min = min(int(rth_last_minute), int(sched_last))
        session_key = pd.Timestamp(session).strftime("%Y-%m-%d")
        out[session_key] = (int(start_min), int(end_min))
    return out


def bars_records_to_frame(records: Sequence[Dict[str, Any]]) -> Any:
    """Normalize raw Alpaca records into canonical timestamp/OHLCV rows before dedup."""
    pd = _require_pandas()
    if not records:
        return pd.DataFrame(columns=[*CANONICAL_COLUMNS, "ts_raw"])

    raw = pd.DataFrame(list(records))
    ts_col = _pick_col(raw, ("t", "timestamp", "time"), "timestamp")
    o_col = _pick_col(raw, ("o", "open"), "open")
    h_col = _pick_col(raw, ("h", "high"), "high")
    l_col = _pick_col(raw, ("l", "low"), "low")
    c_col = _pick_col(raw, ("c", "close"), "close")
    v_col = _pick_col(raw, ("v", "volume"), "volume")

    ts = pd.to_datetime(raw[ts_col], utc=True, errors="coerce")

    out = pd.DataFrame(
        {
            "ts_raw": ts,
            "open": pd.to_numeric(raw[o_col], errors="coerce").astype(np.float64),
            "high": pd.to_numeric(raw[h_col], errors="coerce").astype(np.float64),
            "low": pd.to_numeric(raw[l_col], errors="coerce").astype(np.float64),
            "close": pd.to_numeric(raw[c_col], errors="coerce").astype(np.float64),
            "volume": pd.to_numeric(raw[v_col], errors="coerce").astype(np.float64),
        }
    )

    out = out.dropna(subset=["ts_raw"]).copy()
    out["timestamp"] = out["ts_raw"].dt.floor("min")
    out = out.sort_values(["timestamp", "ts_raw"], kind="mergesort")
    out = out.reset_index(drop=True)
    return out


def deduplicate_canonical_minutes(df: Any) -> Tuple[Any, int]:
    """
    Deduplicate per-minute rows without losing volume.

    Aggregation:
    - open=first, close=last, high=max, low=min, volume=sum
    """
    pd = _require_pandas()
    if df.empty:
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS)), 0

    work = df.copy()
    if "ts_raw" in work.columns:
        work = work.sort_values(["timestamp", "ts_raw"], kind="mergesort")
    else:
        has_duplicate_minutes = bool(work.duplicated(subset=["timestamp"], keep=False).any())
        if has_duplicate_minutes:
            raise RuntimeError(
                "Intraminute ordering is undefined without ts_raw when duplicate minute timestamps exist"
            )
        work = work.sort_values(["timestamp"], kind="mergesort")

    orig_rows = int(work.shape[0])
    agg = (
        work.groupby("timestamp", sort=True, as_index=False)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .loc[:, list(CANONICAL_COLUMNS)]
    )
    agg["open"] = agg["open"].to_numpy(dtype=np.float64)
    agg["high"] = agg["high"].to_numpy(dtype=np.float64)
    agg["low"] = agg["low"].to_numpy(dtype=np.float64)
    agg["close"] = agg["close"].to_numpy(dtype=np.float64)
    agg["volume"] = agg["volume"].to_numpy(dtype=np.float64)
    dup_count = int(orig_rows - int(agg.shape[0]))
    return agg, dup_count


def _apply_session_policy(
    df: Any,
    timezone: str,
    session_policy: str,
    rth_open: str,
    rth_close: str,
    rth_close_inclusive: bool = False,
    calendar_mode: str = "naive",
) -> Tuple[Any, Dict[str, Any]]:
    pd = _require_pandas()
    policy = str(session_policy).upper()
    if policy not in {"RTH", "ETH"}:
        raise RuntimeError(f"Unsupported session_policy={session_policy!r}; expected RTH or ETH")

    calendar_mode_lc = str(calendar_mode).strip().lower()
    if calendar_mode_lc not in {"naive", "nyse"}:
        raise RuntimeError(f"Unsupported calendar_mode={calendar_mode!r}; expected 'naive' or 'nyse'")
    qa_mode = "nyse" if (calendar_mode_lc == "nyse" and policy == "RTH") else "naive"
    meta_warnings = []
    rth_close_inclusive_effective = bool(rth_close_inclusive)
    nyse_last_minute_policy = None
    if qa_mode == "nyse":
        nyse_last_minute_policy = "close_minus_1"
        rth_close_inclusive_effective = False
        if bool(rth_close_inclusive):
            meta_warnings.append("nyse_calendar_mode_ignores_rth_close_inclusive")

    if df.empty:
        meta = {
            "session_policy": policy,
            "timezone": str(timezone),
            "rth_open": str(rth_open),
            "rth_close": str(rth_close),
            "rth_close_inclusive": bool(rth_close_inclusive),
            "calendar_mode": calendar_mode_lc,
            "qa_mode": qa_mode,
            "warnings": list(meta_warnings),
            "n_sessions": 0,
            "expected_minutes_total": 0,
            "observed_minutes_total": 0,
            "missing_minutes_total": 0,
            "expected_minutes": 0,
            "missing_minutes": 0,
            "missing_minutes_preview": [],
            "missing_minutes_pct": 0.0,
            "coverage_pct": 0.0,
        }
        if qa_mode == "naive":
            meta["coverage_pct_naive"] = 0.0
            meta["missing_minutes_pct_naive"] = 0.0
        else:
            meta["nyse_last_minute_policy"] = str(nyse_last_minute_policy)
            meta["rth_close_inclusive_effective"] = bool(rth_close_inclusive_effective)
        return df, meta

    work = df.copy()
    ex = work["timestamp"].dt.tz_convert(str(timezone))
    mod = ex.dt.hour * 60 + ex.dt.minute
    work["exchange_time"] = ex
    work["minute_of_day"] = mod.astype(np.int32)
    work["session_date"] = ex.dt.strftime("%Y-%m-%d")

    session_dates_all = sorted(set(ex.dt.strftime("%Y-%m-%d").tolist()))
    n_sessions = 0
    expected_total = 0
    missing_preview: list[str] = []
    expected_windows: Dict[str, Tuple[int, int]] = {}

    if policy == "RTH":
        open_min = parse_hhmm(rth_open)
        close_min = _effective_rth_close_minute(
            rth_close,
            False if qa_mode == "nyse" else bool(rth_close_inclusive),
        )
        keep = (work["minute_of_day"] >= open_min) & (work["minute_of_day"] <= close_min)
        work = work.loc[keep].copy()
        if qa_mode == "nyse":
            expected_windows = _build_nyse_expected_windows(
                session_dates=session_dates_all,
                timezone=timezone,
                rth_open_minute=open_min,
                rth_last_minute=close_min,
            )
            expected_session_dates = set(expected_windows.keys())
            work = work.loc[work["session_date"].isin(expected_session_dates)].copy()
            n_sessions = int(len(expected_windows))
        else:
            session_dates = sorted(set(work["session_date"].tolist()))
            expected_windows = {d: (int(open_min), int(close_min)) for d in session_dates}
            n_sessions = int(len(expected_windows))
        expected_total = int(
            sum(int(max(window[1] - window[0] + 1, 0)) for window in expected_windows.values())
        )
    else:
        expected_per_session = 1440
        n_sessions = int(work["session_date"].nunique()) if not work.empty else 0
        expected_total = int(n_sessions * expected_per_session)
    observed_total = int(work.shape[0])

    if policy == "RTH":
        observed_local_minutes = set(work["exchange_time"].dt.strftime("%Y-%m-%dT%H:%M").tolist())
        missing = 0
        for session_key in sorted(expected_windows.keys()):
            start_min, end_min = expected_windows[session_key]
            if int(end_min) < int(start_min):
                continue
            for minute in range(int(start_min), int(end_min) + 1):
                stamp = f"{session_key}T{minute // 60:02d}:{minute % 60:02d}"
                if stamp not in observed_local_minutes:
                    missing += 1
                    if len(missing_preview) < 200:
                        missing_preview.append(stamp)
    else:
        missing = int(max(expected_total - observed_total, 0))

    coverage_pct = float(100.0 * observed_total / expected_total) if expected_total > 0 else 0.0
    missing_pct = float(100.0 * missing / expected_total) if expected_total > 0 else 0.0

    meta = {
        "session_policy": policy,
        "timezone": str(timezone),
        "rth_open": str(rth_open),
        "rth_close": str(rth_close),
        "rth_close_inclusive": bool(rth_close_inclusive),
        "calendar_mode": calendar_mode_lc,
        "qa_mode": qa_mode,
        "warnings": list(meta_warnings),
        "n_sessions": n_sessions,
        "expected_minutes_total": expected_total,
        "observed_minutes_total": observed_total,
        "missing_minutes_total": missing,
        "expected_minutes": expected_total,
        "missing_minutes": missing,
        "missing_minutes_preview": list(missing_preview),
        "missing_minutes_pct": missing_pct,
        "coverage_pct": coverage_pct,
    }
    if qa_mode == "naive":
        meta["coverage_pct_naive"] = coverage_pct
        meta["missing_minutes_pct_naive"] = missing_pct
    else:
        meta["nyse_last_minute_policy"] = str(nyse_last_minute_policy)
        meta["rth_close_inclusive_effective"] = bool(rth_close_inclusive_effective)
    return work, meta


def _drop_invalid_ohlcv(df: Any) -> Tuple[Any, int]:
    pd = _require_pandas()
    if df.empty:
        return pd.DataFrame(columns=list(df.columns)), 0

    work = df.copy()
    finite = (
        np.isfinite(work["open"].to_numpy(dtype=np.float64))
        & np.isfinite(work["high"].to_numpy(dtype=np.float64))
        & np.isfinite(work["low"].to_numpy(dtype=np.float64))
        & np.isfinite(work["close"].to_numpy(dtype=np.float64))
        & np.isfinite(work["volume"].to_numpy(dtype=np.float64))
    )
    open_v = work["open"].to_numpy(dtype=np.float64)
    high_v = work["high"].to_numpy(dtype=np.float64)
    low_v = work["low"].to_numpy(dtype=np.float64)
    close_v = work["close"].to_numpy(dtype=np.float64)
    vol_v = work["volume"].to_numpy(dtype=np.float64)

    ohlc_ok = (
        (high_v >= np.maximum(open_v, close_v))
        & (low_v <= np.minimum(open_v, close_v))
        & (vol_v >= 0.0)
    )

    keep = finite & ohlc_ok
    drops = int(np.size(keep) - int(np.sum(keep)))
    out = work.loc[keep].copy()
    return out, drops


def _ensure_strictly_increasing_timestamp(df: Any) -> Any:
    pd = _require_pandas()
    if df.empty:
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS))

    out = df.sort_values("timestamp", kind="mergesort").copy()
    # Should already be deduplicated; keep deterministic final guard.
    out = out.drop_duplicates(subset=["timestamp"], keep="first")

    ts = out["timestamp"]
    if not bool(ts.is_monotonic_increasing):
        raise RuntimeError("Timestamp index is not monotonic increasing after cleaning")

    if bool(ts.duplicated(keep=False).any()):
        raise RuntimeError("Duplicate timestamps remain after cleaning")

    return out.loc[:, list(CANONICAL_COLUMNS)]


def canonicalize_alpaca_bars(
    records: Sequence[Dict[str, Any]],
    symbol: str,
    timezone: str,
    session_policy: str,
    rth_open: str,
    rth_close: str,
    rth_close_inclusive: bool = False,
    calendar_mode: str = "naive",
) -> Tuple[Any, Dict[str, Any]]:
    """
    Full deterministic canonicalization pipeline for Alpaca bars.

    Returns:
    - clean DataFrame with canonical columns only
    - QA dictionary for reports
    """
    pd = _require_pandas()

    raw_rows = int(len(records))
    pre = bars_records_to_frame(records)
    parsed_rows = int(pre.shape[0])

    deduped, dup_count = deduplicate_canonical_minutes(pre)
    sessioned, session_meta = _apply_session_policy(
        deduped,
        timezone=timezone,
        session_policy=session_policy,
        rth_open=rth_open,
        rth_close=rth_close,
        rth_close_inclusive=rth_close_inclusive,
        calendar_mode=calendar_mode,
    )

    valid, drop_count = _drop_invalid_ohlcv(sessioned)
    clean = _ensure_strictly_increasing_timestamp(valid)

    qa = {
        "symbol": str(symbol).upper(),
        "raw_rows": raw_rows,
        "parsed_rows": parsed_rows,
        "dedup_aggregated_rows": int(deduped.shape[0]),
        "duplicate_rows_collapsed": int(dup_count),
        "rows_after_session_policy": int(sessioned.shape[0]),
        "rows_dropped_invalid": int(drop_count),
        "repairs_count": 0,
        "rows_final": int(clean.shape[0]),
        "index_monotonic_increasing": bool(clean["timestamp"].is_monotonic_increasing),
        "session": session_meta,
    }

    # Compute invariant flags on final output.
    if not clean.empty:
        open_v = clean["open"].to_numpy(dtype=np.float64)
        high_v = clean["high"].to_numpy(dtype=np.float64)
        low_v = clean["low"].to_numpy(dtype=np.float64)
        close_v = clean["close"].to_numpy(dtype=np.float64)
        vol_v = clean["volume"].to_numpy(dtype=np.float64)
        qa["invariants_ok"] = bool(
            np.all(high_v >= np.maximum(open_v, close_v))
            and np.all(low_v <= np.minimum(open_v, close_v))
            and np.all(vol_v >= 0.0)
        )
    else:
        qa["invariants_ok"] = True

    return clean, qa


def run_post_clean_qa_or_raise(
    clean: Any,
    session_meta: Dict[str, Any],
    timezone: str,
    session_policy: str,
    rth_open: str,
    rth_close: str,
    rth_close_inclusive: bool = False,
    calendar_mode: str = "naive",
) -> Dict[str, Any]:
    """
    Strict post-clean QA checks for ingestion fail-closed behavior.

    This validator operates only on the cleaned canonical frame.
    """
    pd = _require_pandas()
    if clean is None or int(clean.shape[0]) <= 0:
        raise RuntimeError("Post-clean QA failed: clean frame is empty")

    cols = list(clean.columns)
    if cols != list(CANONICAL_COLUMNS):
        raise RuntimeError(
            "Post-clean QA failed: clean columns mismatch "
            f"(expected={list(CANONICAL_COLUMNS)}, got={cols})"
        )

    nan_count = int(clean.loc[:, list(CANONICAL_COLUMNS)].isna().sum().sum())
    if nan_count > 0:
        raise RuntimeError(f"Post-clean QA failed: NaN values in canonical columns (count={nan_count})")

    if not bool(clean["timestamp"].is_monotonic_increasing):
        raise RuntimeError("Post-clean QA failed: timestamps are not strictly increasing")
    dup_after = int(clean["timestamp"].duplicated(keep=False).sum())
    if dup_after > 0:
        raise RuntimeError(f"Post-clean QA failed: duplicate timestamps remain (count={dup_after})")

    ts = clean["timestamp"]
    if not isinstance(ts.dtype, pd.DatetimeTZDtype):
        raise RuntimeError("Post-clean QA failed: timestamp dtype must be timezone-aware UTC datetime")
    tz_name = str(getattr(ts.dt, "tz", ""))
    if tz_name.upper() != "UTC":
        raise RuntimeError(f"Post-clean QA failed: timestamp timezone must be UTC (got={tz_name!r})")

    policy = str(session_policy).upper()
    calendar_mode_lc = str(calendar_mode).strip().lower()
    rth_last = _effective_rth_close_minute(
        rth_close,
        False if (policy == "RTH" and calendar_mode_lc == "nyse") else bool(rth_close_inclusive),
    )
    rth_open_min = parse_hhmm(rth_open)

    minute_of_day_max = None
    minute_of_day_min = None
    if policy == "RTH":
        local = ts.dt.tz_convert(str(timezone))
        minute_of_day = (local.dt.hour * 60 + local.dt.minute).to_numpy(dtype=np.int32)
        minute_of_day_max = int(np.max(minute_of_day))
        minute_of_day_min = int(np.min(minute_of_day))
        if bool(np.any(minute_of_day < rth_open_min)) or bool(np.any(minute_of_day > rth_last)):
            raise RuntimeError(
                "Post-clean QA failed: RTH boundary violation "
                f"(expected [{rth_open_min},{rth_last}] minute-of-day)"
            )
        if str(rth_close) == "16:00" and not bool(rth_close_inclusive):
            if bool(np.any(minute_of_day == 960)):
                raise RuntimeError("Post-clean QA failed: found 16:00 minute while rth_close_inclusive=false")

    missing_minutes_total = int(session_meta.get("missing_minutes_total", 0))
    expected_minutes_total = int(session_meta.get("expected_minutes_total", 0))

    return {
        "canonical_columns_ok": True,
        "nan_count": int(nan_count),
        "duplicate_minutes_after_clean": int(dup_after),
        "timestamps_monotonic_increasing": True,
        "timestamps_utc_tz_aware": True,
        "rth_last_minute_observed": minute_of_day_max,
        "rth_first_minute_observed": minute_of_day_min,
        "missing_minutes_total": int(missing_minutes_total),
        "expected_minutes_total": int(expected_minutes_total),
    }


def summarize_session_meta_for_clean_frame(
    clean: Any,
    timezone: str,
    session_policy: str,
    rth_open: str,
    rth_close: str,
    rth_close_inclusive: bool = False,
    calendar_mode: str = "naive",
) -> Dict[str, Any]:
    pd = _require_pandas()
    if clean is None or int(clean.shape[0]) <= 0:
        base = pd.DataFrame(columns=list(CANONICAL_COLUMNS))
    else:
        base = clean.loc[:, list(CANONICAL_COLUMNS)].copy()
    _, meta = _apply_session_policy(
        base,
        timezone=timezone,
        session_policy=session_policy,
        rth_open=rth_open,
        rth_close=rth_close,
        rth_close_inclusive=rth_close_inclusive,
        calendar_mode=calendar_mode,
    )
    return meta


def merge_canonical_frames(existing: Any, new: Any) -> Any:
    """Merge existing/new canonical frames deterministically and deduplicate by timestamp."""
    pd = _require_pandas()

    if existing is None or int(existing.shape[0]) == 0:
        base = new.copy()
    elif new is None or int(new.shape[0]) == 0:
        base = existing.copy()
    else:
        base = pd.concat([existing, new], axis=0, ignore_index=True)

    deduped, _ = deduplicate_canonical_minutes(base)
    valid, _ = _drop_invalid_ohlcv(deduped)
    return _ensure_strictly_increasing_timestamp(valid)
