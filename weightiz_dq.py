from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore[assignment]


DQ_ACCEPT = "ACCEPT"
DQ_DEGRADE = "DEGRADE"
DQ_REJECT = "REJECT"

_ALLOWED_TIMEFRAMES = np.array([1, 2, 3, 5, 10, 15, 30, 60], dtype=np.float64)

# Deterministic thresholds/constants.
_SHORT_SESSION_SPAN_RATIO_MAX = 0.70
_SHORT_SESSION_COVERAGE_MIN = 0.90
_MISSING_REJECT_PCT = 20.0
_GAP_EVENT_PENALTY = 0.05
_GAP_EVENT_PENALTY_CAP = 0.30
_IB_PENALTY = 0.50
_VOLUME_PENALTY = 0.10
_BAD_TICK_PCT_THRESHOLD = 0.10
_BAD_TICK_K = 8.0
_BAD_TICK_WINDOW = 20
_IB_WINDOW_MINUTES = 60
_VOLUME_SPIKE_MULTIPLIER = 100.0
_CADENCE_MIN_DELTAS_KNOWN = 10
_CADENCE_MIN_DELTAS_STABLE = 20
_CADENCE_CV_MAX_STABLE = 0.30

# Per-bar issue bitmask.
DQ_ISSUE_FILLED_BAR = 1 << 0
DQ_ISSUE_GAP_EVENT = 1 << 1
DQ_ISSUE_IB_MISSING = 1 << 2
DQ_ISSUE_VOLUME_SPIKE = 1 << 3
DQ_ISSUE_SHORT_SESSION = 1 << 4
DQ_ISSUE_DUPLICATE_TS = 1 << 5
DQ_ISSUE_NON_MONOTONIC = 1 << 6


@dataclass(frozen=True)
class DQDayReport:
    symbol: str
    session_date: str
    decision: str
    reason_codes: tuple[str, ...]
    timeframe_min: int
    session_open_minute: int
    session_close_minute: int
    expected_bars_nominal: int
    expected_bars_effective: int
    observed_bars: int
    missing_bars_nominal: int
    missing_bars_effective: int
    missing_pct_nominal: float
    missing_pct_effective: float
    gap_events: int
    duplicate_timestamp_count: int
    non_monotonic_count: int
    nan_ohlcv_count: int
    invalid_ohlc_count: int
    zero_volume_day: bool
    total_volume: float
    short_session_inferred: bool
    observed_span_ratio: float
    observed_span_coverage: float
    ib_missing: bool
    volume_spike_count: int
    bad_tick_count: int
    halt_proxy_count: int
    max_abs_ret_1bar: float
    max_dynamic_bad_tick_threshold: float
    dqs_base: float
    dqs_gap_penalty: float
    dqs_ib_penalty: float
    dqs_volume_penalty: float
    dqs_final: float
    cadence_day_min: int
    cadence_day_stable: bool
    cadence_day_delta_count: int
    cadence_day_cv: float
    expected_bars_reference: int
    missing_bars_reference: int
    missing_pct_reference: float
    fill_minutes_effective: tuple[int, ...] = field(default_factory=tuple)

    def to_row(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "session_date": self.session_date,
            "decision": self.decision,
            "reason_codes": "|".join(self.reason_codes),
            "timeframe_min": int(self.timeframe_min),
            "session_open_minute": int(self.session_open_minute),
            "session_close_minute": int(self.session_close_minute),
            "expected_bars_nominal": int(self.expected_bars_nominal),
            "expected_bars_effective": int(self.expected_bars_effective),
            "observed_bars": int(self.observed_bars),
            "missing_bars_nominal": int(self.missing_bars_nominal),
            "missing_bars_effective": int(self.missing_bars_effective),
            "missing_pct_nominal": float(self.missing_pct_nominal),
            "missing_pct_effective": float(self.missing_pct_effective),
            "gap_events": int(self.gap_events),
            "duplicate_timestamp_count": int(self.duplicate_timestamp_count),
            "non_monotonic_count": int(self.non_monotonic_count),
            "nan_ohlcv_count": int(self.nan_ohlcv_count),
            "invalid_ohlc_count": int(self.invalid_ohlc_count),
            "zero_volume_day": bool(self.zero_volume_day),
            "total_volume": float(self.total_volume),
            "short_session_inferred": bool(self.short_session_inferred),
            "observed_span_ratio": float(self.observed_span_ratio),
            "observed_span_coverage": float(self.observed_span_coverage),
            "ib_missing": bool(self.ib_missing),
            "volume_spike_count": int(self.volume_spike_count),
            "bad_tick_count": int(self.bad_tick_count),
            "halt_proxy_count": int(self.halt_proxy_count),
            "max_abs_ret_1bar": float(self.max_abs_ret_1bar),
            "max_dynamic_bad_tick_threshold": float(self.max_dynamic_bad_tick_threshold),
            "dqs_base": float(self.dqs_base),
            "dqs_gap_penalty": float(self.dqs_gap_penalty),
            "dqs_ib_penalty": float(self.dqs_ib_penalty),
            "dqs_volume_penalty": float(self.dqs_volume_penalty),
            "dqs_final": float(self.dqs_final),
            "cadence_day_min": int(self.cadence_day_min),
            "cadence_day_stable": bool(self.cadence_day_stable),
            "cadence_day_delta_count": int(self.cadence_day_delta_count),
            "cadence_day_cv": float(self.cadence_day_cv),
            "expected_bars_reference": int(self.expected_bars_reference),
            "missing_bars_reference": int(self.missing_bars_reference),
            "missing_pct_reference": float(self.missing_pct_reference),
        }


def _require_pandas() -> Any:
    if pd is None:
        raise RuntimeError("pandas is required for DQ validation")
    return pd


def _canonicalize_input_df(df: Any, tz_name: str) -> Any:
    pdx = _require_pandas()
    if not isinstance(df, pdx.DataFrame):
        raise RuntimeError(f"DQ input must be pandas.DataFrame, got {type(df)!r}")

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"DQ input missing required columns: {missing}")

    if isinstance(df.index, pdx.DatetimeIndex):
        ts = pdx.to_datetime(df.index, utc=True, errors="coerce")
    elif "timestamp" in df.columns:
        ts = pdx.to_datetime(df["timestamp"], utc=True, errors="coerce")
    else:
        raise RuntimeError("DQ input must have DatetimeIndex or timestamp column")

    work = pdx.DataFrame(
        {
            "timestamp": ts,
            "open": pdx.to_numeric(df["open"], errors="coerce"),
            "high": pdx.to_numeric(df["high"], errors="coerce"),
            "low": pdx.to_numeric(df["low"], errors="coerce"),
            "close": pdx.to_numeric(df["close"], errors="coerce"),
            "volume": pdx.to_numeric(df["volume"], errors="coerce"),
        }
    )
    work = work.reset_index(drop=True)
    work.index.name = None
    work = work.loc[work["timestamp"].notna()].copy()
    work["_seq"] = np.arange(work.shape[0], dtype=np.int64)
    if work.empty:
        work["timestamp_local"] = pdx.to_datetime([], utc=True).tz_convert(tz_name)
        work["session_date"] = pdx.Series(dtype=str)
        work["minute_of_day"] = pdx.Series(dtype=np.int64)
        return work

    work["timestamp_local"] = work["timestamp"].dt.tz_convert(tz_name)
    work["session_date"] = work["timestamp_local"].dt.strftime("%Y-%m-%d")
    work["minute_of_day"] = (
        work["timestamp_local"].dt.hour.astype(np.int64) * 60
        + work["timestamp_local"].dt.minute.astype(np.int64)
    )
    return work


def _round_timeframe(minutes: float) -> int:
    if not np.isfinite(minutes) or minutes <= 0.0:
        return 1
    idx = int(np.argmin(np.abs(_ALLOWED_TIMEFRAMES - float(minutes))))
    return int(_ALLOWED_TIMEFRAMES[idx])


def _day_cadence_stats(session_day: Any, fallback_min: int) -> tuple[int, bool, int, float]:
    ts_ns = (
        session_day["timestamp"]
        .sort_values(kind="mergesort")
        .drop_duplicates()
        .astype("int64")
        .to_numpy(dtype=np.int64)
    )
    if ts_ns.size < 2:
        return int(fallback_min), False, 0, float("nan")

    deltas = np.diff(ts_ns).astype(np.float64) / float(60 * 1_000_000_000)
    pos = deltas[deltas > 0.0]
    delta_count = int(pos.size)
    if delta_count < int(_CADENCE_MIN_DELTAS_KNOWN):
        return int(fallback_min), False, int(delta_count), float("nan")

    med = float(np.median(pos))
    if not np.isfinite(med) or med <= 0.0:
        return int(fallback_min), False, int(delta_count), float("nan")
    std = float(np.std(pos))
    cv = float(std / med) if med > 0.0 and np.isfinite(std) else float("nan")
    cadence_q = int(_round_timeframe(med))
    stable = bool(
        delta_count >= int(_CADENCE_MIN_DELTAS_STABLE)
        and np.isfinite(cv)
        and float(cv) <= float(_CADENCE_CV_MAX_STABLE)
    )
    if stable:
        return cadence_q, True, int(delta_count), float(cv)
    return int(fallback_min), False, int(delta_count), float(cv)


def _infer_timeframe_min(work: Any, fallback: int | None) -> int:
    if fallback is not None and int(fallback) > 0:
        return _round_timeframe(float(fallback))
    if work.shape[0] < 2:
        return 1
    ts_ns = (
        work["timestamp"]
        .sort_values(kind="mergesort")
        .drop_duplicates()
        .astype("int64")
        .to_numpy(dtype=np.int64)
    )
    if ts_ns.size < 2:
        return 1
    deltas = np.diff(ts_ns).astype(np.float64) / float(60 * 1_000_000_000)
    pos = deltas[deltas > 0.0]
    if pos.size == 0:
        return 1
    return _round_timeframe(float(np.median(pos)))


def _expected_minutes(open_minute: int, close_minute: int, timeframe_min: int) -> np.ndarray:
    if close_minute < open_minute:
        return np.zeros(0, dtype=np.int64)
    return np.arange(int(open_minute), int(close_minute) + 1, int(timeframe_min), dtype=np.int64)


def _expected_minutes_aligned(
    open_minute: int,
    close_minute: int,
    timeframe_min: int,
    anchor_minute: int | None,
) -> np.ndarray:
    if close_minute < open_minute:
        return np.zeros(0, dtype=np.int64)
    step = int(max(1, int(timeframe_min)))
    start = int(open_minute)
    if anchor_minute is not None:
        # Use day-local first observed minute as deterministic anchor.
        # This preserves provider bar-end semantics (e.g., 09:31 for 1m, 09:35 for 5m)
        # instead of forcing open-minute start and fabricating one missing bar per day.
        start = int(anchor_minute)
        if start < int(open_minute):
            offset = int((int(open_minute) - int(start)) % step)
            start = int(open_minute) + ((int(step) - offset) % int(step))
    if start > int(close_minute):
        return np.zeros(0, dtype=np.int64)
    return np.arange(int(start), int(close_minute) + 1, int(step), dtype=np.int64)


def _cluster_gap_events(missing_minutes: np.ndarray, timeframe_min: int) -> int:
    if missing_minutes.size == 0:
        return 0
    missing = np.sort(np.asarray(missing_minutes, dtype=np.int64), kind="mergesort")
    d = np.diff(missing)
    splits = np.flatnonzero(d > int(timeframe_min))
    return int(splits.size + 1)


def _halt_proxy_mask(close_s: Any, volume_s: Any) -> np.ndarray:
    pdx = _require_pandas()
    prev_close = close_s.shift(1)
    stall = (volume_s <= 0.0) & (close_s.sub(prev_close).abs() <= 1e-12)
    if stall.empty:
        return np.zeros(0, dtype=bool)
    run_id = (stall != stall.shift(1, fill_value=False)).cumsum()
    run_len = stall.groupby(run_id).transform("sum").astype(np.int64)
    out = (stall & (run_len >= 3)).to_numpy(dtype=bool)
    # Keep deterministic shape for empty frame.
    return out if out.size > 0 else np.zeros(int(close_s.shape[0]), dtype=bool)


def _session_report(
    symbol: str,
    session_date: str,
    day_raw: Any,
    tz_name: str,
    session_open_minute: int,
    session_close_minute: int,
    timeframe_min: int,
) -> DQDayReport:
    pdx = _require_pandas()
    required_cols = ["open", "high", "low", "close", "volume"]

    day_in_order = day_raw.sort_values(["_seq"], kind="mergesort")
    raw_ts = day_in_order["timestamp"].astype("int64").to_numpy(dtype=np.int64)
    non_monotonic_count = int(np.sum(np.diff(raw_ts) < 0)) if raw_ts.size > 1 else 0
    duplicate_timestamp_count = int(day_in_order["timestamp"].duplicated(keep="last").sum())

    day = day_raw.sort_values(["timestamp", "_seq"], kind="mergesort")
    day = day.drop_duplicates(subset=["timestamp"], keep="last")

    session_mask = (day["minute_of_day"] >= int(session_open_minute)) & (
        day["minute_of_day"] <= int(session_close_minute)
    )
    session_day = day.loc[session_mask].copy()

    cadence_day_min, cadence_day_stable, cadence_day_delta_count, cadence_day_cv = _day_cadence_stats(
        session_day=session_day,
        fallback_min=int(timeframe_min),
    )
    timeframe_eval = int(cadence_day_min if cadence_day_stable else int(timeframe_min))

    observed_minutes = np.sort(session_day["minute_of_day"].drop_duplicates().to_numpy(dtype=np.int64), kind="mergesort")
    cadence_anchor = int(observed_minutes[0]) if (cadence_day_stable and observed_minutes.size > 0) else None
    expected_nominal = _expected_minutes(session_open_minute, session_close_minute, timeframe_min)
    expected_reference = _expected_minutes_aligned(
        session_open_minute,
        session_close_minute,
        timeframe_eval,
        cadence_anchor,
    )

    missing_nominal = np.setdiff1d(expected_nominal, observed_minutes, assume_unique=False)
    missing_reference = np.setdiff1d(expected_reference, observed_minutes, assume_unique=False)
    expected_nominal_n = int(expected_nominal.size)
    expected_reference_n = int(expected_reference.size)
    observed_n = int(observed_minutes.size)
    missing_nominal_n = int(missing_nominal.size)
    missing_reference_n = int(missing_reference.size)
    missing_pct_nominal = 100.0 * float(missing_nominal_n) / float(max(expected_nominal_n, 1))
    missing_pct_reference = 100.0 * float(missing_reference_n) / float(max(expected_reference_n, 1))

    if observed_n > 0:
        span_lo = int(max(int(expected_reference[0]), int(observed_minutes[0])))
        span_hi = int(min(int(expected_reference[-1]), int(observed_minutes[-1])))
        expected_span = expected_reference[
            (expected_reference >= int(span_lo)) & (expected_reference <= int(span_hi))
        ]
    else:
        expected_span = np.zeros(0, dtype=np.int64)

    observed_in_span = np.intersect1d(observed_minutes, expected_span, assume_unique=False)
    span_ratio = float(expected_span.size) / float(max(expected_reference.size, 1))
    span_coverage = float(observed_in_span.size) / float(max(expected_span.size, 1)) if expected_span.size > 0 else 0.0
    short_session_inferred = (
        expected_span.size > 0
        and span_ratio < float(_SHORT_SESSION_SPAN_RATIO_MAX)
        and span_coverage >= float(_SHORT_SESSION_COVERAGE_MIN)
    )

    expected_effective = expected_span if short_session_inferred else expected_reference
    missing_effective = np.setdiff1d(expected_effective, observed_minutes, assume_unique=False)
    expected_effective_n = int(expected_effective.size)
    missing_effective_n = int(missing_effective.size)
    missing_pct_effective = 100.0 * float(missing_effective_n) / float(max(expected_effective_n, 1))
    gap_events = _cluster_gap_events(missing_effective, timeframe_eval)

    nan_ohlcv_count = int(session_day[required_cols].isna().sum().sum())

    invalid_ohlc_mask = (
        (session_day["high"] < session_day["low"])
        | (session_day["close"] < session_day["low"])
        | (session_day["close"] > session_day["high"])
        | (session_day["open"] < session_day["low"])
        | (session_day["open"] > session_day["high"])
    )
    invalid_ohlc_count = int(invalid_ohlc_mask.sum())

    total_volume = float(np.nansum(session_day["volume"].to_numpy(dtype=np.float64)))
    zero_volume_day = bool(observed_n > 0 and np.all(session_day["volume"].to_numpy(dtype=np.float64) <= 0.0))

    vol_vals = session_day["volume"].to_numpy(dtype=np.float64)
    pos_vol = vol_vals[vol_vals > 0.0]
    med_vol = float(np.median(pos_vol)) if pos_vol.size > 0 else float("nan")
    if np.isfinite(med_vol) and med_vol > 0.0:
        volume_spike_count = int(np.sum(vol_vals > (float(_VOLUME_SPIKE_MULTIPLIER) * med_vol)))
    else:
        volume_spike_count = 0

    close_s = session_day["close"].astype(np.float64)
    prev_close = close_s.shift(1)
    ret_abs = (close_s / prev_close - 1.0).abs()
    tr_abs = close_s.sub(prev_close).abs()
    atr_like = tr_abs.rolling(window=int(_BAD_TICK_WINDOW), min_periods=5).median()
    dynamic_threshold = np.maximum(
        float(_BAD_TICK_PCT_THRESHOLD),
        (float(_BAD_TICK_K) * atr_like / prev_close.abs()).to_numpy(dtype=np.float64),
    )

    halt_proxy = _halt_proxy_mask(close_s=close_s, volume_s=session_day["volume"].astype(np.float64))
    ret_arr = ret_abs.to_numpy(dtype=np.float64)
    dyn_arr = np.asarray(dynamic_threshold, dtype=np.float64)
    valid_cmp = np.isfinite(ret_arr) & np.isfinite(dyn_arr)
    bad_tick_mask = valid_cmp & (ret_arr > dyn_arr)
    if bad_tick_mask.size > 0 and halt_proxy.size == bad_tick_mask.size:
        bad_tick_mask = bad_tick_mask & (~halt_proxy)
    bad_tick_count = int(np.sum(bad_tick_mask))
    halt_proxy_count = int(np.sum(halt_proxy)) if halt_proxy.size > 0 else 0

    ib_minutes = expected_nominal[expected_nominal < int(session_open_minute) + int(_IB_WINDOW_MINUTES)]
    ib_missing = bool(observed_n == 0 or np.intersect1d(observed_minutes, ib_minutes, assume_unique=False).size == 0)

    reject_reasons: set[str] = set()
    degrade_reasons: set[str] = set()

    if observed_n == 0:
        reject_reasons.add("NO_SESSION_BARS")
    if missing_pct_effective > float(_MISSING_REJECT_PCT) and (not short_session_inferred):
        reject_reasons.add("MISSING_OVER_20_PCT")
    if nan_ohlcv_count > 0:
        reject_reasons.add("NAN_OHLCV")
    if invalid_ohlc_count > 0:
        reject_reasons.add("MATH_IMPOSSIBLE_OHLC")
    if zero_volume_day:
        reject_reasons.add("TOTAL_VOLUME_ZERO")
    if bad_tick_count > 0:
        reject_reasons.add("BAD_TICK_EXTREME")

    if short_session_inferred:
        degrade_reasons.add("SHORT_SESSION_INFERRED")
    if gap_events > 0 and "MISSING_OVER_20_PCT" not in reject_reasons:
        degrade_reasons.add("MICRO_GAPS")
    if ib_missing:
        degrade_reasons.add("IB_WINDOW_MISSING")
    if volume_spike_count > 0:
        degrade_reasons.add("VOLUME_SPIKE")
    if duplicate_timestamp_count > 0:
        degrade_reasons.add("DUPLICATE_TIMESTAMP")
    if non_monotonic_count > 0:
        degrade_reasons.add("NON_MONOTONIC_TIMESTAMP")

    decision = dq_decide(reject_reasons=reject_reasons, degrade_reasons=degrade_reasons)

    dqs_base = 1.0
    dqs_gap_penalty = min(float(_GAP_EVENT_PENALTY) * float(gap_events), float(_GAP_EVENT_PENALTY_CAP))
    dqs_ib_penalty = float(_IB_PENALTY) if ib_missing else 0.0
    dqs_volume_penalty = float(_VOLUME_PENALTY) if volume_spike_count > 0 else 0.0
    dqs_final = float(np.clip(dqs_base - dqs_gap_penalty - dqs_ib_penalty - dqs_volume_penalty, 0.0, 1.0))
    if decision == DQ_REJECT:
        dqs_final = 0.0

    reasons = tuple(sorted((reject_reasons | degrade_reasons), key=str))

    max_abs_ret = float(np.nanmax(ret_arr)) if ret_arr.size > 0 and np.any(np.isfinite(ret_arr)) else 0.0
    max_dyn = float(np.nanmax(dyn_arr)) if dyn_arr.size > 0 and np.any(np.isfinite(dyn_arr)) else 0.0

    return DQDayReport(
        symbol=str(symbol),
        session_date=str(session_date),
        decision=str(decision),
        reason_codes=reasons,
        timeframe_min=int(timeframe_min),
        session_open_minute=int(session_open_minute),
        session_close_minute=int(session_close_minute),
        expected_bars_nominal=int(expected_nominal_n),
        expected_bars_effective=int(expected_effective_n),
        observed_bars=int(observed_n),
        missing_bars_nominal=int(missing_nominal_n),
        missing_bars_effective=int(missing_effective_n),
        missing_pct_nominal=float(missing_pct_nominal),
        missing_pct_effective=float(missing_pct_effective),
        gap_events=int(gap_events),
        duplicate_timestamp_count=int(duplicate_timestamp_count),
        non_monotonic_count=int(non_monotonic_count),
        nan_ohlcv_count=int(nan_ohlcv_count),
        invalid_ohlc_count=int(invalid_ohlc_count),
        zero_volume_day=bool(zero_volume_day),
        total_volume=float(total_volume),
        short_session_inferred=bool(short_session_inferred),
        observed_span_ratio=float(span_ratio),
        observed_span_coverage=float(span_coverage),
        ib_missing=bool(ib_missing),
        volume_spike_count=int(volume_spike_count),
        bad_tick_count=int(bad_tick_count),
        halt_proxy_count=int(halt_proxy_count),
        max_abs_ret_1bar=float(max_abs_ret),
        max_dynamic_bad_tick_threshold=float(max_dyn),
        dqs_base=float(dqs_base),
        dqs_gap_penalty=float(dqs_gap_penalty),
        dqs_ib_penalty=float(dqs_ib_penalty),
        dqs_volume_penalty=float(dqs_volume_penalty),
        dqs_final=float(dqs_final),
        cadence_day_min=int(cadence_day_min),
        cadence_day_stable=bool(cadence_day_stable),
        cadence_day_delta_count=int(cadence_day_delta_count),
        cadence_day_cv=float(cadence_day_cv) if np.isfinite(cadence_day_cv) else float("nan"),
        expected_bars_reference=int(expected_reference_n),
        missing_bars_reference=int(missing_reference_n),
        missing_pct_reference=float(missing_pct_reference),
        fill_minutes_effective=tuple(int(x) for x in missing_effective.tolist()),
    )


def dq_decide(
    report: DQDayReport | None = None,
    reject_reasons: set[str] | None = None,
    degrade_reasons: set[str] | None = None,
) -> str:
    if report is not None:
        if report.decision in {DQ_ACCEPT, DQ_DEGRADE, DQ_REJECT}:
            return str(report.decision)
        raise RuntimeError(f"Unknown DQ decision value: {report.decision!r}")
    if reject_reasons is not None and len(reject_reasons) > 0:
        return DQ_REJECT
    if degrade_reasons is not None and len(degrade_reasons) > 0:
        return DQ_DEGRADE
    return DQ_ACCEPT


def dq_validate(
    df: Any,
    symbol: str,
    tz_name: str,
    session_open_minute: int,
    session_close_minute: int,
    timeframe_min: int | None,
) -> list[DQDayReport]:
    pdx = _require_pandas()
    work = _canonicalize_input_df(df, tz_name=tz_name)
    tfm = _infer_timeframe_min(work, fallback=timeframe_min)
    if work.empty:
        return []

    out: list[DQDayReport] = []
    grouped = work.groupby("session_date", sort=True)
    for session_date in sorted(grouped.groups.keys()):
        day = grouped.get_group(session_date).copy()
        rep = _session_report(
            symbol=symbol,
            session_date=str(session_date),
            day_raw=day,
            tz_name=tz_name,
            session_open_minute=int(session_open_minute),
            session_close_minute=int(session_close_minute),
            timeframe_min=int(tfm),
        )
        out.append(rep)
    return out


def dq_apply(
    df: Any,
    reports: list[DQDayReport],
    tz_name: str,
) -> tuple[Any, list[DQDayReport], Any]:
    pdx = _require_pandas()
    work = _canonicalize_input_df(df, tz_name=tz_name)
    cols = ["open", "high", "low", "close", "volume"]

    if work.empty:
        empty_idx = pdx.DatetimeIndex([], tz="UTC")
        repaired = pdx.DataFrame(columns=cols + ["dqs_day"], index=empty_idx)
        bar_flags = pdx.DataFrame(columns=["timestamp", "symbol", "dq_filled_bar", "dq_issue_flags", "dqs_day"])
        return repaired, list(reports), bar_flags

    report_by_day = {r.session_date: r for r in reports}

    repaired_chunks: list[Any] = []
    flag_chunks: list[Any] = []

    for session_date in sorted(work["session_date"].drop_duplicates().astype(str).tolist()):
        day_raw = work.loc[work["session_date"] == session_date].copy()
        day_raw = day_raw.sort_values(["timestamp", "_seq"], kind="mergesort")
        day_raw = day_raw.drop_duplicates(subset=["timestamp"], keep="last")
        rep = report_by_day.get(session_date)
        if rep is None:
            rep = DQDayReport(
                symbol="",
                session_date=session_date,
                decision=DQ_ACCEPT,
                reason_codes=(),
                timeframe_min=1,
                session_open_minute=570,
                session_close_minute=960,
                expected_bars_nominal=0,
                expected_bars_effective=0,
                observed_bars=0,
                missing_bars_nominal=0,
                missing_bars_effective=0,
                missing_pct_nominal=0.0,
                missing_pct_effective=0.0,
                gap_events=0,
                duplicate_timestamp_count=0,
                non_monotonic_count=0,
                nan_ohlcv_count=0,
                invalid_ohlc_count=0,
                zero_volume_day=False,
                total_volume=0.0,
                short_session_inferred=False,
                observed_span_ratio=0.0,
                observed_span_coverage=0.0,
                ib_missing=False,
                volume_spike_count=0,
                bad_tick_count=0,
                halt_proxy_count=0,
                max_abs_ret_1bar=0.0,
                max_dynamic_bad_tick_threshold=0.0,
                dqs_base=1.0,
                dqs_gap_penalty=0.0,
                dqs_ib_penalty=0.0,
                dqs_volume_penalty=0.0,
                dqs_final=1.0,
                cadence_day_min=1,
                cadence_day_stable=False,
                cadence_day_delta_count=0,
                cadence_day_cv=float("nan"),
                expected_bars_reference=0,
                missing_bars_reference=0,
                missing_pct_reference=0.0,
                fill_minutes_effective=(),
            )

        if rep.decision == DQ_REJECT:
            continue

        day_raw["timestamp_local"] = day_raw["timestamp"].dt.tz_convert(tz_name)
        day_raw["minute_of_day"] = (
            day_raw["timestamp_local"].dt.hour.astype(np.int64) * 60
            + day_raw["timestamp_local"].dt.minute.astype(np.int64)
        )

        session_mask = (day_raw["minute_of_day"] >= int(rep.session_open_minute)) & (
            day_raw["minute_of_day"] <= int(rep.session_close_minute)
        )

        non_session = day_raw.loc[~session_mask, ["timestamp"] + cols].copy()
        session_obs = day_raw.loc[session_mask, ["timestamp", "minute_of_day"] + cols].copy()

        day_local = pdx.Timestamp(str(session_date), tz=tz_name)
        expected_ts_local = [day_local + pdx.Timedelta(minutes=int(m)) for m in rep.fill_minutes_effective]
        missing_expected_utc = pdx.DatetimeIndex(expected_ts_local).tz_convert("UTC") if expected_ts_local else pdx.DatetimeIndex([], tz="UTC")

        # Build full session grid as observed + missing points in deterministic order.
        all_session_ts = np.unique(
            np.concatenate(
                [
                    session_obs["timestamp"].astype("int64").to_numpy(dtype=np.int64),
                    missing_expected_utc.asi8.astype(np.int64),
                ]
            )
        )
        all_session_ts = np.sort(all_session_ts, kind="mergesort")
        session_grid = pdx.DataFrame({"timestamp": pdx.to_datetime(all_session_ts, utc=True)})
        session_grid = session_grid.merge(session_obs[["timestamp"] + cols], on="timestamp", how="left", sort=True)

        close_arr = session_grid["close"].to_numpy(dtype=np.float64)
        open_arr = session_grid["open"].to_numpy(dtype=np.float64)
        high_arr = session_grid["high"].to_numpy(dtype=np.float64)
        low_arr = session_grid["low"].to_numpy(dtype=np.float64)
        vol_arr = session_grid["volume"].to_numpy(dtype=np.float64)

        finite_row = (
            np.isfinite(open_arr)
            & np.isfinite(high_arr)
            & np.isfinite(low_arr)
            & np.isfinite(close_arr)
            & np.isfinite(vol_arr)
        )
        first_valid = int(np.flatnonzero(finite_row)[0]) if np.any(finite_row) else -1

        filled = np.zeros(session_grid.shape[0], dtype=bool)
        issue_flags = np.zeros(session_grid.shape[0], dtype=np.int64)

        for i in range(session_grid.shape[0]):
            if finite_row[i]:
                continue
            if first_valid < 0 or i <= first_valid:
                continue
            prev_close = close_arr[i - 1]
            if not np.isfinite(prev_close):
                continue
            open_arr[i] = prev_close
            high_arr[i] = prev_close
            low_arr[i] = prev_close
            close_arr[i] = prev_close
            vol_arr[i] = 0.0
            filled[i] = True
            issue_flags[i] |= int(DQ_ISSUE_FILLED_BAR)

        session_grid["open"] = open_arr
        session_grid["high"] = high_arr
        session_grid["low"] = low_arr
        session_grid["close"] = close_arr
        session_grid["volume"] = vol_arr

        finite_after = (
            np.isfinite(open_arr)
            & np.isfinite(high_arr)
            & np.isfinite(low_arr)
            & np.isfinite(close_arr)
            & np.isfinite(vol_arr)
        )
        structural_ok = (
            (high_arr >= low_arr)
            & (high_arr >= open_arr)
            & (high_arr >= close_arr)
            & (low_arr <= open_arr)
            & (low_arr <= close_arr)
            & (vol_arr >= 0.0)
        )
        keep_rows = finite_after & structural_ok
        session_grid = session_grid.loc[keep_rows, ["timestamp"] + cols].copy()
        filled = filled[keep_rows]
        issue_flags = issue_flags[keep_rows]

        if rep.gap_events > 0:
            issue_flags |= int(DQ_ISSUE_GAP_EVENT)
        if rep.ib_missing:
            issue_flags |= int(DQ_ISSUE_IB_MISSING)
        if rep.volume_spike_count > 0:
            issue_flags |= int(DQ_ISSUE_VOLUME_SPIKE)
        if rep.short_session_inferred:
            issue_flags |= int(DQ_ISSUE_SHORT_SESSION)
        if rep.duplicate_timestamp_count > 0:
            issue_flags |= int(DQ_ISSUE_DUPLICATE_TS)
        if rep.non_monotonic_count > 0:
            issue_flags |= int(DQ_ISSUE_NON_MONOTONIC)

        day_out = pdx.concat([non_session[["timestamp"] + cols], session_grid], axis=0, ignore_index=True)
        day_out = day_out.sort_values(["timestamp"], kind="mergesort")
        day_out = day_out.drop_duplicates(subset=["timestamp"], keep="last")
        day_out["dqs_day"] = float(rep.dqs_final)

        # Purge any residual non-finite rows for fail-closed state safety.
        day_out = day_out.replace([np.inf, -np.inf], np.nan)
        day_out = day_out.dropna(subset=cols)
        if day_out.empty:
            continue

        repaired_chunks.append(day_out)

        # Flag frame only for session rows kept after repair.
        if session_grid.shape[0] > 0:
            flags_df = pdx.DataFrame(
                {
                    "timestamp": session_grid["timestamp"].to_numpy(),
                    "symbol": np.repeat(rep.symbol, session_grid.shape[0]),
                    "dq_filled_bar": filled.astype(bool),
                    "dq_issue_flags": issue_flags.astype(np.int64),
                    "dqs_day": np.repeat(float(rep.dqs_final), session_grid.shape[0]).astype(np.float64),
                }
            )
            flag_chunks.append(flags_df)

    if repaired_chunks:
        repaired = pdx.concat(repaired_chunks, axis=0, ignore_index=True)
        repaired = repaired.sort_values(["timestamp"], kind="mergesort")
        repaired = repaired.drop_duplicates(subset=["timestamp"], keep="last")
        repaired = repaired.set_index("timestamp")
    else:
        repaired = pdx.DataFrame(columns=cols + ["dqs_day"], index=pdx.DatetimeIndex([], tz="UTC"))

    if flag_chunks:
        bar_flags = pdx.concat(flag_chunks, axis=0, ignore_index=True)
        bar_flags = bar_flags.sort_values(["symbol", "timestamp"], kind="mergesort")
        bar_flags = bar_flags.reset_index(drop=True)
    else:
        bar_flags = pdx.DataFrame(columns=["timestamp", "symbol", "dq_filled_bar", "dq_issue_flags", "dqs_day"])

    return repaired, list(reports), bar_flags
