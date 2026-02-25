"""
Weightiz Institutional Engine - Module 1 (Core Tensor Engine)
==============================================================

This module defines the deterministic tensor skeleton for a multi-asset
Weightiz research/execution engine. It contains:

1) Canonical state schema with pre-allocated tensors.
2) Vectorized UTC -> US/Eastern session clock construction (DST aware).
3) Multi-asset typed epsilon handling from per-asset tick sizes.
4) Portfolio and margin state tensors required for strict Zimtra-like constraints.
5) Hard fail-closed validation.

Design principles:
- No pandas/polars/vectorbt/backtrader in core path.
- No hidden alignment or auto-indexing behavior.
- Deterministic replay with digest hashing.
- Float64 for all core numeric tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional, Sequence, Tuple
import datetime as dt
import hashlib
import numpy as np
from zoneinfo import ZoneInfo


# -----------------------------------------------------------------------------
# Time constants
# -----------------------------------------------------------------------------

NS_PER_SEC: int = 1_000_000_000
NS_PER_MIN: int = 60 * NS_PER_SEC
NS_PER_HOUR: int = 60 * NS_PER_MIN
NS_PER_DAY: int = 24 * 60 * NS_PER_MIN


# -----------------------------------------------------------------------------
# Enumerations for stable tensor channel indices
# -----------------------------------------------------------------------------

class Phase(IntEnum):
    """Execution phase over the trading day."""

    WARMUP = 0
    LIVE = 1
    OVERNIGHT_SELECT = 2
    FLATTEN = 3


class ProfileStatIdx(IntEnum):
    """Channels for profile_stats[T, A, K]."""

    MU_PROF = 0
    SIGMA_PROF = 1
    SIGMA_EFF = 2
    D = 3
    DCLIP = 4
    A_AFFINITY = 5
    DELTA0 = 6
    DELTA_POC = 7
    DELTA_EFF = 8
    Z_DELTA = 9
    GBREAK = 10
    GREJECT = 11
    IPOC = 12
    IVAH = 13
    IVAL = 14
    N_FIELDS = 15


class ScoreIdx(IntEnum):
    """Channels for scores[T, A, S]."""

    SCORE_BO_LONG = 0
    SCORE_BO_SHORT = 1
    SCORE_REJECT = 2
    SCORE_REJ_LONG = 3
    SCORE_REJ_SHORT = 4
    N_FIELDS = 5


class OrderIdx(IntEnum):
    """Channels for orders[T, A, O]."""

    TARGET_QTY = 0
    LIMIT_PX = 1
    STOP_PX = 2
    TAKE_PX = 3
    MAX_SLIP_BPS = 4
    CONVICTION = 5
    N_FIELDS = 6


# -----------------------------------------------------------------------------
# Config / state dataclasses
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class EngineConfig:
    """
    Global configuration for Module 1 state initialization.

    tick_size is mandatory and asset-specific, shape (A,).
    """

    # Dimensions
    T: int
    A: int
    tick_size: np.ndarray
    B: int = 240

    # Grid
    x_min: float = -6.0
    dx: float = 0.05

    # Session logic (US equities style)
    rth_open_minute: int = 9 * 60 + 30
    warmup_minutes: int = 15
    flat_time_minute: int = 15 * 60 + 45
    gap_reset_minutes: float = 5.0

    # Epsilons: scalar + typed by tick_size
    eps_pdf: float = 1e-12
    eps_vol: float = 1e-12

    # Portfolio constraints
    initial_cash: float = 1_000_000.0
    intraday_leverage_max: float = 6.0
    overnight_leverage: float = 2.0
    overnight_positions_max: int = 1
    daily_loss_limit_abs: float = 50_000.0

    # Determinism
    seed: int = 17
    fail_on_nan: bool = True
    mode: str = "research"  # "research" | "sealed"
    timezone: str = "America/New_York"


@dataclass
class TypedEps:
    """Typed epsilon bundle used by downstream modules."""

    eps_pdf: float
    eps_vol: float
    eps_div: np.ndarray      # (A,), eps_div[a] = tick_size[a]
    eps_range: np.ndarray    # (A,), eps_range[a] = tick_size[a]


@dataclass
class TensorState:
    """
    Canonical deterministic state for Module 1 and downstream modules.
    """

    symbols: Tuple[str, ...]
    cfg: EngineConfig
    eps: TypedEps

    # Clock / timeline tensors (T,)
    ts_ns: np.ndarray
    minute_of_day: np.ndarray
    tod: np.ndarray
    session_id: np.ndarray
    gap_min: np.ndarray
    reset_flag: np.ndarray
    phase: np.ndarray

    # Static profile grid (B,)
    x_grid: np.ndarray

    # Input market tensors (T, A)
    open_px: np.ndarray
    high_px: np.ndarray
    low_px: np.ndarray
    close_px: np.ndarray
    volume: np.ndarray
    rvol: np.ndarray
    atr_floor: np.ndarray
    bar_valid: np.ndarray

    # Core profile tensors (T, A, B)
    vp: np.ndarray
    vp_delta: np.ndarray

    # Derived outputs
    profile_stats: np.ndarray
    scores: np.ndarray

    # Orders / positions
    orders: np.ndarray
    order_side: np.ndarray
    order_flags: np.ndarray
    position_qty: np.ndarray
    overnight_mask: np.ndarray

    # Portfolio / margin state (T,)
    available_cash: np.ndarray
    equity: np.ndarray
    margin_used: np.ndarray
    buying_power: np.ndarray
    realized_pnl: np.ndarray
    unrealized_pnl: np.ndarray
    daily_loss: np.ndarray
    daily_loss_breach_flag: np.ndarray
    leverage_limit: np.ndarray


# -----------------------------------------------------------------------------
# Deterministic helper checks
# -----------------------------------------------------------------------------

def _assert_shape(name: str, arr: np.ndarray, expected: Tuple[int, ...]) -> None:
    if arr.shape != expected:
        raise RuntimeError(f"{name} shape mismatch: got {arr.shape}, expected {expected}")


def _assert_finite(name: str, arr: np.ndarray) -> None:
    if not np.all(np.isfinite(arr)):
        bad = np.argwhere(~np.isfinite(arr))[:8]
        raise RuntimeError(f"{name} contains non-finite values at indices {bad.tolist()}")


def _assert_monotonic_ts_ns(ts_ns: np.ndarray) -> None:
    if ts_ns.dtype != np.int64:
        raise RuntimeError(f"ts_ns must be int64, got {ts_ns.dtype}")
    if ts_ns.ndim != 1:
        raise RuntimeError(f"ts_ns must be 1D, got ndim={ts_ns.ndim}")
    diff = np.diff(ts_ns)
    if np.any(diff <= 0):
        idx = int(np.where(diff <= 0)[0][0])
        raise RuntimeError(f"Non-monotonic timestamps at idx={idx}, delta_ns={int(diff[idx])}")


# -----------------------------------------------------------------------------
# Config normalization and typed eps construction
# -----------------------------------------------------------------------------

def _validate_config(cfg: EngineConfig, symbols: Sequence[str]) -> np.ndarray:
    if cfg.T <= 0 or cfg.A <= 0 or cfg.B <= 2:
        raise RuntimeError("Invalid dimensions in EngineConfig")
    if len(symbols) != cfg.A:
        raise RuntimeError(f"symbols length {len(symbols)} must match A={cfg.A}")
    if cfg.dx <= 0:
        raise RuntimeError("dx must be > 0")
    if cfg.initial_cash <= 0:
        raise RuntimeError("initial_cash must be > 0")
    if cfg.intraday_leverage_max <= 0 or cfg.overnight_leverage <= 0:
        raise RuntimeError("leverage limits must be > 0")
    if cfg.overnight_positions_max != 1:
        raise RuntimeError("Strict rule: overnight_positions_max must be exactly 1")
    if cfg.daily_loss_limit_abs <= 0:
        raise RuntimeError("daily_loss_limit_abs must be > 0")
    mode = str(cfg.mode).strip().lower()
    if mode not in {"research", "sealed"}:
        raise RuntimeError(f"engine.mode must be 'research' or 'sealed', got {cfg.mode!r}")

    tick = np.asarray(cfg.tick_size, dtype=np.float64)
    _assert_shape("tick_size", tick, (cfg.A,))
    _assert_finite("tick_size", tick)
    if np.any(tick <= 0):
        idx = int(np.where(tick <= 0)[0][0])
        raise RuntimeError(f"tick_size[{idx}] must be > 0")
    return tick


def _build_typed_eps(cfg: EngineConfig, tick_size: np.ndarray) -> TypedEps:
    return TypedEps(
        eps_pdf=float(cfg.eps_pdf),
        eps_vol=float(cfg.eps_vol),
        eps_div=tick_size.astype(np.float64, copy=True),
        eps_range=tick_size.astype(np.float64, copy=True),
    )


# -----------------------------------------------------------------------------
# Grid initialization
# -----------------------------------------------------------------------------

def build_x_grid(cfg: EngineConfig) -> np.ndarray:
    """
    x_i = x_min + i * dx, i=0..B-1
    """
    x_grid = cfg.x_min + cfg.dx * np.arange(cfg.B, dtype=np.float64)
    _assert_shape("x_grid", x_grid, (cfg.B,))
    _assert_finite("x_grid", x_grid)

    d = np.diff(x_grid)
    if not np.all(d > 0):
        raise RuntimeError("x_grid must be strictly increasing")
    # Float64-safe spacing validation for decimal steps (e.g., 0.05).
    # Using allclose avoids false positives from IEEE754 representation.
    if not np.allclose(d, cfg.dx, rtol=0.0, atol=1e-12):
        raise RuntimeError("x_grid spacing drift detected")
    return x_grid


def _offset_seconds_at_ns(ns: int, tz_local: ZoneInfo, tz_utc: ZoneInfo) -> int:
    sec = int(ns // NS_PER_SEC)
    nsec = int(ns % NS_PER_SEC)
    dt_utc = dt.datetime.fromtimestamp(sec, tz=tz_utc).replace(microsecond=int(nsec // 1000))
    offset = dt_utc.astimezone(tz_local).utcoffset()
    if offset is None:
        raise RuntimeError(f"Timezone offset unavailable for ns={ns}")
    return int(offset.total_seconds())


def _derive_offset_segments_utc(
    ts_ns: np.ndarray,
    tz_local: ZoneInfo,
    tz_utc: ZoneInfo,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Derive UTC offset segments [start_ns, next_start_ns) for the timestamp domain.
    Sampling is hourly with deterministic transition refinement by binary search.
    """
    _assert_monotonic_ts_ns(ts_ns)
    if ts_ns.size == 0:
        raise RuntimeError("Cannot derive offset segments for empty ts_ns")

    start_day = (int(ts_ns[0]) // NS_PER_DAY) * NS_PER_DAY
    end_day = ((int(ts_ns[-1]) // NS_PER_DAY) + 1) * NS_PER_DAY
    pad = 48 * NS_PER_HOUR
    scan_start = start_day - pad
    scan_end = end_day + pad

    sample_ns = np.arange(scan_start, scan_end + NS_PER_HOUR, NS_PER_HOUR, dtype=np.int64)
    sample_offsets = np.empty(sample_ns.shape[0], dtype=np.int32)
    for i in range(sample_ns.shape[0]):
        sample_offsets[i] = np.int32(_offset_seconds_at_ns(int(sample_ns[i]), tz_local, tz_utc))

    change_idx = np.flatnonzero(sample_offsets[1:] != sample_offsets[:-1])
    seg_starts: list[int] = [int(scan_start)]
    seg_offsets: list[int] = [int(sample_offsets[0])]
    for j in change_idx.tolist():
        left_ns = int(sample_ns[j])
        right_ns = int(sample_ns[j + 1])
        left_offset = int(sample_offsets[j])
        right_offset = int(sample_offsets[j + 1])

        lo = left_ns
        hi = right_ns
        while hi - lo > 1:
            mid = lo + ((hi - lo) // 2)
            if _offset_seconds_at_ns(mid, tz_local, tz_utc) == left_offset:
                lo = mid
            else:
                hi = mid
        transition_ns = hi

        if seg_starts[-1] == transition_ns:
            seg_offsets[-1] = right_offset
        else:
            seg_starts.append(transition_ns)
            seg_offsets.append(right_offset)

    return (
        np.asarray(seg_starts, dtype=np.int64),
        np.asarray(seg_offsets, dtype=np.int32),
    )


def _compute_phase(minute_of_day: np.ndarray, tod: np.ndarray, cfg: EngineConfig) -> np.ndarray:
    phase = np.full(minute_of_day.shape[0], np.int8(Phase.WARMUP), dtype=np.int8)
    is_live = (tod >= int(cfg.warmup_minutes)) & (minute_of_day < int(cfg.flat_time_minute))
    phase[is_live] = np.int8(Phase.LIVE)

    is_select = (minute_of_day == int(cfg.flat_time_minute)) & (tod >= int(cfg.warmup_minutes))
    phase[is_select] = np.int8(Phase.OVERNIGHT_SELECT)

    is_flatten = minute_of_day > int(cfg.flat_time_minute)
    phase[is_flatten] = np.int8(Phase.FLATTEN)
    return phase


def _build_session_clock_reference(
    ts_ns: np.ndarray,
    cfg: EngineConfig,
    tz_name: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Reference DST-safe clock path using per-row zoneinfo conversion.
    Kept for audit and equivalence testing against the fast path.
    """
    _assert_monotonic_ts_ns(ts_ns)
    T = ts_ns.shape[0]

    tz_local = ZoneInfo(str(tz_name) if tz_name is not None else str(cfg.timezone))
    tz_utc = ZoneInfo("UTC")

    minute_of_day = np.empty(T, dtype=np.int16)
    local_day_index = np.empty(T, dtype=np.int64)
    for i in range(T):
        ns = int(ts_ns[i])
        sec = ns // NS_PER_SEC
        nsec = ns % NS_PER_SEC
        dt_utc = dt.datetime.fromtimestamp(sec, tz=tz_utc).replace(microsecond=int(nsec // 1000))
        dt_loc = dt_utc.astimezone(tz_local)
        minute_of_day[i] = np.int16(int(dt_loc.hour) * 60 + int(dt_loc.minute))
        local_day_index[i] = np.int64(int(dt_loc.date().toordinal()))

    tod = (minute_of_day.astype(np.int32) - int(cfg.rth_open_minute)).astype(np.int16)
    session_change = np.empty(T, dtype=bool)
    session_change[0] = True
    session_change[1:] = local_day_index[1:] != local_day_index[:-1]
    session_id = np.cumsum(session_change, dtype=np.int64) - 1

    gap_min = np.zeros(T, dtype=np.float64)
    gap_min[1:] = (ts_ns[1:] - ts_ns[:-1]) / float(NS_PER_MIN)
    reset_flag = ((gap_min > float(cfg.gap_reset_minutes)) | session_change).astype(np.int8)
    reset_flag[0] = np.int8(1)

    phase = _compute_phase(minute_of_day, tod, cfg)
    return {
        "minute_of_day": minute_of_day,
        "tod": tod,
        "session_id": session_id,
        "gap_min": gap_min,
        "reset_flag": reset_flag,
        "phase": phase,
    }


def build_session_clock_vectorized(
    ts_ns: np.ndarray,
    cfg: EngineConfig,
    tz_name: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Fast DST-safe clock path:
    - derives UTC offset segments deterministically from zoneinfo,
    - computes local minute/day features using numpy arithmetic.
    """
    _assert_monotonic_ts_ns(ts_ns)
    T = ts_ns.shape[0]

    tz_local = ZoneInfo(str(tz_name) if tz_name is not None else str(cfg.timezone))
    tz_utc = ZoneInfo("UTC")

    seg_starts, seg_offsets = _derive_offset_segments_utc(ts_ns, tz_local, tz_utc)
    seg_idx = np.searchsorted(seg_starts, ts_ns, side="right") - 1
    if np.any(seg_idx < 0):
        raise RuntimeError("Offset segment assignment failed for clock build")

    offset_sec = seg_offsets[seg_idx].astype(np.int64)
    local_ns = ts_ns + offset_sec * np.int64(NS_PER_SEC)

    local_minute_index = np.floor_divide(local_ns, np.int64(NS_PER_MIN))
    minute_of_day = np.remainder(local_minute_index, np.int64(24 * 60)).astype(np.int16)
    tod = (minute_of_day.astype(np.int32) - int(cfg.rth_open_minute)).astype(np.int16)

    local_day_index = np.floor_divide(local_ns, np.int64(NS_PER_DAY)).astype(np.int64)
    session_change = np.empty(T, dtype=bool)
    session_change[0] = True
    session_change[1:] = local_day_index[1:] != local_day_index[:-1]
    session_id = np.cumsum(session_change, dtype=np.int64) - 1

    gap_min = np.zeros(T, dtype=np.float64)
    gap_min[1:] = (ts_ns[1:] - ts_ns[:-1]) / float(NS_PER_MIN)
    reset_flag = ((gap_min > float(cfg.gap_reset_minutes)) | session_change).astype(np.int8)
    reset_flag[0] = np.int8(1)

    phase = _compute_phase(minute_of_day, tod, cfg)
    return {
        "minute_of_day": minute_of_day,
        "tod": tod,
        "session_id": session_id,
        "gap_min": gap_min,
        "reset_flag": reset_flag,
        "phase": phase,
    }


def _validate_clock_override(clock: Dict[str, np.ndarray], T: int) -> None:
    expected = ("minute_of_day", "tod", "session_id", "gap_min", "reset_flag", "phase")
    if not isinstance(clock, dict):
        raise RuntimeError(f"clock_override must be dict[str, np.ndarray], got {type(clock)!r}")

    expected_keys = set(expected)
    got_keys = set(clock.keys())
    if got_keys != expected_keys:
        missing = sorted(expected_keys - got_keys)
        extra = sorted(got_keys - expected_keys)
        raise RuntimeError(
            f"clock_override keys mismatch: missing={missing}, extra={extra}, expected={list(expected)}"
        )

    expected_dtype = {
        "minute_of_day": np.dtype(np.int16),
        "tod": np.dtype(np.int16),
        "session_id": np.dtype(np.int64),
        "gap_min": np.dtype(np.float64),
        "reset_flag": np.dtype(np.int8),
        "phase": np.dtype(np.int8),
    }
    for key in expected:
        arr = clock[key]
        if not isinstance(arr, np.ndarray):
            raise RuntimeError(f"clock_override[{key!r}] must be np.ndarray, got {type(arr)!r}")
        if arr.shape != (T,):
            raise RuntimeError(
                f"clock_override[{key!r}] shape mismatch: got {arr.shape}, expected {(T,)}"
            )
        if arr.dtype != expected_dtype[key]:
            raise RuntimeError(
                f"clock_override[{key!r}] dtype mismatch: got {arr.dtype}, expected {expected_dtype[key]}"
            )

    minute_of_day = clock["minute_of_day"]
    if np.any((minute_of_day < 0) | (minute_of_day > 1439)):
        idx = int(np.where((minute_of_day < 0) | (minute_of_day > 1439))[0][0])
        raise RuntimeError(
            f"clock_override['minute_of_day'] out of range at t={idx}: {int(minute_of_day[idx])}"
        )

    session_id = clock["session_id"]
    if int(session_id[0]) != 0:
        raise RuntimeError(f"clock_override['session_id'][0] must be 0, got {int(session_id[0])}")
    sid_diff = np.diff(session_id)
    if np.any(sid_diff < 0):
        idx = int(np.where(sid_diff < 0)[0][0])
        raise RuntimeError(
            f"clock_override['session_id'] must be nondecreasing: "
            f"t={idx}->{idx + 1}, values={int(session_id[idx])}->{int(session_id[idx + 1])}"
        )

    gap_min = clock["gap_min"]
    if not np.all(np.isfinite(gap_min)):
        idx = int(np.where(~np.isfinite(gap_min))[0][0])
        raise RuntimeError(f"clock_override['gap_min'] non-finite at t={idx}: {gap_min[idx]!r}")
    if float(gap_min[0]) != 0.0:
        raise RuntimeError(f"clock_override['gap_min'][0] must be 0.0, got {float(gap_min[0])}")
    if np.any(gap_min < 0.0):
        idx = int(np.where(gap_min < 0.0)[0][0])
        raise RuntimeError(
            f"clock_override['gap_min'] must be non-negative at t={idx}: {float(gap_min[idx])}"
        )

    reset_flag = clock["reset_flag"]
    valid_reset = np.isin(reset_flag, np.array([0, 1], dtype=np.int8))
    if not np.all(valid_reset):
        idx = int(np.where(~valid_reset)[0][0])
        raise RuntimeError(
            f"clock_override['reset_flag'] must be in {{0,1}} at t={idx}: {int(reset_flag[idx])}"
        )
    if int(reset_flag[0]) != 1:
        raise RuntimeError(f"clock_override['reset_flag'][0] must be 1, got {int(reset_flag[0])}")

    phase = clock["phase"]
    valid_phase_values = np.array(
        [np.int8(Phase.WARMUP), np.int8(Phase.LIVE), np.int8(Phase.OVERNIGHT_SELECT), np.int8(Phase.FLATTEN)],
        dtype=np.int8,
    )
    valid_phase = np.isin(phase, valid_phase_values)
    if not np.all(valid_phase):
        idx = int(np.where(~valid_phase)[0][0])
        raise RuntimeError(f"clock_override['phase'] invalid code at t={idx}: {int(phase[idx])}")


# -----------------------------------------------------------------------------
# Portfolio state initialization
# -----------------------------------------------------------------------------

def _init_portfolio_vectors(cfg: EngineConfig, phase: np.ndarray) -> Dict[str, np.ndarray]:
    T = cfg.T

    available_cash = np.full(T, cfg.initial_cash, dtype=np.float64)
    equity = np.full(T, cfg.initial_cash, dtype=np.float64)
    margin_used = np.zeros(T, dtype=np.float64)
    realized_pnl = np.zeros(T, dtype=np.float64)
    unrealized_pnl = np.zeros(T, dtype=np.float64)
    daily_loss = np.zeros(T, dtype=np.float64)
    daily_loss_breach_flag = np.zeros(T, dtype=np.int8)

    # Module 1 uses an intraday enforcement limit across ticks.
    # Overnight leverage is enforced explicitly at session close in
    # _validate_portfolio_constraints(), which avoids false fail-close during
    # the liquidation window immediately after FLATTEN transition.
    leverage_limit = np.full(T, float(cfg.intraday_leverage_max), dtype=np.float64)

    buying_power = equity * leverage_limit - margin_used

    return {
        "available_cash": available_cash,
        "equity": equity,
        "margin_used": margin_used,
        "buying_power": buying_power,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "daily_loss": daily_loss,
        "daily_loss_breach_flag": daily_loss_breach_flag,
        "leverage_limit": leverage_limit,
    }


# -----------------------------------------------------------------------------
# State pre-allocation
# -----------------------------------------------------------------------------

def preallocate_state(
    ts_ns: np.ndarray,
    cfg: EngineConfig,
    symbols: Sequence[str],
    clock_override: Optional[Dict[str, np.ndarray]] = None,
) -> TensorState:
    tick_size = _validate_config(cfg, symbols)
    eps = _build_typed_eps(cfg, tick_size)

    ts_ns = np.ascontiguousarray(np.asarray(ts_ns, dtype=np.int64))
    _assert_shape("ts_ns", ts_ns, (cfg.T,))

    x_grid = build_x_grid(cfg)
    if clock_override is not None:
        _validate_clock_override(clock_override, cfg.T)
        clk = clock_override
    else:
        clk = build_session_clock_vectorized(ts_ns, cfg)

    shape_ta = (cfg.T, cfg.A)
    shape_tab = (cfg.T, cfg.A, cfg.B)

    # Input tensors (T, A)
    open_px = np.full(shape_ta, np.nan, dtype=np.float64)
    high_px = np.full(shape_ta, np.nan, dtype=np.float64)
    low_px = np.full(shape_ta, np.nan, dtype=np.float64)
    close_px = np.full(shape_ta, np.nan, dtype=np.float64)
    volume = np.full(shape_ta, np.nan, dtype=np.float64)
    rvol = np.full(shape_ta, np.nan, dtype=np.float64)
    atr_floor = np.full(shape_ta, np.nan, dtype=np.float64)
    bar_valid = np.zeros(shape_ta, dtype=bool)

    # Core tensors (T, A, B)
    vp = np.zeros(shape_tab, dtype=np.float64)
    vp_delta = np.zeros(shape_tab, dtype=np.float64)

    # Outputs
    profile_stats = np.full(
        (cfg.T, cfg.A, int(ProfileStatIdx.N_FIELDS)),
        np.nan,
        dtype=np.float64,
    )
    scores = np.full(
        (cfg.T, cfg.A, int(ScoreIdx.N_FIELDS)),
        np.nan,
        dtype=np.float64,
    )

    # Orders / positions
    orders = np.full(
        (cfg.T, cfg.A, int(OrderIdx.N_FIELDS)),
        np.nan,
        dtype=np.float64,
    )
    order_side = np.zeros(shape_ta, dtype=np.int8)
    order_flags = np.zeros(shape_ta, dtype=np.uint16)
    position_qty = np.zeros(shape_ta, dtype=np.float64)
    overnight_mask = np.zeros(shape_ta, dtype=np.int8)

    pf = _init_portfolio_vectors(cfg, clk["phase"])

    state = TensorState(
        symbols=tuple(symbols),
        cfg=cfg,
        eps=eps,
        ts_ns=ts_ns.copy(),
        minute_of_day=clk["minute_of_day"],
        tod=clk["tod"],
        session_id=clk["session_id"],
        gap_min=clk["gap_min"],
        reset_flag=clk["reset_flag"],
        phase=clk["phase"],
        x_grid=x_grid,
        open_px=open_px,
        high_px=high_px,
        low_px=low_px,
        close_px=close_px,
        volume=volume,
        rvol=rvol,
        atr_floor=atr_floor,
        bar_valid=bar_valid,
        vp=vp,
        vp_delta=vp_delta,
        profile_stats=profile_stats,
        scores=scores,
        orders=orders,
        order_side=order_side,
        order_flags=order_flags,
        position_qty=position_qty,
        overnight_mask=overnight_mask,
        available_cash=pf["available_cash"],
        equity=pf["equity"],
        margin_used=pf["margin_used"],
        buying_power=pf["buying_power"],
        realized_pnl=pf["realized_pnl"],
        unrealized_pnl=pf["unrealized_pnl"],
        daily_loss=pf["daily_loss"],
        daily_loss_breach_flag=pf["daily_loss_breach_flag"],
        leverage_limit=pf["leverage_limit"],
    )

    validate_state_hard(state)
    return state


# -----------------------------------------------------------------------------
# Constraint and integrity validation
# -----------------------------------------------------------------------------

def _validate_portfolio_constraints(state: TensorState) -> None:
    cfg = state.cfg
    tol = 1e-10

    if np.any(state.margin_used < -tol):
        idx = int(np.where(state.margin_used < -tol)[0][0])
        raise RuntimeError(f"margin_used negative at t={idx}")

    # Intraday cap applies continuously across all ticks.
    max_margin_intraday = state.equity * float(cfg.intraday_leverage_max)
    if np.any(state.margin_used > max_margin_intraday + tol):
        idx = int(np.where(state.margin_used > max_margin_intraday + tol)[0][0])
        raise RuntimeError(
            f"Intraday leverage breach at t={idx}: "
            f"margin_used={state.margin_used[idx]:.6f}, max={max_margin_intraday[idx]:.6f}"
        )

    implied_bp = state.equity * state.leverage_limit - state.margin_used
    err = np.abs(implied_bp - state.buying_power)
    if np.max(err) > 1e-8:
        idx = int(np.argmax(err))
        raise RuntimeError(
            f"buying_power identity violated at t={idx}: "
            f"stored={state.buying_power[idx]:.6f}, implied={implied_bp[idx]:.6f}"
        )

    if np.any(state.buying_power < -tol):
        idx = int(np.where(state.buying_power < -tol)[0][0])
        raise RuntimeError(f"buying_power negative at t={idx}: {state.buying_power[idx]:.6f}")

    # Overnight cap must hold at end-of-session only (close liquidation complete).
    # This avoids a false fail-close exactly at FLATTEN transition, when orders are
    # still being unwound.
    T = state.session_id.shape[0]
    session_end = np.empty(T, dtype=bool)
    session_end[:-1] = state.session_id[:-1] != state.session_id[1:]
    session_end[-1] = True

    max_margin_overnight = state.equity * float(cfg.overnight_leverage)
    if np.any(state.margin_used[session_end] > max_margin_overnight[session_end] + tol):
        local_idx = int(np.where(state.margin_used[session_end] > max_margin_overnight[session_end] + tol)[0][0])
        idx = int(np.where(session_end)[0][local_idx])
        raise RuntimeError(
            f"Overnight leverage breach at session close t={idx}: "
            f"margin_used={state.margin_used[idx]:.6f}, max={max_margin_overnight[idx]:.6f}"
        )

    overnight_count = np.sum(state.overnight_mask.astype(np.int64), axis=1)
    if np.any(overnight_count > cfg.overnight_positions_max):
        idx = int(np.where(overnight_count > cfg.overnight_positions_max)[0][0])
        raise RuntimeError(
            f"Overnight count breach at t={idx}: count={int(overnight_count[idx])}, max={cfg.overnight_positions_max}"
        )

    expected_breach = (state.daily_loss >= cfg.daily_loss_limit_abs).astype(np.int8)
    if np.any(state.daily_loss_breach_flag != expected_breach):
        idx = int(np.where(state.daily_loss_breach_flag != expected_breach)[0][0])
        raise RuntimeError(
            f"daily_loss_breach_flag mismatch at t={idx}: "
            f"flag={int(state.daily_loss_breach_flag[idx])}, expected={int(expected_breach[idx])}"
        )


def validate_state_hard(state: TensorState) -> None:
    cfg = state.cfg
    T, A, B = cfg.T, cfg.A, cfg.B

    # Shapes
    _assert_shape("ts_ns", state.ts_ns, (T,))
    _assert_shape("minute_of_day", state.minute_of_day, (T,))
    _assert_shape("tod", state.tod, (T,))
    _assert_shape("session_id", state.session_id, (T,))
    _assert_shape("gap_min", state.gap_min, (T,))
    _assert_shape("reset_flag", state.reset_flag, (T,))
    _assert_shape("phase", state.phase, (T,))
    _assert_shape("x_grid", state.x_grid, (B,))
    _assert_shape("eps_div", state.eps.eps_div, (A,))
    _assert_shape("eps_range", state.eps.eps_range, (A,))

    _assert_shape("open_px", state.open_px, (T, A))
    _assert_shape("high_px", state.high_px, (T, A))
    _assert_shape("low_px", state.low_px, (T, A))
    _assert_shape("close_px", state.close_px, (T, A))
    _assert_shape("volume", state.volume, (T, A))
    _assert_shape("rvol", state.rvol, (T, A))
    _assert_shape("atr_floor", state.atr_floor, (T, A))
    _assert_shape("bar_valid", state.bar_valid, (T, A))

    _assert_shape("vp", state.vp, (T, A, B))
    _assert_shape("vp_delta", state.vp_delta, (T, A, B))
    _assert_shape("profile_stats", state.profile_stats, (T, A, int(ProfileStatIdx.N_FIELDS)))
    _assert_shape("scores", state.scores, (T, A, int(ScoreIdx.N_FIELDS)))
    _assert_shape("orders", state.orders, (T, A, int(OrderIdx.N_FIELDS)))
    _assert_shape("order_side", state.order_side, (T, A))
    _assert_shape("order_flags", state.order_flags, (T, A))
    _assert_shape("position_qty", state.position_qty, (T, A))
    _assert_shape("overnight_mask", state.overnight_mask, (T, A))

    _assert_shape("available_cash", state.available_cash, (T,))
    _assert_shape("equity", state.equity, (T,))
    _assert_shape("margin_used", state.margin_used, (T,))
    _assert_shape("buying_power", state.buying_power, (T,))
    _assert_shape("realized_pnl", state.realized_pnl, (T,))
    _assert_shape("unrealized_pnl", state.unrealized_pnl, (T,))
    _assert_shape("daily_loss", state.daily_loss, (T,))
    _assert_shape("daily_loss_breach_flag", state.daily_loss_breach_flag, (T,))
    _assert_shape("leverage_limit", state.leverage_limit, (T,))

    # Dtypes
    if state.ts_ns.dtype != np.int64:
        raise RuntimeError("ts_ns must be int64")
    if state.minute_of_day.dtype != np.int16:
        raise RuntimeError("minute_of_day must be int16")
    if state.tod.dtype != np.int16:
        raise RuntimeError("tod must be int16")
    if state.session_id.dtype != np.int64:
        raise RuntimeError("session_id must be int64")
    if state.gap_min.dtype != np.float64:
        raise RuntimeError("gap_min must be float64")
    if state.reset_flag.dtype != np.int8:
        raise RuntimeError("reset_flag must be int8")
    if state.phase.dtype != np.int8:
        raise RuntimeError("phase must be int8")
    if state.order_side.dtype != np.int8:
        raise RuntimeError("order_side must be int8")
    if state.order_flags.dtype != np.uint16:
        raise RuntimeError("order_flags must be uint16")

    # Clock / grid
    _assert_monotonic_ts_ns(state.ts_ns)
    _assert_finite("x_grid", state.x_grid)
    if not np.all(np.diff(state.x_grid) > 0):
        raise RuntimeError("x_grid must be strictly increasing")

    # Eps
    _assert_finite("eps_div", state.eps.eps_div)
    _assert_finite("eps_range", state.eps.eps_range)
    if np.any(state.eps.eps_div <= 0):
        raise RuntimeError("eps_div must be > 0")
    if np.any(state.eps.eps_range <= 0):
        raise RuntimeError("eps_range must be > 0")

    # Mandatory finite tensors
    for name, arr in [
        ("gap_min", state.gap_min),
        ("vp", state.vp),
        ("vp_delta", state.vp_delta),
        ("position_qty", state.position_qty),
        ("available_cash", state.available_cash),
        ("equity", state.equity),
        ("margin_used", state.margin_used),
        ("buying_power", state.buying_power),
        ("realized_pnl", state.realized_pnl),
        ("unrealized_pnl", state.unrealized_pnl),
        ("daily_loss", state.daily_loss),
        ("leverage_limit", state.leverage_limit),
    ]:
        _assert_finite(name, arr)

    if np.any((state.minute_of_day < 0) | (state.minute_of_day > 1439)):
        idx = int(np.where((state.minute_of_day < 0) | (state.minute_of_day > 1439))[0][0])
        raise RuntimeError(f"minute_of_day out of range at t={idx}: {int(state.minute_of_day[idx])}")

    if int(state.session_id[0]) != 0:
        raise RuntimeError(f"session_id[0] must be 0, got {int(state.session_id[0])}")
    sid_diff = np.diff(state.session_id)
    if np.any(sid_diff < 0):
        idx = int(np.where(sid_diff < 0)[0][0])
        raise RuntimeError(
            f"session_id must be nondecreasing: "
            f"t={idx}->{idx + 1}, values={int(state.session_id[idx])}->{int(state.session_id[idx + 1])}"
        )

    valid_reset_flag = np.isin(state.reset_flag, np.array([0, 1], dtype=np.int8))
    if not np.all(valid_reset_flag):
        idx = int(np.where(~valid_reset_flag)[0][0])
        raise RuntimeError(f"reset_flag must be in {{0,1}} at t={idx}: {int(state.reset_flag[idx])}")
    if int(state.reset_flag[0]) != 1:
        raise RuntimeError(f"reset_flag[0] must be 1, got {int(state.reset_flag[0])}")

    if float(state.gap_min[0]) != 0.0:
        raise RuntimeError(f"gap_min[0] must be 0.0, got {float(state.gap_min[0])}")
    if np.any(state.gap_min < 0.0):
        idx = int(np.where(state.gap_min < 0.0)[0][0])
        raise RuntimeError(f"gap_min must be non-negative at t={idx}: {float(state.gap_min[idx])}")

    expected_tod = state.minute_of_day.astype(np.int32) - int(cfg.rth_open_minute)
    tod_i32 = state.tod.astype(np.int32)
    if not np.array_equal(tod_i32, expected_tod):
        idx = int(np.where(tod_i32 != expected_tod)[0][0])
        raise RuntimeError(
            f"tod mismatch at t={idx}: got={int(tod_i32[idx])}, "
            f"expected={int(expected_tod[idx])}"
        )

    valid_phase = np.isin(
        state.phase,
        np.array([np.int8(Phase.WARMUP), np.int8(Phase.LIVE), np.int8(Phase.OVERNIGHT_SELECT), np.int8(Phase.FLATTEN)], dtype=np.int8),
    )
    if not np.all(valid_phase):
        idx = int(np.where(~valid_phase)[0][0])
        raise RuntimeError(f"Invalid phase code at t={idx}: {int(state.phase[idx])}")

    _validate_portfolio_constraints(state)


def validate_loaded_market_slice(state: TensorState, t_start: int, t_end: int) -> None:
    """
    Validate loaded OHLCV/rvol/atr values on [t_start, t_end) where bar_valid is True.
    """
    cfg = state.cfg
    if not (0 <= t_start < t_end <= cfg.T):
        raise RuntimeError("Invalid slice bounds")

    sl = slice(t_start, t_end)
    valid = state.bar_valid[sl]

    if not np.any(valid):
        return

    for name, arr in [
        ("open_px", state.open_px[sl][valid]),
        ("high_px", state.high_px[sl][valid]),
        ("low_px", state.low_px[sl][valid]),
        ("close_px", state.close_px[sl][valid]),
        ("volume", state.volume[sl][valid]),
        ("rvol", state.rvol[sl][valid]),
        ("atr_floor", state.atr_floor[sl][valid]),
    ]:
        _assert_finite(name, arr)

    h = state.high_px[sl][valid]
    l = state.low_px[sl][valid]
    o = state.open_px[sl][valid]
    c = state.close_px[sl][valid]
    v = state.volume[sl][valid]
    a = state.atr_floor[sl][valid]

    if np.any(h < l):
        raise RuntimeError("OHLC violation: high < low")
    if np.any(h < o):
        raise RuntimeError("OHLC violation: high < open")
    if np.any(h < c):
        raise RuntimeError("OHLC violation: high < close")
    if np.any(l > o):
        raise RuntimeError("OHLC violation: low > open")
    if np.any(l > c):
        raise RuntimeError("OHLC violation: low > close")
    if np.any(v < 0):
        raise RuntimeError("Volume violation: negative volume")
    if np.any(a <= 0):
        raise RuntimeError("ATR floor violation: atr_floor <= 0")


# -----------------------------------------------------------------------------
# Diagnostics and reproducibility helpers
# -----------------------------------------------------------------------------

def memory_report_bytes(state: TensorState) -> Dict[str, int]:
    arrays = {
        "ts_ns": state.ts_ns,
        "minute_of_day": state.minute_of_day,
        "tod": state.tod,
        "session_id": state.session_id,
        "gap_min": state.gap_min,
        "reset_flag": state.reset_flag,
        "phase": state.phase,
        "x_grid": state.x_grid,
        "eps_div": state.eps.eps_div,
        "eps_range": state.eps.eps_range,
        "open_px": state.open_px,
        "high_px": state.high_px,
        "low_px": state.low_px,
        "close_px": state.close_px,
        "volume": state.volume,
        "rvol": state.rvol,
        "atr_floor": state.atr_floor,
        "bar_valid": state.bar_valid,
        "vp": state.vp,
        "vp_delta": state.vp_delta,
        "profile_stats": state.profile_stats,
        "scores": state.scores,
        "orders": state.orders,
        "order_side": state.order_side,
        "order_flags": state.order_flags,
        "position_qty": state.position_qty,
        "overnight_mask": state.overnight_mask,
        "available_cash": state.available_cash,
        "equity": state.equity,
        "margin_used": state.margin_used,
        "buying_power": state.buying_power,
        "realized_pnl": state.realized_pnl,
        "unrealized_pnl": state.unrealized_pnl,
        "daily_loss": state.daily_loss,
        "daily_loss_breach_flag": state.daily_loss_breach_flag,
        "leverage_limit": state.leverage_limit,
    }
    out = {k: int(v.nbytes) for k, v in arrays.items()}
    out["TOTAL"] = int(sum(out.values()))
    return out


def deterministic_digest_sha256(state: TensorState) -> str:
    """
    Hash critical arrays for reproducibility checks.
    """
    h = hashlib.sha256()
    critical = [
        state.ts_ns,
        state.minute_of_day,
        state.tod,
        state.session_id,
        state.gap_min,
        state.reset_flag,
        state.phase,
        state.x_grid,
        state.eps.eps_div,
        state.eps.eps_range,
        state.vp,
        state.vp_delta,
        state.position_qty,
        state.overnight_mask,
        state.available_cash,
        state.equity,
        state.margin_used,
        state.buying_power,
        state.daily_loss,
        state.daily_loss_breach_flag,
    ]
    for arr in critical:
        h.update(np.ascontiguousarray(arr).view(np.uint8))
    return h.hexdigest()


# -----------------------------------------------------------------------------
# Example bootstrap (smoke run)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    T = 390
    A = 4
    symbols = ("SPY", "QQQ", "GLD", "TLT")
    tick_size = np.array([0.01, 0.01, 0.01, 0.01], dtype=np.float64)

    cfg = EngineConfig(
        T=T,
        A=A,
        B=240,
        tick_size=tick_size,
        seed=42,
        initial_cash=2_000_000.0,
        intraday_leverage_max=6.0,
        overnight_leverage=2.0,
        overnight_positions_max=1,
        daily_loss_limit_abs=75_000.0,
    )

    start_ns = np.datetime64("2024-01-02T14:31:00", "ns").astype(np.int64)
    ts_ns = start_ns + np.arange(T, dtype=np.int64) * np.int64(NS_PER_MIN)

    st = preallocate_state(ts_ns=ts_ns, cfg=cfg, symbols=symbols)
    rep = memory_report_bytes(st)

    print("MODULE1_OK")
    print("TOTAL_BYTES", rep["TOTAL"])
    print("DIGEST", deterministic_digest_sha256(st))
