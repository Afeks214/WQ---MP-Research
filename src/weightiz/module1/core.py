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

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple
import datetime as dt
import hashlib
import json
import os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from zoneinfo import ZoneInfo

from weightiz.shared.config.paths import resolve_repo_path


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
    tol = 0.0

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
# Feature engine (declarative, deterministic, NumPy-first)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureSpec:
    name: str
    windows: Tuple[int, ...]
    input_fields: Tuple[str, ...]
    dtype: str = "float64"
    dependencies: Tuple[str, ...] = ()


@dataclass(frozen=True)
class FeatureEngineConfig:
    tensor_backend: str = "ram"  # "ram" | "memmap"
    compute_backend: str = "numpy"  # "numpy" | "cupy"
    parallel_backend: str = "serial"  # "serial" | "process_pool"
    seed: int = 17
    cache_dir: str = "artifacts/feature_cache"
    artifacts_dir: str = "artifacts"
    memmap_path: str = "artifacts/feature_tensor.memmap"
    max_workers: int = 1
    ffill_gap_limit: int = 3
    mad_clip_k: float = 8.0
    drop_corrupted_rows: bool = False
    use_cache: bool = True


def _stable_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _ensure_feature_specs(feature_specs: Sequence[FeatureSpec]) -> list[FeatureSpec]:
    out: list[FeatureSpec] = []
    seen: set[str] = set()
    for spec in feature_specs:
        if not isinstance(spec, FeatureSpec):
            raise RuntimeError(f"Invalid feature spec type: {type(spec)!r}")
        name = str(spec.name).strip()
        if name == "":
            raise RuntimeError("Feature name must be non-empty")
        if name in seen:
            raise RuntimeError(f"Duplicate feature name in registry: {name}")
        seen.add(name)
        if str(spec.dtype).strip().lower() != "float64":
            raise RuntimeError(f"Feature '{name}' dtype must be float64")
        if len(spec.windows) == 0:
            raise RuntimeError(f"Feature '{name}' requires at least one window")
        if any(int(w) <= 0 for w in spec.windows):
            raise RuntimeError(f"Feature '{name}' windows must be > 0")
        out.append(
            FeatureSpec(
                name=name,
                windows=tuple(sorted(set(int(w) for w in spec.windows))),
                input_fields=tuple(str(x).strip().lower() for x in spec.input_fields if str(x).strip() != ""),
                dtype="float64",
                dependencies=tuple(sorted(set(str(x).strip() for x in spec.dependencies if str(x).strip() != ""))),
            )
        )
    return out


def _feature_specs_payload(feature_specs: Sequence[FeatureSpec]) -> list[dict[str, Any]]:
    specs = _ensure_feature_specs(feature_specs)
    payload: list[dict[str, Any]] = []
    for s in specs:
        payload.append(
            {
                "name": s.name,
                "windows": [int(w) for w in s.windows],
                "input_fields": [str(x) for x in s.input_fields],
                "dtype": "float64",
                "dependencies": [str(x) for x in s.dependencies],
            }
        )
    return payload


def feature_registry_hash(feature_specs: Sequence[FeatureSpec]) -> str:
    payload = _feature_specs_payload(feature_specs)
    return hashlib.sha256(_stable_dumps(payload).encode("utf-8")).hexdigest()


def _parse_csv_or_list(v: Any, *, field_name: str) -> tuple[str, ...]:
    if isinstance(v, str):
        parts = [p.strip() for p in v.split(",")]
    elif isinstance(v, (list, tuple)):
        parts = [str(x).strip() for x in v]
    else:
        raise RuntimeError(f"feature registry field '{field_name}' must be string or list")
    parts = [p for p in parts if p != ""]
    if len(parts) == 0:
        raise RuntimeError(f"feature registry field '{field_name}' cannot be empty")
    return tuple(parts)


def _parse_windows(v: Any) -> tuple[int, ...]:
    if isinstance(v, int):
        wins = [int(v)]
    elif isinstance(v, (list, tuple)):
        wins = [int(x) for x in v]
    else:
        raise RuntimeError("feature registry 'window/windows' must be int or list[int]")
    if len(wins) == 0:
        raise RuntimeError("feature registry windows cannot be empty")
    if any(int(w) <= 0 for w in wins):
        raise RuntimeError("feature registry windows must be > 0")
    return tuple(sorted(set(int(w) for w in wins)))


def load_feature_registry(path: str | os.PathLike[str]) -> list[FeatureSpec]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pyyaml is required to load feature registry") from exc

    p = resolve_repo_path(path)
    if not p.exists():
        raise RuntimeError(f"Feature registry file not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("Feature registry must be a YAML mapping")
    features = raw.get("features")
    if not isinstance(features, list) or len(features) == 0:
        raise RuntimeError("Feature registry must contain non-empty 'features' list")

    out: list[FeatureSpec] = []
    allowed = {"name", "window", "windows", "input", "input_fields", "dtype", "dependencies", "depends_on"}
    for i, item in enumerate(features):
        if not isinstance(item, dict):
            raise RuntimeError(f"features[{i}] must be a mapping")
        extra = sorted(set(item.keys()) - allowed)
        if extra:
            raise RuntimeError(f"features[{i}] contains unknown fields: {extra}")

        if "name" not in item:
            raise RuntimeError(f"features[{i}] missing required field 'name'")
        name = str(item["name"]).strip()
        if name == "":
            raise RuntimeError(f"features[{i}] has empty name")

        w_raw = item["windows"] if "windows" in item else item.get("window")
        if w_raw is None:
            raise RuntimeError(f"features[{i}] missing required field 'window/windows'")
        windows = _parse_windows(w_raw)

        in_raw = item["input_fields"] if "input_fields" in item else item.get("input")
        if in_raw is None:
            raise RuntimeError(f"features[{i}] missing required field 'input/input_fields'")
        input_fields = tuple(x.lower() for x in _parse_csv_or_list(in_raw, field_name="input"))

        dep_raw = item["dependencies"] if "dependencies" in item else item.get("depends_on", [])
        if dep_raw in (None, ""):
            dep_raw = []
        dependencies = _parse_csv_or_list(dep_raw, field_name="dependencies") if dep_raw != [] else ()

        dtype = str(item.get("dtype", "float64")).strip().lower()
        if dtype != "float64":
            raise RuntimeError(f"features[{i}] dtype must be float64")

        out.append(
            FeatureSpec(
                name=name,
                windows=windows,
                input_fields=input_fields,
                dtype="float64",
                dependencies=tuple(str(x).strip() for x in dependencies),
            )
        )
    return _ensure_feature_specs(out)


def build_feature_dag(feature_specs: Sequence[FeatureSpec]) -> dict[str, set[str]]:
    specs = _ensure_feature_specs(feature_specs)
    names = {s.name for s in specs}
    dag: dict[str, set[str]] = {}
    for s in specs:
        deps = set(s.dependencies)
        missing = sorted(d for d in deps if d not in names)
        if missing:
            raise RuntimeError(f"Feature '{s.name}' has unknown dependencies: {missing}")
        dag[s.name] = deps
    return dag


def resolve_feature_execution_order(dag: Mapping[str, Iterable[str]]) -> list[str]:
    nodes = sorted(str(k) for k in dag.keys())
    dep_map: dict[str, set[str]] = {n: set(str(d) for d in dag[n]) for n in nodes}
    indeg: dict[str, int] = {n: len(dep_map[n]) for n in nodes}
    out_adj: dict[str, set[str]] = {n: set() for n in nodes}
    for n in nodes:
        for d in dep_map[n]:
            if d not in out_adj:
                raise RuntimeError(f"Dependency graph references unknown node: {d}")
            out_adj[d].add(n)

    ready = sorted([n for n, k in indeg.items() if k == 0])
    order: list[str] = []
    while ready:
        n = ready.pop(0)
        order.append(n)
        for m in sorted(out_adj[n]):
            indeg[m] -= 1
            if indeg[m] == 0:
                ready.append(m)
        ready.sort()

    if len(order) != len(nodes):
        cycle_nodes = sorted([n for n in nodes if indeg[n] > 0])
        raise RuntimeError(f"Cyclic feature dependency graph detected: {cycle_nodes}")
    return order


def rolling_view(array: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float64)
    w = int(window)
    if w <= 0:
        raise RuntimeError("rolling_view window must be > 0")
    if arr.ndim not in {1, 2}:
        raise RuntimeError(f"rolling_view expects 1D or 2D array, got ndim={arr.ndim}")
    if arr.ndim == 1:
        if arr.shape[0] < w:
            raise RuntimeError("rolling_view window exceeds array length")
        return sliding_window_view(arr, window_shape=w, axis=0)
    if arr.shape[1] < w:
        raise RuntimeError("rolling_view window exceeds time dimension")
    return sliding_window_view(arr, window_shape=w, axis=1)


def _safe_mad_clip(arr_ta: np.ndarray, mad_clip_k: float) -> tuple[np.ndarray, int]:
    arr = np.asarray(arr_ta, dtype=np.float64).copy()
    med = np.nanmedian(arr, axis=0)
    mad = np.nanmedian(np.abs(arr - med[None, :]), axis=0)
    scale = np.maximum(1.4826 * mad, 1e-12)
    lo = med - float(mad_clip_k) * scale
    hi = med + float(mad_clip_k) * scale
    finite = np.isfinite(arr)
    clipped = np.clip(arr, lo[None, :], hi[None, :])
    n = int(np.sum(finite & (arr != clipped)))
    arr[finite] = clipped[finite]
    return arr, n


def _forward_fill_small_gaps(arr_ta: np.ndarray, max_gap: int) -> tuple[np.ndarray, int]:
    arr = np.asarray(arr_ta, dtype=np.float64).copy()
    T, A = arr.shape
    filled = 0
    for a in range(A):
        x = arr[:, a]
        bad = ~np.isfinite(x)
        if not np.any(bad):
            continue
        idx = np.where(bad)[0]
        if idx.size == 0:
            continue
        seg_start = idx[0]
        prev = idx[0]
        for k in range(1, idx.size + 1):
            boundary = (k == idx.size) or (idx[k] != prev + 1)
            if boundary:
                seg_end = prev + 1
                seg_len = seg_end - seg_start
                if seg_start > 0 and seg_len <= int(max_gap) and np.isfinite(x[seg_start - 1]):
                    x[seg_start:seg_end] = x[seg_start - 1]
                    filled += seg_len
                if k < idx.size:
                    seg_start = idx[k]
                    prev = idx[k]
            else:
                prev = idx[k]
        arr[:, a] = x
    return arr, int(filled)


def sanitize_market_data(data: Mapping[str, np.ndarray], cfg: FeatureEngineConfig | None = None) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    c = cfg if cfg is not None else FeatureEngineConfig()
    required = ("open", "high", "low", "close", "volume")
    out: dict[str, np.ndarray] = {}
    logs: list[dict[str, object]] = []

    for k in required:
        if k not in data:
            raise RuntimeError(f"sanitize_market_data missing required field: {k}")
        arr = np.asarray(data[k], dtype=np.float64)
        if arr.ndim != 2:
            raise RuntimeError(f"sanitize_market_data field '{k}' must be 2D [T,A], got {arr.shape}")
        out[k] = arr.copy()

    T, A = out["close"].shape
    for k in required:
        if out[k].shape != (T, A):
            raise RuntimeError(f"sanitize_market_data shape mismatch for '{k}': {out[k].shape} vs {(T, A)}")

    if "ts_ns" in data:
        ts_ns = np.asarray(data["ts_ns"], dtype=np.int64).copy()
        if ts_ns.shape != (T,):
            raise RuntimeError(f"sanitize_market_data ts_ns shape mismatch: {ts_ns.shape} vs {(T,)}")
        _assert_monotonic_ts_ns(ts_ns)
        out["ts_ns"] = ts_ns
        if T > 1:
            d = np.diff(ts_ns)
            med = int(np.median(d))
            if med > 0:
                gap_idx = np.where(d > med)[0]
                if gap_idx.size > 0:
                    logs.append(
                        {
                            "event": "timestamp_gaps_detected",
                            "count": int(gap_idx.size),
                            "first_index": int(gap_idx[0]),
                            "median_delta_ns": int(med),
                        }
                    )

    bar_valid = np.asarray(data.get("bar_valid", np.ones((T, A), dtype=bool)), dtype=bool).copy()
    if bar_valid.shape != (T, A):
        raise RuntimeError(f"sanitize_market_data bar_valid shape mismatch: {bar_valid.shape} vs {(T, A)}")

    for k in required:
        n_bad_before = int(np.sum(~np.isfinite(out[k])))
        filled_arr, n_ffill = _forward_fill_small_gaps(out[k], max_gap=int(c.ffill_gap_limit))
        clipped_arr, n_clip = _safe_mad_clip(filled_arr, mad_clip_k=float(c.mad_clip_k))
        out[k] = clipped_arr
        n_bad_after = int(np.sum(~np.isfinite(clipped_arr)))
        if n_bad_before > 0 or n_ffill > 0 or n_clip > 0 or n_bad_after > 0:
            logs.append(
                {
                    "event": "sanitize_field",
                    "field": k,
                    "non_finite_before": int(n_bad_before),
                    "forward_filled": int(n_ffill),
                    "outliers_clipped": int(n_clip),
                    "non_finite_after": int(n_bad_after),
                }
            )

    finite_mask = np.isfinite(out["open"]) & np.isfinite(out["high"]) & np.isfinite(out["low"]) & np.isfinite(out["close"]) & np.isfinite(out["volume"])
    phys_mask = (out["high"] >= out["low"]) & (out["high"] >= out["open"]) & (out["high"] >= out["close"]) & (out["low"] <= out["open"]) & (out["low"] <= out["close"]) & (out["volume"] >= 0.0)
    clean_mask = finite_mask & phys_mask
    bar_valid &= clean_mask

    n_invalid = int(np.sum(~bar_valid))
    if n_invalid > 0:
        logs.append({"event": "corrupted_points_masked", "count": int(n_invalid)})

    if bool(c.drop_corrupted_rows):
        keep_t = np.all(bar_valid, axis=1)
        dropped = int(np.sum(~keep_t))
        if dropped > 0:
            for k in ("open", "high", "low", "close", "volume", "bar_valid"):
                arr = bar_valid if k == "bar_valid" else out[k]
                out[k] = arr[keep_t]
            if "ts_ns" in out:
                out["ts_ns"] = out["ts_ns"][keep_t]
            logs.append({"event": "dropped_corrupted_rows", "count": dropped})
            return out, logs

    out["bar_valid"] = bar_valid
    return out, logs


def _resolve_backend(compute_backend: str) -> tuple[Any, str]:
    backend = str(compute_backend).strip().lower()
    if backend not in {"numpy", "cupy"}:
        raise RuntimeError(f"Unsupported compute backend: {compute_backend}")
    if backend == "numpy":
        return np, "numpy"
    try:
        import cupy as cp  # type: ignore
    except Exception:
        return np, "numpy"
    return cp, "cupy"


def _rolling_sum_prefix(x_aw: Any, w: int, xp: Any) -> Any:
    w = int(w)
    if w <= 0:
        raise RuntimeError("rolling window must be > 0")
    T = int(x_aw.shape[1])
    c = xp.cumsum(x_aw, axis=1, dtype=x_aw.dtype)
    out = c.copy()
    if w < T:
        out[:, w:] = c[:, w:] - c[:, :-w]
    return out


def _rolling_mean_prefix(x_aw: Any, w: int, xp: Any) -> Any:
    T = int(x_aw.shape[1])
    s = _rolling_sum_prefix(x_aw, int(w), xp)
    denom = xp.minimum(xp.arange(1, T + 1), int(w)).astype(x_aw.dtype)
    return s / denom[None, :]


def _rolling_std_prefix(x_aw: Any, w: int, xp: Any) -> Any:
    m = _rolling_mean_prefix(x_aw, int(w), xp)
    m2 = _rolling_mean_prefix(x_aw * x_aw, int(w), xp)
    var = xp.maximum(m2 - m * m, 0.0)
    return xp.sqrt(var)


def _as_aw_inputs(data_ta: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for k, v in data_ta.items():
        if k in {"ts_ns"}:
            continue
        arr = np.asarray(v)
        if arr.ndim == 2:
            out[k] = np.ascontiguousarray(arr.T.astype(np.float64, copy=False))
    if "returns" not in out and "close" in out:
        close = out["close"]
        ret = np.zeros_like(close, dtype=np.float64)
        ret[:, 1:] = np.diff(close, axis=1) / np.maximum(close[:, :-1], 1e-12)
        out["returns"] = ret
    if "hl_spread" not in out and {"high", "low", "close"} <= set(out.keys()):
        out["hl_spread"] = (out["high"] - out["low"]) / np.maximum(out["close"], 1e-12)
    if "price" not in out and "close" in out:
        out["price"] = out["close"]
    return out


def _compute_feature_window_numpy(
    *,
    spec: FeatureSpec,
    w: int,
    inputs_aw: Mapping[str, np.ndarray],
    deps: Mapping[str, Mapping[int, np.ndarray]],
) -> np.ndarray:
    name = spec.name
    w = int(w)

    if name in {"rolling_volatility", "roll_std_ret"}:
        src = np.asarray(inputs_aw.get("returns"), dtype=np.float64)
        return _rolling_std_prefix(src, w, np)
    if name in {"momentum"}:
        price = np.asarray(inputs_aw.get("price", inputs_aw.get("close")), dtype=np.float64)
        idx = np.arange(price.shape[1], dtype=np.int64)
        start = np.maximum(idx - (w - 1), 0)
        base = price[:, start]
        return price / (base + 1e-12) - 1.0
    if name in {"normalized_momentum"}:
        if "momentum" not in deps or int(w) not in deps["momentum"]:
            raise RuntimeError("normalized_momentum requires dependency 'momentum' with matching window")
        mom = np.asarray(deps["momentum"][int(w)], dtype=np.float64)
        scale = _rolling_std_prefix(mom, w, np)
        return mom / (scale + 1e-12)
    if name in {"vwap_deviation"}:
        px = np.asarray(inputs_aw.get("price", inputs_aw.get("close")), dtype=np.float64)
        vol = np.asarray(inputs_aw.get("volume"), dtype=np.float64)
        vwap = _rolling_sum_prefix(px * vol, w, np) / np.maximum(_rolling_sum_prefix(vol, w, np), 1e-12)
        return px / np.maximum(vwap, 1e-12) - 1.0
    if name in {"roll_mean_ret"}:
        src = np.asarray(inputs_aw.get("returns"), dtype=np.float64)
        return _rolling_mean_prefix(src, w, np)
    if name in {"roll_mean_hl_spread"}:
        src = np.asarray(inputs_aw.get("hl_spread"), dtype=np.float64)
        return _rolling_mean_prefix(src, w, np)
    if name in {"roll_mean_close"}:
        src = np.asarray(inputs_aw.get("close"), dtype=np.float64)
        return _rolling_mean_prefix(src, w, np)
    if name in {"roll_std_close"}:
        src = np.asarray(inputs_aw.get("close"), dtype=np.float64)
        return _rolling_std_prefix(src, w, np)
    if name in {"roll_mean_vol"}:
        src = np.asarray(inputs_aw.get("volume"), dtype=np.float64)
        return _rolling_mean_prefix(src, w, np)

    if name.startswith("roll_mean_"):
        if len(spec.input_fields) == 0:
            raise RuntimeError(f"Feature '{name}' requires input_fields")
        src = np.asarray(inputs_aw.get(spec.input_fields[0]), dtype=np.float64)
        return _rolling_mean_prefix(src, w, np)
    if name.startswith("roll_std_"):
        if len(spec.input_fields) == 0:
            raise RuntimeError(f"Feature '{name}' requires input_fields")
        src = np.asarray(inputs_aw.get(spec.input_fields[0]), dtype=np.float64)
        return _rolling_std_prefix(src, w, np)
    raise RuntimeError(f"Unsupported feature kernel: {name}")


def _compute_feature_windows_numpy(
    *,
    spec: FeatureSpec,
    inputs_aw: Mapping[str, np.ndarray],
    deps: Mapping[str, Mapping[int, np.ndarray]],
) -> tuple[str, dict[int, np.ndarray]]:
    out: dict[int, np.ndarray] = {}
    for w in spec.windows:
        arr = _compute_feature_window_numpy(spec=spec, w=int(w), inputs_aw=inputs_aw, deps=deps)
        out[int(w)] = np.asarray(arr, dtype=np.float64)
    return spec.name, out


def _compute_feature_windows_gpu(
    *,
    spec: FeatureSpec,
    inputs_aw: Mapping[str, np.ndarray],
    deps: Mapping[str, Mapping[int, np.ndarray]],
    cp: Any,
) -> tuple[str, dict[int, np.ndarray]]:
    xp_inputs = {k: cp.asarray(np.asarray(v, dtype=np.float64)) for k, v in inputs_aw.items()}
    xp_deps: dict[str, dict[int, Any]] = {}
    for k, d in deps.items():
        xp_deps[k] = {int(w): cp.asarray(np.asarray(v, dtype=np.float64)) for w, v in d.items()}

    out: dict[int, np.ndarray] = {}
    for w in spec.windows:
        name = spec.name
        if name in {"rolling_volatility", "roll_std_ret"}:
            arr = _rolling_std_prefix(xp_inputs["returns"], int(w), cp)
        elif name in {"momentum"}:
            price = xp_inputs.get("price", xp_inputs.get("close"))
            idx = cp.arange(price.shape[1], dtype=cp.int64)
            start = cp.maximum(idx - (int(w) - 1), 0)
            base = price[:, start]
            arr = price / (base + 1e-12) - 1.0
        elif name in {"normalized_momentum"}:
            if "momentum" not in xp_deps or int(w) not in xp_deps["momentum"]:
                raise RuntimeError("normalized_momentum requires dependency 'momentum' with matching window")
            mom = xp_deps["momentum"][int(w)]
            scale = _rolling_std_prefix(mom, int(w), cp)
            arr = mom / (scale + 1e-12)
        elif name in {"vwap_deviation"}:
            px = xp_inputs.get("price", xp_inputs.get("close"))
            vol = xp_inputs["volume"]
            vwap = _rolling_sum_prefix(px * vol, int(w), cp) / cp.maximum(_rolling_sum_prefix(vol, int(w), cp), 1e-12)
            arr = px / cp.maximum(vwap, 1e-12) - 1.0
        elif name in {"roll_mean_ret"}:
            arr = _rolling_mean_prefix(xp_inputs["returns"], int(w), cp)
        elif name in {"roll_mean_hl_spread"}:
            arr = _rolling_mean_prefix(xp_inputs["hl_spread"], int(w), cp)
        elif name in {"roll_mean_close"}:
            arr = _rolling_mean_prefix(xp_inputs["close"], int(w), cp)
        elif name in {"roll_std_close"}:
            arr = _rolling_std_prefix(xp_inputs["close"], int(w), cp)
        elif name in {"roll_mean_vol"}:
            arr = _rolling_mean_prefix(xp_inputs["volume"], int(w), cp)
        elif name.startswith("roll_mean_"):
            arr = _rolling_mean_prefix(xp_inputs[spec.input_fields[0]], int(w), cp)
        elif name.startswith("roll_std_"):
            arr = _rolling_std_prefix(xp_inputs[spec.input_fields[0]], int(w), cp)
        else:
            raise RuntimeError(f"Unsupported feature kernel: {name}")
        out[int(w)] = cp.asnumpy(arr).astype(np.float64, copy=False)
    return spec.name, out


def _feature_worker_job(
    spec_payload: dict[str, Any],
    inputs_aw: dict[str, np.ndarray],
    deps: dict[str, dict[int, np.ndarray]],
    seed: int,
) -> tuple[str, dict[int, np.ndarray]]:
    np.random.seed(int(seed))
    spec = FeatureSpec(
        name=str(spec_payload["name"]),
        windows=tuple(int(x) for x in spec_payload["windows"]),
        input_fields=tuple(str(x) for x in spec_payload["input_fields"]),
        dtype="float64",
        dependencies=tuple(str(x) for x in spec_payload["dependencies"]),
    )
    return _compute_feature_windows_numpy(spec=spec, inputs_aw=inputs_aw, deps=deps)


def _dataset_hash_from_data(data_ta: Mapping[str, np.ndarray], feature_specs: Sequence[FeatureSpec], ts_ns: np.ndarray | None = None) -> str:
    h = hashlib.sha256()
    for key in sorted(data_ta.keys()):
        if key == "ts_ns":
            continue
        arr = np.ascontiguousarray(np.asarray(data_ta[key]))
        h.update(key.encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        buf = arr.view(np.uint8).ravel()
        step = 8 * 1024 * 1024
        for i in range(0, buf.size, step):
            h.update(buf[i : i + step].tobytes())
    if ts_ns is not None:
        ts = np.ascontiguousarray(np.asarray(ts_ns, dtype=np.int64))
        h.update(ts.view(np.uint8).tobytes())
    h.update(feature_registry_hash(feature_specs).encode("utf-8"))
    return h.hexdigest()


def _cache_key(dataset_hash: str, registry_hash: str) -> str:
    return hashlib.sha256(f"{dataset_hash}|{registry_hash}".encode("utf-8")).hexdigest()[:24]


def _atomic_save_npz(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        np.savez_compressed(f, **payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _allocate_tensor(A: int, T: int, F: int, W: int, cfg: FeatureEngineConfig) -> tuple[np.ndarray, Path | None]:
    backend = str(cfg.tensor_backend).strip().lower()
    if backend == "ram":
        return np.zeros((A, T, F, W), dtype=np.float64), None
    if backend != "memmap":
        raise RuntimeError(f"Unsupported tensor backend: {cfg.tensor_backend}")

    p = resolve_repo_path(cfg.memmap_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        p.unlink(missing_ok=True)
    arr = np.memmap(p, mode="w+", dtype=np.float64, shape=(A, T, F, W))
    arr[:] = 0.0
    arr.flush()
    return arr, p


def build_feature_tensor(
    data: Mapping[str, np.ndarray],
    feature_specs: Sequence[FeatureSpec],
    *,
    engine_cfg: FeatureEngineConfig | None = None,
    ts_ns: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, int], dict[str, int], dict[str, Any]]:
    cfg = engine_cfg if engine_cfg is not None else FeatureEngineConfig()
    specs = _ensure_feature_specs(feature_specs)
    dag = build_feature_dag(specs)
    order = resolve_feature_execution_order(dag)
    spec_by_name = {s.name: s for s in specs}
    ordered_specs = [spec_by_name[n] for n in order]

    sanitized_ta, sanitize_logs = sanitize_market_data(
        {
            "open": np.asarray(data["open"], dtype=np.float64),
            "high": np.asarray(data["high"], dtype=np.float64),
            "low": np.asarray(data["low"], dtype=np.float64),
            "close": np.asarray(data["close"], dtype=np.float64),
            "volume": np.asarray(data["volume"], dtype=np.float64),
            "bar_valid": np.asarray(data.get("bar_valid", np.ones_like(np.asarray(data["close"], dtype=np.float64), dtype=bool)), dtype=bool),
            "ts_ns": np.asarray(ts_ns if ts_ns is not None else data.get("ts_ns", np.arange(np.asarray(data["close"]).shape[0], dtype=np.int64)), dtype=np.int64),
        },
        cfg,
    )
    if "bar_valid" not in sanitized_ta:
        raise RuntimeError("sanitize_market_data must produce bar_valid")
    T, A = sanitized_ta["close"].shape

    all_windows = sorted(set(int(w) for s in ordered_specs for w in s.windows))
    if len(all_windows) == 0:
        raise RuntimeError("No windows found for feature tensor build")

    feature_map = {s.name: i for i, s in enumerate(ordered_specs)}
    window_map = {str(i): int(w) for i, w in enumerate(all_windows)}
    window_to_idx = {int(w): i for i, w in enumerate(all_windows)}

    registry_hash = feature_registry_hash(ordered_specs)
    dataset_hash = _dataset_hash_from_data(sanitized_ta, ordered_specs, ts_ns=sanitized_ta.get("ts_ns"))
    key = _cache_key(dataset_hash, registry_hash)
    cache_dir = resolve_repo_path(cfg.cache_dir)
    cache_path = cache_dir / f"feature_tensor_{key}.npz"
    memmap_path = resolve_repo_path(cfg.memmap_path)

    if bool(cfg.use_cache) and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as d:
            meta_raw = d.get("metadata")
            if meta_raw is None:
                raise RuntimeError("Feature cache missing metadata")
            meta = json.loads(str(meta_raw[0]))
            if "tensor" in d.files:
                tensor = np.asarray(d["tensor"], dtype=np.float64)
            elif str(meta.get("backend", "")).lower() == "memmap":
                mm_path = Path(str(meta.get("memmap_path", memmap_path))).expanduser().resolve()
                shape = tuple(int(x) for x in meta["shape"])
                tensor = np.memmap(mm_path, mode="r", dtype=np.float64, shape=shape)
            else:
                raise RuntimeError("Invalid cached feature tensor format")
        if tuple(tensor.shape) != (A, T, len(ordered_specs), len(all_windows)):
            raise RuntimeError("Cached feature tensor shape mismatch")
        if tensor.dtype != np.float64:
            raise RuntimeError("Cached feature tensor dtype mismatch")
        if np.any(~np.isfinite(np.asarray(tensor))):
            raise RuntimeError("Cached feature tensor contains non-finite values")
        tensor.flags.writeable = False
        return tensor, feature_map, window_map, {
            "cache_hit": True,
            "cache_path": str(cache_path),
            "registry_hash": registry_hash,
            "dataset_hash": dataset_hash,
            "sanitize_logs": sanitize_logs,
            "compute_backend_effective": str(meta.get("compute_backend_effective", "numpy")),
            "feature_order": [s.name for s in ordered_specs],
        }

    tensor, tensor_memmap_path = _allocate_tensor(A, T, len(ordered_specs), len(all_windows), cfg)

    clean_ta = {k: np.where(np.isfinite(v), np.asarray(v, dtype=np.float64), 0.0) for k, v in sanitized_ta.items() if k in {"open", "high", "low", "close", "volume", "bar_valid"}}
    clean_aw = _as_aw_inputs(clean_ta)
    for k in ("open", "high", "low", "close", "volume", "returns", "hl_spread", "price"):
        if k in clean_aw:
            clean_aw[k] = np.where(np.isfinite(clean_aw[k]), clean_aw[k], 0.0).astype(np.float64, copy=False)

    backend_mod, backend_name = _resolve_backend(cfg.compute_backend)
    if backend_name == "cupy" and str(cfg.parallel_backend).strip().lower() == "process_pool":
        backend_mod, backend_name = np, "numpy"

    feature_results: dict[str, dict[int, np.ndarray]] = {}
    pending = set(order)
    worker_seed_base = int(cfg.seed)
    while pending:
        ready = [n for n in order if n in pending and dag[n].issubset(feature_results.keys())]
        if len(ready) == 0:
            raise RuntimeError("Unable to resolve feature execution order during compute")

        if str(cfg.parallel_backend).strip().lower() == "process_pool" and backend_name == "numpy" and len(ready) > 1:
            max_workers = int(max(1, cfg.max_workers))
            jobs: list[tuple[str, dict[int, np.ndarray]]] = []
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futs = []
                for j, name in enumerate(sorted(ready)):
                    spec = spec_by_name[name]
                    dep = {d: feature_results[d] for d in spec.dependencies}
                    payload = {
                        "name": spec.name,
                        "windows": [int(w) for w in spec.windows],
                        "input_fields": [str(x) for x in spec.input_fields],
                        "dependencies": [str(x) for x in spec.dependencies],
                    }
                    futs.append(ex.submit(_feature_worker_job, payload, dict(clean_aw), dep, worker_seed_base + j))
                for f in futs:
                    jobs.append(f.result())
            jobs.sort(key=lambda x: x[0])
            for name, out_w in jobs:
                feature_results[name] = out_w
                pending.remove(name)
        else:
            for name in sorted(ready):
                spec = spec_by_name[name]
                dep = {d: feature_results[d] for d in spec.dependencies}
                if backend_name == "cupy":
                    out_name, out_w = _compute_feature_windows_gpu(spec=spec, inputs_aw=clean_aw, deps=dep, cp=backend_mod)
                else:
                    out_name, out_w = _compute_feature_windows_numpy(spec=spec, inputs_aw=clean_aw, deps=dep)
                feature_results[out_name] = out_w
                pending.remove(name)

    for s in ordered_specs:
        fi = feature_map[s.name]
        for w in s.windows:
            wi = window_to_idx[int(w)]
            arr_aw = np.asarray(feature_results[s.name][int(w)], dtype=np.float64)
            if arr_aw.shape != (A, T):
                raise RuntimeError(
                    f"Feature '{s.name}' window {w} shape mismatch: got {arr_aw.shape}, expected {(A, T)}"
                )
            tensor[:, :, fi, wi] = arr_aw

    arr_tensor = np.asarray(tensor)
    if np.any(~np.isfinite(arr_tensor)):
        bad = np.argwhere(~np.isfinite(arr_tensor))[0]
        raise RuntimeError(
            f"Feature tensor contains non-finite values at [a,t,f,w]={bad.tolist()}"
        )
    if arr_tensor.dtype != np.float64:
        raise RuntimeError("Feature tensor dtype must be float64")

    if isinstance(tensor, np.memmap):
        tensor.flush()
    tensor.flags.writeable = False

    meta = {
        "shape": [int(x) for x in tensor.shape],
        "dtype": "float64",
        "backend": str(cfg.tensor_backend).strip().lower(),
        "compute_backend_effective": backend_name,
        "dataset_hash": dataset_hash,
        "registry_hash": registry_hash,
        "feature_order": [s.name for s in ordered_specs],
        "window_values": [int(w) for w in all_windows],
        "memmap_path": str(tensor_memmap_path) if tensor_memmap_path is not None else "",
    }

    if bool(cfg.use_cache):
        cache_payload: dict[str, Any] = {"metadata": np.asarray([json.dumps(meta, sort_keys=True)], dtype=np.str_)}
        if str(cfg.tensor_backend).strip().lower() == "ram":
            cache_payload["tensor"] = np.asarray(tensor, dtype=np.float64)
        _atomic_save_npz(cache_path, cache_payload)

    return tensor, feature_map, window_map, {
        "cache_hit": False,
        "cache_path": str(cache_path),
        "registry_hash": registry_hash,
        "dataset_hash": dataset_hash,
        "sanitize_logs": sanitize_logs,
        "compute_backend_effective": backend_name,
        "feature_order": [s.name for s in ordered_specs],
        "window_values": [int(w) for w in all_windows],
    }


def make_compat_feature_specs(windows: Sequence[int]) -> list[FeatureSpec]:
    wins = tuple(sorted(set(int(w) for w in windows)))
    if len(wins) == 0:
        raise RuntimeError("windows must be non-empty")
    return [
        FeatureSpec(name="roll_mean_ret", windows=wins, input_fields=("returns",)),
        FeatureSpec(name="roll_std_ret", windows=wins, input_fields=("returns",)),
        FeatureSpec(name="roll_mean_hl_spread", windows=wins, input_fields=("hl_spread",)),
        FeatureSpec(name="roll_mean_close", windows=wins, input_fields=("close",)),
        FeatureSpec(name="roll_std_close", windows=wins, input_fields=("close",)),
        FeatureSpec(name="roll_mean_vol", windows=wins, input_fields=("volume",)),
    ]


def build_feature_tensor_from_arrays(
    open_ta: np.ndarray,
    high_ta: np.ndarray,
    low_ta: np.ndarray,
    close_ta: np.ndarray,
    volume_ta: np.ndarray,
    *,
    feature_specs: Sequence[FeatureSpec],
    engine_cfg: FeatureEngineConfig | None = None,
    ts_ns: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, int], dict[str, int], dict[str, Any]]:
    o = np.asarray(open_ta, dtype=np.float64)
    h = np.asarray(high_ta, dtype=np.float64)
    l = np.asarray(low_ta, dtype=np.float64)
    c = np.asarray(close_ta, dtype=np.float64)
    v = np.asarray(volume_ta, dtype=np.float64)
    if not (o.shape == h.shape == l.shape == c.shape == v.shape):
        raise RuntimeError("Raw array shape mismatch for feature tensor build")
    if o.ndim != 2:
        raise RuntimeError("Raw arrays must be 2D [T,A]")
    T, A = o.shape
    bar_valid = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c) & np.isfinite(v)
    data = {
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": v,
        "bar_valid": bar_valid,
    }
    if ts_ns is not None:
        ts = np.asarray(ts_ns, dtype=np.int64)
        if ts.shape != (T,):
            raise RuntimeError(f"ts_ns shape mismatch: got {ts.shape}, expected {(T,)}")
        data["ts_ns"] = ts
    return build_feature_tensor(data, feature_specs, engine_cfg=engine_cfg, ts_ns=np.asarray(data.get("ts_ns")) if "ts_ns" in data else None)


def build_feature_tensor_from_state(
    state: TensorState,
    *,
    feature_specs: Sequence[FeatureSpec] | None = None,
    registry_path: str | os.PathLike[str] = "feature_registry.yaml",
    engine_cfg: FeatureEngineConfig | None = None,
) -> tuple[np.ndarray, dict[str, int], dict[str, int], dict[str, Any]]:
    specs = list(feature_specs) if feature_specs is not None else load_feature_registry(registry_path)
    return build_feature_tensor_from_arrays(
        state.open_px,
        state.high_px,
        state.low_px,
        state.close_px,
        state.volume,
        feature_specs=specs,
        engine_cfg=engine_cfg,
        ts_ns=state.ts_ns,
    )


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
