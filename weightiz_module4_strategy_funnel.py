"""
Weightiz Institutional Engine - Module 4 (Strategy + Zimtra Funnel)
====================================================================

Deterministic execution layer on top of Modules 1-3.

Key guarantees:
- Numpy-only core path.
- Causal next-bar-open execution for LIVE bars.
- Same-bar close-auction execution at OVERNIGHT_SELECT (15:45) for flatten/winner.
- Inverse-volatility (ATR-based) Top-K allocation.
- Dynamic structural exits based on current Module 3 context.
- Multiplicative overnight conviction score.
"""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Tuple

import numpy as np

from weightiz_module1_core import (
    OrderIdx,
    Phase,
    ProfileStatIdx,
    ScoreIdx,
    TensorState,
    validate_state_hard,
)
from weightiz_module3_structure import (
    ContextIdx,
    IB_MISSING_POLICY,
    IB_POLICY_NO_TRADE,
    Module3Output,
    Struct30mIdx,
)
from weightiz_dtype_guard import assert_float64
from weightiz_system_logger import get_logger, log_event


class RegimeIdx(IntEnum):
    NONE = 0
    NEUTRAL = 1
    TREND = 2
    P_SHAPE = 3
    B_SHAPE = 4
    DOUBLE_DISTRIBUTION = 5


class OrderFlagBit(IntEnum):
    ENTRY = 1 << 0
    EXIT = 1 << 1
    FLATTEN = 1 << 2
    OVERNIGHT_CANDIDATE = 1 << 3
    OVERNIGHT_SELECTED = 1 << 4
    KILL_SWITCH = 1 << 5
    MOC_EXEC = 1 << 6


@dataclass(frozen=True)
class Module4Config:
    # Decision-layer runtime safety
    fail_on_non_finite_input: bool = True
    fail_on_non_finite_output: bool = True
    eps: float = 1e-12
    enforce_causal_source_validation: bool = True
    enforce_window_causal_sanity: bool = True

    # Window adapter
    window_selection_mode: str = "multi_window"
    fixed_window_index: int = 0
    anchor_window_index: int = 0

    # Optional pre-intent risk filters
    max_volatility: float = np.inf
    max_spread: float = np.inf
    min_liquidity: float = 0.0

    # Regime decision control
    regime_confidence_min: float = 0.55

    # Signal thresholds
    entry_threshold: float = 0.55
    exit_threshold: float = 0.25

    # Conviction controls
    conviction_scale: float = 1.0
    conviction_clip: float = 1.0

    # Allocation controls
    max_abs_weight: float = 1.0

    # Allocation
    top_k_intraday: int = 5
    max_asset_cap_frac: float = 0.30
    max_turnover_frac_per_bar: float = 0.35

    # Overnight selection
    overnight_min_conviction: float = 0.65
    allow_cash_overnight: bool = True

    # Taxonomy thresholds
    trend_spread_min: float = 0.05
    trend_poc_drift_min_abs: float = 0.35
    neutral_poc_drift_max_abs: float = 0.15
    shape_skew_min_abs: float = 0.35
    double_dist_sep_x: float = 1.0
    double_dist_valley_frac: float = 0.35

    # Costs
    commission_bps: float = 0.40
    spread_tick_mult: float = 1.50
    slippage_bps_low_rvol: float = 3.0
    slippage_bps_mid_rvol: float = 2.0
    slippage_bps_high_rvol: float = 1.5
    stress_slippage_mult: float = 1.0

    # Risk behavior
    hard_kill_on_daily_loss_breach: bool = True

    # Bridge control
    enable_degraded_bridge_mode: bool = True
    execution_strict_prices: bool = True

    # Compatibility-only schema fields for Cell-6 nomenclature.
    # Core Module4 decision logic ignores these fields.
    strategy_type: str = "legacy"
    score_gate: str = ""
    score_gate_rule: str = ""
    deviation_signal: str = ""
    deviation_rule: str = ""
    entry_model: str = ""
    exit_model: str = ""
    origin_level: str = "POC"
    direction: str = "long"
    delta_th: float = 0.55
    dev_th: float = 1.0
    tp_mult: float = 1.0
    atr_stop_mult: float = 1.0


@dataclass
class Module4Output:
    regime_primary_ta: np.ndarray
    regime_confidence_ta: np.ndarray
    intent_long_ta: np.ndarray
    intent_short_ta: np.ndarray
    target_qty_ta: np.ndarray
    filled_qty_ta: np.ndarray
    exec_price_ta: np.ndarray
    trade_cost_ta: np.ndarray
    overnight_score_ta: np.ndarray
    overnight_winner_t: np.ndarray
    kill_switch_t: np.ndarray


@dataclass(frozen=True)
class Module4SignalOutput:
    regime_primary_ta: np.ndarray
    regime_confidence_ta: np.ndarray
    intent_long_ta: np.ndarray
    intent_short_ta: np.ndarray
    target_qty_ta: np.ndarray

    def __post_init__(self) -> None:
        regime = np.asarray(self.regime_primary_ta)
        confidence = np.asarray(self.regime_confidence_ta)
        intent_long = np.asarray(self.intent_long_ta)
        intent_short = np.asarray(self.intent_short_ta)
        target_qty = np.asarray(self.target_qty_ta)

        if regime.ndim != 2:
            raise RuntimeError(f"regime_primary_ta must be [T,A], got shape={regime.shape}")
        T, A = regime.shape
        checks = [
            ("regime_confidence_ta", confidence, np.float64),
            ("intent_long_ta", intent_long, np.bool_),
            ("intent_short_ta", intent_short, np.bool_),
            ("target_qty_ta", target_qty, np.float64),
        ]
        if regime.dtype != np.int8:
            raise RuntimeError(f"regime_primary_ta dtype must be int8, got {regime.dtype}")
        for name, arr, dtype in checks:
            if arr.shape != (T, A):
                raise RuntimeError(f"{name} must be [T,A], got shape={arr.shape}")
            if arr.dtype != dtype:
                raise RuntimeError(f"{name} dtype must be {dtype}, got {arr.dtype}")


class NonFiniteExecutionPriceError(RuntimeError):
    def __init__(self, reason_code: str, exec_px_dump: dict[str, Any], message: str) -> None:
        super().__init__(str(message))
        self.reason_code = str(reason_code)
        self.exec_px_dump = dict(exec_px_dump)


def _first_true_idx(mask: np.ndarray) -> int:
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return -1
    return int(np.where(m)[0][0])


def _assert_shape(name: str, arr: np.ndarray, expected: Tuple[int, ...]) -> None:
    if arr.shape != expected:
        raise RuntimeError(f"{name} shape mismatch: got {arr.shape}, expected {expected}")


def _assert_finite_masked(name: str, arr: np.ndarray, mask: np.ndarray) -> None:
    m = np.asarray(mask, dtype=bool)
    if arr.ndim == m.ndim + 1:
        m = m[:, :, None]
    bad = m & (~np.isfinite(arr))
    if np.any(bad):
        loc = np.argwhere(bad)[:8]
        raise RuntimeError(f"{name} contains non-finite values on required rows at indices {loc.tolist()}")


def _slippage_bps_from_rvol(rvol: np.ndarray, cfg: Module4Config) -> np.ndarray:
    out = np.full(rvol.shape, float(cfg.slippage_bps_mid_rvol), dtype=np.float64)
    out[rvol < 0.8] = float(cfg.slippage_bps_low_rvol)
    out[rvol >= 1.5] = float(cfg.slippage_bps_high_rvol)
    return out


def _double_distribution_flags(
    vp_ab: np.ndarray,
    x_grid: np.ndarray,
    min_sep_x: float,
    valley_frac: float,
) -> np.ndarray:
    """
    Deterministic double-distribution detection for a single time slice.
    Returns bool[A].
    """
    vp = np.asarray(vp_ab, dtype=np.float64)
    x = np.asarray(x_grid, dtype=np.float64)
    A, B = vp.shape
    out = np.zeros(A, dtype=bool)
    if B < 5:
        return out

    left = vp[:, :-2]
    mid = vp[:, 1:-1]
    right = vp[:, 2:]
    is_peak_mid = (mid >= left) & (mid >= right)
    is_peak = np.zeros_like(vp, dtype=bool)
    is_peak[:, 1:-1] = is_peak_mid

    peak_vals = np.where(is_peak, vp, -np.inf)
    top2 = np.argpartition(-peak_vals, kth=1, axis=1)[:, :2]
    p1 = top2[:, 0]
    p2 = top2[:, 1]

    v1 = peak_vals[np.arange(A), p1]
    v2 = peak_vals[np.arange(A), p2]
    swap = v2 > v1
    p1, p2 = np.where(swap, p2, p1), np.where(swap, p1, p2)
    v1 = peak_vals[np.arange(A), p1]
    v2 = peak_vals[np.arange(A), p2]

    sep = np.abs(x[p1] - x[p2])
    sep_ok = sep >= float(min_sep_x)
    for a in range(A):
        i1 = int(min(p1[a], p2[a]))
        i2 = int(max(p1[a], p2[a]))
        if not np.isfinite(v1[a]) or not np.isfinite(v2[a]) or i2 <= i1 + 1:
            continue
        valley = float(np.min(vp[a, i1 + 1 : i2]))
        thr = float(valley_frac) * float(min(v1[a], v2[a]))
        out[a] = bool(sep_ok[a] and (valley < thr))
    return out


def _weighted_argmax_with_tie(score: np.ndarray, tie: np.ndarray) -> int:
    """
    Returns index of deterministic winner:
    - max score
    - max tie-break value
    - min index
    """
    if score.size == 0:
        return -1
    finite = np.isfinite(score) & np.isfinite(tie)
    if not np.any(finite):
        return -1
    idx = np.where(finite)[0]
    s = score[idx]
    z = tie[idx]
    order = np.lexsort((idx, -z, -s))
    return int(idx[order[0]])


def _execute_to_target(
    pos: np.ndarray,
    avg_cost: np.ndarray,
    cash: float,
    realized: float,
    target: np.ndarray,
    price: np.ndarray,
    rvol: np.ndarray,
    tick_size: np.ndarray,
    cfg: Module4Config,
    strict: bool,
    eps: float,
    px_source_name: str = "unknown",
    dump_builder: Callable[[int, float, str], dict[str, Any]] | None = None,
    error_reason_code: str = "NONFINITE_EXEC_PX",
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Execute transition pos -> target at supplied prices.
    Mutates pos/avg_cost; returns updated cash/realized and per-asset delta/cost.
    The skipped mask is legacy helper compatibility only and remains unreachable
    from the canonical signal-only runtime path.
    """
    A = pos.shape[0]
    delta = target - pos
    trade_cost = np.zeros(A, dtype=np.float64)
    skipped = np.zeros(A, dtype=bool)
    slip_bps = _slippage_bps_from_rvol(rvol, cfg)
    exec_needed = np.abs(delta) > eps
    if np.any(exec_needed):
        bad = exec_needed & ((~np.isfinite(price)) | (price <= 0.0))
        if np.any(bad):
            # Institutional behavior: missing quote => no fill (skip), never crash the run.
            delta[bad] = 0.0
            skipped[bad] = True

    for a in range(A):
        dq = float(delta[a])
        if abs(dq) <= eps:
            continue
        px = float(price[a])
        if (not np.isfinite(px)) or (px <= 0.0):
            delta[a] = 0.0
            skipped[a] = True
            continue

        ts = float(tick_size[a])
        notional = abs(dq) * px
        comm = notional * float(cfg.commission_bps) * 1e-4
        spread = 0.5 * float(cfg.spread_tick_mult) * ts * abs(dq)
        slip = notional * slip_bps[a] * float(cfg.stress_slippage_mult) * 1e-4
        cost = comm + spread + slip
        trade_cost[a] = cost

        old_pos = float(pos[a])
        old_avg = float(avg_cost[a])
        new_pos = old_pos + dq

        # Realized PnL when closing against existing position.
        if old_pos != 0.0 and np.sign(old_pos) != np.sign(dq):
            closed = min(abs(dq), abs(old_pos))
            realized += closed * (px - old_avg) * np.sign(old_pos)

        # Update average cost.
        if abs(new_pos) <= eps:
            avg_cost[a] = 0.0
            new_pos = 0.0
        else:
            if old_pos == 0.0:
                avg_cost[a] = px
            elif np.sign(old_pos) == np.sign(new_pos):
                if np.sign(old_pos) == np.sign(dq):
                    # Adding to same-direction position -> weighted avg.
                    avg_cost[a] = (abs(old_pos) * old_avg + abs(dq) * px) / (abs(old_pos) + abs(dq) + eps)
                else:
                    # Partial close without direction flip -> keep old avg.
                    avg_cost[a] = old_avg
            else:
                # Flipped direction -> remaining inventory opened at current px.
                avg_cost[a] = px

        pos[a] = new_pos
        cash -= dq * px + cost

    return cash, realized, delta, trade_cost, skipped


def _decision_to_signal_output(decision: Any) -> Module4SignalOutput:
    return Module4SignalOutput(
        regime_primary_ta=np.ascontiguousarray(np.asarray(decision.regime_id).T, dtype=np.int8),
        regime_confidence_ta=np.ascontiguousarray(np.asarray(decision.regime_confidence).T, dtype=np.float64),
        intent_long_ta=np.ascontiguousarray(np.asarray(decision.intent_long).T, dtype=bool),
        intent_short_ta=np.ascontiguousarray(np.asarray(decision.intent_short).T, dtype=bool),
        target_qty_ta=np.ascontiguousarray(np.asarray(decision.target_weight).T, dtype=np.float64),
    )


def _build_exec_px_dump(
    state: TensorState,
    m3: Module3Output,
    t_signal: int,
    t_fill: int,
    a: int,
    px_source_name: str,
    px_value: float,
    target_qty: float,
    reason_code: str,
    pending_order_id: str,
    quarantine_applied: bool,
    run_context: dict[str, Any],
) -> dict[str, Any]:
    t_sig = int(min(max(int(t_signal), 0), state.cfg.T - 1))
    t_fil = int(min(max(int(t_fill), 0), state.cfg.T - 1))
    aa = int(min(max(int(a), 0), state.cfg.A - 1))
    ts_ns_sig = int(state.ts_ns[t_sig])
    ts_ns_fil = int(state.ts_ns[t_fil])
    ts_utc_sig = datetime.fromtimestamp(float(ts_ns_sig) / 1_000_000_000.0, tz=timezone.utc).isoformat()
    ts_utc_fil = datetime.fromtimestamp(float(ts_ns_fil) / 1_000_000_000.0, tz=timezone.utc).isoformat()
    dqs_day = np.nan
    if hasattr(state, "dqs_day_ta"):
        dqs_arr = np.asarray(getattr(state, "dqs_day_ta"), dtype=np.float64)
        if dqs_arr.shape == (state.cfg.T, state.cfg.A):
            dqs_day = float(dqs_arr[t_fil, aa])
    ib_defined = True
    if m3.ib_defined_ta is not None:
        ib_defined = bool(np.asarray(m3.ib_defined_ta, dtype=bool)[t_fil, aa])
    return {
        "run_context": {
            "candidate_id": str(run_context.get("candidate_id", "unknown")),
            "split_id": str(run_context.get("split_id", "unknown")),
            "scenario_id": str(run_context.get("scenario_id", "unknown")),
        },
        "reason_code": str(reason_code),
        "t_signal": int(t_sig),
        "t_fill": int(t_fil),
        "ts_utc_signal": ts_utc_sig,
        "ts_utc_fill": ts_utc_fil,
        "asset_index": int(aa),
        "asset_symbol": str(state.symbols[aa]),
        "px_source_name": str(px_source_name),
        "px_value": float(px_value),
        "open_px_signal": float(state.open_px[t_sig, aa]),
        "high_px_signal": float(state.high_px[t_sig, aa]),
        "low_px_signal": float(state.low_px[t_sig, aa]),
        "close_px_signal": float(state.close_px[t_sig, aa]),
        "open_px_fill": float(state.open_px[t_fil, aa]),
        "high_px_fill": float(state.high_px[t_fil, aa]),
        "low_px_fill": float(state.low_px[t_fil, aa]),
        "close_px_fill": float(state.close_px[t_fil, aa]),
        "bar_valid_signal": bool(state.bar_valid[t_sig, aa]),
        "bar_valid_fill": bool(state.bar_valid[t_fil, aa]),
        "target_qty": float(target_qty),
        "position_qty_fill": float(state.position_qty[t_fil, aa]),
        "limit_px": float(state.orders[t_sig, aa, int(OrderIdx.LIMIT_PX)]),
        "stop_px": float(state.orders[t_sig, aa, int(OrderIdx.STOP_PX)]),
        "take_px": float(state.orders[t_sig, aa, int(OrderIdx.TAKE_PX)]),
        "conviction": float(state.orders[t_sig, aa, int(OrderIdx.CONVICTION)]),
        "dqs_day": float(dqs_day),
        "ib_defined": bool(ib_defined),
        "phase_signal": int(state.phase[t_sig]),
        "phase_fill": int(state.phase[t_fil]),
        "tod_signal": int(state.tod[t_sig]),
        "tod_fill": int(state.tod[t_fil]),
        "minute_of_day_signal": int(state.minute_of_day[t_sig]),
        "minute_of_day_fill": int(state.minute_of_day[t_fil]),
        "pending_order_id": str(pending_order_id),
        "quarantine_applied": bool(quarantine_applied),
    }


def _accumulate_exec_row(
    filled_row: np.ndarray,
    exec_px_row: np.ndarray,
    cost_row: np.ndarray,
    delta: np.ndarray,
    px: np.ndarray,
    cost: np.ndarray,
    eps: float,
) -> None:
    """
    Aggregate multiple fills in the same bar (e.g., open fill + close MOC).
    """
    A = filled_row.shape[0]
    for a in range(A):
        dq = float(delta[a])
        if abs(dq) <= eps:
            continue
        old_q = float(filled_row[a])
        if abs(old_q) <= eps:
            filled_row[a] = dq
            exec_px_row[a] = float(px[a])
        else:
            w_old = abs(old_q)
            w_new = abs(dq)
            exec_px_row[a] = (w_old * float(exec_px_row[a]) + w_new * float(px[a])) / (w_old + w_new + eps)
            filled_row[a] = old_q + dq
        cost_row[a] += float(cost[a])


def run_module4_strategy_funnel(
    state: TensorState,
    m3: Module3Output,
    cfg4: Module4Config,
    run_context: dict[str, Any] | None = None,
) -> Module4Output:
    """
    Legacy compatibility shim for the historical in-place strategy funnel.
    The canonical research path must use `run_module4_signal_funnel`.
    """
    if state is None or m3 is None or cfg4 is None:
        raise RuntimeError(
            "MODULE4_EXECUTION_FORBIDDEN_IN_CANONICAL_PATH: "
            "use run_module4_signal_funnel + risk_engine.simulate_portfolio_from_signals"
        )
    if not isinstance(m3, Module3Output):
        raise RuntimeError(
            "MODULE4_EXECUTION_FORBIDDEN_IN_CANONICAL_PATH: "
            "use run_module4_signal_funnel + risk_engine.simulate_portfolio_from_signals"
        )

    sig = run_module4_signal_funnel(state, m3, cfg4)
    T = int(state.cfg.T)
    A = int(state.cfg.A)
    state.orders[:] = np.nan
    state.order_side[:] = 0
    state.order_flags[:] = 0
    state.position_qty[:] = 0.0
    state.overnight_mask[:] = 0
    state.available_cash[:] = float(state.cfg.initial_cash)
    state.equity[:] = float(state.cfg.initial_cash)
    state.margin_used[:] = 0.0
    state.buying_power[:] = float(state.cfg.initial_cash)
    state.realized_pnl[:] = 0.0
    state.unrealized_pnl[:] = 0.0
    state.daily_loss[:] = 0.0
    state.daily_loss_breach_flag[:] = 0
    return Module4Output(
        regime_primary_ta=np.asarray(sig.regime_primary_ta, dtype=np.int8),
        regime_confidence_ta=np.asarray(sig.regime_confidence_ta, dtype=np.float64),
        intent_long_ta=np.asarray(sig.intent_long_ta, dtype=bool),
        intent_short_ta=np.asarray(sig.intent_short_ta, dtype=bool),
        target_qty_ta=np.asarray(sig.target_qty_ta, dtype=np.float64),
        filled_qty_ta=np.zeros((T, A), dtype=np.float64),
        exec_price_ta=np.full((T, A), np.nan, dtype=np.float64),
        trade_cost_ta=np.zeros((T, A), dtype=np.float64),
        overnight_score_ta=np.zeros((T, A), dtype=np.float64),
        overnight_winner_t=np.full(T, -1, dtype=np.int64),
        kill_switch_t=np.zeros(T, dtype=bool),
    )
    T = state.cfg.T
    A = state.cfg.A
    B = state.cfg.B
    eps = float(cfg4.eps)
    if run_context is None:
        run_context = {}
    assert_float64("module4.input.scores", state.scores)
    assert_float64("module4.input.profile_stats", state.profile_stats)

    # Shape checks
    _assert_shape("profile_stats", state.profile_stats, (T, A, int(ProfileStatIdx.N_FIELDS)))
    _assert_shape("scores", state.scores, (T, A, int(ScoreIdx.N_FIELDS)))
    _assert_shape("vp", state.vp, (T, A, B))
    _assert_shape("open_px", state.open_px, (T, A))
    _assert_shape("close_px", state.close_px, (T, A))
    _assert_shape("atr_floor", state.atr_floor, (T, A))
    _assert_shape("rvol", state.rvol, (T, A))
    _assert_shape("bar_valid", state.bar_valid, (T, A))
    _assert_shape("phase", state.phase, (T,))
    _assert_shape("session_id", state.session_id, (T,))

    _assert_shape("m3.context_tac", m3.context_tac, (T, A, int(ContextIdx.N_FIELDS)))
    _assert_shape("m3.context_valid_ta", m3.context_valid_ta, (T, A))
    _assert_shape("m3.context_source_t_index_ta", m3.context_source_t_index_ta, (T, A))
    _assert_shape("m3.block_features_tak", m3.block_features_tak, (T, A, int(Struct30mIdx.N_FIELDS)))
    if m3.ib_defined_ta is not None:
        _assert_shape("m3.ib_defined_ta", m3.ib_defined_ta, (T, A))

    phase_live = np.int8(Phase.LIVE)
    phase_os = np.int8(Phase.OVERNIGHT_SELECT)
    phase_flat = np.int8(Phase.FLATTEN)
    in_exec_phase_t = np.isin(state.phase, np.array([phase_live, phase_os, phase_flat], dtype=np.int8))
    tradable_ta = state.bar_valid & m3.context_valid_ta & in_exec_phase_t[:, None]
    dqs_day_ta = getattr(state, "dqs_day_ta", None)
    if dqs_day_ta is None:
        dqs_day_ta = np.ones((T, A), dtype=np.float64)
    else:
        dqs_day_ta = np.asarray(dqs_day_ta, dtype=np.float64)
        _assert_shape("state.dqs_day_ta", dqs_day_ta, (T, A))
    dqs_day_ta = np.clip(np.where(np.isfinite(dqs_day_ta), dqs_day_ta, 0.0), 0.0, 1.0)
    ib_missing_no_trade = str(IB_MISSING_POLICY).upper().strip() == str(IB_POLICY_NO_TRADE)
    if m3.ib_defined_ta is None:
        ib_defined_ta = np.ones((T, A), dtype=bool)
    else:
        ib_defined_ta = np.asarray(m3.ib_defined_ta, dtype=bool)

    if cfg4.fail_on_non_finite_input:
        _assert_finite_masked("close_px", state.close_px, tradable_ta)
        _assert_finite_masked("open_px", state.open_px, tradable_ta)
        _assert_finite_masked("atr_floor", state.atr_floor, tradable_ta)
        _assert_finite_masked("rvol", state.rvol, tradable_ta)
        _assert_finite_masked("scores", state.scores, tradable_ta)
        _assert_finite_masked("profile_stats", state.profile_stats, tradable_ta)
        _assert_finite_masked("context_tac", m3.context_tac, tradable_ta)

    # Reset mutable execution state tensors for deterministic replay.
    state.orders[:] = np.nan
    state.order_side[:] = 0
    state.order_flags[:] = 0
    state.position_qty[:] = 0.0
    state.overnight_mask[:] = 0
    state.available_cash[:] = 0.0
    state.equity[:] = 0.0
    state.margin_used[:] = 0.0
    state.buying_power[:] = 0.0
    state.realized_pnl[:] = 0.0
    state.unrealized_pnl[:] = 0.0
    state.daily_loss[:] = 0.0
    state.daily_loss_breach_flag[:] = 0

    # Precompute skew at each (t,a) from source block rows.
    src_ta = m3.context_source_t_index_ta.astype(np.int64)
    src_ok = src_ta >= 0
    safe_src_ta = np.where(src_ok, src_ta, 0)
    flat_src = safe_src_ta.ravel()
    flat_a = np.tile(np.arange(A, dtype=np.int64), T)
    skew_ref = m3.block_features_tak[:, :, int(Struct30mIdx.SKEW_ANCHOR)]
    skew_flat = skew_ref[flat_src, flat_a]
    skew_ta = skew_flat.reshape(T, A)
    skew_ta = np.where(src_ok, skew_ta, np.nan)

    # Output tensors
    regime_primary_ta = np.full((T, A), np.int8(RegimeIdx.NONE), dtype=np.int8)
    regime_confidence_ta = np.zeros((T, A), dtype=np.float64)
    intent_long_ta = np.zeros((T, A), dtype=bool)
    intent_short_ta = np.zeros((T, A), dtype=bool)
    target_qty_ta = np.zeros((T, A), dtype=np.float64)
    filled_qty_ta = np.zeros((T, A), dtype=np.float64)
    exec_price_ta = np.full((T, A), np.nan, dtype=np.float64)
    trade_cost_ta = np.zeros((T, A), dtype=np.float64)
    overnight_score_ta = np.zeros((T, A), dtype=np.float64)
    overnight_winner_t = np.full(T, -1, dtype=np.int64)
    kill_switch_t = np.zeros(T, dtype=bool)

    # Local causal state
    pos = np.zeros(A, dtype=np.float64)
    avg_cost = np.zeros(A, dtype=np.float64)
    cash = float(state.cfg.initial_cash)
    realized = 0.0
    pending_target = np.zeros(A, dtype=np.float64)
    pending_active = False
    pending_signal_t = -1
    pending_signal_session_id = -1
    overnight_idx = -1
    kill_switch_session = False
    quarantined_asset = np.zeros(A, dtype=bool)

    # Session baseline for daily loss
    session_start_equity = float(state.cfg.initial_cash)

    tick_size = state.eps.eps_div.astype(np.float64)
    loss_limit = float(state.cfg.daily_loss_limit_abs)

    # Causal bar loop (required due portfolio state recursion)
    for t in range(T):
        if t == 0 or state.session_id[t] != state.session_id[t - 1]:
            kill_switch_session = False
            quarantined_asset[:] = False
            # Mark new session baseline at open mark-to-market.
            open_mark = np.where(np.isfinite(state.open_px[t]), state.open_px[t], np.where(np.isfinite(state.close_px[t]), state.close_px[t], 0.0))
            session_start_equity = float(cash + np.sum(pos * open_mark))

        # 1) Execute pending next-open target from prior bar.
        if pending_active:
            t_fill = int(t)
            t_signal = int(pending_signal_t)
            same_session = (t_signal >= 0) and (int(state.session_id[t_fill]) == int(pending_signal_session_id))
            exec_needed = np.abs(pending_target - pos) > eps
            if np.any(exec_needed) and (not same_session):
                a_bad = _first_true_idx(exec_needed)
                target_before_cancel = float(pending_target[a_bad])
                quarantined_asset[a_bad] = True
                pending_target[a_bad] = pos[a_bad]
                pending_active = False
                dump = _build_exec_px_dump(
                    state=state,
                    m3=m3,
                    t_signal=max(t_signal, 0),
                    t_fill=t_fill,
                    a=a_bad,
                    px_source_name="next_open",
                    px_value=float("nan"),
                    target_qty=target_before_cancel,
                    reason_code="NEXT_OPEN_UNAVAILABLE",
                    pending_order_id=f"sig{max(t_signal,0)}_fill{t_fill}_a{a_bad}",
                    quarantine_applied=True,
                    run_context=run_context,
                )
                raise NonFiniteExecutionPriceError(
                    reason_code="NEXT_OPEN_UNAVAILABLE",
                    exec_px_dump=dump,
                    message=f"Next-open unavailable for pending order at a={a_bad}: signal_t={t_signal}, fill_t={t_fill}",
                )

            open_px = state.open_px[t_fill]
            invalid_fill_bar = exec_needed & (~np.asarray(state.bar_valid[t_fill], dtype=bool))
            if np.any(invalid_fill_bar):
                a_bad = _first_true_idx(invalid_fill_bar)
                target_before_cancel = float(pending_target[a_bad])
                quarantined_asset[a_bad] = True
                pending_target[a_bad] = pos[a_bad]
                pending_active = False
                dump = _build_exec_px_dump(
                    state=state,
                    m3=m3,
                    t_signal=max(t_signal, 0),
                    t_fill=t_fill,
                    a=a_bad,
                    px_source_name="next_open",
                    px_value=float(open_px[a_bad]),
                    target_qty=target_before_cancel,
                    reason_code="NEXT_OPEN_UNAVAILABLE",
                    pending_order_id=f"sig{max(t_signal,0)}_fill{t_fill}_a{a_bad}",
                    quarantine_applied=True,
                    run_context=run_context,
                )
                raise NonFiniteExecutionPriceError(
                    reason_code="NEXT_OPEN_UNAVAILABLE",
                    exec_px_dump=dump,
                    message=f"Next-open unavailable at invalid fill bar for pending order at a={a_bad}: signal_t={t_signal}, fill_t={t_fill}",
                )
            bad_px = exec_needed & ((~np.isfinite(open_px)) | (open_px <= 0.0))
            if np.any(bad_px):
                a_bad = _first_true_idx(bad_px)
                target_before_cancel = float(pending_target[a_bad])
                quarantined_asset[a_bad] = True
                pending_target[a_bad] = pos[a_bad]
                pending_active = False
                dump = _build_exec_px_dump(
                    state=state,
                    m3=m3,
                    t_signal=max(t_signal, 0),
                    t_fill=t_fill,
                    a=a_bad,
                    px_source_name="next_open",
                    px_value=float(open_px[a_bad]),
                    target_qty=target_before_cancel,
                    reason_code="NONFINITE_EXEC_PX",
                    pending_order_id=f"sig{max(t_signal,0)}_fill{t_fill}_a{a_bad}",
                    quarantine_applied=True,
                    run_context=run_context,
                )
                raise NonFiniteExecutionPriceError(
                    reason_code="NONFINITE_EXEC_PX",
                    exec_px_dump=dump,
                    message=f"Non-finite/non-positive execution price at a={a_bad}: {float(open_px[a_bad])}",
                )

            rvol_t = state.rvol[t_fill]
            cash, realized, delta_open, cost_open = _execute_to_target(
                pos=pos,
                avg_cost=avg_cost,
                cash=cash,
                realized=realized,
                target=pending_target,
                price=open_px,
                rvol=rvol_t,
                tick_size=tick_size,
                cfg=cfg4,
                strict=cfg4.fail_on_non_finite_input,
                eps=eps,
                px_source_name="next_open",
                dump_builder=lambda a_bad, px_bad, src_name: _build_exec_px_dump(
                    state=state,
                    m3=m3,
                    t_signal=max(t_signal, 0),
                    t_fill=t_fill,
                    a=a_bad,
                    px_source_name=src_name,
                    px_value=px_bad,
                    target_qty=float(pending_target[int(a_bad)]),
                    reason_code="NONFINITE_EXEC_PX",
                    pending_order_id=f"sig{max(t_signal,0)}_fill{t_fill}_a{int(a_bad)}",
                    quarantine_applied=False,
                    run_context=run_context,
                ),
                error_reason_code="NONFINITE_EXEC_PX",
            )
            _accumulate_exec_row(
                filled_row=filled_qty_ta[t],
                exec_px_row=exec_price_ta[t],
                cost_row=trade_cost_ta[t],
                delta=delta_open,
                px=open_px,
                cost=cost_open,
                eps=eps,
            )
            pending_active = False
            pending_signal_t = -1
            pending_signal_session_id = -1

        phase_t = np.int8(state.phase[t])
        tradable = tradable_ta[t].copy()
        dqs_t = dqs_day_ta[t]
        dqs_force_neutral = dqs_t < 0.50
        ib_force_neutral = (~ib_defined_ta[t]) if ib_missing_no_trade else np.zeros(A, dtype=bool)
        force_neutral = dqs_force_neutral | ib_force_neutral
        tradable &= (~force_neutral)
        tradable &= (~quarantined_asset)

        # Pull channels
        ctx = m3.context_tac[t]
        dclip = state.profile_stats[t, :, int(ProfileStatIdx.DCLIP)]
        z_delta = state.profile_stats[t, :, int(ProfileStatIdx.Z_DELTA)]
        gbreak = state.profile_stats[t, :, int(ProfileStatIdx.GBREAK)]
        greject = state.profile_stats[t, :, int(ProfileStatIdx.GREJECT)]
        bo_l = state.scores[t, :, int(ScoreIdx.SCORE_BO_LONG)]
        bo_s = state.scores[t, :, int(ScoreIdx.SCORE_BO_SHORT)]
        rj_l = state.scores[t, :, int(ScoreIdx.SCORE_REJ_LONG)]
        rj_s = state.scores[t, :, int(ScoreIdx.SCORE_REJ_SHORT)]
        close_t = state.close_px[t]
        atr_eff_t = state.atr_floor[t]
        rvol_t = state.rvol[t]
        skew_t = skew_ta[t]

        ctx_x_vah = ctx[:, int(ContextIdx.CTX_X_VAH)]
        ctx_x_val = ctx[:, int(ContextIdx.CTX_X_VAL)]
        ctx_valid_ratio = ctx[:, int(ContextIdx.CTX_VALID_RATIO)]
        ctx_tgs = ctx[:, int(ContextIdx.CTX_TREND_GATE_SPREAD_MEAN)]
        ctx_poc_drift = ctx[:, int(ContextIdx.CTX_POC_DRIFT_X)]
        ctx_poc_vs_prev_va = ctx[:, int(ContextIdx.CTX_POC_VS_PREV_VA)]

        finite_core = (
            np.isfinite(close_t)
            & np.isfinite(atr_eff_t)
            & np.isfinite(bo_l)
            & np.isfinite(bo_s)
            & np.isfinite(rj_l)
            & np.isfinite(rj_s)
            & np.isfinite(gbreak)
            & np.isfinite(greject)
            & np.isfinite(z_delta)
            & np.isfinite(dclip)
            & np.isfinite(ctx_x_vah)
            & np.isfinite(ctx_x_val)
            & np.isfinite(ctx_valid_ratio)
            & np.isfinite(ctx_tgs)
            & np.isfinite(ctx_poc_drift)
            & np.isfinite(ctx_poc_vs_prev_va)
        )
        tradable &= finite_core

        # 2) Regime logic
        trend_up = (
            tradable
            & (ctx_tgs >= float(cfg4.trend_spread_min))
            & (ctx_poc_drift >= float(cfg4.trend_poc_drift_min_abs))
            & (ctx_poc_vs_prev_va > 1.0)
        )
        trend_down = (
            tradable
            & (ctx_tgs >= float(cfg4.trend_spread_min))
            & (ctx_poc_drift <= -float(cfg4.trend_poc_drift_min_abs))
            & (ctx_poc_vs_prev_va < -1.0)
        )
        trend_any = trend_up | trend_down

        p_shape = tradable & np.isfinite(skew_t) & (skew_t <= -float(cfg4.shape_skew_min_abs))
        b_shape = tradable & np.isfinite(skew_t) & (skew_t >= float(cfg4.shape_skew_min_abs))

        dd = np.zeros(A, dtype=bool)
        if np.any(tradable):
            dd_all = _double_distribution_flags(
                vp_ab=state.vp[t],
                x_grid=state.x_grid,
                min_sep_x=float(cfg4.double_dist_sep_x),
                valley_frac=float(cfg4.double_dist_valley_frac),
            )
            dd = tradable & dd_all
        double_up = dd & (dclip >= 0.0)
        double_down = dd & (dclip < 0.0)

        neutral = (
            tradable
            & (np.abs(ctx_poc_drift) <= float(cfg4.neutral_poc_drift_max_abs))
            & (np.abs(ctx_poc_vs_prev_va) <= 1.0)
            & (ctx_valid_ratio >= 0.70)
        )

        regime = np.full(A, np.int8(RegimeIdx.NONE), dtype=np.int8)
        regime[neutral] = np.int8(RegimeIdx.NEUTRAL)
        regime[b_shape] = np.int8(RegimeIdx.B_SHAPE)
        regime[p_shape] = np.int8(RegimeIdx.P_SHAPE)
        regime[trend_any] = np.int8(RegimeIdx.TREND)
        regime[dd] = np.int8(RegimeIdx.DOUBLE_DISTRIBUTION)
        regime_primary_ta[t] = regime

        reg_conf = np.zeros(A, dtype=np.float64)
        reg_conf[neutral] = np.clip(
            1.0 - np.abs(ctx_poc_drift[neutral]) / (float(cfg4.neutral_poc_drift_max_abs) + eps),
            0.0,
            1.0,
        )
        reg_conf[p_shape | b_shape] = np.clip(
            np.abs(skew_t[p_shape | b_shape]) / (2.0 * float(cfg4.shape_skew_min_abs) + eps),
            0.0,
            1.0,
        )
        reg_conf[trend_any] = np.clip(
            0.5 * (np.abs(ctx_poc_drift[trend_any]) / (float(cfg4.trend_poc_drift_min_abs) + eps))
            + 0.5 * (np.abs(ctx_poc_vs_prev_va[trend_any]) / 2.0),
            0.0,
            1.0,
        )
        reg_conf[dd] = 1.0
        regime_confidence_ta[t] = reg_conf

        # 3) Deterministic intents
        bo_l_eff = bo_l * dqs_t
        bo_s_eff = bo_s * dqs_t
        rj_l_eff = rj_l * dqs_t
        rj_s_eff = rj_s * dqs_t

        intent_bo_long = tradable & (trend_up | p_shape | double_up) & (bo_l_eff > float(cfg4.entry_threshold)) & (gbreak > 0.5)
        intent_bo_short = tradable & (trend_down | b_shape | double_down) & (bo_s_eff > float(cfg4.entry_threshold)) & (gbreak > 0.5)
        intent_rej_long = tradable & (neutral | p_shape) & (rj_l_eff > float(cfg4.entry_threshold)) & (greject > 0.5)
        intent_rej_short = tradable & (neutral | b_shape) & (rj_s_eff > float(cfg4.entry_threshold)) & (greject > 0.5)

        intent_long = intent_bo_long | intent_rej_long
        intent_short = intent_bo_short | intent_rej_short

        conv_long = np.maximum(bo_l_eff, rj_l_eff)
        conv_short = np.maximum(bo_s_eff, rj_s_eff)
        both = intent_long & intent_short
        long_better = conv_long > conv_short
        short_better = conv_short > conv_long
        intent_long[both] = long_better[both]
        intent_short[both] = short_better[both]

        # 4) Dynamic exits (corrected: context-only structural stops)
        exit_long = tradable & (pos > 0.0) & ((z_delta < 0.0) | (close_t < ctx_x_val))
        exit_short = tradable & (pos < 0.0) & ((z_delta > 0.0) | (close_t > ctx_x_vah))
        exit_any = exit_long | exit_short

        # Exits override entries.
        intent_long = intent_long & (~exit_any)
        intent_short = intent_short & (~exit_any)

        # Kill switch blocks new entries.
        if kill_switch_session:
            intent_long[:] = False
            intent_short[:] = False
        if np.any(force_neutral):
            intent_long[force_neutral] = False
            intent_short[force_neutral] = False

        intent_long_ta[t] = intent_long
        intent_short_ta[t] = intent_short

        # Build target at bar t
        target = pos.copy()
        target[exit_any] = 0.0
        target[force_neutral] = 0.0

        # 5) Intraday allocation (LIVE only, next-open fill)
        if phase_t == phase_live and (not kill_switch_session):
            candidates = tradable & (intent_long ^ intent_short)
            # Top-K policy is strict: rebalance tradable, non-exit names to zero
            # unless explicitly selected by current conviction ranking.
            rebalance_mask = tradable & (~exit_any)
            target[rebalance_mask] = 0.0
            if np.any(candidates):
                conv_mag = np.maximum(conv_long, conv_short)
                conv_mag = np.where(candidates, conv_mag, -np.inf)
                k = int(min(max(cfg4.top_k_intraday, 0), int(np.sum(candidates))))
                if k > 0:
                    idx_all = np.arange(A, dtype=np.int64)
                    # Top-k by conviction, deterministic tie-break by lower index.
                    sel_pool = np.where(candidates)[0]
                    sel_scores = conv_mag[sel_pool]
                    order = np.lexsort((sel_pool, -sel_scores))
                    top = sel_pool[order[:k]]

                    # --- Institutional numeric guard: dynamic ATR floor + capped inverse-ATR weights ---
                    px_sel_for_floor = np.maximum(close_t[top], tick_size[top])
                    atr_floor_tick = 2.0 * tick_size[top]
                    atr_floor_rel = 1e-5 * px_sel_for_floor
                    atr_floor = np.maximum(eps, np.maximum(atr_floor_tick, atr_floor_rel))
                    atr_sel = np.maximum(atr_eff_t[top], atr_floor)

                    w_raw = 1.0 / atr_sel
                    med_atr = float(np.median(atr_sel)) if atr_sel.size else 0.0
                    base = max(med_atr, float(np.min(atr_floor)) if atr_floor.size else eps)
                    w_cap = (1.0 / base) * 50.0
                    w = np.minimum(w_raw, w_cap)

                    w_sum = float(np.sum(w))
                    if np.isfinite(w_sum) and (w_sum > eps):
                        w = w / w_sum
                        equity_now = float(cash + np.sum(pos * close_t))
                        gross_budget = max(0.0, equity_now * float(state.cfg.intraday_leverage_max))
                        raw_notional = gross_budget * w
                        cap_notional = equity_now * float(cfg4.max_asset_cap_frac)
                        alloc_notional = np.minimum(raw_notional, cap_notional)

                        # NaN Guard 1: alloc_notional
                        if not np.all(np.isfinite(alloc_notional)):
                            alloc_notional = np.where(np.isfinite(alloc_notional), alloc_notional, 0.0)
                        alloc_notional = np.maximum(alloc_notional, 0.0)

                        tot_alloc = float(np.sum(alloc_notional))
                        if np.isfinite(tot_alloc) and (tot_alloc > gross_budget + eps):
                            alloc_notional *= gross_budget / (tot_alloc + eps)

                        sign_sel = np.where(intent_long[top], 1.0, -1.0)
                        px_sel = np.maximum(close_t[top], tick_size[top])
                        desired_qty = sign_sel * (alloc_notional / (px_sel + eps))

                        # NaN Guard 2: desired_qty
                        if not np.all(np.isfinite(desired_qty)):
                            desired_qty = np.where(np.isfinite(desired_qty), desired_qty, pos[top])

                        # Turnover cap (notional change cap per bar).
                        delta_qty = desired_qty - pos[top]
                        delta_notional = np.abs(delta_qty) * px_sel
                        total_delta_notional = float(np.sum(delta_notional))
                        max_turn = max(0.0, equity_now * float(cfg4.max_turnover_frac_per_bar))

                        # NaN Guard 3: turnover logic (keep scope linear)
                        if (not np.isfinite(total_delta_notional)) or (total_delta_notional <= eps) or (not np.isfinite(max_turn)):
                            desired_qty = pos[top]
                        elif total_delta_notional > max_turn + eps:
                            scale = max_turn / total_delta_notional
                            if np.isfinite(scale) and (scale >= 0.0):
                                desired_qty = pos[top] + delta_qty * scale
                            else:
                                desired_qty = pos[top]
                    else:
                        # Unsafe weight sum -> no-op
                        desired_qty = pos[top]

                    target[top] = desired_qty

        # 6) Overnight select / flatten phases
        if phase_t == phase_os:
            # Force flatten at close first.
            zero_target = np.zeros(A, dtype=np.float64)
            cash, realized, delta_close_flat, cost_close_flat = _execute_to_target(
                pos=pos,
                avg_cost=avg_cost,
                cash=cash,
                realized=realized,
                target=zero_target,
                price=close_t,
                rvol=rvol_t,
                tick_size=tick_size,
                cfg=cfg4,
                strict=cfg4.fail_on_non_finite_input,
                eps=eps,
                px_source_name="close_flatten",
                dump_builder=lambda a_bad, px_bad, src_name: _build_exec_px_dump(
                    state=state,
                    m3=m3,
                    t_signal=t,
                    t_fill=t,
                    a=a_bad,
                    px_source_name=src_name,
                    px_value=px_bad,
                    target_qty=float(zero_target[int(a_bad)]),
                    reason_code="NONFINITE_EXEC_PX",
                    pending_order_id=f"samebar_close_flat_t{t}_a{int(a_bad)}",
                    quarantine_applied=False,
                    run_context=run_context,
                ),
            )
            _accumulate_exec_row(
                filled_row=filled_qty_ta[t],
                exec_px_row=exec_price_ta[t],
                cost_row=trade_cost_ta[t],
                delta=delta_close_flat,
                px=close_t,
                cost=cost_close_flat,
                eps=eps,
            )

            # Multiplicative OCS
            structural_w = np.zeros(A, dtype=np.float64)
            structural_w[p_shape | b_shape] = 1.5
            structural_w[trend_any] = np.maximum(structural_w[trend_any], 1.2)

            ocs = structural_w * np.abs(dclip) * np.abs(z_delta) * np.maximum(rvol_t, 0.0)
            ocs = np.where(tradable, ocs, 0.0)
            overnight_score_ta[t] = ocs

            z_abs = np.abs(z_delta)
            win = _weighted_argmax_with_tie(ocs, z_abs)
            max_ocs = float(ocs[win]) if win >= 0 else 0.0

            if (win < 0) or (max_ocs < float(cfg4.overnight_min_conviction)):
                if bool(cfg4.allow_cash_overnight):
                    win = -1
            overnight_winner_t[t] = win

            if win >= 0:
                reg_win = int(regime[win])
                if reg_win == int(RegimeIdx.P_SHAPE):
                    side = 1.0
                elif reg_win == int(RegimeIdx.B_SHAPE):
                    side = -1.0
                else:
                    side = 1.0 if dclip[win] >= 0.0 else -1.0

                equity_now = float(cash + np.sum(pos * close_t))
                overnight_notional = max(0.0, equity_now * float(state.cfg.overnight_leverage))
                px = max(float(close_t[win]), float(tick_size[win]))
                one_target = np.zeros(A, dtype=np.float64)
                one_target[win] = side * (overnight_notional / (px + eps))

                cash, realized, delta_close_on, cost_close_on = _execute_to_target(
                    pos=pos,
                    avg_cost=avg_cost,
                    cash=cash,
                    realized=realized,
                    target=one_target,
                    price=close_t,
                    rvol=rvol_t,
                    tick_size=tick_size,
                    cfg=cfg4,
                    strict=cfg4.fail_on_non_finite_input,
                    eps=eps,
                    px_source_name="close_overnight_select",
                    dump_builder=lambda a_bad, px_bad, src_name: _build_exec_px_dump(
                        state=state,
                        m3=m3,
                        t_signal=t,
                        t_fill=t,
                        a=a_bad,
                        px_source_name=src_name,
                        px_value=px_bad,
                        target_qty=float(one_target[int(a_bad)]),
                        reason_code="NONFINITE_EXEC_PX",
                        pending_order_id=f"samebar_overnight_t{t}_a{int(a_bad)}",
                        quarantine_applied=False,
                        run_context=run_context,
                    ),
                )
                _accumulate_exec_row(
                    filled_row=filled_qty_ta[t],
                    exec_px_row=exec_price_ta[t],
                    cost_row=trade_cost_ta[t],
                    delta=delta_close_on,
                    px=close_t,
                    cost=cost_close_on,
                    eps=eps,
                )
                overnight_idx = win
            else:
                overnight_idx = -1

            target = pos.copy()
            pending_active = False

        elif phase_t == phase_flat:
            # No new entries in FLATTEN phase; maintain current holdings only.
            target = pos.copy()
            pending_active = False

        elif phase_t == phase_live:
            # Schedule target for next open only if the structural next bar remains in-session.
            # This is a calendar/clock guard (no price peeking) that prevents orphan pending orders
            # across short-session and holiday boundaries.
            has_next_bar = (t + 1) < T
            same_session_next = has_next_bar and (int(state.session_id[t + 1]) == int(state.session_id[t]))
            if same_session_next:
                pending_target = target.copy()
                pending_active = True
                pending_signal_t = int(t)
                pending_signal_session_id = int(state.session_id[t])
            else:
                blocked = np.abs(target - pos) > eps
                if np.any(blocked):
                    target[blocked] = pos[blocked]
                    quarantined_asset[blocked] = True
                pending_active = False
                pending_signal_t = -1
                pending_signal_session_id = -1
        else:
            # WARMUP or unknown: no new orders.
            target = pos.copy()
            pending_active = False
            pending_signal_t = -1
            pending_signal_session_id = -1

        target_qty_ta[t] = target

        # Order tensor mutation for this bar.
        delta_target = target - pos
        side = np.sign(delta_target).astype(np.int8)
        flags = np.zeros(A, dtype=np.uint16)
        entry = np.abs(target) > np.abs(pos) + eps
        exit_ = np.abs(target) + eps < np.abs(pos)
        flat_ = (np.abs(target) <= eps) & (np.abs(pos) > eps)
        flags[entry] |= np.uint16(OrderFlagBit.ENTRY)
        flags[exit_] |= np.uint16(OrderFlagBit.EXIT)
        flags[flat_] |= np.uint16(OrderFlagBit.FLATTEN)
        if phase_t == phase_os:
            flags |= np.uint16(OrderFlagBit.MOC_EXEC)
            cands = overnight_score_ta[t] > 0.0
            flags[cands] |= np.uint16(OrderFlagBit.OVERNIGHT_CANDIDATE)
            if overnight_winner_t[t] >= 0:
                flags[int(overnight_winner_t[t])] |= np.uint16(OrderFlagBit.OVERNIGHT_SELECTED)

        state.orders[t, :, int(OrderIdx.TARGET_QTY)] = target
        state.orders[t, :, int(OrderIdx.CONVICTION)] = np.maximum(conv_long, conv_short)
        state.orders[t, :, int(OrderIdx.MAX_SLIP_BPS)] = _slippage_bps_from_rvol(np.maximum(rvol_t, 0.0), cfg4)
        state.order_side[t] = side
        state.order_flags[t] = flags

        # Mark to close.
        mtm_close = np.where(np.isfinite(close_t), close_t, np.where(np.isfinite(state.open_px[t]), state.open_px[t], 0.0))
        market_value = float(np.sum(pos * mtm_close))
        equity_now = float(cash + market_value)
        unreal = float(np.sum(pos * (mtm_close - avg_cost)))
        margin_used = float(np.sum(np.abs(pos * mtm_close)))
        buying_power = float(equity_now * float(state.leverage_limit[t]) - margin_used)
        daily_loss = max(0.0, float(session_start_equity - equity_now))
        breached = bool(daily_loss >= loss_limit)

        # Hard kill switch: immediate same-bar flatten at close.
        if breached and bool(cfg4.hard_kill_on_daily_loss_breach):
            if np.any(np.abs(pos) > eps):
                zero_target = np.zeros(A, dtype=np.float64)
                cash, realized, delta_kill, cost_kill = _execute_to_target(
                    pos=pos,
                    avg_cost=avg_cost,
                    cash=cash,
                    realized=realized,
                    target=zero_target,
                    price=mtm_close,
                    rvol=rvol_t,
                    tick_size=tick_size,
                    cfg=cfg4,
                    strict=cfg4.fail_on_non_finite_input,
                    eps=eps,
                    px_source_name="kill_switch_close",
                    dump_builder=lambda a_bad, px_bad, src_name: _build_exec_px_dump(
                        state=state,
                        m3=m3,
                        t_signal=t,
                        t_fill=t,
                        a=a_bad,
                        px_source_name=src_name,
                        px_value=px_bad,
                        target_qty=float(zero_target[int(a_bad)]),
                        reason_code="NONFINITE_EXEC_PX",
                        pending_order_id=f"samebar_kill_t{t}_a{int(a_bad)}",
                        quarantine_applied=False,
                        run_context=run_context,
                    ),
                )
                _accumulate_exec_row(
                    filled_row=filled_qty_ta[t],
                    exec_px_row=exec_price_ta[t],
                    cost_row=trade_cost_ta[t],
                    delta=delta_kill,
                    px=mtm_close,
                    cost=cost_kill,
                    eps=eps,
                )
                state.order_flags[t] |= np.uint16(OrderFlagBit.KILL_SWITCH | OrderFlagBit.FLATTEN | OrderFlagBit.MOC_EXEC)
                target_qty_ta[t] = pos.copy()
            kill_switch_session = True
            pending_active = False
            overnight_idx = -1

            market_value = float(np.sum(pos * mtm_close))
            equity_now = float(cash + market_value)
            unreal = float(np.sum(pos * (mtm_close - avg_cost)))
            margin_used = float(np.sum(np.abs(pos * mtm_close)))
            buying_power = float(equity_now * float(state.leverage_limit[t]) - margin_used)
            daily_loss = max(0.0, float(session_start_equity - equity_now))
            breached = bool(daily_loss >= loss_limit)

        kill_switch_t[t] = kill_switch_session

        if overnight_idx >= 0 and abs(pos[overnight_idx]) <= eps:
            overnight_idx = -1

        # State mutation for row t.
        state.position_qty[t] = pos
        state.available_cash[t] = cash
        state.equity[t] = equity_now
        state.margin_used[t] = margin_used
        state.buying_power[t] = buying_power
        state.realized_pnl[t] = realized
        state.unrealized_pnl[t] = unreal
        state.daily_loss[t] = daily_loss
        state.daily_loss_breach_flag[t] = np.int8(1 if breached else 0)
        state.overnight_mask[t] = 0
        if overnight_idx >= 0:
            state.overnight_mask[t, overnight_idx] = 1

        # Ensure no stale NaN in exec price row for no-fill assets.
        nofill = np.abs(filled_qty_ta[t]) <= eps
        exec_price_ta[t, nofill] = np.nan

    out = Module4Output(
        regime_primary_ta=regime_primary_ta,
        regime_confidence_ta=regime_confidence_ta,
        intent_long_ta=intent_long_ta,
        intent_short_ta=intent_short_ta,
        target_qty_ta=target_qty_ta,
        filled_qty_ta=filled_qty_ta,
        exec_price_ta=exec_price_ta,
        trade_cost_ta=trade_cost_ta,
        overnight_score_ta=overnight_score_ta,
        overnight_winner_t=overnight_winner_t,
        kill_switch_t=kill_switch_t,
    )

    if cfg4.fail_on_non_finite_output:
        valid_out = tradable_ta
        _assert_finite_masked("regime_confidence_ta", out.regime_confidence_ta, valid_out)
        _assert_finite_masked("target_qty_ta", out.target_qty_ta, np.ones((T, A), dtype=bool))
        _assert_finite_masked("filled_qty_ta", out.filled_qty_ta, np.ones((T, A), dtype=bool))
        _assert_finite_masked("trade_cost_ta", out.trade_cost_ta, np.ones((T, A), dtype=bool))
        _assert_finite_masked("state.position_qty", state.position_qty, np.ones((T, A), dtype=bool))
        _assert_finite_masked("state.equity", state.equity, np.ones(T, dtype=bool))
        _assert_finite_masked("state.margin_used", state.margin_used, np.ones(T, dtype=bool))
        _assert_finite_masked("state.buying_power", state.buying_power, np.ones(T, dtype=bool))
    assert_float64("module4.output.exec_price_ta", out.exec_price_ta)
    assert_float64("module4.output.trade_cost_ta", out.trade_cost_ta)

    # Reuse Module 1 hard invariants for leverage/overnight integrity.
    validate_state_hard(state)
    return out


def run_module4_signal_funnel(
    state: TensorState,
    m3: Module3Output,
    cfg4: Module4Config,
) -> Module4SignalOutput:
    """
    Signal-only Module4 API for canonical orchestration.
    This function does not mutate execution state.
    """
    from module4.contracts import build_module4_input_contracts
    from module4.strategy_funnel_engine import run_module4_funnel

    T = int(state.cfg.T)
    A = int(state.cfg.A)
    _assert_shape("scores", state.scores, (T, A, int(ScoreIdx.N_FIELDS)))
    _assert_shape("profile_stats", state.profile_stats, (T, A, int(ProfileStatIdx.N_FIELDS)))
    assert_float64("module4.signal_input.scores", state.scores)
    assert_float64("module4.signal_input.profile_stats", state.profile_stats)

    if getattr(m3, "structure_tensor", None) is not None:
        structure_tensor = np.asarray(m3.structure_tensor, dtype=np.float64)
    elif getattr(m3, "block_features_tak", None) is not None:
        structure_tensor = np.swapaxes(np.asarray(m3.block_features_tak, dtype=np.float64), 0, 1)[:, :, :, None]
    else:
        raise RuntimeError("MODULE4_BRIDGE_MISSING_STRUCTURE_TENSOR")

    if getattr(m3, "context_tensor", None) is not None:
        context_tensor = np.asarray(m3.context_tensor, dtype=np.float64)
    elif getattr(m3, "context_tac", None) is not None:
        context_tensor = np.swapaxes(np.asarray(m3.context_tac, dtype=np.float64), 0, 1)[:, :, :, None]
    else:
        raise RuntimeError("MODULE4_BRIDGE_MISSING_CONTEXT_TENSOR")

    degraded_mode_mask_at = np.zeros((A, T), dtype=bool)
    # Canonical neutral alpha bridge tensor for legacy callers.
    alpha_signal_tensor = np.zeros((A, T, 1), dtype=np.float64)

    if getattr(m3, "profile_fingerprint_tensor", None) is not None:
        profile_fingerprint_tensor = np.asarray(m3.profile_fingerprint_tensor, dtype=np.float64)
    else:
        if not bool(cfg4.enable_degraded_bridge_mode):
            raise RuntimeError("MODULE4_BRIDGE_MISSING_FINGERPRINT_TENSOR")
        profile_fingerprint_tensor = np.zeros((A, T, 1, 1), dtype=np.float64)
        degraded_mode_mask_at |= True

    if getattr(m3, "profile_regime_tensor", None) is not None:
        profile_regime_tensor = np.asarray(m3.profile_regime_tensor, dtype=np.float64)
    else:
        if not bool(cfg4.enable_degraded_bridge_mode):
            raise RuntimeError("MODULE4_BRIDGE_MISSING_PROFILE_REGIME_TENSOR")
        profile_regime_tensor = np.zeros((A, T, 1, 1), dtype=np.float64)
        degraded_mode_mask_at |= True

    if getattr(m3, "context_valid_ta", None) is not None:
        context_valid_ta = np.asarray(m3.context_valid_ta, dtype=bool)
    else:
        context_valid_ta = np.all(np.isfinite(np.asarray(context_tensor, dtype=np.float64)), axis=(2, 3)).T
    _assert_shape("context_valid_ta", context_valid_ta, (T, A))
    tradable_mask = np.asarray(state.bar_valid, dtype=bool) & context_valid_ta

    source_time_index_at = None
    if getattr(m3, "context_source_index_atw", None) is not None:
        src_atw = np.asarray(m3.context_source_index_atw, dtype=np.int64)
        if src_atw.ndim != 3:
            raise RuntimeError("MODULE4_BRIDGE_BAD_CONTEXT_SOURCE_INDEX_ATW")
        t_grid = np.broadcast_to(np.arange(T, dtype=np.int64)[None, :, None], src_atw.shape)
        if np.any(src_atw > t_grid):
            bad = np.argwhere(src_atw > t_grid)[0]
            raise RuntimeError(
                "CAUSALITY_VIOLATION: context_source_index_atw contains forward-looking index "
                f"at a={int(bad[0])}, t={int(bad[1])}, w={int(bad[2])}"
            )
        source_time_index_at = np.max(src_atw, axis=2)
    elif getattr(m3, "context_source_t_index_ta", None) is not None:
        source_time_index_at = np.swapaxes(np.asarray(m3.context_source_t_index_ta, dtype=np.int64), 0, 1)

    contracts = build_module4_input_contracts(
        alpha_signal_tensor=alpha_signal_tensor,
        score_tensor=np.swapaxes(np.asarray(state.scores, dtype=np.float64), 0, 1),
        profile_stat_tensor=np.swapaxes(np.asarray(state.profile_stats, dtype=np.float64), 0, 1),
        structure_tensor=structure_tensor,
        context_tensor=context_tensor,
        profile_fingerprint_tensor=profile_fingerprint_tensor,
        profile_regime_tensor=profile_regime_tensor,
        tradable_mask=np.swapaxes(np.asarray(tradable_mask, dtype=bool), 0, 1),
        phase_code=np.asarray(state.phase, dtype=np.int64),
        asset_enabled_mask=np.asarray(getattr(state, "asset_enabled_mask", np.ones(A, dtype=bool)), dtype=bool),
        source_time_index_at=source_time_index_at,
        fail_on_non_finite_input=bool(cfg4.fail_on_non_finite_input),
    )
    decision = run_module4_funnel(
        contracts,
        cfg4,
        degraded_mode_mask_at=degraded_mode_mask_at,
    )
    return _decision_to_signal_output(decision)


if __name__ == "__main__":
    log_event(get_logger("module4"), "INFO", "module4_ready", event_type="module4_ready")
