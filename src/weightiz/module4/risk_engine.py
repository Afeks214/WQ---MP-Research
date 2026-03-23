from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from weightiz.shared.validation.dtype_guard import assert_float64

REASON_COST_MODEL_VIOLATION = "cost_model_violation"
TRADING_DAYS_PER_YEAR = 252.0
EXECUTION_COST_MODEL_STATIC = "static_mid_rvol_legacy"
EXECUTION_COST_MODEL_DYNAMIC = "dynamic_bucketed_v1"
TARGET_SIGNAL_MODE_QUANTITY = "quantity_target"
TARGET_SIGNAL_MODE_WEIGHT = "weight_target"


@dataclass(frozen=True)
class CostConfig:
    commission_per_share: float = 0.0
    finra_taf_per_share_sell: float = 0.0
    reg_fee_per_share_sell: float | None = None
    sec_fee_per_dollar_sell: float = 0.0
    short_borrow_apr: float = 0.0
    locate_fee_per_share_short_entry: float = 0.0
    slippage_bps: float = 0.0
    debit_apr: float = 0.0

    def __post_init__(self) -> None:
        if self.reg_fee_per_share_sell is None:
            object.__setattr__(self, "reg_fee_per_share_sell", float(self.finra_taf_per_share_sell))
        object.__setattr__(self, "finra_taf_per_share_sell", float(self.reg_fee_per_share_sell))


@dataclass(frozen=True)
class ExecutionRealismConfig:
    cost_model: str = EXECUTION_COST_MODEL_STATIC
    max_volume_participation: float = 1.0
    # RVOL buckets model liquidity scarcity, not volatility stress:
    # lower RVOL defaults to wider slippage and higher RVOL to tighter slippage.
    low_rvol_slippage_bps: float = 3.0
    mid_rvol_slippage_bps: float = 2.0
    high_rvol_slippage_bps: float = 1.5
    spread_tick_mult: float = 1.5
    open_bucket_minutes: int = 30
    close_bucket_minutes: int = 30
    open_slippage_mult: float = 1.25
    mid_slippage_mult: float = 1.0
    close_slippage_mult: float = 1.15
    participation_slippage_coeff: float = 1.0
    dynamic_slippage_bps_cap: float = 50.0
    rth_open_minute: int = 570
    flat_time_minute: int = 945
    rvol_ta: np.ndarray | None = None
    tick_size_a: np.ndarray | None = None
    minute_of_day_t: np.ndarray | None = None

    def __post_init__(self) -> None:
        if str(self.cost_model).strip() not in {EXECUTION_COST_MODEL_STATIC, EXECUTION_COST_MODEL_DYNAMIC}:
            raise RuntimeError(f"unsupported execution cost model: {self.cost_model!r}")
        if not (0.0 < float(self.max_volume_participation) <= 1.0):
            raise RuntimeError("execution_max_volume_participation must be in (0, 1]")
        if int(self.open_bucket_minutes) < 0:
            raise RuntimeError("execution_open_bucket_minutes must be >= 0")
        if int(self.close_bucket_minutes) < 0:
            raise RuntimeError("execution_close_bucket_minutes must be >= 0")
        if float(self.open_slippage_mult) < 0.0:
            raise RuntimeError("execution_open_slippage_mult must be >= 0")
        if float(self.mid_slippage_mult) < 0.0:
            raise RuntimeError("execution_mid_slippage_mult must be >= 0")
        if float(self.close_slippage_mult) < 0.0:
            raise RuntimeError("execution_close_slippage_mult must be >= 0")
        if float(self.participation_slippage_coeff) < 0.0:
            raise RuntimeError("execution_participation_slippage_coeff must be >= 0")
        if float(self.dynamic_slippage_bps_cap) < 0.0:
            raise RuntimeError("execution_dynamic_slippage_bps_cap must be >= 0")


@dataclass(frozen=True)
class RiskConfig:
    max_position_buying_power_frac: float = 0.25
    overnight_exposure_equity_mult: float = 2.0
    daily_loss_limit_frac: float = 0.10
    account_disable_equity: float = 1000.0
    daily_max_loss_frac: float | None = None

    def __post_init__(self) -> None:
        if self.daily_max_loss_frac is None:
            object.__setattr__(self, "daily_max_loss_frac", float(self.daily_loss_limit_frac))
        object.__setattr__(self, "daily_loss_limit_frac", float(self.daily_max_loss_frac))


@dataclass(frozen=True)
class SimulationResult:
    equity_curve: np.ndarray
    daily_returns: np.ndarray
    filled_qty_ta: np.ndarray
    exec_price_ta: np.ndarray
    trade_cost_ta: np.ndarray
    position_qty_ta: np.ndarray
    margin_used_t: np.ndarray
    buying_power_t: np.ndarray
    daily_loss_t: np.ndarray
    trades: int
    final_equity: float
    max_drawdown: float
    sharpe: float
    sortino: float
    gross_exposure_peak: float = 0.0
    desired_qty_ta: np.ndarray | None = None
    unfilled_qty_ta: np.ndarray | None = None
    fill_cap_qty_ta: np.ndarray | None = None
    participation_rate_ta: np.ndarray | None = None
    fill_capped_flag_ta: np.ndarray | None = None
    fill_rejected_flag_ta: np.ndarray | None = None
    slippage_cost_ta: np.ndarray | None = None
    commission_cost_ta: np.ndarray | None = None
    regulatory_cost_ta: np.ndarray | None = None
    locate_cost_ta: np.ndarray | None = None
    borrow_cost_t: np.ndarray | None = None
    debit_cost_t: np.ndarray | None = None
    session_start_equity_t: np.ndarray | None = None
    execution_cost_model: str = EXECUTION_COST_MODEL_STATIC
    trade_log: list[dict[str, Any]] | None = None
    per_asset_cumret: dict[str, float] | None = None
    execution_diagnostics: dict[str, Any] | None = None


def _max_drawdown(eq: np.ndarray) -> float:
    if eq.size == 0:
        return 0.0
    roll_max = np.maximum.accumulate(eq)
    dd = (roll_max - eq) / np.maximum(roll_max, 1e-12)
    return float(np.max(dd))


def _trade_costs(
    *,
    shares: int,
    side: int,
    is_short_entry: bool,
    cost_cfg: CostConfig,
    price: float = 1.0,
) -> float:
    # Legacy fail-closed compatibility shim for the historical cost-model API.
    if int(shares) < 0:
        raise RuntimeError(REASON_COST_MODEL_VIOLATION)
    if int(side) not in (-1, 1):
        raise RuntimeError(REASON_COST_MODEL_VIOLATION)
    if not np.isfinite(float(price)) or float(price) <= 0.0:
        raise RuntimeError(REASON_COST_MODEL_VIOLATION)

    qty = float(int(shares))
    notional = qty * float(price)
    total = qty * float(cost_cfg.commission_per_share)
    if int(side) < 0:
        total += qty * float(cost_cfg.finra_taf_per_share_sell)
        total += notional * float(cost_cfg.sec_fee_per_dollar_sell)
        if bool(is_short_entry):
            total += qty * float(cost_cfg.locate_fee_per_share_short_entry)
    return float(total)


def _legacy_exec_buy(price: float) -> float:
    return float(price + 0.01)


def _legacy_exec_sell(price: float) -> float:
    return float(price - 0.01)


def _session_financing_cost(
    *,
    qty: np.ndarray,
    price: np.ndarray,
    cash: float,
    cost_cfg: CostConfig,
) -> float:
    borrow_cost, debit_cost = _session_financing_cost_breakdown(
        qty=qty,
        price=price,
        cash=cash,
        cost_cfg=cost_cfg,
    )
    return float(borrow_cost + debit_cost)


def _session_financing_cost_breakdown(
    *,
    qty: np.ndarray,
    price: np.ndarray,
    cash: float,
    cost_cfg: CostConfig,
) -> tuple[float, float]:
    short_qty = np.where(qty < 0.0, -qty, 0.0)
    short_notional = float(np.sum(short_qty * price))
    borrow_cost = short_notional * float(cost_cfg.short_borrow_apr) / TRADING_DAYS_PER_YEAR
    debit_cost = max(-float(cash), 0.0) * float(cost_cfg.debit_apr) / TRADING_DAYS_PER_YEAR
    return float(borrow_cost), float(debit_cost)


def _validate_execution_realism_inputs(
    *,
    execution_realism: ExecutionRealismConfig | None,
    t_count: int,
    a_count: int,
) -> ExecutionRealismConfig:
    realism = execution_realism if execution_realism is not None else ExecutionRealismConfig()
    if str(realism.cost_model).strip() != EXECUTION_COST_MODEL_DYNAMIC:
        return realism

    if realism.rvol_ta is None:
        raise RuntimeError("dynamic execution cost model requires rvol_ta")
    if realism.tick_size_a is None:
        raise RuntimeError("dynamic execution cost model requires tick_size_a")
    if realism.minute_of_day_t is None:
        raise RuntimeError("dynamic execution cost model requires minute_of_day_t")

    rvol_ta = np.asarray(realism.rvol_ta)
    assert_float64("risk_engine.execution_realism.rvol_ta", rvol_ta)
    if rvol_ta.shape != (t_count, a_count):
        raise RuntimeError(
            f"dynamic execution cost model rvol_ta shape mismatch: got {rvol_ta.shape}, expected {(t_count, a_count)}"
        )
    tick_size_a = np.asarray(realism.tick_size_a)
    assert_float64("risk_engine.execution_realism.tick_size_a", tick_size_a)
    if tick_size_a.shape != (a_count,):
        raise RuntimeError(
            f"dynamic execution cost model tick_size_a shape mismatch: got {tick_size_a.shape}, expected {(a_count,)}"
        )
    minute_of_day_t = np.asarray(realism.minute_of_day_t)
    if minute_of_day_t.shape != (t_count,):
        raise RuntimeError(
            f"dynamic execution cost model minute_of_day_t shape mismatch: got {minute_of_day_t.shape}, expected {(t_count,)}"
        )
    if np.any(~np.isfinite(tick_size_a)) or np.any(tick_size_a <= 0.0):
        raise RuntimeError("dynamic execution cost model tick_size_a must be finite and > 0")
    return realism


def _bucket_slippage_bps(
    *,
    rvol: float,
    execution_realism: ExecutionRealismConfig,
) -> float:
    # These buckets intentionally treat RVOL as a liquidity proxy.
    if rvol < 0.8:
        return float(execution_realism.low_rvol_slippage_bps)
    if rvol < 1.5:
        return float(execution_realism.mid_rvol_slippage_bps)
    return float(execution_realism.high_rvol_slippage_bps)


def _session_slippage_mult(
    *,
    minute_of_day: int,
    execution_realism: ExecutionRealismConfig,
) -> float:
    open_cut = int(execution_realism.rth_open_minute) + int(execution_realism.open_bucket_minutes)
    close_cut = int(execution_realism.flat_time_minute) - int(execution_realism.close_bucket_minutes)
    if minute_of_day < open_cut:
        return float(execution_realism.open_slippage_mult)
    if minute_of_day >= close_cut:
        return float(execution_realism.close_slippage_mult)
    return float(execution_realism.mid_slippage_mult)


def _resolve_target_signal_mode(target_signal_mode: str) -> str:
    mode = str(target_signal_mode).strip().lower()
    if mode not in {TARGET_SIGNAL_MODE_QUANTITY, TARGET_SIGNAL_MODE_WEIGHT}:
        raise RuntimeError(
            "target_signal_mode must be one of: "
            f"{TARGET_SIGNAL_MODE_QUANTITY}, {TARGET_SIGNAL_MODE_WEIGHT}"
        )
    return mode


def _hard_stop_breached(
    *,
    qty: float,
    entry_price: float,
    mark_price: float,
    hard_stop_loss_frac: float,
) -> bool:
    if abs(float(qty)) <= 0.0:
        return False
    if float(hard_stop_loss_frac) <= 0.0:
        return False
    if (not np.isfinite(float(entry_price))) or float(entry_price) <= 0.0:
        return False
    if (not np.isfinite(float(mark_price))) or float(mark_price) <= 0.0:
        return False
    pnl_frac = np.sign(float(qty)) * (float(mark_price) - float(entry_price)) / float(entry_price)
    return bool(pnl_frac <= -float(hard_stop_loss_frac))


def _resolve_target_qty(
    *,
    raw_target: float,
    price: float,
    current_qty: float,
    equity: float,
    target_signal_mode: str,
    target_weight_deadband_frac: float,
) -> float:
    if str(target_signal_mode) == TARGET_SIGNAL_MODE_WEIGHT:
        if not np.isfinite(float(raw_target)):
            raise RuntimeError("weight-target mode requires finite target weights")
        if (not np.isfinite(float(price))) or float(price) <= 0.0:
            raise RuntimeError("weight-target mode requires finite positive prices")
        if (not np.isfinite(float(equity))) or float(equity) <= 0.0:
            return float(current_qty)
        target_qty = float(np.rint(float(raw_target) * float(equity) / float(price)))
    else:
        target_qty = float(np.trunc(float(raw_target)))

    current = float(current_qty)
    deadband = float(target_weight_deadband_frac)
    if deadband > 0.0:
        if abs(current) > 0.0:
            if abs(target_qty - current) < deadband * abs(current):
                return current
    return target_qty


def _dynamic_slippage_bps(
    *,
    t_index: int,
    asset_index: int,
    price: float,
    participation_rate: float,
    rvol_ta: np.ndarray,
    tick_size_a: np.ndarray,
    minute_of_day_t: np.ndarray,
    execution_realism: ExecutionRealismConfig,
) -> float:
    rvol = float(rvol_ta[t_index, asset_index])
    tick_size = float(tick_size_a[asset_index])
    minute_of_day = int(minute_of_day_t[t_index])
    if not np.isfinite(rvol):
        raise RuntimeError("dynamic execution cost model requires finite rvol on traded bars")
    if not np.isfinite(tick_size) or tick_size <= 0.0:
        raise RuntimeError("dynamic execution cost model requires finite positive tick_size on traded bars")
    half_spread_bps = 0.5 * float(execution_realism.spread_tick_mult) * tick_size / max(float(price), tick_size) * 1.0e4
    session_mult = _session_slippage_mult(minute_of_day=minute_of_day, execution_realism=execution_realism)
    safe_participation = float(max(participation_rate, 0.0)) if np.isfinite(float(participation_rate)) else 0.0
    raw_bps = (
        _bucket_slippage_bps(rvol=rvol, execution_realism=execution_realism) + half_spread_bps
    ) * session_mult * (1.0 + float(execution_realism.participation_slippage_coeff) * safe_participation)
    return float(min(float(raw_bps), float(execution_realism.dynamic_slippage_bps_cap)))


def simulate_portfolio_from_signals(
    close_px_ta: np.ndarray,
    target_qty_ta: np.ndarray,
    initial_cash: float,
    cost_cfg: CostConfig,
    risk_cfg: RiskConfig,
    session_id_t: np.ndarray | None = None,
    volume_ta: np.ndarray | None = None,
    execution_realism: ExecutionRealismConfig | None = None,
    *,
    target_signal_mode: str = TARGET_SIGNAL_MODE_QUANTITY,
    target_weight_deadband_frac: float = 0.0,
    min_holding_minutes: int = 0,
    hard_stop_loss_frac: float = 0.0,
) -> SimulationResult:
    close_px_ta = np.asarray(close_px_ta)
    target_qty_ta = np.asarray(target_qty_ta)
    assert_float64("risk_engine.close_px_ta", close_px_ta)
    assert_float64("risk_engine.target_qty_ta", target_qty_ta)
    close_px_ta = close_px_ta.astype(np.float64, copy=False)
    target_qty_ta = target_qty_ta.astype(np.float64, copy=False)
    target_signal_mode = _resolve_target_signal_mode(target_signal_mode)
    if float(target_weight_deadband_frac) < 0.0:
        raise RuntimeError("target_weight_deadband_frac must be >= 0")
    if int(min_holding_minutes) < 0:
        raise RuntimeError("min_holding_minutes must be >= 0")
    if float(hard_stop_loss_frac) < 0.0:
        raise RuntimeError("hard_stop_loss_frac must be >= 0")

    if close_px_ta.shape != target_qty_ta.shape:
        raise RuntimeError("risk_engine input shape mismatch")

    T, A = close_px_ta.shape
    if session_id_t is not None:
        session_id_t = np.asarray(session_id_t, dtype=np.int64)
        if session_id_t.shape != (T,):
            raise RuntimeError(f"risk_engine session_id_t shape mismatch: got {session_id_t.shape}, expected {(T,)}")
        if T > 1 and np.any(np.diff(session_id_t) < 0):
            raise RuntimeError("risk_engine session_id_t must be nondecreasing")
    if volume_ta is not None:
        volume_ta = np.asarray(volume_ta)
        assert_float64("risk_engine.volume_ta", volume_ta)
        volume_ta = volume_ta.astype(np.float64, copy=False)
        if volume_ta.shape != (T, A):
            raise RuntimeError(f"risk_engine volume_ta shape mismatch: got {volume_ta.shape}, expected {(T, A)}")
    realism = _validate_execution_realism_inputs(
        execution_realism=execution_realism,
        t_count=T,
        a_count=A,
    )
    dynamic_rvol_ta: np.ndarray | None = None
    dynamic_tick_size_a: np.ndarray | None = None
    dynamic_minute_of_day_t: np.ndarray | None = None
    if str(realism.cost_model).strip() == EXECUTION_COST_MODEL_DYNAMIC:
        dynamic_rvol_ta = np.asarray(realism.rvol_ta, dtype=np.float64)
        dynamic_tick_size_a = np.asarray(realism.tick_size_a, dtype=np.float64)
        dynamic_minute_of_day_t = np.asarray(realism.minute_of_day_t, dtype=np.int32)
    qty = np.zeros(A, dtype=np.float64)
    position_entry_bar = np.full(A, -1, dtype=np.int64)
    position_entry_price = np.zeros(A, dtype=np.float64)
    cash = float(initial_cash)
    eq = np.zeros(T, dtype=np.float64)
    filled_qty = np.zeros((T, A), dtype=np.float64)
    exec_price = np.full((T, A), np.nan, dtype=np.float64)
    trade_cost = np.zeros((T, A), dtype=np.float64)
    desired_qty = np.zeros((T, A), dtype=np.float64)
    unfilled_qty = np.zeros((T, A), dtype=np.float64)
    fill_cap_qty = np.full((T, A), np.nan, dtype=np.float64)
    participation_rate_ta = np.zeros((T, A), dtype=np.float64)
    fill_capped_flag_ta = np.zeros((T, A), dtype=np.int8)
    fill_rejected_flag_ta = np.zeros((T, A), dtype=np.int8)
    slippage_cost_ta = np.zeros((T, A), dtype=np.float64)
    commission_cost_ta = np.zeros((T, A), dtype=np.float64)
    regulatory_cost_ta = np.zeros((T, A), dtype=np.float64)
    locate_cost_ta = np.zeros((T, A), dtype=np.float64)
    position_qty = np.zeros((T, A), dtype=np.float64)
    margin_used_t = np.zeros(T, dtype=np.float64)
    buying_power_t = np.zeros(T, dtype=np.float64)
    daily_loss_t = np.zeros(T, dtype=np.float64)
    borrow_cost_t = np.zeros(T, dtype=np.float64)
    debit_cost_t = np.zeros(T, dtype=np.float64)
    session_start_equity_t = np.zeros(T, dtype=np.float64)
    trades = 0
    day_start_eq = float(initial_cash)
    execution_diagnostics: dict[str, Any] = {
        "desired_fill_attempt_count": 0,
        "desired_fill_qty_abs_sum": 0.0,
        "filled_trade_count": 0,
        "filled_qty_abs_sum": 0.0,
        "volume_cap_hit_count": 0,
        "volume_cap_rejected_count": 0,
        "volume_cap_desired_qty_abs_sum": 0.0,
        "volume_cap_filled_qty_abs_sum": 0.0,
        "volume_cap_clipped_qty_abs_sum": 0.0,
        "buying_power_cap_hit_count": 0,
        "buying_power_cap_desired_qty_abs_sum": 0.0,
        "buying_power_cap_filled_qty_abs_sum": 0.0,
        "buying_power_cap_clipped_qty_abs_sum": 0.0,
        "fill_rejected_count": 0,
        "slippage_cost_total": 0.0,
        "commission_cost_total": 0.0,
        "regulatory_cost_total": 0.0,
        "locate_cost_total": 0.0,
        "borrow_cost_total": 0.0,
        "debit_cost_total": 0.0,
    }

    for t in range(T):
        px = close_px_ta[t]
        tgt = target_qty_ta[t]
        if not np.all(np.isfinite(px)):
            raise RuntimeError("risk_engine non-finite price")
        if session_id_t is not None and t > 0 and int(session_id_t[t]) != int(session_id_t[t - 1]):
            borrow_cost, debit_cost = _session_financing_cost_breakdown(
                qty=qty,
                price=close_px_ta[t - 1],
                cash=cash,
                cost_cfg=cost_cfg,
            )
            cash -= borrow_cost + debit_cost
            borrow_cost_t[t] = float(borrow_cost)
            debit_cost_t[t] = float(debit_cost)
            execution_diagnostics["borrow_cost_total"] += float(borrow_cost)
            execution_diagnostics["debit_cost_total"] += float(debit_cost)
            day_start_eq = float(cash + np.sum(qty * px))
        session_start_equity_t[t] = float(day_start_eq)
        mark_equity = float(cash + np.sum(qty * px))

        for a in range(A):
            target_qty = _resolve_target_qty(
                raw_target=float(tgt[a]),
                price=float(px[a]),
                current_qty=float(qty[a]),
                equity=mark_equity,
                target_signal_mode=target_signal_mode,
                target_weight_deadband_frac=float(target_weight_deadband_frac),
            )
            if abs(float(qty[a])) > 0.0 and int(min_holding_minutes) > 0:
                holding_age = t - int(position_entry_bar[a]) if int(position_entry_bar[a]) >= 0 else int(min_holding_minutes)
                stop_breached = _hard_stop_breached(
                    qty=float(qty[a]),
                    entry_price=float(position_entry_price[a]),
                    mark_price=float(px[a]),
                    hard_stop_loss_frac=float(hard_stop_loss_frac),
                )
                if (holding_age < int(min_holding_minutes)) and (not stop_breached):
                    if abs(target_qty) + 1.0e-12 < abs(float(qty[a])) or (
                        (abs(target_qty) > 0.0) and (np.sign(target_qty) != np.sign(float(qty[a])))
                    ):
                        target_qty = float(qty[a])

            desired_dq = float(target_qty - qty[a])
            desired_qty[t, a] = float(desired_dq)
            if abs(desired_dq) <= 0.0:
                continue
            execution_diagnostics["desired_fill_attempt_count"] += 1
            execution_diagnostics["desired_fill_qty_abs_sum"] += abs(desired_dq)
            dq = float(desired_dq)
            fill_cap = np.inf
            if volume_ta is not None:
                volume_bar = float(volume_ta[t, a])
                if (not np.isfinite(volume_bar)) or volume_bar <= 0.0:
                    execution_diagnostics["volume_cap_hit_count"] += 1
                    execution_diagnostics["volume_cap_rejected_count"] += 1
                    execution_diagnostics["fill_rejected_count"] += 1
                    execution_diagnostics["volume_cap_desired_qty_abs_sum"] += abs(desired_dq)
                    execution_diagnostics["volume_cap_clipped_qty_abs_sum"] += abs(desired_dq)
                    fill_rejected_flag_ta[t, a] = np.int8(1)
                    fill_capped_flag_ta[t, a] = np.int8(1)
                    unfilled_qty[t, a] = float(desired_dq)
                    continue
                fill_cap = float(np.floor(volume_bar * float(realism.max_volume_participation)))
                fill_cap_qty[t, a] = float(fill_cap)
                dq_before_volume_cap = float(dq)
                dq = np.sign(dq) * min(abs(dq), fill_cap)
                if abs(dq) + 1.0e-12 < abs(dq_before_volume_cap):
                    execution_diagnostics["volume_cap_hit_count"] += 1
                    execution_diagnostics["volume_cap_desired_qty_abs_sum"] += abs(dq_before_volume_cap)
                    execution_diagnostics["volume_cap_filled_qty_abs_sum"] += abs(dq)
                    execution_diagnostics["volume_cap_clipped_qty_abs_sum"] += abs(dq_before_volume_cap) - abs(dq)
                    fill_capped_flag_ta[t, a] = np.int8(1)
                    if abs(dq) <= 0.0:
                        execution_diagnostics["volume_cap_rejected_count"] += 1
                        execution_diagnostics["fill_rejected_count"] += 1
                        fill_rejected_flag_ta[t, a] = np.int8(1)
                if abs(dq) <= 0.0:
                    unfilled_qty[t, a] = float(desired_dq)
                    continue
            else:
                fill_cap_qty[t, a] = np.nan
            notional = abs(dq) * float(px[a])
            buying_power = max(0.0, day_start_eq)
            if notional > float(risk_cfg.max_position_buying_power_frac) * buying_power + 1e-12:
                allowed = float(risk_cfg.max_position_buying_power_frac) * buying_power
                dq_before_buying_power_cap = float(dq)
                dq = np.sign(dq) * np.floor(allowed / max(float(px[a]), 1e-12))
                if abs(dq) + 1.0e-12 < abs(dq_before_buying_power_cap):
                    execution_diagnostics["buying_power_cap_hit_count"] += 1
                    execution_diagnostics["buying_power_cap_desired_qty_abs_sum"] += abs(dq_before_buying_power_cap)
                    execution_diagnostics["buying_power_cap_filled_qty_abs_sum"] += abs(dq)
                    execution_diagnostics["buying_power_cap_clipped_qty_abs_sum"] += abs(dq_before_buying_power_cap) - abs(dq)
                if abs(dq) <= 0.0:
                    unfilled_qty[t, a] = float(desired_dq)
                    continue
            notional = abs(dq) * float(px[a])
            participation = 0.0 if not np.isfinite(fill_cap) else abs(dq) / max(fill_cap, 1.0)
            participation_rate_ta[t, a] = float(participation)
            if str(realism.cost_model).strip() == EXECUTION_COST_MODEL_DYNAMIC:
                if dynamic_rvol_ta is None or dynamic_tick_size_a is None or dynamic_minute_of_day_t is None:
                    raise RuntimeError("dynamic execution cost model arrays missing after validation")
                slippage_bps = _dynamic_slippage_bps(
                    t_index=t,
                    asset_index=a,
                    price=float(px[a]),
                    participation_rate=float(participation),
                    rvol_ta=dynamic_rvol_ta,
                    tick_size_a=dynamic_tick_size_a,
                    minute_of_day_t=dynamic_minute_of_day_t,
                    execution_realism=realism,
                )
                slip = notional * slippage_bps * 1.0e-4
            else:
                slip = notional * float(cost_cfg.slippage_bps) * 1e-4 * (1.0 + participation)
            comm = abs(dq) * float(cost_cfg.commission_per_share)
            short_before = max(-float(qty[a]), 0.0)
            short_after = max(-(float(qty[a]) + float(dq)), 0.0)
            short_entry_shares = max(short_after - short_before, 0.0)
            loc = short_entry_shares * float(cost_cfg.locate_fee_per_share_short_entry)
            reg = 0.0
            if dq < 0:
                reg += abs(dq) * float(cost_cfg.finra_taf_per_share_sell)
                reg += notional * float(cost_cfg.sec_fee_per_dollar_sell)
            total_cost = slip + comm + loc + reg
            old_qty = float(qty[a])
            old_entry_price = float(position_entry_price[a])
            cash -= dq * float(px[a]) + total_cost
            qty[a] += dq
            filled_qty[t, a] = float(dq)
            exec_price[t, a] = float(px[a])
            trade_cost[t, a] = float(total_cost)
            unfilled_qty[t, a] = float(desired_dq - dq)
            slippage_cost_ta[t, a] = float(slip)
            commission_cost_ta[t, a] = float(comm)
            regulatory_cost_ta[t, a] = float(reg)
            locate_cost_ta[t, a] = float(loc)
            trades += 1
            execution_diagnostics["filled_trade_count"] += 1
            execution_diagnostics["filled_qty_abs_sum"] += abs(dq)
            execution_diagnostics["slippage_cost_total"] += float(slip)
            execution_diagnostics["commission_cost_total"] += float(comm)
            execution_diagnostics["regulatory_cost_total"] += float(reg)
            execution_diagnostics["locate_cost_total"] += float(loc)

            new_qty = float(qty[a])
            if abs(new_qty) <= 0.0:
                position_entry_bar[a] = np.int64(-1)
                position_entry_price[a] = 0.0
            elif abs(old_qty) <= 0.0 or np.sign(old_qty) != np.sign(new_qty):
                position_entry_bar[a] = np.int64(t)
                position_entry_price[a] = float(px[a])
            elif np.sign(old_qty) == np.sign(new_qty) and abs(new_qty) > abs(old_qty):
                added = abs(new_qty) - abs(old_qty)
                position_entry_price[a] = (
                    abs(old_qty) * old_entry_price + added * float(px[a])
                ) / max(abs(new_qty), 1.0)

        market_value = float(np.sum(qty * px))
        equity = float(cash + market_value)
        if equity < float(risk_cfg.account_disable_equity):
            raise RuntimeError("RISK_ACCOUNT_DISABLE_THRESHOLD")
        gross = float(np.sum(np.abs(qty * px)))
        if gross > float(risk_cfg.overnight_exposure_equity_mult) * max(equity, 1e-12):
            raise RuntimeError("RISK_OVERNIGHT_EXPOSURE_BREACH")
        if (day_start_eq - equity) > float(risk_cfg.daily_loss_limit_frac) * max(day_start_eq, 1e-12):
            raise RuntimeError("RISK_DAILY_LOSS_BREACH")
        eq[t] = equity
        position_qty[t] = qty.copy()
        margin_used_t[t] = gross
        buying_power_t[t] = max(0.0, float(equity) - gross)
        daily_loss_t[t] = max(0.0, day_start_eq - equity)

    pnl = np.diff(eq)
    if pnl.size >= 2 and float(np.std(pnl, ddof=1)) > 0:
        sharpe = float(np.mean(pnl) / np.std(pnl, ddof=1) * np.sqrt(252.0))
        down = pnl[pnl < 0.0]
        sortino = float(np.mean(pnl) / np.std(down, ddof=1) * np.sqrt(252.0)) if down.size >= 2 and float(np.std(down, ddof=1)) > 0 else 0.0
    else:
        sharpe = 0.0
        sortino = 0.0

    dr = np.zeros(T, dtype=np.float64)
    dr[1:] = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
    assert_float64("risk_engine.equity_curve", eq)
    assert_float64("risk_engine.daily_returns", dr)
    assert_float64("risk_engine.filled_qty_ta", filled_qty)
    assert_float64("risk_engine.exec_price_ta", np.nan_to_num(exec_price, nan=0.0))
    assert_float64("risk_engine.trade_cost_ta", trade_cost)
    assert_float64("risk_engine.position_qty_ta", position_qty)
    return SimulationResult(
        equity_curve=eq,
        daily_returns=dr,
        filled_qty_ta=filled_qty,
        exec_price_ta=exec_price,
        trade_cost_ta=trade_cost,
        position_qty_ta=position_qty,
        margin_used_t=margin_used_t,
        buying_power_t=buying_power_t,
        daily_loss_t=daily_loss_t,
        trades=int(trades),
        final_equity=float(eq[-1]) if T else float(initial_cash),
        max_drawdown=_max_drawdown(eq),
        sharpe=float(sharpe),
        sortino=float(sortino),
        gross_exposure_peak=float(np.max(margin_used_t)) if margin_used_t.size else 0.0,
        desired_qty_ta=desired_qty,
        unfilled_qty_ta=unfilled_qty,
        fill_cap_qty_ta=fill_cap_qty,
        participation_rate_ta=participation_rate_ta,
        fill_capped_flag_ta=fill_capped_flag_ta,
        fill_rejected_flag_ta=fill_rejected_flag_ta,
        slippage_cost_ta=slippage_cost_ta,
        commission_cost_ta=commission_cost_ta,
        regulatory_cost_ta=regulatory_cost_ta,
        locate_cost_ta=locate_cost_ta,
        borrow_cost_t=borrow_cost_t,
        debit_cost_t=debit_cost_t,
        session_start_equity_t=session_start_equity_t,
        execution_cost_model=str(realism.cost_model),
        trade_log=[],
        per_asset_cumret={},
        execution_diagnostics=execution_diagnostics,
    )


# Backward-compat shim for potential legacy callers.
def simulate_portfolio_task(*args: Any, **kwargs: Any) -> SimulationResult:
    if args:
        return simulate_portfolio_from_signals(*args, **kwargs)

    strategy = dict(kwargs["strategy"])
    signals = {str(k): np.asarray(v, dtype=np.float64) for k, v in dict(kwargs["signals"]).items()}
    symbols = tuple(str(x) for x in kwargs["symbols"])
    cost_cfg = kwargs["cost_cfg"]
    risk_cfg = kwargs["risk_cfg"]
    initial_cash = float(kwargs["initial_cash"])
    minute_of_day = np.asarray(kwargs["minute_of_day"], dtype=np.int16)
    bar_valid = np.asarray(kwargs["bar_valid"], dtype=bool)
    active_asset_indices = kwargs.get("active_asset_indices")

    open_px = np.asarray(signals["open"], dtype=np.float64)
    high_px = np.asarray(signals["high"], dtype=np.float64)
    low_px = np.asarray(signals["low"], dtype=np.float64)
    close_px = np.asarray(signals["close"], dtype=np.float64)
    atr = np.asarray(signals.get("ATR", np.ones_like(close_px)), dtype=np.float64)
    s_break = np.asarray(signals.get("S_BREAK", np.zeros_like(close_px)), dtype=np.float64)

    if open_px.shape != close_px.shape:
        raise RuntimeError("legacy risk_engine signal shape mismatch")

    T, A = open_px.shape
    active_mask = np.zeros(A, dtype=bool)
    if active_asset_indices is None:
        active_mask[:] = True
    else:
        active_mask[np.asarray(active_asset_indices, dtype=np.int64)] = True

    lev_target = float(strategy.get("lev_target", 1.0) or 1.0)
    entry_threshold = float(strategy.get("s_break_thr", 0.0) or 0.0)
    exit_model = str(strategy.get("exit_model", "E5"))
    atr_stop_mult = strategy.get("atr_stop_mult")

    equity_curve = np.full(T, initial_cash, dtype=np.float64)
    daily_returns = np.zeros(T, dtype=np.float64)
    filled_qty = np.zeros((T, A), dtype=np.float64)
    exec_price = np.full((T, A), np.nan, dtype=np.float64)
    trade_cost = np.zeros((T, A), dtype=np.float64)
    position_qty = np.zeros((T, A), dtype=np.float64)
    margin_used_t = np.zeros(T, dtype=np.float64)
    buying_power_t = np.full(T, initial_cash, dtype=np.float64)
    daily_loss_t = np.zeros(T, dtype=np.float64)
    trade_log: list[dict[str, Any]] = []
    per_asset_cumret = {sym: 0.0 for sym in symbols}
    qty = np.zeros(A, dtype=np.float64)
    entry_px = np.full(A, np.nan, dtype=np.float64)
    entry_t = np.full(A, -1, dtype=np.int64)
    max_gross = 0.0
    realized_total = 0.0

    active_count = max(1, int(np.sum(active_mask)))
    per_asset_budget = float(initial_cash * lev_target / active_count)

    def _log_trade(t: int, a: int, dq: float, px: float, reason: str) -> None:
        filled_qty[t, a] = float(dq)
        exec_price[t, a] = float(px)
        position_qty[t, a] = float(qty[a])
        trade_log.append(
            {
                "t_index": int(t),
                "symbol": str(symbols[a]),
                "qty": float(dq),
                "exec_price": float(px),
                "reason": str(reason),
            }
        )

    for t in range(T):
        for a in range(A):
            if not active_mask[a] or not bool(bar_valid[t, a]):
                continue
            if not np.isfinite(open_px[t, a]):
                continue

            if qty[a] == 0.0 and float(s_break[t, a]) >= entry_threshold:
                buy_px = _legacy_exec_buy(float(open_px[t, a]))
                shares = int(np.floor(per_asset_budget / max(buy_px, 1e-12)))
                shares = max(0, shares)
                if shares > 0:
                    qty[a] = float(shares)
                    entry_px[a] = float(buy_px)
                    entry_t[a] = int(t)
                    cost = _trade_costs(shares=shares, side=1, is_short_entry=False, cost_cfg=cost_cfg, price=buy_px)
                    trade_cost[t, a] = float(cost)
                    _log_trade(t, a, float(shares), buy_px, "ENTRY")

            if qty[a] > 0.0:
                exit_reason: str | None = None
                exit_px = float(open_px[t, a])
                if int(t) >= int(entry_t[a]) + 2 and exit_model == "E1" and atr_stop_mult is not None:
                    stop_px = float(entry_px[a] - float(atr_stop_mult) * float(atr[t, a]))
                    if np.isfinite(low_px[t, a]) and float(low_px[t, a]) <= stop_px:
                        exit_reason = "E1_FIXED_ATR_STOP"
                        exit_px = _legacy_exec_sell(float(open_px[t, a]))
                if exit_reason is None and t > 0 and int(minute_of_day[t - 1]) == 945 and int(minute_of_day[t]) == 946:
                    exit_reason = "AUTO_DELEVER_1545_TO_1546"
                    exit_px = _legacy_exec_sell(float(open_px[t, a]))
                if exit_reason is None:
                    dd_frac = (float(entry_px[a]) - float(low_px[t, a])) * float(qty[a]) / max(initial_cash, 1e-12)
                    if np.isfinite(dd_frac) and dd_frac > float(risk_cfg.daily_loss_limit_frac):
                        exit_reason = "KILL_SWITCH"
                        exit_bar = min(t + 1, T - 1)
                        exit_px = _legacy_exec_sell(float(open_px[exit_bar, a]))
                        t = int(exit_bar)
                if exit_reason is None and t == T - 1:
                    exit_reason = "FINAL_FLATTEN"
                    exit_px = _legacy_exec_sell(float(open_px[t, a]))

                if exit_reason is not None:
                    shares = int(qty[a])
                    realized = (float(exit_px) - float(entry_px[a])) * float(shares)
                    per_asset_cumret[str(symbols[a])] += float(realized)
                    realized_total += float(realized)
                    cost = _trade_costs(shares=shares, side=-1, is_short_entry=False, cost_cfg=cost_cfg, price=exit_px)
                    trade_cost[t, a] += float(cost)
                    qty[a] = 0.0
                    _log_trade(t, a, -float(shares), exit_px, exit_reason)

        gross = float(np.sum(np.abs(qty) * close_px[t]))
        max_gross = max(max_gross, gross)
        margin_used_t[t] = gross
        buying_power_t[t] = max(0.0, float(initial_cash * lev_target) - gross)
        equity_curve[t] = float(initial_cash + realized_total)
        position_qty[t] = qty.copy()

    if T > 1:
        daily_returns[1:] = np.diff(equity_curve) / np.maximum(equity_curve[:-1], 1e-12)

    return SimulationResult(
        equity_curve=equity_curve,
        daily_returns=daily_returns,
        filled_qty_ta=filled_qty,
        exec_price_ta=exec_price,
        trade_cost_ta=trade_cost,
        position_qty_ta=position_qty,
        margin_used_t=margin_used_t,
        buying_power_t=buying_power_t,
        daily_loss_t=daily_loss_t,
        trades=int(len(trade_log)),
        final_equity=float(equity_curve[-1]) if T else float(initial_cash),
        max_drawdown=_max_drawdown(equity_curve),
        sharpe=0.0,
        sortino=0.0,
        gross_exposure_peak=float(max_gross),
        trade_log=trade_log,
        per_asset_cumret=per_asset_cumret,
        execution_diagnostics=None,
    )
