from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from weightiz_dtype_guard import assert_float64


@dataclass(frozen=True)
class CostConfig:
    commission_per_share: float = 0.0
    finra_taf_per_share_sell: float = 0.0
    sec_fee_per_dollar_sell: float = 0.0
    short_borrow_apr: float = 0.0
    locate_fee_per_share_short_entry: float = 0.0
    slippage_bps: float = 0.0


@dataclass(frozen=True)
class RiskConfig:
    max_position_buying_power_frac: float = 0.25
    overnight_exposure_equity_mult: float = 2.0
    daily_loss_limit_frac: float = 0.10
    account_disable_equity: float = 1000.0


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


def _max_drawdown(eq: np.ndarray) -> float:
    if eq.size == 0:
        return 0.0
    roll_max = np.maximum.accumulate(eq)
    dd = (roll_max - eq) / np.maximum(roll_max, 1e-12)
    return float(np.max(dd))


def simulate_portfolio_from_signals(
    close_px_ta: np.ndarray,
    target_qty_ta: np.ndarray,
    initial_cash: float,
    cost_cfg: CostConfig,
    risk_cfg: RiskConfig,
) -> SimulationResult:
    close_px_ta = np.asarray(close_px_ta)
    target_qty_ta = np.asarray(target_qty_ta)
    assert_float64("risk_engine.close_px_ta", close_px_ta)
    assert_float64("risk_engine.target_qty_ta", target_qty_ta)
    close_px_ta = close_px_ta.astype(np.float64, copy=False)
    target_qty_ta = target_qty_ta.astype(np.float64, copy=False)

    if close_px_ta.shape != target_qty_ta.shape:
        raise RuntimeError("risk_engine input shape mismatch")

    T, A = close_px_ta.shape
    qty = np.zeros(A, dtype=np.float64)
    cash = float(initial_cash)
    eq = np.zeros(T, dtype=np.float64)
    filled_qty = np.zeros((T, A), dtype=np.float64)
    exec_price = np.full((T, A), np.nan, dtype=np.float64)
    trade_cost = np.zeros((T, A), dtype=np.float64)
    position_qty = np.zeros((T, A), dtype=np.float64)
    margin_used_t = np.zeros(T, dtype=np.float64)
    buying_power_t = np.zeros(T, dtype=np.float64)
    daily_loss_t = np.zeros(T, dtype=np.float64)
    trades = 0
    day_start_eq = float(initial_cash)

    for t in range(T):
        px = close_px_ta[t]
        tgt = target_qty_ta[t]
        if not np.all(np.isfinite(px)):
            raise RuntimeError("risk_engine non-finite price")

        for a in range(A):
            dq = float(tgt[a] - qty[a])
            if abs(dq) <= 0.0:
                continue
            notional = abs(dq) * float(px[a])
            buying_power = max(0.0, day_start_eq)
            if notional > float(risk_cfg.max_position_buying_power_frac) * buying_power + 1e-12:
                allowed = float(risk_cfg.max_position_buying_power_frac) * buying_power
                dq = np.sign(dq) * np.floor(allowed / max(float(px[a]), 1e-12))
                if abs(dq) <= 0.0:
                    continue
            slip = notional * float(cost_cfg.slippage_bps) * 1e-4
            comm = abs(dq) * float(cost_cfg.commission_per_share)
            loc = abs(dq) * float(cost_cfg.locate_fee_per_share_short_entry) if (dq < 0) else 0.0
            reg = 0.0
            if dq < 0:
                reg += abs(dq) * float(cost_cfg.finra_taf_per_share_sell)
                reg += notional * float(cost_cfg.sec_fee_per_dollar_sell)
            total_cost = slip + comm + loc + reg
            cash -= dq * float(px[a]) + total_cost
            qty[a] += dq
            filled_qty[t, a] = float(dq)
            exec_price[t, a] = float(px[a])
            trade_cost[t, a] = float(total_cost)
            trades += 1

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
    )


# Backward-compat shim for potential legacy callers.
def simulate_portfolio_task(*args: Any, **kwargs: Any) -> SimulationResult:
    return simulate_portfolio_from_signals(*args, **kwargs)
