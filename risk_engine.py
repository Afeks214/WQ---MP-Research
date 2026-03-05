from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


REASON_WORKER_IO_VIOLATION = "WORKER_IO_VIOLATION"
REASON_STRATEGY_GRID_CARDINALITY_ERROR = "STRATEGY_GRID_CARDINALITY_ERROR"
REASON_COST_MODEL_VIOLATION = "COST_MODEL_VIOLATION"
REASON_RISK_CONSTRAINT_BREACH = "RISK_CONSTRAINT_BREACH"


@dataclass(frozen=True)
class CostConfig:
    tick_size: float = 0.01
    slippage_ticks: int = 1
    missing_bar_slippage_ticks: int = 5
    commission_per_share: float = 0.0015
    reg_fee_per_share_sell: float = 0.000119
    locate_fee_per_share_short_entry: float = 0.005


@dataclass(frozen=True)
class RiskConfig:
    per_asset_notional_cap_mult: float = 2.5
    max_position_buying_power_frac: float = 1.0
    overnight_gross_cap_mult: float = 1.6
    daily_max_loss_frac: float = 0.10
    account_disable_equity: float = 0.0
    account_disable_buffer_scale: float = 1.0
    delever_check_minute_et: int = 15 * 60 + 45
    delever_exec_minute_et: int = 15 * 60 + 46
    kill_switch_lockout_same_day: bool = True


@dataclass
class SimulationResult:
    strategy_id: str
    wf_split_idx: int
    cpcv_fold_idx: int
    scenario_id: str
    passed: bool
    reason_code: str
    error_msg: str
    trades: int
    win_rate: float
    sharpe: float
    sortino: float
    avg_ret: float
    med_ret: float
    avg_holding_time_bars: float
    profit_factor: float
    max_drawdown: float
    risk_breaches: int
    daily_loss_breaches: int
    gross_exposure_peak: float
    exposure_utilization: float
    reset_events: int
    final_equity: float
    per_asset_cumret: dict[str, float]
    daily_returns: list[dict[str, Any]]
    equity_curve: list[dict[str, Any]]
    trade_log: list[dict[str, Any]]


def _execution_price(raw_open: float, side: int, cost_cfg: CostConfig, severe_missing: bool) -> float:
    ticks = int(cost_cfg.missing_bar_slippage_ticks if severe_missing else cost_cfg.slippage_ticks)
    return float(raw_open + float(side) * float(ticks) * float(cost_cfg.tick_size))


def _trade_costs(shares: int, side: int, is_short_entry: bool, cost_cfg: CostConfig) -> float:
    if shares < 0:
        raise RuntimeError(f"{REASON_COST_MODEL_VIOLATION}: shares must be non-negative")
    commission_cost = float(shares) * float(cost_cfg.commission_per_share)
    reg_fee = float(shares) * float(cost_cfg.reg_fee_per_share_sell) if int(side) < 0 else 0.0
    locate = float(shares) * float(cost_cfg.locate_fee_per_share_short_entry) if bool(is_short_entry) else 0.0
    return float(commission_cost + reg_fee + locate)


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = np.divide(peak - equity, np.maximum(peak, 1e-12), out=np.zeros_like(equity), where=peak > 0)
    return float(np.max(dd))


def _score_for_entry(strategy: dict[str, Any], signals: dict[str, np.ndarray], t: int, a: int) -> tuple[int, float]:
    family = str(strategy["family"])
    D = float(signals["D"][t, a])
    A_aff = float(signals["A"][t, a])
    de = float(signals["DELTA_EFF"][t, a])
    sb = float(signals["S_BREAK"][t, a])
    sr = float(signals["S_REJECT"][t, a])
    rv = float(signals["RVOL"][t, a])
    close_px = float(signals["close"][t, a])
    vah = float(signals["VAH"][t, a]) if np.isfinite(signals["VAH"][t, a]) else np.nan
    val = float(signals["VAL"][t, a]) if np.isfinite(signals["VAL"][t, a]) else np.nan

    if family == "F1":
        ok = (sb > float(strategy["s_break_thr"])) and (rv > float(strategy["rvol_thr"]))
        if ok:
            return (1 if de >= 0.0 else -1, float(sb))
        return (0, 0.0)
    if family == "F2":
        ok = (sr > float(strategy["s_reject_thr"])) and (A_aff < float(strategy["a_thr"]))
        if ok:
            return (1 if D < 0.0 else -1, float(sr))
        return (0, 0.0)
    if family == "F3":
        ok = (abs(D) > float(strategy["d_thr"])) and (A_aff > float(strategy["a_accept_thr"]))
        if ok:
            return (-1 if D > 0.0 else 1, float(abs(D)))
        return (0, 0.0)
    if family == "F4":
        lo = float(strategy["s_break_mid_low"])
        hi = float(strategy["s_break_mid_high"])
        dist = float(strategy["va_dist"])
        if not np.isfinite(vah) or not np.isfinite(val):
            return (0, 0.0)
        outside = (close_px >= vah + dist) or (close_px <= val - dist)
        ok = outside and (sb >= lo) and (sb <= hi)
        if ok:
            return (1 if close_px > vah else -1, float(sb))
        return (0, 0.0)
    if family == "F5":
        ok = (de > float(strategy["de_thr"])) and (rv > float(strategy["rvol_thr"]))
        if ok:
            return (1 if de >= 0.0 else -1, float(abs(de)))
        return (0, 0.0)
    if family == "F6":
        d_low = float(strategy["d_low"])
        d_high = float(strategy["d_high"])
        ok = (A_aff < float(strategy["a_thr"])) and (abs(D) >= d_low) and (abs(D) <= d_high)
        if ok:
            return (-1 if D > 0.0 else 1, float(abs(D)))
        return (0, 0.0)
    if family == "SWING":
        de_th = float(strategy["deltaeff_threshold"])
        dist_th = float(strategy["distance_to_poc_atr"])
        acc_th = float(strategy["acceptance_threshold"])
        rv_th = float(strategy["rvol_filter"])
        atr = float(max(signals["ATR"][t, a], 1e-12))
        dist_to_poc = abs(close_px - float(signals["POC"][t, a])) / atr
        ok = (
            abs(de) >= de_th
            and dist_to_poc >= dist_th
            and A_aff <= acc_th
            and rv >= rv_th
        )
        if ok:
            side = 1 if de > 0.0 else (-1 if de < 0.0 else 0)
            if side != 0:
                return (side, float(abs(de) * max(rv, 0.0)))
        return (0, 0.0)

    return (0, 0.0)


def simulate_portfolio_task(
    *,
    strategy: dict[str, Any],
    signals: dict[str, np.ndarray],
    symbols: tuple[str, ...],
    split_mask: np.ndarray,
    cost_cfg: CostConfig,
    risk_cfg: RiskConfig,
    initial_cash: float,
    ts_ns: np.ndarray,
    minute_of_day: np.ndarray,
    session_id: np.ndarray,
    bar_valid: np.ndarray,
    last_valid_close: np.ndarray,
    wf_split_idx: int,
    cpcv_fold_idx: int,
    scenario_id: str,
    active_asset_indices: list[int] | None = None,
) -> SimulationResult:
    T, A = signals["close"].shape
    if split_mask.shape != (T,):
        raise RuntimeError("split_mask shape mismatch")

    qty = np.zeros(A, dtype=np.int64)
    avg_px = np.zeros(A, dtype=np.float64)
    entry_atr = np.zeros(A, dtype=np.float64)
    stop_px = np.full(A, np.nan, dtype=np.float64)
    entry_t = np.full(A, -1, dtype=np.int64)
    lockout_session = -1
    active_mask = np.ones(A, dtype=bool)
    if active_asset_indices is not None:
        active_mask[:] = False
        for idx in active_asset_indices:
            if 0 <= int(idx) < A:
                active_mask[int(idx)] = True

    cash = float(initial_cash)
    day_start_equity = float(initial_cash)
    prev_session_id = int(session_id[0])

    pending: dict[int, dict[str, Any]] = {}

    trades = 0
    wins = 0
    trade_rets: list[float] = []
    risk_breaches = 0
    gross_peak = 0.0
    reset_events = 0

    equity_series = np.zeros(T, dtype=np.float64)
    daily_records: list[dict[str, Any]] = []
    trade_log: list[dict[str, Any]] = []
    per_asset_pnl = np.zeros(A, dtype=np.float64)
    daily_loss_breaches = 0
    holding_bars: list[int] = []
    exposure_util_sum = 0.0
    exposure_util_n = 0

    def mark_close_equity(t: int) -> tuple[float, float]:
        close_t = np.where(bar_valid[t], signals["close"][t], last_valid_close[t])
        pos_val = qty.astype(np.float64) * close_t
        eq_close = float(cash + np.nansum(pos_val))

        worst_marks = np.where(
            qty > 0,
            np.where(bar_valid[t], signals["low"][t], last_valid_close[t]),
            np.where(qty < 0, np.where(bar_valid[t], signals["high"][t], last_valid_close[t]), close_t),
        )
        worst_val = qty.astype(np.float64) * worst_marks
        eq_worst = float(cash + np.nansum(worst_val))
        return eq_close, eq_worst

    def execute_order(t: int, a: int, target_qty: int, severe_missing: bool, reason: str) -> None:
        nonlocal cash, trades, wins
        current_qty = int(qty[a])
        avg_before = float(avg_px[a])
        if target_qty == current_qty:
            return
        if not np.isfinite(signals["open"][t, a]):
            raise RuntimeError("DATA_INCONSISTENCY: missing open for execution")

        side = 1 if target_qty > current_qty else -1
        delta = int(abs(target_qty - current_qty))
        px = _execution_price(float(signals["open"][t, a]), side, cost_cfg, severe_missing)
        if not np.isfinite(px) or px <= 0.0:
            raise RuntimeError("Invalid execution price")

        if side > 0:
            cash -= float(delta) * px
        else:
            cash += float(delta) * px

        is_short_entry = (target_qty < current_qty) and (target_qty < 0)
        cost = _trade_costs(delta, side, is_short_entry, cost_cfg)
        cash -= cost

        prev_notional = float(abs(current_qty) * avg_before)
        new_qty = int(target_qty)
        close_qty = 0
        if current_qty != 0:
            if np.sign(current_qty) != np.sign(new_qty):
                close_qty = int(abs(current_qty))
            elif abs(new_qty) < abs(current_qty):
                close_qty = int(abs(current_qty) - abs(new_qty))

        # realize pnl for closes/reductions using pre-update average price
        if close_qty > 0:
            signed_close = int(np.sign(current_qty) * close_qty)
            pnl = float((px - avg_before) * signed_close)
            trade_rets.append(float(pnl / max(abs(avg_before * signed_close), 1e-12)))
            per_asset_pnl[a] += pnl
            if entry_t[a] >= 0:
                holding_bars.append(int(max(0, t - int(entry_t[a]))))
            if pnl > 0:
                wins += 1

        if current_qty == 0 or np.sign(current_qty) == np.sign(new_qty):
            signed = int(target_qty - current_qty)
            if new_qty != 0:
                new_notional = prev_notional + float(abs(signed)) * px
                avg_px[a] = new_notional / float(abs(new_qty))
            else:
                avg_px[a] = 0.0
        else:
            if abs(new_qty) < abs(current_qty):
                pass
            elif abs(new_qty) == abs(current_qty):
                avg_px[a] = 0.0
            else:
                avg_px[a] = px

        if current_qty == 0 and new_qty != 0:
            entry_t[a] = int(t)
            entry_atr[a] = float(signals["ATR"][t, a])
            if strategy["exit_model"] == "E1":
                atr_mult = float(strategy["atr_stop_mult"])
                side_pos = 1 if new_qty > 0 else -1
                stop_px[a] = float(px - side_pos * (atr_mult * max(entry_atr[a], 1e-12)))

        qty[a] = int(new_qty)
        if qty[a] == 0:
            stop_px[a] = np.nan
            entry_t[a] = -1
            entry_atr[a] = 0.0
        trades += 1
        trade_log.append(
            {
                "t": int(t),
                "ts_ns": int(ts_ns[t]),
                "symbol": symbols[a],
                "reason": reason,
                "qty_before": int(current_qty),
                "qty_after": int(new_qty),
                "price": float(px),
                "shares": int(delta),
                "cost": float(cost),
                "severe_missing": bool(severe_missing),
            }
        )

    for t in range(T):
        if not bool(split_mask[t]):
            if t > 0:
                equity_series[t] = equity_series[t - 1]
            else:
                equity_series[t] = float(initial_cash)
            continue

        sid = int(session_id[t])
        if t > 0 and sid != prev_session_id:
            day_start_equity = float(equity_series[t - 1])
            prev_session_id = sid
            if int(minute_of_day[t]) >= 0:
                reset_events += 1

        # mandatory pending executions at first valid open
        for a in range(A):
            if not bool(active_mask[a]):
                continue
            if a not in pending:
                continue
            ord_ = pending[a]
            if t < int(ord_["due_t"]):
                continue
            if not bool(bar_valid[t, a]) or not np.isfinite(signals["open"][t, a]):
                ord_["severe"] = True
                ord_["due_t"] = int(t + 1)
                pending[a] = ord_
                continue
            execute_order(t, a, int(ord_["target_qty"]), bool(ord_["severe"]), str(ord_["reason"]))
            del pending[a]

        eq_close, eq_worst = mark_close_equity(t)
        equity_series[t] = float(eq_close)

        if day_start_equity <= 0.0:
            return SimulationResult(
                strategy_id=str(strategy["strategy_id"]),
                wf_split_idx=int(wf_split_idx),
                cpcv_fold_idx=int(cpcv_fold_idx),
                scenario_id=str(scenario_id),
                passed=False,
                reason_code=REASON_RISK_CONSTRAINT_BREACH,
                error_msg="day_start_equity <= 0",
                trades=trades,
                win_rate=0.0,
                avg_ret=0.0,
                med_ret=0.0,
                avg_holding_time_bars=0.0,
                sharpe=0.0,
                sortino=0.0,
                profit_factor=0.0,
                max_drawdown=1.0,
                risk_breaches=risk_breaches + 1,
                daily_loss_breaches=risk_breaches + 1,
                gross_exposure_peak=gross_peak,
                exposure_utilization=0.0,
                reset_events=reset_events,
                final_equity=float(eq_close),
                per_asset_cumret={symbols[i]: float(per_asset_pnl[i]) for i in range(A)},
                daily_returns=daily_records,
                equity_curve=[{"ts_ns": int(ts_ns[i]), "equity": float(equity_series[i])} for i in range(T) if split_mask[i]],
                trade_log=trade_log,
            )

        dd = float((day_start_equity - eq_worst) / day_start_equity)
        if dd >= float(risk_cfg.daily_max_loss_frac):
            risk_breaches += 1
            daily_loss_breaches += 1
            lockout_session = sid
            for a in range(A):
                if qty[a] != 0:
                    pending[a] = {"target_qty": 0, "due_t": int(t + 1), "severe": False, "reason": "KILL_SWITCH"}

        gross = float(np.nansum(np.abs(qty.astype(np.float64) * np.where(bar_valid[t], signals["close"][t], last_valid_close[t]))))
        gross_peak = max(gross_peak, gross)

        # deterministic auto-deleveraging timing: evaluate at 15:45 close, execute at 15:46 open
        if int(minute_of_day[t]) == int(risk_cfg.delever_check_minute_et):
            target_gross = float(eq_close) * float(risk_cfg.overnight_gross_cap_mult)
            excess = gross - target_gross
            if excess > 0.0 and eq_close > 0.0:
                exposure = np.abs(qty.astype(np.float64) * np.where(bar_valid[t], signals["close"][t], last_valid_close[t]))
                total_exp = float(np.sum(exposure))
                if total_exp > 0:
                    order_idx = np.argsort(-exposure, kind="mergesort")
                    remain = float(excess)
                    for a in order_idx.tolist():
                        if remain <= 0.0:
                            break
                        if qty[a] == 0:
                            continue
                        px_ref = float(np.where(bar_valid[t, a], signals["close"][t, a], last_valid_close[t, a]))
                        if not np.isfinite(px_ref) or px_ref <= 0.0:
                            continue
                        trim_shares = int(min(abs(qty[a]), np.floor(remain / px_ref)))
                        if trim_shares <= 0:
                            continue
                        tgt = int(qty[a] - np.sign(qty[a]) * trim_shares)
                        pending[a] = {
                            "target_qty": tgt,
                            "due_t": int(t + 1),
                            "severe": False,
                            "reason": "AUTO_DELEVER_1545_TO_1546",
                        }
                        remain -= float(trim_shares) * px_ref

        # exits (mandatory)
        for a in range(A):
            if not bool(active_mask[a]):
                continue
            q = int(qty[a])
            if q == 0:
                continue
            side = 1 if q > 0 else -1
            close_a = float(np.where(bar_valid[t, a], signals["close"][t, a], last_valid_close[t, a]))
            low_a = float(np.where(bar_valid[t, a], signals["low"][t, a], last_valid_close[t, a]))
            high_a = float(np.where(bar_valid[t, a], signals["high"][t, a], last_valid_close[t, a]))
            if "SIGNED_SCORE" in signals:
                score_signed = float(signals["SIGNED_SCORE"][t, a])
            else:
                score_signed = float(signals["S_BREAK"][t, a]) * float(np.sign(signals["DELTA_EFF"][t, a]))

            exit_reason = ""
            model = str(strategy["exit_model"])
            if model == "E1":
                sp = float(stop_px[a])
                if np.isfinite(sp):
                    if (side > 0 and low_a <= sp) or (side < 0 and high_a >= sp):
                        exit_reason = "E1_FIXED_ATR_STOP"
            elif model == "E2":
                mp_mean = float(signals["POC"][t, a]) if np.isfinite(signals["POC"][t, a]) else np.nan
                if np.isfinite(mp_mean):
                    if (side > 0 and close_a >= mp_mean) or (side < 0 and close_a <= mp_mean):
                        exit_reason = "E2_PROFILE_MEAN"
            elif model == "E3":
                vah = float(signals["VAH"][t, a]) if np.isfinite(signals["VAH"][t, a]) else np.nan
                val = float(signals["VAL"][t, a]) if np.isfinite(signals["VAL"][t, a]) else np.nan
                if side > 0 and np.isfinite(vah) and high_a >= vah:
                    exit_reason = "E3_VA_TP"
                if side < 0 and np.isfinite(val) and low_a <= val:
                    exit_reason = "E3_VA_TP"
            elif model == "E4":
                # zero-cross hysteresis invariant
                if (side > 0 and score_signed <= 0.0) or (side < 0 and score_signed >= 0.0):
                    exit_reason = "E4_SCORE_ZERO_CROSS"
            elif model == "E5":
                n_bars = int(strategy["time_exit_bars"])
                if entry_t[a] >= 0 and (t - int(entry_t[a]) >= n_bars):
                    exit_reason = "E5_TIME_EXIT"

            if exit_reason and a not in pending:
                pending[a] = {"target_qty": 0, "due_t": int(t + 1), "severe": False, "reason": exit_reason}

        # entries blocked during lockout session
        if lockout_session == sid:
            continue

        lev_target = float(strategy["lev_target"])
        if float(eq_close) < float(risk_cfg.account_disable_equity):
            lev_target = float(lev_target * float(risk_cfg.account_disable_buffer_scale))
        target_gross_notional = float(eq_close * lev_target)
        prices = np.where(bar_valid[t], signals["close"][t], last_valid_close[t])
        current_gross_exposure = float(np.nansum(np.abs(qty.astype(np.float64) * prices)))
        if target_gross_notional > 0:
            exposure_util_sum += float(current_gross_exposure / target_gross_notional)
            exposure_util_n += 1
        available_buying_power = max(0.0, target_gross_notional - current_gross_exposure)
        if available_buying_power <= 0.0:
            continue

        intents: list[tuple[int, int, float]] = []
        for a in range(A):
            if not bool(active_mask[a]):
                continue
            if not bool(bar_valid[t, a]):
                continue
            if a in pending:
                continue
            dir_sign, strength = _score_for_entry(strategy, signals, t, a)
            if dir_sign == 0 or not np.isfinite(strength) or strength <= 0.0:
                continue
            intents.append((a, dir_sign, float(strength)))

        if not intents:
            continue

        strengths = np.array([x[2] for x in intents], dtype=np.float64)
        denom = float(np.sum(strengths))
        if not np.isfinite(denom) or denom <= 0.0:
            continue
        weights = strengths / denom

        for idx, (a, dir_sign, _) in enumerate(intents):
            px_open = float(signals["open"][t, a])
            if not np.isfinite(px_open) or px_open <= 0.0:
                continue

            raw_alloc = float(available_buying_power * weights[idx])
            max_pos_notional = float(available_buying_power * float(risk_cfg.max_position_buying_power_frac))
            per_asset_cap = float(eq_close * risk_cfg.per_asset_notional_cap_mult)
            alloc_notional = min(raw_alloc, per_asset_cap, max_pos_notional)
            exec_px = _execution_price(px_open, 1 if dir_sign > 0 else -1, cost_cfg, False)
            if exec_px <= 0.0:
                continue
            target_shares = int(np.floor(abs(alloc_notional) / exec_px))
            if target_shares <= 0:
                continue

            target_qty = int(dir_sign * target_shares)
            # buying-power hard cap on incremental gross
            inc = float(abs(target_qty - int(qty[a])) * exec_px)
            if inc > available_buying_power + 1e-12:
                allowed_shares = int(np.floor(max(0.0, available_buying_power) / exec_px))
                if allowed_shares <= 0:
                    continue
                target_qty = int(dir_sign * allowed_shares)
            execute_order(t, a, target_qty, False, "ENTRY")
            available_buying_power = max(
                0.0,
                float(eq_close * lev_target)
                - float(np.nansum(np.abs(qty.astype(np.float64) * np.where(bar_valid[t], signals["close"][t], last_valid_close[t])))),
            )

    valid_mask = split_mask.astype(bool)
    eq = equity_series[valid_mask]
    final_equity = float(eq[-1]) if eq.size else float(initial_cash)

    pnl = np.diff(eq) if eq.size > 1 else np.array([], dtype=np.float64)
    if pnl.size:
        pos = pnl[pnl > 0]
        neg = -pnl[pnl < 0]
        pf = float(np.sum(pos) / max(np.sum(neg), 1e-12))
    else:
        pf = 0.0

    win_rate = float((wins / max(trades, 1)))
    avg_ret = float(np.mean(trade_rets)) if trade_rets else 0.0
    med_ret = float(np.median(trade_rets)) if trade_rets else 0.0
    avg_holding = float(np.mean(holding_bars)) if holding_bars else 0.0
    max_dd = _max_drawdown(eq if eq.size else np.array([initial_cash], dtype=np.float64))
    dr = np.array([float(x["return"]) for x in daily_records], dtype=np.float64) if daily_records else np.array([], dtype=np.float64)
    if dr.size >= 2 and float(np.std(dr, ddof=1)) > 0:
        sharpe = float(np.mean(dr) / np.std(dr, ddof=1) * np.sqrt(252.0))
        down = dr[dr < 0.0]
        if down.size >= 2 and float(np.std(down, ddof=1)) > 0:
            sortino = float(np.mean(dr) / np.std(down, ddof=1) * np.sqrt(252.0))
        else:
            sortino = 0.0
    else:
        sharpe = 0.0
        sortino = 0.0
    exposure_util = float(exposure_util_sum / max(exposure_util_n, 1))

    # per-day returns
    day_keys = (ts_ns[valid_mask] // (24 * 60 * 60 * 1_000_000_000)).astype(np.int64)
    if eq.size > 0:
        uniq = np.unique(day_keys)
        for d in uniq.tolist():
            idx = np.where(day_keys == d)[0]
            if idx.size == 0:
                continue
            day_start = float(eq[idx[0]])
            day_end = float(eq[idx[-1]])
            ret = float((day_end - day_start) / max(day_start, 1e-12))
            daily_records.append({"day_key": int(d), "return": ret, "strategy_id": str(strategy["strategy_id"])})

    return SimulationResult(
        strategy_id=str(strategy["strategy_id"]),
        wf_split_idx=int(wf_split_idx),
        cpcv_fold_idx=int(cpcv_fold_idx),
        scenario_id=str(scenario_id),
        passed=True,
        reason_code="OK",
        error_msg="",
        trades=int(trades),
        win_rate=float(win_rate),
        sharpe=float(sharpe),
        sortino=float(sortino),
        avg_ret=float(avg_ret),
        med_ret=float(med_ret),
        avg_holding_time_bars=float(avg_holding),
        profit_factor=float(pf),
        max_drawdown=float(max_dd),
        risk_breaches=int(risk_breaches),
        daily_loss_breaches=int(daily_loss_breaches),
        gross_exposure_peak=float(gross_peak),
        exposure_utilization=float(exposure_util),
        reset_events=int(reset_events),
        final_equity=float(final_equity),
        per_asset_cumret={symbols[i]: float(per_asset_pnl[i]) for i in range(A)},
        daily_returns=daily_records,
        equity_curve=[{"ts_ns": int(ts_ns[i]), "equity": float(equity_series[i])} for i in np.where(valid_mask)[0].tolist()],
        trade_log=trade_log,
    )
