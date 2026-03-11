from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from module6.config import Module6Config
from module6.constants import (
    AVAIL_FORCED_CASH_BY_RISK,
    AVAIL_FORCED_ZERO_BY_PORTFOLIO,
    AVAIL_INVALIDATED_BY_DQ,
    AVAIL_OBSERVED_ACTIVE,
    AVAIL_OBSERVED_FLAT,
    AVAIL_STRUCTURALLY_MISSING,
)
from module6.constraints import check_path_constraints, project_to_feasible_weights
from module6.utils import Module6ValidationError


@dataclass(frozen=True)
class SessionSimulationArtifacts:
    session_paths: pd.DataFrame
    portfolio_summary: pd.DataFrame
    weight_history: pd.DataFrame


def _policy_rebalance_due(policy: str, session_idx: int, drift_weights: np.ndarray, target_weights: np.ndarray, band: float) -> bool:
    if policy == "daily_close":
        return True
    if policy == "weekly_monday_close":
        return bool(session_idx % 5 == 0)
    if policy == "band_10pct":
        return bool(0.5 * np.sum(np.abs(drift_weights - target_weights)) > float(band))
    raise Module6ValidationError(f"unsupported rebalance policy: {policy}")


def simulate_session_batch(
    *,
    portfolio_candidates: pd.DataFrame,
    portfolio_weights: pd.DataFrame,
    strategy_frame: pd.DataFrame,
    matrices: dict[str, np.ndarray | object],
    calendar: np.ndarray,
    config: Module6Config,
    return_weight_history: bool = False,
) -> SessionSimulationArtifacts:
    if portfolio_candidates.shape[0] <= 0:
        return SessionSimulationArtifacts(
            session_paths=pd.DataFrame(),
            portfolio_summary=pd.DataFrame(),
            weight_history=pd.DataFrame(),
        )
    col_map = dict(strategy_frame[["strategy_instance_pk", "column_idx"]].itertuples(index=False, name=None))
    cluster_map = dict(strategy_frame[["strategy_instance_pk", "cluster_id"]].itertuples(index=False, name=None))
    family_map = dict(strategy_frame[["strategy_instance_pk", "family_id"]].itertuples(index=False, name=None))
    priority_map = dict(strategy_frame[["strategy_instance_pk", "robustness_score"]].itertuples(index=False, name=None))
    r_exec = np.asarray(matrices["R_exec"], dtype=np.float64)
    a = np.asarray(matrices["A"], dtype=bool)
    u = np.asarray(matrices["U"], dtype=np.float64)
    state_codes = np.asarray(matrices["state_codes"], dtype=np.int16)
    gross_peak = np.asarray(matrices["gross_peak"], dtype=np.float64)
    overnight = np.asarray(matrices["overnight_flag"], dtype=np.int8)
    session_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []

    for candidate in portfolio_candidates.itertuples(index=False):
        weights_df = portfolio_weights.loc[portfolio_weights["portfolio_pk"] == candidate.portfolio_pk].copy()
        weights_df = weights_df.sort_values(["strategy_instance_pk"], kind="mergesort")
        if weights_df.shape[0] <= 0:
            raise Module6ValidationError(f"missing weights for portfolio {candidate.portfolio_pk}")
        strategy_ids = weights_df["strategy_instance_pk"].astype(str).tolist()
        cols = np.asarray([int(col_map[pk]) for pk in strategy_ids], dtype=np.int64)
        cluster_ids = np.asarray([int(cluster_map[pk]) for pk in strategy_ids], dtype=np.int64)
        family_ids = np.asarray([str(family_map[pk]) for pk in strategy_ids], dtype=object)
        priorities = np.asarray([float(priority_map.get(pk, 0.0)) for pk in strategy_ids], dtype=np.float64)
        target = np.asarray(weights_df["target_weight"], dtype=np.float64)
        target_cash = float(candidate.cash_weight)
        if np.sum(target) > 1.0 + 1.0e-12:
            raise Module6ValidationError(f"non-simplex target weights for portfolio {candidate.portfolio_pk}")
        weights = target.copy()
        cash_weight = target_cash
        equity = 1.0
        day_start_equity = float(equity)
        peak_equity = float(equity)
        breach_count = 0
        disable_flag = False
        total_turnover = 0.0
        total_forced_cash = 0.0
        total_missing = 0.0
        path_returns: list[float] = []
        for t_idx, session_id in enumerate(np.asarray(calendar, dtype=np.int64).tolist()):
            base_codes = np.asarray(state_codes[t_idx, cols], dtype=np.int16)
            base_active = np.asarray([int(code) in (AVAIL_OBSERVED_ACTIVE, AVAIL_OBSERVED_FLAT) for code in base_codes], dtype=bool)
            effective_returns = np.asarray(r_exec[t_idx, cols], dtype=np.float64)
            effective_returns = np.where(base_active, effective_returns, 0.0)
            gross_return = float(np.dot(weights, effective_returns))
            pre_cost_equity = float(equity * (1.0 + gross_return))
            if pre_cost_equity <= 0.0 or not np.isfinite(pre_cost_equity):
                raise Module6ValidationError(f"invalid pre-cost equity evolution for portfolio {candidate.portfolio_pk}")
            drift = np.asarray(weights * (1.0 + effective_returns) / max(1.0 + gross_return, 1.0e-12), dtype=np.float64)
            drift = np.where(np.isfinite(drift), drift, 0.0)
            drift = np.maximum(drift, 0.0)
            drift_total = float(np.sum(drift))
            if drift_total > 1.0:
                drift /= drift_total
                drift_total = float(np.sum(drift))
            drift_cash = max(0.0, 1.0 - drift_total)
            unavailable_target = np.where(base_active, target, 0.0)
            projected = project_to_feasible_weights(
                target_weights=unavailable_target,
                gross_mult=np.asarray(gross_peak[t_idx, cols], dtype=np.float64),
                overnight_flags=np.asarray(overnight[t_idx, cols], dtype=np.int8),
                cluster_ids=cluster_ids,
                family_ids=family_ids,
                priority_scores=priorities,
                config=config,
            )
            rebalance_due = False if disable_flag else _policy_rebalance_due(
                str(candidate.rebalance_policy),
                t_idx,
                drift,
                unavailable_target,
                config.simulator.rebalance_band_l1,
            )
            if disable_flag:
                projected_weights = np.zeros_like(weights)
                projected_cash = 1.0
                path_state_codes = np.where(base_active, AVAIL_FORCED_CASH_BY_RISK, base_codes)
                turnover = 0.0
                cost_frac = 0.0
            elif rebalance_due:
                if projected.infeasible:
                    disable_flag = True
                    breach_count += 1
                    projected_weights = np.zeros_like(weights)
                    projected_cash = 1.0
                    path_state_codes = np.where(base_active, AVAIL_FORCED_CASH_BY_RISK, base_codes)
                    turnover = 0.0
                    cost_frac = 0.0
                else:
                    projected_weights = projected.weights
                    projected_cash = projected.cash_weight
                    path_state_codes = np.where(base_active, base_codes, AVAIL_FORCED_ZERO_BY_PORTFOLIO)
                    turnover = float(0.5 * np.sum(np.abs(drift - projected_weights)))
                    liquidity_penalty = 1.0 + float(np.sum(np.asarray(u[t_idx, cols], dtype=np.float64) * np.maximum(drift, 0.0)))
                    cost_abs = (
                        float(config.simulator.fixed_fee) * float(turnover > 0.0)
                        + float(config.simulator.linear_cost_bps) * 1.0e-4 * turnover * pre_cost_equity
                        + float(config.simulator.slippage_cost_bps) * 1.0e-4 * turnover * pre_cost_equity * liquidity_penalty
                    )
                    cost_frac = float(cost_abs / max(pre_cost_equity, 1.0e-12))
            else:
                projected_weights = drift
                projected_cash = drift_cash
                path_state_codes = base_codes
                turnover = 0.0
                cost_frac = 0.0
            equity = float(pre_cost_equity * (1.0 - cost_frac))
            if not np.isfinite(equity):
                raise Module6ValidationError(f"non-finite equity for portfolio {candidate.portfolio_pk}")
            overnight_active_count = int(np.sum((np.asarray(overnight[t_idx, cols], dtype=np.int8) > 0) & (projected_weights > 0.0)))
            gross_exposure_mult = float(np.sum(projected_weights * np.asarray(gross_peak[t_idx, cols], dtype=np.float64)))
            flags = list(
                check_path_constraints(
                    equity=equity,
                    day_start_equity=day_start_equity,
                    gross_exposure_mult=gross_exposure_mult,
                    cash_weight=projected_cash,
                    config=config,
                    overnight_active_count=overnight_active_count,
                )
            )
            if projected.projected:
                flags.append("rebalance_projected")
            if projected.infeasible:
                flags.append("rebalance_infeasible")
            if "capital_floor_hit" in flags or "daily_loss_breach" in flags or "gross_limit_breach" in flags:
                disable_flag = True
                breach_count += 1
            weights = np.asarray(projected_weights, dtype=np.float64)
            cash_weight = float(projected_cash)
            peak_equity = max(peak_equity, equity)
            drawdown = float(1.0 - equity / max(peak_equity, 1.0e-12))
            session_return = float(equity / max(day_start_equity, 1.0e-12) - 1.0)
            total_turnover += float(turnover)
            total_forced_cash += float(np.mean(np.isin(path_state_codes, [AVAIL_FORCED_CASH_BY_RISK, AVAIL_FORCED_ZERO_BY_PORTFOLIO]).astype(np.float64)))
            total_missing += float(np.mean(np.isin(path_state_codes, [AVAIL_STRUCTURALLY_MISSING, AVAIL_INVALIDATED_BY_DQ]).astype(np.float64)))
            path_returns.append(session_return)
            session_rows.append(
                {
                    "portfolio_pk": str(candidate.portfolio_pk),
                    "reduced_universe_id": str(candidate.reduced_universe_id),
                    "session_id": int(session_id),
                    "session_return": float(session_return),
                    "equity": float(equity),
                    "drawdown": float(drawdown),
                    "turnover": float(turnover),
                    "cost_frac": float(cost_frac),
                    "gross_exposure_mult": float(gross_exposure_mult),
                    "cash_weight": float(cash_weight),
                    "forced_cash_share": float(np.mean(np.isin(path_state_codes, [AVAIL_FORCED_CASH_BY_RISK, AVAIL_FORCED_ZERO_BY_PORTFOLIO]).astype(np.float64))),
                    "missing_share": float(np.mean(np.isin(path_state_codes, [AVAIL_STRUCTURALLY_MISSING, AVAIL_INVALIDATED_BY_DQ]).astype(np.float64))),
                    "breach_count_cum": int(breach_count),
                    "disable_flag": int(disable_flag),
                    "rebalance_due": int(rebalance_due),
                    "compliance_flags": "|".join(sorted(set(flags))),
                }
            )
            if return_weight_history:
                for pk, start_weight, end_weight, state_code in zip(strategy_ids, drift.tolist(), weights.tolist(), path_state_codes.tolist()):
                    weight_rows.append(
                        {
                            "portfolio_pk": str(candidate.portfolio_pk),
                            "session_id": int(session_id),
                            "strategy_instance_pk": str(pk),
                            "start_weight": float(start_weight),
                            "end_weight": float(end_weight),
                            "availability_state_code": int(state_code),
                            "reduced_universe_id": str(candidate.reduced_universe_id),
                        }
                    )
            day_start_equity = float(equity)
        ann_return = float(np.mean(path_returns) * 252.0) if path_returns else 0.0
        max_dd = float(max(row["drawdown"] for row in session_rows if row["portfolio_pk"] == str(candidate.portfolio_pk))) if path_returns else 0.0
        summary_rows.append(
            {
                "portfolio_pk": str(candidate.portfolio_pk),
                "reduced_universe_id": str(candidate.reduced_universe_id),
                "final_equity": float(equity),
                "annualized_return": float(ann_return),
                "max_drawdown": float(max_dd),
                "turnover": float(total_turnover / max(len(calendar), 1)),
                "forced_cash_burden": float(total_forced_cash / max(len(calendar), 1)),
                "missingness_burden": float(total_missing / max(len(calendar), 1)),
                "support_coverage": float(1.0 - total_missing / max(len(calendar), 1)),
                "breach_count": int(breach_count),
                "disable_flag": int(disable_flag),
            }
        )
    return SessionSimulationArtifacts(
        session_paths=pd.DataFrame(session_rows).sort_values(["portfolio_pk", "session_id"], kind="mergesort").reset_index(drop=True),
        portfolio_summary=pd.DataFrame(summary_rows).sort_values(["portfolio_pk"], kind="mergesort").reset_index(drop=True),
        weight_history=pd.DataFrame(weight_rows).sort_values(["portfolio_pk", "session_id", "strategy_instance_pk"], kind="mergesort").reset_index(drop=True) if weight_rows else pd.DataFrame(columns=["portfolio_pk", "session_id", "strategy_instance_pk", "start_weight", "end_weight", "availability_state_code", "reduced_universe_id"]),
    )
