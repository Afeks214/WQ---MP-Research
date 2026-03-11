from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from module6.config import Module6Config
from module6.utils import Module6ValidationError


@dataclass(frozen=True)
class MinuteReplayArtifacts:
    minute_paths: pd.DataFrame
    component_diagnostics: pd.DataFrame
    divergence: pd.DataFrame
    minute_summary: pd.DataFrame


def replay_finalists_minute(
    *,
    finalist_candidates: pd.DataFrame,
    strategy_frame: pd.DataFrame,
    session_paths: pd.DataFrame,
    session_summary: pd.DataFrame,
    weight_history: pd.DataFrame,
    run,
    config: Module6Config,
) -> MinuteReplayArtifacts:
    if finalist_candidates.shape[0] <= 0:
        return MinuteReplayArtifacts(
            minute_paths=pd.DataFrame(),
            component_diagnostics=pd.DataFrame(),
            divergence=pd.DataFrame(),
            minute_summary=pd.DataFrame(),
        )
    instance_lookup = strategy_frame[["strategy_instance_pk", "candidate_id", "split_id", "scenario_id"]].drop_duplicates().copy()
    instance_lookup["instance_key"] = (
        instance_lookup["candidate_id"].astype(str)
        + "|"
        + instance_lookup["split_id"].astype(str)
        + "|"
        + instance_lookup["scenario_id"].astype(str)
    )
    if run.micro_diagnostics is None:
        raise Module6ValidationError("minute replay requires micro_diagnostics truth input")
    eq = run.equity_curves.copy()
    eq["instance_key"] = (
        eq["candidate_id"].astype(str)
        + "|"
        + eq["split_id"].astype(str)
        + "|"
        + eq["scenario_id"].astype(str)
    )
    eq = eq.loc[eq["instance_key"].isin(instance_lookup["instance_key"])].copy()
    if eq.shape[0] <= 0:
        raise Module6ValidationError("minute replay requires equity_curves for finalist instances")
    trade = run.trade_log.copy()
    trade["instance_key"] = (
        trade["candidate_id"].astype(str)
        + "|"
        + trade["split_id"].astype(str)
        + "|"
        + trade["scenario_id"].astype(str)
    )
    trade = trade.loc[trade["instance_key"].isin(instance_lookup["instance_key"])].copy()
    micro = run.micro_diagnostics.copy()
    micro["instance_key"] = (
        micro["candidate_id"].astype(str)
        + "|"
        + micro["split_id"].astype(str)
        + "|"
        + micro["scenario_id"].astype(str)
    )
    micro = micro.loc[micro["instance_key"].isin(instance_lookup["instance_key"])].copy()

    minute_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    divergence_rows: list[dict[str, Any]] = []

    for candidate in finalist_candidates.itertuples(index=False):
        detail = weight_history.loc[weight_history["portfolio_pk"] == candidate.portfolio_pk].copy()
        if detail.shape[0] <= 0:
            raise Module6ValidationError(f"missing weight history for finalist {candidate.portfolio_pk}")
        path_df = session_paths.loc[session_paths["portfolio_pk"] == candidate.portfolio_pk].copy()
        if path_df.shape[0] <= 0:
            raise Module6ValidationError(f"missing session path for finalist {candidate.portfolio_pk}")
        equity = 1.0
        peak = 1.0
        gross_peak_seen = 0.0
        breach_count = 0
        disable = False
        portfolio_session_returns: list[float] = []
        portfolio_session_turnover: list[float] = []
        for session_row in path_df.sort_values("session_id", kind="mergesort").itertuples(index=False):
            session_detail = detail.loc[detail["session_id"] == int(session_row.session_id)].copy()
            if session_detail.shape[0] <= 0:
                continue
            session_detail = session_detail.merge(
                instance_lookup[["strategy_instance_pk", "instance_key"]],
                on="strategy_instance_pk",
                how="left",
            )
            if session_detail["instance_key"].isna().any():
                raise Module6ValidationError(f"unresolved instance mapping for finalist {candidate.portfolio_pk}")
            session_eq = eq.loc[eq["session_id"] == int(session_row.session_id)].copy()
            if session_eq.shape[0] <= 0:
                raise Module6ValidationError(f"missing minute equity session for finalist {candidate.portfolio_pk} session={int(session_row.session_id)}")
            session_trade = trade.loc[trade["session_id"] == int(session_row.session_id)].copy()
            session_micro = micro.loc[micro["session_id"] == int(session_row.session_id)].copy()
            times = np.asarray(
                sorted(
                    set(pd.unique(session_eq["ts_ns"]).tolist())
                    | set(pd.unique(session_trade["ts_ns"]).tolist())
                    | set(pd.unique(session_micro["ts_ns"]).tolist())
                ),
                dtype=np.int64,
            )
            if times.size <= 0:
                raise Module6ValidationError(
                    f"missing minute truth timestamps for finalist {candidate.portfolio_pk} session={int(session_row.session_id)}"
                )
            component_caps: dict[str, np.ndarray] = {}
            component_gross: dict[str, np.ndarray] = {}
            session_start_equity = float(equity)
            session_start_cash_cap = float(session_row.session_start_cash_weight) * session_start_equity
            embedded_trade_notional = 0.0
            embedded_trade_cost_abs = 0.0
            micro_trade_cost_abs = 0.0
            micro_trade_notional = 0.0
            for comp in session_detail.itertuples(index=False):
                loc = session_eq.loc[session_eq["instance_key"] == str(comp.instance_key)].copy()
                loc = loc.sort_values("ts_ns", kind="mergesort")
                if float(comp.start_weight) > 1.0e-12 and int(comp.availability_state_code) in (1, 2) and loc.shape[0] <= 0:
                    raise Module6ValidationError(
                        f"missing equity truth for active sleeve; portfolio_pk={candidate.portfolio_pk} session_id={int(session_row.session_id)} strategy_instance_pk={comp.strategy_instance_pk}"
                    )
                if loc.shape[0] <= 0:
                    vals = np.full(times.shape[0], float(comp.start_weight) * session_start_equity, dtype=np.float64)
                    gross_frac = np.zeros(times.shape[0], dtype=np.float64)
                    base_eq = float(session_start_equity)
                else:
                    loc = (
                        loc.drop_duplicates("ts_ns", keep="last")
                        .set_index("ts_ns")
                        .reindex(times)
                        .sort_index()
                        .ffill()
                        .bfill()
                        .reset_index()
                    )
                    eq_series = np.asarray(loc["equity"], dtype=np.float64)
                    base_eq = float(eq_series[0])
                    if not np.isfinite(base_eq) or base_eq <= 0.0:
                        raise Module6ValidationError(
                            f"invalid minute equity base; portfolio_pk={candidate.portfolio_pk} session_id={int(session_row.session_id)} strategy_instance_pk={comp.strategy_instance_pk}"
                        )
                    vals = float(comp.start_weight) * session_start_equity * (eq_series / max(base_eq, 1.0e-12))
                    gross_frac = np.asarray(loc["margin_used"], dtype=np.float64) / np.maximum(np.asarray(loc["equity"], dtype=np.float64), 1.0e-12)
                component_caps[str(comp.strategy_instance_pk)] = vals
                component_gross[str(comp.strategy_instance_pk)] = vals * gross_frac
                loc_trade = session_trade.loc[session_trade["instance_key"] == str(comp.instance_key)].copy()
                component_trade_notional = 0.0
                component_trade_cost_abs = 0.0
                if loc_trade.shape[0] > 0:
                    scale = float(comp.start_weight) * session_start_equity / max(base_eq, 1.0e-12)
                    trade_notional = np.abs(np.asarray(loc_trade["filled_qty"], dtype=np.float64) * np.asarray(loc_trade["exec_price"], dtype=np.float64))
                    component_trade_notional = float(np.sum(trade_notional) * scale)
                    component_trade_cost_abs = float(np.sum(np.asarray(loc_trade["trade_cost"], dtype=np.float64)) * scale)
                    embedded_trade_notional += component_trade_notional
                    embedded_trade_cost_abs += component_trade_cost_abs
                loc_micro = session_micro.loc[session_micro["instance_key"] == str(comp.instance_key)].copy()
                component_micro_trade_cost_abs = 0.0
                component_micro_trade_notional = 0.0
                if loc_trade.shape[0] > 0 and loc_micro.shape[0] <= 0:
                    raise Module6ValidationError(
                        f"missing micro truth for traded sleeve; portfolio_pk={candidate.portfolio_pk} session_id={int(session_row.session_id)} strategy_instance_pk={comp.strategy_instance_pk}"
                    )
                if loc_micro.shape[0] > 0:
                    scale = float(comp.start_weight) * session_start_equity / max(base_eq, 1.0e-12)
                    component_micro_trade_cost_abs = float(np.sum(np.asarray(loc_micro["trade_cost"], dtype=np.float64)) * scale)
                    component_micro_trade_notional = float(
                        np.sum(
                            np.abs(
                                np.asarray(loc_micro["filled_qty"], dtype=np.float64)
                                * np.asarray(loc_micro["exec_price"], dtype=np.float64)
                            )
                        )
                        * scale
                    )
                    micro_trade_cost_abs += component_micro_trade_cost_abs
                    micro_trade_notional += component_micro_trade_notional
                component_rows.append(
                    {
                        "portfolio_pk": str(candidate.portfolio_pk),
                        "session_id": int(session_row.session_id),
                        "strategy_instance_pk": str(comp.strategy_instance_pk),
                        "availability_state_code": int(comp.availability_state_code),
                        "forced_cash_weight": float(max(float(comp.available_target_weight) - float(comp.end_weight), 0.0)) if int(comp.availability_state_code) in (4, 5) else 0.0,
                        "unavailable_target_weight": float(max(float(comp.requested_target_weight) - float(comp.available_target_weight), 0.0)) if int(comp.availability_state_code) in (3, 6) else 0.0,
                        "embedded_trade_cost_abs": float(component_trade_cost_abs),
                        "embedded_trade_notional": float(component_trade_notional),
                        "micro_trade_cost_abs": float(component_micro_trade_cost_abs),
                        "micro_trade_notional": float(component_micro_trade_notional),
                    }
                )
            cash_cap = float(session_start_cash_cap)
            flattened = False
            for k, ts_ns in enumerate(times.tolist()):
                sleeve_total = float(sum(vals[k] for vals in component_caps.values()))
                gross_total = float(sum(vals[k] for vals in component_gross.values()))
                portfolio_equity = cash_cap + sleeve_total
                dd = float(1.0 - portfolio_equity / max(peak, 1.0e-12))
                if portfolio_equity < float(config.simulator.account_disable_equity):
                    breach_count += 1
                    disable = True
                    flattened = True
                if (session_start_equity - portfolio_equity) > float(config.simulator.daily_loss_limit_frac) * max(session_start_equity, 1.0e-12):
                    breach_count += 1
                    disable = True
                    flattened = True
                peak = max(peak, portfolio_equity)
                gross_peak_seen = max(gross_peak_seen, gross_total / max(portfolio_equity, 1.0e-12))
                minute_rows.append(
                    {
                        "portfolio_pk": str(candidate.portfolio_pk),
                        "reduced_universe_id": str(candidate.reduced_universe_id),
                        "ts_ns": int(ts_ns),
                        "session_id": int(session_row.session_id),
                        "equity": float(portfolio_equity),
                        "drawdown": float(dd),
                        "gross_exposure_mult": float(gross_total / max(portfolio_equity, 1.0e-12)),
                        "disable_flag": int(disable),
                    }
                )
                if flattened:
                    for kk in range(k + 1, len(times)):
                        minute_rows.append(
                            {
                                "portfolio_pk": str(candidate.portfolio_pk),
                                "reduced_universe_id": str(candidate.reduced_universe_id),
                                "ts_ns": int(times[kk]),
                                "session_id": int(session_row.session_id),
                                "equity": float(portfolio_equity),
                                "drawdown": float(dd),
                                "gross_exposure_mult": 0.0,
                                "disable_flag": int(disable),
                            }
                        )
                    equity = float(portfolio_equity)
                    break
            if not flattened:
                equity = float(cash_cap + sum(vals[-1] for vals in component_caps.values()))
            portfolio_turnover = float(
                0.5
                * np.sum(
                    np.abs(
                        np.asarray(session_detail["pre_rebalance_weight"], dtype=np.float64)
                        - np.asarray(session_detail["end_weight"], dtype=np.float64)
                    )
                )
            )
            embedded_turnover_frac = float(embedded_trade_notional / max(session_start_equity, 1.0e-12))
            liquidity_penalty = 1.0 + embedded_turnover_frac
            rebalance_cost_abs = (
                float(config.simulator.fixed_fee) * float(portfolio_turnover > 0.0)
                + float(config.simulator.linear_cost_bps) * 1.0e-4 * portfolio_turnover * equity
                + float(config.simulator.slippage_cost_bps) * 1.0e-4 * portfolio_turnover * equity * liquidity_penalty
            )
            equity = max(0.0, float(equity - rebalance_cost_abs))
            portfolio_session_returns.append(float(equity / max(session_start_equity, 1.0e-12) - 1.0))
            portfolio_session_turnover.append(float(portfolio_turnover))
        minute_score = float(np.mean(portfolio_session_returns) * 252.0 - max((1.0 - equity / max(peak, 1.0e-12)), 0.0))
        max_dd = float(max((row["drawdown"] for row in minute_rows if row["portfolio_pk"] == str(candidate.portfolio_pk)), default=0.0))
        minute_turnover = float(np.mean(portfolio_session_turnover) if portfolio_session_turnover else 0.0)
        summary_rows.append(
            {
                "portfolio_pk": str(candidate.portfolio_pk),
                "minute_score": float(minute_score),
                "minute_annualized_return": float(np.mean(portfolio_session_returns) * 252.0 if portfolio_session_returns else 0.0),
                "minute_max_drawdown": float(max_dd),
                "minute_turnover": float(minute_turnover),
                "minute_gross_exposure_peak": float(gross_peak_seen),
                "minute_final_equity": float(equity),
                "minute_breach_count": int(breach_count),
            }
        )

    minute_summary = pd.DataFrame(summary_rows).sort_values(["portfolio_pk"], kind="mergesort").reset_index(drop=True)
    session_score_map = dict(
        session_summary[["portfolio_pk", "first_pass_score"]].itertuples(index=False, name=None)
    ) if "first_pass_score" in session_summary.columns else {str(row.portfolio_pk): 0.0 for row in session_summary.itertuples(index=False)}
    session_return_map = dict(session_summary[["portfolio_pk", "annualized_return"]].itertuples(index=False, name=None))
    session_dd_map = dict(session_summary[["portfolio_pk", "max_drawdown"]].itertuples(index=False, name=None))
    session_to_map = dict(session_summary[["portfolio_pk", "turnover"]].itertuples(index=False, name=None))
    session_gross_map = dict(session_summary[["portfolio_pk", "gross_exposure_peak"]].itertuples(index=False, name=None)) if "gross_exposure_peak" in session_summary.columns else {str(row.portfolio_pk): 0.0 for row in session_summary.itertuples(index=False)}
    session_breach_map = dict(session_summary[["portfolio_pk", "breach_count"]].itertuples(index=False, name=None))

    minute_summary["session_score"] = minute_summary["portfolio_pk"].map(lambda x: float(session_score_map.get(str(x), 0.0)))
    minute_summary["session_annualized_return"] = minute_summary["portfolio_pk"].map(lambda x: float(session_return_map.get(str(x), 0.0)))
    minute_summary["session_max_drawdown"] = minute_summary["portfolio_pk"].map(lambda x: float(session_dd_map.get(str(x), 0.0)))
    minute_summary["session_turnover"] = minute_summary["portfolio_pk"].map(lambda x: float(session_to_map.get(str(x), 0.0)))
    minute_summary["session_gross_exposure_peak"] = minute_summary["portfolio_pk"].map(lambda x: float(session_gross_map.get(str(x), 0.0)))
    minute_summary["session_breach_count"] = minute_summary["portfolio_pk"].map(lambda x: int(session_breach_map.get(str(x), 0)))
    minute_summary["return_drift"] = minute_summary["minute_annualized_return"] - minute_summary["session_annualized_return"]
    minute_summary["drawdown_drift"] = minute_summary["minute_max_drawdown"] - minute_summary["session_max_drawdown"]
    minute_summary["turnover_drift"] = minute_summary["minute_turnover"] - minute_summary["session_turnover"]
    minute_summary["gross_exposure_drift"] = minute_summary["minute_gross_exposure_peak"] - minute_summary["session_gross_exposure_peak"]
    minute_summary["breach_count_delta"] = minute_summary["minute_breach_count"] - minute_summary["session_breach_count"]
    minute_summary["rank_session"] = minute_summary["session_score"].rank(method="dense", ascending=False).astype(int)
    minute_summary["rank_minute"] = minute_summary["minute_score"].rank(method="dense", ascending=False).astype(int)
    minute_summary["rank_delta"] = minute_summary["rank_minute"] - minute_summary["rank_session"]
    minute_summary["rejected"] = False
    for row in minute_summary.itertuples(index=False):
        reason = None
        if float(row.minute_score) < float(config.scoring.min_truth_score_ratio) * float(row.session_score):
            reason = "MINUTE_REPLAY_SCORE_COLLAPSE"
        elif abs(float(row.return_drift)) > max(float(config.scoring.return_drift_floor), float(config.scoring.max_allowed_return_drift_frac) * max(abs(float(row.session_annualized_return)), float(config.scoring.return_scale_floor))):
            reason = "MINUTE_REPLAY_RETURN_DRIFT"
        elif abs(float(row.drawdown_drift)) > float(config.scoring.max_allowed_drawdown_drift):
            reason = "MINUTE_REPLAY_DD_DRIFT"
        elif abs(float(row.turnover_drift)) > max(float(config.scoring.turnover_drift_floor), float(config.scoring.max_allowed_turnover_drift_frac) * max(float(row.session_turnover), float(config.scoring.turnover_scale_floor))):
            reason = "MINUTE_REPLAY_TURNOVER_DRIFT"
        elif abs(float(row.gross_exposure_drift)) > max(float(config.scoring.gross_exposure_drift_floor), float(config.scoring.max_allowed_gross_exposure_drift_frac) * max(float(row.session_gross_exposure_peak), float(config.scoring.gross_exposure_scale_floor))):
            reason = "MINUTE_REPLAY_GROSS_DRIFT"
        elif int(row.breach_count_delta) > int(config.scoring.max_allowed_breach_count_delta):
            reason = "MINUTE_REPLAY_NEW_BREACH"
        divergence_rows.append(
            {
                "portfolio_pk": str(row.portfolio_pk),
                "session_score": float(row.session_score),
                "minute_score": float(row.minute_score),
                "return_drift": float(row.return_drift),
                "drawdown_drift": float(row.drawdown_drift),
                "turnover_drift": float(row.turnover_drift),
                "gross_exposure_drift": float(row.gross_exposure_drift),
                "breach_count_delta": int(row.breach_count_delta),
                "rank_session": int(row.rank_session),
                "rank_minute": int(row.rank_minute),
                "rank_delta": int(row.rank_delta),
                "reject_reason": "" if reason is None else str(reason),
            }
        )
    divergence = pd.DataFrame(divergence_rows).sort_values(["portfolio_pk"], kind="mergesort").reset_index(drop=True)
    corr = 1.0
    if divergence.shape[0] > 1:
        session_scores_arr = np.asarray(divergence["session_score"], dtype=np.float64)
        minute_scores_arr = np.asarray(divergence["minute_score"], dtype=np.float64)
        if np.allclose(session_scores_arr, session_scores_arr[0]) or np.allclose(minute_scores_arr, minute_scores_arr[0]):
            corr = 1.0
        else:
            corr = float(spearmanr(session_scores_arr, minute_scores_arr).correlation)
            if not np.isfinite(corr):
                corr = 1.0
    if corr < float(config.scoring.min_rank_stability):
        raise Module6ValidationError("SCREENING_TRUTH_RANK_INSTABILITY")
    if float(np.percentile(np.abs(np.asarray(divergence["rank_delta"], dtype=np.float64)), 95)) > float(config.scoring.max_abs_rank_delta_p95):
        raise Module6ValidationError("SCREENING_TRUTH_RANK_DRIFT_P95")
    return MinuteReplayArtifacts(
        minute_paths=pd.DataFrame(minute_rows).sort_values(["portfolio_pk", "ts_ns"], kind="mergesort").reset_index(drop=True),
        component_diagnostics=pd.DataFrame(component_rows).sort_values(["portfolio_pk", "session_id", "strategy_instance_pk"], kind="mergesort").reset_index(drop=True),
        divergence=divergence,
        minute_summary=minute_summary.sort_values(["portfolio_pk"], kind="mergesort").reset_index(drop=True),
    )
