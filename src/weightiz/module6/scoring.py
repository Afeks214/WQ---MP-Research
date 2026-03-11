from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from weightiz.module6.config import Module6Config
from weightiz.module6.utils import Module6ValidationError, normalized_rank


@dataclass(frozen=True)
class ScoredPortfolios:
    session_scores: pd.DataFrame
    finalist_scores: pd.DataFrame
    comparable_scores: pd.DataFrame


def _weight_concentrations(portfolio_weights: pd.DataFrame, strategy_frame: pd.DataFrame) -> pd.DataFrame:
    merged = portfolio_weights.merge(
        strategy_frame[["strategy_instance_pk", "cluster_id", "family_id"]],
        on="strategy_instance_pk",
        how="left",
    )
    cluster = (
        merged.groupby(["portfolio_pk", "cluster_id"], dropna=False)["target_weight"].sum().groupby("portfolio_pk").max().rename("cluster_concentration")
    )
    family = (
        merged.groupby(["portfolio_pk", "family_id"], dropna=False)["target_weight"].sum().groupby("portfolio_pk").max().rename("family_concentration")
    )
    sleeve = merged.groupby("portfolio_pk", dropna=False)["target_weight"].max().rename("sleeve_concentration")
    return pd.concat([cluster, family, sleeve], axis=1).reset_index()


def _portfolio_internal_overlap(portfolio_weights: pd.DataFrame, strategy_frame: pd.DataFrame, execution_overlap_proxy) -> pd.DataFrame:
    if execution_overlap_proxy is None or "overlap_proxy_idx" not in strategy_frame.columns:
        return pd.DataFrame(columns=["portfolio_pk", "internal_overlap_proxy"])
    idx_map = dict(strategy_frame[["strategy_instance_pk", "overlap_proxy_idx"]].itertuples(index=False, name=None))
    overlap = execution_overlap_proxy.composite.tocsr()
    rows: list[dict[str, float | str]] = []
    for portfolio_pk, grp in portfolio_weights.groupby("portfolio_pk", dropna=False, sort=True):
        local_idx = [idx_map.get(str(pk)) for pk in grp["strategy_instance_pk"].astype(str).tolist()]
        if any(idx is None for idx in local_idx):
            continue
        weights = np.asarray(grp["target_weight"], dtype=np.float64)
        mat = overlap[np.asarray(local_idx, dtype=np.int64)][:, np.asarray(local_idx, dtype=np.int64)].toarray()
        score = float(weights @ mat @ weights)
        rows.append({"portfolio_pk": str(portfolio_pk), "internal_overlap_proxy": float(score)})
    return pd.DataFrame(rows)


def score_session_paths(
    *,
    session_paths: pd.DataFrame,
    session_summary: pd.DataFrame,
    portfolio_weights: pd.DataFrame,
    strategy_frame: pd.DataFrame,
    config: Module6Config,
    execution_overlap_proxy=None,
) -> pd.DataFrame:
    if session_summary.shape[0] <= 0:
        return pd.DataFrame()
    gross_peak = session_paths.groupby("portfolio_pk", dropna=False)["gross_exposure_mult"].max().rename("gross_exposure_peak").reset_index()
    concentrations = _weight_concentrations(portfolio_weights, strategy_frame)
    internal_overlap = _portfolio_internal_overlap(portfolio_weights, strategy_frame, execution_overlap_proxy)
    scored = (
        session_summary.merge(gross_peak, on="portfolio_pk", how="left")
        .merge(concentrations, on="portfolio_pk", how="left")
        .merge(internal_overlap, on="portfolio_pk", how="left")
    )
    scored["availability_burden"] = 0.70 * scored["missingness_burden"].fillna(0.0) + 0.30 * scored["forced_cash_burden"].fillna(0.0)
    scored["headroom"] = 1.0 - np.minimum(
        1.0,
        scored["gross_exposure_peak"].fillna(0.0) / max(config.simulator.intraday_leverage_max, 1.0e-12),
    )
    scored["hard_reject"] = (
        scored["breach_count"].fillna(0).astype(int) > 0
    ) | (
        scored["support_coverage"].fillna(0.0) < float(config.intake.min_availability_ratio)
    ) | (
        scored["final_equity"].fillna(0.0) <= 0.0
    )
    calmar = scored["annualized_return"].fillna(0.0) / np.maximum(scored["max_drawdown"].fillna(0.0), 1.0e-6)
    ranked = pd.DataFrame(
        {
            "portfolio_pk": scored["portfolio_pk"].astype(str),
            "calmar_rank": normalized_rank(calmar.to_numpy(dtype=np.float64), ascending=False),
            "return_rank": normalized_rank(scored["annualized_return"].fillna(0.0).to_numpy(dtype=np.float64), ascending=False),
            "drawdown_rank": normalized_rank(scored["max_drawdown"].fillna(0.0).to_numpy(dtype=np.float64), ascending=True),
            "headroom_rank": normalized_rank(scored["headroom"].fillna(0.0).to_numpy(dtype=np.float64), ascending=False),
            "turnover_rank": normalized_rank(scored["turnover"].fillna(0.0).to_numpy(dtype=np.float64), ascending=True),
            "concentration_rank": normalized_rank(scored["cluster_concentration"].fillna(1.0).to_numpy(dtype=np.float64), ascending=True),
            "overlap_rank": normalized_rank(pd.to_numeric(scored["internal_overlap_proxy"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64), ascending=True),
        }
    )
    scored = scored.merge(ranked, on="portfolio_pk", how="left")
    scored["first_pass_score"] = (
        0.30 * scored["calmar_rank"]
        + 0.20 * scored["drawdown_rank"]
        + 0.15 * scored["return_rank"]
        + 0.15 * scored["headroom_rank"]
        + 0.10 * scored["turnover_rank"]
        + 0.05 * scored["concentration_rank"]
        + 0.05 * scored["overlap_rank"]
    )
    scored.loc[scored["hard_reject"], "first_pass_score"] = -np.inf
    return scored.sort_values(["first_pass_score", "portfolio_pk"], ascending=[False, True], kind="mergesort").reset_index(drop=True)


def build_cross_universe_comparable_scores(
    *,
    finalist_scores: pd.DataFrame,
    config: Module6Config,
    comparison_support: pd.DataFrame | np.ndarray,
) -> pd.DataFrame:
    if finalist_scores.shape[0] <= 0:
        return pd.DataFrame()
    if isinstance(comparison_support, pd.DataFrame):
        support_count = int(comparison_support.shape[0])
    else:
        support_count = int(np.asarray(comparison_support, dtype=np.int64).shape[0])
    if support_count <= 0:
        raise Module6ValidationError("comparison_support must be non-empty")
    required = {
        "portfolio_pk",
        "calendar_version",
        "support_policy_version",
        "comparison_support_recomputed",
        "minute_annualized_return",
        "minute_max_drawdown",
        "minute_turnover",
        "support_coverage",
        "availability_burden",
    }
    if not required.issubset(set(finalist_scores.columns)):
        missing = sorted(required - set(finalist_scores.columns))
        raise Module6ValidationError(f"comparable scoring missing required columns: {missing}")
    if finalist_scores["calendar_version"].astype(str).nunique() != 1:
        raise Module6ValidationError("CROSS_UNIVERSE_CALENDAR_MISMATCH")
    if finalist_scores["support_policy_version"].astype(str).nunique() != 1:
        raise Module6ValidationError("CROSS_UNIVERSE_SUPPORT_POLICY_MISMATCH")
    comparable = finalist_scores.copy()
    if not comparable["comparison_support_recomputed"].fillna(False).astype(bool).all():
        raise Module6ValidationError("cross-universe comparable scoring requires canonical comparison-support recomputation")
    required_support = float(max(config.intake.required_comparison_support, config.scoring.min_cross_universe_support))
    comparable["comparison_support_session_count"] = int(support_count)
    comparable["required_comparison_support"] = float(required_support)
    comparable["cross_universe_reject"] = comparable["support_coverage"].fillna(0.0) < float(required_support)
    comparable.loc[comparable["cross_universe_reject"], "cross_universe_reject_reason"] = "CROSS_UNIVERSE_SUPPORT_TOO_SHORT"
    comparable["return_rank_truth"] = normalized_rank(comparable["minute_annualized_return"].fillna(0.0).to_numpy(dtype=np.float64), ascending=False)
    comparable["drawdown_rank_truth"] = normalized_rank(comparable["minute_max_drawdown"].fillna(0.0).to_numpy(dtype=np.float64), ascending=True)
    comparable["turnover_rank_truth"] = normalized_rank(comparable["minute_turnover"].fillna(0.0).to_numpy(dtype=np.float64), ascending=True)
    comparable["availability_rank_truth"] = normalized_rank(comparable["availability_burden"].fillna(1.0).to_numpy(dtype=np.float64), ascending=True)
    comparable["comparable_truth_score"] = (
        0.35 * comparable["return_rank_truth"]
        + 0.25 * comparable["drawdown_rank_truth"]
        + 0.20 * comparable["turnover_rank_truth"]
        + 0.20 * comparable["availability_rank_truth"]
    )
    comparable.loc[comparable["cross_universe_reject"], "comparable_truth_score"] = -np.inf
    return comparable.sort_values(["comparable_truth_score", "portfolio_pk"], ascending=[False, True], kind="mergesort").reset_index(drop=True)


def score_finalists(
    *,
    session_scores: pd.DataFrame,
    minute_summary: pd.DataFrame,
    divergence: pd.DataFrame,
    portfolio_weights: pd.DataFrame,
    strategy_frame: pd.DataFrame,
    config: Module6Config,
    execution_overlap_proxy=None,
) -> pd.DataFrame:
    if minute_summary.shape[0] <= 0:
        return pd.DataFrame()
    concentrations = _weight_concentrations(portfolio_weights, strategy_frame)
    internal_overlap = _portfolio_internal_overlap(portfolio_weights, strategy_frame, execution_overlap_proxy)
    finalists = (
        minute_summary.merge(
            session_scores[
                [
                    "portfolio_pk",
                    "reduced_universe_id",
                    "support_coverage",
                    "availability_burden",
                    "first_pass_score",
                    "calendar_version",
                    "support_policy_version",
                ]
            ],
            on="portfolio_pk",
            how="left",
        )
        .merge(divergence, on="portfolio_pk", how="left", suffixes=("", "_div"))
        .merge(concentrations, on="portfolio_pk", how="left")
        .merge(internal_overlap, on="portfolio_pk", how="left")
    )
    finalists["rejected"] = finalists["reject_reason"].fillna("").astype(str).str.len() > 0
    finalists["truth_calmar"] = finalists["minute_annualized_return"].fillna(0.0) / np.maximum(finalists["minute_max_drawdown"].fillna(0.0), 1.0e-6)
    finalists["truth_return_rank"] = normalized_rank(finalists["minute_annualized_return"].fillna(0.0).to_numpy(dtype=np.float64), ascending=False)
    finalists["truth_calmar_rank"] = normalized_rank(finalists["truth_calmar"].to_numpy(dtype=np.float64), ascending=False)
    finalists["truth_turnover_rank"] = normalized_rank(finalists["minute_turnover"].fillna(0.0).to_numpy(dtype=np.float64), ascending=True)
    finalists["truth_overlap_rank"] = normalized_rank(pd.to_numeric(finalists["internal_overlap_proxy"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64), ascending=True)
    finalists["truth_availability_rank"] = normalized_rank(finalists["availability_burden"].fillna(1.0).to_numpy(dtype=np.float64), ascending=True)
    finalists["final_score"] = (
        0.25 * finalists["truth_calmar_rank"]
        + 0.20 * finalists["truth_return_rank"]
        + 0.15 * finalists["first_pass_score"].replace([-np.inf], 0.0)
        + 0.15 * finalists["truth_availability_rank"]
        + 0.10 * finalists["truth_overlap_rank"]
        + 0.10 * finalists["truth_turnover_rank"]
        + 0.05 * normalized_rank((1.0 - finalists["cluster_concentration"].fillna(1.0)).to_numpy(dtype=np.float64), ascending=False)
    )
    finalists.loc[finalists["rejected"], "final_score"] = -np.inf
    return finalists.sort_values(["final_score", "portfolio_pk"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
