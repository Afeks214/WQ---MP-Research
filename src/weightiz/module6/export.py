from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from weightiz.module6.types import PortfolioSelectionReport
from weightiz.module6.utils import ensure_directory


def write_module6_outputs(
    *,
    output_dir: Path,
    candidates: pd.DataFrame,
    portfolio_weights: pd.DataFrame,
    session_paths: pd.DataFrame,
    comparison_support_session_paths: pd.DataFrame,
    minute_paths: pd.DataFrame,
    minute_component_diagnostics: pd.DataFrame,
    session_scores: pd.DataFrame,
    comparison_support_session_scores: pd.DataFrame,
    finalist_scores: pd.DataFrame,
    comparable_scores: pd.DataFrame,
    divergence: pd.DataFrame,
    weight_history: pd.DataFrame,
    comparison_support_calendar: pd.DataFrame,
    dependence_artifacts: dict[str, Any],
    overlap_proxy: Any,
    overlap_proxy_index: pd.DataFrame,
    exact_overlap: pd.DataFrame,
    global_frontier: pd.DataFrame,
    risk_return_frontier: pd.DataFrame,
    operational_frontier: pd.DataFrame,
    selected_frontier: pd.DataFrame,
    selection_report: PortfolioSelectionReport,
) -> None:
    out = ensure_directory(output_dir)
    frontiers_dir = ensure_directory(out / "frontiers")
    dependence_dir = ensure_directory(out / "dependence")
    overlap_dir = ensure_directory(out / "overlap")
    candidates.to_parquet(out / "portfolio_candidates.parquet", index=False)
    portfolio_weights.to_parquet(out / "portfolio_weights.parquet", index=False)
    session_paths.to_parquet(out / "portfolio_paths_session.parquet", index=False)
    comparison_support_session_paths.to_parquet(out / "portfolio_paths_session_comparison_support.parquet", index=False)
    minute_paths.to_parquet(out / "portfolio_paths_minute_finalists.parquet", index=False)
    minute_component_diagnostics.to_parquet(out / "minute_component_diagnostics.parquet", index=False)
    session_scores.to_parquet(out / "portfolio_scores_session.parquet", index=False)
    comparison_support_session_scores.to_parquet(out / "portfolio_scores_session_comparison_support.parquet", index=False)
    finalist_scores.to_parquet(out / "portfolio_scores.parquet", index=False)
    comparable_scores.to_parquet(out / "cross_universe_comparison_scores.parquet", index=False)
    divergence.to_parquet(out / "screening_truth_divergence.parquet", index=False)
    weight_history.to_parquet(out / "portfolio_weight_history.parquet", index=False)
    comparison_support_calendar.to_parquet(out / "comparison_support_calendar.parquet", index=False)
    global_frontier.to_parquet(frontiers_dir / "global_frontier.parquet", index=False)
    risk_return_frontier.to_parquet(frontiers_dir / "risk_return_frontier.parquet", index=False)
    operational_frontier.to_parquet(frontiers_dir / "operational_frontier.parquet", index=False)
    selected_frontier.to_parquet(frontiers_dir / "selected_frontier.parquet", index=False)
    overlap_proxy_index.to_parquet(overlap_dir / "execution_overlap_proxy_index.parquet", index=False)
    sparse.save_npz(overlap_dir / "execution_overlap_proxy_symbol_support.npz", overlap_proxy.symbol_support)
    sparse.save_npz(overlap_dir / "execution_overlap_proxy_activity_concurrence.npz", overlap_proxy.activity_concurrence)
    sparse.save_npz(overlap_dir / "execution_overlap_proxy_gross_concurrence.npz", overlap_proxy.gross_exposure_concurrence)
    sparse.save_npz(overlap_dir / "execution_overlap_proxy_rebalance_collision.npz", overlap_proxy.rebalance_collision)
    sparse.save_npz(overlap_dir / "execution_overlap_proxy_composite.npz", overlap_proxy.composite)
    exact_overlap.to_parquet(overlap_dir / "finalist_exact_overlap.parquet", index=False)
    for reduced_universe_id, bundle in sorted(dependence_artifacts.items(), key=lambda item: str(item[0])):
        dep_out = ensure_directory(dependence_dir / str(reduced_universe_id))
        np.save(dep_out / "covariance.npy", np.asarray(bundle.covariance, dtype=np.float64))
        np.save(dep_out / "correlation.npy", np.asarray(bundle.correlation, dtype=np.float64))
        np.save(dep_out / "downside_covariance.npy", np.asarray(bundle.downside_covariance, dtype=np.float64))
        np.save(dep_out / "regime_overlap.npy", np.asarray(bundle.regime_overlap, dtype=np.float64))
        np.save(dep_out / "common_support.npy", np.asarray(bundle.common_support, dtype=bool))
        sparse.save_npz(dep_out / "drawdown_concurrence.npz", bundle.drawdown_concurrence.tocsr())
        (dep_out / "metadata.json").write_text(
            json.dumps(
                {
                    "shrinkage": float(bundle.shrinkage),
                    "negative_mass": float(bundle.negative_mass),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    selected_details = comparable_scores.loc[
        comparable_scores["portfolio_pk"].isin(selection_report.selected_portfolio_pks)
    ].sort_values(["comparable_truth_score", "portfolio_pk"], ascending=[False, True], kind="mergesort")
    alternate_details = comparable_scores.loc[
        comparable_scores["portfolio_pk"].isin(selection_report.alternate_portfolio_pks)
    ].sort_values(["comparable_truth_score", "portfolio_pk"], ascending=[False, True], kind="mergesort")
    (out / "portfolio_selection_report.json").write_text(
        json.dumps(
            {
                "run_id": str(selection_report.run_id),
                "output_dir": str(selection_report.output_dir),
                "selected_portfolio_pks": list(selection_report.selected_portfolio_pks),
                "alternate_portfolio_pks": list(selection_report.alternate_portfolio_pks),
                "summary": dict(selection_report.summary),
                "selected_details": selected_details.to_dict("records"),
                "alternate_details": alternate_details.to_dict("records"),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
