from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from module6.types import PortfolioSelectionReport
from module6.utils import ensure_directory


def write_module6_outputs(
    *,
    output_dir: Path,
    candidates: pd.DataFrame,
    portfolio_weights: pd.DataFrame,
    session_paths: pd.DataFrame,
    minute_paths: pd.DataFrame,
    minute_component_diagnostics: pd.DataFrame,
    session_scores: pd.DataFrame,
    finalist_scores: pd.DataFrame,
    comparable_scores: pd.DataFrame,
    divergence: pd.DataFrame,
    global_frontier: pd.DataFrame,
    risk_return_frontier: pd.DataFrame,
    operational_frontier: pd.DataFrame,
    selected_frontier: pd.DataFrame,
    selection_report: PortfolioSelectionReport,
) -> None:
    out = ensure_directory(output_dir)
    frontiers_dir = ensure_directory(out / "frontiers")
    candidates.to_parquet(out / "portfolio_candidates.parquet", index=False)
    portfolio_weights.to_parquet(out / "portfolio_weights.parquet", index=False)
    session_paths.to_parquet(out / "portfolio_paths_session.parquet", index=False)
    minute_paths.to_parquet(out / "portfolio_paths_minute_finalists.parquet", index=False)
    minute_component_diagnostics.to_parquet(out / "minute_component_diagnostics.parquet", index=False)
    session_scores.to_parquet(out / "portfolio_scores_session.parquet", index=False)
    finalist_scores.to_parquet(out / "portfolio_scores.parquet", index=False)
    comparable_scores.to_parquet(out / "cross_universe_comparison_scores.parquet", index=False)
    divergence.to_parquet(out / "screening_truth_divergence.parquet", index=False)
    global_frontier.to_parquet(frontiers_dir / "global_frontier.parquet", index=False)
    risk_return_frontier.to_parquet(frontiers_dir / "risk_return_frontier.parquet", index=False)
    operational_frontier.to_parquet(frontiers_dir / "operational_frontier.parquet", index=False)
    selected_frontier.to_parquet(frontiers_dir / "selected_frontier.parquet", index=False)
    (out / "portfolio_selection_report.json").write_text(
        json.dumps(
            {
                "run_id": str(selection_report.run_id),
                "output_dir": str(selection_report.output_dir),
                "selected_portfolio_pks": list(selection_report.selected_portfolio_pks),
                "alternate_portfolio_pks": list(selection_report.alternate_portfolio_pks),
                "summary": dict(selection_report.summary),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
