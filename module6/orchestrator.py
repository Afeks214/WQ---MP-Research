from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from module6.config import Module6Config
from module6.dependence import build_covariance_bundle
from module6.export import write_module6_outputs
from module6.frontier import select_diverse_finalists
from module6.generators import generate_all_portfolios
from module6.io import load_module5_run
from module6.ledger import materialize_canonical_ledgers
from module6.matrices import build_matrix_store
from module6.reduction import reduce_universe
from module6.runtime import open_matrix_store
from module6.scoring import build_cross_universe_comparable_scores, score_finalists, score_session_paths
from module6.simulator.minute_refine import replay_finalists_minute
from module6.simulator.session_path import simulate_session_batch
from module6.types import PortfolioSelectionReport
from module6.utils import Module6ValidationError, ensure_directory


def run_module6_portfolio_research(
    run_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    config: Module6Config | None = None,
) -> PortfolioSelectionReport:
    cfg = config if config is not None else Module6Config()
    run = load_module5_run(run_dir=run_dir, config=cfg)
    out_dir = ensure_directory(Path(output_dir).resolve() if output_dir is not None else Path(run.paths.run_dir) / cfg.export.output_subdir_name)
    ledgers = materialize_canonical_ledgers(run=run, output_dir=out_dir, config=cfg)
    matrix_store = build_matrix_store(ledgers=ledgers, run=run, output_dir=out_dir, config=cfg)
    matrices = open_matrix_store(matrix_store)
    matrices["column_index"] = matrix_store.column_index
    reduction = reduce_universe(
        ledgers=ledgers,
        matrices=matrices,
        run=run,
        output_dir=out_dir,
        config=cfg,
    )
    all_candidates: list[pd.DataFrame] = []
    all_weights: list[pd.DataFrame] = []
    all_session_paths: list[pd.DataFrame] = []
    all_session_scores: list[pd.DataFrame] = []
    all_finalist_scores: list[pd.DataFrame] = []
    all_comparable_scores: list[pd.DataFrame] = []
    all_minute_paths: list[pd.DataFrame] = []
    all_component_diag: list[pd.DataFrame] = []
    all_divergence: list[pd.DataFrame] = []

    for reduced_universe in reduction.reduced_universes:
        strategy_frame = reduction.admitted_instances.loc[
            reduction.admitted_instances["strategy_instance_pk"].isin(reduced_universe.strategy_instance_pks)
        ].copy()
        strategy_frame = strategy_frame.sort_values(["strategy_instance_pk"], kind="mergesort").reset_index(drop=True)
        if strategy_frame.shape[0] <= 0:
            continue
        column_indices = strategy_frame["column_idx"].to_numpy(dtype="int64")
        covariance_bundle = build_covariance_bundle(
            returns_exec=matrices["R_exec"],
            availability=matrices["A"],
            regime_exposure=matrices["G"],
            column_indices=column_indices,
            config=cfg.dependence,
        )
        candidates_df, weights_df = generate_all_portfolios(
            reduced_universe=reduced_universe,
            strategy_frame=strategy_frame,
            covariance_bundle=covariance_bundle,
            returns_exec=matrices["R_exec"],
            column_indices=column_indices,
            config=cfg,
            calendar_version=str(strategy_frame["calendar_version"].iloc[0]),
        )
        if candidates_df.shape[0] <= 0:
            continue
        session_artifacts = simulate_session_batch(
            portfolio_candidates=candidates_df,
            portfolio_weights=weights_df,
            strategy_frame=strategy_frame,
            matrices=matrices,
            calendar=matrix_store.calendar,
            config=cfg,
            return_weight_history=False,
        )
        session_scores = score_session_paths(
            session_paths=session_artifacts.session_paths,
            session_summary=session_artifacts.portfolio_summary,
            portfolio_weights=weights_df,
            strategy_frame=strategy_frame,
            config=cfg,
        )
        session_scores["calendar_version"] = str(strategy_frame["calendar_version"].iloc[0])
        session_scores["support_policy_version"] = cfg.simulator.support_policy_version
        shortlist = session_scores.head(int(cfg.scoring.shortlist_minute_keep)).copy()
        finalist_candidates = candidates_df.loc[candidates_df["portfolio_pk"].isin(shortlist["portfolio_pk"])].copy()
        detailed_artifacts = simulate_session_batch(
            portfolio_candidates=finalist_candidates,
            portfolio_weights=weights_df,
            strategy_frame=strategy_frame,
            matrices=matrices,
            calendar=matrix_store.calendar,
            config=cfg,
            return_weight_history=True,
        )
        minute_artifacts = replay_finalists_minute(
            finalist_candidates=finalist_candidates,
            strategy_frame=strategy_frame,
            session_paths=detailed_artifacts.session_paths,
            session_summary=session_scores,
            weight_history=detailed_artifacts.weight_history,
            run=run,
            config=cfg,
        )
        finalist_scores = score_finalists(
            session_scores=session_scores,
            minute_summary=minute_artifacts.minute_summary,
            divergence=minute_artifacts.divergence,
            portfolio_weights=weights_df.loc[weights_df["portfolio_pk"].isin(finalist_candidates["portfolio_pk"])],
            strategy_frame=strategy_frame,
            config=cfg,
        )
        comparable_scores = build_cross_universe_comparable_scores(
            finalist_scores=finalist_scores,
            config=cfg,
        )
        all_candidates.append(candidates_df)
        all_weights.append(weights_df)
        all_session_paths.append(session_artifacts.session_paths)
        all_session_scores.append(session_scores)
        all_finalist_scores.append(finalist_scores)
        all_comparable_scores.append(comparable_scores)
        all_minute_paths.append(minute_artifacts.minute_paths)
        all_component_diag.append(minute_artifacts.component_diagnostics)
        all_divergence.append(minute_artifacts.divergence)

    if not all_finalist_scores:
        raise Module6ValidationError("no finalist portfolios were produced by Module 6")
    portfolio_candidates = pd.concat(all_candidates, axis=0, ignore_index=True).drop_duplicates("portfolio_pk", keep="first")
    portfolio_weights = pd.concat(all_weights, axis=0, ignore_index=True)
    session_paths = pd.concat(all_session_paths, axis=0, ignore_index=True)
    session_scores = pd.concat(all_session_scores, axis=0, ignore_index=True)
    finalist_scores = pd.concat(all_finalist_scores, axis=0, ignore_index=True).drop_duplicates("portfolio_pk", keep="first")
    comparable_scores = build_cross_universe_comparable_scores(
        finalist_scores=pd.concat(all_comparable_scores, axis=0, ignore_index=True).drop_duplicates("portfolio_pk", keep="first"),
        config=cfg,
    )
    minute_paths = pd.concat(all_minute_paths, axis=0, ignore_index=True)
    component_diag = pd.concat(all_component_diag, axis=0, ignore_index=True)
    divergence = pd.concat(all_divergence, axis=0, ignore_index=True).drop_duplicates("portfolio_pk", keep="first")
    selected_input = comparable_scores.copy()
    global_frontier, risk_return_frontier, operational_frontier, selected_frontier = select_diverse_finalists(
        scores=selected_input.sort_values(["comparable_truth_score", "portfolio_pk"], ascending=[False, True], kind="mergesort").reset_index(drop=True),
        portfolio_weights=portfolio_weights,
        strategy_frame=reduction.admitted_instances,
        config=cfg,
    )
    selected_pks = tuple(selected_frontier["portfolio_pk"].head(int(cfg.scoring.final_primary_count)).astype(str).tolist())
    alternate_pks = tuple(
        selected_input.loc[~selected_input["portfolio_pk"].isin(selected_pks), "portfolio_pk"]
        .head(int(cfg.scoring.final_alternate_count))
        .astype(str)
        .tolist()
    )
    report = PortfolioSelectionReport(
        run_id=str(run.run_manifest["run_id"]),
        output_dir=out_dir,
        selected_portfolio_pks=selected_pks,
        alternate_portfolio_pks=alternate_pks,
        summary={
            "config": asdict(cfg),
            "n_portfolio_candidates": int(portfolio_candidates.shape[0]),
            "n_finalists": int(finalist_scores.shape[0]),
            "n_selected": int(len(selected_pks)),
            "n_alternates": int(len(alternate_pks)),
        },
    )
    write_module6_outputs(
        output_dir=out_dir,
        candidates=portfolio_candidates,
        portfolio_weights=portfolio_weights,
        session_paths=session_paths,
        minute_paths=minute_paths,
        minute_component_diagnostics=component_diag,
        session_scores=session_scores,
        finalist_scores=finalist_scores,
        comparable_scores=comparable_scores,
        divergence=divergence,
        global_frontier=global_frontier,
        risk_return_frontier=risk_return_frontier,
        operational_frontier=operational_frontier,
        selected_frontier=selected_frontier,
        selection_report=report,
    )
    return report
