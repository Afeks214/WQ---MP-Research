from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from weightiz.module6.config import Module6Config
from weightiz.module6.dependence import build_covariance_bundle
from weightiz.module6.execution_overlap import finalist_exact_overlap
from weightiz.module6.export import write_module6_outputs
from weightiz.module6.frontier import select_diverse_finalists
from weightiz.module6.generators import generate_all_portfolios
from weightiz.module6.io import load_module5_run
from weightiz.module6.ledger import materialize_canonical_ledgers
from weightiz.module6.matrices import build_matrix_store
from weightiz.module6.reduction import reduce_universe
from weightiz.module6.runtime import open_matrix_store
from weightiz.module6.scoring import build_cross_universe_comparable_scores, score_finalists, score_session_paths
from weightiz.module6.simulator.minute_refine import replay_finalists_minute
from weightiz.module6.simulator.session_path import simulate_session_batch
from weightiz.module6.types import PortfolioSelectionReport
from weightiz.module6.utils import Module6ValidationError, ensure_directory


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
    comparison_support_calendar = pd.read_parquet(matrix_store.calendar_index_path)
    reduction = reduce_universe(
        ledgers=ledgers,
        matrices=matrices,
        run=run,
        output_dir=out_dir,
        config=cfg,
    )
    overlap_proxy_index = reduction.admitted_instances[
        ["strategy_instance_pk", "candidate_id", "split_id", "scenario_id", "column_idx", "overlap_proxy_idx"]
    ].drop_duplicates().sort_values(["strategy_instance_pk"], kind="mergesort").reset_index(drop=True)
    all_candidates: list[pd.DataFrame] = []
    all_weights: list[pd.DataFrame] = []
    all_session_paths: list[pd.DataFrame] = []
    all_session_scores: list[pd.DataFrame] = []
    dependence_artifacts: dict[str, Any] = {}
    shortlist_portfolio_pks: list[str] = []
    calendar_version = str(run.run_manifest["module6_bridge"]["calendar_version"])

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
        dependence_artifacts[str(reduced_universe.reduced_universe_id)] = covariance_bundle
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
            calendar=comparison_support_calendar,
            config=cfg,
            return_weight_history=False,
        )
        session_scores = score_session_paths(
            session_paths=session_artifacts.session_paths,
            session_summary=session_artifacts.portfolio_summary,
            portfolio_weights=weights_df,
            strategy_frame=strategy_frame,
            config=cfg,
            execution_overlap_proxy=reduction.overlap_proxy,
        )
        session_scores["calendar_version"] = calendar_version
        session_scores["support_policy_version"] = cfg.simulator.support_policy_version
        session_scores["comparison_support_recomputed"] = False
        session_stage_shortlist = session_scores.head(int(cfg.scoring.shortlist_session_keep)).copy()
        shortlist = session_stage_shortlist.head(int(cfg.scoring.shortlist_minute_keep)).copy()
        shortlist_portfolio_pks.extend(shortlist["portfolio_pk"].astype(str).tolist())
        all_candidates.append(candidates_df)
        all_weights.append(weights_df)
        all_session_paths.append(session_artifacts.session_paths)
        all_session_scores.append(session_scores)

    if not all_candidates:
        raise Module6ValidationError("no portfolio candidates were produced by Module 6")
    portfolio_candidates = pd.concat(all_candidates, axis=0, ignore_index=True).drop_duplicates("portfolio_pk", keep="first")
    portfolio_weights = pd.concat(all_weights, axis=0, ignore_index=True)
    session_paths = pd.concat(all_session_paths, axis=0, ignore_index=True)
    session_scores = pd.concat(all_session_scores, axis=0, ignore_index=True)
    shortlist_unique = sorted(set(shortlist_portfolio_pks))
    if not shortlist_unique:
        raise Module6ValidationError("no shortlisted portfolios available for comparison-support recomputation")
    finalist_candidates = portfolio_candidates.loc[portfolio_candidates["portfolio_pk"].isin(shortlist_unique)].copy()
    finalist_weights = portfolio_weights.loc[portfolio_weights["portfolio_pk"].isin(shortlist_unique)].copy()
    finalist_instances = reduction.admitted_instances.loc[
        reduction.admitted_instances["strategy_instance_pk"].isin(finalist_weights["strategy_instance_pk"].astype(str))
    ][["strategy_instance_pk", "candidate_id", "split_id", "scenario_id"]].drop_duplicates()
    exact_overlap = finalist_exact_overlap(run.trade_log, finalist_instances)
    comparison_session_artifacts = simulate_session_batch(
        portfolio_candidates=finalist_candidates,
        portfolio_weights=finalist_weights,
        strategy_frame=reduction.admitted_instances,
        matrices=matrices,
        calendar=comparison_support_calendar,
        config=cfg,
        return_weight_history=True,
    )
    comparison_session_scores = score_session_paths(
        session_paths=comparison_session_artifacts.session_paths,
        session_summary=comparison_session_artifacts.portfolio_summary,
        portfolio_weights=finalist_weights,
        strategy_frame=reduction.admitted_instances,
        config=cfg,
        execution_overlap_proxy=reduction.overlap_proxy,
    )
    comparison_session_scores["calendar_version"] = calendar_version
    comparison_session_scores["support_policy_version"] = cfg.simulator.support_policy_version
    comparison_session_scores["comparison_support_recomputed"] = True
    minute_artifacts = replay_finalists_minute(
        finalist_candidates=finalist_candidates,
        strategy_frame=reduction.admitted_instances,
        session_paths=comparison_session_artifacts.session_paths,
        session_summary=comparison_session_scores,
        weight_history=comparison_session_artifacts.weight_history,
        run=run,
        config=cfg,
    )
    finalist_scores = score_finalists(
        session_scores=comparison_session_scores,
        minute_summary=minute_artifacts.minute_summary,
        divergence=minute_artifacts.divergence,
        portfolio_weights=finalist_weights,
        strategy_frame=reduction.admitted_instances,
        config=cfg,
        execution_overlap_proxy=reduction.overlap_proxy,
    )
    finalist_scores["comparison_support_recomputed"] = True
    comparable_scores = build_cross_universe_comparable_scores(
        finalist_scores=finalist_scores.drop_duplicates("portfolio_pk", keep="first"),
        config=cfg,
        comparison_support=comparison_support_calendar,
    )
    minute_paths = minute_artifacts.minute_paths
    component_diag = minute_artifacts.component_diagnostics
    divergence = minute_artifacts.divergence.drop_duplicates("portfolio_pk", keep="first")
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
            "comparison_support_session_count": int(comparison_support_calendar.shape[0]),
            "canonical_reference_policy": str(run.run_manifest["module6_bridge"]["canonical_reference_policy"]),
        },
    )
    write_module6_outputs(
        output_dir=out_dir,
        candidates=portfolio_candidates,
        portfolio_weights=portfolio_weights,
        session_paths=session_paths,
        comparison_support_session_paths=comparison_session_artifacts.session_paths,
        minute_paths=minute_paths,
        minute_component_diagnostics=component_diag,
        session_scores=session_scores,
        comparison_support_session_scores=comparison_session_scores,
        finalist_scores=finalist_scores,
        comparable_scores=comparable_scores,
        divergence=divergence,
        weight_history=comparison_session_artifacts.weight_history,
        comparison_support_calendar=comparison_support_calendar,
        dependence_artifacts=dependence_artifacts,
        overlap_proxy=reduction.overlap_proxy,
        overlap_proxy_index=overlap_proxy_index,
        exact_overlap=exact_overlap,
        global_frontier=global_frontier,
        risk_return_frontier=risk_return_frontier,
        operational_frontier=operational_frontier,
        selected_frontier=selected_frontier,
        selection_report=report,
    )
    return report
