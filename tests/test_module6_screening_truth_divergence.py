from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from module6.io import load_module5_run
from module6.ledger import materialize_canonical_ledgers
from module6.matrices import build_matrix_store
from module6.reduction import reduce_universe
from module6.runtime import open_matrix_store
from module6.simulator.minute_refine import replay_finalists_minute
from module6.simulator.session_path import simulate_session_batch
from module6.dependence import build_covariance_bundle
from module6.generators import generate_all_portfolios
from module6.scoring import score_session_paths
from module6.utils import Module6ValidationError
from tests.module6_testkit import build_synthetic_module5_run, make_test_config


def test_screening_truth_rank_instability_gate_blocks_export(tmp_path):
    base_cfg = make_test_config()
    cfg = replace(base_cfg, scoring=replace(base_cfg.scoring, min_rank_stability=0.99))
    run_dir = build_synthetic_module5_run(tmp_path)
    loaded = load_module5_run(run_dir, cfg)
    ledgers = materialize_canonical_ledgers(loaded, run_dir / "ledgers", cfg)
    store = build_matrix_store(ledgers=ledgers, run=loaded, output_dir=run_dir / "matrix_out", config=cfg)
    matrices = open_matrix_store(store)
    matrices["column_index"] = store.column_index
    reduction = reduce_universe(ledgers=ledgers, matrices=matrices, run=loaded, output_dir=run_dir / "reduce_out", config=cfg)
    strategy_frame = reduction.admitted_instances.loc[reduction.admitted_instances["strategy_instance_pk"].isin(reduction.reduced_universes[0].strategy_instance_pks)].copy()
    cols = strategy_frame["column_idx"].to_numpy(dtype="int64")
    bundle = build_covariance_bundle(matrices["R_exec"], matrices["A"], matrices["G"], cols, cfg.dependence)
    candidates, weights = generate_all_portfolios(
        reduced_universe=reduction.reduced_universes[0],
        strategy_frame=strategy_frame,
        covariance_bundle=bundle,
        returns_exec=matrices["R_exec"],
        column_indices=cols,
        config=cfg,
        calendar_version=str(strategy_frame["calendar_version"].iloc[0]),
    )
    finalists = candidates.head(3).copy()
    session_art = simulate_session_batch(
        portfolio_candidates=finalists,
        portfolio_weights=weights,
        strategy_frame=strategy_frame,
        matrices=matrices,
        calendar=pd.read_parquet(store.calendar_index_path),
        config=cfg,
        return_weight_history=True,
    )
    session_scores = score_session_paths(
        session_paths=session_art.session_paths,
        session_summary=session_art.portfolio_summary,
        portfolio_weights=weights,
        strategy_frame=strategy_frame,
        config=cfg,
    ).assign(calendar_version=str(strategy_frame["calendar_version"].iloc[0]), support_policy_version=cfg.simulator.support_policy_version)
    session_scores = session_scores.loc[session_scores["portfolio_pk"].isin(finalists["portfolio_pk"])].copy()
    session_scores["first_pass_score"] = -session_scores["annualized_return"].to_numpy()
    with pytest.raises(Module6ValidationError):
        replay_finalists_minute(
            finalist_candidates=finalists,
            strategy_frame=strategy_frame,
            session_paths=session_art.session_paths,
            session_summary=session_scores,
            weight_history=session_art.weight_history,
            run=loaded,
            config=cfg,
        )
