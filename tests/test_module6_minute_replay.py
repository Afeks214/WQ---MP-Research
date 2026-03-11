from __future__ import annotations

import pandas as pd
import pytest

from weightiz.module6.io import load_module5_run
from weightiz.module6.ledger import materialize_canonical_ledgers
from weightiz.module6.matrices import build_matrix_store
from weightiz.module6.reduction import reduce_universe
from weightiz.module6.runtime import open_matrix_store
from weightiz.module6.simulator.session_path import simulate_session_batch
from weightiz.module6.simulator.minute_refine import replay_finalists_minute
from weightiz.module6.dependence import build_covariance_bundle
from weightiz.module6.generators import generate_all_portfolios
from weightiz.module6.scoring import score_session_paths
from weightiz.module6.utils import Module6ValidationError
from tests.module6_testkit import build_synthetic_module5_run, make_test_config


def test_minute_replay_emits_divergence_rows(tmp_path):
    run_dir = build_synthetic_module5_run(tmp_path)
    cfg = make_test_config()
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
    session_art = simulate_session_batch(
        portfolio_candidates=candidates.head(3),
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
    minute = replay_finalists_minute(
        finalist_candidates=candidates.head(3),
        strategy_frame=strategy_frame,
        session_paths=session_art.session_paths,
        session_summary=session_scores,
        weight_history=session_art.weight_history,
        run=loaded,
        config=cfg,
    )
    assert minute.divergence.shape[0] == 3
    assert {"portfolio_pk", "session_score", "minute_score", "rank_delta"}.issubset(minute.divergence.columns)


def test_minute_replay_requires_micro_truth_input(tmp_path):
    run_dir = build_synthetic_module5_run(tmp_path)
    cfg = make_test_config()
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
    session_art = simulate_session_batch(
        portfolio_candidates=candidates.head(2),
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
    loaded.micro_diagnostics = None
    with pytest.raises(Module6ValidationError, match="micro_diagnostics truth input"):
        replay_finalists_minute(
            finalist_candidates=candidates.head(2),
            strategy_frame=strategy_frame,
            session_paths=session_art.session_paths,
            session_summary=session_scores,
            weight_history=session_art.weight_history,
            run=loaded,
            config=cfg,
        )
