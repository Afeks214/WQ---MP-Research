from __future__ import annotations

import numpy as np
import pandas as pd

from weightiz.module6.io import load_module5_run
from weightiz.module6.ledger import materialize_canonical_ledgers
from weightiz.module6.matrices import build_matrix_store
from weightiz.module6.reduction import reduce_universe
from weightiz.module6.runtime import open_matrix_store
from weightiz.module6.simulator.session_path import simulate_session_batch
from weightiz.module6.dependence import build_covariance_bundle
from weightiz.module6.generators import generate_all_portfolios
from weightiz.module6.scoring import build_cross_universe_comparable_scores, score_session_paths
from tests.module6_testkit import build_synthetic_module5_run, make_test_config


def test_score_session_paths_adds_first_pass_score(tmp_path):
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
        portfolio_candidates=candidates.head(4),
        portfolio_weights=weights,
        strategy_frame=strategy_frame,
        matrices=matrices,
        calendar=pd.read_parquet(store.calendar_index_path),
        config=cfg,
        return_weight_history=False,
    )
    scored = score_session_paths(
        session_paths=session_art.session_paths,
        session_summary=session_art.portfolio_summary,
        portfolio_weights=weights,
        strategy_frame=strategy_frame,
        config=cfg,
    )
    assert "first_pass_score" in scored.columns


def test_score_session_paths_uses_finite_reject_floor(tmp_path):
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
        return_weight_history=False,
    )
    forced_reject_summary = session_art.portfolio_summary.copy()
    forced_reject_summary["final_equity"] = 0.0
    scored = score_session_paths(
        session_paths=session_art.session_paths,
        session_summary=forced_reject_summary,
        portfolio_weights=weights,
        strategy_frame=strategy_frame,
        config=cfg,
    )
    assert np.isfinite(np.asarray(scored["first_pass_score"], dtype=np.float64)).all()
    assert (scored["first_pass_score"] == -1.0).all()


def test_score_session_paths_penalizes_low_support_without_hard_reject(tmp_path):
    run_dir = build_synthetic_module5_run(tmp_path)
    cfg = make_test_config()
    loaded = load_module5_run(run_dir, cfg)
    ledgers = materialize_canonical_ledgers(loaded, run_dir / "ledgers_penalty", cfg)
    store = build_matrix_store(ledgers=ledgers, run=loaded, output_dir=run_dir / "matrix_penalty", config=cfg)
    matrices = open_matrix_store(store)
    matrices["column_index"] = store.column_index
    reduction = reduce_universe(ledgers=ledgers, matrices=matrices, run=loaded, output_dir=run_dir / "reduce_penalty", config=cfg)
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
        portfolio_candidates=candidates.head(1),
        portfolio_weights=weights,
        strategy_frame=strategy_frame,
        matrices=matrices,
        calendar=pd.read_parquet(store.calendar_index_path),
        config=cfg,
        return_weight_history=False,
    )
    penalized_summary = session_art.portfolio_summary.copy()
    penalized_summary["support_coverage"] = 0.10
    scored = score_session_paths(
        session_paths=session_art.session_paths,
        session_summary=penalized_summary,
        portfolio_weights=weights,
        strategy_frame=strategy_frame,
        config=cfg,
    )
    assert bool(scored["hard_reject"].iloc[0]) is False
    assert float(scored["support_penalty"].iloc[0]) > 0.0
    assert np.isfinite(float(scored["first_pass_score"].iloc[0]))


def test_score_session_paths_hard_rejects_zero_gross_and_disable_flag(tmp_path):
    run_dir = build_synthetic_module5_run(tmp_path)
    cfg = make_test_config()
    loaded = load_module5_run(run_dir, cfg)
    ledgers = materialize_canonical_ledgers(loaded, run_dir / "ledgers_dead", cfg)
    store = build_matrix_store(ledgers=ledgers, run=loaded, output_dir=run_dir / "matrix_dead", config=cfg)
    matrices = open_matrix_store(store)
    matrices["column_index"] = store.column_index
    reduction = reduce_universe(ledgers=ledgers, matrices=matrices, run=loaded, output_dir=run_dir / "reduce_dead", config=cfg)
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
    art = simulate_session_batch(
        portfolio_candidates=candidates.head(1),
        portfolio_weights=weights,
        strategy_frame=strategy_frame,
        matrices=matrices,
        calendar=pd.read_parquet(store.calendar_index_path),
        config=cfg,
        return_weight_history=False,
    )
    dead_summary = art.portfolio_summary.copy()
    dead_summary["disable_flag"] = 1
    dead_paths = art.session_paths.copy()
    dead_paths["gross_exposure_mult"] = 0.0
    scored = score_session_paths(
        session_paths=dead_paths,
        session_summary=dead_summary,
        portfolio_weights=weights,
        strategy_frame=strategy_frame,
        config=cfg,
    )
    assert bool(scored["hard_reject"].iloc[0]) is True
    assert str(scored["hard_reject_reason"].iloc[0]) == "SESSION_DISABLE_FLAG"
    assert float(scored["first_pass_score"].iloc[0]) == -1.0


def test_cross_universe_scores_reject_dead_portfolios():
    cfg = make_test_config()
    finalists = pd.DataFrame(
        {
            "portfolio_pk": ["p0", "p1"],
            "calendar_version": ["cv1", "cv1"],
            "support_policy_version": ["sp1", "sp1"],
            "comparison_support_recomputed": [True, True],
            "minute_annualized_return": [0.2, 0.1],
            "minute_max_drawdown": [0.05, 0.04],
            "minute_turnover": [0.1, 0.2],
            "support_coverage": [0.9, 0.9],
            "availability_burden": [0.0, 0.0],
            "session_disable_flag": [0, 1],
            "session_breach_count": [0, 0],
            "session_gross_exposure_peak": [1.2, 1.1],
            "minute_disable_flag": [0, 0],
            "minute_breach_count": [0, 0],
            "minute_gross_exposure_peak": [1.2, 1.1],
        }
    )
    comparable = build_cross_universe_comparable_scores(
        finalist_scores=finalists,
        config=cfg,
        comparison_support=pd.DataFrame({"session_id": [1, 2, 3]}),
    )
    rejected = comparable.loc[comparable["portfolio_pk"] == "p1"].iloc[0]
    assert bool(rejected["cross_universe_reject"]) is True
    assert str(rejected["cross_universe_reject_reason"]) == "SESSION_DISABLE_FLAG"
    assert float(rejected["comparable_truth_score"]) == -1.0
