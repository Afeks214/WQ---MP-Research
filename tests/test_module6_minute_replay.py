from __future__ import annotations

from dataclasses import replace

import numpy as np
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


def test_minute_replay_session_score_uses_truth_scale(tmp_path):
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
    session_scores["first_pass_score"] = 1.0e6
    minute = replay_finalists_minute(
        finalist_candidates=candidates.head(3),
        strategy_frame=strategy_frame,
        session_paths=session_art.session_paths,
        session_summary=session_scores,
        weight_history=session_art.weight_history,
        run=loaded,
        config=cfg,
    )
    expected = minute.minute_summary["session_annualized_return"] - minute.minute_summary["session_max_drawdown"].clip(lower=0.0)
    assert np.allclose(
        np.asarray(minute.minute_summary["session_score"], dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
    )


def test_minute_replay_micro_notional_nan_guard(tmp_path):
    run_dir = build_synthetic_module5_run(tmp_path)
    cfg = make_test_config()
    loaded = load_module5_run(run_dir, cfg)
    loaded.micro_diagnostics = loaded.micro_diagnostics.copy()
    mask = np.asarray(loaded.micro_diagnostics["filled_qty"], dtype=np.float64) == 0.0
    loaded.micro_diagnostics.loc[mask, "exec_price"] = np.nan
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
    minute = replay_finalists_minute(
        finalist_candidates=candidates.head(2),
        strategy_frame=strategy_frame,
        session_paths=session_art.session_paths,
        session_summary=session_scores,
        weight_history=session_art.weight_history,
        run=loaded,
        config=cfg,
    )
    arr = np.asarray(minute.component_diagnostics["micro_trade_notional"], dtype=np.float64)
    assert np.isfinite(arr).all()


def test_minute_replay_uses_explicit_starting_capital_and_post_cost_floor(tmp_path):
    run_dir = build_synthetic_module5_run(tmp_path)
    base_cfg = make_test_config()
    session_cfg = replace(
        base_cfg,
        simulator=replace(
            base_cfg.simulator,
            starting_capital=2000.0,
            account_disable_equity=1000.0,
            fixed_fee=0.0,
        ),
    )
    replay_cfg = replace(
        session_cfg,
        simulator=replace(
            session_cfg.simulator,
            fixed_fee=1200.0,
        ),
    )
    loaded = load_module5_run(run_dir, session_cfg)
    ledgers = materialize_canonical_ledgers(loaded, run_dir / "ledgers_floor", session_cfg)
    store = build_matrix_store(ledgers=ledgers, run=loaded, output_dir=run_dir / "matrix_out_floor", config=session_cfg)
    matrices = open_matrix_store(store)
    matrices["column_index"] = store.column_index
    reduction = reduce_universe(ledgers=ledgers, matrices=matrices, run=loaded, output_dir=run_dir / "reduce_out_floor", config=session_cfg)
    strategy_frame = reduction.admitted_instances.loc[reduction.admitted_instances["strategy_instance_pk"].isin(reduction.reduced_universes[0].strategy_instance_pks)].copy()
    cols = strategy_frame["column_idx"].to_numpy(dtype="int64")
    bundle = build_covariance_bundle(matrices["R_exec"], matrices["A"], matrices["G"], cols, session_cfg.dependence)
    candidates, weights = generate_all_portfolios(
        reduced_universe=reduction.reduced_universes[0],
        strategy_frame=strategy_frame,
        covariance_bundle=bundle,
        returns_exec=matrices["R_exec"],
        column_indices=cols,
        config=session_cfg,
        calendar_version=str(strategy_frame["calendar_version"].iloc[0]),
    )
    finalists = candidates.head(1).copy()
    session_art = simulate_session_batch(
        portfolio_candidates=finalists,
        portfolio_weights=weights,
        strategy_frame=strategy_frame,
        matrices=matrices,
        calendar=pd.read_parquet(store.calendar_index_path),
        config=session_cfg,
        return_weight_history=True,
    )
    session_scores = score_session_paths(
        session_paths=session_art.session_paths,
        session_summary=session_art.portfolio_summary,
        portfolio_weights=weights,
        strategy_frame=strategy_frame,
        config=session_cfg,
    ).assign(
        calendar_version=str(strategy_frame["calendar_version"].iloc[0]),
        support_policy_version=session_cfg.simulator.support_policy_version,
    )
    minute = replay_finalists_minute(
        finalist_candidates=finalists,
        strategy_frame=strategy_frame,
        session_paths=session_art.session_paths,
        session_summary=session_scores,
        weight_history=session_art.weight_history,
        run=loaded,
        config=replay_cfg,
    )
    assert float(minute.minute_summary["starting_equity"].iloc[0]) == 2000.0
    assert float(minute.minute_summary["session_starting_equity"].iloc[0]) == 2000.0
    assert int(minute.minute_summary["minute_disable_flag"].iloc[0]) == 1
    assert int(minute.minute_summary["minute_breach_count"].iloc[0]) > 0
    assert float(minute.minute_summary["minute_final_equity"].iloc[0]) < float(replay_cfg.simulator.account_disable_equity)
