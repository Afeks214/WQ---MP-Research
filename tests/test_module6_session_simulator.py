from __future__ import annotations

import pandas as pd

from module6.io import load_module5_run
from module6.ledger import materialize_canonical_ledgers
from module6.matrices import build_matrix_store
from module6.reduction import reduce_universe
from module6.runtime import open_matrix_store
from module6.simulator.session_path import _policy_rebalance_due, simulate_session_batch
from module6.dependence import build_covariance_bundle
from module6.generators import generate_all_portfolios
from tests.module6_testkit import build_synthetic_module5_run, make_test_config


def test_session_simulator_emits_paths_and_costs(tmp_path):
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
    art = simulate_session_batch(
        portfolio_candidates=candidates.head(4),
        portfolio_weights=weights,
        strategy_frame=strategy_frame,
        matrices=matrices,
        calendar=pd.read_parquet(store.calendar_index_path),
        config=cfg,
        return_weight_history=True,
    )
    assert art.session_paths.shape[0] > 0
    assert "cost_frac" in art.session_paths.columns
    assert art.weight_history.shape[0] > 0


def test_weekly_rebalance_uses_calendar_flag_not_session_index():
    cfg = make_test_config()
    drift = pd.Series([0.6, 0.4], dtype="float64").to_numpy()
    target = pd.Series([0.5, 0.5], dtype="float64").to_numpy()
    assert not _policy_rebalance_due("weekly_monday_close", session_idx=1, session_meta={"is_monday_close": 0}, drift_weights=drift, target_weights=target, band=cfg.simulator.rebalance_band_l1)
    assert _policy_rebalance_due("weekly_monday_close", session_idx=1, session_meta={"is_monday_close": 1}, drift_weights=drift, target_weights=target, band=cfg.simulator.rebalance_band_l1)
