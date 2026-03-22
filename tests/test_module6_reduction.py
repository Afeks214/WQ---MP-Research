from __future__ import annotations

import json
from dataclasses import replace

import pytest

from weightiz.module6.io import load_module5_run
from weightiz.module6.ledger import materialize_canonical_ledgers
from weightiz.module6.matrices import build_matrix_store
from weightiz.module6.reduction import reduce_universe
from weightiz.module6.runtime import open_matrix_store
from weightiz.module6.utils import Module6ValidationError
from tests.module6_testkit import build_synthetic_module5_run, make_test_config


def test_reduction_collapses_duplicates_and_keeps_hedge(tmp_path):
    run_dir = build_synthetic_module5_run(tmp_path)
    cfg = make_test_config()
    loaded = load_module5_run(run_dir, cfg)
    ledgers = materialize_canonical_ledgers(loaded, run_dir / "ledgers", cfg)
    store = build_matrix_store(ledgers=ledgers, run=loaded, output_dir=run_dir / "matrix_out", config=cfg)
    matrices = open_matrix_store(store)
    matrices["column_index"] = store.column_index
    reduction = reduce_universe(ledgers=ledgers, matrices=matrices, run=loaded, output_dir=run_dir / "reduce_out", config=cfg)
    membership = reduction.cluster_membership
    dup = membership.loc[membership["candidate_id"].isin(["cand_000", "cand_001"])]
    assert dup["cluster_id"].nunique() == 1
    retained = membership.loc[membership["retained_in_reduced_universe"].astype(bool), ["candidate_id", "strategy_instance_pk"]]
    assert "cand_002" in set(retained["candidate_id"].astype(str))
    assert len(reduction.reduced_universes) >= 2
    assert len(reduction.reduced_universes[0].strategy_instance_pks) <= cfg.reduction.reduced_universe_cap
    assert len(reduction.reduced_universes[1].strategy_instance_pks) <= cfg.reduction.mv_universe_cap


def test_pre_reduction_fail_closed_writes_intake_gate_diagnostics(tmp_path):
    run_dir = build_synthetic_module5_run(tmp_path)
    base_cfg = make_test_config()
    cfg = replace(
        base_cfg,
        intake=replace(
            base_cfg.intake,
            min_availability_ratio=0.0,
            min_observed_sessions=10_000,
        ),
    )
    loaded = load_module5_run(run_dir, cfg)
    ledgers = materialize_canonical_ledgers(loaded, run_dir / "ledgers_fail", cfg)
    store = build_matrix_store(ledgers=ledgers, run=loaded, output_dir=run_dir / "matrix_fail", config=cfg)
    matrices = open_matrix_store(store)
    matrices["column_index"] = store.column_index
    reduce_dir = run_dir / "reduce_fail"
    with pytest.raises(Module6ValidationError, match="no admitted strategies survived pre-reduction intake gates"):
        reduce_universe(
            ledgers=ledgers,
            matrices=matrices,
            run=loaded,
            output_dir=reduce_dir,
            config=cfg,
        )
    diagnostics_path = reduce_dir / "diagnostics" / "module6_intake_pre_reduction_intake.json"
    assert diagnostics_path.exists()
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["stage"] == "pre_reduction_intake"
    assert diagnostics["module6_policy_class"] == "standard"
    assert diagnostics["availability_ratio_gate_mode"] == "warn"
    assert diagnostics["availability_ratio_warning"] is False
    assert int(diagnostics["intake_candidate_count"]) > 0
    assert int(diagnostics["admitted_candidate_count"]) == 0
    assert diagnostics["first_zero_gate"] == "observed_session_count_gate"
    thresholds = diagnostics["resolved_thresholds"]
    assert float(thresholds["min_availability_ratio"]) == 0.0
    assert int(thresholds["min_observed_sessions"]) == 10_000
    assert thresholds["availability_ratio_gate_mode"] == "warn"
    assert int(thresholds["availability_ratio_warning_count"]) == 0
    gate_names = [str(row["gate"]) for row in diagnostics["gates"]]
    assert gate_names == [
        "portfolio_admit_flag_gate",
        "failed_status_gate",
        "reject_gate",
        "availability_ratio_gate",
        "observed_session_count_gate",
        "turnover_sanity_gate",
    ]
    assert all(int(row["survivor_count"]) >= 0 for row in diagnostics["gates"])
    assert not (reduce_dir / "reduced_universes" / "reduced_universe_000.parquet").exists()


def test_availability_ratio_gate_hard_mode_blocks_candidates(tmp_path):
    run_dir = build_synthetic_module5_run(tmp_path)
    base_cfg = make_test_config()
    cfg = replace(
        base_cfg,
        intake=replace(
            base_cfg.intake,
            availability_ratio_gate_mode="hard",
            min_observed_sessions=10_000,
        ),
    )
    loaded = load_module5_run(run_dir, cfg)
    ledgers = materialize_canonical_ledgers(loaded, run_dir / "ledgers_avail_hard", cfg)
    ledgers["strategy_master"] = ledgers["strategy_master"].assign(availability_ratio=0.0)
    store = build_matrix_store(ledgers=ledgers, run=loaded, output_dir=run_dir / "matrix_avail_hard", config=cfg)
    matrices = open_matrix_store(store)
    matrices["column_index"] = store.column_index
    reduce_dir = run_dir / "reduce_avail_hard"
    with pytest.raises(Module6ValidationError, match="no admitted strategies survived pre-reduction intake gates"):
        reduce_universe(
            ledgers=ledgers,
            matrices=matrices,
            run=loaded,
            output_dir=reduce_dir,
            config=cfg,
        )
    diagnostics = json.loads((reduce_dir / "diagnostics" / "module6_intake_pre_reduction_intake.json").read_text(encoding="utf-8"))
    assert diagnostics["availability_ratio_gate_mode"] == "hard"
    assert diagnostics["availability_ratio_gate_effective_mode"] == "hard"
    assert diagnostics["availability_ratio_warning"] is False
    assert diagnostics["first_zero_gate"] == "availability_ratio_gate"
    thresholds = diagnostics["resolved_thresholds"]
    assert thresholds["availability_ratio_gate_mode"] == "hard"
    assert thresholds["availability_ratio_gate_effective_mode"] == "hard"
    assert int(thresholds["availability_ratio_warning_count"]) == 0
    availability_gate = next(row for row in diagnostics["gates"] if row["gate"] == "availability_ratio_gate")
    assert availability_gate["gate_active"] is True
    assert int(availability_gate["dropped_count"]) > 0


def test_availability_ratio_gate_warn_mode_emits_warning_without_hard_block(tmp_path):
    run_dir = build_synthetic_module5_run(tmp_path)
    base_cfg = make_test_config()
    cfg = replace(
        base_cfg,
        intake=replace(
            base_cfg.intake,
            availability_ratio_gate_mode="warn",
            min_observed_sessions=10_000,
        ),
    )
    loaded = load_module5_run(run_dir, cfg)
    ledgers = materialize_canonical_ledgers(loaded, run_dir / "ledgers_avail_warn", cfg)
    ledgers["strategy_master"] = ledgers["strategy_master"].assign(availability_ratio=0.0)
    store = build_matrix_store(ledgers=ledgers, run=loaded, output_dir=run_dir / "matrix_avail_warn", config=cfg)
    matrices = open_matrix_store(store)
    matrices["column_index"] = store.column_index
    reduce_dir = run_dir / "reduce_avail_warn"
    with pytest.raises(Module6ValidationError, match="no admitted strategies survived pre-reduction intake gates"):
        reduce_universe(
            ledgers=ledgers,
            matrices=matrices,
            run=loaded,
            output_dir=reduce_dir,
            config=cfg,
        )
    diagnostics = json.loads((reduce_dir / "diagnostics" / "module6_intake_pre_reduction_intake.json").read_text(encoding="utf-8"))
    assert diagnostics["availability_ratio_gate_mode"] == "warn"
    assert diagnostics["availability_ratio_warning"] is True
    assert diagnostics["first_zero_gate"] == "observed_session_count_gate"
    thresholds = diagnostics["resolved_thresholds"]
    assert thresholds["availability_ratio_gate_mode"] == "warn"
    assert int(thresholds["availability_ratio_warning_count"]) > 0
    availability_gate = next(row for row in diagnostics["gates"] if row["gate"] == "availability_ratio_gate")
    assert availability_gate["gate_active"] is False
    assert int(availability_gate["dropped_count"]) == 0


def test_availability_ratio_gate_off_mode_suppresses_warning_and_hard_block(tmp_path):
    run_dir = build_synthetic_module5_run(tmp_path)
    base_cfg = make_test_config()
    cfg = replace(
        base_cfg,
        intake=replace(
            base_cfg.intake,
            availability_ratio_gate_mode="off",
            min_observed_sessions=10_000,
        ),
    )
    loaded = load_module5_run(run_dir, cfg)
    ledgers = materialize_canonical_ledgers(loaded, run_dir / "ledgers_avail_off", cfg)
    ledgers["strategy_master"] = ledgers["strategy_master"].assign(availability_ratio=0.0)
    store = build_matrix_store(ledgers=ledgers, run=loaded, output_dir=run_dir / "matrix_avail_off", config=cfg)
    matrices = open_matrix_store(store)
    matrices["column_index"] = store.column_index
    reduce_dir = run_dir / "reduce_avail_off"
    with pytest.raises(Module6ValidationError, match="no admitted strategies survived pre-reduction intake gates"):
        reduce_universe(
            ledgers=ledgers,
            matrices=matrices,
            run=loaded,
            output_dir=reduce_dir,
            config=cfg,
        )
    diagnostics = json.loads((reduce_dir / "diagnostics" / "module6_intake_pre_reduction_intake.json").read_text(encoding="utf-8"))
    assert diagnostics["availability_ratio_gate_mode"] == "off"
    assert diagnostics["availability_ratio_warning"] is False
    assert diagnostics["first_zero_gate"] == "observed_session_count_gate"
    thresholds = diagnostics["resolved_thresholds"]
    assert thresholds["availability_ratio_gate_mode"] == "off"
    assert int(thresholds["availability_ratio_warning_count"]) == 0
    availability_gate = next(row for row in diagnostics["gates"] if row["gate"] == "availability_ratio_gate")
    assert availability_gate["gate_active"] is False
    assert int(availability_gate["dropped_count"]) == 0


def test_matrix_entry_fail_closed_writes_intake_gate_diagnostics(tmp_path):
    run_dir = build_synthetic_module5_run(tmp_path)
    cfg = make_test_config()
    loaded = load_module5_run(run_dir, cfg)
    ledgers = materialize_canonical_ledgers(loaded, run_dir / "ledgers_matrix_fail", cfg)
    ledgers["strategy_instance_master"] = ledgers["strategy_instance_master"].assign(portfolio_admit_flag=False)
    with pytest.raises(Module6ValidationError, match="no admitted canonical portfolio instances available for matrix build"):
        build_matrix_store(
            ledgers=ledgers,
            run=loaded,
            output_dir=run_dir / "matrix_fail_no_admit",
            config=cfg,
        )
    diagnostics_path = run_dir / "matrix_fail_no_admit" / "diagnostics" / "module6_intake_matrix_entry.json"
    assert diagnostics_path.exists()
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["stage"] == "matrix_entry"
    assert diagnostics["module6_policy_class"] == "standard"
    assert diagnostics["first_zero_gate"] == "portfolio_admit_flag_gate"
    gate_names = [str(row["gate"]) for row in diagnostics["gates"]]
    assert gate_names == [
        "canonical_portfolio_role_gate",
        "portfolio_admit_flag_gate",
    ]
