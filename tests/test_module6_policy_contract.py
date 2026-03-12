from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from weightiz.cli.run_research import _load_config
from weightiz.module5.harness.robustness_support import (
    resolve_module6_admission_verdict,
    resolve_module6_policy_class_from_research_mode,
)
from weightiz.module6.config import (
    MODULE6_RUN_POLICY_REPRESENTATIVE_DISCOVERY,
    MODULE6_RUN_POLICY_STANDARD,
    resolve_intake_gate_thresholds,
)
from weightiz.module6.io import load_module5_run
from weightiz.module6.ledger import materialize_canonical_ledgers
from weightiz.module6.matrices import build_matrix_store
from weightiz.module6.reduction import reduce_universe
from weightiz.module6.runtime import open_matrix_store
from tests.module6_testkit import build_synthetic_module5_run, make_test_config


def _rewrite_policy_contract(
    run_dir: Path,
    *,
    policy_class: str,
    reject: bool,
    module6_admit: bool,
) -> None:
    for name in ("leaderboard.csv", "robustness_leaderboard.csv"):
        path = run_dir / name
        frame = pd.read_csv(path)
        frame["research_mode"] = "discovery" if policy_class == MODULE6_RUN_POLICY_REPRESENTATIVE_DISCOVERY else "standard"
        frame["discovery_included"] = policy_class == MODULE6_RUN_POLICY_REPRESENTATIVE_DISCOVERY
        frame["module6_policy_class"] = policy_class
        frame["module6_admit"] = bool(module6_admit)
        frame["module6_admission_basis"] = (
            "representative_discovery_mcs_included"
            if policy_class == MODULE6_RUN_POLICY_REPRESENTATIVE_DISCOVERY and module6_admit
            else "standard_reject_blocked"
        )
        frame["standard_reject"] = bool(reject)
        frame["standard_pass"] = not bool(reject)
        frame["reject"] = bool(reject)
        frame["pass"] = not bool(reject)
        frame.to_csv(path, index=False)


def _drop_trade_log_support_for_split(run_dir: Path, *, split_id: str) -> None:
    path = run_dir / "trade_log.parquet"
    frame = pd.read_parquet(path)
    frame = frame.loc[frame["split_id"].astype(str) != str(split_id)].copy()
    frame.to_parquet(path, index=False)


def _clear_trade_log(run_dir: Path) -> None:
    path = run_dir / "trade_log.parquet"
    frame = pd.read_parquet(path)
    frame.iloc[0:0].copy().to_parquet(path, index=False)


def test_representative_discovery_config_resolves_explicit_policy() -> None:
    cfg = _load_config(Path("configs/local_discovery_short_7core.yaml"))
    base_cfg = make_test_config()
    assert cfg.harness.research_mode == "discovery"
    assert int(cfg.module6["generator"]["random_sparse_quota"]) == 64
    assert int(cfg.module6["generator"]["cluster_balanced_quota"]) == 16
    assert int(cfg.module6["scoring"]["shortlist_session_keep"]) == 32
    assert int(cfg.module6["scoring"]["final_primary_count"]) == 2
    policy_class, min_availability_ratio, min_observed_sessions = resolve_intake_gate_thresholds(
        replace(
            base_cfg.intake,
            run_policy_class=str(cfg.module6["intake"]["run_policy_class"]),
            min_availability_ratio=float(cfg.module6["intake"]["min_availability_ratio"]),
            min_observed_sessions=int(cfg.module6["intake"]["min_observed_sessions"]),
        )
    )
    assert policy_class == MODULE6_RUN_POLICY_REPRESENTATIVE_DISCOVERY
    assert min_availability_ratio == 0.25
    assert min_observed_sessions == 2


def test_adaptive_discovery_config_resolves_explicit_policy() -> None:
    cfg = _load_config(Path("configs/local_adaptive_discovery_7core.yaml"))
    base_cfg = make_test_config()
    assert cfg.harness.research_mode == "discovery"
    assert int(cfg.module6["generator"]["random_sparse_quota"]) == 64
    assert int(cfg.module6["generator"]["cluster_balanced_quota"]) == 16
    assert int(cfg.module6["scoring"]["shortlist_session_keep"]) == 32
    assert int(cfg.module6["scoring"]["final_primary_count"]) == 2
    policy_class, min_availability_ratio, min_observed_sessions = resolve_intake_gate_thresholds(
        replace(
            base_cfg.intake,
            run_policy_class=str(cfg.module6["intake"]["run_policy_class"]),
            min_availability_ratio=float(cfg.module6["intake"]["min_availability_ratio"]),
            min_observed_sessions=int(cfg.module6["intake"]["min_observed_sessions"]),
        )
    )
    assert policy_class == MODULE6_RUN_POLICY_REPRESENTATIVE_DISCOVERY
    assert min_availability_ratio == 0.25
    assert min_observed_sessions == 2


def test_policy_contract_distinguishes_standard_reject_from_representative_discovery_admission() -> None:
    standard_admit, standard_basis = resolve_module6_admission_verdict(
        policy_class=MODULE6_RUN_POLICY_STANDARD,
        discovery_included=False,
        in_mcs=True,
        reject=True,
        robustness_score=0.37,
    )
    discovery_admit, discovery_basis = resolve_module6_admission_verdict(
        policy_class=MODULE6_RUN_POLICY_REPRESENTATIVE_DISCOVERY,
        discovery_included=True,
        in_mcs=True,
        reject=True,
        robustness_score=0.37,
    )
    assert resolve_module6_policy_class_from_research_mode("standard") == MODULE6_RUN_POLICY_STANDARD
    assert resolve_module6_policy_class_from_research_mode("discovery") == MODULE6_RUN_POLICY_REPRESENTATIVE_DISCOVERY
    assert standard_admit is False
    assert standard_basis == "standard_reject_blocked"
    assert discovery_admit is True
    assert discovery_basis == "representative_discovery_mcs_included"


def test_ledger_uses_explicit_module6_admission_contract_for_representative_discovery(tmp_path: Path) -> None:
    run_dir = build_synthetic_module5_run(tmp_path)
    _rewrite_policy_contract(
        run_dir,
        policy_class=MODULE6_RUN_POLICY_REPRESENTATIVE_DISCOVERY,
        reject=True,
        module6_admit=True,
    )
    base_cfg = make_test_config()
    cfg = replace(
        base_cfg,
        intake=replace(
            base_cfg.intake,
            run_policy_class=MODULE6_RUN_POLICY_REPRESENTATIVE_DISCOVERY,
            min_availability_ratio=0.75,
            min_observed_sessions=20,
        ),
    )
    loaded = load_module5_run(run_dir, cfg)
    ledgers = materialize_canonical_ledgers(loaded, run_dir / "module6_ledgers", cfg)
    assert ledgers["strategy_master"]["reject"].astype(bool).all()
    assert ledgers["strategy_master"]["module6_admit"].astype(bool).all()
    assert ledgers["strategy_master"]["portfolio_admit_flag"].astype(bool).all()


def test_reduction_uses_policy_bound_intake_for_representative_discovery(tmp_path: Path) -> None:
    run_dir = build_synthetic_module5_run(tmp_path)
    _rewrite_policy_contract(
        run_dir,
        policy_class=MODULE6_RUN_POLICY_REPRESENTATIVE_DISCOVERY,
        reject=True,
        module6_admit=True,
    )
    base_cfg = make_test_config()
    cfg = replace(
        base_cfg,
        intake=replace(
            base_cfg.intake,
            run_policy_class=MODULE6_RUN_POLICY_REPRESENTATIVE_DISCOVERY,
            min_availability_ratio=0.75,
            min_observed_sessions=20,
        ),
    )
    loaded = load_module5_run(run_dir, cfg)
    ledgers = materialize_canonical_ledgers(loaded, run_dir / "ledgers", cfg)
    store = build_matrix_store(ledgers=ledgers, run=loaded, output_dir=run_dir / "matrix_out", config=cfg)
    matrices = open_matrix_store(store)
    matrices["column_index"] = store.column_index
    reduction = reduce_universe(ledgers=ledgers, matrices=matrices, run=loaded, output_dir=run_dir / "reduce_out", config=cfg)
    assert ledgers["strategy_master"]["reject"].astype(bool).all()
    assert reduction.admitted_instances.shape[0] > 0


def test_reduction_uses_candidate_level_overlap_support_for_representative_discovery(tmp_path: Path) -> None:
    run_dir = build_synthetic_module5_run(tmp_path)
    _rewrite_policy_contract(
        run_dir,
        policy_class=MODULE6_RUN_POLICY_REPRESENTATIVE_DISCOVERY,
        reject=True,
        module6_admit=True,
    )
    _drop_trade_log_support_for_split(run_dir, split_id="wf_000")
    base_cfg = make_test_config()
    cfg = replace(
        base_cfg,
        intake=replace(
            base_cfg.intake,
            run_policy_class=MODULE6_RUN_POLICY_REPRESENTATIVE_DISCOVERY,
            min_availability_ratio=0.75,
            min_observed_sessions=20,
        ),
    )
    loaded = load_module5_run(run_dir, cfg)
    ledgers = materialize_canonical_ledgers(loaded, run_dir / "ledgers_support_fallback", cfg)
    store = build_matrix_store(ledgers=ledgers, run=loaded, output_dir=run_dir / "matrix_support_fallback", config=cfg)
    matrices = open_matrix_store(store)
    matrices["column_index"] = store.column_index
    reduction = reduce_universe(
        ledgers=ledgers,
        matrices=matrices,
        run=loaded,
        output_dir=run_dir / "reduce_support_fallback",
        config=cfg,
    )
    assert reduction.admitted_instances.shape[0] > 0


def test_reduction_allows_empty_trade_log_for_representative_discovery(tmp_path: Path) -> None:
    run_dir = build_synthetic_module5_run(tmp_path)
    _rewrite_policy_contract(
        run_dir,
        policy_class=MODULE6_RUN_POLICY_REPRESENTATIVE_DISCOVERY,
        reject=True,
        module6_admit=True,
    )
    _clear_trade_log(run_dir)
    base_cfg = make_test_config()
    cfg = replace(
        base_cfg,
        intake=replace(
            base_cfg.intake,
            run_policy_class=MODULE6_RUN_POLICY_REPRESENTATIVE_DISCOVERY,
            min_availability_ratio=0.75,
            min_observed_sessions=20,
        ),
    )
    loaded = load_module5_run(run_dir, cfg)
    ledgers = materialize_canonical_ledgers(loaded, run_dir / "ledgers_empty_trade_log", cfg)
    store = build_matrix_store(ledgers=ledgers, run=loaded, output_dir=run_dir / "matrix_empty_trade_log", config=cfg)
    matrices = open_matrix_store(store)
    matrices["column_index"] = store.column_index
    reduction = reduce_universe(
        ledgers=ledgers,
        matrices=matrices,
        run=loaded,
        output_dir=run_dir / "reduce_empty_trade_log",
        config=cfg,
    )
    assert reduction.admitted_instances.shape[0] > 0
    assert int(reduction.overlap_proxy.symbol_support.nnz) == 0
    assert int(reduction.overlap_proxy.composite.nnz) == 0


def test_standard_policy_stays_fail_closed_when_canonical_symbol_support_is_missing(tmp_path: Path) -> None:
    run_dir = build_synthetic_module5_run(tmp_path)
    _drop_trade_log_support_for_split(run_dir, split_id="wf_000")
    base_cfg = make_test_config()
    loaded = load_module5_run(run_dir, base_cfg)
    ledgers = materialize_canonical_ledgers(loaded, run_dir / "ledgers_standard_strict", base_cfg)
    store = build_matrix_store(ledgers=ledgers, run=loaded, output_dir=run_dir / "matrix_standard_strict", config=base_cfg)
    matrices = open_matrix_store(store)
    matrices["column_index"] = store.column_index
    try:
        reduce_universe(
            ledgers=ledgers,
            matrices=matrices,
            run=loaded,
            output_dir=run_dir / "reduce_standard_strict",
            config=base_cfg,
        )
    except Exception as exc:
        assert "execution overlap proxy input missing symbol support" in str(exc)
    else:
        raise AssertionError("standard policy unexpectedly allowed missing canonical symbol support")
