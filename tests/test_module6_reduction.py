from __future__ import annotations

from module6.io import load_module5_run
from module6.ledger import materialize_canonical_ledgers
from module6.matrices import build_matrix_store
from module6.reduction import reduce_universe
from module6.runtime import open_matrix_store
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
