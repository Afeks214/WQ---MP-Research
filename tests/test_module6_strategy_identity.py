from __future__ import annotations

from weightiz.module6.io import load_module5_run
from weightiz.module6.ledger import materialize_canonical_ledgers
from tests.module6_testkit import build_synthetic_module5_run, make_test_config


def test_strategy_pk_is_stable_while_strategy_instance_pk_varies(tmp_path):
    run_dir = build_synthetic_module5_run(tmp_path)
    loaded = load_module5_run(run_dir, make_test_config())
    ledgers = materialize_canonical_ledgers(loaded, run_dir / "ledgers", make_test_config())
    instance_master = ledgers["strategy_instance_master"]
    subset = instance_master.loc[instance_master["candidate_id"] == "cand_000"]
    assert subset["strategy_pk"].nunique() == 1
    assert subset["strategy_instance_pk"].nunique() == 2

