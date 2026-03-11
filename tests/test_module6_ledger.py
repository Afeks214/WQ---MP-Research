from __future__ import annotations

from module6.io import load_module5_run
from module6.ledger import materialize_canonical_ledgers
from tests.module6_testkit import build_synthetic_module5_run, make_test_config


def test_materialize_canonical_ledgers_writes_expected_files(tmp_path):
    run_dir = build_synthetic_module5_run(tmp_path)
    loaded = load_module5_run(run_dir, make_test_config())
    out = materialize_canonical_ledgers(loaded, run_dir / "module6_ledgers", make_test_config())
    assert out["strategy_master"]["strategy_pk"].is_unique
    assert out["strategy_instance_master"]["strategy_instance_pk"].is_unique
    assert not out["strategy_session_ledger"].duplicated(["strategy_instance_pk", "session_id"]).any()

