from __future__ import annotations

import json

from weightiz.module6.io import load_module5_run
from weightiz.module6.ledger import materialize_canonical_ledgers
from tests.module6_testkit import build_synthetic_module5_run, make_test_config


def test_materialize_canonical_ledgers_writes_expected_files(tmp_path):
    run_dir = build_synthetic_module5_run(tmp_path)
    loaded = load_module5_run(run_dir, make_test_config())
    out = materialize_canonical_ledgers(loaded, run_dir / "module6_ledgers", make_test_config())
    assert out["strategy_master"]["strategy_pk"].is_unique
    assert out["strategy_instance_master"]["strategy_instance_pk"].is_unique
    assert not out["strategy_session_ledger"].duplicated(["strategy_instance_pk", "session_id"]).any()


def test_materialize_canonical_ledgers_accepts_list_backed_stats_scalars(tmp_path):
    run_dir = build_synthetic_module5_run(tmp_path)
    stats_path = next((run_dir / "candidates").glob("*/candidate_stats.json"))
    stats_doc = json.loads(stats_path.read_text(encoding="utf-8"))
    stats_doc["dsr"]["dsr"] = [stats_doc["dsr"]["dsr"]]
    stats_path.write_text(json.dumps(stats_doc), encoding="utf-8")

    loaded = load_module5_run(run_dir, make_test_config())
    out = materialize_canonical_ledgers(loaded, run_dir / "module6_ledgers_list", make_test_config())
    assert out["strategy_instance_master"]["stats_dsr"].notna().all()
