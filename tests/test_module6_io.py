from __future__ import annotations

from module6.io import load_module5_run
from tests.module6_testkit import build_synthetic_module5_run, make_test_config


def test_load_module5_run_reads_bridge_artifacts(tmp_path):
    run_dir = build_synthetic_module5_run(tmp_path)
    loaded = load_module5_run(run_dir, make_test_config())
    assert loaded.strategy_instance_selection.shape[0] > 0
    assert loaded.strategy_instance_session_returns.shape[0] > 0
    counts = loaded.strategy_instance_selection.loc[
        loaded.strategy_instance_selection["portfolio_instance_role"] == "canonical_portfolio"
    ].groupby("candidate_id").size()
    assert counts.min() == 1
    assert counts.max() == 1

