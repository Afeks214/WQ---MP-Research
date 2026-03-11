from __future__ import annotations

import pandas as pd
import pytest

from weightiz.module6.io import load_module5_run
from weightiz.module6.utils import Module6ValidationError
from tests.module6_testkit import build_synthetic_module5_run, make_test_config


def test_load_module5_run_reads_bridge_artifacts(tmp_path):
    run_dir = build_synthetic_module5_run(tmp_path)
    loaded = load_module5_run(run_dir, make_test_config())
    assert loaded.strategy_instance_selection.shape[0] > 0
    assert loaded.strategy_instance_session_returns.shape[0] > 0
    assert "canonical_reference_policy" in loaded.strategy_instance_selection.columns
    assert "session_id" in loaded.trade_log.columns
    counts = loaded.strategy_instance_selection.loc[
        loaded.strategy_instance_selection["portfolio_instance_role"] == "canonical_portfolio"
    ].groupby("candidate_id").size()
    assert counts.min() == 1
    assert counts.max() == 1


def test_load_module5_run_missing_canonical_reference_fails_closed(tmp_path):
    run_dir = build_synthetic_module5_run(tmp_path)
    selection_path = run_dir / "strategy_instance_selection.parquet"
    selection = pd.read_parquet(selection_path).drop(columns=["canonical_reference_policy"])
    selection.to_parquet(selection_path, index=False)
    with pytest.raises(Module6ValidationError, match="canonical_reference_policy"):
        load_module5_run(run_dir, make_test_config())
