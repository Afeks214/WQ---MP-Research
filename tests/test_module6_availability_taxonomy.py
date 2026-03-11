from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from module6.constants import AVAIL_FORCED_CASH_BY_RISK, AVAIL_FORCED_ZERO_BY_PORTFOLIO, AVAIL_INVALIDATED_BY_DQ, AVAIL_OBSERVED_ACTIVE, AVAIL_OBSERVED_FLAT, AVAIL_STRUCTURALLY_MISSING
from module6.io import load_module5_run
from module6.ledger import materialize_canonical_ledgers
from module6.utils import Module6ValidationError
from module6.utils import state_code_to_bool
from tests.module6_testkit import build_synthetic_module5_run, make_test_config


def test_availability_state_code_maps_to_boolean_hot_path():
    assert state_code_to_bool(AVAIL_OBSERVED_ACTIVE)
    assert state_code_to_bool(AVAIL_OBSERVED_FLAT)
    assert not state_code_to_bool(AVAIL_STRUCTURALLY_MISSING)
    assert not state_code_to_bool(AVAIL_INVALIDATED_BY_DQ)
    assert not state_code_to_bool(AVAIL_FORCED_ZERO_BY_PORTFOLIO)
    assert not state_code_to_bool(AVAIL_FORCED_CASH_BY_RISK)


def test_invalid_base_availability_code_fails_closed(tmp_path):
    run_dir = build_synthetic_module5_run(tmp_path)
    session_path = run_dir / "strategy_instance_session_returns.parquet"
    df = pd.read_parquet(session_path)
    df.loc[df.index[0], "availability_state_code"] = 5
    df.to_parquet(session_path, index=False)
    loaded = load_module5_run(run_dir, make_test_config())
    with pytest.raises(Module6ValidationError):
        materialize_canonical_ledgers(loaded, run_dir / "ledgers", make_test_config())
