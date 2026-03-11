from __future__ import annotations

import numpy as np

from module6.constants import AVAIL_FORCED_CASH_BY_RISK, AVAIL_FORCED_ZERO_BY_PORTFOLIO, AVAIL_INVALIDATED_BY_DQ, AVAIL_OBSERVED_ACTIVE, AVAIL_OBSERVED_FLAT, AVAIL_STRUCTURALLY_MISSING
from module6.utils import state_code_to_bool


def test_availability_state_code_maps_to_boolean_hot_path():
    assert state_code_to_bool(AVAIL_OBSERVED_ACTIVE)
    assert state_code_to_bool(AVAIL_OBSERVED_FLAT)
    assert not state_code_to_bool(AVAIL_STRUCTURALLY_MISSING)
    assert not state_code_to_bool(AVAIL_INVALIDATED_BY_DQ)
    assert not state_code_to_bool(AVAIL_FORCED_ZERO_BY_PORTFOLIO)
    assert not state_code_to_bool(AVAIL_FORCED_CASH_BY_RISK)

