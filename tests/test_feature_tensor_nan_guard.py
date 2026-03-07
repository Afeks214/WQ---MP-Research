from __future__ import annotations

import numpy as np
import pytest

from weightiz_module2_core import validate_feature_tensor_contract


def test_feature_tensor_nan_guard_raises():
    t = np.zeros((2, 3, 4, 2), dtype=np.float64)
    t[0, 0, 0, 0] = np.nan
    with pytest.raises(RuntimeError, match="FEATURE_TENSOR_CONTAINS_NAN"):
        validate_feature_tensor_contract(t)


def test_feature_tensor_inf_guard_raises():
    t = np.zeros((2, 3, 4, 2), dtype=np.float64)
    t[0, 0, 0, 0] = np.inf
    with pytest.raises(RuntimeError, match="FEATURE_TENSOR_CONTAINS_INF"):
        validate_feature_tensor_contract(t)
