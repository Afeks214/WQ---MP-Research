from __future__ import annotations

import numpy as np

from weightiz.module2.core import build_feature_tensor_multiaxis, validate_feature_tensor_contract


def test_feature_tensor_schema_shape_and_dtype():
    T, A = 12, 3
    open_ta = np.ones((T, A), dtype=np.float64)
    high_ta = open_ta + 0.1
    low_ta = open_ta - 0.1
    close_ta = open_ta + 0.01
    vol_ta = np.ones((T, A), dtype=np.float64) * 100

    tensor, feature_map, window_map = build_feature_tensor_multiaxis(
        open_ta,
        high_ta,
        low_ta,
        close_ta,
        vol_ta,
        windows=[15, 30],
    )
    assert tensor.shape == (A, T, len(feature_map), len(window_map))
    assert tensor.dtype == np.float64
    validate_feature_tensor_contract(tensor, {"shape": list(tensor.shape)})
