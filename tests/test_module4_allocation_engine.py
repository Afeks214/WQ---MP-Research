from __future__ import annotations

import numpy as np

from module4.allocation_engine import compute_normalized_signal_allocation
from weightiz_module4_strategy_funnel import Module4Config


def test_allocation_math_and_ranking_are_locked() -> None:
    conviction = np.array([[0.6, 0.0], [-0.3, 0.0]], dtype=np.float64)
    confidence = np.array([[0.5, 0.5], [1.0, 0.5]], dtype=np.float64)
    tradable = np.array([[True, True], [True, True]], dtype=bool)
    asset_enabled = np.array([True, True], dtype=bool)

    out = compute_normalized_signal_allocation(
        conviction_net=conviction,
        regime_confidence=confidence,
        tradable_mask=tradable,
        asset_enabled_mask=asset_enabled,
        cfg4=Module4Config(max_abs_weight=1.0),
    )

    np.testing.assert_allclose(out.allocation_score[:, 0], np.array([0.3, -0.3], dtype=np.float64))
    np.testing.assert_allclose(out.target_weight[:, 0], np.array([0.5, -0.5], dtype=np.float64))
    np.testing.assert_array_equal(out.allocation_rank[:, 1], np.array([0, 1], dtype=np.int16))


def test_allocation_zeroes_masked_assets() -> None:
    conviction = np.array([[0.4], [0.4]], dtype=np.float64)
    confidence = np.ones((2, 1), dtype=np.float64)
    tradable = np.array([[True], [False]], dtype=bool)
    asset_enabled = np.array([True, True], dtype=bool)
    out = compute_normalized_signal_allocation(
        conviction_net=conviction,
        regime_confidence=confidence,
        tradable_mask=tradable,
        asset_enabled_mask=asset_enabled,
        cfg4=Module4Config(max_abs_weight=1.0),
    )
    assert out.target_weight[1, 0] == 0.0
