from __future__ import annotations

import numpy as np
import pytest

from weightiz.module4.risk_engine import CostConfig, RiskConfig, simulate_portfolio_from_signals
from weightiz.shared.validation.dtype_guard import assert_float64
from weightiz.module2.core import build_feature_tensor_multiaxis
from weightiz.shared.io.shared_feature_store import attach_shared_feature_store, close_shared_feature_store, create_shared_feature_store


def test_assert_float64_failure():
    with pytest.raises(RuntimeError, match="FLOAT64_ENFORCEMENT_FAILURE"):
        assert_float64("x", np.zeros((2, 2), dtype=np.float32))


def test_tensor_and_shm_are_float64():
    T, A = 8, 2
    open_ta = np.ones((T, A), dtype=np.float64)
    high_ta = np.ones((T, A), dtype=np.float64) * 1.01
    low_ta = np.ones((T, A), dtype=np.float64) * 0.99
    close_ta = np.ones((T, A), dtype=np.float64)
    volume_ta = np.ones((T, A), dtype=np.float64) * 100
    tensor, _fm, _wm = build_feature_tensor_multiaxis(open_ta, high_ta, low_ta, close_ta, volume_ta, windows=[15, 30])
    assert tensor.dtype == np.float64

    reg, h = create_shared_feature_store(tensor)
    try:
        wh = attach_shared_feature_store(reg)
        try:
            assert wh.array.dtype == np.float64
        finally:
            close_shared_feature_store(wh, is_master=False)
    finally:
        close_shared_feature_store(h, is_master=True, owner_pid=reg.owner_pid)


def test_risk_engine_dtype_enforced():
    px = np.ones((6, 2), dtype=np.float64)
    tgt = np.zeros((6, 2), dtype=np.float64)
    out = simulate_portfolio_from_signals(px, tgt, 1_000_000.0, CostConfig(), RiskConfig())
    assert out.equity_curve.dtype == np.float64
    assert out.daily_returns.dtype == np.float64

    with pytest.raises(RuntimeError, match="FLOAT64_ENFORCEMENT_FAILURE"):
        simulate_portfolio_from_signals(px.astype(np.float32), tgt, 1_000_000.0, CostConfig(), RiskConfig())
