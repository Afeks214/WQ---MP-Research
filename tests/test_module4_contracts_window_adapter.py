from __future__ import annotations

import numpy as np
import pytest

from weightiz.module4.contracts import (
    RiskFilterConfig,
    apply_optional_risk_filters,
    build_module4_input_contracts,
)
from weightiz.module4.window_adapter import WindowAdapterConfig, adapt_windows


def _sample_inputs(A: int = 2, T: int = 5, W: int = 3):
    rng = np.random.default_rng(11)
    alpha = rng.normal(size=(T, A, 2)).astype(np.float64)  # [T,A,S] to test canonicalization
    score = rng.normal(size=(A, T, 4)).astype(np.float64)
    prof = rng.normal(size=(A, T, 3)).astype(np.float64)
    structure = rng.normal(size=(A, T, 6, W)).astype(np.float64)
    context = rng.normal(size=(A, T, 5, W)).astype(np.float64)
    fp = rng.normal(size=(A, T, 2, W)).astype(np.float64)
    regime = rng.normal(size=(A, T, 1, W)).astype(np.float64)
    tradable = np.ones((A, T), dtype=bool)
    phase = np.zeros(T, dtype=np.int8)
    enabled = np.array([True] * A, dtype=bool)
    src_idx = np.broadcast_to(np.arange(T, dtype=np.int64)[None, :], (A, T)).copy()
    return {
        "alpha_signal_tensor": alpha,
        "score_tensor": score,
        "profile_stat_tensor": prof,
        "structure_tensor": structure,
        "context_tensor": context,
        "profile_fingerprint_tensor": fp,
        "profile_regime_tensor": regime,
        "tradable_mask": tradable,
        "phase_code": phase,
        "asset_enabled_mask": enabled,
        "source_time_index_at": src_idx,
    }


def test_contracts_canonicalize_to_a_t_prefix():
    raw = _sample_inputs(A=2, T=5, W=4)
    c = build_module4_input_contracts(**raw)
    assert c.alpha_signal_tensor.shape == (2, 5, 2)
    assert c.score_tensor.shape == (2, 5, 4)
    assert c.profile_stat_tensor.shape == (2, 5, 3)
    assert c.structure_tensor.shape == (2, 5, 6, 4)
    assert c.context_tensor.shape == (2, 5, 5, 4)
    assert c.profile_fingerprint_tensor.shape == (2, 5, 2, 4)
    assert c.profile_regime_tensor.shape == (2, 5, 1, 4)


def test_contracts_reject_forward_looking_source_index():
    raw = _sample_inputs(A=1, T=4, W=2)
    bad = raw["source_time_index_at"].copy()
    bad[0, 1] = 3
    raw["source_time_index_at"] = bad
    with pytest.raises(RuntimeError, match="CAUSALITY_VIOLATION"):
        build_module4_input_contracts(**raw)


def test_risk_filters_optional_behavior():
    tradable = np.array([[True, True, False], [True, False, True]], dtype=bool)
    vol = np.array([[0.1, 2.0, 0.2], [0.2, 0.1, 0.4]], dtype=np.float64)
    spr = np.array([[0.01, 0.03, 0.01], [0.02, 0.01, 0.05]], dtype=np.float64)
    liq = np.array([[0.9, 0.7, 0.8], [0.95, 0.4, 0.99]], dtype=np.float64)

    out = apply_optional_risk_filters(
        tradable_mask_at=tradable,
        volatility_tensor_at=vol,
        spread_tensor_at=spr,
        liquidity_score_at=liq,
        cfg=RiskFilterConfig(max_volatility=1.0, max_spread=0.04, min_liquidity=0.8),
    )
    exp = np.array([[True, False, False], [True, False, False]], dtype=bool)
    np.testing.assert_array_equal(out, exp)


def test_window_adapter_multi_mode_shapes_and_dtype():
    raw = _sample_inputs(A=2, T=6, W=3)
    c = build_module4_input_contracts(**raw)
    out = adapt_windows(c, WindowAdapterConfig(mode="multi_window", anchor_window_index=1))

    assert out.structure_adapted.shape == (2, 6, 6)
    assert out.context_adapted.shape == (2, 6, 5)
    assert out.fingerprint_adapted.shape == (2, 6, 2)
    assert out.regime_hint.shape == (2, 6, 1)
    assert out.selected_window_idx.shape == (2, 6)
    assert out.window_score.shape == (2, 6, 3)
    assert out.regime_confidence_window.shape == (2, 6, 3)
    assert out.structure_adapted.dtype == np.float64


def test_window_adapter_tie_break_prefers_anchor_then_lowest_index():
    A, T, W = 1, 3, 4
    raw = _sample_inputs(A=A, T=T, W=W)

    # Force identical utility and confidence across windows.
    raw["structure_tensor"][:] = 0.0
    raw["context_tensor"][:] = 0.0
    raw["profile_fingerprint_tensor"][:] = 0.0
    raw["profile_regime_tensor"][:] = 1.0

    c = build_module4_input_contracts(**raw)
    out = adapt_windows(c, WindowAdapterConfig(mode="multi_window", anchor_window_index=2))
    np.testing.assert_array_equal(out.selected_window_idx, np.full((A, T), 2, dtype=np.int16))

    out2 = adapt_windows(c, WindowAdapterConfig(mode="multi_window", anchor_window_index=0))
    np.testing.assert_array_equal(out2.selected_window_idx, np.zeros((A, T), dtype=np.int16))
