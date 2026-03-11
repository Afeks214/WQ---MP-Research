from __future__ import annotations

import numpy as np

from weightiz.module3 import ContextIdx, StructIdx
from weightiz.module4.regime_classifier import (
    REGIME_B_SHAPE,
    REGIME_DOUBLE_DISTRIBUTION,
    REGIME_NONE,
    REGIME_P_SHAPE,
    REGIME_TREND,
    classify_regime,
)
from weightiz.module4.strategy_funnel import Module4Config


def _base_inputs(A: int = 2, T: int = 4):
    structure = np.zeros((A, T, int(StructIdx.N_FIELDS)), dtype=np.float64)
    context = np.zeros((A, T, int(ContextIdx.N_FIELDS)), dtype=np.float64)
    hint = np.zeros((A, T, 1), dtype=np.float64)
    tradable = np.ones((A, T), dtype=bool)
    structure[:, :, int(StructIdx.VALID_RATIO)] = 1.0
    context[:, :, int(ContextIdx.CTX_VALID_RATIO)] = 1.0
    return structure, context, hint, tradable


def test_regime_classifier_matches_locked_policy() -> None:
    structure, context, hint, tradable = _base_inputs()

    structure[0, 0, int(StructIdx.TREND_GATE_SPREAD_MEAN)] = 0.20
    context[0, 0, int(ContextIdx.CTX_TREND_GATE_SPREAD_MEAN)] = 0.20
    structure[0, 0, int(StructIdx.POC_DRIFT_X)] = 0.70
    context[0, 0, int(ContextIdx.CTX_POC_DRIFT_X)] = 0.70
    context[0, 0, int(ContextIdx.CTX_REGIME_CODE)] = 1.0

    structure[0, 1, int(StructIdx.SKEW_ANCHOR)] = 0.80
    context[0, 1, int(ContextIdx.CTX_POC_DRIFT_X)] = 0.20
    context[0, 1, int(ContextIdx.CTX_REGIME_CODE)] = 4.0
    hint[0, 1, 0] = 4.0

    structure[0, 2, int(StructIdx.SKEW_ANCHOR)] = -0.80
    context[0, 2, int(ContextIdx.CTX_POC_DRIFT_X)] = 0.20
    context[0, 2, int(ContextIdx.CTX_REGIME_CODE)] = 4.0
    hint[0, 2, 0] = 4.0

    context[0, 3, int(ContextIdx.CTX_REGIME_CODE)] = 3.0
    context[0, 3, int(ContextIdx.CTX_REGIME_PERSISTENCE)] = 1.0
    context[0, 3, int(ContextIdx.CTX_POC_DRIFT_X)] = 0.20
    hint[0, 3, 0] = 3.0

    tradable[1, 0] = False

    cfg = Module4Config()
    out = classify_regime(
        structure_adapted=structure,
        context_adapted=context,
        regime_hint=hint,
        tradable_mask=tradable,
        cfg4=cfg,
    )

    assert out.regime_id[0, 0] == REGIME_TREND
    assert out.regime_id[0, 1] == REGIME_P_SHAPE
    assert out.regime_id[0, 2] == REGIME_B_SHAPE
    assert out.regime_id[0, 3] == REGIME_DOUBLE_DISTRIBUTION
    assert out.regime_id[1, 0] == REGIME_NONE
    assert out.regime_valid_mask[1, 0] == np.False_
    assert out.regime_confidence.dtype == np.float64
    assert out.regime_score.shape == (2, 4, 6)


def test_regime_classifier_low_confidence_mask_triggers() -> None:
    structure, context, hint, tradable = _base_inputs(A=1, T=1)
    structure[0, 0, int(StructIdx.VALID_RATIO)] = 0.5
    context[0, 0, int(ContextIdx.CTX_VALID_RATIO)] = 0.5
    cfg = Module4Config(regime_confidence_min=0.5)
    out = classify_regime(
        structure_adapted=structure,
        context_adapted=context,
        regime_hint=hint,
        tradable_mask=tradable,
        cfg4=cfg,
    )
    assert out.low_regime_confidence_mask[0, 0]


def test_regime_classifier_responds_to_trend_spread_threshold() -> None:
    structure, context, hint, tradable = _base_inputs(A=1, T=1)
    structure[0, 0, int(StructIdx.TREND_GATE_SPREAD_MEAN)] = 0.20
    context[0, 0, int(ContextIdx.CTX_TREND_GATE_SPREAD_MEAN)] = 0.20
    structure[0, 0, int(StructIdx.POC_DRIFT_X)] = 0.70
    context[0, 0, int(ContextIdx.CTX_POC_DRIFT_X)] = 0.70
    context[0, 0, int(ContextIdx.CTX_REGIME_CODE)] = 1.0

    out_lo = classify_regime(
        structure_adapted=structure,
        context_adapted=context,
        regime_hint=hint,
        tradable_mask=tradable,
        cfg4=Module4Config(trend_spread_min=0.05),
    )
    out_hi = classify_regime(
        structure_adapted=structure,
        context_adapted=context,
        regime_hint=hint,
        tradable_mask=tradable,
        cfg4=Module4Config(trend_spread_min=0.50),
    )
    assert out_lo.regime_id[0, 0] == REGIME_TREND
    assert out_lo.regime_score[0, 0, int(REGIME_TREND)] > out_hi.regime_score[0, 0, int(REGIME_TREND)]


def test_regime_classifier_responds_to_shape_skew_threshold() -> None:
    structure, context, hint, tradable = _base_inputs(A=1, T=1)
    structure[0, 0, int(StructIdx.SKEW_ANCHOR)] = 0.80
    context[0, 0, int(ContextIdx.CTX_POC_DRIFT_X)] = 0.20
    context[0, 0, int(ContextIdx.CTX_REGIME_CODE)] = 4.0
    hint[0, 0, 0] = 4.0

    out_lo = classify_regime(
        structure_adapted=structure,
        context_adapted=context,
        regime_hint=hint,
        tradable_mask=tradable,
        cfg4=Module4Config(shape_skew_min_abs=0.35),
    )
    out_hi = classify_regime(
        structure_adapted=structure,
        context_adapted=context,
        regime_hint=hint,
        tradable_mask=tradable,
        cfg4=Module4Config(shape_skew_min_abs=2.0),
    )
    assert out_lo.regime_id[0, 0] == REGIME_P_SHAPE
    assert out_lo.regime_score[0, 0, int(REGIME_P_SHAPE)] > out_hi.regime_score[0, 0, int(REGIME_P_SHAPE)]
