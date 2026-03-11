from __future__ import annotations

import numpy as np

from weightiz.module3 import ContextIdx
from weightiz.module4.conviction_engine import compute_conviction
from weightiz.module1.core import ScoreIdx
from weightiz.module4.strategy_funnel import Module4Config


def test_conviction_is_zero_for_flat_and_clipped_for_directional_intent() -> None:
    A, T = 1, 3
    intent_long = np.array([[True, False, False]], dtype=bool)
    intent_short = np.array([[False, True, False]], dtype=bool)
    intent_flat = np.array([[False, False, True]], dtype=bool)
    utility = np.array([[2.0, -2.0, 1.0]], dtype=np.float64)
    confidence = np.ones((A, T), dtype=np.float64)
    score = np.ones((A, T, int(ScoreIdx.N_FIELDS)), dtype=np.float64)
    context = np.zeros((A, T, int(ContextIdx.N_FIELDS)), dtype=np.float64)
    context[:, :, int(ContextIdx.CTX_REGIME_PERSISTENCE)] = 1.0
    tradable = np.ones((A, T), dtype=bool)

    out = compute_conviction(
        intent_long=intent_long,
        intent_short=intent_short,
        intent_flat=intent_flat,
        signed_intent_utility=utility,
        regime_confidence=confidence,
        score_tensor=score,
        context_adapted=context,
        tradable_mask=tradable,
        cfg4=Module4Config(conviction_scale=1.0, conviction_clip=0.5),
    )

    assert out.conviction_net[0, 0] == 0.5
    assert out.conviction_net[0, 1] == -0.5
    assert out.conviction_net[0, 2] == 0.0
    assert out.conviction_long[0, 2] == 0.0
    assert out.conviction_short[0, 2] == 0.0
