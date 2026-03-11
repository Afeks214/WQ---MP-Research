from __future__ import annotations

import numpy as np

from weightiz.module4.strategy_intent_engine import generate_strategy_intent
from weightiz.module1.core import ProfileStatIdx, ScoreIdx
from weightiz.module4.strategy_funnel import Module4Config


def test_intent_generation_is_mutually_exclusive_and_flat_on_mask() -> None:
    A, T = 2, 3
    alpha = np.zeros((A, T, 1), dtype=np.float64)
    score = np.zeros((A, T, int(ScoreIdx.N_FIELDS)), dtype=np.float64)
    profile = np.zeros((A, T, int(ProfileStatIdx.N_FIELDS)), dtype=np.float64)
    regime_id = np.array([[2, 2, 1], [1, 4, 2]], dtype=np.int8)
    regime_conf = np.ones((A, T), dtype=np.float64)
    tradable = np.array([[True, True, True], [False, True, True]], dtype=bool)

    score[0, 0, int(ScoreIdx.SCORE_BO_LONG)] = 0.9
    score[0, 1, int(ScoreIdx.SCORE_BO_SHORT)] = 0.9
    score[0, 2, int(ScoreIdx.SCORE_BO_LONG)] = 0.8
    score[0, 2, int(ScoreIdx.SCORE_BO_SHORT)] = 0.8

    out = generate_strategy_intent(
        alpha_signal_tensor=alpha,
        score_tensor=score,
        profile_stat_tensor=profile,
        regime_id=regime_id,
        regime_confidence=regime_conf,
        tradable_mask=tradable,
        cfg4=Module4Config(entry_threshold=0.55, exit_threshold=0.25),
    )

    assert out.intent_long[0, 0]
    assert out.intent_short[0, 1]
    assert out.intent_flat[0, 2]
    assert out.intent_flat[1, 0]
    assert not np.any(out.intent_long & out.intent_short)
    assert out.intent_gate_mask.shape == (A, T, 6)
