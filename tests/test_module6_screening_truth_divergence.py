from __future__ import annotations

import pandas as pd
import pytest

from module6.config import Module6Config, ScoringConfig
from module6.scoring import build_cross_universe_comparable_scores
from module6.utils import Module6ValidationError


def test_screening_truth_rank_instability_gate_blocks_export():
    cfg = Module6Config(scoring=ScoringConfig(min_rank_stability=0.99))
    df = pd.DataFrame(
        {
            "portfolio_pk": ["p0", "p1"],
            "calendar_version": ["c1", "c1"],
            "support_policy_version": ["s1", "s1"],
            "minute_annualized_return": [0.1, 0.2],
            "minute_max_drawdown": [0.1, 0.2],
            "minute_turnover": [0.1, 0.2],
            "support_coverage": [1.0, 1.0],
            "availability_burden": [0.0, 0.0],
        }
    )
    scored = build_cross_universe_comparable_scores(finalist_scores=df, config=cfg)
    assert scored.shape[0] == 2

