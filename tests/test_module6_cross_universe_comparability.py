from __future__ import annotations

import pandas as pd
import pytest

from weightiz.module6.scoring import build_cross_universe_comparable_scores
from weightiz.module6.utils import Module6ValidationError
from tests.module6_testkit import make_test_config


def test_cross_universe_support_short_rejected():
    df = pd.DataFrame(
        {
            "portfolio_pk": ["p0", "p1"],
            "calendar_version": ["c1", "c1"],
            "support_policy_version": ["s1", "s1"],
            "comparison_support_recomputed": [True, True],
            "minute_annualized_return": [0.1, 0.2],
            "minute_max_drawdown": [0.1, 0.05],
            "minute_turnover": [0.1, 0.1],
            "support_coverage": [0.6, 1.0],
            "availability_burden": [0.4, 0.0],
        }
    )
    comparison_support = pd.DataFrame({"session_id": list(range(10))})
    scored = build_cross_universe_comparable_scores(finalist_scores=df, config=make_test_config(), comparison_support=comparison_support)
    assert scored.loc[scored["portfolio_pk"] == "p0", "cross_universe_reject"].iloc[0]


def test_cross_universe_calendar_mismatch_blocks():
    df = pd.DataFrame(
        {
            "portfolio_pk": ["p0", "p1"],
            "calendar_version": ["c1", "c2"],
            "support_policy_version": ["s1", "s1"],
            "comparison_support_recomputed": [True, True],
            "minute_annualized_return": [0.1, 0.2],
            "minute_max_drawdown": [0.1, 0.05],
            "minute_turnover": [0.1, 0.1],
            "support_coverage": [1.0, 1.0],
            "availability_burden": [0.0, 0.0],
        }
    )
    with pytest.raises(Module6ValidationError):
        build_cross_universe_comparable_scores(finalist_scores=df, config=make_test_config(), comparison_support=pd.DataFrame({"session_id": [1, 2]}))


def test_cross_universe_requires_canonical_recompute():
    df = pd.DataFrame(
        {
            "portfolio_pk": ["p0", "p1"],
            "calendar_version": ["c1", "c1"],
            "support_policy_version": ["s1", "s1"],
            "comparison_support_recomputed": [True, False],
            "minute_annualized_return": [0.1, 0.2],
            "minute_max_drawdown": [0.1, 0.05],
            "minute_turnover": [0.1, 0.1],
            "support_coverage": [1.0, 1.0],
            "availability_burden": [0.0, 0.0],
        }
    )
    with pytest.raises(Module6ValidationError, match="comparison-support recomputation"):
        build_cross_universe_comparable_scores(finalist_scores=df, config=make_test_config(), comparison_support=pd.DataFrame({"session_id": [1, 2]}))
