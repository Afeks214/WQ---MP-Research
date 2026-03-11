from __future__ import annotations

import pandas as pd

from module6.frontier import select_diverse_finalists
from tests.module6_testkit import make_test_config


def test_frontier_selection_returns_unique_portfolios():
    scores = pd.DataFrame(
        {
            "portfolio_pk": ["p0", "p1", "p2"],
            "final_score": [0.9, 0.8, 0.7],
            "minute_annualized_return": [0.2, 0.15, 0.1],
            "minute_max_drawdown": [0.1, 0.05, 0.02],
            "minute_turnover": [0.1, 0.2, 0.3],
            "availability_burden": [0.0, 0.1, 0.2],
            "headroom": [0.8, 0.7, 0.9],
            "comparable_truth_score": [0.9, 0.8, 0.7],
        }
    )
    weights = pd.DataFrame(
        {
            "portfolio_pk": ["p0", "p0", "p1", "p1", "p2"],
            "strategy_instance_pk": ["a", "b", "a", "c", "d"],
            "target_weight": [0.5, 0.4, 0.6, 0.3, 0.9],
        }
    )
    strategy_frame = pd.DataFrame({"strategy_instance_pk": ["a", "b", "c", "d"], "cluster_id": [0, 1, 2, 3]})
    global_front, risk_front, operational_front, selected = select_diverse_finalists(
        scores=scores,
        portfolio_weights=weights,
        strategy_frame=strategy_frame,
        config=make_test_config(),
    )
    assert global_front["portfolio_pk"].is_unique
    assert risk_front["portfolio_pk"].is_unique
    assert operational_front["portfolio_pk"].is_unique
    assert selected["portfolio_pk"].is_unique

