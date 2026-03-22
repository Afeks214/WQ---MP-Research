from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from weightiz.module6.frontier import select_diverse_finalists
from weightiz.module6.utils import Module6ValidationError
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
            "minute_average_gross_exposure": [0.3, 0.4, 0.5],
            "minute_realized_volatility": [0.2, 0.2, 0.2],
            "session_average_gross_exposure": [0.3, 0.4, 0.5],
            "session_realized_volatility": [0.2, 0.2, 0.2],
            "comparable_truth_score": [0.9, 0.8, 0.7],
            "cross_universe_reject": [False, False, False],
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


def test_frontier_selection_excludes_cross_universe_rejected_rows():
    scores = pd.DataFrame(
        {
            "portfolio_pk": ["p0", "p1", "p2"],
            "final_score": [0.9, 0.8, 0.7],
            "minute_annualized_return": [0.2, 0.15, 0.1],
            "minute_max_drawdown": [0.1, 0.05, 0.02],
            "minute_turnover": [0.1, 0.2, 0.3],
            "availability_burden": [0.0, 0.1, 0.2],
            "headroom": [0.8, 0.7, 0.9],
            "minute_average_gross_exposure": [0.3, 0.4, 0.5],
            "minute_realized_volatility": [0.2, 0.2, 0.2],
            "session_average_gross_exposure": [0.3, 0.4, 0.5],
            "session_realized_volatility": [0.2, 0.2, 0.2],
            "comparable_truth_score": [0.9, -1.0, 0.7],
            "cross_universe_reject": [False, True, False],
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
        scores=scores.sort_values(["comparable_truth_score", "portfolio_pk"], ascending=[False, True], kind="mergesort").reset_index(drop=True),
        portfolio_weights=weights,
        strategy_frame=strategy_frame,
        config=make_test_config(),
    )
    rejected = {"p1"}
    assert rejected.isdisjoint(set(global_front["portfolio_pk"].astype(str)))
    assert rejected.isdisjoint(set(risk_front["portfolio_pk"].astype(str)))
    assert rejected.isdisjoint(set(operational_front["portfolio_pk"].astype(str)))
    assert rejected.isdisjoint(set(selected["portfolio_pk"].astype(str)))


def test_frontier_selection_fails_closed_when_all_rows_are_rejected():
    scores = pd.DataFrame(
        {
            "portfolio_pk": ["p0", "p1"],
            "final_score": [0.9, 0.8],
            "minute_annualized_return": [0.2, 0.15],
            "minute_max_drawdown": [0.1, 0.05],
            "minute_turnover": [0.1, 0.2],
            "availability_burden": [0.0, 0.1],
            "headroom": [0.8, 0.7],
            "minute_average_gross_exposure": [0.3, 0.4],
            "minute_realized_volatility": [0.2, 0.2],
            "session_average_gross_exposure": [0.3, 0.4],
            "session_realized_volatility": [0.2, 0.2],
            "comparable_truth_score": [-1.0, -1.0],
            "cross_universe_reject": [True, True],
        }
    )
    weights = pd.DataFrame(
        {
            "portfolio_pk": ["p0", "p1"],
            "strategy_instance_pk": ["a", "b"],
            "target_weight": [1.0, 1.0],
        }
    )
    strategy_frame = pd.DataFrame({"strategy_instance_pk": ["a", "b"], "cluster_id": [0, 1]})
    with pytest.raises(Module6ValidationError, match="NO_COMPARABLE_FINALISTS_SURVIVED"):
        select_diverse_finalists(
            scores=scores,
            portfolio_weights=weights,
            strategy_frame=strategy_frame,
            config=make_test_config(),
        )


def test_frontier_selection_excludes_dead_portfolios_even_if_cross_universe_flag_is_false():
    scores = pd.DataFrame(
        {
            "portfolio_pk": ["p0", "p1", "p2"],
            "final_score": [0.9, 0.95, 0.7],
            "minute_annualized_return": [0.2, 0.3, 0.1],
            "minute_max_drawdown": [0.1, 0.01, 0.02],
            "minute_turnover": [0.1, 0.05, 0.3],
            "availability_burden": [0.0, 0.0, 0.2],
            "headroom": [0.8, 0.9, 0.9],
            "minute_average_gross_exposure": [0.4, 0.4, 0.4],
            "minute_realized_volatility": [0.2, 0.2, 0.2],
            "session_average_gross_exposure": [0.4, 0.4, 0.4],
            "session_realized_volatility": [0.2, 0.2, 0.2],
            "comparable_truth_score": [0.9, 0.95, 0.7],
            "cross_universe_reject": [False, False, False],
            "session_disable_flag": [0, 1, 0],
            "session_breach_count": [0, 1, 0],
            "session_gross_exposure_peak": [1.0, 1.1, 0.8],
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
        scores=scores.sort_values(["comparable_truth_score", "portfolio_pk"], ascending=[False, True], kind="mergesort").reset_index(drop=True),
        portfolio_weights=weights,
        strategy_frame=strategy_frame,
        config=make_test_config(),
    )
    rejected = {"p1"}
    assert rejected.isdisjoint(set(global_front["portfolio_pk"].astype(str)))
    assert rejected.isdisjoint(set(risk_front["portfolio_pk"].astype(str)))
    assert rejected.isdisjoint(set(operational_front["portfolio_pk"].astype(str)))
    assert rejected.isdisjoint(set(selected["portfolio_pk"].astype(str)))


def test_frontier_selection_excludes_cash_like_portfolios_below_risk_budget():
    base_cfg = make_test_config()
    cfg = replace(
        base_cfg,
        scoring=replace(
            base_cfg.scoring,
            min_average_gross_exposure=0.15,
            min_realized_volatility=0.05,
        ),
    )
    scores = pd.DataFrame(
        {
            "portfolio_pk": ["p0", "p1"],
            "final_score": [0.9, 0.8],
            "minute_annualized_return": [0.2, 0.15],
            "minute_max_drawdown": [0.1, 0.05],
            "minute_turnover": [0.1, 0.2],
            "availability_burden": [0.0, 0.1],
            "minute_average_gross_exposure": [0.05, 0.25],
            "minute_realized_volatility": [0.02, 0.2],
            "session_average_gross_exposure": [0.05, 0.25],
            "session_realized_volatility": [0.02, 0.2],
            "comparable_truth_score": [0.9, 0.8],
            "cross_universe_reject": [False, False],
            "session_disable_flag": [0, 0],
            "session_breach_count": [0, 0],
            "session_gross_exposure_peak": [0.2, 0.4],
            "minute_disable_flag": [0, 0],
            "minute_breach_count": [0, 0],
            "minute_gross_exposure_peak": [0.2, 0.4],
        }
    )
    weights = pd.DataFrame(
        {
            "portfolio_pk": ["p0", "p1"],
            "strategy_instance_pk": ["a", "b"],
            "target_weight": [1.0, 1.0],
        }
    )
    strategy_frame = pd.DataFrame({"strategy_instance_pk": ["a", "b"], "cluster_id": [0, 1]})
    global_front, risk_front, operational_front, selected = select_diverse_finalists(
        scores=scores.sort_values(["comparable_truth_score", "portfolio_pk"], ascending=[False, True], kind="mergesort").reset_index(drop=True),
        portfolio_weights=weights,
        strategy_frame=strategy_frame,
        config=cfg,
    )
    excluded = {"p0"}
    assert excluded.isdisjoint(set(global_front["portfolio_pk"].astype(str)))
    assert excluded.isdisjoint(set(risk_front["portfolio_pk"].astype(str)))
    assert excluded.isdisjoint(set(operational_front["portfolio_pk"].astype(str)))
    assert excluded.isdisjoint(set(selected["portfolio_pk"].astype(str)))
