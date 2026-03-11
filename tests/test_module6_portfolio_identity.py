from __future__ import annotations

from weightiz.module6.utils import portfolio_pk


def test_portfolio_pk_changes_when_policy_changes():
    base = dict(
        reduced_universe_id="ru",
        generator_family="random_sparse",
        rebalance_policy="band_10pct",
        target_weights_hash_value="abc",
        cash_policy="explicit_cash_residual",
        constraint_policy_version="c1",
        ranking_policy_version="r1",
        overnight_policy_version="o1",
        friction_policy_version="f1",
        support_policy_version="s1",
        calendar_version="cal1",
    )
    a = portfolio_pk(**base)
    b = portfolio_pk(**{**base, "rebalance_policy": "daily_close"})
    c = portfolio_pk(**{**base, "overnight_policy_version": "o2"})
    assert a != b
    assert a != c

