from __future__ import annotations

import numpy as np

from weightiz.module4.risk_engine import CostConfig, RiskConfig, simulate_portfolio_from_signals


def _risk_cfg() -> RiskConfig:
    return RiskConfig(
        max_position_buying_power_frac=10.0,
        overnight_exposure_equity_mult=100.0,
        daily_loss_limit_frac=1.0,
        account_disable_equity=0.0,
    )


def test_locate_fee_applies_only_to_incremental_short_exposure() -> None:
    px = np.full((4, 1), 100.0, dtype=np.float64)
    tgt = np.array([[10.0], [0.0], [-5.0], [-10.0]], dtype=np.float64)
    out = simulate_portfolio_from_signals(
        px,
        tgt,
        10_000.0,
        CostConfig(locate_fee_per_share_short_entry=0.25),
        _risk_cfg(),
    )

    np.testing.assert_allclose(out.locate_cost_ta[:, 0], np.array([0.0, 0.0, 1.25, 1.25], dtype=np.float64))
    np.testing.assert_allclose(out.trade_cost_ta[:, 0], out.locate_cost_ta[:, 0], rtol=0.0, atol=1e-12)


def test_long_reduction_does_not_pay_locate_fee() -> None:
    px = np.full((2, 1), 100.0, dtype=np.float64)
    tgt = np.array([[10.0], [5.0]], dtype=np.float64)
    out = simulate_portfolio_from_signals(
        px,
        tgt,
        10_000.0,
        CostConfig(locate_fee_per_share_short_entry=0.50),
        _risk_cfg(),
    )

    np.testing.assert_allclose(out.locate_cost_ta[:, 0], np.zeros(2, dtype=np.float64))


def test_flat_to_short_pays_locate_on_full_short_size() -> None:
    px = np.full((1, 1), 100.0, dtype=np.float64)
    tgt = np.array([[-8.0]], dtype=np.float64)
    out = simulate_portfolio_from_signals(
        px,
        tgt,
        10_000.0,
        CostConfig(locate_fee_per_share_short_entry=0.125),
        _risk_cfg(),
    )

    np.testing.assert_allclose(out.locate_cost_ta[:, 0], np.array([1.0], dtype=np.float64))
