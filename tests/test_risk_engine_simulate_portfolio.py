from __future__ import annotations

import numpy as np

from weightiz.module4.risk_engine import CostConfig, RiskConfig, simulate_portfolio_from_signals


def test_simulate_portfolio_truncates_target_quantities_to_integer_shares() -> None:
    px = np.full((3, 1), 100.0, dtype=np.float64)
    tgt = np.array([[1.7], [2.2], [-1.4]], dtype=np.float64)

    out = simulate_portfolio_from_signals(
        px,
        tgt,
        10_000.0,
        CostConfig(),
        RiskConfig(
            max_position_buying_power_frac=1.0,
            overnight_exposure_equity_mult=100.0,
            daily_loss_limit_frac=1.0,
            account_disable_equity=0.0,
        ),
    )

    np.testing.assert_allclose(out.filled_qty_ta[:, 0], np.array([1.0, 1.0, -3.0], dtype=np.float64))
    np.testing.assert_allclose(out.position_qty_ta[:, 0], np.array([1.0, 2.0, -1.0], dtype=np.float64))
    assert np.all(np.isclose(out.filled_qty_ta, np.trunc(out.filled_qty_ta), rtol=0.0, atol=0.0))
    assert np.all(np.isclose(out.position_qty_ta, np.trunc(out.position_qty_ta), rtol=0.0, atol=0.0))


def test_simulate_portfolio_resets_daily_loss_baseline_on_session_change() -> None:
    px = np.array([[100.0], [92.0], [92.0], [88.0]], dtype=np.float64)
    tgt = np.array([[10.0], [10.0], [10.0], [10.0]], dtype=np.float64)
    cfg = RiskConfig(
        max_position_buying_power_frac=1.0,
        overnight_exposure_equity_mult=100.0,
        daily_loss_limit_frac=0.1,
        account_disable_equity=0.0,
    )

    try:
        simulate_portfolio_from_signals(px, tgt, 1_000.0, CostConfig(), cfg)
    except RuntimeError as exc:
        assert str(exc) == "RISK_DAILY_LOSS_BREACH"
    else:
        raise AssertionError("expected stale day_start_eq path to breach without session reset")

    out = simulate_portfolio_from_signals(
        px,
        tgt,
        1_000.0,
        CostConfig(),
        cfg,
        session_id_t=np.array([0, 0, 1, 1], dtype=np.int64),
    )

    np.testing.assert_allclose(out.daily_loss_t, np.array([0.0, 80.0, 0.0, 40.0], dtype=np.float64))
