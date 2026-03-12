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


def test_simulate_portfolio_rejects_nonmonotone_session_ids() -> None:
    px = np.full((3, 1), 100.0, dtype=np.float64)
    tgt = np.zeros((3, 1), dtype=np.float64)

    try:
        simulate_portfolio_from_signals(
            px,
            tgt,
            1_000.0,
            CostConfig(),
            RiskConfig(),
            session_id_t=np.array([0, 2, 1], dtype=np.int64),
        )
    except RuntimeError as exc:
        assert str(exc) == "risk_engine session_id_t must be nondecreasing"
    else:
        raise AssertionError("expected nonmonotone session ids to fail closed")


def test_simulate_portfolio_accrues_short_borrow_on_session_change_only() -> None:
    px = np.full((3, 1), 100.0, dtype=np.float64)
    tgt = np.full((3, 1), -10.0, dtype=np.float64)
    out = simulate_portfolio_from_signals(
        px,
        tgt,
        1_000.0,
        CostConfig(short_borrow_apr=0.252),
        RiskConfig(
            max_position_buying_power_frac=10.0,
            overnight_exposure_equity_mult=100.0,
            daily_loss_limit_frac=1.0,
            account_disable_equity=0.0,
        ),
        session_id_t=np.array([0, 0, 1], dtype=np.int64),
    )

    expected_borrow = 1_000.0 * 0.252 / 252.0
    np.testing.assert_allclose(out.equity_curve, np.array([1_000.0, 1_000.0, 1_000.0 - expected_borrow], dtype=np.float64))

    out_long = simulate_portfolio_from_signals(
        px,
        np.full((3, 1), 10.0, dtype=np.float64),
        1_000.0,
        CostConfig(short_borrow_apr=0.252),
        RiskConfig(
            max_position_buying_power_frac=10.0,
            overnight_exposure_equity_mult=100.0,
            daily_loss_limit_frac=1.0,
            account_disable_equity=0.0,
        ),
        session_id_t=np.array([0, 0, 1], dtype=np.int64),
    )
    np.testing.assert_allclose(out_long.equity_curve, np.array([1_000.0, 1_000.0, 1_000.0], dtype=np.float64))


def test_simulate_portfolio_accrues_debit_interest_only_when_cash_is_negative() -> None:
    px = np.full((3, 1), 100.0, dtype=np.float64)
    tgt = np.full((3, 1), 15.0, dtype=np.float64)
    out = simulate_portfolio_from_signals(
        px,
        tgt,
        1_000.0,
        CostConfig(debit_apr=0.252),
        RiskConfig(
            max_position_buying_power_frac=2.0,
            overnight_exposure_equity_mult=100.0,
            daily_loss_limit_frac=1.0,
            account_disable_equity=0.0,
        ),
        session_id_t=np.array([0, 0, 1], dtype=np.int64),
    )

    expected_debit = 500.0 * 0.252 / 252.0
    np.testing.assert_allclose(out.equity_curve, np.array([1_000.0, 1_000.0, 1_000.0 - expected_debit], dtype=np.float64))

    out_flat_cash = simulate_portfolio_from_signals(
        px,
        np.full((3, 1), 10.0, dtype=np.float64),
        1_000.0,
        CostConfig(debit_apr=0.252),
        RiskConfig(
            max_position_buying_power_frac=2.0,
            overnight_exposure_equity_mult=100.0,
            daily_loss_limit_frac=1.0,
            account_disable_equity=0.0,
        ),
        session_id_t=np.array([0, 0, 1], dtype=np.int64),
    )
    np.testing.assert_allclose(out_flat_cash.equity_curve, np.array([1_000.0, 1_000.0, 1_000.0], dtype=np.float64))


def test_simulate_portfolio_caps_fills_to_reported_bar_volume() -> None:
    px = np.full((4, 1), 100.0, dtype=np.float64)
    volume = np.full((4, 1), 5.0, dtype=np.float64)
    tgt = np.array([[10.0], [10.0], [0.0], [0.0]], dtype=np.float64)

    out = simulate_portfolio_from_signals(
        px,
        tgt,
        10_000.0,
        CostConfig(),
        RiskConfig(
            max_position_buying_power_frac=10.0,
            overnight_exposure_equity_mult=100.0,
            daily_loss_limit_frac=1.0,
            account_disable_equity=0.0,
        ),
        volume_ta=volume,
    )

    np.testing.assert_allclose(out.filled_qty_ta[:, 0], np.array([5.0, 5.0, -5.0, -5.0], dtype=np.float64))
    np.testing.assert_allclose(out.position_qty_ta[:, 0], np.array([5.0, 10.0, 5.0, 0.0], dtype=np.float64))


def test_simulate_portfolio_slippage_scales_with_participation_and_zero_trade_is_stable() -> None:
    px = np.full((2, 1), 100.0, dtype=np.float64)
    volume = np.full((2, 1), 100.0, dtype=np.float64)

    small = simulate_portfolio_from_signals(
        px[:1],
        np.array([[10.0]], dtype=np.float64),
        10_000.0,
        CostConfig(slippage_bps=10.0),
        RiskConfig(
            max_position_buying_power_frac=10.0,
            overnight_exposure_equity_mult=100.0,
            daily_loss_limit_frac=1.0,
            account_disable_equity=0.0,
        ),
        volume_ta=volume[:1],
    )
    large = simulate_portfolio_from_signals(
        px[:1],
        np.array([[50.0]], dtype=np.float64),
        10_000.0,
        CostConfig(slippage_bps=10.0),
        RiskConfig(
            max_position_buying_power_frac=10.0,
            overnight_exposure_equity_mult=100.0,
            daily_loss_limit_frac=1.0,
            account_disable_equity=0.0,
        ),
        volume_ta=volume[:1],
    )
    idle = simulate_portfolio_from_signals(
        px,
        np.zeros((2, 1), dtype=np.float64),
        10_000.0,
        CostConfig(slippage_bps=10.0),
        RiskConfig(),
        volume_ta=volume,
    )

    np.testing.assert_allclose(small.trade_cost_ta[0, 0], 1.1, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(large.trade_cost_ta[0, 0], 7.5, rtol=0.0, atol=1e-12)
    assert float(large.trade_cost_ta[0, 0]) > float(small.trade_cost_ta[0, 0])
    np.testing.assert_allclose(idle.trade_cost_ta, np.zeros((2, 1), dtype=np.float64))
