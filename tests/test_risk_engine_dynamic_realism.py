from __future__ import annotations

import numpy as np
import pytest

from weightiz.module4.risk_engine import CostConfig, ExecutionRealismConfig, RiskConfig, simulate_portfolio_from_signals


def _risk_cfg() -> RiskConfig:
    return RiskConfig(
        max_position_buying_power_frac=10.0,
        overnight_exposure_equity_mult=100.0,
        daily_loss_limit_frac=1.0,
        account_disable_equity=0.0,
    )


def test_dynamic_cost_model_respects_configured_bucket_coefficients_and_session_multipliers() -> None:
    px = np.full((3, 1), 100.0, dtype=np.float64)
    volume = np.full((3, 1), 100.0, dtype=np.float64)
    tgt = np.array([[10.0], [20.0], [30.0]], dtype=np.float64)
    realism = ExecutionRealismConfig(
        cost_model="dynamic_bucketed_v1",
        max_volume_participation=1.0,
        low_rvol_slippage_bps=4.0,
        mid_rvol_slippage_bps=7.0,
        high_rvol_slippage_bps=9.0,
        spread_tick_mult=2.0,
        open_bucket_minutes=30,
        close_bucket_minutes=15,
        open_slippage_mult=1.5,
        mid_slippage_mult=1.0,
        close_slippage_mult=1.25,
        participation_slippage_coeff=1.0,
        dynamic_slippage_bps_cap=100.0,
        rth_open_minute=570,
        flat_time_minute=945,
        rvol_ta=np.array([[0.5], [1.0], [2.0]], dtype=np.float64),
        tick_size_a=np.array([0.01], dtype=np.float64),
        minute_of_day_t=np.array([570, 700, 940], dtype=np.int16),
    )

    out = simulate_portfolio_from_signals(
        px,
        tgt,
        100_000.0,
        CostConfig(),
        _risk_cfg(),
        volume_ta=volume,
        execution_realism=realism,
    )

    expected_bps = np.array(
        [
            (4.0 + 1.0) * 1.5 * 1.1,
            (7.0 + 1.0) * 1.0 * 1.1,
            (9.0 + 1.0) * 1.25 * 1.1,
        ],
        dtype=np.float64,
    )
    expected_slippage = np.array([10.0, 10.0, 10.0], dtype=np.float64) * 100.0 * expected_bps * 1.0e-4
    np.testing.assert_allclose(out.slippage_cost_ta[:, 0], expected_slippage, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(out.trade_cost_ta[:, 0], expected_slippage, rtol=0.0, atol=1e-12)
    assert out.execution_cost_model == "dynamic_bucketed_v1"


def test_dynamic_cost_model_without_volume_does_not_assume_full_participation() -> None:
    px = np.full((1, 1), 100.0, dtype=np.float64)
    tgt = np.array([[10.0]], dtype=np.float64)
    realism = ExecutionRealismConfig(
        cost_model="dynamic_bucketed_v1",
        low_rvol_slippage_bps=4.0,
        mid_rvol_slippage_bps=4.0,
        high_rvol_slippage_bps=4.0,
        spread_tick_mult=0.0,
        open_bucket_minutes=0,
        close_bucket_minutes=0,
        open_slippage_mult=1.0,
        mid_slippage_mult=1.0,
        close_slippage_mult=1.0,
        participation_slippage_coeff=1.0,
        dynamic_slippage_bps_cap=100.0,
        rth_open_minute=570,
        flat_time_minute=945,
        rvol_ta=np.array([[1.0]], dtype=np.float64),
        tick_size_a=np.array([0.01], dtype=np.float64),
        minute_of_day_t=np.array([700], dtype=np.int16),
    )

    out = simulate_portfolio_from_signals(
        px,
        tgt,
        100_000.0,
        CostConfig(),
        _risk_cfg(),
        execution_realism=realism,
    )

    np.testing.assert_allclose(out.fill_cap_qty_ta[:, 0], np.array([np.nan], dtype=np.float64), rtol=0.0, atol=0.0, equal_nan=True)
    np.testing.assert_allclose(out.participation_rate_ta[:, 0], np.array([0.0], dtype=np.float64), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(out.slippage_cost_ta[:, 0], np.array([0.4], dtype=np.float64), rtol=0.0, atol=1e-12)


def test_dynamic_cost_model_fails_closed_when_required_inputs_are_missing() -> None:
    px = np.full((1, 1), 100.0, dtype=np.float64)
    tgt = np.array([[10.0]], dtype=np.float64)
    volume = np.full((1, 1), 100.0, dtype=np.float64)

    with pytest.raises(RuntimeError, match="requires rvol_ta"):
        simulate_portfolio_from_signals(
            px,
            tgt,
            10_000.0,
            CostConfig(),
            _risk_cfg(),
            volume_ta=volume,
            execution_realism=ExecutionRealismConfig(
                cost_model="dynamic_bucketed_v1",
                tick_size_a=np.array([0.01], dtype=np.float64),
                minute_of_day_t=np.array([570], dtype=np.int16),
            ),
        )


def test_dynamic_cost_model_rejects_shape_mismatches() -> None:
    px = np.full((1, 1), 100.0, dtype=np.float64)
    tgt = np.array([[10.0]], dtype=np.float64)
    volume = np.full((1, 1), 100.0, dtype=np.float64)

    with pytest.raises(RuntimeError, match="tick_size_a shape mismatch"):
        simulate_portfolio_from_signals(
            px,
            tgt,
            10_000.0,
            CostConfig(),
            _risk_cfg(),
            volume_ta=volume,
            execution_realism=ExecutionRealismConfig(
                cost_model="dynamic_bucketed_v1",
                rvol_ta=np.array([[1.0]], dtype=np.float64),
                tick_size_a=np.array([0.01, 0.02], dtype=np.float64),
                minute_of_day_t=np.array([570], dtype=np.int16),
            ),
        )
