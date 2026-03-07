import numpy as np

from risk_engine import CostConfig, RiskConfig, simulate_portfolio_task


def test_realized_pnl_uses_pre_update_average_price() -> None:
    # Build scenario with entry at t0, stop trigger at t1, mandatory exit at t2.
    T, A = 3, 1
    open_px = np.array([[100.0], [100.0], [120.0]], dtype=np.float64)
    high_px = np.array([[101.0], [101.0], [121.0]], dtype=np.float64)
    low_px = np.array([[99.0], [98.0], [119.0]], dtype=np.float64)
    close_px = np.array([[100.0], [100.0], [120.0]], dtype=np.float64)

    signals = {
        "open": open_px,
        "high": high_px,
        "low": low_px,
        "close": close_px,
        "ATR": np.ones((T, A), dtype=np.float64),
        "D": np.zeros((T, A), dtype=np.float64),
        "A": np.full((T, A), 0.1, dtype=np.float64),
        "DELTA_EFF": np.ones((T, A), dtype=np.float64),
        "S_BREAK": np.array([[1.0], [1.0], [0.0]], dtype=np.float64),
        "S_REJECT": np.zeros((T, A), dtype=np.float64),
        "RVOL": np.full((T, A), 2.0, dtype=np.float64),
        "POC": close_px.copy(),
        "VAH": (close_px + 1.0).copy(),
        "VAL": (close_px - 1.0).copy(),
    }

    strategy = {
        "strategy_id": "pnl_avg_test",
        "family": "F1",
        "W": 60,
        "lev_target": 2.0,
        "exit_model": "E1",
        "atr_stop_mult": 1.0,
        "time_exit_bars": None,
        "s_break_thr": 0.5,
        "s_reject_thr": None,
        "d_thr": None,
        "d_low": None,
        "d_high": None,
        "a_thr": None,
        "a_accept_thr": None,
        "rvol_thr": 1.0,
        "de_thr": None,
        "va_dist": None,
        "s_break_mid_low": None,
        "s_break_mid_high": None,
    }

    out = simulate_portfolio_task(
        strategy=strategy,
        signals=signals,
        symbols=("SPY",),
        split_mask=np.ones(T, dtype=bool),
        cost_cfg=CostConfig(),
        risk_cfg=RiskConfig(),
        initial_cash=100000.0,
        ts_ns=np.arange(T, dtype=np.int64) * 60 * 1_000_000_000,
        minute_of_day=np.array([600, 601, 602], dtype=np.int16),
        session_id=np.zeros(T, dtype=np.int64),
        bar_valid=np.ones((T, A), dtype=bool),
        last_valid_close=signals["close"],
        wf_split_idx=0,
        cpcv_fold_idx=0,
        scenario_id="baseline",
    )

    # Entry exec price uses +1 tick slippage: 100.01.
    # Exit exec price uses -1 tick slippage: 119.99.
    # Shares opened at t0: floor(200000 / 100.01) = 1999.
    # Expected realized pnl = (119.99 - 100.01) * 1999 = 39940.02.
    pnl = float(out.per_asset_cumret["SPY"])
    assert abs(pnl - 39940.02) < 1e-2
