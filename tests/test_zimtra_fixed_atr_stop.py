import numpy as np

from risk_engine import CostConfig, RiskConfig, simulate_portfolio_task


def test_fixed_atr_stop_trigger_by_intrabar_low_high() -> None:
    T, A = 5, 1
    open_px = np.array([[100.0], [100.0], [100.0], [100.0], [100.0]], dtype=np.float64)
    high_px = np.array([[101.0], [101.0], [101.0], [101.0], [101.0]], dtype=np.float64)
    low_px = np.array([[99.0], [99.0], [95.0], [100.0], [100.0]], dtype=np.float64)
    close_px = np.array([[100.0], [100.0], [100.0], [100.0], [100.0]], dtype=np.float64)

    signals = {
        "open": open_px,
        "high": high_px,
        "low": low_px,
        "close": close_px,
        "ATR": np.full((T, A), 1.0, dtype=np.float64),
        "D": np.zeros((T, A), dtype=np.float64),
        "A": np.full((T, A), 0.1, dtype=np.float64),
        "DELTA_EFF": np.full((T, A), 1.0, dtype=np.float64),
        "S_BREAK": np.full((T, A), 1.0, dtype=np.float64),
        "S_REJECT": np.zeros((T, A), dtype=np.float64),
        "RVOL": np.full((T, A), 2.0, dtype=np.float64),
        "POC": close_px.copy(),
        "VAH": (close_px + 1.0).copy(),
        "VAL": (close_px - 1.0).copy(),
        "SIGNED_SCORE": np.full((T, A), 1.0, dtype=np.float64),
    }
    strategy = {
        "strategy_id": "atr_stop_test",
        "family": "F1",
        "W": 60,
        "lev_target": 1.0,
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
        minute_of_day=np.array([600, 601, 602, 603, 604], dtype=np.int16),
        session_id=np.zeros(T, dtype=np.int64),
        bar_valid=np.ones((T, A), dtype=bool),
        last_valid_close=signals["close"],
        wf_split_idx=0,
        cpcv_fold_idx=0,
        scenario_id="baseline",
    )
    assert any(r["reason"] == "E1_FIXED_ATR_STOP" for r in out.trade_log)
