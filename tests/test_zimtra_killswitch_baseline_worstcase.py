import numpy as np

from weightiz.module4.risk_engine import CostConfig, RiskConfig, simulate_portfolio_task


def test_killswitch_uses_worstcase_and_flattens_next_open() -> None:
    T, A = 4, 1
    open_px = np.array([[100.0], [100.0], [50.0], [50.0]], dtype=np.float64)
    high_px = np.array([[101.0], [101.0], [55.0], [55.0]], dtype=np.float64)
    low_px = np.array([[99.0], [99.0], [1.0], [50.0]], dtype=np.float64)
    close_px = np.array([[100.0], [100.0], [50.0], [50.0]], dtype=np.float64)
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
        "strategy_id": "kill_test",
        "family": "F1",
        "W": 60,
        "lev_target": 2.0,
        "exit_model": "E5",
        "atr_stop_mult": None,
        "time_exit_bars": 1170,
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
        risk_cfg=RiskConfig(daily_max_loss_frac=0.10),
        initial_cash=100000.0,
        ts_ns=np.arange(T, dtype=np.int64) * 60 * 1_000_000_000,
        minute_of_day=np.array([600, 601, 602, 603], dtype=np.int16),
        session_id=np.zeros(T, dtype=np.int64),
        bar_valid=np.ones((T, A), dtype=bool),
        last_valid_close=signals["close"],
        wf_split_idx=0,
        cpcv_fold_idx=0,
        scenario_id="baseline",
    )
    assert any(r["reason"] == "KILL_SWITCH" for r in out.trade_log)
