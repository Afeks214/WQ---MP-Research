import numpy as np

from weightiz.module4.risk_engine import CostConfig, RiskConfig, simulate_portfolio_task


def test_delever_executes_at_1546_after_1545_check() -> None:
    T, A = 4, 1
    px = np.array([[100.0], [100.0], [100.0], [100.0]], dtype=np.float64)
    signals = {
        "open": px.copy(),
        "high": px.copy(),
        "low": px.copy(),
        "close": px.copy(),
        "ATR": np.full((T, A), 1.0, dtype=np.float64),
        "D": np.zeros((T, A), dtype=np.float64),
        "A": np.full((T, A), 0.1, dtype=np.float64),
        "DELTA_EFF": np.full((T, A), 1.0, dtype=np.float64),
        "S_BREAK": np.full((T, A), 1.0, dtype=np.float64),
        "S_REJECT": np.zeros((T, A), dtype=np.float64),
        "RVOL": np.full((T, A), 2.0, dtype=np.float64),
        "POC": px.copy(),
        "VAH": (px + 1.0).copy(),
        "VAL": (px - 1.0).copy(),
        "SIGNED_SCORE": np.full((T, A), 1.0, dtype=np.float64),
    }
    strategy = {
        "strategy_id": "delever_test",
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
        risk_cfg=RiskConfig(),
        initial_cash=100000.0,
        ts_ns=np.arange(T, dtype=np.int64) * 60 * 1_000_000_000,
        minute_of_day=np.array([944, 945, 946, 947], dtype=np.int16),
        session_id=np.zeros(T, dtype=np.int64),
        bar_valid=np.ones((T, A), dtype=bool),
        last_valid_close=signals["close"],
        wf_split_idx=0,
        cpcv_fold_idx=0,
        scenario_id="baseline",
    )
    reasons = [r["reason"] for r in out.trade_log]
    assert any("AUTO_DELEVER_1545_TO_1546" in r for r in reasons)
