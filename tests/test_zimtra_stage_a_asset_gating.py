import numpy as np

from risk_engine import CostConfig, RiskConfig, simulate_portfolio_task


def test_stage_a_active_asset_gating_only_trades_selected_assets() -> None:
    T, A = 8, 5
    px = np.full((T, A), 100.0, dtype=np.float64)
    signals = {
        "open": px.copy(),
        "high": (px + 1.0).copy(),
        "low": (px - 1.0).copy(),
        "close": px.copy(),
        "ATR": np.ones((T, A), dtype=np.float64),
        "D": np.zeros((T, A), dtype=np.float64),
        "A": np.full((T, A), 0.1, dtype=np.float64),
        "DELTA_EFF": np.ones((T, A), dtype=np.float64),
        "S_BREAK": np.ones((T, A), dtype=np.float64),
        "S_REJECT": np.zeros((T, A), dtype=np.float64),
        "RVOL": np.full((T, A), 2.0, dtype=np.float64),
        "POC": px.copy(),
        "VAH": (px + 1.0).copy(),
        "VAL": (px - 1.0).copy(),
    }
    strategy = {
        "strategy_id": "stage_a_gate_test",
        "family": "F1",
        "W": 60,
        "lev_target": 1.0,
        "exit_model": "E5",
        "atr_stop_mult": None,
        "time_exit_bars": 390,
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

    symbols = ("SPY", "QQQ", "IWM", "XLK", "XLE")
    out = simulate_portfolio_task(
        strategy=strategy,
        signals=signals,
        symbols=symbols,
        split_mask=np.ones(T, dtype=bool),
        cost_cfg=CostConfig(),
        risk_cfg=RiskConfig(),
        initial_cash=100000.0,
        ts_ns=np.arange(T, dtype=np.int64) * 60 * 1_000_000_000,
        minute_of_day=np.full(T, 600, dtype=np.int16),
        session_id=np.zeros(T, dtype=np.int64),
        bar_valid=np.ones((T, A), dtype=bool),
        last_valid_close=signals["close"],
        wf_split_idx=0,
        cpcv_fold_idx=0,
        scenario_id="baseline",
        active_asset_indices=[0, 1, 2],
    )

    traded_symbols = {str(row["symbol"]) for row in out.trade_log}
    assert traded_symbols.issubset({"SPY", "QQQ", "IWM"})
    assert "XLK" not in traded_symbols
    assert "XLE" not in traded_symbols
