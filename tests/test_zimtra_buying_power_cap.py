import numpy as np

from weightiz.module4.risk_engine import CostConfig, RiskConfig, simulate_portfolio_task


def _common_signals(T: int, A: int) -> dict[str, np.ndarray]:
    px = np.full((T, A), 100.0, dtype=np.float64)
    return {
        "open": px.copy(),
        "high": px.copy(),
        "low": px.copy(),
        "close": px.copy(),
        "ATR": np.full((T, A), 1.0, dtype=np.float64),
        "D": np.full((T, A), 0.0, dtype=np.float64),
        "A": np.full((T, A), 0.1, dtype=np.float64),
        "DELTA_EFF": np.full((T, A), 1.0, dtype=np.float64),
        "S_BREAK": np.full((T, A), 1.0, dtype=np.float64),
        "S_REJECT": np.full((T, A), 0.0, dtype=np.float64),
        "RVOL": np.full((T, A), 2.0, dtype=np.float64),
        "POC": px.copy(),
        "VAH": (px + 1.0).copy(),
        "VAL": (px - 1.0).copy(),
        "SIGNED_SCORE": np.full((T, A), 1.0, dtype=np.float64),
    }


def test_buying_power_cap_never_exceeds_target_gross() -> None:
    T, A = 10, 2
    signals = _common_signals(T, A)
    split_mask = np.ones(T, dtype=bool)
    strategy = {
        "strategy_id": "cap_test",
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

    out = simulate_portfolio_task(
        strategy=strategy,
        signals=signals,
        symbols=("SPY", "QQQ"),
        split_mask=split_mask,
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
    )
    assert out.gross_exposure_peak <= 100000.0 * 1.0 + 1e-6
