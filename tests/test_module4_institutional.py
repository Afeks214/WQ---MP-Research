import numpy as np
import pytest

from weightiz_module1_core import EngineConfig, OrderIdx, Phase, ProfileStatIdx, ScoreIdx, preallocate_state
from weightiz_module3_structure import ContextIdx, Module3Output, Struct30mIdx
from weightiz_module4_strategy_funnel import (
    Module4Config,
    OrderFlagBit,
    _execute_to_target,
    run_module4_strategy_funnel,
)


def _mk_state(
    T: int = 8,
    A: int = 2,
    *,
    initial_cash: float = 1_000_000.0,
    daily_loss_limit_abs: float = 50_000.0,
) -> object:
    start_ns = np.datetime64("2025-01-06T14:30:00", "ns").astype(np.int64)
    ts_ns = start_ns + np.arange(T, dtype=np.int64) * np.int64(60_000_000_000)
    cfg = EngineConfig(
        T=T,
        A=A,
        B=240,
        tick_size=np.full(A, 0.01, dtype=np.float64),
        initial_cash=float(initial_cash),
        daily_loss_limit_abs=float(daily_loss_limit_abs),
    )
    st = preallocate_state(ts_ns, cfg, tuple(f"A{i}" for i in range(A)))
    st.phase[:] = np.int8(Phase.LIVE)
    st.bar_valid[:] = True

    base = 100.0 + np.arange(T, dtype=np.float64)[:, None] * 0.10
    st.open_px[:] = base
    st.high_px[:] = base + 0.20
    st.low_px[:] = base - 0.20
    st.close_px[:] = base + 0.05
    st.volume[:] = 10_000.0
    st.rvol[:] = 1.2
    st.atr_floor[:] = 0.5

    st.scores[:] = 0.0
    st.profile_stats[:] = 0.0
    st.scores[:, :, int(ScoreIdx.SCORE_BO_LONG)] = 0.90
    st.scores[:, :, int(ScoreIdx.SCORE_BO_SHORT)] = 0.10
    st.scores[:, :, int(ScoreIdx.SCORE_REJ_LONG)] = 0.20
    st.scores[:, :, int(ScoreIdx.SCORE_REJ_SHORT)] = 0.10

    st.profile_stats[:, :, int(ProfileStatIdx.DCLIP)] = 1.0
    st.profile_stats[:, :, int(ProfileStatIdx.Z_DELTA)] = 1.0
    st.profile_stats[:, :, int(ProfileStatIdx.GBREAK)] = 1.0
    st.profile_stats[:, :, int(ProfileStatIdx.GREJECT)] = 0.1
    st.profile_stats[:, :, int(ProfileStatIdx.SIGMA_EFF)] = 0.5

    st.vp[:] = 0.0
    st.vp[:, :, 100] = 5.0
    st.vp[:, :, 130] = 4.0
    st.vp[:, :, 115] = 0.5
    return st


def _mk_m3(st) -> Module3Output:
    T, A = st.cfg.T, st.cfg.A
    c3 = int(ContextIdx.N_FIELDS)
    k3 = int(Struct30mIdx.N_FIELDS)
    ctx = np.full((T, A, c3), np.nan, dtype=np.float64)
    ctx[:, :, int(ContextIdx.CTX_X_POC)] = 0.5
    ctx[:, :, int(ContextIdx.CTX_X_VAH)] = 1.0
    ctx[:, :, int(ContextIdx.CTX_X_VAL)] = -1.0
    ctx[:, :, int(ContextIdx.CTX_VA_WIDTH_X)] = 2.0
    ctx[:, :, int(ContextIdx.CTX_DCLIP_MEAN)] = 1.0
    ctx[:, :, int(ContextIdx.CTX_AFFINITY_MEAN)] = 0.7
    ctx[:, :, int(ContextIdx.CTX_ZDELTA_MEAN)] = 1.2
    ctx[:, :, int(ContextIdx.CTX_DELTA_EFF_MEAN)] = 0.6
    ctx[:, :, int(ContextIdx.CTX_TREND_GATE_SPREAD_MEAN)] = 0.2
    ctx[:, :, int(ContextIdx.CTX_POC_DRIFT_X)] = 0.5
    ctx[:, :, int(ContextIdx.CTX_VALID_RATIO)] = 1.0
    ctx[:, :, int(ContextIdx.CTX_IB_HIGH_X)] = 1.2
    ctx[:, :, int(ContextIdx.CTX_IB_LOW_X)] = -1.2
    ctx[:, :, int(ContextIdx.CTX_POC_VS_PREV_VA)] = 1.2

    block_features = np.full((T, A, k3), np.nan, dtype=np.float64)
    block_features[:, :, int(Struct30mIdx.SKEW_ANCHOR)] = -0.5
    block_features[:, :, int(Struct30mIdx.X_POC)] = 0.5
    block_features[:, :, int(Struct30mIdx.X_VAH)] = 1.0
    block_features[:, :, int(Struct30mIdx.X_VAL)] = -1.0

    src = np.tile(np.arange(T, dtype=np.int64)[:, None], (1, A))
    valid = np.ones((T, A), dtype=bool)
    return Module3Output(
        block_id_t=np.arange(T, dtype=np.int64),
        block_seq_t=np.zeros(T, dtype=np.int16),
        block_end_flag_t=np.ones(T, dtype=bool),
        block_start_t_index_t=np.arange(T, dtype=np.int64),
        block_end_t_index_t=np.arange(T, dtype=np.int64),
        block_features_tak=block_features,
        block_valid_ta=valid.copy(),
        context_tac=ctx,
        context_valid_ta=valid,
        context_source_t_index_ta=src,
    )


def test_pending_open_nan_does_not_crash_when_non_strict() -> None:
    st = _mk_state(T=5, A=2)
    m3 = _mk_m3(st)
    st.bar_valid[1, 0] = False
    st.open_px[1, 0] = np.nan
    st.close_px[1, 0] = np.nan

    cfg = Module4Config(execution_strict_prices=False)
    out = run_module4_strategy_funnel(st, m3, cfg)

    assert out.filled_qty_ta[1, 0] == pytest.approx(0.0, abs=1e-12)
    assert (int(st.order_flags[1, 0]) & int(OrderFlagBit.EXEC_SKIPPED_BAD_PRICE)) != 0
    assert abs(float(out.filled_qty_ta[1, 1])) > 1e-12


def test_pending_open_nan_raises_when_strict() -> None:
    st = _mk_state(T=5, A=2)
    m3 = _mk_m3(st)
    st.bar_valid[1, 0] = False
    st.open_px[1, 0] = np.nan
    st.close_px[1, 0] = np.nan

    cfg = Module4Config(execution_strict_prices=True, fail_on_non_finite_input=False)
    with pytest.raises(RuntimeError):
        run_module4_strategy_funnel(st, m3, cfg)


def test_kill_switch_records_zero_target() -> None:
    st = _mk_state(T=5, A=1, initial_cash=1_000.0, daily_loss_limit_abs=1.0)
    m3 = _mk_m3(st)
    st.open_px[:, 0] = 100.0
    st.high_px[:, 0] = 101.0
    st.low_px[:, 0] = 0.5
    st.close_px[:, 0] = 100.0
    st.close_px[1, 0] = 1.0

    out = run_module4_strategy_funnel(st, m3, Module4Config(execution_strict_prices=False))
    kill_rows = np.flatnonzero(out.kill_switch_t)
    assert kill_rows.size > 0
    t = int(kill_rows[0])

    np.testing.assert_allclose(
        st.orders[t, :, int(OrderIdx.TARGET_QTY)],
        np.zeros(st.cfg.A, dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        out.target_qty_ta[t],
        np.zeros(st.cfg.A, dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        st.position_qty[t],
        np.zeros(st.cfg.A, dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )
    flag_val = int(st.order_flags[t, 0])
    assert (flag_val & int(OrderFlagBit.KILL_SWITCH)) != 0
    assert (flag_val & int(OrderFlagBit.FLATTEN)) != 0
    assert (flag_val & int(OrderFlagBit.MOC_EXEC)) != 0


def test_prefix_invariance_no_lookahead_module4() -> None:
    st1 = _mk_state(T=32, A=2)
    st2 = _mk_state(T=32, A=2)
    m31 = _mk_m3(st1)
    m32 = _mk_m3(st2)
    cfg = Module4Config(execution_strict_prices=False)

    run_module4_strategy_funnel(st1, m31, cfg)
    run_module4_strategy_funnel(st2, m32, cfg)

    t0 = 18
    st2.open_px[t0 + 1 :] *= 1.03
    st2.high_px[t0 + 1 :] *= 1.05
    st2.low_px[t0 + 1 :] *= 0.97
    st2.close_px[t0 + 1 :] *= 1.04
    m32.context_tac[t0 + 1 :, :, int(ContextIdx.CTX_X_VAH)] += 0.30
    m32.context_tac[t0 + 1 :, :, int(ContextIdx.CTX_X_VAL)] -= 0.30
    m32.block_features_tak[t0 + 1 :, :, int(Struct30mIdx.SKEW_ANCHOR)] *= -1.0

    out1 = run_module4_strategy_funnel(st1, m31, cfg)
    out2 = run_module4_strategy_funnel(st2, m32, cfg)

    np.testing.assert_allclose(
        out1.target_qty_ta[: t0 + 1],
        out2.target_qty_ta[: t0 + 1],
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        out1.filled_qty_ta[: t0 + 1],
        out2.filled_qty_ta[: t0 + 1],
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        st1.equity[: t0 + 1],
        st2.equity[: t0 + 1],
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )


def test_accounting_flip_and_avg_cost_invariants() -> None:
    cfg = Module4Config(
        commission_bps=0.0,
        spread_tick_mult=0.0,
        slippage_bps_low_rvol=0.0,
        slippage_bps_mid_rvol=0.0,
        slippage_bps_high_rvol=0.0,
        stress_slippage_mult=1.0,
    )
    eps = float(cfg.eps)
    pos = np.array([0.0], dtype=np.float64)
    avg_cost = np.array([0.0], dtype=np.float64)
    cash = 0.0
    realized = 0.0
    rvol = np.array([1.0], dtype=np.float64)
    tick_size = np.array([0.01], dtype=np.float64)

    path = [
        (10.0, 100.0),
        (15.0, 110.0),
        (8.0, 120.0),
        (-6.0, 90.0),
        (0.0, 80.0),
    ]
    for target_q, px in path:
        cash, realized, delta, cost, skipped = _execute_to_target(
            pos=pos,
            avg_cost=avg_cost,
            cash=cash,
            realized=realized,
            target=np.array([target_q], dtype=np.float64),
            price=np.array([px], dtype=np.float64),
            rvol=rvol,
            tick_size=tick_size,
            cfg=cfg,
            strict=True,
            eps=eps,
        )
        assert np.all(~skipped)
        assert np.all(np.isfinite(delta))
        assert np.all(np.isfinite(cost))
        assert np.all(np.isfinite(pos))
        assert np.all(np.isfinite(avg_cost))
        assert np.isfinite(realized)

    assert pos[0] == pytest.approx(0.0, abs=1e-12)
    assert avg_cost[0] == pytest.approx(0.0, abs=1e-12)
    assert realized == pytest.approx(70.0, abs=1e-9)
    assert realized > 0.0
