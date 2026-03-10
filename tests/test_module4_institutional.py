from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from module3 import ContextIdx, StructIdx
from weightiz_module1_core import EngineConfig, Phase, ProfileStatIdx, ScoreIdx, preallocate_state
from weightiz_module4_strategy_funnel import (
    Module4Config,
    _execute_to_target,
    run_module4_signal_funnel,
    run_module4_strategy_funnel,
)


def _mk_state(T: int = 8, A: int = 2) -> object:
    start_ns = np.datetime64("2025-01-06T14:30:00", "ns").astype(np.int64)
    ts_ns = start_ns + np.arange(T, dtype=np.int64) * np.int64(60_000_000_000)
    cfg = EngineConfig(T=T, A=A, B=240, tick_size=np.full(A, 0.01, dtype=np.float64))
    st = preallocate_state(ts_ns, cfg, tuple(f"A{i}" for i in range(A)))
    st.phase[:] = np.int8(Phase.LIVE)
    st.bar_valid[:] = True
    st.scores[:] = 0.0
    st.profile_stats[:] = 0.0
    st.scores[:, :, int(ScoreIdx.SCORE_BO_LONG)] = 0.90
    st.scores[:, :, int(ScoreIdx.SCORE_BO_SHORT)] = 0.10
    st.profile_stats[:, :, int(ProfileStatIdx.DCLIP)] = 1.0
    st.profile_stats[:, :, int(ProfileStatIdx.Z_DELTA)] = 1.0
    return st


def _mk_m3(T: int, A: int) -> object:
    structure = np.zeros((A, T, int(StructIdx.N_FIELDS), 1), dtype=np.float64)
    context = np.zeros((A, T, int(ContextIdx.N_FIELDS), 1), dtype=np.float64)
    structure[:, :, int(StructIdx.VALID_RATIO), 0] = 1.0
    structure[:, :, int(StructIdx.TREND_GATE_SPREAD_MEAN), 0] = 0.2
    structure[:, :, int(StructIdx.POC_DRIFT_X), 0] = 0.6
    context[:, :, int(ContextIdx.CTX_VALID_RATIO), 0] = 1.0
    context[:, :, int(ContextIdx.CTX_TREND_GATE_SPREAD_MEAN), 0] = 0.2
    context[:, :, int(ContextIdx.CTX_POC_DRIFT_X), 0] = 0.6
    context[:, :, int(ContextIdx.CTX_REGIME_CODE), 0] = 1.0
    context[:, :, int(ContextIdx.CTX_REGIME_PERSISTENCE), 0] = 1.0
    return SimpleNamespace(
        structure_tensor=structure,
        context_tensor=context,
        profile_fingerprint_tensor=np.zeros((A, T, 1, 1), dtype=np.float64),
        profile_regime_tensor=np.zeros((A, T, 1, 1), dtype=np.float64),
        context_valid_ta=np.ones((T, A), dtype=bool),
        context_source_index_atw=np.broadcast_to(np.arange(T, dtype=np.int64)[None, :, None], (A, T, 1)).copy(),
    )


def test_execution_entrypoint_is_forbidden_with_non_strict_compat_flag() -> None:
    st = _mk_state(T=5, A=2)
    with pytest.raises(RuntimeError, match="MODULE4_EXECUTION_FORBIDDEN_IN_CANONICAL_PATH"):
        run_module4_strategy_funnel(st, _mk_m3(st.cfg.T, st.cfg.A), Module4Config(execution_strict_prices=False))


def test_execution_entrypoint_is_forbidden_with_strict_compat_flag() -> None:
    st = _mk_state(T=5, A=2)
    with pytest.raises(RuntimeError, match="MODULE4_EXECUTION_FORBIDDEN_IN_CANONICAL_PATH"):
        run_module4_strategy_funnel(
            st,
            _mk_m3(st.cfg.T, st.cfg.A),
            Module4Config(execution_strict_prices=True, fail_on_non_finite_input=False),
        )


def test_signal_path_is_prefix_invariant_to_future_only_changes() -> None:
    st1 = _mk_state(T=32, A=2)
    st2 = _mk_state(T=32, A=2)
    m31 = _mk_m3(st1.cfg.T, st1.cfg.A)
    m32 = _mk_m3(st2.cfg.T, st2.cfg.A)

    t0 = 18
    st2.scores[t0 + 1 :, :, int(ScoreIdx.SCORE_BO_LONG)] = 0.05
    st2.profile_stats[t0 + 1 :, :, int(ProfileStatIdx.Z_DELTA)] = 0.1
    m32.context_tensor[:, t0 + 1 :, int(ContextIdx.CTX_POC_DRIFT_X), 0] += 0.3
    m32.structure_tensor[:, t0 + 1 :, int(StructIdx.SKEW_ANCHOR), 0] = -0.5

    out1 = run_module4_signal_funnel(st1, m31, Module4Config())
    out2 = run_module4_signal_funnel(st2, m32, Module4Config())

    np.testing.assert_array_equal(out1.regime_primary_ta[: t0 + 1], out2.regime_primary_ta[: t0 + 1])
    np.testing.assert_allclose(out1.regime_confidence_ta[: t0 + 1], out2.regime_confidence_ta[: t0 + 1], rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(out1.intent_long_ta[: t0 + 1], out2.intent_long_ta[: t0 + 1])
    np.testing.assert_array_equal(out1.intent_short_ta[: t0 + 1], out2.intent_short_ta[: t0 + 1])
    np.testing.assert_allclose(out1.target_qty_ta[: t0 + 1], out2.target_qty_ta[: t0 + 1], rtol=0.0, atol=0.0)


def test_execute_to_target_returns_skipped_mask_for_bad_price() -> None:
    cfg = Module4Config(
        commission_bps=0.0,
        spread_tick_mult=0.0,
        slippage_bps_low_rvol=0.0,
        slippage_bps_mid_rvol=0.0,
        slippage_bps_high_rvol=0.0,
        stress_slippage_mult=1.0,
    )
    cash, realized, delta, cost, skipped = _execute_to_target(
        pos=np.array([0.0, 0.0], dtype=np.float64),
        avg_cost=np.array([0.0, 0.0], dtype=np.float64),
        cash=0.0,
        realized=0.0,
        target=np.array([1.0, 1.0], dtype=np.float64),
        price=np.array([100.0, np.nan], dtype=np.float64),
        rvol=np.array([1.0, 1.0], dtype=np.float64),
        tick_size=np.array([0.01, 0.01], dtype=np.float64),
        cfg=cfg,
        strict=True,
        eps=float(cfg.eps),
    )
    assert np.isfinite(cash)
    assert np.isfinite(realized)
    np.testing.assert_allclose(delta, np.array([1.0, 0.0], dtype=np.float64), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(cost, np.zeros(2, dtype=np.float64), rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(skipped, np.array([False, True], dtype=bool))


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
