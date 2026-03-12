from __future__ import annotations

from types import SimpleNamespace
import warnings

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from weightiz.module5.harness.evaluation_path import (
    assert_artifact_dependency_contract,
    benchmark_daily_returns,
    collect_funnel_payload,
    collect_micro_diagnostics_payload,
    collect_micro_profile_blocks_payload,
    equity_curve_payload,
    materialize_risk_outputs_into_state,
    trade_log_payload,
)
from weightiz.module5.harness.state_overlay import (
    BaseTensorState,
    CandidateScratch,
    CombinedStateView,
    FeatureOverlay,
    MarketOverlay,
)
from weightiz.module1.core import EngineConfig, Phase, ProfileStatIdx, ScoreIdx, preallocate_state
from weightiz.module3.bridge import ContextIdx
from weightiz.module4.strategy_funnel import RegimeIdx


def _timestamps_two_sessions() -> np.ndarray:
    vals = [
        np.datetime64("2024-01-02T14:30:00", "ns").astype(np.int64),
        np.datetime64("2024-01-02T14:31:00", "ns").astype(np.int64),
        np.datetime64("2024-01-03T14:30:00", "ns").astype(np.int64),
        np.datetime64("2024-01-03T14:31:00", "ns").astype(np.int64),
    ]
    return np.asarray(vals, dtype=np.int64)


def _engine_cfg(*, t_count: int, a_count: int, daily_loss_limit_abs: float = 15.0) -> EngineConfig:
    return EngineConfig(
        T=t_count,
        A=a_count,
        B=16,
        tick_size=np.full(a_count, 0.01, dtype=np.float64),
        mode="sealed",
        timezone="America/New_York",
        daily_loss_limit_abs=float(daily_loss_limit_abs),
        initial_cash=1_000.0,
    )


def _build_state(*, a_count: int = 2, daily_loss_limit_abs: float = 15.0):
    ts_ns = _timestamps_two_sessions()
    cfg = _engine_cfg(t_count=int(ts_ns.shape[0]), a_count=a_count, daily_loss_limit_abs=daily_loss_limit_abs)
    state = preallocate_state(ts_ns=ts_ns, cfg=cfg, symbols=tuple(f"S{i}" for i in range(a_count)))
    state.open_px[:, :] = np.asarray(
        [
            [100.0, 50.0][:a_count],
            [100.0, 50.0][:a_count],
            [80.0, 51.0][:a_count],
            [80.0, 51.0][:a_count],
        ],
        dtype=np.float64,
    )
    state.close_px[:, :] = np.asarray(
        [
            [100.0, 50.0][:a_count],
            [100.0, 50.0][:a_count],
            [70.0, 51.0][:a_count],
            [60.0, 51.0][:a_count],
        ],
        dtype=np.float64,
    )
    state.high_px[:, :] = np.maximum(state.open_px, state.close_px) + 0.5
    state.low_px[:, :] = np.minimum(state.open_px, state.close_px) - 0.5
    state.volume[:, :] = 1_000.0
    state.bar_valid[:, :] = True
    state.phase[:] = np.int8(Phase.LIVE)
    return state


def _dummy_m4_signal(t_count: int, a_count: int) -> SimpleNamespace:
    shape = (t_count, a_count)
    return SimpleNamespace(
        regime_primary_ta=np.full(shape, np.int8(RegimeIdx.TREND), dtype=np.int8),
        regime_confidence_ta=np.full(shape, 0.7, dtype=np.float64),
        intent_long_ta=np.ones(shape, dtype=bool),
        intent_short_ta=np.zeros(shape, dtype=bool),
        target_qty_ta=np.zeros(shape, dtype=np.float64),
    )


def _dummy_risk_result_negative_sign() -> SimpleNamespace:
    return SimpleNamespace(
        equity_curve=np.asarray([1_000.0, 1_000.0, 970.0, 960.0], dtype=np.float64),
        filled_qty_ta=np.zeros((4, 1), dtype=np.float64),
        exec_price_ta=np.full((4, 1), np.nan, dtype=np.float64),
        trade_cost_ta=np.zeros((4, 1), dtype=np.float64),
        position_qty_ta=np.asarray([[0.0], [1.0], [1.0], [1.0]], dtype=np.float64),
        margin_used_t=np.asarray([0.0, 100.0, 70.0, 60.0], dtype=np.float64),
        buying_power_t=np.asarray([1_000.0, 900.0, 900.0, 900.0], dtype=np.float64),
        daily_loss_t=np.asarray([0.0, 0.0, -1.0, 2.0], dtype=np.float64),
        trades=0,
        final_equity=960.0,
        max_drawdown=0.04,
        sharpe=0.0,
        sortino=0.0,
    )


def _dummy_risk_result_session_gap() -> SimpleNamespace:
    return SimpleNamespace(
        equity_curve=np.asarray([1_000.0, 1_000.0, 970.0, 960.0], dtype=np.float64),
        filled_qty_ta=np.zeros((4, 1), dtype=np.float64),
        exec_price_ta=np.full((4, 1), np.nan, dtype=np.float64),
        trade_cost_ta=np.zeros((4, 1), dtype=np.float64),
        position_qty_ta=np.asarray([[0.0], [1.0], [1.0], [1.0]], dtype=np.float64),
        margin_used_t=np.asarray([0.0, 100.0, 70.0, 60.0], dtype=np.float64),
        buying_power_t=np.asarray([1_000.0, 900.0, 900.0, 900.0], dtype=np.float64),
        daily_loss_t=np.asarray([0.0, 0.0, 30.0, 40.0], dtype=np.float64),
        trades=0,
        final_equity=960.0,
        max_drawdown=0.04,
        sharpe=0.0,
        sortino=0.0,
    )


def test_materialize_risk_outputs_recomputes_session_daily_loss_and_breach_flag() -> None:
    state = _build_state(a_count=1, daily_loss_limit_abs=15.0)
    view = materialize_risk_outputs_into_state(
        state=state,
        m4_sig=_dummy_m4_signal(t_count=4, a_count=1),
        risk_res=_dummy_risk_result_session_gap(),
        execution_view_cls=SimpleNamespace,
    )

    np.testing.assert_allclose(state.daily_loss, np.asarray([0.0, 0.0, 10.0, 20.0], dtype=np.float64), rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(state.daily_loss_breach_flag, np.asarray([0, 0, 0, 1], dtype=np.int8))
    np.testing.assert_array_equal(view.filled_qty_ta, np.zeros((4, 1), dtype=np.float64))


def test_materialize_risk_outputs_fails_closed_on_negative_daily_loss_sign() -> None:
    state = _build_state(a_count=1, daily_loss_limit_abs=15.0)
    with pytest.raises(RuntimeError, match="AMBIGUOUS_DAILY_LOSS_CONTRACT_NEGATIVE"):
        materialize_risk_outputs_into_state(
            state=state,
            m4_sig=_dummy_m4_signal(t_count=4, a_count=1),
            risk_res=_dummy_risk_result_negative_sign(),
            execution_view_cls=SimpleNamespace,
        )


def _build_compact_execution_view() -> tuple[CombinedStateView, SimpleNamespace, SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    base_state = _build_state(a_count=2, daily_loss_limit_abs=25.0)
    base_state.phase[:] = np.asarray(
        [np.int8(Phase.LIVE), np.int8(Phase.LIVE), np.int8(Phase.OVERNIGHT_SELECT), np.int8(Phase.LIVE)],
        dtype=np.int8,
    )
    base = BaseTensorState.from_tensor_state(base_state)
    market = MarketOverlay.from_base(base)
    feature = FeatureOverlay.allocate(base)
    scratch = CandidateScratch.allocate(base, "compact")
    scratch.order_side[:, :] = np.asarray([[0, 0], [1, 0], [0, 0], [-1, 0]], dtype=np.int8)
    scratch.order_flags[:, :] = np.asarray([[0, 0], [1, 0], [0, 0], [2, 0]], dtype=np.uint16)
    scratch.position_qty[:, :] = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
        dtype=np.float64,
    )
    scratch.equity[:] = np.asarray([1_000.0, 1_005.0, 995.0, 990.0], dtype=np.float64)
    scratch.margin_used[:] = np.asarray([0.0, 100.0, 80.0, 0.0], dtype=np.float64)
    scratch.buying_power[:] = np.asarray([1_000.0, 905.0, 915.0, 990.0], dtype=np.float64)
    scratch.daily_loss[:] = np.asarray([0.0, 0.0, 5.0, 10.0], dtype=np.float64)
    scratch.daily_loss_breach_flag[:] = np.asarray([0, 0, 0, 0], dtype=np.int8)

    feature.rvol[:, :] = 1.2
    feature.atr_floor[:, :] = 0.5
    feature.vp[:, :, :] = 1.0
    feature.vp_delta[:, :, :] = 0.1
    feature.profile_stats[:, :, :] = 0.0
    feature.profile_stats[:, :, int(ProfileStatIdx.DCLIP)] = 1.0
    feature.profile_stats[:, :, int(ProfileStatIdx.Z_DELTA)] = 1.1
    feature.profile_stats[:, :, int(ProfileStatIdx.GBREAK)] = 1.0
    feature.profile_stats[:, :, int(ProfileStatIdx.GREJECT)] = 1.0
    feature.scores[:, :, :] = 0.0
    feature.scores[:, :, int(ScoreIdx.SCORE_BO_LONG)] = 0.8
    feature.scores[:, :, int(ScoreIdx.SCORE_BO_SHORT)] = 0.2
    feature.scores[:, :, int(ScoreIdx.SCORE_REJ_LONG)] = 0.7
    feature.scores[:, :, int(ScoreIdx.SCORE_REJ_SHORT)] = 0.1

    view = CombinedStateView(base, market, feature, scratch, asset_enabled_mask=np.asarray([True, True], dtype=bool))

    context = np.zeros((base.cfg.T, base.cfg.A, int(ContextIdx.N_FIELDS)), dtype=np.float64)
    context[:, :, int(ContextIdx.CTX_X_POC)] = 0.1
    context[:, :, int(ContextIdx.CTX_X_VAH)] = 0.2
    context[:, :, int(ContextIdx.CTX_X_VAL)] = -0.2
    context[:, :, int(ContextIdx.CTX_TREND_GATE_SPREAD_MEAN)] = 0.3
    context[:, :, int(ContextIdx.CTX_POC_DRIFT_X)] = 0.4
    context[:, :, int(ContextIdx.CTX_POC_VS_PREV_VA)] = 0.5
    context[:, :, int(ContextIdx.CTX_IB_HIGH_X)] = 0.6
    context[:, :, int(ContextIdx.CTX_IB_LOW_X)] = -0.6
    m3 = SimpleNamespace(
        context_tac=context,
        context_valid_ta=np.ones((base.cfg.T, base.cfg.A), dtype=np.int8),
        context_source_t_index_ta=np.broadcast_to(np.arange(base.cfg.T, dtype=np.int64)[:, None], (base.cfg.T, base.cfg.A)).copy(),
        context_valid_atw=None,
        context_source_index_atw=None,
        block_end_flag_t=np.asarray([0, 1, 0, 1], dtype=bool),
        block_valid_ta=np.ones((base.cfg.T, base.cfg.A), dtype=bool),
        block_seq_t=np.asarray([0, 1, 1, 2], dtype=np.int16),
    )
    m4_out = SimpleNamespace(
        regime_primary_ta=np.full((base.cfg.T, base.cfg.A), np.int8(RegimeIdx.TREND), dtype=np.int8),
        regime_confidence_ta=np.full((base.cfg.T, base.cfg.A), 0.8, dtype=np.float64),
        intent_long_ta=np.ones((base.cfg.T, base.cfg.A), dtype=np.int8),
        intent_short_ta=np.zeros((base.cfg.T, base.cfg.A), dtype=np.int8),
        target_qty_ta=np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
            dtype=np.float64,
        ),
        filled_qty_ta=np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [-1.0, 0.0]],
            dtype=np.float64,
        ),
        exec_price_ta=np.asarray(
            [[np.nan, np.nan], [100.0, np.nan], [np.nan, np.nan], [60.0, np.nan]],
            dtype=np.float64,
        ),
        trade_cost_ta=np.asarray(
            [[0.0, 0.0], [0.2, 0.0], [0.0, 0.0], [0.2, 0.0]],
            dtype=np.float64,
        ),
        overnight_score_ta=np.full((base.cfg.T, base.cfg.A), 0.6, dtype=np.float64),
        overnight_winner_t=np.asarray([-1, -1, 0, -1], dtype=np.int16),
    )
    split = SimpleNamespace(test_idx=np.arange(base.cfg.T, dtype=np.int64))
    cfg = SimpleNamespace(
        export_micro_diagnostics=True,
        micro_diag_mode="events_only",
        micro_diag_symbols=(),
        micro_diag_session_ids=(),
        micro_diag_trade_window_pre=0,
        micro_diag_trade_window_post=0,
        micro_diag_export_block_profiles=True,
        micro_diag_export_funnel=True,
        micro_diag_max_rows=10_000,
    )
    return view, m3, m4_out, split, cfg


def test_compact_artifact_builders_enforce_declared_dependencies_only() -> None:
    view, m3, m4_out, split, cfg = _build_compact_execution_view()

    assert view.candidate_scratch is not None
    assert view.candidate_scratch.available_cash is None
    assert view.candidate_scratch.realized_pnl is None
    assert view.candidate_scratch.unrealized_pnl is None

    eq = equity_curve_payload(view, "cand", "wf_000", "baseline")
    trade = trade_log_payload(view, m4_out, "cand", "wf_000", "baseline")
    micro = collect_micro_diagnostics_payload(
        state=view,
        m3=m3,
        m4_out=m4_out,
        candidate_id="cand",
        split_id="wf_000",
        scenario_id="baseline",
        split=split,
        enabled_assets_mask=np.asarray([True, True], dtype=bool),
        cfg=cfg,
    )
    profile = collect_micro_profile_blocks_payload(
        state=view,
        m3=m3,
        candidate_id="cand",
        split_id="wf_000",
        scenario_id="baseline",
        enabled_assets_mask=np.asarray([True, True], dtype=bool),
        cfg=cfg,
    )
    funnel = collect_funnel_payload(
        state=view,
        m4_out=m4_out,
        candidate_id="cand",
        split_id="wf_000",
        scenario_id="baseline",
        enabled_assets_mask=np.asarray([True, True], dtype=bool),
        cfg=cfg,
        require_pandas_fn=lambda: pd,
    )

    assert eq["daily_loss"].shape == (view.cfg.T,)
    assert trade["filled_qty"].shape[0] == 2
    assert micro is not None and micro["position_qty"].shape[0] > 0
    assert profile is not None and profile["vp_block_blob"].shape[0] == 4
    assert funnel is not None and funnel["is_winner"].shape[0] == 2


def test_benchmark_daily_returns_skips_all_invalid_basket_rows_without_warning() -> None:
    state = _build_state(a_count=2)
    state.bar_valid[1, :] = False
    state.close_px[1, :] = np.nan

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        sess, ret = benchmark_daily_returns(
            state,
            benchmark_symbol="MISSING",
            session_bounds_fn=lambda sid: (
                np.asarray([0, 2], dtype=np.int64),
                np.asarray([2, 4], dtype=np.int64),
                np.asarray([0, 1], dtype=np.int64),
            ),
        )

    assert sess.tolist() == [0, 1]
    np.testing.assert_allclose(ret, np.array([0.0, -0.26], dtype=np.float64))
    assert not any("Mean of empty slice" in str(w.message) for w in caught)


def test_equity_curve_payload_avoids_zero_peak_division_warning() -> None:
    view, _m3, _m4_out, _split, _cfg = _build_compact_execution_view()
    view.equity[:] = 0.0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        eq = equity_curve_payload(view, "cand", "wf_000", "baseline")

    np.testing.assert_allclose(eq["drawdown"], np.zeros(view.cfg.T, dtype=np.float64))
    assert not any("divide by zero" in str(w.message) or "invalid value" in str(w.message) for w in caught)


def test_strategy_results_contract_fails_closed_when_required_field_missing() -> None:
    row = {
        "daily_returns_exec": np.zeros(2, dtype=np.float64),
        "daily_returns_raw": np.zeros(2, dtype=np.float64),
        "risk_engine_metrics": {},
        "trade_payload": {},
        "equity_payload": {},
        "quality_reason_codes": [],
    }
    with pytest.raises(RuntimeError, match="candidate_row.asset_pnl_by_symbol"):
        assert_artifact_dependency_contract("strategy_results", candidate_row=row)
