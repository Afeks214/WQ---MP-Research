from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import weightiz.module4.strategy_funnel_engine as funnel_engine
from weightiz.module3 import ContextIdx, StructIdx
from weightiz.module4.contracts import build_module4_input_contracts
from weightiz.module4.strategy_funnel_engine import Module4DecisionOutput, run_module4_funnel
from weightiz.module4.telemetry import DecisionReasonCode
from weightiz.module1.core import ProfileStatIdx, ScoreIdx
from weightiz.module4.strategy_funnel import Module4Config, Module4SignalOutput, run_module4_signal_funnel


def _sample_contracts(A: int = 2, T: int = 4, W: int = 2, *, with_risk: bool = False):
    alpha = np.zeros((A, T, 1), dtype=np.float64)
    score = np.zeros((A, T, int(ScoreIdx.N_FIELDS)), dtype=np.float64)
    profile = np.zeros((A, T, int(ProfileStatIdx.N_FIELDS)), dtype=np.float64)
    structure = np.zeros((A, T, int(StructIdx.N_FIELDS), W), dtype=np.float64)
    context = np.zeros((A, T, int(ContextIdx.N_FIELDS), W), dtype=np.float64)
    fp = np.zeros((A, T, 1, W), dtype=np.float64)
    regime = np.zeros((A, T, 1, W), dtype=np.float64)
    tradable = np.ones((A, T), dtype=bool)
    phase = np.zeros(T, dtype=np.int64)
    enabled = np.ones(A, dtype=bool)
    src = np.broadcast_to(np.arange(T, dtype=np.int64)[None, :], (A, T)).copy()

    structure[:, :, int(StructIdx.VALID_RATIO), :] = 1.0
    context[:, :, int(ContextIdx.CTX_VALID_RATIO), :] = 1.0
    context[:, :, int(ContextIdx.CTX_REGIME_PERSISTENCE), :] = 1.0

    score[0, 0, int(ScoreIdx.SCORE_BO_LONG)] = 0.9
    if T > 1:
        score[0, 1, int(ScoreIdx.SCORE_BO_SHORT)] = 0.9
    if T > 2:
        score[0, 2, int(ScoreIdx.SCORE_BO_LONG)] = 0.9
        if A > 1:
            score[1, 2, int(ScoreIdx.SCORE_BO_SHORT)] = 0.9

    structure[0, :, int(StructIdx.TREND_GATE_SPREAD_MEAN), :] = 0.2
    structure[0, :, int(StructIdx.POC_DRIFT_X), :] = 0.6
    context[0, :, int(ContextIdx.CTX_TREND_GATE_SPREAD_MEAN), :] = 0.2
    context[0, :, int(ContextIdx.CTX_POC_DRIFT_X), :] = 0.6
    context[0, :, int(ContextIdx.CTX_REGIME_CODE), :] = 1.0

    kwargs = {}
    if with_risk:
        kwargs["volatility_tensor"] = np.array([[0.1, 2.0, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]], dtype=np.float64)

    return build_module4_input_contracts(
        alpha_signal_tensor=alpha,
        score_tensor=score,
        profile_stat_tensor=profile,
        structure_tensor=structure,
        context_tensor=context,
        profile_fingerprint_tensor=fp,
        profile_regime_tensor=regime,
        tradable_mask=tradable,
        phase_code=phase,
        asset_enabled_mask=enabled,
        source_time_index_at=src,
        **kwargs,
    )


def test_orchestrator_output_shapes_and_telemetry_population() -> None:
    out = run_module4_funnel(_sample_contracts(), Module4Config())
    assert isinstance(out, Module4DecisionOutput)
    assert out.target_weight.shape == (2, 4)
    assert out.target_position.shape == (2, 4)
    assert out.target_delta_signal.shape == (2, 4)
    assert out.selected_window_idx.shape == (2, 4)
    assert out.telemetry.window_score.shape[:2] == (2, 4)
    assert out.telemetry.intent_gate_mask.shape == (2, 4, 6)
    assert out.telemetry.allocation_rank.shape == (2, 4)


def test_risk_filtered_cell_flattens_and_zeroes_weight() -> None:
    contracts = _sample_contracts(with_risk=True)
    out = run_module4_funnel(contracts, Module4Config(max_volatility=1.0))
    assert out.intent_flat[0, 1]
    assert out.target_weight[0, 1] == 0.0
    assert out.decision_valid_mask[0, 1]
    assert out.telemetry.decision_reason_code[0, 1] == np.int16(DecisionReasonCode.RISK_FILTER_BLOCK)


def test_target_position_and_delta_are_causal() -> None:
    out = run_module4_funnel(_sample_contracts(), Module4Config())
    np.testing.assert_array_equal(
        out.target_position[0],
        np.array([1.0, -1.0, 1.0, 0.0], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        out.target_delta_signal[0],
        np.array([1.0, -2.0, 2.0, -1.0], dtype=np.float64),
    )


def test_repeatability_is_deterministic() -> None:
    contracts = _sample_contracts()
    cfg = Module4Config()
    out1 = run_module4_funnel(contracts, cfg)
    out2 = run_module4_funnel(contracts, cfg)
    np.testing.assert_array_equal(out1.target_weight, out2.target_weight)
    np.testing.assert_array_equal(out1.telemetry.decision_reason_code, out2.telemetry.decision_reason_code)


def test_no_execution_metadata_leakage() -> None:
    names = set(Module4DecisionOutput.__dataclass_fields__.keys())
    forbidden = {"filled_qty", "exec_price", "trade_cost", "position_qty", "pnl", "orders"}
    assert names.isdisjoint(forbidden)


def test_low_confidence_flat_remains_decision_valid() -> None:
    contracts = _sample_contracts(A=1, T=2)
    out = run_module4_funnel(contracts, Module4Config(regime_confidence_min=0.95))
    assert np.all(out.intent_flat)
    assert np.all(out.decision_valid_mask)
    assert out.telemetry.decision_reason_code[0, 0] == np.int16(DecisionReasonCode.LOW_REGIME_CONFIDENCE)


def test_asset_disabled_cell_is_not_decision_valid() -> None:
    contracts = _sample_contracts(A=1, T=2)
    disabled = build_module4_input_contracts(
        alpha_signal_tensor=contracts.alpha_signal_tensor,
        score_tensor=contracts.score_tensor,
        profile_stat_tensor=contracts.profile_stat_tensor,
        structure_tensor=contracts.structure_tensor,
        context_tensor=contracts.context_tensor,
        profile_fingerprint_tensor=contracts.profile_fingerprint_tensor,
        profile_regime_tensor=contracts.profile_regime_tensor,
        tradable_mask=contracts.tradable_mask,
        phase_code=contracts.phase_code,
        asset_enabled_mask=np.array([False], dtype=bool),
        source_time_index_at=contracts.source_time_index_at,
    )
    out = run_module4_funnel(disabled, Module4Config())
    assert not np.any(out.decision_valid_mask)
    assert np.all(out.telemetry.decision_reason_code == np.int16(DecisionReasonCode.MASKED_NOT_TRADABLE))


def test_degraded_allowed_cells_remain_valid_when_outputs_are_usable() -> None:
    contracts = _sample_contracts(A=1, T=2)
    degraded = np.ones((1, 2), dtype=bool)
    out = run_module4_funnel(contracts, Module4Config(), degraded_mode_mask_at=degraded)
    assert np.all(out.telemetry.degraded_mode_mask)
    assert np.all(out.decision_valid_mask)
    assert np.all(out.telemetry.decision_reason_code != np.int16(DecisionReasonCode.DEGRADED_BRIDGE_RESTRICTED))


def test_non_finite_stage_output_invalidates_cell_when_fail_soft(monkeypatch: pytest.MonkeyPatch) -> None:
    contracts = _sample_contracts(A=1, T=2)
    original = funnel_engine.compute_normalized_signal_allocation

    def bad_allocation(**kwargs):
        out = original(**kwargs)
        return type(out)(
            allocation_score=np.array([[np.nan, 0.0]], dtype=np.float64),
            target_weight=out.target_weight,
            allocation_rank=out.allocation_rank,
            allocation_valid_mask=out.allocation_valid_mask,
        )

    monkeypatch.setattr(funnel_engine, "compute_normalized_signal_allocation", bad_allocation)
    out = run_module4_funnel(contracts, Module4Config(fail_on_non_finite_output=False))
    assert not out.decision_valid_mask[0, 0]
    assert out.target_weight[0, 0] == 0.0
    assert out.telemetry.decision_reason_code[0, 0] == np.int16(DecisionReasonCode.INVALID_INPUT)


def test_orchestration_order_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    import weightiz.module4.strategy_funnel_engine as eng

    calls: list[str] = []
    contracts = _sample_contracts(A=1, T=1)
    cfg = Module4Config()

    def wrap(name, ret):
        def _f(*args, **kwargs):
            calls.append(name)
            return ret
        return _f

    adapted = SimpleNamespace(
        structure_adapted=np.zeros((1, 1, int(StructIdx.N_FIELDS)), dtype=np.float64),
        context_adapted=np.zeros((1, 1, int(ContextIdx.N_FIELDS)), dtype=np.float64),
        fingerprint_adapted=np.zeros((1, 1, 1), dtype=np.float64),
        regime_hint=np.zeros((1, 1, 1), dtype=np.float64),
        selected_window_idx=np.zeros((1, 1), dtype=np.int16),
        window_score=np.zeros((1, 1, 2), dtype=np.float64),
        regime_confidence_window=np.zeros((1, 1, 2), dtype=np.float64),
    )
    regime = SimpleNamespace(
        regime_id=np.ones((1, 1), dtype=np.int8),
        regime_confidence=np.ones((1, 1), dtype=np.float64),
        regime_valid_mask=np.ones((1, 1), dtype=bool),
        low_regime_confidence_mask=np.zeros((1, 1), dtype=bool),
        regime_score=np.ones((1, 1, 6), dtype=np.float64),
    )
    intent = SimpleNamespace(
        intent_long=np.zeros((1, 1), dtype=bool),
        intent_short=np.zeros((1, 1), dtype=bool),
        intent_flat=np.ones((1, 1), dtype=bool),
        intent_valid_mask=np.ones((1, 1), dtype=bool),
        intent_gate_mask=np.ones((1, 1, 6), dtype=bool),
        signed_intent_utility=np.zeros((1, 1), dtype=np.float64),
    )
    conviction = SimpleNamespace(
        conviction_long=np.zeros((1, 1), dtype=np.float64),
        conviction_short=np.zeros((1, 1), dtype=np.float64),
        conviction_net=np.zeros((1, 1), dtype=np.float64),
        conviction_valid_mask=np.ones((1, 1), dtype=bool),
    )
    allocation = SimpleNamespace(
        allocation_score=np.zeros((1, 1), dtype=np.float64),
        target_weight=np.zeros((1, 1), dtype=np.float64),
        allocation_rank=np.zeros((1, 1), dtype=np.int16),
        allocation_valid_mask=np.ones((1, 1), dtype=bool),
    )

    monkeypatch.setattr(eng, "apply_optional_risk_filters", wrap("risk", contracts.tradable_mask))
    monkeypatch.setattr(eng, "adapt_windows", wrap("window", adapted))
    monkeypatch.setattr(eng, "classify_regime", wrap("regime", regime))
    monkeypatch.setattr(eng, "generate_strategy_intent", wrap("intent", intent))
    monkeypatch.setattr(eng, "compute_conviction", wrap("conviction", conviction))
    monkeypatch.setattr(eng, "compute_normalized_signal_allocation", wrap("allocation", allocation))

    eng.run_module4_funnel(contracts, cfg)
    assert calls == ["risk", "window", "regime", "intent", "conviction", "allocation"]


def test_legacy_bridge_returns_frozen_signal_output() -> None:
    T, A, W = 4, 2, 2
    state = SimpleNamespace(
        cfg=SimpleNamespace(T=T, A=A),
        scores=np.zeros((T, A, int(ScoreIdx.N_FIELDS)), dtype=np.float64),
        profile_stats=np.zeros((T, A, int(ProfileStatIdx.N_FIELDS)), dtype=np.float64),
        bar_valid=np.ones((T, A), dtype=bool),
        phase=np.zeros(T, dtype=np.int64),
    )
    state.scores[0, 0, int(ScoreIdx.SCORE_BO_LONG)] = 0.9
    structure = np.zeros((A, T, int(StructIdx.N_FIELDS), W), dtype=np.float64)
    context = np.zeros((A, T, int(ContextIdx.N_FIELDS), W), dtype=np.float64)
    structure[:, :, int(StructIdx.VALID_RATIO), :] = 1.0
    context[:, :, int(ContextIdx.CTX_VALID_RATIO), :] = 1.0
    m3 = SimpleNamespace(
        structure_tensor=structure,
        context_tensor=context,
        profile_fingerprint_tensor=np.zeros((A, T, 1, W), dtype=np.float64),
        profile_regime_tensor=np.zeros((A, T, 1, W), dtype=np.float64),
        context_valid_ta=np.ones((T, A), dtype=bool),
        context_source_index_atw=np.broadcast_to(np.arange(T, dtype=np.int64)[None, :, None], (A, T, W)).copy(),
    )

    out = run_module4_signal_funnel(state, m3, Module4Config())
    assert isinstance(out, Module4SignalOutput)
    assert out.regime_primary_ta.shape == (T, A)
    assert out.target_qty_ta.shape == (T, A)
    assert out.regime_primary_ta.dtype == np.int8
    assert out.target_qty_ta.dtype == np.float64


def test_bridge_fails_on_forward_looking_source_index() -> None:
    T, A, W = 3, 1, 1
    state = SimpleNamespace(
        cfg=SimpleNamespace(T=T, A=A),
        scores=np.zeros((T, A, int(ScoreIdx.N_FIELDS)), dtype=np.float64),
        profile_stats=np.zeros((T, A, int(ProfileStatIdx.N_FIELDS)), dtype=np.float64),
        bar_valid=np.ones((T, A), dtype=bool),
        phase=np.zeros(T, dtype=np.int64),
    )
    structure = np.zeros((A, T, int(StructIdx.N_FIELDS), W), dtype=np.float64)
    context = np.zeros((A, T, int(ContextIdx.N_FIELDS), W), dtype=np.float64)
    structure[:, :, int(StructIdx.VALID_RATIO), :] = 1.0
    context[:, :, int(ContextIdx.CTX_VALID_RATIO), :] = 1.0
    bad = np.broadcast_to(np.arange(T, dtype=np.int64)[None, :, None], (A, T, W)).copy()
    bad[0, 0, 0] = 2
    m3 = SimpleNamespace(
        structure_tensor=structure,
        context_tensor=context,
        profile_fingerprint_tensor=np.zeros((A, T, 1, W), dtype=np.float64),
        profile_regime_tensor=np.zeros((A, T, 1, W), dtype=np.float64),
        context_valid_ta=np.ones((T, A), dtype=bool),
        context_source_index_atw=bad,
    )
    with pytest.raises(RuntimeError, match="CAUSALITY_VIOLATION"):
        run_module4_signal_funnel(state, m3, Module4Config())
