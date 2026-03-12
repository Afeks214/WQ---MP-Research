from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .allocation_engine import AllocationResult, compute_normalized_signal_allocation
from .contracts import Module4InputContracts, RiskFilterConfig, apply_optional_risk_filters, assert_causal_source_index
from .conviction_engine import ConvictionResult, compute_conviction
from .regime_classifier import RegimeClassificationResult, classify_regime
from .strategy_intent_engine import StrategyIntentResult, generate_strategy_intent
from .telemetry import DecisionReasonCode, Module4Telemetry
from .window_adapter import WindowAdapterConfig, WindowAdapterOutput, adapt_windows


@dataclass(frozen=True)
class Module4DecisionOutput:
    intent_long: np.ndarray
    intent_short: np.ndarray
    intent_flat: np.ndarray
    regime_id: np.ndarray
    regime_confidence: np.ndarray
    conviction_long: np.ndarray
    conviction_short: np.ndarray
    conviction_net: np.ndarray
    allocation_score: np.ndarray
    target_weight: np.ndarray
    target_position: np.ndarray
    target_delta_signal: np.ndarray
    selected_window_idx: np.ndarray
    decision_valid_mask: np.ndarray
    telemetry: Module4Telemetry

    def __post_init__(self) -> None:
        intent_long = np.asarray(self.intent_long)
        intent_short = np.asarray(self.intent_short)
        intent_flat = np.asarray(self.intent_flat)
        regime_id = np.asarray(self.regime_id)
        regime_confidence = np.asarray(self.regime_confidence)
        conviction_long = np.asarray(self.conviction_long)
        conviction_short = np.asarray(self.conviction_short)
        conviction_net = np.asarray(self.conviction_net)
        allocation_score = np.asarray(self.allocation_score)
        target_weight = np.asarray(self.target_weight)
        target_position = np.asarray(self.target_position)
        target_delta_signal = np.asarray(self.target_delta_signal)
        selected_window_idx = np.asarray(self.selected_window_idx)
        decision_valid_mask = np.asarray(self.decision_valid_mask)

        if intent_long.ndim != 2:
            raise RuntimeError(f"intent_long must be [A,T], got shape={intent_long.shape}")
        A, T = intent_long.shape
        checks = [
            ("intent_short", intent_short, np.bool_),
            ("intent_flat", intent_flat, np.bool_),
            ("regime_id", regime_id, np.int8),
            ("regime_confidence", regime_confidence, np.float64),
            ("conviction_long", conviction_long, np.float64),
            ("conviction_short", conviction_short, np.float64),
            ("conviction_net", conviction_net, np.float64),
            ("allocation_score", allocation_score, np.float64),
            ("target_weight", target_weight, np.float64),
            ("target_position", target_position, np.float64),
            ("target_delta_signal", target_delta_signal, np.float64),
            ("selected_window_idx", selected_window_idx, np.int16),
            ("decision_valid_mask", decision_valid_mask, np.bool_),
        ]
        if intent_long.dtype != np.bool_:
            raise RuntimeError(f"intent_long dtype must be bool, got {intent_long.dtype}")
        for name, arr, dtype in checks:
            if arr.shape != (A, T):
                raise RuntimeError(f"{name} must be [A,T], got shape={arr.shape}")
            if arr.dtype != dtype:
                raise RuntimeError(f"{name} dtype must be {dtype}, got {arr.dtype}")


def _sanitize_numeric(name: str, arr: np.ndarray, *, fail_on_non_finite_output: bool) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(arr)
    finite = np.isfinite(x)
    if np.all(finite):
        return x, np.ones(x.shape[:2], dtype=bool) if x.ndim == 3 else np.ones(x.shape, dtype=bool)
    if fail_on_non_finite_output:
        bad = np.argwhere(~finite)[:8]
        raise RuntimeError(f"{name} contains non-finite outputs at indices {bad.tolist()}")
    out = np.where(finite, x, 0.0)
    valid = np.all(finite, axis=2) if x.ndim == 3 else finite
    return out, valid


def _target_position_from_weight(target_weight: np.ndarray) -> np.ndarray:
    out = np.zeros(target_weight.shape, dtype=np.float64)
    out[target_weight > 0.0] = 1.0
    out[target_weight < 0.0] = -1.0
    return out


def _target_delta_signal(target_position: np.ndarray) -> np.ndarray:
    delta = np.zeros(target_position.shape, dtype=np.float64)
    if target_position.shape[1] == 0:
        return delta
    delta[:, 0] = target_position[:, 0]
    if target_position.shape[1] > 1:
        delta[:, 1:] = target_position[:, 1:] - target_position[:, :-1]
    return delta


def _build_reason_codes(
    *,
    contracts: Module4InputContracts,
    tradable_after_risk: np.ndarray,
    degraded_mode_mask: np.ndarray,
    low_regime_confidence_mask: np.ndarray,
    intent_long: np.ndarray,
    intent_short: np.ndarray,
    intent_flat: np.ndarray,
    conviction_net: np.ndarray,
    allocation_score: np.ndarray,
    decision_valid_mask: np.ndarray,
) -> np.ndarray:
    A, T = tradable_after_risk.shape
    reason = np.full((A, T), np.int16(DecisionReasonCode.INTENT_FLAT), dtype=np.int16)

    base_tradable = np.asarray(contracts.tradable_mask, dtype=bool)
    asset_enabled = np.broadcast_to(
        np.asarray(contracts.asset_enabled_mask, dtype=bool)[:, None],
        (A, T),
    )
    risk_block = base_tradable & (~tradable_after_risk)
    zero_score = tradable_after_risk & np.isclose(allocation_score, 0.0, rtol=0.0, atol=1e-12)
    zero_conv = tradable_after_risk & np.isclose(conviction_net, 0.0, rtol=0.0, atol=1e-12)

    reason[intent_short] = np.int16(DecisionReasonCode.INTENT_SHORT)
    reason[intent_long] = np.int16(DecisionReasonCode.INTENT_LONG)
    reason[intent_flat] = np.int16(DecisionReasonCode.INTENT_FLAT)
    reason[zero_conv] = np.int16(DecisionReasonCode.ZERO_CONVICTION)
    reason[zero_score] = np.int16(DecisionReasonCode.ZERO_SCORE_AFTER_MASK)
    reason[low_regime_confidence_mask] = np.int16(DecisionReasonCode.LOW_REGIME_CONFIDENCE)
    reason[risk_block] = np.int16(DecisionReasonCode.RISK_FILTER_BLOCK)
    reason[~asset_enabled] = np.int16(DecisionReasonCode.MASKED_NOT_TRADABLE)
    reason[~base_tradable] = np.int16(DecisionReasonCode.MASKED_NOT_TRADABLE)
    reason[degraded_mode_mask & (~decision_valid_mask)] = np.int16(DecisionReasonCode.DEGRADED_BRIDGE_RESTRICTED)
    return reason


def run_module4_funnel(
    contracts: Module4InputContracts,
    cfg4: object,
    *,
    degraded_mode_mask_at: np.ndarray | None = None,
) -> Module4DecisionOutput:
    if not isinstance(contracts, Module4InputContracts):
        raise RuntimeError("run_module4_funnel expects Module4InputContracts")

    A = int(contracts.asset_enabled_mask.shape[0])
    T = int(contracts.phase_code.shape[0])
    if contracts.source_time_index_at is not None and bool(getattr(cfg4, "enforce_causal_source_validation", True)):
        assert_causal_source_index(contracts.source_time_index_at)

    tradable_after_risk = apply_optional_risk_filters(
        tradable_mask_at=contracts.tradable_mask,
        volatility_tensor_at=contracts.volatility_tensor,
        spread_tensor_at=contracts.spread_tensor,
        liquidity_score_at=contracts.liquidity_score,
        cfg=RiskFilterConfig(
            max_volatility=float(getattr(cfg4, "max_volatility", np.inf)),
            max_spread=float(getattr(cfg4, "max_spread", np.inf)),
            min_liquidity=float(getattr(cfg4, "min_liquidity", 0.0)),
        ),
    )

    adapted = adapt_windows(
        contracts,
        WindowAdapterConfig(
            mode=str(getattr(cfg4, "window_selection_mode", "multi_window")),
            fixed_window_index=int(getattr(cfg4, "fixed_window_index", 0)),
            anchor_window_index=int(getattr(cfg4, "anchor_window_index", 0)),
            epsilon=float(getattr(cfg4, "eps", 1e-12)),
            enforce_causal_checks=bool(getattr(cfg4, "enforce_window_causal_sanity", True)),
        ),
    )

    regime = classify_regime(
        structure_adapted=adapted.structure_adapted,
        context_adapted=adapted.context_adapted,
        regime_hint=adapted.regime_hint,
        tradable_mask=tradable_after_risk,
        cfg4=cfg4,
    )
    regime_confidence, regime_finite = _sanitize_numeric(
        "regime_confidence", regime.regime_confidence, fail_on_non_finite_output=bool(getattr(cfg4, "fail_on_non_finite_output", True))
    )
    regime_score, regime_score_finite = _sanitize_numeric(
        "regime_score", regime.regime_score, fail_on_non_finite_output=bool(getattr(cfg4, "fail_on_non_finite_output", True))
    )

    intent = generate_strategy_intent(
        alpha_signal_tensor=contracts.alpha_signal_tensor,
        score_tensor=contracts.score_tensor,
        profile_stat_tensor=contracts.profile_stat_tensor,
        regime_id=regime.regime_id,
        regime_confidence=regime_confidence,
        tradable_mask=tradable_after_risk,
        cfg4=cfg4,
    )
    signed_intent_utility, intent_utility_finite = _sanitize_numeric(
        "signed_intent_utility", intent.signed_intent_utility, fail_on_non_finite_output=bool(getattr(cfg4, "fail_on_non_finite_output", True))
    )

    conviction = compute_conviction(
        intent_long=intent.intent_long,
        intent_short=intent.intent_short,
        intent_flat=intent.intent_flat,
        signed_intent_utility=signed_intent_utility,
        regime_confidence=regime_confidence,
        score_tensor=contracts.score_tensor,
        context_adapted=adapted.context_adapted,
        tradable_mask=tradable_after_risk,
        cfg4=cfg4,
    )
    conviction_long, conviction_long_finite = _sanitize_numeric(
        "conviction_long", conviction.conviction_long, fail_on_non_finite_output=bool(getattr(cfg4, "fail_on_non_finite_output", True))
    )
    conviction_short, conviction_short_finite = _sanitize_numeric(
        "conviction_short", conviction.conviction_short, fail_on_non_finite_output=bool(getattr(cfg4, "fail_on_non_finite_output", True))
    )
    conviction_net, conviction_net_finite = _sanitize_numeric(
        "conviction_net", conviction.conviction_net, fail_on_non_finite_output=bool(getattr(cfg4, "fail_on_non_finite_output", True))
    )

    allocation = compute_normalized_signal_allocation(
        conviction_net=conviction_net,
        regime_confidence=regime_confidence,
        tradable_mask=tradable_after_risk,
        asset_enabled_mask=contracts.asset_enabled_mask,
        cfg4=cfg4,
    )
    allocation_score, allocation_score_finite = _sanitize_numeric(
        "allocation_score", allocation.allocation_score, fail_on_non_finite_output=bool(getattr(cfg4, "fail_on_non_finite_output", True))
    )
    target_weight, target_weight_finite = _sanitize_numeric(
        "target_weight", allocation.target_weight, fail_on_non_finite_output=bool(getattr(cfg4, "fail_on_non_finite_output", True))
    )

    asset_enabled_mask = np.broadcast_to(
        np.asarray(contracts.asset_enabled_mask, dtype=bool)[:, None],
        (A, T),
    )
    low_conf_flat_mask = np.asarray(regime.low_regime_confidence_mask, dtype=bool)
    if np.any(low_conf_flat_mask):
        conviction_long = np.asarray(conviction_long, dtype=np.float64).copy()
        conviction_short = np.asarray(conviction_short, dtype=np.float64).copy()
        conviction_net = np.asarray(conviction_net, dtype=np.float64).copy()
        allocation_score = np.asarray(allocation_score, dtype=np.float64).copy()
        target_weight = np.asarray(target_weight, dtype=np.float64).copy()
        conviction_long[low_conf_flat_mask] = 0.0
        conviction_short[low_conf_flat_mask] = 0.0
        conviction_net[low_conf_flat_mask] = 0.0
        allocation_score[low_conf_flat_mask] = 0.0
        target_weight[low_conf_flat_mask] = 0.0

    target_position = _target_position_from_weight(target_weight)
    target_delta_signal = _target_delta_signal(target_position)

    degraded_mode_mask = np.zeros((A, T), dtype=bool)
    if degraded_mode_mask_at is not None:
        degraded_mode_mask = np.asarray(degraded_mode_mask_at, dtype=bool)
        if degraded_mode_mask.shape != (A, T):
            raise RuntimeError(
                f"degraded_mode_mask_at shape mismatch: got {degraded_mode_mask.shape}, expected {(A, T)}"
            )

    signal_usable_mask = (
        regime_finite
        & regime_score_finite
        & intent.intent_valid_mask
        & intent_utility_finite
        & conviction.conviction_valid_mask
        & conviction_long_finite
        & conviction_short_finite
        & conviction_net_finite
        & allocation_score_finite
        & target_weight_finite
    )
    decision_valid_mask = signal_usable_mask & asset_enabled_mask

    invalid_cells = ~decision_valid_mask
    if np.any(invalid_cells):
        regime_id = np.asarray(regime.regime_id, dtype=np.int8).copy()
        regime_confidence = np.asarray(regime_confidence, dtype=np.float64).copy()
        intent_long = np.asarray(intent.intent_long, dtype=bool).copy()
        intent_short = np.asarray(intent.intent_short, dtype=bool).copy()
        intent_flat = np.asarray(intent.intent_flat, dtype=bool).copy()
        conviction_long = np.asarray(conviction_long, dtype=np.float64).copy()
        conviction_short = np.asarray(conviction_short, dtype=np.float64).copy()
        conviction_net = np.asarray(conviction_net, dtype=np.float64).copy()
        allocation_score = np.asarray(allocation_score, dtype=np.float64).copy()
        target_weight = np.asarray(target_weight, dtype=np.float64).copy()
        target_position = np.asarray(target_position, dtype=np.float64).copy()
        target_delta_signal = np.asarray(target_delta_signal, dtype=np.float64).copy()

        regime_id[invalid_cells] = np.int8(0)
        regime_confidence[invalid_cells] = 0.0
        intent_long[invalid_cells] = False
        intent_short[invalid_cells] = False
        intent_flat[invalid_cells] = True
        conviction_long[invalid_cells] = 0.0
        conviction_short[invalid_cells] = 0.0
        conviction_net[invalid_cells] = 0.0
        allocation_score[invalid_cells] = 0.0
        target_weight[invalid_cells] = 0.0
        target_position = _target_position_from_weight(target_weight)
        target_delta_signal = _target_delta_signal(target_position)
    else:
        regime_id = np.asarray(regime.regime_id, dtype=np.int8)
        intent_long = np.asarray(intent.intent_long, dtype=bool)
        intent_short = np.asarray(intent.intent_short, dtype=bool)
        intent_flat = np.asarray(intent.intent_flat, dtype=bool)

    if np.any(low_conf_flat_mask):
        intent_long = np.asarray(intent_long, dtype=bool).copy()
        intent_short = np.asarray(intent_short, dtype=bool).copy()
        intent_flat = np.asarray(intent_flat, dtype=bool).copy()
        intent_long[low_conf_flat_mask] = False
        intent_short[low_conf_flat_mask] = False
        intent_flat[low_conf_flat_mask] = True

    reason = _build_reason_codes(
        contracts=contracts,
        tradable_after_risk=tradable_after_risk,
        degraded_mode_mask=degraded_mode_mask,
        low_regime_confidence_mask=regime.low_regime_confidence_mask,
        intent_long=intent.intent_long,
        intent_short=intent.intent_short,
        intent_flat=intent.intent_flat,
        conviction_net=conviction_net,
        allocation_score=allocation_score,
        decision_valid_mask=decision_valid_mask,
    )
    invalid_runtime = (
        invalid_cells
        & np.asarray(contracts.asset_enabled_mask, dtype=bool)[:, None]
        & np.asarray(contracts.tradable_mask, dtype=bool)
        & (~degraded_mode_mask)
    )
    reason[invalid_runtime] = np.int16(DecisionReasonCode.INVALID_INPUT)

    telemetry = Module4Telemetry(
        decision_reason_code=np.ascontiguousarray(reason, dtype=np.int16),
        window_score=np.ascontiguousarray(adapted.window_score, dtype=np.float64),
        intent_gate_mask=np.ascontiguousarray(intent.intent_gate_mask, dtype=bool),
        allocation_rank=np.ascontiguousarray(allocation.allocation_rank, dtype=np.int16),
        regime_score=np.ascontiguousarray(regime_score, dtype=np.float64),
        decision_valid_mask=np.ascontiguousarray(decision_valid_mask, dtype=bool),
        degraded_mode_mask=np.ascontiguousarray(degraded_mode_mask, dtype=bool),
    )

    return Module4DecisionOutput(
        intent_long=np.ascontiguousarray(intent_long, dtype=bool),
        intent_short=np.ascontiguousarray(intent_short, dtype=bool),
        intent_flat=np.ascontiguousarray(intent_flat, dtype=bool),
        regime_id=np.ascontiguousarray(regime_id, dtype=np.int8),
        regime_confidence=np.ascontiguousarray(regime_confidence, dtype=np.float64),
        conviction_long=np.ascontiguousarray(conviction_long, dtype=np.float64),
        conviction_short=np.ascontiguousarray(conviction_short, dtype=np.float64),
        conviction_net=np.ascontiguousarray(conviction_net, dtype=np.float64),
        allocation_score=np.ascontiguousarray(allocation_score, dtype=np.float64),
        target_weight=np.ascontiguousarray(target_weight, dtype=np.float64),
        target_position=np.ascontiguousarray(target_position, dtype=np.float64),
        target_delta_signal=np.ascontiguousarray(target_delta_signal, dtype=np.float64),
        selected_window_idx=np.ascontiguousarray(adapted.selected_window_idx, dtype=np.int16),
        decision_valid_mask=np.ascontiguousarray(decision_valid_mask, dtype=bool),
        telemetry=telemetry,
    )
