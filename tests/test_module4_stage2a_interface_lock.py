from __future__ import annotations

import inspect
from dataclasses import fields

from weightiz.module3 import ContextIdx, StructIdx
from weightiz.module4.allocation_engine import AllocationResult, compute_normalized_signal_allocation
from weightiz.module4.conviction_engine import ConvictionResult, compute_conviction
from weightiz.module4.regime_classifier import RegimeClassificationResult, classify_regime
from weightiz.module4.strategy_intent_engine import StrategyIntentResult, generate_strategy_intent
from weightiz.module4.strategy_funnel import Module4SignalOutput


def test_struct_and_context_indices_are_locked() -> None:
    assert int(StructIdx.VALID_RATIO) == 0
    assert int(StructIdx.SKEW_ANCHOR) == 8
    assert int(StructIdx.TREND_GATE_SPREAD_MEAN) == 21
    assert int(StructIdx.POC_DRIFT_X) == 22
    assert int(ContextIdx.CTX_VALID_RATIO) == 10
    assert int(ContextIdx.CTX_TREND_GATE_SPREAD_MEAN) == 8
    assert int(ContextIdx.CTX_POC_DRIFT_X) == 9
    assert int(ContextIdx.CTX_REGIME_CODE) == 14
    assert int(ContextIdx.CTX_REGIME_PERSISTENCE) == 15


def test_stage2a_function_signatures_are_locked() -> None:
    assert list(inspect.signature(classify_regime).parameters) == [
        "structure_adapted",
        "context_adapted",
        "regime_hint",
        "tradable_mask",
        "cfg4",
    ]
    assert list(inspect.signature(generate_strategy_intent).parameters) == [
        "alpha_signal_tensor",
        "score_tensor",
        "profile_stat_tensor",
        "regime_id",
        "regime_confidence",
        "tradable_mask",
        "cfg4",
    ]
    assert list(inspect.signature(compute_conviction).parameters) == [
        "intent_long",
        "intent_short",
        "intent_flat",
        "signed_intent_utility",
        "regime_confidence",
        "score_tensor",
        "context_adapted",
        "tradable_mask",
        "cfg4",
    ]
    assert list(inspect.signature(compute_normalized_signal_allocation).parameters) == [
        "conviction_net",
        "regime_confidence",
        "tradable_mask",
        "asset_enabled_mask",
        "cfg4",
    ]


def test_stage2a_output_field_sets_are_locked() -> None:
    assert [f.name for f in fields(RegimeClassificationResult)] == [
        "regime_id",
        "regime_confidence",
        "regime_valid_mask",
        "low_regime_confidence_mask",
        "regime_score",
    ]
    assert [f.name for f in fields(StrategyIntentResult)] == [
        "intent_long",
        "intent_short",
        "intent_flat",
        "intent_valid_mask",
        "intent_gate_mask",
        "signed_intent_utility",
    ]
    assert [f.name for f in fields(ConvictionResult)] == [
        "conviction_long",
        "conviction_short",
        "conviction_net",
        "conviction_valid_mask",
    ]
    assert [f.name for f in fields(AllocationResult)] == [
        "allocation_score",
        "target_weight",
        "allocation_rank",
        "allocation_valid_mask",
    ]


def test_legacy_signal_output_fields_are_frozen_to_five() -> None:
    assert Module4SignalOutput.__dataclass_params__.frozen
    assert [f.name for f in fields(Module4SignalOutput)] == [
        "regime_primary_ta",
        "regime_confidence_ta",
        "intent_long_ta",
        "intent_short_ta",
        "target_qty_ta",
    ]
