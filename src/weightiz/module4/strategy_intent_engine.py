from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from weightiz.module1.core import ProfileStatIdx, ScoreIdx


REGIME_NONE = np.int8(0)
REGIME_NEUTRAL = np.int8(1)
REGIME_TREND = np.int8(2)
REGIME_P_SHAPE = np.int8(3)
REGIME_B_SHAPE = np.int8(4)
REGIME_DOUBLE_DISTRIBUTION = np.int8(5)
GATE_COUNT = 6


@dataclass(frozen=True)
class StrategyIntentResult:
    intent_long: np.ndarray
    intent_short: np.ndarray
    intent_flat: np.ndarray
    intent_valid_mask: np.ndarray
    intent_gate_mask: np.ndarray
    signed_intent_utility: np.ndarray


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)


def generate_strategy_intent(
    *,
    alpha_signal_tensor: np.ndarray,
    score_tensor: np.ndarray,
    profile_stat_tensor: np.ndarray,
    regime_id: np.ndarray,
    regime_confidence: np.ndarray,
    tradable_mask: np.ndarray,
    cfg4: object,
) -> StrategyIntentResult:
    alpha = np.asarray(alpha_signal_tensor, dtype=np.float64)
    score = np.asarray(score_tensor, dtype=np.float64)
    profile = np.asarray(profile_stat_tensor, dtype=np.float64)
    regime = np.asarray(regime_id, dtype=np.int8)
    confidence = np.asarray(regime_confidence, dtype=np.float64)
    tradable = np.asarray(tradable_mask, dtype=bool)

    if alpha.ndim != 3 or score.ndim != 3 or profile.ndim != 3:
        raise RuntimeError("Module4 intent inputs must be [A,T,*]")
    A, T, _ = score.shape
    if alpha.shape[0] != A or alpha.shape[1] != T:
        raise RuntimeError("alpha_signal_tensor shape mismatch")
    if profile.shape[0] != A or profile.shape[1] != T:
        raise RuntimeError("profile_stat_tensor shape mismatch")
    if regime.shape != (A, T):
        raise RuntimeError(f"regime_id shape mismatch: got {regime.shape}, expected {(A, T)}")
    if confidence.shape != (A, T):
        raise RuntimeError(f"regime_confidence shape mismatch: got {confidence.shape}, expected {(A, T)}")
    if tradable.shape != (A, T):
        raise RuntimeError(f"tradable_mask shape mismatch: got {tradable.shape}, expected {(A, T)}")

    finite_mask = (
        np.all(np.isfinite(alpha), axis=2)
        & np.all(np.isfinite(score), axis=2)
        & np.all(np.isfinite(profile), axis=2)
        & np.isfinite(confidence)
    )

    alpha_mean = np.mean(alpha, axis=2) if alpha.shape[2] > 0 else np.zeros((A, T), dtype=np.float64)
    bo_long = score[:, :, int(ScoreIdx.SCORE_BO_LONG)]
    bo_short = score[:, :, int(ScoreIdx.SCORE_BO_SHORT)]
    reject = score[:, :, int(ScoreIdx.SCORE_REJECT)] if score.shape[2] > int(ScoreIdx.SCORE_REJECT) else np.zeros((A, T), dtype=np.float64)
    dclip = profile[:, :, int(ProfileStatIdx.DCLIP)] if profile.shape[2] > int(ProfileStatIdx.DCLIP) else np.zeros((A, T), dtype=np.float64)
    z_delta = profile[:, :, int(ProfileStatIdx.Z_DELTA)] if profile.shape[2] > int(ProfileStatIdx.Z_DELTA) else np.zeros((A, T), dtype=np.float64)

    long_utility = bo_long + 0.10 * _clip01(alpha_mean) + 0.05 * _clip01(dclip) + 0.05 * _clip01(z_delta) - 0.05 * _clip01(np.abs(reject))
    short_utility = bo_short + 0.10 * _clip01(-alpha_mean) + 0.05 * _clip01(-dclip) + 0.05 * _clip01(-z_delta) - 0.05 * _clip01(np.abs(reject))

    regime_allows_long = np.isin(regime, np.array([REGIME_NEUTRAL, REGIME_TREND, REGIME_P_SHAPE, REGIME_DOUBLE_DISTRIBUTION], dtype=np.int8))
    regime_allows_short = np.isin(regime, np.array([REGIME_NEUTRAL, REGIME_TREND, REGIME_B_SHAPE, REGIME_DOUBLE_DISTRIBUTION], dtype=np.int8))
    long_threshold_pass = regime_allows_long & tradable & (long_utility >= float(cfg4.entry_threshold))
    short_threshold_pass = regime_allows_short & tradable & (short_utility >= float(cfg4.entry_threshold))

    signed_intent_utility = long_utility - short_utility
    directional_edge = np.abs(signed_intent_utility) > float(cfg4.exit_threshold)

    intent_long = long_threshold_pass & directional_edge & (signed_intent_utility > 0.0)
    intent_short = short_threshold_pass & directional_edge & (signed_intent_utility < 0.0)

    tie = long_threshold_pass & short_threshold_pass & np.isclose(long_utility, short_utility, rtol=0.0, atol=float(cfg4.eps))
    stronger_long = long_threshold_pass & short_threshold_pass & (long_utility > short_utility)
    stronger_short = long_threshold_pass & short_threshold_pass & (short_utility > long_utility)
    intent_long = (intent_long | stronger_long) & (~tie)
    intent_short = (intent_short | stronger_short) & (~tie)

    invalid = ~finite_mask
    intent_long[invalid] = False
    intent_short[invalid] = False
    intent_long[~tradable] = False
    intent_short[~tradable] = False

    intent_flat = ~(intent_long | intent_short)
    signed_intent_utility = np.where(intent_flat, 0.0, signed_intent_utility)
    signed_intent_utility[invalid] = 0.0

    gate_mask = np.zeros((A, T, GATE_COUNT), dtype=bool)
    gate_mask[:, :, 0] = finite_mask
    gate_mask[:, :, 1] = tradable
    gate_mask[:, :, 2] = regime_allows_long
    gate_mask[:, :, 3] = regime_allows_short
    gate_mask[:, :, 4] = long_threshold_pass
    gate_mask[:, :, 5] = short_threshold_pass

    return StrategyIntentResult(
        intent_long=np.ascontiguousarray(intent_long, dtype=bool),
        intent_short=np.ascontiguousarray(intent_short, dtype=bool),
        intent_flat=np.ascontiguousarray(intent_flat, dtype=bool),
        intent_valid_mask=np.ascontiguousarray(finite_mask, dtype=bool),
        intent_gate_mask=np.ascontiguousarray(gate_mask, dtype=bool),
        signed_intent_utility=np.ascontiguousarray(signed_intent_utility, dtype=np.float64),
    )
