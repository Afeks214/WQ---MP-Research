from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from module3 import ContextIdx


@dataclass(frozen=True)
class ConvictionResult:
    conviction_long: np.ndarray
    conviction_short: np.ndarray
    conviction_net: np.ndarray
    conviction_valid_mask: np.ndarray


def _clip(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float64), lo, hi)


def _support_from_context(context_adapted: np.ndarray) -> np.ndarray:
    if context_adapted.shape[2] <= int(ContextIdx.CTX_REGIME_PERSISTENCE):
        return np.ones(context_adapted.shape[:2], dtype=np.float64)
    persistence = context_adapted[:, :, int(ContextIdx.CTX_REGIME_PERSISTENCE)]
    return 0.5 + 0.5 * np.clip(persistence, 0.0, 1.0)


def compute_conviction(
    *,
    intent_long: np.ndarray,
    intent_short: np.ndarray,
    intent_flat: np.ndarray,
    signed_intent_utility: np.ndarray,
    regime_confidence: np.ndarray,
    score_tensor: np.ndarray,
    context_adapted: np.ndarray,
    tradable_mask: np.ndarray,
    cfg4: object,
) -> ConvictionResult:
    long_int = np.asarray(intent_long, dtype=bool)
    short_int = np.asarray(intent_short, dtype=bool)
    flat_int = np.asarray(intent_flat, dtype=bool)
    utility = np.asarray(signed_intent_utility, dtype=np.float64)
    confidence = np.asarray(regime_confidence, dtype=np.float64)
    score = np.asarray(score_tensor, dtype=np.float64)
    context = np.asarray(context_adapted, dtype=np.float64)
    tradable = np.asarray(tradable_mask, dtype=bool)

    if utility.shape != confidence.shape or utility.shape != long_int.shape or utility.shape != short_int.shape or utility.shape != flat_int.shape:
        raise RuntimeError("conviction input shape mismatch on [A,T] tensors")
    A, T = utility.shape
    if score.shape[:2] != (A, T):
        raise RuntimeError("score_tensor shape mismatch")
    if context.shape[:2] != (A, T):
        raise RuntimeError("context_adapted shape mismatch")
    if tradable.shape != (A, T):
        raise RuntimeError("tradable_mask shape mismatch")

    finite_mask = (
        np.isfinite(utility)
        & np.isfinite(confidence)
        & np.all(np.isfinite(score), axis=2)
        & np.all(np.isfinite(context), axis=2)
    )

    score_support = np.clip(np.mean(np.abs(score), axis=2), 0.0, 1.0)
    context_support = _support_from_context(context)
    support = 0.5 + 0.25 * score_support + 0.25 * context_support
    raw = float(cfg4.conviction_scale) * utility * confidence * support
    conviction_net = _clip(raw, -float(cfg4.conviction_clip), float(cfg4.conviction_clip))
    conviction_net = np.where(flat_int | (~tradable) | (~finite_mask), 0.0, conviction_net)
    conviction_long = np.where(conviction_net > 0.0, conviction_net, 0.0)
    conviction_short = np.where(conviction_net < 0.0, -conviction_net, 0.0)

    return ConvictionResult(
        conviction_long=np.ascontiguousarray(conviction_long, dtype=np.float64),
        conviction_short=np.ascontiguousarray(conviction_short, dtype=np.float64),
        conviction_net=np.ascontiguousarray(conviction_net, dtype=np.float64),
        conviction_valid_mask=np.ascontiguousarray(finite_mask, dtype=bool),
    )
