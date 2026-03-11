from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from weightiz.module3 import ContextIdx, StructIdx


REGIME_NONE = np.int8(0)
REGIME_NEUTRAL = np.int8(1)
REGIME_TREND = np.int8(2)
REGIME_P_SHAPE = np.int8(3)
REGIME_B_SHAPE = np.int8(4)
REGIME_DOUBLE_DISTRIBUTION = np.int8(5)
REGIME_COUNT = 6

_PRECEDENCE = np.asarray([0, 1, 4, 3, 2, 5], dtype=np.int64)


@dataclass(frozen=True)
class RegimeClassificationResult:
    regime_id: np.ndarray
    regime_confidence: np.ndarray
    regime_valid_mask: np.ndarray
    low_regime_confidence_mask: np.ndarray
    regime_score: np.ndarray


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)


def _require_shape(name: str, arr: np.ndarray, expected: tuple[int, ...]) -> None:
    if arr.shape != expected:
        raise RuntimeError(f"{name} shape mismatch: got {arr.shape}, expected {expected}")


def _require_float64(name: str, arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr)
    if x.dtype != np.float64:
        raise RuntimeError(f"{name} dtype must be float64, got {x.dtype}")
    return x


def _winner_from_scores(score_atr: np.ndarray) -> np.ndarray:
    A, T, R = score_atr.shape
    winner = np.zeros((A, T), dtype=np.int8)
    regime_ids = np.arange(R, dtype=np.int64)
    for a in range(A):
        for t in range(T):
            s = score_atr[a, t]
            order = np.lexsort((regime_ids, -_PRECEDENCE, -s))
            winner[a, t] = np.int8(order[0])
    return winner


def classify_regime(
    *,
    structure_adapted: np.ndarray,
    context_adapted: np.ndarray,
    regime_hint: np.ndarray,
    tradable_mask: np.ndarray,
    cfg4: object,
) -> RegimeClassificationResult:
    structure = _require_float64("structure_adapted", structure_adapted)
    context = _require_float64("context_adapted", context_adapted)
    hint = _require_float64("regime_hint", regime_hint)
    tradable = np.asarray(tradable_mask, dtype=bool)

    if structure.ndim != 3:
        raise RuntimeError(f"structure_adapted must be [A,T,F_struct], got ndim={structure.ndim}")
    if context.ndim != 3:
        raise RuntimeError(f"context_adapted must be [A,T,C], got ndim={context.ndim}")
    if hint.ndim != 3:
        raise RuntimeError(f"regime_hint must be [A,T,1], got ndim={hint.ndim}")

    A, T, _ = structure.shape
    _require_shape("context_adapted", context, (A, T, int(ContextIdx.N_FIELDS)))
    _require_shape("regime_hint", hint, (A, T, 1))
    _require_shape("tradable_mask", tradable, (A, T))

    finite_mask = (
        np.all(np.isfinite(structure), axis=2)
        & np.all(np.isfinite(context), axis=2)
        & np.all(np.isfinite(hint[:, :, 0]))
    )
    valid_mask = finite_mask & tradable

    struct_valid_ratio = structure[:, :, int(StructIdx.VALID_RATIO)]
    struct_skew = structure[:, :, int(StructIdx.SKEW_ANCHOR)]
    struct_trend_spread = structure[:, :, int(StructIdx.TREND_GATE_SPREAD_MEAN)]
    struct_poc_drift = structure[:, :, int(StructIdx.POC_DRIFT_X)]

    context_valid_ratio = context[:, :, int(ContextIdx.CTX_VALID_RATIO)]
    ctx_trend_spread = context[:, :, int(ContextIdx.CTX_TREND_GATE_SPREAD_MEAN)]
    ctx_poc_drift = context[:, :, int(ContextIdx.CTX_POC_DRIFT_X)]
    ctx_regime_code = context[:, :, int(ContextIdx.CTX_REGIME_CODE)]
    ctx_regime_persistence = context[:, :, int(ContextIdx.CTX_REGIME_PERSISTENCE)]

    v = _clip01(0.5 * (struct_valid_ratio + context_valid_ratio))
    trend_e = _clip01(
        0.5
        * (
            np.abs(struct_trend_spread) / float(cfg4.trend_spread_min)
            + np.abs(ctx_trend_spread) / float(cfg4.trend_spread_min)
        )
    )
    drift_e = _clip01(
        0.5
        * (
            np.abs(struct_poc_drift) / float(cfg4.trend_poc_drift_min_abs)
            + np.abs(ctx_poc_drift) / float(cfg4.trend_poc_drift_min_abs)
        )
    )
    neutral_e = _clip01(1.0 - np.abs(ctx_poc_drift) / float(cfg4.neutral_poc_drift_max_abs))
    p_shape_e = _clip01(struct_skew / float(cfg4.shape_skew_min_abs))
    b_shape_e = _clip01((-struct_skew) / float(cfg4.shape_skew_min_abs))
    ctx_code = np.rint(_clip01(ctx_regime_code / 4.0) * 4.0).astype(np.int64)
    hint_code = np.rint(_clip01(hint[:, :, 0] / 4.0) * 4.0).astype(np.int64)
    persist_e = _clip01(ctx_regime_persistence)

    regime_score = np.zeros((A, T, REGIME_COUNT), dtype=np.float64)
    regime_score[:, :, int(REGIME_NONE)] = 1.0 - v
    regime_score[:, :, int(REGIME_NEUTRAL)] = v * (
        0.60 * neutral_e + 0.20 * (1.0 - trend_e) + 0.20 * (1.0 - drift_e)
    )
    regime_score[:, :, int(REGIME_TREND)] = v * (
        0.45 * trend_e
        + 0.35 * drift_e
        + 0.10 * np.isin(ctx_code, np.array([1, 2], dtype=np.int64)).astype(np.float64)
        + 0.10 * np.isin(hint_code, np.array([1, 2], dtype=np.int64)).astype(np.float64)
    )
    regime_score[:, :, int(REGIME_P_SHAPE)] = v * (
        0.70 * p_shape_e
        + 0.20 * (ctx_code == 4).astype(np.float64)
        + 0.10 * (hint_code == 4).astype(np.float64)
    )
    regime_score[:, :, int(REGIME_B_SHAPE)] = v * (
        0.70 * b_shape_e
        + 0.20 * (ctx_code == 4).astype(np.float64)
        + 0.10 * (hint_code == 4).astype(np.float64)
    )
    regime_score[:, :, int(REGIME_DOUBLE_DISTRIBUTION)] = v * (
        0.70 * (ctx_code == 3).astype(np.float64)
        + 0.20 * (hint_code == 3).astype(np.float64)
        + 0.10 * persist_e
    )

    regime_score[~valid_mask] = 0.0
    regime_id = _winner_from_scores(regime_score)
    top1 = np.take_along_axis(regime_score, regime_id[:, :, None].astype(np.int64), axis=2)[:, :, 0]
    sorted_scores = np.sort(regime_score, axis=2)
    top2 = sorted_scores[:, :, -2]
    denom = np.abs(top1) + np.abs(top2) + float(cfg4.eps)
    confidence = np.clip((top1 - top2) / denom, 0.0, 1.0)
    confidence = np.where(valid_mask, confidence, 0.0)
    low_conf = valid_mask & (confidence < float(cfg4.regime_confidence_min))

    regime_id = np.where(valid_mask, regime_id, REGIME_NONE).astype(np.int8, copy=False)
    return RegimeClassificationResult(
        regime_id=np.ascontiguousarray(regime_id, dtype=np.int8),
        regime_confidence=np.ascontiguousarray(confidence, dtype=np.float64),
        regime_valid_mask=np.ascontiguousarray(valid_mask, dtype=bool),
        low_regime_confidence_mask=np.ascontiguousarray(low_conf, dtype=bool),
        regime_score=np.ascontiguousarray(regime_score, dtype=np.float64),
    )
