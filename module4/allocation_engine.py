from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AllocationResult:
    allocation_score: np.ndarray
    target_weight: np.ndarray
    allocation_rank: np.ndarray
    allocation_valid_mask: np.ndarray


def compute_normalized_signal_allocation(
    *,
    conviction_net: np.ndarray,
    regime_confidence: np.ndarray,
    tradable_mask: np.ndarray,
    asset_enabled_mask: np.ndarray,
    cfg4: object,
) -> AllocationResult:
    conviction = np.asarray(conviction_net, dtype=np.float64)
    confidence = np.asarray(regime_confidence, dtype=np.float64)
    tradable = np.asarray(tradable_mask, dtype=bool)
    asset_enabled = np.asarray(asset_enabled_mask, dtype=bool)

    if conviction.shape != confidence.shape or conviction.shape != tradable.shape:
        raise RuntimeError("allocation input [A,T] shape mismatch")
    A, T = conviction.shape
    if asset_enabled.shape != (A,):
        raise RuntimeError(f"asset_enabled_mask shape mismatch: got {asset_enabled.shape}, expected {(A,)}")

    allocation_score = conviction * confidence
    allocation_score = np.where(tradable, allocation_score, 0.0)
    allocation_score = np.where(asset_enabled[:, None], allocation_score, 0.0)

    norm = np.sum(np.abs(allocation_score), axis=0) + 1e-12
    target_weight = allocation_score / norm[None, :]
    target_weight = np.clip(target_weight, -float(cfg4.max_abs_weight), float(cfg4.max_abs_weight))
    target_weight = np.where(tradable, target_weight, 0.0)
    target_weight = np.where(asset_enabled[:, None], target_weight, 0.0)

    rank = np.zeros((A, T), dtype=np.int16)
    idx = np.arange(A, dtype=np.int64)
    for t in range(T):
        score_t = allocation_score[:, t]
        abs_conv_t = np.abs(conviction[:, t])
        order = np.lexsort((idx, -abs_conv_t, -score_t))
        inv = np.empty(A, dtype=np.int16)
        inv[order] = np.arange(A, dtype=np.int16)
        rank[:, t] = inv

    valid = np.isfinite(conviction) & np.isfinite(confidence)
    valid &= asset_enabled[:, None]
    return AllocationResult(
        allocation_score=np.ascontiguousarray(allocation_score, dtype=np.float64),
        target_weight=np.ascontiguousarray(target_weight, dtype=np.float64),
        allocation_rank=np.ascontiguousarray(rank, dtype=np.int16),
        allocation_valid_mask=np.ascontiguousarray(valid, dtype=bool),
    )
