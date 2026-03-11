from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class DecisionReasonCode(IntEnum):
    INVALID_INPUT = 1
    CAUSAL_VIOLATION = 2
    MASKED_NOT_TRADABLE = 3
    RISK_FILTER_BLOCK = 4
    LOW_REGIME_CONFIDENCE = 5
    ZERO_SCORE_AFTER_MASK = 6
    ZERO_CONVICTION = 7
    INTENT_FLAT = 8
    INTENT_LONG = 9
    INTENT_SHORT = 10
    DEGRADED_BRIDGE_RESTRICTED = 11


@dataclass(frozen=True)
class Module4Telemetry:
    decision_reason_code: np.ndarray
    window_score: np.ndarray
    intent_gate_mask: np.ndarray
    allocation_rank: np.ndarray
    regime_score: np.ndarray
    decision_valid_mask: np.ndarray
    degraded_mode_mask: np.ndarray

    def __post_init__(self) -> None:
        reason = np.asarray(self.decision_reason_code)
        window = np.asarray(self.window_score)
        gate = np.asarray(self.intent_gate_mask)
        rank = np.asarray(self.allocation_rank)
        regime_score = np.asarray(self.regime_score)
        valid = np.asarray(self.decision_valid_mask)
        degraded = np.asarray(self.degraded_mode_mask)

        if reason.ndim != 2:
            raise RuntimeError(f"decision_reason_code must be [A,T], got shape={reason.shape}")
        A, T = reason.shape
        if reason.dtype != np.int16:
            raise RuntimeError(f"decision_reason_code dtype must be int16, got {reason.dtype}")
        if window.ndim != 3 or window.shape[:2] != (A, T) or window.dtype != np.float64:
            raise RuntimeError(f"window_score must be [A,T,W] float64, got shape={window.shape}, dtype={window.dtype}")
        if gate.ndim != 3 or gate.shape[:2] != (A, T) or gate.dtype != np.bool_:
            raise RuntimeError(f"intent_gate_mask must be [A,T,G] bool, got shape={gate.shape}, dtype={gate.dtype}")
        if rank.shape != (A, T) or rank.dtype != np.int16:
            raise RuntimeError(f"allocation_rank must be [A,T] int16, got shape={rank.shape}, dtype={rank.dtype}")
        if regime_score.ndim != 3 or regime_score.shape[:2] != (A, T) or regime_score.dtype != np.float64:
            raise RuntimeError(
                f"regime_score must be [A,T,R] float64, got shape={regime_score.shape}, dtype={regime_score.dtype}"
            )
        if valid.shape != (A, T) or valid.dtype != np.bool_:
            raise RuntimeError(f"decision_valid_mask must be [A,T] bool, got shape={valid.shape}, dtype={valid.dtype}")
        if degraded.shape != (A, T) or degraded.dtype != np.bool_:
            raise RuntimeError(f"degraded_mode_mask must be [A,T] bool, got shape={degraded.shape}, dtype={degraded.dtype}")
