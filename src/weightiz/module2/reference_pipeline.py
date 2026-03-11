from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .market_profile_engine import run_streaming_profile_engine


@dataclass(frozen=True)
class ReferenceOutputs:
    mixture: dict[str, np.ndarray]
    profiles: dict[str, np.ndarray]
    metrics: dict[str, np.ndarray]


def run_reference_pipeline(
    *,
    state: Any,
    cfg: Any,
    physics: Any,
    mode: str,
    open_use: np.ndarray,
    high_use: np.ndarray,
    low_use: np.ndarray,
    close_use: np.ndarray,
    vol_use: np.ndarray,
    valid: np.ndarray,
    build_poc_rank_fn: Any,
    compute_value_area_fn: Any,
    rolling_median_mad_fn: Any,
    profile_stat_idx: Any,
    score_idx: Any,
    phase_enum: Any,
) -> ReferenceOutputs:
    """Deterministic reference module for forensic parity (no notebook execution)."""
    artifacts = run_streaming_profile_engine(
        state=state,
        cfg=cfg,
        physics=physics,
        mode=mode,
        open_use=open_use,
        high_use=high_use,
        low_use=low_use,
        close_use=close_use,
        vol_use=vol_use,
        valid=valid,
        build_poc_rank_fn=build_poc_rank_fn,
        compute_value_area_fn=compute_value_area_fn,
        rolling_median_mad_fn=rolling_median_mad_fn,
        profile_stat_idx=profile_stat_idx,
        score_idx=score_idx,
        phase_enum=phase_enum,
        collect_forensics=True,
    )
    return ReferenceOutputs(
        mixture=artifacts.mixture_history,
        profiles=artifacts.profile_history,
        metrics=artifacts.metric_history,
    )
