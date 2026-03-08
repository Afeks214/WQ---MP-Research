from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class RollingProfileState:
    """Rolling profile state without storing a window dimension in outputs."""

    inj_total_wan: np.ndarray
    inj_delta_wan: np.ndarray
    mom0_wa: np.ndarray
    mom1_wa: np.ndarray
    mom2_wa: np.ndarray
    vp_total_an: np.ndarray
    vp_delta_an: np.ndarray
    agg_m0_a: np.ndarray
    agg_m1_a: np.ndarray
    agg_m2_a: np.ndarray


@dataclass(frozen=True)
class RollingMoments:
    m0: np.ndarray
    m1: np.ndarray
    m2: np.ndarray


def init_rolling_profile_state(*, window: int, assets: int, bins: int) -> RollingProfileState:
    if window <= 0:
        raise RuntimeError("window must be > 0")
    if assets <= 0 or bins <= 0:
        raise RuntimeError("assets and bins must be > 0")

    return RollingProfileState(
        inj_total_wan=np.zeros((window, assets, bins), dtype=np.float64, order="C"),
        inj_delta_wan=np.zeros((window, assets, bins), dtype=np.float64, order="C"),
        mom0_wa=np.zeros((window, assets), dtype=np.float64, order="C"),
        mom1_wa=np.zeros((window, assets), dtype=np.float64, order="C"),
        mom2_wa=np.zeros((window, assets), dtype=np.float64, order="C"),
        vp_total_an=np.zeros((assets, bins), dtype=np.float64, order="C"),
        vp_delta_an=np.zeros((assets, bins), dtype=np.float64, order="C"),
        agg_m0_a=np.zeros(assets, dtype=np.float64),
        agg_m1_a=np.zeros(assets, dtype=np.float64),
        agg_m2_a=np.zeros(assets, dtype=np.float64),
    )


def apply_rolling_update(
    state: RollingProfileState,
    *,
    t_index: int,
    inj_total_an: np.ndarray,
    inj_delta_an: np.ndarray,
    moments: RollingMoments,
) -> None:
    """Apply VP_t = VP_{t-1} - inj_{t-W} + inj_t for both total and delta."""
    slot = int(t_index) % int(state.inj_total_wan.shape[0])

    old_total = state.inj_total_wan[slot]
    old_delta = state.inj_delta_wan[slot]

    state.vp_total_an += inj_total_an - old_total
    state.vp_delta_an += inj_delta_an - old_delta

    old_m0 = state.mom0_wa[slot]
    old_m1 = state.mom1_wa[slot]
    old_m2 = state.mom2_wa[slot]

    state.agg_m0_a += moments.m0 - old_m0
    state.agg_m1_a += moments.m1 - old_m1
    state.agg_m2_a += moments.m2 - old_m2

    state.inj_total_wan[slot] = inj_total_an
    state.inj_delta_wan[slot] = inj_delta_an
    state.mom0_wa[slot] = moments.m0
    state.mom1_wa[slot] = moments.m1
    state.mom2_wa[slot] = moments.m2
