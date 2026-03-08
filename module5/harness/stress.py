from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from weightiz_module1_core import TensorState


def compute_bar_valid(
    open_px: np.ndarray,
    high_px: np.ndarray,
    low_px: np.ndarray,
    close_px: np.ndarray,
    volume: np.ndarray,
) -> np.ndarray:
    finite = (
        np.isfinite(open_px)
        & np.isfinite(high_px)
        & np.isfinite(low_px)
        & np.isfinite(close_px)
        & np.isfinite(volume)
    )
    phys = (
        (high_px >= low_px)
        & (high_px >= open_px)
        & (high_px >= close_px)
        & (low_px <= open_px)
        & (low_px <= close_px)
        & (volume >= 0.0)
    )
    return finite & phys


def apply_missing_bursts(
    state: TensorState,
    active_t: np.ndarray,
    scenario: Any,
    rng: np.random.Generator,
) -> None:
    if scenario.missing_burst_prob <= 0.0 or scenario.missing_burst_max <= 0:
        return

    t_count, a_count = state.bar_valid.shape
    start_mask = (rng.random((t_count, a_count)) < float(scenario.missing_burst_prob)) & active_t[:, None]
    starts = np.argwhere(start_mask)

    lo_len = int(max(1, scenario.missing_burst_min))
    hi_len = int(max(lo_len, scenario.missing_burst_max))

    for t0, a in starts.tolist():
        length = int(rng.integers(lo_len, hi_len + 1))
        t1 = min(t_count, int(t0) + length)
        state.open_px[t0:t1, a] = np.nan
        state.high_px[t0:t1, a] = np.nan
        state.low_px[t0:t1, a] = np.nan
        state.close_px[t0:t1, a] = np.nan
        state.volume[t0:t1, a] = np.nan
        state.rvol[t0:t1, a] = np.nan
        state.atr_floor[t0:t1, a] = np.nan
        state.bar_valid[t0:t1, a] = False


def apply_jitter(
    state: TensorState,
    active_t: np.ndarray,
    scenario: Any,
    rng: np.random.Generator,
) -> None:
    sigma_bps = float(scenario.jitter_sigma_bps)
    if sigma_bps <= 0.0:
        return

    eps = rng.normal(
        loc=0.0,
        scale=sigma_bps / 1e4,
        size=state.open_px.shape,
    ).astype(np.float64)
    mask = active_t[:, None] & state.bar_valid

    mult = 1.0 + eps
    state.open_px = np.where(mask, state.open_px * mult, state.open_px)
    state.high_px = np.where(mask, state.high_px * mult, state.high_px)
    state.low_px = np.where(mask, state.low_px * mult, state.low_px)
    state.close_px = np.where(mask, state.close_px * mult, state.close_px)

    stacked = np.stack([state.open_px, state.high_px, state.low_px, state.close_px], axis=2)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        hi = np.nanmax(stacked, axis=2)
        lo = np.nanmin(stacked, axis=2)
    state.high_px = np.where(mask, hi, state.high_px)
    state.low_px = np.where(mask, lo, state.low_px)


def recompute_bar_valid_inplace(state: TensorState) -> None:
    state.bar_valid[:, :] = compute_bar_valid(
        state.open_px,
        state.high_px,
        state.low_px,
        state.close_px,
        state.volume,
    )


def set_placeholders_from_bar_valid(state: TensorState) -> None:
    tick = np.asarray(state.eps.eps_div, dtype=np.float64)[None, :]
    atr0 = np.maximum(4.0 * tick, 1e-12)
    state.rvol[:, :] = np.where(state.bar_valid, 1.0, np.nan)
    state.atr_floor[:, :] = np.where(state.bar_valid, atr0, np.nan)


def assert_placeholder_consistency(state: TensorState) -> None:
    valid = np.asarray(state.bar_valid, dtype=bool)
    if state.rvol.shape != valid.shape or state.atr_floor.shape != valid.shape:
        raise RuntimeError(
            f"Placeholder shape mismatch: rvol={state.rvol.shape}, atr_floor={state.atr_floor.shape}, "
            f"bar_valid={valid.shape}"
        )
    if np.any(valid):
        if not np.all(np.isfinite(state.rvol[valid])):
            bad = np.argwhere(valid & (~np.isfinite(state.rvol)))[0]
            raise RuntimeError(
                f"rvol must be finite on valid bars; first_offending_index={[int(bad[0]), int(bad[1])]}"
            )
        if not np.all(np.isfinite(state.atr_floor[valid])):
            bad = np.argwhere(valid & (~np.isfinite(state.atr_floor)))[0]
            raise RuntimeError(
                f"atr_floor must be finite on valid bars; first_offending_index={[int(bad[0]), int(bad[1])]}"
            )
    invalid = ~valid
    if np.any(invalid & np.isfinite(state.rvol)):
        bad = np.argwhere(invalid & np.isfinite(state.rvol))[0]
        raise RuntimeError(
            f"rvol must be NaN on invalid bars; first_offending_index={[int(bad[0]), int(bad[1])]}"
        )
    if np.any(invalid & np.isfinite(state.atr_floor)):
        bad = np.argwhere(invalid & np.isfinite(state.atr_floor))[0]
        raise RuntimeError(
            f"atr_floor must be NaN on invalid bars; first_offending_index={[int(bad[0]), int(bad[1])]}"
        )
