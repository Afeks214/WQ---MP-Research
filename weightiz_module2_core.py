"""
Weightiz Institutional Engine - Module 2 (Core Weightiz Profile Engine)
=======================================================================

This module implements the deterministic, numpy-only core profile engine on top
of Module 1 tensors.

Key guarantees:
- No pandas/polars/vectorbt/backtrader in core path.
- Causal ATR and RVOL precomputation (no look-ahead).
- Strict POC tie-break: max mass -> min |x| -> left (smaller x).
- Candle micro-physics (range/body/CLV/sigma/weights) precomputed once over (T, A).
- Rolling profile loop is causal over t and vectorized over A/B/W.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from weightiz_module1_core import (
    EngineConfig,
    Phase,
    ProfileStatIdx,
    ScoreIdx,
    TensorState,
    validate_state_hard,
)


SQRT_2PI: float = float(np.sqrt(2.0 * np.pi))


class DayTypeIdx(IntEnum):
    """Channels for profile taxonomy snapshots (A, D)."""

    NORMAL_NEUTRAL = 0
    TREND = 1
    P_SHAPE = 2
    B_SHAPE = 3
    DOUBLE_DISTRIBUTION = 4
    N_FIELDS = 5


@dataclass(frozen=True)
class Module2Config:
    """Configuration for Module 2 math and gating."""

    # Rolling profile window (bars)
    profile_window_bars: int = 60
    profile_warmup_bars: int = 60

    # ATR settings
    atr_span: int = 14
    atr_alpha: Optional[float] = None
    atr_floor_mult_tick: float = 4.0
    atr_floor_abs: float | np.ndarray = 0.0

    # RVOL settings
    rvol_lookback_sessions: int = 20
    rvol_policy: str = "neutral_one"  # one of: neutral_one, warmup_mask, nan_fail
    rvol_vol_eps_mult_tick: float = 1.0
    rvol_vol_eps_abs: float | np.ndarray = 1e-12
    rvol_clip_min: float = 0.0
    rvol_clip_max: float = 50.0

    # Robust volume cap (Section 9.2 style)
    volume_cap_window_bars: int = 60
    volume_cap_mad_mult: float = 5.0
    volume_cap_min_mult: float = 0.25

    # Mixture center shifts (in x-space)
    mu1_clv_shift: float = 0.0
    mu2_clv_shift: float = 0.35

    # Mixture sigma and weight model
    sigma1_base: float = 0.18
    sigma1_body_coeff: float = 0.22
    sigma1_rvol_coeff: float = 0.06
    sigma1_range_coeff: float = 0.08
    sigma1_min: float = 0.05
    sigma1_max: float = 1.50

    sigma2_ratio_base: float = 1.80
    sigma2_body_coeff: float = 0.60
    sigma2_clv_coeff: float = 0.30
    sigma2_min: float = 0.08
    sigma2_max: float = 3.00

    w1_base: float = 0.62
    w1_body_coeff: float = 0.28
    w1_rvol_coeff: float = 0.04
    w1_clv_coeff: float = 0.12
    w1_min: float = 0.05
    w1_max: float = 0.95

    # Hybrid delta (Section 10 style)
    ret_scale_window_bars: int = 60
    ret_scale_min_periods: int = 10
    ret_scale_min: float = 0.05

    # Profile extraction
    va_threshold: float = 0.70
    poc_eq_atol: float = 0.0

    # Score and delta gates (Section 13-14 style)
    d_clip: float = 6.0
    break_bias: float = 1.0
    reject_center: float = 2.0
    rvol_trend_cutoff: float = 2.0
    body_trend_cutoff: float = 0.60

    delta_mad_lookback_bars: int = 180
    delta_mad_min_periods: int = 10
    sigma_delta_min: float = 0.05
    delta_gate_threshold: float = 1.0

    # Taxonomy thresholds (for optional snapshot output)
    normal_concentration_threshold: float = 0.05
    trend_spread_threshold_x: float = 2.50
    trend_delta_confirm_z: float = 1.50
    double_dist_min_sep_x: float = 1.0
    double_dist_valley_frac: float = 0.35

    # Runtime checks
    fail_on_non_finite_output: bool = True


@dataclass
class MarketPhysics:
    """
    Precomputed market-physics tensors over (T, A).

    Required fields follow the Module 2 plan and include one extra auditing
    field (`rvol_eligible`) used to maintain warmup eligibility deterministically.
    """

    atr_raw: np.ndarray
    atr_floor: np.ndarray
    atr_eff: np.ndarray
    cumvol: np.ndarray
    rvol: np.ndarray
    range_: np.ndarray
    body_pct: np.ndarray
    clv: np.ndarray
    sigma1: np.ndarray
    sigma2: np.ndarray
    w1: np.ndarray
    w2: np.ndarray
    med_v: np.ndarray
    mad_v: np.ndarray
    cap_v_eff: np.ndarray
    ret: np.ndarray
    ret_norm: np.ndarray
    s_r: np.ndarray
    rvol_eligible: np.ndarray


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _assert_shape(name: str, arr: np.ndarray, expected: Tuple[int, ...]) -> None:
    if arr.shape != expected:
        raise RuntimeError(f"{name} shape mismatch: got {arr.shape}, expected {expected}")


def _assert_finite(name: str, arr: np.ndarray) -> None:
    if not np.all(np.isfinite(arr)):
        bad = np.argwhere(~np.isfinite(arr))[:8]
        raise RuntimeError(f"{name} contains non-finite values at indices {bad.tolist()}")


def _as_asset_vector(name: str, value: float | np.ndarray, A: int) -> np.ndarray:
    vec = np.asarray(value, dtype=np.float64)
    if vec.ndim == 0:
        out = np.full(A, float(vec), dtype=np.float64)
    else:
        out = np.asarray(vec, dtype=np.float64)
        _assert_shape(name, out, (A,))
    _assert_finite(name, out)
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x64))


def _ema_causal(arr_ta: np.ndarray, alpha: float) -> np.ndarray:
    """
    Causal EMA over t, vectorized over assets.

    out[t] = alpha * arr[t] + (1-alpha) * out[t-1]
    """
    T, A = arr_ta.shape
    out = np.zeros((T, A), dtype=np.float64)
    out[0] = arr_ta[0]
    one_minus = 1.0 - alpha
    for t in range(1, T):
        out[t] = alpha * arr_ta[t] + one_minus * out[t - 1]
    return out


def _rolling_median_mad_causal(
    arr_ta: np.ndarray,
    window: int,
    min_periods: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Causal rolling median and MAD over t, vectorized over assets.

    Uses sliding_window_view for full windows and deterministic prefix handling
    for early bars where fewer than `window` observations exist.
    """
    arr = np.asarray(arr_ta, dtype=np.float64)
    T, A = arr.shape
    if window <= 0:
        raise RuntimeError("rolling window must be > 0")
    if min_periods <= 0:
        raise RuntimeError("min_periods must be > 0")

    med = np.full((T, A), np.nan, dtype=np.float64)
    mad = np.full((T, A), np.nan, dtype=np.float64)

    w = min(int(window), T)
    if T == 0:
        return med, mad

    if T >= w:
        wins = sliding_window_view(arr, window_shape=w, axis=0)  # (T-w+1, w, A)
        med_full = np.nanmedian(wins, axis=1)
        mad_full = np.nanmedian(np.abs(wins - med_full[:, None, :]), axis=1)
        med[w - 1 :] = med_full
        mad[w - 1 :] = mad_full

    prefix = min(w - 1, T)
    for t in range(prefix):
        n = t + 1
        if n < min_periods:
            continue
        seg = arr[:n]
        m = np.nanmedian(seg, axis=0)
        d = np.nanmedian(np.abs(seg - m[None, :]), axis=0)
        med[t] = m
        mad[t] = d

    if min_periods > 1:
        cnt = np.minimum(np.arange(1, T + 1, dtype=np.int64), w)
        eligible = cnt >= int(min_periods)
        med[~eligible] = np.nan
        mad[~eligible] = np.nan

    return med, mad


def _session_starts(session_id: np.ndarray) -> np.ndarray:
    T = session_id.shape[0]
    out = np.zeros(T, dtype=np.int64)
    start = 0
    out[0] = 0
    for t in range(1, T):
        if session_id[t] != session_id[t - 1]:
            start = t
        out[t] = start
    return out


def _build_poc_rank(x_grid: np.ndarray) -> np.ndarray:
    """
    Rank bins by strict tie-break order:
    1) minimal |x|
    2) then smaller x (left)
    """
    B = x_grid.shape[0]
    order = np.lexsort((x_grid, np.abs(x_grid)))
    rank = np.empty(B, dtype=np.int64)
    rank[order] = np.arange(B, dtype=np.int64)
    return rank


def _make_va_offsets(B: int) -> np.ndarray:
    offs = np.empty(B, dtype=np.int64)
    offs[0] = 0
    pos = 1
    k = 1
    while pos < B:
        offs[pos] = +k
        pos += 1
        if pos < B:
            offs[pos] = -k
            pos += 1
        k += 1
    return offs


def _compute_value_area_idx_offsetscan(
    vp_ab: np.ndarray,
    ipoc_a: np.ndarray,
    va_threshold: float,
    offsets_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Deterministic Value Area extraction via offset scan around POC.

    Inputs:
    - vp_ab: (A, B)
    - ipoc_a: (A,)

    Returns:
    - ipoc, ivah, ival each shape (A,)
    """
    A, B = vp_ab.shape
    ipoc = np.asarray(ipoc_a, dtype=np.int64)

    total = np.sum(vp_ab, axis=1)
    target = total * float(va_threshold)

    idx = ipoc[:, None] + offsets_b[None, :]
    valid = (idx >= 0) & (idx < B)
    idx_clip = np.clip(idx, 0, B - 1)

    scanned = np.take_along_axis(vp_ab, idx_clip, axis=1)
    scanned = np.where(valid, scanned, 0.0)

    csum = np.cumsum(scanned, axis=1)
    reached = csum >= target[:, None]
    k = np.argmax(reached, axis=1).astype(np.int64)
    empty = total <= 0.0
    k = np.where(empty, 0, k)

    j = np.arange(B, dtype=np.int64)[None, :]
    included = j <= k[:, None]
    idx_included = np.where(included & valid, idx, -1)

    ivah = np.max(idx_included, axis=1)
    big = np.int64(B + 10)
    tmp = np.where(idx_included >= 0, idx_included, big)
    ival = np.min(tmp, axis=1)

    ivah = np.where(ivah < 0, ipoc, ivah)
    ival = np.where((ival >= big) | (ival < 0), ipoc, ival)

    return ipoc, ivah, ival


def _validate_stage_a_inputs(state: TensorState, cfg: Module2Config) -> None:
    validate_state_hard(state)

    T = state.cfg.T
    A = state.cfg.A

    for name, arr in [
        ("open_px", state.open_px),
        ("high_px", state.high_px),
        ("low_px", state.low_px),
        ("close_px", state.close_px),
        ("volume", state.volume),
        ("bar_valid", state.bar_valid),
    ]:
        _assert_shape(name, arr, (T, A))

    valid = state.bar_valid
    if not np.any(valid):
        raise RuntimeError("No valid market bars available (bar_valid all False)")

    # Finiteness checks are required on loaded/valid bars.
    for name, arr in [
        ("open_px", state.open_px[valid]),
        ("high_px", state.high_px[valid]),
        ("low_px", state.low_px[valid]),
        ("close_px", state.close_px[valid]),
        ("volume", state.volume[valid]),
    ]:
        _assert_finite(name, arr)

    o = state.open_px[valid]
    h = state.high_px[valid]
    l = state.low_px[valid]
    c = state.close_px[valid]
    v = state.volume[valid]

    if np.any(h < l):
        bad = int(np.where(h < l)[0][0])
        raise RuntimeError(f"OHLC violation high < low at valid-row index {bad}")
    if np.any(h < o):
        bad = int(np.where(h < o)[0][0])
        raise RuntimeError(f"OHLC violation high < open at valid-row index {bad}")
    if np.any(h < c):
        bad = int(np.where(h < c)[0][0])
        raise RuntimeError(f"OHLC violation high < close at valid-row index {bad}")
    if np.any(l > o):
        bad = int(np.where(l > o)[0][0])
        raise RuntimeError(f"OHLC violation low > open at valid-row index {bad}")
    if np.any(l > c):
        bad = int(np.where(l > c)[0][0])
        raise RuntimeError(f"OHLC violation low > close at valid-row index {bad}")
    if np.any(v < 0.0):
        bad = int(np.where(v < 0.0)[0][0])
        raise RuntimeError(f"Volume violation negative volume at valid-row index {bad}")

    tick = np.asarray(state.eps.eps_div, dtype=np.float64)
    _assert_shape("tick_size(eps_div)", tick, (A,))
    _assert_finite("tick_size(eps_div)", tick)
    if np.any(tick <= 0.0):
        idx = int(np.where(tick <= 0.0)[0][0])
        raise RuntimeError(f"tick_size[{idx}] must be > 0")

    if cfg.profile_window_bars <= 1:
        raise RuntimeError("profile_window_bars must be > 1")
    if cfg.profile_window_bars > T:
        raise RuntimeError("profile_window_bars cannot exceed T")
    if cfg.rvol_lookback_sessions <= 0:
        raise RuntimeError("rvol_lookback_sessions must be > 0")
    if cfg.rvol_policy not in {"neutral_one", "warmup_mask", "nan_fail"}:
        raise RuntimeError("rvol_policy must be one of: neutral_one, warmup_mask, nan_fail")


# -----------------------------------------------------------------------------
# Stage B: Precompute market physics
# -----------------------------------------------------------------------------

def precompute_market_physics(state: TensorState, cfg: Module2Config) -> MarketPhysics:
    """
    Compute causal ATR/ATR-floor, RVOL baseline, and micro-physics tensors.

    No look-ahead guarantee:
    - ATR uses only <= t bars.
    - RVOL baseline at (session d, minute m) uses only prior sessions d' < d.
    """
    _validate_stage_a_inputs(state, cfg)

    T = state.cfg.T
    A = state.cfg.A
    tick = np.asarray(state.eps.eps_div, dtype=np.float64)

    valid = state.bar_valid.astype(bool)

    # Forward-filled close for deterministic handling of invalid bars.
    close_ff = np.zeros((T, A), dtype=np.float64)
    close_ff[0] = np.where(valid[0], state.close_px[0], 0.0)
    for t in range(1, T):
        close_ff[t] = np.where(valid[t], state.close_px[t], close_ff[t - 1])

    open_use = np.where(valid, state.open_px, close_ff)
    high_use = np.where(valid, state.high_px, close_ff)
    low_use = np.where(valid, state.low_px, close_ff)
    close_use = np.where(valid, state.close_px, close_ff)

    # 1) True range and ATR EMA (causal)
    prev_close = np.empty((T, A), dtype=np.float64)
    prev_close[0] = close_ff[0]
    prev_close[1:] = close_ff[:-1]

    tr1 = high_use - low_use
    tr2 = np.abs(high_use - prev_close)
    tr3 = np.abs(low_use - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))

    alpha = float(cfg.atr_alpha) if cfg.atr_alpha is not None else 2.0 / (float(cfg.atr_span) + 1.0)
    if not (0.0 < alpha <= 1.0):
        raise RuntimeError(f"Invalid ATR alpha={alpha}; must be in (0,1]")
    atr_raw = _ema_causal(tr, alpha)

    # 2) ATR hard floor and effective ATR
    atr_floor_abs = _as_asset_vector("atr_floor_abs", cfg.atr_floor_abs, A)
    floor_asset = np.maximum(float(cfg.atr_floor_mult_tick) * tick, atr_floor_abs)
    atr_floor = np.broadcast_to(floor_asset[None, :], (T, A)).astype(np.float64, copy=True)
    atr_eff = np.maximum(atr_raw, atr_floor)

    # 3) Cumulative per-session volume
    vol_clean = np.where(valid, state.volume, 0.0)
    cumvol = np.zeros((T, A), dtype=np.float64)
    cumvol[0] = vol_clean[0]
    for t in range(1, T):
        if state.session_id[t] != state.session_id[t - 1]:
            cumvol[t] = vol_clean[t]
        else:
            cumvol[t] = cumvol[t - 1] + vol_clean[t]

    # 3b) Robust volume cap statistics (Section 9.2 style)
    cap_window = int(cfg.volume_cap_window_bars)
    if cap_window <= 0:
        raise RuntimeError("volume_cap_window_bars must be > 0")
    vol_for_stats = np.where(valid, state.volume, np.nan)
    med_v, mad_v = _rolling_median_mad_causal(vol_for_stats, window=cap_window, min_periods=1)
    med_v = np.where(np.isfinite(med_v), med_v, 0.0)
    mad_v = np.where(np.isfinite(mad_v), mad_v, 0.0)
    mad_v_scaled = 1.4826 * mad_v
    cap_v_base = med_v + float(cfg.volume_cap_mad_mult) * mad_v_scaled
    cap_min = float(cfg.volume_cap_min_mult) * np.where(np.isfinite(med_v), med_v, 0.0)

    # 4) RVOL via causal ToD baseline median over prior sessions
    sid_rel = (state.session_id - np.min(state.session_id)).astype(np.int64)
    S = int(np.max(sid_rel)) + 1

    minute = state.minute_of_day.astype(np.int64)
    if np.any((minute < 0) | (minute >= 1440)):
        bad = int(np.where((minute < 0) | (minute >= 1440))[0][0])
        raise RuntimeError(f"minute_of_day out of range at t={bad}: {int(minute[bad])}")

    M = 1440
    cumvol_cube = np.full((S, M, A), np.nan, dtype=np.float64)
    cumvol_cube[sid_rel, minute, :] = cumvol

    baseline_cube = np.full((S, M, A), np.nan, dtype=np.float64)
    used_minutes = np.unique(minute)
    L = int(cfg.rvol_lookback_sessions)

    for m in used_minutes.tolist():
        vals = cumvol_cube[:, m, :]  # (S, A)
        vals_prev = np.empty_like(vals)
        vals_prev[0] = np.nan
        vals_prev[1:] = vals[:-1]

        baseline = np.full((S, A), np.nan, dtype=np.float64)
        if S >= L:
            windows = sliding_window_view(vals_prev, window_shape=L, axis=0)  # (S-L+1, L, A)
            baseline[L - 1 :] = np.nanmedian(windows, axis=1)
            prefix = min(L - 1, S)
        else:
            prefix = S

        for d in range(prefix):
            baseline[d] = np.nanmedian(vals_prev[: d + 1], axis=0)

        baseline_cube[:, m, :] = baseline

    baseline_ta = baseline_cube[sid_rel, minute, :]  # (T, A)
    rvol_eligible = np.isfinite(baseline_ta) & (baseline_ta > 0.0)

    vol_eps_abs = _as_asset_vector("rvol_vol_eps_abs", cfg.rvol_vol_eps_abs, A)
    vol_eps = np.maximum(float(cfg.rvol_vol_eps_mult_tick) * tick[None, :], vol_eps_abs[None, :])

    denom = np.maximum(np.where(np.isfinite(baseline_ta), baseline_ta, np.nan), vol_eps)
    rvol = np.divide(cumvol, denom, out=np.full((T, A), np.nan, dtype=np.float64), where=np.isfinite(denom))

    policy = cfg.rvol_policy
    if policy == "nan_fail":
        bad = ~np.isfinite(rvol)
        if np.any(bad):
            idx = np.argwhere(bad)[0]
            raise RuntimeError(f"RVOL baseline unavailable at t={int(idx[0])}, a={int(idx[1])}")
    elif policy in {"neutral_one", "warmup_mask"}:
        rvol = np.where(np.isfinite(rvol), rvol, 1.0)
    else:
        raise RuntimeError(f"Unsupported rvol_policy={policy}")

    rvol = np.clip(rvol, float(cfg.rvol_clip_min), float(cfg.rvol_clip_max))

    # RVOL expansion on robust cap (Sec 9.2.1): base_cap * (1 + log(max(1, RVOL)))
    cap_v_eff = cap_v_base * (1.0 + np.log(np.maximum(1.0, rvol)))
    cap_v_eff = np.maximum(np.where(np.isfinite(cap_v_eff), cap_v_eff, 0.0), cap_min)
    cap_v_eff = np.where(valid, cap_v_eff, 0.0)

    # 4b) Signed returns and robust return scale for Hybrid Delta (Section 10)
    ret = np.zeros((T, A), dtype=np.float64)
    ret[1:] = close_use[1:] - close_use[:-1]
    ret_norm = ret / (atr_eff + state.eps.eps_div[None, :])
    ret_norm = np.where(valid, ret_norm, 0.0)

    rs_window = int(cfg.ret_scale_window_bars)
    if rs_window <= 0:
        raise RuntimeError("ret_scale_window_bars must be > 0")
    _, ret_mad = _rolling_median_mad_causal(
        ret_norm,
        window=rs_window,
        min_periods=int(cfg.ret_scale_min_periods),
    )
    s_r = np.maximum(1.4826 * np.where(np.isfinite(ret_mad), ret_mad, 0.0), float(cfg.ret_scale_min))
    s_r = np.where(valid, s_r, float(cfg.ret_scale_min))

    # 5) Candle micro-physics precompute over full (T, A)
    range_ = np.maximum(high_use - low_use, state.eps.eps_range[None, :])
    body_pct = np.abs(close_use - open_use) / (range_ + state.eps.eps_range[None, :])
    body_pct = np.clip(body_pct, 0.0, 1.0)

    clv = ((close_use - low_use) - (high_use - close_use)) / (range_ + state.eps.eps_range[None, :])
    clv = np.clip(clv, -1.0, 1.0)

    range_ratio = np.clip(range_ / (atr_eff + state.eps.eps_div[None, :]), 0.0, 10.0)
    rvol_log = np.log1p(np.maximum(rvol, 0.0))

    sigma1 = (
        float(cfg.sigma1_base)
        + float(cfg.sigma1_body_coeff) * (1.0 - body_pct)
        + float(cfg.sigma1_rvol_coeff) * rvol_log
        + float(cfg.sigma1_range_coeff) * range_ratio
    )
    sigma1 = np.clip(sigma1, float(cfg.sigma1_min), float(cfg.sigma1_max))

    sigma2 = sigma1 * (
        float(cfg.sigma2_ratio_base)
        + float(cfg.sigma2_body_coeff) * body_pct
        + float(cfg.sigma2_clv_coeff) * np.abs(clv)
    )
    sigma2 = np.clip(sigma2, float(cfg.sigma2_min), float(cfg.sigma2_max))

    w1 = (
        float(cfg.w1_base)
        + float(cfg.w1_body_coeff) * body_pct
        + float(cfg.w1_rvol_coeff) * rvol_log
        - float(cfg.w1_clv_coeff) * np.abs(clv)
    )
    w1 = np.clip(w1, float(cfg.w1_min), float(cfg.w1_max))
    w2 = 1.0 - w1

    # Deterministic neutral defaults on invalid bars
    sigma1 = np.where(valid, sigma1, float(cfg.sigma1_base))
    sigma2 = np.where(valid, sigma2, float(cfg.sigma2_min))
    w1 = np.where(valid, w1, 1.0)
    w2 = np.where(valid, w2, 0.0)
    body_pct = np.where(valid, body_pct, 0.0)
    clv = np.where(valid, clv, 0.0)

    # 6) Finite and conservation checks
    for name, arr in [
        ("atr_raw", atr_raw),
        ("atr_floor", atr_floor),
        ("atr_eff", atr_eff),
        ("cumvol", cumvol),
        ("rvol", rvol),
        ("med_v", med_v),
        ("mad_v", mad_v),
        ("cap_v_eff", cap_v_eff),
        ("ret", ret),
        ("ret_norm", ret_norm),
        ("s_r", s_r),
        ("range_", range_),
        ("body_pct", body_pct),
        ("clv", clv),
        ("sigma1", sigma1),
        ("sigma2", sigma2),
        ("w1", w1),
        ("w2", w2),
    ]:
        _assert_finite(name, arr)

    if np.any(sigma1 <= 0.0):
        idx = np.argwhere(sigma1 <= 0.0)[0]
        raise RuntimeError(f"sigma1 <= 0 at t={int(idx[0])}, a={int(idx[1])}")
    if np.any(sigma2 <= 0.0):
        idx = np.argwhere(sigma2 <= 0.0)[0]
        raise RuntimeError(f"sigma2 <= 0 at t={int(idx[0])}, a={int(idx[1])}")

    if not np.allclose(w1 + w2, 1.0, atol=1e-12, rtol=0.0):
        max_err = float(np.max(np.abs((w1 + w2) - 1.0)))
        raise RuntimeError(f"w1+w2 conservation violated; max error={max_err:.3e}")

    return MarketPhysics(
        atr_raw=atr_raw,
        atr_floor=atr_floor,
        atr_eff=atr_eff,
        cumvol=cumvol,
        rvol=rvol,
        range_=range_,
        body_pct=body_pct,
        clv=clv,
        sigma1=sigma1,
        sigma2=sigma2,
        w1=w1,
        w2=w2,
        med_v=med_v,
        mad_v=mad_v,
        cap_v_eff=cap_v_eff,
        ret=ret,
        ret_norm=ret_norm,
        s_r=s_r,
        rvol_eligible=rvol_eligible,
    )


# -----------------------------------------------------------------------------
# Stage C: Core profile loop
# -----------------------------------------------------------------------------

def run_weightiz_profile_engine(state: TensorState, cfg: Module2Config) -> None:
    """
    Run the deterministic profile engine and write outputs in-place to state:
    - state.rvol and state.atr_floor (effective ATR denominator)
    - state.vp, state.vp_delta
    - state.profile_stats, state.scores
    """
    _validate_stage_a_inputs(state, cfg)

    t0_wall = time.perf_counter()

    T = state.cfg.T
    A = state.cfg.A
    B = state.cfg.B
    W = int(cfg.profile_window_bars)

    physics = precompute_market_physics(state, cfg)

    # Persist computed physics channels required downstream.
    state.rvol[:, :] = physics.rvol
    state.atr_floor[:, :] = physics.atr_eff

    # Reset outputs in a deterministic way.
    state.vp.fill(0.0)
    state.vp_delta.fill(0.0)
    state.profile_stats.fill(np.nan)
    state.scores.fill(np.nan)

    # Precompute windows once (no per-window reconstruction).
    close_w = sliding_window_view(np.asarray(state.close_px, dtype=np.float64), window_shape=W, axis=0)
    vol_w = sliding_window_view(np.asarray(state.volume, dtype=np.float64), window_shape=W, axis=0)
    valid_w = sliding_window_view(state.bar_valid.astype(bool), window_shape=W, axis=0)

    clv_w = sliding_window_view(physics.clv, window_shape=W, axis=0)
    sig1_w = sliding_window_view(physics.sigma1, window_shape=W, axis=0)
    sig2_w = sliding_window_view(physics.sigma2, window_shape=W, axis=0)
    w1_w = sliding_window_view(physics.w1, window_shape=W, axis=0)
    w2_w = sliding_window_view(physics.w2, window_shape=W, axis=0)
    body_w = sliding_window_view(physics.body_pct, window_shape=W, axis=0)
    rvol_w = sliding_window_view(physics.rvol, window_shape=W, axis=0)
    cap_w = sliding_window_view(physics.cap_v_eff, window_shape=W, axis=0)
    ret_norm_w = sliding_window_view(physics.ret_norm, window_shape=W, axis=0)
    s_r_w = sliding_window_view(physics.s_r, window_shape=W, axis=0)

    n_win = close_w.shape[0]
    if n_win != (T - W + 1):
        raise RuntimeError("Unexpected window count from sliding_window_view")

    x = np.asarray(state.x_grid, dtype=np.float64)
    x2 = x * x
    dx = float(state.cfg.dx)
    idx_zero = int(np.argmin(np.abs(x)))

    poc_rank = _build_poc_rank(x)
    va_offsets = _make_va_offsets(B)
    computed_mask = np.zeros((T, A), dtype=bool)

    # Causal loop over t; vectorized over A/B/W.
    for win_idx in range(n_win):
        t = win_idx + W - 1

        if t < int(cfg.profile_warmup_bars - 1):
            continue

        # Optional phase-based eligibility (WARMUP bars should remain non-tradable).
        if state.phase[t] == np.int8(Phase.WARMUP):
            continue

        # (W, A)
        close_win = close_w[win_idx]
        vol_win = np.maximum(vol_w[win_idx], 0.0)
        valid_win = valid_w[win_idx]

        clv_win = clv_w[win_idx]
        sig1_win = sig1_w[win_idx]
        sig2_win = sig2_w[win_idx]
        w1_win = w1_w[win_idx]
        w2_win = w2_w[win_idx]
        body_pct_win = body_w[win_idx]
        rvol_win = rvol_w[win_idx]
        cap_win = cap_w[win_idx]
        ret_norm_win = ret_norm_w[win_idx]
        s_r_win = s_r_w[win_idx]

        close_t = state.close_px[t]
        atr_t = physics.atr_eff[t]

        tradable_t = (
            state.bar_valid[t]
            & np.isfinite(close_t)
            & np.isfinite(atr_t)
            & (atr_t > 0.0)
        )

        if not np.any(tradable_t):
            continue

        # Coordinate reprojection to x-space.
        mu = (close_win - close_t[None, :]) / (atr_t[None, :] + state.eps.eps_div[None, :])
        mu1 = mu + float(cfg.mu1_clv_shift) * clv_win
        mu2 = mu + float(cfg.mu2_clv_shift) * clv_win

        # Gaussian mixture injection
        sig1_safe = np.maximum(sig1_win, state.eps.eps_range[None, :])
        sig2_safe = np.maximum(sig2_win, state.eps.eps_range[None, :])

        z1 = (x[None, None, :] - mu1[:, :, None]) / sig1_safe[:, :, None]
        z2 = (x[None, None, :] - mu2[:, :, None]) / sig2_safe[:, :, None]

        pdf1 = np.exp(-0.5 * z1 * z1) / (sig1_safe[:, :, None] * SQRT_2PI)
        pdf2 = np.exp(-0.5 * z2 * z2) / (sig2_safe[:, :, None] * SQRT_2PI)

        mix = w1_win[:, :, None] * pdf1 + w2_win[:, :, None] * pdf2

        # Normalize each candle-injection over x-grid (mass conservation).
        mix_sum = np.sum(mix, axis=2) * dx
        mix = np.divide(
            mix,
            mix_sum[:, :, None] + state.eps.eps_pdf,
            out=np.zeros_like(mix),
            where=(mix_sum[:, :, None] > 0.0),
        )

        # Robust capped profile volume (Section 9.2): V_prof = min(V, cap_eff)
        vprof_win = np.minimum(vol_win, np.maximum(cap_win, 0.0))
        vol_weight = vprof_win * np.maximum(rvol_win, 0.0)
        mass = vol_weight[:, :, None] * mix

        # Apply bar validity mask.
        mass = np.where(valid_win[:, :, None], mass, 0.0)

        vp_t = np.sum(mass, axis=0)  # (A, B)

        # Hybrid Delta (Section 10.1/10.2 paper constants):
        # pSR_buy = sigmoid(ln(9) * r_norm / s_r), pCLV_buy = sigmoid(6 * CLV)
        # p_buy = w_trend * pSR_buy + (1-w_trend) * pCLV_buy, w_trend=body_pct
        # signed blend = 2*p_buy - 1
        p_sr_buy = _sigmoid(
            float(np.log(9.0)) * ret_norm_win / (s_r_win + state.eps.eps_div[None, :])
        )
        p_clv_buy = _sigmoid(6.0 * clv_win)
        w_trend = np.clip(body_pct_win, 0.0, 1.0)
        p_buy = w_trend * p_sr_buy + (1.0 - w_trend) * p_clv_buy
        p_buy = np.clip(p_buy, 0.0, 1.0)
        signed_blend = 2.0 * p_buy - 1.0
        vpd_t = np.sum(mass * signed_blend[:, :, None], axis=0)

        # Force non-tradable assets to zero profile at this t.
        vp_t[~tradable_t] = 0.0
        vpd_t[~tradable_t] = 0.0

        state.vp[t] = vp_t
        state.vp_delta[t] = vpd_t

        # ---- Profile stats / scores extraction ----
        total = np.sum(vp_t, axis=1)
        denom = total + state.eps.eps_vol

        mu_prof = (vp_t @ x) / denom
        ex2 = (vp_t @ x2) / denom
        sigma_prof = np.sqrt(np.maximum(ex2 - mu_prof * mu_prof, 0.0))
        sigma_eff = np.maximum(sigma_prof, 2.0 * dx)

        D = (-mu_prof) / (sigma_eff + state.eps.eps_pdf)
        Dclip = np.clip(D, -float(cfg.d_clip), float(cfg.d_clip))

        vp_max = np.max(vp_t, axis=1)
        A_aff = vp_t[:, idx_zero] / (vp_max + state.eps.eps_vol)

        # Strict POC tie-break
        max_mass = np.max(vp_t, axis=1, keepdims=True)
        is_max = np.isclose(vp_t, max_mass, atol=float(cfg.poc_eq_atol), rtol=0.0)
        masked_rank = np.where(is_max, poc_rank[None, :], B + 1)
        ipoc = np.argmin(masked_rank, axis=1).astype(np.int64)
        ipoc = np.where(total > state.eps.eps_vol, ipoc, idx_zero)

        row = np.arange(A, dtype=np.int64)
        delta0 = vpd_t[:, idx_zero] / (vp_t[:, idx_zero] + state.eps.eps_vol)
        delta_poc = vpd_t[row, ipoc] / (vp_t[row, ipoc] + state.eps.eps_vol)

        wpoc = 1.0 - A_aff
        delta_eff = wpoc * delta_poc + (1.0 - wpoc) * delta0

        ipoc, ivah, ival = _compute_value_area_idx_offsetscan(
            vp_ab=vp_t,
            ipoc_a=ipoc,
            va_threshold=float(cfg.va_threshold),
            offsets_b=va_offsets,
        )

        # Write channels for tradable assets only.
        tradable_idx = np.where(tradable_t)[0]
        if tradable_idx.size > 0:
            a = tradable_idx
            ps = state.profile_stats[t, a]

            ps[:, int(ProfileStatIdx.MU_PROF)] = mu_prof[a]
            ps[:, int(ProfileStatIdx.SIGMA_PROF)] = sigma_prof[a]
            ps[:, int(ProfileStatIdx.SIGMA_EFF)] = sigma_eff[a]
            ps[:, int(ProfileStatIdx.D)] = D[a]
            ps[:, int(ProfileStatIdx.DCLIP)] = Dclip[a]
            ps[:, int(ProfileStatIdx.A_AFFINITY)] = A_aff[a]
            ps[:, int(ProfileStatIdx.DELTA0)] = delta0[a]
            ps[:, int(ProfileStatIdx.DELTA_POC)] = delta_poc[a]
            ps[:, int(ProfileStatIdx.DELTA_EFF)] = delta_eff[a]
            ps[:, int(ProfileStatIdx.IPOC)] = ipoc[a].astype(np.float64)
            ps[:, int(ProfileStatIdx.IVAH)] = ivah[a].astype(np.float64)
            ps[:, int(ProfileStatIdx.IVAL)] = ival[a].astype(np.float64)
            computed_mask[t, a] = True

        if cfg.fail_on_non_finite_output:
            core_cols = np.array(
                [
                    int(ProfileStatIdx.MU_PROF),
                    int(ProfileStatIdx.SIGMA_PROF),
                    int(ProfileStatIdx.SIGMA_EFF),
                    int(ProfileStatIdx.D),
                    int(ProfileStatIdx.DCLIP),
                    int(ProfileStatIdx.A_AFFINITY),
                    int(ProfileStatIdx.DELTA0),
                    int(ProfileStatIdx.DELTA_POC),
                    int(ProfileStatIdx.DELTA_EFF),
                    int(ProfileStatIdx.IPOC),
                    int(ProfileStatIdx.IVAH),
                    int(ProfileStatIdx.IVAL),
                ],
                dtype=np.int64,
            )
            arr_ps = state.profile_stats[t, tradable_t][:, core_cols]
            if arr_ps.size and not np.all(np.isfinite(arr_ps)):
                bad = np.argwhere(~np.isfinite(arr_ps))[0]
                raise RuntimeError(
                    f"Non-finite profile_stats at t={t}, local_idx={bad.tolist()}"
                )

    # Stage F: sequential post-processing for delta gating and scores.
    # This removes expensive rolling nanmedian calls from the main t loop.
    delta_eff_all = np.where(
        computed_mask,
        state.profile_stats[:, :, int(ProfileStatIdx.DELTA_EFF)],
        np.nan,
    )

    d_delta = np.full((T, A), np.nan, dtype=np.float64)
    for t in range(T):
        mask_t = computed_mask[t]
        if not np.any(mask_t):
            continue
        if t == 0 or state.reset_flag[t] == 1:
            d_delta[t, mask_t] = 0.0
        else:
            prev = delta_eff_all[t - 1]
            curr = delta_eff_all[t]
            ok = mask_t & np.isfinite(prev) & np.isfinite(curr)
            d_delta[t, ok] = curr[ok] - prev[ok]
            d_delta[t, mask_t & ~ok] = 0.0

    sigma_delta = np.full((T, A), np.nan, dtype=np.float64)
    sid = state.session_id
    starts = np.where(np.r_[True, sid[1:] != sid[:-1]])[0]
    ends = np.r_[starts[1:], T]

    for s, e in zip(starts.tolist(), ends.tolist()):
        seg_level = delta_eff_all[s:e]
        seg_chg = d_delta[s:e]

        _, mad_level = _rolling_median_mad_causal(
            seg_level,
            window=int(cfg.delta_mad_lookback_bars),
            min_periods=int(cfg.delta_mad_min_periods),
        )
        _, mad_chg = _rolling_median_mad_causal(
            seg_chg,
            window=int(cfg.delta_mad_lookback_bars),
            min_periods=int(cfg.delta_mad_min_periods),
        )

        sig_seg = np.maximum(
            np.maximum(
                1.4826 * np.where(np.isfinite(mad_level), mad_level, 0.0),
                1.4826 * np.where(np.isfinite(mad_chg), mad_chg, 0.0),
            ),
            float(cfg.sigma_delta_min),
        )
        sigma_delta[s:e] = sig_seg

    valid_post = computed_mask & np.isfinite(delta_eff_all) & np.isfinite(sigma_delta)
    z_delta = np.divide(
        delta_eff_all,
        sigma_delta + state.eps.eps_pdf,
        out=np.full((T, A), np.nan, dtype=np.float64),
        where=valid_post,
    )

    ln9 = float(np.log(9.0))
    gbreak = _sigmoid(ln9 * (z_delta - float(cfg.delta_gate_threshold)))
    greject = _sigmoid(ln9 * (-z_delta - float(cfg.delta_gate_threshold)))

    dclip_all = state.profile_stats[:, :, int(ProfileStatIdx.DCLIP)]
    aff_all = state.profile_stats[:, :, int(ProfileStatIdx.A_AFFINITY)]
    rvol_all = physics.rvol
    body_all = physics.body_pct

    sbase_bo_long = _sigmoid(dclip_all - float(cfg.break_bias)) * rvol_all
    sbase_bo_short = _sigmoid((-dclip_all) - float(cfg.break_bias)) * rvol_all
    rvoltrend = (
        (rvol_all > float(cfg.rvol_trend_cutoff))
        & (body_all > float(cfg.body_trend_cutoff))
    ).astype(np.float64)
    sbase_reject = _sigmoid(float(cfg.reject_center) - np.abs(dclip_all)) * aff_all * (1.0 - rvoltrend)

    score_bo_long = sbase_bo_long * gbreak
    score_bo_short = sbase_bo_short * gbreak
    score_reject = sbase_reject * greject
    score_rej_long = score_reject * _sigmoid(-dclip_all)
    score_rej_short = score_reject * _sigmoid(dclip_all)

    z_chan = state.profile_stats[:, :, int(ProfileStatIdx.Z_DELTA)]
    gb_chan = state.profile_stats[:, :, int(ProfileStatIdx.GBREAK)]
    gr_chan = state.profile_stats[:, :, int(ProfileStatIdx.GREJECT)]
    z_chan[valid_post] = z_delta[valid_post]
    gb_chan[valid_post] = gbreak[valid_post]
    gr_chan[valid_post] = greject[valid_post]

    sc_bo_l = state.scores[:, :, int(ScoreIdx.SCORE_BO_LONG)]
    sc_bo_s = state.scores[:, :, int(ScoreIdx.SCORE_BO_SHORT)]
    sc_rej = state.scores[:, :, int(ScoreIdx.SCORE_REJECT)]
    sc_rej_l = state.scores[:, :, int(ScoreIdx.SCORE_REJ_LONG)]
    sc_rej_s = state.scores[:, :, int(ScoreIdx.SCORE_REJ_SHORT)]

    sc_bo_l[valid_post] = score_bo_long[valid_post]
    sc_bo_s[valid_post] = score_bo_short[valid_post]
    sc_rej[valid_post] = score_reject[valid_post]
    sc_rej_l[valid_post] = score_rej_long[valid_post]
    sc_rej_s[valid_post] = score_rej_short[valid_post]

    if cfg.fail_on_non_finite_output:
        arr_sc = state.scores[valid_post]
        if arr_sc.size and not np.all(np.isfinite(arr_sc)):
            bad = np.argwhere(~np.isfinite(arr_sc))[0]
            raise RuntimeError(f"Non-finite scores after Stage F at local_idx={bad.tolist()}")

    # Warmup/insufficient RVOL handling for warmup_mask policy.
    if cfg.rvol_policy == "warmup_mask":
        bad_mask = ~physics.rvol_eligible
        # Keep outputs NaN where RVOL baseline was unavailable.
        state.profile_stats[bad_mask] = np.nan
        state.scores[bad_mask] = np.nan

    # Final hard checks.
    _assert_finite("vp", state.vp)
    _assert_finite("vp_delta", state.vp_delta)

    elapsed = time.perf_counter() - t0_wall
    # Deterministic, concise telemetry.
    print(
        f"MODULE2_OK elapsed_sec={elapsed:.6f} "
        f"T={T} A={A} B={B} W={W}"
    )


# -----------------------------------------------------------------------------
# Optional taxonomy snapshot (Module 4 can supersede this)
# -----------------------------------------------------------------------------

def compute_profile_taxonomy_snapshot(
    vp_ab: np.ndarray,
    profile_stats_ak: np.ndarray,
    x_grid: np.ndarray,
    cfg: Module2Config,
) -> np.ndarray:
    """
    Compute deterministic day-type booleans from current profile slice.

    Returns boolean array of shape (A, DayTypeIdx.N_FIELDS).
    """
    vp = np.asarray(vp_ab, dtype=np.float64)
    ps = np.asarray(profile_stats_ak, dtype=np.float64)
    x = np.asarray(x_grid, dtype=np.float64)

    A, B = vp.shape
    if ps.shape[0] != A:
        raise RuntimeError("profile_stats_ak asset axis mismatch")
    if x.shape[0] != B:
        raise RuntimeError("x_grid length mismatch")

    out = np.zeros((A, int(DayTypeIdx.N_FIELDS)), dtype=bool)

    ipoc = np.clip(ps[:, int(ProfileStatIdx.IPOC)].astype(np.int64), 0, B - 1)
    ivah = np.clip(ps[:, int(ProfileStatIdx.IVAH)].astype(np.int64), 0, B - 1)
    ival = np.clip(ps[:, int(ProfileStatIdx.IVAL)].astype(np.int64), 0, B - 1)

    poc_x = x[ipoc]
    vah_x = x[ivah]
    val_x = x[ival]

    mu_prof = ps[:, int(ProfileStatIdx.MU_PROF)]
    sigma_eff = ps[:, int(ProfileStatIdx.SIGMA_EFF)]
    z_delta = ps[:, int(ProfileStatIdx.Z_DELTA)]
    D = ps[:, int(ProfileStatIdx.D)]

    total = np.sum(vp, axis=1)
    conc = vp[np.arange(A, dtype=np.int64), ipoc] / (total + 1e-12)

    dx = float(x[1] - x[0]) if B > 1 else 1.0
    balance = np.abs((vah_x - poc_x) - (poc_x - val_x))

    normal = (
        (np.abs(poc_x - mu_prof) <= 2.0 * dx)
        & (conc > float(cfg.normal_concentration_threshold))
        & (balance <= 3.0 * dx)
    )
    out[:, int(DayTypeIdx.NORMAL_NEUTRAL)] = normal

    spread = vah_x - val_x
    edge = np.minimum(vah_x - poc_x, poc_x - val_x)
    trend_dir = np.where(D >= 0.0, 1.0, -1.0)
    trend = (
        (spread > float(cfg.trend_spread_threshold_x))
        & (edge <= 3.0 * dx)
        & ((z_delta * trend_dir) > float(cfg.trend_delta_confirm_z))
    )
    out[:, int(DayTypeIdx.TREND)] = trend

    pshape = (
        ((poc_x - mu_prof) > 0.5 * sigma_eff)
        & ((poc_x - val_x) > 2.0 * (vah_x - poc_x))
    )
    out[:, int(DayTypeIdx.P_SHAPE)] = pshape

    bshape = (
        ((mu_prof - poc_x) > 0.5 * sigma_eff)
        & ((vah_x - poc_x) > 2.0 * (poc_x - val_x))
    )
    out[:, int(DayTypeIdx.B_SHAPE)] = bshape

    # Double distribution: two local peaks + valley vacuum.
    if B >= 5:
        left = vp[:, :-2]
        mid = vp[:, 1:-1]
        right = vp[:, 2:]
        is_peak_mid = (mid >= left) & (mid >= right)
        is_peak = np.zeros_like(vp, dtype=bool)
        is_peak[:, 1:-1] = is_peak_mid

        peak_vals = np.where(is_peak, vp, -np.inf)
        top2 = np.argpartition(-peak_vals, kth=1, axis=1)[:, :2]
        p1 = top2[:, 0]
        p2 = top2[:, 1]

        # Ensure p1 has higher value.
        v1 = peak_vals[np.arange(A), p1]
        v2 = peak_vals[np.arange(A), p2]
        swap = v2 > v1
        p1, p2 = np.where(swap, p2, p1), np.where(swap, p1, p2)
        v1 = peak_vals[np.arange(A), p1]
        v2 = peak_vals[np.arange(A), p2]

        sep = np.abs(x[p1] - x[p2])
        sep_ok = sep >= float(cfg.double_dist_min_sep_x)

        dd = np.zeros(A, dtype=bool)
        for a in range(A):
            i1 = int(min(p1[a], p2[a]))
            i2 = int(max(p1[a], p2[a]))
            if not np.isfinite(v1[a]) or not np.isfinite(v2[a]) or i2 <= i1 + 1:
                continue
            valley = float(np.min(vp[a, i1 + 1 : i2]))
            thresh = float(cfg.double_dist_valley_frac) * float(min(v1[a], v2[a]))
            dd[a] = bool(sep_ok[a] and (valley < thresh))

        out[:, int(DayTypeIdx.DOUBLE_DISTRIBUTION)] = dd

    return out


# -----------------------------------------------------------------------------
# Example smoke test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    from weightiz_module1_core import NS_PER_MIN, preallocate_state

    T = 390 * 4
    A = 3
    symbols = ("SPY", "QQQ", "GLD")
    tick_size = np.array([0.01, 0.01, 0.01], dtype=np.float64)

    e_cfg = EngineConfig(
        T=T,
        A=A,
        B=240,
        tick_size=tick_size,
        seed=123,
    )

    start_ns = np.datetime64("2025-01-02T14:30:00", "ns").astype(np.int64)
    ts_ns = start_ns + np.arange(T, dtype=np.int64) * np.int64(NS_PER_MIN)

    st = preallocate_state(ts_ns=ts_ns, cfg=e_cfg, symbols=symbols)

    rng = np.random.default_rng(7)
    base = np.array([500.0, 430.0, 190.0], dtype=np.float64)

    close = np.zeros((T, A), dtype=np.float64)
    close[0] = base
    for t in range(1, T):
        close[t] = np.maximum(0.01, close[t - 1] * (1.0 + 0.0002 * rng.standard_normal(A)))

    open_px = close * (1.0 + 0.0001 * rng.standard_normal((T, A)))
    high_px = np.maximum(open_px, close) * (1.0 + 0.0005 * np.abs(rng.standard_normal((T, A))))
    low_px = np.minimum(open_px, close) * (1.0 - 0.0005 * np.abs(rng.standard_normal((T, A))))
    volume = np.maximum(1.0, 1e6 * (1.0 + 0.1 * rng.standard_normal((T, A))))

    st.open_px[:, :] = open_px
    st.high_px[:, :] = high_px
    st.low_px[:, :] = low_px
    st.close_px[:, :] = close
    st.volume[:, :] = volume
    st.bar_valid[:, :] = True

    m2_cfg = Module2Config()
    run_weightiz_profile_engine(st, m2_cfg)

    print("PROFILE_STATS_FINITE", np.isfinite(st.profile_stats[np.isfinite(st.profile_stats)]).all())
    print("SCORES_FINITE", np.isfinite(st.scores[np.isfinite(st.scores)]).all())
