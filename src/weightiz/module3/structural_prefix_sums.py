from __future__ import annotations

import numpy as np


def _assert_atw(name: str, arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    if out.ndim != 3:
        raise RuntimeError(f"{name} must have shape [A,T,W], got ndim={out.ndim}, shape={out.shape}")
    return out


def build_prefix_sum(series_atw: np.ndarray) -> np.ndarray:
    """Build prefix sum along T (axis=1). Returns [A,T+1,W]."""
    x = _assert_atw("series_atw", np.asarray(series_atw, dtype=np.float64))
    A, T, W = x.shape
    prefix = np.zeros((A, T + 1, W), dtype=np.float64)
    prefix[:, 1:, :] = np.cumsum(x, axis=1, dtype=np.float64)
    return prefix


def build_prefix_count(valid_atw: np.ndarray) -> np.ndarray:
    """Build prefix count along T (axis=1). Returns [A,T+1,W]."""
    v = _assert_atw("valid_atw", np.asarray(valid_atw, dtype=bool))
    A, T, W = v.shape
    prefix = np.zeros((A, T + 1, W), dtype=np.float64)
    prefix[:, 1:, :] = np.cumsum(v.astype(np.float64), axis=1, dtype=np.float64)
    return prefix


def rolling_sum_from_prefix(prefix_sum_at1w: np.ndarray, window: int) -> np.ndarray:
    """Causal full-window rolling sum from prefix sums. Returns [A,T,W] with NaN warmup."""
    p = _assert_atw("prefix_sum_at1w", np.asarray(prefix_sum_at1w, dtype=np.float64))
    A, T1, W = p.shape
    T = T1 - 1
    ww = int(window)
    if ww <= 0:
        raise RuntimeError("window must be > 0")

    out = np.full((A, T, W), np.nan, dtype=np.float64)
    if T <= 0:
        return out
    t_idx = np.arange(T, dtype=np.int64)
    hi = t_idx + 1
    lo = hi - ww
    valid = lo >= 0
    if not np.any(valid):
        return out
    hi_v = hi[valid]
    lo_v = lo[valid]
    out[:, valid, :] = p[:, hi_v, :] - p[:, lo_v, :]
    return out


def rolling_count_from_prefix(prefix_count_at1w: np.ndarray, window: int) -> np.ndarray:
    p = _assert_atw("prefix_count_at1w", np.asarray(prefix_count_at1w, dtype=np.float64))
    return rolling_sum_from_prefix(p, window)


def rolling_mean_from_prefix(
    prefix_sum_at1w: np.ndarray,
    prefix_count_at1w: np.ndarray,
    window: int,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    s = rolling_sum_from_prefix(prefix_sum_at1w, int(window))
    c = rolling_count_from_prefix(prefix_count_at1w, int(window))
    out = np.full(s.shape, np.nan, dtype=np.float64)
    np.divide(s, np.maximum(c, float(eps)), out=out, where=np.isfinite(c) & (c > 0.0))
    return out
