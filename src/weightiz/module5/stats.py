"""
Weightiz Institutional Engine - Module 5 Stats (Pure Math Library)
===================================================================

Standalone statistical engine for institutional strategy validation.

Implemented tests:
- Deflated Sharpe Ratio (DSR; Bailey/LdP style, PSR with deflation)
- Probability of Backtest Overfitting (PBO via CSCV)
- White's Reality Check (WRC)
- SPA (Superior Predictive Ability)
- MCS (Model Confidence Set)

Hard constraints:
- NumPy-only math, no dependency on Weightiz state/tensors.
- Inputs are daily/session-level returns or losses.
- Guard: T <= 10_000 (to avoid bootstrap gather memory explosion).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Any

import numpy as np


EULER_GAMMA: float = 0.5772156649015329


@dataclass(frozen=True)
class BootstrapSpec:
    B: int = 2000
    avg_block_len: int = 20
    seed: int = 47


def validate_returns_1d(x: np.ndarray, name: str = "x", t_max: int = 10_000) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    if arr.ndim != 1:
        raise RuntimeError(f"{name} must be 1D, got ndim={arr.ndim}")
    T = int(arr.shape[0])
    if T < 3:
        raise RuntimeError(f"{name} must have T>=3, got T={T}")
    if T > int(t_max):
        raise RuntimeError(f"{name} length exceeds guard: T={T}, t_max={int(t_max)}")
    if not np.all(np.isfinite(arr)):
        bad = np.argwhere(~np.isfinite(arr))[:8]
        raise RuntimeError(f"{name} contains non-finite values at indices {bad.tolist()}")
    return arr


def validate_returns_2d(x: np.ndarray, name: str = "x", t_max: int = 10_000) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    if arr.ndim != 2:
        raise RuntimeError(f"{name} must be 2D, got ndim={arr.ndim}")
    T, N = arr.shape
    if T < 3:
        raise RuntimeError(f"{name} must have T>=3, got T={T}")
    if N < 1:
        raise RuntimeError(f"{name} must have N>=1, got N={N}")
    if T > int(t_max):
        raise RuntimeError(f"{name} length exceeds guard: T={T}, t_max={int(t_max)}")
    if not np.all(np.isfinite(arr)):
        bad = np.argwhere(~np.isfinite(arr))[:8]
        raise RuntimeError(f"{name} contains non-finite values at indices {bad.tolist()}")
    return arr


def norm_cdf(z: np.ndarray) -> np.ndarray:
    """
    Fast normal CDF approximation (Abramowitz-Stegun 7.1.26 style), vectorized.
    """
    z64 = np.asarray(z, dtype=np.float64)
    x = np.clip(z64, -12.0, 12.0)

    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.2316419 * ax)
    poly = (
        (
            (
                ((1.330274429 * t - 1.821255978) * t + 1.781477937) * t
                - 0.356563782
            )
            * t
            + 0.319381530
        )
        * t
    )
    phi = np.exp(-0.5 * ax * ax) / np.sqrt(2.0 * np.pi)
    cdf_pos = 1.0 - phi * poly
    out = np.where(x >= 0.0, cdf_pos, 1.0 - cdf_pos)
    return np.asarray(np.clip(out, 0.0, 1.0), dtype=np.float64)


def norm_ppf(p: np.ndarray) -> np.ndarray:
    """
    Acklam inverse-normal approximation (vectorized).
    """
    x = np.asarray(p, dtype=np.float64)
    if np.any(~np.isfinite(x)):
        raise RuntimeError("norm_ppf input contains non-finite values")
    if np.any((x <= 0.0) | (x >= 1.0)):
        raise RuntimeError("norm_ppf input must lie strictly in (0,1)")

    # Coefficients
    a = np.array(
        [
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.383577518672690e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        ],
        dtype=np.float64,
    )
    b = np.array(
        [
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        ],
        dtype=np.float64,
    )
    c = np.array(
        [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        ],
        dtype=np.float64,
    )
    d = np.array(
        [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e00,
            3.754408661907416e00,
        ],
        dtype=np.float64,
    )

    plow = 0.02425
    phigh = 1.0 - plow

    out = np.empty_like(x)
    low = x < plow
    high = x > phigh
    mid = (~low) & (~high)

    if np.any(low):
        q = np.sqrt(-2.0 * np.log(x[low]))
        out[low] = (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )

    if np.any(high):
        q = np.sqrt(-2.0 * np.log(1.0 - x[high]))
        out[high] = -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )

    if np.any(mid):
        q = x[mid] - 0.5
        r = q * q
        out[mid] = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)

    return out


def annualized_sharpe(returns: np.ndarray, periods_per_year: int = 252, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    mu = np.mean(arr, axis=0)
    sd = np.std(arr, axis=0, ddof=1)
    sr_daily = mu / (sd + float(eps))
    return np.sqrt(float(periods_per_year)) * sr_daily


def sample_skew_kurtosis_excess(returns: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    mu = np.mean(arr, axis=0)
    cen = arr - mu[None, :]
    sd = np.std(arr, axis=0, ddof=1)
    z = cen / (sd[None, :] + float(eps))
    skew = np.mean(z**3, axis=0)
    kurt_excess = np.mean(z**4, axis=0) - 3.0
    return skew, kurt_excess


def effective_num_trials_from_corr(returns_matrix: np.ndarray, min_trials: int = 1, eps: float = 1e-12) -> int:
    r = validate_returns_2d(returns_matrix, name="returns_matrix")
    N = int(r.shape[1])
    if N <= 1:
        return 1
    corr = np.corrcoef(r, rowvar=False)
    corr = np.asarray(corr, dtype=np.float64)
    if corr.shape != (N, N):
        raise RuntimeError("Invalid correlation shape in effective_num_trials_from_corr")
    corr = np.nan_to_num(corr, nan=1.0, posinf=1.0, neginf=-1.0)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)

    eig = np.linalg.eigvalsh(corr).astype(np.float64, copy=False)
    eig = np.clip(eig, 0.0, np.inf)
    s1 = float(np.sum(eig))
    s2 = float(np.sum(eig * eig))
    if (not np.isfinite(s1)) or (not np.isfinite(s2)) or s1 <= 0.0:
        return int(max(int(min_trials), 1))

    pr = (s1 * s1) / (s2 + float(eps))
    n_eff = int(np.rint(pr))
    n_eff = max(int(min_trials), min(N, max(1, n_eff)))
    return int(n_eff)


def expected_max_z(n_trials: int) -> float:
    n = int(n_trials)
    if n <= 1:
        return 0.0
    eps = 1e-12
    p1 = 1.0 - 1.0 / float(n)
    p2 = 1.0 - math.exp(-1.0) / float(n)
    p1 = float(np.clip(p1, eps, 1.0 - eps))
    p2 = float(np.clip(p2, eps, 1.0 - eps))
    z1 = float(norm_ppf(np.array([p1], dtype=np.float64))[0])
    z2 = float(norm_ppf(np.array([p2], dtype=np.float64))[0])
    return (1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2


def _psr_denom_sq(sr: np.ndarray, sk: np.ndarray, ke: np.ndarray) -> np.ndarray:
    return 1.0 - sk * sr + ((ke + 2.0) / 4.0) * (sr * sr)


def psr_against_threshold(
    sr_daily: np.ndarray,
    sr_star_daily: np.ndarray,
    skew: np.ndarray,
    kurt_excess: np.ndarray,
    n_obs: int | np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Probabilistic Sharpe Ratio in daily-space (UNANNUALIZED Sharpe).
    kurt_excess is converted to standard-kurtosis term via:
      (gamma4 - 1)/4 = (kurt_excess + 2)/4
    """
    sr = np.asarray(sr_daily, dtype=np.float64)
    sr_star = np.asarray(sr_star_daily, dtype=np.float64)
    sk = np.asarray(skew, dtype=np.float64)
    ke = np.asarray(kurt_excess, dtype=np.float64)
    n = np.asarray(n_obs, dtype=np.float64)

    if sr.shape != sk.shape or sr.shape != ke.shape:
        raise RuntimeError("psr_against_threshold shape mismatch among sr/skew/kurt_excess")
    if sr_star.ndim == 0:
        sr_star = np.full(sr.shape, float(sr_star), dtype=np.float64)
    if sr_star.shape != sr.shape:
        raise RuntimeError("psr_against_threshold sr_star_daily shape mismatch")
    if n.ndim == 0:
        n = np.full(sr.shape, float(n), dtype=np.float64)
    if n.shape != sr.shape:
        raise RuntimeError("psr_against_threshold n_obs shape mismatch")

    denom_sq = _psr_denom_sq(sr=sr, sk=sk, ke=ke)
    bad = (denom_sq <= 0.0) | (~np.isfinite(denom_sq))
    denom = np.sqrt(np.where(bad, 1.0, denom_sq))
    z = (sr - sr_star) * np.sqrt(np.maximum(n - 1.0, 1.0)) / np.maximum(denom, float(eps))
    out = norm_cdf(z)
    out = np.clip(out, 0.0, 1.0)
    out = np.where(bad, 0.0, out)
    return out.astype(np.float64, copy=False)


def deflated_sharpe_ratio(
    returns_matrix: np.ndarray,
    n_trials: int | None = None,
    periods_per_year: int = 252,
    eps: float = 1e-12,
) -> dict[str, np.ndarray | float | int]:
    r = validate_returns_2d(returns_matrix, name="returns_matrix")
    T, N = r.shape

    mu = np.mean(r, axis=0)
    sd = np.std(r, axis=0, ddof=1)
    sr_daily = mu / (sd + float(eps))
    sr_ann = np.sqrt(float(periods_per_year)) * sr_daily
    skew, kurt_excess = sample_skew_kurtosis_excess(r, eps=eps)

    if n_trials is None:
        n_eff = effective_num_trials_from_corr(r, min_trials=1, eps=eps)
    else:
        n_eff = _validate_effective_trial_count(n_trials, N=N)

    emz = expected_max_z(n_eff)
    sr_mean = float(np.mean(sr_daily))
    sr_std = float(np.std(sr_daily, ddof=1)) if N > 1 else 0.0
    sr_star_daily = sr_mean + sr_std * emz
    denom_sq = _psr_denom_sq(sr=sr_daily, sk=skew, ke=kurt_excess)
    denom_bad = (denom_sq <= 0.0) | (~np.isfinite(denom_sq))

    dsr = psr_against_threshold(
        sr_daily=sr_daily,
        sr_star_daily=np.full(N, sr_star_daily, dtype=np.float64),
        skew=skew,
        kurt_excess=kurt_excess,
        n_obs=T,
        eps=eps,
    )

    return {
        "sharpe_daily": sr_daily,
        "sharpe_ann": sr_ann,
        "skew": skew,
        "kurt_excess": kurt_excess,
        "n_trials_effective": int(n_eff),
        "expected_max_z": float(emz),
        "sharpe_deflated_threshold_daily": float(sr_star_daily),
        "dsr": dsr,
        "psr_denom_sq_min": float(np.min(denom_sq)),
        "psr_denom_bad_count": int(np.sum(denom_bad)),
    }


def build_cscv_slices(T: int, S: int = 10) -> np.ndarray:
    t = int(T)
    s = int(S)
    if t < 3:
        raise RuntimeError("T must be >=3")
    if s < 2:
        raise RuntimeError("S must be >=2")
    if s > t:
        raise RuntimeError(f"S must be <= T, got S={s}, T={t}")
    edges = np.floor(np.linspace(0, t, s + 1)).astype(np.int64)
    edges[-1] = t
    starts = edges[:-1]
    ends = edges[1:]
    if np.any(ends <= starts):
        raise RuntimeError("Invalid CSCV slice construction (empty slice detected)")
    return np.stack([starts, ends], axis=1)


def build_cscv_incidence(S: int = 10, k: int = 5) -> np.ndarray:
    s = int(S)
    kk = int(k)
    if s < 2:
        raise RuntimeError("S must be >=2")
    if kk < 1 or kk >= s:
        raise RuntimeError(f"k must be in [1, S-1], got k={kk}, S={s}")
    if s > 62:
        raise RuntimeError("S > 62 not supported by uint64 bit-incidence implementation")

    max_mask = np.uint64(1) << np.uint64(s)
    vals = np.arange(max_mask, dtype=np.uint64)
    bits = ((vals[:, None] >> np.arange(s, dtype=np.uint64)[None, :]) & np.uint64(1)).astype(np.uint8)
    keep = np.sum(bits, axis=1) == kk
    inc = bits[keep].astype(bool)
    if inc.size == 0:
        raise RuntimeError("No CSCV combinations generated")
    return inc


def _validate_effective_trial_count(n_trials_effective: Any, *, N: int) -> int:
    if np.ndim(n_trials_effective) != 0:
        raise RuntimeError("n_trials_effective must be a scalar integer-like value")
    if isinstance(n_trials_effective, (bool, np.bool_)):
        raise RuntimeError("n_trials_effective must not be bool")

    scalar = np.asarray(n_trials_effective)
    if np.issubdtype(scalar.dtype, np.integer):
        n_eff = int(scalar)
    elif np.issubdtype(scalar.dtype, np.floating):
        value = float(scalar)
        if not np.isfinite(value) or not float(value).is_integer():
            raise RuntimeError("n_trials_effective must be integer-like")
        n_eff = int(value)
    else:
        raise RuntimeError("n_trials_effective must be integer-like")

    if n_eff <= 0:
        raise RuntimeError(f"n_trials_effective must be >=1, got {n_eff}")
    if n_eff > int(N):
        raise RuntimeError(f"n_trials_effective must be <= candidate count N={int(N)}, got {n_eff}")
    return int(n_eff)


def pbo_cscv(
    returns_matrix: np.ndarray,
    S: int = 10,
    k: int = 5,
    n_trials_effective: int | None = None,
    periods_per_year: int = 252,
    eps: float = 1e-12,
) -> dict[str, np.ndarray | float | int]:
    r = validate_returns_2d(returns_matrix, name="returns_matrix")
    T, N = r.shape
    if N == 1:
        return {
            "pbo": float("nan"),
            "lambda_logits": np.array([], dtype=np.float64),
            "u_percentiles": np.array([], dtype=np.float64),
            "is_best_idx": np.array([], dtype=np.int64),
            "oos_rank_of_is_best": np.array([], dtype=np.int64),
            "n_combinations": 0,
            "S": int(S),
            "k": int(k),
            "n_trials_effective_used": 1,
            "insufficient_candidates": True,
        }

    if n_trials_effective is None:
        n_eff_used = int(N)
    else:
        n_eff_used = _validate_effective_trial_count(n_trials_effective, N=N)

    s_int = int(S)
    k_int = int(k)
    try:
        comb = int(math.comb(s_int, k_int))
    except ValueError as exc:
        raise RuntimeError(f"Invalid PBO parameters S={s_int}, k={k_int}") from exc
    if comb > 100_000:
        raise RuntimeError("PBO combinatorial space too large (>100k). Reduce S or adjust k.")
    if comb * s_int > 5_000_000:
        raise RuntimeError("PBO incidence matrix too large (comb*S > 5,000,000). Reduce S or adjust k.")

    slices = build_cscv_slices(T=T, S=S)
    inc_test = build_cscv_incidence(S=S, k=k).astype(np.float64)  # (M,S)
    inc_is = 1.0 - inc_test
    M = inc_test.shape[0]

    # Slice sums/sumsq
    cs = np.vstack([np.zeros((1, N), dtype=np.float64), np.cumsum(r, axis=0)])
    cs2 = np.vstack([np.zeros((1, N), dtype=np.float64), np.cumsum(r * r, axis=0)])
    starts = slices[:, 0]
    ends = slices[:, 1]
    lens = (ends - starts).astype(np.float64)  # (S,)
    sl_sum = cs[ends] - cs[starts]  # (S,N)
    sl_s2 = cs2[ends] - cs2[starts]  # (S,N)

    n_test = inc_test @ lens[:, None]  # (M,1)
    n_is = inc_is @ lens[:, None]  # (M,1)
    sum_test = inc_test @ sl_sum  # (M,N)
    sum_is = inc_is @ sl_sum
    s2_test = inc_test @ sl_s2
    s2_is = inc_is @ sl_s2

    mu_test = sum_test / np.maximum(n_test, 1.0)
    mu_is = sum_is / np.maximum(n_is, 1.0)

    var_test = (s2_test - (sum_test * sum_test) / np.maximum(n_test, 1.0)) / np.maximum(n_test - 1.0, 1.0)
    var_is = (s2_is - (sum_is * sum_is) / np.maximum(n_is, 1.0)) / np.maximum(n_is - 1.0, 1.0)
    sd_test = np.sqrt(np.maximum(var_test, 0.0))
    sd_is = np.sqrt(np.maximum(var_is, 0.0))

    scale = np.sqrt(float(periods_per_year))
    sr_test = scale * mu_test / (sd_test + float(eps))
    sr_is = scale * mu_is / (sd_is + float(eps))

    is_best_idx = np.argmax(sr_is, axis=1).astype(np.int64)  # (M,)

    # OOS ranks (1=best), stable via mergesort.
    order = np.argsort(-sr_test, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.int64)
    rows = np.arange(M)[:, None]
    ranks[rows, order] = np.arange(1, N + 1, dtype=np.int64)[None, :]
    oos_rank = ranks[np.arange(M, dtype=np.int64), is_best_idx]  # (M,)

    u = oos_rank.astype(np.float64) / float(n_eff_used + 1)
    u_clip = np.clip(u, float(eps), 1.0 - float(eps))
    lam = np.log(u_clip / (1.0 - u_clip))
    pbo = float(np.mean(lam <= 0.0))

    return {
        "pbo": pbo,
        "lambda_logits": lam,
        "u_percentiles": u,
        "is_best_idx": is_best_idx,
        "oos_rank_of_is_best": oos_rank,
        "n_combinations": int(M),
        "S": int(S),
        "k": int(k),
        "n_trials_effective_used": int(n_eff_used),
        "insufficient_candidates": False,
    }


def stationary_bootstrap_indices(
    T: int,
    B: int = 2000,
    avg_block_len: int = 20,
    seed: int = 47,
) -> np.ndarray:
    t = int(T)
    b = int(B)
    l = int(avg_block_len)
    if t < 3:
        raise RuntimeError("T must be >=3")
    if b < 1:
        raise RuntimeError("B must be >=1")
    if l < 1:
        raise RuntimeError("avg_block_len must be >=1")
    if t > 10_000:
        raise RuntimeError(f"T exceeds guard in stationary_bootstrap_indices: T={t} > 10000")
    if b * t > 25_000_000:
        raise RuntimeError(
            "stationary_bootstrap_indices workload too large (B*T > 25,000,000). Reduce B or T."
        )

    rng = np.random.default_rng(int(seed))
    p_restart = 1.0 / float(l)

    J = rng.random((b, t)) < p_restart
    J[:, 0] = True
    U = rng.integers(0, t, size=(b, t), dtype=np.int64)

    idx = np.empty((b, t), dtype=np.int64)
    idx[:, 0] = U[:, 0]
    for i in range(1, t):
        cont = idx[:, i - 1] + 1
        cont = np.where(cont >= t, 0, cont)
        idx[:, i] = np.where(J[:, i], U[:, i], cont)
    return idx


def bootstrap_means_2d(x_tn: np.ndarray, idx_bt: np.ndarray) -> np.ndarray:
    x = validate_returns_2d(x_tn, name="x_tn")
    idx = np.asarray(idx_bt, dtype=np.int64)
    if idx.ndim != 2:
        raise RuntimeError(f"idx_bt must be 2D, got ndim={idx.ndim}")
    B, T = idx.shape
    if T != x.shape[0]:
        raise RuntimeError(f"idx_bt second dimension must equal T={x.shape[0]}, got {T}")
    if np.any(idx < 0) or np.any(idx >= x.shape[0]):
        raise RuntimeError("idx_bt contains out-of-range indices")
    # Memory-safe path: avoid materializing (B,T,N).
    # We iterate over candidate axis only and keep temporary gather at (B,T).
    _, N = x.shape
    out = np.empty((B, N), dtype=np.float64)
    for j in range(N):
        xj = x[:, j]
        out[:, j] = np.mean(xj[idx], axis=1)
    return out


def white_reality_check(
    returns_matrix: np.ndarray,
    benchmark_returns: np.ndarray,
    B: int = 2000,
    avg_block_len: int = 20,
    seed: int = 47,
    eps: float = 1e-12,
) -> dict[str, np.ndarray | float]:
    r = validate_returns_2d(returns_matrix, name="returns_matrix")
    bmk = validate_returns_1d(benchmark_returns, name="benchmark_returns")
    T, N = r.shape
    if bmk.shape[0] != T:
        raise RuntimeError("benchmark_returns length mismatch")
    if N < 1:
        raise RuntimeError("returns_matrix must have N>=1")

    d = r - bmk[:, None]
    mu = np.mean(d, axis=0)  # (N,)
    d_centered = d - mu[None, :]
    if not np.all(np.isfinite(d_centered)):
        raise RuntimeError("white_reality_check encountered non-finite centered differentials")

    t_obs = float(np.sqrt(float(T)) * np.max(mu))
    if not np.isfinite(t_obs):
        t_obs = np.inf

    idx = stationary_bootstrap_indices(T=T, B=B, avg_block_len=avg_block_len, seed=seed)
    mu_boot = bootstrap_means_2d(d_centered, idx)  # (B,N)
    if not np.all(np.isfinite(mu_boot)):
        raise RuntimeError("white_reality_check encountered non-finite bootstrap means")
    t_boot = np.sqrt(float(T)) * np.max(mu_boot, axis=1)
    t_boot = np.where(np.isfinite(t_boot), t_boot, np.inf)
    p_value = float(np.mean(t_boot >= t_obs - float(eps)))
    if not np.isfinite(p_value):
        p_value = 1.0

    return {
        "t_obs": t_obs,
        "t_boot": t_boot,
        "p_value": p_value,
        "mean_diff": mu,
    }


def spa_test(
    returns_matrix: np.ndarray,
    benchmark_returns: np.ndarray,
    B: int = 2000,
    avg_block_len: int = 20,
    seed: int = 47,
    eps: float = 1e-12,
) -> dict[str, np.ndarray | float]:
    r = validate_returns_2d(returns_matrix, name="returns_matrix")
    bmk = validate_returns_1d(benchmark_returns, name="benchmark_returns")
    T, N = r.shape
    if bmk.shape[0] != T:
        raise RuntimeError("benchmark_returns length mismatch")
    if N < 1:
        raise RuntimeError("returns_matrix must have N>=1")

    d = r - bmk[:, None]
    mu = np.mean(d, axis=0)
    sd = np.std(d, axis=0, ddof=1)
    sd_zero = sd <= float(eps)
    sd_safe = np.where(sd_zero, float(eps), sd)
    t_i_obs = np.sqrt(float(T)) * mu / sd_safe
    t_i_obs = np.where(sd_zero & (mu == 0.0), 0.0, t_i_obs)
    t_i_obs = np.where(np.isfinite(t_i_obs), t_i_obs, 0.0)
    t_obs = float(np.maximum(0.0, np.max(t_i_obs)))

    d_centered = d - mu[None, :]
    idx = stationary_bootstrap_indices(T=T, B=B, avg_block_len=avg_block_len, seed=seed)
    mu_boot = bootstrap_means_2d(d_centered, idx)
    t_i_boot = np.sqrt(float(T)) * mu_boot / sd_safe[None, :]
    if np.any(sd_zero & (mu == 0.0)):
        t_i_boot[:, sd_zero & (mu == 0.0)] = 0.0
    t_i_boot = np.where(np.isfinite(t_i_boot), t_i_boot, 0.0)
    t_boot = np.maximum(0.0, np.max(t_i_boot, axis=1))
    p_value = float(np.mean(t_boot >= t_obs - float(eps)))
    if not np.isfinite(p_value):
        p_value = 1.0

    return {
        "t_obs": t_obs,
        "t_boot": t_boot,
        "p_value": p_value,
        "t_i_obs": t_i_obs,
    }


def model_confidence_set(
    losses_matrix: np.ndarray,
    alpha: float = 0.10,
    B: int = 2000,
    avg_block_len: int = 20,
    seed: int = 47,
    eps: float = 1e-12,
) -> dict[str, Any]:
    L = validate_returns_2d(losses_matrix, name="losses_matrix")
    T, N = L.shape
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise RuntimeError(f"alpha must be in (0,1), got {a}")

    if N == 1:
        return {
            "survivors": np.array([0], dtype=np.int64),
            "elimination_order": [],
            "step_pvalues": [],
            "step_stats": [],
            "alpha": a,
            "insufficient_candidates": True,
        }

    active = np.arange(N, dtype=np.int64)
    elimination_order: list[int] = []
    step_pvalues: list[float] = []
    step_stats: list[float] = []

    step = 0
    while active.size > 1:
        if active.size <= 1:
            break
        m = int(active.size)
        X = L[:, active]  # (T,m)
        mu = np.mean(X, axis=0)  # (m,)
        d_bar = mu[:, None] - mu[None, :]  # (m,m)
        if not np.all(np.isfinite(d_bar)):
            raise RuntimeError("model_confidence_set encountered non-finite d_bar")

        # Hansen-style studentization:
        # se_ij is estimated from centered bootstrap mean-differentials
        # (d_boot_ij - d_bar_ij), then used consistently for t_obs and t_boot.
        idx = stationary_bootstrap_indices(
            T=T,
            B=B,
            avg_block_len=avg_block_len,
            seed=int(seed + 1009 * step),
        )
        mu_boot = bootstrap_means_2d(X, idx)  # (B,m) bootstrap means of losses
        if not np.all(np.isfinite(mu_boot)):
            raise RuntimeError("model_confidence_set encountered non-finite bootstrap means")
        d_boot = mu_boot[:, :, None] - mu_boot[:, None, :]  # (B,m,m)
        d_tilde = d_boot - d_bar[None, :, :]  # centered bootstrap mean-diffs
        if not np.all(np.isfinite(d_tilde)):
            raise RuntimeError("model_confidence_set encountered non-finite centered bootstrap diffs")

        se = np.std(d_tilde, axis=0, ddof=1)  # (m,m), se of mean-differentials
        if not np.all(np.isfinite(se)):
            raise RuntimeError("model_confidence_set encountered non-finite studentization scale")
        se = np.maximum(se, float(eps))

        t_obs = np.abs(d_bar) / se
        if not np.all(np.isfinite(t_obs)):
            raise RuntimeError("model_confidence_set encountered non-finite observed t-statistics")
        np.fill_diagonal(t_obs, 0.0)
        tr_obs = float(np.max(t_obs))

        t_boot = np.abs(d_tilde / se[None, :, :])
        if not np.all(np.isfinite(t_boot)):
            raise RuntimeError("model_confidence_set encountered non-finite bootstrap t-statistics")
        t_boot[:, np.arange(m), np.arange(m)] = 0.0
        tr_boot = np.max(t_boot, axis=(1, 2))
        p_value = float(np.mean(tr_boot >= tr_obs - float(eps)))
        if not np.isfinite(p_value):
            raise RuntimeError("model_confidence_set produced non-finite p-value")

        step_pvalues.append(p_value)
        step_stats.append(tr_obs)

        if p_value >= a:
            break

        # Eliminate model with highest average relative loss.
        avg_rel = np.sum(d_bar, axis=1) / np.maximum(float(m - 1), 1.0)
        worst_local = int(np.argmax(avg_rel))
        worst_global = int(active[worst_local])
        elimination_order.append(worst_global)
        active = np.delete(active, worst_local)
        step += 1

    return {
        "survivors": active.copy(),
        "elimination_order": elimination_order,
        "step_pvalues": step_pvalues,
        "step_stats": step_stats,
        "alpha": a,
        "insufficient_candidates": False,
    }


def run_full_stats(
    returns_matrix: np.ndarray,
    benchmark: np.ndarray,
    losses: np.ndarray | None = None,
    bootstrap_spec: BootstrapSpec | dict[str, Any] | None = None,
    cpcv_params: dict[str, int] | None = None,
    n_trials_effective: int | None = None,
) -> dict[str, Any]:
    """
    Convenience wrapper to run the full Module 5 statistical battery with one call.

    Args:
    - returns_matrix: (T,N) or (T,) returns.
    - benchmark: (T,) benchmark returns.
    - losses: (T,N) or (T,) losses; if None, defaults to -returns_matrix.
    - bootstrap_spec: BootstrapSpec or dict(B, avg_block_len, seed).
    - cpcv_params: dict(S, k).
    - n_trials_effective: optional effective strategy count used by DSR/PBO.
    """
    r = np.asarray(returns_matrix, dtype=np.float64)
    if r.ndim == 1:
        r = r[:, None]
    r = validate_returns_2d(r, name="returns_matrix")

    bmk = validate_returns_1d(np.asarray(benchmark, dtype=np.float64), name="benchmark")
    if bmk.shape[0] != r.shape[0]:
        raise RuntimeError(
            f"benchmark length mismatch: benchmark={bmk.shape[0]}, returns T={r.shape[0]}"
        )

    if losses is None:
        L = -r
    else:
        L = np.asarray(losses, dtype=np.float64)
        if L.ndim == 1:
            L = L[:, None]
        L = validate_returns_2d(L, name="losses")
        if L.shape != r.shape:
            raise RuntimeError(f"losses shape mismatch: got {L.shape}, expected {r.shape}")

    if bootstrap_spec is None:
        bs = BootstrapSpec()
    elif isinstance(bootstrap_spec, BootstrapSpec):
        bs = bootstrap_spec
    elif isinstance(bootstrap_spec, dict):
        bs = BootstrapSpec(
            B=int(bootstrap_spec.get("B", 2000)),
            avg_block_len=int(bootstrap_spec.get("avg_block_len", 20)),
            seed=int(bootstrap_spec.get("seed", 47)),
        )
    else:
        raise RuntimeError("bootstrap_spec must be None, BootstrapSpec, or dict")

    cpcv = dict(cpcv_params or {})
    S = int(cpcv.get("S", 10))
    k = int(cpcv.get("k", 5))

    dsr = deflated_sharpe_ratio(r, n_trials=n_trials_effective)
    pbo = pbo_cscv(r, S=S, k=k, n_trials_effective=n_trials_effective)
    wrc = white_reality_check(
        r,
        bmk,
        B=int(bs.B),
        avg_block_len=int(bs.avg_block_len),
        seed=int(bs.seed),
    )
    spa = spa_test(
        r,
        bmk,
        B=int(bs.B),
        avg_block_len=int(bs.avg_block_len),
        seed=int(bs.seed + 1),
    )
    mcs = model_confidence_set(
        L,
        alpha=0.10,
        B=int(bs.B),
        avg_block_len=int(bs.avg_block_len),
        seed=int(bs.seed + 2),
    )

    return {
        "bootstrap_spec": {
            "B": int(bs.B),
            "avg_block_len": int(bs.avg_block_len),
            "seed": int(bs.seed),
        },
        "cpcv_params": {"S": int(S), "k": int(k)},
        "dsr": dsr,
        "pbo": pbo,
        "wrc": wrc,
        "spa": spa,
        "mcs": mcs,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    T, N = 600, 8
    ret = rng.normal(0.0004, 0.01, size=(T, N)).astype(np.float64)
    bmk = rng.normal(0.0002, 0.009, size=T).astype(np.float64)
    loss = -ret

    dsr = deflated_sharpe_ratio(ret)
    pbo = pbo_cscv(ret, S=10, k=5)
    wrc = white_reality_check(ret, bmk, B=256, avg_block_len=20, seed=11)
    spa = spa_test(ret, bmk, B=256, avg_block_len=20, seed=11)
    mcs = model_confidence_set(loss, alpha=0.10, B=256, avg_block_len=20, seed=11)

    print("MODULE5_STATS_READY")
    print("DSR_MEAN", float(np.mean(dsr["dsr"])))
    print("PBO", float(pbo["pbo"]))
    print("WRC_P", float(wrc["p_value"]))
    print("SPA_P", float(spa["p_value"]))
    print("MCS_SURVIVORS", mcs["survivors"].tolist())
