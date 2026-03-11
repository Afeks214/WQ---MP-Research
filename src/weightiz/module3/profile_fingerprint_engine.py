from __future__ import annotations

import numpy as np

from .schema import FingerprintIdx, RegimeCode
from .structural_kernels import (
    compute_poc_distance,
    compute_profile_balance_ratio,
    compute_profile_entropy,
    compute_profile_kurtosis,
    compute_profile_peak_count,
    compute_profile_skew,
    compute_value_area_width,
)


def compute_profile_fingerprint_tensor(
    shared_feature_tensor: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute profile geometry fingerprints for all [A,T,W]."""
    x = np.asarray(shared_feature_tensor, dtype=np.float64)
    if x.ndim != 4:
        raise RuntimeError(f"shared_feature_tensor must be [A,T,F,W], got shape={x.shape}")

    A, T, _, W = x.shape
    out = np.full((A, T, int(FingerprintIdx.N_FIELDS), W), np.nan, dtype=np.float64)
    out[:, :, int(FingerprintIdx.PROFILE_SKEW), :] = compute_profile_skew(x)
    out[:, :, int(FingerprintIdx.PROFILE_KURTOSIS), :] = compute_profile_kurtosis(x)
    out[:, :, int(FingerprintIdx.PROFILE_ENTROPY), :] = compute_profile_entropy(x, eps=float(eps))
    out[:, :, int(FingerprintIdx.PROFILE_BALANCE_RATIO), :] = compute_profile_balance_ratio(x, eps=float(eps))
    out[:, :, int(FingerprintIdx.PROFILE_PEAK_COUNT), :] = compute_profile_peak_count(x)
    out[:, :, int(FingerprintIdx.PROFILE_POC_DISTANCE), :] = compute_poc_distance(x, eps=float(eps))
    out[:, :, int(FingerprintIdx.PROFILE_VALUE_AREA_WIDTH), :] = compute_value_area_width(x)
    return out


def compute_profile_regime_tensor(
    profile_fingerprint_tensor: np.ndarray,
) -> np.ndarray:
    """Deterministic regime classifier encoded as float64."""
    fp = np.asarray(profile_fingerprint_tensor, dtype=np.float64)
    if fp.ndim != 4:
        raise RuntimeError(
            f"profile_fingerprint_tensor must be [A,T,F_fp,W], got shape={fp.shape}"
        )

    skew = fp[:, :, int(FingerprintIdx.PROFILE_SKEW), :]
    kurt = fp[:, :, int(FingerprintIdx.PROFILE_KURTOSIS), :]
    entropy = fp[:, :, int(FingerprintIdx.PROFILE_ENTROPY), :]
    balance = fp[:, :, int(FingerprintIdx.PROFILE_BALANCE_RATIO), :]
    peaks = fp[:, :, int(FingerprintIdx.PROFILE_PEAK_COUNT), :]
    poc_dist = fp[:, :, int(FingerprintIdx.PROFILE_POC_DISTANCE), :]
    va_width = np.maximum(fp[:, :, int(FingerprintIdx.PROFILE_VALUE_AREA_WIDTH), :], 1e-12)

    regime = np.full(skew.shape, float(RegimeCode.BALANCED), dtype=np.float64)

    extreme = (balance < 0.20) | (np.abs(skew) >= 1.20) | (np.abs(poc_dist) >= 2.0)
    double_dist = (peaks >= 2.0) & (kurt < -0.50) & (entropy >= 0.85)
    trend_up = (poc_dist > 0.35 * va_width) & (skew > 0.25)
    trend_down = (poc_dist < -0.35 * va_width) & (skew < -0.25)

    regime[trend_up] = float(RegimeCode.TREND_UP)
    regime[trend_down] = float(RegimeCode.TREND_DOWN)
    regime[double_dist] = float(RegimeCode.DOUBLE_DISTRIBUTION)
    regime[extreme] = float(RegimeCode.EXTREME_IMBALANCE)

    return regime[:, :, None, :].astype(np.float64, copy=False)
