from __future__ import annotations

import numpy as np

from .schema import FeatureIdx


def _assert_shared(shared_feature_tensor: np.ndarray) -> np.ndarray:
    x = np.asarray(shared_feature_tensor)
    if x.ndim != 4:
        raise RuntimeError(f"shared_feature_tensor must be [A,T,F,W], got shape={x.shape}")
    if x.dtype != np.float64:
        raise RuntimeError(f"shared_feature_tensor dtype must be float64, got {x.dtype}")
    return x


def extract_feature_view(shared_feature_tensor: np.ndarray, feature_idx: FeatureIdx) -> np.ndarray:
    x = _assert_shared(shared_feature_tensor)
    f = int(feature_idx)
    if f < 0 or f >= x.shape[2]:
        raise RuntimeError(f"Feature index out of bounds: idx={f}, F={x.shape[2]}")
    return x[:, :, f, :]


def compute_value_area_width(shared_feature_tensor: np.ndarray) -> np.ndarray:
    x_vah = extract_feature_view(shared_feature_tensor, FeatureIdx.X_VAH)
    x_val = extract_feature_view(shared_feature_tensor, FeatureIdx.X_VAL)
    return np.maximum(x_vah - x_val, 0.0)


def compute_poc_distance(shared_feature_tensor: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    x_poc = extract_feature_view(shared_feature_tensor, FeatureIdx.X_POC)
    x_vah = extract_feature_view(shared_feature_tensor, FeatureIdx.X_VAH)
    x_val = extract_feature_view(shared_feature_tensor, FeatureIdx.X_VAL)
    mid = 0.5 * (x_vah + x_val)
    width = np.maximum(x_vah - x_val, float(eps))
    return (x_poc - mid) / width


def compute_profile_skew(shared_feature_tensor: np.ndarray) -> np.ndarray:
    skew = extract_feature_view(shared_feature_tensor, FeatureIdx.SKEW_PROF)
    zdelta = extract_feature_view(shared_feature_tensor, FeatureIdx.Z_DELTA)
    return np.where(np.isfinite(skew), skew, zdelta)


def compute_profile_kurtosis(shared_feature_tensor: np.ndarray) -> np.ndarray:
    kurt = extract_feature_view(shared_feature_tensor, FeatureIdx.KURT_PROF)
    skew = compute_profile_skew(shared_feature_tensor)
    fallback = skew * skew - 1.0
    return np.where(np.isfinite(kurt), kurt, fallback)


def compute_profile_entropy(shared_feature_tensor: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    entropy = extract_feature_view(shared_feature_tensor, FeatureIdx.PROFILE_ENTROPY)
    zdelta = np.abs(extract_feature_view(shared_feature_tensor, FeatureIdx.Z_DELTA))
    fallback = np.exp(-np.maximum(zdelta, 0.0))
    out = np.where(np.isfinite(entropy), entropy, fallback)
    return np.clip(out, float(eps), 1.0)


def compute_profile_balance_ratio(shared_feature_tensor: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    ratio = extract_feature_view(shared_feature_tensor, FeatureIdx.PROFILE_BALANCE_RATIO)
    skew = np.abs(compute_profile_skew(shared_feature_tensor))
    fallback = 1.0 / (1.0 + skew + float(eps))
    out = np.where(np.isfinite(ratio), ratio, fallback)
    return np.clip(out, 0.0, 1.0)


def compute_profile_peak_count(shared_feature_tensor: np.ndarray) -> np.ndarray:
    peaks = extract_feature_view(shared_feature_tensor, FeatureIdx.PROFILE_PEAK_COUNT)
    kurt = np.abs(compute_profile_kurtosis(shared_feature_tensor))
    fallback = 1.0 + (kurt >= 1.0).astype(np.float64)
    out = np.where(np.isfinite(peaks), peaks, fallback)
    return np.maximum(np.rint(out), 0.0)


def compute_tail_imbalance(shared_feature_tensor: np.ndarray) -> np.ndarray:
    zdelta = extract_feature_view(shared_feature_tensor, FeatureIdx.Z_DELTA)
    spread = extract_feature_view(shared_feature_tensor, FeatureIdx.GBREAK) - extract_feature_view(
        shared_feature_tensor, FeatureIdx.GREJECT
    )
    return zdelta * spread


def compute_drift(series_atw: np.ndarray) -> np.ndarray:
    x = np.asarray(series_atw, dtype=np.float64)
    if x.ndim != 3:
        raise RuntimeError(f"series_atw must be [A,T,W], got shape={x.shape}")
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if x.shape[1] > 1:
        out[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
    return out


def compute_poc_vs_prev_va(shared_feature_tensor: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    x_poc = extract_feature_view(shared_feature_tensor, FeatureIdx.X_POC)
    x_vah = extract_feature_view(shared_feature_tensor, FeatureIdx.X_VAH)
    x_val = extract_feature_view(shared_feature_tensor, FeatureIdx.X_VAL)

    prev_vah = np.full(x_vah.shape, np.nan, dtype=np.float64)
    prev_val = np.full(x_val.shape, np.nan, dtype=np.float64)
    if x_vah.shape[1] > 1:
        prev_vah[:, 1:, :] = x_vah[:, :-1, :]
        prev_val[:, 1:, :] = x_val[:, :-1, :]

    width = np.maximum(prev_vah - prev_val, float(eps))
    out = np.zeros(x_poc.shape, dtype=np.float64)
    gt = x_poc > prev_vah
    lt = x_poc < prev_val
    mid = ~(gt | lt)

    out[gt] = 1.0 + (x_poc[gt] - prev_vah[gt]) / width[gt]
    out[lt] = -1.0 - (prev_val[lt] - x_poc[lt]) / width[lt]
    out[mid] = -1.0 + 2.0 * (x_poc[mid] - prev_val[mid]) / width[mid]
    out[:, 0, :] = 0.0
    out[~np.isfinite(out)] = 0.0
    return out
