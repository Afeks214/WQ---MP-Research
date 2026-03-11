from __future__ import annotations

from typing import Tuple

import numpy as np

from weightiz.module3 import FeatureIdx, Module3Config
from weightiz.shared.validation.dtype_guard import assert_float64
from weightiz.module1.core import ProfileStatIdx, ScoreIdx, TensorState


def _assert_shape(name: str, arr: np.ndarray, expected: Tuple[int, ...]) -> None:
    if arr.shape != expected:
        raise RuntimeError(f"{name} shape mismatch: got {arr.shape}, expected {expected}")


def session_spans(session_id_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    s = np.asarray(session_id_t, dtype=np.int64)
    T = int(s.shape[0])
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]]).astype(np.int64)
    ends = np.r_[starts[1:], T].astype(np.int64)
    return starts, ends


def safe_bin_from_channel(idx_ta: np.ndarray, B: int) -> np.ndarray:
    src = np.asarray(idx_ta, dtype=np.float64)
    out = np.zeros(src.shape, dtype=np.int64)
    m = np.isfinite(src)
    out[m] = np.rint(src[m]).astype(np.int64)
    out = np.clip(out, 0, int(B) - 1)
    return out


def broadcast_ta_to_atw(arr_ta: np.ndarray, W: int) -> np.ndarray:
    x = np.asarray(arr_ta, dtype=np.float64)
    if x.ndim != 2:
        raise RuntimeError(f"arr_ta must be [T,A], got shape={x.shape}")
    T, A = x.shape
    out = np.empty((A, T, W), dtype=np.float64)
    for a in range(A):
        out[a, :, :] = x[:, a][:, None]
    return out


def profile_geometry_from_state(state: TensorState, *, eps: float) -> tuple[np.ndarray, ...]:
    vp = np.asarray(state.vp, dtype=np.float64)
    _, _, B = vp.shape
    x = np.asarray(state.x_grid, dtype=np.float64)

    mass = np.sum(vp, axis=2, keepdims=True)
    p = np.where(mass > float(eps), vp / (mass + float(eps)), 0.0)

    mu = np.sum(p * x[None, None, :], axis=2)
    cen = x[None, None, :] - mu[:, :, None]
    var = np.sum(p * (cen * cen), axis=2)
    sigma = np.sqrt(np.maximum(var, 0.0))
    z = cen / (sigma[:, :, None] + float(eps))

    skew = np.sum(p * (z**3), axis=2)
    kurt = np.sum(p * (z**4), axis=2) - 3.0

    entropy = -np.sum(np.where(p > 0.0, p * np.log(np.maximum(p, float(eps))), 0.0), axis=2)
    entropy /= float(np.log(max(B, 2)))
    entropy = np.clip(entropy, float(eps), 1.0)

    left = vp[:, :, :-2]
    mid = vp[:, :, 1:-1]
    right = vp[:, :, 2:]
    peak_count = np.sum((mid >= left) & (mid >= right), axis=2).astype(np.float64)

    poc_idx = safe_bin_from_channel(state.profile_stats[:, :, int(ProfileStatIdx.IPOC)], B)
    bins = np.arange(B, dtype=np.int64)[None, None, :]
    up = np.sum(np.where(bins > poc_idx[:, :, None], p, 0.0), axis=2)
    dn = np.sum(np.where(bins < poc_idx[:, :, None], p, 0.0), axis=2)
    balance = 1.0 - np.abs(up - dn) / np.maximum(up + dn, float(eps))
    balance = np.clip(balance, 0.0, 1.0)

    return skew, kurt, entropy, balance, peak_count


def build_shared_feature_tensor_from_state(state: TensorState, cfg: Module3Config) -> np.ndarray:
    T = int(state.cfg.T)
    A = int(state.cfg.A)
    B = int(state.cfg.B)
    W = int(len(cfg.structural_windows))
    F = int(FeatureIdx.N_FIELDS)

    _assert_shape("profile_stats", state.profile_stats, (T, A, int(ProfileStatIdx.N_FIELDS)))
    _assert_shape("scores", state.scores, (T, A, int(ScoreIdx.N_FIELDS)))
    _assert_shape("bar_valid", state.bar_valid, (T, A))
    _assert_shape("vp", state.vp, (T, A, B))
    assert_float64("module3.bridge.input.profile_stats", state.profile_stats)
    assert_float64("module3.bridge.input.scores", state.scores)
    assert_float64("module3.bridge.input.vp", state.vp)

    shared = np.full((A, T, F, W), np.nan, dtype=np.float64)

    ipoc_i = safe_bin_from_channel(state.profile_stats[:, :, int(ProfileStatIdx.IPOC)], B)
    ivah_i = safe_bin_from_channel(state.profile_stats[:, :, int(ProfileStatIdx.IVAH)], B)
    ival_i = safe_bin_from_channel(state.profile_stats[:, :, int(ProfileStatIdx.IVAL)], B)

    x = np.asarray(state.x_grid, dtype=np.float64)
    x_poc = x[ipoc_i]
    x_vah = x[ivah_i]
    x_val = x[ival_i]

    mu_prof = np.asarray(state.profile_stats[:, :, int(ProfileStatIdx.MU_PROF)], dtype=np.float64)
    sigma_prof = np.asarray(state.profile_stats[:, :, int(ProfileStatIdx.SIGMA_PROF)], dtype=np.float64)

    skew, kurt, entropy, balance, peak_count = profile_geometry_from_state(state, eps=float(cfg.eps))
    va_width = np.maximum(x_vah - x_val, 0.0)
    poc_dist = (x_poc - mu_prof) / np.maximum(va_width, float(cfg.eps))

    feature_ta = {
        FeatureIdx.DCLIP: state.profile_stats[:, :, int(ProfileStatIdx.DCLIP)],
        FeatureIdx.A_AFFINITY: state.profile_stats[:, :, int(ProfileStatIdx.A_AFFINITY)],
        FeatureIdx.Z_DELTA: state.profile_stats[:, :, int(ProfileStatIdx.Z_DELTA)],
        FeatureIdx.GBREAK: state.profile_stats[:, :, int(ProfileStatIdx.GBREAK)],
        FeatureIdx.GREJECT: state.profile_stats[:, :, int(ProfileStatIdx.GREJECT)],
        FeatureIdx.DELTA_EFF: state.profile_stats[:, :, int(ProfileStatIdx.DELTA_EFF)],
        FeatureIdx.SCORE_BO_LONG: state.scores[:, :, int(ScoreIdx.SCORE_BO_LONG)],
        FeatureIdx.SCORE_BO_SHORT: state.scores[:, :, int(ScoreIdx.SCORE_BO_SHORT)],
        FeatureIdx.SCORE_REJECT: state.scores[:, :, int(ScoreIdx.SCORE_REJECT)],
        FeatureIdx.X_POC: x_poc,
        FeatureIdx.X_VAH: x_vah,
        FeatureIdx.X_VAL: x_val,
        FeatureIdx.MU_PROF: mu_prof,
        FeatureIdx.SIGMA_PROF: sigma_prof,
        FeatureIdx.SKEW_PROF: skew,
        FeatureIdx.KURT_PROF: kurt,
        FeatureIdx.PROFILE_ENTROPY: entropy,
        FeatureIdx.PROFILE_BALANCE_RATIO: balance,
        FeatureIdx.PROFILE_PEAK_COUNT: peak_count,
        FeatureIdx.PROFILE_POC_DISTANCE: poc_dist,
        FeatureIdx.PROFILE_VALUE_AREA_WIDTH: va_width,
        FeatureIdx.BAR_VALID: state.bar_valid.astype(np.float64),
        FeatureIdx.IB_HIGH: x_vah,
        FeatureIdx.IB_LOW: x_val,
    }

    for f_idx, arr_ta in feature_ta.items():
        shared[:, :, int(f_idx), :] = broadcast_ta_to_atw(np.asarray(arr_ta, dtype=np.float64), W)

    shared = np.where(np.isfinite(shared), shared, 0.0).astype(np.float64, copy=False)
    return shared


def resolve_shared_feature_tensor(state: TensorState, cfg: Module3Config) -> np.ndarray:
    candidate = getattr(state, "shared_feature_tensor", None)
    if candidate is not None:
        shared = np.asarray(candidate, dtype=np.float64)
        if shared.ndim != 4:
            raise RuntimeError(
                f"state.shared_feature_tensor must be [A,T,F,W], got shape={shared.shape}"
            )
        if shared.shape[0] != int(state.cfg.A) or shared.shape[1] != int(state.cfg.T):
            raise RuntimeError(
                "state.shared_feature_tensor shape mismatch with TensorState dimensions"
            )
        if shared.shape[3] != int(len(cfg.structural_windows)):
            raise RuntimeError(
                "state.shared_feature_tensor W axis does not match Module3Config.structural_windows"
            )
        return shared
    return build_shared_feature_tensor_from_state(state, cfg)


def to_ta_from_at(arr_at: np.ndarray) -> np.ndarray:
    x = np.asarray(arr_at)
    if x.ndim != 2:
        raise RuntimeError(f"arr_at must be [A,T], got shape={x.shape}")
    A, T = x.shape
    out = np.empty((T, A), dtype=x.dtype)
    for a in range(A):
        out[:, a] = x[a, :]
    return out


def to_tac_from_atc(arr_atc: np.ndarray) -> np.ndarray:
    x = np.asarray(arr_atc)
    if x.ndim != 3:
        raise RuntimeError(f"arr_atc must be [A,T,C], got shape={x.shape}")
    A, T, C = x.shape
    out = np.empty((T, A, C), dtype=x.dtype)
    for a in range(A):
        out[:, a, :] = x[a, :, :]
    return out
