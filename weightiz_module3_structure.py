"""
Weightiz Institutional Engine - Module 3 Bridge
===============================================

Public Module3 API bridge that routes execution through the refactored
multi-window Module3 package while preserving legacy downstream fields.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from module3 import (
    ContextIdx,
    FeatureIdx,
    Module3Config,
    Module3Output,
    StructIdx,
    deterministic_digest_sha256_module3 as _deterministic_digest_sha256_module3,
    resolve_window_index,
    run_structural_window_engine,
    validate_output_contract,
)
from weightiz_dtype_guard import assert_float64
from weightiz_module1_core import ProfileStatIdx, ScoreIdx, TensorState
from weightiz_system_logger import get_logger, log_event

IB_POLICY_NO_TRADE = "NO_TRADE"
IB_POLICY_DEGRADE = "DEGRADE"
IB_MISSING_POLICY = IB_POLICY_NO_TRADE

# Backward-compatible enum aliases used across the repo.
Struct30mIdx = StructIdx


def _assert_shape(name: str, arr: np.ndarray, expected: Tuple[int, ...]) -> None:
    if arr.shape != expected:
        raise RuntimeError(f"{name} shape mismatch: got {arr.shape}, expected {expected}")


def _session_spans(session_id_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    s = np.asarray(session_id_t, dtype=np.int64)
    T = int(s.shape[0])
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]]).astype(np.int64)
    ends = np.r_[starts[1:], T].astype(np.int64)
    return starts, ends


def _safe_bin_from_channel(idx_ta: np.ndarray, B: int) -> np.ndarray:
    src = np.asarray(idx_ta, dtype=np.float64)
    out = np.zeros(src.shape, dtype=np.int64)
    m = np.isfinite(src)
    out[m] = np.rint(src[m]).astype(np.int64)
    out = np.clip(out, 0, int(B) - 1)
    return out


def _broadcast_ta_to_atw(arr_ta: np.ndarray, W: int) -> np.ndarray:
    x = np.asarray(arr_ta, dtype=np.float64)
    if x.ndim != 2:
        raise RuntimeError(f"arr_ta must be [T,A], got shape={x.shape}")
    T, A = x.shape
    out = np.empty((A, T, W), dtype=np.float64)
    for a in range(A):
        out[a, :, :] = x[:, a][:, None]
    return out


def _profile_geometry_from_state(state: TensorState, *, eps: float) -> tuple[np.ndarray, ...]:
    vp = np.asarray(state.vp, dtype=np.float64)  # [T,A,B]
    T, A, B = vp.shape
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

    poc_idx = _safe_bin_from_channel(state.profile_stats[:, :, int(ProfileStatIdx.IPOC)], B)
    bins = np.arange(B, dtype=np.int64)[None, None, :]
    up = np.sum(np.where(bins > poc_idx[:, :, None], p, 0.0), axis=2)
    dn = np.sum(np.where(bins < poc_idx[:, :, None], p, 0.0), axis=2)
    balance = 1.0 - np.abs(up - dn) / np.maximum(up + dn, float(eps))
    balance = np.clip(balance, 0.0, 1.0)

    return skew, kurt, entropy, balance, peak_count


def _build_shared_feature_tensor_from_state(state: TensorState, cfg: Module3Config) -> np.ndarray:
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

    ipoc_i = _safe_bin_from_channel(state.profile_stats[:, :, int(ProfileStatIdx.IPOC)], B)
    ivah_i = _safe_bin_from_channel(state.profile_stats[:, :, int(ProfileStatIdx.IVAH)], B)
    ival_i = _safe_bin_from_channel(state.profile_stats[:, :, int(ProfileStatIdx.IVAL)], B)

    x = np.asarray(state.x_grid, dtype=np.float64)
    x_poc = x[ipoc_i]
    x_vah = x[ivah_i]
    x_val = x[ival_i]

    mu_prof = np.asarray(state.profile_stats[:, :, int(ProfileStatIdx.MU_PROF)], dtype=np.float64)
    sigma_prof = np.asarray(state.profile_stats[:, :, int(ProfileStatIdx.SIGMA_PROF)], dtype=np.float64)

    skew, kurt, entropy, balance, peak_count = _profile_geometry_from_state(state, eps=float(cfg.eps))
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
        shared[:, :, int(f_idx), :] = _broadcast_ta_to_atw(np.asarray(arr_ta, dtype=np.float64), W)

    shared = np.where(np.isfinite(shared), shared, 0.0).astype(np.float64, copy=False)
    return shared


def _resolve_shared_feature_tensor(state: TensorState, cfg: Module3Config) -> np.ndarray:
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
    return _build_shared_feature_tensor_from_state(state, cfg)


def _build_legacy_block_maps(state: TensorState, selected_window: int) -> tuple[np.ndarray, ...]:
    T = int(state.cfg.T)
    window = max(1, int(selected_window))
    sid = np.asarray(state.session_id, dtype=np.int64)

    block_seq_t = np.full(T, -1, dtype=np.int16)
    block_id_t = np.full(T, -1, dtype=np.int64)
    block_end_flag_t = np.zeros(T, dtype=bool)
    block_start_t_index_t = np.full(T, -1, dtype=np.int64)
    block_end_t_index_t = np.full(T, -1, dtype=np.int64)

    starts, ends = _session_spans(sid)
    for s0, s1 in zip(starts.tolist(), ends.tolist()):
        n = int(s1 - s0)
        local = np.arange(n, dtype=np.int64)
        seq = (local // window).astype(np.int16)
        block_seq_t[int(s0) : int(s1)] = seq
        block_id_t[int(s0) : int(s1)] = sid[int(s0)] * np.int64(4096) + seq.astype(np.int64)

        end_mask = ((local + 1) % window == 0)
        if n > 0:
            end_mask[-1] = True
        end_local = np.flatnonzero(end_mask).astype(np.int64)
        end_global = end_local + int(s0)
        block_end_flag_t[end_global] = True
        block_end_t_index_t[end_global] = end_global

        for eg in end_global.tolist():
            start = max(int(s0), int(eg) - window + 1)
            block_start_t_index_t[int(eg)] = int(start)

    return block_id_t, block_seq_t, block_end_flag_t, block_start_t_index_t, block_end_t_index_t


def _to_ta_from_at(arr_at: np.ndarray) -> np.ndarray:
    x = np.asarray(arr_at)
    if x.ndim != 2:
        raise RuntimeError(f"arr_at must be [A,T], got shape={x.shape}")
    A, T = x.shape
    out = np.empty((T, A), dtype=x.dtype)
    for a in range(A):
        out[:, a] = x[a, :]
    return out


def _to_tac_from_atc(arr_atc: np.ndarray) -> np.ndarray:
    x = np.asarray(arr_atc)
    if x.ndim != 3:
        raise RuntimeError(f"arr_atc must be [A,T,C], got shape={x.shape}")
    A, T, C = x.shape
    out = np.empty((T, A, C), dtype=x.dtype)
    for a in range(A):
        out[:, a, :] = x[a, :, :]
    return out


def run_module3_structural_aggregation(state: TensorState, cfg: Module3Config) -> Module3Output:
    """Public Module3 entrypoint. Builds shared tensor bridge and runs new engine."""
    shared_feature_tensor = _resolve_shared_feature_tensor(state, cfg)
    out = run_structural_window_engine(
        shared_feature_tensor=shared_feature_tensor,
        session_id_t=np.asarray(state.session_id, dtype=np.int64),
        cfg=cfg,
    )

    w_idx = resolve_window_index(int(cfg.selected_window), cfg.structural_windows)

    selected_struct = np.asarray(out.structure_tensor[:, :, :, w_idx], dtype=np.float64)
    selected_ctx = np.asarray(out.context_tensor[:, :, :, w_idx], dtype=np.float64)

    block_features_tak = _to_tac_from_atc(selected_struct)
    context_tac = _to_tac_from_atc(selected_ctx)

    valid_ratio_at = selected_struct[:, :, int(Struct30mIdx.VALID_RATIO)]
    block_valid_ta = _to_ta_from_at(np.isfinite(valid_ratio_at) & (valid_ratio_at >= float(cfg.min_block_valid_ratio)))

    if out.context_valid_atw is not None:
        context_valid_ta = _to_ta_from_at(out.context_valid_atw[:, :, w_idx])
    else:
        context_valid_ta = np.all(np.isfinite(context_tac), axis=2)

    if out.context_source_index_atw is not None:
        context_source_t_index_ta = _to_ta_from_at(out.context_source_index_atw[:, :, w_idx]).astype(np.int64)
    else:
        T, A = context_valid_ta.shape
        context_source_t_index_ta = np.full((T, A), -1, dtype=np.int64)
        for a in range(A):
            last = -1
            for t in range(T):
                if bool(context_valid_ta[t, a]):
                    last = t
                context_source_t_index_ta[t, a] = int(last)

    block_id_t, block_seq_t, block_end_flag_t, block_start_t_index_t, block_end_t_index_t = _build_legacy_block_maps(
        state, int(cfg.selected_window)
    )

    ib_hi = block_features_tak[:, :, int(Struct30mIdx.IB_HIGH_X)]
    ib_lo = block_features_tak[:, :, int(Struct30mIdx.IB_LOW_X)]
    ib_defined_ta = np.isfinite(ib_hi) & np.isfinite(ib_lo)

    out.block_id_t = block_id_t
    out.block_seq_t = block_seq_t
    out.block_end_flag_t = block_end_flag_t
    out.block_start_t_index_t = block_start_t_index_t
    out.block_end_t_index_t = block_end_t_index_t
    out.block_features_tak = block_features_tak
    out.block_valid_ta = block_valid_ta
    out.context_tac = context_tac
    out.context_valid_ta = context_valid_ta
    out.context_source_t_index_ta = context_source_t_index_ta
    out.ib_defined_ta = ib_defined_ta

    if bool(cfg.validate_outputs):
        validate_module3_output(state, out, cfg)
    return out


def run_module3(state: TensorState, cfg: Module3Config) -> Module3Output:
    return run_module3_structural_aggregation(state, cfg)


def validate_module3_output(state: TensorState, out: Module3Output, cfg: Module3Config) -> None:
    validate_output_contract(out)

    T = int(state.cfg.T)
    A = int(state.cfg.A)

    _assert_shape("structure_tensor", out.structure_tensor, (A, T, int(Struct30mIdx.N_FIELDS), len(cfg.structural_windows)))
    _assert_shape("context_tensor", out.context_tensor, (A, T, int(ContextIdx.N_FIELDS), len(cfg.structural_windows)))

    if out.block_features_tak is not None:
        _assert_shape("block_features_tak", out.block_features_tak, (T, A, int(Struct30mIdx.N_FIELDS)))
    if out.block_valid_ta is not None:
        _assert_shape("block_valid_ta", out.block_valid_ta, (T, A))
    if out.context_tac is not None:
        _assert_shape("context_tac", out.context_tac, (T, A, int(ContextIdx.N_FIELDS)))
    if out.context_valid_ta is not None:
        _assert_shape("context_valid_ta", out.context_valid_ta, (T, A))
    if out.context_source_t_index_ta is not None:
        _assert_shape("context_source_t_index_ta", out.context_source_t_index_ta, (T, A))


def deterministic_digest_sha256_module3_bridge(out: Module3Output) -> str:
    return _deterministic_digest_sha256_module3(out)


# Backward-compatible exported name.
deterministic_digest_sha256_module3 = deterministic_digest_sha256_module3_bridge


if __name__ == "__main__":
    log_event(get_logger("module3"), "INFO", "module3_ready", event_type="module3_ready")
