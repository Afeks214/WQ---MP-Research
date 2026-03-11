from __future__ import annotations

import numpy as np

from .profile_fingerprint_engine import compute_profile_fingerprint_tensor, compute_profile_regime_tensor
from .schema import FeatureIdx, StructIdx
from .structural_context_builder import build_context_tensor
from .structural_kernels import (
    compute_drift,
    compute_poc_distance,
    compute_poc_vs_prev_va,
    compute_profile_kurtosis,
    compute_profile_skew,
    compute_tail_imbalance,
    compute_value_area_width,
    extract_feature_view,
)
from .structural_prefix_sums import (
    build_prefix_count,
    build_prefix_sum,
    rolling_count_from_prefix,
    rolling_mean_from_prefix,
)
from .structural_validation import run_forensic_validation, validate_shared_tensor_contract
from .types import Module3Config, Module3Output


def _session_spans(session_id_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(session_id_t, dtype=np.int64)
    if s.ndim != 1:
        raise RuntimeError(f"session_id_t must be 1D, got shape={s.shape}")
    T = int(s.shape[0])
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]]).astype(np.int64)
    ends = np.r_[starts[1:], T].astype(np.int64)
    return starts, ends


def _compute_ib_cache(shared_feature_tensor: np.ndarray, session_id_t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(shared_feature_tensor, dtype=np.float64)
    A, T, _, W = x.shape
    ib_hi = np.full((A, T, W), np.nan, dtype=np.float64)
    ib_lo = np.full((A, T, W), np.nan, dtype=np.float64)
    ib_defined = np.zeros((A, T, W), dtype=bool)

    hi_src = x[:, :, int(FeatureIdx.IB_HIGH), :]
    lo_src = x[:, :, int(FeatureIdx.IB_LOW), :]

    starts, ends = _session_spans(session_id_t)
    for s0, s1 in zip(starts.tolist(), ends.tolist()):
        seg_hi = hi_src[:, int(s0) : int(s1), :]
        seg_lo = lo_src[:, int(s0) : int(s1), :]
        seed_hi = np.full((A, W), np.nan, dtype=np.float64)
        seed_lo = np.full((A, W), np.nan, dtype=np.float64)

        for a in range(A):
            for w in range(W):
                col_hi = seg_hi[a, :, w]
                col_lo = seg_lo[a, :, w]
                m = np.isfinite(col_hi) & np.isfinite(col_lo)
                if np.any(m):
                    first = int(np.flatnonzero(m)[0])
                    seed_hi[a, w] = float(col_hi[first])
                    seed_lo[a, w] = float(col_lo[first])

        ib_hi[:, int(s0) : int(s1), :] = seed_hi[:, None, :]
        ib_lo[:, int(s0) : int(s1), :] = seed_lo[:, None, :]
        ib_defined[:, int(s0) : int(s1), :] = np.isfinite(seed_hi[:, None, :]) & np.isfinite(seed_lo[:, None, :])

    return ib_hi, ib_lo, ib_defined


def _rolling_mean_for_series(
    series_atw: np.ndarray,
    valid_atw: np.ndarray,
    window: int,
    *,
    eps: float,
) -> np.ndarray:
    s = np.asarray(series_atw, dtype=np.float64)
    v = np.asarray(valid_atw, dtype=bool)
    masked = np.where(v, s, 0.0)
    ps = build_prefix_sum(masked)
    pc = build_prefix_count(v)
    return rolling_mean_from_prefix(ps, pc, int(window), eps=float(eps))


def run_structural_window_engine(
    shared_feature_tensor: np.ndarray,
    session_id_t: np.ndarray,
    cfg: Module3Config,
) -> Module3Output:
    """Deterministic multi-window structural aggregation over [A,T,F,W]."""
    validate_shared_tensor_contract(shared_feature_tensor)
    x = np.asarray(shared_feature_tensor, dtype=np.float64)
    A, T, F, W = x.shape

    windows = tuple(int(w) for w in cfg.structural_windows)
    if W != len(windows):
        raise RuntimeError(
            f"shared_feature_tensor W mismatch: tensor.W={W}, config.structural_windows={windows}"
        )
    if np.asarray(session_id_t).shape != (T,):
        raise RuntimeError(f"session_id_t shape mismatch: got {np.asarray(session_id_t).shape}, expected {(T,)}")

    structure = np.full((A, T, int(StructIdx.N_FIELDS), W), np.nan, dtype=np.float64)

    valid_atw = extract_feature_view(x, FeatureIdx.BAR_VALID) > 0.5
    ib_hi_cache, ib_lo_cache, ib_defined = _compute_ib_cache(x, session_id_t)

    width = compute_value_area_width(x)
    skew = compute_profile_skew(x)
    kurt = compute_profile_kurtosis(x)
    tail = compute_tail_imbalance(x)
    poc_dist_rel = compute_poc_distance(x, eps=float(cfg.eps))
    poc_vs_prev_va = compute_poc_vs_prev_va(x, eps=float(cfg.eps))

    x_poc = extract_feature_view(x, FeatureIdx.X_POC)
    x_vah = extract_feature_view(x, FeatureIdx.X_VAH)
    x_val = extract_feature_view(x, FeatureIdx.X_VAL)

    drift_poc = compute_drift(x_poc)
    drift_vah = compute_drift(x_vah)
    drift_val = compute_drift(x_val)
    delta_eff = extract_feature_view(x, FeatureIdx.DELTA_EFF)
    delta_shift = compute_drift(delta_eff)

    for w_idx, window in enumerate(windows):
        valid_aw = valid_atw[:, :, w_idx : w_idx + 1]

        cnt_prefix = build_prefix_count(valid_aw)
        valid_cnt = rolling_count_from_prefix(cnt_prefix, int(window))[:, :, 0]
        valid_ratio = valid_cnt / float(window)

        structure[:, :, int(StructIdx.VALID_RATIO), w_idx] = valid_ratio
        structure[:, :, int(StructIdx.N_VALID_BARS), w_idx] = valid_cnt

        structure[:, :, int(StructIdx.X_POC), w_idx] = x_poc[:, :, w_idx]
        structure[:, :, int(StructIdx.X_VAH), w_idx] = x_vah[:, :, w_idx]
        structure[:, :, int(StructIdx.X_VAL), w_idx] = x_val[:, :, w_idx]
        structure[:, :, int(StructIdx.VA_WIDTH_X), w_idx] = width[:, :, w_idx]

        structure[:, :, int(StructIdx.MU_ANCHOR), w_idx] = extract_feature_view(x, FeatureIdx.MU_PROF)[:, :, w_idx]
        structure[:, :, int(StructIdx.SIGMA_ANCHOR), w_idx] = extract_feature_view(x, FeatureIdx.SIGMA_PROF)[:, :, w_idx]
        structure[:, :, int(StructIdx.SKEW_ANCHOR), w_idx] = skew[:, :, w_idx]
        structure[:, :, int(StructIdx.KURT_EXCESS_ANCHOR), w_idx] = kurt[:, :, w_idx]
        structure[:, :, int(StructIdx.TAIL_IMBALANCE), w_idx] = tail[:, :, w_idx]

        dclip_series = x[:, :, int(FeatureIdx.DCLIP) : int(FeatureIdx.DCLIP) + 1, w_idx]
        dclip_valid = valid_aw[:, :, 0:1]
        dclip_m = _rolling_mean_for_series(dclip_series, dclip_valid, int(window), eps=float(cfg.eps))[:, :, 0]
        dclip2_m = _rolling_mean_for_series(
            dclip_series * dclip_series, dclip_valid, int(window), eps=float(cfg.eps)
        )[:, :, 0]
        dclip_std = np.sqrt(np.maximum(dclip2_m - dclip_m * dclip_m, 0.0))

        structure[:, :, int(StructIdx.DCLIP_MEAN), w_idx] = dclip_m
        structure[:, :, int(StructIdx.DCLIP_STD), w_idx] = dclip_std
        structure[:, :, int(StructIdx.AFFINITY_MEAN), w_idx] = _rolling_mean_for_series(
            x[:, :, int(FeatureIdx.A_AFFINITY) : int(FeatureIdx.A_AFFINITY) + 1, w_idx], dclip_valid, int(window), eps=float(cfg.eps)
        )[:, :, 0]
        structure[:, :, int(StructIdx.ZDELTA_MEAN), w_idx] = _rolling_mean_for_series(
            x[:, :, int(FeatureIdx.Z_DELTA) : int(FeatureIdx.Z_DELTA) + 1, w_idx], dclip_valid, int(window), eps=float(cfg.eps)
        )[:, :, 0]
        structure[:, :, int(StructIdx.GBREAK_MEAN), w_idx] = _rolling_mean_for_series(
            x[:, :, int(FeatureIdx.GBREAK) : int(FeatureIdx.GBREAK) + 1, w_idx], dclip_valid, int(window), eps=float(cfg.eps)
        )[:, :, 0]
        structure[:, :, int(StructIdx.GREJECT_MEAN), w_idx] = _rolling_mean_for_series(
            x[:, :, int(FeatureIdx.GREJECT) : int(FeatureIdx.GREJECT) + 1, w_idx], dclip_valid, int(window), eps=float(cfg.eps)
        )[:, :, 0]
        structure[:, :, int(StructIdx.DELTA_EFF_MEAN), w_idx] = _rolling_mean_for_series(
            x[:, :, int(FeatureIdx.DELTA_EFF) : int(FeatureIdx.DELTA_EFF) + 1, w_idx], dclip_valid, int(window), eps=float(cfg.eps)
        )[:, :, 0]
        structure[:, :, int(StructIdx.SCORE_BO_LONG_MEAN), w_idx] = _rolling_mean_for_series(
            x[:, :, int(FeatureIdx.SCORE_BO_LONG) : int(FeatureIdx.SCORE_BO_LONG) + 1, w_idx], dclip_valid, int(window), eps=float(cfg.eps)
        )[:, :, 0]
        structure[:, :, int(StructIdx.SCORE_BO_SHORT_MEAN), w_idx] = _rolling_mean_for_series(
            x[:, :, int(FeatureIdx.SCORE_BO_SHORT) : int(FeatureIdx.SCORE_BO_SHORT) + 1, w_idx], dclip_valid, int(window), eps=float(cfg.eps)
        )[:, :, 0]
        structure[:, :, int(StructIdx.SCORE_REJECT_MEAN), w_idx] = _rolling_mean_for_series(
            x[:, :, int(FeatureIdx.SCORE_REJECT) : int(FeatureIdx.SCORE_REJECT) + 1, w_idx], dclip_valid, int(window), eps=float(cfg.eps)
        )[:, :, 0]

        structure[:, :, int(StructIdx.TREND_GATE_SPREAD_MEAN), w_idx] = (
            structure[:, :, int(StructIdx.GBREAK_MEAN), w_idx]
            - structure[:, :, int(StructIdx.GREJECT_MEAN), w_idx]
        )

        structure[:, :, int(StructIdx.POC_DRIFT_X), w_idx] = drift_poc[:, :, w_idx]
        structure[:, :, int(StructIdx.VAH_DRIFT_X), w_idx] = drift_vah[:, :, w_idx]
        structure[:, :, int(StructIdx.VAL_DRIFT_X), w_idx] = drift_val[:, :, w_idx]
        structure[:, :, int(StructIdx.DELTA_SHIFT), w_idx] = delta_shift[:, :, w_idx]

        structure[:, :, int(StructIdx.IB_HIGH_X), w_idx] = ib_hi_cache[:, :, w_idx]
        structure[:, :, int(StructIdx.IB_LOW_X), w_idx] = ib_lo_cache[:, :, w_idx]
        structure[:, :, int(StructIdx.POC_VS_PREV_VA), w_idx] = poc_vs_prev_va[:, :, w_idx]

        invalid = ~valid_aw[:, :, 0]
        structure[:, :, int(StructIdx.VALID_RATIO), w_idx] = np.where(
            np.isfinite(structure[:, :, int(StructIdx.VALID_RATIO), w_idx]),
            structure[:, :, int(StructIdx.VALID_RATIO), w_idx],
            0.0,
        )
        structure[:, :, int(StructIdx.N_VALID_BARS), w_idx] = np.where(
            np.isfinite(structure[:, :, int(StructIdx.N_VALID_BARS), w_idx]),
            structure[:, :, int(StructIdx.N_VALID_BARS), w_idx],
            0.0,
        )
        structure[:, :, int(StructIdx.POC_VS_PREV_VA), w_idx] = np.where(
            np.isfinite(structure[:, :, int(StructIdx.POC_VS_PREV_VA), w_idx]),
            structure[:, :, int(StructIdx.POC_VS_PREV_VA), w_idx],
            0.0,
        )
        structure[:, :, int(StructIdx.IB_HIGH_X), w_idx] = np.where(
            ib_defined[:, :, w_idx],
            structure[:, :, int(StructIdx.IB_HIGH_X), w_idx],
            np.nan,
        )
        structure[:, :, int(StructIdx.IB_LOW_X), w_idx] = np.where(
            ib_defined[:, :, w_idx],
            structure[:, :, int(StructIdx.IB_LOW_X), w_idx],
            np.nan,
        )
        structure[:, :, :, w_idx] = np.where(invalid[:, :, None], np.nan, structure[:, :, :, w_idx])

    structure[:, :, int(StructIdx.POC_VS_PREV_VA), :] = np.where(
        np.isfinite(structure[:, :, int(StructIdx.POC_VS_PREV_VA), :]),
        structure[:, :, int(StructIdx.POC_VS_PREV_VA), :],
        poc_dist_rel,
    )

    fp = compute_profile_fingerprint_tensor(x, eps=float(cfg.eps))
    rg = compute_profile_regime_tensor(fp)
    ctx, ctx_valid, ctx_src = build_context_tensor(
        structure,
        rg,
        np.asarray(session_id_t, dtype=np.int64),
        mode=str(cfg.context_mode),
        rolling_period=int(cfg.rolling_context_period),
    )

    out = Module3Output(
        structure_tensor=structure,
        context_tensor=ctx,
        profile_fingerprint_tensor=fp,
        profile_regime_tensor=rg,
        context_valid_atw=ctx_valid,
        context_source_index_atw=ctx_src,
    )

    if bool(cfg.validate_outputs):
        run_forensic_validation(
            x,
            out,
            session_id_t=np.asarray(session_id_t, dtype=np.int64),
            structural_windows=windows,
            atol=1e-12,
        )

    return out
