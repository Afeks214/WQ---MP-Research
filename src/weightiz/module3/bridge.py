"""
Weightiz Institutional Engine - Module 3 Bridge
===============================================

Public Module3 API bridge that routes execution through the refactored
multi-window Module3 package while preserving legacy downstream fields.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from weightiz.module3 import (
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
from weightiz.module3.bridge_support import (
    broadcast_ta_to_atw as _bridge_broadcast_ta_to_atw,
    build_shared_feature_tensor_from_state as _bridge_build_shared_feature_tensor_from_state,
    profile_geometry_from_state as _bridge_profile_geometry_from_state,
    resolve_shared_feature_tensor as _bridge_resolve_shared_feature_tensor,
    safe_bin_from_channel as _bridge_safe_bin_from_channel,
    session_spans as _bridge_session_spans,
    to_ta_from_at as _bridge_to_ta_from_at,
    to_tac_from_atc as _bridge_to_tac_from_atc,
)
from weightiz.shared.validation.dtype_guard import assert_float64
from weightiz.module1.core import ProfileStatIdx, ScoreIdx, TensorState
from weightiz.shared.logging.system_logger import get_logger, log_event

IB_POLICY_NO_TRADE = "NO_TRADE"
IB_POLICY_DEGRADE = "DEGRADE"
IB_MISSING_POLICY = IB_POLICY_NO_TRADE

# Backward-compatible enum aliases used across the repo.
Struct30mIdx = StructIdx


def _assert_shape(name: str, arr: np.ndarray, expected: Tuple[int, ...]) -> None:
    if arr.shape != expected:
        raise RuntimeError(f"{name} shape mismatch: got {arr.shape}, expected {expected}")


def _assert_finite_masked(name: str, arr: np.ndarray, mask: np.ndarray) -> None:
    m = np.asarray(mask, dtype=bool)
    if arr.ndim == m.ndim + 1:
        m = m[:, :, None]
    bad = m & (~np.isfinite(arr))
    if np.any(bad):
        loc = np.argwhere(bad)[:8]
        raise RuntimeError(f"{name} contains non-finite values on required rows at indices {loc.tolist()}")


def _assert_float64_bridge_visibility(state: TensorState) -> None:
    # Preserve wrapper-visible float64 enforcement for self-audit and the bridge seam.
    assert_float64("module3.bridge.input.profile_stats", state.profile_stats)


def _session_spans(session_id_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return _bridge_session_spans(session_id_t)


def _build_block_map(
    state: TensorState, cfg: Module3Config
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    T = state.cfg.T

    phase_mask_arr = np.asarray(cfg.phase_mask, dtype=np.int8)
    in_scope = np.isin(state.phase, phase_mask_arr)

    if cfg.use_rth_minutes_only:
        in_scope &= (
            (state.minute_of_day.astype(np.int64) >= int(cfg.rth_open_minute))
            & (state.minute_of_day.astype(np.int64) <= int(cfg.last_minute_inclusive))
        )

    m = state.minute_of_day.astype(np.int64) - int(cfg.rth_open_minute)
    block_seq = np.full(T, -1, dtype=np.int16)
    in_calc = in_scope & (m >= 0)
    seq_raw = (m[in_calc] // int(cfg.block_minutes)).astype(np.int64)
    if np.any(seq_raw > 4095):
        raise RuntimeError("block_seq exceeds 4095 safety stride")
    block_seq[in_calc] = seq_raw.astype(np.int16)

    block_id = np.full(T, -1, dtype=np.int64)
    block_id[in_calc] = state.session_id[in_calc].astype(np.int64) * np.int64(4096) + block_seq[in_calc].astype(np.int64)

    prev_id = np.r_[-2, block_id[:-1]]
    next_id = np.r_[block_id[1:], -3]
    block_start = in_calc & (prev_id != block_id)
    block_end = in_calc & (next_id != block_id)

    return in_scope, block_seq, block_id, block_start, block_end


def _validate_block_map_integrity(
    in_scope_t: np.ndarray,
    block_id_t: np.ndarray,
    block_start_flag_t: np.ndarray,
    block_end_flag_t: np.ndarray,
) -> None:
    in_scope = np.asarray(in_scope_t, dtype=bool)
    block_id = np.asarray(block_id_t, dtype=np.int64)
    start_flag = np.asarray(block_start_flag_t, dtype=bool)
    end_flag = np.asarray(block_end_flag_t, dtype=bool)

    prev_id = np.r_[-2, block_id[:-1]]
    next_id = np.r_[block_id[1:], -3]
    expected_start = in_scope & (prev_id != block_id)
    expected_end = in_scope & (next_id != block_id)
    if not np.array_equal(start_flag, expected_start) or not np.array_equal(end_flag, expected_end):
        raise RuntimeError("Block start/end integrity violation")


def _safe_bin_from_channel(idx_ta: np.ndarray, B: int) -> np.ndarray:
    return _bridge_safe_bin_from_channel(idx_ta, B)


def _safe_int_index(idx_f: np.ndarray, B: int) -> Tuple[np.ndarray, np.ndarray]:
    fin = np.isfinite(idx_f)
    out = np.full(idx_f.shape, -1, dtype=np.int64)
    out[fin] = np.rint(idx_f[fin]).astype(np.int64)
    ok = fin & (out >= 0) & (out < B)
    out = np.where(ok, out, -1)
    return out, ok


def _broadcast_ta_to_atw(arr_ta: np.ndarray, W: int) -> np.ndarray:
    return _bridge_broadcast_ta_to_atw(arr_ta, W)


def _profile_geometry_from_state(state: TensorState, *, eps: float) -> tuple[np.ndarray, ...]:
    return _bridge_profile_geometry_from_state(state, eps=eps)


def _build_shared_feature_tensor_from_state(state: TensorState, cfg: Module3Config) -> np.ndarray:
    return _bridge_build_shared_feature_tensor_from_state(state, cfg)


def _resolve_shared_feature_tensor(state: TensorState, cfg: Module3Config) -> np.ndarray:
    return _bridge_resolve_shared_feature_tensor(state, cfg)


def _segment_prefix_sum_count(src_taf: np.ndarray, valid_ta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    T, A, F = src_taf.shape
    v = valid_ta[:, :, None] & np.isfinite(src_taf)
    s = np.where(v, src_taf, 0.0)

    pref_sum = np.zeros((T + 1, A, F), dtype=np.float64)
    pref_cnt = np.zeros((T + 1, A, F), dtype=np.int64)

    pref_sum[1:] = np.cumsum(s, axis=0)
    pref_cnt[1:] = np.cumsum(v.astype(np.int64), axis=0)
    return pref_sum, pref_cnt


def _segment_prefix_scalar(src_ta: np.ndarray, valid_ta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    T, A = src_ta.shape
    v = valid_ta & np.isfinite(src_ta)
    s = np.where(v, src_ta, 0.0)
    s2 = s * s

    pref_sum = np.zeros((T + 1, A), dtype=np.float64)
    pref_s2 = np.zeros((T + 1, A), dtype=np.float64)
    pref_cnt = np.zeros((T + 1, A), dtype=np.int64)

    pref_sum[1:] = np.cumsum(s, axis=0)
    pref_s2[1:] = np.cumsum(s2, axis=0)
    pref_cnt[1:] = np.cumsum(v.astype(np.int64), axis=0)
    return pref_sum, pref_s2, pref_cnt


def _to_ta_from_at(arr_at: np.ndarray) -> np.ndarray:
    return _bridge_to_ta_from_at(arr_at)


def _to_tac_from_atc(arr_atc: np.ndarray) -> np.ndarray:
    return _bridge_to_tac_from_atc(arr_atc)


def run_module3_structural_aggregation(state: TensorState, cfg: Module3Config) -> Module3Output:
    """Public Module3 entrypoint. Builds shared tensor bridge and runs new engine."""
    T = int(state.cfg.T)
    A = int(state.cfg.A)
    B = int(state.cfg.B)

    _assert_float64_bridge_visibility(state)
    _assert_shape("vp", state.vp, (T, A, B))
    _assert_shape("profile_stats", state.profile_stats, (T, A, int(ProfileStatIdx.N_FIELDS)))
    _assert_shape("scores", state.scores, (T, A, int(ScoreIdx.N_FIELDS)))
    _assert_shape("bar_valid", state.bar_valid, (T, A))
    _assert_shape("session_id", state.session_id, (T,))
    _assert_shape("minute_of_day", state.minute_of_day, (T,))
    _assert_shape("phase", state.phase, (T,))

    in_scope_t, block_seq_t, block_id_t, block_start_flag_t, block_end_flag_t = _build_block_map(state, cfg)
    _validate_block_map_integrity(in_scope_t, block_id_t, block_start_flag_t, block_end_flag_t)
    req_ta = in_scope_t[:, None] & state.bar_valid

    if cfg.fail_on_non_finite_input:
        _assert_finite_masked("vp", state.vp, req_ta)
        for ch_name, ch_idx in [
            ("DCLIP", int(ProfileStatIdx.DCLIP)),
            ("A_AFFINITY", int(ProfileStatIdx.A_AFFINITY)),
            ("Z_DELTA", int(ProfileStatIdx.Z_DELTA)),
            ("GBREAK", int(ProfileStatIdx.GBREAK)),
            ("GREJECT", int(ProfileStatIdx.GREJECT)),
            ("DELTA_EFF", int(ProfileStatIdx.DELTA_EFF)),
            ("IPOC", int(ProfileStatIdx.IPOC)),
            ("IVAH", int(ProfileStatIdx.IVAH)),
            ("IVAL", int(ProfileStatIdx.IVAL)),
        ]:
            _assert_finite_masked(f"profile_stats[{ch_name}]", state.profile_stats[:, :, ch_idx], req_ta)
        for sc_name, sc_idx in [
            ("SCORE_BO_LONG", int(ScoreIdx.SCORE_BO_LONG)),
            ("SCORE_BO_SHORT", int(ScoreIdx.SCORE_BO_SHORT)),
            ("SCORE_REJECT", int(ScoreIdx.SCORE_REJECT)),
        ]:
            _assert_finite_masked(f"scores[{sc_name}]", state.scores[:, :, sc_idx], req_ta)

    shared_feature_tensor = _resolve_shared_feature_tensor(state, cfg)
    out = run_structural_window_engine(
        shared_feature_tensor=shared_feature_tensor,
        session_id_t=np.asarray(state.session_id, dtype=np.int64),
        cfg=cfg,
    )

    te_idx = np.flatnonzero(block_end_flag_t).astype(np.int64)
    ts_candidates = np.flatnonzero(block_start_flag_t).astype(np.int64)

    if te_idx.size == 0:
        out.block_id_t = block_id_t
        out.block_seq_t = block_seq_t
        out.block_end_flag_t = block_end_flag_t
        out.block_start_t_index_t = np.full(T, -1, dtype=np.int64)
        out.block_end_t_index_t = np.full(T, -1, dtype=np.int64)
        out.block_features_tak = np.full((T, A, int(Struct30mIdx.N_FIELDS)), np.nan, dtype=np.float64)
        out.block_valid_ta = np.zeros((T, A), dtype=bool)
        out.context_tac = np.full((T, A, int(ContextIdx.N_FIELDS)), np.nan, dtype=np.float64)
        out.context_valid_ta = np.zeros((T, A), dtype=bool)
        out.context_source_t_index_ta = np.full((T, A), -1, dtype=np.int64)
        out.ib_defined_ta = np.zeros((T, A), dtype=bool)
        validate_module3_output(state, out, cfg)
        return out

    ts_lookup_pos = np.searchsorted(ts_candidates, te_idx, side="right") - 1
    if np.any(ts_lookup_pos < 0):
        raise RuntimeError("Block start lookup failed for some block_end indices")
    ts_idx = ts_candidates[ts_lookup_pos]
    n_total_e = (
        state.minute_of_day[te_idx].astype(np.int64)
        - state.minute_of_day[ts_idx].astype(np.int64)
        + np.int64(1)
    )

    if not cfg.include_partial_last_block:
        full = n_total_e == int(cfg.block_minutes)
        te_idx = te_idx[full]
        ts_idx = ts_idx[full]
        n_total_e = n_total_e[full]
        block_end_flag_t = np.zeros(T, dtype=bool)
        block_end_flag_t[te_idx] = True

    E = int(te_idx.shape[0])
    if E == 0:
        out.block_id_t = block_id_t
        out.block_seq_t = block_seq_t
        out.block_end_flag_t = block_end_flag_t
        out.block_start_t_index_t = np.full(T, -1, dtype=np.int64)
        out.block_end_t_index_t = np.full(T, -1, dtype=np.int64)
        out.block_features_tak = np.full((T, A, int(Struct30mIdx.N_FIELDS)), np.nan, dtype=np.float64)
        out.block_valid_ta = np.zeros((T, A), dtype=bool)
        out.context_tac = np.full((T, A, int(ContextIdx.N_FIELDS)), np.nan, dtype=np.float64)
        out.context_valid_ta = np.zeros((T, A), dtype=bool)
        out.context_source_t_index_ta = np.full((T, A), -1, dtype=np.int64)
        out.ib_defined_ta = np.zeros((T, A), dtype=bool)
        validate_module3_output(state, out, cfg)
        return out

    vp_eab = state.vp[te_idx]
    ps_eak = state.profile_stats[te_idx]

    bar_v = state.bar_valid.astype(np.int64)
    pref_bar = np.zeros((T + 1, A), dtype=np.int64)
    pref_bar[1:] = np.cumsum(bar_v, axis=0)
    n_valid_ea = pref_bar[te_idx + 1] - pref_bar[ts_idx]
    valid_ratio_ea = n_valid_ea.astype(np.float64) / np.maximum(n_total_e[:, None].astype(np.float64), float(cfg.eps))

    dclip_ta = state.profile_stats[:, :, int(ProfileStatIdx.DCLIP)]
    aff_ta = state.profile_stats[:, :, int(ProfileStatIdx.A_AFFINITY)]
    zdelta_ta = state.profile_stats[:, :, int(ProfileStatIdx.Z_DELTA)]
    gbreak_ta = state.profile_stats[:, :, int(ProfileStatIdx.GBREAK)]
    greject_ta = state.profile_stats[:, :, int(ProfileStatIdx.GREJECT)]
    deltaeff_ta = state.profile_stats[:, :, int(ProfileStatIdx.DELTA_EFF)]
    sbo_l_ta = state.scores[:, :, int(ScoreIdx.SCORE_BO_LONG)]
    sbo_s_ta = state.scores[:, :, int(ScoreIdx.SCORE_BO_SHORT)]
    srej_ta = state.scores[:, :, int(ScoreIdx.SCORE_REJECT)]
    trend_spread_ta = gbreak_ta - greject_ta

    src_taf = np.stack(
        [
            dclip_ta,
            aff_ta,
            zdelta_ta,
            gbreak_ta,
            greject_ta,
            deltaeff_ta,
            sbo_l_ta,
            sbo_s_ta,
            srej_ta,
            trend_spread_ta,
        ],
        axis=2,
    )

    src_valid_ta = state.bar_valid & np.all(np.isfinite(src_taf), axis=2)
    pref_sum, pref_cnt = _segment_prefix_sum_count(src_taf, src_valid_ta)
    seg_sum = pref_sum[te_idx + 1] - pref_sum[ts_idx]
    seg_cnt = pref_cnt[te_idx + 1] - pref_cnt[ts_idx]
    seg_mean = np.divide(seg_sum, seg_cnt, out=np.full_like(seg_sum, np.nan, dtype=np.float64), where=seg_cnt > 0)

    d_sum_pref, d_s2_pref, d_cnt_pref = _segment_prefix_scalar(dclip_ta, src_valid_ta)
    d_sum = d_sum_pref[te_idx + 1] - d_sum_pref[ts_idx]
    d_s2 = d_s2_pref[te_idx + 1] - d_s2_pref[ts_idx]
    d_cnt = d_cnt_pref[te_idx + 1] - d_cnt_pref[ts_idx]
    d_mean = np.divide(d_sum, d_cnt, out=np.full((E, A), np.nan), where=d_cnt > 0)
    d_var = np.divide(d_s2, d_cnt, out=np.zeros((E, A), dtype=np.float64), where=d_cnt > 0) - np.where(np.isfinite(d_mean), d_mean * d_mean, 0.0)
    d_var = np.maximum(d_var, 0.0)
    d_std = np.where(d_cnt > 0, np.sqrt(d_var), np.nan)

    ipoc_i, ipoc_ok = _safe_int_index(ps_eak[:, :, int(ProfileStatIdx.IPOC)], B)
    ivah_i, ivah_ok = _safe_int_index(ps_eak[:, :, int(ProfileStatIdx.IVAH)], B)
    ival_i, ival_ok = _safe_int_index(ps_eak[:, :, int(ProfileStatIdx.IVAL)], B)
    idx_ok = ipoc_ok & ivah_ok & ival_ok
    if cfg.fail_on_bad_indices:
        bad = ~idx_ok
        if np.any(bad):
            loc = np.argwhere(bad)[0]
            raise RuntimeError(
                f"Bad index channels at e={int(loc[0])}, a={int(loc[1])} (IPOC/IVAH/IVAL out-of-range or non-finite)"
            )

    x = np.asarray(state.x_grid, dtype=np.float64)
    x_poc = x[np.clip(ipoc_i, 0, B - 1)]
    x_vah = x[np.clip(ivah_i, 0, B - 1)]
    x_val = x[np.clip(ival_i, 0, B - 1)]
    va_width = x_vah - x_val

    vp_sum_ea = np.sum(vp_eab, axis=2)
    p_eab = vp_eab / (vp_sum_ea[:, :, None] + float(cfg.eps))
    mu_ea = np.sum(p_eab * x[None, None, :], axis=2)
    cen = x[None, None, :] - mu_ea[:, :, None]
    var_ea = np.sum(p_eab * (cen * cen), axis=2)
    sigma_ea = np.sqrt(np.maximum(var_ea, 0.0))
    z = cen / (sigma_ea[:, :, None] + float(cfg.eps))
    skew_ea = np.sum(p_eab * (z**3), axis=2)
    kurt_ea = np.sum(p_eab * (z**4), axis=2) - 3.0
    up_mask = x[None, None, :] > (mu_ea + sigma_ea)[:, :, None]
    dn_mask = x[None, None, :] < (mu_ea - sigma_ea)[:, :, None]
    tail_up = np.sum(np.where(up_mask, p_eab, 0.0), axis=2)
    tail_dn = np.sum(np.where(dn_mask, p_eab, 0.0), axis=2)
    tail_imb_ea = tail_up - tail_dn

    vp_max_ea = np.max(vp_eab, axis=2)
    pop_thr = float(cfg.ib_pop_frac) * vp_max_ea[:, :, None]
    pop_mask = vp_eab >= pop_thr
    x_hi_ea = np.max(np.where(pop_mask, x[None, None, :], -np.inf), axis=2)
    x_lo_ea = np.min(np.where(pop_mask, x[None, None, :], np.inf), axis=2)
    x_hi_ea = np.where(np.isfinite(x_hi_ea), x_hi_ea, np.nan)
    x_lo_ea = np.where(np.isfinite(x_lo_ea), x_lo_ea, np.nan)

    block_valid_ea = (
        (n_valid_ea >= int(cfg.min_block_valid_bars))
        & (valid_ratio_ea >= float(cfg.min_block_valid_ratio))
        & idx_ok
        & np.isfinite(mu_ea)
        & np.isfinite(sigma_ea)
        & (vp_sum_ea > float(cfg.eps))
    )

    session_e = state.session_id[te_idx].astype(np.int64)
    block_seq_e = block_seq_t[te_idx].astype(np.int16)
    s_starts, s_ends = _session_spans(session_e)

    poc_drift_ea = np.full((E, A), np.nan, dtype=np.float64)
    vah_drift_ea = np.full((E, A), np.nan, dtype=np.float64)
    val_drift_ea = np.full((E, A), np.nan, dtype=np.float64)
    delta_shift_ea = np.full((E, A), np.nan, dtype=np.float64)
    rel_prev_va_ea = np.full((E, A), np.nan, dtype=np.float64)
    ib_hi_out_ea = np.full((E, A), np.nan, dtype=np.float64)
    ib_lo_out_ea = np.full((E, A), np.nan, dtype=np.float64)
    ib_defined_ea = np.zeros((E, A), dtype=bool)

    delta_mean_ea = seg_mean[:, :, 5]

    for s0, s1 in zip(s_starts.tolist(), s_ends.tolist()):
        idx = np.arange(s0, s1, dtype=np.int64)
        Es = int(idx.shape[0])
        if Es <= 0:
            continue

        valid_s = block_valid_ea[idx]
        rid = np.arange(Es, dtype=np.int64)[:, None]
        cand = np.where(valid_s, rid, -1)
        last = np.maximum.accumulate(cand, axis=0)
        prev = np.vstack([np.full((1, A), -1, dtype=np.int64), last[:-1]])
        prev_exists = prev >= 0
        safe_prev = np.clip(prev, 0, max(Es - 1, 0))

        x_poc_s = x_poc[idx]
        x_vah_s = x_vah[idx]
        x_val_s = x_val[idx]
        de_s = delta_mean_ea[idx]

        prev_poc = np.where(prev_exists, np.take_along_axis(x_poc_s, safe_prev, axis=0), np.nan)
        prev_vah = np.where(prev_exists, np.take_along_axis(x_vah_s, safe_prev, axis=0), np.nan)
        prev_val = np.where(prev_exists, np.take_along_axis(x_val_s, safe_prev, axis=0), np.nan)
        prev_de = np.where(prev_exists, np.take_along_axis(de_s, safe_prev, axis=0), np.nan)

        if cfg.fail_on_missing_prev_va:
            missing_prev = valid_s & (~prev_exists)
            if np.any(missing_prev):
                loc = np.argwhere(missing_prev)[0]
                raise RuntimeError(
                    f"Missing previous VA for POC_VS_PREV_VA at session-local row={int(loc[0])}, a={int(loc[1])}"
                )

        pd = np.where(prev_exists, x_poc_s - prev_poc, 0.0)
        vd_h = np.where(prev_exists, x_vah_s - prev_vah, 0.0)
        vd_l = np.where(prev_exists, x_val_s - prev_val, 0.0)
        dsh = np.where(prev_exists, de_s - prev_de, 0.0)

        pd = np.where(valid_s, pd, np.nan)
        vd_h = np.where(valid_s, vd_h, np.nan)
        vd_l = np.where(valid_s, vd_l, np.nan)
        dsh = np.where(valid_s, dsh, np.nan)

        Wprev = np.maximum(prev_vah - prev_val, float(cfg.eps))
        xcur = x_poc_s
        rel = np.where(
            xcur > prev_vah,
            1.0 + (xcur - prev_vah) / Wprev,
            np.where(
                xcur < prev_val,
                -1.0 - (prev_val - xcur) / Wprev,
                -1.0 + 2.0 * (xcur - prev_val) / Wprev,
            ),
        )
        rel = np.where(prev_exists, rel, 0.0)
        rel = np.where(valid_s, rel, np.nan)

        seq_s = block_seq_e[idx].astype(np.int64)
        x_hi_s = x_hi_ea[idx]
        x_lo_s = x_lo_ea[idx]
        mask0 = (seq_s[:, None] == 0) & valid_s
        mask01 = ((seq_s[:, None] == 0) | (seq_s[:, None] == 1)) & valid_s

        ib0_hi = np.max(np.where(mask0, x_hi_s, -np.inf), axis=0)
        ib0_lo = np.min(np.where(mask0, x_lo_s, np.inf), axis=0)
        ib01_hi = np.max(np.where(mask01, x_hi_s, -np.inf), axis=0)
        ib01_lo = np.min(np.where(mask01, x_lo_s, np.inf), axis=0)

        ib0_hi = np.where(np.isfinite(ib0_hi), ib0_hi, np.nan)
        ib0_lo = np.where(np.isfinite(ib0_lo), ib0_lo, np.nan)
        ib01_hi = np.where(np.isfinite(ib01_hi), ib01_hi, np.nan)
        ib01_lo = np.where(np.isfinite(ib01_lo), ib01_lo, np.nan)

        use0 = seq_s[:, None] == 0
        ib0_hi_b = np.broadcast_to(ib0_hi[None, :], (Es, A))
        ib0_lo_b = np.broadcast_to(ib0_lo[None, :], (Es, A))
        ib01_hi_b = np.broadcast_to(ib01_hi[None, :], (Es, A))
        ib01_lo_b = np.broadcast_to(ib01_lo[None, :], (Es, A))
        ib_hi_rows = np.where(use0, ib0_hi_b, ib01_hi_b)
        ib_lo_rows = np.where(use0, ib0_lo_b, ib01_lo_b)

        ib_defined_rows = np.isfinite(ib_hi_rows) & np.isfinite(ib_lo_rows)
        if IB_MISSING_POLICY == IB_POLICY_NO_TRADE:
            valid_s = valid_s & ib_defined_rows
        elif IB_MISSING_POLICY == IB_POLICY_DEGRADE:
            pass
        else:
            raise RuntimeError(f"Unsupported IB_MISSING_POLICY={IB_MISSING_POLICY}")

        ib_hi_rows = np.where(valid_s & ib_defined_rows, ib_hi_rows, np.nan)
        ib_lo_rows = np.where(valid_s & ib_defined_rows, ib_lo_rows, np.nan)

        poc_drift_ea[idx] = pd
        vah_drift_ea[idx] = vd_h
        val_drift_ea[idx] = vd_l
        delta_shift_ea[idx] = dsh
        rel_prev_va_ea[idx] = rel
        ib_hi_out_ea[idx] = ib_hi_rows
        ib_lo_out_ea[idx] = ib_lo_rows
        ib_defined_ea[idx] = ib_defined_rows
        block_valid_ea[idx] = valid_s

    K3 = int(Struct30mIdx.N_FIELDS)
    feat_eak = np.full((E, A, K3), np.nan, dtype=np.float64)
    feat_eak[:, :, int(Struct30mIdx.VALID_RATIO)] = valid_ratio_ea
    feat_eak[:, :, int(Struct30mIdx.N_VALID_BARS)] = n_valid_ea.astype(np.float64)
    feat_eak[:, :, int(Struct30mIdx.X_POC)] = x_poc
    feat_eak[:, :, int(Struct30mIdx.X_VAH)] = x_vah
    feat_eak[:, :, int(Struct30mIdx.X_VAL)] = x_val
    feat_eak[:, :, int(Struct30mIdx.VA_WIDTH_X)] = va_width
    feat_eak[:, :, int(Struct30mIdx.MU_ANCHOR)] = mu_ea
    feat_eak[:, :, int(Struct30mIdx.SIGMA_ANCHOR)] = sigma_ea
    feat_eak[:, :, int(Struct30mIdx.SKEW_ANCHOR)] = skew_ea
    feat_eak[:, :, int(Struct30mIdx.KURT_EXCESS_ANCHOR)] = kurt_ea
    feat_eak[:, :, int(Struct30mIdx.TAIL_IMBALANCE)] = tail_imb_ea
    feat_eak[:, :, int(Struct30mIdx.DCLIP_MEAN)] = seg_mean[:, :, 0]
    feat_eak[:, :, int(Struct30mIdx.DCLIP_STD)] = d_std
    feat_eak[:, :, int(Struct30mIdx.AFFINITY_MEAN)] = seg_mean[:, :, 1]
    feat_eak[:, :, int(Struct30mIdx.ZDELTA_MEAN)] = seg_mean[:, :, 2]
    feat_eak[:, :, int(Struct30mIdx.GBREAK_MEAN)] = seg_mean[:, :, 3]
    feat_eak[:, :, int(Struct30mIdx.GREJECT_MEAN)] = seg_mean[:, :, 4]
    feat_eak[:, :, int(Struct30mIdx.DELTA_EFF_MEAN)] = seg_mean[:, :, 5]
    feat_eak[:, :, int(Struct30mIdx.SCORE_BO_LONG_MEAN)] = seg_mean[:, :, 6]
    feat_eak[:, :, int(Struct30mIdx.SCORE_BO_SHORT_MEAN)] = seg_mean[:, :, 7]
    feat_eak[:, :, int(Struct30mIdx.SCORE_REJECT_MEAN)] = seg_mean[:, :, 8]
    feat_eak[:, :, int(Struct30mIdx.TREND_GATE_SPREAD_MEAN)] = seg_mean[:, :, 9]
    feat_eak[:, :, int(Struct30mIdx.POC_DRIFT_X)] = poc_drift_ea
    feat_eak[:, :, int(Struct30mIdx.VAH_DRIFT_X)] = vah_drift_ea
    feat_eak[:, :, int(Struct30mIdx.VAL_DRIFT_X)] = val_drift_ea
    feat_eak[:, :, int(Struct30mIdx.DELTA_SHIFT)] = delta_shift_ea
    feat_eak[:, :, int(Struct30mIdx.IB_HIGH_X)] = ib_hi_out_ea
    feat_eak[:, :, int(Struct30mIdx.IB_LOW_X)] = ib_lo_out_ea
    feat_eak[:, :, int(Struct30mIdx.POC_VS_PREV_VA)] = rel_prev_va_ea
    feat_eak = np.where(block_valid_ea[:, :, None], feat_eak, np.nan)

    block_features_tak = np.full((T, A, K3), np.nan, dtype=np.float64)
    block_features_tak[te_idx] = feat_eak
    block_valid_ta = np.zeros((T, A), dtype=bool)
    block_valid_ta[te_idx] = block_valid_ea
    ib_defined_ta = np.zeros((T, A), dtype=bool)
    ib_defined_ta[te_idx] = ib_defined_ea

    block_start_t_index_t = np.full(T, -1, dtype=np.int64)
    block_end_t_index_t = np.full(T, -1, dtype=np.int64)
    block_start_t_index_t[te_idx] = ts_idx
    block_end_t_index_t[te_idx] = te_idx

    C3 = int(ContextIdx.N_FIELDS)
    context_tac = np.full((T, A, C3), np.nan, dtype=np.float64)
    map_idx = np.array([
        int(Struct30mIdx.X_POC),
        int(Struct30mIdx.X_VAH),
        int(Struct30mIdx.X_VAL),
        int(Struct30mIdx.VA_WIDTH_X),
        int(Struct30mIdx.DCLIP_MEAN),
        int(Struct30mIdx.AFFINITY_MEAN),
        int(Struct30mIdx.ZDELTA_MEAN),
        int(Struct30mIdx.DELTA_EFF_MEAN),
        int(Struct30mIdx.TREND_GATE_SPREAD_MEAN),
        int(Struct30mIdx.POC_DRIFT_X),
        int(Struct30mIdx.VALID_RATIO),
        int(Struct30mIdx.IB_HIGH_X),
        int(Struct30mIdx.IB_LOW_X),
        int(Struct30mIdx.POC_VS_PREV_VA),
    ], dtype=np.int64)
    bf_ctx_tac = block_features_tak[:, :, map_idx]
    context_tac[:, :, : map_idx.shape[0]] = bf_ctx_tac

    starts, ends = _session_spans(state.session_id.astype(np.int64))
    valid_end_ta = np.zeros((T, A), dtype=bool)
    valid_end_ta[te_idx] = block_valid_ea
    row_t1 = np.arange(T, dtype=np.int64)[:, None]
    cand_ta = np.where(valid_end_ta, row_t1, -1)

    src_ta = np.full((T, A), -1, dtype=np.int64)
    for s0, s1 in zip(starts.tolist(), ends.tolist()):
        src_ta[s0:s1] = np.maximum.accumulate(cand_ta[s0:s1], axis=0)

    context_valid_ta = src_ta >= 0
    safe_src_ta = np.where(context_valid_ta, src_ta, 0)
    gathered = np.take_along_axis(bf_ctx_tac, safe_src_ta[:, :, None], axis=0)
    context_tac[:, :, : map_idx.shape[0]] = np.where(context_valid_ta[:, :, None], gathered, np.nan)
    context_tac[:, :, int(ContextIdx.CTX_REGIME_CODE)] = np.where(context_valid_ta, 0.0, np.nan)
    context_tac[:, :, int(ContextIdx.CTX_REGIME_PERSISTENCE)] = np.where(context_valid_ta, 0.0, np.nan)
    context_source_t_index_ta = np.where(context_valid_ta, src_ta, -1)

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

    validate_module3_output(state, out, cfg)
    return out


def run_module3(state: TensorState, cfg: Module3Config) -> Module3Output:
    return run_module3_structural_aggregation(state, cfg)


def validate_module3_output(state: TensorState, out: Module3Output, cfg: Module3Config) -> None:
    validate_output_contract(out)

    T = int(state.cfg.T)
    A = int(state.cfg.A)
    K3 = int(Struct30mIdx.N_FIELDS)
    C3 = int(ContextIdx.N_FIELDS)

    _assert_shape("block_id_t", out.block_id_t, (T,))
    _assert_shape("block_seq_t", out.block_seq_t, (T,))
    _assert_shape("block_end_flag_t", out.block_end_flag_t, (T,))
    _assert_shape("block_start_t_index_t", out.block_start_t_index_t, (T,))
    _assert_shape("block_end_t_index_t", out.block_end_t_index_t, (T,))
    _assert_shape("block_features_tak", out.block_features_tak, (T, A, K3))
    _assert_shape("block_valid_ta", out.block_valid_ta, (T, A))
    _assert_shape("context_tac", out.context_tac, (T, A, C3))
    _assert_shape("context_valid_ta", out.context_valid_ta, (T, A))
    _assert_shape("context_source_t_index_ta", out.context_source_t_index_ta, (T, A))
    if out.ib_defined_ta is not None:
        _assert_shape("ib_defined_ta", out.ib_defined_ta, (T, A))

    t_idx = np.arange(T, dtype=np.int64)[:, None]
    bad_future = out.context_valid_ta & (out.context_source_t_index_ta > t_idx)
    if np.any(bad_future):
        loc = np.argwhere(bad_future)[0]
        raise RuntimeError(f"Context source index is in the future at t={int(loc[0])}, a={int(loc[1])}")

    bad_cross = out.context_valid_ta & (state.session_id[out.context_source_t_index_ta.clip(0)] != state.session_id[:, None])
    if np.any(bad_cross):
        loc = np.argwhere(bad_cross)[0]
        raise RuntimeError(f"Context source crosses session boundary at t={int(loc[0])}, a={int(loc[1])}")

    s_starts, s_ends = _session_spans(state.session_id.astype(np.int64))
    for s0, s1 in zip(s_starts.tolist(), s_ends.tolist()):
        src_s = out.context_source_t_index_ta[s0:s1]
        v_s = out.context_valid_ta[s0:s1]
        if src_s.shape[0] <= 1:
            continue
        pair = v_s[1:] & v_s[:-1]
        dec = pair & (src_s[1:] < src_s[:-1])
        if np.any(dec):
            loc = np.argwhere(dec)[0]
            raise RuntimeError(
                f"Context source index not monotonic in session slice at local_t={int(loc[0])}, a={int(loc[1])}"
            )

    if cfg.fail_on_non_finite_output:
        bf_valid_mask = np.broadcast_to(out.block_valid_ta[:, :, None], out.block_features_tak.shape).copy()
        if out.ib_defined_ta is not None:
            allow_ib_nan = out.block_valid_ta & (~out.ib_defined_ta)
            bf_valid_mask[:, :, int(Struct30mIdx.IB_HIGH_X)] &= ~allow_ib_nan
            bf_valid_mask[:, :, int(Struct30mIdx.IB_LOW_X)] &= ~allow_ib_nan
        bad_bf = bf_valid_mask & (~np.isfinite(out.block_features_tak))
        if np.any(bad_bf):
            loc = np.argwhere(bad_bf)[0]
            raise RuntimeError(f"Non-finite block feature on valid block row at local_idx={loc.tolist()}")

        ctx_valid_mask = np.broadcast_to(out.context_valid_ta[:, :, None], out.context_tac.shape).copy()
        if out.ib_defined_ta is not None:
            allow_ctx_ib_nan = out.context_valid_ta & (~out.ib_defined_ta)
            ctx_valid_mask[:, :, int(ContextIdx.CTX_IB_HIGH_X)] &= ~allow_ctx_ib_nan
            ctx_valid_mask[:, :, int(ContextIdx.CTX_IB_LOW_X)] &= ~allow_ctx_ib_nan
        bad_ctx = ctx_valid_mask & (~np.isfinite(out.context_tac))
        if np.any(bad_ctx):
            loc = np.argwhere(bad_ctx)[0]
            raise RuntimeError(f"Non-finite context on valid row at local_idx={loc.tolist()}")

        ib_hi = out.block_features_tak[:, :, int(Struct30mIdx.IB_HIGH_X)]
        ib_lo = out.block_features_tak[:, :, int(Struct30mIdx.IB_LOW_X)]
        ib_valid = out.block_valid_ta & np.isfinite(ib_hi) & np.isfinite(ib_lo)
        bad_ib = ib_valid & (ib_lo > ib_hi)
        if np.any(bad_ib):
            loc = np.argwhere(bad_ib)[0]
            raise RuntimeError(f"IB bounds invalid (IB_LOW_X > IB_HIGH_X) at t={int(loc[0])}, a={int(loc[1])}")


def deterministic_digest_sha256_module3_bridge(out: Module3Output) -> str:
    return _deterministic_digest_sha256_module3(out)


# Backward-compatible exported name.
deterministic_digest_sha256_module3 = deterministic_digest_sha256_module3_bridge


if __name__ == "__main__":
    log_event(get_logger("module3"), "INFO", "module3_ready", event_type="module3_ready")
