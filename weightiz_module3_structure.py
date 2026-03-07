"""
Weightiz Institutional Engine - Module 3 (30m Structural Aggregation)
=====================================================================

Deterministic 30-minute structural layer built on Module 1/2 tensors.

Key guarantees:
- Numpy-only core path.
- No loop over block-end rows (E) for block physics/moments/tails.
- No loop over minute rows (T) for context forward-fill.
- Fail-closed validation for non-finite/index violations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple

import numpy as np

from weightiz_module1_core import Phase, ProfileStatIdx, ScoreIdx, TensorState
from weightiz_dtype_guard import assert_float64
from weightiz_system_logger import get_logger, log_event

IB_POLICY_NO_TRADE = "NO_TRADE"
IB_POLICY_DEGRADE = "DEGRADE"
IB_MISSING_POLICY = IB_POLICY_NO_TRADE


class Struct30mIdx(IntEnum):
    VALID_RATIO = 0
    N_VALID_BARS = 1
    X_POC = 2
    X_VAH = 3
    X_VAL = 4
    VA_WIDTH_X = 5
    MU_ANCHOR = 6
    SIGMA_ANCHOR = 7
    SKEW_ANCHOR = 8
    KURT_EXCESS_ANCHOR = 9
    TAIL_IMBALANCE = 10
    DCLIP_MEAN = 11
    DCLIP_STD = 12
    AFFINITY_MEAN = 13
    ZDELTA_MEAN = 14
    GBREAK_MEAN = 15
    GREJECT_MEAN = 16
    DELTA_EFF_MEAN = 17
    SCORE_BO_LONG_MEAN = 18
    SCORE_BO_SHORT_MEAN = 19
    SCORE_REJECT_MEAN = 20
    TREND_GATE_SPREAD_MEAN = 21
    POC_DRIFT_X = 22
    VAH_DRIFT_X = 23
    VAL_DRIFT_X = 24
    DELTA_SHIFT = 25
    IB_HIGH_X = 26
    IB_LOW_X = 27
    POC_VS_PREV_VA = 28
    N_FIELDS = 29


class ContextIdx(IntEnum):
    CTX_X_POC = 0
    CTX_X_VAH = 1
    CTX_X_VAL = 2
    CTX_VA_WIDTH_X = 3
    CTX_DCLIP_MEAN = 4
    CTX_AFFINITY_MEAN = 5
    CTX_ZDELTA_MEAN = 6
    CTX_DELTA_EFF_MEAN = 7
    CTX_TREND_GATE_SPREAD_MEAN = 8
    CTX_POC_DRIFT_X = 9
    CTX_VALID_RATIO = 10
    CTX_IB_HIGH_X = 11
    CTX_IB_LOW_X = 12
    CTX_POC_VS_PREV_VA = 13
    N_FIELDS = 14


@dataclass(frozen=True)
class Module3Config:
    block_minutes: int = 30
    phase_mask: Tuple[int, ...] = (int(Phase.LIVE), int(Phase.OVERNIGHT_SELECT))
    use_rth_minutes_only: bool = True
    rth_open_minute: int = 570
    last_minute_inclusive: int = 945
    include_partial_last_block: bool = True
    min_block_valid_bars: int = 12
    min_block_valid_ratio: float = 0.70
    ib_pop_frac: float = 0.01
    context_mode: str = "ffill_last_complete"
    fail_on_non_finite_input: bool = True
    fail_on_non_finite_output: bool = True
    fail_on_bad_indices: bool = True
    fail_on_missing_prev_va: bool = False
    eps: float = 1e-12


@dataclass
class Module3Output:
    block_id_t: np.ndarray
    block_seq_t: np.ndarray
    block_end_flag_t: np.ndarray
    block_start_t_index_t: np.ndarray
    block_end_t_index_t: np.ndarray
    block_features_tak: np.ndarray
    block_valid_ta: np.ndarray
    context_tac: np.ndarray
    context_valid_ta: np.ndarray
    context_source_t_index_ta: np.ndarray
    ib_defined_ta: np.ndarray | None = None


def _assert_shape(name: str, arr: np.ndarray, expected: Tuple[int, ...]) -> None:
    if arr.shape != expected:
        raise RuntimeError(f"{name} shape mismatch: got {arr.shape}, expected {expected}")


def _assert_finite(name: str, arr: np.ndarray) -> None:
    if not np.all(np.isfinite(arr)):
        bad = np.argwhere(~np.isfinite(arr))[:8]
        raise RuntimeError(f"{name} contains non-finite values at indices {bad.tolist()}")


def _assert_finite_masked(name: str, arr: np.ndarray, mask: np.ndarray) -> None:
    m = np.asarray(mask, dtype=bool)
    if arr.ndim == m.ndim + 1:
        m = m[:, :, None]
    bad = m & (~np.isfinite(arr))
    if np.any(bad):
        loc = np.argwhere(bad)[:8]
        raise RuntimeError(f"{name} contains non-finite values on required rows at indices {loc.tolist()}")


def _session_spans(session_id_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    T = int(session_id_t.shape[0])
    starts = np.flatnonzero(np.r_[True, session_id_t[1:] != session_id_t[:-1]])
    ends = np.r_[starts[1:], T]
    return starts.astype(np.int64), ends.astype(np.int64)


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
    block_seq_i64 = np.full(T, -1, dtype=np.int64)
    block_seq_i64[in_calc] = m[in_calc] // int(cfg.block_minutes)
    bad_stride = in_calc & (block_seq_i64 >= np.int64(4096))
    if np.any(bad_stride):
        max_block_seq = int(np.max(block_seq_i64[in_calc]))
        t_bad = int(np.flatnonzero(bad_stride)[0])
        raise RuntimeError(
            f"block_seq exceeds 4095 safety stride: max_block_seq={max_block_seq}, "
            f"block_minutes={int(cfg.block_minutes)}, t={t_bad}"
        )
    block_seq[in_calc] = block_seq_i64[in_calc].astype(np.int16)

    block_id = np.full(T, -1, dtype=np.int64)
    # 4096 is an upper safety stride beyond any expected block_seq cardinality.
    block_id[in_calc] = state.session_id[in_calc].astype(np.int64) * np.int64(4096) + block_seq_i64[in_calc]

    prev_id = np.r_[-2, block_id[:-1]]
    next_id = np.r_[block_id[1:], -3]

    block_start = in_calc & (prev_id != block_id)
    block_end = in_calc & (next_id != block_id)

    return in_scope, block_seq, block_id, block_start, block_end


def _assert_block_start_end_integrity(
    block_id_t: np.ndarray,
    block_start_flag_t: np.ndarray,
    block_end_flag_t: np.ndarray,
    ts_idx: np.ndarray,
    te_idx: np.ndarray,
    prefix: str,
) -> None:
    if te_idx.size == 0:
        return
    same_block = block_id_t[ts_idx] == block_id_t[te_idx]
    ordered = ts_idx <= te_idx
    start_ok = block_start_flag_t[ts_idx]
    end_ok = block_end_flag_t[te_idx]
    ok = same_block & ordered & start_ok & end_ok
    if np.all(ok):
        return

    e = int(np.flatnonzero(~ok)[0])
    ts = int(ts_idx[e])
    te = int(te_idx[e])
    raise RuntimeError(
        f"{prefix}: e={e}, ts={ts}, te={te}, "
        f"block_id_ts={int(block_id_t[ts])}, block_id_te={int(block_id_t[te])}, "
        f"ts_le_te={bool(ordered[e])}, start_flag_ts={bool(start_ok[e])}, end_flag_te={bool(end_ok[e])}"
    )


def _safe_int_index(idx_f: np.ndarray, B: int) -> Tuple[np.ndarray, np.ndarray]:
    fin = np.isfinite(idx_f)
    out = np.full(idx_f.shape, -1, dtype=np.int64)
    out[fin] = np.rint(idx_f[fin]).astype(np.int64)
    ok = fin & (out >= 0) & (out < B)
    out = np.where(ok, out, -1)
    return out, ok


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


def run_module3_structural_aggregation(state: TensorState, cfg: Module3Config) -> Module3Output:
    """
    Build deterministic 30m structural snapshots and minute-aligned context.
    """
    T = state.cfg.T
    A = state.cfg.A
    B = state.cfg.B

    if cfg.block_minutes <= 0:
        raise RuntimeError("block_minutes must be > 0")
    if cfg.context_mode != "ffill_last_complete":
        raise RuntimeError("Only context_mode='ffill_last_complete' is supported")

    # Input shape checks
    _assert_shape("vp", state.vp, (T, A, B))
    assert_float64("module3.input.vp", state.vp)
    assert_float64("module3.input.profile_stats", state.profile_stats)
    assert_float64("module3.input.scores", state.scores)
    _assert_shape("profile_stats", state.profile_stats, (T, A, int(ProfileStatIdx.N_FIELDS)))
    _assert_shape("scores", state.scores, (T, A, int(ScoreIdx.N_FIELDS)))
    _assert_shape("bar_valid", state.bar_valid, (T, A))
    _assert_shape("session_id", state.session_id, (T,))
    _assert_shape("minute_of_day", state.minute_of_day, (T,))
    _assert_shape("phase", state.phase, (T,))

    in_scope_t, block_seq_t, block_id_t, block_start_flag_t, block_end_flag_t = _build_block_map(state, cfg)
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

    te_idx = np.flatnonzero(block_end_flag_t).astype(np.int64)
    ts_candidates = np.flatnonzero(block_start_flag_t).astype(np.int64)

    if te_idx.size == 0:
        block_features_tak = np.full((T, A, int(Struct30mIdx.N_FIELDS)), np.nan, dtype=np.float64)
        block_valid_ta = np.zeros((T, A), dtype=bool)
        context_tac = np.full((T, A, int(ContextIdx.N_FIELDS)), np.nan, dtype=np.float64)
        context_valid_ta = np.zeros((T, A), dtype=bool)
        src = np.full((T, A), -1, dtype=np.int64)

        out = Module3Output(
            block_id_t=block_id_t,
            block_seq_t=block_seq_t,
            block_end_flag_t=block_end_flag_t,
            block_start_t_index_t=np.full(T, -1, dtype=np.int64),
            block_end_t_index_t=np.full(T, -1, dtype=np.int64),
            block_features_tak=block_features_tak,
            block_valid_ta=block_valid_ta,
            context_tac=context_tac,
            context_valid_ta=context_valid_ta,
            context_source_t_index_ta=src,
            ib_defined_ta=np.zeros((T, A), dtype=bool),
        )
        validate_module3_output(state, out, cfg)
        return out

    # vectorized block start lookup for each te
    ts_lookup_pos = np.searchsorted(ts_candidates, te_idx, side="right") - 1
    if np.any(ts_lookup_pos < 0):
        raise RuntimeError("Block start lookup failed for some block_end indices")
    ts_idx = ts_candidates[ts_lookup_pos]
    _assert_block_start_end_integrity(
        block_id_t=block_id_t,
        block_start_flag_t=block_start_flag_t,
        block_end_flag_t=block_end_flag_t,
        ts_idx=ts_idx,
        te_idx=te_idx,
        prefix="Block start/end integrity violation",
    )

    observed_len_e = (te_idx - ts_idx + 1).astype(np.int64)

    if not cfg.include_partial_last_block:
        full = observed_len_e == int(cfg.block_minutes)
        te_idx = te_idx[full]
        ts_idx = ts_idx[full]
        observed_len_e = observed_len_e[full]

        block_end_flag_t = np.zeros(T, dtype=bool)
        block_end_flag_t[te_idx] = True
        _assert_block_start_end_integrity(
            block_id_t=block_id_t,
            block_start_flag_t=block_start_flag_t,
            block_end_flag_t=block_end_flag_t,
            ts_idx=ts_idx,
            te_idx=te_idx,
            prefix="Block start/end integrity violation after partial-block filter",
        )

    E = int(te_idx.shape[0])
    if E == 0:
        block_features_tak = np.full((T, A, int(Struct30mIdx.N_FIELDS)), np.nan, dtype=np.float64)
        block_valid_ta = np.zeros((T, A), dtype=bool)
        context_tac = np.full((T, A, int(ContextIdx.N_FIELDS)), np.nan, dtype=np.float64)
        context_valid_ta = np.zeros((T, A), dtype=bool)
        src = np.full((T, A), -1, dtype=np.int64)
        out = Module3Output(
            block_id_t=block_id_t,
            block_seq_t=block_seq_t,
            block_end_flag_t=block_end_flag_t,
            block_start_t_index_t=np.full(T, -1, dtype=np.int64),
            block_end_t_index_t=np.full(T, -1, dtype=np.int64),
            block_features_tak=block_features_tak,
            block_valid_ta=block_valid_ta,
            context_tac=context_tac,
            context_valid_ta=context_valid_ta,
            context_source_t_index_ta=src,
            ib_defined_ta=np.zeros((T, A), dtype=bool),
        )
        validate_module3_output(state, out, cfg)
        return out

    # E-batched block snapshots (strict zero-loop over E)
    vp_eab = state.vp[te_idx]  # (E, A, B)
    ps_eak = state.profile_stats[te_idx]  # (E, A, K)
    sc_eas = state.scores[te_idx]  # (E, A, S)

    # Segment validity counts from bar_valid
    bar_v = state.bar_valid.astype(np.int64)
    pref_bar = np.zeros((T + 1, A), dtype=np.int64)
    pref_bar[1:] = np.cumsum(bar_v, axis=0)

    expected_len_e = np.full(E, int(cfg.block_minutes), dtype=np.int64)
    if cfg.include_partial_last_block:
        session_e = state.session_id[te_idx].astype(np.int64)
        is_last_in_session = np.zeros(E, dtype=bool)
        is_last_in_session[-1] = True
        if E > 1:
            is_last_in_session[:-1] = session_e[:-1] != session_e[1:]
        expected_len_e[is_last_in_session] = np.minimum(
            expected_len_e[is_last_in_session],
            observed_len_e[is_last_in_session],
        )
    if np.any(expected_len_e <= 0):
        e_bad = int(np.flatnonzero(expected_len_e <= 0)[0])
        raise RuntimeError(
            f"Non-positive expected block length at e={e_bad}: expected_len={int(expected_len_e[e_bad])}"
        )

    n_valid_ea = pref_bar[te_idx + 1] - pref_bar[ts_idx]
    valid_ratio_ea = n_valid_ea.astype(np.float64) / np.maximum(
        expected_len_e[:, None].astype(np.float64),
        float(cfg.eps),
    )

    # Segment features via prefix sums (vectorized)
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
    )  # (T, A, F=10)

    src_valid_ta = state.bar_valid & np.all(np.isfinite(src_taf), axis=2)
    pref_sum, pref_cnt = _segment_prefix_sum_count(src_taf, src_valid_ta)

    seg_sum = pref_sum[te_idx + 1] - pref_sum[ts_idx]  # (E, A, F)
    seg_cnt = pref_cnt[te_idx + 1] - pref_cnt[ts_idx]  # (E, A, F)

    seg_mean = np.divide(
        seg_sum,
        seg_cnt,
        out=np.full_like(seg_sum, np.nan, dtype=np.float64),
        where=seg_cnt > 0,
    )

    # DCLIP std via scalar prefix sums
    d_sum_pref, d_s2_pref, d_cnt_pref = _segment_prefix_scalar(dclip_ta, src_valid_ta)
    d_sum = d_sum_pref[te_idx + 1] - d_sum_pref[ts_idx]
    d_s2 = d_s2_pref[te_idx + 1] - d_s2_pref[ts_idx]
    d_cnt = d_cnt_pref[te_idx + 1] - d_cnt_pref[ts_idx]
    d_mean = np.divide(d_sum, d_cnt, out=np.full((E, A), np.nan), where=d_cnt > 0)
    d_var = np.divide(d_s2, d_cnt, out=np.zeros((E, A), dtype=np.float64), where=d_cnt > 0) - np.where(
        np.isfinite(d_mean), d_mean * d_mean, 0.0
    )
    d_var = np.maximum(d_var, 0.0)
    d_std = np.sqrt(d_var)
    d_std = np.where(d_cnt > 0, d_std, np.nan)

    # Anchor indices and x-levels
    ipoc_i, ipoc_ok = _safe_int_index(ps_eak[:, :, int(ProfileStatIdx.IPOC)], B)
    ivah_i, ivah_ok = _safe_int_index(ps_eak[:, :, int(ProfileStatIdx.IVAH)], B)
    ival_i, ival_ok = _safe_int_index(ps_eak[:, :, int(ProfileStatIdx.IVAL)], B)
    idx_ok = ipoc_ok & ivah_ok & ival_ok

    if cfg.fail_on_bad_indices:
        bad = ~idx_ok
        if np.any(bad):
            loc = np.argwhere(bad)[0]
            raise RuntimeError(
                f"Bad index channels at e={int(loc[0])}, a={int(loc[1])} "
                f"(IPOC/IVAH/IVAL out-of-range or non-finite)"
            )

    x = np.asarray(state.x_grid, dtype=np.float64)
    x_poc = x[np.clip(ipoc_i, 0, B - 1)]
    x_vah = x[np.clip(ivah_i, 0, B - 1)]
    x_val = x[np.clip(ival_i, 0, B - 1)]
    va_width = x_vah - x_val

    # Anchor moments and tails from vp_eab (vectorized E,A,B)
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

    # IB extrema candidates per block snapshot
    vp_max_ea = np.max(vp_eab, axis=2)
    pop_thr = float(cfg.ib_pop_frac) * vp_max_ea[:, :, None]
    pop_mask = vp_eab >= pop_thr
    x_hi_ea = np.max(np.where(pop_mask, x[None, None, :], -np.inf), axis=2)
    x_lo_ea = np.min(np.where(pop_mask, x[None, None, :], np.inf), axis=2)
    x_hi_ea = np.where(np.isfinite(x_hi_ea), x_hi_ea, np.nan)
    x_lo_ea = np.where(np.isfinite(x_lo_ea), x_lo_ea, np.nan)

    # Block validity (base)
    block_valid_ea = (
        (n_valid_ea >= int(cfg.min_block_valid_bars))
        & (valid_ratio_ea >= float(cfg.min_block_valid_ratio))
        & idx_ok
        & np.isfinite(mu_ea)
        & np.isfinite(sigma_ea)
        & (vp_sum_ea > float(cfg.eps))
    )

    # Session-level processing (allowed loop over small session count)
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

    delta_mean_ea = seg_mean[:, :, 5]

    for s0, s1 in zip(s_starts.tolist(), s_ends.tolist()):
        idx = np.arange(s0, s1, dtype=np.int64)
        Es = int(idx.shape[0])
        if Es <= 0:
            continue

        valid_s = block_valid_ea[idx]  # (Es, A)

        # Previous valid block index per (row, asset) with no E-loop
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
            no_prev_fill = np.nan
        else:
            # Deterministic neutral defaults keep downstream context finite on first valid block.
            no_prev_fill = 0.0

        # Drifts: first valid block can use deterministic neutral default when strict mode is disabled.
        pd = np.where(prev_exists, x_poc_s - prev_poc, no_prev_fill)
        vd_h = np.where(prev_exists, x_vah_s - prev_vah, no_prev_fill)
        vd_l = np.where(prev_exists, x_val_s - prev_val, no_prev_fill)
        dsh = np.where(prev_exists, de_s - prev_de, no_prev_fill)

        pd = np.where(valid_s, pd, np.nan)
        vd_h = np.where(valid_s, vd_h, np.nan)
        vd_l = np.where(valid_s, vd_l, np.nan)
        dsh = np.where(valid_s, dsh, np.nan)

        # POC vs previous VA state
        W = np.maximum(prev_vah - prev_val, float(cfg.eps))
        xcur = x_poc_s

        rel = np.where(
            xcur > prev_vah,
            1.0 + (xcur - prev_vah) / W,
            np.where(
                xcur < prev_val,
                -1.0 - (prev_val - xcur) / W,
                -1.0 + 2.0 * (xcur - prev_val) / W,
            ),
        )

        # If no previous valid VA, strict mode raises above; otherwise use deterministic neutral default.
        rel = np.where(prev_exists, rel, no_prev_fill)
        rel = np.where(valid_s, rel, np.nan)

        # IB from seq 0 / seq {0,1}
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

        ib_hi_rows = np.where(valid_s, ib_hi_rows, np.nan)
        ib_lo_rows = np.where(valid_s, ib_lo_rows, np.nan)

        poc_drift_ea[idx] = pd
        vah_drift_ea[idx] = vd_h
        val_drift_ea[idx] = vd_l
        delta_shift_ea[idx] = dsh
        rel_prev_va_ea[idx] = rel
        ib_hi_out_ea[idx] = ib_hi_rows
        ib_lo_out_ea[idx] = ib_lo_rows

    # Assemble E-batch structural features
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

    # IB may be undefined when split-domain masking removes opening seed blocks.
    # Expose explicit policy state instead of crashing.
    ib_defined_ea = np.isfinite(ib_hi_out_ea) & np.isfinite(ib_lo_out_ea)
    policy = str(IB_MISSING_POLICY).upper().strip()
    if policy not in {IB_POLICY_NO_TRADE, IB_POLICY_DEGRADE}:
        raise RuntimeError(f"Unsupported IB policy: {IB_MISSING_POLICY!r}")
    if policy == IB_POLICY_NO_TRADE:
        block_valid_ea = block_valid_ea & ib_defined_ea
    else:
        # Research degrade mode keeps block features finite while exposing ib_defined flag.
        feat_eak[:, :, int(Struct30mIdx.IB_HIGH_X)] = np.where(ib_defined_ea, ib_hi_out_ea, x_poc)
        feat_eak[:, :, int(Struct30mIdx.IB_LOW_X)] = np.where(ib_defined_ea, ib_lo_out_ea, x_poc)

    # Invalid block rows -> NaN features
    feat_eak = np.where(block_valid_ea[:, :, None], feat_eak, np.nan)

    # Build full-T outputs
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

    # Context mapping (vectorized global gather to avoid absolute/relative indexing mismatches)
    C3 = int(ContextIdx.N_FIELDS)
    map_idx = np.array(
        [
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
        ],
        dtype=np.int64,
    )

    bf_ctx_tac = block_features_tak[:, :, map_idx]  # (T, A, C3)

    starts, ends = _session_spans(state.session_id.astype(np.int64))
    valid_end_ta = np.zeros((T, A), dtype=bool)
    valid_end_ta[te_idx] = block_valid_ea
    row_t1 = np.arange(T, dtype=np.int64)[:, None]
    cand_ta = np.where(valid_end_ta, row_t1, -1)

    # 1) Session-safe source accumulation (no cross-session leak)
    src_ta = np.full((T, A), -1, dtype=np.int64)
    for s0, s1 in zip(starts.tolist(), ends.tolist()):
        src_ta[s0:s1] = np.maximum.accumulate(cand_ta[s0:s1], axis=0)

    # 2) Global gather using absolute T indices
    context_valid_ta = src_ta >= 0
    safe_src_ta = np.where(context_valid_ta, src_ta, 0)
    context_tac = np.take_along_axis(bf_ctx_tac, safe_src_ta[:, :, None], axis=0)
    context_tac = np.where(context_valid_ta[:, :, None], context_tac, np.nan)
    context_source_t_index_ta = np.where(context_valid_ta, src_ta, -1)

    out = Module3Output(
        block_id_t=block_id_t,
        block_seq_t=block_seq_t,
        block_end_flag_t=block_end_flag_t,
        block_start_t_index_t=block_start_t_index_t,
        block_end_t_index_t=block_end_t_index_t,
        block_features_tak=block_features_tak,
        block_valid_ta=block_valid_ta,
        context_tac=context_tac,
        context_valid_ta=context_valid_ta,
        context_source_t_index_ta=context_source_t_index_ta,
        ib_defined_ta=ib_defined_ta,
    )

    validate_module3_output(state, out, cfg)
    return out


def validate_module3_output(state: TensorState, out: Module3Output, cfg: Module3Config) -> None:
    T = state.cfg.T
    A = state.cfg.A
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
    assert_float64("module3.output.block_features_tak", out.block_features_tak)
    assert_float64("module3.output.context_tac", out.context_tac)

    # Source must be causal
    t_idx = np.arange(T, dtype=np.int64)[:, None]
    bad_future = out.context_valid_ta & (out.context_source_t_index_ta > t_idx)
    if np.any(bad_future):
        loc = np.argwhere(bad_future)[0]
        raise RuntimeError(
            f"Context source index is in the future at t={int(loc[0])}, a={int(loc[1])}"
        )
    bad_bounds = out.context_valid_ta & (
        (out.context_source_t_index_ta < 0) | (out.context_source_t_index_ta >= np.int64(T))
    )
    if np.any(bad_bounds):
        loc = np.argwhere(bad_bounds)[0]
        raise RuntimeError(
            f"Context source index out of bounds at t={int(loc[0])}, a={int(loc[1])}, "
            f"src={int(out.context_source_t_index_ta[int(loc[0]), int(loc[1])])}, T={T}"
        )

    safe_src = np.where(out.context_valid_ta, out.context_source_t_index_ta, 0)
    src_session = state.session_id[safe_src]
    dst_session = state.session_id[:, None]
    bad_cross_session = out.context_valid_ta & (src_session != dst_session)
    if np.any(bad_cross_session):
        loc = np.argwhere(bad_cross_session)[0]
        t = int(loc[0])
        a = int(loc[1])
        src_t = int(out.context_source_t_index_ta[t, a])
        raise RuntimeError(
            f"Context source crosses session boundary at t={t}, a={a}, src_t={src_t}, "
            f"src_session={int(state.session_id[src_t])}, dst_session={int(state.session_id[t])}"
        )

    # Session-local monotonic source index
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
        # block features must be finite where block is valid
        bf = out.block_features_tak[out.block_valid_ta]
        if bf.size and not np.all(np.isfinite(bf)):
            loc = np.argwhere(~np.isfinite(bf))[0]
            raise RuntimeError(f"Non-finite block feature on valid block row at local_idx={loc.tolist()}")

        # context must be finite where context_valid is true
        ctx = out.context_tac[out.context_valid_ta]
        if ctx.size and not np.all(np.isfinite(ctx)):
            loc = np.argwhere(~np.isfinite(ctx))[0]
            raise RuntimeError(f"Non-finite context on valid row at local_idx={loc.tolist()}")

        ib_hi = out.block_features_tak[:, :, int(Struct30mIdx.IB_HIGH_X)]
        ib_lo = out.block_features_tak[:, :, int(Struct30mIdx.IB_LOW_X)]
        ib_valid = out.block_valid_ta & np.isfinite(ib_hi) & np.isfinite(ib_lo)
        bad_ib = ib_valid & (ib_lo > ib_hi)
        if np.any(bad_ib):
            loc = np.argwhere(bad_ib)[0]
            raise RuntimeError(f"IB bounds invalid (IB_LOW_X > IB_HIGH_X) at t={int(loc[0])}, a={int(loc[1])}")


def deterministic_digest_sha256_module3(out: Module3Output) -> str:
    import hashlib

    h = hashlib.sha256()
    arrs = [
        out.block_id_t,
        out.block_seq_t,
        out.block_end_flag_t,
        out.block_start_t_index_t,
        out.block_end_t_index_t,
        out.block_features_tak,
        out.block_valid_ta,
        out.context_tac,
        out.context_valid_ta,
        out.context_source_t_index_ta,
    ]
    if out.ib_defined_ta is not None:
        arrs.append(out.ib_defined_ta)
    for a in arrs:
        h.update(np.ascontiguousarray(a).view(np.uint8))
    return h.hexdigest()


if __name__ == "__main__":
    # Lightweight parser-level smoke that does not execute heavy engine logic.
    log_event(get_logger("module3"), "INFO", "module3_ready", event_type="module3_ready")
