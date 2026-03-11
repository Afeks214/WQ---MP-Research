from __future__ import annotations

from typing import Any, Callable

import numpy as np
from weightiz.module3.schema import ContextIdx, StructIdx


def _assert_or_flag_window_finite(
    *,
    features: dict[str, np.ndarray],
    valid_mask_atw: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    mask = np.asarray(valid_mask_atw, dtype=bool)
    if mask.ndim != 3:
        raise RuntimeError(f"valid_mask_atw must be [A,T,W], got shape={mask.shape}")

    updated = mask.copy()
    invalid_count = 0
    details: dict[str, Any] = {}

    for name, arr in features.items():
        x = np.asarray(arr, dtype=np.float64)
        if x.ndim != 4:
            raise RuntimeError(f"{name} must be [A,T,F,W], got shape={x.shape}")
        if x.shape[0] != mask.shape[0] or x.shape[1] != mask.shape[1] or x.shape[3] != mask.shape[2]:
            raise RuntimeError(
                f"{name} shape mismatch vs valid_mask_atw: arr={x.shape}, mask={mask.shape}"
            )
        row_finite = np.all(np.isfinite(x), axis=2)
        bad = mask & (~row_finite)
        if np.any(bad):
            details[name] = np.argwhere(bad)[:8].tolist()
            invalid_count += int(np.sum(bad))
            updated &= row_finite

    return updated, {"invalid_count": invalid_count, "details": details}


def _context_tensor_for_invariant(context_tensor: np.ndarray) -> np.ndarray:
    x = np.asarray(context_tensor, dtype=np.float64).copy()
    # Regime-context channels are optional outside regime_context mode; treat them as neutral when absent.
    x[:, :, int(ContextIdx.CTX_REGIME_CODE), :] = 0.0
    x[:, :, int(ContextIdx.CTX_REGIME_PERSISTENCE), :] = 0.0
    return x


def _structure_valid_mask_for_invariant(structure_tensor: np.ndarray) -> np.ndarray:
    x = np.asarray(structure_tensor, dtype=np.float64)
    if x.ndim != 4:
        raise RuntimeError(f"structure_tensor must be [A,T,F,W], got shape={x.shape}")
    # VALID_RATIO is zero-filled during warmup and cannot own the required-finite domain.
    return np.isfinite(x[:, :, int(StructIdx.DCLIP_MEAN), :])


def apply_post_m2_invariants(
    state: Any,
    active_t: np.ndarray,
    *,
    assert_or_flag_finite_fn: Callable[..., tuple[np.ndarray, dict[str, Any]]],
    set_placeholders_from_bar_valid_fn: Callable[[Any], None],
) -> list[str]:
    scope = np.asarray(active_t, dtype=bool)[:, None] & np.asarray(state.bar_valid, dtype=bool)
    updated, flags = assert_or_flag_finite_fn(
        features={
            "profile_stats": np.asarray(state.profile_stats, dtype=np.float64),
            "scores": np.asarray(state.scores, dtype=np.float64),
        },
        valid_mask=scope,
        context="post_m2",
    )
    state.bar_valid[:, :] = np.where(scope, updated, state.bar_valid)
    set_placeholders_from_bar_valid_fn(state)
    return ["INVARIANT_POST_M2_NONFINITE"] if int(flags.get("invalid_count", 0)) > 0 else []


def apply_post_m3_invariants(
    m3: Any,
    *,
    assert_or_flag_finite_fn: Callable[..., tuple[np.ndarray, dict[str, Any]]],
    ib_missing_policy: str,
    ib_policy_no_trade: str,
) -> list[str]:
    reasons: list[str] = []
    block_updated, block_flags = assert_or_flag_finite_fn(
        features={"block_features_tak": np.asarray(m3.block_features_tak, dtype=np.float64)},
        valid_mask=np.asarray(m3.block_valid_ta, dtype=bool),
        context="post_m3_block",
    )
    m3.block_valid_ta[:, :] = block_updated
    if int(block_flags.get("invalid_count", 0)) > 0:
        reasons.append("INVARIANT_POST_M3_BLOCK_NONFINITE")

    ctx_updated, ctx_flags = assert_or_flag_finite_fn(
        features={"context_tac": np.asarray(m3.context_tac, dtype=np.float64)},
        valid_mask=np.asarray(m3.context_valid_ta, dtype=bool),
        context="post_m3_context",
    )
    m3.context_valid_ta[:, :] = ctx_updated
    m3.context_source_t_index_ta[:, :] = np.where(m3.context_valid_ta, m3.context_source_t_index_ta, -1)
    m3.context_tac[:, :, :] = np.where(m3.context_valid_ta[:, :, None], m3.context_tac, np.nan)
    if int(ctx_flags.get("invalid_count", 0)) > 0:
        reasons.append("INVARIANT_POST_M3_CONTEXT_NONFINITE")

    if (m3.ib_defined_ta is not None) and (str(ib_missing_policy).upper().strip() == str(ib_policy_no_trade)):
        ib_ok = np.asarray(m3.ib_defined_ta, dtype=bool)
        before = np.asarray(m3.block_valid_ta, dtype=bool)
        after = before & ib_ok
        if int(np.sum(before & (~after))) > 0:
            reasons.append("IB_MISSING_NO_TRADE")
        m3.block_valid_ta[:, :] = after

    return sorted(set(reasons))


def apply_pre_m4_invariants(
    state: Any,
    m3: Any,
    *,
    assert_or_flag_finite_fn: Callable[..., tuple[np.ndarray, dict[str, Any]]],
    set_placeholders_from_bar_valid_fn: Callable[[Any], None],
    ib_missing_policy: str,
    ib_policy_no_trade: str,
) -> list[str]:
    reasons: list[str] = []
    updated_bar, bar_flags = assert_or_flag_finite_fn(
        features={
            "open_px": np.asarray(state.open_px, dtype=np.float64),
            "high_px": np.asarray(state.high_px, dtype=np.float64),
            "low_px": np.asarray(state.low_px, dtype=np.float64),
            "close_px": np.asarray(state.close_px, dtype=np.float64),
            "volume": np.asarray(state.volume, dtype=np.float64),
            "profile_stats": np.asarray(state.profile_stats, dtype=np.float64),
            "scores": np.asarray(state.scores, dtype=np.float64),
        },
        valid_mask=np.asarray(state.bar_valid, dtype=bool),
        context="pre_m4_state",
    )
    if int(bar_flags.get("invalid_count", 0)) > 0:
        reasons.append("INVARIANT_PRE_M4_STATE_NONFINITE")
    state.bar_valid[:, :] = updated_bar
    set_placeholders_from_bar_valid_fn(state)

    structure_valid_atw = _structure_valid_mask_for_invariant(m3.structure_tensor)
    context_valid_atw = (
        np.asarray(m3.context_valid_atw, dtype=bool)
        if getattr(m3, "context_valid_atw", None) is not None
        else np.all(np.isfinite(np.asarray(m3.context_tensor, dtype=np.float64)), axis=2)
    )

    updated_structure_atw, structure_flags = _assert_or_flag_window_finite(
        features={
            "structure_tensor": np.asarray(m3.structure_tensor, dtype=np.float64),
            "profile_fingerprint_tensor": np.asarray(m3.profile_fingerprint_tensor, dtype=np.float64),
            "profile_regime_tensor": np.asarray(m3.profile_regime_tensor, dtype=np.float64),
        },
        valid_mask_atw=structure_valid_atw,
    )
    updated_context_atw, context_flags = _assert_or_flag_window_finite(
        features={"context_tensor": _context_tensor_for_invariant(m3.context_tensor)},
        valid_mask_atw=context_valid_atw,
    )

    if int(structure_flags.get("invalid_count", 0)) > 0 or int(context_flags.get("invalid_count", 0)) > 0:
        reasons.append("INVARIANT_PRE_M4_M3_WINDOW_NONFINITE")
    m3.structure_tensor[:, :, :, :] = np.where(updated_structure_atw[:, :, None, :], m3.structure_tensor, 0.0)
    m3.profile_fingerprint_tensor[:, :, :, :] = np.where(
        updated_structure_atw[:, :, None, :], m3.profile_fingerprint_tensor, 0.0
    )
    m3.profile_regime_tensor[:, :, :, :] = np.where(
        updated_structure_atw[:, :, None, :], m3.profile_regime_tensor, 0.0
    )
    m3.context_tensor[:, :, :, :] = np.where(updated_context_atw[:, :, None, :], m3.context_tensor, 0.0)
    if m3.context_valid_atw is not None:
        m3.context_valid_atw[:, :, :] = updated_context_atw
    if m3.context_source_index_atw is not None:
        m3.context_source_index_atw[:, :, :] = np.where(updated_context_atw, m3.context_source_index_atw, -1)

    updated_ctx, ctx_flags = assert_or_flag_finite_fn(
        features={"context_tac": np.asarray(m3.context_tac, dtype=np.float64)},
        valid_mask=np.asarray(m3.context_valid_ta, dtype=bool),
        context="pre_m4_context",
    )
    if int(ctx_flags.get("invalid_count", 0)) > 0:
        reasons.append("INVARIANT_PRE_M4_CONTEXT_NONFINITE")
    m3.context_valid_ta[:, :] = updated_ctx
    m3.context_source_t_index_ta[:, :] = np.where(m3.context_valid_ta, m3.context_source_t_index_ta, -1)
    m3.context_tac[:, :, :] = np.where(m3.context_valid_ta[:, :, None], m3.context_tac, np.nan)

    if (m3.ib_defined_ta is not None) and (str(ib_missing_policy).upper().strip() == str(ib_policy_no_trade)):
        ib_ok = np.asarray(m3.ib_defined_ta, dtype=bool)
        before = np.asarray(m3.block_valid_ta, dtype=bool)
        after = before & ib_ok
        if int(np.sum(before & (~after))) > 0:
            reasons.append("IB_MISSING_NO_TRADE")
        m3.block_valid_ta[:, :] = after

    return sorted(set(reasons))


def assert_active_domain_ohlc(state: Any, active_t: np.ndarray) -> None:
    mask = np.asarray(active_t, dtype=bool)[:, None] & np.asarray(state.bar_valid, dtype=bool)
    if not np.any(mask):
        return

    finite_bad = mask & (
        ~np.isfinite(state.open_px)
        | ~np.isfinite(state.high_px)
        | ~np.isfinite(state.low_px)
        | ~np.isfinite(state.close_px)
        | ~np.isfinite(state.volume)
    )
    if np.any(finite_bad):
        bad = np.argwhere(finite_bad)[0]
        raise RuntimeError(
            f"Active-domain OHLC contains non-finite values at t={int(bad[0])}, a={int(bad[1])}"
        )

    hl_bad = mask & (state.high_px < state.low_px)
    if np.any(hl_bad):
        bad = np.argwhere(hl_bad)[0]
        raise RuntimeError(f"Active-domain OHLC violation high<low at t={int(bad[0])}, a={int(bad[1])}")

    open_bad = mask & ((state.open_px < state.low_px) | (state.open_px > state.high_px))
    if np.any(open_bad):
        bad = np.argwhere(open_bad)[0]
        raise RuntimeError(
            f"Active-domain OHLC violation open outside [low,high] at t={int(bad[0])}, a={int(bad[1])}"
        )

    close_bad = mask & ((state.close_px < state.low_px) | (state.close_px > state.high_px))
    if np.any(close_bad):
        bad = np.argwhere(close_bad)[0]
        raise RuntimeError(
            f"Active-domain OHLC violation close outside [low,high] at t={int(bad[0])}, a={int(bad[1])}"
        )

    vol_bad = mask & (state.volume < 0.0)
    if np.any(vol_bad):
        bad = np.argwhere(vol_bad)[0]
        raise RuntimeError(f"Active-domain volume violation (negative) at t={int(bad[0])}, a={int(bad[1])}")


def validate_loaded_market_slice_active_domain(
    state: Any,
    active_t: np.ndarray,
    *,
    contiguous_segments_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    validate_loaded_market_slice_fn: Callable[[Any, int, int], Any],
) -> None:
    active_idx = np.flatnonzero(np.asarray(active_t, dtype=bool)).astype(np.int64)
    seg_starts, seg_ends = contiguous_segments_fn(active_idx)
    for s0, s1 in zip(seg_starts.tolist(), seg_ends.tolist()):
        validate_loaded_market_slice_fn(state, int(s0), int(s1))


def apply_enabled_assets(state: Any, m3: Any, enabled_mask: np.ndarray) -> None:
    A = state.cfg.A
    mask = np.asarray(enabled_mask, dtype=bool)
    if mask.shape != (A,):
        raise RuntimeError(f"enabled_assets_mask shape mismatch: got {mask.shape}, expected {(A,)}")

    off = ~mask
    if not np.any(off):
        return

    state.open_px[:, off] = np.nan
    state.high_px[:, off] = np.nan
    state.low_px[:, off] = np.nan
    state.close_px[:, off] = np.nan
    state.volume[:, off] = np.nan
    state.rvol[:, off] = np.nan
    state.atr_floor[:, off] = np.nan
    state.bar_valid[:, off] = False

    m3.block_features_tak[:, off, :] = np.nan
    m3.block_valid_ta[:, off] = False
    m3.context_tac[:, off, :] = np.nan
    m3.context_valid_ta[:, off] = False
    m3.context_source_t_index_ta[:, off] = -1
    if m3.ib_defined_ta is not None:
        m3.ib_defined_ta[:, off] = False
