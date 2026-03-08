from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import EPSILON, Module4InputContracts, assert_causal_source_index


@dataclass(frozen=True)
class WindowAdapterConfig:
    mode: str = "multi_window"  # single_window | multi_window
    fixed_window_index: int = 0
    anchor_window_index: int = 0
    epsilon: float = EPSILON
    enforce_causal_checks: bool = True


@dataclass(frozen=True)
class WindowAdapterOutput:
    structure_adapted: np.ndarray
    context_adapted: np.ndarray
    fingerprint_adapted: np.ndarray
    regime_hint: np.ndarray
    selected_window_idx: np.ndarray
    window_score: np.ndarray
    regime_confidence_window: np.ndarray


def _validate_inputs(inp: Module4InputContracts) -> tuple[int, int, int, int, int, int]:
    s = inp.structure_tensor
    c = inp.context_tensor
    f = inp.profile_fingerprint_tensor
    r = inp.profile_regime_tensor

    if s.ndim != 4 or c.ndim != 4 or f.ndim != 4 or r.ndim != 4:
        raise RuntimeError("Window adapter expects 4D tensors [A,T,*,W]")

    A, T, F_struct, W = s.shape
    if c.shape[0] != A or c.shape[1] != T or c.shape[3] != W:
        raise RuntimeError(f"context_tensor shape mismatch: {c.shape} vs expected (A={A},T={T},C,W={W})")
    if f.shape[0] != A or f.shape[1] != T or f.shape[3] != W:
        raise RuntimeError(f"profile_fingerprint_tensor shape mismatch: {f.shape} vs expected (A={A},T={T},F_fp,W={W})")
    if r.shape != (A, T, 1, W):
        raise RuntimeError(f"profile_regime_tensor must be [A,T,1,W], got {r.shape}")

    if inp.tradable_mask.shape != (A, T):
        raise RuntimeError(f"tradable_mask must be [A,T], got {inp.tradable_mask.shape}")

    if inp.phase_code.shape != (T,):
        raise RuntimeError(f"phase_code must be [T], got {inp.phase_code.shape}")

    if inp.asset_enabled_mask.shape != (A,):
        raise RuntimeError(f"asset_enabled_mask must be [A], got {inp.asset_enabled_mask.shape}")

    if inp.source_time_index_at is not None:
        if inp.source_time_index_at.shape != (A, T):
            raise RuntimeError(
                f"source_time_index_at must be [A,T], got {inp.source_time_index_at.shape}"
            )

    return A, T, F_struct, int(c.shape[2]), int(f.shape[2]), W


def _compute_window_utility(inp: Module4InputContracts) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute deterministic per-window utility U[A,T,W] using time-local slices only.
    Causality: no reduction is allowed over time axis.
    """
    regime_conf = np.abs(inp.profile_regime_tensor[:, :, 0, :])
    context_mag = np.mean(np.abs(inp.context_tensor), axis=2)
    struct_mag = np.mean(np.abs(inp.structure_tensor), axis=2)
    fp_mag = np.mean(np.abs(inp.profile_fingerprint_tensor), axis=2)

    utility = regime_conf + 0.50 * context_mag + 0.25 * struct_mag + 0.25 * fp_mag
    utility = np.where(np.isfinite(utility), utility, 0.0)
    regime_conf = np.where(np.isfinite(regime_conf), regime_conf, 0.0)
    return np.ascontiguousarray(utility), np.ascontiguousarray(regime_conf)


def _causality_sanity_check(inp: Module4InputContracts, utility: np.ndarray) -> None:
    """
    Runtime causality guard:
    recompute U at each t using only [:,t,:,:] and ensure identical utility[:,t,:].
    """
    A, T, _F_struct, _C, _F_fp, W = _validate_inputs(inp)

    for t in range(T):
        reg = np.abs(inp.profile_regime_tensor[:, t, 0, :])
        ctx = np.mean(np.abs(inp.context_tensor[:, t, :, :]), axis=1)
        stc = np.mean(np.abs(inp.structure_tensor[:, t, :, :]), axis=1)
        fpr = np.mean(np.abs(inp.profile_fingerprint_tensor[:, t, :, :]), axis=1)
        u_t = reg + 0.50 * ctx + 0.25 * stc + 0.25 * fpr
        u_t = np.where(np.isfinite(u_t), u_t, 0.0)
        if not np.allclose(u_t, utility[:, t, :], rtol=0.0, atol=1e-12):
            raise RuntimeError(f"CAUSALITY_VIOLATION: non-local utility aggregation detected at t={t}")


def _select_window_multi(
    *,
    utility: np.ndarray,
    regime_confidence_window: np.ndarray,
    tradable_mask: np.ndarray,
    asset_enabled_mask: np.ndarray,
    anchor_window: int,
) -> np.ndarray:
    A, T, W = utility.shape
    anchor = int(np.clip(anchor_window, 0, W - 1))
    idx_grid = np.arange(W, dtype=np.int64)
    dist = np.abs(idx_grid - anchor)

    selected = np.full((A, T), np.int16(anchor), dtype=np.int16)

    for a in range(A):
        for t in range(T):
            if (not bool(asset_enabled_mask[a])) or (not bool(tradable_mask[a, t])):
                selected[a, t] = np.int16(anchor)
                continue

            s = utility[a, t]
            c = regime_confidence_window[a, t]
            # Tie-break hierarchy:
            # 1) highest utility
            # 2) highest regime confidence
            # 3) smallest |window-anchor|
            # 4) smallest window index
            order = np.lexsort((idx_grid, dist, -c, -s))
            selected[a, t] = np.int16(order[0])

    return selected


def _select_window_single(
    *,
    A: int,
    T: int,
    W: int,
    fixed_window_index: int,
) -> np.ndarray:
    w = int(np.clip(fixed_window_index, 0, W - 1))
    return np.full((A, T), np.int16(w), dtype=np.int16)


def _gather_by_selected(src_atfw: np.ndarray, selected_at: np.ndarray) -> np.ndarray:
    A, T, F, W = src_atfw.shape
    out = np.empty((A, T, F), dtype=np.float64)
    for a in range(A):
        sel = selected_at[a].astype(np.int64)
        for t in range(T):
            out[a, t] = src_atfw[a, t, :, sel[t]]
    return out


def adapt_windows(
    inp: Module4InputContracts,
    cfg: WindowAdapterConfig = WindowAdapterConfig(),
) -> WindowAdapterOutput:
    """
    Deterministic window adapter for Module4 decision layer.

    Input tensors are canonicalized [A,T,*,W].
    Output tensors are [A,T,*] and selected_window_idx[A,T].
    """
    A, T, _F_struct, _C, _F_fp, W = _validate_inputs(inp)
    if cfg.enforce_causal_checks and inp.source_time_index_at is not None:
        assert_causal_source_index(inp.source_time_index_at)

    utility, regime_conf_w = _compute_window_utility(inp)

    if cfg.enforce_causal_checks:
        _causality_sanity_check(inp, utility)

    mode = str(cfg.mode).strip().lower()
    if mode == "single_window":
        selected = _select_window_single(A=A, T=T, W=W, fixed_window_index=int(cfg.fixed_window_index))
    elif mode == "multi_window":
        selected = _select_window_multi(
            utility=utility,
            regime_confidence_window=regime_conf_w,
            tradable_mask=inp.tradable_mask,
            asset_enabled_mask=inp.asset_enabled_mask,
            anchor_window=int(cfg.anchor_window_index),
        )
    else:
        raise RuntimeError(f"Unsupported window adapter mode={cfg.mode!r}; expected single_window|multi_window")

    structure_adapted = _gather_by_selected(inp.structure_tensor, selected)
    context_adapted = _gather_by_selected(inp.context_tensor, selected)
    fingerprint_adapted = _gather_by_selected(inp.profile_fingerprint_tensor, selected)
    regime_adapted = _gather_by_selected(inp.profile_regime_tensor, selected)

    # Enforce [A,T,1] invariant for regime hint.
    if regime_adapted.shape != (A, T, 1):
        raise RuntimeError(f"regime_hint shape mismatch: got {regime_adapted.shape}, expected {(A, T, 1)}")

    return WindowAdapterOutput(
        structure_adapted=np.ascontiguousarray(structure_adapted, dtype=np.float64),
        context_adapted=np.ascontiguousarray(context_adapted, dtype=np.float64),
        fingerprint_adapted=np.ascontiguousarray(fingerprint_adapted, dtype=np.float64),
        regime_hint=np.ascontiguousarray(regime_adapted, dtype=np.float64),
        selected_window_idx=np.ascontiguousarray(selected, dtype=np.int16),
        window_score=np.ascontiguousarray(utility, dtype=np.float64),
        regime_confidence_window=np.ascontiguousarray(regime_conf_w, dtype=np.float64),
    )
