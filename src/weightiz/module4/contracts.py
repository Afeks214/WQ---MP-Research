from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

EPSILON: float = 1e-12


@dataclass(frozen=True)
class Module4InputContracts:
    """Canonicalized Module4 inputs with internal [A,T,*] orientation."""

    alpha_signal_tensor: np.ndarray
    score_tensor: np.ndarray
    profile_stat_tensor: np.ndarray
    structure_tensor: np.ndarray
    context_tensor: np.ndarray
    profile_fingerprint_tensor: np.ndarray
    profile_regime_tensor: np.ndarray
    tradable_mask: np.ndarray
    phase_code: np.ndarray
    asset_enabled_mask: np.ndarray
    profile_grid_summary_tensor: Optional[np.ndarray] = None
    volatility_tensor: Optional[np.ndarray] = None
    spread_tensor: Optional[np.ndarray] = None
    liquidity_score: Optional[np.ndarray] = None
    source_time_index_at: Optional[np.ndarray] = None


@dataclass(frozen=True)
class RiskFilterConfig:
    max_volatility: float = np.inf
    max_spread: float = np.inf
    min_liquidity: float = 0.0


def _require_ndarray(name: str, arr: np.ndarray) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        raise RuntimeError(f"{name} must be np.ndarray, got {type(arr)!r}")
    return arr


def _canonicalize_at_prefix(name: str, arr: np.ndarray, A: int, T: int) -> np.ndarray:
    """Normalize tensors from [A,T,*] or [T,A,*] into [A,T,*]."""
    x = _require_ndarray(name, np.asarray(arr))
    if x.ndim < 2:
        raise RuntimeError(f"{name} must have ndim>=2, got {x.ndim}")
    if x.shape[0] == A and x.shape[1] == T:
        out = x
    elif x.shape[0] == T and x.shape[1] == A:
        out = np.swapaxes(x, 0, 1)
    else:
        raise RuntimeError(
            f"{name} must be [A,T,*] or [T,A,*] with A={A}, T={T}; got shape={x.shape}"
        )
    return np.ascontiguousarray(out)


def _as_float64(name: str, arr: np.ndarray, *, allow_non_finite: bool = False) -> np.ndarray:
    x = np.asarray(arr)
    if not np.issubdtype(x.dtype, np.number):
        raise RuntimeError(f"{name} must be numeric, got dtype={x.dtype}")
    x64 = np.asarray(x, dtype=np.float64)
    if (not allow_non_finite) and (not np.all(np.isfinite(x64))):
        bad = np.argwhere(~np.isfinite(x64))[:8]
        raise RuntimeError(f"{name} contains non-finite values at indices {bad.tolist()}")
    return np.ascontiguousarray(x64)


def _as_bool(name: str, arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr)
    if x.dtype == np.bool_:
        return np.ascontiguousarray(x.astype(bool, copy=False))
    # Allow integer 0/1 inputs but enforce boolean semantic.
    if np.issubdtype(x.dtype, np.integer):
        valid = np.isin(x, np.array([0, 1], dtype=x.dtype))
        if not np.all(valid):
            bad = np.argwhere(~valid)[:8]
            raise RuntimeError(f"{name} integer mask must contain only 0/1; invalid at {bad.tolist()}")
        return np.ascontiguousarray(x.astype(bool, copy=False))
    raise RuntimeError(f"{name} must be bool or integer 0/1 mask, got dtype={x.dtype}")


def _as_int64(name: str, arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr)
    if not np.issubdtype(x.dtype, np.integer):
        raise RuntimeError(f"{name} must be integer dtype, got {x.dtype}")
    return np.ascontiguousarray(x.astype(np.int64, copy=False))


def _validate_tensor_shape(name: str, arr: np.ndarray, expected: tuple[int, ...]) -> None:
    if arr.shape != expected:
        raise RuntimeError(f"{name} shape mismatch: got {arr.shape}, expected {expected}")


def _validate_profile_window_shapes(
    *,
    structure_tensor: np.ndarray,
    context_tensor: np.ndarray,
    profile_fingerprint_tensor: np.ndarray,
    profile_regime_tensor: np.ndarray,
) -> None:
    if structure_tensor.ndim != 4:
        raise RuntimeError(f"structure_tensor must be 4D [A,T,F_struct,W], got ndim={structure_tensor.ndim}")
    if context_tensor.ndim != 4:
        raise RuntimeError(f"context_tensor must be 4D [A,T,C,W], got ndim={context_tensor.ndim}")
    if profile_fingerprint_tensor.ndim != 4:
        raise RuntimeError(
            f"profile_fingerprint_tensor must be 4D [A,T,F_fp,W], got ndim={profile_fingerprint_tensor.ndim}"
        )
    if profile_regime_tensor.ndim != 4:
        raise RuntimeError(f"profile_regime_tensor must be 4D [A,T,1,W], got ndim={profile_regime_tensor.ndim}")

    A, T, _f_struct, W = structure_tensor.shape
    _validate_tensor_shape("context_tensor", context_tensor, (A, T, context_tensor.shape[2], W))
    _validate_tensor_shape(
        "profile_fingerprint_tensor",
        profile_fingerprint_tensor,
        (A, T, profile_fingerprint_tensor.shape[2], W),
    )
    _validate_tensor_shape("profile_regime_tensor", profile_regime_tensor, (A, T, 1, W))


def assert_causal_source_index(source_time_index_at: np.ndarray) -> None:
    """
    Causality invariant: source index used for decision at time t must satisfy source_idx <= t.
    Input is canonicalized [A,T].
    """
    idx = _as_int64("source_time_index_at", source_time_index_at)
    if idx.ndim != 2:
        raise RuntimeError(f"source_time_index_at must be 2D [A,T], got ndim={idx.ndim}")
    A, T = idx.shape
    t_grid = np.broadcast_to(np.arange(T, dtype=np.int64)[None, :], (A, T))
    bad = idx > t_grid
    if np.any(bad):
        loc = np.argwhere(bad)[0]
        a = int(loc[0])
        t = int(loc[1])
        raise RuntimeError(
            "CAUSALITY_VIOLATION: source_time_index_at contains forward-looking index "
            f"at a={a}, t={t}, source_idx={int(idx[a, t])}"
        )


def build_module4_input_contracts(
    *,
    alpha_signal_tensor: np.ndarray,
    score_tensor: np.ndarray,
    profile_stat_tensor: np.ndarray,
    structure_tensor: np.ndarray,
    context_tensor: np.ndarray,
    profile_fingerprint_tensor: np.ndarray,
    profile_regime_tensor: np.ndarray,
    tradable_mask: np.ndarray,
    phase_code: np.ndarray,
    asset_enabled_mask: np.ndarray,
    profile_grid_summary_tensor: np.ndarray | None = None,
    volatility_tensor: np.ndarray | None = None,
    spread_tensor: np.ndarray | None = None,
    liquidity_score: np.ndarray | None = None,
    source_time_index_at: np.ndarray | None = None,
    fail_on_non_finite_input: bool = True,
) -> Module4InputContracts:
    """Validate, canonicalize, and freeze Module4 decision-layer input contracts."""
    phase = _as_int64("phase_code", phase_code)
    if phase.ndim != 1:
        raise RuntimeError(f"phase_code must be 1D [T], got ndim={phase.ndim}")
    T = int(phase.shape[0])

    asset_enabled = _as_bool("asset_enabled_mask", asset_enabled_mask)
    if asset_enabled.ndim != 1:
        raise RuntimeError(f"asset_enabled_mask must be 1D [A], got ndim={asset_enabled.ndim}")
    A = int(asset_enabled.shape[0])

    alpha = _as_float64(
        "alpha_signal_tensor",
        _canonicalize_at_prefix("alpha_signal_tensor", alpha_signal_tensor, A, T),
        allow_non_finite=not bool(fail_on_non_finite_input),
    )
    score = _as_float64(
        "score_tensor",
        _canonicalize_at_prefix("score_tensor", score_tensor, A, T),
        allow_non_finite=not bool(fail_on_non_finite_input),
    )
    profile_stat = _as_float64(
        "profile_stat_tensor",
        _canonicalize_at_prefix("profile_stat_tensor", profile_stat_tensor, A, T),
        allow_non_finite=not bool(fail_on_non_finite_input),
    )

    structure = _as_float64(
        "structure_tensor",
        _canonicalize_at_prefix("structure_tensor", structure_tensor, A, T),
        allow_non_finite=not bool(fail_on_non_finite_input),
    )
    context = _as_float64(
        "context_tensor",
        _canonicalize_at_prefix("context_tensor", context_tensor, A, T),
        allow_non_finite=not bool(fail_on_non_finite_input),
    )
    fingerprint = _as_float64(
        "profile_fingerprint_tensor",
        _canonicalize_at_prefix("profile_fingerprint_tensor", profile_fingerprint_tensor, A, T),
        allow_non_finite=not bool(fail_on_non_finite_input),
    )
    regime = _as_float64(
        "profile_regime_tensor",
        _canonicalize_at_prefix("profile_regime_tensor", profile_regime_tensor, A, T),
        allow_non_finite=not bool(fail_on_non_finite_input),
    )

    _validate_profile_window_shapes(
        structure_tensor=structure,
        context_tensor=context,
        profile_fingerprint_tensor=fingerprint,
        profile_regime_tensor=regime,
    )

    tradable = _as_bool(
        "tradable_mask",
        _canonicalize_at_prefix("tradable_mask", tradable_mask, A, T),
    )
    if tradable.ndim != 2:
        raise RuntimeError(f"tradable_mask must be 2D [A,T], got ndim={tradable.ndim}")

    grid_summary = None
    if profile_grid_summary_tensor is not None:
        grid_summary = _as_float64(
            "profile_grid_summary_tensor",
            _canonicalize_at_prefix("profile_grid_summary_tensor", profile_grid_summary_tensor, A, T),
            allow_non_finite=not bool(fail_on_non_finite_input),
        )

    vol = None
    if volatility_tensor is not None:
        vol = _as_float64(
            "volatility_tensor",
            _canonicalize_at_prefix("volatility_tensor", volatility_tensor, A, T),
            allow_non_finite=not bool(fail_on_non_finite_input),
        )
        _validate_tensor_shape("volatility_tensor", vol, (A, T))

    spr = None
    if spread_tensor is not None:
        spr = _as_float64(
            "spread_tensor",
            _canonicalize_at_prefix("spread_tensor", spread_tensor, A, T),
            allow_non_finite=not bool(fail_on_non_finite_input),
        )
        _validate_tensor_shape("spread_tensor", spr, (A, T))

    liq = None
    if liquidity_score is not None:
        liq = _as_float64(
            "liquidity_score",
            _canonicalize_at_prefix("liquidity_score", liquidity_score, A, T),
            allow_non_finite=not bool(fail_on_non_finite_input),
        )
        _validate_tensor_shape("liquidity_score", liq, (A, T))

    source_idx = None
    if source_time_index_at is not None:
        source_idx = _as_int64(
            "source_time_index_at",
            _canonicalize_at_prefix("source_time_index_at", source_time_index_at, A, T),
        )
        _validate_tensor_shape("source_time_index_at", source_idx, (A, T))
        assert_causal_source_index(source_idx)

    return Module4InputContracts(
        alpha_signal_tensor=alpha,
        score_tensor=score,
        profile_stat_tensor=profile_stat,
        structure_tensor=structure,
        context_tensor=context,
        profile_fingerprint_tensor=fingerprint,
        profile_regime_tensor=regime,
        tradable_mask=tradable,
        phase_code=phase,
        asset_enabled_mask=asset_enabled,
        profile_grid_summary_tensor=grid_summary,
        volatility_tensor=vol,
        spread_tensor=spr,
        liquidity_score=liq,
        source_time_index_at=source_idx,
    )


def apply_optional_risk_filters(
    *,
    tradable_mask_at: np.ndarray,
    volatility_tensor_at: np.ndarray | None,
    spread_tensor_at: np.ndarray | None,
    liquidity_score_at: np.ndarray | None,
    cfg: RiskFilterConfig,
) -> np.ndarray:
    """Apply deterministic pre-intent risk-aware tradability filters."""
    tradable = _as_bool("tradable_mask_at", tradable_mask_at)
    if tradable.ndim != 2:
        raise RuntimeError(f"tradable_mask_at must be 2D [A,T], got ndim={tradable.ndim}")

    out = tradable.copy()
    if volatility_tensor_at is not None:
        vol = _as_float64("volatility_tensor_at", volatility_tensor_at, allow_non_finite=True)
        _validate_tensor_shape("volatility_tensor_at", vol, out.shape)
        out &= np.isfinite(vol) & (vol < float(cfg.max_volatility))
    if spread_tensor_at is not None:
        spr = _as_float64("spread_tensor_at", spread_tensor_at, allow_non_finite=True)
        _validate_tensor_shape("spread_tensor_at", spr, out.shape)
        out &= np.isfinite(spr) & (spr < float(cfg.max_spread))
    if liquidity_score_at is not None:
        liq = _as_float64("liquidity_score_at", liquidity_score_at, allow_non_finite=True)
        _validate_tensor_shape("liquidity_score_at", liq, out.shape)
        out &= np.isfinite(liq) & (liq > float(cfg.min_liquidity))
    return out
