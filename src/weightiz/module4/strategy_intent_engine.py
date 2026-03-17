from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from weightiz.module1.core import ProfileStatIdx, ScoreIdx


REGIME_NONE = np.int8(0)
REGIME_NEUTRAL = np.int8(1)
REGIME_TREND = np.int8(2)
REGIME_P_SHAPE = np.int8(3)
REGIME_B_SHAPE = np.int8(4)
REGIME_DOUBLE_DISTRIBUTION = np.int8(5)
GATE_COUNT = 6

WAVE1_ALPHA_RVOL = 0
WAVE1_ALPHA_TOD = 1
WAVE1_ALPHA_X_POC = 2
WAVE1_ALPHA_X_VAH = 3
WAVE1_ALPHA_X_VAL = 4
WAVE1_ALPHA_VA_WIDTH = 5
WAVE1_ALPHA_SESSION_ID = 6

WAVE1_REGIME_INDET = np.int8(0)
WAVE1_REGIME_BALANCE = np.int8(1)
WAVE1_REGIME_TREND = np.int8(2)
WAVE1_REGIME_TRANSITION = np.int8(3)

FAMILY_NONE = np.int8(0)
FAMILY_F3 = np.int8(3)
FAMILY_F5 = np.int8(5)
FAMILY_F6 = np.int8(6)
FAMILY_F5_OVERLAY = np.int8(15)


@dataclass(frozen=True)
class StrategyIntentResult:
    intent_long: np.ndarray
    intent_short: np.ndarray
    intent_flat: np.ndarray
    intent_valid_mask: np.ndarray
    intent_gate_mask: np.ndarray
    signed_intent_utility: np.ndarray


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x64))


def _alpha_or_default(alpha: np.ndarray, idx: int, default: float) -> np.ndarray:
    if alpha.shape[2] <= idx:
        return np.full(alpha.shape[:2], float(default), dtype=np.float64)
    out = np.asarray(alpha[:, :, idx], dtype=np.float64)
    return np.where(np.isfinite(out), out, float(default))


def _compute_wave1_regime(
    *,
    d_val: np.ndarray,
    bo_dom: np.ndarray,
    reject: np.ndarray,
    rvol: np.ndarray,
    gbreak: np.ndarray,
) -> np.ndarray:
    regime = np.full(d_val.shape, WAVE1_REGIME_INDET, dtype=np.int8)
    balance = (np.abs(d_val) <= 1.2) & (np.abs(bo_dom - reject) <= 0.10) & (rvol < 1.5)
    trend = (gbreak >= 0.55) & (bo_dom > 1.25 * np.maximum(reject, 1.0e-12)) & (np.abs(d_val) > 1.0)
    transition = (
        (gbreak > 0.45)
        & (gbreak < 0.55)
        & (bo_dom > 1.10 * np.maximum(reject, 1.0e-12))
        & (np.abs(d_val) > 0.8)
    )
    regime[balance] = WAVE1_REGIME_BALANCE
    regime[transition] = WAVE1_REGIME_TRANSITION
    regime[trend] = WAVE1_REGIME_TREND
    return regime


def _generate_wave1_strategy_intent(
    *,
    alpha: np.ndarray,
    score: np.ndarray,
    profile: np.ndarray,
    tradable: np.ndarray,
    finite_mask: np.ndarray,
    cfg4: object,
) -> StrategyIntentResult:
    A, T, _ = score.shape

    bo_long = score[:, :, int(ScoreIdx.SCORE_BO_LONG)]
    bo_short = score[:, :, int(ScoreIdx.SCORE_BO_SHORT)]
    reject = score[:, :, int(ScoreIdx.SCORE_REJECT)]
    bo_dom = np.maximum(bo_long, bo_short)

    d_val = profile[:, :, int(ProfileStatIdx.D)]
    dclip = profile[:, :, int(ProfileStatIdx.DCLIP)]
    z_delta = profile[:, :, int(ProfileStatIdx.Z_DELTA)]
    gbreak = profile[:, :, int(ProfileStatIdx.GBREAK)]
    greject = profile[:, :, int(ProfileStatIdx.GREJECT)]

    rvol = _alpha_or_default(alpha, WAVE1_ALPHA_RVOL, 1.0)
    tod = _alpha_or_default(alpha, WAVE1_ALPHA_TOD, 0.0)
    x_vah = _alpha_or_default(alpha, WAVE1_ALPHA_X_VAH, np.nan)
    x_val = _alpha_or_default(alpha, WAVE1_ALPHA_X_VAL, np.nan)
    session_id = _alpha_or_default(alpha, WAVE1_ALPHA_SESSION_ID, 0.0)

    # F6 reconstructed pre-gated breakout base from canonical exported DCLIP + RVOL.
    bo_base_long = _sigmoid(dclip - 1.0) * rvol
    bo_base_short = _sigmoid((-dclip) - 1.0) * rvol

    dead_low = 0.40
    dead_high = 0.55

    breakout_up = x_vah < -0.10
    breakout_dn = x_val > 0.10
    breakout_ctx = breakout_up | breakout_dn | (np.abs(d_val) >= 1.2)

    f5_long = (
        breakout_up
        & (gbreak >= dead_high)
        & (z_delta >= 1.0)
        & (rvol >= 1.30)
        & (bo_long >= 0.45)
        & (reject < 0.80 * np.maximum(bo_dom, 1.0e-12))
    )
    f5_short = (
        breakout_dn
        & (gbreak >= dead_high)
        & (z_delta <= -1.0)
        & (rvol >= 1.30)
        & (bo_short >= 0.45)
        & (reject < 0.80 * np.maximum(bo_dom, 1.0e-12))
    )

    f6_enabled = bool(getattr(cfg4, "f6_enabled", True))
    if bool(getattr(cfg4, "f6_parity_required", False)) and (not f6_enabled):
        f6_enabled = False
    f6_ext_up = (
        (bo_base_long >= 0.60)
        & (d_val > 1.20)
        & (gbreak <= dead_low)
        & (bo_long <= 0.35)
        & (rvol <= 1.40)
        & ((reject >= 0.45) | (greject >= 0.50))
    )
    f6_ext_dn = (
        (bo_base_short >= 0.60)
        & (d_val < -1.20)
        & (gbreak <= dead_low)
        & (bo_short <= 0.35)
        & (rvol <= 1.40)
        & ((reject >= 0.45) | (greject >= 0.50))
    )
    if not f6_enabled:
        f6_ext_up[:] = False
        f6_ext_dn[:] = False

    upper_edge = np.abs(x_vah) <= 0.20
    lower_edge = np.abs(x_val) <= 0.20
    f3_reject_dom = (reject >= 0.55) & (reject >= 1.20 * np.maximum(bo_dom, 1.0e-12)) & (rvol < 1.50)
    f3_long = lower_edge & (d_val < 0.0) & f3_reject_dom & (z_delta >= 1.0) & (greject >= 0.55)
    f3_short = upper_edge & (d_val > 0.0) & f3_reject_dom & (z_delta <= -1.0) & (greject >= 0.55)

    wave1_regime = _compute_wave1_regime(
        d_val=d_val,
        bo_dom=bo_dom,
        reject=reject,
        rvol=rvol,
        gbreak=gbreak,
    )

    intent_long = np.zeros((A, T), dtype=bool)
    intent_short = np.zeros((A, T), dtype=bool)
    signed_utility = np.zeros((A, T), dtype=np.float64)
    family_code = np.full((A, T), FAMILY_NONE, dtype=np.int8)
    gate_mask = np.zeros((A, T, GATE_COUNT), dtype=bool)
    gate_mask[:, :, 0] = finite_mask
    gate_mask[:, :, 1] = tradable
    gate_mask[:, :, 2] = breakout_ctx

    family_filter = str(getattr(cfg4, "family_id", "")).strip().upper()
    family_f3_allowed = family_filter in {"", "AUTO", "F3", "ALL"}
    family_f5_allowed = family_filter in {"", "AUTO", "F5", "ALL"}
    family_f6_allowed = family_filter in {"", "AUTO", "F6", "ALL"} and f6_enabled

    enable_overlay = bool(getattr(cfg4, "enable_f5_close_overlay", True))

    for a in range(A):
        overlay_active = False
        overlay_dir = 0
        overlay_sessions = 0
        prev_sid = int(np.rint(session_id[a, 0])) if T > 0 else 0
        for t in range(T):
            sid_t = int(np.rint(session_id[a, t]))
            if sid_t != prev_sid:
                if overlay_active:
                    overlay_sessions += 1
                prev_sid = sid_t

            valid = bool(finite_mask[a, t] and tradable[a, t] and np.isfinite(tod[a, t]) and (tod[a, t] >= 15.0))
            if not valid:
                overlay_active = False if overlay_sessions >= 3 else overlay_active
                continue

            f5_sig = 0
            if family_f5_allowed:
                if bool(f5_long[a, t]):
                    f5_sig = 1
                elif bool(f5_short[a, t]):
                    f5_sig = -1
            f6_sig = 0
            if family_f6_allowed:
                if bool(f6_ext_up[a, t]):
                    f6_sig = -1
                elif bool(f6_ext_dn[a, t]):
                    f6_sig = 1
            f3_sig = 0
            if family_f3_allowed:
                if bool(f3_long[a, t]):
                    f3_sig = 1
                elif bool(f3_short[a, t]):
                    f3_sig = -1

            dead_zone = bool(breakout_ctx[a, t] and (gbreak[a, t] >= dead_low) and (gbreak[a, t] <= dead_high))
            if dead_zone:
                f5_sig = 0
                f6_sig = 0
            gate_mask[a, t, 3] = bool(f5_sig != 0)
            gate_mask[a, t, 4] = bool(f6_sig != 0)
            gate_mask[a, t, 5] = bool(f3_sig != 0)

            sel_sig = 0
            sel_family = FAMILY_NONE

            # Step 3/4: breakout trichotomy + dead-zone
            if breakout_ctx[a, t]:
                if (f5_sig != 0) and (f6_sig == 0):
                    sel_sig = int(f5_sig)
                    sel_family = FAMILY_F5
                elif (f6_sig != 0) and (f5_sig == 0):
                    sel_sig = int(f6_sig)
                    sel_family = FAMILY_F6
                elif (f5_sig != 0) and (f6_sig != 0):
                    sel_sig = 0
                    sel_family = FAMILY_NONE

            # Step 5/6/7: responsive path + collisions.
            if sel_sig == 0 and (f3_sig != 0) and (not dead_zone):
                sel_sig = int(f3_sig)
                sel_family = FAMILY_F3
            elif (sel_sig != 0) and (f3_sig != 0):
                if sel_family == FAMILY_F5:
                    rg = int(wave1_regime[a, t])
                    if rg == int(WAVE1_REGIME_BALANCE):
                        sel_sig = int(f3_sig)
                        sel_family = FAMILY_F3
                    elif rg in {int(WAVE1_REGIME_TREND), int(WAVE1_REGIME_TRANSITION)}:
                        pass
                    else:
                        sel_sig = 0
                        sel_family = FAMILY_NONE
                elif sel_family == FAMILY_F6:
                    if float(np.abs(d_val[a, t])) < 1.8:
                        sel_sig = int(f3_sig)
                        sel_family = FAMILY_F3

            # Step 8: F5 close-window extension overlay (F7 demoted).
            if enable_overlay:
                if overlay_active:
                    # Overlay fail-closed exits.
                    directional_bo = bo_long[a, t] if overlay_dir > 0 else bo_short[a, t]
                    delta_sign_ok = (z_delta[a, t] * float(overlay_dir)) > 0.0
                    reject_flip = (reject[a, t] > directional_bo) and (greject[a, t] > 0.55)
                    target_reached = (overlay_sessions >= 1) and (d_val[a, t] * float(overlay_dir) >= 1.5)
                    if (overlay_sessions >= 3) or (not delta_sign_ok) or reject_flip or target_reached:
                        overlay_active = False
                    else:
                        sel_sig = int(overlay_dir)
                        sel_family = FAMILY_F5_OVERLAY

                if (sel_family == FAMILY_F5) and (330.0 <= tod[a, t] <= 375.0):
                    dir_bo = bo_long[a, t] if sel_sig > 0 else bo_short[a, t]
                    if (
                        (np.abs(z_delta[a, t]) > 1.8)
                        and (gbreak[a, t] >= 0.75)
                        and (rvol[a, t] >= 1.0)
                        and (dir_bo >= 0.45)
                    ):
                        overlay_active = True
                        overlay_dir = int(np.sign(sel_sig))
                        overlay_sessions = 0

            if sel_sig > 0:
                intent_long[a, t] = True
                signed_utility[a, t] = float(max(bo_long[a, t], 0.0))
                family_code[a, t] = np.int8(sel_family)
            elif sel_sig < 0:
                intent_short[a, t] = True
                signed_utility[a, t] = -float(max(bo_short[a, t], 0.0))
                family_code[a, t] = np.int8(sel_family)

    invalid = ~finite_mask
    intent_long[invalid] = False
    intent_short[invalid] = False
    intent_long[~tradable] = False
    intent_short[~tradable] = False
    intent_flat = ~(intent_long | intent_short)
    signed_utility = np.where(intent_flat, 0.0, signed_utility)
    signed_utility[invalid] = 0.0

    out = StrategyIntentResult(
        intent_long=np.ascontiguousarray(intent_long, dtype=bool),
        intent_short=np.ascontiguousarray(intent_short, dtype=bool),
        intent_flat=np.ascontiguousarray(intent_flat, dtype=bool),
        intent_valid_mask=np.ascontiguousarray(finite_mask, dtype=bool),
        intent_gate_mask=np.ascontiguousarray(gate_mask, dtype=bool),
        signed_intent_utility=np.ascontiguousarray(signed_utility, dtype=np.float64),
    )
    # Side-channel family provenance for downstream diagnostics.
    object.__setattr__(out, "family_code", np.ascontiguousarray(family_code, dtype=np.int8))
    object.__setattr__(out, "wave1_regime_code", np.ascontiguousarray(wave1_regime, dtype=np.int8))
    return out


def _generate_legacy_strategy_intent(
    *,
    alpha: np.ndarray,
    score: np.ndarray,
    profile: np.ndarray,
    regime: np.ndarray,
    confidence: np.ndarray,
    tradable: np.ndarray,
    finite_mask: np.ndarray,
    cfg4: object,
) -> StrategyIntentResult:
    alpha_mean = np.mean(alpha, axis=2) if alpha.shape[2] > 0 else np.zeros((alpha.shape[0], alpha.shape[1]), dtype=np.float64)
    bo_long = score[:, :, int(ScoreIdx.SCORE_BO_LONG)]
    bo_short = score[:, :, int(ScoreIdx.SCORE_BO_SHORT)]
    reject = score[:, :, int(ScoreIdx.SCORE_REJECT)] if score.shape[2] > int(ScoreIdx.SCORE_REJECT) else np.zeros((alpha.shape[0], alpha.shape[1]), dtype=np.float64)
    dclip = profile[:, :, int(ProfileStatIdx.DCLIP)] if profile.shape[2] > int(ProfileStatIdx.DCLIP) else np.zeros((alpha.shape[0], alpha.shape[1]), dtype=np.float64)
    z_delta = profile[:, :, int(ProfileStatIdx.Z_DELTA)] if profile.shape[2] > int(ProfileStatIdx.Z_DELTA) else np.zeros((alpha.shape[0], alpha.shape[1]), dtype=np.float64)

    long_utility = bo_long + 0.10 * _clip01(alpha_mean) + 0.05 * _clip01(dclip) + 0.05 * _clip01(z_delta) - 0.05 * _clip01(np.abs(reject))
    short_utility = bo_short + 0.10 * _clip01(-alpha_mean) + 0.05 * _clip01(-dclip) + 0.05 * _clip01(-z_delta) - 0.05 * _clip01(np.abs(reject))

    regime_allows_long = np.isin(regime, np.array([REGIME_NEUTRAL, REGIME_TREND, REGIME_P_SHAPE, REGIME_DOUBLE_DISTRIBUTION], dtype=np.int8))
    regime_allows_short = np.isin(regime, np.array([REGIME_NEUTRAL, REGIME_TREND, REGIME_B_SHAPE, REGIME_DOUBLE_DISTRIBUTION], dtype=np.int8))
    long_threshold_pass = regime_allows_long & tradable & (long_utility >= float(cfg4.entry_threshold))
    short_threshold_pass = regime_allows_short & tradable & (short_utility >= float(cfg4.entry_threshold))

    signed_intent_utility = long_utility - short_utility
    directional_edge = np.abs(signed_intent_utility) > float(cfg4.exit_threshold)

    intent_long = long_threshold_pass & directional_edge & (signed_intent_utility > 0.0)
    intent_short = short_threshold_pass & directional_edge & (signed_intent_utility < 0.0)

    tie = long_threshold_pass & short_threshold_pass & np.isclose(long_utility, short_utility, rtol=0.0, atol=float(cfg4.eps))
    stronger_long = long_threshold_pass & short_threshold_pass & (long_utility > short_utility)
    stronger_short = long_threshold_pass & short_threshold_pass & (short_utility > long_utility)
    intent_long = (intent_long | stronger_long) & (~tie)
    intent_short = (intent_short | stronger_short) & (~tie)

    invalid = ~finite_mask
    intent_long[invalid] = False
    intent_short[invalid] = False
    intent_long[~tradable] = False
    intent_short[~tradable] = False

    intent_flat = ~(intent_long | intent_short)
    signed_intent_utility = np.where(intent_flat, 0.0, signed_intent_utility)
    signed_intent_utility[invalid] = 0.0

    gate_mask = np.zeros((alpha.shape[0], alpha.shape[1], GATE_COUNT), dtype=bool)
    gate_mask[:, :, 0] = finite_mask
    gate_mask[:, :, 1] = tradable
    gate_mask[:, :, 2] = regime_allows_long
    gate_mask[:, :, 3] = regime_allows_short
    gate_mask[:, :, 4] = long_threshold_pass
    gate_mask[:, :, 5] = short_threshold_pass

    return StrategyIntentResult(
        intent_long=np.ascontiguousarray(intent_long, dtype=bool),
        intent_short=np.ascontiguousarray(intent_short, dtype=bool),
        intent_flat=np.ascontiguousarray(intent_flat, dtype=bool),
        intent_valid_mask=np.ascontiguousarray(finite_mask, dtype=bool),
        intent_gate_mask=np.ascontiguousarray(gate_mask, dtype=bool),
        signed_intent_utility=np.ascontiguousarray(signed_intent_utility, dtype=np.float64),
    )


def generate_strategy_intent(
    *,
    alpha_signal_tensor: np.ndarray,
    score_tensor: np.ndarray,
    profile_stat_tensor: np.ndarray,
    regime_id: np.ndarray,
    regime_confidence: np.ndarray,
    tradable_mask: np.ndarray,
    cfg4: object,
) -> StrategyIntentResult:
    alpha = np.asarray(alpha_signal_tensor, dtype=np.float64)
    score = np.asarray(score_tensor, dtype=np.float64)
    profile = np.asarray(profile_stat_tensor, dtype=np.float64)
    regime = np.asarray(regime_id, dtype=np.int8)
    confidence = np.asarray(regime_confidence, dtype=np.float64)
    tradable = np.asarray(tradable_mask, dtype=bool)

    if alpha.ndim != 3 or score.ndim != 3 or profile.ndim != 3:
        raise RuntimeError("Module4 intent inputs must be [A,T,*]")
    A, T, _ = score.shape
    if alpha.shape[0] != A or alpha.shape[1] != T:
        raise RuntimeError("alpha_signal_tensor shape mismatch")
    if profile.shape[0] != A or profile.shape[1] != T:
        raise RuntimeError("profile_stat_tensor shape mismatch")
    if regime.shape != (A, T):
        raise RuntimeError(f"regime_id shape mismatch: got {regime.shape}, expected {(A, T)}")
    if confidence.shape != (A, T):
        raise RuntimeError(f"regime_confidence shape mismatch: got {confidence.shape}, expected {(A, T)}")
    if tradable.shape != (A, T):
        raise RuntimeError(f"tradable_mask shape mismatch: got {tradable.shape}, expected {(A, T)}")

    finite_mask = (
        np.all(np.isfinite(alpha), axis=2)
        & np.all(np.isfinite(score), axis=2)
        & np.all(np.isfinite(profile), axis=2)
        & np.isfinite(confidence)
    )

    strategy_type = str(getattr(cfg4, "strategy_type", "legacy")).strip().lower()
    if strategy_type == "institutional_wave1":
        return _generate_wave1_strategy_intent(
            alpha=alpha,
            score=score,
            profile=profile,
            tradable=tradable,
            finite_mask=finite_mask,
            cfg4=cfg4,
        )

    return _generate_legacy_strategy_intent(
        alpha=alpha,
        score=score,
        profile=profile,
        regime=regime,
        confidence=confidence,
        tradable=tradable,
        finite_mask=finite_mask,
        cfg4=cfg4,
    )
