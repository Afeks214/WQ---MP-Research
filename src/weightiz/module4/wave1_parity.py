from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x64))


@dataclass(frozen=True)
class F6ParityReport:
    passed: bool
    numeric_p95_abs_err: float
    numeric_max_abs_err: float
    behavioral_mismatch_rate: float
    branch_mismatch_rate: float
    firing_mismatch_rate: float
    collision_mismatch_rate: float
    deadzone_violation_rate: float
    total_points: int
    numeric_tol: float
    behavioral_tol: float
    strict_behavior_checked: bool
    reason: str


def evaluate_f6_parity(
    *,
    dclip: np.ndarray,
    rvol: np.ndarray,
    gbreak: np.ndarray,
    score_bo_long: np.ndarray,
    score_bo_short: np.ndarray,
    d_value: np.ndarray | None = None,
    score_reject: np.ndarray | None = None,
    greject: np.ndarray | None = None,
    z_delta: np.ndarray | None = None,
    x_vah: np.ndarray | None = None,
    x_val: np.ndarray | None = None,
    numeric_tol: float = 1.0e-4,
    behavioral_tol: float = 0.02,
    deadzone_low: float = 0.40,
    deadzone_high: float = 0.55,
    require_strict_behavior: bool = False,
) -> F6ParityReport:
    """
    F6 parity gate for reconstructed pre-gated breakout base.

    Method A (paper reconstruction):
      bo_base_long_A  = sigmoid(Dclip - 1.0) * RVOL
      bo_base_short_A = sigmoid(-Dclip - 1.0) * RVOL

    Method B (observable de-gating):
      bo_base_long_B  = SCORE_BO_LONG  / max(GBREAK, eps)
      bo_base_short_B = SCORE_BO_SHORT / max(GBREAK, eps)

    We compare only on rows where G_break is informative and out of dead-zone.
    """
    dclip_a = np.asarray(dclip, dtype=np.float64)
    rvol_a = np.asarray(rvol, dtype=np.float64)
    gbreak_a = np.asarray(gbreak, dtype=np.float64)
    sbo_l = np.asarray(score_bo_long, dtype=np.float64)
    sbo_s = np.asarray(score_bo_short, dtype=np.float64)

    if dclip_a.shape != rvol_a.shape or dclip_a.shape != gbreak_a.shape:
        raise RuntimeError("F6 parity arrays dclip/rvol/gbreak must share shape")
    if dclip_a.shape != sbo_l.shape or dclip_a.shape != sbo_s.shape:
        raise RuntimeError("F6 parity score arrays must share shape with dclip")

    finite = (
        np.isfinite(dclip_a)
        & np.isfinite(rvol_a)
        & np.isfinite(gbreak_a)
        & np.isfinite(sbo_l)
        & np.isfinite(sbo_s)
    )
    informative = finite & (gbreak_a >= 0.10) & ((gbreak_a < deadzone_low) | (gbreak_a > deadzone_high))
    total_points = int(np.sum(informative))
    if total_points <= 0:
        return F6ParityReport(
            passed=False,
            numeric_p95_abs_err=float("inf"),
            numeric_max_abs_err=float("inf"),
            behavioral_mismatch_rate=1.0,
            branch_mismatch_rate=1.0,
            firing_mismatch_rate=1.0,
            collision_mismatch_rate=1.0,
            deadzone_violation_rate=1.0,
            total_points=0,
            numeric_tol=float(numeric_tol),
            behavioral_tol=float(behavioral_tol),
            strict_behavior_checked=False,
            reason="NO_INFORMATIVE_POINTS",
        )

    base_l_a = _sigmoid(dclip_a - 1.0) * rvol_a
    base_s_a = _sigmoid((-dclip_a) - 1.0) * rvol_a

    eps = 1.0e-12
    base_l_b = np.divide(sbo_l, np.maximum(gbreak_a, eps))
    base_s_b = np.divide(sbo_s, np.maximum(gbreak_a, eps))

    abs_err_l = np.abs(base_l_a - base_l_b)
    abs_err_s = np.abs(base_s_a - base_s_b)
    abs_err = np.maximum(abs_err_l, abs_err_s)
    err_obs = abs_err[informative]
    numeric_p95 = float(np.quantile(err_obs, 0.95))
    numeric_max = float(np.max(err_obs))

    # Behavioral consistency for ext-up/ext-down branching.
    ext_up_a = (base_l_a >= 0.60) & (gbreak_a <= deadzone_low) & informative
    ext_dn_a = (base_s_a >= 0.60) & (gbreak_a <= deadzone_low) & informative
    ext_up_b = (base_l_b >= 0.60) & (gbreak_a <= deadzone_low) & informative
    ext_dn_b = (base_s_b >= 0.60) & (gbreak_a <= deadzone_low) & informative
    branch_mismatch = ((ext_up_a != ext_up_b) | (ext_dn_a != ext_dn_b)) & informative
    branch_mismatch_rate = float(np.mean(branch_mismatch[informative])) if total_points > 0 else 1.0

    strict_inputs_present = all(
        arr is not None
        for arr in (d_value, score_reject, greject, z_delta, x_vah, x_val)
    )
    strict_behavior_checked = False
    firing_mismatch_rate = branch_mismatch_rate
    collision_mismatch_rate = branch_mismatch_rate
    deadzone_violation_rate = 0.0

    strict_missing_reason = ""
    if require_strict_behavior and (not strict_inputs_present):
        strict_missing_reason = "STRICT_BEHAVIOR_CONTEXT_MISSING"

    if strict_inputs_present:
        d_val = np.asarray(d_value, dtype=np.float64)
        reject = np.asarray(score_reject, dtype=np.float64)
        grej = np.asarray(greject, dtype=np.float64)
        zdel = np.asarray(z_delta, dtype=np.float64)
        xv = np.asarray(x_vah, dtype=np.float64)
        xl = np.asarray(x_val, dtype=np.float64)
        extra = (d_val, reject, grej, zdel, xv, xl)
        if any(arr.shape != dclip_a.shape for arr in extra):
            raise RuntimeError("F6 strict parity context arrays must share shape with dclip")

        strict_finite = finite
        for arr in extra:
            strict_finite &= np.isfinite(arr)
        strict_mask = informative & strict_finite
        strict_points = int(np.sum(strict_mask))
        if strict_points > 0:
            strict_behavior_checked = True

            f6_up_a = (
                (base_l_a >= 0.60)
                & (d_val > 1.20)
                & (gbreak_a <= deadzone_low)
                & (sbo_l <= 0.35)
                & (rvol_a <= 1.40)
                & ((reject >= 0.45) | (grej >= 0.50))
                & strict_mask
            )
            f6_dn_a = (
                (base_s_a >= 0.60)
                & (d_val < -1.20)
                & (gbreak_a <= deadzone_low)
                & (sbo_s <= 0.35)
                & (rvol_a <= 1.40)
                & ((reject >= 0.45) | (grej >= 0.50))
                & strict_mask
            )
            f6_up_b = (
                (base_l_b >= 0.60)
                & (d_val > 1.20)
                & (gbreak_a <= deadzone_low)
                & (sbo_l <= 0.35)
                & (rvol_a <= 1.40)
                & ((reject >= 0.45) | (grej >= 0.50))
                & strict_mask
            )
            f6_dn_b = (
                (base_s_b >= 0.60)
                & (d_val < -1.20)
                & (gbreak_a <= deadzone_low)
                & (sbo_s <= 0.35)
                & (rvol_a <= 1.40)
                & ((reject >= 0.45) | (grej >= 0.50))
                & strict_mask
            )
            f6_sig_a = np.where(f6_up_a, -1, np.where(f6_dn_a, 1, 0))
            f6_sig_b = np.where(f6_up_b, -1, np.where(f6_dn_b, 1, 0))
            firing_mismatch_rate = float(np.mean((f6_sig_a != f6_sig_b)[strict_mask]))

            bo_dom = np.maximum(sbo_l, sbo_s)
            breakout_up = xv < -0.10
            breakout_dn = xl > 0.10
            breakout_ctx = breakout_up | breakout_dn | (np.abs(d_val) >= 1.2)

            f5_long = (
                breakout_up
                & (gbreak_a >= deadzone_high)
                & (zdel >= 1.0)
                & (rvol_a >= 1.30)
                & (sbo_l >= 0.45)
                & (reject < 0.80 * np.maximum(bo_dom, 1.0e-12))
                & strict_mask
            )
            f5_short = (
                breakout_dn
                & (gbreak_a >= deadzone_high)
                & (zdel <= -1.0)
                & (rvol_a >= 1.30)
                & (sbo_s >= 0.45)
                & (reject < 0.80 * np.maximum(bo_dom, 1.0e-12))
                & strict_mask
            )

            dead_zone_mask = breakout_ctx & (gbreak_a >= deadzone_low) & (gbreak_a <= deadzone_high) & strict_mask

            f6_sig_a = np.where(dead_zone_mask, 0, f6_sig_a)
            f6_sig_b = np.where(dead_zone_mask, 0, f6_sig_b)
            f5_sig = np.where(f5_long, 1, np.where(f5_short, -1, 0))
            f5_sig = np.where(dead_zone_mask, 0, f5_sig)

            out_a = np.zeros(dclip_a.shape, dtype=np.int8)
            out_b = np.zeros(dclip_a.shape, dtype=np.int8)
            # 11/12: F5 long/short, 21/22: F6 long/short
            out_a[(breakout_ctx & strict_mask) & (f5_sig == 1) & (f6_sig_a == 0)] = 11
            out_a[(breakout_ctx & strict_mask) & (f5_sig == -1) & (f6_sig_a == 0)] = 12
            out_a[(breakout_ctx & strict_mask) & (f6_sig_a == 1) & (f5_sig == 0)] = 21
            out_a[(breakout_ctx & strict_mask) & (f6_sig_a == -1) & (f5_sig == 0)] = 22

            out_b[(breakout_ctx & strict_mask) & (f5_sig == 1) & (f6_sig_b == 0)] = 11
            out_b[(breakout_ctx & strict_mask) & (f5_sig == -1) & (f6_sig_b == 0)] = 12
            out_b[(breakout_ctx & strict_mask) & (f6_sig_b == 1) & (f5_sig == 0)] = 21
            out_b[(breakout_ctx & strict_mask) & (f6_sig_b == -1) & (f5_sig == 0)] = 22

            collision_mismatch_rate = float(np.mean((out_a != out_b)[strict_mask]))
            dead_count = int(np.sum(dead_zone_mask))
            if dead_count > 0:
                deadzone_violation_rate = float(
                    np.mean(((out_a != 0) | (out_b != 0))[dead_zone_mask])
                )
            else:
                deadzone_violation_rate = 0.0
        elif require_strict_behavior:
            strict_missing_reason = "STRICT_BEHAVIOR_NO_POINTS"

    behavioral_mismatch_rate = float(
        max(
            branch_mismatch_rate,
            firing_mismatch_rate,
            collision_mismatch_rate,
            deadzone_violation_rate,
        )
    )

    numeric_ok = bool(numeric_p95 <= float(numeric_tol))
    strict_ok = bool((not require_strict_behavior) or strict_behavior_checked)
    deadzone_ok = bool(deadzone_violation_rate <= 1.0e-12)
    behavioral_ok = bool(behavioral_mismatch_rate <= float(behavioral_tol))
    passed = bool(numeric_ok and strict_ok and deadzone_ok and behavioral_ok)

    if passed:
        reason = "PASS"
    elif not numeric_ok:
        reason = "NUMERIC_FAIL"
    elif not strict_ok:
        reason = strict_missing_reason if strict_missing_reason else "STRICT_BEHAVIOR_FAIL"
    elif not deadzone_ok:
        reason = "DEADZONE_VIOLATION"
    else:
        reason = "BEHAVIORAL_FAIL"

    return F6ParityReport(
        passed=passed,
        numeric_p95_abs_err=numeric_p95,
        numeric_max_abs_err=numeric_max,
        behavioral_mismatch_rate=behavioral_mismatch_rate,
        branch_mismatch_rate=branch_mismatch_rate,
        firing_mismatch_rate=firing_mismatch_rate,
        collision_mismatch_rate=collision_mismatch_rate,
        deadzone_violation_rate=deadzone_violation_rate,
        total_points=total_points,
        numeric_tol=float(numeric_tol),
        behavioral_tol=float(behavioral_tol),
        strict_behavior_checked=bool(strict_behavior_checked),
        reason=reason,
    )
