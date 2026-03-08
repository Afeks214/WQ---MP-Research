from __future__ import annotations

from dataclasses import dataclass
import numpy as np

SQRT_2PI: float = float(np.sqrt(2.0 * np.pi))


@dataclass(frozen=True)
class MixtureBarParams:
    mu1: np.ndarray
    mu2: np.ndarray
    sigma1: np.ndarray
    sigma2: np.ndarray
    w1: np.ndarray
    w2: np.ndarray
    vprof: np.ndarray
    pbuy: np.ndarray
    delta_coeff: np.ndarray


@dataclass(frozen=True)
class InjectionResult:
    total_an: np.ndarray
    delta_an: np.ndarray
    m0_a: np.ndarray
    m1_a: np.ndarray
    m2_a: np.ndarray


@dataclass(frozen=True)
class ScoreInputs:
    ret_norm: np.ndarray
    s_r: np.ndarray
    clv: np.ndarray
    body_pct: np.ndarray


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _gaussian_pdf(x_grid: np.ndarray, mu_a: np.ndarray, sigma_a: np.ndarray) -> np.ndarray:
    z = (x_grid[None, :] - mu_a[:, None]) / sigma_a[:, None]
    pdf = np.exp(-0.5 * z * z) / (sigma_a[:, None] * SQRT_2PI)
    return np.where(np.isfinite(pdf), pdf, 0.0)


def compute_pbuy_and_delta_coeff(inputs: ScoreInputs, eps_div_a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Locked sealed constants from the mathematical spec section used in module2 core.
    p_sr_buy = _sigmoid(np.log(9.0) * inputs.ret_norm / (inputs.s_r + eps_div_a))
    p_clv_buy = _sigmoid(6.0 * inputs.clv)
    w_trend = np.clip(inputs.body_pct, 0.0, 1.0)
    p_buy = np.clip(w_trend * p_sr_buy + (1.0 - w_trend) * p_clv_buy, 0.0, 1.0)
    delta_coeff = 2.0 * p_buy - 1.0
    delta_coeff = np.where(np.isfinite(delta_coeff), delta_coeff, 0.0)
    return p_buy, delta_coeff


def build_bar_mixture_params(
    *,
    open_a: np.ndarray,
    high_a: np.ndarray,
    low_a: np.ndarray,
    close_a: np.ndarray,
    atr_eff_a: np.ndarray,
    rvol_a: np.ndarray,
    clv_a: np.ndarray,
    body_pct_a: np.ndarray,
    sigma1_a: np.ndarray,
    sigma2_a: np.ndarray,
    w1_a: np.ndarray,
    w2_a: np.ndarray,
    volume_a: np.ndarray,
    cap_v_eff_a: np.ndarray,
    score_inputs: ScoreInputs,
    eps_div_a: np.ndarray,
    dx: float,
    sealed_mode: bool,
    mu1_clv_shift: float,
    mu2_clv_shift: float,
) -> MixtureBarParams:
    denom = atr_eff_a + eps_div_a
    mid_a = 0.5 * (open_a + close_a)
    mu_base = (mid_a - close_a) / denom
    if sealed_mode:
        mu1 = mu_base
        mu2 = mu_base
    else:
        mu1 = mu_base + float(mu1_clv_shift) * clv_a
        mu2 = mu_base + float(mu2_clv_shift) * clv_a

    s1 = np.maximum(np.asarray(sigma1_a, dtype=np.float64), float(dx))
    s2 = np.maximum(np.asarray(sigma2_a, dtype=np.float64), float(dx))
    w1 = np.clip(np.asarray(w1_a, dtype=np.float64), 0.0, 1.0)
    w2 = np.clip(np.asarray(w2_a, dtype=np.float64), 0.0, 1.0)
    wsum = w1 + w2
    w1 = np.divide(w1, wsum, out=np.ones_like(w1), where=wsum > 0.0)
    w2 = 1.0 - w1

    vprof = np.minimum(np.maximum(volume_a, 0.0), np.maximum(cap_v_eff_a, 0.0))
    vprof = np.where(np.isfinite(vprof), vprof * np.maximum(rvol_a, 0.0), 0.0)

    pbuy, delta_coeff = compute_pbuy_and_delta_coeff(score_inputs, eps_div_a)

    return MixtureBarParams(
        mu1=mu1,
        mu2=mu2,
        sigma1=s1,
        sigma2=s2,
        w1=w1,
        w2=w2,
        vprof=vprof,
        pbuy=pbuy,
        delta_coeff=delta_coeff,
    )


def inject_profile_mass(
    *,
    params: MixtureBarParams,
    x_grid: np.ndarray,
    dx: float,
    eps_pdf: float,
    valid_a: np.ndarray,
) -> InjectionResult:
    pdf1 = _gaussian_pdf(x_grid, params.mu1, params.sigma1)
    pdf2 = _gaussian_pdf(x_grid, params.mu2, params.sigma2)

    mix = params.w1[:, None] * pdf1 + params.w2[:, None] * pdf2
    mix = np.where(np.isfinite(mix), mix, 0.0)

    norm = np.sum(mix, axis=1) * float(dx)
    mix = np.divide(
        mix,
        norm[:, None] + float(eps_pdf),
        out=np.zeros_like(mix),
        where=norm[:, None] > 0.0,
    )
    total_an = params.vprof[:, None] * mix
    total_an = np.where(valid_a[:, None], total_an, 0.0)
    total_an = np.where(np.isfinite(total_an), total_an, 0.0)

    delta_an = total_an * params.delta_coeff[:, None]
    delta_an = np.where(np.isfinite(delta_an), delta_an, 0.0)

    # Closed-form mixture moments before discretization; used for grid-optional profile stats.
    mu_mix = params.w1 * params.mu1 + params.w2 * params.mu2
    var_mix = (
        params.w1 * (params.sigma1 * params.sigma1 + (params.mu1 - mu_mix) ** 2)
        + params.w2 * (params.sigma2 * params.sigma2 + (params.mu2 - mu_mix) ** 2)
    )
    m0 = params.vprof
    m1 = params.vprof * mu_mix
    m2 = params.vprof * (mu_mix * mu_mix + var_mix)
    m0 = np.where(valid_a, np.where(np.isfinite(m0), m0, 0.0), 0.0)
    m1 = np.where(valid_a, np.where(np.isfinite(m1), m1, 0.0), 0.0)
    m2 = np.where(valid_a, np.where(np.isfinite(m2), m2, 0.0), 0.0)

    return InjectionResult(total_an=total_an, delta_an=delta_an, m0_a=m0, m1_a=m1, m2_a=m2)
