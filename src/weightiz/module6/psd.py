from __future__ import annotations

import numpy as np

from weightiz.module6.config import DependenceConfig
from weightiz.module6.utils import Module6ValidationError


def ledoit_wolf_diag_shrink(returns: np.ndarray) -> tuple[np.ndarray, float]:
    x = np.asarray(returns, dtype=np.float64)
    if x.ndim != 2:
        raise Module6ValidationError(f"returns must be 2D for covariance shrinkage; ndim={x.ndim}")
    t_count, n_count = x.shape
    if t_count <= 1:
        raise Module6ValidationError("at least two observations required for covariance shrinkage")
    xc = x - np.mean(x, axis=0, keepdims=True)
    sample_cov = (xc.T @ xc) / float(t_count - 1)
    target = np.diag(np.diag(sample_cov))
    gamma = float(np.sum((sample_cov - target) ** 2))
    if gamma <= 1.0e-18:
        return target.astype(np.float64), 1.0
    phi_accum = 0.0
    for row in xc:
        outer = np.outer(row, row)
        phi_accum += float(np.sum((outer - sample_cov) ** 2))
    phi = phi_accum / float(t_count)
    shrink = min(1.0, max(0.0, float(phi / max(gamma * float(t_count), 1.0e-18))))
    cov = (1.0 - shrink) * sample_cov + shrink * target
    return cov.astype(np.float64), float(shrink)


def eigen_floor_psd(covariance: np.ndarray, config: DependenceConfig) -> tuple[np.ndarray, float]:
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise Module6ValidationError("covariance matrix must be square for PSD projection")
    cov = 0.5 * (cov + cov.T)
    trace = float(np.trace(cov))
    eps = float(config.shrinkage_floor_eps_mult * trace / max(cov.shape[0], 1))
    eigvals, eigvecs = np.linalg.eigh(cov)
    neg_mass = float(np.sum(np.abs(eigvals[eigvals < 0.0])))
    eigvals = np.maximum(eigvals, eps)
    repaired = (eigvecs * eigvals) @ eigvecs.T
    repaired = 0.5 * (repaired + repaired.T)
    return repaired.astype(np.float64), neg_mass


def enforce_psd(covariance: np.ndarray, config: DependenceConfig) -> tuple[np.ndarray, float]:
    repaired, neg_mass = eigen_floor_psd(covariance, config)
    trace = float(np.trace(np.asarray(covariance, dtype=np.float64)))
    if neg_mass > float(config.negative_mass_reject_mult * max(trace, 1.0e-18)):
        raise Module6ValidationError(
            f"covariance rejected due to excessive negative eigen-mass; negative_mass={neg_mass} trace={trace}"
        )
    return repaired, neg_mass

