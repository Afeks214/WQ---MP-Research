from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from weightiz.module6.config import DependenceConfig
from weightiz.module6.utils import Module6ValidationError


@dataclass(frozen=True)
class PsdRepairDiagnostics:
    repaired: np.ndarray
    distortion: float
    negative_mass: float
    negative_mass_ratio: float
    condition_number: float
    off_diagonal_sign_flip_rate: float
    spurious_extreme_correlation_rate: float
    min_eigenvalue_pre: float
    min_eigenvalue_post: float


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    return 0.5 * (arr + arr.T)


def covariance_to_correlation(covariance: np.ndarray, *, diag_eps: float) -> np.ndarray:
    cov = _symmetrize(covariance)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise Module6ValidationError("covariance matrix must be square for correlation conversion")
    diag = np.sqrt(np.maximum(np.diag(cov), float(diag_eps)))
    corr = cov / np.maximum(np.outer(diag, diag), float(diag_eps))
    corr = np.clip(_symmetrize(corr), -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr.astype(np.float64)


def repair_psd_with_diagnostics(covariance: np.ndarray, config: DependenceConfig) -> PsdRepairDiagnostics:
    cov = _symmetrize(covariance)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise Module6ValidationError("covariance matrix must be square for PSD projection")
    if cov.shape[0] <= 0:
        raise Module6ValidationError("covariance matrix must be non-empty for PSD projection")
    if not np.isfinite(cov).all():
        raise Module6ValidationError("non-finite covariance detected before PSD projection")

    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError as exc:
        raise Module6ValidationError("PSD_EIGEN_DECOMPOSITION_FAILED") from exc

    neg_eigs = eigvals[eigvals < 0.0]
    negative_mass = float(np.sum(np.abs(neg_eigs)))
    eig_abs_total = float(np.sum(np.abs(eigvals)))
    negative_mass_ratio = float(negative_mass / max(eig_abs_total, float(config.eps)))
    min_eigenvalue_pre = float(np.min(eigvals))
    trace = float(np.trace(cov))
    eps_psd = float(config.psd_eig_floor_mult * trace / max(cov.shape[0], 1))
    eigvals_repaired = np.maximum(eigvals, eps_psd)
    repaired = (eigvecs * eigvals_repaired) @ eigvecs.T
    repaired = _symmetrize(repaired)
    min_eigenvalue_post = float(np.min(eigvals_repaired))

    repaired_eigs = np.linalg.eigvalsh(repaired)
    max_eig = float(np.max(repaired_eigs))
    min_eig = float(np.min(repaired_eigs))
    condition_number = float(max_eig / max(min_eig, float(config.eps)))
    distortion = float(
        np.linalg.norm(repaired - cov, ord="fro") / max(np.linalg.norm(cov, ord="fro"), float(config.eps))
    )

    corr_before = covariance_to_correlation(cov, diag_eps=float(config.corr_diag_eps))
    corr_after = covariance_to_correlation(repaired, diag_eps=float(config.corr_diag_eps))
    if cov.shape[0] <= 1:
        sign_flip_rate = 0.0
        spurious_extreme_rate = 0.0
    else:
        i_idx, j_idx = np.triu_indices(cov.shape[0], k=1)
        corr_before_off = corr_before[i_idx, j_idx]
        corr_after_off = corr_after[i_idx, j_idx]
        significant_mask = np.abs(corr_before_off) >= float(config.sign_flip_significant_abs_corr_min)
        sign_flips = significant_mask & ((corr_before_off * corr_after_off) < 0.0)
        sign_flip_rate = float(sign_flips.sum() / max(significant_mask.sum(), 1))
        spurious_extreme = (
            (np.abs(corr_after_off) >= float(config.spurious_extreme_post_abs_corr_min))
            & (np.abs(corr_before_off) <= float(config.spurious_extreme_pre_abs_corr_max))
        )
        spurious_extreme_rate = float(spurious_extreme.sum() / max(corr_before_off.size, 1))

    return PsdRepairDiagnostics(
        repaired=repaired.astype(np.float64),
        distortion=float(distortion),
        negative_mass=float(negative_mass),
        negative_mass_ratio=float(negative_mass_ratio),
        condition_number=float(condition_number),
        off_diagonal_sign_flip_rate=float(sign_flip_rate),
        spurious_extreme_correlation_rate=float(spurious_extreme_rate),
        min_eigenvalue_pre=float(min_eigenvalue_pre),
        min_eigenvalue_post=float(min_eigenvalue_post),
    )


def enforce_psd(covariance: np.ndarray, config: DependenceConfig) -> tuple[np.ndarray, float]:
    diagnostics = repair_psd_with_diagnostics(covariance, config)
    if diagnostics.negative_mass_ratio > float(config.negative_eigen_mass_ratio_max):
        raise Module6ValidationError(
            "PSD_NEGATIVE_EIGEN_MASS_EXCESSIVE"
            f": negative_mass_ratio={diagnostics.negative_mass_ratio:.6f}"
        )
    if diagnostics.distortion > float(config.psd_max_distortion):
        raise Module6ValidationError(
            "PSD_DISTORTION_EXCESSIVE"
            f": delta_psd={diagnostics.distortion:.6f}"
        )
    if diagnostics.condition_number > float(config.condition_number_max):
        raise Module6ValidationError(
            "PSD_CONDITION_NUMBER_EXCESSIVE"
            f": condition_number={diagnostics.condition_number:.6f}"
        )
    if diagnostics.off_diagonal_sign_flip_rate > float(config.sign_flip_rate_max):
        raise Module6ValidationError(
            "PSD_SIGN_FLIP_EXCESSIVE"
            f": sign_flip_rate={diagnostics.off_diagonal_sign_flip_rate:.6f}"
        )
    if diagnostics.spurious_extreme_correlation_rate > float(config.spurious_extreme_rate_max):
        raise Module6ValidationError(
            "PSD_SPURIOUS_EXTREME_CORRELATION_EXCESSIVE"
            f": rate={diagnostics.spurious_extreme_correlation_rate:.6f}"
        )
    return diagnostics.repaired.astype(np.float64), float(diagnostics.negative_mass)
