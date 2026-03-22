from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from weightiz.module6.config import DependenceConfig
from weightiz.module6.psd import covariance_to_correlation, repair_psd_with_diagnostics
from weightiz.module6.utils import Module6ValidationError


@dataclass(frozen=True)
class DependenceColumnSelection:
    keep_mask: np.ndarray
    observed_counts: np.ndarray
    support_minimum: int


@dataclass(frozen=True)
class CovarianceBundle:
    covariance: np.ndarray
    covariance_pre_psd: np.ndarray
    correlation: np.ndarray
    downside_covariance: np.ndarray
    regime_overlap: np.ndarray
    drawdown_concurrence: sparse.csr_matrix
    asset_column_indices: np.ndarray
    pair_overlap_counts: np.ndarray
    pair_reliability: np.ndarray
    pair_completion_reason_codes: np.ndarray
    completion_mask: np.ndarray
    asset_observed_counts: np.ndarray
    asset_support_minimum: int
    pair_support_minimum: int
    pair_support_full: int
    completion_prior_used: bool
    completion_reason_codes: tuple[str, ...]
    shrinkage: float
    negative_mass: float
    negative_eigen_mass_ratio: float
    condition_number: float
    psd_projection_distortion: float
    min_eigenvalue_pre: float
    min_eigenvalue_post: float
    off_diagonal_sign_flip_rate: float
    spurious_extreme_correlation_rate: float
    regime_mismatch_rate: float
    effective_pair_count: int
    effective_pair_reliability: float
    prior_only_pair_count: int
    zero_overlap_pair_count: int
    submin_overlap_pair_count: int
    repair_status: str


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    return 0.5 * (arr + arr.T)


def _support_floor(
    *,
    total_sessions: int,
    frac: float,
    min_sessions: int,
    max_sessions: int,
) -> int:
    dynamic_floor = int(np.ceil(float(frac) * float(total_sessions)))
    return int(max(int(min_sessions), min(int(max_sessions), dynamic_floor)))


def resolve_dependence_column_selection(
    *,
    availability: np.ndarray,
    column_indices: np.ndarray,
    config: DependenceConfig,
    min_observed_sessions_override: int | None = None,
) -> DependenceColumnSelection:
    idx = np.asarray(column_indices, dtype=np.int64).reshape(-1)
    mask = np.asarray(availability[:, idx], dtype=bool)
    if mask.ndim != 2:
        raise Module6ValidationError("availability matrix must be 2D for dependence selection")
    observed_counts = np.sum(mask, axis=0, dtype=np.int64)
    support_minimum = _support_floor(
        total_sessions=int(mask.shape[0]),
        frac=float(config.asset_support_frac),
        min_sessions=int(config.asset_support_min_sessions),
        max_sessions=int(config.asset_support_max_sessions),
    )
    if min_observed_sessions_override is not None:
        support_minimum = int(max(support_minimum, int(min_observed_sessions_override)))
    keep_mask = observed_counts >= int(support_minimum)
    return DependenceColumnSelection(
        keep_mask=np.asarray(keep_mask, dtype=bool),
        observed_counts=np.asarray(observed_counts, dtype=np.int64),
        support_minimum=int(support_minimum),
    )


def _pairwise_overlap_counts(mask: np.ndarray) -> np.ndarray:
    mask_i = np.asarray(mask, dtype=np.int64)
    return np.asarray(mask_i.T @ mask_i, dtype=np.int64)


def _pairwise_covariance(values: np.ndarray, mask: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=np.float64)
    m = np.asarray(mask, dtype=np.float64)
    xm = x * m
    counts = m.T @ m
    sum_x = xm.T @ m
    sumsq_x = (xm * xm).T @ m
    cross = xm.T @ xm

    count_safe = np.maximum(counts, 1.0)
    denom = np.maximum(counts - 1.0, 1.0)
    mean_i = sum_x / count_safe
    mean_j = mean_i.T
    cov = (cross - counts * mean_i * mean_j) / denom
    cov = np.where(counts > 1.0, cov, 0.0)
    cov = _symmetrize(cov)

    var_i = np.maximum((sumsq_x - counts * mean_i * mean_i) / denom, 0.0)
    corr = cov / np.maximum(np.sqrt(var_i * var_i.T), float(eps))
    corr = np.where(counts > 1.0, corr, 0.0)
    corr = np.clip(_symmetrize(corr), -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return np.asarray(counts, dtype=np.int64), cov.astype(np.float64), corr.astype(np.float64)


def _prepare_winsorized_returns(
    *,
    returns_exec: np.ndarray,
    availability: np.ndarray,
    config: DependenceConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    returns = np.asarray(returns_exec, dtype=np.float64)
    mask = np.asarray(availability, dtype=bool)
    if returns.ndim != 2 or mask.ndim != 2 or returns.shape != mask.shape:
        raise Module6ValidationError("returns and availability must be aligned 2D matrices for dependence")
    if not np.isfinite(returns).all():
        raise Module6ValidationError("non-finite returns detected during dependence preparation")

    mask_f = mask.astype(np.float64)
    observed_counts = np.sum(mask, axis=0, dtype=np.int64)
    sum_r = np.sum(returns * mask_f, axis=0, dtype=np.float64)
    mean_r = sum_r / np.maximum(observed_counts.astype(np.float64), 1.0)
    centered = (returns - mean_r[None, :]) * mask_f
    denom = np.maximum(observed_counts.astype(np.float64) - 1.0, 1.0)
    base_var = np.sum(centered * centered, axis=0, dtype=np.float64) / denom
    base_var = np.maximum(base_var, float(config.variance_floor))
    sigma = np.sqrt(base_var)
    standardized = centered / np.maximum(sigma[None, :], float(config.variance_floor))
    standardized = np.clip(standardized, -float(config.return_vol_clip), float(config.return_vol_clip)) * mask_f
    winsorized = standardized * sigma[None, :]
    winsorized_var = np.sum(winsorized * winsorized, axis=0, dtype=np.float64) / denom
    winsorized_var = np.maximum(winsorized_var, float(config.variance_floor))
    return mask, observed_counts, winsorized.astype(np.float64), standardized.astype(np.float64), winsorized_var.astype(np.float64)


def _regime_prior(
    *,
    regime_exposure: np.ndarray,
    column_indices: np.ndarray,
    variances: np.ndarray,
    config: DependenceConfig,
) -> tuple[np.ndarray, np.ndarray]:
    regime = np.asarray(regime_exposure[np.asarray(column_indices, dtype=np.int64)], dtype=np.float64)
    regime = np.nan_to_num(regime, nan=0.0, posinf=0.0, neginf=0.0)
    row_norm = np.linalg.norm(regime, axis=1, keepdims=True)
    regime_rows = regime / np.maximum(row_norm, float(config.regime_row_norm_eps))
    regime_overlap = np.clip(_symmetrize(regime_rows @ regime_rows.T), -1.0, 1.0)
    np.fill_diagonal(regime_overlap, 1.0)
    corr_prior = (
        (1.0 - float(config.completion_alpha_regime)) * np.eye(regime_overlap.shape[0], dtype=np.float64)
        + float(config.completion_alpha_regime) * regime_overlap
    )
    scale = np.outer(np.sqrt(np.maximum(variances, float(config.variance_floor))), np.sqrt(np.maximum(variances, float(config.variance_floor))))
    prior = _symmetrize(corr_prior * scale)
    np.fill_diagonal(prior, np.maximum(variances, float(config.variance_floor)))
    return prior.astype(np.float64), regime_overlap.astype(np.float64)


def _pair_reliability(
    *,
    pair_counts: np.ndarray,
    regime_overlap: np.ndarray,
    pair_support_minimum: int,
    pair_support_full: int,
    config: DependenceConfig,
) -> tuple[np.ndarray, float, int, int, int]:
    counts = np.asarray(pair_counts, dtype=np.float64)
    denom = max(float(pair_support_full - pair_support_minimum), 1.0)
    reliability = np.where(
        counts < float(pair_support_minimum),
        0.0,
        np.minimum(1.0, (counts - float(pair_support_minimum)) / denom),
    )
    reliability = np.where(
        (np.asarray(regime_overlap, dtype=np.float64) < 0.0) & (counts < float(pair_support_full)),
        0.0,
        reliability,
    )
    reliability = _symmetrize(reliability)
    np.fill_diagonal(reliability, 1.0)

    upper = np.triu_indices(reliability.shape[0], k=1)
    observed_pairs = counts[upper] > 0.0
    mismatch_pairs = observed_pairs & (np.asarray(regime_overlap, dtype=np.float64)[upper] < 0.0)
    mismatch_rate = float(mismatch_pairs.sum() / max(observed_pairs.sum(), 1))
    zero_overlap_pair_count = int(np.sum(counts[upper] == 0.0))
    submin_overlap_pair_count = int(np.sum((counts[upper] > 0.0) & (counts[upper] < float(pair_support_minimum))))
    prior_only_pair_count = int(np.sum(reliability[upper] <= 0.0))
    return reliability.astype(np.float64), float(mismatch_rate), prior_only_pair_count, zero_overlap_pair_count, submin_overlap_pair_count


def _pair_completion_reason_codes(
    *,
    pair_counts: np.ndarray,
    pair_support_minimum: int,
) -> np.ndarray:
    counts = np.asarray(pair_counts, dtype=np.int64)
    codes = np.full(counts.shape, "", dtype="<U24")
    if counts.shape[0] <= 1:
        return codes
    off_diag = ~np.eye(counts.shape[0], dtype=bool)
    zero_mask = off_diag & (counts == 0)
    submin_mask = off_diag & (counts > 0) & (counts < int(pair_support_minimum))
    codes[zero_mask] = "PAIR_OVERLAP_ZERO"
    codes[submin_mask] = "PAIR_OVERLAP_SUBMIN"
    return codes


def _shrink_complete_covariance(
    *,
    complete_covariance: np.ndarray,
    pair_reliability: np.ndarray,
    regime_mismatch_rate: float,
    config: DependenceConfig,
) -> tuple[np.ndarray, float, float]:
    n_count = int(complete_covariance.shape[0])
    if n_count <= 1:
        effective_pair_reliability = 1.0
    else:
        upper = np.triu_indices(n_count, k=1)
        rel_off = np.asarray(pair_reliability[upper], dtype=np.float64)
        effective_pair_reliability = float(np.mean(rel_off)) if rel_off.size > 0 else 1.0
    reliability_deficit = 1.0 - float(effective_pair_reliability)
    mismatch_pressure = np.clip(
        (
            float(regime_mismatch_rate) - float(config.regime_mismatch_warn)
        )
        / max(float(config.regime_mismatch_reject) - float(config.regime_mismatch_warn), float(config.eps)),
        0.0,
        1.0,
    )
    lam = float(
        np.clip(
            float(config.shrinkage_base)
            + float(config.shrinkage_reliability_mult) * float(reliability_deficit)
            + float(config.shrinkage_regime_mult) * float(mismatch_pressure),
            float(config.shrinkage_min),
            float(config.shrinkage_max),
        )
    )
    target = np.diag(np.diag(np.asarray(complete_covariance, dtype=np.float64)))
    shrunk = _symmetrize((1.0 - lam) * np.asarray(complete_covariance, dtype=np.float64) + lam * target)
    return shrunk.astype(np.float64), float(lam), float(effective_pair_reliability)


def _repair_covariance(
    *,
    covariance: np.ndarray,
    config: DependenceConfig,
) -> tuple[np.ndarray, float, float, float, float, float, float, float, float, str]:
    diagnostics = repair_psd_with_diagnostics(covariance, config)
    if diagnostics.distortion > float(config.psd_max_distortion):
        raise Module6ValidationError(
            "PSD_DISTORTION_EXCESSIVE"
            f": delta_psd={diagnostics.distortion:.6f}"
        )
    if diagnostics.negative_mass_ratio > float(config.negative_eigen_mass_ratio_max):
        raise Module6ValidationError(
            "PSD_NEGATIVE_EIGEN_MASS_EXCESSIVE"
            f": negative_mass_ratio={diagnostics.negative_mass_ratio:.6f}"
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
    repair_status = "warn" if diagnostics.distortion > float(config.psd_warn_distortion) else "clean"
    return (
        diagnostics.repaired.astype(np.float64),
        float(diagnostics.distortion),
        float(diagnostics.negative_mass),
        float(diagnostics.negative_mass_ratio),
        float(diagnostics.condition_number),
        float(diagnostics.min_eigenvalue_pre),
        float(diagnostics.min_eigenvalue_post),
        float(diagnostics.off_diagonal_sign_flip_rate),
        float(diagnostics.spurious_extreme_correlation_rate),
        str(repair_status),
    )


def build_covariance_bundle(
    returns_exec: np.ndarray,
    availability: np.ndarray,
    regime_exposure: np.ndarray,
    column_indices: np.ndarray,
    config: DependenceConfig,
    asset_support_minimum: int | None = None,
) -> CovarianceBundle:
    idx = np.asarray(column_indices, dtype=np.int64).reshape(-1)
    if idx.size < 2:
        raise Module6ValidationError("DEPENDENCE_UNIVERSE_TOO_SMALL")
    availability_local = np.asarray(availability[:, idx], dtype=bool)
    asset_support_floor = int(
        asset_support_minimum
        if asset_support_minimum is not None
        else _support_floor(
            total_sessions=int(availability_local.shape[0]),
            frac=float(config.asset_support_frac),
            min_sessions=int(config.asset_support_min_sessions),
            max_sessions=int(config.asset_support_max_sessions),
        )
    )
    asset_observed_counts = np.sum(availability_local, axis=0, dtype=np.int64)
    keep_mask = np.asarray(asset_observed_counts >= int(asset_support_floor), dtype=bool)
    if not bool(np.any(keep_mask)):
        raise Module6ValidationError("DEPENDENCE_ASSET_SUPPORT_TOO_SHORT")
    idx = idx[keep_mask]
    if idx.size < 2:
        raise Module6ValidationError("DEPENDENCE_UNIVERSE_TOO_SMALL")

    mask, observed_counts, winsorized_returns, standardized_returns, variances = _prepare_winsorized_returns(
        returns_exec=np.asarray(returns_exec[:, idx], dtype=np.float64),
        availability=np.asarray(availability[:, idx], dtype=bool),
        config=config,
    )
    if bool(np.any(observed_counts < int(asset_support_floor))):
        raise Module6ValidationError("DEPENDENCE_ASSET_SUPPORT_TOO_SHORT")
    if int(np.sum(observed_counts > 0)) < 2:
        raise Module6ValidationError("DEPENDENCE_ASSET_SUPPORT_TOO_SHORT")

    pair_counts = _pairwise_overlap_counts(mask)
    pair_support_minimum = _support_floor(
        total_sessions=int(mask.shape[0]),
        frac=float(config.pair_support_frac),
        min_sessions=int(config.pair_support_min_sessions),
        max_sessions=int(config.pair_support_max_sessions),
    )
    pair_support_full = int(
        min(
            int(config.pair_support_full_max_sessions),
            max(
                int(config.pair_support_full_min_sessions),
                int(np.ceil(float(config.pair_support_full_multiplier) * float(pair_support_minimum))),
            ),
        )
    )

    full_prior, regime_overlap = _regime_prior(
        regime_exposure=np.asarray(regime_exposure, dtype=np.float64),
        column_indices=idx,
        variances=variances,
        config=config,
    )
    pair_reliability, regime_mismatch_rate, prior_only_pair_count, zero_overlap_pair_count, submin_overlap_pair_count = _pair_reliability(
        pair_counts=pair_counts,
        regime_overlap=regime_overlap,
        pair_support_minimum=pair_support_minimum,
        pair_support_full=pair_support_full,
        config=config,
    )
    pair_completion_reason_codes = _pair_completion_reason_codes(
        pair_counts=pair_counts,
        pair_support_minimum=pair_support_minimum,
    )
    if idx.size >= 3 and prior_only_pair_count >= int((idx.size * (idx.size - 1)) // 2):
        raise Module6ValidationError("PAIRWISE_STRUCTURE_UNRELIABLE")

    _, _, raw_corr = _pairwise_covariance(
        standardized_returns,
        mask,
        eps=float(config.eps),
    )
    raw_covariance = _symmetrize(raw_corr * np.outer(np.sqrt(variances), np.sqrt(variances)))
    np.fill_diagonal(raw_covariance, np.maximum(variances, float(config.variance_floor)))
    complete_covariance = _symmetrize(pair_reliability * raw_covariance + (1.0 - pair_reliability) * full_prior)
    np.fill_diagonal(complete_covariance, np.maximum(variances, float(config.variance_floor)))
    if not np.isfinite(complete_covariance).all():
        raise Module6ValidationError("DEPENDENCE_COMPLETION_NONFINITE")

    shrunk_covariance, shrinkage, effective_pair_reliability = _shrink_complete_covariance(
        complete_covariance=complete_covariance,
        pair_reliability=pair_reliability,
        regime_mismatch_rate=regime_mismatch_rate,
        config=config,
    )
    repaired_covariance, distortion, negative_mass, negative_mass_ratio, condition_number, min_eigenvalue_pre, min_eigenvalue_post, sign_flip_rate, spurious_extreme_rate, repair_status = _repair_covariance(
        covariance=shrunk_covariance,
        config=config,
    )
    if repaired_covariance.shape[0] < 2:
        raise Module6ValidationError("DEPENDENCE_UNIVERSE_TOO_SMALL")
    if np.any(np.diag(repaired_covariance) <= 0.0):
        raise Module6ValidationError("DEPENDENCE_NONPOSITIVE_DIAGONAL")
    if regime_mismatch_rate > float(config.regime_mismatch_reject):
        raise Module6ValidationError(
            "REGIME_MISMATCH_EXCESSIVE"
            f": regime_mismatch_rate={regime_mismatch_rate:.6f}"
        )

    downside_returns = np.minimum(winsorized_returns, 0.0)
    _, downside_raw, _ = _pairwise_covariance(downside_returns, mask, eps=float(config.eps))
    downside_variances = np.maximum(np.diag(downside_raw), float(config.variance_floor))
    downside_prior = _symmetrize(
        ((1.0 - float(config.completion_alpha_regime)) * np.eye(idx.size, dtype=np.float64) + float(config.completion_alpha_regime) * regime_overlap)
        * np.outer(np.sqrt(downside_variances), np.sqrt(downside_variances))
    )
    np.fill_diagonal(downside_prior, downside_variances)
    downside_complete = _symmetrize(pair_reliability * downside_raw + (1.0 - pair_reliability) * downside_prior)
    np.fill_diagonal(downside_complete, downside_variances)
    downside_shrunk, _, _ = _shrink_complete_covariance(
        complete_covariance=downside_complete,
        pair_reliability=pair_reliability,
        regime_mismatch_rate=regime_mismatch_rate,
        config=config,
    )
    downside_covariance, _, _, _, _, _, _, _, _, _ = _repair_covariance(
        covariance=downside_shrunk,
        config=config,
    )

    correlation = covariance_to_correlation(repaired_covariance, diag_eps=float(config.corr_diag_eps))
    upper = np.triu_indices(idx.size, k=1)
    effective_pair_count = int(np.sum(pair_reliability[upper] > 0.0))
    completion_mask = np.asarray(pair_reliability < 1.0, dtype=bool)
    completion_prior_used = bool(np.any(pair_reliability[upper] < 1.0)) if upper[0].size > 0 else False
    pair_reason_summary = tuple(
        sorted(
            {
                str(code)
                for code in np.unique(pair_completion_reason_codes)
                if str(code)
            }
        )
    )
    drawdown_concurrence = build_drawdown_concurrence(winsorized_returns, config)
    return CovarianceBundle(
        covariance=repaired_covariance.astype(np.float64),
        covariance_pre_psd=shrunk_covariance.astype(np.float64),
        correlation=correlation.astype(np.float64),
        downside_covariance=downside_covariance.astype(np.float64),
        regime_overlap=np.asarray(regime_overlap, dtype=np.float64),
        drawdown_concurrence=drawdown_concurrence,
        asset_column_indices=np.asarray(idx, dtype=np.int64),
        pair_overlap_counts=np.asarray(pair_counts, dtype=np.int64),
        pair_reliability=np.asarray(pair_reliability, dtype=np.float64),
        pair_completion_reason_codes=np.asarray(pair_completion_reason_codes, dtype="<U24"),
        completion_mask=np.asarray(completion_mask, dtype=bool),
        asset_observed_counts=np.asarray(observed_counts, dtype=np.int64),
        asset_support_minimum=int(asset_support_floor),
        pair_support_minimum=int(pair_support_minimum),
        pair_support_full=int(pair_support_full),
        completion_prior_used=bool(completion_prior_used),
        completion_reason_codes=pair_reason_summary,
        shrinkage=float(shrinkage),
        negative_mass=float(negative_mass),
        negative_eigen_mass_ratio=float(negative_mass_ratio),
        condition_number=float(condition_number),
        psd_projection_distortion=float(distortion),
        min_eigenvalue_pre=float(min_eigenvalue_pre),
        min_eigenvalue_post=float(min_eigenvalue_post),
        off_diagonal_sign_flip_rate=float(sign_flip_rate),
        spurious_extreme_correlation_rate=float(spurious_extreme_rate),
        regime_mismatch_rate=float(regime_mismatch_rate),
        effective_pair_count=int(effective_pair_count),
        effective_pair_reliability=float(effective_pair_reliability),
        prior_only_pair_count=int(prior_only_pair_count),
        zero_overlap_pair_count=int(zero_overlap_pair_count),
        submin_overlap_pair_count=int(submin_overlap_pair_count),
        repair_status=str(repair_status),
    )


def build_drawdown_concurrence(returns_common: np.ndarray, config: DependenceConfig) -> sparse.csr_matrix:
    x = np.asarray(returns_common, dtype=np.float64)
    equity = np.cumprod(1.0 + x, axis=0)
    roll_max = np.maximum.accumulate(equity, axis=0)
    drawdown = (equity - roll_max) / np.maximum(roll_max, float(config.eps))
    threshold = float(np.quantile(drawdown.reshape(-1), config.drawdown_tail_threshold))
    tail = (drawdown <= threshold).astype(np.int64)
    inter = tail.T @ tail
    tail_sum = np.sum(tail, axis=0, dtype=np.int64)
    union = tail_sum[:, None] + tail_sum[None, :] - inter
    concurrence = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    concurrence = _symmetrize(np.asarray(concurrence, dtype=np.float64))
    np.fill_diagonal(concurrence, 0.0)
    concurrence = np.where(inter > 0, concurrence, 0.0)
    return sparse.csr_matrix(concurrence)
