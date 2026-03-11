from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from weightiz.module6.calendar import common_support_mask
from weightiz.module6.config import DependenceConfig
from weightiz.module6.psd import enforce_psd, ledoit_wolf_diag_shrink
from weightiz.module6.utils import Module6ValidationError, safe_divide


@dataclass(frozen=True)
class CovarianceBundle:
    covariance: np.ndarray
    correlation: np.ndarray
    downside_covariance: np.ndarray
    regime_overlap: np.ndarray
    drawdown_concurrence: sparse.csr_matrix
    common_support: np.ndarray
    shrinkage: float
    negative_mass: float


def build_covariance_bundle(
    returns_exec: np.ndarray,
    availability: np.ndarray,
    regime_exposure: np.ndarray,
    column_indices: np.ndarray,
    config: DependenceConfig,
) -> CovarianceBundle:
    idx = np.asarray(column_indices, dtype=np.int64).reshape(-1)
    support = common_support_mask(availability, idx)
    if int(np.sum(support)) <= 1:
        raise Module6ValidationError("insufficient common support for covariance bundle")
    x = np.asarray(returns_exec[np.asarray(support, dtype=bool)][:, idx], dtype=np.float64)
    if not np.isfinite(x).all():
        raise Module6ValidationError("non-finite returns detected on covariance support")
    cov, shrink = ledoit_wolf_diag_shrink(x)
    cov, neg_mass = enforce_psd(cov, config)
    diag = np.sqrt(np.maximum(np.diag(cov), 1.0e-18))
    corr = safe_divide(cov, np.outer(diag, diag))
    x_down = np.minimum(x - np.mean(x, axis=0, keepdims=True), 0.0)
    downside = (x_down.T @ x_down) / float(max(x_down.shape[0] - 1, 1))
    downside, _ = enforce_psd(downside, config)
    regime = np.asarray(regime_exposure[idx], dtype=np.float64)
    regime_norm = np.linalg.norm(regime, axis=1, keepdims=True)
    regime_overlap = safe_divide(regime @ regime.T, regime_norm @ regime_norm.T)
    drawdown_concurrence = build_drawdown_concurrence(x, config)
    return CovarianceBundle(
        covariance=cov,
        correlation=np.asarray(corr, dtype=np.float64),
        downside_covariance=downside,
        regime_overlap=np.asarray(regime_overlap, dtype=np.float64),
        drawdown_concurrence=drawdown_concurrence,
        common_support=np.asarray(support, dtype=bool),
        shrinkage=float(shrink),
        negative_mass=float(neg_mass),
    )


def build_drawdown_concurrence(returns_common: np.ndarray, config: DependenceConfig) -> sparse.csr_matrix:
    x = np.asarray(returns_common, dtype=np.float64)
    equity = np.cumprod(1.0 + x, axis=0)
    roll_max = np.maximum.accumulate(equity, axis=0)
    drawdown = safe_divide(equity - roll_max, roll_max)
    threshold = float(np.quantile(drawdown.reshape(-1), config.drawdown_tail_threshold))
    tail = drawdown <= threshold
    n = x.shape[1]
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            inter = int(np.sum(tail[:, i] & tail[:, j]))
            union = int(np.sum(tail[:, i] | tail[:, j]))
            if union <= 0 or inter <= 0:
                continue
            score = float(inter / union)
            rows.extend([i, j])
            cols.extend([j, i])
            vals.extend([score, score])
    return sparse.csr_matrix((np.asarray(vals, dtype=np.float64), (rows, cols)), shape=(n, n))

