from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from module6.config import Module6Config
from module6.types import ReducedUniverseSpec
from module6.utils import normalize_long_only_weights, portfolio_pk, target_weights_hash


def _cluster_var(cov: np.ndarray, items: np.ndarray) -> float:
    sub = cov[np.ix_(items, items)]
    inv_diag = 1.0 / np.maximum(np.diag(sub), 1.0e-12)
    weights = inv_diag / np.sum(inv_diag)
    return float(weights @ sub @ weights)


def _hrp_weights(correlation: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    dist = np.sqrt(np.maximum(0.0, 0.5 * (1.0 - correlation)))
    np.fill_diagonal(dist, 0.0)
    order = leaves_list(linkage(squareform(dist, checks=False), method="single"))
    sorted_items = order.astype(np.int64)
    weights = pd.Series(1.0, index=sorted_items.tolist(), dtype=np.float64)
    clusters = [sorted_items]
    while clusters:
        cluster = clusters.pop(0)
        if cluster.size <= 1:
            continue
        split = cluster.size // 2
        left = cluster[:split]
        right = cluster[split:]
        var_left = _cluster_var(covariance, left)
        var_right = _cluster_var(covariance, right)
        alpha = 1.0 - var_left / max(var_left + var_right, 1.0e-12)
        weights[left.tolist()] *= alpha
        weights[right.tolist()] *= 1.0 - alpha
        clusters.extend([left, right])
    out = np.zeros(correlation.shape[0], dtype=np.float64)
    for idx, weight in weights.items():
        out[int(idx)] = float(weight)
    return out / np.maximum(np.sum(out), 1.0e-12)


def generate_hrp_variants(
    *,
    reduced_universe: ReducedUniverseSpec,
    strategy_frame: pd.DataFrame,
    covariance_bundle,
    config: Module6Config,
    calendar_version: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if covariance_bundle.common_support.sum() < 2 * len(reduced_universe.strategy_instance_pks):
        return pd.DataFrame(columns=["portfolio_pk"]), pd.DataFrame(columns=["portfolio_pk", "strategy_instance_pk", "target_weight"])
    strategy_ids = list(reduced_universe.strategy_instance_pks)
    weights = _hrp_weights(
        np.asarray(covariance_bundle.correlation, dtype=np.float64),
        np.asarray(covariance_bundle.covariance, dtype=np.float64),
    )
    rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    lookbacks = (63, 126, 252)
    policies = ("daily_close", "weekly_monday_close", "band_10pct")
    idx = 0
    for lookback in lookbacks:
        for policy in policies:
            normalized, cash_weight = normalize_long_only_weights(
                {pk: float(w) for pk, w in zip(strategy_ids, weights.tolist())},
                config.generator.minimum_cash_weight,
            )
            weights_hash = target_weights_hash(normalized)
            pk = portfolio_pk(
                reduced_universe_id=reduced_universe.reduced_universe_id,
                generator_family="hrp_risk",
                rebalance_policy=policy,
                target_weights_hash_value=weights_hash,
                cash_policy=f"explicit_cash_residual_lb{lookback}",
                constraint_policy_version=config.simulator.constraint_policy_version,
                ranking_policy_version=config.scoring.ranking_policy_version,
                overnight_policy_version=config.simulator.overnight_policy_version,
                friction_policy_version=config.simulator.friction_policy_version,
                support_policy_version=config.simulator.support_policy_version,
                calendar_version=calendar_version,
            )
            rows.append(
                {
                    "portfolio_pk": pk,
                    "reduced_universe_id": reduced_universe.reduced_universe_id,
                    "generator_family": "hrp_risk",
                    "rebalance_policy": policy,
                    "target_weights_hash": weights_hash,
                    "cash_policy": f"explicit_cash_residual_lb{lookback}",
                    "constraint_policy_version": config.simulator.constraint_policy_version,
                    "ranking_policy_version": config.scoring.ranking_policy_version,
                    "overnight_policy_version": config.simulator.overnight_policy_version,
                    "friction_policy_version": config.simulator.friction_policy_version,
                    "support_policy_version": config.simulator.support_policy_version,
                    "calendar_version": calendar_version,
                    "cash_weight": float(cash_weight),
                    "seed": int(config.generator.random_seed) + 202,
                    "batch_id": f"hrp_{idx:04d}",
                }
            )
            idx += 1
            for strategy_instance_pk, weight in sorted(normalized.items()):
                weight_rows.append(
                    {
                        "portfolio_pk": pk,
                        "strategy_instance_pk": str(strategy_instance_pk),
                        "target_weight": float(weight),
                        "cash_weight": float(cash_weight),
                        "reduced_universe_id": reduced_universe.reduced_universe_id,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(weight_rows)

