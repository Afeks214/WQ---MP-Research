from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from weightiz.module6.config import Module6Config
from weightiz.module6.constraints import apply_long_only_weight_caps
from weightiz.module6.types import ReducedUniverseSpec
from weightiz.module6.utils import portfolio_pk, target_weights_hash


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
    base_weights = _hrp_weights(
        np.asarray(covariance_bundle.correlation, dtype=np.float64),
        np.asarray(covariance_bundle.covariance, dtype=np.float64),
    )
    cluster_map = dict(strategy_frame[["strategy_instance_pk", "cluster_id"]].itertuples(index=False, name=None))
    family_map = dict(strategy_frame[["strategy_instance_pk", "family_id"]].itertuples(index=False, name=None))
    rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    policies = ("daily_close", "weekly_monday_close", "band_10pct")
    blend_levels = tuple(float(x) for x in np.linspace(0.0, 0.55, 12).tolist())
    equal_weights = np.full(len(strategy_ids), 1.0 / max(len(strategy_ids), 1), dtype=np.float64)
    idx = 0
    quota = max(0, int(config.generator.hrp_variant_quota))
    for blend in blend_levels:
        for policy in policies:
            if idx >= quota:
                break
            raw_weights = (1.0 - float(blend)) * np.asarray(base_weights, dtype=np.float64) + float(blend) * equal_weights
            projection = apply_long_only_weight_caps(
                target_weights=raw_weights,
                cluster_ids=np.asarray([int(cluster_map.get(pk, -1)) for pk in strategy_ids], dtype=np.int64),
                family_ids=np.asarray([str(family_map.get(pk, "")) for pk in strategy_ids], dtype=object),
                per_sleeve_cap=float(config.generator.per_sleeve_cap),
                per_cluster_cap=float(config.generator.per_cluster_cap),
                per_family_cap=float(config.generator.per_family_cap),
                min_cash_weight=float(config.generator.minimum_cash_weight),
            )
            if projection.infeasible:
                continue
            normalized = {
                str(pk): float(w)
                for pk, w in zip(strategy_ids, projection.weights.tolist())
                if float(w) > 0.0
            }
            cash_weight = float(projection.cash_weight)
            weights_hash = target_weights_hash(normalized)
            pk = portfolio_pk(
                reduced_universe_id=reduced_universe.reduced_universe_id,
                generator_family="hrp_risk",
                rebalance_policy=policy,
                target_weights_hash_value=weights_hash,
                cash_policy=f"explicit_cash_residual_blend_{int(round(blend * 100)):02d}",
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
                    "cash_policy": f"explicit_cash_residual_blend_{int(round(blend * 100)):02d}",
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
        if idx >= quota:
            break
    return pd.DataFrame(rows), pd.DataFrame(weight_rows)
