from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from module6.config import Module6Config
from module6.types import ReducedUniverseSpec
from module6.utils import normalize_long_only_weights, portfolio_pk, target_weights_hash


def generate_cluster_balanced_batch(
    *,
    reduced_universe: ReducedUniverseSpec,
    strategy_frame: pd.DataFrame,
    config: Module6Config,
    calendar_version: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    weights_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(config.generator.random_seed) + 101)
    grouped = {
        int(cluster_id): grp.sort_values(
            ["robustness_score", "availability_ratio", "strategy_instance_pk"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        for cluster_id, grp in strategy_frame.groupby("cluster_id", dropna=False, sort=True)
    }
    cluster_ids = sorted(grouped.keys())
    if len(cluster_ids) <= 1:
        return pd.DataFrame(columns=["portfolio_pk"]), pd.DataFrame(columns=["portfolio_pk", "strategy_instance_pk", "target_weight"])
    for idx in range(int(config.generator.cluster_balanced_quota)):
        active_clusters = cluster_ids.copy()
        rng.shuffle(active_clusters)
        active_clusters = active_clusters[: max(2, min(len(active_clusters), 6))]
        raw_cluster = rng.dirichlet(np.ones(len(active_clusters), dtype=np.float64))
        candidate_weights: dict[str, float] = {}
        for cluster_id, cluster_w in zip(active_clusters, raw_cluster.tolist()):
            grp = grouped[int(cluster_id)]
            pick_count = min(max(1, int(rng.integers(1, 3))), grp.shape[0])
            chosen = grp.head(pick_count)
            local_raw = np.asarray([1.0 / max(float(x), 1.0e-6) for x in chosen["avg_turnover_metrics"].fillna(0.0).astype(float).tolist()], dtype=np.float64)
            if np.sum(local_raw) <= 0.0:
                local_raw = np.ones(pick_count, dtype=np.float64)
            local_raw = local_raw / np.sum(local_raw)
            for pk, lw in zip(chosen["strategy_instance_pk"].astype(str).tolist(), local_raw.tolist()):
                candidate_weights[pk] = candidate_weights.get(pk, 0.0) + float(cluster_w) * float(lw)
        normalized, cash_weight = normalize_long_only_weights(candidate_weights, config.generator.minimum_cash_weight)
        weights_hash = target_weights_hash(normalized)
        pk = portfolio_pk(
            reduced_universe_id=reduced_universe.reduced_universe_id,
            generator_family="cluster_balanced",
            rebalance_policy="weekly_monday_close",
            target_weights_hash_value=weights_hash,
            cash_policy="explicit_cash_residual",
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
                "generator_family": "cluster_balanced",
                "rebalance_policy": "weekly_monday_close",
                "target_weights_hash": weights_hash,
                "cash_policy": "explicit_cash_residual",
                "constraint_policy_version": config.simulator.constraint_policy_version,
                "ranking_policy_version": config.scoring.ranking_policy_version,
                "overnight_policy_version": config.simulator.overnight_policy_version,
                "friction_policy_version": config.simulator.friction_policy_version,
                "support_policy_version": config.simulator.support_policy_version,
                "calendar_version": calendar_version,
                "cash_weight": float(cash_weight),
                "seed": int(config.generator.random_seed) + 101,
                "batch_id": f"cluster_balanced_{idx:04d}",
            }
        )
        for strategy_instance_pk, weight in sorted(normalized.items()):
            weights_rows.append(
                {
                    "portfolio_pk": pk,
                    "strategy_instance_pk": str(strategy_instance_pk),
                    "target_weight": float(weight),
                    "cash_weight": float(cash_weight),
                    "reduced_universe_id": reduced_universe.reduced_universe_id,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(weights_rows)

