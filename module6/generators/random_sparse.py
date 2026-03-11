from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from module6.config import Module6Config
from module6.types import ReducedUniverseSpec
from module6.utils import normalize_long_only_weights, portfolio_pk, target_weights_hash


def generate_random_sparse_batch(
    *,
    reduced_universe: ReducedUniverseSpec,
    strategy_frame: pd.DataFrame,
    config: Module6Config,
    calendar_version: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    weights_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(config.generator.random_seed))
    ids = [pk for pk in reduced_universe.strategy_instance_pks if pk in set(strategy_frame["strategy_instance_pk"].astype(str).tolist())]
    score_map = dict(strategy_frame[["strategy_instance_pk", "robustness_score"]].itertuples(index=False, name=None))
    cluster_map = dict(strategy_frame[["strategy_instance_pk", "cluster_id"]].itertuples(index=False, name=None))
    family_map = dict(strategy_frame[["strategy_instance_pk", "family_id"]].itertuples(index=False, name=None))
    probs = np.asarray([max(float(score_map.get(pk, 0.0)), 0.0) + 1.0 for pk in ids], dtype=np.float64)
    probs = probs / np.sum(probs)
    for idx in range(int(config.generator.random_sparse_quota)):
        card = int(rng.choice(np.asarray(config.generator.active_cardinality_choices, dtype=np.int64)))
        card = min(card, len(ids))
        chosen = rng.choice(np.asarray(ids, dtype=object), size=card, replace=False, p=probs)
        raw = rng.dirichlet(np.ones(card, dtype=np.float64))
        candidate_weights = {str(pk): float(w) for pk, w in zip(chosen.tolist(), raw.tolist())}
        cluster_totals: dict[int, float] = {}
        family_totals: dict[str, float] = {}
        for pk, w in candidate_weights.items():
            cluster_totals[int(cluster_map.get(pk, -1))] = cluster_totals.get(int(cluster_map.get(pk, -1)), 0.0) + float(w)
            family_totals[str(family_map.get(pk, ""))] = family_totals.get(str(family_map.get(pk, "")), 0.0) + float(w)
        for pk in list(candidate_weights.keys()):
            candidate_weights[pk] = min(float(candidate_weights[pk]), float(config.generator.per_sleeve_cap))
        for cluster_id, total in cluster_totals.items():
            if total > float(config.generator.per_cluster_cap):
                scale = float(config.generator.per_cluster_cap) / total
                for pk in list(candidate_weights.keys()):
                    if int(cluster_map.get(pk, -1)) == cluster_id:
                        candidate_weights[pk] *= scale
        for family_id, total in family_totals.items():
            if total > float(config.generator.per_family_cap):
                scale = float(config.generator.per_family_cap) / total
                for pk in list(candidate_weights.keys()):
                    if str(family_map.get(pk, "")) == family_id:
                        candidate_weights[pk] *= scale
        normalized, cash_weight = normalize_long_only_weights(candidate_weights, config.generator.minimum_cash_weight)
        weights_hash = target_weights_hash(normalized)
        pk = portfolio_pk(
            reduced_universe_id=reduced_universe.reduced_universe_id,
            generator_family="random_sparse",
            rebalance_policy="band_10pct",
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
                "generator_family": "random_sparse",
                "rebalance_policy": "band_10pct",
                "target_weights_hash": weights_hash,
                "cash_policy": "explicit_cash_residual",
                "constraint_policy_version": config.simulator.constraint_policy_version,
                "ranking_policy_version": config.scoring.ranking_policy_version,
                "overnight_policy_version": config.simulator.overnight_policy_version,
                "friction_policy_version": config.simulator.friction_policy_version,
                "support_policy_version": config.simulator.support_policy_version,
                "calendar_version": calendar_version,
                "cash_weight": float(cash_weight),
                "seed": int(config.generator.random_seed),
                "batch_id": f"random_sparse_{idx // max(config.generator.random_sparse_batch_size, 1):04d}",
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

