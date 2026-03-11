from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from module6.config import Module6Config
from module6.constraints import apply_long_only_weight_caps
from module6.types import ReducedUniverseSpec
from module6.utils import portfolio_pk, target_weights_hash


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
        projection = apply_long_only_weight_caps(
            target_weights=np.asarray([candidate_weights[pk] for pk in chosen.tolist()], dtype=np.float64),
            cluster_ids=np.asarray([int(cluster_map.get(str(pk), -1)) for pk in chosen.tolist()], dtype=np.int64),
            family_ids=np.asarray([str(family_map.get(str(pk), "")) for pk in chosen.tolist()], dtype=object),
            per_sleeve_cap=float(config.generator.per_sleeve_cap),
            per_cluster_cap=float(config.generator.per_cluster_cap),
            per_family_cap=float(config.generator.per_family_cap),
            min_cash_weight=float(config.generator.minimum_cash_weight),
        )
        if projection.infeasible:
            continue
        normalized = {
            str(pk): float(w)
            for pk, w in zip(chosen.tolist(), projection.weights.tolist())
            if float(w) > 0.0
        }
        cash_weight = float(projection.cash_weight)
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
