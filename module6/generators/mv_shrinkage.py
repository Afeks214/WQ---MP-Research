from __future__ import annotations

import numpy as np
import pandas as pd

from module6.config import Module6Config
from module6.constraints import apply_long_only_weight_caps
from module6.types import ReducedUniverseSpec
from module6.utils import portfolio_pk, target_weights_hash


def generate_mv_variants(
    *,
    reduced_universe: ReducedUniverseSpec,
    strategy_frame: pd.DataFrame,
    covariance_bundle,
    returns_exec: np.ndarray,
    column_indices: np.ndarray,
    config: Module6Config,
    calendar_version: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not bool(config.generator.enable_mv_diagnostic):
        return pd.DataFrame(columns=["portfolio_pk"]), pd.DataFrame(columns=["portfolio_pk", "strategy_instance_pk", "target_weight"])
    if len(reduced_universe.strategy_instance_pks) > int(config.reduction.mv_universe_cap):
        return pd.DataFrame(columns=["portfolio_pk"]), pd.DataFrame(columns=["portfolio_pk", "strategy_instance_pk", "target_weight"])
    support_count = int(np.sum(covariance_bundle.common_support))
    if support_count < 3 * len(reduced_universe.strategy_instance_pks):
        return pd.DataFrame(columns=["portfolio_pk"]), pd.DataFrame(columns=["portfolio_pk", "strategy_instance_pk", "target_weight"])
    cov = np.asarray(covariance_bundle.covariance, dtype=np.float64)
    mu = np.mean(np.asarray(returns_exec, dtype=np.float64)[:, np.asarray(column_indices, dtype=np.int64)], axis=0)
    inv = np.linalg.pinv(cov)
    raw = inv @ mu
    raw = np.maximum(raw, 0.0)
    if np.sum(raw) <= 0.0:
        raw = np.ones_like(raw, dtype=np.float64)
    cluster_map = dict(strategy_frame[["strategy_instance_pk", "cluster_id"]].itertuples(index=False, name=None))
    family_map = dict(strategy_frame[["strategy_instance_pk", "family_id"]].itertuples(index=False, name=None))
    rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    blend_levels = tuple(float(x) for x in np.linspace(0.0, 0.35, max(int(config.generator.mv_variant_quota), 1)).tolist())
    equal = np.full(len(reduced_universe.strategy_instance_pks), 1.0 / max(len(reduced_universe.strategy_instance_pks), 1), dtype=np.float64)
    idx = 0
    for blend in blend_levels:
        blended = (1.0 - float(blend)) * raw + float(blend) * equal
        projection = apply_long_only_weight_caps(
            target_weights=blended,
            cluster_ids=np.asarray([int(cluster_map.get(pk, -1)) for pk in reduced_universe.strategy_instance_pks], dtype=np.int64),
            family_ids=np.asarray([str(family_map.get(pk, "")) for pk in reduced_universe.strategy_instance_pks], dtype=object),
            per_sleeve_cap=float(config.generator.per_sleeve_cap),
            per_cluster_cap=float(config.generator.per_cluster_cap),
            per_family_cap=float(config.generator.per_family_cap),
            min_cash_weight=float(config.generator.minimum_cash_weight),
        )
        if projection.infeasible:
            continue
        normalized = {
            str(pk): float(w)
            for pk, w in zip(reduced_universe.strategy_instance_pks, projection.weights.tolist())
            if float(w) > 0.0
        }
        cash_weight = float(projection.cash_weight)
        weights_hash = target_weights_hash(normalized)
        pk = portfolio_pk(
            reduced_universe_id=reduced_universe.reduced_universe_id,
            generator_family="mv_shrinkage",
            rebalance_policy="weekly_monday_close",
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
                "generator_family": "mv_shrinkage",
                "rebalance_policy": "weekly_monday_close",
                "target_weights_hash": weights_hash,
                "cash_policy": f"explicit_cash_residual_blend_{int(round(blend * 100)):02d}",
                "constraint_policy_version": config.simulator.constraint_policy_version,
                "ranking_policy_version": config.scoring.ranking_policy_version,
                "overnight_policy_version": config.simulator.overnight_policy_version,
                "friction_policy_version": config.simulator.friction_policy_version,
                "support_policy_version": config.simulator.support_policy_version,
                "calendar_version": calendar_version,
                "cash_weight": float(cash_weight),
                "seed": int(config.generator.random_seed) + 303,
                "batch_id": f"mv_{idx:04d}",
            }
        )
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
        idx += 1
        if idx >= int(config.generator.mv_variant_quota):
            break
    return pd.DataFrame(rows), pd.DataFrame(weight_rows)
