from __future__ import annotations

import numpy as np
import pandas as pd

from module6.config import Module6Config
from module6.types import ReducedUniverseSpec
from module6.utils import normalize_long_only_weights, portfolio_pk, target_weights_hash


def generate_mv_variants(
    *,
    reduced_universe: ReducedUniverseSpec,
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
    normalized, cash_weight = normalize_long_only_weights(
        {pk: float(w) for pk, w in zip(reduced_universe.strategy_instance_pks, raw.tolist())},
        config.generator.minimum_cash_weight,
    )
    weights_hash = target_weights_hash(normalized)
    pk = portfolio_pk(
        reduced_universe_id=reduced_universe.reduced_universe_id,
        generator_family="mv_shrinkage",
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
    return (
        pd.DataFrame(
            [
                {
                    "portfolio_pk": pk,
                    "reduced_universe_id": reduced_universe.reduced_universe_id,
                    "generator_family": "mv_shrinkage",
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
                    "seed": int(config.generator.random_seed) + 303,
                    "batch_id": "mv_0000",
                }
            ]
        ),
        pd.DataFrame(
            [
                {
                    "portfolio_pk": pk,
                    "strategy_instance_pk": str(strategy_instance_pk),
                    "target_weight": float(weight),
                    "cash_weight": float(cash_weight),
                    "reduced_universe_id": reduced_universe.reduced_universe_id,
                }
                for strategy_instance_pk, weight in sorted(normalized.items())
            ]
        ),
    )

