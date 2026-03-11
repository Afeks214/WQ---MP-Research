from __future__ import annotations

import numpy as np
import pandas as pd

from weightiz.module6.config import GeneratorConfig, Module6Config
from weightiz.module6.dependence import build_covariance_bundle
from weightiz.module6.generators import generate_all_portfolios
from weightiz.module6.types import ReducedUniverseSpec


def test_generator_outputs_are_simplex_and_nonnegative():
    strategy_frame = pd.DataFrame(
        {
            "strategy_instance_pk": ["a", "b", "c", "d"],
            "cluster_id": [0, 1, 2, 3],
            "family_id": ["f0", "f1", "f2", "f3"],
            "robustness_score": [0.9, 0.8, 0.7, 0.6],
            "availability_ratio": [1.0, 1.0, 1.0, 1.0],
            "avg_turnover_metrics": [0.1, 0.2, 0.3, 0.4],
        }
    )
    returns = np.asarray([[0.01, 0.0, -0.01, 0.02], [0.0, 0.01, -0.01, 0.01], [0.01, 0.01, -0.02, 0.0], [0.0, 0.0, 0.0, 0.01]], dtype=np.float64)
    bundle = build_covariance_bundle(returns, np.ones_like(returns, dtype=bool), np.ones((4, 4), dtype=np.float64), np.asarray([0, 1, 2, 3]), Module6Config().dependence)
    cfg = Module6Config(generator=GeneratorConfig(random_sparse_quota=4, cluster_balanced_quota=2, hrp_variant_quota=9, active_cardinality_choices=(2, 3)))
    candidates, weights = generate_all_portfolios(
        reduced_universe=ReducedUniverseSpec("ru", ("a", "b", "c", "d"), ("a",), ("c",), 4),
        strategy_frame=strategy_frame,
        covariance_bundle=bundle,
        returns_exec=returns,
        column_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        config=cfg,
        calendar_version="calv1",
    )
    grouped = weights.groupby("portfolio_pk")["target_weight"].sum()
    assert (grouped <= 1.0 + 1.0e-12).all()
    assert (weights["target_weight"] >= 0.0).all()
    merged = weights.merge(strategy_frame[["strategy_instance_pk", "cluster_id", "family_id"]], on="strategy_instance_pk", how="left")
    assert merged.groupby(["portfolio_pk", "cluster_id"])["target_weight"].sum().max() <= cfg.generator.per_cluster_cap + 1.0e-12
    assert merged.groupby(["portfolio_pk", "family_id"])["target_weight"].sum().max() <= cfg.generator.per_family_cap + 1.0e-12
    assert merged.groupby("portfolio_pk")["target_weight"].max().max() <= cfg.generator.per_sleeve_cap + 1.0e-12
    counts = candidates.groupby("generator_family").size().to_dict()
    assert counts.get("random_sparse", 0) <= cfg.generator.random_sparse_quota
    assert counts.get("cluster_balanced", 0) <= cfg.generator.cluster_balanced_quota
    assert counts.get("hrp_risk", 0) <= cfg.generator.hrp_variant_quota


def test_generators_handle_single_strategy_universe():
    strategy_frame = pd.DataFrame(
        {
            "strategy_instance_pk": ["a"],
            "cluster_id": [0],
            "family_id": ["f0"],
            "robustness_score": [0.9],
            "availability_ratio": [1.0],
            "avg_turnover_metrics": [0.1],
        }
    )
    returns = np.asarray([[0.01], [0.0], [0.01], [0.0]], dtype=np.float64)
    bundle = build_covariance_bundle(
        returns,
        np.ones_like(returns, dtype=bool),
        np.ones((4, 1), dtype=np.float64),
        np.asarray([0], dtype=np.int64),
        Module6Config().dependence,
    )
    cfg = Module6Config(generator=GeneratorConfig(random_sparse_quota=2, cluster_balanced_quota=2, hrp_variant_quota=2, active_cardinality_choices=(1,)))
    candidates, weights = generate_all_portfolios(
        reduced_universe=ReducedUniverseSpec("ru", ("a",), ("a",), tuple(), 1),
        strategy_frame=strategy_frame,
        covariance_bundle=bundle,
        returns_exec=returns,
        column_indices=np.asarray([0], dtype=np.int64),
        config=cfg,
        calendar_version="calv1",
    )
    assert not candidates.empty
    assert not weights.empty
    grouped = weights.groupby("portfolio_pk")["target_weight"].sum()
    assert (grouped <= 1.0 + 1.0e-12).all()
