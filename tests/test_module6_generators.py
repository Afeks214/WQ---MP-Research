from __future__ import annotations

import numpy as np

from module6.config import GeneratorConfig, Module6Config
from module6.dependence import build_covariance_bundle
from module6.generators import generate_all_portfolios
from module6.types import ReducedUniverseSpec


def test_generator_outputs_are_simplex_and_nonnegative():
    strategy_frame = __import__("pandas").DataFrame(
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

