from __future__ import annotations

import numpy as np

from module6.config import GeneratorConfig, Module6Config
from module6.dependence import build_covariance_bundle
from module6.generators.mv_shrinkage import generate_mv_variants
from module6.types import ReducedUniverseSpec


def test_mv_generator_disabled_by_default():
    returns = np.asarray([[0.01, 0.02], [0.0, 0.01], [0.01, 0.0], [0.02, 0.01], [0.01, 0.03], [0.0, 0.01]], dtype=np.float64)
    bundle = build_covariance_bundle(returns, np.ones_like(returns, dtype=bool), np.ones((2, 4), dtype=np.float64), np.asarray([0, 1], dtype=np.int64), Module6Config().dependence)
    frame, weights = generate_mv_variants(
        reduced_universe=ReducedUniverseSpec("ru", ("a", "b"), ("a",), tuple(), 2),
        covariance_bundle=bundle,
        returns_exec=returns,
        column_indices=np.asarray([0, 1], dtype=np.int64),
        config=Module6Config(generator=GeneratorConfig(enable_mv_diagnostic=False)),
        calendar_version="cal1",
    )
    assert frame.shape[0] == 0
    assert weights.shape[0] == 0

