from __future__ import annotations

import numpy as np
import pandas as pd

from weightiz.module6.config import GeneratorConfig, Module6Config
from weightiz.module6.dependence import build_covariance_bundle
from weightiz.module6.generators.mv_shrinkage import generate_mv_variants
from weightiz.module6.types import ReducedUniverseSpec


def test_mv_generator_disabled_by_default():
    returns = np.asarray([[0.01, 0.02], [0.0, 0.01], [0.01, 0.0], [0.02, 0.01], [0.01, 0.03], [0.0, 0.01]], dtype=np.float64)
    bundle = build_covariance_bundle(returns, np.ones_like(returns, dtype=bool), np.ones((2, 4), dtype=np.float64), np.asarray([0, 1], dtype=np.int64), Module6Config().dependence)
    strategy_frame = pd.DataFrame(
        {
            "strategy_instance_pk": ["a", "b"],
            "cluster_id": [0, 1],
            "family_id": ["f0", "f1"],
        }
    )
    frame, weights = generate_mv_variants(
        reduced_universe=ReducedUniverseSpec("ru", ("a", "b"), ("a",), tuple(), 2),
        strategy_frame=strategy_frame,
        covariance_bundle=bundle,
        returns_exec=returns,
        column_indices=np.asarray([0, 1], dtype=np.int64),
        config=Module6Config(generator=GeneratorConfig(enable_mv_diagnostic=False)),
        calendar_version="cal1",
    )
    assert frame.shape[0] == 0
    assert weights.shape[0] == 0
