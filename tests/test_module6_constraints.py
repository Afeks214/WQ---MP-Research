from __future__ import annotations

import numpy as np

from weightiz.module6.constraints import check_path_constraints, project_to_feasible_weights
from tests.module6_testkit import make_test_config


def test_project_to_feasible_weights_respects_caps():
    cfg = make_test_config()
    result = project_to_feasible_weights(
        target_weights=np.asarray([0.8, 0.3], dtype=np.float64),
        gross_mult=np.asarray([2.0, 2.0], dtype=np.float64),
        overnight_flags=np.asarray([1, 1], dtype=np.int8),
        cluster_ids=np.asarray([0, 0], dtype=np.int64),
        family_ids=np.asarray(["f0", "f0"], dtype=object),
        priority_scores=np.asarray([1.0, 0.5], dtype=np.float64),
        config=cfg,
    )
    assert result.cash_weight >= cfg.simulator.min_cash_weight - 1.0e-12
    assert np.all(result.weights <= cfg.simulator.max_sleeve_weight + 1.0e-12)


def test_check_path_constraints_flags_breach():
    flags = check_path_constraints(
        equity=500.0,
        day_start_equity=1000.0,
        gross_exposure_mult=10.0,
        cash_weight=0.0,
        config=make_test_config(),
        overnight_active_count=10,
        buying_power_headroom=-0.1,
    )
    assert "capital_floor_hit" in flags
    assert "gross_limit_breach" in flags
    assert "buying_power_breach" in flags
