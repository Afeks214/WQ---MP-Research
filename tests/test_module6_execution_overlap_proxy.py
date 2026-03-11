from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from weightiz.module6.execution_overlap import build_execution_overlap_proxy
from weightiz.module6.config import DependenceConfig
from weightiz.module6.utils import Module6ValidationError


def test_execution_overlap_proxy_component_correctness():
    instances = pd.DataFrame(
        {
            "strategy_instance_pk": ["a", "b"],
            "candidate_id": ["ca", "cb"],
            "split_id": ["wf_000", "wf_000"],
            "scenario_id": ["baseline", "baseline"],
        }
    )
    trade_log = pd.DataFrame(
        {
            "candidate_id": ["ca", "cb"],
            "split_id": ["wf_000", "wf_000"],
            "scenario_id": ["baseline", "baseline"],
            "symbol": ["SYM0", "SYM0"],
            "ts_ns": [1, 1],
            "filled_qty": [1.0, 1.0],
            "exec_price": [100.0, 100.0],
        }
    )
    overlap = build_execution_overlap_proxy(
        instance_rows=instances,
        trade_log=trade_log,
        turnover_matrix=np.asarray([[1.0, 1.0], [0.0, 0.0]], dtype=np.float64),
        gross_peak_matrix=np.asarray([[1.0, 1.0], [0.0, 0.0]], dtype=np.float64),
        config=DependenceConfig(),
        candidate_pairs=[(0, 1)],
    )
    assert overlap.composite.shape == (2, 2)
    assert overlap.composite[0, 1] > 0.0


def test_execution_overlap_proxy_weights_are_locked():
    instances = pd.DataFrame(
        {
            "strategy_instance_pk": ["a", "b"],
            "candidate_id": ["ca", "cb"],
            "split_id": ["wf_000", "wf_000"],
            "scenario_id": ["baseline", "baseline"],
        }
    )
    trade_log = pd.DataFrame(
        {
            "candidate_id": ["ca", "cb"],
            "split_id": ["wf_000", "wf_000"],
            "scenario_id": ["baseline", "baseline"],
            "symbol": ["SYM0", "SYM0"],
            "ts_ns": [1, 1],
            "filled_qty": [1.0, 1.0],
            "exec_price": [100.0, 100.0],
        }
    )
    with pytest.raises(Module6ValidationError):
        build_execution_overlap_proxy(
            instance_rows=instances,
            trade_log=trade_log,
            turnover_matrix=np.asarray([[1.0, 1.0]], dtype=np.float64),
            gross_peak_matrix=np.asarray([[1.0, 1.0]], dtype=np.float64),
            config=DependenceConfig(overlap_weight_activity=0.20),
            candidate_pairs=[(0, 1)],
        )


def test_execution_overlap_proxy_allows_single_instance_without_trade_support():
    instances = pd.DataFrame(
        {
            "strategy_instance_pk": ["a"],
            "candidate_id": ["ca"],
            "split_id": ["wf_000"],
            "scenario_id": ["baseline"],
        }
    )
    overlap = build_execution_overlap_proxy(
        instance_rows=instances,
        trade_log=pd.DataFrame(columns=["candidate_id", "split_id", "scenario_id", "symbol"]),
        turnover_matrix=np.asarray([[0.0]], dtype=np.float64),
        gross_peak_matrix=np.asarray([[0.0]], dtype=np.float64),
        config=DependenceConfig(),
        candidate_pairs=[],
    )
    assert overlap.composite.shape == (1, 1)
    assert overlap.composite.nnz == 0
