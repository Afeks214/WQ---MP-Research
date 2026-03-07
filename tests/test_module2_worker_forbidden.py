from __future__ import annotations

import os

import numpy as np
import pytest

from weightiz_module1_core import EngineConfig, preallocate_state
from weightiz_module2_core import Module2Config, run_weightiz_profile_engine


def test_module2_worker_execution_forbidden() -> None:
    cfg = EngineConfig(T=4, A=1, B=8, tick_size=np.array([0.01], dtype=np.float64), seed=1)
    st = preallocate_state(
        ts_ns=np.arange(4, dtype=np.int64),
        cfg=cfg,
        symbols=("X",),
    )
    st.open_px[:] = 1.0
    st.high_px[:] = 1.0
    st.low_px[:] = 1.0
    st.close_px[:] = 1.0
    st.volume[:] = 1.0
    st.bar_valid[:] = True

    os.environ["WEIGHTIZ_WORKER_PROCESS"] = "1"
    try:
        with pytest.raises(RuntimeError, match="MODULE2_WORKER_EXECUTION_FORBIDDEN"):
            run_weightiz_profile_engine(st, Module2Config())
    finally:
        os.environ.pop("WEIGHTIZ_WORKER_PROCESS", None)
