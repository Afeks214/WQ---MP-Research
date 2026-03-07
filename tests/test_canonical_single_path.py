from __future__ import annotations

import pytest

import run_research as rr


def test_parallel_engine_forbidden_guard():
    cfg = rr.RunConfigModel.model_validate(
        {
            "symbols": ["AAA", "BBB"],
            "zimtra_sweep": {"enabled": True},
        }
    )
    with pytest.raises(RuntimeError, match="PARALLEL_ENGINE_FORBIDDEN"):
        rr._enforce_canonical_runtime_path(cfg)
