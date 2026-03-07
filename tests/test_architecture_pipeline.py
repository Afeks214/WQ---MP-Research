from __future__ import annotations

import inspect

import pytest

import run_research
import sweep_runner
import weightiz_module2_core as m2
import weightiz_module4_strategy_funnel as m4
import weightiz_module5_harness as h


def test_single_runtime_dispatch_and_sweep_stub() -> None:
    src = inspect.getsource(run_research)
    assert "run_weightiz_harness(" in src
    assert "run_zimtra_sweep(" not in src
    with pytest.raises(RuntimeError, match="PARALLEL_ENGINE_FORBIDDEN"):
        sweep_runner.run_zimtra_sweep()


def test_worker_pipeline_uses_shm_module3_module4_signal_and_risk_engine() -> None:
    init_src = inspect.getsource(h._init_worker_context)
    group_src = inspect.getsource(h._run_group_task)
    assert "attach_shared_feature_store(" in init_src
    assert "create_shared_feature_store(" not in group_src
    assert "run_module3_structural_aggregation(" in group_src
    assert "run_module4_signal_funnel(" in group_src
    assert "simulate_portfolio_from_signals(" in group_src
    assert "run_module4_strategy_funnel(" not in group_src
    assert "run_weightiz_profile_engine(" not in group_src


def test_module2_has_worker_execution_guard() -> None:
    src = inspect.getsource(m2.run_weightiz_profile_engine)
    assert "MODULE2_WORKER_EXECUTION_FORBIDDEN" in src


def test_module4_execution_api_is_forbidden_in_canonical_path() -> None:
    with pytest.raises(RuntimeError, match="MODULE4_EXECUTION_FORBIDDEN_IN_CANONICAL_PATH"):
        m4.run_module4_strategy_funnel(None, None, None)  # type: ignore[arg-type]


def test_runtime_modules_are_print_free() -> None:
    files = [run_research, m2, h, m4]
    for mod in files:
        src = inspect.getsource(mod)
        assert "print(" not in src
