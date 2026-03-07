from __future__ import annotations

import inspect

import weightiz_module5_harness as h


def test_worker_group_task_uses_risk_engine_only() -> None:
    src = inspect.getsource(h._run_group_task)
    assert "simulate_portfolio_from_signals(" in src
    assert "run_module4_signal_funnel(" in src
    assert "run_module4_strategy_funnel(" not in src
