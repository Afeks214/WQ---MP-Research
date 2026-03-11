from __future__ import annotations

import inspect
from pathlib import Path

import run_research
import weightiz.module2.core as m2
import weightiz.module3.bridge as m3
import weightiz.module4.risk_engine as risk_engine
import weightiz.module4.strategy_funnel as m4
import weightiz.module5.orchestrator as m5

RUNTIME_MODULES = [run_research, m2, m3, m4, m5, risk_engine]


def test_runtime_modules_have_no_print_statements() -> None:
    for module in RUNTIME_MODULES:
        src = inspect.getsource(module)
        assert "print(" not in src, f"print() found in {module.__name__}"
