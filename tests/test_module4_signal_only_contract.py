from __future__ import annotations

import inspect
from pathlib import Path

import weightiz_module4_strategy_funnel as m4


def test_signal_only_api_exists():
    src = Path("weightiz_module4_strategy_funnel.py").read_text(encoding="utf-8")
    assert "def run_module4_signal_funnel" in src
    assert "class Module4SignalOutput" in src


def test_signal_only_bridge_is_thin_and_uses_new_orchestrator() -> None:
    src = inspect.getsource(m4.run_module4_signal_funnel)
    assert "run_module4_funnel(" in src
    assert "bo_l =" not in src
    assert "bo_s =" not in src
    assert "target = np.zeros" not in src
    assert "regime = np.full" not in src
