from __future__ import annotations

from pathlib import Path


def test_signal_only_api_exists():
    src = Path("weightiz_module4_strategy_funnel.py").read_text(encoding="utf-8")
    assert "def run_module4_signal_funnel" in src
    assert "class Module4SignalOutput" in src
