from __future__ import annotations

from pathlib import Path


RUNTIME_FILES = [
    "run_research.py",
    "weightiz_module2_core.py",
    "weightiz_module3_structure.py",
    "weightiz_module4_strategy_funnel.py",
    "weightiz_module5_harness.py",
    "risk_engine.py",
]


def test_runtime_modules_have_no_print_statements() -> None:
    root = Path(__file__).resolve().parents[1]
    for rel in RUNTIME_FILES:
        src = (root / rel).read_text(encoding="utf-8")
        assert "print(" not in src, f"print() found in {rel}"
