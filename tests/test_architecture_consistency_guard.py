from __future__ import annotations

from weightiz.shared.validation.architecture_guard import run_architecture_consistency_check


def test_architecture_consistency_check_passes_on_repo_state() -> None:
    run_architecture_consistency_check()
