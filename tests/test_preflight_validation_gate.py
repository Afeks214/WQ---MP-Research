from __future__ import annotations

import pytest

from weightiz.shared.validation.validation_suite import run_preflight_validation_suite


def test_preflight_rejects_parallel_runtime():
    with pytest.raises(RuntimeError, match="parallel runtime path"):
        run_preflight_validation_suite({"search": {"seed": 1}}, context={"parallel_runtime_enabled": True})


def test_preflight_accepts_valid_min_config():
    run_preflight_validation_suite({"search": {"seed": 7}}, context={"parallel_runtime_enabled": False})
