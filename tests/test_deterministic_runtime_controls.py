from __future__ import annotations

import random

import numpy as np
import pytest

from weightiz.cli import run_research as rr
from weightiz.shared.validation.validation_suite import run_preflight_validation_suite


def test_configure_deterministic_runtime_sets_env_and_seed(monkeypatch):
    rr._configure_deterministic_runtime(123)

    assert rr.os.environ["PYTHONHASHSEED"] == "123"
    assert rr.os.environ["OMP_NUM_THREADS"] == "1"
    assert rr.os.environ["MKL_NUM_THREADS"] == "1"
    assert rr.os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert rr.os.environ["NUMEXPR_NUM_THREADS"] == "1"
    assert rr.os.environ["VECLIB_MAXIMUM_THREADS"] == "1"

    rr._configure_deterministic_runtime(123)
    a1 = np.random.random(5)
    b1 = [random.random() for _ in range(5)]
    rr._configure_deterministic_runtime(123)
    a2 = np.random.random(5)
    b2 = [random.random() for _ in range(5)]

    assert np.allclose(a1, a2)
    assert b1 == b2


def test_preflight_requires_seed():
    with pytest.raises(RuntimeError, match="config.search.seed"):
        run_preflight_validation_suite({"search": {"seed": None}}, context={"parallel_runtime_enabled": False})
