from __future__ import annotations

import numpy as np


def assert_float64(name: str, array: np.ndarray) -> None:
    """Fail-fast dtype guard for all numeric pipeline boundaries."""
    arr = np.asarray(array)
    if arr.dtype != np.float64:
        raise RuntimeError(f"FLOAT64_ENFORCEMENT_FAILURE: {name}")
