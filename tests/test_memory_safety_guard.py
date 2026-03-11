from __future__ import annotations

import pytest

from weightiz.shared.io.shared_feature_store import enforce_memory_safety, estimate_tensor_bytes


def test_memory_safety_guard_raises():
    b = estimate_tensor_bytes(100, 100, 100, 100)
    with pytest.raises(RuntimeError, match="safe memory limit"):
        enforce_memory_safety(b, available_ram_bytes=10)


def test_memory_safety_guard_passes():
    b = estimate_tensor_bytes(2, 10, 3, 2)
    enforce_memory_safety(b, available_ram_bytes=10**9)
