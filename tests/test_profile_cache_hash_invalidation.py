from __future__ import annotations

from weightiz.shared.io.feature_tensor_cache import compute_tensor_hash


def test_profile_cache_hash_changes_on_config_change():
    a = {
        "data_hash": "d1",
        "module2_config": {"x": 1},
        "profile_windows": [15, 30],
        "schema_version": "1",
    }
    b = {
        "data_hash": "d1",
        "module2_config": {"x": 2},
        "profile_windows": [15, 30],
        "schema_version": "1",
    }
    assert compute_tensor_hash(a) != compute_tensor_hash(b)
