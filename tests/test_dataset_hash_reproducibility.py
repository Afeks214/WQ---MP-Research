from __future__ import annotations

import json

from weightiz_feature_tensor_cache import compute_tensor_hash


def test_dataset_hash_reproducibility_inputs_stable():
    payload = {
        "data_hash": "dataset123",
        "module2_config": {"w": 60},
        "profile_windows": [15, 30, 60],
        "schema_version": "1",
    }
    h1 = compute_tensor_hash(payload)
    h2 = compute_tensor_hash(json.loads(json.dumps(payload)))
    assert h1 == h2
