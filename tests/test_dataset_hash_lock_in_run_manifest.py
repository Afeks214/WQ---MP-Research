from __future__ import annotations

import inspect

import weightiz.module5.orchestrator as h


def test_dataset_hash_key_exists_in_manifest_build():
    src = inspect.getsource(h)
    assert '"dataset_hash"' in src
    assert '"feature_tensor"' in src
