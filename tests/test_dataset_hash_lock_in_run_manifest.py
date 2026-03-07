from __future__ import annotations

from pathlib import Path


def test_dataset_hash_key_exists_in_manifest_build():
    src = Path("weightiz_module5_harness.py").read_text(encoding="utf-8")
    assert '"dataset_hash"' in src
    assert '"feature_tensor"' in src
