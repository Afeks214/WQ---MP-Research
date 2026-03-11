from __future__ import annotations

import numpy as np
import pytest

from weightiz.shared.io.feature_tensor_cache import (
    build_manifest,
    load_tensor_cache,
    save_tensor_cache,
)


def test_atomic_cache_write_survives_interrupted_write(tmp_path, monkeypatch):
    tensor = np.ones((2, 4, 3, 2), dtype=np.float64)
    npz = tmp_path / "profile_tensor_deadbeef.npz"
    js = tmp_path / "profile_tensor_deadbeef.json"

    manifest = build_manifest(
        tensor,
        feature_map={"f0": 0, "f1": 1, "f2": 2},
        window_map={"0": 15, "1": 30},
        hash_inputs={"data_hash": "d", "module2_config": {}, "profile_windows": [15, 30], "schema_version": "1"},
        dataset_hash="ds",
        dataset_version="v1",
        asset_universe=["A", "B"],
        rows_per_asset=4,
        timestamp_start="1",
        timestamp_end="2",
    )
    save_tensor_cache(npz, js, tensor, manifest)

    orig = np.load(npz)["tensor"].copy()

    def _broken_savez(file_obj, **_kwargs):
        file_obj.write(b"PARTIAL")
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(np, "savez_compressed", _broken_savez)
    with pytest.raises(RuntimeError):
        save_tensor_cache(npz, js, tensor * 2.0, manifest)

    loaded, _m = load_tensor_cache(npz, js)
    assert np.array_equal(loaded, orig)


def test_corrupted_cache_cannot_be_loaded(tmp_path):
    npz = tmp_path / "profile_tensor_bad.npz"
    js = tmp_path / "profile_tensor_bad.json"
    npz.write_bytes(b"not-a-valid-npz")
    js.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError):
        load_tensor_cache(npz, js)
