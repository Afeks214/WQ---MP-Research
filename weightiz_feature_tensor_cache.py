from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from weightiz_dtype_guard import assert_float64


PROFILE_CACHE_SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class FeatureTensorManifest:
    shape: list[int]
    dtype: str
    feature_map: dict[str, int]
    window_map: dict[str, int]
    hash_inputs: dict[str, Any]
    created_utc: str
    dataset_hash: str
    dataset_version: str
    asset_universe: list[str]
    rows_per_asset: int
    timestamp_start: str
    timestamp_end: str


def _stable_hash_obj(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_tensor_hash(hash_inputs: dict[str, Any]) -> str:
    required = {"data_hash", "module2_config", "profile_windows", "schema_version"}
    missing = sorted(required - set(hash_inputs.keys()))
    if missing:
        raise RuntimeError(f"Missing hash inputs: {missing}")
    return _stable_hash_obj(hash_inputs)[:16]


def profile_cache_paths(cache_dir: Path, tensor_hash: str) -> tuple[Path, Path]:
    cache_dir = cache_dir.resolve()
    return (
        cache_dir / f"profile_tensor_{tensor_hash}.npz",
        cache_dir / f"profile_tensor_{tensor_hash}.json",
    )


def cleanup_stale_tmp_cache_files(cache_dir: Path) -> int:
    cache_dir = cache_dir.resolve()
    if not cache_dir.exists():
        return 0
    n = 0
    for p in cache_dir.glob("profile_tensor_*.npz.tmp"):
        try:
            p.unlink(missing_ok=True)
            n += 1
        except Exception:
            continue
    return n


def _atomic_write_bytes(final_path: Path, payload: bytes) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, final_path)


def save_tensor_cache(
    npz_path: Path,
    json_path: Path,
    tensor: np.ndarray,
    manifest: FeatureTensorManifest,
) -> None:
    assert_float64("feature_tensor_cache_save", tensor)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    cleanup_stale_tmp_cache_files(npz_path.parent)

    # Write NPZ atomically via tmp + fsync + replace.
    tmp_npz = npz_path.with_suffix(npz_path.suffix + ".tmp")
    with tmp_npz.open("wb") as f:
        np.savez_compressed(f, tensor=tensor)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_npz, npz_path)

    manifest_payload = json.dumps(asdict(manifest), ensure_ascii=False, indent=2).encode("utf-8")
    _atomic_write_bytes(json_path, manifest_payload)


def load_tensor_cache(npz_path: Path, json_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    if not npz_path.exists() or not json_path.exists():
        raise RuntimeError(f"Profile cache missing: npz={npz_path.exists()} json={json_path.exists()}")
    try:
        with np.load(npz_path, allow_pickle=False) as d:
            if "tensor" not in d:
                raise RuntimeError("Corrupted tensor cache: missing 'tensor' array")
            tensor = np.asarray(d["tensor"])
    except Exception as exc:
        raise RuntimeError("Failed loading profile tensor cache (possible partial/corrupt write)") from exc
    assert_float64("feature_tensor_cache_load", tensor)

    manifest = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError("Invalid tensor manifest format")
    return tensor, manifest


def build_manifest(
    tensor: np.ndarray,
    *,
    feature_map: dict[str, int],
    window_map: dict[str, int],
    hash_inputs: dict[str, Any],
    dataset_hash: str,
    dataset_version: str,
    asset_universe: list[str],
    rows_per_asset: int,
    timestamp_start: str,
    timestamp_end: str,
) -> FeatureTensorManifest:
    assert_float64("feature_tensor_manifest", tensor)
    return FeatureTensorManifest(
        shape=[int(x) for x in tensor.shape],
        dtype=str(tensor.dtype),
        feature_map={str(k): int(v) for k, v in feature_map.items()},
        window_map={str(k): int(v) for k, v in window_map.items()},
        hash_inputs=hash_inputs,
        created_utc=datetime.now(timezone.utc).isoformat(),
        dataset_hash=str(dataset_hash),
        dataset_version=str(dataset_version),
        asset_universe=[str(x) for x in asset_universe],
        rows_per_asset=int(rows_per_asset),
        timestamp_start=str(timestamp_start),
        timestamp_end=str(timestamp_end),
    )
