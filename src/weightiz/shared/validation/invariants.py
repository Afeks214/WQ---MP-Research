from __future__ import annotations

from typing import Any

import numpy as np


def assert_or_flag_finite(
    features: dict[str, np.ndarray],
    valid_mask: np.ndarray,
    context: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Data-quality finite guard.

    - Never raises for data issues; masks invalid rows and returns flags.
    - Raises only for programmer misuse (shape/type mismatch).
    - Guarantees returned valid_mask implies all provided features are finite.
    """
    if not isinstance(features, dict) or not features:
        raise RuntimeError(f"{context}: features must be a non-empty dict")

    vm = np.asarray(valid_mask, dtype=bool)
    if vm.ndim != 2:
        raise RuntimeError(f"{context}: valid_mask must be 2D bool, got shape={vm.shape}")

    combined = np.ones(vm.shape, dtype=bool)
    offending_features: list[str] = []

    for name in sorted(features.keys()):
        arr = np.asarray(features[name])
        if arr.shape[:2] != vm.shape:
            raise RuntimeError(
                f"{context}: feature '{name}' shape prefix mismatch: got {arr.shape}, expected prefix {vm.shape}"
            )

        if arr.ndim == 2:
            row_finite = np.isfinite(arr)
        else:
            tail_axes = tuple(range(2, arr.ndim))
            row_finite = np.all(np.isfinite(arr), axis=tail_axes)

        if row_finite.shape != vm.shape:
            raise RuntimeError(
                f"{context}: feature '{name}' finite mask shape mismatch: got {row_finite.shape}, expected {vm.shape}"
            )

        combined &= row_finite
        if np.any(vm & (~row_finite)):
            offending_features.append(str(name))

    updated = vm & combined
    invalid_points = np.argwhere(vm & (~combined))

    flags = {
        "context": str(context),
        "checked_features": sorted([str(k) for k in features.keys()]),
        "offending_features": sorted(offending_features),
        "invalid_count": int(invalid_points.shape[0]),
        "invalid_preview": invalid_points[:8].astype(np.int64).tolist(),
    }
    return updated, flags
