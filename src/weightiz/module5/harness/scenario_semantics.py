from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np


def apply_threshold_perturbation(m4_cfg: Any, scenario: Any) -> Any:
    entry_shift = float(getattr(scenario, "entry_threshold_shift", 0.0))
    exit_shift = float(getattr(scenario, "exit_threshold_shift", 0.0))
    if abs(entry_shift) <= 0.0 and abs(exit_shift) <= 0.0:
        return m4_cfg
    entry_threshold = float(getattr(m4_cfg, "entry_threshold")) + entry_shift
    exit_threshold = float(getattr(m4_cfg, "exit_threshold")) + exit_shift
    if not np.isfinite(entry_threshold) or entry_threshold < 0.0:
        raise RuntimeError(f"scenario entry_threshold_shift produced invalid threshold: {entry_threshold}")
    if not np.isfinite(exit_threshold) or exit_threshold < 0.0:
        raise RuntimeError(f"scenario exit_threshold_shift produced invalid threshold: {exit_threshold}")
    return replace(
        m4_cfg,
        entry_threshold=float(entry_threshold),
        exit_threshold=float(exit_threshold),
    )


def apply_signal_lag(target_qty_ta: np.ndarray, lag_bars: int) -> np.ndarray:
    arr = np.asarray(target_qty_ta, dtype=np.float64)
    lag = int(lag_bars)
    if lag <= 0:
        return arr.copy()
    out = np.zeros_like(arr, dtype=np.float64)
    if lag < arr.shape[0]:
        out[lag:] = arr[:-lag]
    return out


def apply_target_scale(target_qty_ta: np.ndarray, target_scale_mult: float) -> np.ndarray:
    arr = np.asarray(target_qty_ta, dtype=np.float64)
    scale = float(target_scale_mult)
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError(f"scenario target_scale_mult must be finite and >0, got {target_scale_mult!r}")
    if abs(scale - 1.0) <= 1.0e-12:
        return arr.copy()
    return arr * scale
