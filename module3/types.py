from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .window_registry import normalize_structural_windows


@dataclass
class Module3Config:
    structural_windows: tuple[int, ...] = (5, 15, 30, 60)
    selected_window: int = 30
    validate_outputs: bool = True

    # Context mode controls.
    context_mode: str = "ffill_last_complete"
    rolling_context_period: int = 5
    eps: float = 1e-12

    # Legacy compatibility fields (accepted but not primary).
    block_minutes: int = 30
    phase_mask: tuple[int, ...] = (1, 2)
    use_rth_minutes_only: bool = True
    rth_open_minute: int = 570
    last_minute_inclusive: int = 945
    include_partial_last_block: bool = True
    min_block_valid_bars: int = 1
    min_block_valid_ratio: float = 0.0
    ib_pop_frac: float = 0.01
    fail_on_non_finite_input: bool = True
    fail_on_non_finite_output: bool = True
    fail_on_bad_indices: bool = True
    fail_on_missing_prev_va: bool = False

    def __post_init__(self) -> None:
        sw = normalize_structural_windows(tuple(int(w) for w in self.structural_windows), fallback=(int(self.block_minutes),))
        self.structural_windows = sw
        if int(self.selected_window) not in self.structural_windows:
            self.selected_window = int(self.structural_windows[0])
        if str(self.context_mode) not in {"ffill_last_complete", "rolling_context", "regime_context"}:
            raise RuntimeError(
                "context_mode must be one of: ffill_last_complete, rolling_context, regime_context"
            )
        if int(self.rolling_context_period) <= 0:
            raise RuntimeError("rolling_context_period must be > 0")
        if float(self.eps) <= 0.0:
            raise RuntimeError("eps must be > 0")


@dataclass
class Module3Output:
    structure_tensor: np.ndarray
    context_tensor: np.ndarray
    profile_fingerprint_tensor: np.ndarray
    profile_regime_tensor: np.ndarray

    # Optional engine metadata tensors.
    context_valid_atw: np.ndarray | None = None
    context_source_index_atw: np.ndarray | None = None

    # Legacy compatibility fields expected by existing downstream modules.
    block_id_t: np.ndarray | None = None
    block_seq_t: np.ndarray | None = None
    block_end_flag_t: np.ndarray | None = None
    block_start_t_index_t: np.ndarray | None = None
    block_end_t_index_t: np.ndarray | None = None
    block_features_tak: np.ndarray | None = None
    block_valid_ta: np.ndarray | None = None
    context_tac: np.ndarray | None = None
    context_valid_ta: np.ndarray | None = None
    context_source_t_index_ta: np.ndarray | None = None
    ib_defined_ta: np.ndarray | None = None

    def assert_float64(self) -> None:
        for name, arr in [
            ("structure_tensor", self.structure_tensor),
            ("context_tensor", self.context_tensor),
            ("profile_fingerprint_tensor", self.profile_fingerprint_tensor),
            ("profile_regime_tensor", self.profile_regime_tensor),
        ]:
            if np.asarray(arr).dtype != np.float64:
                raise RuntimeError(f"{name} dtype must be float64, got {np.asarray(arr).dtype}")
