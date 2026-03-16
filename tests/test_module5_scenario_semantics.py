from __future__ import annotations

import numpy as np
import pytest

from weightiz.module4.strategy_funnel import Module4Config
from weightiz.module5.harness.scenario_semantics import apply_signal_lag, apply_target_scale, apply_threshold_perturbation


class _Scenario:
    def __init__(self, *, signal_lag_bars: int = 0, entry_threshold_shift: float = 0.0, exit_threshold_shift: float = 0.0, target_scale_mult: float = 1.0) -> None:
        self.signal_lag_bars = int(signal_lag_bars)
        self.entry_threshold_shift = float(entry_threshold_shift)
        self.exit_threshold_shift = float(exit_threshold_shift)
        self.target_scale_mult = float(target_scale_mult)


def test_apply_signal_lag_zero_fills_front_and_shifts_target_qty() -> None:
    arr = np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=np.float64)
    lag1 = apply_signal_lag(arr, 1)
    lag2 = apply_signal_lag(arr, 2)

    np.testing.assert_allclose(lag1[:, 0], np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64))
    np.testing.assert_allclose(lag2[:, 0], np.asarray([0.0, 0.0, 1.0, 2.0], dtype=np.float64))


def test_apply_threshold_perturbation_uses_live_module4_threshold_fields() -> None:
    cfg = Module4Config(entry_threshold=0.55, exit_threshold=0.25)
    scenario = _Scenario(entry_threshold_shift=0.05, exit_threshold_shift=0.10)
    shifted = apply_threshold_perturbation(cfg, scenario)

    np.testing.assert_allclose(float(shifted.entry_threshold), 0.60, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(float(shifted.exit_threshold), 0.35, rtol=0.0, atol=1e-12)


def test_apply_threshold_perturbation_allows_live_thresholds_above_one_when_result_stays_non_negative() -> None:
    cfg = Module4Config(entry_threshold=1.10, exit_threshold=0.25)
    scenario = _Scenario(entry_threshold_shift=0.15)
    shifted = apply_threshold_perturbation(cfg, scenario)

    np.testing.assert_allclose(float(shifted.entry_threshold), 1.25, rtol=0.0, atol=1e-12)


def test_apply_threshold_perturbation_fails_closed_when_shift_is_invalid() -> None:
    cfg = Module4Config(entry_threshold=0.05, exit_threshold=0.25)
    scenario = _Scenario(entry_threshold_shift=-0.10)

    with pytest.raises(RuntimeError, match="entry_threshold_shift"):
        apply_threshold_perturbation(cfg, scenario)


def test_apply_target_scale_scales_once_and_requires_positive_multiplier() -> None:
    arr = np.asarray([[1.0], [2.0]], dtype=np.float64)
    scaled = apply_target_scale(arr, 5.0)
    np.testing.assert_allclose(scaled[:, 0], np.asarray([5.0, 10.0], dtype=np.float64))

    with pytest.raises(RuntimeError, match="target_scale_mult"):
        apply_target_scale(arr, 0.0)
