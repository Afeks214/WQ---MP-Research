from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from weightiz.module1.core import EngineConfig, NS_PER_MIN, preallocate_state
from weightiz.module5.harness.stress import apply_missing_bursts


def _make_state(*, t_count: int = 6, a_count: int = 2):
    start_ns = np.datetime64("2025-01-06T14:30:00", "ns").astype(np.int64)
    ts_ns = start_ns + np.arange(t_count, dtype=np.int64) * np.int64(NS_PER_MIN)
    cfg = EngineConfig(T=t_count, A=a_count, B=64, tick_size=np.full(a_count, 0.01, dtype=np.float64))
    state = preallocate_state(ts_ns=ts_ns, cfg=cfg, symbols=tuple(f"S{i}" for i in range(a_count)))
    base = np.arange(t_count * a_count, dtype=np.float64).reshape(t_count, a_count) + 100.0
    state.open_px[:] = base
    state.high_px[:] = base + 0.2
    state.low_px[:] = base - 0.2
    state.close_px[:] = base + 0.1
    state.volume[:] = 1000.0
    state.rvol[:] = 1.0
    state.atr_floor[:] = 0.5
    state.bar_valid[:] = True
    return state


class _StubRng:
    def __init__(self, random_outputs: list[np.ndarray], integer_outputs: list[int]):
        self._random_outputs = [np.asarray(x, dtype=np.float64) for x in random_outputs]
        self._integer_outputs = [int(x) for x in integer_outputs]
        self.random_shapes: list[object] = []

    def random(self, size=None):
        self.random_shapes.append(size)
        if not self._random_outputs:
            raise AssertionError("unexpected random() call")
        out = self._random_outputs.pop(0)
        expected = (int(size),) if isinstance(size, (int, np.integer)) else tuple(size)
        if out.shape != expected:
            raise AssertionError(f"random() shape mismatch: got {out.shape}, expected {expected}")
        return out.copy()

    def integers(self, low, high=None, size=None):
        if size is not None:
            raise AssertionError("size-based integers() not expected in apply_missing_bursts")
        if not self._integer_outputs:
            raise AssertionError("unexpected integers() call")
        return int(self._integer_outputs.pop(0))


def test_apply_missing_bursts_marks_contiguous_ranges_per_asset() -> None:
    state = _make_state()
    scenario = SimpleNamespace(missing_burst_prob=0.5, missing_burst_min=2, missing_burst_max=3)
    rng = _StubRng(
        random_outputs=[
            np.array([0.0, 1.0, 1.0, 0.0, 1.0, 1.0], dtype=np.float64),
            np.array([1.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        ],
        integer_outputs=[2, 2, 3],
    )

    apply_missing_bursts(state, np.ones(state.cfg.T, dtype=bool), scenario, rng)

    invalid_col0 = np.flatnonzero(~state.bar_valid[:, 0]).tolist()
    invalid_col1 = np.flatnonzero(~state.bar_valid[:, 1]).tolist()
    assert invalid_col0 == [0, 1, 3, 4]
    assert invalid_col1 == [1, 2, 3]


def test_apply_missing_bursts_is_deterministic_for_identical_seed() -> None:
    scenario = SimpleNamespace(missing_burst_prob=0.35, missing_burst_min=1, missing_burst_max=3)
    active_t = np.array([True, True, False, True, True, True], dtype=bool)
    left = _make_state()
    right = _make_state()

    apply_missing_bursts(left, active_t, scenario, np.random.default_rng(47))
    apply_missing_bursts(right, active_t, scenario, np.random.default_rng(47))

    np.testing.assert_array_equal(left.bar_valid, right.bar_valid)
    np.testing.assert_array_equal(np.isnan(left.close_px), np.isnan(right.close_px))


def test_apply_missing_bursts_does_not_request_full_dense_random_matrix() -> None:
    state = _make_state(t_count=5, a_count=3)
    scenario = SimpleNamespace(missing_burst_prob=0.1, missing_burst_min=1, missing_burst_max=1)
    rng = _StubRng(
        random_outputs=[
            np.ones(5, dtype=np.float64),
            np.ones(5, dtype=np.float64),
            np.ones(5, dtype=np.float64),
        ],
        integer_outputs=[],
    )

    apply_missing_bursts(state, np.ones(state.cfg.T, dtype=bool), scenario, rng)

    assert rng.random_shapes == [5, 5, 5]
