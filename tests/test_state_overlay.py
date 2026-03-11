from __future__ import annotations

import numpy as np
import pytest

from module5.harness.state_overlay import (
    BaseTensorState,
    CandidateScratch,
    CombinedStateView,
    FeatureOverlay,
    MarketOverlay,
)
from weightiz_module1_core import EngineConfig, preallocate_state


def _cfg(*, T: int, A: int):
    return EngineConfig(T=T, A=A, B=64, tick_size=np.full(A, 0.01, dtype=np.float64))


def _ts_ns(start: str, periods: int) -> np.ndarray:
    base = np.datetime64(start, "m").astype("datetime64[ns]").astype(np.int64)
    return base + np.arange(periods, dtype=np.int64) * 60_000_000_000


def _base_tensor_state(*, T: int = 6, A: int = 2) -> BaseTensorState:
    state = preallocate_state(
        ts_ns=_ts_ns("2024-01-03T14:30", T),
        cfg=_cfg(T=T, A=A),
        symbols=tuple(f"S{i}" for i in range(A)),
    )
    state.open_px[:, :] = 100.0
    state.high_px[:, :] = 101.0
    state.low_px[:, :] = 99.0
    state.close_px[:, :] = 100.5
    state.volume[:, :] = 1_000.0
    state.bar_valid[:, :] = True
    return BaseTensorState.from_tensor_state(state)


def test_base_tensor_state_arrays_are_read_only() -> None:
    base = _base_tensor_state()

    assert base.open_px.flags.writeable is False
    assert base.bar_valid.flags.writeable is False
    with pytest.raises(ValueError):
        base.open_px[0, 0] = 123.0


def test_combined_state_view_routes_overlay_and_blocks_base_mutation() -> None:
    base = _base_tensor_state()
    market = MarketOverlay.from_base(base)
    feature = FeatureOverlay.allocate(base)
    scratch = CandidateScratch.allocate(base, "compact")
    view = CombinedStateView(base, market, feature, scratch)

    view.open_px = np.full_like(view.open_px, 111.0)
    view.rvol = np.full_like(view.rvol, 2.0)
    view.order_side = np.full_like(view.order_side, 3, dtype=np.int8)

    assert np.all(market.open_px == 111.0)
    assert np.all(feature.rvol == 2.0)
    assert np.all(scratch.order_side == 3)
    assert np.all(base.open_px == 100.0)

    with pytest.raises(AttributeError):
        view.ts_ns = np.arange(base.cfg.T, dtype=np.int64)


def test_candidate_scratch_compact_reset_restores_template_values() -> None:
    base = _base_tensor_state(T=4, A=2)
    scratch = CandidateScratch.allocate(base, "compact")

    scratch.order_side[:, :] = 7
    scratch.position_qty[:, :] = 9.0
    scratch.equity[:] = 1.0
    scratch.margin_used[:] = 2.0
    scratch.buying_power[:] = 3.0
    scratch.daily_loss[:] = 4.0
    scratch.daily_loss_breach_flag[:] = 1

    scratch.reset_from_base(base)

    assert np.all(scratch.order_side == 0)
    assert np.all(scratch.position_qty == 0.0)
    assert np.all(scratch.equity == float(base.cfg.initial_cash))
    assert np.all(scratch.margin_used == 0.0)
    assert np.all(scratch.buying_power == float(base.cfg.initial_cash))
    assert np.all(scratch.daily_loss == 0.0)
    assert np.all(scratch.daily_loss_breach_flag == 0)


def test_combined_state_view_compact_mode_fails_closed_for_missing_full_only_field() -> None:
    base = _base_tensor_state(T=4, A=2)
    market = MarketOverlay.from_base(base)
    feature = FeatureOverlay.allocate(base)
    scratch = CandidateScratch.allocate(base, "compact")
    view = CombinedStateView(base, market, feature, scratch)

    with pytest.raises(RuntimeError, match="required writable field orders"):
        _ = view.orders


def test_template_scratch_exposes_read_only_structural_templates() -> None:
    base = _base_tensor_state(T=4, A=2)
    market = MarketOverlay.from_base(base)
    feature = FeatureOverlay.allocate(base)
    scratch = CandidateScratch.template_from_base(base)
    view = CombinedStateView(base, market, feature, scratch)

    assert scratch.mode == "template"
    assert scratch.nbytes == 0
    assert view.orders.flags.writeable is False
    with pytest.raises((RuntimeError, ValueError)):
        view.orders[:, :, :] = 0.0
