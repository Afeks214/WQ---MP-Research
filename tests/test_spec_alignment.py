from __future__ import annotations

import inspect
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from weightiz.module1.core import EngineConfig, preallocate_state
import weightiz.module2.core as m2core
from weightiz.module1.core import Phase, ProfileStatIdx, ScoreIdx
from weightiz.module2.core import (
    Module2Config,
    _build_poc_rank,
    _rolling_median_mad_causal,
    compute_value_area_greedy,
    precompute_market_physics,
    run_weightiz_profile_engine,
)
from weightiz.module2.market_profile_engine import MarketProfileRunArtifacts, run_streaming_profile_engine
from weightiz.module2.market_profile_kernels import (
    MixtureBarParams,
    ScoreInputs,
    build_bar_mixture_params,
    compute_pbuy_and_delta_coeff,
    inject_profile_mass,
)

from tests.fixtures.spec_cases import (
    EXECUTED_PATH,
    M2_CORE,
    M2_ENGINE,
    M2_KERNELS,
    M2_TENSOR,
    fill_bars,
    make_clock_override,
    make_fixed_physics,
    make_state,
    paper_bar_profile,
    paper_profile_stats_from_vp,
)


def _run_streaming_with_fixed_physics(
    state: object,
    cfg: Module2Config,
    physics: SimpleNamespace,
) -> MarketProfileRunArtifacts:
    return run_streaming_profile_engine(
        state=state,
        cfg=cfg,
        physics=physics,
        mode=str(state.cfg.mode),
        open_use=np.asarray(state.open_px, dtype=np.float64),
        high_use=np.asarray(state.high_px, dtype=np.float64),
        low_use=np.asarray(state.low_px, dtype=np.float64),
        close_use=np.asarray(state.close_px, dtype=np.float64),
        vol_use=np.asarray(state.volume, dtype=np.float64),
        valid=np.asarray(state.bar_valid, dtype=bool),
        build_poc_rank_fn=_build_poc_rank,
        compute_value_area_fn=compute_value_area_greedy,
        rolling_median_mad_fn=_rolling_median_mad_causal,
        profile_stat_idx=ProfileStatIdx,
        score_idx=ScoreIdx,
        phase_enum=Phase,
        collect_forensics=False,
    )


def test_execution_path_public_api_routes_to_streaming_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    state = make_state(T=2, tick_size=1.0, mode="sealed")
    fill_bars(
        state,
        [
            (100.0, 101.0, 99.0, 100.0, 10.0),
            (100.0, 101.0, 99.0, 100.0, 10.0),
        ],
    )
    cfg = Module2Config(profile_window_bars=2, profile_warmup_bars=1, atr_alpha=1.0)
    fake_physics = make_fixed_physics(T=2)
    called: dict[str, object] = {}

    def fake_precompute(_state: object, _cfg: Module2Config) -> object:
        called["precompute"] = True
        return fake_physics

    def fake_streaming(**kwargs: object) -> MarketProfileRunArtifacts:
        called["streaming_mode"] = kwargs["mode"]
        called["streaming_state_id"] = id(kwargs["state"])
        return MarketProfileRunArtifacts(
            computed_mask=np.zeros((2, 1), dtype=bool),
            mixture_history={},
            profile_history={},
            metric_history={},
        )

    monkeypatch.setattr(m2core, "precompute_market_physics", fake_precompute)
    monkeypatch.setattr(m2core, "run_streaming_profile_engine", fake_streaming)

    run_weightiz_profile_engine(state, cfg)

    assert called["precompute"] is True
    assert called["streaming_mode"] == "sealed"
    assert called["streaming_state_id"] == id(state)


def test_locked_defaults_match_spec() -> None:
    state = make_state(T=2, tick_size=0.01, mode="sealed")

    assert state.eps.eps_pdf == pytest.approx(1.0e-12, abs=0.0)
    assert state.eps.eps_vol == pytest.approx(1.0e-12, abs=0.0)
    np.testing.assert_allclose(state.eps.eps_div, np.array([0.01], dtype=np.float64), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(state.eps.eps_range, np.array([0.01], dtype=np.float64), rtol=0.0, atol=0.0)
    assert state.cfg.B == 240
    assert state.cfg.dx == pytest.approx(0.05, abs=0.0)
    assert state.x_grid[0] == pytest.approx(-6.0, abs=0.0)
    assert state.x_grid[-1] == pytest.approx(5.95, abs=1.0e-12)
    np.testing.assert_allclose(
        state.x_grid,
        -6.0 + 0.05 * np.arange(240, dtype=np.float64),
        rtol=0.0,
        atol=1.0e-12,
    )


@pytest.mark.parametrize(
    "cfg_kwargs",
    [
        {"eps_pdf": 1.0e-9},
        {"eps_vol": 1.0e-9},
        {"B": 120, "x_min": -5.0, "dx": 0.1},
    ],
)
def test_locked_constants_are_not_fail_closed(cfg_kwargs: dict[str, float]) -> None:
    cfg = EngineConfig(
        T=2,
        A=1,
        tick_size=np.array([0.01], dtype=np.float64),
        mode="sealed",
        **cfg_kwargs,
    )
    ts_ns = np.array([0, 60_000_000_000], dtype=np.int64)
    with pytest.raises(RuntimeError):
        preallocate_state(ts_ns=ts_ns, cfg=cfg, symbols=("A0",))


def test_float64_tensors_and_kernels_stay_float64() -> None:
    state = make_state(T=3, tick_size=0.01, mode="sealed")
    fill_bars(
        state,
        [
            (100.0, 101.0, 99.0, 100.0, 10.0),
            (100.0, 101.0, 99.0, 100.5, 11.0),
            (100.5, 101.5, 100.0, 101.0, 12.0),
        ],
    )
    cfg = Module2Config(
        profile_window_bars=2,
        profile_warmup_bars=1,
        atr_alpha=1.0,
        ret_scale_min_periods=1,
        delta_mad_min_periods=1,
    )
    run_weightiz_profile_engine(state, cfg)

    assert state.x_grid.dtype == np.float64
    assert state.vp.dtype == np.float64
    assert state.vp_delta.dtype == np.float64
    assert state.profile_stats.dtype == np.float64
    assert state.scores.dtype == np.float64

    params = build_bar_mixture_params(
        open_a=np.array([100.0], dtype=np.float64),
        high_a=np.array([101.0], dtype=np.float64),
        low_a=np.array([99.0], dtype=np.float64),
        close_a=np.array([100.5], dtype=np.float64),
        atr_eff_a=np.array([1.0], dtype=np.float64),
        rvol_a=np.array([1.0], dtype=np.float64),
        clv_a=np.array([0.0], dtype=np.float64),
        body_pct_a=np.array([0.25], dtype=np.float64),
        sigma1_a=np.array([0.1], dtype=np.float64),
        sigma2_a=np.array([0.2], dtype=np.float64),
        w1_a=np.array([0.25], dtype=np.float64),
        w2_a=np.array([0.75], dtype=np.float64),
        volume_a=np.array([10.0], dtype=np.float64),
        cap_v_eff_a=np.array([10.0], dtype=np.float64),
        score_inputs=ScoreInputs(
            ret_norm=np.array([0.0], dtype=np.float64),
            s_r=np.array([0.05], dtype=np.float64),
            clv=np.array([0.0], dtype=np.float64),
            body_pct=np.array([0.25], dtype=np.float64),
        ),
        eps_div_a=np.array([0.01], dtype=np.float64),
        eps_pdf=1.0e-12,
        dx=0.05,
        sealed_mode=True,
        mu1_clv_shift=0.0,
        mu2_clv_shift=0.35,
    )
    inj = inject_profile_mass(
        params=params,
        x_grid=np.asarray(state.x_grid, dtype=np.float64),
        dx=float(state.cfg.dx),
        eps_pdf=float(state.eps.eps_pdf),
        valid_a=np.array([True]),
    )
    assert inj.total_an.dtype == np.float64
    assert inj.delta_an.dtype == np.float64
    assert inj.m0_a.dtype == np.float64
    assert inj.m1_a.dtype == np.float64
    assert inj.m2_a.dtype == np.float64


def test_true_range_and_atr_floor_numeric() -> None:
    state = make_state(T=3, tick_size=1.0, mode="sealed")
    fill_bars(
        state,
        [
            (100.0, 101.0, 99.0, 100.0, 10.0),
            (100.0, 105.0, 98.0, 104.0, 10.0),
            (104.0, 106.0, 103.0, 105.0, 10.0),
        ],
    )
    phys = precompute_market_physics(
        state,
        Module2Config(
            profile_window_bars=2,
            profile_warmup_bars=1,
            atr_alpha=1.0,
            ret_scale_min_periods=1,
            delta_mad_min_periods=1,
        ),
    )

    expected_tr = np.array([2.0, 7.0, 3.0], dtype=np.float64)
    np.testing.assert_allclose(phys.atr_raw[:, 0], expected_tr, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        phys.atr_floor[:, 0],
        np.array([4.0, 7.0, 4.0], dtype=np.float64),
        rtol=0.0,
        atol=1.0e-12,
    )

    phys_half = precompute_market_physics(
        state,
        Module2Config(
            profile_window_bars=2,
            profile_warmup_bars=1,
            atr_alpha=0.5,
            ret_scale_min_periods=1,
            delta_mad_min_periods=1,
        ),
    )
    np.testing.assert_allclose(
        phys_half.atr_raw[:, 0],
        np.array([2.0, 4.5, 3.75], dtype=np.float64),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_rvol_baseline_excludes_today_numeric() -> None:
    T = 21
    clock = make_clock_override(
        minute_of_day=[570] * T,
        tod=[0] * T,
        session_id=list(range(T)),
        gap_min=[0.0] + [24.0 * 60.0] * (T - 1),
        reset_flag=[1] * T,
        phase=[1] * T,
    )
    state = make_state(T=T, tick_size=0.01, mode="sealed", clock_override=clock)
    fill_bars(
        state,
        [(100.0, 101.0, 99.0, 100.0, float(v)) for v in range(1, T + 1)],
    )
    phys = precompute_market_physics(
        state,
        Module2Config(
            profile_window_bars=2,
            profile_warmup_bars=1,
            atr_alpha=1.0,
            ret_scale_min_periods=1,
            delta_mad_min_periods=1,
        ),
    )
    assert phys.rvol[20, 0] == pytest.approx(2.0, abs=1.0e-12)


def test_current_bar_sealed_sigma_and_weights_match_spec() -> None:
    state = make_state(T=2, tick_size=0.01, mode="sealed")
    fill_bars(
        state,
        [
            (1.5, 2.0, 1.0, 1.5, 10.0),
            (1.5, 2.0, 1.0, 1.5, 10.0),
        ],
    )
    phys = precompute_market_physics(
        state,
        Module2Config(
            profile_window_bars=2,
            profile_warmup_bars=1,
            atr_alpha=1.0,
            ret_scale_min_periods=1,
            delta_mad_min_periods=1,
        ),
    )
    sigma_base = 1.0 / (4.0 * 1.01)
    sigma1 = sigma_base / (1.0 + np.log1p(1.0))
    sigma2 = sigma_base
    assert phys.range_[0, 0] == pytest.approx(1.0, abs=1.0e-12)
    assert phys.body_pct[0, 0] == pytest.approx(0.0, abs=1.0e-12)
    assert phys.sigma1[0, 0] == pytest.approx(sigma1, abs=1.0e-12)
    assert phys.sigma2[0, 0] == pytest.approx(sigma2, abs=1.0e-12)
    assert phys.w1[0, 0] == pytest.approx(0.0, abs=1.0e-12)
    assert phys.w2[0, 0] == pytest.approx(1.0, abs=1.0e-12)


def test_value_area_and_poc_tie_breaks_match_spec() -> None:
    x = np.asarray([-0.10, -0.05, 0.0, 0.05, 0.10], dtype=np.float64)
    vp = np.asarray([[0.0, 1.0, 0.5, 1.0, 0.0]], dtype=np.float64)
    rank = _build_poc_rank(x)
    max_mass = np.max(vp, axis=1, keepdims=True)
    ipoc = np.argmin(np.where(vp == max_mass, rank[None, :], x.shape[0] + 1), axis=1)
    assert int(ipoc[0]) == 1

    vp_va = np.asarray([[0.0, 0.20, 0.60, 0.20, 0.0]], dtype=np.float64)
    ipoc_va = np.asarray([2], dtype=np.int64)
    _, ivah, ival = compute_value_area_greedy(
        vp_ab=vp_va,
        ipoc_a=ipoc_va,
        x_grid=x,
        va_threshold=0.70,
        eps_vol=1.0e-12,
    )
    assert int(ival[0]) == 1
    assert int(ivah[0]) == 2


def test_streaming_backend_does_not_reproject_window_to_current_anchor() -> None:
    params = build_bar_mixture_params(
        open_a=np.array([100.0], dtype=np.float64),
        high_a=np.array([100.0], dtype=np.float64),
        low_a=np.array([100.0], dtype=np.float64),
        close_a=np.array([100.0], dtype=np.float64),
        atr_eff_a=np.array([10.0], dtype=np.float64),
        rvol_a=np.array([1.0], dtype=np.float64),
        clv_a=np.array([0.0], dtype=np.float64),
        body_pct_a=np.array([0.0], dtype=np.float64),
        sigma1_a=np.array([0.1], dtype=np.float64),
        sigma2_a=np.array([0.1], dtype=np.float64),
        w1_a=np.array([1.0], dtype=np.float64),
        w2_a=np.array([0.0], dtype=np.float64),
        volume_a=np.array([10.0], dtype=np.float64),
        cap_v_eff_a=np.array([10.0], dtype=np.float64),
        score_inputs=ScoreInputs(
            ret_norm=np.array([0.0], dtype=np.float64),
            s_r=np.array([0.05], dtype=np.float64),
            clv=np.array([0.0], dtype=np.float64),
            body_pct=np.array([0.0], dtype=np.float64),
        ),
        eps_div_a=np.array([1.0], dtype=np.float64),
        eps_pdf=1.0e-12,
        dx=0.05,
        sealed_mode=True,
        mu1_clv_shift=0.0,
        mu2_clv_shift=0.35,
        anchor_close_a=np.array([110.0], dtype=np.float64),
        anchor_atr_eff_a=np.array([10.0], dtype=np.float64),
        anchor_rvol_a=np.array([1.0], dtype=np.float64),
    )
    expected_mu = (100.0 - 110.0) / (10.0 + 1.0)
    assert params.mu1[0] == pytest.approx(expected_mu, abs=1.0e-12)


def test_cap_is_not_recomputed_on_working_window() -> None:
    clock = make_clock_override(minute_of_day=[570, 571, 572], tod=[0, 1, 2], phase=[1, 1, 1])
    state = make_state(T=3, tick_size=1.0, mode="sealed", clock_override=clock)
    fill_bars(
        state,
        [
            (100.0, 100.0, 100.0, 100.0, 1.0),
            (100.0, 100.0, 100.0, 100.0, 1000.0),
            (100.0, 100.0, 100.0, 100.0, 1.0),
        ],
    )
    cfg = Module2Config(
        profile_window_bars=3,
        profile_warmup_bars=1,
        atr_alpha=1.0,
        ret_scale_min_periods=1,
        delta_mad_min_periods=1,
    )
    run_weightiz_profile_engine(state, cfg)
    expected_total_mass = 3.0
    actual_total_mass = float(np.sum(state.vp[2, 0]))
    assert actual_total_mass == pytest.approx(expected_total_mass, abs=1.0e-12)


def test_cap_window_matches_manual_median_mad() -> None:
    clock = make_clock_override(minute_of_day=[590, 591, 592], phase=[2, 2, 2])
    state = make_state(T=3, tick_size=1.0, mode="sealed", clock_override=clock)
    fill_bars(
        state,
        [
            (100.0, 100.0, 100.0, 100.0, 1.0),
            (100.0, 100.0, 100.0, 100.0, 3.0),
            (100.0, 100.0, 100.0, 100.0, 100.0),
        ],
    )
    cfg = Module2Config(
        profile_window_bars=3,
        profile_warmup_bars=1,
        atr_alpha=1.0,
        ret_scale_min_periods=1,
        delta_mad_min_periods=1,
    )
    run_weightiz_profile_engine(state, cfg)
    # Window volumes at t=2: [1, 3, 100] => median=3, MAD=2, cap=13, RVOL=1 => cap_eff=13.
    expected_total_mass = 1.0 + 3.0 + 13.0
    actual_total_mass = float(np.sum(state.vp[2, 0]))
    assert actual_total_mass == pytest.approx(expected_total_mass, abs=1.0e-9)


def test_session_reset_does_not_clear_rolling_state() -> None:
    clock = make_clock_override(
        minute_of_day=[570, 571, 570],
        tod=[0, 1, 0],
        session_id=[0, 0, 1],
        gap_min=[0.0, 1.0, 1000.0],
        reset_flag=[1, 0, 1],
        phase=[1, 1, 1],
    )
    state = make_state(T=3, tick_size=1.0, mode="sealed", clock_override=clock)
    fill_bars(
        state,
        [
            (100.0, 100.0, 100.0, 100.0, 10.0),
            (100.0, 100.0, 100.0, 100.0, 10.0),
            (110.0, 110.0, 110.0, 110.0, 10.0),
        ],
    )
    cfg = Module2Config(
        profile_window_bars=2,
        profile_warmup_bars=1,
        atr_alpha=1.0,
        ret_scale_min_periods=1,
        delta_mad_min_periods=1,
    )
    run_weightiz_profile_engine(state, cfg)
    actual_total_mass = float(np.sum(state.vp[2, 0]))
    assert actual_total_mass == pytest.approx(10.0, abs=1.0e-9)


def test_profile_warmup_uses_profile_window_not_tod_15() -> None:
    T = 21
    state = make_state(T=T, tick_size=1.0, mode="sealed")
    fill_bars(state, [(100.0, 100.0, 100.0, 100.0, 1.0)] * T)
    physics = make_fixed_physics(T=T, atr_eff=1.0, rvol=1.0, sigma1=0.1, sigma2=0.1, w1=1.0, w2=0.0)
    cfg = Module2Config(profile_window_bars=60, profile_warmup_bars=60)
    _run_streaming_with_fixed_physics(state, cfg, physics)
    assert int(state.phase[20]) == int(Phase.LIVE)
    assert float(np.sum(state.vp[20, 0])) > 0.0


def test_gap_reset_threshold_is_strict_gt_5() -> None:
    clock = make_clock_override(
        minute_of_day=[590, 591, 592, 593],
        session_id=[0, 0, 0, 0],
        gap_min=[0.0, 5.0, 6.0, 1.0],
        reset_flag=[1, 0, 0, 0],
        phase=[2, 2, 2, 2],
    )
    state = make_state(T=4, tick_size=1.0, mode="sealed", clock_override=clock)
    fill_bars(
        state,
        [
            (100.0, 100.0, 100.0, 100.0, 1.0),
            (100.0, 100.0, 100.0, 100.0, 1.0),
            (100.0, 100.0, 100.0, 100.0, 1.0),
            (100.0, 100.0, 100.0, 100.0, 1.0),
        ],
    )
    cfg = Module2Config(
        profile_window_bars=4,
        profile_warmup_bars=1,
        atr_alpha=1.0,
        ret_scale_min_periods=1,
        delta_mad_min_periods=1,
    )
    run_weightiz_profile_engine(state, cfg)
    mass_t1 = float(np.sum(state.vp[1, 0]))
    mass_t2 = float(np.sum(state.vp[2, 0]))
    mass_t3 = float(np.sum(state.vp[3, 0]))
    assert mass_t1 == pytest.approx(2.0, abs=1.0e-9)
    assert mass_t2 == pytest.approx(1.0, abs=1.0e-9)
    assert mass_t3 == pytest.approx(2.0, abs=1.0e-9)


def test_session_local_Wt_resets_after_boundary() -> None:
    clock = make_clock_override(
        minute_of_day=[590, 591, 590],
        session_id=[0, 0, 1],
        gap_min=[0.0, 1.0, 1.0],
        reset_flag=[1, 0, 0],
        phase=[2, 2, 2],
    )
    state = make_state(T=3, tick_size=1.0, mode="sealed", clock_override=clock)
    fill_bars(
        state,
        [
            (100.0, 100.0, 100.0, 100.0, 10.0),
            (100.0, 100.0, 100.0, 100.0, 10.0),
            (110.0, 110.0, 110.0, 110.0, 10.0),
        ],
    )
    cfg = Module2Config(
        profile_window_bars=2,
        profile_warmup_bars=1,
        atr_alpha=1.0,
        ret_scale_min_periods=1,
        delta_mad_min_periods=1,
    )
    run_weightiz_profile_engine(state, cfg)
    actual_total_mass = float(np.sum(state.vp[2, 0]))
    assert actual_total_mass == pytest.approx(10.0, abs=1.0e-9)


def test_warmup_updates_state_outputs_neutral() -> None:
    clock = make_clock_override(
        minute_of_day=[570, 571, 585],
        session_id=[0, 0, 0],
        gap_min=[0.0, 1.0, 1.0],
        reset_flag=[1, 0, 0],
        phase=[0, 0, 1],
    )
    state = make_state(T=3, tick_size=1.0, mode="sealed", clock_override=clock)
    fill_bars(
        state,
        [
            (100.0, 100.0, 100.0, 100.0, 1.0),
            (100.0, 100.0, 100.0, 100.0, 1.0),
            (100.0, 100.0, 100.0, 100.0, 1.0),
        ],
    )
    physics = make_fixed_physics(T=3, atr_eff=1.0, rvol=1.0, sigma1=0.1, sigma2=0.1, w1=1.0, w2=0.0)
    cfg = Module2Config(profile_window_bars=3, profile_warmup_bars=60, delta_mad_min_periods=1, ret_scale_min_periods=1)
    _run_streaming_with_fixed_physics(state, cfg, physics)
    assert float(np.sum(state.vp[0, 0])) == pytest.approx(1.0, abs=1.0e-9)
    assert float(np.sum(state.vp[1, 0])) == pytest.approx(2.0, abs=1.0e-9)
    assert float(np.sum(state.vp[2, 0])) == pytest.approx(3.0, abs=1.0e-9)
    np.testing.assert_allclose(state.scores[:2], 0.0, rtol=0.0, atol=0.0)


def test_body_pct_uses_extra_eps_div_in_denominator() -> None:
    state = make_state(T=2, tick_size=0.01, mode="sealed")
    fill_bars(
        state,
        [
            (1.0, 2.0, 1.0, 1.5, 10.0),
            (1.0, 2.0, 1.0, 1.5, 10.0),
        ],
    )
    phys = precompute_market_physics(
        state,
        Module2Config(
            profile_window_bars=2,
            profile_warmup_bars=1,
            atr_alpha=1.0,
            ret_scale_min_periods=1,
            delta_mad_min_periods=1,
        ),
    )
    assert phys.body_pct[0, 0] == pytest.approx(0.5, abs=1.0e-12)


def test_wick_ratio_is_not_implemented_on_executed_path() -> None:
    state = make_state(T=2, tick_size=0.01, mode="sealed")
    fill_bars(
        state,
        [
            (1.0, 2.0, 1.0, 1.5, 10.0),
            (1.0, 2.0, 1.0, 1.5, 10.0),
        ],
    )
    phys = precompute_market_physics(
        state,
        Module2Config(
            profile_window_bars=2,
            profile_warmup_bars=1,
            atr_alpha=1.0,
            ret_scale_min_periods=1,
            delta_mad_min_periods=1,
        ),
    )
    assert hasattr(phys, "wick_ratio")


def test_wick_ratio_formula_numeric() -> None:
    state = make_state(T=2, tick_size=0.01, mode="sealed")
    fill_bars(
        state,
        [
            (1.0, 2.0, 0.5, 1.5, 10.0),
            (1.0, 2.0, 0.5, 1.5, 10.0),
        ],
    )
    phys = precompute_market_physics(
        state,
        Module2Config(
            profile_window_bars=2,
            profile_warmup_bars=1,
            atr_alpha=1.0,
            ret_scale_min_periods=1,
            delta_mad_min_periods=1,
        ),
    )
    expected = (1.5 - 0.5) / 1.5
    assert phys.wick_ratio[0, 0] == pytest.approx(expected, abs=1.0e-12)


def test_reprojection_matches_closed_fixture() -> None:
    clock = make_clock_override(minute_of_day=[590, 591], phase=[2, 2])
    state = make_state(T=2, tick_size=1.0, mode="sealed", clock_override=clock)
    bars = [
        (100.0, 102.0, 98.0, 101.0, 10.0),
        (109.0, 111.0, 109.0, 110.0, 5.0),
    ]
    fill_bars(state, bars)
    physics = make_fixed_physics(T=2, atr_eff=10.0, rvol=1.0, sigma1=0.1, sigma2=0.1, w1=1.0, w2=0.0, cap_v_eff=1.0e9)
    cfg = Module2Config(profile_window_bars=2, profile_warmup_bars=1, delta_mad_min_periods=1, ret_scale_min_periods=1)
    _run_streaming_with_fixed_physics(state, cfg, physics)

    x_grid = np.asarray(state.x_grid, dtype=np.float64)
    expected_k0 = paper_bar_profile(
        open_px=bars[0][0],
        high_px=bars[0][1],
        low_px=bars[0][2],
        close_px=bars[0][3],
        volume=bars[0][4],
        current_close=bars[1][3],
        atr_floor_t=10.0,
        rvol_t=1.0,
        cap_eff_t=1.0e9,
        tick_size=1.0,
        x_grid=x_grid,
        dx=state.cfg.dx,
    )
    expected_k1 = paper_bar_profile(
        open_px=bars[1][0],
        high_px=bars[1][1],
        low_px=bars[1][2],
        close_px=bars[1][3],
        volume=bars[1][4],
        current_close=bars[1][3],
        atr_floor_t=10.0,
        rvol_t=1.0,
        cap_eff_t=1.0e9,
        tick_size=1.0,
        x_grid=x_grid,
        dx=state.cfg.dx,
    )
    expected = expected_k0 + expected_k1
    actual = np.asarray(state.vp[1, 0], dtype=np.float64)
    actual_norm = actual / (np.sum(actual) + 1.0e-12)
    expected_norm = expected / (np.sum(expected) + 1.0e-12)
    np.testing.assert_allclose(actual_norm, expected_norm, rtol=0.0, atol=1.0e-10)


def test_kernel_row_normalization_uses_dx_and_breaks_discrete_mass() -> None:
    x_grid = -6.0 + 0.05 * np.arange(240, dtype=np.float64)
    params = MixtureBarParams(
        mu1=np.array([0.0], dtype=np.float64),
        mu2=np.array([0.0], dtype=np.float64),
        sigma1=np.array([0.2], dtype=np.float64),
        sigma2=np.array([0.2], dtype=np.float64),
        w1=np.array([1.0], dtype=np.float64),
        w2=np.array([0.0], dtype=np.float64),
        vprof=np.array([10.0], dtype=np.float64),
        pbuy=np.array([0.5], dtype=np.float64),
        delta_coeff=np.array([0.0], dtype=np.float64),
    )
    inj = inject_profile_mass(
        params=params,
        x_grid=x_grid,
        dx=0.05,
        eps_pdf=1.0e-12,
        valid_a=np.array([True]),
    )
    assert float(np.sum(inj.total_an[0])) == pytest.approx(10.0, abs=1.0e-9)


def test_per_row_mass_equals_vprof() -> None:
    x_grid = -6.0 + 0.05 * np.arange(240, dtype=np.float64)
    params = MixtureBarParams(
        mu1=np.array([0.0, 0.1], dtype=np.float64),
        mu2=np.array([0.0, 0.1], dtype=np.float64),
        sigma1=np.array([0.2, 0.2], dtype=np.float64),
        sigma2=np.array([0.2, 0.2], dtype=np.float64),
        w1=np.array([1.0, 1.0], dtype=np.float64),
        w2=np.array([0.0, 0.0], dtype=np.float64),
        vprof=np.array([3.0, 7.0], dtype=np.float64),
        pbuy=np.array([0.5, 0.5], dtype=np.float64),
        delta_coeff=np.array([0.0, 0.0], dtype=np.float64),
    )
    inj = inject_profile_mass(
        params=params,
        x_grid=x_grid,
        dx=0.05,
        eps_pdf=1.0e-12,
        valid_a=np.array([True, True]),
    )
    np.testing.assert_allclose(
        np.sum(inj.total_an, axis=1),
        np.array([3.0, 7.0], dtype=np.float64),
        rtol=0.0,
        atol=1.0e-9,
    )


def test_vprof_is_scaled_by_rvol_in_executed_kernel() -> None:
    params = build_bar_mixture_params(
        open_a=np.array([100.0], dtype=np.float64),
        high_a=np.array([101.0], dtype=np.float64),
        low_a=np.array([99.0], dtype=np.float64),
        close_a=np.array([100.0], dtype=np.float64),
        atr_eff_a=np.array([1.0], dtype=np.float64),
        rvol_a=np.array([3.0], dtype=np.float64),
        clv_a=np.array([0.0], dtype=np.float64),
        body_pct_a=np.array([0.5], dtype=np.float64),
        sigma1_a=np.array([0.1], dtype=np.float64),
        sigma2_a=np.array([0.2], dtype=np.float64),
        w1_a=np.array([0.5], dtype=np.float64),
        w2_a=np.array([0.5], dtype=np.float64),
        volume_a=np.array([10.0], dtype=np.float64),
        cap_v_eff_a=np.array([8.0], dtype=np.float64),
        score_inputs=ScoreInputs(
            ret_norm=np.array([0.0], dtype=np.float64),
            s_r=np.array([0.05], dtype=np.float64),
            clv=np.array([0.0], dtype=np.float64),
            body_pct=np.array([0.5], dtype=np.float64),
        ),
        eps_div_a=np.array([0.01], dtype=np.float64),
        eps_pdf=1.0e-12,
        dx=0.05,
        sealed_mode=True,
        mu1_clv_shift=0.0,
        mu2_clv_shift=0.35,
    )
    assert params.vprof[0] == pytest.approx(8.0, abs=1.0e-12)


def test_vprof_equals_min_volume_cap() -> None:
    params = build_bar_mixture_params(
        open_a=np.array([100.0, 100.0], dtype=np.float64),
        high_a=np.array([101.0, 101.0], dtype=np.float64),
        low_a=np.array([99.0, 99.0], dtype=np.float64),
        close_a=np.array([100.0, 100.0], dtype=np.float64),
        atr_eff_a=np.array([1.0, 1.0], dtype=np.float64),
        rvol_a=np.array([3.0, 3.0], dtype=np.float64),
        clv_a=np.array([0.0, 0.0], dtype=np.float64),
        body_pct_a=np.array([0.5, 0.5], dtype=np.float64),
        sigma1_a=np.array([0.1, 0.1], dtype=np.float64),
        sigma2_a=np.array([0.2, 0.2], dtype=np.float64),
        w1_a=np.array([0.5, 0.5], dtype=np.float64),
        w2_a=np.array([0.5, 0.5], dtype=np.float64),
        volume_a=np.array([10.0, 3.0], dtype=np.float64),
        cap_v_eff_a=np.array([8.0, 8.0], dtype=np.float64),
        score_inputs=ScoreInputs(
            ret_norm=np.array([0.0, 0.0], dtype=np.float64),
            s_r=np.array([0.05, 0.05], dtype=np.float64),
            clv=np.array([0.0, 0.0], dtype=np.float64),
            body_pct=np.array([0.5, 0.5], dtype=np.float64),
        ),
        eps_div_a=np.array([0.01, 0.01], dtype=np.float64),
        eps_pdf=1.0e-12,
        dx=0.05,
        sealed_mode=True,
        mu1_clv_shift=0.0,
        mu2_clv_shift=0.35,
    )
    np.testing.assert_allclose(params.vprof, np.array([8.0, 3.0], dtype=np.float64), rtol=0.0, atol=1.0e-12)


def test_profile_moments_are_not_derived_from_discrete_grid() -> None:
    state = make_state(T=3, tick_size=1.0, mode="sealed")
    fill_bars(
        state,
        [
            (100.0, 103.0, 97.0, 102.0, 10.0),
            (102.0, 104.0, 100.0, 103.0, 9.0),
            (103.0, 105.0, 101.0, 104.0, 8.0),
        ],
    )
    cfg = Module2Config(
        profile_window_bars=3,
        profile_warmup_bars=1,
        atr_alpha=1.0,
        ret_scale_min_periods=1,
        delta_mad_min_periods=1,
    )
    run_weightiz_profile_engine(state, cfg)
    x_grid = np.asarray(state.x_grid, dtype=np.float64)
    mu_manual, sigma_manual = paper_profile_stats_from_vp(state.vp[2, 0], x_grid, 1.0e-12)
    mu_state = float(state.profile_stats[2, 0, int(ProfileStatIdx.MU_PROF)])
    sigma_state = float(state.profile_stats[2, 0, int(ProfileStatIdx.SIGMA_PROF)])
    assert mu_state == pytest.approx(mu_manual, abs=1.0e-12)
    assert sigma_state == pytest.approx(sigma_manual, abs=1.0e-12)


def test_hybrid_delta_signed_return_path_differs_from_paper() -> None:
    inputs = ScoreInputs(
        ret_norm=np.array([0.01], dtype=np.float64),
        s_r=np.array([0.01], dtype=np.float64),
        clv=np.array([0.0], dtype=np.float64),
        body_pct=np.array([1.0], dtype=np.float64),
    )
    pbuy, _ = compute_pbuy_and_delta_coeff(
        inputs,
        np.array([0.01], dtype=np.float64),
        dx=0.05,
        eps_pdf=1.0e-12,
        sealed_mode=True,
    )
    expected = 1.0 / (1.0 + np.exp(-(np.log(9.0) * 0.01 / (0.025 + 1.0e-12))))
    assert pbuy[0] == pytest.approx(expected, abs=1.0e-12)


def test_sr_slope_ln9_numeric() -> None:
    inputs = ScoreInputs(
        ret_norm=np.array([0.02], dtype=np.float64),
        s_r=np.array([0.01], dtype=np.float64),
        clv=np.array([0.0], dtype=np.float64),
        body_pct=np.array([1.0], dtype=np.float64),
    )
    pbuy, _ = compute_pbuy_and_delta_coeff(
        inputs,
        np.array([0.01], dtype=np.float64),
        dx=0.05,
        eps_pdf=1.0e-12,
        sealed_mode=True,
    )
    expected = 1.0 / (1.0 + np.exp(-(np.log(9.0) * 0.02 / (0.025 + 1.0e-12))))
    assert pbuy[0] == pytest.approx(expected, abs=1.0e-12)


def test_buy_sell_profile_outputs_are_missing_from_public_state() -> None:
    state = make_state(T=2, tick_size=0.01, mode="sealed")
    fill_bars(
        state,
        [
            (100.0, 101.0, 99.0, 100.0, 10.0),
            (100.0, 101.0, 99.0, 100.5, 11.0),
        ],
    )
    cfg = Module2Config(
        profile_window_bars=2,
        profile_warmup_bars=1,
        atr_alpha=1.0,
        ret_scale_min_periods=1,
        delta_mad_min_periods=1,
    )
    run_weightiz_profile_engine(state, cfg)
    assert hasattr(state, "vp_buy")
    assert hasattr(state, "vp_sell")


def test_vp_buy_plus_vp_sell_equals_vp() -> None:
    state = make_state(T=3, tick_size=0.01, mode="sealed")
    fill_bars(
        state,
        [
            (100.0, 101.0, 99.0, 100.0, 10.0),
            (100.0, 101.0, 99.0, 100.5, 11.0),
            (100.5, 102.0, 100.0, 101.0, 12.0),
        ],
    )
    cfg = Module2Config(
        profile_window_bars=3,
        profile_warmup_bars=1,
        atr_alpha=1.0,
        ret_scale_min_periods=1,
        delta_mad_min_periods=1,
    )
    run_weightiz_profile_engine(state, cfg)
    np.testing.assert_allclose(state.vp_buy + state.vp_sell, state.vp, rtol=0.0, atol=1.0e-12)


def test_delta_equals_buy_minus_sell_per_bin() -> None:
    state = make_state(T=3, tick_size=0.01, mode="sealed")
    fill_bars(
        state,
        [
            (100.0, 101.0, 99.0, 100.0, 10.0),
            (100.0, 101.0, 99.0, 100.5, 11.0),
            (100.5, 102.0, 100.0, 101.0, 12.0),
        ],
    )
    cfg = Module2Config(
        profile_window_bars=3,
        profile_warmup_bars=1,
        atr_alpha=1.0,
        ret_scale_min_periods=1,
        delta_mad_min_periods=1,
    )
    run_weightiz_profile_engine(state, cfg)
    np.testing.assert_allclose(state.vp_buy - state.vp_sell, state.vp_delta, rtol=0.0, atol=1.0e-12)


def test_discrete_mu_sigma_matches_manual() -> None:
    state = make_state(T=3, tick_size=1.0, mode="sealed")
    fill_bars(
        state,
        [
            (100.0, 103.0, 97.0, 102.0, 10.0),
            (102.0, 104.0, 100.0, 103.0, 9.0),
            (103.0, 105.0, 101.0, 104.0, 8.0),
        ],
    )
    cfg = Module2Config(
        profile_window_bars=3,
        profile_warmup_bars=1,
        atr_alpha=1.0,
        ret_scale_min_periods=1,
        delta_mad_min_periods=1,
    )
    run_weightiz_profile_engine(state, cfg)
    x_grid = np.asarray(state.x_grid, dtype=np.float64)
    mu_manual, sigma_manual = paper_profile_stats_from_vp(state.vp[2, 0], x_grid, 1.0e-12)
    assert state.profile_stats[2, 0, int(ProfileStatIdx.MU_PROF)] == pytest.approx(mu_manual, abs=1.0e-12)
    assert state.profile_stats[2, 0, int(ProfileStatIdx.SIGMA_PROF)] == pytest.approx(sigma_manual, abs=1.0e-12)


def test_delta_noise_uses_current_row_instead_of_past_only() -> None:
    arr = np.array([[0.0], [10.0], [10.0]], dtype=np.float64)
    arr_past = np.full_like(arr, np.nan)
    arr_past[1:] = arr[:-1]
    _, mad = _rolling_median_mad_causal(arr_past, window=180, min_periods=1)
    assert mad[2, 0] == pytest.approx(5.0, abs=1.0e-12)


def test_delta_noise_excludes_current_row() -> None:
    arr = np.array([[0.0], [10.0], [10.0]], dtype=np.float64)
    _, mad_current = _rolling_median_mad_causal(arr, window=180, min_periods=1)
    arr_past = np.full_like(arr, np.nan)
    arr_past[1:] = arr[:-1]
    _, mad_past = _rolling_median_mad_causal(arr_past, window=180, min_periods=1)
    assert mad_current[2, 0] == pytest.approx(0.0, abs=1.0e-12)
    assert mad_past[2, 0] == pytest.approx(5.0, abs=1.0e-12)


def test_delta_change_is_zeroed_on_session_reset() -> None:
    clock = make_clock_override(
        minute_of_day=[570, 571, 570],
        tod=[0, 1, 0],
        session_id=[0, 0, 1],
        gap_min=[0.0, 1.0, 1000.0],
        reset_flag=[1, 0, 1],
        phase=[1, 1, 1],
    )
    state = make_state(T=3, tick_size=1.0, mode="sealed", clock_override=clock)
    fill_bars(
        state,
        [
            (100.0, 100.0, 100.0, 100.0, 1.0),
            (100.0, 100.0, 100.0, 100.0, 1.0),
            (100.0, 100.0, 100.0, 100.0, 1.0),
        ],
    )
    physics = make_fixed_physics(T=3, atr_eff=1.0, rvol=1.0, sigma1=0.1, sigma2=0.1, w1=1.0, w2=0.0)
    captured: list[np.ndarray] = []

    def rolling_spy(arr: np.ndarray, window: int, min_periods: int) -> tuple[np.ndarray, np.ndarray]:
        captured.append(np.asarray(arr, dtype=np.float64).copy())
        return _rolling_median_mad_causal(arr, window=window, min_periods=min_periods)

    run_streaming_profile_engine(
        state=state,
        cfg=Module2Config(profile_window_bars=2, profile_warmup_bars=1, delta_mad_min_periods=1, ret_scale_min_periods=1),
        physics=physics,
        mode="sealed",
        open_use=np.asarray(state.open_px, dtype=np.float64),
        high_use=np.asarray(state.high_px, dtype=np.float64),
        low_use=np.asarray(state.low_px, dtype=np.float64),
        close_use=np.asarray(state.close_px, dtype=np.float64),
        vol_use=np.asarray(state.volume, dtype=np.float64),
        valid=np.asarray(state.bar_valid, dtype=bool),
        build_poc_rank_fn=_build_poc_rank,
        compute_value_area_fn=compute_value_area_greedy,
        rolling_median_mad_fn=rolling_spy,
        profile_stat_idx=ProfileStatIdx,
        score_idx=ScoreIdx,
        phase_enum=Phase,
        collect_forensics=False,
    )
    assert any(arr.shape[0] == 1 and arr[0, 0] == pytest.approx(0.0, abs=1.0e-12) for arr in captured)


def test_warmup_mask_policy_leaves_nonfinite_outputs() -> None:
    state = make_state(T=2, tick_size=0.01, mode="sealed")
    fill_bars(
        state,
        [
            (100.0, 101.0, 99.0, 100.0, 10.0),
            (100.0, 101.0, 99.0, 100.5, 11.0),
        ],
    )
    cfg = Module2Config(
        profile_window_bars=2,
        profile_warmup_bars=1,
        atr_alpha=1.0,
        rvol_policy="warmup_mask",
        ret_scale_min_periods=1,
        delta_mad_min_periods=1,
    )
    with pytest.raises(RuntimeError):
        run_weightiz_profile_engine(state, cfg)


def test_sealed_nonfinite_raises() -> None:
    state = make_state(T=1, tick_size=0.01, mode="sealed")
    fill_bars(
        state,
        [
            (100.0, 101.0, 99.0, 100.0, 10.0),
        ],
    )
    physics = make_fixed_physics(T=1, atr_eff=1.0, rvol=1.0, sigma1=0.1, sigma2=0.1, w1=1.0, w2=0.0)

    def bad_value_area_fn(*, vp_ab: np.ndarray, ipoc_a: np.ndarray, x_grid: np.ndarray, va_threshold: float, eps_vol: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = int(vp_ab.shape[0])
        bad = np.full(n, np.nan, dtype=np.float64)
        return bad, bad, bad

    with pytest.raises(RuntimeError):
        run_streaming_profile_engine(
            state=state,
            cfg=Module2Config(profile_window_bars=2, profile_warmup_bars=1, delta_mad_min_periods=1, ret_scale_min_periods=1),
            physics=physics,
            mode="sealed",
            open_use=np.asarray(state.open_px, dtype=np.float64),
            high_use=np.asarray(state.high_px, dtype=np.float64),
            low_use=np.asarray(state.low_px, dtype=np.float64),
            close_use=np.asarray(state.close_px, dtype=np.float64),
            vol_use=np.asarray(state.volume, dtype=np.float64),
            valid=np.asarray(state.bar_valid, dtype=bool),
            build_poc_rank_fn=_build_poc_rank,
            compute_value_area_fn=bad_value_area_fn,
            rolling_median_mad_fn=_rolling_median_mad_causal,
            profile_stat_idx=ProfileStatIdx,
            score_idx=ScoreIdx,
            phase_enum=Phase,
            collect_forensics=False,
        )


def test_source_line_anchors_still_point_at_executed_files() -> None:
    # This keeps the artifact file:line evidence tied to the current repo state.
    for path, pattern in [
        (M2_CORE, "def run_weightiz_profile_engine"),
        (M2_ENGINE, "def run_streaming_profile_engine"),
        (M2_KERNELS, "def build_bar_mixture_params"),
        (M2_TENSOR, "def apply_rolling_update"),
    ]:
        text = Path(path).read_text(encoding="utf-8")
        assert pattern in text
    assert EXECUTED_PATH
