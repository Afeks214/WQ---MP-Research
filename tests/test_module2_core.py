import unittest
import numpy as np

from weightiz_module1_core import (
    EngineConfig,
    Phase,
    ProfileStatIdx,
    ScoreIdx,
    deterministic_digest_sha256,
    preallocate_state,
)
from weightiz_module2_core import (
    Module2Config,
    _rolling_median_mad_causal,
    compute_value_area_greedy,
    precompute_market_physics,
    run_weightiz_profile_engine,
)


def _state_from_minute_series(T: int, A: int = 4, mode: str = "research"):
    start_ns = np.datetime64("2025-01-06T14:30:00", "ns").astype(np.int64)
    ts_ns = start_ns + np.arange(T, dtype=np.int64) * np.int64(60_000_000_000)
    cfg = EngineConfig(T=T, A=A, B=240, tick_size=np.full(A, 0.01, dtype=np.float64), mode=mode)
    st = preallocate_state(ts_ns=ts_ns, cfg=cfg, symbols=tuple(f"A{i}" for i in range(A)))
    return st


def _state_from_daily_points(S: int, A: int = 4, mode: str = "research"):
    # One point per business day at the same minute-of-day; sessions increase across days.
    day_idx = np.arange(S, dtype=np.int64)
    day0 = np.datetime64("2025-01-02", "D")
    ts_days = day0 + day_idx
    ts_ns = (ts_days.astype("datetime64[ns]") + np.timedelta64(14, "h") + np.timedelta64(30, "m")).astype(np.int64)
    cfg = EngineConfig(T=S, A=A, B=240, tick_size=np.full(A, 0.01, dtype=np.float64), mode=mode)
    st = preallocate_state(ts_ns=ts_ns, cfg=cfg, symbols=tuple(f"A{i}" for i in range(A)))
    return st


def _fill_ohlcv_valid(st, seed: int = 7):
    T, A = st.cfg.T, st.cfg.A
    rng = np.random.default_rng(seed)
    base = np.linspace(100.0, 104.0, A, dtype=np.float64)

    close = np.zeros((T, A), dtype=np.float64)
    close[0] = base
    for t in range(1, T):
        close[t] = np.maximum(0.01, close[t - 1] * (1.0 + 0.0005 * rng.standard_normal(A)))

    open_px = close * (1.0 + 0.0002 * rng.standard_normal((T, A)))
    high_px = np.maximum(open_px, close) * (1.0 + 0.0006 * np.abs(rng.standard_normal((T, A))))
    low_px = np.minimum(open_px, close) * (1.0 - 0.0006 * np.abs(rng.standard_normal((T, A))))
    volume = np.maximum(100.0, 1e6 * (1.0 + 0.05 * rng.standard_normal((T, A))))

    st.open_px[:, :] = open_px
    st.high_px[:, :] = high_px
    st.low_px[:, :] = low_px
    st.close_px[:, :] = close
    st.volume[:, :] = volume
    st.bar_valid[:, :] = True


class TestModule2Core(unittest.TestCase):
    def test_value_area_greedy_differs_from_offset_scan_on_asymmetric_shoulders(self):
        x = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
        # POC at 0 with asymmetric shoulders: greedy should select left first.
        vp = np.asarray([[0.00, 0.39, 0.60, 0.01, 0.00]], dtype=np.float64)
        ipoc = np.asarray([2], dtype=np.int64)
        _, ivah, ival = compute_value_area_greedy(
            vp_ab=vp,
            ipoc_a=ipoc,
            x_grid=x,
            va_threshold=0.70,
            eps_vol=1e-12,
        )
        self.assertEqual(int(ival[0]), 1)
        self.assertEqual(int(ivah[0]), 2)

    def test_value_area_greedy_tie_prefers_left_after_zero_distance_tie(self):
        x = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
        vp = np.asarray([[0.0, 0.20, 0.60, 0.20, 0.0]], dtype=np.float64)
        ipoc = np.asarray([2], dtype=np.int64)
        _, ivah, ival = compute_value_area_greedy(
            vp_ab=vp,
            ipoc_a=ipoc,
            x_grid=x,
            va_threshold=0.70,
            eps_vol=1e-12,
        )
        # Left/right masses are tied and |x| is tied, so deterministic choice is left.
        self.assertEqual(int(ival[0]), 1)
        self.assertEqual(int(ivah[0]), 2)

    def test_rolling_median_mad_matches_naive_for_rectangular_ta(self):
        rng = np.random.default_rng(1)
        arr = rng.normal(0.0, 1.0, size=(40, 4)).astype(np.float64)
        arr[5, 1] = np.nan
        arr[12, 3] = np.nan
        arr[24, 0] = np.nan

        window = 7
        min_periods = 3
        med, mad = _rolling_median_mad_causal(arr, window=window, min_periods=min_periods)

        med_ref = np.full_like(med, np.nan)
        mad_ref = np.full_like(mad, np.nan)
        for t in range(arr.shape[0]):
            lo = max(0, t - window + 1)
            seg = arr[lo : t + 1]
            if seg.shape[0] < min_periods:
                continue
            m = np.nanmedian(seg, axis=0)
            d = np.nanmedian(np.abs(seg - m[None, :]), axis=0)
            med_ref[t] = m
            mad_ref[t] = d

        np.testing.assert_allclose(med, med_ref, atol=1e-12, rtol=0.0, equal_nan=True)
        np.testing.assert_allclose(mad, mad_ref, atol=1e-12, rtol=0.0, equal_nan=True)

    def test_rvol_baseline_window_does_not_crash_for_a_ne_window(self):
        st = _state_from_daily_points(S=35, A=4)
        _fill_ohlcv_valid(st, seed=10)

        cfg = Module2Config(
            profile_window_bars=5,
            profile_warmup_bars=5,
            rvol_lookback_sessions=20,
        )
        phys = precompute_market_physics(st, cfg)

        self.assertEqual(phys.rvol.shape, (st.cfg.T, st.cfg.A))
        self.assertEqual(phys.cap_v_eff.shape, (st.cfg.T, st.cfg.A))
        self.assertTrue(np.all(np.isfinite(phys.rvol)))
        self.assertTrue(np.all(np.isfinite(phys.cap_v_eff)))

    def test_profile_engine_runs_for_a4_w60(self):
        st = _state_from_minute_series(T=220, A=4)
        _fill_ohlcv_valid(st, seed=22)

        cfg = Module2Config(
            profile_window_bars=60,
            profile_warmup_bars=60,
            ret_scale_min_periods=1,
            delta_mad_min_periods=1,
            fail_on_non_finite_output=True,
        )
        run_weightiz_profile_engine(st, cfg)

        t_idx = np.arange(st.cfg.T, dtype=np.int64)[:, None]
        computed_mask = (t_idx >= 59) & (st.phase[:, None] != np.int8(Phase.WARMUP)) & st.bar_valid
        self.assertTrue(np.any(computed_mask))
        req_mask = (
            ((st.phase[:, None] == np.int8(Phase.LIVE)) | (st.phase[:, None] == np.int8(Phase.OVERNIGHT_SELECT)))
            & st.bar_valid
        )
        self.assertTrue(np.any(req_mask))

        req_profile = [
            int(ProfileStatIdx.DCLIP),
            int(ProfileStatIdx.A_AFFINITY),
            int(ProfileStatIdx.Z_DELTA),
            int(ProfileStatIdx.GBREAK),
            int(ProfileStatIdx.GREJECT),
            int(ProfileStatIdx.DELTA_EFF),
            int(ProfileStatIdx.IPOC),
            int(ProfileStatIdx.IVAH),
            int(ProfileStatIdx.IVAL),
        ]
        for ch in req_profile:
            arr = st.profile_stats[:, :, ch]
            self.assertTrue(np.all(np.isfinite(arr[req_mask])))

        req_scores = [
            int(ScoreIdx.SCORE_BO_LONG),
            int(ScoreIdx.SCORE_BO_SHORT),
            int(ScoreIdx.SCORE_REJECT),
        ]
        for ch in req_scores:
            arr = st.scores[:, :, ch]
            self.assertTrue(np.all(np.isfinite(arr[req_mask])))

    def test_sealed_deterministic_digest_replay(self):
        st1 = _state_from_minute_series(T=220, A=4, mode="sealed")
        st2 = _state_from_minute_series(T=220, A=4, mode="sealed")
        _fill_ohlcv_valid(st1, seed=123)
        _fill_ohlcv_valid(st2, seed=123)

        cfg = Module2Config(
            profile_window_bars=60,
            profile_warmup_bars=60,
            ret_scale_min_periods=1,
            delta_mad_min_periods=1,
            fail_on_non_finite_output=True,
        )
        run_weightiz_profile_engine(st1, cfg)
        run_weightiz_profile_engine(st2, cfg)

        self.assertEqual(deterministic_digest_sha256(st1), deterministic_digest_sha256(st2))

    def test_sealed_sigma_never_below_dx(self):
        st = _state_from_daily_points(S=40, A=3, mode="sealed")
        _fill_ohlcv_valid(st, seed=77)
        cfg = Module2Config(profile_window_bars=5, profile_warmup_bars=5, rvol_lookback_sessions=20)
        phys = precompute_market_physics(st, cfg)
        dx = float(st.cfg.dx)
        self.assertTrue(np.all(phys.sigma1[st.bar_valid] >= dx))
        self.assertTrue(np.all(phys.sigma2[st.bar_valid] >= dx))

    def test_sealed_prefix_invariance_no_lookahead(self):
        st1 = _state_from_minute_series(T=260, A=4, mode="sealed")
        st2 = _state_from_minute_series(T=260, A=4, mode="sealed")
        _fill_ohlcv_valid(st1, seed=901)
        _fill_ohlcv_valid(st2, seed=901)

        t0 = 170
        st2.open_px[t0 + 1 :] *= 1.03
        st2.high_px[t0 + 1 :] *= 1.05
        st2.low_px[t0 + 1 :] *= 0.97
        st2.close_px[t0 + 1 :] *= 1.04
        st2.volume[t0 + 1 :] *= 1.50

        cfg = Module2Config(
            profile_window_bars=60,
            profile_warmup_bars=60,
            ret_scale_min_periods=1,
            delta_mad_min_periods=1,
            fail_on_non_finite_output=True,
        )
        run_weightiz_profile_engine(st1, cfg)
        run_weightiz_profile_engine(st2, cfg)

        np.testing.assert_allclose(
            st1.profile_stats[: t0 + 1],
            st2.profile_stats[: t0 + 1],
            atol=0.0,
            rtol=0.0,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            st1.scores[: t0 + 1],
            st2.scores[: t0 + 1],
            atol=0.0,
            rtol=0.0,
            equal_nan=True,
        )


if __name__ == "__main__":
    unittest.main()
