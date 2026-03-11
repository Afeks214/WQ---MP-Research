import unittest

import numpy as np

from weightiz.module1.core import EngineConfig, Phase, ProfileStatIdx, preallocate_state
from weightiz.module2.core import (
    Module2Config,
    _build_poc_rank,
    _rolling_median_mad_causal,
    compute_value_area_greedy,
    precompute_market_physics,
    run_weightiz_profile_engine,
)


def _state_from_minute_series(T: int, A: int = 3, mode: str = "sealed"):
    start_ns = np.datetime64("2025-01-06T14:30:00", "ns").astype(np.int64)
    ts_ns = start_ns + np.arange(T, dtype=np.int64) * np.int64(60_000_000_000)
    cfg = EngineConfig(T=T, A=A, B=240, tick_size=np.full(A, 0.01, dtype=np.float64), mode=mode)
    return preallocate_state(ts_ns=ts_ns, cfg=cfg, symbols=tuple(f"A{i}" for i in range(A)))


def _state_with_gap(T: int, gap_at: int, A: int = 1, mode: str = "sealed"):
    start_ns = np.datetime64("2025-01-06T14:30:00", "ns").astype(np.int64)
    ts_ns = start_ns + np.arange(T, dtype=np.int64) * np.int64(60_000_000_000)
    ts_ns[gap_at:] += np.int64(6 * 60_000_000_000)
    cfg = EngineConfig(T=T, A=A, B=240, tick_size=np.full(A, 0.01, dtype=np.float64), mode=mode)
    return preallocate_state(ts_ns=ts_ns, cfg=cfg, symbols=tuple(f"A{i}" for i in range(A)))


def _fill_ohlcv_valid(st, seed: int = 7):
    T, A = st.cfg.T, st.cfg.A
    rng = np.random.default_rng(seed)
    base = np.linspace(100.0, 100.0 + (A - 1), A, dtype=np.float64)

    close = np.zeros((T, A), dtype=np.float64)
    close[0] = base
    for t in range(1, T):
        close[t] = np.maximum(0.01, close[t - 1] * (1.0 + 0.0006 * rng.standard_normal(A)))

    open_px = close * (1.0 + 0.0002 * rng.standard_normal((T, A)))
    high_px = np.maximum(open_px, close) * (1.0 + 0.0007 * np.abs(rng.standard_normal((T, A))))
    low_px = np.minimum(open_px, close) * (1.0 - 0.0007 * np.abs(rng.standard_normal((T, A))))
    volume = np.maximum(100.0, 1e6 * (1.0 + 0.08 * rng.standard_normal((T, A))))

    st.open_px[:, :] = open_px
    st.high_px[:, :] = high_px
    st.low_px[:, :] = low_px
    st.close_px[:, :] = close
    st.volume[:, :] = volume
    st.bar_valid[:, :] = True


class TestModule2Institutional(unittest.TestCase):
    def test_atr_floor_locked_formula_in_sealed_mode(self):
        st = _state_from_minute_series(T=40, A=2, mode="sealed")

        close = np.full((st.cfg.T, st.cfg.A), 100.0, dtype=np.float64)
        st.open_px[:, :] = close
        st.high_px[:, :] = close
        st.low_px[:, :] = close
        st.close_px[:, :] = close
        st.volume[:, :] = 1_000_000.0
        st.bar_valid[:, :] = True

        cfg = Module2Config(profile_window_bars=20, profile_warmup_bars=20, rvol_lookback_sessions=20)
        run_weightiz_profile_engine(st, cfg)

        expected_floor = np.maximum(4.0 * np.asarray(st.eps.eps_div, dtype=np.float64), 0.0002 * close)
        np.testing.assert_allclose(st.atr_floor, expected_floor, rtol=0.0, atol=1e-12)

    def test_prefix_invariance_includes_physics_outputs(self):
        st1 = _state_from_minute_series(T=280, A=3, mode="sealed")
        st2 = _state_from_minute_series(T=280, A=3, mode="sealed")
        _fill_ohlcv_valid(st1, seed=901)
        _fill_ohlcv_valid(st2, seed=901)

        t0 = 180
        st2.open_px[t0 + 1 :] *= 1.03
        st2.high_px[t0 + 1 :] *= 1.05
        st2.low_px[t0 + 1 :] *= 0.97
        st2.close_px[t0 + 1 :] *= 1.04
        st2.volume[t0 + 1 :] *= 1.45

        cfg = Module2Config(
            profile_window_bars=60,
            profile_warmup_bars=60,
            ret_scale_min_periods=1,
            delta_mad_min_periods=1,
            fail_on_non_finite_output=True,
        )
        run_weightiz_profile_engine(st1, cfg)
        run_weightiz_profile_engine(st2, cfg)

        np.testing.assert_array_equal(st1.rvol[: t0 + 1], st2.rvol[: t0 + 1])
        np.testing.assert_array_equal(st1.atr_floor[: t0 + 1], st2.atr_floor[: t0 + 1])
        np.testing.assert_array_equal(st1.vp[: t0 + 1], st2.vp[: t0 + 1])
        np.testing.assert_array_equal(st1.vp_delta[: t0 + 1], st2.vp_delta[: t0 + 1])
        np.testing.assert_array_equal(st1.profile_stats[: t0 + 1], st2.profile_stats[: t0 + 1])
        np.testing.assert_array_equal(st1.scores[: t0 + 1], st2.scores[: t0 + 1])

    def test_invalid_bars_do_not_poison_outputs(self):
        st = _state_from_minute_series(T=170, A=3, mode="sealed")
        _fill_ohlcv_valid(st, seed=77)

        invalid = np.zeros((st.cfg.T, st.cfg.A), dtype=bool)
        invalid[40:45, 1] = True
        invalid[90, 2] = True
        invalid[130:133, 0] = True
        st.bar_valid[invalid] = False

        for arr in (st.open_px, st.high_px, st.low_px, st.close_px, st.volume):
            arr[invalid] = np.nan

        cfg = Module2Config(
            profile_window_bars=30,
            profile_warmup_bars=30,
            ret_scale_min_periods=1,
            delta_mad_min_periods=1,
            fail_on_non_finite_output=True,
        )
        run_weightiz_profile_engine(st, cfg)

        self.assertTrue(np.all(np.isfinite(st.vp)))
        self.assertTrue(np.all(np.isfinite(st.vp_delta)))

        idx_zero = int(np.argmin(np.abs(st.x_grid)))
        self.assertTrue(np.all(st.vp[invalid] == 0.0))
        self.assertTrue(np.all(st.vp_delta[invalid] == 0.0))
        self.assertTrue(np.all(st.scores[invalid] == 0.0))

        ps_invalid = st.profile_stats[invalid]
        expected = np.zeros((int(ProfileStatIdx.N_FIELDS),), dtype=np.float64)
        expected[int(ProfileStatIdx.IPOC)] = float(idx_zero)
        expected[int(ProfileStatIdx.IVAH)] = float(idx_zero)
        expected[int(ProfileStatIdx.IVAL)] = float(idx_zero)
        self.assertTrue(np.all(ps_invalid == expected[None, :]))

    def test_reset_flag_resets_delta_diff(self):
        t_reset = 45
        st = _state_with_gap(T=95, gap_at=t_reset, A=1, mode="sealed")

        close = np.zeros((st.cfg.T, 1), dtype=np.float64)
        close[0, 0] = 100.0
        for t in range(1, st.cfg.T):
            if t < t_reset:
                close[t, 0] = close[t - 1, 0] * (1.0 + 0.002 * ((-1) ** t))
            else:
                close[t, 0] = close[t - 1, 0] * 1.006

        st.open_px[:, :] = close * 1.0001
        st.high_px[:, :] = np.maximum(st.open_px, close) * 1.001
        st.low_px[:, :] = np.minimum(st.open_px, close) * 0.999
        st.close_px[:, :] = close
        st.volume[:, :] = 1_000_000.0
        st.bar_valid[:, :] = True

        cfg = Module2Config(
            profile_window_bars=15,
            profile_warmup_bars=15,
            ret_scale_min_periods=1,
            delta_mad_lookback_bars=5,
            delta_mad_min_periods=1,
            sigma_delta_min=0.05,
            fail_on_non_finite_output=True,
        )
        run_weightiz_profile_engine(st, cfg)

        self.assertEqual(int(st.reset_flag[t_reset]), 1)

        T = st.cfg.T
        A = st.cfg.A
        W = int(cfg.profile_window_bars)
        computed = (np.arange(T, dtype=np.int64)[:, None] >= (W - 1)) & st.bar_valid

        delta_eff = st.profile_stats[:, :, int(ProfileStatIdx.DELTA_EFF)]
        delta_eff_all = np.where(computed, delta_eff, np.nan)

        d_delta = np.full((T, A), np.nan, dtype=np.float64)
        for t in range(T):
            mask_t = computed[t]
            if not np.any(mask_t):
                continue
            if t == 0 or st.reset_flag[t] == 1 or st.session_id[t] != st.session_id[t - 1]:
                d_delta[t, mask_t] = 0.0
            else:
                prev = delta_eff_all[t - 1]
                curr = delta_eff_all[t]
                ok = mask_t & np.isfinite(prev) & np.isfinite(curr)
                d_delta[t, ok] = curr[ok] - prev[ok]
                d_delta[t, mask_t & ~ok] = 0.0

        self.assertEqual(float(d_delta[t_reset, 0]), 0.0)

        sigma_delta = np.full((T, A), np.nan, dtype=np.float64)
        sid = st.session_id
        starts = np.where(np.r_[True, (sid[1:] != sid[:-1]) | (st.reset_flag[1:] == 1)])[0]
        ends = np.r_[starts[1:], T]
        for s, e in zip(starts.tolist(), ends.tolist()):
            seg_level = delta_eff_all[s:e]
            seg_chg = d_delta[s:e]
            _, mad_level = _rolling_median_mad_causal(
                seg_level,
                window=int(cfg.delta_mad_lookback_bars),
                min_periods=int(cfg.delta_mad_min_periods),
            )
            _, mad_chg = _rolling_median_mad_causal(
                seg_chg,
                window=int(cfg.delta_mad_lookback_bars),
                min_periods=int(cfg.delta_mad_min_periods),
            )
            sig_seg = np.maximum(
                np.maximum(
                    1.4826 * np.where(np.isfinite(mad_level), mad_level, 0.0),
                    1.4826 * np.where(np.isfinite(mad_chg), mad_chg, 0.0),
                ),
                float(cfg.sigma_delta_min),
            )
            sigma_delta[s:e] = sig_seg

        valid_post = computed & np.isfinite(delta_eff_all) & np.isfinite(sigma_delta)
        z_expected = np.divide(
            delta_eff_all,
            sigma_delta + st.eps.eps_pdf,
            out=np.full((T, A), np.nan, dtype=np.float64),
            where=valid_post,
        )

        z_engine = st.profile_stats[:, :, int(ProfileStatIdx.Z_DELTA)]
        self.assertAlmostEqual(float(z_engine[t_reset, 0]), float(z_expected[t_reset, 0]), places=12)

    def test_poc_tie_break_min_abs_then_left(self):
        x = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
        rank = _build_poc_rank(x)

        vp_1 = np.asarray([[1.0, 4.0, 4.0, 4.0, 1.0]], dtype=np.float64)
        max_1 = np.max(vp_1, axis=1, keepdims=True)
        is_max_1 = np.isclose(vp_1, max_1, atol=0.0, rtol=0.0)
        ipoc_1 = np.argmin(np.where(is_max_1, rank[None, :], vp_1.shape[1] + 1), axis=1)
        self.assertEqual(int(ipoc_1[0]), 2)

        vp_2 = np.asarray([[1.0, 4.0, 1.0, 4.0, 1.0]], dtype=np.float64)
        max_2 = np.max(vp_2, axis=1, keepdims=True)
        is_max_2 = np.isclose(vp_2, max_2, atol=0.0, rtol=0.0)
        ipoc_2 = np.argmin(np.where(is_max_2, rank[None, :], vp_2.shape[1] + 1), axis=1)
        self.assertEqual(int(ipoc_2[0]), 1)

    def test_value_area_greedy_tie_break(self):
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
        self.assertEqual(int(ival[0]), 1)
        self.assertEqual(int(ivah[0]), 2)

    def test_sealed_sigma_floor_dx(self):
        st = _state_from_minute_series(T=120, A=3, mode="sealed")
        _fill_ohlcv_valid(st, seed=55)

        cfg = Module2Config(
            profile_window_bars=20,
            profile_warmup_bars=20,
            rvol_lookback_sessions=20,
        )
        phys = precompute_market_physics(st, cfg)
        dx = float(st.cfg.dx)

        self.assertTrue(np.all(np.isfinite(phys.sigma1)))
        self.assertTrue(np.all(np.isfinite(phys.sigma2)))
        self.assertTrue(np.all(phys.sigma1[st.bar_valid] >= dx))
        self.assertTrue(np.all(phys.sigma2[st.bar_valid] >= dx))

    def test_warmup_computes_profiles_scores_zero(self):
        st = _state_from_minute_series(T=100, A=2, mode="sealed")
        _fill_ohlcv_valid(st, seed=120)

        cfg = Module2Config(
            profile_window_bars=5,
            profile_warmup_bars=5,
            ret_scale_min_periods=1,
            delta_mad_min_periods=1,
            fail_on_non_finite_output=True,
        )
        run_weightiz_profile_engine(st, cfg)

        warmup_rows = np.where(st.phase == np.int8(Phase.WARMUP))[0]
        computable_warmup_rows = warmup_rows[warmup_rows >= (cfg.profile_window_bars - 1)]
        self.assertGreater(computable_warmup_rows.size, 0)

        vp_mass = np.sum(st.vp[computable_warmup_rows], axis=2)
        self.assertTrue(np.all(vp_mass > 0.0))

        ps_warmup = st.profile_stats[computable_warmup_rows]
        self.assertTrue(np.all(np.isfinite(ps_warmup)))

        scores_warmup = st.scores[computable_warmup_rows]
        self.assertTrue(np.all(scores_warmup == 0.0))


if __name__ == "__main__":
    unittest.main()
