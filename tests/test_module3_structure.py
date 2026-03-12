import unittest
from unittest.mock import patch

import numpy as np

from weightiz.module3.structural_context_builder import build_context_tensor
from weightiz.module1.core import EngineConfig, Phase, ProfileStatIdx, ScoreIdx, preallocate_state
import weightiz.module3.bridge as module3
from weightiz.module3.bridge import (
    ContextIdx,
    Module3Config,
    Struct30mIdx,
    run_module3_structural_aggregation,
)
from weightiz.module3.structural_prefix_sums import build_prefix_count, build_prefix_sum, rolling_mean_from_prefix
from weightiz.module3.structural_window_engine import _rolling_mean_for_series


def _fill_required_channels(state) -> None:
    state.phase[:] = np.int8(Phase.LIVE)
    state.bar_valid[:] = True

    # Fill required channels with finite defaults over all in-scope rows.
    state.profile_stats[:, :, int(ProfileStatIdx.DCLIP)] = 0.1
    state.profile_stats[:, :, int(ProfileStatIdx.A_AFFINITY)] = 0.6
    state.profile_stats[:, :, int(ProfileStatIdx.Z_DELTA)] = 0.2
    state.profile_stats[:, :, int(ProfileStatIdx.GBREAK)] = 0.55
    state.profile_stats[:, :, int(ProfileStatIdx.GREJECT)] = 0.35
    state.profile_stats[:, :, int(ProfileStatIdx.DELTA_EFF)] = 0.05

    # Default valid indices.
    state.profile_stats[:, :, int(ProfileStatIdx.IPOC)] = 120.0
    state.profile_stats[:, :, int(ProfileStatIdx.IVAH)] = 130.0
    state.profile_stats[:, :, int(ProfileStatIdx.IVAL)] = 110.0

    state.scores[:, :, int(ScoreIdx.SCORE_BO_LONG)] = 0.2
    state.scores[:, :, int(ScoreIdx.SCORE_BO_SHORT)] = -0.1
    state.scores[:, :, int(ScoreIdx.SCORE_REJECT)] = 0.05


def _make_state(T: int = 90, A: int = 1):
    start_ns = np.datetime64("2025-01-06T14:30:00", "ns").astype(np.int64)
    ts_ns = start_ns + np.arange(T, dtype=np.int64) * np.int64(60_000_000_000)
    cfg = EngineConfig(T=T, A=A, B=240, tick_size=np.full(A, 0.01, dtype=np.float64))
    symbols = tuple(f"A{i}" for i in range(A))
    state = preallocate_state(ts_ns, cfg, symbols)
    _fill_required_channels(state)
    return state


def _make_state_with_clock_override(minute_of_day: np.ndarray, session_id: np.ndarray, A: int = 1):
    minute_of_day = np.asarray(minute_of_day, dtype=np.int16)
    session_id = np.asarray(session_id, dtype=np.int64)
    T = int(minute_of_day.shape[0])
    if session_id.shape != (T,):
        raise RuntimeError("session_id shape must match minute_of_day shape")

    start_ns = np.datetime64("2025-01-06T14:30:00", "ns").astype(np.int64)
    ts_ns = start_ns + np.arange(T, dtype=np.int64) * np.int64(60_000_000_000)
    cfg = EngineConfig(T=T, A=A, B=240, tick_size=np.full(A, 0.01, dtype=np.float64))

    tod = (minute_of_day.astype(np.int32) - int(cfg.rth_open_minute)).astype(np.int16)
    gap_min = np.zeros(T, dtype=np.float64)
    if T > 1:
        gap_min[1:] = 1.0
    reset_flag = np.zeros(T, dtype=np.int8)
    reset_flag[0] = np.int8(1)
    if T > 1:
        reset_flag[1:] = (session_id[1:] != session_id[:-1]).astype(np.int8)
    phase = np.full(T, np.int8(Phase.LIVE), dtype=np.int8)

    clock = {
        "minute_of_day": minute_of_day,
        "tod": tod,
        "session_id": session_id,
        "gap_min": gap_min,
        "reset_flag": reset_flag,
        "phase": phase,
    }
    state = preallocate_state(ts_ns, cfg, tuple(f"A{i}" for i in range(A)), clock_override=clock)
    _fill_required_channels(state)
    return state


def _set_vp_peak(state, t: int, a: int, bins, values):
    state.vp[t, a, :] = 0.0
    for b, v in zip(bins, values):
        state.vp[t, a, int(b)] = float(v)


class TestModule3Structure(unittest.TestCase):
    def test_build_prefix_sum_neutralizes_mid_series_nan(self):
        series = np.array([[[1.0], [2.0], [np.nan], [4.0], [5.0]]], dtype=np.float64)
        prefix = build_prefix_sum(series)
        np.testing.assert_allclose(
            prefix[:, :, 0],
            np.array([[0.0, 1.0, 3.0, 3.0, 7.0, 12.0]], dtype=np.float64),
            atol=0.0,
            rtol=0.0,
        )

    def test_prefix_mean_matches_finite_reference_when_nan_appears_mid_series(self):
        series = np.array([[[1.0], [2.0], [np.nan], [4.0], [5.0]]], dtype=np.float64)
        valid = np.isfinite(series)
        pref = rolling_mean_from_prefix(build_prefix_sum(series), build_prefix_count(valid), 2)
        naive = np.full(series.shape, np.nan, dtype=np.float64)
        for t in range(series.shape[1]):
            lo = t - 1
            if lo < 0:
                continue
            seg = series[:, lo : t + 1, :]
            naive[:, t, :] = np.nanmean(seg, axis=1)
        np.testing.assert_allclose(pref, naive, atol=1e-12, rtol=0.0, equal_nan=True)

    def test_active_rolling_mean_counts_only_finite_samples(self):
        series = np.array([[[1.0], [2.0], [np.nan], [4.0], [5.0]]], dtype=np.float64)
        valid = np.ones_like(series, dtype=bool)
        got = _rolling_mean_for_series(series, valid, 2, eps=1e-12)
        expected = np.array([[[np.nan], [1.5], [2.0], [4.0], [4.5]]], dtype=np.float64)
        np.testing.assert_allclose(got, expected, atol=1e-12, rtol=0.0, equal_nan=True)

    def test_window_context_validity_ignores_optional_regime_channels_in_non_regime_modes(self):
        A = 1
        T = 3
        W = 1
        structure = np.full((A, T, int(module3.StructIdx.N_FIELDS), W), np.nan, dtype=np.float64)
        for s_idx in [
            module3.StructIdx.X_POC,
            module3.StructIdx.X_VAH,
            module3.StructIdx.X_VAL,
            module3.StructIdx.VA_WIDTH_X,
            module3.StructIdx.DCLIP_MEAN,
            module3.StructIdx.AFFINITY_MEAN,
            module3.StructIdx.ZDELTA_MEAN,
            module3.StructIdx.DELTA_EFF_MEAN,
            module3.StructIdx.TREND_GATE_SPREAD_MEAN,
            module3.StructIdx.POC_DRIFT_X,
            module3.StructIdx.VALID_RATIO,
            module3.StructIdx.IB_HIGH_X,
            module3.StructIdx.IB_LOW_X,
            module3.StructIdx.POC_VS_PREV_VA,
        ]:
            structure[:, :, int(s_idx), :] = 1.0
        regime = np.zeros((A, T, 1, W), dtype=np.float64)
        session_id = np.zeros(T, dtype=np.int64)

        ctx, valid, src = build_context_tensor(structure, regime, session_id, mode="ffill_last_complete")

        self.assertTrue(np.all(valid))
        np.testing.assert_array_equal(src[:, :, 0], np.array([[0, 1, 2]], dtype=np.int64))
        self.assertTrue(np.all(ctx[:, :, int(ContextIdx.CTX_REGIME_CODE), :] == 0.0))
        self.assertTrue(np.all(ctx[:, :, int(ContextIdx.CTX_REGIME_PERSISTENCE), :] == 0.0))

    def test_block_start_end_integrity_same_block_id(self):
        state = _make_state(T=20, A=1)
        state.vp[:, :, 120] = 1.0

        def _bad_block_map(_state, _cfg):
            T = _state.cfg.T
            in_scope = np.ones(T, dtype=bool)
            block_seq = np.zeros(T, dtype=np.int16)
            block_id = np.arange(T, dtype=np.int64)
            block_start = np.zeros(T, dtype=bool)
            block_end = np.zeros(T, dtype=bool)
            block_start[0] = True
            block_end[10] = True
            return in_scope, block_seq, block_id, block_start, block_end

        with patch.object(module3, "_build_block_map", side_effect=_bad_block_map):
            with self.assertRaisesRegex(RuntimeError, "Block start/end integrity violation"):
                run_module3_structural_aggregation(state, Module3Config(block_minutes=5))

        with self.assertRaisesRegex(RuntimeError, "block_seq exceeds 4095 safety stride"):
            run_module3_structural_aggregation(
                state,
                Module3Config(
                    block_minutes=1,
                    use_rth_minutes_only=False,
                    rth_open_minute=-5000,
                ),
            )

    def test_valid_ratio_uses_expected_minutes_not_observed_rows(self):
        minute = np.array(
            [
                570,
                571,
                572,
                573,
                574,
                576,
                577,
                578,
                579,
                580,
                581,
                582,
                583,
                584,
                585,
                586,
                587,
                588,
                589,
                590,
                591,
            ],
            dtype=np.int16,
        )
        session = np.zeros(minute.shape[0], dtype=np.int64)
        state = _make_state_with_clock_override(minute, session, A=1)
        state.vp[:, :, 120] = 10.0
        state.vp[:, :, 121] = 8.0

        out = run_module3_structural_aggregation(
            state,
            Module3Config(
                block_minutes=10,
                include_partial_last_block=True,
                min_block_valid_bars=1,
                min_block_valid_ratio=0.0,
            ),
        )
        te_idx = np.flatnonzero(out.block_end_flag_t)
        self.assertGreaterEqual(int(te_idx.size), 3)

        vr0 = float(out.block_features_tak[int(te_idx[0]), 0, int(Struct30mIdx.VALID_RATIO)])
        self.assertAlmostEqual(vr0, 0.9, places=12)
        vr_last = float(out.block_features_tak[int(te_idx[-1]), 0, int(Struct30mIdx.VALID_RATIO)])
        self.assertAlmostEqual(vr_last, 1.0, places=12)

    def test_context_forward_fill_is_session_safe_no_leak(self):
        minute = np.array([570, 571, 572, 573, 570, 571, 572, 573], dtype=np.int16)
        session = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        state = _make_state_with_clock_override(minute, session, A=1)
        state.vp[:, :, 120] = 10.0
        state.vp[:, :, 121] = 8.0
        cfg = Module3Config(
            block_minutes=4,
            include_partial_last_block=True,
            min_block_valid_bars=1,
            min_block_valid_ratio=0.0,
        )
        out = run_module3_structural_aggregation(state, cfg)

        # Force a cross-session source leak and assert fail-closed validation.
        out.context_valid_ta[4, 0] = True
        out.context_source_t_index_ta[4, 0] = 3
        out.context_tac[4, 0, :] = out.context_tac[3, 0, :]

        with self.assertRaisesRegex(RuntimeError, "crosses session boundary"):
            module3.validate_module3_output(state, out, cfg)

    def test_prefix_invariance_module3_outputs(self):
        st1 = _make_state(T=120, A=1)
        st2 = _make_state(T=120, A=1)
        st1.vp[:, :, 120] = 10.0
        st1.vp[:, :, 118] = 8.0
        st2.vp[:, :, 120] = 10.0
        st2.vp[:, :, 118] = 8.0

        t0 = 70
        st2.vp[t0 + 1 :, 0, 120] *= 3.0
        st2.vp[t0 + 1 :, 0, 118] *= 0.25
        st2.profile_stats[t0 + 1 :, 0, int(ProfileStatIdx.DCLIP)] += 0.7
        st2.profile_stats[t0 + 1 :, 0, int(ProfileStatIdx.A_AFFINITY)] -= 0.2
        st2.scores[t0 + 1 :, 0, int(ScoreIdx.SCORE_BO_LONG)] -= 0.4
        st2.bar_valid[t0 + 1 :, 0] = False

        cfg = Module3Config(block_minutes=30, min_block_valid_bars=1, min_block_valid_ratio=0.0)
        out1 = run_module3_structural_aggregation(st1, cfg)
        out2 = run_module3_structural_aggregation(st2, cfg)

        np.testing.assert_array_equal(out1.block_id_t[: t0 + 1], out2.block_id_t[: t0 + 1])
        np.testing.assert_array_equal(out1.block_seq_t[: t0 + 1], out2.block_seq_t[: t0 + 1])
        np.testing.assert_array_equal(out1.block_end_flag_t[: t0 + 1], out2.block_end_flag_t[: t0 + 1])
        np.testing.assert_array_equal(out1.block_start_t_index_t[: t0 + 1], out2.block_start_t_index_t[: t0 + 1])
        np.testing.assert_array_equal(out1.block_end_t_index_t[: t0 + 1], out2.block_end_t_index_t[: t0 + 1])
        np.testing.assert_array_equal(out1.block_valid_ta[: t0 + 1], out2.block_valid_ta[: t0 + 1])
        np.testing.assert_array_equal(out1.context_valid_ta[: t0 + 1], out2.context_valid_ta[: t0 + 1])
        np.testing.assert_array_equal(
            out1.context_source_t_index_ta[: t0 + 1],
            out2.context_source_t_index_ta[: t0 + 1],
        )
        np.testing.assert_allclose(
            out1.block_features_tak[: t0 + 1],
            out2.block_features_tak[: t0 + 1],
            atol=0.0,
            rtol=0.0,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            out1.context_tac[: t0 + 1],
            out2.context_tac[: t0 + 1],
            atol=0.0,
            rtol=0.0,
            equal_nan=True,
        )

    def test_ib_bounds_consistency(self):
        state = _make_state(T=90, A=1)
        for t in [29, 59, 89]:
            _set_vp_peak(state, t, 0, bins=[110, 120], values=[10.0, 9.0])
        cfg = Module3Config(min_block_valid_bars=1, min_block_valid_ratio=0.0)
        out = run_module3_structural_aggregation(state, cfg)

        ib_hi = out.block_features_tak[:, 0, int(Struct30mIdx.IB_HIGH_X)]
        ib_lo = out.block_features_tak[:, 0, int(Struct30mIdx.IB_LOW_X)]
        ib_mask = out.block_valid_ta[:, 0] & np.isfinite(ib_hi) & np.isfinite(ib_lo)
        self.assertTrue(np.all(ib_lo[ib_mask] <= ib_hi[ib_mask]))

        te_idx = np.flatnonzero(out.block_end_flag_t)
        self.assertGreater(int(te_idx.size), 0)
        t_bad = int(te_idx[0])
        out.block_valid_ta[t_bad, 0] = True
        out.block_features_tak[t_bad, 0, int(Struct30mIdx.IB_HIGH_X)] = 0.0
        out.block_features_tak[t_bad, 0, int(Struct30mIdx.IB_LOW_X)] = 1.0
        with self.assertRaisesRegex(RuntimeError, "IB bounds invalid"):
            module3.validate_module3_output(state, out, cfg)

    def test_block_end_boundaries_and_context_causality(self):
        state = _make_state(T=120, A=2)

        # Put profile mass on each expected 30m block-end.
        te_idx = np.array([29, 59, 89, 119], dtype=np.int64)
        for t in te_idx:
            for a in range(2):
                _set_vp_peak(state, int(t), a, bins=[118 + a, 120 + a], values=[10.0, 9.0])

        out = run_module3_structural_aggregation(state, Module3Config())

        got = np.flatnonzero(out.block_end_flag_t)
        self.assertTrue(np.array_equal(got, te_idx))

        t_idx = np.broadcast_to(
            np.arange(state.cfg.T, dtype=np.int64)[:, None],
            out.context_source_t_index_ta.shape,
        )
        self.assertTrue(
            np.all(out.context_source_t_index_ta[out.context_valid_ta] <= t_idx[out.context_valid_ta])
        )

    def test_ib_high_low_forward_logic(self):
        state = _make_state(T=90, A=1)
        x = state.x_grid
        te_idx = np.array([29, 59, 89], dtype=np.int64)

        # Block 0: populated bins [110, 120]
        _set_vp_peak(state, int(te_idx[0]), 0, bins=[110, 120], values=[10.0, 10.0])
        # Block 1: populated bins [100, 130]
        _set_vp_peak(state, int(te_idx[1]), 0, bins=[100, 130], values=[10.0, 10.0])
        # Block 2: arbitrary, IB should remain based on first two blocks
        _set_vp_peak(state, int(te_idx[2]), 0, bins=[115, 125], values=[10.0, 10.0])

        out = run_module3_structural_aggregation(state, Module3Config())

        ib_hi = out.block_features_tak[:, 0, int(Struct30mIdx.IB_HIGH_X)]
        ib_lo = out.block_features_tak[:, 0, int(Struct30mIdx.IB_LOW_X)]

        # seq==0 uses IB0
        self.assertAlmostEqual(float(ib_hi[29]), float(x[120]), places=12)
        self.assertAlmostEqual(float(ib_lo[29]), float(x[110]), places=12)

        # seq>=1 uses IB01 union(first two blocks)
        self.assertAlmostEqual(float(ib_hi[59]), float(x[130]), places=12)
        self.assertAlmostEqual(float(ib_lo[59]), float(x[100]), places=12)
        self.assertAlmostEqual(float(ib_hi[89]), float(x[130]), places=12)
        self.assertAlmostEqual(float(ib_lo[89]), float(x[100]), places=12)

    def test_missing_ib_seed_blocks_are_marked_invalid_not_exception(self):
        state = _make_state(T=120, A=1)
        # Only seq=2 has usable mass; seq=0/1 are absent.
        _set_vp_peak(state, 89, 0, bins=[118, 120], values=[10.0, 9.0])
        cfg = Module3Config(min_block_valid_bars=1, min_block_valid_ratio=0.0, fail_on_non_finite_output=True)
        out = run_module3_structural_aggregation(state, cfg)

        # seq=2 block-end should be invalidated because IB seed is undefined.
        self.assertFalse(bool(out.block_valid_ta[89, 0]))
        self.assertIsNotNone(out.ib_defined_ta)
        self.assertFalse(bool(out.ib_defined_ta[89, 0]))
        ib_hi = out.block_features_tak[89, 0, int(Struct30mIdx.IB_HIGH_X)]
        ib_lo = out.block_features_tak[89, 0, int(Struct30mIdx.IB_LOW_X)]
        self.assertTrue(np.isnan(float(ib_hi)))
        self.assertTrue(np.isnan(float(ib_lo)))

    def test_ib_policy_degrade_keeps_block_valid(self):
        state = _make_state(T=120, A=1)
        _set_vp_peak(state, 89, 0, bins=[118, 120], values=[10.0, 9.0])
        cfg = Module3Config(min_block_valid_bars=1, min_block_valid_ratio=0.0, fail_on_non_finite_output=True)

        with patch.object(module3, "IB_MISSING_POLICY", "DEGRADE"):
            out = run_module3_structural_aggregation(state, cfg)

        self.assertTrue(bool(out.block_end_flag_t[89]))
        self.assertTrue(bool(out.block_valid_ta[89, 0]))
        self.assertIsNotNone(out.ib_defined_ta)
        self.assertFalse(bool(out.ib_defined_ta[89, 0]))

    def test_poc_vs_prev_va_piecewise_metric(self):
        state = _make_state(T=90, A=1)

        # Non-zero vp mass so block_valid can pass.
        for t in [29, 59, 89]:
            _set_vp_peak(state, t, 0, bins=[118, 120], values=[10.0, 8.0])

        # Block 0: POC=0.0, VA=[-0.5, 0.5]
        state.profile_stats[29, 0, int(ProfileStatIdx.IPOC)] = 120.0
        state.profile_stats[29, 0, int(ProfileStatIdx.IVAH)] = 130.0
        state.profile_stats[29, 0, int(ProfileStatIdx.IVAL)] = 110.0

        # Block 1: POC=0.75 above prev VAH=0.5 -> rel = 1 + 0.25/1 = 1.25
        state.profile_stats[59, 0, int(ProfileStatIdx.IPOC)] = 135.0
        state.profile_stats[59, 0, int(ProfileStatIdx.IVAH)] = 140.0
        state.profile_stats[59, 0, int(ProfileStatIdx.IVAL)] = 120.0

        # Block 2: POC=-1.0 below prev VAL=0.0 -> rel = -1 - 1/1 = -2
        state.profile_stats[89, 0, int(ProfileStatIdx.IPOC)] = 100.0
        state.profile_stats[89, 0, int(ProfileStatIdx.IVAH)] = 105.0
        state.profile_stats[89, 0, int(ProfileStatIdx.IVAL)] = 95.0

        out = run_module3_structural_aggregation(state, Module3Config())
        rel = out.block_features_tak[:, 0, int(Struct30mIdx.POC_VS_PREV_VA)]

        self.assertAlmostEqual(float(rel[29]), 0.0, places=12)
        self.assertAlmostEqual(float(rel[59]), 1.25, places=12)
        self.assertAlmostEqual(float(rel[89]), -2.0, places=12)

        # Context channel mapping integrity at the same rows.
        ctx_rel = out.context_tac[:, 0, int(ContextIdx.CTX_POC_VS_PREV_VA)]
        self.assertTrue(np.isfinite(ctx_rel[59]))

    def test_fail_closed_non_finite_input(self):
        state = _make_state(T=60, A=1)
        _set_vp_peak(state, 29, 0, bins=[120], values=[10.0])
        _set_vp_peak(state, 59, 0, bins=[120], values=[10.0])

        # Inject non-finite on required in-scope row.
        state.profile_stats[40, 0, int(ProfileStatIdx.DCLIP)] = np.nan

        with self.assertRaises(RuntimeError):
            run_module3_structural_aggregation(state, Module3Config(fail_on_non_finite_input=True))


if __name__ == "__main__":
    unittest.main()
