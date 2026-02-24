import unittest
import numpy as np

from weightiz_module1_core import EngineConfig, Phase, ProfileStatIdx, ScoreIdx, preallocate_state
from weightiz_module3_structure import (
    ContextIdx,
    Module3Config,
    Struct30mIdx,
    run_module3_structural_aggregation,
)


def _make_state(T: int = 90, A: int = 1):
    start_ns = np.datetime64("2025-01-06T14:30:00", "ns").astype(np.int64)
    ts_ns = start_ns + np.arange(T, dtype=np.int64) * np.int64(60_000_000_000)
    cfg = EngineConfig(T=T, A=A, B=240, tick_size=np.full(A, 0.01, dtype=np.float64))
    symbols = tuple(f"A{i}" for i in range(A))
    state = preallocate_state(ts_ns, cfg, symbols)

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

    return state


def _set_vp_peak(state, t: int, a: int, bins, values):
    state.vp[t, a, :] = 0.0
    for b, v in zip(bins, values):
        state.vp[t, a, int(b)] = float(v)


class TestModule3Structure(unittest.TestCase):
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

        t_idx = np.arange(state.cfg.T, dtype=np.int64)[:, None]
        self.assertTrue(np.all(out.context_source_t_index_ta[out.context_valid_ta] <= t_idx[out.context_valid_ta]))

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
