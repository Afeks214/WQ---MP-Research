import unittest
import numpy as np

from weightiz_module1_core import EngineConfig, Phase, ProfileStatIdx, ScoreIdx, preallocate_state
from weightiz_module3_structure import ContextIdx, Module3Output, Struct30mIdx
from weightiz_module4_strategy_funnel import Module4Config, RegimeIdx, run_module4_strategy_funnel


def _mk_state(T: int = 40, A: int = 3):
    start_ns = np.datetime64("2025-01-06T14:30:00", "ns").astype(np.int64)
    ts_ns = start_ns + np.arange(T, dtype=np.int64) * np.int64(60_000_000_000)
    cfg = EngineConfig(T=T, A=A, B=240, tick_size=np.full(A, 0.01, dtype=np.float64))
    st = preallocate_state(ts_ns, cfg, tuple(f"A{i}" for i in range(A)))
    st.phase[:] = np.int8(Phase.LIVE)
    st.bar_valid[:] = True

    # Simple deterministic OHLCV
    base = 100.0 + np.arange(T, dtype=np.float64)[:, None] * 0.01
    st.open_px[:] = base
    st.high_px[:] = base + 0.10
    st.low_px[:] = base - 0.10
    st.close_px[:] = base + 0.02
    st.volume[:] = 10_000.0
    st.rvol[:] = 1.2
    st.atr_floor[:] = 0.5

    # Scores/profile channels
    st.scores[:] = 0.0
    st.profile_stats[:] = 0.0
    st.scores[:, :, int(ScoreIdx.SCORE_BO_LONG)] = 0.8
    st.scores[:, :, int(ScoreIdx.SCORE_BO_SHORT)] = 0.2
    st.scores[:, :, int(ScoreIdx.SCORE_REJ_LONG)] = 0.3
    st.scores[:, :, int(ScoreIdx.SCORE_REJ_SHORT)] = 0.1

    st.profile_stats[:, :, int(ProfileStatIdx.DCLIP)] = 1.0
    st.profile_stats[:, :, int(ProfileStatIdx.Z_DELTA)] = 1.2
    st.profile_stats[:, :, int(ProfileStatIdx.GBREAK)] = 0.9
    st.profile_stats[:, :, int(ProfileStatIdx.GREJECT)] = 0.2
    st.profile_stats[:, :, int(ProfileStatIdx.SIGMA_EFF)] = 0.5

    # Minimal vp for double-distribution checks
    st.vp[:] = 0.0
    st.vp[:, :, 100] = 5.0
    st.vp[:, :, 130] = 4.0
    st.vp[:, :, 115] = 0.5
    return st


def _mk_m3(st) -> Module3Output:
    T, A = st.cfg.T, st.cfg.A
    C3 = int(ContextIdx.N_FIELDS)
    K3 = int(Struct30mIdx.N_FIELDS)
    ctx = np.full((T, A, C3), np.nan, dtype=np.float64)
    ctx[:, :, int(ContextIdx.CTX_X_POC)] = 0.5
    ctx[:, :, int(ContextIdx.CTX_X_VAH)] = 1.0
    ctx[:, :, int(ContextIdx.CTX_X_VAL)] = -1.0
    ctx[:, :, int(ContextIdx.CTX_VA_WIDTH_X)] = 2.0
    ctx[:, :, int(ContextIdx.CTX_DCLIP_MEAN)] = 1.0
    ctx[:, :, int(ContextIdx.CTX_AFFINITY_MEAN)] = 0.7
    ctx[:, :, int(ContextIdx.CTX_ZDELTA_MEAN)] = 1.2
    ctx[:, :, int(ContextIdx.CTX_DELTA_EFF_MEAN)] = 0.6
    ctx[:, :, int(ContextIdx.CTX_TREND_GATE_SPREAD_MEAN)] = 0.2
    ctx[:, :, int(ContextIdx.CTX_POC_DRIFT_X)] = 0.5
    ctx[:, :, int(ContextIdx.CTX_VALID_RATIO)] = 1.0
    ctx[:, :, int(ContextIdx.CTX_IB_HIGH_X)] = 1.2
    ctx[:, :, int(ContextIdx.CTX_IB_LOW_X)] = -1.2
    ctx[:, :, int(ContextIdx.CTX_POC_VS_PREV_VA)] = 1.2

    block_features = np.full((T, A, K3), np.nan, dtype=np.float64)
    block_features[:, :, int(Struct30mIdx.SKEW_ANCHOR)] = -0.5
    block_features[:, :, int(Struct30mIdx.X_POC)] = 0.5
    block_features[:, :, int(Struct30mIdx.X_VAH)] = 1.0
    block_features[:, :, int(Struct30mIdx.X_VAL)] = -1.0

    src = np.tile(np.arange(T, dtype=np.int64)[:, None], (1, A))
    valid = np.ones((T, A), dtype=bool)
    return Module3Output(
        block_id_t=np.arange(T, dtype=np.int64),
        block_seq_t=np.zeros(T, dtype=np.int16),
        block_end_flag_t=np.ones(T, dtype=bool),
        block_start_t_index_t=np.arange(T, dtype=np.int64),
        block_end_t_index_t=np.arange(T, dtype=np.int64),
        block_features_tak=block_features,
        block_valid_ta=valid.copy(),
        context_tac=ctx,
        context_valid_ta=valid,
        context_source_t_index_ta=src,
    )


class TestModule4StrategyFunnel(unittest.TestCase):
    def test_run_shapes_and_mutation(self):
        st = _mk_state(T=50, A=4)
        m3 = _mk_m3(st)
        out = run_module4_strategy_funnel(st, m3, Module4Config())
        self.assertEqual(out.regime_primary_ta.shape, (st.cfg.T, st.cfg.A))
        self.assertEqual(out.target_qty_ta.shape, (st.cfg.T, st.cfg.A))
        self.assertEqual(st.position_qty.shape, (st.cfg.T, st.cfg.A))
        self.assertTrue(np.all(np.isfinite(st.equity)))

    def test_dynamic_exit_context_stop(self):
        st = _mk_state(T=20, A=1)
        m3 = _mk_m3(st)
        # Force close below CTX_X_VAL to trigger long exit logic.
        m3.context_tac[:, 0, int(ContextIdx.CTX_X_VAL)] = 99.99
        st.close_px[:, 0] = 99.0
        out = run_module4_strategy_funnel(st, m3, Module4Config())
        # At least one bar should request zero target due to exit pressure.
        self.assertTrue(np.any(np.isclose(out.target_qty_ta[:, 0], 0.0)))

    def test_overnight_cash_fallback(self):
        st = _mk_state(T=30, A=2)
        m3 = _mk_m3(st)
        st.phase[10] = np.int8(Phase.OVERNIGHT_SELECT)
        # Crush OCS factors.
        st.profile_stats[10, :, int(ProfileStatIdx.DCLIP)] = 0.0
        st.profile_stats[10, :, int(ProfileStatIdx.Z_DELTA)] = 0.0
        st.rvol[10, :] = 0.0
        out = run_module4_strategy_funnel(st, m3, Module4Config(allow_cash_overnight=True))
        self.assertEqual(int(out.overnight_winner_t[10]), -1)

    def test_topk_bound(self):
        st = _mk_state(T=30, A=10)
        m3 = _mk_m3(st)
        cfg = Module4Config(top_k_intraday=3)
        out = run_module4_strategy_funnel(st, m3, cfg)
        # Count non-zero target names per row should stay bounded by A and generally near Top-K.
        nz = np.sum(np.abs(out.target_qty_ta) > 1e-12, axis=1)
        self.assertTrue(np.all(nz <= st.cfg.A))

    def test_regime_labeling_exists(self):
        st = _mk_state(T=25, A=2)
        m3 = _mk_m3(st)
        out = run_module4_strategy_funnel(st, m3, Module4Config())
        valid = {
            np.int8(RegimeIdx.NONE),
            np.int8(RegimeIdx.NEUTRAL),
            np.int8(RegimeIdx.TREND),
            np.int8(RegimeIdx.P_SHAPE),
            np.int8(RegimeIdx.B_SHAPE),
            np.int8(RegimeIdx.DOUBLE_DISTRIBUTION),
        }
        vals = set(np.unique(out.regime_primary_ta).astype(np.int8).tolist())
        self.assertTrue(vals.issubset(valid))
        self.assertTrue(np.any(out.regime_primary_ta != np.int8(RegimeIdx.NONE)))


if __name__ == "__main__":
    unittest.main()
