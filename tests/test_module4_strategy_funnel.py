from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from weightiz.module3 import ContextIdx, StructIdx
from weightiz.module1.core import EngineConfig, Phase, ProfileStatIdx, ScoreIdx, preallocate_state
from weightiz.module4.strategy_funnel import Module4Config, RegimeIdx, run_module4_signal_funnel


def _mk_state(T: int = 40, A: int = 3):
    start_ns = np.datetime64("2025-01-06T14:30:00", "ns").astype(np.int64)
    ts_ns = start_ns + np.arange(T, dtype=np.int64) * np.int64(60_000_000_000)
    cfg = EngineConfig(T=T, A=A, B=240, tick_size=np.full(A, 0.01, dtype=np.float64))
    st = preallocate_state(ts_ns, cfg, tuple(f"A{i}" for i in range(A)))
    st.phase[:] = np.int8(Phase.LIVE)
    st.bar_valid[:] = True
    st.scores[:] = 0.0
    st.profile_stats[:] = 0.0
    st.scores[:, :, int(ScoreIdx.SCORE_BO_LONG)] = 0.8
    st.scores[:, :, int(ScoreIdx.SCORE_BO_SHORT)] = 0.2
    st.profile_stats[:, :, int(ProfileStatIdx.DCLIP)] = 1.0
    st.profile_stats[:, :, int(ProfileStatIdx.Z_DELTA)] = 1.2
    return st


def _mk_canonical_m3(T: int, A: int) -> object:
    structure = np.zeros((A, T, int(StructIdx.N_FIELDS), 1), dtype=np.float64)
    context = np.zeros((A, T, int(ContextIdx.N_FIELDS), 1), dtype=np.float64)
    structure[:, :, int(StructIdx.VALID_RATIO), 0] = 1.0
    structure[:, :, int(StructIdx.TREND_GATE_SPREAD_MEAN), 0] = 0.2
    structure[:, :, int(StructIdx.POC_DRIFT_X), 0] = 0.6
    context[:, :, int(ContextIdx.CTX_VALID_RATIO), 0] = 1.0
    context[:, :, int(ContextIdx.CTX_TREND_GATE_SPREAD_MEAN), 0] = 0.2
    context[:, :, int(ContextIdx.CTX_POC_DRIFT_X), 0] = 0.6
    context[:, :, int(ContextIdx.CTX_REGIME_CODE), 0] = 1.0
    context[:, :, int(ContextIdx.CTX_REGIME_PERSISTENCE), 0] = 1.0
    return SimpleNamespace(
        structure_tensor=structure,
        context_tensor=context,
        profile_fingerprint_tensor=np.zeros((A, T, 1, 1), dtype=np.float64),
        profile_regime_tensor=np.zeros((A, T, 1, 1), dtype=np.float64),
        context_valid_ta=np.ones((T, A), dtype=bool),
        context_source_index_atw=np.broadcast_to(np.arange(T, dtype=np.int64)[None, :, None], (A, T, 1)).copy(),
    )


def _mk_legacy_bridge_m3(T: int, A: int) -> object:
    block_features = np.zeros((T, A, int(StructIdx.N_FIELDS)), dtype=np.float64)
    context = np.zeros((T, A, int(ContextIdx.N_FIELDS)), dtype=np.float64)
    block_features[:, :, int(StructIdx.VALID_RATIO)] = 1.0
    block_features[:, :, int(StructIdx.TREND_GATE_SPREAD_MEAN)] = 0.2
    block_features[:, :, int(StructIdx.POC_DRIFT_X)] = 0.6
    context[:, :, int(ContextIdx.CTX_VALID_RATIO)] = 1.0
    context[:, :, int(ContextIdx.CTX_TREND_GATE_SPREAD_MEAN)] = 0.2
    context[:, :, int(ContextIdx.CTX_POC_DRIFT_X)] = 0.6
    context[:, :, int(ContextIdx.CTX_REGIME_CODE)] = 1.0
    context[:, :, int(ContextIdx.CTX_REGIME_PERSISTENCE)] = 1.0
    return SimpleNamespace(
        block_features_tak=block_features,
        context_tac=context,
        context_valid_ta=np.ones((T, A), dtype=bool),
        context_source_t_index_ta=np.broadcast_to(np.arange(T, dtype=np.int64)[:, None], (T, A)).copy(),
    )


class TestModule4StrategyFunnel(unittest.TestCase):
    def test_run_shapes_and_no_execution_mutation(self):
        st = _mk_state(T=50, A=4)
        m3 = _mk_canonical_m3(st.cfg.T, st.cfg.A)
        position_before = st.position_qty.copy()
        equity_before = st.equity.copy()
        out = run_module4_signal_funnel(st, m3, Module4Config())

        self.assertEqual(out.regime_primary_ta.shape, (st.cfg.T, st.cfg.A))
        self.assertEqual(out.target_qty_ta.shape, (st.cfg.T, st.cfg.A))
        self.assertEqual(out.allocation_score_ta.shape, (st.cfg.T, st.cfg.A))
        self.assertEqual(out.conviction_net_ta.shape, (st.cfg.T, st.cfg.A))
        self.assertEqual(out.target_weight_ta.shape, (st.cfg.T, st.cfg.A))
        self.assertEqual(out.decision_reason_code_ta.shape, (st.cfg.T, st.cfg.A))
        np.testing.assert_array_equal(st.position_qty, position_before)
        np.testing.assert_array_equal(st.equity, equity_before)

    def test_legacy_bridge_fallback_accepts_context_tac_and_block_features(self):
        st = _mk_state(T=20, A=2)
        out = run_module4_signal_funnel(st, _mk_legacy_bridge_m3(st.cfg.T, st.cfg.A), Module4Config())
        self.assertEqual(out.regime_primary_ta.shape, (st.cfg.T, st.cfg.A))
        self.assertEqual(out.regime_primary_ta.dtype, np.int8)
        self.assertEqual(out.target_qty_ta.dtype, np.float64)

    def test_regime_labeling_exists(self):
        st = _mk_state(T=25, A=2)
        out = run_module4_signal_funnel(st, _mk_canonical_m3(st.cfg.T, st.cfg.A), Module4Config())
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

    def test_target_qty_is_discrete_position_signal_and_weight_telemetry_is_preserved(self):
        st = _mk_state(T=30, A=10)
        out = run_module4_signal_funnel(st, _mk_canonical_m3(st.cfg.T, st.cfg.A), Module4Config(max_abs_weight=0.8))
        self.assertTrue(np.all(np.isfinite(out.target_qty_ta)))
        self.assertTrue(np.all(np.isin(out.target_qty_ta, np.array([-1.0, 0.0, 1.0], dtype=np.float64))))
        self.assertIsNotNone(out.target_weight_ta)
        assert out.target_weight_ta is not None
        self.assertTrue(np.all(np.isfinite(out.target_weight_ta)))
        self.assertTrue(np.all(np.abs(out.target_weight_ta) <= 0.8 + 1e-12))
        np.testing.assert_array_equal(out.target_qty_ta, np.sign(out.target_weight_ta))

    def test_topk_intraday_is_compatibility_only_in_signal_path(self):
        st = _mk_state(T=30, A=5)
        m3 = _mk_canonical_m3(st.cfg.T, st.cfg.A)
        out_default = run_module4_signal_funnel(st, m3, Module4Config())
        out_compat = run_module4_signal_funnel(st, m3, Module4Config(top_k_intraday=1))
        np.testing.assert_array_equal(out_default.regime_primary_ta, out_compat.regime_primary_ta)
        np.testing.assert_allclose(out_default.regime_confidence_ta, out_compat.regime_confidence_ta, rtol=0.0, atol=0.0)
        np.testing.assert_array_equal(out_default.intent_long_ta, out_compat.intent_long_ta)
        np.testing.assert_array_equal(out_default.intent_short_ta, out_compat.intent_short_ta)
        np.testing.assert_allclose(out_default.target_qty_ta, out_compat.target_qty_ta, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
