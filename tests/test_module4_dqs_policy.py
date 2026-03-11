from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from weightiz.module3 import ContextIdx, StructIdx
from weightiz.module1.core import EngineConfig, Phase, ProfileStatIdx, ScoreIdx, preallocate_state
from weightiz.module4.strategy_funnel import Module4Config, run_module4_signal_funnel


def _mk_state(T: int = 20, A: int = 2):
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


def _mk_m3(T: int, A: int, *, ib_defined: bool = True) -> object:
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
        ib_defined_ta=np.full((T, A), ib_defined, dtype=bool),
    )


class TestModule4DQSPolicy(unittest.TestCase):
    def test_dqs_day_is_ignored_by_signal_only_module4(self) -> None:
        st_lo = _mk_state(T=16, A=1)
        st_hi = _mk_state(T=16, A=1)
        st_lo.dqs_day_ta = np.full((st_lo.cfg.T, st_lo.cfg.A), 0.10, dtype=np.float64)
        st_hi.dqs_day_ta = np.full((st_hi.cfg.T, st_hi.cfg.A), 0.95, dtype=np.float64)
        m3_lo = _mk_m3(st_lo.cfg.T, st_lo.cfg.A)
        m3_hi = _mk_m3(st_hi.cfg.T, st_hi.cfg.A)

        out_lo = run_module4_signal_funnel(st_lo, m3_lo, Module4Config(entry_threshold=0.55))
        out_hi = run_module4_signal_funnel(st_hi, m3_hi, Module4Config(entry_threshold=0.55))

        np.testing.assert_array_equal(out_lo.regime_primary_ta, out_hi.regime_primary_ta)
        np.testing.assert_allclose(out_lo.regime_confidence_ta, out_hi.regime_confidence_ta, rtol=0.0, atol=0.0)
        np.testing.assert_array_equal(out_lo.intent_long_ta, out_hi.intent_long_ta)
        np.testing.assert_array_equal(out_lo.intent_short_ta, out_hi.intent_short_ta)
        np.testing.assert_allclose(out_lo.target_qty_ta, out_hi.target_qty_ta, rtol=0.0, atol=0.0)

    def test_ib_defined_is_ignored_by_signal_only_bridge(self) -> None:
        st = _mk_state(T=16, A=2)
        out_defined = run_module4_signal_funnel(st, _mk_m3(st.cfg.T, st.cfg.A, ib_defined=True), Module4Config())
        out_missing = run_module4_signal_funnel(st, _mk_m3(st.cfg.T, st.cfg.A, ib_defined=False), Module4Config())

        np.testing.assert_array_equal(out_defined.regime_primary_ta, out_missing.regime_primary_ta)
        np.testing.assert_allclose(out_defined.regime_confidence_ta, out_missing.regime_confidence_ta, rtol=0.0, atol=0.0)
        np.testing.assert_array_equal(out_defined.intent_long_ta, out_missing.intent_long_ta)
        np.testing.assert_array_equal(out_defined.intent_short_ta, out_missing.intent_short_ta)
        np.testing.assert_allclose(out_defined.target_qty_ta, out_missing.target_qty_ta, rtol=0.0, atol=0.0)

    def test_execution_entrypoint_remains_forbidden(self) -> None:
        st = _mk_state(T=8, A=1)
        with self.assertRaises(RuntimeError, msg="legacy execution path must remain forbidden"):
            from weightiz.module4.strategy_funnel import run_module4_strategy_funnel

            run_module4_strategy_funnel(st, _mk_m3(st.cfg.T, st.cfg.A), Module4Config())


if __name__ == "__main__":
    unittest.main()
