from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from module3 import ContextIdx, StructIdx
from weightiz_module1_core import EngineConfig, NS_PER_MIN, Phase, ProfileStatIdx, ScoreIdx, preallocate_state
from weightiz_module4_strategy_funnel import Module4Config, run_module4_signal_funnel, run_module4_strategy_funnel


def _mk_state(T: int = 4, A: int = 1):
    start_ns = np.datetime64("2025-01-06T14:30:00", "ns").astype(np.int64)
    ts_ns = start_ns + np.arange(T, dtype=np.int64) * np.int64(NS_PER_MIN)
    cfg = EngineConfig(T=T, A=A, B=240, tick_size=np.full(A, 0.01, dtype=np.float64))
    st = preallocate_state(ts_ns=ts_ns, cfg=cfg, symbols=("AAA",))
    st.phase[:] = np.int8(Phase.LIVE)
    st.bar_valid[:] = True
    st.open_px[:] = 100.0
    st.open_px[1, 0] = np.nan
    st.scores[:] = 0.0
    st.profile_stats[:] = 0.0
    st.scores[:, :, int(ScoreIdx.SCORE_BO_LONG)] = 0.9
    st.profile_stats[:, :, int(ProfileStatIdx.DCLIP)] = 1.0
    st.profile_stats[:, :, int(ProfileStatIdx.Z_DELTA)] = 1.0
    return st


def _mk_m3(T: int, A: int) -> object:
    structure = np.zeros((A, T, int(StructIdx.N_FIELDS), 1), dtype=np.float64)
    context = np.zeros((A, T, int(ContextIdx.N_FIELDS), 1), dtype=np.float64)
    structure[:, :, int(StructIdx.VALID_RATIO), 0] = 1.0
    structure[:, :, int(StructIdx.TREND_GATE_SPREAD_MEAN), 0] = 0.3
    structure[:, :, int(StructIdx.POC_DRIFT_X), 0] = 0.5
    context[:, :, int(ContextIdx.CTX_VALID_RATIO), 0] = 1.0
    context[:, :, int(ContextIdx.CTX_TREND_GATE_SPREAD_MEAN), 0] = 0.3
    context[:, :, int(ContextIdx.CTX_POC_DRIFT_X), 0] = 0.5
    context[:, :, int(ContextIdx.CTX_REGIME_CODE), 0] = 1.0
    return SimpleNamespace(
        structure_tensor=structure,
        context_tensor=context,
        profile_fingerprint_tensor=np.zeros((A, T, 1, 1), dtype=np.float64),
        profile_regime_tensor=np.zeros((A, T, 1, 1), dtype=np.float64),
        context_valid_ta=np.ones((T, A), dtype=bool),
        context_source_index_atw=np.broadcast_to(np.arange(T, dtype=np.int64)[None, :, None], (A, T, 1)).copy(),
        ib_defined_ta=np.ones((T, A), dtype=bool),
    )


class TestModule4NextOpenNoLookahead(unittest.TestCase):
    def test_execution_entry_is_forbidden_even_with_fill_time_pathology(self) -> None:
        st = _mk_state()
        with self.assertRaisesRegex(RuntimeError, "MODULE4_EXECUTION_FORBIDDEN_IN_CANONICAL_PATH"):
            run_module4_strategy_funnel(
                st,
                _mk_m3(st.cfg.T, st.cfg.A),
                Module4Config(entry_threshold=0.55, fail_on_non_finite_input=False),
                run_context={"candidate_id": "c0", "split_id": "wf_000", "scenario_id": "baseline"},
            )

    def test_execution_entry_is_forbidden_even_when_fill_bar_is_invalid(self) -> None:
        st = _mk_state()
        st.bar_valid[1, 0] = False
        with self.assertRaisesRegex(RuntimeError, "MODULE4_EXECUTION_FORBIDDEN_IN_CANONICAL_PATH"):
            run_module4_strategy_funnel(
                st,
                _mk_m3(st.cfg.T, st.cfg.A),
                Module4Config(entry_threshold=0.55, fail_on_non_finite_input=False),
                run_context={"candidate_id": "c0", "split_id": "wf_000", "scenario_id": "baseline"},
            )

    def test_signal_funnel_ignores_next_open_execution_pathologies(self) -> None:
        st = _mk_state()
        out = run_module4_signal_funnel(st, _mk_m3(st.cfg.T, st.cfg.A), Module4Config(entry_threshold=0.55))
        self.assertEqual(out.regime_primary_ta.shape, (st.cfg.T, st.cfg.A))
        self.assertEqual(out.target_qty_ta.shape, (st.cfg.T, st.cfg.A))


if __name__ == "__main__":
    unittest.main()
