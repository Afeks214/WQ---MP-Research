from __future__ import annotations

import unittest

import numpy as np

from weightiz_module1_core import EngineConfig, NS_PER_MIN, Phase, ProfileStatIdx, ScoreIdx, preallocate_state
from weightiz_module3_structure import ContextIdx, Module3Output, Struct30mIdx
from weightiz_module4_strategy_funnel import Module4Config, NonFiniteExecutionPriceError, run_module4_strategy_funnel


def _mk_state(T: int = 4, A: int = 1):
    start_ns = np.datetime64("2025-01-06T14:30:00", "ns").astype(np.int64)
    ts_ns = start_ns + np.arange(T, dtype=np.int64) * np.int64(NS_PER_MIN)
    cfg = EngineConfig(T=T, A=A, B=240, tick_size=np.full(A, 0.01, dtype=np.float64))
    st = preallocate_state(ts_ns, cfg, tuple(f"A{i}" for i in range(A)))
    st.phase[:] = np.int8(Phase.LIVE)
    st.bar_valid[:] = True

    base = 100.0 + np.arange(T, dtype=np.float64)[:, None] * 0.1
    st.open_px[:] = base
    st.high_px[:] = base + 0.2
    st.low_px[:] = base - 0.2
    st.close_px[:] = base + 0.05
    st.volume[:] = 10_000.0
    st.rvol[:] = 1.2
    st.atr_floor[:] = 0.5

    st.scores[:] = 0.0
    st.profile_stats[:] = 0.0
    st.scores[:, :, int(ScoreIdx.SCORE_BO_LONG)] = 0.9
    st.profile_stats[:, :, int(ProfileStatIdx.GBREAK)] = 1.0
    st.profile_stats[:, :, int(ProfileStatIdx.GREJECT)] = 0.0
    st.profile_stats[:, :, int(ProfileStatIdx.DCLIP)] = 1.0
    st.profile_stats[:, :, int(ProfileStatIdx.Z_DELTA)] = 1.0

    st.open_px[1, 0] = np.nan
    return st


def _mk_m3(st) -> Module3Output:
    T, A = st.cfg.T, st.cfg.A
    c3 = int(ContextIdx.N_FIELDS)
    k3 = int(Struct30mIdx.N_FIELDS)
    ctx = np.full((T, A, c3), np.nan, dtype=np.float64)
    ctx[:, :, int(ContextIdx.CTX_X_VAH)] = 1.0
    ctx[:, :, int(ContextIdx.CTX_X_VAL)] = -1.0
    ctx[:, :, int(ContextIdx.CTX_VALID_RATIO)] = 1.0
    ctx[:, :, int(ContextIdx.CTX_TREND_GATE_SPREAD_MEAN)] = 0.3
    ctx[:, :, int(ContextIdx.CTX_POC_DRIFT_X)] = 0.5
    ctx[:, :, int(ContextIdx.CTX_POC_VS_PREV_VA)] = 1.2

    blocks = np.full((T, A, k3), np.nan, dtype=np.float64)
    blocks[:, :, int(Struct30mIdx.SKEW_ANCHOR)] = -0.5
    src = np.tile(np.arange(T, dtype=np.int64)[:, None], (1, A))
    valid = np.ones((T, A), dtype=bool)
    return Module3Output(
        block_id_t=np.arange(T, dtype=np.int64),
        block_seq_t=np.zeros(T, dtype=np.int16),
        block_end_flag_t=np.ones(T, dtype=bool),
        block_start_t_index_t=np.arange(T, dtype=np.int64),
        block_end_t_index_t=np.arange(T, dtype=np.int64),
        block_features_tak=blocks,
        block_valid_ta=valid.copy(),
        context_tac=ctx,
        context_valid_ta=valid.copy(),
        context_source_t_index_ta=src,
        ib_defined_ta=np.ones((T, A), dtype=bool),
    )


class TestModule4NonFiniteExecPx(unittest.TestCase):
    def test_nonfinite_exec_px_raises_typed_exception_with_dump(self) -> None:
        st = _mk_state()
        m3 = _mk_m3(st)
        with self.assertRaises(NonFiniteExecutionPriceError) as cm:
            run_module4_strategy_funnel(
                st,
                m3,
                Module4Config(entry_threshold=0.55, fail_on_non_finite_input=False),
                run_context={"candidate_id": "c0", "split_id": "wf_000", "scenario_id": "baseline"},
            )
        exc = cm.exception
        self.assertEqual(exc.reason_code, "NONFINITE_EXEC_PX")
        dump = exc.exec_px_dump
        required = {
            "run_context",
            "reason_code",
            "t_signal",
            "t_fill",
            "ts_utc_signal",
            "ts_utc_fill",
            "asset_index",
            "asset_symbol",
            "px_source_name",
            "px_value",
            "open_px_signal",
            "high_px_signal",
            "low_px_signal",
            "close_px_signal",
            "open_px_fill",
            "high_px_fill",
            "low_px_fill",
            "close_px_fill",
            "bar_valid_signal",
            "bar_valid_fill",
            "target_qty",
            "position_qty_fill",
            "limit_px",
            "stop_px",
            "take_px",
            "conviction",
            "dqs_day",
            "ib_defined",
            "phase_signal",
            "phase_fill",
            "tod_signal",
            "tod_fill",
            "minute_of_day_signal",
            "minute_of_day_fill",
            "pending_order_id",
            "quarantine_applied",
        }
        self.assertTrue(required.issubset(set(dump.keys())))
        self.assertEqual(dump["run_context"]["candidate_id"], "c0")
        self.assertEqual(dump["px_source_name"], "next_open")
        self.assertTrue(np.isnan(float(dump["px_value"])))


if __name__ == "__main__":
    unittest.main()
