from __future__ import annotations

import unittest

import numpy as np

from weightiz_module1_core import EngineConfig, NS_PER_MIN, Phase, ProfileStatIdx, ScoreIdx, preallocate_state
from weightiz_module3_structure import ContextIdx, Module3Output, Struct30mIdx
from weightiz_module4_strategy_funnel import Module4Config, run_module4_strategy_funnel


class TestNoNanEquityOnInvalidBars(unittest.TestCase):
    def test_invalid_bar_with_nan_close_does_not_poison_equity(self) -> None:
        start_ns = np.datetime64("2025-01-06T14:30:00", "ns").astype(np.int64)
        ts_ns = start_ns + np.arange(8, dtype=np.int64) * np.int64(NS_PER_MIN)
        cfg = EngineConfig(T=8, A=1, B=120, tick_size=np.asarray([0.01], dtype=np.float64))
        st = preallocate_state(ts_ns=ts_ns, cfg=cfg, symbols=("AAA",))
        st.phase[:] = np.int8(Phase.LIVE)
        st.bar_valid[:] = True

        base = 100.0 + np.arange(8, dtype=np.float64)[:, None] * 0.05
        st.open_px[:] = base
        st.high_px[:] = base + 0.1
        st.low_px[:] = base - 0.1
        st.close_px[:] = base + 0.02
        st.volume[:] = 1000.0
        st.rvol[:] = 1.0
        st.atr_floor[:] = 0.5

        # Make a bar invalid with non-finite close/open; this must not poison equity.
        st.bar_valid[3, 0] = False
        st.close_px[3, 0] = np.nan
        st.open_px[3, 0] = np.nan

        st.scores[:] = 0.0
        st.profile_stats[:] = 0.0
        st.scores[:, :, int(ScoreIdx.SCORE_BO_LONG)] = 0.0
        st.scores[:, :, int(ScoreIdx.SCORE_BO_SHORT)] = 0.0
        st.profile_stats[:, :, int(ProfileStatIdx.GBREAK)] = 0.0
        st.profile_stats[:, :, int(ProfileStatIdx.GREJECT)] = 0.0

        c3 = int(ContextIdx.N_FIELDS)
        k3 = int(Struct30mIdx.N_FIELDS)
        ctx = np.zeros((8, 1, c3), dtype=np.float64)
        ctx[:, :, int(ContextIdx.CTX_X_VAH)] = 1.0
        ctx[:, :, int(ContextIdx.CTX_X_VAL)] = -1.0
        ctx[:, :, int(ContextIdx.CTX_VALID_RATIO)] = 1.0
        ctx[:, :, int(ContextIdx.CTX_TREND_GATE_SPREAD_MEAN)] = 0.0
        ctx[:, :, int(ContextIdx.CTX_POC_DRIFT_X)] = 0.0
        ctx[:, :, int(ContextIdx.CTX_POC_VS_PREV_VA)] = 0.0
        blocks = np.zeros((8, 1, k3), dtype=np.float64)

        m3 = Module3Output(
            block_id_t=np.arange(8, dtype=np.int64),
            block_seq_t=np.zeros(8, dtype=np.int16),
            block_end_flag_t=np.ones(8, dtype=bool),
            block_start_t_index_t=np.arange(8, dtype=np.int64),
            block_end_t_index_t=np.arange(8, dtype=np.int64),
            block_features_tak=blocks,
            block_valid_ta=np.ones((8, 1), dtype=bool),
            context_tac=ctx,
            context_valid_ta=np.ones((8, 1), dtype=bool),
            context_source_t_index_ta=np.tile(np.arange(8, dtype=np.int64)[:, None], (1, 1)),
            ib_defined_ta=np.ones((8, 1), dtype=bool),
        )

        run_module4_strategy_funnel(st, m3, Module4Config(entry_threshold=0.99, fail_on_non_finite_input=True))
        self.assertTrue(np.all(np.isfinite(st.equity)))
        self.assertTrue(np.all(np.isfinite(st.margin_used)))
        self.assertTrue(np.all(np.isfinite(st.buying_power)))


if __name__ == "__main__":
    unittest.main()
