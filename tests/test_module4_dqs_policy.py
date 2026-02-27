from __future__ import annotations

import unittest

import numpy as np

from weightiz_module1_core import EngineConfig, Phase, ProfileStatIdx, ScoreIdx, preallocate_state
from weightiz_module3_structure import ContextIdx, Module3Output, Struct30mIdx
from weightiz_module4_strategy_funnel import Module4Config, run_module4_strategy_funnel


def _mk_state(T: int = 20, A: int = 2):
    start_ns = np.datetime64("2025-01-06T14:30:00", "ns").astype(np.int64)
    ts_ns = start_ns + np.arange(T, dtype=np.int64) * np.int64(60_000_000_000)
    cfg = EngineConfig(T=T, A=A, B=240, tick_size=np.full(A, 0.01, dtype=np.float64))
    st = preallocate_state(ts_ns, cfg, tuple(f"A{i}" for i in range(A)))
    st.phase[:] = np.int8(Phase.LIVE)
    st.bar_valid[:] = True

    base = 100.0 + np.arange(T, dtype=np.float64)[:, None] * 0.01
    st.open_px[:] = base
    st.high_px[:] = base + 0.10
    st.low_px[:] = base - 0.10
    st.close_px[:] = base + 0.02
    st.volume[:] = 10_000.0
    st.rvol[:] = 1.2
    st.atr_floor[:] = 0.5

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
        context_valid_ta=valid.copy(),
        context_source_t_index_ta=src,
        ib_defined_ta=np.ones((T, A), dtype=bool),
    )


class TestModule4DQSPolicy(unittest.TestCase):
    def test_low_dqs_forces_neutral(self) -> None:
        st = _mk_state(T=16, A=2)
        m3 = _mk_m3(st)
        st.dqs_day_ta = np.full((st.cfg.T, st.cfg.A), 0.40, dtype=np.float64)

        out = run_module4_strategy_funnel(st, m3, Module4Config(entry_threshold=0.55))

        self.assertTrue(np.all(out.intent_long_ta == 0))
        self.assertTrue(np.all(out.intent_short_ta == 0))
        self.assertTrue(np.allclose(out.target_qty_ta, 0.0))

    def test_ib_missing_no_trade_forces_neutral(self) -> None:
        st = _mk_state(T=16, A=2)
        m3 = _mk_m3(st)
        st.dqs_day_ta = np.ones((st.cfg.T, st.cfg.A), dtype=np.float64)
        m3.ib_defined_ta[:, 0] = False

        out = run_module4_strategy_funnel(st, m3, Module4Config(entry_threshold=0.55))

        self.assertTrue(np.all(out.intent_long_ta[:, 0] == 0))
        self.assertTrue(np.allclose(out.target_qty_ta[:, 0], 0.0))

    def test_effective_conviction_scaled_by_dqs(self) -> None:
        st_low = _mk_state(T=16, A=1)
        m3_low = _mk_m3(st_low)
        st_low.dqs_day_ta = np.full((st_low.cfg.T, st_low.cfg.A), 0.60, dtype=np.float64)  # 0.8*0.6=0.48 < 0.55
        out_low = run_module4_strategy_funnel(st_low, m3_low, Module4Config(entry_threshold=0.55))
        self.assertTrue(np.all(out_low.intent_long_ta == 0))

        st_hi = _mk_state(T=16, A=1)
        m3_hi = _mk_m3(st_hi)
        st_hi.dqs_day_ta = np.full((st_hi.cfg.T, st_hi.cfg.A), 0.80, dtype=np.float64)  # 0.8*0.8=0.64 > 0.55
        out_hi = run_module4_strategy_funnel(st_hi, m3_hi, Module4Config(entry_threshold=0.55))
        self.assertTrue(np.any(out_hi.intent_long_ta))


if __name__ == "__main__":
    unittest.main()
