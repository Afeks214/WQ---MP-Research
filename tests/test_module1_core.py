import unittest

import numpy as np

from weightiz_module1_core import (
    EngineConfig,
    NS_PER_MIN,
    _build_session_clock_reference,
    build_session_clock_vectorized,
    deterministic_digest_sha256,
    preallocate_state,
)


def _minute_range_utc(start_iso: str, end_iso: str) -> np.ndarray:
    start_ns = np.datetime64(start_iso, "ns").astype(np.int64)
    end_ns = np.datetime64(end_iso, "ns").astype(np.int64)
    n = int((end_ns - start_ns) // np.int64(NS_PER_MIN)) + 1
    return start_ns + np.arange(n, dtype=np.int64) * np.int64(NS_PER_MIN)


def _cfg(T: int, A: int = 2) -> EngineConfig:
    return EngineConfig(
        T=T,
        A=A,
        B=240,
        tick_size=np.full(A, 0.01, dtype=np.float64),
        mode="sealed",
        timezone="America/New_York",
    )


class TestModule1Core(unittest.TestCase):
    def test_deterministic_replay_digest(self):
        ts_ns = _minute_range_utc("2024-01-03T14:30:00", "2024-01-03T20:00:00")
        cfg = _cfg(ts_ns.shape[0], A=3)
        symbols = ("A0", "A1", "A2")

        st1 = preallocate_state(ts_ns=ts_ns, cfg=cfg, symbols=symbols)
        st2 = preallocate_state(ts_ns=ts_ns, cfg=cfg, symbols=symbols)

        self.assertEqual(deterministic_digest_sha256(st1), deterministic_digest_sha256(st2))

    def test_dst_boundary_invariants_america_new_york(self):
        dst_start = _minute_range_utc("2024-03-10T06:30:00", "2024-03-10T08:30:00")
        dst_end = _minute_range_utc("2024-11-03T05:30:00", "2024-11-03T07:30:00")
        open_markers = np.asarray(
            [
                np.datetime64("2024-03-08T14:30:00", "ns").astype(np.int64),
                np.datetime64("2024-03-11T13:30:00", "ns").astype(np.int64),
                np.datetime64("2024-11-01T13:30:00", "ns").astype(np.int64),
                np.datetime64("2024-11-04T14:30:00", "ns").astype(np.int64),
            ],
            dtype=np.int64,
        )
        ts_ns = np.unique(np.concatenate([dst_start, dst_end, open_markers])).astype(np.int64)

        cfg = _cfg(ts_ns.shape[0], A=2)
        clk = build_session_clock_vectorized(ts_ns, cfg, tz_name="America/New_York")

        self.assertTrue(np.all(np.diff(clk["session_id"]) >= 0))

        open_mask = clk["minute_of_day"] == np.int16(cfg.rth_open_minute)
        self.assertTrue(np.any(open_mask))
        np.testing.assert_array_equal(
            clk["tod"][open_mask],
            np.zeros(int(np.sum(open_mask)), dtype=np.int16),
        )

        self.assertTrue(np.all((clk["minute_of_day"] >= 0) & (clk["minute_of_day"] <= 1439)))

    def test_gap_reset_triggers_on_six_minute_gap(self):
        ts_ns = np.asarray(
            [
                np.datetime64("2024-01-03T14:30:00", "ns").astype(np.int64),
                np.datetime64("2024-01-03T14:31:00", "ns").astype(np.int64),
                np.datetime64("2024-01-03T14:32:00", "ns").astype(np.int64),
                np.datetime64("2024-01-03T14:38:00", "ns").astype(np.int64),
                np.datetime64("2024-01-03T14:39:00", "ns").astype(np.int64),
            ],
            dtype=np.int64,
        )
        cfg = _cfg(ts_ns.shape[0], A=2)
        state = preallocate_state(ts_ns=ts_ns, cfg=cfg, symbols=("A0", "A1"))

        self.assertAlmostEqual(float(state.gap_min[3]), 6.0, places=12)
        self.assertEqual(int(state.reset_flag[3]), 1)
        self.assertEqual(int(state.reset_flag[0]), 1)

    def test_clock_override_validation_missing_key_wrong_dtype_wrong_shape_and_success(self):
        ts_ns = _minute_range_utc("2024-01-03T14:30:00", "2024-01-03T14:39:00")
        cfg = _cfg(ts_ns.shape[0], A=2)
        base_clk = build_session_clock_vectorized(ts_ns, cfg)
        symbols = ("A0", "A1")

        missing_key = {k: v.copy() for k, v in base_clk.items()}
        del missing_key["phase"]
        with self.assertRaisesRegex(RuntimeError, "keys mismatch"):
            preallocate_state(ts_ns=ts_ns, cfg=cfg, symbols=symbols, clock_override=missing_key)

        wrong_dtype = {k: v.copy() for k, v in base_clk.items()}
        wrong_dtype["minute_of_day"] = wrong_dtype["minute_of_day"].astype(np.int32)
        with self.assertRaisesRegex(RuntimeError, "dtype mismatch"):
            preallocate_state(ts_ns=ts_ns, cfg=cfg, symbols=symbols, clock_override=wrong_dtype)

        wrong_shape = {k: v.copy() for k, v in base_clk.items()}
        wrong_shape["tod"] = wrong_shape["tod"][:-1]
        with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
            preallocate_state(ts_ns=ts_ns, cfg=cfg, symbols=symbols, clock_override=wrong_shape)

        good_override = {k: v.copy() for k, v in base_clk.items()}
        state = preallocate_state(ts_ns=ts_ns, cfg=cfg, symbols=symbols, clock_override=good_override)
        self.assertEqual(state.minute_of_day.shape[0], ts_ns.shape[0])

    def test_portfolio_invariants_at_init(self):
        ts_ns = _minute_range_utc("2024-01-03T14:30:00", "2024-01-03T14:45:00")
        cfg = _cfg(ts_ns.shape[0], A=3)
        state = preallocate_state(ts_ns=ts_ns, cfg=cfg, symbols=("A0", "A1", "A2"))

        implied_bp = state.equity * state.leverage_limit - state.margin_used
        np.testing.assert_allclose(state.buying_power, implied_bp, rtol=0.0, atol=1e-12)
        self.assertAlmostEqual(
            float(state.buying_power[0]),
            float(cfg.initial_cash * cfg.intraday_leverage_max),
            places=12,
        )
        self.assertEqual(float(state.margin_used[0]), 0.0)
        self.assertEqual(float(state.equity[0]), float(cfg.initial_cash))

    def test_reference_vs_fast_clock_equivalence_across_dst(self):
        ts_start = _minute_range_utc("2024-03-09T18:00:00", "2024-03-11T18:00:00")
        ts_end = _minute_range_utc("2024-11-02T18:00:00", "2024-11-04T18:00:00")
        ts_ns = np.unique(np.concatenate([ts_start, ts_end])).astype(np.int64)

        cfg = _cfg(ts_ns.shape[0], A=2)
        fast_clk = build_session_clock_vectorized(ts_ns, cfg, tz_name="America/New_York")
        ref_clk = _build_session_clock_reference(ts_ns, cfg, tz_name="America/New_York")

        for key in ("minute_of_day", "tod", "session_id", "gap_min", "reset_flag", "phase"):
            np.testing.assert_array_equal(fast_clk[key], ref_clk[key], err_msg=f"clock mismatch for {key}")


if __name__ == "__main__":
    unittest.main()
