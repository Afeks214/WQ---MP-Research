import unittest
import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

from weightiz_module6_data import (
    build_funnel_table,
    compute_episode_mfe_mae,
    leverage_utilization,
    x_to_price,
)


@unittest.skipIf(pd is None, "pandas not available")
class TestModule6Data(unittest.TestCase):
    def test_x_to_price(self):
        close = np.array([100.0, 101.0], dtype=np.float64)
        x = np.array([0.5, -1.0], dtype=np.float64)
        atr = np.array([2.0, 1.5], dtype=np.float64)
        out = x_to_price(close, x, atr)
        np.testing.assert_allclose(out, np.array([101.0, 99.5], dtype=np.float64), rtol=0.0, atol=1e-12)

    def test_leverage_utilization(self):
        equity = np.array([1000.0, 1200.0], dtype=np.float64)
        margin = np.array([400.0, 600.0], dtype=np.float64)
        bp = np.array([5600.0, 6600.0], dtype=np.float64)
        u_margin, u_bp = leverage_utilization(equity, margin, bp, leverage_ref=6.0)
        np.testing.assert_allclose(u_margin, np.array([400 / 6000, 600 / 7200]), atol=1e-12)
        np.testing.assert_allclose(u_bp, 1.0 - np.array([5600 / 6000, 6600 / 7200]), atol=1e-12)

    def test_compute_episode_mfe_mae(self):
        micro = pd.DataFrame(
            {
                "symbol": ["SPY"] * 6,
                "ts_ns": [1, 2, 3, 4, 5, 6],
                "close": [100.0, 101.0, 103.0, 102.0, 104.0, 105.0],
                "filled_qty": [0.0, 10.0, 0.0, 0.0, -10.0, 0.0],
            }
        )
        out = compute_episode_mfe_mae(micro, pd.DataFrame())
        self.assertEqual(len(out), 1)
        row = out.iloc[0]
        self.assertEqual(row["side"], "LONG")
        self.assertGreater(float(row["mfe"]), 0.0)
        self.assertLessEqual(float(row["mae"]), float(row["mfe"]))

    def test_build_funnel_table_from_micro(self):
        micro = pd.DataFrame(
            {
                "ts_ns": [10, 10, 10],
                "session_id": [1, 1, 1],
                "symbol": ["SPY", "QQQ", "TLT"],
                "dclip": [1.0, 0.4, -0.2],
                "z_delta": [1.2, 0.5, -0.3],
                "regime_primary": [2, 3, 1],
                "rvol": [1.5, 1.1, 0.9],
                "overnight_winner_flag": [1, 0, 0],
            }
        )
        out = build_funnel_table(None, micro, selected_session=1, selected_ts_ns=10)
        self.assertEqual(len(out), 3)
        self.assertIn("ocs", out.columns)
        self.assertIn("structural_weight", out.columns)
        self.assertEqual(int(out["is_winner"].sum()), 1)


if __name__ == "__main__":
    unittest.main()
