import unittest
import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    import plotly.graph_objects as go  # noqa: F401
    HAVE_PLOTLY = True
except Exception:  # pragma: no cover
    HAVE_PLOTLY = False

from weightiz_module6_views import (
    build_brain_figure,
    build_funnel_bar_figure,
    build_macro_figure,
    build_micro_matrix_figure,
)


@unittest.skipIf((pd is None) or (not HAVE_PLOTLY), "pandas/plotly not available")
class TestModule6Figures(unittest.TestCase):
    def test_macro_figure(self):
        eq = pd.DataFrame(
            {
                "ts_ns": np.array([1, 2, 3, 4, 5], dtype=np.int64) * 60_000_000_000,
                "equity": [1000, 1010, 1020, 1015, 1030],
                "drawdown": [0.0, 0.0, 0.0, -0.005, 0.0],
                "margin_used": [100, 120, 140, 130, 150],
                "buying_power": [5900, 5880, 5860, 5870, 5850],
            }
        )
        daily = pd.DataFrame({"session_id": [1, 2, 3, 4], "task": [0.01, -0.005, 0.002, 0.004]})
        fig = build_macro_figure(eq, daily, "task", 2, 2, 0.0, 6.0, "America/New_York")
        self.assertGreaterEqual(len(fig.data), 4)

    def test_brain_figure(self):
        md = pd.DataFrame(
            {
                "ts_ns": np.array([1, 2, 3, 4], dtype=np.int64) * 60_000_000_000,
                "score_bo_long": [0.2, 0.7, 0.6, 0.3],
                "score_bo_short": [0.1, 0.1, 0.2, 0.1],
                "score_rej_long": [0.1, 0.2, 0.3, 0.2],
                "score_rej_short": [0.05, 0.06, 0.1, 0.07],
                "regime_primary": [1, 2, 2, 3],
                "intent_long": [0, 1, 1, 0],
                "intent_short": [0, 0, 0, 1],
            }
        )
        fig = build_brain_figure(md, entry_threshold=0.55, exit_threshold=0.25, timezone="America/New_York")
        self.assertGreaterEqual(len(fig.data), 2)

    def test_micro_matrix_figure(self):
        md = pd.DataFrame(
            {
                "ts_ns": np.array([1, 2, 3, 4], dtype=np.int64) * 60_000_000_000,
                "open": [100, 101, 102, 101],
                "high": [101, 102, 103, 102],
                "low": [99, 100, 101, 100],
                "close": [100.5, 101.5, 102.5, 101.2],
                "ctx_x_poc": [0.0, 0.0, 0.0, 0.0],
                "ctx_x_vah": [0.5, 0.5, 0.5, 0.5],
                "ctx_x_val": [-0.5, -0.5, -0.5, -0.5],
                "atr_eff": [1.0, 1.0, 1.0, 1.0],
                "z_delta": [0.2, -0.3, 0.5, -0.1],
            }
        )
        fig = build_micro_matrix_figure(md, profile_blocks_df=None, max_profile_blocks_render=10, timezone="America/New_York")
        self.assertGreaterEqual(len(fig.data), 2)

    def test_funnel_figure(self):
        ft = pd.DataFrame(
            {
                "symbol": ["SPY", "QQQ"],
                "ocs": [1.2, 0.8],
                "is_winner": [1, 0],
            }
        )
        fig = build_funnel_bar_figure(ft, timezone="America/New_York")
        self.assertGreaterEqual(len(fig.data), 1)


if __name__ == "__main__":
    unittest.main()
