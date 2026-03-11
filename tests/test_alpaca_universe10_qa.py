import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import yaml

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

from weightiz.shared.io.market_data.cleaning import canonicalize_alpaca_bars, run_post_clean_qa_or_raise

HAS_XCALS = find_spec("exchange_calendars") is not None


@unittest.skipIf(pd is None, "pandas not available")
class TestAlpacaUniverse10QA(unittest.TestCase):
    def test_universe10_config_exists_and_has_expected_window_and_symbols(self):
        repo_root = Path(__file__).resolve().parents[1]
        cfg_path = repo_root / "configs" / "data_alpaca_universe10_3y.yaml"
        self.assertTrue(cfg_path.exists(), f"Missing config file: {cfg_path}")
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

        expected_symbols = ["SPY", "QQQ", "IWM", "DIA", "TLT", "HYG", "GLD", "XLF", "XLK", "XLE"]
        self.assertEqual(list(cfg["alpaca"]["symbols"]), expected_symbols)
        self.assertEqual(str(cfg["alpaca"]["start"]), "2023-01-01T00:00:00Z")
        self.assertEqual(str(cfg["alpaca"]["end"]), "2025-12-31T23:59:59Z")
        self.assertEqual(str(cfg["alpaca"]["calendar_mode"]), "nyse")
        self.assertEqual(int(cfg["alpaca"]["min_ok_symbols"]), 8)

    @unittest.skipUnless(HAS_XCALS, "exchange_calendars not available")
    def test_post_clean_qa_reports_missing_minute_holes_in_nyse_rth(self):
        records = [
            {"t": "2024-03-11T13:30:00Z", "o": 100.0, "h": 101.0, "l": 99.5, "c": 100.5, "v": 10},
            {"t": "2024-03-11T13:31:00Z", "o": 100.5, "h": 101.2, "l": 100.4, "c": 101.0, "v": 11},
        ]
        clean, qa = canonicalize_alpaca_bars(
            records=records,
            symbol="SPY",
            timezone="America/New_York",
            session_policy="RTH",
            rth_open="09:30",
            rth_close="16:00",
            rth_close_inclusive=False,
            calendar_mode="nyse",
        )
        self.assertEqual(int(clean.shape[0]), 2)
        self.assertGreater(int(qa["session"]["missing_minutes_total"]), 0)

        post = run_post_clean_qa_or_raise(
            clean=clean,
            session_meta=dict(qa["session"]),
            timezone="America/New_York",
            session_policy="RTH",
            rth_open="09:30",
            rth_close="16:00",
            rth_close_inclusive=False,
            calendar_mode="nyse",
        )
        self.assertGreater(int(post["missing_minutes_total"]), 0)

    def test_post_clean_qa_rejects_nan_non_monotonic_and_duplicates(self):
        base_ts = pd.to_datetime(["2024-01-02T14:30:00Z", "2024-01-02T14:31:00Z"], utc=True)

        df_nan = pd.DataFrame(
            {
                "timestamp": base_ts,
                "open": np.asarray([100.0, 101.0], dtype=np.float64),
                "high": np.asarray([101.0, 102.0], dtype=np.float64),
                "low": np.asarray([99.0, 100.0], dtype=np.float64),
                "close": np.asarray([100.5, 101.5], dtype=np.float64),
                "volume": np.asarray([10.0, np.nan], dtype=np.float64),
            }
        )
        with self.assertRaisesRegex(RuntimeError, "NaN values in canonical columns"):
            run_post_clean_qa_or_raise(
                clean=df_nan,
                session_meta={},
                timezone="America/New_York",
                session_policy="ETH",
                rth_open="09:30",
                rth_close="16:00",
            )

        df_non_mono = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-02T14:31:00Z", "2024-01-02T14:30:00Z"], utc=True),
                "open": np.asarray([101.0, 100.0], dtype=np.float64),
                "high": np.asarray([102.0, 101.0], dtype=np.float64),
                "low": np.asarray([100.0, 99.0], dtype=np.float64),
                "close": np.asarray([101.5, 100.5], dtype=np.float64),
                "volume": np.asarray([11.0, 10.0], dtype=np.float64),
            }
        )
        with self.assertRaisesRegex(RuntimeError, "timestamps are not strictly increasing"):
            run_post_clean_qa_or_raise(
                clean=df_non_mono,
                session_meta={},
                timezone="America/New_York",
                session_policy="ETH",
                rth_open="09:30",
                rth_close="16:00",
            )

        df_dup = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-02T14:30:00Z", "2024-01-02T14:30:00Z"], utc=True),
                "open": np.asarray([100.0, 100.0], dtype=np.float64),
                "high": np.asarray([101.0, 101.0], dtype=np.float64),
                "low": np.asarray([99.0, 99.0], dtype=np.float64),
                "close": np.asarray([100.5, 100.5], dtype=np.float64),
                "volume": np.asarray([10.0, 10.0], dtype=np.float64),
            }
        )
        with self.assertRaisesRegex(RuntimeError, "duplicate timestamps remain"):
            run_post_clean_qa_or_raise(
                clean=df_dup,
                session_meta={},
                timezone="America/New_York",
                session_policy="ETH",
                rth_open="09:30",
                rth_close="16:00",
            )

    def test_duplicates_are_collapsed_deterministically_before_post_clean_qa(self):
        records = [
            {"t": "2024-01-02T14:30:40Z", "o": 100.2, "h": 102.0, "l": 98.5, "c": 101.2, "v": 20},
            {"t": "2024-01-02T14:30:10Z", "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.5, "v": 10},
            {"t": "2024-01-02T14:31:00Z", "o": 101.2, "h": 101.5, "l": 100.8, "c": 101.0, "v": 30},
        ]
        clean, qa = canonicalize_alpaca_bars(
            records=records,
            symbol="SPY",
            timezone="America/New_York",
            session_policy="ETH",
            rth_open="09:30",
            rth_close="16:00",
            calendar_mode="naive",
        )
        self.assertEqual(int(clean.shape[0]), 2)
        first_row = clean.iloc[0]
        self.assertAlmostEqual(float(first_row["open"]), 100.0)
        self.assertAlmostEqual(float(first_row["close"]), 101.2)
        self.assertAlmostEqual(float(first_row["volume"]), 30.0)

        post = run_post_clean_qa_or_raise(
            clean=clean,
            session_meta=dict(qa["session"]),
            timezone="America/New_York",
            session_policy="ETH",
            rth_open="09:30",
            rth_close="16:00",
            calendar_mode="naive",
        )
        self.assertEqual(int(post["duplicate_minutes_after_clean"]), 0)

    def test_dst_canary_config_spans_dst_start_and_end(self):
        repo_root = Path(__file__).resolve().parents[1]
        cfg_path = repo_root / "configs" / "data_alpaca_canary_dst_both.yaml"
        self.assertTrue(cfg_path.exists(), f"Missing config file: {cfg_path}")
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        self.assertEqual(str(cfg["alpaca"]["start"]), "2024-03-08T00:00:00Z")
        self.assertEqual(str(cfg["alpaca"]["end"]), "2024-11-06T23:59:59Z")


if __name__ == "__main__":
    unittest.main()
