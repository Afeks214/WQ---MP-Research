import unittest
from importlib.util import find_spec

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

from weightiz_data.alpaca_client import AlpacaClient
from weightiz_data.cleaning import _apply_session_policy, canonicalize_alpaca_bars, deduplicate_canonical_minutes


HAS_XCALS = find_spec("exchange_calendars") is not None


@unittest.skipIf(pd is None, "pandas not available")
class TestAlpacaIngestCleaning(unittest.TestCase):
    def test_dedup_uses_true_intraminute_ordering_even_if_input_unsorted(self):
        records = [
            {"t": "2024-01-02T14:30:40Z", "o": 200.0, "h": 202.0, "l": 199.0, "c": 201.0, "v": 11},
            {"t": "2024-01-02T14:30:10Z", "o": 100.0, "h": 102.0, "l": 99.0, "c": 101.0, "v": 7},
        ]

        clean, qa = canonicalize_alpaca_bars(
            records=records,
            symbol="SPY",
            timezone="America/New_York",
            session_policy="ETH",
            rth_open="09:30",
            rth_close="16:00",
        )

        self.assertEqual(int(clean.shape[0]), 1)
        row0 = clean.iloc[0]
        self.assertAlmostEqual(float(row0["open"]), 100.0)
        self.assertAlmostEqual(float(row0["close"]), 201.0)
        self.assertAlmostEqual(float(row0["volume"]), 18.0)
        self.assertEqual(int(qa["duplicate_rows_collapsed"]), 1)

    def test_dedup_aggregation_volume_open_close_high_low(self):
        records = [
            {"t": "2024-01-02T14:30:10Z", "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.5, "v": 10},
            {"t": "2024-01-02T14:30:40Z", "o": 100.2, "h": 102.0, "l": 98.5, "c": 101.2, "v": 20},
            {"t": "2024-01-02T14:31:05Z", "o": 101.2, "h": 101.5, "l": 100.8, "c": 101.0, "v": 30},
        ]

        clean, qa = canonicalize_alpaca_bars(
            records=records,
            symbol="SPY",
            timezone="America/New_York",
            session_policy="ETH",
            rth_open="09:30",
            rth_close="16:00",
        )

        self.assertEqual(int(clean.shape[0]), 2)
        row0 = clean.iloc[0]
        self.assertAlmostEqual(float(row0["open"]), 100.0)
        self.assertAlmostEqual(float(row0["close"]), 101.2)
        self.assertAlmostEqual(float(row0["high"]), 102.0)
        self.assertAlmostEqual(float(row0["low"]), 98.5)
        self.assertAlmostEqual(float(row0["volume"]), 30.0)
        self.assertEqual(int(qa["duplicate_rows_collapsed"]), 1)

    def test_deduplicate_fails_closed_without_ts_raw_on_duplicate_minutes(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2024-01-02T14:30:00Z", "2024-01-02T14:30:00Z"],
                    utc=True,
                ),
                "open": np.asarray([100.0, 101.0], dtype=np.float64),
                "high": np.asarray([101.0, 102.0], dtype=np.float64),
                "low": np.asarray([99.0, 100.0], dtype=np.float64),
                "close": np.asarray([100.5, 101.5], dtype=np.float64),
                "volume": np.asarray([10.0, 20.0], dtype=np.float64),
            }
        )

        with self.assertRaisesRegex(RuntimeError, "Intraminute ordering is undefined without ts_raw"):
            deduplicate_canonical_minutes(df)

    def test_deduplicate_allows_missing_ts_raw_when_minutes_unique(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2024-01-02T14:30:00Z", "2024-01-02T14:31:00Z"],
                    utc=True,
                ),
                "open": np.asarray([100.0, 101.0], dtype=np.float64),
                "high": np.asarray([101.0, 102.0], dtype=np.float64),
                "low": np.asarray([99.0, 100.0], dtype=np.float64),
                "close": np.asarray([100.5, 101.5], dtype=np.float64),
                "volume": np.asarray([10.0, 20.0], dtype=np.float64),
            }
        )

        out, dup_count = deduplicate_canonical_minutes(df)
        self.assertEqual(int(out.shape[0]), 2)
        self.assertEqual(int(dup_count), 0)

    def test_rth_filtering_dst_boundary_is_stable(self):
        records = [
            # RTH open bars around DST start/end.
            {"t": "2024-03-08T14:30:00Z", "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 1},
            {"t": "2024-03-11T13:30:00Z", "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 1},
            {"t": "2024-11-01T13:30:00Z", "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 1},
            {"t": "2024-11-04T14:30:00Z", "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 1},
            # RTH close edge around DST period.
            {"t": "2024-03-11T19:59:00Z", "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 1},  # 15:59 ET
            {"t": "2024-03-11T20:00:00Z", "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 1},  # 16:00 ET
            # Off-hours bars that should be filtered by RTH policy.
            {"t": "2024-03-11T12:30:00Z", "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 1},
            {"t": "2024-11-04T22:00:00Z", "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 1},
        ]

        clean_exclusive, _ = canonicalize_alpaca_bars(
            records=records,
            symbol="QQQ",
            timezone="America/New_York",
            session_policy="RTH",
            rth_open="09:30",
            rth_close="16:00",
            rth_close_inclusive=False,
        )

        clean_inclusive, _ = canonicalize_alpaca_bars(
            records=records,
            symbol="QQQ",
            timezone="America/New_York",
            session_policy="RTH",
            rth_open="09:30",
            rth_close="16:00",
            rth_close_inclusive=True,
        )

        et_exclusive = clean_exclusive["timestamp"].dt.tz_convert("America/New_York")
        minute_of_day_exclusive = (et_exclusive.dt.hour * 60 + et_exclusive.dt.minute).to_numpy(dtype=np.int32)
        self.assertEqual(int(clean_exclusive.shape[0]), 5)
        self.assertTrue(np.all((minute_of_day_exclusive >= 570) & (minute_of_day_exclusive <= 959)))
        self.assertFalse(bool(np.any(minute_of_day_exclusive == 960)))

        et_inclusive = clean_inclusive["timestamp"].dt.tz_convert("America/New_York")
        minute_of_day_inclusive = (et_inclusive.dt.hour * 60 + et_inclusive.dt.minute).to_numpy(dtype=np.int32)
        self.assertEqual(int(clean_inclusive.shape[0]), 6)
        self.assertTrue(np.all((minute_of_day_inclusive >= 570) & (minute_of_day_inclusive <= 960)))
        self.assertTrue(bool(np.any(minute_of_day_inclusive == 960)))
        self.assertEqual(int(np.sum(minute_of_day_inclusive == 570)), 4)

    def test_index_monotonicity_and_ohlc_invariants(self):
        records = [
            # valid
            {"t": "2024-01-02T14:31:00Z", "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.5, "v": 10},
            # invalid high < max(open, close) -> dropped
            {"t": "2024-01-02T14:30:00Z", "o": 100.0, "h": 99.0, "l": 98.0, "c": 100.5, "v": 10},
            # invalid negative volume -> dropped
            {"t": "2024-01-02T14:32:00Z", "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.2, "v": -1},
            # valid later
            {"t": "2024-01-02T14:33:00Z", "o": 101.0, "h": 101.1, "l": 100.9, "c": 101.0, "v": 5},
        ]

        clean, qa = canonicalize_alpaca_bars(
            records=records,
            symbol="IWM",
            timezone="America/New_York",
            session_policy="ETH",
            rth_open="09:30",
            rth_close="16:00",
        )

        self.assertEqual(int(clean.shape[0]), 2)
        self.assertTrue(bool(clean["timestamp"].is_monotonic_increasing))

        o = clean["open"].to_numpy(dtype=np.float64)
        h = clean["high"].to_numpy(dtype=np.float64)
        l = clean["low"].to_numpy(dtype=np.float64)
        c = clean["close"].to_numpy(dtype=np.float64)
        v = clean["volume"].to_numpy(dtype=np.float64)

        self.assertTrue(np.all(h >= np.maximum(o, c)))
        self.assertTrue(np.all(l <= np.minimum(o, c)))
        self.assertTrue(np.all(v >= 0.0))
        self.assertTrue(bool(qa["invariants_ok"]))

    @unittest.skipUnless(HAS_XCALS, "exchange_calendars not available")
    def test_nyse_calendar_mode_uses_half_day_expected_minutes(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2024-11-29T14:30:00Z", "2024-11-29T17:59:00Z"],
                    utc=True,
                ),
                "open": np.asarray([1.0, 1.0], dtype=np.float64),
                "high": np.asarray([1.0, 1.0], dtype=np.float64),
                "low": np.asarray([1.0, 1.0], dtype=np.float64),
                "close": np.asarray([1.0, 1.0], dtype=np.float64),
                "volume": np.asarray([1.0, 1.0], dtype=np.float64),
            }
        )

        _, session = _apply_session_policy(
            df=df,
            timezone="America/New_York",
            session_policy="RTH",
            rth_open="09:30",
            rth_close="16:00",
            rth_close_inclusive=False,
            calendar_mode="nyse",
        )

        self.assertEqual(str(session["qa_mode"]), "nyse")
        self.assertEqual(int(session["expected_minutes_total"]), 210)
        self.assertEqual(int(session["observed_minutes_total"]), 2)
        self.assertEqual(int(session["missing_minutes_total"]), 208)
        self.assertEqual(int(session["n_sessions"]), 1)
        self.assertEqual(str(session["nyse_last_minute_policy"]), "close_minus_1")

        _, session_inclusive = _apply_session_policy(
            df=df,
            timezone="America/New_York",
            session_policy="RTH",
            rth_open="09:30",
            rth_close="16:00",
            rth_close_inclusive=True,
            calendar_mode="nyse",
        )
        self.assertEqual(bool(session_inclusive["rth_close_inclusive"]), True)
        self.assertEqual(bool(session_inclusive["rth_close_inclusive_effective"]), False)
        self.assertIn(
            "nyse_calendar_mode_ignores_rth_close_inclusive",
            list(session_inclusive.get("warnings", [])),
        )


class TestAlpacaClientRetry(unittest.TestCase):
    def test_retry_backoff_seconds_is_deterministic_and_bounded(self):
        values = [
            AlpacaClient._retry_backoff_seconds(attempt=k, base_sec=0.5, max_sec=8.0) for k in range(8)
        ]
        self.assertListEqual(values, [0.5, 1.0, 2.0, 4.0, 8.0, 8.0, 8.0, 8.0])


if __name__ == "__main__":
    unittest.main()
