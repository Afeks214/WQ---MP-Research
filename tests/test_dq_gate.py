from __future__ import annotations

import unittest

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore[assignment]

from weightiz_dq import DQ_ACCEPT, DQ_DEGRADE, DQ_REJECT, dq_apply, dq_validate


@unittest.skipIf(pd is None, "pandas not available")
class TestDQGate(unittest.TestCase):
    def _session_index(self, day_local: str, freq: str = "1min") -> "pd.DatetimeIndex":
        start = pd.Timestamp(f"{day_local} 09:30:00", tz="America/New_York")
        end = pd.Timestamp(f"{day_local} 15:45:00", tz="America/New_York")
        return pd.date_range(start, end, freq=freq, tz="America/New_York").tz_convert("UTC")

    def _frame_from_index(self, idx: "pd.DatetimeIndex", base: float = 100.0) -> "pd.DataFrame":
        n = int(idx.shape[0])
        t = np.arange(n, dtype=np.float64)
        close = base + 0.01 * t
        open_px = close - 0.01
        high = np.maximum(open_px, close) + 0.02
        low = np.minimum(open_px, close) - 0.02
        volume = 1000.0 + (t % 11.0)
        return pd.DataFrame(
            {
                "open": open_px,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=idx,
        )

    def test_normal_session_expected_bars(self) -> None:
        idx = self._session_index("2024-01-03", freq="1min")
        df = self._frame_from_index(idx)
        reports = dq_validate(
            df=df,
            symbol="SPY",
            tz_name="America/New_York",
            session_open_minute=570,
            session_close_minute=945,
            timeframe_min=None,
        )
        self.assertEqual(len(reports), 1)
        rep = reports[0]
        self.assertEqual(rep.timeframe_min, 1)
        self.assertEqual(rep.expected_bars_nominal, 376)
        self.assertEqual(rep.missing_bars_nominal, 0)
        self.assertEqual(rep.decision, DQ_ACCEPT)

    def test_short_session_inferred_degrade_not_reject(self) -> None:
        idx = pd.date_range(
            pd.Timestamp("2024-11-29 09:30:00", tz="America/New_York"),
            pd.Timestamp("2024-11-29 12:00:00", tz="America/New_York"),
            freq="1min",
            tz="America/New_York",
        ).tz_convert("UTC")
        df = self._frame_from_index(idx)
        reports = dq_validate(
            df=df,
            symbol="SPY",
            tz_name="America/New_York",
            session_open_minute=570,
            session_close_minute=945,
            timeframe_min=None,
        )
        self.assertEqual(len(reports), 1)
        rep = reports[0]
        self.assertEqual(rep.decision, DQ_DEGRADE)
        self.assertTrue(rep.short_session_inferred)
        self.assertIn("SHORT_SESSION_INFERRED", rep.reason_codes)
        self.assertGreater(rep.missing_pct_nominal, 20.0)

    def test_timeframe_5m_expected_bars(self) -> None:
        idx = self._session_index("2024-01-04", freq="5min")
        df = self._frame_from_index(idx)
        reports = dq_validate(
            df=df,
            symbol="QQQ",
            tz_name="America/New_York",
            session_open_minute=570,
            session_close_minute=945,
            timeframe_min=None,
        )
        rep = reports[0]
        self.assertEqual(rep.timeframe_min, 5)
        self.assertEqual(rep.expected_bars_nominal, 76)
        self.assertEqual(rep.missing_bars_nominal, 0)
        self.assertEqual(rep.decision, DQ_ACCEPT)

    def test_gap_event_clustering(self) -> None:
        idx = self._session_index("2024-01-05", freq="1min")
        df = self._frame_from_index(idx)

        miss_consecutive = {
            pd.Timestamp("2024-01-05 10:00:00", tz="America/New_York").tz_convert("UTC"),
            pd.Timestamp("2024-01-05 10:01:00", tz="America/New_York").tz_convert("UTC"),
            pd.Timestamp("2024-01-05 10:02:00", tz="America/New_York").tz_convert("UTC"),
            pd.Timestamp("2024-01-05 10:03:00", tz="America/New_York").tz_convert("UTC"),
            pd.Timestamp("2024-01-05 10:04:00", tz="America/New_York").tz_convert("UTC"),
        }
        d1 = df.loc[~df.index.isin(miss_consecutive)].copy()
        rep1 = dq_validate(
            df=d1,
            symbol="IWM",
            tz_name="America/New_York",
            session_open_minute=570,
            session_close_minute=945,
            timeframe_min=None,
        )[0]
        self.assertEqual(rep1.decision, DQ_DEGRADE)
        self.assertEqual(rep1.gap_events, 1)

        miss_separated = {
            pd.Timestamp("2024-01-05 10:00:00", tz="America/New_York").tz_convert("UTC"),
            pd.Timestamp("2024-01-05 10:10:00", tz="America/New_York").tz_convert("UTC"),
            pd.Timestamp("2024-01-05 10:20:00", tz="America/New_York").tz_convert("UTC"),
            pd.Timestamp("2024-01-05 10:30:00", tz="America/New_York").tz_convert("UTC"),
            pd.Timestamp("2024-01-05 10:40:00", tz="America/New_York").tz_convert("UTC"),
        }
        d2 = df.loc[~df.index.isin(miss_separated)].copy()
        rep2 = dq_validate(
            df=d2,
            symbol="IWM",
            tz_name="America/New_York",
            session_open_minute=570,
            session_close_minute=945,
            timeframe_min=None,
        )[0]
        self.assertEqual(rep2.decision, DQ_DEGRADE)
        self.assertEqual(rep2.gap_events, 5)

    def test_volatility_aware_bad_tick(self) -> None:
        idx = self._session_index("2024-01-08", freq="1min")
        n = int(idx.shape[0])

        # High-vol day: large but persistent swings should not auto-reject.
        close_hv = np.empty(n, dtype=np.float64)
        close_hv[0] = 100.0
        for i in range(1, n):
            close_hv[i] = close_hv[i - 1] * (1.0 + (0.11 if (i % 2 == 0) else -0.11))
        open_hv = np.r_[close_hv[0], close_hv[:-1]]
        high_hv = np.maximum(open_hv, close_hv) + 0.01
        low_hv = np.minimum(open_hv, close_hv) - 0.01
        vol = np.full(n, 1000.0, dtype=np.float64)
        df_hv = pd.DataFrame(
            {"open": open_hv, "high": high_hv, "low": low_hv, "close": close_hv, "volume": vol},
            index=idx,
        )
        rep_hv = dq_validate(
            df=df_hv,
            symbol="QQQ",
            tz_name="America/New_York",
            session_open_minute=570,
            session_close_minute=945,
            timeframe_min=None,
        )[0]
        self.assertNotEqual(rep_hv.decision, DQ_REJECT)

        # Low-vol day with one extreme jump should reject.
        close_lv = 100.0 + 0.01 * np.arange(n, dtype=np.float64)
        close_lv[n // 2] = close_lv[n // 2 - 1] * 1.30
        open_lv = np.r_[close_lv[0], close_lv[:-1]]
        high_lv = np.maximum(open_lv, close_lv) + 0.01
        low_lv = np.minimum(open_lv, close_lv) - 0.01
        df_lv = pd.DataFrame(
            {"open": open_lv, "high": high_lv, "low": low_lv, "close": close_lv, "volume": vol},
            index=idx,
        )
        rep_lv = dq_validate(
            df=df_lv,
            symbol="QQQ",
            tz_name="America/New_York",
            session_open_minute=570,
            session_close_minute=945,
            timeframe_min=None,
        )[0]
        self.assertEqual(rep_lv.decision, DQ_REJECT)
        self.assertIn("BAD_TICK_EXTREME", rep_lv.reason_codes)

    def test_intrasession_fill_no_overnight_carry(self) -> None:
        idx1 = self._session_index("2024-01-09", freq="1min")
        idx2 = self._session_index("2024-01-10", freq="1min")

        d1 = self._frame_from_index(idx1, base=100.0)
        d2 = self._frame_from_index(idx2, base=200.0)

        miss = {
            pd.Timestamp("2024-01-10 09:30:00", tz="America/New_York").tz_convert("UTC"),
            pd.Timestamp("2024-01-10 09:32:00", tz="America/New_York").tz_convert("UTC"),
        }
        d2 = d2.loc[~d2.index.isin(miss)].copy()

        df = pd.concat([d1, d2], axis=0).sort_index(kind="mergesort")

        reports = dq_validate(
            df=df,
            symbol="SPY",
            tz_name="America/New_York",
            session_open_minute=570,
            session_close_minute=945,
            timeframe_min=None,
        )
        repaired, _day_reports, flags = dq_apply(df=df, reports=reports, tz_name="America/New_York")

        ts_930 = pd.Timestamp("2024-01-10 09:30:00", tz="America/New_York").tz_convert("UTC")
        ts_931 = pd.Timestamp("2024-01-10 09:31:00", tz="America/New_York").tz_convert("UTC")
        ts_932 = pd.Timestamp("2024-01-10 09:32:00", tz="America/New_York").tz_convert("UTC")

        self.assertNotIn(ts_930, repaired.index)
        self.assertIn(ts_931, repaired.index)
        self.assertIn(ts_932, repaired.index)

        close_931 = float(repaired.loc[ts_931, "close"])
        row_932 = repaired.loc[ts_932]
        self.assertAlmostEqual(float(row_932["open"]), close_931, places=10)
        self.assertAlmostEqual(float(row_932["high"]), close_931, places=10)
        self.assertAlmostEqual(float(row_932["low"]), close_931, places=10)
        self.assertAlmostEqual(float(row_932["close"]), close_931, places=10)
        self.assertEqual(float(row_932["volume"]), 0.0)

        f932 = flags.loc[flags["timestamp"] == ts_932]
        self.assertEqual(int(f932.shape[0]), 1)
        self.assertTrue(bool(f932.iloc[0]["dq_filled_bar"]))

    def test_sparse_regular_day_uses_cadence_aware_expected(self) -> None:
        # Day 1 dense 1m bars + day 2 sparse-but-regular 10m bars.
        idx1 = self._session_index("2024-01-11", freq="1min")
        idx2 = self._session_index("2024-01-12", freq="10min")
        d1 = self._frame_from_index(idx1, base=100.0)
        d2 = self._frame_from_index(idx2, base=200.0)
        df = pd.concat([d1, d2], axis=0).sort_index(kind="mergesort")

        reports = dq_validate(
            df=df,
            symbol="DIA",
            tz_name="America/New_York",
            session_open_minute=570,
            session_close_minute=945,
            timeframe_min=None,
        )
        by_day = {r.session_date: r for r in reports}
        rep_sparse = by_day["2024-01-12"]
        self.assertTrue(rep_sparse.cadence_day_stable)
        self.assertEqual(rep_sparse.cadence_day_min, 10)
        self.assertEqual(rep_sparse.expected_bars_reference, 38)
        self.assertEqual(rep_sparse.observed_bars, 38)
        self.assertNotEqual(rep_sparse.decision, DQ_REJECT)

    def test_bar_end_semantics_do_not_create_fake_gap(self) -> None:
        # Provider-like minute bars often start at 09:31 and end at 16:00 (bar-end timestamp).
        idx_1m = pd.date_range(
            pd.Timestamp("2024-01-16 09:31:00", tz="America/New_York"),
            pd.Timestamp("2024-01-16 16:00:00", tz="America/New_York"),
            freq="1min",
            tz="America/New_York",
        ).tz_convert("UTC")
        df_1m = self._frame_from_index(idx_1m, base=150.0)
        rep_1m = dq_validate(
            df=df_1m,
            symbol="SPY",
            tz_name="America/New_York",
            session_open_minute=570,
            session_close_minute=960,
            timeframe_min=None,
        )[0]
        self.assertEqual(rep_1m.expected_bars_effective, rep_1m.observed_bars)
        self.assertNotIn("MICRO_GAPS", rep_1m.reason_codes)

        idx_5m = pd.date_range(
            pd.Timestamp("2024-01-16 09:35:00", tz="America/New_York"),
            pd.Timestamp("2024-01-16 16:00:00", tz="America/New_York"),
            freq="5min",
            tz="America/New_York",
        ).tz_convert("UTC")
        df_5m = self._frame_from_index(idx_5m, base=250.0)
        rep_5m = dq_validate(
            df=df_5m,
            symbol="QQQ",
            tz_name="America/New_York",
            session_open_minute=570,
            session_close_minute=960,
            timeframe_min=None,
        )[0]
        self.assertEqual(rep_5m.expected_bars_effective, rep_5m.observed_bars)
        self.assertNotIn("MICRO_GAPS", rep_5m.reason_codes)

    def test_true_feed_drop_still_rejects_with_cadence_awareness(self) -> None:
        idx = self._session_index("2024-01-15", freq="10min")
        df = self._frame_from_index(idx, base=300.0)
        # Keep only every third bar: large true drop should still reject.
        keep = np.arange(df.shape[0], dtype=np.int64) % 3 == 0
        dropped = df.iloc[keep].copy()

        reports = dq_validate(
            df=dropped,
            symbol="GLD",
            tz_name="America/New_York",
            session_open_minute=570,
            session_close_minute=945,
            timeframe_min=10,
        )
        rep = reports[0]
        self.assertEqual(rep.decision, DQ_REJECT)
        self.assertIn("MISSING_OVER_20_PCT", rep.reason_codes)

        reports_again = dq_validate(
            df=dropped,
            symbol="GLD",
            tz_name="America/New_York",
            session_open_minute=570,
            session_close_minute=945,
            timeframe_min=10,
        )
        rep2 = reports_again[0]
        self.assertEqual(rep.cadence_day_min, rep2.cadence_day_min)
        self.assertEqual(rep.reason_codes, rep2.reason_codes)


if __name__ == "__main__":
    unittest.main()
