import unittest
import numpy as np
import pandas as pd

from weightiz.module1.core import EngineConfig, build_session_clock_vectorized


class TestTimezoneDST(unittest.TestCase):
    def test_zoneinfo_dst_open_alignment_start_and_end(self):
        # Around 2024 DST transitions:
        # - Before start: 2024-03-08 14:30 UTC == 09:30 ET
        # - After start : 2024-03-11 13:30 UTC == 09:30 ET
        # - Before end  : 2024-11-01 13:30 UTC == 09:30 ET
        # - After end   : 2024-11-04 14:30 UTC == 09:30 ET
        ts = np.asarray(
            [
                np.datetime64("2024-03-08T14:30:00", "ns").astype(np.int64),
                np.datetime64("2024-03-11T13:30:00", "ns").astype(np.int64),
                np.datetime64("2024-11-01T13:30:00", "ns").astype(np.int64),
                np.datetime64("2024-11-04T14:30:00", "ns").astype(np.int64),
            ],
            dtype=np.int64,
        )
        cfg = EngineConfig(T=ts.shape[0], A=2, tick_size=np.asarray([0.01, 0.01], dtype=np.float64), mode="sealed")
        clk = build_session_clock_vectorized(ts, cfg, tz_name="America/New_York")

        # RTH open minute should remain 09:30 ET across DST boundaries.
        self.assertTrue(np.all(clk["minute_of_day"] == 570))
        self.assertTrue(np.all(clk["tod"] == 0))
        self.assertListEqual(clk["session_id"].tolist(), [0, 1, 2, 3])

    def test_session_id_changes_only_on_local_date_change(self):
        ts = np.asarray(
            [
                np.datetime64("2024-03-11T13:30:00", "ns").astype(np.int64),  # 09:30 ET
                np.datetime64("2024-03-11T15:30:00", "ns").astype(np.int64),  # 11:30 ET
                np.datetime64("2024-03-11T19:59:00", "ns").astype(np.int64),  # 15:59 ET
                np.datetime64("2024-03-12T13:30:00", "ns").astype(np.int64),  # next day 09:30 ET
            ],
            dtype=np.int64,
        )
        cfg = EngineConfig(T=ts.shape[0], A=2, tick_size=np.asarray([0.01, 0.01], dtype=np.float64), mode="sealed")
        clk = build_session_clock_vectorized(ts, cfg, tz_name="America/New_York")

        self.assertListEqual(clk["session_id"].tolist(), [0, 0, 0, 1])

        local_dates = (
            pd.to_datetime(ts, utc=True)
            .tz_convert("America/New_York")
            .date
        )
        change = np.r_[True, local_dates[1:] != local_dates[:-1]]
        expected_sid = np.cumsum(change.astype(np.int64)) - 1
        np.testing.assert_array_equal(clk["session_id"], expected_sid.astype(np.int64))


if __name__ == "__main__":
    unittest.main()
