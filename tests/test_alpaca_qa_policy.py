import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from scripts.fetch_alpaca_data import run_fetch


class _FakeClientSuccess:
    def __init__(self, *args, **kwargs):
        pass

    def fetch_bars_multi_with_meta(
        self,
        symbols,
        timeframe,
        start,
        end,
        feed,
        adjustment,
        sort="asc",
        max_symbols_per_request=100,
        limit=10_000,
    ):
        bars = {}
        metas = {}
        for sym in symbols:
            bars[sym] = [
                {"t": "2024-03-11T13:30:00Z", "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.5, "v": 10},
                {"t": "2024-03-11T13:31:00Z", "o": 100.5, "h": 101.2, "l": 100.4, "c": 101.0, "v": 12},
            ]
            metas[sym] = {
                "limit_requested": int(limit),
                "limit_effective": int(limit),
                "limit_fallback_used": False,
                "limit_fallback_reason": "",
                "retry_429_count": 0,
                "total_sleep_seconds": 0.0,
                "backoff_schedule_seconds": [],
            }
        feed_map = {str(s): str(feed) for s in symbols}
        return bars, feed_map, [], metas


def _fake_clean_df():
    import numpy as np
    import pandas as pd

    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-03-11T13:30:00Z", "2024-03-11T13:31:00Z"],
                utc=True,
            ),
            "open": np.asarray([100.0, 100.5], dtype=np.float64),
            "high": np.asarray([101.0, 101.2], dtype=np.float64),
            "low": np.asarray([99.0, 100.4], dtype=np.float64),
            "close": np.asarray([100.5, 101.0], dtype=np.float64),
            "volume": np.asarray([10.0, 12.0], dtype=np.float64),
        }
    )


def _fake_canonicalize(records, symbol, timezone, session_policy, rth_open, rth_close, rth_close_inclusive=False, calendar_mode="naive"):
    return _fake_clean_df(), {
        "symbol": str(symbol).upper(),
        "invariants_ok": True,
        "session": {
            "coverage_pct": 100.0,
            "missing_minutes_total": 0,
            "missing_minutes_pct": 0.0,
            "missing_minutes_preview": [],
        },
    }


def _fake_post_clean(*args, **kwargs):
    return {
        "canonical_columns_ok": True,
        "nan_count": 0,
        "duplicate_minutes_after_clean": 0,
        "timestamps_monotonic_increasing": True,
        "timestamps_utc_tz_aware": True,
    }


class TestAlpacaQAPolicy(unittest.TestCase):
    def _write_config(
        self,
        root: Path,
        symbols: list[str],
        qa_policy: str,
        coverage_min_pct: float,
        min_ok_symbols: int,
    ) -> Path:
        cfg = {
            "alpaca": {
                "api_key_env": "ALPACA_API_KEY",
                "secret_key_env": "ALPACA_SECRET_KEY",
                "base_url": "https://data.alpaca.markets",
                "feed": "iex",
                "adjustment": "raw",
                "timeframe": "1Min",
                "sort": "asc",
                "limit_per_page": 10000,
                "start": "2024-03-08T00:00:00Z",
                "end": "2024-03-14T23:59:59Z",
                "symbols": list(symbols),
                "session_policy": "RTH",
                "timezone": "America/New_York",
                "rth_open": "09:30",
                "rth_close": "16:00",
                "rth_close_inclusive": False,
                "calendar_mode": "nyse",
                "qa_policy": str(qa_policy),
                "coverage_min_pct": float(coverage_min_pct),
                "rate_limit_sleep_sec": 0.0,
                "max_symbols_per_request": 100,
                "max_retries_429": 2,
                "backoff_base_sec": 0.5,
                "backoff_max_sec": 8.0,
                "failure_rate_threshold": 1.0,
                "min_ok_symbols": int(min_ok_symbols),
            },
            "storage": {
                "root": str(root),
                "write_format": "csv",
                "overwrite_clean": False,
            },
        }
        config_path = root / "cfg.yaml"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        return config_path

    @staticmethod
    def _load_json(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    def test_staging_written_when_qa_fails_coverage_threshold(self):
        def _low_cov(*args, **kwargs):
            return {
                "coverage_pct": 95.0,
                "missing_minutes_total": 10,
                "missing_minutes_pct": 5.0,
                "missing_minutes_preview": ["2024-03-11T09:32"],
            }

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = self._write_config(
                root=root,
                symbols=["SPY"],
                qa_policy="coverage_threshold",
                coverage_min_pct=99.0,
                min_ok_symbols=1,
            )
            with patch.dict(
                os.environ, {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}, clear=False
            ), patch("scripts.fetch_alpaca_data.AlpacaClient", _FakeClientSuccess), patch(
                "scripts.fetch_alpaca_data.canonicalize_alpaca_bars", _fake_canonicalize
            ), patch(
                "scripts.fetch_alpaca_data.run_post_clean_qa_or_raise", _fake_post_clean
            ), patch(
                "scripts.fetch_alpaca_data.summarize_session_meta_for_clean_frame", _low_cov
            ):
                with self.assertRaisesRegex(RuntimeError, "all_symbols_failed"):
                    run_fetch(cfg)

            staging = root / "clean_staging" / "SPY.csv"
            clean = root / "clean" / "SPY.csv"
            self.assertTrue(staging.exists())
            self.assertFalse(clean.exists())
            qa_doc = self._load_json(sorted((root / "reports").glob("*_qa.json"))[0])
            self.assertEqual(str(qa_doc["symbols"]["SPY"]["status"]), "failed_qa")

    def test_promotion_occurs_when_qa_passes(self):
        def _high_cov(*args, **kwargs):
            return {
                "coverage_pct": 99.5,
                "missing_minutes_total": 1,
                "missing_minutes_pct": 0.5,
                "missing_minutes_preview": ["2024-03-11T09:32"],
            }

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = self._write_config(
                root=root,
                symbols=["SPY"],
                qa_policy="coverage_threshold",
                coverage_min_pct=99.0,
                min_ok_symbols=1,
            )
            with patch.dict(
                os.environ, {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}, clear=False
            ), patch("scripts.fetch_alpaca_data.AlpacaClient", _FakeClientSuccess), patch(
                "scripts.fetch_alpaca_data.canonicalize_alpaca_bars", _fake_canonicalize
            ), patch(
                "scripts.fetch_alpaca_data.run_post_clean_qa_or_raise", _fake_post_clean
            ), patch(
                "scripts.fetch_alpaca_data.summarize_session_meta_for_clean_frame", _high_cov
            ):
                summary = run_fetch(cfg)

            self.assertEqual(str(summary["exit_reason"]), "")
            self.assertTrue((root / "clean_staging" / "SPY.csv").exists())
            self.assertTrue((root / "clean" / "SPY.csv").exists())

    def test_coverage_threshold_run_exits_zero_when_ok_symbols_meet_gate(self):
        def _coverage_by_rows(clean, *args, **kwargs):
            # SPY has 2 rows (pass), QQQ has 1 row (fail)
            nrows = int(clean.shape[0])
            if nrows >= 2:
                return {
                    "coverage_pct": 99.2,
                    "missing_minutes_total": 1,
                    "missing_minutes_pct": 0.8,
                    "missing_minutes_preview": [],
                }
            return {
                "coverage_pct": 90.0,
                "missing_minutes_total": 20,
                "missing_minutes_pct": 10.0,
                "missing_minutes_preview": ["2024-03-11T09:40"],
            }

        def _canonicalize_by_symbol(records, symbol, *args, **kwargs):
            df = _fake_clean_df()
            if str(symbol).upper() == "QQQ":
                df = df.iloc[:1].copy()
            return df, {"symbol": str(symbol).upper(), "invariants_ok": True, "session": {}}

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = self._write_config(
                root=root,
                symbols=["SPY", "QQQ"],
                qa_policy="coverage_threshold",
                coverage_min_pct=99.0,
                min_ok_symbols=1,
            )
            with patch.dict(
                os.environ, {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}, clear=False
            ), patch("scripts.fetch_alpaca_data.AlpacaClient", _FakeClientSuccess), patch(
                "scripts.fetch_alpaca_data.canonicalize_alpaca_bars", _canonicalize_by_symbol
            ), patch(
                "scripts.fetch_alpaca_data.run_post_clean_qa_or_raise", _fake_post_clean
            ), patch(
                "scripts.fetch_alpaca_data.summarize_session_meta_for_clean_frame", _coverage_by_rows
            ):
                summary = run_fetch(cfg)

            self.assertEqual(str(summary["exit_reason"]), "")
            self.assertTrue((root / "clean" / "SPY.csv").exists())
            self.assertFalse((root / "clean" / "QQQ.csv").exists())

    def test_strict_no_holes_fails_and_does_not_promote(self):
        def _holes(*args, **kwargs):
            return {
                "coverage_pct": 99.9,
                "missing_minutes_total": 1,
                "missing_minutes_pct": 0.1,
                "missing_minutes_preview": ["2024-03-11T09:35"],
            }

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = self._write_config(
                root=root,
                symbols=["SPY"],
                qa_policy="strict_no_holes",
                coverage_min_pct=99.0,
                min_ok_symbols=1,
            )
            with patch.dict(
                os.environ, {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}, clear=False
            ), patch("scripts.fetch_alpaca_data.AlpacaClient", _FakeClientSuccess), patch(
                "scripts.fetch_alpaca_data.canonicalize_alpaca_bars", _fake_canonicalize
            ), patch(
                "scripts.fetch_alpaca_data.run_post_clean_qa_or_raise", _fake_post_clean
            ), patch(
                "scripts.fetch_alpaca_data.summarize_session_meta_for_clean_frame", _holes
            ):
                with self.assertRaisesRegex(RuntimeError, "all_symbols_failed"):
                    run_fetch(cfg)

            self.assertTrue((root / "clean_staging" / "SPY.csv").exists())
            self.assertFalse((root / "clean" / "SPY.csv").exists())
            qa_doc = self._load_json(sorted((root / "reports").glob("*_qa.json"))[0])
            self.assertEqual(str(qa_doc["symbols"]["SPY"]["status"]), "failed_qa")
            self.assertIn("strict_no_holes", str(qa_doc["symbols"]["SPY"]["qa_fail_reason"]))


if __name__ == "__main__":
    unittest.main()
