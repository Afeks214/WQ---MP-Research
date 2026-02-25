import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from scripts.fetch_alpaca_data import run_fetch


class _FakeClientEntitlementLimited:
    def __init__(self, *args, **kwargs):
        self.calls = 0

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
        self.calls += 1
        sym = str(symbols[0])
        meta = {
            sym: {
                "limit_requested": int(limit),
                "limit_effective": int(limit),
                "limit_fallback_used": False,
                "limit_fallback_reason": "",
                "retry_429_count": 0,
                "total_sleep_seconds": 0.0,
                "backoff_schedule_seconds": [],
            }
        }
        # 1st call = recent preflight (has data), 2nd call = historical preflight (no data)
        if self.calls == 1:
            bars = {
                sym: [
                    {"t": "2026-02-25T15:00:00Z", "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 1},
                ]
            }
            return bars, {sym: str(feed)}, [], meta
        return {sym: []}, {sym: str(feed)}, [], meta


class _FakeClientPreflightOk:
    def __init__(self, *args, **kwargs):
        self.calls = 0

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
        self.calls += 1
        sym = str(symbols[0])
        meta = {
            sym: {
                "limit_requested": int(limit),
                "limit_effective": int(limit),
                "limit_fallback_used": False,
                "limit_fallback_reason": "",
                "retry_429_count": 0,
                "total_sleep_seconds": 0.0,
                "backoff_schedule_seconds": [],
            }
        }
        bars = {
            sym: [
                {"t": "2024-03-11T13:30:00Z", "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.5, "v": 10},
                {"t": "2024-03-11T13:31:00Z", "o": 100.5, "h": 101.2, "l": 100.4, "c": 101.0, "v": 11},
            ]
        }
        return bars, {sym: str(feed)}, [], meta


class TestAlpacaEntitlementPreflight(unittest.TestCase):
    def _write_config(self, root: Path, symbols: list[str]) -> Path:
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
                "calendar_mode": "naive",
                "qa_policy": "coverage_threshold",
                "coverage_min_pct": 0.01,
                "rate_limit_sleep_sec": 0.0,
                "max_symbols_per_request": 100,
                "max_retries_429": 2,
                "backoff_base_sec": 0.5,
                "backoff_max_sec": 8.0,
                "failure_rate_threshold": 0.2,
                "min_ok_symbols": 1,
            },
            "storage": {
                "root": str(root),
                "write_format": "csv",
                "overwrite_clean": False,
            },
        }
        path = root / "cfg.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        return path

    @staticmethod
    def _load(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    def test_entitlement_limited_fails_early_with_reports(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = self._write_config(root, ["SPY"])
            with patch.dict(
                os.environ,
                {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"},
                clear=False,
            ), patch("scripts.fetch_alpaca_data.AlpacaClient", _FakeClientEntitlementLimited):
                with self.assertRaisesRegex(RuntimeError, "historical_data_not_available_for_requested_range"):
                    run_fetch(cfg)

            qa_path = sorted((root / "reports").glob("*_qa.json"))[0]
            manifest_path = sorted((root / "reports").glob("*_manifest.json"))[0]
            qa = self._load(qa_path)
            manifest = self._load(manifest_path)

            self.assertEqual(str(qa["preflight_entitlement_class"]), "historical_denied_or_limited")
            self.assertGreater(int(qa["preflight_recent_bar_count"]), 0)
            self.assertEqual(int(qa["preflight_historical_bar_count"]), 0)
            self.assertTrue(str(qa["preflight_actionable_fix"]))
            self.assertEqual(
                str(qa["summary"]["abort_reason"]),
                "historical_data_not_available_for_requested_range",
            )

            self.assertEqual(str(manifest["preflight_entitlement_class"]), "historical_denied_or_limited")
            self.assertIn("preflight_recent_start", manifest)
            self.assertIn("preflight_historical_start", manifest)

    def test_entitlement_preflight_passes_when_historical_has_data(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = self._write_config(root, ["SPY"])
            with patch.dict(
                os.environ,
                {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"},
                clear=False,
            ), patch("scripts.fetch_alpaca_data.AlpacaClient", _FakeClientPreflightOk):
                summary = run_fetch(cfg)

            qa = self._load(Path(summary["qa_report"]))
            self.assertEqual(str(qa["preflight_entitlement_class"]), "ok")
            self.assertGreater(int(qa["preflight_historical_bar_count"]), 0)
            self.assertEqual(str(qa["symbols"]["SPY"]["status"]), "ok")


if __name__ == "__main__":
    unittest.main()
