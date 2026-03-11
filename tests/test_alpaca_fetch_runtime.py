import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Optional, Union
from unittest.mock import Mock, patch

import yaml

from scripts.fetch_alpaca_data import run_fetch
from weightiz.shared.io.market_data.alpaca_client import AlpacaAPIError, AlpacaClient


class _FakeResponse:
    def __init__(self, status_code: int, text: str = "", payload: Optional[dict] = None):
        self.status_code = int(status_code)
        self.text = text
        self._payload = payload or {}

    def json(self):
        return self._payload


class _FakeClientSuccess:
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


class _FakeClientPreflightDenied:
    def __init__(self, *args, **kwargs):
        pass

    def fetch_bars_multi_with_meta(self, *args, **kwargs):
        raise AlpacaAPIError(
            "Alpaca API error status=403: forbidden permission denied",
            status_code=403,
            response_text="forbidden permission denied",
        )


class _FakeClientOther4xx:
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
        # First two calls are preflight (recent + historical): return success.
        if self.calls in (1, 2):
            sym = str(symbols[0])
            bars = {
                sym: [{"t": "2024-03-11T13:30:00Z", "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 1}]
            }
            feed_map = {sym: str(feed)}
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
            return bars, feed_map, [], meta
        raise AlpacaAPIError(
            "Alpaca API error status=404: resource missing",
            status_code=404,
            response_text="resource missing",
        )


class TestAlpacaFetchRuntime(unittest.TestCase):
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
        config_path = root / "cfg.yaml"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        return config_path

    @staticmethod
    def _load_json(path: Union[str, Path]) -> dict:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def test_symbols_sorted_and_run_id_contains_config_hash_prefix(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = self._write_config(root=root, symbols=["SPY", "QQQ"])

            with patch.dict(
                os.environ,
                {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"},
                clear=False,
            ), patch("scripts.fetch_alpaca_data.AlpacaClient", _FakeClientSuccess):
                summary = run_fetch(config_path)

            qa_doc = self._load_json(summary["qa_report"])
            manifest_doc = self._load_json(summary["manifest_report"])

            self.assertEqual(list(qa_doc["symbols"].keys()), ["QQQ", "SPY"])
            self.assertEqual(list(manifest_doc["symbols"].keys()), ["QQQ", "SPY"])
            self.assertEqual(str(qa_doc["preflight_symbol"]), "QQQ")
            self.assertIn(str(summary["config_sha256"])[:10], str(summary["run_id"]))
            self.assertIn(str(summary["config_sha256"])[:10], str(qa_doc["run_id"]))
            self.assertIn(str(summary["config_sha256"])[:10], str(manifest_doc["run_id"]))

    def test_preflight_permission_denied_writes_artifacts_with_diagnostics(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = self._write_config(root=root, symbols=["SPY", "QQQ"])
            reports_root = root / "reports"

            with patch.dict(
                os.environ,
                {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"},
                clear=False,
            ), patch("scripts.fetch_alpaca_data.AlpacaClient", _FakeClientPreflightDenied):
                with self.assertRaisesRegex(RuntimeError, "permission_denied_feed"):
                    run_fetch(config_path)

            qa_files = sorted(reports_root.glob("*_qa.json"))
            manifest_files = sorted(reports_root.glob("*_manifest.json"))
            self.assertEqual(len(qa_files), 1)
            self.assertEqual(len(manifest_files), 1)

            qa_doc = self._load_json(qa_files[0])
            manifest_doc = self._load_json(manifest_files[0])

            self.assertEqual(str(qa_doc["summary"]["abort_reason"]), "permission_denied_feed")
            self.assertEqual(str(manifest_doc["summary"]["abort_reason"]), "permission_denied_feed")
            self.assertEqual(int(qa_doc["preflight_status_code"]), 403)
            self.assertEqual(str(qa_doc["preflight_error_class"]), "permission_denied_feed")
            self.assertTrue(str(qa_doc["preflight_error_msg"]))

    def test_invalid_limit_fallback_to_safe_limit_1000(self):
        client = object.__new__(AlpacaClient)
        client._base_url = "https://data.alpaca.markets"
        client._sleep = 0.0
        client._timeout = 5.0
        client._max_retries_429 = 2
        client._backoff_base_sec = 0.5
        client._backoff_max_sec = 8.0
        responses = [
            _FakeResponse(status_code=400, text="invalid request: limit exceeds max"),
            _FakeResponse(
                status_code=200,
                payload={
                    "bars": {
                        "SPY": [
                            {"t": "2024-03-11T13:30:00Z", "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 1}
                        ]
                    },
                    "next_page_token": None,
                },
            ),
        ]
        client._session = type("Session", (), {})()
        client._session.get = Mock(side_effect=responses)

        _, _, _, meta = client.fetch_bars_multi_with_meta(
            symbols=["SPY"],
            timeframe="1Min",
            start="2024-03-08T00:00:00Z",
            end="2024-03-14T23:59:59Z",
            feed="iex",
            adjustment="raw",
            sort="asc",
            limit=10000,
        )
        self.assertEqual(int(meta["SPY"]["limit_requested"]), 10000)
        self.assertEqual(int(meta["SPY"]["limit_effective"]), 1000)
        self.assertEqual(bool(meta["SPY"]["limit_fallback_used"]), True)

    def test_other_4xx_records_failed_without_limit_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = self._write_config(root=root, symbols=["SPY"])
            reports_root = root / "reports"

            with patch.dict(
                os.environ,
                {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"},
                clear=False,
            ), patch("scripts.fetch_alpaca_data.AlpacaClient", _FakeClientOther4xx):
                with self.assertRaisesRegex(RuntimeError, "all_symbols_failed"):
                    run_fetch(config_path)

            manifest_doc = self._load_json(sorted(reports_root.glob("*_manifest.json"))[0])
            self.assertEqual(str(manifest_doc["symbols"]["SPY"]["status"]), "failed")
            self.assertEqual(bool(manifest_doc["symbols"]["SPY"]["limit_fallback_used"]), False)

    def test_canary_reports_include_source_params_and_calendar_availability(self):
        repo_root = Path(__file__).resolve().parents[1]
        src_cfg = repo_root / "configs" / "data_alpaca_canary_dst.yaml"
        raw_cfg = yaml.safe_load(src_cfg.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw_cfg["storage"]["root"] = str(root)
            raw_cfg["storage"]["write_format"] = "csv"
            raw_cfg["alpaca"]["calendar_mode"] = "naive"
            raw_cfg["alpaca"]["qa_policy"] = "coverage_threshold"
            raw_cfg["alpaca"]["coverage_min_pct"] = 0.01
            config_path = root / "canary_dst.yaml"
            config_path.write_text(yaml.safe_dump(raw_cfg, sort_keys=False), encoding="utf-8")

            with patch.dict(
                os.environ,
                {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"},
                clear=False,
            ), patch("scripts.fetch_alpaca_data.AlpacaClient", _FakeClientSuccess):
                summary = run_fetch(config_path)

            qa_doc = self._load_json(summary["qa_report"])
            manifest_doc = self._load_json(summary["manifest_report"])
            for doc in (qa_doc, manifest_doc):
                self.assertEqual(str(doc["source"]), "alpaca")
                self.assertIn("base_url", doc)
                self.assertIn("feed", doc)
                self.assertIn("adjustment", doc)
                self.assertIn("timeframe", doc)
                self.assertIn("sort", doc)
                self.assertIn("limit_requested", doc)
                self.assertIn("limit_effective", doc)
                self.assertIn("calendar_expectations_available", doc)

    def test_dst_canary_config_exists_and_expected_window(self):
        repo_root = Path(__file__).resolve().parents[1]
        cfg_path = repo_root / "configs" / "data_alpaca_canary_dst.yaml"
        self.assertTrue(cfg_path.exists())
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        self.assertEqual(str(cfg["alpaca"]["start"]), "2024-03-08T00:00:00Z")
        self.assertEqual(str(cfg["alpaca"]["end"]), "2024-03-14T23:59:59Z")

    def test_symbol_filter_runs_single_symbol(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = self._write_config(root=root, symbols=["SPY", "QQQ"])

            with patch.dict(
                os.environ,
                {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"},
                clear=False,
            ), patch("scripts.fetch_alpaca_data.AlpacaClient", _FakeClientSuccess):
                summary = run_fetch(config_path, symbol_filter="SPY")

            qa_doc = self._load_json(summary["qa_report"])
            self.assertEqual(list(qa_doc["symbols"].keys()), ["SPY"])
            self.assertEqual(str(qa_doc["preflight_symbol"]), "SPY")


if __name__ == "__main__":
    unittest.main()
