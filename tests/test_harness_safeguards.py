import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore[assignment]

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


@unittest.skipIf(pd is None or yaml is None, "pandas/pyyaml not available")
class TestHarnessSafeguards(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parent.parent

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="weightiz_safeguards_test_")
        self.tmp_root = Path(self.tmp.name)
        self.data_root = self.tmp_root / "data" / "minute"
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.artifacts_root = self.tmp_root / "artifacts"
        self.artifacts_root.mkdir(parents=True, exist_ok=True)
        self._write_synth_symbol("AAA", seed=5)
        self._write_synth_symbol("BBB", seed=9)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_synth_symbol(self, symbol: str, seed: int) -> None:
        rng = np.random.default_rng(seed)
        sessions = pd.date_range("2024-01-02", periods=6, freq="B", tz="America/New_York")
        chunks = []
        for day_idx, d in enumerate(sessions):
            start = d.replace(hour=9, minute=30, second=0)
            ts = pd.date_range(start, periods=30, freq="1min", tz="America/New_York").tz_convert("UTC")
            t = np.arange(ts.shape[0], dtype=np.float64)
            base = 100.0 + 0.2 * day_idx + 0.01 * t + (0.25 if symbol == "BBB" else 0.0)
            noise = rng.normal(0.0, 0.01, size=ts.shape[0])
            close = base + noise
            open_px = close - 0.01
            high = np.maximum(open_px, close) + 0.02
            low = np.minimum(open_px, close) - 0.02
            volume = 1000.0 + (t % 5.0) * 10.0 + day_idx * 4.0
            chunks.append(
                pd.DataFrame(
                    {
                        "timestamp": ts,
                        "open": open_px,
                        "high": high,
                        "low": low,
                        "close": close,
                        "volume": volume,
                    }
                )
            )
        pd.concat(chunks, axis=0, ignore_index=True).to_parquet(self.data_root / f"{symbol}.parquet", index=False)

    def _base_config(self) -> dict:
        return {
            "run_name": "safeguard_test",
            "symbols": ["AAA", "BBB"],
            "data": {
                "root": str(self.data_root),
                "format": "parquet",
                "path_by_symbol": {"AAA": "AAA.parquet", "BBB": "BBB.parquet"},
                "timestamp_column": "timestamp",
                "start": "2024-01-02T00:00:00Z",
                "end": "2024-12-31T23:59:59Z",
            },
            "engine": {
                "mode": "sealed",
                "B": 120,
                "x_min": -6.0,
                "dx": 0.05,
                "seed": 17,
                "tick_size_default": 0.01,
            },
            "module2_configs": [
                {
                    "profile_window_bars": 20,
                    "profile_warmup_bars": 20,
                    "atr_span": 14,
                    "rvol_lookback_sessions": 5,
                    "va_threshold": 0.70,
                    "fail_on_non_finite_output": True,
                }
            ],
            "module3_configs": [
                {"block_minutes": 15, "include_partial_last_block": True, "min_block_valid_ratio": 0.68, "ib_pop_frac": 0.009},
                {"block_minutes": 30, "include_partial_last_block": False, "min_block_valid_ratio": 0.75, "ib_pop_frac": 0.012},
            ],
            "module4_configs": [
                {"entry_threshold": 0.55, "exit_threshold": 0.25, "top_k_intraday": 4, "max_asset_cap_frac": 0.30, "max_turnover_frac_per_bar": 0.30, "overnight_min_conviction": 0.65, "hard_kill_on_daily_loss_breach": True},
                {"entry_threshold": 0.67, "exit_threshold": 0.34, "top_k_intraday": 2, "max_asset_cap_frac": 0.22, "max_turnover_frac_per_bar": 0.22, "overnight_min_conviction": 0.80, "hard_kill_on_daily_loss_breach": True},
            ],
            "harness": {
                "seed": 97,
                "timezone": "America/New_York",
                "freq": "1min",
                "min_asset_coverage": 1.0,
                "purge_bars": 0,
                "embargo_bars": 0,
                "wf_train_sessions": 2,
                "wf_test_sessions": 3,
                "wf_step_sessions": 1,
                "cpcv_slices": 3,
                "cpcv_k_test": 1,
                "parallel_backend": "serial",
                "parallel_workers": 1,
                "stress_profile": "baseline_mild_severe",
                "max_ram_utilization_frac": 0.70,
                "enforce_lookahead_guard": True,
                "report_dir": str(self.artifacts_root / "module5_harness"),
                "fail_on_non_finite": True,
                "daily_return_min_days": 3,
                "benchmark_symbol": "AAA",
                "export_micro_diagnostics": False,
                "failure_rate_abort_threshold": 0.99,
                "failure_count_abort_threshold": 999,
                "payload_pickle_threshold_bytes": 131072,
                "test_fail_task_ids": [],
                "test_fail_ratio": 0.0,
            },
            "stress_scenarios": [
                {
                    "scenario_id": "baseline",
                    "name": "baseline",
                    "missing_burst_prob": 0.0,
                    "missing_burst_min": 0,
                    "missing_burst_max": 0,
                    "jitter_sigma_bps": 0.0,
                    "slippage_mult": 1.0,
                    "enabled": True,
                }
            ],
            "candidates": {"mode": "auto_grid", "specs": []},
        }

    def _run_cfg(self, cfg: dict) -> tuple[Path, dict]:
        cfg_path = self.tmp_root / f"cfg_{time.time_ns()}.yaml"
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        cmd = [sys.executable, str(self.repo_root / "run_research.py"), "--config", str(cfg_path)]
        proc = subprocess.run(cmd, cwd=str(self.repo_root), capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"run_research rc={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

        latest = self.artifacts_root / ".latest_run"
        run_dir = Path(latest.read_text(encoding="utf-8").strip())
        with (run_dir / "run_summary.json").open("r", encoding="utf-8") as f:
            summary = json.load(f)
        return run_dir, summary

    def test_fault_tolerance_does_not_abort_on_single_task_failure(self) -> None:
        cfg = self._base_config()
        cfg["harness"]["test_fail_task_ids"] = ["cand_0000_m20_m30_m40|wf_000|baseline"]
        run_dir, summary = self._run_cfg(cfg)

        self.assertFalse(bool(summary.get("aborted", False)))
        self.assertEqual(int(summary.get("failure_count", 0)), 1)

        dead = run_dir / "deadletter_tasks.jsonl"
        self.assertTrue(dead.exists())
        lines = [x for x in dead.read_text(encoding="utf-8").splitlines() if x.strip()]
        self.assertEqual(len(lines), 1)

        rb = pd.read_csv(run_dir / "robustness_leaderboard.csv")
        self.assertTrue((run_dir / "robustness_leaderboard.csv").exists())
        failed_row = rb.loc[rb["candidate_id"] == "cand_0000_m20_m30_m40"].iloc[0]
        self.assertTrue(np.isneginf(float(failed_row["robustness_score"])))

        cm = json.loads((run_dir / "candidates" / "cand_0000_m20_m30_m40" / "candidate_metrics.json").read_text())
        self.assertTrue(bool(cm.get("failed", False)))

    def test_circuit_breaker_aborts_when_failure_rate_exceeds_threshold(self) -> None:
        cfg = self._base_config()
        cfg["harness"]["test_fail_ratio"] = 1.0
        cfg["harness"]["failure_rate_abort_threshold"] = 0.20
        cfg["harness"]["failure_count_abort_threshold"] = 2

        run_dir, summary = self._run_cfg(cfg)

        self.assertTrue(bool(summary.get("aborted", False)))
        self.assertTrue(str(summary.get("abort_reason", "")))
        self.assertGreaterEqual(int(summary.get("failure_count", 0)), 2)
        self.assertTrue((run_dir / "run_summary.json").exists())
        self.assertTrue((run_dir / "robustness_leaderboard.csv").exists())

    def test_large_payload_not_passed_to_workers(self) -> None:
        cfg = self._base_config()
        cfg["harness"]["parallel_backend"] = "process_pool"
        cfg["harness"]["parallel_workers"] = 2

        _, summary = self._run_cfg(cfg)
        self.assertTrue(bool(summary.get("payload_safe", False)))
        self.assertTrue(bool(summary.get("large_payload_passing_avoided", False)))


if __name__ == "__main__":
    unittest.main()
