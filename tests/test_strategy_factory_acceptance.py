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
class TestStrategyFactoryAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.tmp = tempfile.TemporaryDirectory(prefix="weightiz_factory_test_")
        cls.tmp_root = Path(cls.tmp.name)
        cls.data_root = cls.tmp_root / "data" / "minute"
        cls.data_root.mkdir(parents=True, exist_ok=True)
        cls.artifacts_root = cls.tmp_root / "artifacts"
        cls.artifacts_root.mkdir(parents=True, exist_ok=True)

        cls.symbols = ["AAA", "BBB"]
        cls._write_synth_symbol("AAA", seed=11)
        cls._write_synth_symbol("BBB", seed=29)

        cls.config_path = cls.tmp_root / "sweep_test.yaml"
        cls._write_config(cls.config_path)

        cls.run_dir_1 = cls._execute_run(cls.config_path)
        time.sleep(1.10)
        cls.run_dir_2 = cls._execute_run(cls.config_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    @classmethod
    def _write_synth_symbol(cls, symbol: str, seed: int) -> None:
        rng = np.random.default_rng(seed)
        sessions = pd.date_range("2024-01-02", periods=6, freq="B", tz="America/New_York")
        chunks = []
        for day_idx, d in enumerate(sessions):
            start = d.replace(hour=9, minute=30, second=0)
            ts = pd.date_range(start, periods=30, freq="1min", tz="America/New_York").tz_convert("UTC")
            t = np.arange(ts.shape[0], dtype=np.float64)
            base = 100.0 + 0.15 * day_idx + 0.02 * t + (0.35 if symbol == "BBB" else 0.0)
            noise = rng.normal(0.0, 0.01, size=ts.shape[0])
            close = base + noise
            open_px = close - 0.01
            high = np.maximum(open_px, close) + 0.02
            low = np.minimum(open_px, close) - 0.02
            volume = 1000.0 + (t % 7.0) * 12.0 + day_idx * 5.0
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
        df = pd.concat(chunks, axis=0, ignore_index=True)
        df.to_parquet(cls.data_root / f"{symbol}.parquet", index=False)

    @classmethod
    def _module3_grid(cls) -> list[dict[str, object]]:
        return [
            {"block_minutes": 15, "include_partial_last_block": True, "min_block_valid_ratio": 0.65, "ib_pop_frac": 0.008},
            {"block_minutes": 15, "include_partial_last_block": False, "min_block_valid_ratio": 0.70, "ib_pop_frac": 0.010},
            {"block_minutes": 20, "include_partial_last_block": True, "min_block_valid_ratio": 0.68, "ib_pop_frac": 0.009},
            {"block_minutes": 20, "include_partial_last_block": False, "min_block_valid_ratio": 0.72, "ib_pop_frac": 0.011},
            {"block_minutes": 30, "include_partial_last_block": True, "min_block_valid_ratio": 0.70, "ib_pop_frac": 0.010},
            {"block_minutes": 30, "include_partial_last_block": False, "min_block_valid_ratio": 0.75, "ib_pop_frac": 0.012},
            {"block_minutes": 40, "include_partial_last_block": True, "min_block_valid_ratio": 0.74, "ib_pop_frac": 0.010},
            {"block_minutes": 40, "include_partial_last_block": False, "min_block_valid_ratio": 0.78, "ib_pop_frac": 0.012},
            {"block_minutes": 20, "include_partial_last_block": True, "min_block_valid_ratio": 0.80, "ib_pop_frac": 0.014},
            {"block_minutes": 30, "include_partial_last_block": True, "min_block_valid_ratio": 0.76, "ib_pop_frac": 0.013},
        ]

    @classmethod
    def _module4_grid(cls) -> list[dict[str, object]]:
        return [
            {"entry_threshold": 0.52, "exit_threshold": 0.20, "top_k_intraday": 6, "max_asset_cap_frac": 0.35, "max_turnover_frac_per_bar": 0.40, "overnight_min_conviction": 0.60, "hard_kill_on_daily_loss_breach": True},
            {"entry_threshold": 0.54, "exit_threshold": 0.22, "top_k_intraday": 6, "max_asset_cap_frac": 0.33, "max_turnover_frac_per_bar": 0.38, "overnight_min_conviction": 0.62, "hard_kill_on_daily_loss_breach": True},
            {"entry_threshold": 0.56, "exit_threshold": 0.24, "top_k_intraday": 5, "max_asset_cap_frac": 0.31, "max_turnover_frac_per_bar": 0.36, "overnight_min_conviction": 0.64, "hard_kill_on_daily_loss_breach": True},
            {"entry_threshold": 0.58, "exit_threshold": 0.26, "top_k_intraday": 5, "max_asset_cap_frac": 0.30, "max_turnover_frac_per_bar": 0.34, "overnight_min_conviction": 0.66, "hard_kill_on_daily_loss_breach": True},
            {"entry_threshold": 0.60, "exit_threshold": 0.28, "top_k_intraday": 4, "max_asset_cap_frac": 0.28, "max_turnover_frac_per_bar": 0.32, "overnight_min_conviction": 0.68, "hard_kill_on_daily_loss_breach": True},
            {"entry_threshold": 0.62, "exit_threshold": 0.30, "top_k_intraday": 4, "max_asset_cap_frac": 0.27, "max_turnover_frac_per_bar": 0.30, "overnight_min_conviction": 0.70, "hard_kill_on_daily_loss_breach": True},
            {"entry_threshold": 0.64, "exit_threshold": 0.31, "top_k_intraday": 3, "max_asset_cap_frac": 0.25, "max_turnover_frac_per_bar": 0.28, "overnight_min_conviction": 0.72, "hard_kill_on_daily_loss_breach": True},
            {"entry_threshold": 0.66, "exit_threshold": 0.33, "top_k_intraday": 3, "max_asset_cap_frac": 0.24, "max_turnover_frac_per_bar": 0.26, "overnight_min_conviction": 0.75, "hard_kill_on_daily_loss_breach": True},
            {"entry_threshold": 0.67, "exit_threshold": 0.34, "top_k_intraday": 2, "max_asset_cap_frac": 0.22, "max_turnover_frac_per_bar": 0.24, "overnight_min_conviction": 0.78, "hard_kill_on_daily_loss_breach": True},
            {"entry_threshold": 0.68, "exit_threshold": 0.35, "top_k_intraday": 2, "max_asset_cap_frac": 0.20, "max_turnover_frac_per_bar": 0.20, "overnight_min_conviction": 0.80, "hard_kill_on_daily_loss_breach": True},
        ]

    @classmethod
    def _write_config(cls, path: Path) -> None:
        cfg = {
            "run_name": "factory_acceptance",
            "symbols": cls.symbols,
            "data": {
                "root": str(cls.data_root),
                "format": "parquet",
                "path_by_symbol": {
                    "AAA": "AAA.parquet",
                    "BBB": "BBB.parquet",
                },
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
            "module3_configs": cls._module3_grid(),
            "module4_configs": cls._module4_grid(),
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
                "report_dir": str(cls.artifacts_root / "module5_harness"),
                "fail_on_non_finite": True,
                "daily_return_min_days": 3,
                "benchmark_symbol": "AAA",
                "export_micro_diagnostics": False,
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
            "candidates": {
                "mode": "auto_grid",
                "specs": [],
            },
        }
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    @classmethod
    def _execute_run(cls, cfg_path: Path) -> Path:
        cmd = [
            sys.executable,
            "-m",
            "weightiz.cli.run_research",
            "--config",
            str(cfg_path),
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(cls.repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"run_research failed rc={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )

        latest = cls.artifacts_root / ".latest_run"
        if not latest.exists():
            raise RuntimeError(".latest_run missing after run")
        run_dir = Path(latest.read_text(encoding="utf-8").strip())
        if not run_dir.exists():
            raise RuntimeError(f"run_dir from .latest_run does not exist: {run_dir}")
        return run_dir

    def test_factory_outputs_100_candidates(self) -> None:
        run_dir = self.run_dir_2

        candidates_dir = run_dir / "candidates"
        self.assertTrue(candidates_dir.exists())
        cand_dirs = sorted([p for p in candidates_dir.iterdir() if p.is_dir()])
        self.assertEqual(len(cand_dirs), 100)

        lb = pd.read_csv(run_dir / "leaderboard.csv")
        rb = pd.read_csv(run_dir / "robustness_leaderboard.csv")
        self.assertEqual(len(lb), 100)
        self.assertEqual(len(rb), 100)

        scores = rb["robustness_score"].to_numpy(dtype=np.float64)
        self.assertTrue(np.all(scores[:-1] >= scores[1:] - 1e-15))

        with (run_dir / "verdict.json").open("r", encoding="utf-8") as f:
            verdict = json.load(f)
        rows = verdict.get("leaderboard", [])
        self.assertEqual(len(rows), 100)
        self.assertTrue(all("candidate_id" in r for r in rows))
        self.assertTrue(all("task_id" not in r for r in rows))

        daily = pd.read_parquet(run_dir / "daily_returns.parquet")
        self.assertEqual(daily.shape[1], 102)  # session_id + benchmark + 100 candidates
        self.assertTrue((run_dir / "plateaus.json").exists())

    def test_sweep_is_deterministic(self) -> None:
        run1 = self.run_dir_1
        run2 = self.run_dir_2

        with (run1 / "run_summary.json").open("r", encoding="utf-8") as f:
            s1 = json.load(f)
        with (run2 / "run_summary.json").open("r", encoding="utf-8") as f:
            s2 = json.load(f)

        self.assertEqual(s1["resolved_config_sha256"], s2["resolved_config_sha256"])

        b1 = (run1 / "robustness_leaderboard.csv").read_bytes()
        b2 = (run2 / "robustness_leaderboard.csv").read_bytes()
        self.assertEqual(b1, b2)


if __name__ == "__main__":
    unittest.main()
