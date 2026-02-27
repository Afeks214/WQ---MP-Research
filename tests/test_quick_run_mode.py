from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import weightiz_module5_harness as harness
from scripts import run_sweep_auto as sweep_auto


class _ProcResult:
    def __init__(self, returncode: int = 0) -> None:
        self.returncode = int(returncode)


@unittest.skipIf(pd is None, "pandas not available")
class TestQuickRunMode(unittest.TestCase):
    def test_quick_run_reduction_auto_grid_is_deterministic(self) -> None:
        base = {
            "symbols": ["SPY", "QQQ", "IWM"],
            "module2_configs": [{"id": 0}, {"id": 1}],
            "module3_configs": [{"id": 0}, {"id": 1}, {"id": 2}],
            "module4_configs": [{"id": 0}, {"id": 1}, {"id": 2}],
            "harness": {"parallel_backend": "process_pool", "parallel_workers": 8, "min_asset_coverage": 0.8},
            "stress_scenarios": [
                {"scenario_id": "baseline", "enabled": True},
                {"scenario_id": "mild", "enabled": True},
                {"scenario_id": "severe", "enabled": True},
            ],
            "candidates": {"mode": "auto_grid", "specs": []},
        }

        reduced, policy = sweep_auto._apply_quick_run_reduction(
            base,
            chosen_symbols=["XLK", "SPY", "QQQ", "DIA"],
        )

        self.assertEqual(reduced["symbols"], ["DIA", "QQQ"])
        self.assertEqual(policy["quick_symbols"], ["DIA", "QQQ"])
        self.assertEqual(len(reduced["module2_configs"]), 1)
        self.assertGreaterEqual(len(reduced["module3_configs"]), 1)
        self.assertGreaterEqual(len(reduced["module4_configs"]), 1)
        combos = (
            len(reduced["module2_configs"])
            * len(reduced["module3_configs"])
            * len(reduced["module4_configs"])
        )
        self.assertLessEqual(combos, 3)
        self.assertEqual(reduced["harness"]["parallel_backend"], "serial")
        self.assertEqual(int(reduced["harness"]["parallel_workers"]), 1)
        self.assertEqual(float(reduced["harness"]["min_asset_coverage"]), 0.0)
        enabled = [s for s in reduced["stress_scenarios"] if bool(s.get("enabled"))]
        self.assertEqual([str(s["scenario_id"]) for s in enabled], ["baseline"])
        # Source config remains unchanged.
        self.assertEqual(base["symbols"], ["SPY", "QQQ", "IWM"])
        self.assertEqual(len(base["module2_configs"]), 2)

    def test_quick_run_reduction_manual_candidates_keeps_first_two_sorted(self) -> None:
        base = {
            "symbols": ["SPY", "QQQ", "IWM"],
            "module2_configs": [{"id": 0}],
            "module3_configs": [{"id": 0}],
            "module4_configs": [{"id": 0}],
            "candidates": {
                "mode": "manual",
                "specs": [
                    {"candidate_id": "cand_z"},
                    {"candidate_id": "cand_b"},
                    {"candidate_id": "cand_a"},
                ],
            },
        }

        reduced, policy = sweep_auto._apply_quick_run_reduction(base, chosen_symbols=["SPY", "QQQ", "IWM"])
        kept = [str(x.get("candidate_id")) for x in reduced["candidates"]["specs"]]
        self.assertEqual(kept, ["cand_a", "cand_b"])
        self.assertEqual(policy["quick_candidate_mode"], "manual")
        self.assertEqual(int(policy["quick_manual_specs_kept"]), 2)

    def test_run_research_quick_sets_env_and_logs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = root / "cfg.yaml"
            cfg.write_text("run_name: test\n", encoding="utf-8")
            logs = root / "logs"
            with patch.object(sweep_auto.subprocess, "run", return_value=_ProcResult(0)) as run_mock:
                with patch.object(sweep_auto, "_read_latest_run_dir", return_value=root):
                    out = sweep_auto._run_research(cfg, quick_run=True, log_dir=logs)
            self.assertEqual(out, root)
            self.assertTrue((logs / "cfg_stdout.log").exists())
            self.assertTrue((logs / "cfg_stderr.log").exists())
            called_env = run_mock.call_args.kwargs["env"]
            self.assertEqual(called_env.get("QUICK_RUN"), "1")
            self.assertEqual(called_env.get("QUICK_RUN_TASK_TIMEOUT_SEC"), "120")
            self.assertEqual(called_env.get("QUICK_RUN_PROGRESS_EVERY"), "1")

    def test_run_research_nonquick_streams_to_logs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = root / "cfg.yaml"
            cfg.write_text("run_name: test\n", encoding="utf-8")
            logs = root / "logs"
            with patch.object(sweep_auto.subprocess, "run", return_value=_ProcResult(0)) as run_mock:
                with patch.object(sweep_auto, "_read_latest_run_dir", return_value=root):
                    out = sweep_auto._run_research(cfg, quick_run=False, log_dir=logs)
            self.assertEqual(out, root)
            self.assertTrue((logs / "cfg_stdout.log").exists())
            self.assertTrue((logs / "cfg_stderr.log").exists())
            called_env = run_mock.call_args.kwargs["env"]
            self.assertNotIn("QUICK_RUN", called_env)

    def test_verify_quick_run_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"symbol": "SPY", "session_date": "2024-01-02", "decision": "ACCEPT"}]).to_csv(
                run_dir / "dq_report.csv",
                index=False,
            )
            pd.DataFrame(
                [{"timestamp": "2024-01-02T14:30:00Z", "symbol": "SPY", "dq_filled_bar": False, "dq_issue_flags": 0, "dqs_day": 1.0}]
            ).to_parquet(run_dir / "dq_bar_flags.parquet", index=False)
            pd.DataFrame(
                [{"candidate_id": "cand_0001", "robustness_score": 0.25, "pass": True, "failed": False}]
            ).to_csv(run_dir / "robustness_leaderboard.csv", index=False)
            (run_dir / "run_status.json").write_text(
                json.dumps({"failure_rate": 0.0}, sort_keys=True),
                encoding="utf-8",
            )
            (run_dir / "plateaus.json").write_text(json.dumps({"clusters": []}, sort_keys=True), encoding="utf-8")

            summary = sweep_auto._verify_quick_run_artifacts(run_dir)
            self.assertEqual(int(summary["dq_report_rows"]), 1)
            self.assertEqual(int(summary["dq_bar_flags_rows"]), 1)
            self.assertEqual(int(summary["leaderboard_rows"]), 1)
            self.assertEqual(len(summary["top5"]), 1)
            self.assertTrue(bool(summary["wiring_ok"]))
            self.assertTrue(bool(summary["evaluation_ready"]))
            self.assertEqual(str(summary["evaluation_ready_reason"]), "")

    def test_verify_quick_run_artifacts_marks_not_evaluation_ready_when_all_failed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"symbol": "SPY", "session_date": "2024-01-02", "decision": "ACCEPT"}]).to_csv(
                run_dir / "dq_report.csv",
                index=False,
            )
            pd.DataFrame(
                [{"timestamp": "2024-01-02T14:30:00Z", "symbol": "SPY", "dq_filled_bar": False, "dq_issue_flags": 0, "dqs_day": 1.0}]
            ).to_parquet(run_dir / "dq_bar_flags.parquet", index=False)
            pd.DataFrame(
                [{"candidate_id": "cand_0001", "robustness_score": float("-inf"), "pass": False, "failed": True}]
            ).to_csv(run_dir / "robustness_leaderboard.csv", index=False)
            (run_dir / "run_status.json").write_text(
                json.dumps({"failure_rate": 0.0}, sort_keys=True),
                encoding="utf-8",
            )
            (run_dir / "stats_raw.json").write_text(
                json.dumps({"quick_run_stats_error": "RuntimeError: returns_matrix must have T>=3, got T=1"}, sort_keys=True),
                encoding="utf-8",
            )
            (run_dir / "plateaus.json").write_text(json.dumps({"clusters": []}, sort_keys=True), encoding="utf-8")

            summary = sweep_auto._verify_quick_run_artifacts(run_dir)
            self.assertTrue(bool(summary["wiring_ok"]))
            self.assertFalse(bool(summary["evaluation_ready"]))
            self.assertIn("all_candidates_failed", str(summary["evaluation_ready_reason"]))

    def test_harness_reads_quick_run_runtime_overrides(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "QUICK_RUN": "1",
                "QUICK_RUN_TASK_TIMEOUT_SEC": "77",
                "QUICK_RUN_PROGRESS_EVERY": "2",
            },
            clear=False,
        ):
            opts = harness._quick_run_settings_from_env()
        self.assertTrue(opts.enabled)
        self.assertEqual(int(opts.task_timeout_sec), 77)
        self.assertEqual(int(opts.progress_every_groups), 2)
        self.assertTrue(opts.disable_cpcv)
        self.assertTrue(opts.baseline_only)


if __name__ == "__main__":
    unittest.main()
