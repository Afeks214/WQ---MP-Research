from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import weightiz_module5_harness as h


class TestHarnessNonFiniteExecPxLocalized(unittest.TestCase):
    def test_nonfinite_exec_px_is_localized_and_deadletter_has_exec_dump(self) -> None:
        tracker: dict = {}
        rows = []
        for i, cid in enumerate(["c1", "c2", "c3"]):
            rows.append(
                {
                    "task_id": f"{cid}|wf_000|baseline",
                    "candidate_id": cid,
                    "split_id": "wf_000",
                    "scenario_id": "baseline",
                    "status": "error",
                    "error_type": "NonFiniteExecutionPriceError",
                    "error_hash": f"h{i}",
                    "error": "NonFiniteExecutionPriceError: Non-finite/non-positive execution price at a=0: nan",
                    "traceback": 'Traceback\n  File "weightiz_module4_strategy_funnel.py", line 250, in _execute_to_target',
                    "top_frame": "weightiz_module4_strategy_funnel.py:250:_execute_to_target",
                    "exception_signature": "NonFiniteExecutionPriceError|weightiz_module4_strategy_funnel.py:250:_execute_to_target",
                    "task_seed": i,
                    "asset_keys": ["AAA", "BBB"],
                    "quality_reason_codes": ["NONFINITE_EXEC_PX"],
                    "exec_px_dump": {
                        "run_context": {"candidate_id": cid, "split_id": "wf_000", "scenario_id": "baseline"},
                        "t": 42,
                        "ts_utc": "2025-01-06T15:12:00+00:00",
                        "asset_index": 0,
                        "asset_symbol": "AAA",
                        "px_source_name": "next_open",
                        "px_value": float("nan"),
                        "close_px": 100.0,
                        "open_px": float("nan"),
                        "high_px": 100.5,
                        "low_px": 99.5,
                        "target_qty": 10.0,
                        "limit_px": float("nan"),
                        "stop_px": float("nan"),
                        "take_px": float("nan"),
                        "conviction": 0.9,
                        "dqs_day": 1.0,
                        "ib_defined": True,
                        "phase": 1,
                        "tod": 22,
                        "minute_of_day": 572,
                    },
                }
            )

        with tempfile.TemporaryDirectory() as td:
            deadletter = Path(td) / "deadletter_tasks.jsonl"
            aborted = False
            for row in rows:
                h._update_failure_tracker(tracker, row)
                abort, _reason = h._should_abort_systemic(tracker, row)
                aborted = aborted or abort
                h._record_deadletter(deadletter, row)
            self.assertFalse(aborted)

            lines = [json.loads(x) for x in deadletter.read_text(encoding="utf-8").splitlines() if x.strip()]
            self.assertEqual(len(lines), 3)
            self.assertTrue(all(d.get("error_type") == "NonFiniteExecutionPriceError" for d in lines))
            self.assertTrue(all("NONFINITE_EXEC_PX" in d.get("reason_codes", []) for d in lines))
            self.assertTrue(all(isinstance(d.get("exec_px_dump"), dict) for d in lines))
            self.assertTrue(all(d.get("exec_px_dump", {}).get("px_source_name") == "next_open" for d in lines))
            self.assertTrue(all(d.get("exec_px_dump", {}).get("run_context", {}).get("split_id") == "wf_000" for d in lines))

    def test_next_open_unavailable_is_localized(self) -> None:
        tracker: dict = {}
        row = {
            "task_id": "c9|wf_009|baseline",
            "candidate_id": "c9",
            "split_id": "wf_009",
            "scenario_id": "baseline",
            "status": "error",
            "error_type": "NonFiniteExecutionPriceError",
            "error_hash": "h9",
            "error": "NonFiniteExecutionPriceError: Next-open unavailable",
            "traceback": 'Traceback\\n  File "weightiz_module4_strategy_funnel.py", line 580, in run_module4_strategy_funnel',
            "top_frame": "weightiz_module4_strategy_funnel.py:580:run_module4_strategy_funnel",
            "exception_signature": "NonFiniteExecutionPriceError|weightiz_module4_strategy_funnel.py:580:run_module4_strategy_funnel",
            "task_seed": 9,
            "asset_keys": ["AAA", "BBB"],
            "quality_reason_codes": ["NEXT_OPEN_UNAVAILABLE"],
            "exec_px_dump": {
                "run_context": {"candidate_id": "c9", "split_id": "wf_009", "scenario_id": "baseline"},
                "t_signal": 10,
                "t_fill": 11,
                "asset_index": 0,
                "asset_symbol": "AAA",
                "px_source_name": "next_open",
                "px_value": float("nan"),
            },
        }
        h._update_failure_tracker(tracker, row)
        abort, _reason = h._should_abort_systemic(tracker, row)
        self.assertFalse(abort)


if __name__ == "__main__":
    unittest.main()
