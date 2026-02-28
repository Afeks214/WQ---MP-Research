from __future__ import annotations

import unittest

import weightiz_module5_harness as h


class TestHarnessFaultIsolation(unittest.TestCase):
    def test_localized_reason_codes_do_not_trigger_systemic_abort(self) -> None:
        tracker: dict = {}
        rows = [
            {
                "task_id": "c1|wf_000|baseline",
                "candidate_id": "c1",
                "error_type": "RuntimeError",
                "traceback": 'Traceback\n  File "/tmp/a.py", line 10, in run\nRuntimeError: x',
                "quality_reason_codes": ["DQ_DEGRADED_INPUT"],
                "asset_keys": ["AAA", "BBB"],
            },
            {
                "task_id": "c2|wf_001|baseline",
                "candidate_id": "c2",
                "error_type": "RuntimeError",
                "traceback": 'Traceback\n  File "/tmp/a.py", line 10, in run\nRuntimeError: x',
                "quality_reason_codes": ["DQ_DEGRADED_INPUT"],
                "asset_keys": ["AAA", "BBB"],
            },
            {
                "task_id": "c3|wf_002|baseline",
                "candidate_id": "c3",
                "error_type": "RuntimeError",
                "traceback": 'Traceback\n  File "/tmp/a.py", line 10, in run\nRuntimeError: x',
                "quality_reason_codes": ["DQ_DEGRADED_INPUT"],
                "asset_keys": ["AAA", "BBB"],
            },
        ]

        fired = False
        for r in rows:
            h._update_failure_tracker(tracker, r)
            abort, _reason = h._should_abort_systemic(tracker, r)
            fired = fired or abort
        self.assertFalse(fired)

    def test_systemic_signature_rule_triggers_abort(self) -> None:
        tracker: dict = {}
        rows = [
            {
                "task_id": "c1|wf_000|baseline",
                "candidate_id": "c1",
                "error_type": "TypeError",
                "traceback": 'Traceback\n  File "/tmp/core.py", line 77, in step\nTypeError: bad',
                "quality_reason_codes": [],
                "asset_keys": ["AAA", "BBB"],
            },
            {
                "task_id": "c2|wf_001|baseline",
                "candidate_id": "c2",
                "error_type": "TypeError",
                "traceback": 'Traceback\n  File "/tmp/core.py", line 77, in step\nTypeError: bad',
                "quality_reason_codes": [],
                "asset_keys": ["AAA", "BBB"],
            },
            {
                "task_id": "c3|wf_002|baseline",
                "candidate_id": "c3",
                "error_type": "TypeError",
                "traceback": 'Traceback\n  File "/tmp/core.py", line 77, in step\nTypeError: bad',
                "quality_reason_codes": [],
                "asset_keys": ["AAA", "BBB"],
            },
        ]

        last_abort = False
        last_reason = ""
        for r in rows:
            h._update_failure_tracker(tracker, r)
            last_abort, last_reason = h._should_abort_systemic(tracker, r)

        self.assertTrue(last_abort)
        self.assertIn("systemic_exception", last_reason)
        self.assertIn("units=3", last_reason)
        self.assertIn("assets=2", last_reason)
        self.assertIn("candidates=3", last_reason)

    def test_baseline_shortfall_ignores_localized_failures(self) -> None:
        rows = [
            {"status": "ok", "scenario_id": "baseline", "split_id": "wf_000", "quality_reason_codes": []},
            {"status": "ok", "scenario_id": "baseline", "split_id": "wf_001", "quality_reason_codes": []},
            {
                "status": "error",
                "scenario_id": "baseline",
                "split_id": "wf_002",
                "error_type": "NonFiniteExecutionPriceError",
                "quality_reason_codes": ["NONFINITE_EXEC_PX"],
            },
        ]
        reasons = h._baseline_failure_reasons(rows, expected_baseline_tasks=3)
        self.assertEqual(reasons, [])

    def test_baseline_shortfall_keeps_nonlocalized_errors(self) -> None:
        rows = [
            {"status": "ok", "scenario_id": "baseline", "split_id": "wf_000", "quality_reason_codes": []},
            {
                "status": "error",
                "scenario_id": "baseline",
                "split_id": "wf_001",
                "error_type": "ValueError",
                "quality_reason_codes": ["SCHEMA_ERROR"],
            },
        ]
        reasons = h._baseline_failure_reasons(rows, expected_baseline_tasks=2)
        self.assertTrue(any("baseline_ok_tasks=1 expected=2" in r for r in reasons))
        self.assertTrue(any("wf_001:ValueError" in r for r in reasons))


if __name__ == "__main__":
    unittest.main()
