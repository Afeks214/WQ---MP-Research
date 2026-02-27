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


if __name__ == "__main__":
    unittest.main()
