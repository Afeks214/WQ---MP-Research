from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import weightiz_module5_harness as h
from weightiz_module1_core import EngineConfig, NS_PER_MIN, preallocate_state


class TestOptionARiskBreachStateDump(unittest.TestCase):
    def _mk_state(self):
        start_ns = np.datetime64("2025-01-06T14:30:00", "ns").astype(np.int64)
        ts_ns = start_ns + np.arange(6, dtype=np.int64) * np.int64(NS_PER_MIN)
        cfg = EngineConfig(T=6, A=2, B=64, tick_size=np.asarray([0.01, 0.01], dtype=np.float64))
        st = preallocate_state(ts_ns=ts_ns, cfg=cfg, symbols=("AAA", "BBB"))
        st.close_px[:] = np.asarray(
            [[100.0, 50.0], [101.0, 51.0], [102.0, 52.0], [103.0, 53.0], [104.0, 54.0], [105.0, 55.0]],
            dtype=np.float64,
        )
        st.position_qty[:] = np.asarray(
            [[0.0, 0.0], [10.0, -5.0], [10.0, -5.0], [12.0, -6.0], [12.0, -6.0], [12.0, -6.0]],
            dtype=np.float64,
        )
        st.available_cash[:] = np.linspace(1_000_000.0, 990_000.0, 6, dtype=np.float64)
        st.realized_pnl[:] = np.linspace(0.0, 1250.0, 6, dtype=np.float64)
        st.equity[:] = np.linspace(1_000_000.0, 995_000.0, 6, dtype=np.float64)
        st.margin_used[:] = np.linspace(0.0, 2_050_000.0, 6, dtype=np.float64)
        st.leverage_limit[:] = np.full(6, 2.0, dtype=np.float64)
        st.buying_power[:] = st.equity * st.leverage_limit - st.margin_used
        return st

    def test_risk_breach_does_not_trigger_systemic_and_writes_state_dump(self) -> None:
        st = self._mk_state()
        dump = h._build_risk_constraint_state_dump(
            st,
            t=5,
            candidate_id="c0",
            split_id="wf_001",
            scenario_id="baseline",
        )
        required = {
            "t",
            "ts_utc",
            "candidate_id",
            "split_id",
            "scenario_id",
            "equity_t",
            "margin_used_t",
            "leverage_limit_t",
            "buying_power_t",
            "cash_t",
            "realized_pnl_t",
            "max_margin_allowed_t",
            "assets",
        }
        self.assertTrue(required.issubset(set(dump.keys())))
        self.assertEqual(len(dump["assets"]), 2)

        tracker: dict = {}
        rows = []
        for i, cid in enumerate(["c1", "c2", "c3"]):
            rows.append(
                {
                    "task_id": f"{cid}|wf_001|baseline",
                    "candidate_id": cid,
                    "split_id": "wf_001",
                    "scenario_id": "baseline",
                    "error_type": "RuntimeError",
                    "error": "RuntimeError: Intraday leverage breach at t=5: margin_used=2050000.0, max=1990000.0",
                    "traceback": 'Traceback\n  File "weightiz_module1_core.py", line 727, in _validate_portfolio_constraints\nRuntimeError: Intraday leverage breach',
                    "top_frame": "weightiz_module1_core.py:727:_validate_portfolio_constraints",
                    "error_hash": f"h{i}",
                    "task_seed": i,
                    "asset_keys": ["AAA", "BBB"],
                    "quality_reason_codes": ["RISK_CONSTRAINT_BREACH"],
                    "state_dump": dict(dump, candidate_id=cid),
                }
            )

        with tempfile.TemporaryDirectory() as td:
            deadletter = Path(td) / "deadletter.jsonl"
            for row in rows:
                h._update_failure_tracker(tracker, row)
                abort, _reason = h._should_abort_systemic(tracker, row)
                self.assertFalse(abort)
                h._record_deadletter(deadletter, row)

            lines = [json.loads(x) for x in deadletter.read_text(encoding="utf-8").splitlines() if x.strip()]
            self.assertEqual(len(lines), 3)
            first = lines[0]
            self.assertIn("RISK_CONSTRAINT_BREACH", first.get("reason_codes", []))
            sd = first.get("state_dump", {})
            self.assertTrue(required.issubset(set(sd.keys())))
            self.assertIsInstance(sd.get("assets"), list)
            self.assertGreaterEqual(len(sd.get("assets", [])), 1)


if __name__ == "__main__":
    unittest.main()
