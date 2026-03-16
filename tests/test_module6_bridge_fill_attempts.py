from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from weightiz.module5.harness.module6_bridge import build_module6_bridge_artifacts


def test_module6_bridge_uses_session_fill_payload_for_rejected_attempts(tmp_path) -> None:
    report_root = tmp_path / "artifacts"
    report_root.mkdir(parents=True, exist_ok=True)

    build_module6_bridge_artifacts(
        report_root=report_root,
        run_id="run_fill_payload",
        execution_mode="serial",
        common_sessions=np.array([101], dtype=np.int64),
        canonical_reference_split_id="wf_000",
        canonical_reference_scenario_id="baseline",
        canonical_reference_policy="enabled_baseline_first_split_v1",
        baseline_candidate_ids=["cand_fill"],
        candidate_daily_mat=np.array([[0.0]], dtype=np.float64),
        candidates=[SimpleNamespace(candidate_id="cand_fill")],
        candidate_rows=[
            {
                "candidate_id": "cand_fill",
                "parameter_hash": "param_hash",
                "family_id": "F1",
                "hypothesis_id": "hypo_fill",
            }
        ],
        all_results=[
            {
                "candidate_id": "cand_fill",
                "split_id": "wf_000",
                "scenario_id": "baseline",
                "status": "ok",
                "quality_reason_codes": [],
                "dq_invalidated": False,
                "session_ids_exec": np.array([101], dtype=np.int64),
                "daily_returns_exec": np.array([0.0], dtype=np.float64),
                "session_ids_raw": np.array([101], dtype=np.int64),
                "daily_returns_raw": np.array([0.0], dtype=np.float64),
                "availability_state_session_ids": np.array([101], dtype=np.int64),
                "availability_state_codes": np.array([1], dtype=np.int16),
                "equity_payload": {
                    "ts_ns": np.array([1], dtype=np.int64),
                    "session_id": np.array([101], dtype=np.int64),
                    "equity": np.array([1_000_000.0], dtype=np.float64),
                    "margin_used": np.array([0.0], dtype=np.float64),
                    "buying_power": np.array([1_000_000.0], dtype=np.float64),
                    "daily_loss": np.array([0.0], dtype=np.float64),
                },
                "trade_payload": {
                    "ts_ns": np.zeros(0, dtype=np.int64),
                    "session_id": np.zeros(0, dtype=np.int64),
                    "filled_qty": np.zeros(0, dtype=np.float64),
                    "exec_price": np.zeros(0, dtype=np.float64),
                },
                "session_fill_payload": {
                    "session_id": np.array([101], dtype=np.int64),
                    "desired_qty_abs_sum": np.array([10.0], dtype=np.float64),
                    "unfilled_qty_abs_sum": np.array([10.0], dtype=np.float64),
                    "fill_cap_hit_count": np.array([1.0], dtype=np.float64),
                    "fill_reject_count": np.array([1.0], dtype=np.float64),
                },
                "micro_payload": None,
            }
        ],
        engine_cfg=SimpleNamespace(initial_cash=1_000_000.0),
        keep_symbols=["SPY"],
        dataset_hash="dataset_hash_fill_payload",
        require_pandas_fn=lambda: pd,
    )

    session_df = pd.read_parquet(report_root / "strategy_instance_session_returns.parquet")
    row = session_df.iloc[0]
    assert int(row["session_trade_count"]) == 0
    assert int(row["session_fill_cap_hit_count"]) == 1
    assert int(row["session_fill_reject_count"]) == 1
    assert float(row["session_fill_failure_rate"]) == 1.0
