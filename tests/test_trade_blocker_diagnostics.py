from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from weightiz.module5.harness.artifact_writers import write_json
from weightiz.module5.harness.trade_blocker_diagnostics import (
    build_candidate_path_diagnostics,
    write_trade_blocker_artifacts,
)


def test_build_candidate_path_diagnostics_flags_no_setup_and_zero_allocation() -> None:
    m4_sig = SimpleNamespace(
        regime_confidence_ta=np.zeros((3, 2), dtype=np.float64),
        intent_long_ta=np.zeros((3, 2), dtype=bool),
        intent_short_ta=np.zeros((3, 2), dtype=bool),
        target_qty_ta=np.zeros((3, 2), dtype=np.float64),
        conviction_net_ta=np.zeros((3, 2), dtype=np.float64),
        allocation_score_ta=np.zeros((3, 2), dtype=np.float64),
        decision_reason_code_ta=np.zeros((3, 2), dtype=np.int16),
    )
    risk_res = SimpleNamespace(
        execution_diagnostics={
            "desired_fill_attempt_count": 0,
            "desired_fill_qty_abs_sum": 0.0,
            "filled_trade_count": 0,
            "filled_qty_abs_sum": 0.0,
            "volume_cap_hit_count": 0,
            "volume_cap_rejected_count": 0,
            "volume_cap_desired_qty_abs_sum": 0.0,
            "volume_cap_filled_qty_abs_sum": 0.0,
            "volume_cap_clipped_qty_abs_sum": 0.0,
            "buying_power_cap_hit_count": 0,
            "buying_power_cap_desired_qty_abs_sum": 0.0,
            "buying_power_cap_filled_qty_abs_sum": 0.0,
            "buying_power_cap_clipped_qty_abs_sum": 0.0,
        }
    )

    row = build_candidate_path_diagnostics(
        task_id="cand|wf_000|baseline",
        candidate_id="cand",
        split_id="wf_000",
        scenario_id="baseline",
        status="ok",
        m2_idx=0,
        m3_idx=0,
        m4_idx=0,
        enabled_assets_mask=np.array([True, True], dtype=bool),
        quality_reason_codes=[],
        m4_sig=m4_sig,
        target_qty_exec=np.zeros((3, 2), dtype=np.float64),
        risk_res_exec=risk_res,
        trade_payload=None,
    )

    assert row["no_setup_candidate"] is True
    assert row["weak_signal_candidate"] is True
    assert row["zero_allocation_candidate"] is False
    assert row["has_fill_attempt"] is False


def test_write_trade_blocker_artifacts_emits_expected_files(tmp_path: Path) -> None:
    ok_result = {
        "task_id": "cand_a|wf_000|baseline",
        "candidate_id": "cand_a",
        "split_id": "wf_000",
        "scenario_id": "baseline",
        "status": "ok",
        "execution_path_diagnostics": {
            "task_id": "cand_a|wf_000|baseline",
            "candidate_id": "cand_a",
            "split_id": "wf_000",
            "scenario_id": "baseline",
            "status": "ok",
            "m2_idx": 0,
            "m3_idx": 0,
            "m4_idx": 0,
            "enabled_asset_count": 2,
            "quality_reason_codes": [],
            "signal_bars_any": 4,
            "confidence_bars_any": 4,
            "allocation_raw_bars_any": 4,
            "allocation_exec_bars_any": 4,
            "conviction_nonzero_cells": 8,
            "confidence_nonzero_cells": 8,
            "allocation_score_nonzero_cells": 8,
            "allocation_score_abs_mean": 0.2,
            "allocation_score_abs_p50": 0.2,
            "allocation_score_abs_p95": 0.3,
            "allocation_score_abs_max": 0.3,
            "conviction_abs_max": 0.4,
            "meaningful_allocation_score_floor": 0.05,
            "meaningful_signal_bars_any": 4,
            "never_crosses_meaningful_signal": False,
            "reason_low_regime_confidence_cells": 0,
            "reason_zero_score_cells": 0,
            "reason_zero_conviction_cells": 0,
            "reason_risk_filter_block_cells": 0,
            "reason_masked_not_tradable_cells": 0,
            "reason_invalid_input_cells": 0,
            "desired_fill_attempt_count": 2,
            "desired_fill_qty_abs_sum": 10.0,
            "filled_trade_count": 0,
            "filled_qty_abs_sum": 0.0,
            "volume_cap_hit_count": 2,
            "volume_cap_rejected_count": 0,
            "volume_cap_desired_qty_abs_sum": 10.0,
            "volume_cap_filled_qty_abs_sum": 4.0,
            "volume_cap_clipped_qty_abs_sum": 6.0,
            "buying_power_cap_hit_count": 0,
            "buying_power_cap_desired_qty_abs_sum": 0.0,
            "buying_power_cap_filled_qty_abs_sum": 0.0,
            "buying_power_cap_clipped_qty_abs_sum": 0.0,
            "daily_loss_breach_count": 0,
            "account_disable_breach_count": 0,
            "overnight_exposure_breach_count": 0,
            "trade_log_rows": 0,
            "candidate_generated": 1,
            "has_signal_opportunity": True,
            "has_allocation_raw": True,
            "has_allocation_exec": True,
            "has_fill_attempt": True,
            "has_filled_trade": False,
            "has_trade_log": False,
            "no_setup_candidate": False,
            "weak_signal_candidate": False,
            "zero_allocation_candidate": False,
            "risk_choke_candidate": False,
            "volume_cap_choke_candidate": True,
            "other_blocker_candidate": False,
        },
    }
    error_result = {
        "task_id": "cand_b|wf_000|baseline",
        "candidate_id": "cand_b",
        "split_id": "wf_000",
        "scenario_id": "baseline",
        "status": "error",
        "m2_idx": 0,
        "m3_idx": 0,
        "m4_idx": 1,
        "asset_keys": ["SPY", "QQQ"],
        "quality_reason_codes": [],
        "error_type": "RuntimeError",
        "error": "RISK_DAILY_LOSS_BREACH",
    }

    paths, summary = write_trade_blocker_artifacts(
        all_results=[ok_result, error_result],
        report_root=tmp_path,
        require_pandas_fn=lambda: pd,
        write_json_fn=write_json,
    )

    assert Path(paths["candidate_path_diagnostics"]).exists()
    assert Path(paths["execution_funnel_diagnostics"]).exists()
    assert Path(paths["trade_blocker_summary"]).exists()
    candidate_df = pd.read_parquet(paths["candidate_path_diagnostics"])
    funnel_df = pd.read_parquet(paths["execution_funnel_diagnostics"])
    assert candidate_df.shape[0] == 2
    assert funnel_df.shape[0] >= 1
    assert summary["volume_cap_analysis"]["candidate_paths_volume_cap_choke"] == 1
    assert summary["risk_choke_analysis"]["daily_loss_breach_paths"] == 1
