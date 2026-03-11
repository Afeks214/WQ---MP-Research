from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from weightiz.cli import run_research
from weightiz.shared.config.models import HarnessConfigModel, RunConfigModel
from weightiz.module5.stage_a_discovery import (
    STAGE_A_LIVE_ENTRY_THRESHOLD,
    STAGE_A_PROCESS_BACKEND,
    STAGE_A_PROCESS_WORKERS,
    STAGE_A_RESEARCH_THRESHOLD,
    STAGE_A_TOTAL_CANDIDATES,
    STAGE_A_WINDOW_SET,
)
from scripts.build_stage_a_cloud_campaign import build_stage_a_cloud_config
from weightiz.module5.orchestrator import Module5HarnessConfig


def test_stage_a_threshold_is_owned_once_across_loader_and_runtime_defaults() -> None:
    assert HarnessConfigModel().robustness_reject_threshold == STAGE_A_RESEARCH_THRESHOLD
    assert Module5HarnessConfig().robustness_reject_threshold == STAGE_A_RESEARCH_THRESHOLD


def test_stage_a_cloud_builder_hits_locked_campaign_shape() -> None:
    config, plan = build_stage_a_cloud_config()
    validated = RunConfigModel.model_validate(config)

    assert validated.harness.research_mode == "discovery"
    assert validated.harness.parallel_backend == STAGE_A_PROCESS_BACKEND
    assert validated.harness.parallel_workers == STAGE_A_PROCESS_WORKERS
    assert validated.harness.robustness_reject_threshold == STAGE_A_RESEARCH_THRESHOLD
    assert len(validated.module2_configs) == 1
    assert tuple(validated.module3_configs[0].structural_windows) == STAGE_A_WINDOW_SET
    assert len(validated.candidates.specs) == STAGE_A_TOTAL_CANDIDATES
    assert int(plan["total_candidates"]) == STAGE_A_TOTAL_CANDIDATES
    assert int(plan["module2_config_count"]) == 1
    assert float(plan["live_gate_threshold"]) == STAGE_A_LIVE_ENTRY_THRESHOLD
    assert str(plan["live_gate_threshold_ownership"]) == "module4.entry_threshold"
    assert {float(cfg.entry_threshold) for cfg in validated.module4_configs} == {STAGE_A_LIVE_ENTRY_THRESHOLD}
    assert {int(spec.m2_idx) for spec in validated.candidates.specs} == {0}

    family_counts: dict[str, int] = {}
    for spec in validated.candidates.specs:
        family_id = next(tag.split("=", 1)[1] for tag in spec.tags if tag.startswith("family_id="))
        family_counts[family_id] = family_counts.get(family_id, 0) + 1

    assert family_counts == {
        "F1": 840,
        "F2": 840,
        "F3": 840,
        "F4": 840,
        "F5": 840,
        "F6": 800,
    }


def test_research_distribution_report_includes_stage_a_multi_window_surfaces() -> None:
    with tempfile.TemporaryDirectory(prefix="stage_a_report_") as td:
        run_dir = Path(td) / "run"
        run_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            [
                {
                    "candidate_id": "stagea_f1_h000_w015",
                    "family_id": "F1",
                    "family_name": "acceptance_rejection_geometry",
                    "hypothesis_id": "F1H000",
                    "parameter_hash": "p1",
                    "window_set": "15,20,30,40,60,90,240",
                    "evaluation_role": "window_probe",
                    "evaluation_window": 15,
                    "block_minutes": 15,
                    "cluster_id": 0,
                    "robustness_score": 0.52,
                    "execution_robustness": 0.61,
                    "cum_return": 0.03,
                    "cost_adjusted_expectancy": 0.010,
                    "cross_window_consistency_score": 0.72,
                    "cross_window_conflict_score": 0.12,
                    "multi_scale_stability_score": 0.68,
                    "overnight_suitability_score": 0.22,
                    "zimtra_compliance_flags": "",
                    "max_drawdown": 0.04,
                    "standard_reject": False,
                    "standard_pass": True,
                    "discovery_included": True,
                },
                {
                    "candidate_id": "stagea_f1_h000_w060",
                    "family_id": "F1",
                    "family_name": "acceptance_rejection_geometry",
                    "hypothesis_id": "F1H000",
                    "parameter_hash": "p1",
                    "window_set": "15,20,30,40,60,90,240",
                    "evaluation_role": "window_probe",
                    "evaluation_window": 60,
                    "block_minutes": 60,
                    "cluster_id": 0,
                    "robustness_score": 0.51,
                    "execution_robustness": 0.60,
                    "cum_return": 0.02,
                    "cost_adjusted_expectancy": 0.008,
                    "cross_window_consistency_score": 0.72,
                    "cross_window_conflict_score": 0.12,
                    "multi_scale_stability_score": 0.68,
                    "overnight_suitability_score": 0.24,
                    "zimtra_compliance_flags": "",
                    "max_drawdown": 0.03,
                    "standard_reject": False,
                    "standard_pass": True,
                    "discovery_included": True,
                },
                {
                    "candidate_id": "stagea_f5_h000_live",
                    "family_id": "F5",
                    "family_name": "multi_scale_alignment",
                    "hypothesis_id": "F5H000",
                    "parameter_hash": "p5",
                    "window_set": "15,20,30,40,60,90,240",
                    "evaluation_role": "multi_window_live",
                    "evaluation_window": np.nan,
                    "block_minutes": -1,
                    "cluster_id": 1,
                    "robustness_score": 0.49,
                    "execution_robustness": 0.58,
                    "cum_return": 0.01,
                    "cost_adjusted_expectancy": 0.006,
                    "cross_window_consistency_score": 0.64,
                    "cross_window_conflict_score": 0.18,
                    "multi_scale_stability_score": 0.62,
                    "overnight_suitability_score": 0.31,
                    "zimtra_compliance_flags": "fragile_execution",
                    "max_drawdown": 0.05,
                    "standard_reject": False,
                    "standard_pass": True,
                    "discovery_included": True,
                },
                {
                    "candidate_id": "stagea_f5_h000_w240",
                    "family_id": "F5",
                    "family_name": "multi_scale_alignment",
                    "hypothesis_id": "F5H000",
                    "parameter_hash": "p5",
                    "window_set": "15,20,30,40,60,90,240",
                    "evaluation_role": "window_probe",
                    "evaluation_window": 240,
                    "block_minutes": 240,
                    "cluster_id": 1,
                    "robustness_score": 0.44,
                    "execution_robustness": 0.56,
                    "cum_return": -0.01,
                    "cost_adjusted_expectancy": -0.004,
                    "cross_window_consistency_score": 0.64,
                    "cross_window_conflict_score": 0.18,
                    "multi_scale_stability_score": 0.62,
                    "overnight_suitability_score": 0.29,
                    "zimtra_compliance_flags": "fragile_execution",
                    "max_drawdown": 0.06,
                    "standard_reject": True,
                    "standard_pass": False,
                    "discovery_included": True,
                },
            ]
        ).to_csv(run_dir / "robustness_leaderboard.csv", index=False)

        pd.DataFrame(
            {
                "session_id": [1, 2, 3],
                "benchmark": [0.0, 0.0, 0.0],
                "stagea_f1_h000_w015": [0.01, 0.00, 0.02],
                "stagea_f1_h000_w060": [0.01, -0.01, 0.01],
                "stagea_f5_h000_live": [0.00, 0.01, 0.01],
                "stagea_f5_h000_w240": [0.00, -0.02, 0.01],
            }
        ).to_parquet(run_dir / "daily_returns.parquet", index=False)

        pd.DataFrame(
            {
                "candidate_id": [
                    "stagea_f1_h000_w015",
                    "stagea_f1_h000_w060",
                    "stagea_f5_h000_live",
                    "stagea_f5_h000_w240",
                ],
                "filled_qty": [1.0, 1.0, 1.0, 1.0],
            }
        ).to_parquet(run_dir / "trade_log.parquet", index=False)

        report = run_research._build_research_distribution_report(
            run_dir=run_dir,
            research_mode="discovery",
            plan_doc={"adaptive_local_run": {"family_entries": []}},
        )

        for key in [
            "family_level_summary",
            "window_level_summary",
            "candidate_count",
            "effective_return_signature_count",
            "cluster_count",
            "distinct_robustness_score_count",
            "distinct_execution_robustness_count",
            "count_standard_reject",
            "count_standard_pass",
            "count_discovery_included",
            "positive_expectancy_count",
            "positive_sharpe_count",
            "top_expectancy_pockets_by_family_window",
            "cross_window_consistency_summary",
        ]:
            assert key in report

        assert int(report["candidate_count"]) == 4
        assert int(report["cluster_count"]) == 2
        assert int(report["count_discovery_included"]) == 4
        assert int(report["positive_expectancy_count"]) == 3
        assert int(report["cross_window_consistency_summary"]["hypothesis_count"]) == 2
