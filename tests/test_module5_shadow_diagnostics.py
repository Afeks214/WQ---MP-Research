from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from weightiz.module1.core import EngineConfig
from weightiz.module2.core import Module2Config
from weightiz.module3.bridge import Module3Config
from weightiz.module4.strategy_funnel import Module4Config
from weightiz.module5.harness.candidate_artifacts import build_candidate_artifacts


def _engine_cfg() -> EngineConfig:
    return EngineConfig(
        T=1,
        A=1,
        B=16,
        tick_size=np.asarray([0.01], dtype=np.float64),
        mode="sealed",
        timezone="America/New_York",
        initial_cash=1_000.0,
    )


def _candidate() -> SimpleNamespace:
    return SimpleNamespace(
        candidate_id="cand_shadow",
        m2_idx=0,
        m3_idx=0,
        m4_idx=0,
        enabled_assets_mask=np.asarray([True], dtype=bool),
        tags=(),
    )


def _row(*, scenario_id: str, daily_returns: list[float], trades: int, risk_engine_metrics: dict[str, float] | None = None) -> dict[str, object]:
    session_ids = np.asarray([101, 102, 103], dtype=np.int64)
    trade_payload = {
        "session_id": session_ids[: max(trades, 1)],
        "filled_qty": np.asarray([1.0] * max(trades, 1), dtype=np.float64),
        "exec_price": np.asarray([100.0] * max(trades, 1), dtype=np.float64),
    }
    if trades <= 0:
        trade_payload = {
            "session_id": np.zeros(0, dtype=np.int64),
            "filled_qty": np.zeros(0, dtype=np.float64),
            "exec_price": np.zeros(0, dtype=np.float64),
        }
    return {
        "candidate_id": "cand_shadow",
        "split_id": "wf_000",
        "scenario_id": scenario_id,
        "status": "ok",
        "daily_returns": np.asarray(daily_returns, dtype=np.float64),
        "trade_payload": trade_payload,
        "equity_payload": None,
        "m2_idx": 0,
        "m3_idx": 0,
        "m4_idx": 0,
        "tags": [],
        "test_days": len(daily_returns),
        "dqs_min": 1.0,
        "dqs_median": 1.0,
        "quality_reason_codes": [],
        "risk_engine_metrics": dict(risk_engine_metrics or {}),
    }


def _build_shadow_candidate_artifacts(tmp_path: Path, scenarios: list[SimpleNamespace], rows: list[dict[str, object]]):
    return build_candidate_artifacts(
        report_root=tmp_path,
        run_id="shadow_run",
        run_started_utc=datetime.now(timezone.utc),
        git_hash="deadbeef",
        candidates=[_candidate()],
        all_results=rows,
        candidate_daily_mat=np.asarray([[0.02], [0.01], [0.005]], dtype=np.float64),
        daily_bmk=np.zeros(3, dtype=np.float64),
        common_sessions=np.asarray([101, 102, 103], dtype=np.int64),
        baseline_candidate_ids=["cand_shadow"],
        candidate_scenario_series={"cand_shadow": {"baseline": {101: 0.02, 102: 0.01, 103: 0.005}}},
        candidate_verdict={},
        expected_baseline_tasks=1,
        scenarios=scenarios,
        engine_cfg=_engine_cfg(),
        m2_configs=[Module2Config()],
        m3_configs=[Module3Config()],
        m4_configs=[Module4Config()],
        harness_cfg=SimpleNamespace(seed=17, cpcv_slices=2, cpcv_k_test=1),
        require_pandas_fn=lambda: pd,
        write_json_fn=lambda path, payload: Path(path).write_text(
            json.dumps(
                payload,
                sort_keys=True,
                default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value),
            ),
            encoding="utf-8",
        ),
        baseline_failure_reasons_fn=lambda rows_base_all, expected_baseline_tasks: [],
        clip01_fn=lambda x: float(np.clip(float(x), 0.0, 1.0)),
        cum_return_fn=lambda arr: float(np.sum(np.asarray(arr, dtype=np.float64))),
        max_drawdown_from_returns_fn=lambda arr: 0.0,
        turnover_from_trade_payload_fn=lambda payload, initial_cash: float(len(payload.get("filled_qty", []))) if isinstance(payload, dict) else 0.0,
        sharpe_daily_fn=lambda arr: 0.0,
        trade_count_from_payload_fn=lambda payload: int(len(payload.get("filled_qty", []))) if isinstance(payload, dict) else 0,
        margin_exposure_stats_from_equity_payloads_fn=lambda payloads: {"avg_margin_used_frac": 0.0, "peak_margin_used_frac": 0.0},
        asset_pnl_concentration_from_result_rows_fn=lambda rows: 0.0,
        asset_notional_concentration_from_trade_payloads_fn=lambda payloads: 0.0,
        robustness_caps={"dd_cap": 1.0, "std_cap": 1.0, "conc_cap": 1.0},
    )


def test_degradation_shadow_metrics_are_emitted_for_degradation_scenarios(tmp_path: Path) -> None:
    scenarios = [
        SimpleNamespace(scenario_id="baseline", scenario_group="stress"),
        SimpleNamespace(scenario_id="deg_lag_1", scenario_group="degradation"),
    ]
    rows = [
        _row(scenario_id="baseline", daily_returns=[0.02, 0.01, 0.005], trades=4),
        _row(scenario_id="deg_lag_1", daily_returns=[0.015, 0.01, 0.0], trades=3),
    ]

    candidate_rows, _robustness_rows, _plateaus = _build_shadow_candidate_artifacts(tmp_path, scenarios, rows)
    row = candidate_rows[0]

    assert "degradation_score" in row
    assert "degradation_fragile" in row
    assert np.isfinite(float(row["degradation_score"]))
    assert bool(row["degradation_fragile"]) is False


def test_degradation_metrics_remain_inactive_when_no_degradation_scenarios_are_present(tmp_path: Path) -> None:
    scenarios = [SimpleNamespace(scenario_id="baseline", scenario_group="stress")]
    rows = [_row(scenario_id="baseline", daily_returns=[0.02, 0.01, 0.005], trades=4)]

    candidate_rows, _robustness_rows, _plateaus = _build_shadow_candidate_artifacts(tmp_path, scenarios, rows)
    row = candidate_rows[0]

    assert np.isnan(float(row["degradation_score"]))
    assert bool(row["degradation_fragile"]) is False


def test_cost_adjusted_expectancy_is_emitted_without_falsely_surfacing_incomplete_cost_objectives(tmp_path: Path) -> None:
    scenarios = [SimpleNamespace(scenario_id="baseline", scenario_group="stress")]
    rows = [_row(scenario_id="baseline", daily_returns=[0.02, 0.01, 0.005], trades=4)]

    candidate_rows, _robustness_rows, _plateaus = _build_shadow_candidate_artifacts(tmp_path, scenarios, rows)
    row = candidate_rows[0]
    metrics = json.loads((tmp_path / "candidates" / "cand_shadow" / "candidate_metrics.json").read_text(encoding="utf-8"))

    expected = float(np.mean(np.asarray([0.02, 0.01, 0.005], dtype=np.float64)))
    np.testing.assert_allclose(float(row["cost_adjusted_expectancy"]), expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(float(metrics["stage_a_metadata"]["cost_adjusted_expectancy"]), expected, rtol=0.0, atol=1e-12)
    for key in ("net_ir_after_costs", "probability_exceed_explicit_cost", "stressed_cost_survival_score"):
        assert key not in row
        assert key not in metrics


def test_capacity_shadow_metrics_emit_redline_summary_for_scaled_scenarios(tmp_path: Path) -> None:
    scenarios = [
        SimpleNamespace(scenario_id="baseline", scenario_group="stress", target_scale_mult=1.0),
        SimpleNamespace(scenario_id="cap_10x", scenario_group="capacity", target_scale_mult=10.0),
    ]
    rows = [
        _row(
            scenario_id="baseline",
            daily_returns=[0.02, 0.01, 0.005],
            trades=4,
            risk_engine_metrics={
                "slippage_cost_total_exec": 1.0,
                "commission_cost_total_exec": 1.0,
                "desired_fill_qty_abs_sum_exec": 100.0,
                "volume_cap_clipped_qty_abs_sum_exec": 0.0,
            },
        ),
        _row(
            scenario_id="cap_10x",
            daily_returns=[-0.01, -0.005, 0.0],
            trades=8,
            risk_engine_metrics={
                "slippage_cost_total_exec": 8.0,
                "commission_cost_total_exec": 4.0,
                "desired_fill_qty_abs_sum_exec": 1_000.0,
                "volume_cap_clipped_qty_abs_sum_exec": 300.0,
            },
        ),
    ]

    candidate_rows, _robustness_rows, _plateaus = _build_shadow_candidate_artifacts(tmp_path, scenarios, rows)
    row = candidate_rows[0]

    assert float(row["capacity_redline_scale"]) == 10.0
    np.testing.assert_allclose(float(row["capacity_fill_failure_rate_10x"]), 0.3, rtol=0.0, atol=1e-12)
    assert float(row["capacity_expectancy_ratio_10x"]) < 0.0


def test_capacity_shadow_metrics_do_not_redline_on_tiny_negative_expectancy_without_fill_breakage(tmp_path: Path) -> None:
    scenarios = [
        SimpleNamespace(scenario_id="baseline", scenario_group="stress", target_scale_mult=1.0),
        SimpleNamespace(scenario_id="cap_10x", scenario_group="capacity", target_scale_mult=10.0),
    ]
    rows = [
        _row(
            scenario_id="baseline",
            daily_returns=[0.02, 0.01, 0.005],
            trades=4,
            risk_engine_metrics={
                "desired_fill_qty_abs_sum_exec": 100.0,
                "volume_cap_clipped_qty_abs_sum_exec": 0.0,
            },
        ),
        _row(
            scenario_id="cap_10x",
            daily_returns=[-1.0e-6, 0.0, 0.0],
            trades=4,
            risk_engine_metrics={
                "desired_fill_qty_abs_sum_exec": 100.0,
                "volume_cap_clipped_qty_abs_sum_exec": 5.0,
            },
        ),
    ]

    candidate_rows, _robustness_rows, _plateaus = _build_shadow_candidate_artifacts(tmp_path, scenarios, rows)
    row = candidate_rows[0]

    assert np.isnan(float(row["capacity_redline_scale"]))
    np.testing.assert_allclose(float(row["capacity_fill_failure_rate_10x"]), 0.05, rtol=0.0, atol=1e-12)
    assert float(row["capacity_expectancy_ratio_10x"]) < 0.0
