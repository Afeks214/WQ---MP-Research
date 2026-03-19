from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from weightiz.shared.config.builders import (
    apply_calibrated_module6_gate_values,
    build_harness_config,
    build_harness_gate_inputs,
    build_module6_gate_inputs,
    build_run_geometry,
)
from weightiz.shared.config.models import RunConfigModel
from weightiz.shared.gate_calibrator import EmpiricalGateStats, GateCalibrator


def _base_geometry() -> object:
    class _Harness:
        wf_train_sessions = 40
        wf_test_sessions = 20
        wf_step_sessions = 20
        cpcv_slices = 10
        cpcv_k_test = 5
        disable_cpcv_splits = False

    return build_run_geometry(_Harness(), data_sessions=120, common_sessions=60)


class TestGeometryDerivedGate:
    def test_geometry_derived_gates_follow_formula_and_do_not_weaken(self) -> None:
        geometry = _base_geometry()
        calibrator = GateCalibrator()
        baseline_min = 0.20
        baseline_support = 0.30
        out = calibrator.calibrate(
            geometry=geometry,
            gate_inputs={
                "min_availability_ratio": {"value": baseline_min, "force_static": False},
                "required_comparison_support": {"value": baseline_support, "force_static": False},
            },
            empirical_stats=None,
        )
        expected_min = max(
            baseline_min,
            np.clip(max(20.0 / 60.0, 5.0 / 10.0) + 1.0 / 60.0, 0.05, 0.98),
        )
        expected_support = max(
            baseline_support,
            np.clip(max(20.0 / 60.0, 5.0 / 10.0) + 2.0 / 60.0, 0.10, 0.99),
        )
        assert out.derived_values["min_availability_ratio"] == pytest.approx(expected_min)
        assert out.derived_values["required_comparison_support"] == pytest.approx(expected_support)
        assert out.derived_values["min_availability_ratio"] >= baseline_min
        assert out.derived_values["required_comparison_support"] >= baseline_support


class TestDistributionDerivedGate:
    def test_distribution_derived_gate_uses_empirical_quantile_and_mad(self) -> None:
        geometry = _base_geometry()
        robustness = np.asarray([0.61, 0.65, 0.68, 0.72, 0.75, 0.77, 0.79, 0.81, 0.83, 0.85] * 3, dtype=np.float64)
        baseline = 0.40
        calibrator = GateCalibrator()
        out = calibrator.calibrate(
            geometry=geometry,
            gate_inputs={"robustness_reject_threshold": {"value": baseline, "force_static": False}},
            empirical_stats=EmpiricalGateStats(robustness_scores=robustness),
        )
        q30 = float(np.quantile(robustness, 0.30))
        med = float(np.median(robustness))
        mad = float(np.median(np.abs(robustness - med)))
        expected = max(baseline, float(np.clip(max(q30, med - 1.4826 * mad), 0.0, 0.99)))
        assert out.derived_values["robustness_reject_threshold"] == pytest.approx(expected)
        assert out.derived_values["robustness_reject_threshold"] >= baseline


class TestControlLimitGate:
    def test_control_limit_gate_uses_past_only_median_plus_k_mad(self) -> None:
        geometry = _base_geometry()
        fill_failure = np.asarray(
            [
                0.05,
                0.04,
                0.06,
                0.05,
                0.04,
                0.05,
                0.05,
                0.04,
                0.05,
                0.05,
                0.04,
                0.05,
                0.05,
                0.04,
                0.05,
                0.05,
                0.04,
                0.05,
                0.05,
                0.04,
                0.05,
                0.05,
                0.04,
                0.05,
                0.05,
                0.04,
                0.05,
                0.05,
                0.04,
                0.30,
            ],
            dtype=np.float64,
        )
        baseline = 0.25
        calibrator = GateCalibrator()
        out = calibrator.calibrate(
            geometry=geometry,
            gate_inputs={"fill_failure_control_limit": {"value": baseline, "force_static": False}},
            empirical_stats=EmpiricalGateStats(fill_failure_rates=fill_failure, fold_count=5),
        )
        hist = fill_failure[:-1]
        med = float(np.median(hist))
        mad = float(np.median(np.abs(hist - med)))
        expected = max(baseline, float(np.clip(med + 3.0 * 1.4826 * mad, 0.05, 0.95)))
        assert out.derived_values["fill_failure_control_limit"] == pytest.approx(expected)
        assert out.derived_values["fill_failure_control_limit"] >= baseline


class TestStaticPolicyGateNonOverride:
    def test_static_policy_gate_is_never_dynamically_overridden(self) -> None:
        geometry = _base_geometry()
        calibrator = GateCalibrator()
        out = calibrator.calibrate(
            geometry=geometry,
            gate_inputs={"cluster_corr_threshold": {"value": 0.90, "force_static": False}},
            empirical_stats=EmpiricalGateStats(robustness_scores=np.linspace(0.0, 1.0, 100)),
        )
        assert out.derived_values["cluster_corr_threshold"] == pytest.approx(0.90)
        assert "POLICY_GATE_STATIC" in out.calibration_reasons["cluster_corr_threshold"]


class TestFallbackOnInsufficientData:
    def test_distribution_gate_falls_back_when_data_is_insufficient(self) -> None:
        geometry = _base_geometry()
        baseline = 0.45
        calibrator = GateCalibrator()
        out = calibrator.calibrate(
            geometry=geometry,
            gate_inputs={"robustness_reject_threshold": {"value": baseline, "force_static": False}},
            empirical_stats=EmpiricalGateStats(robustness_scores=np.asarray([0.1, 0.2, 0.3], dtype=np.float64)),
        )
        assert out.derived_values["robustness_reject_threshold"] == pytest.approx(baseline)
        assert out.fallback_flags["robustness_reject_threshold"] is True


class TestForceStaticOverride:
    def test_force_static_override_is_respected(self) -> None:
        geometry = _base_geometry()
        baseline = 0.33
        calibrator = GateCalibrator()
        out = calibrator.calibrate(
            geometry=geometry,
            gate_inputs={"min_availability_ratio": {"value": baseline, "force_static": True}},
            empirical_stats=None,
        )
        assert out.derived_values["min_availability_ratio"] == pytest.approx(baseline)
        assert "force_static=True" in out.calibration_reasons["min_availability_ratio"]


class TestIdempotency:
    def test_calibration_is_idempotent_for_identical_inputs(self) -> None:
        geometry = _base_geometry()
        empirical = EmpiricalGateStats(
            robustness_scores=np.linspace(0.3, 0.9, 50),
            execution_robustness_scores=np.linspace(0.2, 0.95, 50),
            fill_failure_rates=np.linspace(0.01, 0.10, 50),
            fold_count=10,
        )
        gate_inputs = {
            "min_availability_ratio": {"value": 0.20, "force_static": False},
            "required_comparison_support": {"value": 0.30, "force_static": False},
            "robustness_reject_threshold": {"value": 0.40, "force_static": False},
            "execution_fragile_threshold": {"value": 0.50, "force_static": False},
            "fill_failure_control_limit": {"value": 0.25, "force_static": False},
        }
        calibrator = GateCalibrator()
        out_a = calibrator.calibrate(geometry=geometry, gate_inputs=gate_inputs, empirical_stats=empirical)
        out_b = calibrator.calibrate(geometry=geometry, gate_inputs=gate_inputs, empirical_stats=empirical)
        assert out_a.to_dict() == out_b.to_dict()


class TestBuilderIntegration:
    def test_builders_integrate_with_gate_calibration_and_module6_payload(self, tmp_path: Path) -> None:
        cfg = RunConfigModel.model_validate(
            {
                "symbols": ["SPY", "QQQ"],
                "harness": {
                    "wf_train_sessions": 40,
                    "wf_test_sessions": 20,
                    "wf_step_sessions": 20,
                    "cpcv_slices": 10,
                    "cpcv_k_test": 5,
                    "disable_cpcv_splits": True,
                    "robustness_reject_threshold": 0.40,
                    "execution_fragile_threshold": 0.50,
                    "gate_overrides": {
                        "robustness_reject_threshold": {"value": 0.44, "force_static": True},
                    },
                },
                "module6": {
                    "intake": {
                        "min_availability_ratio": 0.20,
                        "required_comparison_support": 0.25,
                    }
                },
            }
        )
        harness_cfg = build_harness_config(cfg, tmp_path)
        assert bool(harness_cfg.disable_cpcv_splits) is True
        assert float(harness_cfg.robustness_reject_threshold) == pytest.approx(0.44)
        assert float(harness_cfg.execution_fragile_threshold) == pytest.approx(0.50)

        geometry = build_run_geometry(harness_cfg, data_sessions=120, common_sessions=60)
        assert bool(geometry.disable_cpcv_splits) is True
        pre_cal = GateCalibrator().calibrate(
            geometry=geometry,
            gate_inputs=build_module6_gate_inputs(cfg.module6),
            empirical_stats=None,
        )
        calibrated_module6 = apply_calibrated_module6_gate_values(cfg.module6, pre_cal)
        assert float(calibrated_module6["intake"]["min_availability_ratio"]) >= 0.20
        assert float(calibrated_module6["intake"]["required_comparison_support"]) >= 0.25


class TestHarnessBuilderPassivity:
    def test_build_harness_config_keeps_thresholds_static_except_explicit_override(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _unexpected_calibration_call(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("build_harness_config must not call GateCalibrator.calibrate")

        monkeypatch.setattr(GateCalibrator, "calibrate", _unexpected_calibration_call)

        cfg = RunConfigModel.model_validate(
            {
                "symbols": ["SPY", "QQQ"],
                "harness": {
                    "wf_train_sessions": 40,
                    "wf_test_sessions": 20,
                    "wf_step_sessions": 20,
                    "cpcv_slices": 10,
                    "cpcv_k_test": 5,
                    "robustness_reject_threshold": 0.31,
                    "execution_fragile_threshold": 0.42,
                },
            }
        )
        harness_cfg = build_harness_config(cfg, tmp_path)
        assert float(harness_cfg.robustness_reject_threshold) == pytest.approx(0.31)
        assert float(harness_cfg.execution_fragile_threshold) == pytest.approx(0.42)

        cfg_override = RunConfigModel.model_validate(
            {
                "symbols": ["SPY", "QQQ"],
                "harness": {
                    "wf_train_sessions": 40,
                    "wf_test_sessions": 20,
                    "wf_step_sessions": 20,
                    "cpcv_slices": 10,
                    "cpcv_k_test": 5,
                    "robustness_reject_threshold": 0.31,
                    "execution_fragile_threshold": 0.42,
                    "gate_overrides": {
                        "robustness_reject_threshold": {"value": 0.57, "force_static": True},
                    },
                },
            }
        )
        harness_cfg_override = build_harness_config(cfg_override, tmp_path)
        assert float(harness_cfg_override.robustness_reject_threshold) == pytest.approx(0.57)
        assert float(harness_cfg_override.execution_fragile_threshold) == pytest.approx(0.42)


class TestCalibrationLogCompleteness:
    def test_calibration_log_reasons_and_fallbacks_are_complete(self) -> None:
        geometry = _base_geometry()
        calibrator = GateCalibrator()
        out = calibrator.calibrate(
            geometry=geometry,
            gate_inputs={
                "min_availability_ratio": {"value": 0.20, "force_static": False},
                "robustness_reject_threshold": {"value": 0.40, "force_static": False},
                "cluster_corr_threshold": {"value": 0.90, "force_static": False},
                "min_observed_sessions": {"value": 20, "force_static": False},
            },
            empirical_stats=EmpiricalGateStats(robustness_scores=np.linspace(0.1, 0.9, 50)),
        )
        keys = set(out.derived_values.keys())
        assert keys == set(out.calibration_reasons.keys())
        assert keys == set(out.fallback_flags.keys())
        assert len(out.calibration_log) == len(keys)
        assert any("POLICY_GATE_STATIC" in line for line in out.calibration_log)
        assert any("SAFETY_GATE_STATIC" in line for line in out.calibration_log)


class TestBackwardCompatibility:
    def test_extra_fields_still_rejected_and_post_run_calibration_is_report_only(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            RunConfigModel.model_validate(
                {
                    "symbols": ["SPY", "QQQ"],
                    "harness": {"totally_unknown_field": 1},
                }
            )

        cfg = RunConfigModel.model_validate({"symbols": ["SPY", "QQQ"]})
        harness_cfg = build_harness_config(cfg, tmp_path)
        baseline_reject = float(harness_cfg.robustness_reject_threshold)
        baseline_fragile = float(harness_cfg.execution_fragile_threshold)

        post_cal = GateCalibrator().calibrate(
            geometry=build_run_geometry(harness_cfg, data_sessions=120, common_sessions=60),
            gate_inputs={
                **build_harness_gate_inputs(harness_cfg),
                "fill_failure_control_limit": {"value": 0.25, "force_static": False},
            },
            empirical_stats=EmpiricalGateStats(
                robustness_scores=np.linspace(0.6, 0.95, 100),
                execution_robustness_scores=np.linspace(0.6, 0.95, 100),
                fill_failure_rates=np.linspace(0.01, 0.20, 100),
                fold_count=10,
            ),
        )

        # Post-run calibration is report-only in this pass: base runtime thresholds stay unchanged.
        assert float(harness_cfg.robustness_reject_threshold) == pytest.approx(baseline_reject)
        assert float(harness_cfg.execution_fragile_threshold) == pytest.approx(baseline_fragile)
        assert post_cal.derived_values["robustness_reject_threshold"] >= baseline_reject
        assert post_cal.derived_values["execution_fragile_threshold"] >= baseline_fragile
