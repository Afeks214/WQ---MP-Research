from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


_MAD_SCALE = 1.4826


@dataclass(frozen=True)
class RunGeometry:
    data_sessions: int
    common_sessions: int
    wf_train_sessions: int
    wf_test_sessions: int
    wf_step_sessions: int
    cpcv_slices: int
    cpcv_k_test: int
    disable_cpcv_splits: bool = False


@dataclass(frozen=True)
class EmpiricalGateStats:
    robustness_scores: np.ndarray | list[float] | None = None
    execution_robustness_scores: np.ndarray | list[float] | None = None
    fill_failure_rates: np.ndarray | list[float] | None = None
    support_coverages: np.ndarray | list[float] | None = None
    availability_ratios: np.ndarray | list[float] | None = None
    observed_session_counts: np.ndarray | list[int] | None = None
    fold_count: int | None = None


@dataclass(frozen=True)
class CalibratedGates:
    derived_values: dict[str, float]
    calibration_reasons: dict[str, str]
    fallback_flags: dict[str, bool]
    calibration_log: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "derived_values": {str(k): float(v) for k, v in sorted(self.derived_values.items())},
            "calibration_reasons": {str(k): str(v) for k, v in sorted(self.calibration_reasons.items())},
            "fallback_flags": {str(k): bool(v) for k, v in sorted(self.fallback_flags.items())},
            "calibration_log": [str(x) for x in self.calibration_log],
        }


@dataclass(frozen=True)
class _GateSetting:
    value: float
    force_static: bool


class GateCalibrator:
    """
    Deterministic gate calibrator:
    - geometric gates: min_availability_ratio, required_comparison_support
    - distribution gates: robustness_reject_threshold, execution_fragile_threshold
    - control-limit gate: fill_failure_control_limit (reporting surface)
    """

    def __init__(
        self,
        *,
        min_distribution_samples: int = 20,
        control_limit_warmup: int = 20,
        control_limit_k: float = 3.0,
    ) -> None:
        self._min_distribution_samples = int(max(1, min_distribution_samples))
        self._control_limit_warmup = int(max(1, control_limit_warmup))
        self._control_limit_k = float(max(0.0, control_limit_k))
        self._dynamic_geometry_gates = {
            "min_availability_ratio",
            "required_comparison_support",
        }
        self._dynamic_distribution_gates = {
            "robustness_reject_threshold",
            "execution_fragile_threshold",
        }
        self._dynamic_control_limit_gates = {
            "fill_failure_control_limit",
        }
        self._static_policy_gates = {
            "cluster_corr_threshold",
            "duplicate_corr_threshold",
            "run_policy_class",
            "canonical_selection_stage",
            "support_policy_version",
            "ranking_policy_version",
        }
        self._static_safety_gates = {
            "min_observed_sessions",
            "execution_max_volume_participation",
            "execution_dynamic_slippage_bps_cap",
            "daily_loss_limit_frac",
        }

    def calibrate(
        self,
        *,
        geometry: RunGeometry,
        gate_inputs: dict[str, Any],
        empirical_stats: EmpiricalGateStats | None = None,
    ) -> CalibratedGates:
        derived: dict[str, float] = {}
        reasons: dict[str, str] = {}
        fallback_flags: dict[str, bool] = {}
        log_lines: list[str] = []
        min_availability_ratio_value: float | None = None
        for gate_id in sorted(gate_inputs.keys()):
            setting = self._coerce_gate_setting(gate_id=gate_id, raw=gate_inputs[gate_id])
            if gate_id in self._static_policy_gates:
                self._record_static(
                    gate_id=gate_id,
                    setting=setting,
                    reason="POLICY_GATE_STATIC",
                    derived=derived,
                    reasons=reasons,
                    fallback_flags=fallback_flags,
                    log_lines=log_lines,
                )
                continue
            if gate_id in self._static_safety_gates:
                self._record_static(
                    gate_id=gate_id,
                    setting=setting,
                    reason="SAFETY_GATE_STATIC",
                    derived=derived,
                    reasons=reasons,
                    fallback_flags=fallback_flags,
                    log_lines=log_lines,
                )
                continue
            if gate_id in self._dynamic_geometry_gates:
                value, reason, fallback = self._calibrate_geometry_gate(
                    gate_id=gate_id,
                    setting=setting,
                    geometry=geometry,
                    min_availability_ratio=min_availability_ratio_value,
                )
                derived[gate_id] = float(value)
                reasons[gate_id] = str(reason)
                fallback_flags[gate_id] = bool(fallback)
                log_lines.append(f"{gate_id}: {reason}")
                if gate_id == "min_availability_ratio":
                    min_availability_ratio_value = float(value)
                continue
            if gate_id in self._dynamic_distribution_gates:
                value, reason, fallback = self._calibrate_distribution_gate(
                    gate_id=gate_id,
                    setting=setting,
                    empirical_stats=empirical_stats,
                )
                derived[gate_id] = float(value)
                reasons[gate_id] = str(reason)
                fallback_flags[gate_id] = bool(fallback)
                log_lines.append(f"{gate_id}: {reason}")
                continue
            if gate_id in self._dynamic_control_limit_gates:
                value, reason, fallback = self._calibrate_control_limit_gate(
                    gate_id=gate_id,
                    setting=setting,
                    empirical_stats=empirical_stats,
                )
                derived[gate_id] = float(value)
                reasons[gate_id] = str(reason)
                fallback_flags[gate_id] = bool(fallback)
                log_lines.append(f"{gate_id}: {reason}")
                continue
            self._record_static(
                gate_id=gate_id,
                setting=setting,
                reason="UNRECOGNIZED_GATE_LEFT_STATIC",
                derived=derived,
                reasons=reasons,
                fallback_flags=fallback_flags,
                log_lines=log_lines,
            )
        return CalibratedGates(
            derived_values=derived,
            calibration_reasons=reasons,
            fallback_flags=fallback_flags,
            calibration_log=log_lines,
        )

    def _record_static(
        self,
        *,
        gate_id: str,
        setting: _GateSetting,
        reason: str,
        derived: dict[str, float],
        reasons: dict[str, str],
        fallback_flags: dict[str, bool],
        log_lines: list[str],
    ) -> None:
        derived[gate_id] = float(setting.value)
        fallback_flags[gate_id] = False
        if setting.force_static:
            reasons[gate_id] = f"{reason}: force_static=True respected"
        else:
            reasons[gate_id] = reason
        log_lines.append(f"{gate_id}: {reasons[gate_id]}")

    def _coerce_gate_setting(self, *, gate_id: str, raw: Any) -> _GateSetting:
        if isinstance(raw, dict):
            if "value" not in raw:
                raise ValueError(f"gate override missing value field: {gate_id}")
            value = float(raw.get("value"))
            force_static = bool(raw.get("force_static", False))
        else:
            value = float(raw)
            force_static = False
        if not np.isfinite(value):
            raise ValueError(f"gate value must be finite: {gate_id}")
        return _GateSetting(value=float(value), force_static=bool(force_static))

    def _calibrate_geometry_gate(
        self,
        *,
        gate_id: str,
        setting: _GateSetting,
        geometry: RunGeometry,
        min_availability_ratio: float | None,
    ) -> tuple[float, str, bool]:
        baseline = float(setting.value)
        if setting.force_static:
            return baseline, "force_static=True respected", False
        common_sessions = int(max(0, geometry.common_sessions))
        if common_sessions <= 0:
            return baseline, "fallback_static: common_sessions<=0", True
        wf_test_share = float(max(0.0, float(geometry.wf_test_sessions) / float(common_sessions)))
        if bool(geometry.disable_cpcv_splits):
            cpcv_test_share = 0.0
        else:
            slices = int(max(1, geometry.cpcv_slices))
            k_test = int(max(0, min(geometry.cpcv_k_test, slices)))
            cpcv_test_share = float(k_test / slices)
        if gate_id == "min_availability_ratio":
            raw = max(wf_test_share, cpcv_test_share) + (1.0 / float(common_sessions))
            bounded = float(np.clip(raw, 0.05, 0.98))
            calibrated = float(max(baseline, bounded))
            reason = (
                "dynamic_geometry: max(wf_test/common, cpcv_k_test/cpcv_slices)+1/common "
                f"= {raw:.6f}; bounded={bounded:.6f}; static={baseline:.6f}; final={calibrated:.6f}"
            )
            return calibrated, reason, False
        if gate_id == "required_comparison_support":
            raw = max(wf_test_share, cpcv_test_share) + (2.0 / float(common_sessions))
            bounded = float(np.clip(raw, 0.10, 0.99))
            if min_availability_ratio is not None:
                bounded = float(max(bounded, float(min_availability_ratio)))
            calibrated = float(max(baseline, bounded))
            reason = (
                "dynamic_geometry: max(wf_test/common, cpcv_k_test/cpcv_slices)+2/common "
                f"= {raw:.6f}; bounded={bounded:.6f}; static={baseline:.6f}; final={calibrated:.6f}"
            )
            return calibrated, reason, False
        return baseline, "fallback_static: unsupported_geometry_gate", True

    def _calibrate_distribution_gate(
        self,
        *,
        gate_id: str,
        setting: _GateSetting,
        empirical_stats: EmpiricalGateStats | None,
    ) -> tuple[float, str, bool]:
        baseline = float(setting.value)
        if setting.force_static:
            return baseline, "force_static=True respected", False
        if empirical_stats is None:
            return baseline, "fallback_static: no empirical stats", True
        if gate_id == "robustness_reject_threshold":
            arr = _finite_array(empirical_stats.robustness_scores)
            if arr.size < self._min_distribution_samples:
                return baseline, f"fallback_static: robustness sample<{self._min_distribution_samples}", True
            q30 = float(np.quantile(arr, 0.30))
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med)))
            robust_tail = float(med - _MAD_SCALE * mad)
            raw = float(max(q30, robust_tail))
            bounded = float(np.clip(raw, 0.0, 0.99))
            calibrated = float(max(baseline, bounded))
            reason = (
                "dynamic_distribution: max(q30, median-1.4826*MAD) "
                f"= {raw:.6f}; bounded={bounded:.6f}; static={baseline:.6f}; final={calibrated:.6f}; n={int(arr.size)}"
            )
            return calibrated, reason, False
        if gate_id == "execution_fragile_threshold":
            exec_scores = _finite_array(empirical_stats.execution_robustness_scores)
            if exec_scores.size <= 0:
                fill_failure = _finite_array(empirical_stats.fill_failure_rates)
                if fill_failure.size > 0:
                    exec_scores = np.clip(1.0 - fill_failure, 0.0, 1.0)
            if exec_scores.size < self._min_distribution_samples:
                return baseline, f"fallback_static: execution sample<{self._min_distribution_samples}", True
            q25 = float(np.quantile(exec_scores, 0.25))
            med = float(np.median(exec_scores))
            mad = float(np.median(np.abs(exec_scores - med)))
            robust_tail = float(med - 0.8 * _MAD_SCALE * mad)
            raw = float(max(q25, robust_tail))
            bounded = float(np.clip(raw, 0.0, 0.99))
            calibrated = float(max(baseline, bounded))
            reason = (
                "dynamic_distribution: max(q25, median-0.8*1.4826*MAD) "
                f"= {raw:.6f}; bounded={bounded:.6f}; static={baseline:.6f}; final={calibrated:.6f}; n={int(exec_scores.size)}"
            )
            return calibrated, reason, False
        return baseline, "fallback_static: unsupported_distribution_gate", True

    def _calibrate_control_limit_gate(
        self,
        *,
        gate_id: str,
        setting: _GateSetting,
        empirical_stats: EmpiricalGateStats | None,
    ) -> tuple[float, str, bool]:
        baseline = float(setting.value)
        if setting.force_static:
            return baseline, "force_static=True respected", False
        if empirical_stats is None:
            return baseline, "fallback_static: no empirical stats", True
        if gate_id != "fill_failure_control_limit":
            return baseline, "fallback_static: unsupported_control_limit_gate", True
        arr = _finite_array(empirical_stats.fill_failure_rates)
        arr = np.clip(arr, 0.0, 1.0)
        warmup = int(max(self._control_limit_warmup, int(empirical_stats.fold_count or 0)))
        if arr.size <= warmup:
            return baseline, f"fallback_static: fill_failure sample<=warmup({warmup})", True
        hist = arr[:-1]
        if hist.size < warmup:
            return baseline, f"fallback_static: history sample<warmup({warmup})", True
        med = float(np.median(hist))
        mad = float(np.median(np.abs(hist - med)))
        sigma = float(_MAD_SCALE * mad)
        raw = float(med + self._control_limit_k * sigma)
        bounded = float(np.clip(raw, 0.05, 0.95))
        calibrated = float(max(baseline, bounded))
        reason = (
            "dynamic_control_limit: median(history)+k*1.4826*MAD "
            f"= {raw:.6f}; bounded={bounded:.6f}; static={baseline:.6f}; final={calibrated:.6f}; warmup={warmup}; n={int(arr.size)}"
        )
        return calibrated, reason, False


def _finite_array(values: np.ndarray | list[float] | None) -> np.ndarray:
    if values is None:
        return np.zeros(0, dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size <= 0:
        return np.zeros(0, dtype=np.float64)
    return arr[np.isfinite(arr)]
