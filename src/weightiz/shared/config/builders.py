from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from weightiz.shared.config.models import RunConfigModel
from weightiz.shared.gate_calibrator import CalibratedGates, RunGeometry
from weightiz.module1.core import EngineConfig
from weightiz.module2.core import Module2Config
from weightiz.module3.bridge import Module3Config
from weightiz.module4.strategy_funnel import Module4Config
from weightiz.module5.orchestrator import CandidateSpec, Module5HarnessConfig, StressScenario


def resolve_tick_size(cfg: RunConfigModel) -> np.ndarray:
    syms = [s.strip().upper() for s in cfg.symbols]
    if cfg.engine.tick_size is not None:
        arr = np.asarray(cfg.engine.tick_size, dtype=np.float64)
    else:
        arr = np.full(len(syms), float(cfg.engine.tick_size_default), dtype=np.float64)
        for i, s in enumerate(syms):
            if s in cfg.engine.tick_size_by_symbol:
                arr[i] = float(cfg.engine.tick_size_by_symbol[s])
            elif s.lower() in cfg.engine.tick_size_by_symbol:
                arr[i] = float(cfg.engine.tick_size_by_symbol[s.lower()])
            elif s.upper() in cfg.engine.tick_size_by_symbol:
                arr[i] = float(cfg.engine.tick_size_by_symbol[s.upper()])

    if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
        raise RuntimeError("Resolved tick_size contains non-finite or non-positive values")
    return arr


def build_engine_config(cfg: RunConfigModel) -> EngineConfig:
    e = cfg.engine
    tick_size = resolve_tick_size(cfg)
    return EngineConfig(
        T=1,
        A=len(cfg.symbols),
        mode=e.mode,
        tick_size=tick_size,
        B=e.B,
        x_min=e.x_min,
        dx=e.dx,
        rth_open_minute=e.rth_open_minute,
        warmup_minutes=e.warmup_minutes,
        flat_time_minute=e.flat_time_minute,
        gap_reset_minutes=e.gap_reset_minutes,
        eps_pdf=e.eps_pdf,
        eps_vol=e.eps_vol,
        initial_cash=e.initial_cash,
        intraday_leverage_max=e.intraday_leverage_max,
        overnight_leverage=e.overnight_leverage,
        overnight_positions_max=e.overnight_positions_max,
        daily_loss_limit_abs=e.daily_loss_limit_abs,
        seed=e.seed,
        fail_on_nan=e.fail_on_nan,
    )


def build_module2_configs(cfg: RunConfigModel) -> list[Module2Config]:
    out: list[Module2Config] = []
    for m in cfg.module2_configs:
        out.append(Module2Config(**m.model_dump()))
    return out


def build_module3_configs(cfg: RunConfigModel) -> list[Module3Config]:
    out: list[Module3Config] = []
    for m in cfg.module3_configs:
        d = m.model_dump()
        d["phase_mask"] = tuple(int(x) for x in d["phase_mask"])
        out.append(Module3Config(**d))
    return out


def build_module4_configs(cfg: RunConfigModel) -> list[Module4Config]:
    out: list[Module4Config] = []
    for m in cfg.module4_configs:
        out.append(Module4Config(**m.model_dump()))
    return out


def build_run_geometry(
    harness_cfg: Any,
    *,
    data_sessions: int,
    common_sessions: int,
) -> RunGeometry:
    wf_train = int(max(0, int(getattr(harness_cfg, "wf_train_sessions", 0))))
    wf_test = int(max(0, int(getattr(harness_cfg, "wf_test_sessions", 0))))
    wf_step = int(max(0, int(getattr(harness_cfg, "wf_step_sessions", 0))))
    cpcv_slices = int(max(1, int(getattr(harness_cfg, "cpcv_slices", 1))))
    cpcv_k_test = int(max(0, int(getattr(harness_cfg, "cpcv_k_test", 0))))
    disable_cpcv = bool(getattr(harness_cfg, "disable_cpcv_splits", False))
    resolved_data_sessions = int(max(0, int(data_sessions)))
    resolved_common_sessions = int(max(0, int(common_sessions)))
    return RunGeometry(
        data_sessions=resolved_data_sessions,
        common_sessions=resolved_common_sessions,
        wf_train_sessions=wf_train,
        wf_test_sessions=wf_test,
        wf_step_sessions=wf_step,
        cpcv_slices=cpcv_slices,
        cpcv_k_test=cpcv_k_test,
        disable_cpcv_splits=disable_cpcv,
    )


def build_harness_gate_inputs(harness_cfg: Any) -> dict[str, Any]:
    gate_overrides = getattr(harness_cfg, "gate_overrides", None)
    robust_override = getattr(gate_overrides, "robustness_reject_threshold", None)
    fragile_override = getattr(gate_overrides, "execution_fragile_threshold", None)
    robust_value = float(
        getattr(robust_override, "value", getattr(harness_cfg, "robustness_reject_threshold", 0.0))
    )
    robust_force_static = bool(getattr(robust_override, "force_static", False))
    fragile_value = float(
        getattr(fragile_override, "value", getattr(harness_cfg, "execution_fragile_threshold", 0.0))
    )
    fragile_force_static = bool(getattr(fragile_override, "force_static", False))
    return {
        "robustness_reject_threshold": {
            "value": float(robust_value),
            "force_static": bool(robust_force_static),
        },
        "execution_fragile_threshold": {
            "value": float(fragile_value),
            "force_static": bool(fragile_force_static),
        },
    }


def build_module6_gate_inputs(module6_block: dict[str, Any] | None) -> dict[str, Any]:
    cfg = module6_block if isinstance(module6_block, dict) else {}
    intake = cfg.get("intake", {})
    intake_payload = intake if isinstance(intake, dict) else {}
    gate_overrides = cfg.get("gate_overrides", {})
    gate_override_payload = gate_overrides if isinstance(gate_overrides, dict) else {}
    min_availability_default = intake_payload.get("min_availability_ratio", 0.95)
    required_support_default = intake_payload.get("required_comparison_support", 0.85)
    min_availability_override = gate_override_payload.get("min_availability_ratio")
    required_support_override = gate_override_payload.get("required_comparison_support")
    if isinstance(min_availability_override, dict):
        min_availability_value = float(min_availability_override.get("value", min_availability_default))
        min_availability_force_static = bool(min_availability_override.get("force_static", False))
    elif min_availability_override is None:
        min_availability_value = float(min_availability_default)
        min_availability_force_static = False
    else:
        min_availability_value = float(min_availability_override)
        min_availability_force_static = False
    if isinstance(required_support_override, dict):
        required_support_value = float(required_support_override.get("value", required_support_default))
        required_support_force_static = bool(required_support_override.get("force_static", False))
    elif required_support_override is None:
        required_support_value = float(required_support_default)
        required_support_force_static = False
    else:
        required_support_value = float(required_support_override)
        required_support_force_static = False
    return {
        "min_availability_ratio": {
            "value": float(min_availability_value),
            "force_static": bool(min_availability_force_static),
        },
        "required_comparison_support": {
            "value": float(required_support_value),
            "force_static": bool(required_support_force_static),
        },
    }


def apply_calibrated_module6_gate_values(
    module6_block: dict[str, Any] | None,
    calibration: CalibratedGates,
) -> dict[str, Any]:
    cfg = dict(module6_block) if isinstance(module6_block, dict) else {}
    intake = cfg.get("intake", {})
    intake_payload = dict(intake) if isinstance(intake, dict) else {}
    if "min_availability_ratio" in calibration.derived_values:
        intake_payload["min_availability_ratio"] = float(calibration.derived_values["min_availability_ratio"])
    if "required_comparison_support" in calibration.derived_values:
        intake_payload["required_comparison_support"] = float(calibration.derived_values["required_comparison_support"])
    cfg["intake"] = intake_payload
    return cfg


def build_harness_config(cfg: RunConfigModel, project_root: Path) -> Module5HarnessConfig:
    h = cfg.harness
    report_dir = Path(h.report_dir)
    if not report_dir.is_absolute():
        report_dir = (project_root / report_dir).resolve()
    harness_gate_inputs = build_harness_gate_inputs(h)
    calibrated_robustness_reject_threshold = float(
        harness_gate_inputs["robustness_reject_threshold"]["value"]
    )
    calibrated_execution_fragile_threshold = float(
        harness_gate_inputs["execution_fragile_threshold"]["value"]
    )

    return Module5HarnessConfig(
        seed=h.seed,
        research_mode=h.research_mode,
        timezone=h.timezone,
        freq=h.freq,
        min_asset_coverage=h.min_asset_coverage,
        purge_bars=h.purge_bars,
        embargo_bars=h.embargo_bars,
        wf_train_sessions=h.wf_train_sessions,
        wf_test_sessions=h.wf_test_sessions,
        wf_step_sessions=h.wf_step_sessions,
        cpcv_slices=h.cpcv_slices,
        cpcv_k_test=h.cpcv_k_test,
        disable_cpcv_splits=h.disable_cpcv_splits,
        parallel_backend=h.parallel_backend,
        parallel_workers=h.parallel_workers,
        stress_profile=h.stress_profile,
        max_ram_utilization_frac=h.max_ram_utilization_frac,
        enforce_lookahead_guard=h.enforce_lookahead_guard,
        report_dir=str(report_dir),
        fail_on_non_finite=h.fail_on_non_finite,
        daily_return_min_days=h.daily_return_min_days,
        benchmark_symbol=h.benchmark_symbol,
        export_micro_diagnostics=h.export_micro_diagnostics,
        micro_diag_mode=h.micro_diag_mode,
        micro_diag_symbols=tuple(str(x) for x in h.micro_diag_symbols),
        micro_diag_session_ids=tuple(int(x) for x in h.micro_diag_session_ids),
        micro_diag_trade_window_pre=h.micro_diag_trade_window_pre,
        micro_diag_trade_window_post=h.micro_diag_trade_window_post,
        micro_diag_export_block_profiles=h.micro_diag_export_block_profiles,
        micro_diag_export_funnel=h.micro_diag_export_funnel,
        micro_diag_max_rows=h.micro_diag_max_rows,
        failure_rate_abort_threshold=h.failure_rate_abort_threshold,
        failure_count_abort_threshold=h.failure_count_abort_threshold,
        payload_pickle_threshold_bytes=h.payload_pickle_threshold_bytes,
        process_pool_candidate_chunk_size=h.process_pool_candidate_chunk_size,
        group_bound_execution_enabled=h.group_bound_execution_enabled,
        group_dispatch_policy=h.group_dispatch_policy,
        group_max_in_flight_factor=h.group_max_in_flight_factor,
        group_target_wall_time_sec=h.group_target_wall_time_sec,
        group_max_result_payload_bytes=h.group_max_result_payload_bytes,
        group_max_memory_bytes=h.group_max_memory_bytes,
        group_min_candidates_per_chunk=h.group_min_candidates_per_chunk,
        group_max_candidates_per_chunk_hard=h.group_max_candidates_per_chunk_hard,
        startup_default_candidate_loop_sec=h.startup_default_candidate_loop_sec,
        startup_default_result_payload_bytes=h.startup_default_result_payload_bytes,
        startup_default_candidate_incremental_bytes=h.startup_default_candidate_incremental_bytes,
        startup_default_module3_bytes=h.startup_default_module3_bytes,
        scratch_mode=h.scratch_mode,
        strict_candidate_state_validation=h.strict_candidate_state_validation,
        risk_breach_state_dump_enabled=h.risk_breach_state_dump_enabled,
        debug_full_state_payloads=h.debug_full_state_payloads,
        module3_output_mode=h.module3_output_mode,
        base_sharing_mode=h.base_sharing_mode,
        cow_private_ratio_threshold=h.cow_private_ratio_threshold,
        cow_probe_workers=h.cow_probe_workers,
        safety_margin_frac=h.safety_margin_frac,
        safety_margin_min_bytes=h.safety_margin_min_bytes,
        max_queue_bytes_frac=h.max_queue_bytes_frac,
        max_result_buffer_bytes_frac=h.max_result_buffer_bytes_frac,
        throughput_minimal_observability=h.throughput_minimal_observability,
        health_check_interval=h.health_check_interval,
        progress_interval_seconds=h.progress_interval_seconds,
        test_fail_task_ids=tuple(str(x) for x in h.test_fail_task_ids),
        test_fail_ratio=h.test_fail_ratio,
        cluster_corr_threshold=h.cluster_corr_threshold,
        cluster_distance_block_size=h.cluster_distance_block_size,
        cluster_distance_in_memory_max_n=h.cluster_distance_in_memory_max_n,
        execution_transaction_cost_per_trade=h.execution_transaction_cost_per_trade,
        execution_cost_model=h.execution_cost_model,
        execution_slippage_mult=h.execution_slippage_mult,
        execution_extra_slippage_bps=h.execution_extra_slippage_bps,
        execution_max_volume_participation=h.execution_max_volume_participation,
        execution_short_borrow_apr=h.execution_short_borrow_apr,
        execution_short_locate_fee_per_share=h.execution_short_locate_fee_per_share,
        execution_debit_apr=h.execution_debit_apr,
        execution_open_bucket_minutes=h.execution_open_bucket_minutes,
        execution_close_bucket_minutes=h.execution_close_bucket_minutes,
        execution_open_slippage_mult=h.execution_open_slippage_mult,
        execution_mid_slippage_mult=h.execution_mid_slippage_mult,
        execution_close_slippage_mult=h.execution_close_slippage_mult,
        execution_participation_slippage_coeff=h.execution_participation_slippage_coeff,
        execution_dynamic_slippage_bps_cap=h.execution_dynamic_slippage_bps_cap,
        execution_latency_bars=h.execution_latency_bars,
        regime_vol_window=h.regime_vol_window,
        regime_slope_window=h.regime_slope_window,
        regime_hurst_window=h.regime_hurst_window,
        regime_min_obs_per_mask=h.regime_min_obs_per_mask,
        horizon_minutes=tuple(int(x) for x in h.horizon_minutes),
        robustness_weight_dsr=h.robustness_weight_dsr,
        robustness_weight_pbo=h.robustness_weight_pbo,
        robustness_weight_spa=h.robustness_weight_spa,
        robustness_weight_regime=h.robustness_weight_regime,
        robustness_weight_execution=h.robustness_weight_execution,
        robustness_weight_horizon=h.robustness_weight_horizon,
        robustness_reject_threshold=calibrated_robustness_reject_threshold,
        execution_fragile_threshold=calibrated_execution_fragile_threshold,
    )


def build_stress_scenarios(cfg: RunConfigModel) -> Optional[list[StressScenario]]:
    if cfg.stress_scenarios is None:
        return None

    out: list[StressScenario] = []
    for s in cfg.stress_scenarios:
        out.append(
            StressScenario(
                scenario_id=s.scenario_id,
                name=s.name,
                scenario_group=s.scenario_group,
                missing_burst_prob=s.missing_burst_prob,
                missing_burst_min=s.missing_burst_min,
                missing_burst_max=s.missing_burst_max,
                jitter_sigma_bps=s.jitter_sigma_bps,
                slippage_mult=s.slippage_mult,
                signal_lag_bars=s.signal_lag_bars,
                entry_threshold_shift=s.entry_threshold_shift,
                exit_threshold_shift=s.exit_threshold_shift,
                target_scale_mult=s.target_scale_mult,
                enabled=s.enabled,
            )
        )
    return out


def build_candidates(cfg: RunConfigModel) -> Optional[list[CandidateSpec]]:
    if cfg.candidates.mode == "auto_grid":
        return None

    syms = [s.strip().upper() for s in cfg.symbols]
    A = len(syms)

    out: list[CandidateSpec] = []
    for i, c in enumerate(cfg.candidates.specs):
        raw = c.enabled_assets
        if isinstance(raw, str):
            if raw.lower() != "all":
                raise RuntimeError(f"candidates.specs[{i}].enabled_assets string must be 'all'")
            mask = np.ones(A, dtype=bool)
        else:
            if len(raw) == A and all(isinstance(x, bool) for x in raw):
                mask = np.asarray(raw, dtype=bool)
            else:
                selected = {str(x).strip().upper() for x in raw}
                unknown = sorted(selected - set(syms))
                if unknown:
                    raise RuntimeError(
                        f"candidates.specs[{i}].enabled_assets has unknown symbols: {unknown}"
                    )
                mask = np.asarray([s in selected for s in syms], dtype=bool)

        out.append(
            CandidateSpec(
                candidate_id=c.candidate_id,
                m2_idx=c.m2_idx,
                m3_idx=c.m3_idx,
                m4_idx=c.m4_idx,
                enabled_assets_mask=mask,
                tags=tuple(c.tags),
            )
        )

    return out
