from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from app.config_models import RunConfigModel
from weightiz_module1_core import EngineConfig
from weightiz_module2_core import Module2Config
from weightiz_module3_structure import Module3Config
from weightiz_module4_strategy_funnel import Module4Config
from weightiz_module5_harness import CandidateSpec, Module5HarnessConfig, StressScenario


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


def build_harness_config(cfg: RunConfigModel, project_root: Path) -> Module5HarnessConfig:
    h = cfg.harness
    report_dir = Path(h.report_dir)
    if not report_dir.is_absolute():
        report_dir = (project_root / report_dir).resolve()

    return Module5HarnessConfig(
        seed=h.seed,
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
        test_fail_task_ids=tuple(str(x) for x in h.test_fail_task_ids),
        test_fail_ratio=h.test_fail_ratio,
        cluster_corr_threshold=h.cluster_corr_threshold,
        cluster_distance_block_size=h.cluster_distance_block_size,
        cluster_distance_in_memory_max_n=h.cluster_distance_in_memory_max_n,
        execution_transaction_cost_per_trade=h.execution_transaction_cost_per_trade,
        execution_slippage_mult=h.execution_slippage_mult,
        execution_extra_slippage_bps=h.execution_extra_slippage_bps,
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
        robustness_reject_threshold=h.robustness_reject_threshold,
        execution_fragile_threshold=h.execution_fragile_threshold,
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
                missing_burst_prob=s.missing_burst_prob,
                missing_burst_min=s.missing_burst_min,
                missing_burst_max=s.missing_burst_max,
                jitter_sigma_bps=s.jitter_sigma_bps,
                slippage_mult=s.slippage_mult,
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
