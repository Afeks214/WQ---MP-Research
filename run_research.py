#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union
import warnings

# Deterministic runtime thread caps must be configured before importing numpy/scipy.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import random

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore[assignment]

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise RuntimeError("pyyaml is required. Install with: pip install pyyaml") from exc

try:
    from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
except Exception as exc:  # pragma: no cover
    raise RuntimeError("pydantic>=2 is required. Install with: pip install 'pydantic>=2'") from exc

from weightiz_module1_core import EngineConfig
from weightiz_module2_core import Module2Config
from weightiz_module3_structure import Module3Config
from weightiz_module4_strategy_funnel import Module4Config
from weightiz_module5_harness import (
    CandidateSpec,
    Module5HarnessConfig,
    StressScenario,
    run_weightiz_harness,
)
from weightiz_self_audit import run_full_self_audit
from weightiz_architecture_guard import run_architecture_consistency_check
from weightiz_validation_suite import run_preflight_validation_suite
from weightiz_system_logger import get_logger, log_event


def _require_pandas() -> Any:
    if pd is None:
        raise RuntimeError("pandas is required. Install with: pip install pandas")
    return pd


class DataConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root: str = "./data/minute"
    format: Literal["parquet", "csv"] = "parquet"
    path_by_symbol: dict[str, str] = Field(default_factory=dict)
    timestamp_column: Optional[str] = None
    start: Optional[datetime] = None
    end: Optional[datetime] = None

    @model_validator(mode="after")
    def validate_time_range(self) -> "DataConfigModel":
        if self.start is not None and self.end is not None and self.end < self.start:
            raise ValueError("data.end must be >= data.start")
        return self


class EngineConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["sealed", "research"] = "sealed"
    B: int = 240
    x_min: float = -6.0
    dx: float = 0.05
    rth_open_minute: int = 9 * 60 + 30
    warmup_minutes: int = 15
    flat_time_minute: int = 15 * 60 + 45
    gap_reset_minutes: float = 5.0
    eps_pdf: float = 1e-12
    eps_vol: float = 1e-12
    initial_cash: float = 1_000_000.0
    intraday_leverage_max: float = 6.0
    overnight_leverage: float = 2.0
    overnight_positions_max: int = 1
    daily_loss_limit_abs: float = 50_000.0
    seed: int = 17
    fail_on_nan: bool = True

    tick_size: Optional[list[float]] = None
    tick_size_default: float = 0.01
    tick_size_by_symbol: dict[str, float] = Field(default_factory=dict)


class Module2ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_window_bars: int = 60
    profile_warmup_bars: int = 60
    atr_span: int = 14
    atr_alpha: Optional[float] = None
    atr_floor_mult_tick: float = 4.0
    atr_floor_abs: Union[float, list[float]] = 0.0
    rvol_lookback_sessions: int = 20
    rvol_policy: str = "neutral_one"
    rvol_vol_eps_mult_tick: float = 1.0
    rvol_vol_eps_abs: Union[float, list[float]] = 1e-12
    rvol_clip_min: float = 0.0
    rvol_clip_max: float = 50.0
    volume_cap_window_bars: int = 60
    volume_cap_mad_mult: float = 5.0
    volume_cap_min_mult: float = 0.25
    mu1_clv_shift: float = 0.0
    mu2_clv_shift: float = 0.35
    sigma1_base: float = 0.18
    sigma1_body_coeff: float = 0.22
    sigma1_rvol_coeff: float = 0.06
    sigma1_range_coeff: float = 0.08
    sigma1_min: float = 0.05
    sigma1_max: float = 1.5
    sigma2_ratio_base: float = 1.8
    sigma2_body_coeff: float = 0.6
    sigma2_clv_coeff: float = 0.3
    sigma2_min: float = 0.08
    sigma2_max: float = 3.0
    w1_base: float = 0.62
    w1_body_coeff: float = 0.28
    w1_rvol_coeff: float = 0.04
    w1_clv_coeff: float = 0.12
    w1_min: float = 0.05
    w1_max: float = 0.95
    ret_scale_window_bars: int = 60
    ret_scale_min_periods: int = 10
    ret_scale_min: float = 0.05
    va_threshold: float = 0.7
    poc_eq_atol: float = 0.0
    d_clip: float = 6.0
    break_bias: float = 1.0
    reject_center: float = 2.0
    rvol_trend_cutoff: float = 2.0
    body_trend_cutoff: float = 0.6
    delta_mad_lookback_bars: int = 180
    delta_mad_min_periods: int = 10
    sigma_delta_min: float = 0.05
    delta_gate_threshold: float = 1.0
    normal_concentration_threshold: float = 0.05
    trend_spread_threshold_x: float = 2.5
    trend_delta_confirm_z: float = 1.5
    double_dist_min_sep_x: float = 1.0
    double_dist_valley_frac: float = 0.35
    fail_on_non_finite_output: bool = True


class Module3ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    block_minutes: int = 30
    phase_mask: list[int] = Field(default_factory=lambda: [1, 2])
    use_rth_minutes_only: bool = True
    rth_open_minute: int = 570
    last_minute_inclusive: int = 945
    include_partial_last_block: bool = True
    min_block_valid_bars: int = 12
    min_block_valid_ratio: float = 0.7
    ib_pop_frac: float = 0.01
    context_mode: str = "ffill_last_complete"
    fail_on_non_finite_input: bool = True
    fail_on_non_finite_output: bool = True
    fail_on_bad_indices: bool = True
    fail_on_missing_prev_va: bool = False
    eps: float = 1e-12


class Module4ConfigModel(BaseModel):
    # Schema gate for module4_configs:
    # this strict model is the real validator used by _load_config -> RunConfigModel.
    model_config = ConfigDict(extra="forbid")

    fail_on_non_finite_input: bool = True
    fail_on_non_finite_output: bool = True
    eps: float = 1e-12
    enforce_causal_source_validation: bool = True
    enforce_window_causal_sanity: bool = True
    window_selection_mode: str = "multi_window"
    fixed_window_index: int = 0
    anchor_window_index: int = 0
    max_volatility: float = float("inf")
    max_spread: float = float("inf")
    min_liquidity: float = 0.0
    regime_confidence_min: float = 0.55
    entry_threshold: float = 0.55
    exit_threshold: float = 0.25
    conviction_scale: float = 1.0
    conviction_clip: float = 1.0
    max_abs_weight: float = 1.0
    top_k_intraday: int = 5
    max_asset_cap_frac: float = 0.3
    max_turnover_frac_per_bar: float = 0.35
    overnight_min_conviction: float = 0.65
    allow_cash_overnight: bool = True
    trend_spread_min: float = 0.05
    trend_poc_drift_min_abs: float = 0.35
    neutral_poc_drift_max_abs: float = 0.15
    shape_skew_min_abs: float = 0.35
    double_dist_sep_x: float = 1.0
    double_dist_valley_frac: float = 0.35
    commission_bps: float = 0.4
    spread_tick_mult: float = 1.5
    slippage_bps_low_rvol: float = 3.0
    slippage_bps_mid_rvol: float = 2.0
    slippage_bps_high_rvol: float = 1.5
    stress_slippage_mult: float = 1.0
    hard_kill_on_daily_loss_breach: bool = True
    enable_degraded_bridge_mode: bool = True

    # Additive compatibility fields for Cell-6 nomenclature.
    strategy_type: str = "legacy"
    score_gate: str = ""
    score_gate_rule: str = ""
    deviation_signal: str = ""
    deviation_rule: str = ""
    entry_model: str = ""
    exit_model: str = ""
    origin_level: str = "POC"
    direction: str = "long"
    delta_th: float = 0.55
    dev_th: float = 1.0
    tp_mult: float = 1.0
    atr_stop_mult: float = 1.0

    @model_validator(mode="after")
    def apply_delta_threshold_mapping(self) -> "Module4ConfigModel":
        # Backward-compatible mapping policy:
        # if delta_th was explicitly supplied, mirror it to legacy entry_threshold.
        if "delta_th" in self.model_fields_set:
            self.entry_threshold = float(self.delta_th)
        return self


class HarnessConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 97
    timezone: str = "America/New_York"
    freq: str = "1min"
    min_asset_coverage: float = 0.80
    purge_bars: int = 60
    embargo_bars: int = 30
    wf_train_sessions: int = 60
    wf_test_sessions: int = 20
    wf_step_sessions: int = 20
    cpcv_slices: int = 10
    cpcv_k_test: int = 5
    parallel_backend: str = "process_pool"
    parallel_workers: int = 1
    stress_profile: str = "baseline_mild_severe"
    max_ram_utilization_frac: float = 0.70
    enforce_lookahead_guard: bool = True
    report_dir: str = "./artifacts"
    fail_on_non_finite: bool = True
    daily_return_min_days: int = 60
    benchmark_symbol: str = "SPY"
    export_micro_diagnostics: bool = True
    micro_diag_mode: str = "events_only"
    micro_diag_symbols: list[str] = Field(default_factory=list)
    micro_diag_session_ids: list[int] = Field(default_factory=list)
    micro_diag_trade_window_pre: int = 90
    micro_diag_trade_window_post: int = 180
    micro_diag_export_block_profiles: bool = True
    micro_diag_export_funnel: bool = True
    micro_diag_max_rows: int = 5_000_000
    failure_rate_abort_threshold: float = 0.02
    failure_count_abort_threshold: int = 50
    payload_pickle_threshold_bytes: int = 131_072
    test_fail_task_ids: list[str] = Field(default_factory=list)
    test_fail_ratio: float = 0.0
    cluster_corr_threshold: float = 0.90
    cluster_distance_block_size: int = 256
    cluster_distance_in_memory_max_n: int = 2500
    execution_transaction_cost_per_trade: float = 0.0
    execution_slippage_mult: float = 1.0
    execution_extra_slippage_bps: float = 0.0
    execution_latency_bars: int = 1
    regime_vol_window: int = 60
    regime_slope_window: int = 60
    regime_hurst_window: int = 120
    regime_min_obs_per_mask: int = 20
    horizon_minutes: list[int] = Field(default_factory=lambda: [1, 5, 15, 60])
    robustness_weight_dsr: float = 0.20
    robustness_weight_pbo: float = 0.15
    robustness_weight_spa: float = 0.10
    robustness_weight_regime: float = 0.20
    robustness_weight_execution: float = 0.20
    robustness_weight_horizon: float = 0.15
    robustness_reject_threshold: float = 0.60
    execution_fragile_threshold: float = 0.50

    @field_validator("horizon_minutes", mode="before")
    @classmethod
    def validate_horizon_minutes_input(cls, value: Any) -> Any:
        if not isinstance(value, list):
            raise ValueError("harness.horizon_minutes must be a list of positive integers")
        for entry in value:
            if isinstance(entry, bool) or not isinstance(entry, int):
                raise ValueError("harness.horizon_minutes entries must be positive integers")
        return value

    @model_validator(mode="after")
    def validate_institutional_controls(self) -> "HarnessConfigModel":
        if not (0.0 <= float(self.cluster_corr_threshold) <= 1.0):
            raise ValueError("harness.cluster_corr_threshold must be in [0,1]")
        if int(self.cluster_distance_block_size) < 1:
            raise ValueError("harness.cluster_distance_block_size must be >=1")
        if int(self.cluster_distance_in_memory_max_n) < 1:
            raise ValueError("harness.cluster_distance_in_memory_max_n must be >=1")
        if float(self.execution_transaction_cost_per_trade) < 0.0:
            raise ValueError("harness.execution_transaction_cost_per_trade must be >=0")
        if float(self.execution_slippage_mult) < 0.0:
            raise ValueError("harness.execution_slippage_mult must be >=0")
        if float(self.execution_extra_slippage_bps) < 0.0:
            raise ValueError("harness.execution_extra_slippage_bps must be >=0")
        if int(self.execution_latency_bars) < 0:
            raise ValueError("harness.execution_latency_bars must be >=0")
        if int(self.regime_vol_window) < 2:
            raise ValueError("harness.regime_vol_window must be >=2")
        if int(self.regime_slope_window) < 2:
            raise ValueError("harness.regime_slope_window must be >=2")
        if int(self.regime_hurst_window) < 8:
            raise ValueError("harness.regime_hurst_window must be >=8")
        if int(self.regime_min_obs_per_mask) < 1:
            raise ValueError("harness.regime_min_obs_per_mask must be >=1")
        if len(self.horizon_minutes) == 0:
            raise ValueError("harness.horizon_minutes must be non-empty")
        seen: set[int] = set()
        for raw_horizon in self.horizon_minutes:
            if isinstance(raw_horizon, bool):
                raise ValueError("harness.horizon_minutes entries must be positive integers")
            horizon = int(raw_horizon)
            if horizon <= 0:
                raise ValueError("harness.horizon_minutes entries must be positive integers")
            if horizon in seen:
                raise ValueError("harness.horizon_minutes must not contain duplicates")
            seen.add(horizon)
        weights = {
            "harness.robustness_weight_dsr": float(self.robustness_weight_dsr),
            "harness.robustness_weight_pbo": float(self.robustness_weight_pbo),
            "harness.robustness_weight_spa": float(self.robustness_weight_spa),
            "harness.robustness_weight_regime": float(self.robustness_weight_regime),
            "harness.robustness_weight_execution": float(self.robustness_weight_execution),
            "harness.robustness_weight_horizon": float(self.robustness_weight_horizon),
        }
        for name, value in weights.items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0,1]")
        if abs(sum(weights.values()) - 1.0) > 1e-12:
            raise ValueError("harness robustness weights must sum to 1.0 within tolerance 1e-12")
        if not (0.0 <= float(self.robustness_reject_threshold) <= 1.0):
            raise ValueError("harness.robustness_reject_threshold must be in [0,1]")
        if not (0.0 <= float(self.execution_fragile_threshold) <= 1.0):
            raise ValueError("harness.execution_fragile_threshold must be in [0,1]")
        return self


class SearchConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: Optional[int] = None
    method: Literal["sobol", "lhs", "uniform"] = "sobol"
    elite_pct: float = 0.10
    target_evals: int = 700


class StressScenarioModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    name: str
    missing_burst_prob: float
    missing_burst_min: int
    missing_burst_max: int
    jitter_sigma_bps: float
    slippage_mult: float
    enabled: bool = True


class CandidateSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    m2_idx: int
    m3_idx: int
    m4_idx: int
    enabled_assets: Union[str, list[str], list[bool]] = "all"
    tags: list[str] = Field(default_factory=list)


class CandidatesModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["auto_grid", "manual"] = "auto_grid"
    specs: list[CandidateSpecModel] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_manual_specs(self) -> "CandidatesModel":
        if self.mode == "manual" and len(self.specs) == 0:
            raise ValueError("candidates.specs must be non-empty when mode='manual'")
        return self


class RunConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_name: str = "weightiz_v35"
    symbols: list[str]
    data: DataConfigModel = Field(default_factory=DataConfigModel)
    engine: EngineConfigModel = Field(default_factory=EngineConfigModel)
    module2_configs: list[Module2ConfigModel] = Field(default_factory=lambda: [Module2ConfigModel()])
    module3_configs: list[Module3ConfigModel] = Field(default_factory=lambda: [Module3ConfigModel()])
    module4_configs: list[Module4ConfigModel] = Field(default_factory=lambda: [Module4ConfigModel()])
    harness: HarnessConfigModel = Field(default_factory=HarnessConfigModel)
    search: SearchConfigModel = Field(default_factory=SearchConfigModel)
    zimtra_sweep: Optional[dict[str, Any]] = None
    stress_scenarios: Optional[list[StressScenarioModel]] = None
    candidates: CandidatesModel = Field(default_factory=CandidatesModel)

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "RunConfigModel":
        syms = [s.strip().upper() for s in self.symbols]
        if len(syms) < 2:
            raise ValueError("symbols must contain at least 2 entries")
        if len(set(syms)) != len(syms):
            raise ValueError("symbols must be unique")

        if self.engine.tick_size is not None and len(self.engine.tick_size) != len(syms):
            raise ValueError(
                f"engine.tick_size length mismatch: got {len(self.engine.tick_size)}, expected {len(syms)}"
            )

        n2 = len(self.module2_configs)
        n3 = len(self.module3_configs)
        n4 = len(self.module4_configs)
        if n2 == 0 or n3 == 0 or n4 == 0:
            raise ValueError("module2_configs/module3_configs/module4_configs must be non-empty")

        if self.candidates.mode == "manual":
            for i, spec in enumerate(self.candidates.specs):
                if not (0 <= spec.m2_idx < n2):
                    raise ValueError(f"candidates.specs[{i}].m2_idx out of range [0, {n2 - 1}]")
                if not (0 <= spec.m3_idx < n3):
                    raise ValueError(f"candidates.specs[{i}].m3_idx out of range [0, {n3 - 1}]")
                if not (0 <= spec.m4_idx < n4):
                    raise ValueError(f"candidates.specs[{i}].m4_idx out of range [0, {n4 - 1}]")
        # Backward-compatible aliasing from legacy zimtra section into canonical search.
        if self.search.seed is None and isinstance(self.zimtra_sweep, dict):
            legacy_seed = self.zimtra_sweep.get("seed")
            if legacy_seed is not None:
                self.search.seed = int(legacy_seed)
        if self.search.seed is None:
            self.search.seed = int(self.harness.seed)
        return self


def _resolve_tick_size(cfg: RunConfigModel) -> np.ndarray:
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


def _build_engine_config(cfg: RunConfigModel) -> EngineConfig:
    e = cfg.engine
    tick_size = _resolve_tick_size(cfg)
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


def _build_module2_configs(cfg: RunConfigModel) -> list[Module2Config]:
    out: list[Module2Config] = []
    for m in cfg.module2_configs:
        out.append(Module2Config(**m.model_dump()))
    return out


def _build_module3_configs(cfg: RunConfigModel) -> list[Module3Config]:
    out: list[Module3Config] = []
    for m in cfg.module3_configs:
        d = m.model_dump()
        d["phase_mask"] = tuple(int(x) for x in d["phase_mask"])
        out.append(Module3Config(**d))
    return out


def _build_module4_configs(cfg: RunConfigModel) -> list[Module4Config]:
    out: list[Module4Config] = []
    for m in cfg.module4_configs:
        out.append(Module4Config(**m.model_dump()))
    return out


def _build_harness_config(cfg: RunConfigModel, project_root: Path) -> Module5HarnessConfig:
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


def _resolve_data_paths(cfg: RunConfigModel, project_root: Path) -> list[str]:
    syms = [s.strip().upper() for s in cfg.symbols]
    d = cfg.data

    root = Path(d.root)
    if not root.is_absolute():
        root = (project_root / root).resolve()

    out: list[str] = []
    missing: list[str] = []

    for s in syms:
        mapped = d.path_by_symbol.get(s, d.path_by_symbol.get(s.lower(), d.path_by_symbol.get(s.upper())))
        if mapped is None:
            p = root / f"{s}.{d.format}"
        else:
            p0 = Path(mapped)
            p = p0 if p0.is_absolute() else (root / p0)
        p = p.resolve()
        if not p.exists():
            missing.append(f"{s}: {p}")
        else:
            out.append(str(p))

    if missing:
        raise RuntimeError("Missing data files:\n" + "\n".join(missing))
    return out


def _find_col(df: Any, candidates: tuple[str, ...], name: str) -> str:
    cols = {str(c).strip().lower(): str(c) for c in df.columns}
    for c in candidates:
        if c in cols:
            return cols[c]
    raise RuntimeError(f"Missing required column '{name}'")


def in_memory_date_filter_loader(data_cfg: DataConfigModel) -> Callable[[str, str], Any]:
    pdx = _require_pandas()

    start_utc = pdx.to_datetime(data_cfg.start, utc=True) if data_cfg.start is not None else None
    end_utc = pdx.to_datetime(data_cfg.end, utc=True) if data_cfg.end is not None else None

    def _load(path: str, tz_name: str) -> Any:
        p = Path(path)
        if not p.exists():
            raise RuntimeError(f"Data path does not exist: {path}")

        suffix = p.suffix.lower()
        if suffix == ".parquet":
            df = pdx.read_parquet(p)
        else:
            df = pdx.read_csv(p)

        if data_cfg.timestamp_column is not None:
            ts_col = _find_col(df, (data_cfg.timestamp_column.strip().lower(),), "timestamp")
        else:
            ts_col = _find_col(df, ("timestamp", "ts", "datetime", "date", "time"), "timestamp")

        o_col = _find_col(df, ("open", "o"), "open")
        h_col = _find_col(df, ("high", "h"), "high")
        l_col = _find_col(df, ("low", "l"), "low")
        c_col = _find_col(df, ("close", "c"), "close")
        v_col = _find_col(df, ("volume", "vol", "v"), "volume")

        ts = pdx.to_datetime(df[ts_col], utc=True, errors="coerce")
        keep = ts.notna().to_numpy(dtype=bool)

        if start_utc is not None:
            keep &= (ts >= start_utc).to_numpy(dtype=bool)
        if end_utc is not None:
            keep &= (ts <= end_utc).to_numpy(dtype=bool)

        if not np.any(keep):
            raise RuntimeError(f"No rows after timestamp/date filtering for {path}")

        out = pdx.DataFrame(
            {
                # Keep canonical UTC timestamps at ingestion boundary.
                "timestamp": ts[keep].dt.floor("min"),
                "open": pdx.to_numeric(df.loc[keep, o_col], errors="coerce"),
                "high": pdx.to_numeric(df.loc[keep, h_col], errors="coerce"),
                "low": pdx.to_numeric(df.loc[keep, l_col], errors="coerce"),
                "close": pdx.to_numeric(df.loc[keep, c_col], errors="coerce"),
                "volume": pdx.to_numeric(df.loc[keep, v_col], errors="coerce"),
            }
        )

        out = out.dropna(subset=["timestamp"]).sort_values("timestamp", kind="mergesort")
        out = out.drop_duplicates(subset=["timestamp"], keep="last")
        out = out.set_index("timestamp")
        return out

    return _load


def _build_stress_scenarios(cfg: RunConfigModel) -> Optional[list[StressScenario]]:
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


def _build_candidates(cfg: RunConfigModel) -> Optional[list[CandidateSpec]]:
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


def _append_run_registry(
    artifacts_root: Path,
    run_id: str,
    run_dir: Path,
    symbols: list[str],
    n_candidates: int,
    pass_count: int,
    resolved_config_sha256: str,
) -> None:
    artifacts_root.mkdir(parents=True, exist_ok=True)

    entry = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "path": str(run_dir.resolve()),
        "symbols": symbols,
        "n_candidates": int(n_candidates),
        "pass_count": int(pass_count),
        "resolved_config_sha256": str(resolved_config_sha256),
    }

    index_path = artifacts_root / "run_index.jsonl"
    with index_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    latest_path = artifacts_root / ".latest_run"
    latest_path.write_text(str(run_dir.resolve()) + "\n", encoding="utf-8")


def _resolved_config_sha256(cfg: RunConfigModel) -> str:
    payload = cfg.model_dump(mode="json")
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _ensure_dashboard_handoff(artifacts_root: Path, run_dir: Path) -> Path:
    module5_root = artifacts_root / "module5_harness"
    module5_root.mkdir(parents=True, exist_ok=True)
    target = module5_root / run_dir.name
    if target.resolve() == run_dir.resolve():
        return target
    if target.exists() or target.is_symlink():
        return target
    try:
        target.symlink_to(run_dir.resolve(), target_is_directory=True)
    except Exception:
        # Fallback: create directory marker with absolute pointer.
        target.mkdir(parents=True, exist_ok=True)
        (target / ".run_path").write_text(str(run_dir.resolve()) + "\n", encoding="utf-8")
    return target


def _load_config(path: Path) -> RunConfigModel:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise RuntimeError("YAML config root must be an object/mapping")
    return RunConfigModel.model_validate(raw)


def _enforce_canonical_runtime_path(cfg: RunConfigModel) -> None:
    if isinstance(cfg.zimtra_sweep, dict):
        enabled = bool(cfg.zimtra_sweep.get("enabled", False))
        if enabled:
            raise RuntimeError("PARALLEL_ENGINE_FORBIDDEN: use canonical Module5 pipeline")


def _map_legacy_zimtra_aliases(cfg: RunConfigModel) -> RunConfigModel:
    if not isinstance(cfg.zimtra_sweep, dict):
        return cfg
    legacy_workers = cfg.zimtra_sweep.get("workers")
    if legacy_workers is not None:
        cfg.harness.parallel_workers = int(legacy_workers)
    return cfg


def _configure_deterministic_runtime(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    np.random.seed(int(seed))
    random.seed(int(seed))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger = get_logger("run_research")
    parser = argparse.ArgumentParser(description="Weightiz V3.5 research runner")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    config_path = Path(args.config).expanduser().resolve()
    cfg = _load_config(config_path)
    cfg = _map_legacy_zimtra_aliases(cfg)
    _enforce_canonical_runtime_path(cfg)
    if cfg.search.seed is None:
        raise RuntimeError("DETERMINISTIC_SEED_REQUIRED")
    _configure_deterministic_runtime(int(cfg.search.seed))
    self_audit_report = run_full_self_audit(
        cfg=cfg,
        project_root=project_root,
    )
    run_architecture_consistency_check()
    run_preflight_validation_suite(
        cfg,
        context={
            "parallel_runtime_enabled": False,
            "config_hash": _resolved_config_sha256(cfg),
            "report_dir": str(Path(cfg.harness.report_dir).resolve()),
        },
    )
    resolved_sha = _resolved_config_sha256(cfg)

    symbols = [s.strip().upper() for s in cfg.symbols]
    data_paths = _resolve_data_paths(cfg, project_root)

    engine_cfg = _build_engine_config(cfg)
    m2_cfgs = _build_module2_configs(cfg)
    m3_cfgs = _build_module3_configs(cfg)
    m4_cfgs = _build_module4_configs(cfg)
    harness_cfg = _build_harness_config(cfg, project_root)

    data_loader = in_memory_date_filter_loader(cfg.data)
    stress_scenarios = _build_stress_scenarios(cfg)
    candidate_specs = _build_candidates(cfg)

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always", RuntimeWarning)
        out = run_weightiz_harness(
            data_paths=data_paths,
            symbols=symbols,
            engine_cfg=engine_cfg,
            m2_configs=m2_cfgs,
            m3_configs=m3_cfgs,
            m4_configs=m4_cfgs,
            harness_cfg=harness_cfg,
            candidate_specs=candidate_specs,
            data_loader_func=data_loader,
            stress_scenarios=stress_scenarios,
            self_audit_report=self_audit_report,
        )
    runtime_warnings = [w for w in captured_warnings if issubclass(w.category, RuntimeWarning)]
    runtime_warning_count = int(len(runtime_warnings))

    run_manifest_path = Path(out.artifact_paths["run_manifest"]).resolve()
    run_dir = run_manifest_path.parent
    run_id = str(out.run_manifest.get("run_id", run_dir.name))
    run_status_path = Path(out.artifact_paths.get("run_status", run_dir / "run_status.json")).resolve()

    # Runtime warning telemetry (captured in this process) is attached to both
    # run_manifest.json and run_status.json without changing strict YAML schemas.
    out.run_manifest["runtime_warning_count"] = int(runtime_warning_count)
    if run_manifest_path.exists():
        try:
            manifest_doc = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            if isinstance(manifest_doc, dict):
                manifest_doc["runtime_warning_count"] = int(runtime_warning_count)
                run_manifest_path.write_text(
                    json.dumps(manifest_doc, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        except Exception:
            pass
    if run_status_path.exists():
        try:
            status_doc = json.loads(run_status_path.read_text(encoding="utf-8"))
            if isinstance(status_doc, dict):
                status_doc["runtime_warning_count"] = int(runtime_warning_count)
                run_status_path.write_text(
                    json.dumps(status_doc, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        except Exception:
            pass

    leaderboard = out.stats_verdict.get("leaderboard", [])
    pass_count = int(sum(1 for row in leaderboard if bool(row.get("pass", False))))

    report_root = Path(harness_cfg.report_dir).resolve()
    artifacts_root = report_root.parent if report_root.name == "module5_harness" else report_root
    _ensure_dashboard_handoff(artifacts_root, run_dir)
    _append_run_registry(
        artifacts_root=artifacts_root,
        run_id=run_id,
        run_dir=run_dir,
        symbols=symbols,
        n_candidates=int(out.run_manifest.get("n_candidates", len(out.candidate_results))),
        pass_count=pass_count,
        resolved_config_sha256=resolved_sha,
    )

    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "n_candidate_results": int(out.run_manifest.get("n_candidates", len(out.candidate_results))),
        "pass_count": pass_count,
        "aborted": bool(out.run_manifest.get("aborted", False)),
        "abort_reason": str(out.run_manifest.get("abort_reason", "")),
        "failure_count": int(out.run_manifest.get("failure_count", 0)),
        "failure_rate": float(out.run_manifest.get("failure_rate", 0.0)),
        "parallel_backend": str(out.run_manifest.get("parallel_backend", harness_cfg.parallel_backend)),
        "parallel_workers_effective": int(out.run_manifest.get("parallel_workers_effective", 1)),
        "payload_safe": bool(out.run_manifest.get("payload_safe", True)),
        "large_payload_passing_avoided": bool(out.run_manifest.get("large_payload_passing_avoided", True)),
        "resolved_config_sha256": resolved_sha,
        "run_index": str((artifacts_root / "run_index.jsonl").resolve()),
        "latest_run": str((artifacts_root / ".latest_run").resolve()),
        "runtime_warning_count": int(runtime_warning_count),
    }

    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log_event(logger, "INFO", "run_complete", event_type="run_complete")


if __name__ == "__main__":
    main()
