#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

exe = str(Path(sys.executable).resolve())

valid_markers = (".venv", "envs", "conda", "miniforge")

if not any(marker in exe for marker in valid_markers):
    raise RuntimeError(
        "\nFATAL ENVIRONMENT LOCK\n"
        f"Detected interpreter: {exe}\n"
        "run_research.py must run inside the project virtual environment\n"
        "Example:\n"
        "./.venv/bin/python run_research.py --config configs/..."
    )

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import platform
import socket
import subprocess
import time
from typing import Any, Callable, Literal, Optional, Union
import warnings

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore[assignment]

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise RuntimeError("pyyaml is required. Install with: pip install pyyaml") from exc

try:
    from pydantic import BaseModel, ConfigDict, Field, model_validator
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

    entry_threshold: float = 0.55
    exit_threshold: float = 0.25
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
    fail_on_non_finite_input: bool = True
    fail_on_non_finite_output: bool = True
    eps: float = 1e-12

    # Additive compatibility fields for Cell-6 nomenclature.
    strategy_type: str = "legacy"
    score_gate: float = 0.0
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


class ZimtraCostConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tick_size: float = 0.01
    slippage_ticks: int = 1
    missing_bar_slippage_ticks: int = 5
    commission_per_share: float = 0.0015
    reg_fee_per_share_sell: float = 0.000119
    locate_fee_per_share_short_entry: float = 0.005


class ZimtraRiskConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    per_asset_notional_cap_mult: float = 2.5
    max_position_buying_power_frac: float = 1.0
    overnight_gross_cap_mult: float = 1.6
    daily_max_loss_frac: float = 0.10
    account_disable_equity: float = 0.0
    account_disable_buffer_scale: float = 1.0
    delever_check_minute_et: int = 15 * 60 + 45
    delever_exec_minute_et: int = 15 * 60 + 46
    kill_switch_lockout_same_day: bool = True


class ZimtraSwingGridConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_window_minutes: list[int]
    profile_memory_sessions: list[int]
    deltaeff_threshold: list[float]
    distance_to_poc_atr: list[float]
    acceptance_threshold: list[float]
    rvol_filter: list[float]
    holding_period_days: list[int]
    lev_target: float = 1.5

    @model_validator(mode="after")
    def validate_non_empty(self) -> "ZimtraSwingGridConfigModel":
        fields = [
            "profile_window_minutes",
            "profile_memory_sessions",
            "deltaeff_threshold",
            "distance_to_poc_atr",
            "acceptance_threshold",
            "rvol_filter",
            "holding_period_days",
        ]
        for f in fields:
            vals = getattr(self, f)
            if len(vals) == 0:
                raise ValueError(f"zimtra_sweep.swing_grid.{f} must be non-empty")
        return self


class ZimtraSamplingConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["grid", "sobol"] = "grid"
    n_samples: int = 65536
    seed: int = 42
    lev_target: float = 1.5
    param_ranges: dict[str, list[float]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_sampling(self) -> "ZimtraSamplingConfigModel":
        if self.method == "sobol":
            if self.n_samples <= 0:
                raise ValueError("zimtra_sweep.sampling.n_samples must be > 0")
            if (self.n_samples & (self.n_samples - 1)) != 0:
                raise ValueError(
                    "zimtra_sweep.sampling.n_samples must be a power-of-two for Sobol random_base2"
                )
            for k, v in self.param_ranges.items():
                if len(v) != 2:
                    raise ValueError(
                        f"zimtra_sweep.sampling.param_ranges.{k} must have exactly 2 values [low, high]"
                    )
                lo = float(v[0])
                hi = float(v[1])
                if hi < lo:
                    raise ValueError(
                        f"zimtra_sweep.sampling.param_ranges.{k} invalid bounds: high < low"
                    )
        return self


class ZimtraAdaptiveConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    new_samples: int = 10000
    noise: float = 0.15

    @model_validator(mode="after")
    def validate_adaptive(self) -> "ZimtraAdaptiveConfigModel":
        if int(self.new_samples) < 0:
            raise ValueError("zimtra_sweep.adaptive.new_samples must be >= 0")
        if float(self.noise) < 0.0:
            raise ValueError("zimtra_sweep.adaptive.noise must be >= 0")
        return self


class ZimtraCVConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wf_train_months: int = 6
    wf_test_months: int = 3
    wf_splits: int = 10
    purge_trading_days: int = 5
    cpcv_n: int = 10
    cpcv_k: int = 3


class ZimtraStageConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    workers: int = 48
    max_workers_cap: int = 50
    screen_assets: list[str] = Field(default_factory=lambda: ["SPY", "QQQ", "IWM"])
    screen_wf_splits: int = 4
    gate_profit_factor_min: float = 1.05
    gate_max_drawdown_max: float = 0.08
    gate_positive_assets_min: int = 2
    gate_assets_total: int = 3

    @model_validator(mode="after")
    def validate_stage_workers(self) -> "ZimtraStageConfigModel":
        if self.workers <= 0:
            raise ValueError("workers must be > 0")
        if self.workers > self.max_workers_cap:
            raise ValueError(
                f"workers must be <= max_workers_cap ({self.max_workers_cap}), got {self.workers}"
            )
        return self


class ZimtraScenarioConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requested: list[str] = Field(default_factory=lambda: ["baseline", "mild", "severe"])
    allow_baseline_only_without_hooks: bool = True


class ZimtraSweepConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    workers: int = 48
    max_workers_cap: int = 50
    deterministic_required: bool = True
    forbid_randomness: bool = True
    rth_only: bool = True
    union_master_timeline: bool = True
    shared_memory_required: bool = True
    memory_transport: Literal["shared_memory", "memmap"] = "shared_memory"
    stage_a: ZimtraStageConfigModel = Field(default_factory=ZimtraStageConfigModel)
    stage_b: ZimtraStageConfigModel = Field(default_factory=lambda: ZimtraStageConfigModel(enabled=True))
    cv: ZimtraCVConfigModel = Field(default_factory=ZimtraCVConfigModel)
    risk: ZimtraRiskConfigModel = Field(default_factory=ZimtraRiskConfigModel)
    cost: ZimtraCostConfigModel = Field(default_factory=ZimtraCostConfigModel)
    scenarios: ZimtraScenarioConfigModel = Field(default_factory=ZimtraScenarioConfigModel)
    swing_grid: Optional[ZimtraSwingGridConfigModel] = None
    sampling: ZimtraSamplingConfigModel = Field(default_factory=ZimtraSamplingConfigModel)
    adaptive: ZimtraAdaptiveConfigModel = Field(default_factory=ZimtraAdaptiveConfigModel)

    @model_validator(mode="after")
    def validate_workers(self) -> "ZimtraSweepConfigModel":
        if self.workers <= 0:
            raise ValueError("zimtra_sweep.workers must be > 0")
        if self.workers > self.max_workers_cap:
            raise ValueError(
                f"zimtra_sweep.workers must be <= {self.max_workers_cap}, got {self.workers}"
            )
        if not self.rth_only:
            raise ValueError("zimtra_sweep.rth_only must be true (fail-closed)")
        if not self.union_master_timeline:
            raise ValueError("zimtra_sweep.union_master_timeline must be true (fail-closed)")
        if self.memory_transport not in {"shared_memory", "memmap"}:
            raise ValueError("zimtra_sweep.memory_transport must be shared_memory or memmap")
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
    stress_scenarios: Optional[list[StressScenarioModel]] = None
    candidates: CandidatesModel = Field(default_factory=CandidatesModel)
    zimtra_sweep: Optional[ZimtraSweepConfigModel] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_family_d_grid_payload(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw

        # Bridge accepted alias: universe.symbols -> symbols
        if "symbols" not in raw and isinstance(raw.get("universe"), dict):
            u = raw.get("universe", {})
            if isinstance(u, dict) and isinstance(u.get("symbols"), list):
                raw = dict(raw)
                raw["symbols"] = u.get("symbols")
            raw.pop("universe", None)

        # Bridge accepted alias: module4_configs.grid -> expanded list[Module4ConfigModel]
        m4 = raw.get("module4_configs")
        if isinstance(m4, dict) and isinstance(m4.get("grid"), dict):
            grid = m4.get("grid", {})
            if not isinstance(grid, dict):
                return raw
            # Keep strictness by requiring only this exact Family-D grid keyset.
            allowed_keys = [
                "strategy_type",
                "score_gate",
                "dev_th",
                "delta_th",
                "atr_stop_mult",
                "tp_mult",
                "top_k_intraday",
                "max_asset_cap_frac",
                "max_turnover_frac_per_bar",
                "overnight_min_conviction",
                "allow_cash_overnight",
            ]
            extra = sorted(k for k in grid.keys() if k not in allowed_keys)
            if extra:
                raise ValueError(f"module4_configs.grid contains unsupported keys: {extra}")

            # Validate required keys and strict numeric bounds for the requested factors.
            for k in allowed_keys:
                if k not in grid:
                    raise ValueError(f"module4_configs.grid missing required key: {k}")
                if not isinstance(grid[k], list) or len(grid[k]) == 0:
                    raise ValueError(f"module4_configs.grid.{k} must be a non-empty list")

            for v in grid["score_gate"]:
                x = float(v)
                if not (0.0 <= x <= 1.0):
                    raise ValueError("module4_configs.grid.score_gate values must be in [0,1]")
            for v in grid["delta_th"]:
                x = float(v)
                if not (0.0 <= x <= 1.0):
                    raise ValueError("module4_configs.grid.delta_th values must be in [0,1]")
            for key in ("dev_th", "atr_stop_mult", "tp_mult"):
                for v in grid[key]:
                    if float(v) <= 0.0:
                        raise ValueError(f"module4_configs.grid.{key} values must be > 0")

            order = [
                "strategy_type",
                "score_gate",
                "dev_th",
                "delta_th",
                "atr_stop_mult",
                "tp_mult",
                "top_k_intraday",
                "max_asset_cap_frac",
                "max_turnover_frac_per_bar",
                "overnight_min_conviction",
                "allow_cash_overnight",
            ]
            combos = []
            for values in itertools.product(*[grid[k] for k in order]):
                row = dict(zip(order, values))
                # Deterministic scalar coercions and backward-compatible mapping.
                row["strategy_type"] = str(row["strategy_type"])
                row["score_gate"] = float(row["score_gate"])
                row["dev_th"] = float(row["dev_th"])
                row["delta_th"] = float(row["delta_th"])
                row["atr_stop_mult"] = float(row["atr_stop_mult"])
                row["tp_mult"] = float(row["tp_mult"])
                row["top_k_intraday"] = int(row["top_k_intraday"])
                row["max_asset_cap_frac"] = float(row["max_asset_cap_frac"])
                row["max_turnover_frac_per_bar"] = float(row["max_turnover_frac_per_bar"])
                row["overnight_min_conviction"] = float(row["overnight_min_conviction"])
                row["allow_cash_overnight"] = bool(row["allow_cash_overnight"])
                row["entry_threshold"] = float(row["score_gate"])
                combos.append(row)

            raw = dict(raw)
            raw["module4_configs"] = combos

        return raw

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
        keep = ts.notna().to_numpy(dtype=bool).copy()

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


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if np.isfinite(x):
            return float(x)
    except Exception:
        pass
    return float(default)


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _json_load(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _family_mode_enabled(run_name: str) -> bool:
    return str(run_name).strip().lower().startswith("sweep_family_")


def _deterministic_jitter_seconds(run_name: str, seed: int) -> int:
    token = f"{str(run_name)}{int(seed)}".encode("utf-8")
    h = int.from_bytes(hashlib.sha256(token).digest()[:8], byteorder="big", signed=False)
    return int(10 + (h % 21))


def _family_log_append(log_path: Path, message: str) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {message}\n")


def _safe_git_hash(project_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return str(out)
    except Exception:
        return "UNKNOWN"


def _flatten_dict(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for k in sorted(value.keys()):
            key = f"{prefix}.{k}" if prefix else str(k)
            _flatten_dict(key, value[k], out)
    else:
        out[prefix] = value


def _candidate_returns_stats(candidate_dir: Path) -> tuple[float, float]:
    ret_path = candidate_dir / "candidate_returns.parquet"
    if not ret_path.exists():
        return 0.0, 0.0
    pdx = _require_pandas()
    try:
        df = pdx.read_parquet(ret_path)
    except Exception:
        return 0.0, 0.0
    if "returns" not in df.columns or len(df) == 0:
        return 0.0, 0.0
    arr = pdx.to_numeric(df["returns"], errors="coerce").to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.median(arr))


def _results_required_columns() -> list[str]:
    return [
        "family",
        "config_id",
        "seed",
        "W",
        "T",
        "A",
        "B",
        "start_date",
        "end_date",
        "bars",
        "trades",
        "win_rate",
        "avg_ret",
        "med_ret",
        "profit_factor",
        "max_drawdown",
    ]


def _assert_results_integrity(df: Any) -> None:
    pdx = _require_pandas()
    required = _results_required_columns()
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Family results missing required columns: {missing}")

    if len(df) > 0:
        dup = df.duplicated(subset=["config_id", "seed"], keep=False)
        if bool(dup.any()):
            first = int(np.where(dup.to_numpy(dtype=bool))[0][0])
            raise RuntimeError(
                f"Family results must be unique by (config_id, seed). Duplicate at row={first}"
            )

    numeric_required = ["seed", "W", "T", "A", "B", "bars", "trades", "win_rate", "avg_ret", "med_ret", "profit_factor", "max_drawdown"]
    for col in numeric_required:
        vals = pdx.to_numeric(df[col], errors="raise").to_numpy(dtype=np.float64)
        if vals.size and (not np.all(np.isfinite(vals))):
            bad_idx = int(np.where(~np.isfinite(vals))[0][0])
            raise RuntimeError(f"Family results column {col!r} has non-finite value at row={bad_idx}")


def _canonical_results_sha256(df: Any) -> str:
    pdx = _require_pandas()
    norm = df.copy()
    if norm.shape[0] > 0:
        norm = norm.sort_values(["config_id", "seed"], kind="mergesort").reset_index(drop=True)

    for col in norm.columns:
        if pdx.api.types.is_numeric_dtype(norm[col]):
            norm[col] = pdx.to_numeric(norm[col], errors="raise").astype(np.float64)
            norm[col] = norm[col].round(8)
        else:
            norm[col] = norm[col].astype("string")

    canonical_bytes = norm.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


def _estimate_gap_reset_stats(
    data_paths: list[str],
    data_loader: Callable[[str, str], Any],
    tz_name: str,
    gap_reset_minutes: float,
) -> dict[str, Any]:
    if not data_paths:
        return {"symbol_path": "", "rows": 0, "gap_resets": 0, "gap_reset_rate": 0.0}
    path = str(data_paths[0])
    pdx = _require_pandas()
    try:
        df = data_loader(path, tz_name)
        if not isinstance(df.index, pdx.DatetimeIndex):
            return {"symbol_path": path, "rows": 0, "gap_resets": 0, "gap_reset_rate": 0.0}
        idx = pdx.DatetimeIndex(df.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        idx = idx.tz_convert("UTC")
        ts_ns = idx.asi8.astype(np.int64)
        if ts_ns.size == 0:
            return {"symbol_path": path, "rows": 0, "gap_resets": 0, "gap_reset_rate": 0.0}
        gap = np.zeros(ts_ns.size, dtype=np.float64)
        gap[1:] = (ts_ns[1:] - ts_ns[:-1]) / float(60 * 1_000_000_000)
        resets = int(1 + np.sum(gap[1:] > float(gap_reset_minutes)))
        rate = float(resets / max(int(ts_ns.size), 1))
        return {
            "symbol_path": path,
            "rows": int(ts_ns.size),
            "gap_resets": int(resets),
            "gap_reset_rate": float(rate),
        }
    except Exception:
        return {"symbol_path": path, "rows": 0, "gap_resets": 0, "gap_reset_rate": 0.0}


def _build_family_results_rows(
    family_name: str,
    run_dir: Path,
    cfg: RunConfigModel,
    bars_total: int,
) -> list[dict[str, Any]]:
    pdx = _require_pandas()
    lb_path = run_dir / "leaderboard.csv"
    rb_path = run_dir / "robustness_leaderboard.csv"
    if lb_path.exists():
        lb_df = pdx.read_csv(lb_path)
    elif rb_path.exists():
        lb_df = pdx.read_csv(rb_path)
    else:
        return []

    rows: list[dict[str, Any]] = []
    if "candidate_id" not in lb_df.columns:
        return rows

    start_date = cfg.data.start.isoformat() if cfg.data.start is not None else ""
    end_date = cfg.data.end.isoformat() if cfg.data.end is not None else ""
    seed = int(cfg.harness.seed)

    for _, lb_row in lb_df.sort_values("candidate_id", kind="mergesort").iterrows():
        candidate_id = str(lb_row.get("candidate_id", "")).strip()
        if not candidate_id:
            continue
        candidate_dir = run_dir / "candidates" / candidate_id
        if not candidate_dir.exists():
            raise RuntimeError(f"Missing candidate directory for candidate_id={candidate_id}: {candidate_dir}")
        metrics_doc = _json_load(candidate_dir / "candidate_metrics.json", default={})
        config_doc = _json_load(candidate_dir / "candidate_config.json", default={})
        base_metrics = metrics_doc.get("base_metrics", {}) if isinstance(metrics_doc, dict) else {}
        if not isinstance(base_metrics, dict):
            raise RuntimeError(f"candidate_metrics.json base_metrics missing/invalid for candidate_id={candidate_id}")
        required_metric_keys = ("n_trades", "win_rate", "profit_factor", "max_drawdown")
        missing_metric_keys = [k for k in required_metric_keys if k not in base_metrics]
        if missing_metric_keys:
            raise RuntimeError(
                f"candidate_metrics.json missing keys for candidate_id={candidate_id}: {missing_metric_keys}"
            )

        m2_idx = _safe_int(lb_row.get("m2_idx", 0), default=0)
        m4_idx = _safe_int(lb_row.get("m4_idx", 0), default=0)
        if not (0 <= m2_idx < len(cfg.module2_configs)):
            m2_idx = 0
        if not (0 <= m4_idx < len(cfg.module4_configs)):
            m4_idx = 0

        m2_cfg = cfg.module2_configs[m2_idx]
        m4_cfg = cfg.module4_configs[m4_idx]
        avg_ret, med_ret = _candidate_returns_stats(candidate_dir)

        row: dict[str, Any] = {
            "family": family_name,
            "config_id": candidate_id,
            "seed": int(seed),
            "W": int(m2_cfg.profile_window_bars),
            "T": float(m4_cfg.entry_threshold),
            "A": float(m4_cfg.trend_poc_drift_min_abs),
            "B": float(m4_cfg.neutral_poc_drift_max_abs),
            "start_date": start_date,
            "end_date": end_date,
            "bars": int(bars_total),
            "trades": int(_safe_int(base_metrics.get("n_trades", 0), default=0)),
            "win_rate": float(_safe_float(base_metrics.get("win_rate", 0.0), default=0.0)),
            "avg_ret": float(avg_ret),
            "med_ret": float(med_ret),
            "profit_factor": float(_safe_float(base_metrics.get("profit_factor", 0.0), default=0.0)),
            "max_drawdown": float(_safe_float(base_metrics.get("max_drawdown", 0.0), default=0.0)),
            "m2_idx": int(m2_idx),
            "m4_idx": int(m4_idx),
        }

        # Merge existing leaderboard features.
        for k in lb_df.columns:
            if k not in row:
                row[str(k)] = lb_row.get(k)

        # Merge candidate module4 and base metrics as flattened fields.
        if isinstance(config_doc, dict):
            m4_payload = config_doc.get("module4_config", {})
            if isinstance(m4_payload, dict):
                flat_m4: dict[str, Any] = {}
                _flatten_dict("module4", m4_payload, flat_m4)
                for k, v in flat_m4.items():
                    row.setdefault(k, v)
        if isinstance(base_metrics, dict):
            flat_metrics: dict[str, Any] = {}
            _flatten_dict("base_metrics", base_metrics, flat_metrics)
            for k, v in flat_metrics.items():
                row.setdefault(k, v)

        # Re-assert required row contract after merges.
        row["family"] = family_name
        row["config_id"] = candidate_id
        row["seed"] = int(seed)
        row["W"] = int(m2_cfg.profile_window_bars)
        row["T"] = float(m4_cfg.entry_threshold)
        row["A"] = float(m4_cfg.trend_poc_drift_min_abs)
        row["B"] = float(m4_cfg.neutral_poc_drift_max_abs)
        row["start_date"] = start_date
        row["end_date"] = end_date
        row["bars"] = int(bars_total)
        row["trades"] = int(_safe_int(base_metrics.get("n_trades", 0), default=0))
        row["win_rate"] = float(_safe_float(base_metrics.get("win_rate", 0.0), default=0.0))
        row["avg_ret"] = float(avg_ret)
        row["med_ret"] = float(med_ret)
        row["profit_factor"] = float(_safe_float(base_metrics.get("profit_factor", 0.0), default=0.0))
        row["max_drawdown"] = float(_safe_float(base_metrics.get("max_drawdown", 0.0), default=0.0))
        rows.append(row)

    rows.sort(key=lambda x: (str(x.get("config_id", "")), int(_safe_int(x.get("seed", 0), 0))))
    return rows


def _write_family_artifacts(
    *,
    family_root: Path,
    family_name: str,
    config_path: Path,
    cfg: RunConfigModel,
    resolved_sha: str,
    run_dir: Path,
    run_id: str,
    run_summary: dict[str, Any],
    data_paths: list[str],
    data_loader: Callable[[str, str], Any],
    jitter_seconds: float,
    family_start_utc: datetime,
) -> dict[str, Any]:
    family_root.mkdir(parents=True, exist_ok=True)
    audit_dir = family_root / "audit_bundle"
    audit_dir.mkdir(parents=True, exist_ok=True)

    run_manifest = _json_load(run_dir / "run_manifest.json", default={})
    run_status = _json_load(run_dir / "run_status.json", default={})
    bars_total = int(run_manifest.get("ingestion", {}).get("master_rows", 0)) if isinstance(run_manifest, dict) else 0

    rows = _build_family_results_rows(
        family_name=family_name,
        run_dir=run_dir,
        cfg=cfg,
        bars_total=bars_total,
    )
    pdx = _require_pandas()
    results_df = pdx.DataFrame(rows)
    if not results_df.empty:
        results_df = results_df.sort_values(["config_id", "seed"], kind="mergesort").reset_index(drop=True)
    _assert_results_integrity(results_df)
    results_path = family_root / "results.parquet"
    results_df.to_parquet(results_path, index=False)
    results_sha = _canonical_results_sha256(results_df)
    results_file_sha = hashlib.sha256(results_path.read_bytes()).hexdigest()

    gap_stats = _estimate_gap_reset_stats(
        data_paths=data_paths,
        data_loader=data_loader,
        tz_name=cfg.harness.timezone,
        gap_reset_minutes=float(cfg.engine.gap_reset_minutes),
    )

    summary_doc = {
        "family": family_name,
        "run_id": str(run_id),
        "source_run_dir": str(run_dir),
        "config_path": str(config_path),
        "resolved_config_sha256": str(resolved_sha),
        "rows": int(results_df.shape[0]),
        "aborted": bool(run_summary.get("aborted", False)),
        "failure_count": int(run_summary.get("failure_count", 0)),
        "runtime_warning_count": int(run_summary.get("runtime_warning_count", 0)),
        "jitter_seconds": float(jitter_seconds),
        "gap_reset_minutes": float(cfg.engine.gap_reset_minutes),
        "gap_reset_stats": gap_stats,
        "family_started_utc": family_start_utc.isoformat(),
        "family_finished_utc": datetime.now(timezone.utc).isoformat(),
        "results_sha256_canonical": str(results_sha),
        "results_sha256_file": str(results_file_sha),
    }
    (family_root / "summary.json").write_text(
        json.dumps(summary_doc, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Audit bundle
    (audit_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (audit_dir / "git_commit.txt").write_text(_safe_git_hash(Path(__file__).resolve().parent) + "\n", encoding="utf-8")
    (audit_dir / "python_version.txt").write_text(sys.version + "\n", encoding="utf-8")
    try:
        pip_out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    except Exception as exc:
        pip_out = f"ERROR: {type(exc).__name__}: {exc}\n"
    (audit_dir / "pip_freeze.txt").write_text(pip_out, encoding="utf-8")
    env_lines = [f"{k}={v}" for k, v in sorted(os.environ.items(), key=lambda x: x[0])]
    (audit_dir / "env_vars.txt").write_text("\n".join(env_lines) + "\n", encoding="utf-8")
    machine_doc = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_executable": sys.executable,
        "hostname": socket.gethostname(),
    }
    (audit_dir / "machine_info.json").write_text(
        json.dumps(machine_doc, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    time_doc = {
        "family_start_utc": family_start_utc.isoformat(),
        "family_end_utc": datetime.now(timezone.utc).isoformat(),
    }
    (audit_dir / "timestamps.json").write_text(
        json.dumps(time_doc, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (audit_dir / "results_parquet_sha256_canonical.txt").write_text(results_sha + "\n", encoding="utf-8")
    (audit_dir / "results_parquet_sha256_file.txt").write_text(results_file_sha + "\n", encoding="utf-8")
    if isinstance(run_manifest, dict):
        (audit_dir / "run_manifest.json").write_text(
            json.dumps(run_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if isinstance(run_status, dict):
        (audit_dir / "run_status.json").write_text(
            json.dumps(run_status, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return summary_doc


def main() -> None:
    parser = argparse.ArgumentParser(description="Weightiz V3.5 research runner")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Load config/data and initialize workers, then exit")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    config_path = Path(args.config).expanduser().resolve()
    cfg = _load_config(config_path)
    resolved_sha = _resolved_config_sha256(cfg)

    if cfg.zimtra_sweep is not None and bool(cfg.zimtra_sweep.enabled):
        from sweep_runner import run_zimtra_sweep

        zimtra_summary = run_zimtra_sweep(
            cfg=cfg,
            project_root=project_root,
            config_path=config_path,
            resolved_config_sha256=resolved_sha,
        )
        print("RUN_COMPLETE")
        print(json.dumps(zimtra_summary, ensure_ascii=False, indent=2))
        return

    family_mode = _family_mode_enabled(cfg.run_name)
    family_root = (project_root / "artifacts" / str(cfg.run_name)).resolve()
    family_log_path = family_root / "run.log"
    family_started_utc = datetime.now(timezone.utc)

    if family_mode:
        family_root.mkdir(parents=True, exist_ok=True)
        (family_root / "pid").write_text(str(os.getpid()) + "\n", encoding="utf-8")
        _family_log_append(family_log_path, f"FAMILY_MODE_START run_name={cfg.run_name} config={config_path}")
        if str(cfg.harness.parallel_backend) != "process_pool":
            raise RuntimeError(
                "Family sweeps require process_pool backend (fail-closed). "
                f"Got harness.parallel_backend={cfg.harness.parallel_backend!r}"
            )
        if int(cfg.harness.parallel_workers) > 14:
            raise ValueError(
                f"Strict cap exceeded: {int(cfg.harness.parallel_workers)} > 14. Fail-closed."
            )

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

    if bool(args.dry_run):
        # Deterministic infrastructure check only (no strategy execution).
        loaded_rows = 0
        for p in data_paths:
            fr = data_loader(p, str(cfg.harness.timezone))
            loaded_rows += int(getattr(fr, "shape", [0])[0])

        workers = int(max(1, cfg.harness.parallel_workers))
        if str(cfg.harness.parallel_backend) == "process_pool":
            with mp.Pool(processes=workers) as pool:
                _ = pool.map(int, range(min(workers, 4)))

        summary = {
            "dry_run": True,
            "config_path": str(config_path),
            "symbols": len(symbols),
            "data_files": len(data_paths),
            "loaded_rows": int(loaded_rows),
            "parallel_backend": str(cfg.harness.parallel_backend),
            "parallel_workers": int(workers),
            "status": "ok",
        }
        print("DRY_RUN_COMPLETE")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    jitter_seconds = 0.0
    if family_mode:
        jitter_seconds = float(_deterministic_jitter_seconds(str(cfg.run_name), int(cfg.harness.seed)))
        _family_log_append(
            family_log_path,
            f"FAMILY_JITTER_SECONDS={jitter_seconds:.6f} formula=10+(sha256(run_name+seed)%21)",
        )
        time.sleep(jitter_seconds)

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always", RuntimeWarning)
        if family_mode:
            _family_log_append(
                family_log_path,
                (
                    "HARNESS_START "
                    f"symbols={len(symbols)} m2={len(m2_cfgs)} m3={len(m3_cfgs)} "
                    f"m4={len(m4_cfgs)} workers={harness_cfg.parallel_workers}"
                ),
            )
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

    if family_mode:
        family_summary = _write_family_artifacts(
            family_root=family_root,
            family_name=str(cfg.run_name),
            config_path=config_path,
            cfg=cfg,
            resolved_sha=resolved_sha,
            run_dir=run_dir,
            run_id=run_id,
            run_summary=summary,
            data_paths=data_paths,
            data_loader=data_loader,
            jitter_seconds=jitter_seconds,
            family_start_utc=family_started_utc,
        )
        _family_log_append(
            family_log_path,
            (
                "FAMILY_MODE_COMPLETE "
                f"rows={int(family_summary.get('rows', 0))} "
                f"results_sha256_canonical={family_summary.get('results_sha256_canonical', '')}"
            ),
        )

    print("RUN_COMPLETE")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
