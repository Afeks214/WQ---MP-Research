from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.stage_a_discovery import STAGE_A_RESEARCH_THRESHOLD


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

    structural_windows: list[int] = Field(default_factory=lambda: [5, 15, 30, 60])
    selected_window: int = 30
    validate_outputs: bool = True
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
    rolling_context_period: int = 5
    fail_on_non_finite_input: bool = True
    fail_on_non_finite_output: bool = True
    fail_on_bad_indices: bool = True
    fail_on_missing_prev_va: bool = False
    eps: float = 1e-12

    @field_validator("structural_windows", mode="before")
    @classmethod
    def validate_structural_windows_input(cls, value: Any) -> Any:
        if not isinstance(value, list):
            raise ValueError("module3.structural_windows must be a list of positive integers")
        for entry in value:
            if isinstance(entry, bool) or not isinstance(entry, int):
                raise ValueError("module3.structural_windows entries must be positive integers")
        return value


class Module4ConfigModel(BaseModel):
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
    execution_strict_prices: bool = True

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
        if "delta_th" in self.model_fields_set:
            self.entry_threshold = float(self.delta_th)
        return self


class HarnessConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 97
    research_mode: Literal["standard", "discovery"] = "standard"
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
    process_pool_candidate_chunk_size: int = 1
    group_bound_execution_enabled: bool = True
    group_dispatch_policy: str = "largest_first_stable"
    group_max_in_flight_factor: int = 2
    group_target_wall_time_sec: float = 30.0
    group_max_result_payload_bytes: int = 4 * 1024 * 1024
    group_max_memory_bytes: int = 0
    group_min_candidates_per_chunk: int = 1
    group_max_candidates_per_chunk_hard: int = 64
    startup_default_candidate_loop_sec: float = 0.50
    startup_default_result_payload_bytes: int = 32 * 1024
    startup_default_candidate_incremental_bytes: int = 1 * 1024 * 1024
    startup_default_module3_bytes: int = 512 * 1024 * 1024
    scratch_mode: Literal["auto", "compact", "full"] = "auto"
    strict_candidate_state_validation: Literal["compact_execution_view", "full_tensorstate"] = "compact_execution_view"
    risk_breach_state_dump_enabled: bool = False
    debug_full_state_payloads: bool = False
    module3_output_mode: Literal["full_legacy", "signal_only"] = "full_legacy"
    # Configured base-state transport mode:
    # - auto: resolve to fork_cow only on linux+fork, otherwise fallback-only serialized_copy
    # - fork_cow: require copy-on-write transport
    # - explicit_shm: reserved and fail-closed until implemented
    base_sharing_mode: Literal["auto", "fork_cow", "explicit_shm"] = "auto"
    cow_private_ratio_threshold: float = 0.10
    cow_probe_workers: int = 8
    safety_margin_frac: float = 0.15
    safety_margin_min_bytes: int = 8 * 1024 * 1024 * 1024
    max_queue_bytes_frac: float = 0.02
    max_result_buffer_bytes_frac: float = 0.05
    throughput_minimal_observability: bool = False
    health_check_interval: int = 50
    progress_interval_seconds: int = 10
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
    robustness_reject_threshold: float = STAGE_A_RESEARCH_THRESHOLD
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
        if int(self.process_pool_candidate_chunk_size) < 1:
            raise ValueError("harness.process_pool_candidate_chunk_size must be >=1")
        if int(self.group_max_in_flight_factor) < 1:
            raise ValueError("harness.group_max_in_flight_factor must be >=1")
        if float(self.group_target_wall_time_sec) <= 0.0:
            raise ValueError("harness.group_target_wall_time_sec must be >0")
        if int(self.group_max_result_payload_bytes) < 1024:
            raise ValueError("harness.group_max_result_payload_bytes must be >=1024")
        if int(self.group_max_memory_bytes) < 0:
            raise ValueError("harness.group_max_memory_bytes must be >=0")
        if int(self.group_min_candidates_per_chunk) < 1:
            raise ValueError("harness.group_min_candidates_per_chunk must be >=1")
        if int(self.group_max_candidates_per_chunk_hard) < int(self.group_min_candidates_per_chunk):
            raise ValueError("harness.group_max_candidates_per_chunk_hard must be >= harness.group_min_candidates_per_chunk")
        if float(self.startup_default_candidate_loop_sec) <= 0.0:
            raise ValueError("harness.startup_default_candidate_loop_sec must be >0")
        if int(self.startup_default_result_payload_bytes) < 1024:
            raise ValueError("harness.startup_default_result_payload_bytes must be >=1024")
        if int(self.startup_default_candidate_incremental_bytes) < 1024:
            raise ValueError("harness.startup_default_candidate_incremental_bytes must be >=1024")
        if int(self.startup_default_module3_bytes) < 1024:
            raise ValueError("harness.startup_default_module3_bytes must be >=1024")
        if not (0.0 <= float(self.cow_private_ratio_threshold) <= 1.0):
            raise ValueError("harness.cow_private_ratio_threshold must be in [0,1]")
        if int(self.cow_probe_workers) < 1:
            raise ValueError("harness.cow_probe_workers must be >=1")
        if not (0.0 <= float(self.safety_margin_frac) <= 1.0):
            raise ValueError("harness.safety_margin_frac must be in [0,1]")
        if int(self.safety_margin_min_bytes) < 1024:
            raise ValueError("harness.safety_margin_min_bytes must be >=1024")
        if not (0.0 <= float(self.max_queue_bytes_frac) <= 1.0):
            raise ValueError("harness.max_queue_bytes_frac must be in [0,1]")
        if not (0.0 <= float(self.max_result_buffer_bytes_frac) <= 1.0):
            raise ValueError("harness.max_result_buffer_bytes_frac must be in [0,1]")
        if int(self.health_check_interval) < 1:
            raise ValueError("harness.health_check_interval must be >=1")
        if int(self.progress_interval_seconds) < 1:
            raise ValueError("harness.progress_interval_seconds must be >=1")
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

        if self.search.seed is None and isinstance(self.zimtra_sweep, dict):
            legacy_seed = self.zimtra_sweep.get("seed")
            if legacy_seed is not None:
                self.search.seed = int(legacy_seed)
        if self.search.seed is None:
            self.search.seed = int(self.harness.seed)
        return self
