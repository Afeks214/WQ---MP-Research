from __future__ import annotations

from dataclasses import dataclass, field

from module6.constants import (
    MODULE6_CONSTRAINT_POLICY_VERSION,
    MODULE6_FRICTION_POLICY_VERSION,
    MODULE6_OVERNIGHT_POLICY_VERSION,
    MODULE6_RANKING_POLICY_VERSION,
    MODULE6_SUPPORT_POLICY_VERSION,
)


@dataclass(frozen=True)
class IntakeConfig:
    min_availability_ratio: float = 0.95
    min_observed_sessions: int = 126
    require_bridge_artifacts: bool = True
    canonical_selection_stage: str = "module5_bridge_canonical_baseline_v1"
    require_zero_filled_daily_returns_non_authoritative: bool = True
    required_comparison_support: float = 0.85


@dataclass(frozen=True)
class ReductionConfig:
    projection_width: int = 32
    ann_top_k: int = 24
    duplicate_corr_threshold: float = 0.85
    drawdown_concurrence_threshold: float = 0.60
    pre_reduction_cap: int = 256
    reduced_universe_cap: int = 128
    mv_universe_cap: int = 64
    hedge_keep_count: int = 16
    min_exact_duplicate_overlap: float = 0.999999


@dataclass(frozen=True)
class DependenceConfig:
    shrinkage_floor_eps_mult: float = 1.0e-10
    negative_mass_reject_mult: float = 1.0e-6
    drawdown_tail_threshold: float = 0.05
    overlap_weight_symbol_support: float = 0.25
    overlap_weight_activity: float = 0.35
    overlap_weight_gross: float = 0.25
    overlap_weight_rebalance: float = 0.15
    activity_signature_buckets: int = 8


@dataclass(frozen=True)
class GeneratorConfig:
    random_sparse_quota: int = 24000
    cluster_balanced_quota: int = 4000
    hrp_variant_quota: int = 36
    mv_variant_quota: int = 18
    random_sparse_batch_size: int = 2000
    active_cardinality_choices: tuple[int, ...] = (4, 6, 8, 12, 16, 24)
    per_sleeve_cap: float = 0.25
    per_cluster_cap: float = 0.25
    per_family_cap: float = 0.35
    minimum_cash_weight: float = 0.02
    enable_mv_diagnostic: bool = False
    random_seed: int = 17


@dataclass(frozen=True)
class SimulatorConfig:
    fixed_fee: float = 0.0
    linear_cost_bps: float = 1.0
    slippage_cost_bps: float = 2.0
    rebalance_band_l1: float = 0.10
    daily_loss_limit_frac: float = 0.10
    account_disable_equity: float = 1000.0
    intraday_leverage_max: float = 6.0
    overnight_leverage: float = 2.0
    max_overnight_sleeves: int = 4
    max_cluster_weight: float = 0.35
    max_family_weight: float = 0.35
    max_sleeve_weight: float = 0.25
    min_cash_weight: float = 0.02
    constraint_policy_version: str = MODULE6_CONSTRAINT_POLICY_VERSION
    overnight_policy_version: str = MODULE6_OVERNIGHT_POLICY_VERSION
    friction_policy_version: str = MODULE6_FRICTION_POLICY_VERSION
    support_policy_version: str = MODULE6_SUPPORT_POLICY_VERSION


@dataclass(frozen=True)
class ScoringConfig:
    min_truth_score_ratio: float = 0.80
    max_allowed_return_drift_frac: float = 0.20
    return_drift_floor: float = 0.005
    return_scale_floor: float = 0.02
    max_allowed_drawdown_drift: float = 0.02
    max_allowed_turnover_drift_frac: float = 0.25
    turnover_drift_floor: float = 0.02
    turnover_scale_floor: float = 0.05
    max_allowed_gross_exposure_drift_frac: float = 0.15
    gross_exposure_drift_floor: float = 0.10
    gross_exposure_scale_floor: float = 0.50
    max_allowed_breach_count_delta: int = 0
    min_rank_stability: float = 0.90
    max_abs_rank_delta_p95: int = 16
    ranking_policy_version: str = MODULE6_RANKING_POLICY_VERSION
    min_cross_universe_support: float = 0.85
    shortlist_session_keep: int = 1024
    shortlist_minute_keep: int = 256
    final_scalar_keep: int = 64
    final_primary_count: int = 6
    final_alternate_count: int = 6


@dataclass(frozen=True)
class ExportConfig:
    output_subdir_name: str = "module6"
    write_matrix_store: bool = True
    write_frontiers: bool = True


@dataclass(frozen=True)
class RuntimeConfig:
    portfolio_batch_size: int = 2000
    minute_replay_batch_size: int = 32
    overlap_pair_block_size: int = 128
    random_projection_seed: int = 17


@dataclass(frozen=True)
class Module6Config:
    intake: IntakeConfig = field(default_factory=IntakeConfig)
    reduction: ReductionConfig = field(default_factory=ReductionConfig)
    dependence: DependenceConfig = field(default_factory=DependenceConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

