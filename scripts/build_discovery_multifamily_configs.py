from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
LOCAL_CONFIG_PATH = ROOT / "configs" / "local_discovery_short_7core.yaml"
CLOUD_PLAN_PATH = ROOT / "configs" / "cloud_discovery_20k_plan.yaml"
LOCAL_ADAPTIVE_CONFIG_PATH = ROOT / "configs" / "local_adaptive_discovery_7core.yaml"
LOCAL_ADAPTIVE_PLAN_PATH = ROOT / "configs" / "local_adaptive_discovery_7core_plan.yaml"


LIVE_AXES = [
    "module3.block_minutes",
    "module4.entry_threshold",
    "module4.exit_threshold",
    "module4.regime_confidence_min",
    "module4.window_selection_mode",
    "module4.fixed_window_index",
    "module4.anchor_window_index",
    "module4.max_abs_weight",
    "module4.conviction_scale",
    "module4.conviction_clip",
    "module4.trend_spread_min",
    "module4.trend_poc_drift_min_abs",
    "module4.neutral_poc_drift_max_abs",
    "module4.shape_skew_min_abs",
    "module4.double_dist_sep_x",
    "module4.double_dist_valley_frac",
    "module4.max_volatility",
    "module4.max_spread",
    "module4.min_liquidity",
]

DEAD_AXES = [
    "module4.top_k_intraday",
    "module4.max_asset_cap_frac",
    "module4.max_turnover_frac_per_bar",
    "module4.overnight_min_conviction",
    "module4.allow_cash_overnight",
    "module4.commission_bps",
    "module4.spread_tick_mult",
    "module4.slippage_bps_low_rvol",
    "module4.slippage_bps_mid_rvol",
    "module4.slippage_bps_high_rvol",
    "module4.stress_slippage_mult",
    "module4.hard_kill_on_daily_loss_breach",
    "module4.execution_strict_prices",
    "module4.strategy_type",
    "module4.score_gate",
    "module4.score_gate_rule",
    "module4.deviation_signal",
    "module4.deviation_rule",
    "module4.entry_model",
    "module4.exit_model",
    "module4.origin_level",
    "module4.direction",
    "module4.delta_th",
    "module4.dev_th",
    "module4.tp_mult",
    "module4.atr_stop_mult",
]

UNCERTAIN_AXES = [
    "module3.block_minutes",
    "module3.include_partial_last_block",
    "module3.min_block_valid_ratio",
    "module3.ib_pop_frac",
    "module4.max_volatility",
    "module4.max_spread",
    "module4.min_liquidity",
]

MODULE3_WINDOWS = [15, 20, 30, 40]
CANONICAL_MODULE4_WINDOWS = [5, 15, 30, 60]


def _base_module4_config() -> dict[str, object]:
    return {
        "entry_threshold": 0.55,
        "exit_threshold": 0.28,
        "regime_confidence_min": 0.58,
        "top_k_intraday": 5,
        "max_asset_cap_frac": 0.30,
        "max_turnover_frac_per_bar": 0.35,
        "overnight_min_conviction": 0.60,
        "allow_cash_overnight": True,
        "fail_on_non_finite_input": False,
        "fail_on_non_finite_output": False,
        "conviction_scale": 1.0,
        "conviction_clip": 1.0,
        "max_abs_weight": 1.0,
        "window_selection_mode": "multi_window",
        "fixed_window_index": 2,
        "anchor_window_index": 2,
        "trend_spread_min": 0.05,
        "trend_poc_drift_min_abs": 0.35,
        "neutral_poc_drift_max_abs": 0.15,
        "shape_skew_min_abs": 0.35,
        "double_dist_sep_x": 1.0,
        "double_dist_valley_frac": 0.35,
    }


def _family_a_configs() -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []
    for entry_threshold in [0.50, 0.55, 0.60, 0.65, 0.70]:
        for regime_confidence_min in [0.50, 0.58, 0.66, 0.74]:
            cfg = _base_module4_config()
            cfg["entry_threshold"] = float(entry_threshold)
            cfg["regime_confidence_min"] = float(regime_confidence_min)
            configs.append(cfg)
    return configs


def _family_b_configs() -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []
    for entry_threshold in [0.50, 0.56, 0.62, 0.68, 0.74]:
        for exit_threshold in [0.18, 0.24, 0.30, 0.36]:
            cfg = _base_module4_config()
            cfg["entry_threshold"] = float(entry_threshold)
            cfg["exit_threshold"] = float(exit_threshold)
            configs.append(cfg)
    return configs


def _family_c_configs() -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []
    for trend_spread_min in [0.03, 0.05, 0.07, 0.09, 0.11]:
        for shape_skew_min_abs in [0.20, 0.35, 0.50, 0.65]:
            cfg = _base_module4_config()
            cfg["trend_spread_min"] = float(trend_spread_min)
            cfg["shape_skew_min_abs"] = float(shape_skew_min_abs)
            configs.append(cfg)
    return configs


def _family_d_configs() -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []
    for fixed_window_index in [0, 1, 2, 3]:
        for max_abs_weight in [0.20, 0.35, 0.50, 0.65, 0.80]:
            cfg = _base_module4_config()
            cfg["window_selection_mode"] = "single_window"
            cfg["fixed_window_index"] = int(fixed_window_index)
            cfg["anchor_window_index"] = int(fixed_window_index)
            cfg["max_abs_weight"] = float(max_abs_weight)
            configs.append(cfg)
    return configs


def _adaptive_family_a_configs() -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []
    for entry_threshold in [0.48, 0.54, 0.60, 0.66, 0.72, 0.78]:
        for regime_confidence_min in [0.40, 0.52, 0.64, 0.76]:
            cfg = _base_module4_config()
            cfg["entry_threshold"] = float(entry_threshold)
            cfg["regime_confidence_min"] = float(regime_confidence_min)
            configs.append(cfg)
    return configs


def _adaptive_family_b_configs() -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []
    for entry_threshold in [0.50, 0.58, 0.66, 0.74]:
        for exit_threshold in [0.10, 0.20, 0.30]:
            for conviction_scale in [0.85, 1.15]:
                cfg = _base_module4_config()
                cfg["entry_threshold"] = float(entry_threshold)
                cfg["exit_threshold"] = float(exit_threshold)
                cfg["conviction_scale"] = float(conviction_scale)
                configs.append(cfg)
    return configs


def _adaptive_family_c_configs() -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []
    for trend_spread_min in [0.03, 0.05, 0.07, 0.09]:
        for shape_skew_min_abs in [0.20, 0.35, 0.50]:
            for regime_confidence_min in [0.50, 0.68]:
                cfg = _base_module4_config()
                cfg["trend_spread_min"] = float(trend_spread_min)
                cfg["shape_skew_min_abs"] = float(shape_skew_min_abs)
                cfg["regime_confidence_min"] = float(regime_confidence_min)
                configs.append(cfg)
    return configs


def _adaptive_family_d_configs() -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []
    for fixed_window_index in [0, 1, 2, 3]:
        for max_abs_weight in [0.25, 0.50, 0.75]:
            for conviction_clip in [0.75, 1.25]:
                cfg = _base_module4_config()
                cfg["window_selection_mode"] = "single_window"
                cfg["fixed_window_index"] = int(fixed_window_index)
                cfg["anchor_window_index"] = int(fixed_window_index)
                cfg["max_abs_weight"] = float(max_abs_weight)
                cfg["conviction_clip"] = float(conviction_clip)
                configs.append(cfg)
    return configs


def _module3_configs() -> list[dict[str, object]]:
    return [
        {
            "block_minutes": 15,
            "phase_mask": [1, 2],
            "min_block_valid_bars": 8,
            "min_block_valid_ratio": 0.65,
            "fail_on_non_finite_output": False,
            "include_partial_last_block": True,
            "ib_pop_frac": 0.008,
        },
        {
            "block_minutes": 20,
            "phase_mask": [1, 2],
            "min_block_valid_bars": 8,
            "min_block_valid_ratio": 0.68,
            "fail_on_non_finite_output": False,
            "include_partial_last_block": True,
            "ib_pop_frac": 0.009,
        },
        {
            "block_minutes": 30,
            "phase_mask": [1, 2],
            "min_block_valid_bars": 8,
            "min_block_valid_ratio": 0.70,
            "fail_on_non_finite_output": False,
            "include_partial_last_block": True,
            "ib_pop_frac": 0.01,
        },
        {
            "block_minutes": 40,
            "phase_mask": [1, 2],
            "min_block_valid_bars": 8,
            "min_block_valid_ratio": 0.74,
            "fail_on_non_finite_output": False,
            "include_partial_last_block": True,
            "ib_pop_frac": 0.01,
        },
    ]


def build_local_short_run_config() -> dict[str, object]:
    module4_configs = (
        _family_a_configs()
        + _family_b_configs()
        + _family_c_configs()
        + _family_d_configs()
    )
    return {
        "run_name": "local_discovery_short_7core",
        "symbols": ["SPY", "QQQ", "IWM", "TLT"],
        "data": {
            "root": "./data/minute",
            "format": "parquet",
            "timestamp_column": "timestamp",
            "start": "2024-01-02T00:00:00Z",
            "end": "2024-01-11T23:59:59Z",
        },
        "engine": {
            "B": 240,
            "x_min": -6.0,
            "dx": 0.05,
            "initial_cash": 1_000_000.0,
            "intraday_leverage_max": 6.0,
            "overnight_leverage": 2.0,
            "overnight_positions_max": 1,
            "daily_loss_limit_abs": 50_000.0,
            "seed": 17,
            "fail_on_nan": True,
            "tick_size_default": 0.01,
        },
        "module2_configs": [
            {
                "profile_window_bars": 60,
                "profile_warmup_bars": 60,
                "atr_span": 10,
                "rvol_lookback_sessions": 12,
                "va_threshold": 0.68,
                "fail_on_non_finite_output": True,
            }
        ],
        "module3_configs": _module3_configs(),
        "module4_configs": module4_configs,
        "harness": {
            "seed": 97,
            "timezone": "America/New_York",
            "freq": "1min",
            "min_asset_coverage": 0.8,
            "purge_bars": 30,
            "embargo_bars": 15,
            "wf_train_sessions": 2,
            "wf_test_sessions": 2,
            "wf_step_sessions": 2,
            "cpcv_slices": 2,
            "cpcv_k_test": 1,
            "parallel_backend": "process_pool",
            "parallel_workers": 7,
            "stress_profile": "baseline_only",
            "max_ram_utilization_frac": 0.7,
            "enforce_lookahead_guard": True,
            "report_dir": "./artifacts",
            "fail_on_non_finite": True,
            "daily_return_min_days": 2,
            "benchmark_symbol": "SPY",
            "export_micro_diagnostics": True,
            "micro_diag_mode": "events_only",
            "micro_diag_export_block_profiles": True,
            "micro_diag_export_funnel": True,
            "micro_diag_max_rows": 5_000_000,
        },
        "stress_scenarios": [
            {
                "scenario_id": "baseline",
                "name": "baseline",
                "missing_burst_prob": 0.0,
                "missing_burst_min": 0,
                "missing_burst_max": 0,
                "jitter_sigma_bps": 0.0,
                "slippage_mult": 1.0,
                "enabled": True,
            }
        ],
        "candidates": {
            "mode": "auto_grid",
            "specs": [],
        },
    }


def build_cloud_plan() -> dict[str, object]:
    family_entries = [
        {
            "family_name": "family_a_activation_frontier",
            "research_purpose": "Entry/regime activation frontier under adaptive window selection.",
            "local_live_axes": {
                "entry_threshold": [0.50, 0.55, 0.60, 0.65, 0.70],
                "regime_confidence_min": [0.50, 0.58, 0.66, 0.74],
            },
            "cloud_live_axes": {
                "entry_threshold": {"start": 0.48, "stop": 0.72, "step": 0.01, "count": 25},
                "regime_confidence_min": {"start": 0.40, "stop": 0.89, "step": 0.01, "count": 50},
            },
            "local_m4_index_range": [0, 19],
            "local_m4_config_count": 20,
            "local_candidate_count": 80,
            "cloud_m4_config_count": 1250,
            "cloud_candidate_count": 5000,
        },
        {
            "family_name": "family_b_hysteresis_hold",
            "research_purpose": "Entry/exit hysteresis to test hold length versus churn under the same adaptive window policy.",
            "local_live_axes": {
                "entry_threshold": [0.50, 0.56, 0.62, 0.68, 0.74],
                "exit_threshold": [0.18, 0.24, 0.30, 0.36],
            },
            "cloud_live_axes": {
                "entry_threshold": {"start": 0.50, "stop": 0.74, "step": 0.01, "count": 25},
                "exit_threshold": {"start": 0.10, "stop": 0.59, "step": 0.01, "count": 50},
            },
            "local_m4_index_range": [20, 39],
            "local_m4_config_count": 20,
            "local_candidate_count": 80,
            "cloud_m4_config_count": 1250,
            "cloud_candidate_count": 5000,
        },
        {
            "family_name": "family_c_regime_taxonomy",
            "research_purpose": "Regime taxonomy sensitivity across trend-versus-shape classification thresholds.",
            "local_live_axes": {
                "trend_spread_min": [0.03, 0.05, 0.07, 0.09, 0.11],
                "shape_skew_min_abs": [0.20, 0.35, 0.50, 0.65],
            },
            "cloud_live_axes": {
                "trend_spread_min": {"start": 0.02, "stop": 0.26, "step": 0.01, "count": 25},
                "shape_skew_min_abs": {"start": 0.10, "stop": 0.59, "step": 0.01, "count": 50},
            },
            "local_m4_index_range": [40, 59],
            "local_m4_config_count": 20,
            "local_candidate_count": 80,
            "cloud_m4_config_count": 1250,
            "cloud_candidate_count": 5000,
        },
        {
            "family_name": "family_d_window_specialists",
            "research_purpose": "Single-window specialists that force the canonical 5/15/30/60 structural windows and vary concentration caps.",
            "local_live_axes": {
                "fixed_window_index": [0, 1, 2, 3],
                "max_abs_weight": [0.20, 0.35, 0.50, 0.65, 0.80],
            },
            "cloud_live_axes": {
                "fixed_window_index": {"values": [0, 1, 2, 3], "count": 4},
                "max_abs_weight": {"start": 0.20, "stop": 0.80, "step": 0.05, "count": 13},
                "entry_threshold": {"start": 0.50, "stop": 0.74, "step": 0.01, "count": 25},
            },
            "local_m4_index_range": [60, 79],
            "local_m4_config_count": 20,
            "local_candidate_count": 80,
            "cloud_m4_config_count": 1300,
            "cloud_candidate_count": 5200,
        },
    ]
    return {
        "plan_name": "cloud_discovery_20k_plan",
        "source_local_config": "configs/local_discovery_short_7core.yaml",
        "live_axes": LIVE_AXES,
        "dead_axes": DEAD_AXES,
        "uncertain_axes": UNCERTAIN_AXES,
        "chosen_module3_block_windows": MODULE3_WINDOWS,
        "canonical_internal_module4_windows": CANONICAL_MODULE4_WINDOWS,
        "local_short_run": {
            "config_path": "configs/local_discovery_short_7core.yaml",
            "module2_window_count": 1,
            "module3_window_count": 4,
            "total_m4_configs": 80,
            "total_candidates": 320,
            "family_entries": family_entries,
            "candidate_count_per_block_window": 80,
        },
        "later_cloud_run_20k": {
            "plan_path": "configs/cloud_discovery_20k_plan.yaml",
            "module2_window_count": 1,
            "module3_window_count": 4,
            "total_m4_configs": 5050,
            "total_candidates": 20200,
            "candidate_count_per_block_window": 5050,
            "family_entries": family_entries,
        },
    }


def build_local_adaptive_run_config() -> dict[str, object]:
    module4_configs = (
        _adaptive_family_a_configs()
        + _adaptive_family_b_configs()
        + _adaptive_family_c_configs()
        + _adaptive_family_d_configs()
    )
    cfg = build_local_short_run_config()
    cfg["run_name"] = "local_adaptive_discovery_7core"
    cfg["module4_configs"] = module4_configs
    cfg["harness"]["research_mode"] = "discovery"
    cfg["harness"]["report_dir"] = "./artifacts"
    return cfg


def build_local_adaptive_plan() -> dict[str, object]:
    family_entries = [
        {
            "family_name": "family_a_activation_frontier",
            "research_purpose": "Expand the entry/regime activation frontier under adaptive multi-window selection.",
            "local_live_axes": {
                "entry_threshold": [0.48, 0.54, 0.60, 0.66, 0.72, 0.78],
                "regime_confidence_min": [0.40, 0.52, 0.64, 0.76],
            },
            "local_m4_index_range": [0, 23],
            "local_m4_config_count": 24,
            "local_candidate_count": 96,
        },
        {
            "family_name": "family_b_hysteresis_persistence",
            "research_purpose": "Test hold-vs-churn persistence with entry/exit hysteresis and conviction scaling.",
            "local_live_axes": {
                "entry_threshold": [0.50, 0.58, 0.66, 0.74],
                "exit_threshold": [0.10, 0.20, 0.30],
                "conviction_scale": [0.85, 1.15],
            },
            "local_m4_index_range": [24, 47],
            "local_m4_config_count": 24,
            "local_candidate_count": 96,
        },
        {
            "family_name": "family_c_regime_classification",
            "research_purpose": "Probe regime taxonomy sensitivity by moving trend and shape thresholds with confidence filtering.",
            "local_live_axes": {
                "trend_spread_min": [0.03, 0.05, 0.07, 0.09],
                "shape_skew_min_abs": [0.20, 0.35, 0.50],
                "regime_confidence_min": [0.50, 0.68],
            },
            "local_m4_index_range": [48, 71],
            "local_m4_config_count": 24,
            "local_candidate_count": 96,
        },
        {
            "family_name": "family_d_window_scale_specialists",
            "research_purpose": "Force single-window specialists and vary allocation concentration and conviction clipping.",
            "local_live_axes": {
                "fixed_window_index": [0, 1, 2, 3],
                "max_abs_weight": [0.25, 0.50, 0.75],
                "conviction_clip": [0.75, 1.25],
            },
            "local_m4_index_range": [72, 95],
            "local_m4_config_count": 24,
            "local_candidate_count": 96,
        },
    ]
    return {
        "plan_name": "local_adaptive_discovery_7core_plan",
        "source_config": "configs/local_adaptive_discovery_7core.yaml",
        "research_mode": "discovery",
        "live_axes": LIVE_AXES,
        "dead_axes": DEAD_AXES,
        "uncertain_axes": UNCERTAIN_AXES,
        "module3_block_windows": MODULE3_WINDOWS,
        "canonical_internal_module4_windows": CANONICAL_MODULE4_WINDOWS,
        "adaptive_local_run": {
            "config_path": "configs/local_adaptive_discovery_7core.yaml",
            "module2_window_count": 1,
            "module3_window_count": 4,
            "total_m4_configs": 96,
            "total_candidates": 384,
            "candidate_count_per_block_window": 96,
            "family_entries": family_entries,
        },
        "discovery_mode_policy": {
            "hard_fatal": [
                "non-finite active-domain outputs",
                "shape or contract mismatches",
                "impossible state failures",
                "worker contract failures",
                "risk-engine contract failures",
                "canonical execution authority violations",
            ],
            "soft_flag": [
                "localized invariant reason codes with completed evaluation",
                "fragile execution flags",
                "candidate-level execution warnings that preserve runtime trust",
            ],
            "analysis_only": [
                "threshold rejection policy",
                "pass/fail policy thresholds for acceptance",
                "execution robustness gating as a research filter",
            ],
        },
        "report_artifact": {
            "path": "research_distribution_report.json",
            "required_fields": [
                "sharpe_distribution",
                "return_distribution",
                "drawdown_distribution",
                "cluster_analysis",
                "top_5_percent_candidate_metrics",
                "family_level_summary",
                "window_level_summary",
                "standard_reject_counts",
                "standard_pass_counts",
                "discovery_included_candidates",
                "effective_return_signature_count",
                "distinct_robustness_score_count",
                "distinct_execution_robustness_count",
            ],
        },
    }


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def main() -> None:
    _write_yaml(LOCAL_CONFIG_PATH, build_local_short_run_config())
    _write_yaml(CLOUD_PLAN_PATH, build_cloud_plan())
    _write_yaml(LOCAL_ADAPTIVE_CONFIG_PATH, build_local_adaptive_run_config())
    _write_yaml(LOCAL_ADAPTIVE_PLAN_PATH, build_local_adaptive_plan())


if __name__ == "__main__":
    main()
