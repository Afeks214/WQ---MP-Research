#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from weightiz.module5.stage_a_discovery import (
    STAGE_A_CAMPAIGN_ID,
    STAGE_A_DATA_END_UTC,
    STAGE_A_DATA_START_UTC,
    STAGE_A_F6_WINDOW_SET,
    STAGE_A_FAMILY_SPECS,
    STAGE_A_LIVE_ENTRY_THRESHOLD,
    STAGE_A_PROCESS_BACKEND,
    STAGE_A_PROCESS_WORKERS,
    STAGE_A_RESEARCH_MODE,
    STAGE_A_RESEARCH_THRESHOLD,
    STAGE_A_RUN_NAME,
    STAGE_A_SYMBOLS,
    STAGE_A_TOTAL_CANDIDATES,
    STAGE_A_WINDOW_INDEX,
    STAGE_A_WINDOW_SET,
    encode_stage_a_tags,
    stable_stage_a_hash,
)

CONFIG_PATH = ROOT / "configs" / "cloud_stage_a_discovery_5000.yaml"
PLAN_PATH = ROOT / "configs" / "cloud_stage_a_discovery_5000_plan.yaml"


BASE_M2: dict[str, Any] = {
    "profile_window_bars": 60,
    "profile_warmup_bars": 60,
    "atr_span": 14,
    "atr_floor_mult_tick": 4.0,
    "rvol_lookback_sessions": 20,
    "rvol_policy": "neutral_one",
    "volume_cap_window_bars": 60,
    "volume_cap_mad_mult": 5.0,
    "volume_cap_min_mult": 0.25,
    "mu1_clv_shift": 0.0,
    "mu2_clv_shift": 0.35,
    "sigma_delta_min": 0.05,
    "va_threshold": 0.70,
    "d_clip": 6.0,
    "break_bias": 1.0,
    "reject_center": 2.0,
    "rvol_trend_cutoff": 2.0,
    "body_trend_cutoff": 0.60,
    "delta_gate_threshold": 1.0,
    "normal_concentration_threshold": 0.05,
    "trend_delta_confirm_z": 1.5,
    "double_dist_min_sep_x": 1.0,
    "double_dist_valley_frac": 0.35,
    "fail_on_non_finite_output": True,
}

BASE_M3: dict[str, Any] = {
    "structural_windows": list(STAGE_A_WINDOW_SET),
    "selected_window": 60,
    "validate_outputs": True,
    "block_minutes": 60,
    "phase_mask": [1, 2],
    "use_rth_minutes_only": True,
    "rth_open_minute": 570,
    "last_minute_inclusive": 945,
    "include_partial_last_block": True,
    "min_block_valid_bars": 12,
    "min_block_valid_ratio": 0.70,
    "ib_pop_frac": 0.01,
    "context_mode": "ffill_last_complete",
    "rolling_context_period": 5,
    "fail_on_non_finite_input": True,
    "fail_on_non_finite_output": True,
    "fail_on_bad_indices": True,
    "fail_on_missing_prev_va": False,
    "eps": 1.0e-12,
}

BASE_M4: dict[str, Any] = {
    "fail_on_non_finite_input": True,
    "fail_on_non_finite_output": True,
    "eps": 1.0e-12,
    "enforce_causal_source_validation": True,
    "enforce_window_causal_sanity": True,
    "window_selection_mode": "single_window",
    "fixed_window_index": 0,
    "anchor_window_index": 0,
    "max_volatility": float("inf"),
    "max_spread": float("inf"),
    "min_liquidity": 0.0,
    "regime_confidence_min": 0.55,
    "entry_threshold": STAGE_A_LIVE_ENTRY_THRESHOLD,
    "exit_threshold": 0.25,
    "conviction_scale": 1.0,
    "conviction_clip": 1.0,
    "max_abs_weight": 1.0,
    "top_k_intraday": 5,
    "max_asset_cap_frac": 0.30,
    "max_turnover_frac_per_bar": 0.35,
    "overnight_min_conviction": 0.65,
    "allow_cash_overnight": True,
    "trend_spread_min": 0.05,
    "trend_poc_drift_min_abs": 0.35,
    "neutral_poc_drift_max_abs": 0.15,
    "shape_skew_min_abs": 0.35,
    "double_dist_sep_x": 1.0,
    "double_dist_valley_frac": 0.35,
    "commission_bps": 0.40,
    "spread_tick_mult": 1.50,
    "slippage_bps_low_rvol": 3.0,
    "slippage_bps_mid_rvol": 2.0,
    "slippage_bps_high_rvol": 1.5,
    "stress_slippage_mult": 1.0,
    "hard_kill_on_daily_loss_breach": True,
    "enable_degraded_bridge_mode": True,
    "execution_strict_prices": True,
}

BASE_HARNESS: dict[str, Any] = {
    "seed": 97,
    "research_mode": STAGE_A_RESEARCH_MODE,
    "timezone": "America/New_York",
    "freq": "1min",
    "min_asset_coverage": 0.80,
    "purge_bars": 60,
    "embargo_bars": 30,
    "wf_train_sessions": 60,
    "wf_test_sessions": 20,
    "wf_step_sessions": 20,
    "cpcv_slices": 10,
    "cpcv_k_test": 5,
    "parallel_backend": STAGE_A_PROCESS_BACKEND,
    "parallel_workers": STAGE_A_PROCESS_WORKERS,
    "stress_profile": "baseline_mild_severe",
    "max_ram_utilization_frac": 0.70,
    "enforce_lookahead_guard": True,
    "report_dir": "./artifacts/module5_harness",
    "fail_on_non_finite": True,
    "daily_return_min_days": 60,
    "benchmark_symbol": "SPY",
    "export_micro_diagnostics": True,
    "micro_diag_mode": "events_only",
    "micro_diag_trade_window_pre": 45,
    "micro_diag_trade_window_post": 120,
    "micro_diag_export_block_profiles": True,
    "micro_diag_export_funnel": True,
    "micro_diag_max_rows": 250000,
    "cluster_corr_threshold": 0.90,
    "cluster_distance_block_size": 256,
    "cluster_distance_in_memory_max_n": 2500,
    "execution_transaction_cost_per_trade": 0.0,
    "execution_slippage_mult": 1.0,
    "execution_extra_slippage_bps": 0.0,
    "execution_latency_bars": 1,
    "regime_vol_window": 60,
    "regime_slope_window": 60,
    "regime_hurst_window": 120,
    "regime_min_obs_per_mask": 20,
    "horizon_minutes": [1, 5, 15, 60],
    "robustness_weight_dsr": 0.20,
    "robustness_weight_pbo": 0.15,
    "robustness_weight_spa": 0.10,
    "robustness_weight_regime": 0.20,
    "robustness_weight_execution": 0.20,
    "robustness_weight_horizon": 0.15,
    "robustness_reject_threshold": STAGE_A_RESEARCH_THRESHOLD,
    "execution_fragile_threshold": 0.50,
}


@dataclass
class ConfigRegistry:
    module2_configs: list[dict[str, Any]]
    module3_configs: list[dict[str, Any]]
    module4_configs: list[dict[str, Any]]

    def __init__(self) -> None:
        self.module2_configs = []
        self.module3_configs = []
        self.module4_configs = []
        self._m2_idx: dict[str, int] = {}
        self._m3_idx: dict[str, int] = {}
        self._m4_idx: dict[str, int] = {}

    def add_m2(self, payload: dict[str, Any]) -> int:
        return self._add(payload, self.module2_configs, self._m2_idx)

    def add_m3(self, payload: dict[str, Any]) -> int:
        return self._add(payload, self.module3_configs, self._m3_idx)

    def add_m4(self, payload: dict[str, Any]) -> int:
        return self._add(payload, self.module4_configs, self._m4_idx)

    @staticmethod
    def _add(payload: dict[str, Any], bucket: list[dict[str, Any]], idx_map: dict[str, int]) -> int:
        key = stable_stage_a_hash(payload)
        if key in idx_map:
            return int(idx_map[key])
        idx = int(len(bucket))
        idx_map[key] = idx
        bucket.append(dict(payload))
        return idx


def _path_map(symbols: tuple[str, ...]) -> dict[str, str]:
    return {str(sym): f"{sym}.parquet" for sym in symbols}


def _window_probe_m4(base_overrides: dict[str, Any], window: int) -> dict[str, Any]:
    out = dict(BASE_M4)
    out.update(base_overrides)
    out["window_selection_mode"] = "single_window"
    out["fixed_window_index"] = int(STAGE_A_WINDOW_INDEX[int(window)])
    out["anchor_window_index"] = int(STAGE_A_WINDOW_INDEX[int(window)])
    return out


def _multi_window_live_m4(base_overrides: dict[str, Any], anchor_window: int) -> dict[str, Any]:
    out = dict(BASE_M4)
    out.update(base_overrides)
    out["window_selection_mode"] = "multi_window"
    out["fixed_window_index"] = int(STAGE_A_WINDOW_INDEX[int(anchor_window)])
    out["anchor_window_index"] = int(STAGE_A_WINDOW_INDEX[int(anchor_window)])
    return out


def _family_meta(
    *,
    family_id: str,
    family_name: str,
    hypothesis_id: str,
    evaluation_role: str,
    parameter_hash: str,
    window_set: tuple[int, ...],
    evaluation_window: int | None = None,
) -> tuple[str, ...]:
    payload: dict[str, Any] = {
        "campaign_id": STAGE_A_CAMPAIGN_ID,
        "family_id": family_id,
        "family_name": family_name,
        "hypothesis_id": hypothesis_id,
        "evaluation_role": evaluation_role,
        "window_set": window_set,
        "parameter_hash": parameter_hash,
    }
    if evaluation_window is not None:
        payload["evaluation_window"] = int(evaluation_window)
    return encode_stage_a_tags(payload)


def _candidate_spec(candidate_id: str, m2_idx: int, m3_idx: int, m4_idx: int, tags: tuple[str, ...]) -> dict[str, Any]:
    return {
        "candidate_id": str(candidate_id),
        "m2_idx": int(m2_idx),
        "m3_idx": int(m3_idx),
        "m4_idx": int(m4_idx),
        "tags": list(tags),
    }


def _build_f1(reg: ConfigRegistry, m2_idx: int, m3_idx: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    reject_center_values = [1.0, 1.4, 1.8, 2.2, 2.6]
    d_clip_values = [3.5, 4.5, 5.5, 6.5]
    va_threshold_values = [0.62, 0.66, 0.70, 0.74, 0.78, 0.82]
    family_name = "acceptance_rejection_geometry"
    reject_to_spread = {1.0: 0.03, 1.4: 0.05, 1.8: 0.07, 2.2: 0.09, 2.6: 0.11}
    dclip_to_drift = {3.5: 0.20, 4.5: 0.30, 5.5: 0.40, 6.5: 0.50}
    va_to_exit = {0.62: 0.16, 0.66: 0.18, 0.70: 0.20, 0.74: 0.22, 0.78: 0.24, 0.82: 0.26}

    h = 0
    for reject_center in reject_center_values:
        for d_clip in d_clip_values:
            for va_threshold in va_threshold_values:
                hypothesis_id = f"F1H{h:03d}"
                axes = {
                    "reject_center": reject_center,
                    "d_clip": d_clip,
                    "va_threshold": va_threshold,
                }
                parameter_hash = stable_stage_a_hash({"family_id": "F1", "axes": axes})
                probe_base = {
                    "entry_threshold": STAGE_A_LIVE_ENTRY_THRESHOLD,
                    "exit_threshold": float(va_to_exit[float(va_threshold)]),
                    "regime_confidence_min": 0.48,
                    "trend_spread_min": float(reject_to_spread[float(reject_center)]),
                    "trend_poc_drift_min_abs": float(dclip_to_drift[float(d_clip)]),
                }
                for window in STAGE_A_WINDOW_SET:
                    probe_idx = reg.add_m4(_window_probe_m4(probe_base, int(window)))
                    tags = _family_meta(
                        family_id="F1",
                        family_name=family_name,
                        hypothesis_id=hypothesis_id,
                        evaluation_role="window_probe",
                        parameter_hash=parameter_hash,
                        window_set=STAGE_A_WINDOW_SET,
                        evaluation_window=int(window),
                    )
                    specs.append(
                        _candidate_spec(
                            candidate_id=f"stagea_f1_h{h:03d}_w{int(window):03d}",
                            m2_idx=m2_idx,
                            m3_idx=m3_idx,
                            m4_idx=probe_idx,
                            tags=tags,
                        )
                    )
                h += 1
    return specs, {
        "family_id": "F1",
        "family_name": family_name,
        "candidate_budget": int(len(specs)),
        "hypothesis_count": int(h),
        "window_set": list(STAGE_A_WINDOW_SET),
        "live_axes": [
            "module4.exit_threshold",
            "module4.trend_spread_min",
            "module4.trend_poc_drift_min_abs",
        ],
    }


def _build_f2(reg: ConfigRegistry, m2_idx: int, m3_idx: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    delta_gate_values = [0.55, 0.75, 0.95, 1.15, 1.35]
    trend_confirm_values = [1.0, 1.3, 1.6, 1.9]
    clv_shift_values = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    family_name = "delta_confirmation_aggression"
    delta_to_exit = {0.55: 0.12, 0.75: 0.14, 0.95: 0.16, 1.15: 0.18, 1.35: 0.20}
    trend_to_conf = {1.0: 0.40, 1.3: 0.48, 1.6: 0.56, 1.9: 0.64}
    clv_to_scale = {0.10: 0.80, 0.20: 0.90, 0.30: 1.00, 0.40: 1.10, 0.50: 1.20, 0.60: 1.30}

    h = 0
    for delta_gate in delta_gate_values:
        for trend_confirm in trend_confirm_values:
            for clv_shift in clv_shift_values:
                hypothesis_id = f"F2H{h:03d}"
                axes = {
                    "delta_gate_threshold": delta_gate,
                    "trend_delta_confirm_z": trend_confirm,
                    "mu2_clv_shift": clv_shift,
                }
                parameter_hash = stable_stage_a_hash({"family_id": "F2", "axes": axes})
                probe_base = {
                    "entry_threshold": STAGE_A_LIVE_ENTRY_THRESHOLD,
                    "exit_threshold": float(delta_to_exit[float(delta_gate)]),
                    "regime_confidence_min": float(trend_to_conf[float(trend_confirm)]),
                    "conviction_scale": float(clv_to_scale[float(clv_shift)]),
                }
                for window in STAGE_A_WINDOW_SET:
                    probe_idx = reg.add_m4(_window_probe_m4(probe_base, int(window)))
                    tags = _family_meta(
                        family_id="F2",
                        family_name=family_name,
                        hypothesis_id=hypothesis_id,
                        evaluation_role="window_probe",
                        parameter_hash=parameter_hash,
                        window_set=STAGE_A_WINDOW_SET,
                        evaluation_window=int(window),
                    )
                    specs.append(
                        _candidate_spec(
                            candidate_id=f"stagea_f2_h{h:03d}_w{int(window):03d}",
                            m2_idx=m2_idx,
                            m3_idx=m3_idx,
                            m4_idx=probe_idx,
                            tags=tags,
                        )
                    )
                h += 1
    return specs, {
        "family_id": "F2",
        "family_name": family_name,
        "candidate_budget": int(len(specs)),
        "hypothesis_count": int(h),
        "window_set": list(STAGE_A_WINDOW_SET),
        "live_axes": [
            "module4.exit_threshold",
            "module4.regime_confidence_min",
            "module4.conviction_scale",
        ],
    }


def _build_f3(reg: ConfigRegistry, m2_idx: int, m3_idx: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    rvol_cutoff_values = [1.2, 1.5, 1.8, 2.1, 2.4]
    body_cutoff_values = [0.35, 0.45, 0.55, 0.65]
    volume_cap_values = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    family_name = "participation_rvol_conviction"
    rvol_to_conf = {1.2: 0.40, 1.5: 0.46, 1.8: 0.52, 2.1: 0.58, 2.4: 0.64}
    body_to_exit = {0.35: 0.14, 0.45: 0.18, 0.55: 0.22, 0.65: 0.26}
    volume_to_scale = {2.5: 0.80, 3.0: 0.90, 3.5: 1.00, 4.0: 1.10, 4.5: 1.20, 5.0: 1.30}

    h = 0
    for rvol_cutoff in rvol_cutoff_values:
        for body_cutoff in body_cutoff_values:
            for volume_cap in volume_cap_values:
                hypothesis_id = f"F3H{h:03d}"
                axes = {
                    "rvol_trend_cutoff": rvol_cutoff,
                    "body_trend_cutoff": body_cutoff,
                    "volume_cap_mad_mult": volume_cap,
                }
                parameter_hash = stable_stage_a_hash({"family_id": "F3", "axes": axes})
                probe_base = {
                    "entry_threshold": STAGE_A_LIVE_ENTRY_THRESHOLD,
                    "exit_threshold": float(body_to_exit[float(body_cutoff)]),
                    "regime_confidence_min": float(rvol_to_conf[float(rvol_cutoff)]),
                    "conviction_scale": float(volume_to_scale[float(volume_cap)]),
                }
                for window in STAGE_A_WINDOW_SET:
                    probe_idx = reg.add_m4(_window_probe_m4(probe_base, int(window)))
                    tags = _family_meta(
                        family_id="F3",
                        family_name=family_name,
                        hypothesis_id=hypothesis_id,
                        evaluation_role="window_probe",
                        parameter_hash=parameter_hash,
                        window_set=STAGE_A_WINDOW_SET,
                        evaluation_window=int(window),
                    )
                    specs.append(
                        _candidate_spec(
                            candidate_id=f"stagea_f3_h{h:03d}_w{int(window):03d}",
                            m2_idx=m2_idx,
                            m3_idx=m3_idx,
                            m4_idx=probe_idx,
                            tags=tags,
                        )
                    )
                h += 1
    return specs, {
        "family_id": "F3",
        "family_name": family_name,
        "candidate_budget": int(len(specs)),
        "hypothesis_count": int(h),
        "window_set": list(STAGE_A_WINDOW_SET),
        "live_axes": [
            "module4.exit_threshold",
            "module4.regime_confidence_min",
            "module4.conviction_scale",
        ],
    }


def _build_f4(reg: ConfigRegistry, m2_idx: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    ib_pop_values = [0.004, 0.006, 0.008, 0.010, 0.012]
    valid_ratio_values = [0.55, 0.62, 0.69, 0.76]
    rolling_values = [3, 4, 5, 6, 7, 8]
    family_name = "session_state_initial_balance"
    probe_base = {"entry_threshold": STAGE_A_LIVE_ENTRY_THRESHOLD, "exit_threshold": 0.25, "regime_confidence_min": 0.46}
    probe_idx = {int(w): reg.add_m4(_window_probe_m4(probe_base, int(w))) for w in STAGE_A_WINDOW_SET}

    h = 0
    for ib_pop in ib_pop_values:
        for valid_ratio in valid_ratio_values:
            for rolling_period in rolling_values:
                hypothesis_id = f"F4H{h:03d}"
                axes = {
                    "ib_pop_frac": ib_pop,
                    "min_block_valid_ratio": valid_ratio,
                    "rolling_context_period": rolling_period,
                }
                parameter_hash = stable_stage_a_hash({"family_id": "F4", "axes": axes})
                m3_idx = reg.add_m3(
                    {
                        **BASE_M3,
                        "ib_pop_frac": float(ib_pop),
                        "min_block_valid_ratio": float(valid_ratio),
                        "rolling_context_period": int(rolling_period),
                    }
                )
                for window in STAGE_A_WINDOW_SET:
                    tags = _family_meta(
                        family_id="F4",
                        family_name=family_name,
                        hypothesis_id=hypothesis_id,
                        evaluation_role="window_probe",
                        parameter_hash=parameter_hash,
                        window_set=STAGE_A_WINDOW_SET,
                        evaluation_window=int(window),
                    )
                    specs.append(
                        _candidate_spec(
                            candidate_id=f"stagea_f4_h{h:03d}_w{int(window):03d}",
                            m2_idx=m2_idx,
                            m3_idx=m3_idx,
                            m4_idx=probe_idx[int(window)],
                            tags=tags,
                        )
                    )
                h += 1
    return specs, {
        "family_id": "F4",
        "family_name": family_name,
        "candidate_budget": int(len(specs)),
        "hypothesis_count": int(h),
        "window_set": list(STAGE_A_WINDOW_SET),
        "live_axes": [
            "module3.ib_pop_frac",
            "module3.min_block_valid_ratio",
            "module3.rolling_context_period",
        ],
    }


def _build_f5(reg: ConfigRegistry, m2_idx: int, m3_idx: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    anchor_windows = [15, 60, 240]
    regime_conf_values = [0.40, 0.48, 0.56, 0.64, 0.72]
    conviction_scale_values = [0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35]
    family_name = "multi_scale_alignment"

    h = 0
    for anchor_window in anchor_windows:
        for regime_conf in regime_conf_values:
            for conviction_scale in conviction_scale_values:
                hypothesis_id = f"F5H{h:03d}"
                axes = {
                    "anchor_window": anchor_window,
                    "regime_confidence_min": regime_conf,
                    "conviction_scale": conviction_scale,
                }
                parameter_hash = stable_stage_a_hash({"family_id": "F5", "axes": axes})
                live_base = {
                    "entry_threshold": STAGE_A_LIVE_ENTRY_THRESHOLD,
                    "exit_threshold": 0.22,
                    "regime_confidence_min": float(regime_conf),
                    "conviction_scale": float(conviction_scale),
                    "conviction_clip": 1.0,
                }
                live_idx = reg.add_m4(_multi_window_live_m4(live_base, int(anchor_window)))
                live_tags = _family_meta(
                    family_id="F5",
                    family_name=family_name,
                    hypothesis_id=hypothesis_id,
                    evaluation_role="multi_window_live",
                    parameter_hash=parameter_hash,
                    window_set=STAGE_A_WINDOW_SET,
                )
                specs.append(
                    _candidate_spec(
                        candidate_id=f"stagea_f5_h{h:03d}_live",
                        m2_idx=m2_idx,
                        m3_idx=m3_idx,
                        m4_idx=live_idx,
                        tags=live_tags,
                    )
                )
                for window in STAGE_A_WINDOW_SET:
                    probe_idx = reg.add_m4(_window_probe_m4(live_base, int(window)))
                    tags = _family_meta(
                        family_id="F5",
                        family_name=family_name,
                        hypothesis_id=hypothesis_id,
                        evaluation_role="window_probe",
                        parameter_hash=parameter_hash,
                        window_set=STAGE_A_WINDOW_SET,
                        evaluation_window=int(window),
                    )
                    specs.append(
                        _candidate_spec(
                            candidate_id=f"stagea_f5_h{h:03d}_w{int(window):03d}",
                            m2_idx=m2_idx,
                            m3_idx=m3_idx,
                            m4_idx=probe_idx,
                            tags=tags,
                        )
                    )
                h += 1
    return specs, {
        "family_id": "F5",
        "family_name": family_name,
        "candidate_budget": int(len(specs)),
        "hypothesis_count": int(h),
        "window_set": list(STAGE_A_WINDOW_SET),
        "live_axes": [
            "module4.anchor_window_index",
            "module4.regime_confidence_min",
            "module4.conviction_scale",
        ],
        "live_role": "probe_plus_multi_window_live",
    }


def _build_f6(reg: ConfigRegistry, m3_idx: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    regime_conf_values = [0.42, 0.55]
    shape_skew_values = [0.20, 0.30, 0.40, 0.50, 0.60]
    double_sep_values = [0.75, 1.00, 1.25, 1.50, 1.75]
    valley_frac_values = [0.20, 0.30, 0.40, 0.50]
    family_name = "shape_fingerprint_regimes"
    shared_m2_idx = reg.add_m2(BASE_M2)

    h = 0
    for regime_conf in regime_conf_values:
        for shape_skew in shape_skew_values:
            for double_sep in double_sep_values:
                for valley_frac in valley_frac_values:
                    hypothesis_id = f"F6H{h:03d}"
                    axes = {
                        "regime_confidence_min": regime_conf,
                        "shape_skew_min_abs": shape_skew,
                        "double_dist_sep_x": double_sep,
                        "double_dist_valley_frac": valley_frac,
                    }
                    parameter_hash = stable_stage_a_hash({"family_id": "F6", "axes": axes})
                    probe_base = {
                        "entry_threshold": STAGE_A_LIVE_ENTRY_THRESHOLD,
                        "exit_threshold": 0.24,
                        "regime_confidence_min": float(regime_conf),
                        "shape_skew_min_abs": float(shape_skew),
                        "double_dist_sep_x": float(double_sep),
                        "double_dist_valley_frac": float(valley_frac),
                    }
                    for window in STAGE_A_F6_WINDOW_SET:
                        probe_idx = reg.add_m4(_window_probe_m4(probe_base, int(window)))
                        tags = _family_meta(
                            family_id="F6",
                            family_name=family_name,
                            hypothesis_id=hypothesis_id,
                            evaluation_role="window_probe",
                            parameter_hash=parameter_hash,
                            window_set=STAGE_A_F6_WINDOW_SET,
                            evaluation_window=int(window),
                        )
                        specs.append(
                            _candidate_spec(
                                candidate_id=f"stagea_f6_h{h:03d}_w{int(window):03d}",
                                m2_idx=shared_m2_idx,
                                m3_idx=m3_idx,
                                m4_idx=probe_idx,
                                tags=tags,
                            )
                        )
                    h += 1
    return specs, {
        "family_id": "F6",
        "family_name": family_name,
        "candidate_budget": int(len(specs)),
        "hypothesis_count": int(h),
        "window_set": list(STAGE_A_F6_WINDOW_SET),
        "live_axes": [
            "module4.shape_skew_min_abs",
            "module4.double_dist_sep_x",
            "module4.double_dist_valley_frac",
            "module4.regime_confidence_min",
        ],
        "restricted_window_subset_justification": (
            "Shape hypotheses are restricted to 30/60/90/240 minute structure because the shortest "
            "windows do not carry enough stable shape information."
        ),
    }


def build_stage_a_cloud_config() -> tuple[dict[str, Any], dict[str, Any]]:
    reg = ConfigRegistry()
    shared_m2_idx = reg.add_m2(BASE_M2)
    shared_m3_idx = reg.add_m3(BASE_M3)

    candidate_specs: list[dict[str, Any]] = []
    family_entries: list[dict[str, Any]] = []

    family_specs = {spec.family_id: spec for spec in STAGE_A_FAMILY_SPECS}

    for builder in (_build_f1, _build_f2, _build_f3):
        specs, entry = builder(reg, shared_m2_idx, shared_m3_idx)
        candidate_specs.extend(specs)
        entry.update(
            {
                "hypothesis": family_specs[entry["family_id"]].hypothesis,
                "live_role": family_specs[entry["family_id"]].live_role,
            }
        )
        family_entries.append(entry)

    f4_specs, f4_entry = _build_f4(reg, shared_m2_idx)
    candidate_specs.extend(f4_specs)
    f4_entry.update(
        {
            "hypothesis": family_specs["F4"].hypothesis,
            "live_role": family_specs["F4"].live_role,
        }
    )
    family_entries.append(f4_entry)

    f5_specs, f5_entry = _build_f5(reg, shared_m2_idx, shared_m3_idx)
    candidate_specs.extend(f5_specs)
    f5_entry.update(
        {
            "hypothesis": family_specs["F5"].hypothesis,
            "live_role": family_specs["F5"].live_role,
        }
    )
    family_entries.append(f5_entry)

    f6_specs, f6_entry = _build_f6(reg, shared_m3_idx)
    candidate_specs.extend(f6_specs)
    f6_entry.update(
        {
            "hypothesis": family_specs["F6"].hypothesis,
            "live_role": family_specs["F6"].live_role,
        }
    )
    family_entries.append(f6_entry)

    if len(candidate_specs) != STAGE_A_TOTAL_CANDIDATES:
        raise RuntimeError(
            f"Stage A candidate count mismatch: got {len(candidate_specs)}, expected {STAGE_A_TOTAL_CANDIDATES}"
        )

    config = {
        "run_name": STAGE_A_RUN_NAME,
        "symbols": list(STAGE_A_SYMBOLS),
        "data": {
            "root": "./data/alpaca/clean",
            "format": "parquet",
            "path_by_symbol": _path_map(STAGE_A_SYMBOLS),
            "timestamp_column": "timestamp",
            "start": STAGE_A_DATA_START_UTC,
            "end": STAGE_A_DATA_END_UTC,
        },
        "engine": {
            "mode": "research",
            "B": 240,
            "x_min": -6.0,
            "dx": 0.05,
            "rth_open_minute": 570,
            "warmup_minutes": 15,
            "flat_time_minute": 945,
            "gap_reset_minutes": 5.0,
            "initial_cash": 1000000.0,
            "intraday_leverage_max": 6.0,
            "overnight_leverage": 2.0,
            "overnight_positions_max": 1,
            "daily_loss_limit_abs": 50000.0,
            "seed": 17,
            "fail_on_nan": True,
            "tick_size_default": 0.01,
        },
        "module2_configs": reg.module2_configs,
        "module3_configs": reg.module3_configs,
        "module4_configs": reg.module4_configs,
        "harness": dict(BASE_HARNESS),
        "search": {
            "seed": 97,
            "method": "uniform",
            "elite_pct": 0.10,
            "target_evals": STAGE_A_TOTAL_CANDIDATES,
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
            "mode": "manual",
            "specs": candidate_specs,
        },
    }

    plan = {
        "plan_name": "cloud_stage_a_discovery_5000_plan",
        "campaign_id": STAGE_A_CAMPAIGN_ID,
        "config_path": str(CONFIG_PATH.relative_to(ROOT)),
        "run_name": STAGE_A_RUN_NAME,
        "research_mode": STAGE_A_RESEARCH_MODE,
        "live_gate_threshold_ownership": "module4.entry_threshold",
        "live_gate_threshold": STAGE_A_LIVE_ENTRY_THRESHOLD,
        "research_threshold_ownership": "harness.robustness_reject_threshold",
        "research_threshold": STAGE_A_RESEARCH_THRESHOLD,
        "data_horizon": {
            "months": 18,
            "start_utc": STAGE_A_DATA_START_UTC,
            "end_utc": STAGE_A_DATA_END_UTC,
        },
        "execution": {
            "backend": STAGE_A_PROCESS_BACKEND,
            "workers": STAGE_A_PROCESS_WORKERS,
            "deterministic_runtime": True,
            "bounded_artifact_writing": True,
        },
        "structural_window_set": list(STAGE_A_WINDOW_SET),
        "total_candidates": STAGE_A_TOTAL_CANDIDATES,
        "module2_config_count": int(len(reg.module2_configs)),
        "module3_config_count": int(len(reg.module3_configs)),
        "module4_config_count": int(len(reg.module4_configs)),
        "family_entries": family_entries,
        "required_artifacts": [
            "strategy_results.parquet",
            "trade_log.parquet",
            "micro_diagnostics.parquet",
            "micro_profile_blocks.parquet",
            "run_manifest.json",
            "run_summary.json",
            "research_distribution_report.json",
        ],
    }
    return config, plan


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def main() -> None:
    config, plan = build_stage_a_cloud_config()
    _write_yaml(CONFIG_PATH, config)
    _write_yaml(PLAN_PATH, plan)


if __name__ == "__main__":
    main()
