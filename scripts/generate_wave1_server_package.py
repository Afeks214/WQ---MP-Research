#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from weightiz.module4.wave1_parity import evaluate_f6_parity


ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = ROOT / "configs" / "server"
FAMILY_DIR = ROOT / "configs" / "families"
DEFAULT_PARITY_INPUT = SERVER_DIR / "wave1_f6_parity_input.npz"


def _portable_repo_path(path: Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:  # pragma: no cover - defensive portability fallback
        return str(p)


def _missing_f6_parity_report(*, parity_input_path: Path, reason: str) -> dict[str, Any]:
    return {
        "passed": False,
        "numeric_p95_abs_err": None,
        "numeric_max_abs_err": None,
        "behavioral_mismatch_rate": None,
        "branch_mismatch_rate": None,
        "firing_mismatch_rate": None,
        "collision_mismatch_rate": None,
        "deadzone_violation_rate": None,
        "total_points": 0,
        "numeric_tol": 1.0e-4,
        "behavioral_tol": 0.02,
        "strict_behavior_checked": False,
        "reason": str(reason),
        "method": "canonical_sample_required_v1",
        "parity_input_path": _portable_repo_path(parity_input_path),
    }


def _canonical_f6_parity_report(*, parity_input_path: Path) -> dict[str, Any]:
    if not parity_input_path.exists():
        return _missing_f6_parity_report(
            parity_input_path=parity_input_path,
            reason="NO_CANONICAL_PARITY_INPUT",
        )
    try:
        with np.load(parity_input_path) as data:
            required = (
                "dclip",
                "rvol",
                "gbreak",
                "score_bo_long",
                "score_bo_short",
                "d_value",
                "score_reject",
                "greject",
                "z_delta",
                "x_vah",
                "x_val",
            )
            missing = [k for k in required if k not in data]
            if missing:
                return _missing_f6_parity_report(
                    parity_input_path=parity_input_path,
                    reason=f"MISSING_KEYS:{','.join(missing)}",
                )
            report = evaluate_f6_parity(
                dclip=np.asarray(data["dclip"], dtype=np.float64),
                rvol=np.asarray(data["rvol"], dtype=np.float64),
                gbreak=np.asarray(data["gbreak"], dtype=np.float64),
                score_bo_long=np.asarray(data["score_bo_long"], dtype=np.float64),
                score_bo_short=np.asarray(data["score_bo_short"], dtype=np.float64),
                d_value=np.asarray(data["d_value"], dtype=np.float64),
                score_reject=np.asarray(data["score_reject"], dtype=np.float64),
                greject=np.asarray(data["greject"], dtype=np.float64),
                z_delta=np.asarray(data["z_delta"], dtype=np.float64),
                x_vah=np.asarray(data["x_vah"], dtype=np.float64),
                x_val=np.asarray(data["x_val"], dtype=np.float64),
                numeric_tol=1.0e-4,
                behavioral_tol=0.02,
                require_strict_behavior=True,
            )
    except Exception as exc:  # pragma: no cover - defensive fail-closed guard
        return _missing_f6_parity_report(
            parity_input_path=parity_input_path,
            reason=f"PARITY_INPUT_LOAD_ERROR:{type(exc).__name__}",
        )

    min_points = 100
    passed = bool(
        report.passed
        and bool(report.strict_behavior_checked)
        and (int(report.total_points) >= min_points)
    )
    return {
        "passed": passed,
        "numeric_p95_abs_err": float(report.numeric_p95_abs_err),
        "numeric_max_abs_err": float(report.numeric_max_abs_err),
        "behavioral_mismatch_rate": float(report.behavioral_mismatch_rate),
        "branch_mismatch_rate": float(report.branch_mismatch_rate),
        "firing_mismatch_rate": float(report.firing_mismatch_rate),
        "collision_mismatch_rate": float(report.collision_mismatch_rate),
        "deadzone_violation_rate": float(report.deadzone_violation_rate),
        "total_points": int(report.total_points),
        "numeric_tol": float(report.numeric_tol),
        "behavioral_tol": float(report.behavioral_tol),
        "strict_behavior_checked": bool(report.strict_behavior_checked),
        "reason": str(report.reason if int(report.total_points) >= min_points else "INSUFFICIENT_PARITY_POINTS"),
        "method": "canonical_sample_v1",
        "parity_input_path": _portable_repo_path(parity_input_path),
        "min_points_required": int(min_points),
    }


def _family_module4_configs(*, f6_enabled: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    f5_regime = [0.55, 0.60, 0.65, 0.70]
    f5_cap = [0.15, 0.20, 0.25]
    f5_conv = [0.75, 1.00, 1.25, 1.50]
    for regime_conf, cap, conv in itertools.product(f5_regime, f5_cap, f5_conv):
        out.append(
            {
                "strategy_type": "institutional_wave1",
                "family_id": "F5",
                "entry_threshold": 0.55,
                "exit_threshold": 0.25,
                "regime_confidence_min": float(regime_conf),
                "max_asset_cap_frac": float(cap),
                "conviction_scale": float(conv),
                "enable_f5_close_overlay": True,
                "f6_enabled": bool(f6_enabled),
                "f6_parity_required": True,
            }
        )
    assert len(f5_regime) * len(f5_cap) * len(f5_conv) == 48

    f3_regime = [0.55, 0.60, 0.65]
    f3_cap = [0.15, 0.20, 0.25]
    f3_conv = [0.75, 1.00, 1.25, 1.50]
    for regime_conf, cap, conv in itertools.product(f3_regime, f3_cap, f3_conv):
        out.append(
            {
                "strategy_type": "institutional_wave1",
                "family_id": "F3",
                "entry_threshold": 0.55,
                "exit_threshold": 0.25,
                "regime_confidence_min": float(regime_conf),
                "max_asset_cap_frac": float(cap),
                "conviction_scale": float(conv),
                "enable_f5_close_overlay": True,
                "f6_enabled": bool(f6_enabled),
                "f6_parity_required": True,
            }
        )
    assert len(f3_regime) * len(f3_cap) * len(f3_conv) == 36

    if f6_enabled:
        f6_regime = [0.55, 0.60, 0.65]
        f6_cap = [0.10, 0.15]
        f6_conv = [0.75, 1.00, 1.25, 1.50]
        for regime_conf, cap, conv in itertools.product(f6_regime, f6_cap, f6_conv):
            out.append(
                {
                    "strategy_type": "institutional_wave1",
                    "family_id": "F6",
                    "entry_threshold": 0.55,
                    "exit_threshold": 0.25,
                    "regime_confidence_min": float(regime_conf),
                    "max_asset_cap_frac": float(cap),
                    "conviction_scale": float(conv),
                    "enable_f5_close_overlay": True,
                    "f6_enabled": True,
                    "f6_parity_required": True,
                }
            )
        assert len(f6_regime) * len(f6_cap) * len(f6_conv) == 24
    return out


def _run_config(*, f6_enabled: bool, parity_report_path: Path) -> dict[str, Any]:
    module4_configs = _family_module4_configs(f6_enabled=f6_enabled)
    return {
        "run_name": "wave1_server_autopsy_v1",
        "symbols": ["SPY", "QQQ", "GLD", "EEM", "HYG", "IWM", "XLU", "XLK", "XLE", "TLT"],
        "data": {
            "root": str(os.environ.get("WEIGHTIZ_DATA_ROOT", "./data/alpaca/clean")),
            "format": "parquet",
            "start": "2024-09-10T00:00:00Z",
            "end": "2026-02-28T23:59:59Z",
            "timestamp_column": "timestamp",
        },
        "engine": {
            "mode": "research",
            "B": 240,
            "warmup_minutes": 15,
            "flat_time_minute": 945,
            "gap_reset_minutes": 5.0,
            "seed": 17,
            "fail_on_nan": True,
            "initial_cash": 1_000_000.0,
            "tick_size_default": 0.01,
        },
        "module2_configs": [{"profile_window_bars": 60}],
        "module3_configs": [{"block_minutes": 30}],
        "module4_configs": module4_configs,
        "harness": {
            "seed": 97,
            "timezone": "America/New_York",
            "parallel_backend": "process_pool",
            "parallel_workers": 48,
            "report_dir": "./artifacts/module5_harness_wave1",
            "wf_train_sessions": 60,
            "wf_test_sessions": 20,
            "wf_step_sessions": 20,
            "cpcv_slices": 10,
            "cpcv_k_test": 5,
            "research_mode": "discovery",
        },
        "module6": {
            "intake": {"run_policy_class": "representative_discovery", "min_availability_ratio": 0.5, "min_observed_sessions": 30},
            "scoring": {"min_cross_universe_support": 0.8},
            "export": {"output_subdir_name": "module6_wave1"},
        },
        "zimtra_sweep": {
            "enabled": True,
            "deterministic_required": True,
            "wave1_truth": {
                "families_keep": ["F5", "F3"] + (["F6"] if f6_enabled else []),
                "f7_overlay_only": True,
                "f6_parity_report": _portable_repo_path(parity_report_path),
                "candidate_budget_f5": 48,
                "candidate_budget_f3": 36,
                "candidate_budget_f6": 24 if f6_enabled else 0,
            },
        },
    }


def _family_policy(*, f6_enabled: bool, parity_report_path: Path) -> dict[str, Any]:
    return {
        "version": "wave1_locked_v1",
        "family_truth": {
            "standalone_families": ["F5", "F3"] + (["F6"] if f6_enabled else []),
            "overlay_only": ["F7_as_F5_close_extension"],
            "deferred": ["F8", "F9", "F10"],
            "rejected": ["F4"],
        },
        "deterministic_ladder": [
            "global_disqualifiers",
            "breakout_branch_detection",
            "F5_vs_F6_trichotomy",
            "dead_zone_no_trade",
            "responsive_F3_path",
            "F3_vs_F5_collision_resolver",
            "F3_vs_F6_collision_resolver",
            "F5_close_window_overlay_extension",
        ],
        "dead_zone": {"gbreak_low": 0.40, "gbreak_high": 0.55},
        "f6_gate": {
            "required": True,
            "enabled": bool(f6_enabled),
            "report_path": _portable_repo_path(parity_report_path),
        },
        "campaign": {
            "objective": "family-autopsy-first server research run",
            "date_window": ["2024-09-10", "2026-02-28"],
            "assets": ["SPY", "QQQ", "GLD", "EEM", "HYG", "IWM", "XLU", "XLK", "XLE", "TLT"],
            "candidate_budgets": {"F5": 48, "F3": 36, "F6": 24 if f6_enabled else 0},
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Wave-1 server campaign package.")
    parser.add_argument(
        "--parity-input",
        type=Path,
        default=DEFAULT_PARITY_INPUT,
        help=(
            "Path to canonical F6 parity sample .npz containing "
            "dclip/rvol/gbreak/score_bo_long/score_bo_short/"
            "d_value/score_reject/greject/z_delta/x_vah/x_val."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    SERVER_DIR.mkdir(parents=True, exist_ok=True)
    FAMILY_DIR.mkdir(parents=True, exist_ok=True)

    parity = _canonical_f6_parity_report(parity_input_path=args.parity_input)

    parity_path = SERVER_DIR / "wave1_f6_parity_report.json"
    parity_path.write_text(json.dumps(parity, indent=2, sort_keys=True), encoding="utf-8")
    f6_enabled = bool(parity["passed"])

    run_payload = _run_config(f6_enabled=f6_enabled, parity_report_path=parity_path)
    run_cfg_path = SERVER_DIR / "wave1_server_campaign.yaml"
    run_cfg_path.write_text(yaml.safe_dump(run_payload, sort_keys=False), encoding="utf-8")

    policy_payload = _family_policy(f6_enabled=f6_enabled, parity_report_path=parity_path)
    policy_path = FAMILY_DIR / "wave1_family_policy.yaml"
    policy_path.write_text(yaml.safe_dump(policy_payload, sort_keys=False), encoding="utf-8")

    manifest = {
        "run_config_path": _portable_repo_path(run_cfg_path),
        "family_policy_path": _portable_repo_path(policy_path),
        "parity_report_path": _portable_repo_path(parity_path),
        "f6_enabled": bool(f6_enabled),
        "candidate_count": int(len(run_payload["module4_configs"])),
        "family_counts": {
            "F5": int(sum(1 for c in run_payload["module4_configs"] if c.get("family_id") == "F5")),
            "F3": int(sum(1 for c in run_payload["module4_configs"] if c.get("family_id") == "F3")),
            "F6": int(sum(1 for c in run_payload["module4_configs"] if c.get("family_id") == "F6")),
            "F7": 0,
        },
    }
    manifest_path = SERVER_DIR / "wave1_server_package_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
