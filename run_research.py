#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional
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
from app.config_models import (
    CandidateSpecModel,
    CandidatesModel,
    DataConfigModel,
    EngineConfigModel,
    HarnessConfigModel,
    Module2ConfigModel,
    Module3ConfigModel,
    Module4ConfigModel,
    RunConfigModel,
    SearchConfigModel,
    StressScenarioModel,
)
from app.config_builders import (
    build_candidates as _cfg_build_candidates,
    build_engine_config as _cfg_build_engine_config,
    build_harness_config as _cfg_build_harness_config,
    build_module2_configs as _cfg_build_module2_configs,
    build_module3_configs as _cfg_build_module3_configs,
    build_module4_configs as _cfg_build_module4_configs,
    build_stress_scenarios as _cfg_build_stress_scenarios,
    resolve_tick_size as _cfg_resolve_tick_size,
)
from app.runtime_support import (
    append_run_registry as _runtime_append_run_registry,
    ensure_dashboard_handoff as _runtime_ensure_dashboard_handoff,
    resolved_config_sha256 as _runtime_resolved_config_sha256,
)
from app.data_resolution import (
    find_col as _data_find_col,
    in_memory_date_filter_loader as _data_in_memory_date_filter_loader,
    require_pandas as _data_require_pandas,
    resolve_data_paths as _data_resolve_data_paths,
)
from app.stage_a_discovery import parse_stage_a_window_set
from module6 import Module6Config, run_module6_portfolio_research
from module5.harness.artifact_writers import write_json as _artifact_write_json


def _require_pandas() -> Any:
    return _data_require_pandas()


def _resolve_tick_size(cfg: RunConfigModel) -> np.ndarray:
    return _cfg_resolve_tick_size(cfg)


def _build_engine_config(cfg: RunConfigModel) -> EngineConfig:
    return _cfg_build_engine_config(cfg)


def _build_module2_configs(cfg: RunConfigModel) -> list[Module2Config]:
    return _cfg_build_module2_configs(cfg)


def _build_module3_configs(cfg: RunConfigModel) -> list[Module3Config]:
    return _cfg_build_module3_configs(cfg)


def _build_module4_configs(cfg: RunConfigModel) -> list[Module4Config]:
    return _cfg_build_module4_configs(cfg)


def _build_harness_config(cfg: RunConfigModel, project_root: Path) -> Module5HarnessConfig:
    return _cfg_build_harness_config(cfg, project_root)


def _resolve_data_paths(cfg: RunConfigModel, project_root: Path) -> list[str]:
    return _data_resolve_data_paths(cfg, project_root)


def _find_col(df: Any, candidates: tuple[str, ...], name: str) -> str:
    return _data_find_col(df, candidates, name)


def in_memory_date_filter_loader(data_cfg: DataConfigModel) -> Callable[[str, str], Any]:
    return _data_in_memory_date_filter_loader(data_cfg)


def _build_stress_scenarios(cfg: RunConfigModel) -> Optional[list[StressScenario]]:
    return _cfg_build_stress_scenarios(cfg)


def _build_candidates(cfg: RunConfigModel) -> Optional[list[CandidateSpec]]:
    return _cfg_build_candidates(cfg)


def _append_run_registry(
    artifacts_root: Path,
    run_id: str,
    run_dir: Path,
    symbols: list[str],
    n_candidates: int,
    pass_count: int,
    resolved_config_sha256: str,
) -> None:
    return _runtime_append_run_registry(
        artifacts_root=artifacts_root,
        run_id=run_id,
        run_dir=run_dir,
        symbols=symbols,
        n_candidates=n_candidates,
        pass_count=pass_count,
        resolved_config_sha256=resolved_config_sha256,
    )


def _resolved_config_sha256(cfg: RunConfigModel) -> str:
    return _runtime_resolved_config_sha256(cfg)


def _ensure_dashboard_handoff(artifacts_root: Path, run_dir: Path) -> Path:
    return _runtime_ensure_dashboard_handoff(artifacts_root, run_dir)


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


def _distribution_summary(values: Any) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size <= 0:
        return {
            "count": int(arr.size),
            "finite_count": 0,
            "available": False,
            "note": "no finite values",
        }
    return {
        "count": int(arr.size),
        "finite_count": int(finite.size),
        "available": True,
        "min": float(np.min(finite)),
        "p05": float(np.percentile(finite, 5)),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def _load_plan_doc(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw if isinstance(raw, dict) else None


def _extract_family_entries(plan_doc: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(plan_doc, dict):
        return []
    for key in ("adaptive_local_run", "local_short_run", "later_cloud_run_20k"):
        section = plan_doc.get(key)
        if isinstance(section, dict):
            entries = section.get("family_entries")
            if isinstance(entries, list):
                return [dict(x) for x in entries if isinstance(x, dict)]
    return []


def _family_name_from_m4_idx(m4_idx: Any, family_entries: list[dict[str, Any]]) -> str:
    try:
        idx = int(m4_idx)
    except Exception:
        return "unknown"
    for entry in family_entries:
        rng = entry.get("local_m4_index_range")
        if not isinstance(rng, list) or len(rng) != 2:
            continue
        lo = int(rng[0])
        hi = int(rng[1])
        if lo <= idx <= hi:
            return str(entry.get("family_name", "unknown"))
    return "unknown"


def _build_research_distribution_report(
    *,
    run_dir: Path,
    research_mode: str,
    plan_doc: Optional[dict[str, Any]],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "run_dir": str(run_dir),
        "research_mode": str(research_mode),
        "notes": [],
    }
    family_entries = _extract_family_entries(plan_doc)
    leaderboard_path = run_dir / "robustness_leaderboard.csv"
    if pd is None:
        report["notes"].append("pandas_unavailable")
        return report
    if not leaderboard_path.exists():
        report["notes"].append("missing_robustness_leaderboard.csv")
        return report

    leaderboard = pd.read_csv(leaderboard_path)
    if leaderboard.shape[0] <= 0:
        report["notes"].append("empty_robustness_leaderboard.csv")
        return report

    standard_reject_col = "standard_reject" if "standard_reject" in leaderboard.columns else "reject"
    standard_pass_col = "standard_pass" if "standard_pass" in leaderboard.columns else "pass"
    if "discovery_included" not in leaderboard.columns:
        leaderboard["discovery_included"] = bool(str(research_mode).strip().lower() == "discovery")

    if "family_name" not in leaderboard.columns:
        leaderboard["family_name"] = leaderboard["m4_idx"].map(lambda x: _family_name_from_m4_idx(x, family_entries))
    if "family_id" not in leaderboard.columns:
        leaderboard["family_id"] = leaderboard["family_name"].astype(str)
    if "evaluation_window" in leaderboard.columns:
        leaderboard["block_window"] = pd.to_numeric(leaderboard["evaluation_window"], errors="coerce").fillna(-1).astype(int)
    else:
        leaderboard["block_window"] = leaderboard["block_minutes"].astype(int) if "block_minutes" in leaderboard.columns else -1
    if "window_set" not in leaderboard.columns:
        leaderboard["window_set"] = ""
    if "hypothesis_id" not in leaderboard.columns:
        leaderboard["hypothesis_id"] = ""
    if "parameter_hash" not in leaderboard.columns:
        leaderboard["parameter_hash"] = ""
    if "cost_adjusted_expectancy" not in leaderboard.columns:
        if "cum_return" in leaderboard.columns:
            leaderboard["cost_adjusted_expectancy"] = pd.to_numeric(leaderboard["cum_return"], errors="coerce").fillna(0.0)
        else:
            leaderboard["cost_adjusted_expectancy"] = 0.0
    if "overnight_suitability_score" not in leaderboard.columns:
        leaderboard["overnight_suitability_score"] = np.nan
    if "zimtra_compliance_flags" not in leaderboard.columns:
        leaderboard["zimtra_compliance_flags"] = ""
    if "cross_window_consistency_score" not in leaderboard.columns:
        leaderboard["cross_window_consistency_score"] = np.nan
    if "cross_window_conflict_score" not in leaderboard.columns:
        leaderboard["cross_window_conflict_score"] = np.nan
    if "multi_scale_stability_score" not in leaderboard.columns:
        leaderboard["multi_scale_stability_score"] = np.nan
    if "evaluation_role" not in leaderboard.columns:
        leaderboard["evaluation_role"] = ""

    daily_path = run_dir / "daily_returns.parquet"
    sharpe_map: dict[str, float] = {}
    effective_return_signatures = 0
    signature_group_sizes: list[int] = []
    if daily_path.exists():
        daily = pd.read_parquet(daily_path)
        candidate_cols = [c for c in daily.columns if c not in {"session_id", "benchmark"}]
        signature_groups: dict[bytes, list[str]] = {}
        for col in candidate_cols:
            vec = np.asarray(daily[col], dtype=np.float64)
            finite = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
            sd = float(np.std(finite, ddof=1)) if finite.size >= 2 else 0.0
            sharpe_map[str(col)] = float(np.mean(finite) / sd * np.sqrt(252.0)) if finite.size >= 2 and sd > 0.0 else 0.0
            sig = np.round(np.nan_to_num(vec, nan=9.999e99), 12).tobytes()
            signature_groups.setdefault(sig, []).append(str(col))
        effective_return_signatures = int(len(signature_groups))
        signature_group_sizes = sorted((len(v) for v in signature_groups.values()), reverse=True)
    else:
        report["notes"].append("missing_daily_returns.parquet")

    leaderboard["sharpe_daily"] = leaderboard["candidate_id"].map(lambda x: float(sharpe_map.get(str(x), 0.0)))

    trade_path = run_dir / "trade_log.parquet"
    trade_counts: dict[str, int] = {}
    if trade_path.exists():
        trade_df = pd.read_parquet(trade_path)
        if "candidate_id" in trade_df.columns and trade_df.shape[0] > 0:
            trade_counts = {str(k): int(v) for k, v in trade_df["candidate_id"].value_counts().to_dict().items()}
    else:
        report["notes"].append("missing_trade_log.parquet")
    leaderboard["executed_trade_count"] = leaderboard["candidate_id"].map(lambda x: int(trade_counts.get(str(x), 0)))

    standard_reject_counts = {str(k): int(v) for k, v in leaderboard[standard_reject_col].value_counts(dropna=False).to_dict().items()}
    standard_pass_counts = {str(k): int(v) for k, v in leaderboard[standard_pass_col].value_counts(dropna=False).to_dict().items()}
    discovery_included_count = int(np.sum(np.asarray(leaderboard["discovery_included"], dtype=bool)))
    cluster_counts = {str(k): int(v) for k, v in leaderboard["cluster_id"].value_counts(dropna=False).to_dict().items()} if "cluster_id" in leaderboard.columns else {}

    positive_expectancy_mask = np.asarray(leaderboard["cost_adjusted_expectancy"], dtype=np.float64) > 0.0
    positive_sharpe_mask = np.asarray(leaderboard["sharpe_daily"], dtype=np.float64) > 0.0
    traded_mask = np.asarray(leaderboard["executed_trade_count"], dtype=np.int64) > 0

    probe_rows = leaderboard[leaderboard["block_window"] > 0].copy()
    if probe_rows.shape[0] <= 0:
        probe_rows = leaderboard.copy()

    def _nanmean(values: Any) -> float:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size <= 0:
            return 0.0
        return float(np.mean(arr))

    hypothesis_summary: list[dict[str, Any]] = []
    hypothesis_cols = ["family_id", "family_name", "hypothesis_id", "parameter_hash"]
    hypothesis_df = probe_rows.copy()
    hypothesis_df = hypothesis_df[hypothesis_df["hypothesis_id"].astype(str).str.len() > 0]
    if hypothesis_df.shape[0] > 0:
        gb_h = hypothesis_df.groupby(hypothesis_cols, dropna=False, sort=True)
        for key, frame in gb_h:
            family_id, family_name, hypothesis_id, parameter_hash = [str(x) for x in (key if isinstance(key, tuple) else (key,))]
            window_set = parse_stage_a_window_set(str(frame["window_set"].iloc[0])) if "window_set" in frame.columns else ()
            window_count_expected = int(len(window_set))
            window_count_actual = int(frame["block_window"].nunique(dropna=True))
            hypothesis_summary.append(
                {
                    "family_id": family_id,
                    "family_name": family_name,
                    "hypothesis_id": hypothesis_id,
                    "parameter_hash": parameter_hash,
                    "window_set": list(window_set),
                    "window_count_expected": window_count_expected,
                    "window_count_actual": window_count_actual,
                    "window_probe_completion": float(window_count_actual / max(window_count_expected, 1)) if window_count_expected > 0 else 0.0,
                    "cost_adjusted_expectancy": _nanmean(frame["cost_adjusted_expectancy"]),
                    "cross_window_consistency_score": _nanmean(frame["cross_window_consistency_score"]),
                    "cross_window_conflict_score": _nanmean(frame["cross_window_conflict_score"]),
                    "multi_scale_stability_score": _nanmean(frame["multi_scale_stability_score"]),
                    "overnight_suitability_score": _nanmean(frame["overnight_suitability_score"]),
                    "standard_reject_count": int(np.sum(np.asarray(frame[standard_reject_col], dtype=bool))),
                    "standard_pass_count": int(np.sum(np.asarray(frame[standard_pass_col], dtype=bool))),
                    "positive_expectancy_windows": int(np.sum(np.asarray(frame["cost_adjusted_expectancy"], dtype=np.float64) > 0.0)),
                }
            )

    def _group_summary(group_cols: list[str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        gb = leaderboard.groupby(group_cols, dropna=False, sort=True)
        for key, frame in gb:
            key_tuple = key if isinstance(key, tuple) else (key,)
            row = {group_cols[i]: (int(key_tuple[i]) if group_cols[i] == "block_window" else str(key_tuple[i])) for i in range(len(group_cols))}
            row.update(
                {
                    "candidate_count": int(len(frame)),
                    "discovery_included_count": int(np.sum(np.asarray(frame["discovery_included"], dtype=bool))),
                    "standard_reject_count": int(np.sum(np.asarray(frame[standard_reject_col], dtype=bool))),
                    "standard_pass_count": int(np.sum(np.asarray(frame[standard_pass_col], dtype=bool))),
                    "executed_trade_count": int(np.sum(np.asarray(frame["executed_trade_count"], dtype=np.int64) > 0)),
                    "positive_expectancy_count": int(np.sum(np.asarray(frame["cost_adjusted_expectancy"], dtype=np.float64) > 0.0)),
                    "positive_sharpe_count": int(np.sum(np.asarray(frame["sharpe_daily"], dtype=np.float64) > 0.0)),
                    "mean_robustness_score": float(np.mean(np.asarray(frame["robustness_score"], dtype=np.float64))),
                    "mean_execution_robustness": float(np.mean(np.asarray(frame["execution_robustness"], dtype=np.float64))),
                    "mean_cost_adjusted_expectancy": float(np.mean(np.asarray(frame["cost_adjusted_expectancy"], dtype=np.float64))),
                    "mean_sharpe_daily": float(np.mean(np.asarray(frame["sharpe_daily"], dtype=np.float64))),
                    "mean_cross_window_consistency_score": _nanmean(frame["cross_window_consistency_score"]),
                    "mean_multi_scale_stability_score": _nanmean(frame["multi_scale_stability_score"]),
                    "cluster_ids": sorted(int(x) for x in set(np.asarray(frame["cluster_id"], dtype=np.int64).tolist())) if "cluster_id" in frame.columns else [],
                }
            )
            out.append(row)
        return out

    family_summary = _group_summary(["family_id", "family_name"])
    window_summary = []
    for key, frame in probe_rows.groupby(["block_window"], dropna=False, sort=True):
        row = {"block_window": int(key if not isinstance(key, tuple) else key[0])}
        row.update(
            {
                "candidate_count": int(len(frame)),
                "discovery_included_count": int(np.sum(np.asarray(frame["discovery_included"], dtype=bool))),
                "standard_reject_count": int(np.sum(np.asarray(frame[standard_reject_col], dtype=bool))),
                "standard_pass_count": int(np.sum(np.asarray(frame[standard_pass_col], dtype=bool))),
                "executed_trade_count": int(np.sum(np.asarray(frame["executed_trade_count"], dtype=np.int64) > 0)),
                "positive_expectancy_count": int(np.sum(np.asarray(frame["cost_adjusted_expectancy"], dtype=np.float64) > 0.0)),
                "positive_sharpe_count": int(np.sum(np.asarray(frame["sharpe_daily"], dtype=np.float64) > 0.0)),
                "mean_robustness_score": float(np.mean(np.asarray(frame["robustness_score"], dtype=np.float64))),
                "mean_execution_robustness": float(np.mean(np.asarray(frame["execution_robustness"], dtype=np.float64))),
                "mean_cost_adjusted_expectancy": float(np.mean(np.asarray(frame["cost_adjusted_expectancy"], dtype=np.float64))),
                "mean_sharpe_daily": float(np.mean(np.asarray(frame["sharpe_daily"], dtype=np.float64))),
                "mean_cross_window_consistency_score": _nanmean(frame["cross_window_consistency_score"]),
                "mean_multi_scale_stability_score": _nanmean(frame["multi_scale_stability_score"]),
            }
        )
        window_summary.append(row)

    family_window_summary = []
    for key, frame in probe_rows.groupby(["family_id", "family_name", "block_window"], dropna=False, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = {
            "family_id": str(key_tuple[0]),
            "family_name": str(key_tuple[1]),
            "block_window": int(key_tuple[2]),
        }
        row.update(
            {
                "candidate_count": int(len(frame)),
                "discovery_included_count": int(np.sum(np.asarray(frame["discovery_included"], dtype=bool))),
                "standard_reject_count": int(np.sum(np.asarray(frame[standard_reject_col], dtype=bool))),
                "standard_pass_count": int(np.sum(np.asarray(frame[standard_pass_col], dtype=bool))),
                "executed_trade_count": int(np.sum(np.asarray(frame["executed_trade_count"], dtype=np.int64) > 0)),
                "positive_expectancy_count": int(np.sum(np.asarray(frame["cost_adjusted_expectancy"], dtype=np.float64) > 0.0)),
                "positive_sharpe_count": int(np.sum(np.asarray(frame["sharpe_daily"], dtype=np.float64) > 0.0)),
                "mean_robustness_score": float(np.mean(np.asarray(frame["robustness_score"], dtype=np.float64))),
                "mean_execution_robustness": float(np.mean(np.asarray(frame["execution_robustness"], dtype=np.float64))),
                "mean_cost_adjusted_expectancy": float(np.mean(np.asarray(frame["cost_adjusted_expectancy"], dtype=np.float64))),
                "mean_sharpe_daily": float(np.mean(np.asarray(frame["sharpe_daily"], dtype=np.float64))),
                "mean_cross_window_consistency_score": _nanmean(frame["cross_window_consistency_score"]),
                "mean_multi_scale_stability_score": _nanmean(frame["multi_scale_stability_score"]),
            }
        )
        family_window_summary.append(row)
    top_regions = sorted(
        family_window_summary,
        key=lambda x: (
            -float(x["mean_cost_adjusted_expectancy"]),
            -float(x["positive_expectancy_count"]),
            -float(x["mean_cross_window_consistency_score"]),
            -float(x["mean_sharpe_daily"]),
            str(x["family_id"]),
            int(x["block_window"]),
        ),
    )[:8]

    top_n = max(1, int(np.ceil(0.05 * float(len(leaderboard)))))
    top_df = leaderboard.sort_values(
        ["cost_adjusted_expectancy", "cross_window_consistency_score", "robustness_score", "candidate_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).head(top_n)

    consistency_values = np.asarray(
        [float(x["cross_window_consistency_score"]) for x in hypothesis_summary if np.isfinite(float(x["cross_window_consistency_score"]))],
        dtype=np.float64,
    )
    conflict_values = np.asarray(
        [float(x["cross_window_conflict_score"]) for x in hypothesis_summary if np.isfinite(float(x["cross_window_conflict_score"]))],
        dtype=np.float64,
    )
    stability_values = np.asarray(
        [float(x["multi_scale_stability_score"]) for x in hypothesis_summary if np.isfinite(float(x["multi_scale_stability_score"]))],
        dtype=np.float64,
    )
    cross_window_summary = {
        "hypothesis_count": int(len(hypothesis_summary)),
        "mean_cross_window_consistency_score": float(np.mean(consistency_values)) if consistency_values.size > 0 else 0.0,
        "median_cross_window_consistency_score": float(np.median(consistency_values)) if consistency_values.size > 0 else 0.0,
        "mean_cross_window_conflict_score": float(np.mean(conflict_values)) if conflict_values.size > 0 else 0.0,
        "mean_multi_scale_stability_score": float(np.mean(stability_values)) if stability_values.size > 0 else 0.0,
        "high_consistency_hypothesis_count": int(np.sum(consistency_values >= 0.60)) if consistency_values.size > 0 else 0,
        "top_hypotheses": sorted(
            hypothesis_summary,
            key=lambda x: (
                -float(x["cost_adjusted_expectancy"]),
                -float(x["cross_window_consistency_score"]),
                str(x["hypothesis_id"]),
            ),
        )[:12],
    }

    report.update(
        {
            "plan_available": bool(plan_doc is not None),
            "family_entries_count": int(len(family_entries)),
            "candidate_count": int(len(leaderboard)),
            "cluster_count": int(leaderboard["cluster_id"].nunique(dropna=False)) if "cluster_id" in leaderboard.columns else 0,
            "standard_reject_counts": standard_reject_counts,
            "standard_pass_counts": standard_pass_counts,
            "discovery_included_candidates": int(discovery_included_count),
            "count_standard_reject": int(np.sum(np.asarray(leaderboard[standard_reject_col], dtype=bool))),
            "count_standard_pass": int(np.sum(np.asarray(leaderboard[standard_pass_col], dtype=bool))),
            "count_discovery_included": int(discovery_included_count),
            "effective_return_signature_count": int(effective_return_signatures),
            "distinct_robustness_score_count": int(leaderboard["robustness_score"].nunique(dropna=False)),
            "distinct_execution_robustness_count": int(leaderboard["execution_robustness"].nunique(dropna=False)),
            "family_representation_counts": {str(k): int(v) for k, v in leaderboard["family_id"].value_counts(dropna=False).to_dict().items()},
            "window_representation_counts": {str(k): int(v) for k, v in probe_rows["block_window"].value_counts(dropna=False).sort_index().to_dict().items()},
            "count_with_executed_trades": int(np.sum(traded_mask)),
            "count_with_positive_expectancy": int(np.sum(positive_expectancy_mask)),
            "count_with_positive_sharpe": int(np.sum(positive_sharpe_mask)),
            "positive_expectancy_count": int(np.sum(positive_expectancy_mask)),
            "positive_sharpe_count": int(np.sum(positive_sharpe_mask)),
            "sharpe_distribution": _distribution_summary(leaderboard["sharpe_daily"]),
            "return_distribution": _distribution_summary(leaderboard.get("cum_return", pd.Series(np.zeros(len(leaderboard))))),
            "cost_adjusted_expectancy_distribution": _distribution_summary(leaderboard["cost_adjusted_expectancy"]),
            "drawdown_distribution": _distribution_summary(leaderboard["max_drawdown"]),
            "cluster_analysis": {
                "cluster_count": int(leaderboard["cluster_id"].nunique(dropna=False)) if "cluster_id" in leaderboard.columns else 0,
                "cluster_counts": cluster_counts,
                "signature_group_sizes_top20": signature_group_sizes[:20],
            },
            "top_5_percent_candidate_metrics": top_df[
                [
                    "candidate_id",
                    "family_id",
                    "family_name",
                    "hypothesis_id",
                    "block_window",
                    "cost_adjusted_expectancy",
                    "cross_window_consistency_score",
                    "multi_scale_stability_score",
                    "parameter_hash",
                    "window_set",
                    "overnight_suitability_score",
                    "zimtra_compliance_flags",
                    "robustness_score",
                    "execution_robustness",
                    "max_drawdown",
                    "sharpe_daily",
                    standard_reject_col,
                    standard_pass_col,
                    "discovery_included",
                    "cluster_id",
                ]
            ].to_dict(orient="records"),
            "family_level_summary": family_summary,
            "window_level_summary": window_summary,
            "hypothesis_level_summary": hypothesis_summary[:250],
            "top_expectancy_pockets_by_family_window": top_regions,
            "top_family_window_regions": top_regions,
            "cross_window_consistency_summary": cross_window_summary,
        }
    )
    return report


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
                _artifact_write_json(run_manifest_path, manifest_doc)
        except Exception:
            pass
    if run_status_path.exists():
        try:
            status_doc = json.loads(run_status_path.read_text(encoding="utf-8"))
            if isinstance(status_doc, dict):
                status_doc["runtime_warning_count"] = int(runtime_warning_count)
                _artifact_write_json(run_status_path, status_doc)
        except Exception:
            pass

    leaderboard = out.stats_verdict.get("leaderboard", [])
    pass_count = int(sum(1 for row in leaderboard if bool(row.get("pass", False))))
    research_report_path = run_dir / "research_distribution_report.json"
    research_report: Optional[dict[str, Any]] = None
    if str(getattr(harness_cfg, "research_mode", "standard")).strip().lower() == "discovery":
        plan_path = config_path.with_name(f"{config_path.stem}_plan.yaml")
        research_report = _build_research_distribution_report(
            run_dir=run_dir,
            research_mode=str(harness_cfg.research_mode),
            plan_doc=_load_plan_doc(plan_path),
        )
        research_report["config_path"] = str(config_path)
        research_report["plan_path"] = str(plan_path) if plan_path.exists() else None
        _artifact_write_json(research_report_path, research_report)

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
    try:
        module6_report = run_module6_portfolio_research(
            run_dir,
            output_dir=run_dir / "module6",
            config=Module6Config(),
        )
    except Exception as exc:
        raise RuntimeError(f"MODULE6_SUPPORTED_FLOW_BLOCKED: {type(exc).__name__}: {exc}") from exc

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
        "research_mode": str(getattr(harness_cfg, "research_mode", "standard")),
        "module6_output_dir": str(module6_report.output_dir),
        "module6_selected_count": int(len(module6_report.selected_portfolio_pks)),
    }
    if research_report is not None:
        summary["research_distribution_report"] = str(research_report_path)
        summary["discovery_included_candidates"] = int(research_report.get("discovery_included_candidates", 0))
        summary["standard_reject_counts"] = research_report.get("standard_reject_counts", {})

    _artifact_write_json(run_dir / "run_summary.json", summary)

    log_event(logger, "INFO", "run_complete", event_type="run_complete")


if __name__ == "__main__":
    main()
