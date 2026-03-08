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
