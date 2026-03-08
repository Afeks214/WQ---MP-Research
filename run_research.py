#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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


def _require_pandas() -> Any:
    if pd is None:
        raise RuntimeError("pandas is required. Install with: pip install pandas")
    return pd


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
