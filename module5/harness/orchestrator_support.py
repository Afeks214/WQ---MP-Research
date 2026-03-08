from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np


def build_candidate_results(
    all_results: list[dict[str, Any]],
    candidate_verdict: dict[str, dict[str, Any]],
) -> list[dict[str, object]]:
    candidate_results: list[dict[str, object]] = []
    for r in all_results:
        out = {
            "task_id": r.get("task_id"),
            "candidate_id": r.get("candidate_id"),
            "split_id": r.get("split_id"),
            "scenario_id": r.get("scenario_id"),
            "status": r.get("status"),
            "error": r.get("error", ""),
            "session_ids": r.get("session_ids"),
            "session_ids_exec": r.get("session_ids_exec"),
            "session_ids_raw": r.get("session_ids_raw"),
            "daily_returns": r.get("daily_returns"),
            "daily_returns_exec": r.get("daily_returns_exec"),
            "daily_returns_raw": r.get("daily_returns_raw"),
            "m2_idx": r.get("m2_idx"),
            "m3_idx": r.get("m3_idx"),
            "m4_idx": r.get("m4_idx"),
            "tags": r.get("tags", []),
            "test_days": int(r.get("test_days", 0)),
            "quality_reason_codes": sorted([str(x) for x in r.get("quality_reason_codes", [])]),
            "dqs_min": float(r.get("dqs_min", np.nan)),
            "dqs_median": float(r.get("dqs_median", np.nan)),
        }
        lb = candidate_verdict.get(str(r.get("candidate_id", "")))
        if lb is not None:
            out.update(
                {
                    "dsr": lb.get("dsr"),
                    "in_mcs": lb.get("in_mcs"),
                    "wrc_p": lb.get("wrc_p"),
                    "spa_p": lb.get("spa_p"),
                    "pbo": lb.get("pbo"),
                    "cluster_id": lb.get("cluster_id"),
                    "cluster_representative": lb.get("cluster_representative"),
                    "regime_robustness": lb.get("regime_robustness"),
                    "execution_robustness": lb.get("execution_robustness"),
                    "horizon_robustness": lb.get("horizon_robustness"),
                    "robustness_score": lb.get("robustness_score"),
                    "fragile": lb.get("fragile"),
                    "reject": lb.get("reject"),
                    "pass": lb.get("pass"),
                }
            )
        candidate_results.append(out)
    return candidate_results


def finalize_run_outputs(
    *,
    report_root: Path,
    run_id: str,
    run_started_utc: Any,
    engine_cfg: Any,
    harness_cfg: Any,
    candidates: list[Any],
    splits: list[Any],
    scenarios: list[Any],
    all_results: list[dict[str, Any]],
    ok_results: list[dict[str, Any]],
    ledger_rows: list[dict[str, Any]],
    stats_verdict: dict[str, Any],
    candidate_verdict: dict[str, dict[str, Any]],
    common_sessions: np.ndarray,
    daily_mat_exec: np.ndarray,
    daily_mat_raw: np.ndarray,
    daily_bmk: np.ndarray,
    baseline_candidate_ids: list[str],
    candidate_scenario_series: dict[str, dict[str, dict[int, float]]],
    dq_bundle: dict[str, Any],
    feature_tensor: np.ndarray,
    feature_handles_master: Any,
    budget: int,
    avail: int,
    tasks_completed: int,
    tasks_submitted: int,
    groups_completed: int,
    failure_count: int,
    failure_tracker: dict[tuple[str, str], dict[str, set[str] | bool]],
    aborted: bool,
    aborted_early: bool,
    abort_reason: str,
    execution_mode: str,
    use_process_pool: bool,
    effective_workers: int,
    payload_safe: bool,
    payload_arg_max_bytes: int,
    large_payload_passing_avoided: bool,
    mp_start_method: str,
    runtime_seconds: float,
    dataset_hash: str,
    keep_symbols: list[str],
    symbols: list[str],
    keep_idx: np.ndarray,
    ingest_meta: dict[str, Any],
    n_group_tasks: int,
    est_state: int,
    m2_configs: list[Any],
    m3_configs: list[Any],
    m4_configs: list[Any],
    tensor_npz_path: Path,
    tensor_json_path: Path,
    tensor_hash: str,
    run_status_path: Path,
    deadletter_path: Path,
    self_audit_report_path: Path,
    compute_authority: dict[str, Any],
    feature_tensor_role: dict[str, Any],
    robustness_caps: dict[str, float],
    quick_settings: Any,
    first_exception_class: str,
    first_exception_message: str,
    first_exception_hash: str,
    monitor: Any,
    require_pandas_fn: Callable[[], Any],
    stack_payload_frames_fn: Callable[[list[dict[str, np.ndarray]]], Any],
    write_json_fn: Callable[[Path, Any], None],
    write_frozen_json_fn: Callable[[Path, Any], None],
    build_candidate_artifacts_fn: Callable[..., tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]],
    collect_ledger_rows_fn: Callable[[list[dict[str, Any]], str], list[dict[str, Any]]],
    ledger_write_fn: Callable[[list[dict[str, Any]], Path], None],
    git_hash_fn: Callable[[], str],
    stable_hash_obj_fn: Callable[[Any], str],
    execution_topology_fn: Callable[[str, bool], dict[str, object]],
    dq_accept: str,
    dq_degrade: str,
    dq_reject: str,
    harness_output_cls: type,
) -> Any:
    eq_payloads = [r["equity_payload"] for r in ok_results if r.get("equity_payload") is not None]
    tr_payloads = [r["trade_payload"] for r in ok_results if r.get("trade_payload") is not None]
    micro_payloads = [r["micro_payload"] for r in ok_results if r.get("micro_payload") is not None]
    profile_payloads = [r["profile_payload"] for r in ok_results if r.get("profile_payload") is not None]
    funnel_payloads = [r["funnel_payload"] for r in ok_results if r.get("funnel_payload") is not None]

    eq_df = stack_payload_frames_fn(eq_payloads)
    tr_df = stack_payload_frames_fn(tr_payloads)
    micro_df = stack_payload_frames_fn(micro_payloads)
    profile_df = stack_payload_frames_fn(profile_payloads)
    funnel_df = stack_payload_frames_fn(funnel_payloads)

    pdx = require_pandas_fn()

    equity_path = report_root / "equity_curves.parquet"
    trade_path = report_root / "trade_log.parquet"
    daily_path = report_root / "daily_returns.parquet"
    verdict_path = report_root / "verdict.json"
    stats_raw_path = report_root / "stats_raw.json"
    manifest_path = report_root / "run_manifest.json"
    micro_diag_path = report_root / "micro_diagnostics.parquet"
    micro_profile_blocks_path = report_root / "micro_profile_blocks.parquet"
    funnel_1545_path = report_root / "funnel_1545.parquet"
    dq_report_path = report_root / "dq_report.csv"
    dq_bar_flags_path = report_root / "dq_bar_flags.parquet"

    eq_df.to_parquet(equity_path, index=False)
    tr_df.to_parquet(trade_path, index=False)

    daily_cols: dict[str, Any] = {
        "session_id": common_sessions.astype(np.int64),
        "benchmark": daily_bmk,
    }
    baseline_col = {str(cid): j for j, cid in enumerate(baseline_candidate_ids)}
    for c in sorted(candidates, key=lambda x: str(x.candidate_id)):
        cid = str(c.candidate_id)
        if cid in baseline_col:
            daily_cols[cid] = daily_mat_exec[:, int(baseline_col[cid])]
        else:
            daily_cols[cid] = np.zeros(int(common_sessions.shape[0]), dtype=np.float64)
    daily_df = pdx.DataFrame(daily_cols)
    daily_df.to_parquet(daily_path, index=False)

    stats_verdict_to_write = dict(stats_verdict)
    stats_verdict_to_write["leaderboard"] = sorted(
        list(stats_verdict.get("leaderboard", [])),
        key=lambda x: str(x.get("candidate_id", "")),
    )
    write_frozen_json_fn(stats_raw_path, stats_verdict_to_write)

    if bool(harness_cfg.export_micro_diagnostics):
        micro_df.to_parquet(micro_diag_path, index=False)
        if bool(harness_cfg.micro_diag_export_block_profiles):
            profile_df.to_parquet(micro_profile_blocks_path, index=False)
        if bool(harness_cfg.micro_diag_export_funnel):
            funnel_df.to_parquet(funnel_1545_path, index=False)

    dq_day_df = pdx.DataFrame(list(dq_bundle.get("day_reports", [])))
    if dq_day_df.shape[0] > 0:
        dq_day_df = dq_day_df.sort_values(["symbol", "session_date"], kind="mergesort").reset_index(drop=True)
    dq_day_df.to_csv(dq_report_path, index=False)

    dq_bar_df = pdx.DataFrame(list(dq_bundle.get("bar_flags_rows", [])))
    if dq_bar_df.shape[0] > 0:
        dq_bar_df["timestamp"] = pdx.to_datetime(dq_bar_df["timestamp"], utc=True, errors="coerce")
        dq_bar_df = dq_bar_df.sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(drop=True)
    dq_bar_df.to_parquet(dq_bar_flags_path, index=False)

    candidate_rows, robustness_rows, plateaus_doc = build_candidate_artifacts_fn(
        report_root=report_root,
        run_id=run_id,
        run_started_utc=run_started_utc,
        git_hash=git_hash_fn(),
        candidates=candidates,
        all_results=all_results,
        candidate_daily_mat=daily_mat_exec,
        daily_bmk=daily_bmk,
        common_sessions=common_sessions,
        baseline_candidate_ids=baseline_candidate_ids,
        candidate_scenario_series=candidate_scenario_series,
        candidate_verdict=candidate_verdict,
        expected_baseline_tasks=int(len(splits)) if any(str(s.scenario_id) == "baseline" and bool(s.enabled) for s in scenarios) else 0,
        scenarios=scenarios,
        engine_cfg=engine_cfg,
        m2_configs=m2_configs,
        m3_configs=m3_configs,
        m4_configs=m4_configs,
        harness_cfg=harness_cfg,
    )

    leaderboard_csv_path = report_root / "leaderboard.csv"
    leaderboard_json_path = report_root / "leaderboard.json"
    robustness_csv_path = report_root / "robustness_leaderboard.csv"
    plateaus_path = report_root / "plateaus.json"
    strategy_ledger_path = report_root / "strategy_results.parquet"

    pdx.DataFrame(sorted(candidate_rows, key=lambda x: str(x["candidate_id"]))).to_csv(leaderboard_csv_path, index=False)
    write_json_fn(leaderboard_json_path, sorted(candidate_rows, key=lambda x: str(x["candidate_id"])))
    pdx.DataFrame(robustness_rows).to_csv(robustness_csv_path, index=False)
    write_json_fn(plateaus_path, plateaus_doc)
    write_json_fn(
        verdict_path,
        {
            "leaderboard": sorted(candidate_rows, key=lambda x: str(x["candidate_id"])),
            "summary": {
                "n_candidates_with_baseline": int(daily_mat_exec.shape[1]),
                "n_candidates_with_baseline_raw": int(daily_mat_raw.shape[1]),
                "n_candidates_total": int(len(candidates)),
                "n_days": int(daily_mat_exec.shape[0]),
                "benchmark_symbol": harness_cfg.benchmark_symbol,
            },
        },
    )

    validation_rows: list[dict[str, Any]] = []
    for c in sorted(candidates, key=lambda x: str(x.candidate_id)):
        cid = str(c.candidate_id)
        lb = candidate_verdict.get(cid, {})
        validation_rows.append(
            {
                "strategy_id": cid,
                "cluster_id": lb.get("cluster_id"),
                "cluster_representative": lb.get("cluster_representative", cid),
                "dsr": lb.get("dsr"),
                "pbo": lb.get("pbo"),
                "spa_p": lb.get("spa_p"),
                "mcs_inclusion": bool(lb.get("in_mcs", False)),
                "regime_robustness": lb.get("regime_robustness"),
                "execution_robustness": lb.get("execution_robustness"),
                "horizon_robustness": lb.get("horizon_robustness"),
                "robustness_score": lb.get("robustness_score", float("-inf")),
                "reject": bool(lb.get("reject", True)),
                "fragile": bool(lb.get("fragile", False)),
            }
        )
    validation_rows = sorted(validation_rows, key=lambda x: str(x["strategy_id"]))
    validation_report_path = report_root / "validation_report.json"
    write_frozen_json_fn(validation_report_path, validation_rows)
    canonical_root = report_root.parent.parent if report_root.parent.name == "module5_harness" else report_root.parent
    canonical_validation_report_path = canonical_root / "validation_report.json"
    write_frozen_json_fn(canonical_validation_report_path, validation_rows)

    ledger_write_fn(
        collect_ledger_rows_fn(
            ledger_rows,
            evaluation_timestamp=datetime.now(timezone.utc).isoformat(),
        ),
        strategy_ledger_path,
    )
    monitor.check_and_emit(
        strategies_completed=int(tasks_completed),
        tensor=feature_handles_master.array,
        worker_status={"active": 0},
        ledger_path=strategy_ledger_path,
        queue_backlog=0,
        memory_status={"ok": bool(feature_tensor.nbytes <= budget), "available_bytes": int(avail), "budget_bytes": int(budget)},
        require_ledger_exists=True,
    )

    run_status = {
        "run_id": run_id,
        "aborted": bool(aborted),
        "aborted_early": bool(aborted_early),
        "abort_reason": str(abort_reason),
        "execution_mode": str(execution_mode),
        "process_start_method": str(mp_start_method if mp_start_method else ""),
        "tasks_submitted": int(tasks_submitted),
        "tasks_completed": int(tasks_completed),
        "groups_completed": int(groups_completed),
        "failure_count": int(failure_count),
        "failure_rate": float(failure_count / max(tasks_completed, 1)),
        "first_exception": {
            "class": first_exception_class if first_exception_class else None,
            "message": first_exception_message if first_exception_message else None,
            "error_hash": first_exception_hash if first_exception_hash else None,
        },
        "systemic_breaker": {
            "enabled": True,
            "triggered": bool(aborted),
            "reason": str(abort_reason),
            "tracked_signatures": int(len(failure_tracker)),
            "rule": "same_signature && units>=3 && assets>=2 && candidates>=2",
        },
        "quick_run": {
            "enabled": bool(quick_settings.enabled),
            "task_timeout_sec": int(quick_settings.task_timeout_sec),
            "progress_every_groups": int(quick_settings.progress_every_groups),
            "baseline_only": bool(quick_settings.baseline_only),
            "disable_cpcv": bool(quick_settings.disable_cpcv),
        },
        "compute_authority": compute_authority,
        "feature_tensor_role": feature_tensor_role,
        "execution_topology": execution_topology_fn(execution_mode, use_process_pool),
    }
    write_json_fn(run_status_path, run_status)

    manifest = {
        "run_id": run_id,
        "run_started_utc": run_started_utc.isoformat(),
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": git_hash_fn(),
        "git_commit": git_hash_fn(),
        "seed": int(harness_cfg.seed),
        "search_seed": int(harness_cfg.seed),
        "config_hash": stable_hash_obj_fn(asdict(harness_cfg)),
        "dataset_hash": str(dataset_hash),
        "asset_count": int(len(keep_symbols)),
        "strategy_count": int(len(candidates)),
        "start_time": run_started_utc.isoformat(),
        "end_time": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": float(runtime_seconds),
        "symbols_input": list(symbols),
        "symbols_kept": keep_symbols,
        "keep_idx": keep_idx.tolist(),
        "ingestion": ingest_meta,
        "engine_cfg_hash": stable_hash_obj_fn(asdict(engine_cfg)),
        "m2_hashes": [stable_hash_obj_fn(asdict(c)) for c in m2_configs],
        "m3_hashes": [stable_hash_obj_fn(asdict(c)) for c in m3_configs],
        "m4_hashes": [stable_hash_obj_fn(asdict(c)) for c in m4_configs],
        "harness_cfg_hash": stable_hash_obj_fn(asdict(harness_cfg)),
        "n_candidates": len(candidates),
        "n_splits": len(splits),
        "n_scenarios": len(scenarios),
        "n_group_tasks": int(n_group_tasks),
        "n_task_results": len(all_results),
        "n_ok_results": len(ok_results),
        "execution_mode": str(execution_mode),
        "process_start_method": str(mp_start_method if mp_start_method else ""),
        "tasks_submitted": int(tasks_submitted),
        "tasks_completed": int(tasks_completed),
        "groups_completed": int(groups_completed),
        "aborted_early": bool(aborted_early),
        "failure_count": int(failure_count),
        "failure_rate": float(failure_count / max(tasks_completed, 1)),
        "aborted": bool(aborted),
        "abort_reason": str(abort_reason),
        "first_exception_class": first_exception_class if first_exception_class else None,
        "first_exception_message": first_exception_message if first_exception_message else None,
        "first_exception_hash": first_exception_hash if first_exception_hash else None,
        "run_status_path": str(run_status_path),
        "deadletter_path": str(deadletter_path),
        "deadletter_count": int(failure_count),
        "n_candidate_rows": int(len(candidate_rows)),
        "daily_matrix_shape": list(daily_mat_exec.shape),
        "daily_matrix_shape_raw": list(daily_mat_raw.shape),
        "n_candidates_with_baseline": int(len(baseline_candidate_ids)),
        "parallel_backend": harness_cfg.parallel_backend,
        "parallel_workers_effective": int(effective_workers),
        "payload_safe": bool(payload_safe),
        "payload_arg_max_bytes": int(payload_arg_max_bytes),
        "large_payload_passing_avoided": bool(large_payload_passing_avoided),
        "compute_authority": compute_authority,
        "feature_tensor_role": feature_tensor_role,
        "execution_topology": execution_topology_fn(execution_mode, use_process_pool),
        "robustness_score": {
            "formula": "0.20*dsr+0.15*(1-pbo)+0.10*(1-spa_p)+0.20*regime+0.20*execution+0.15*horizon",
            "dsr_source": "cluster_representative_exec",
            "caps": dict(robustness_caps),
        },
        "circuit_breaker": {
            "failure_rate_abort_threshold": float(harness_cfg.failure_rate_abort_threshold),
            "failure_count_abort_threshold": int(harness_cfg.failure_count_abort_threshold),
            "mode": "systemic_signature_distinctness",
            "systemic_rule": "same_signature && units>=3 && assets>=2 && candidates>=2",
            "tracked_signatures": int(len(failure_tracker)),
        },
        "quick_run": {
            "enabled": bool(quick_settings.enabled),
            "task_timeout_sec": int(quick_settings.task_timeout_sec),
            "progress_every_groups": int(quick_settings.progress_every_groups),
            "baseline_only": bool(quick_settings.baseline_only),
            "disable_cpcv": bool(quick_settings.disable_cpcv),
        },
        "memory": {
            "available_bytes": int(avail),
            "estimated_state_bytes": int(est_state),
            "budget_bytes": int(budget),
        },
        "feature_tensor": {
            "cache_npz": str(tensor_npz_path),
            "cache_manifest": str(tensor_json_path),
            "shape": list(feature_tensor.shape),
            "dtype": str(feature_tensor.dtype),
            "hash": str(tensor_hash),
            "compute_authoritative": False,
            "worker_role": "diagnostics_cache_only",
        },
        "micro_diagnostics": {
            "enabled": bool(harness_cfg.export_micro_diagnostics),
            "mode": str(harness_cfg.micro_diag_mode),
            "symbols_filter": list(harness_cfg.micro_diag_symbols),
            "session_ids_filter": [int(x) for x in harness_cfg.micro_diag_session_ids],
            "max_rows": int(harness_cfg.micro_diag_max_rows),
            "rows_exported": int(len(micro_df)),
            "profile_rows_exported": int(len(profile_df)),
            "funnel_rows_exported": int(len(funnel_df)),
        },
        "dq": {
            "report_path": str(dq_report_path),
            "bar_flags_path": str(dq_bar_flags_path),
            "n_day_rows": int(dq_day_df.shape[0]),
            "n_bar_flag_rows": int(dq_bar_df.shape[0]),
            "accept_count": int((dq_day_df.get("decision", pdx.Series(dtype=str)) == dq_accept).sum()) if dq_day_df.shape[0] > 0 else 0,
            "degrade_count": int((dq_day_df.get("decision", pdx.Series(dtype=str)) == dq_degrade).sum()) if dq_day_df.shape[0] > 0 else 0,
            "reject_count": int((dq_day_df.get("decision", pdx.Series(dtype=str)) == dq_reject).sum()) if dq_day_df.shape[0] > 0 else 0,
        },
    }
    write_json_fn(manifest_path, manifest)

    artifact_paths = {
        "equity_curves": str(equity_path),
        "trade_log": str(trade_path),
        "daily_returns": str(daily_path),
        "verdict": str(verdict_path),
        "stats_raw": str(stats_raw_path),
        "run_manifest": str(manifest_path),
        "strategy_results": str(strategy_ledger_path),
        "run_status": str(run_status_path),
        "leaderboard_csv": str(leaderboard_csv_path),
        "leaderboard_json": str(leaderboard_json_path),
        "robustness_leaderboard_csv": str(robustness_csv_path),
        "plateaus": str(plateaus_path),
        "deadletter_tasks": str(deadletter_path),
        "dq_report_csv": str(dq_report_path),
        "dq_bar_flags_parquet": str(dq_bar_flags_path),
        "validation_report": str(validation_report_path),
        "validation_report_latest": str(canonical_validation_report_path),
        "self_audit_report": str(self_audit_report_path),
    }
    if bool(harness_cfg.export_micro_diagnostics):
        artifact_paths["micro_diagnostics"] = str(micro_diag_path)
        if bool(harness_cfg.micro_diag_export_block_profiles):
            artifact_paths["micro_profile_blocks"] = str(micro_profile_blocks_path)
        if bool(harness_cfg.micro_diag_export_funnel):
            artifact_paths["funnel_1545"] = str(funnel_1545_path)

    return harness_output_cls(
        candidate_results=build_candidate_results(all_results, candidate_verdict),
        daily_returns_matrix=daily_mat_exec,
        daily_benchmark_returns=daily_bmk,
        stats_verdict=stats_verdict,
        artifact_paths=artifact_paths,
        run_manifest=manifest,
    )
