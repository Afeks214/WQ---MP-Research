"""
Weightiz Institutional Engine - Module 5 Part 2 (Validation Harness)
=====================================================================

Validation harness and research orchestrator:
- Pandas IO boundary for minute OHLCV ingestion/alignment.
- Leakage-safe WF/CPCV split generation with purge+embargo.
- Adversarial stress perturbations on per-group stressed state views.
- Deterministic orchestration of stressed state -> Module 2 -> Module 3 -> Module 4.
- Close-to-close daily return compression for candidate equity (overnight PnL preserved).
- Artifact export and statistical verdict wiring to Module 5 Part 1.

Important truth surfaces:
- Worker compute authority is the stressed split-masked state path.
- The published feature tensor and shared-memory store are diagnostics/cache only.
- Group-bound execution reuses post-M2/post-M3 state within one semantic group task.

Architecture map:
1) Split construction: `_generate_wf_splits`, `_generate_cpcv_splits`.
2) Task dispatch: group-bound semantic task planning, `_safe_execute_task`, process pool/serial loop in `run_weightiz_harness`.
3) Candidate aggregation: `_aggregate_candidate_baseline_matrix` + `_build_candidate_artifacts`.
4) Artifact writing: `run_weightiz_harness` (run-level) + `_build_candidate_artifacts` (candidate-level).
5) Robustness scoring: `_build_candidate_artifacts` using `ROBUSTNESS_CAPS`.
6) Plateau clustering: `_plateau_key` + deterministic grouping in `_build_candidate_artifacts`.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import dataclasses
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
import atexit
import hashlib
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback
import warnings
from queue import SimpleQueue
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover - runtime guard
    pd = None  # type: ignore[assignment]

from weightiz.module5.stage_a_discovery import STAGE_A_RESEARCH_THRESHOLD

_MANIFEST_REQUIRED_KEYS = (
    "git_commit",
    "config_hash",
    "dataset_hash",
    "search_seed",
    "asset_count",
    "strategy_count",
    "runtime_seconds",
    "start_time",
    "end_time",
)
_MANIFEST_TRUTH_SURFACES = (
    "dataset_hash",
    "feature_tensor",
)

from weightiz.module5.harness.aggregation import (
    aggregate_candidate_baseline_matrices as _aggregation_aggregate_candidate_baseline_matrices,
    aggregate_candidate_baseline_matrix as _aggregation_aggregate_candidate_baseline_matrix,
)
from weightiz.module5.harness.artifact_writers import (
    atomic_write_parquet as _artifact_atomic_write_parquet,
    to_jsonable as _artifact_to_jsonable,
    write_frozen_json as _artifact_write_frozen_json,
    write_json as _artifact_write_json,
)
from weightiz.module5.harness.candidate_artifacts import (
    build_candidate_artifacts as _candidate_build_candidate_artifacts,
    collect_ledger_rows_from_results as _candidate_collect_ledger_rows_from_results,
    plateau_key as _candidate_plateau_key,
    stack_payload_frames as _candidate_stack_payload_frames,
    summarize_fold_stats as _candidate_summarize_fold_stats,
)
from weightiz.module5.harness.evaluation_path import (
    assert_artifact_dependency_contract as _eval_assert_artifact_dependency_contract,
    asset_pnl_by_symbol_from_state as _eval_asset_pnl_by_symbol_from_state,
    benchmark_daily_returns as _eval_benchmark_daily_returns,
    canonical_artifact_dependency_matrix as _eval_canonical_artifact_dependency_matrix,
    candidate_daily_returns_close_to_close as _eval_candidate_daily_returns_close_to_close,
    collect_funnel_payload as _eval_collect_funnel_payload,
    collect_micro_diagnostics_payload as _eval_collect_micro_diagnostics_payload,
    collect_micro_profile_blocks_payload as _eval_collect_micro_profile_blocks_payload,
    equity_curve_payload as _eval_equity_curve_payload,
    event_window_mask as _eval_event_window_mask,
    materialize_risk_outputs_into_state as _eval_materialize_risk_outputs_into_state,
    select_micro_rows as _eval_select_micro_rows,
    structural_weight_from_regime as _eval_structural_weight_from_regime,
    trade_log_payload as _eval_trade_log_payload,
    validate_canonical_artifact_dependencies as _eval_validate_canonical_artifact_dependencies,
)
from weightiz.module5.harness.group_executor import (
    GroupExecutionResult,
    GroupExecutionRuntimeStats,
    GroupExecutionTask,
    build_base_group_execution_tasks as _group_build_base_execution_tasks,
    build_group_execution_tasks as _group_build_execution_tasks,
    chunk_group_execution_task as _group_chunk_execution_task,
    order_group_execution_tasks as _group_order_execution_tasks,
)
from weightiz.module5.harness.failure_policy import (
    baseline_failure_reasons as _failure_baseline_failure_reasons,
    build_risk_constraint_state_dump as _failure_build_risk_constraint_state_dump,
    error_hash as _failure_error_hash,
    exception_signature as _failure_exception_signature,
    extract_breach_index as _failure_extract_breach_index,
    is_high_suspicion_exception as _failure_is_high_suspicion_exception,
    is_localized_reason_codes as _failure_is_localized_reason_codes,
    is_risk_constraint_breach as _failure_is_risk_constraint_breach,
    normalized_top_frame as _failure_normalized_top_frame,
    record_deadletter as _failure_record_deadletter,
    should_abort_run as _failure_should_abort_run,
    should_abort_systemic as _failure_should_abort_systemic,
    update_failure_tracker as _failure_update_failure_tracker,
)
from weightiz.module5.harness.memory_accounting import (
    build_memory_accounting_estimate as _mem_build_memory_accounting_estimate,
    chunk_policy_memory_cap as _mem_chunk_policy_memory_cap,
    estimate_queue_bytes as _mem_estimate_queue_bytes,
    estimate_result_buffer_bytes as _mem_estimate_result_buffer_bytes,
    estimate_worker_pool_capacity as _mem_estimate_worker_pool_capacity,
    resolve_base_sharing_mode as _mem_resolve_base_sharing_mode,
)
from weightiz.module5.harness.module6_bridge import (
    AVAIL_INVALIDATED_BY_DQ as _MODULE6_AVAIL_INVALIDATED_BY_DQ,
    AVAIL_OBSERVED_ACTIVE as _MODULE6_AVAIL_OBSERVED_ACTIVE,
    AVAIL_OBSERVED_FLAT as _MODULE6_AVAIL_OBSERVED_FLAT,
)
from weightiz.module5.harness.runtime_truth import (
    build_compute_authority as _truth_build_compute_authority,
    build_execution_topology as _truth_build_execution_topology,
    build_feature_tensor_role as _truth_build_feature_tensor_role,
)
from weightiz.module5.harness.state_overlay import (
    BaseTensorState,
    CandidateScratch,
    CombinedStateView,
    FeatureOverlay,
    MarketOverlay,
    measure_module3_output_bytes,
    validate_candidate_execution_view,
)
from weightiz.module5.harness.orchestrator_support import (
    finalize_run_outputs as _orchestrator_finalize_run_outputs,
)
from weightiz.module5.harness.robustness_support import compute_stats_verdict as _robustness_compute_stats_verdict
from weightiz.module5.harness.worker_runtime import (
    run_group_task_from_context as _worker_run_group_task_from_context,
    safe_execute_task as _worker_safe_execute_task,
)
from weightiz.module5.harness.ingestion import (
    build_clock_override_from_utc as _ingest_build_clock_override_from_utc,
    ingest_master_aligned as _ingest_ingest_master_aligned,
    load_asset_frame as _ingest_load_asset_frame,
    validate_utc_minute_index as _ingest_validate_utc_minute_index,
)
from weightiz.module5.harness.metrics_support import (
    apply_latency_to_target_qty as _metrics_apply_latency_to_target_qty,
    asset_notional_concentration_from_trade_payloads as _metrics_asset_notional_concentration_from_trade_payloads,
    asset_pnl_concentration_from_result_rows as _metrics_asset_pnl_concentration_from_result_rows,
    clip01 as _metrics_clip01,
    cum_return as _metrics_cum_return,
    effective_benchmark_for_horizon as _metrics_effective_benchmark_for_horizon,
    extract_final_equity as _metrics_extract_final_equity,
    margin_exposure_stats_from_equity_payloads as _metrics_margin_exposure_stats_from_equity_payloads,
    max_drawdown_from_returns as _metrics_max_drawdown_from_returns,
    resample_returns_horizon as _metrics_resample_returns_horizon,
    sharpe_daily as _metrics_sharpe_daily,
    slice_score_from_stats as _metrics_slice_score_from_stats,
    trade_count_from_payload as _metrics_trade_count_from_payload,
    turnover_from_trade_payload as _metrics_turnover_from_trade_payload,
)
from weightiz.module5.harness.invariants import (
    apply_enabled_assets as _inv_apply_enabled_assets,
    apply_post_m2_invariants as _inv_apply_post_m2_invariants,
    apply_post_m3_invariants as _inv_apply_post_m3_invariants,
    apply_pre_m4_invariants as _inv_apply_pre_m4_invariants,
    assert_active_domain_ohlc as _inv_assert_active_domain_ohlc,
    validate_loaded_market_slice_active_domain as _inv_validate_loaded_market_slice_active_domain,
)
from weightiz.module5.harness.splits import (
    apply_purge_embargo as _splits_apply_purge_embargo,
    build_candidate_specs_default as _splits_build_candidate_specs_default,
    build_group_tasks as _splits_build_group_tasks,
    contiguous_segments as _splits_contiguous_segments,
    default_stress_scenarios as _splits_default_stress_scenarios,
    generate_cpcv_splits as _splits_generate_cpcv_splits,
    generate_quick_fallback_split as _splits_generate_quick_fallback_split,
    generate_wf_splits as _splits_generate_wf_splits,
    normalize_candidate_specs as _splits_normalize_candidate_specs,
    session_bounds as _splits_session_bounds,
    sessions_to_idx as _splits_sessions_to_idx,
    validate_split as _splits_validate_split,
)
from weightiz.module5.harness.stress import (
    apply_jitter as _stress_apply_jitter,
    apply_missing_bursts as _stress_apply_missing_bursts,
    assert_placeholder_consistency as _stress_assert_placeholder_consistency,
    compute_bar_valid as _stress_compute_bar_valid,
    recompute_bar_valid_inplace as _stress_recompute_bar_valid_inplace,
    set_placeholders_from_bar_valid as _stress_set_placeholders_from_bar_valid,
)

from weightiz.module1.core import (
    EngineConfig,
    FeatureEngineConfig,
    FeatureSpec,
    Phase,
    ProfileStatIdx,
    ScoreIdx,
    TensorState,
    build_feature_tensor_from_state,
    make_compat_feature_specs,
    preallocate_state,
    validate_loaded_market_slice,
    validate_state_hard,
)
from weightiz.module2.core import Module2Config, run_weightiz_profile_engine
from weightiz.module3.bridge import (
    IB_MISSING_POLICY,
    IB_POLICY_NO_TRADE,
    ContextIdx,
    Module3Config,
    Module3Output,
    run_module3_structural_aggregation,
)
from weightiz.module4.strategy_funnel import (
    Module4Config,
    Module4SignalOutput,
    RegimeIdx,
    run_module4_signal_funnel,
)
from weightiz.module5.stats import run_full_stats
from weightiz.shared.validation.dq import DQ_ACCEPT, DQ_DEGRADE, DQ_REJECT, dq_apply, dq_validate
from weightiz.shared.validation.invariants import assert_or_flag_finite
from weightiz.shared.validation.dtype_guard import assert_float64
from weightiz.shared.io.feature_tensor_cache import (
    PROFILE_CACHE_SCHEMA_VERSION,
    build_manifest as build_feature_manifest,
    cleanup_stale_tmp_cache_files,
    compute_tensor_hash,
    load_tensor_cache,
    profile_cache_paths,
    save_tensor_cache,
)
from weightiz.module2.core import (
    compute_window_correlation_diagnostics,
    validate_feature_tensor_contract,
)
from weightiz.shared.io.shared_feature_store import (
    SharedFeatureHandles,
    SharedFeatureRegistry,
    attach_shared_feature_store,
    cleanup_orphan_shared_memory_segments,
    close_shared_feature_store,
    create_shared_feature_store,
    enforce_memory_safety,
    estimate_tensor_bytes,
)
from weightiz.module4.risk_engine import CostConfig, RiskConfig, simulate_portfolio_from_signals
from weightiz.shared.logging.runtime_monitor import RuntimeMonitor
from weightiz.shared.logging.system_logger import configure_worker_logging, get_logger, init_runtime_logger, log_event


def run_module4_strategy_funnel(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError("MODULE4_EXECUTION_FORBIDDEN_IN_CANONICAL_PATH")


ROBUSTNESS_CAPS: dict[str, float] = {
    "dd_cap": 0.35,
    "std_cap": 1.00,
    "conc_cap": 0.75,
}


@dataclass(frozen=True)
class Module5HarnessConfig:
    seed: int = 97
    research_mode: str = "standard"
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
    parallel_workers: int = max(1, (os.cpu_count() or 2) - 1)
    stress_profile: str = "baseline_mild_severe"
    max_ram_utilization_frac: float = 0.70
    enforce_lookahead_guard: bool = True
    report_dir: str = "./artifacts/module5_harness"
    fail_on_non_finite: bool = True
    daily_return_min_days: int = 60
    benchmark_symbol: str = "SPY"
    export_micro_diagnostics: bool = False
    micro_diag_mode: str = "events_only"
    micro_diag_symbols: tuple[str, ...] = ()
    micro_diag_session_ids: tuple[int, ...] = ()
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
    scratch_mode: str = "auto"
    strict_candidate_state_validation: str = "compact_execution_view"
    risk_breach_state_dump_enabled: bool = False
    debug_full_state_payloads: bool = False
    module3_output_mode: str = "full_legacy"
    # Configured base-state transport mode:
    # - auto: resolve to fork_cow only on linux+fork, otherwise fallback-only serialized_copy
    # - fork_cow: require copy-on-write transport
    # - explicit_shm: reserved and fail-closed until implemented
    base_sharing_mode: str = "auto"
    cow_private_ratio_threshold: float = 0.10
    cow_probe_workers: int = 8
    safety_margin_frac: float = 0.15
    safety_margin_min_bytes: int = 8 * 1024**3
    max_queue_bytes_frac: float = 0.02
    max_result_buffer_bytes_frac: float = 0.05
    throughput_minimal_observability: bool = False
    health_check_interval: int = 50
    progress_interval_seconds: int = 10
    # Test-only deterministic fault hooks.
    test_fail_task_ids: tuple[str, ...] = ()
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
    horizon_minutes: tuple[int, ...] = (1, 5, 15, 60)
    robustness_weight_dsr: float = 0.20
    robustness_weight_pbo: float = 0.15
    robustness_weight_spa: float = 0.10
    robustness_weight_regime: float = 0.20
    robustness_weight_execution: float = 0.20
    robustness_weight_horizon: float = 0.15
    robustness_reject_threshold: float = STAGE_A_RESEARCH_THRESHOLD
    execution_fragile_threshold: float = 0.50


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    m2_idx: int
    m3_idx: int
    m4_idx: int
    enabled_assets_mask: np.ndarray
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class SplitSpec:
    split_id: str
    mode: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    purge_idx: np.ndarray
    embargo_idx: np.ndarray
    session_train_bounds: tuple[int, int]
    session_test_bounds: tuple[int, int]
    purge_bars: int = 0
    embargo_bars: int = 0
    total_bars: int = 0


@dataclass(frozen=True)
class StressScenario:
    scenario_id: str
    name: str
    missing_burst_prob: float
    missing_burst_min: int
    missing_burst_max: int
    jitter_sigma_bps: float
    slippage_mult: float
    enabled: bool = True


@dataclass
class HarnessOutput:
    candidate_results: list[dict[str, object]]
    daily_returns_matrix: np.ndarray
    daily_benchmark_returns: np.ndarray
    stats_verdict: dict[str, object]
    artifact_paths: dict[str, str]
    run_manifest: dict[str, object]


_GroupTask = GroupExecutionTask


@dataclass(frozen=True)
class _ExecutionView:
    regime_primary_ta: np.ndarray
    regime_confidence_ta: np.ndarray
    intent_long_ta: np.ndarray
    intent_short_ta: np.ndarray
    target_qty_ta: np.ndarray
    filled_qty_ta: np.ndarray
    exec_price_ta: np.ndarray
    trade_cost_ta: np.ndarray
    overnight_score_ta: np.ndarray
    overnight_winner_t: np.ndarray
    kill_switch_t: np.ndarray


_WORKER_CONTEXT: dict[str, Any] | None = None
_WORKER_PROCESS: bool = False


@dataclass(frozen=True)
class _QuickRunSettings:
    enabled: bool
    task_timeout_sec: int
    progress_every_groups: int
    baseline_only: bool
    disable_cpcv: bool


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in {"1", "true", "yes", "y", "t", "on"}


def _quick_run_settings_from_env() -> _QuickRunSettings:
    enabled = _env_flag("QUICK_RUN", default=False)
    timeout_raw = str(os.environ.get("QUICK_RUN_TASK_TIMEOUT_SEC", "120")).strip()
    progress_raw = str(os.environ.get("QUICK_RUN_PROGRESS_EVERY", "1")).strip()
    try:
        timeout_sec = max(1, int(timeout_raw))
    except Exception:
        timeout_sec = 120
    try:
        progress_every = max(1, int(progress_raw))
    except Exception:
        progress_every = 1
    return _QuickRunSettings(
        enabled=bool(enabled),
        task_timeout_sec=int(timeout_sec),
        progress_every_groups=int(progress_every),
        baseline_only=True,
        disable_cpcv=True,
    )


def _run_with_timeout_alarm(seconds: int, fn: Callable[[], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    sec = int(max(0, seconds))
    if sec <= 0 or not hasattr(signal, "setitimer"):
        return fn()

    old_handler = signal.getsignal(signal.SIGALRM)

    def _handler(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"quick_run_task_timeout>{sec}s")

    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(sec))
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


def _require_pandas() -> Any:
    if pd is None:
        raise RuntimeError("pandas is required for Module 5 harness IO/export boundary")
    return pd


def _to_jsonable(obj: Any) -> Any:
    return _artifact_to_jsonable(obj)


def _stable_hash_obj(obj: Any) -> str:
    payload = json.dumps(_to_jsonable(obj), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _git_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def _seed_for_task(base_seed: int, *parts: str) -> int:
    h = hashlib.sha256()
    h.update(str(base_seed).encode("utf-8"))
    for p in parts:
        h.update(b"|")
        h.update(p.encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "little", signed=False) % (2**32 - 1)


def _error_hash(error_type: str, error_msg: str) -> str:
    return _failure_error_hash(error_type, error_msg)


def _record_deadletter(path: Path, row: dict[str, Any]) -> None:
    _failure_record_deadletter(path, row)


def _should_abort_run(
    failure_count: int,
    total_tasks: int,
    failure_rate_threshold: float,
    failure_count_threshold: int,
) -> tuple[bool, str]:
    return _failure_should_abort_run(
        failure_count=failure_count,
        total_tasks=total_tasks,
        failure_rate_threshold=failure_rate_threshold,
        failure_count_threshold=failure_count_threshold,
    )


def _normalized_top_frame(traceback_text: str) -> str:
    return _failure_normalized_top_frame(traceback_text)


def _exception_signature(row: dict[str, Any]) -> tuple[str, str]:
    return _failure_exception_signature(row)


def _is_localized_reason_codes(reason_codes: list[str]) -> bool:
    return _failure_is_localized_reason_codes(reason_codes)


def _is_risk_constraint_breach(error_type: str, error_msg: str, top_frame: str, traceback_text: str = "") -> bool:
    return _failure_is_risk_constraint_breach(error_type, error_msg, top_frame, traceback_text)


def _baseline_failure_reasons(
    rows_base_all: list[dict[str, Any]],
    expected_baseline_tasks: int,
) -> list[str]:
    return _failure_baseline_failure_reasons(rows_base_all, expected_baseline_tasks)


def _extract_breach_index(error_msg: str, state: TensorState | None) -> int:
    return _failure_extract_breach_index(error_msg, state)


def _build_risk_constraint_state_dump(
    state: TensorState,
    t: int,
    candidate_id: str,
    split_id: str,
    scenario_id: str,
) -> dict[str, Any]:
    return _failure_build_risk_constraint_state_dump(state, t, candidate_id, split_id, scenario_id)


def _is_high_suspicion_exception(error_type: str) -> bool:
    return _failure_is_high_suspicion_exception(error_type)


def _update_failure_tracker(
    tracker: dict[tuple[str, str], dict[str, set[str] | bool]],
    row: dict[str, Any],
) -> tuple[tuple[str, str], dict[str, set[str] | bool]]:
    return _failure_update_failure_tracker(tracker, row)


def _should_abort_systemic(
    tracker: dict[tuple[str, str], dict[str, set[str] | bool]],
    row: dict[str, Any],
) -> tuple[bool, str]:
    return _failure_should_abort_systemic(tracker, row)


def _is_large_payload_safe(tasks: list[_GroupTask], threshold_bytes: int) -> tuple[bool, int]:
    import pickle

    if not tasks:
        return True, 0
    sample = tasks[: min(len(tasks), 32)]
    sizes = [len(pickle.dumps(t, protocol=pickle.HIGHEST_PROTOCOL)) for t in sample]
    mx = int(max(sizes))
    return bool(mx <= int(threshold_bytes)), mx


def _resolve_mp_context() -> tuple[Any, str]:
    """
    Deterministic process start policy:
    - honor WEIGHTIZ_MP_START_METHOD when explicitly set to spawn|fork|forkserver
    - default to fork on macOS to avoid large spawn bootstrap stalls
    - otherwise use interpreter default for the active Python/runtime
    """
    forced = str(os.environ.get("WEIGHTIZ_MP_START_METHOD", "")).strip().lower()
    if forced in {"spawn", "fork", "forkserver"}:
        return mp.get_context(forced), forced
    if sys.platform == "darwin":
        return mp.get_context("fork"), "fork"
    default_ctx = mp.get_context()
    return default_ctx, str(default_ctx.get_start_method())


def _safe_execute_task(
    group: _GroupTask,
    executor_fn: Callable[[_GroupTask], list[dict[str, Any]]],
    candidates: list[CandidateSpec],
    splits: list[SplitSpec],
    scenarios: list[StressScenario],
    harness_cfg: Module5HarnessConfig,
) -> list[dict[str, Any]]:
    return _worker_safe_execute_task(
        group=group,
        executor_fn=executor_fn,
        candidates=candidates,
        splits=splits,
        scenarios=scenarios,
        harness_cfg=harness_cfg,
        error_hash_fn=_error_hash,
        normalized_top_frame_fn=_normalized_top_frame,
        seed_for_task_fn=_seed_for_task,
    )


def _init_worker_context(
    base_state: Any,
    candidates: list[CandidateSpec],
    splits: list[SplitSpec],
    scenarios: list[StressScenario],
    m2_configs: list[Module2Config],
    m3_configs: list[Module3Config],
    m4_configs: list[Module4Config],
    harness_cfg: Module5HarnessConfig,
    feature_registry: SharedFeatureRegistry | None,
    log_queue: mp.Queue | None = None,
    run_id: str = "",
) -> None:
    global _WORKER_CONTEXT, _WORKER_PROCESS
    _WORKER_PROCESS = True
    # Keep worker BLAS/Arrow threading deterministic and avoid nested oversubscription.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("ARROW_NUM_THREADS", "1")
    os.environ["WEIGHTIZ_WORKER_PROCESS"] = "1"
    if log_queue is not None:
        configure_worker_logging(log_queue, level="INFO")
    feature_handles: SharedFeatureHandles | None = None
    if feature_registry is not None:
        feature_handles = attach_shared_feature_store(feature_registry)
        assert_float64("worker.shared_feature_store", feature_handles.array)
        atexit.register(lambda: close_shared_feature_store(feature_handles, is_master=False))
    _WORKER_CONTEXT = {
        "base_state": base_state,
        "candidates": candidates,
        "splits": splits,
        "scenarios": scenarios,
        "m2_configs": m2_configs,
        "m3_configs": m3_configs,
        "m4_configs": m4_configs,
        "harness_cfg": harness_cfg,
        # The shared feature tensor is published for diagnostics/cache visibility only.
        # Worker execution remains authoritative on the stressed cloned TensorState path.
        "shared_feature_tensor_role": "diagnostics_cache_only",
        "feature_registry": feature_registry,
        "feature_handles": feature_handles,
        "run_id": str(run_id),
        "base_sharing_mode": str(getattr(harness_cfg, "base_sharing_mode", "auto")),
    }


def _run_group_task_from_context(group: _GroupTask) -> list[dict[str, Any]]:
    return _worker_run_group_task_from_context(
        group,
        worker_context=_WORKER_CONTEXT,
        run_group_task_fn=_run_group_task,
    )


def _assert_finite(name: str, arr: np.ndarray) -> None:
    if not np.all(np.isfinite(arr)):
        bad = np.argwhere(~np.isfinite(arr))[:8]
        raise RuntimeError(f"{name} contains non-finite values at indices {bad.tolist()}")


def _recompute_module2_on_stressed_state(state: TensorState, cfg: Module2Config) -> None:
    env_key = "WEIGHTIZ_ALLOW_CANONICAL_HARNESS_MODULE2"
    prev = os.environ.get(env_key)
    os.environ[env_key] = "1"
    try:
        run_weightiz_profile_engine(state, cfg)
    finally:
        if prev is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = prev

    expected_profile_shape = (int(state.cfg.T), int(state.cfg.A), int(ProfileStatIdx.N_FIELDS))
    expected_score_shape = (int(state.cfg.T), int(state.cfg.A), int(ScoreIdx.N_FIELDS))
    if tuple(np.asarray(state.profile_stats).shape) != expected_profile_shape:
        raise RuntimeError(
            f"stressed module2 profile_stats shape mismatch: got {np.asarray(state.profile_stats).shape}, "
            f"expected {expected_profile_shape}"
        )
    if tuple(np.asarray(state.scores).shape) != expected_score_shape:
        raise RuntimeError(
            f"stressed module2 scores shape mismatch: got {np.asarray(state.scores).shape}, "
            f"expected {expected_score_shape}"
        )
    assert_float64("harness.stressed_module2.profile_stats", np.asarray(state.profile_stats, dtype=np.float64))
    assert_float64("harness.stressed_module2.scores", np.asarray(state.scores, dtype=np.float64))


def _tensor_nbytes_total(obj: Any, _seen: set[int] | None = None) -> int:
    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:
        return 0
    _seen.add(oid)
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if dataclasses.is_dataclass(obj):
        total = 0
        for field in dataclasses.fields(obj):
            total += _tensor_nbytes_total(getattr(obj, field.name), _seen)
        return int(total)
    if isinstance(obj, dict):
        return int(sum(_tensor_nbytes_total(v, _seen) for v in obj.values()))
    if isinstance(obj, (list, tuple)):
        return int(sum(_tensor_nbytes_total(v, _seen) for v in obj))
    return 0


def _available_memory_bytes() -> int:
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            return int(parts[1]) * 1024
        except Exception:
            pass
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        return page_size * pages
    except Exception:
        pass
    # Unix/macOS total-RAM fallback.
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        return page_size * pages
    except Exception:
        return 8 * 1024 * 1024 * 1024


def _estimate_state_bytes(base_state: TensorState) -> int:
    # Use real ndarray footprint from the current state, then budget for the
    # stressed clone and task-local buffers in process-pool execution.
    state_bytes = max(1, _tensor_nbytes_total(base_state))
    overhead = 512 * 1024 * 1024
    return int(state_bytes * 2 + overhead)


def _resolve_candidate_scratch_mode(harness_cfg: Module5HarnessConfig) -> str:
    mode = str(harness_cfg.scratch_mode).strip().lower()
    if mode == "auto":
        if bool(harness_cfg.risk_breach_state_dump_enabled) or bool(harness_cfg.debug_full_state_payloads):
            return "full"
        if str(harness_cfg.strict_candidate_state_validation).strip().lower() == "full_tensorstate":
            return "full"
        return "compact"
    if mode not in {"compact", "full"}:
        raise RuntimeError(f"Unsupported candidate scratch mode={harness_cfg.scratch_mode!r}")
    return mode


def _estimate_group_runtime_bytes(
    base_state: TensorState | BaseTensorState,
    harness_cfg: Module5HarnessConfig,
) -> tuple[int, int, int, int]:
    base = base_state if isinstance(base_state, BaseTensorState) else BaseTensorState.from_tensor_state(base_state)
    market_overlay = MarketOverlay.from_base(base)
    feature_overlay = FeatureOverlay.allocate(base, module3_output_mode=str(harness_cfg.module3_output_mode))
    scratch = CandidateScratch.allocate(base, _resolve_candidate_scratch_mode(harness_cfg))
    module3_bytes = int(max(1, int(harness_cfg.startup_default_module3_bytes)))
    return (
        int(market_overlay.nbytes),
        int(feature_overlay.feature_bytes),
        int(module3_bytes),
        int(scratch.nbytes),
    )




def _load_asset_frame(path: str, tz_name: str) -> Any:
    return _ingest_load_asset_frame(path, tz_name, require_pandas_fn=_require_pandas)


def _validate_utc_minute_index(idx: Any, label: str) -> Any:
    return _ingest_validate_utc_minute_index(idx, label, require_pandas_fn=_require_pandas)


def _build_clock_override_from_utc(
    ts_ns_utc: np.ndarray,
    cfg: EngineConfig,
    tz_name: str,
) -> dict[str, np.ndarray]:
    class _PhaseEnum:  # bridge enum values without moving public constants
        WARMUP = Phase.WARMUP
        LIVE = Phase.LIVE
        OVERNIGHT_SELECT = Phase.OVERNIGHT_SELECT
        FLATTEN = Phase.FLATTEN

    return _ingest_build_clock_override_from_utc(ts_ns_utc, cfg, tz_name, phase_enum=_PhaseEnum)


def _compute_bar_valid(
    open_px: np.ndarray,
    high_px: np.ndarray,
    low_px: np.ndarray,
    close_px: np.ndarray,
    volume: np.ndarray,
) -> np.ndarray:
    return _stress_compute_bar_valid(open_px, high_px, low_px, close_px, volume)


def _ingest_master_aligned(
    data_paths: list[str],
    symbols: list[str],
    engine_cfg: EngineConfig,
    harness_cfg: Module5HarnessConfig,
    data_loader_func: Callable[[str, str], Any] | None = None,
) -> tuple[TensorState, np.ndarray, list[str], np.ndarray, dict[str, Any], np.ndarray, dict[str, Any]]:
    return _ingest_ingest_master_aligned(
        data_paths=data_paths,
        symbols=symbols,
        engine_cfg=engine_cfg,
        harness_cfg=harness_cfg,
        data_loader_func=data_loader_func,
        require_pandas_fn=_require_pandas,
        load_asset_frame_fn=_load_asset_frame,
        validate_utc_minute_index_fn=_validate_utc_minute_index,
        compute_bar_valid_fn=_compute_bar_valid,
        build_clock_override_from_utc_fn=_build_clock_override_from_utc,
        replace_fn=replace,
        preallocate_state_fn=preallocate_state,
        validate_loaded_market_slice_fn=validate_loaded_market_slice,
        validate_state_hard_fn=validate_state_hard,
        dq_validate_fn=dq_validate,
        dq_apply_fn=dq_apply,
        dq_accept=DQ_ACCEPT,
        dq_degrade=DQ_DEGRADE,
        dq_reject=DQ_REJECT,
    )


def _session_bounds(session_id: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _splits_session_bounds(session_id)


def _sessions_to_idx(session_id: np.ndarray, sessions: np.ndarray) -> np.ndarray:
    return _splits_sessions_to_idx(session_id, sessions)


def _contiguous_segments(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _splits_contiguous_segments(idx)


def _apply_purge_embargo(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    T: int,
    purge_bars: int,
    embargo_bars: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _splits_apply_purge_embargo(train_idx, test_idx, T, purge_bars, embargo_bars)


def _generate_wf_splits(state: TensorState, cfg: Module5HarnessConfig) -> list[SplitSpec]:
    return _splits_generate_wf_splits(state, cfg, split_spec_cls=SplitSpec)


def _generate_cpcv_splits(state: TensorState, cfg: Module5HarnessConfig) -> list[SplitSpec]:
    return _splits_generate_cpcv_splits(state, cfg, split_spec_cls=SplitSpec)


def _generate_quick_fallback_split(state: TensorState, cfg: Module5HarnessConfig) -> list[SplitSpec]:
    return _splits_generate_quick_fallback_split(state, cfg, split_spec_cls=SplitSpec)


def _validate_split(spec: SplitSpec, enforce_guard: bool) -> None:
    return _splits_validate_split(spec, enforce_guard, contiguous_segments_fn=_contiguous_segments)


def _default_stress_scenarios(cfg: Module5HarnessConfig) -> list[StressScenario]:
    return _splits_default_stress_scenarios(cfg, stress_scenario_cls=StressScenario)


def _apply_split_domain_mask(state: TensorState, split: SplitSpec) -> np.ndarray:
    T = state.cfg.T
    active_t = np.zeros(T, dtype=bool)
    active_t[split.train_idx] = True
    active_t[split.test_idx] = True

    inactive = ~active_t
    if np.any(inactive):
        state.open_px[inactive] = np.nan
        state.high_px[inactive] = np.nan
        state.low_px[inactive] = np.nan
        state.close_px[inactive] = np.nan
        state.volume[inactive] = np.nan
        state.rvol[inactive] = np.nan
        state.atr_floor[inactive] = np.nan
        state.bar_valid[inactive] = False

    return active_t


def _apply_missing_bursts(
    state: TensorState,
    active_t: np.ndarray,
    scenario: StressScenario,
    rng: np.random.Generator,
) -> None:
    _stress_apply_missing_bursts(state, active_t, scenario, rng)


def _apply_jitter(
    state: TensorState,
    active_t: np.ndarray,
    scenario: StressScenario,
    rng: np.random.Generator,
) -> None:
    _stress_apply_jitter(state, active_t, scenario, rng)


def _recompute_bar_valid_inplace(state: TensorState) -> None:
    _stress_recompute_bar_valid_inplace(state)


def _set_placeholders_from_bar_valid(state: TensorState) -> None:
    _stress_set_placeholders_from_bar_valid(state)


def _assert_placeholder_consistency(state: TensorState) -> None:
    _stress_assert_placeholder_consistency(state)


def _apply_post_m2_invariants(state: TensorState, active_t: np.ndarray) -> list[str]:
    return _inv_apply_post_m2_invariants(
        state,
        active_t,
        assert_or_flag_finite_fn=assert_or_flag_finite,
        set_placeholders_from_bar_valid_fn=_set_placeholders_from_bar_valid,
    )


def _apply_post_m3_invariants(m3: Module3Output) -> list[str]:
    return _inv_apply_post_m3_invariants(
        m3,
        assert_or_flag_finite_fn=assert_or_flag_finite,
        ib_missing_policy=IB_MISSING_POLICY,
        ib_policy_no_trade=IB_POLICY_NO_TRADE,
    )


def _apply_pre_m4_invariants(state: TensorState, m3: Module3Output) -> list[str]:
    return _inv_apply_pre_m4_invariants(
        state,
        m3,
        assert_or_flag_finite_fn=assert_or_flag_finite,
        set_placeholders_from_bar_valid_fn=_set_placeholders_from_bar_valid,
        ib_missing_policy=IB_MISSING_POLICY,
        ib_policy_no_trade=IB_POLICY_NO_TRADE,
    )


def _assert_active_domain_ohlc(state: TensorState, active_t: np.ndarray) -> None:
    return _inv_assert_active_domain_ohlc(state, active_t)


def _validate_loaded_market_slice_active_domain(state: TensorState, active_t: np.ndarray) -> None:
    return _inv_validate_loaded_market_slice_active_domain(
        state,
        active_t,
        contiguous_segments_fn=_contiguous_segments,
        validate_loaded_market_slice_fn=validate_loaded_market_slice,
    )


def _apply_enabled_assets(state: TensorState, m3: Module3Output, enabled_mask: np.ndarray) -> None:
    return _inv_apply_enabled_assets(state, m3, enabled_mask)


def _materialize_risk_outputs_into_state(
    state: TensorState,
    m4_sig: Module4SignalOutput,
    risk_res: Any,
) -> _ExecutionView:
    return _eval_materialize_risk_outputs_into_state(
        state=state,
        m4_sig=m4_sig,
        risk_res=risk_res,
        execution_view_cls=_ExecutionView,
    )


def _candidate_daily_returns_close_to_close(
    state: TensorState,
    split: SplitSpec,
    initial_cash: float,
    equity_curve: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _eval_candidate_daily_returns_close_to_close(
        state=state,
        split=split,
        initial_cash=initial_cash,
        session_bounds_fn=_session_bounds,
        equity_curve=equity_curve,
    )


def _asset_pnl_by_symbol_from_state(
    state: TensorState,
    split: SplitSpec,
) -> dict[str, float]:
    return _eval_asset_pnl_by_symbol_from_state(state, split)


def _benchmark_daily_returns(
    state: TensorState,
    benchmark_symbol: str,
) -> tuple[np.ndarray, np.ndarray]:
    return _eval_benchmark_daily_returns(
        state=state,
        benchmark_symbol=benchmark_symbol,
        session_bounds_fn=_session_bounds,
    )


def _build_candidate_specs_default(
    A: int,
    m2_configs: list[Module2Config],
    m3_configs: list[Module3Config],
    m4_configs: list[Module4Config],
) -> list[CandidateSpec]:
    return _splits_build_candidate_specs_default(
        A,
        m2_configs,
        m3_configs,
        m4_configs,
        candidate_spec_cls=CandidateSpec,
    )


def _normalize_candidate_specs(
    specs: list[CandidateSpec],
    keep_idx: np.ndarray,
    A_filtered: int,
    A_input: int,
) -> list[CandidateSpec]:
    return _splits_normalize_candidate_specs(
        specs,
        keep_idx,
        A_filtered,
        A_input,
        candidate_spec_cls=CandidateSpec,
    )


def _build_group_tasks(
    candidates: list[CandidateSpec],
    splits: list[SplitSpec],
    scenarios: list[StressScenario],
) -> list[_GroupTask]:
    return _splits_build_group_tasks(candidates, splits, scenarios, group_task_cls=_GroupTask)


def _split_group_tasks_by_candidate(
    group_tasks: list[_GroupTask],
    chunk_size: int = 1,
) -> list[_GroupTask]:
    """
    Legacy candidate-fanout helper retained for compatibility and focused tests.

    It is not the authoritative V2 hot path when group-bound execution is
    enabled. The live V2 runtime uses semantic base groups plus deterministic
    group-local chunking in module5.harness.group_executor.
    """
    n = int(max(1, chunk_size))
    out: list[_GroupTask] = []
    for g in group_tasks:
        cand = tuple(sorted(int(x) for x in g.candidate_indices))
        if len(cand) <= n:
            out.append(g)
            continue
        for i in range(0, len(cand), n):
            part = tuple(cand[i : i + n])
            out.append(
                _GroupTask(
                    group_id=f"{g.group_id}_p{i // n:03d}",
                    split_idx=int(g.split_idx),
                    scenario_idx=int(g.scenario_idx),
                    m2_idx=int(g.m2_idx),
                    m3_idx=int(g.m3_idx),
                    candidate_indices=part,
                )
            )
    return out


def _build_base_group_execution_tasks(
    candidates: list[CandidateSpec],
    splits: list[SplitSpec],
    scenarios: list[StressScenario],
) -> list[_GroupTask]:
    return [
        _GroupTask(
            group_id=str(task.group_id),
            split_idx=int(task.split_idx),
            scenario_idx=int(task.scenario_idx),
            m2_idx=int(task.m2_idx),
            m3_idx=int(task.m3_idx),
            candidate_indices=tuple(int(x) for x in task.candidate_indices),
            estimated_group_cost=int(task.estimated_group_cost),
            chunk_candidate_cap=int(task.chunk_candidate_cap),
            chunk_start=int(task.chunk_start),
            chunk_end=int(task.chunk_end),
            first_candidate_id=str(getattr(task, "first_candidate_id", "")),
            module3_group_bytes_estimated=int(getattr(task, "module3_group_bytes_estimated", 0)),
        )
        for task in _group_build_base_execution_tasks(candidates, splits, scenarios)
    ]


def _build_group_execution_tasks(
    candidates: list[CandidateSpec],
    splits: list[SplitSpec],
    scenarios: list[StressScenario],
    harness_cfg: Module5HarnessConfig,
    *,
    group_fixed_bytes: int,
    module3_group_bytes_estimated: int,
    max_group_memory_bytes: int | None = None,
    candidate_loop_sec_per_candidate_p95: float | None = None,
    result_payload_bytes_per_candidate_p95: int | None = None,
    candidate_incremental_bytes_p95: int | None = None,
) -> list[_GroupTask]:
    return [
        _GroupTask(
            group_id=str(task.group_id),
            split_idx=int(task.split_idx),
            scenario_idx=int(task.scenario_idx),
            m2_idx=int(task.m2_idx),
            m3_idx=int(task.m3_idx),
            candidate_indices=tuple(int(x) for x in task.candidate_indices),
            estimated_group_cost=int(task.estimated_group_cost),
            chunk_candidate_cap=int(task.chunk_candidate_cap),
            chunk_start=int(task.chunk_start),
            chunk_end=int(task.chunk_end),
            first_candidate_id=str(getattr(task, "first_candidate_id", "")),
            module3_group_bytes_estimated=int(getattr(task, "module3_group_bytes_estimated", 0)),
        )
        for task in _group_build_execution_tasks(
            candidates,
            splits,
            scenarios,
            dispatch_policy=str(harness_cfg.group_dispatch_policy),
            target_group_wall_time_sec=float(harness_cfg.group_target_wall_time_sec),
            max_result_payload_bytes=int(harness_cfg.group_max_result_payload_bytes),
            max_group_memory_bytes=int(
                max(
                    1,
                    max_group_memory_bytes
                    if max_group_memory_bytes is not None
                    else (harness_cfg.group_max_memory_bytes or group_fixed_bytes * max(2, int(harness_cfg.parallel_workers))),
                )
            ),
            min_candidates_per_chunk=int(harness_cfg.group_min_candidates_per_chunk),
            max_candidates_per_chunk_hard=int(max(1, harness_cfg.group_max_candidates_per_chunk_hard)),
            startup_default_candidate_loop_sec=float(harness_cfg.startup_default_candidate_loop_sec),
            startup_default_result_payload_bytes=int(harness_cfg.startup_default_result_payload_bytes),
            startup_default_candidate_incremental_bytes=int(harness_cfg.startup_default_candidate_incremental_bytes),
            group_fixed_bytes=int(group_fixed_bytes),
            module3_group_bytes_estimated=int(module3_group_bytes_estimated),
            candidate_loop_sec_per_candidate_p95=candidate_loop_sec_per_candidate_p95,
            result_payload_bytes_per_candidate_p95=result_payload_bytes_per_candidate_p95,
            candidate_incremental_bytes_p95=candidate_incremental_bytes_p95,
        )
    ]


def _chunk_group_execution_task(
    base_group: _GroupTask,
    candidates: list[CandidateSpec],
    harness_cfg: Module5HarnessConfig,
    *,
    group_fixed_bytes: int,
    module3_group_bytes_estimated: int,
    max_group_memory_bytes: int,
    candidate_loop_sec_per_candidate_p95: float | None = None,
    result_payload_bytes_per_candidate_p95: int | None = None,
    candidate_incremental_bytes_p95: int | None = None,
) -> list[_GroupTask]:
    return [
        _GroupTask(
            group_id=str(task.group_id),
            split_idx=int(task.split_idx),
            scenario_idx=int(task.scenario_idx),
            m2_idx=int(task.m2_idx),
            m3_idx=int(task.m3_idx),
            candidate_indices=tuple(int(x) for x in task.candidate_indices),
            estimated_group_cost=int(task.estimated_group_cost),
            chunk_candidate_cap=int(task.chunk_candidate_cap),
            chunk_start=int(task.chunk_start),
            chunk_end=int(task.chunk_end),
            first_candidate_id=str(getattr(task, "first_candidate_id", "")),
            module3_group_bytes_estimated=int(getattr(task, "module3_group_bytes_estimated", 0)),
        )
        for task in _group_chunk_execution_task(
            base_group,
            candidates,
            module3_group_bytes_estimated=int(module3_group_bytes_estimated),
            target_group_wall_time_sec=float(harness_cfg.group_target_wall_time_sec),
            max_result_payload_bytes=int(harness_cfg.group_max_result_payload_bytes),
            max_group_memory_bytes=int(max(1, max_group_memory_bytes)),
            min_candidates_per_chunk=int(harness_cfg.group_min_candidates_per_chunk),
            max_candidates_per_chunk_hard=int(max(1, harness_cfg.group_max_candidates_per_chunk_hard)),
            startup_default_candidate_loop_sec=float(harness_cfg.startup_default_candidate_loop_sec),
            startup_default_result_payload_bytes=int(harness_cfg.startup_default_result_payload_bytes),
            startup_default_candidate_incremental_bytes=int(harness_cfg.startup_default_candidate_incremental_bytes),
            group_fixed_bytes=int(group_fixed_bytes),
            candidate_loop_sec_per_candidate_p95=candidate_loop_sec_per_candidate_p95,
            result_payload_bytes_per_candidate_p95=result_payload_bytes_per_candidate_p95,
            candidate_incremental_bytes_p95=candidate_incremental_bytes_p95,
        )
    ]


def _order_group_execution_tasks(group_tasks: list[_GroupTask], dispatch_policy: str) -> list[_GroupTask]:
    return [
        _GroupTask(
            group_id=str(task.group_id),
            split_idx=int(task.split_idx),
            scenario_idx=int(task.scenario_idx),
            m2_idx=int(task.m2_idx),
            m3_idx=int(task.m3_idx),
            candidate_indices=tuple(int(x) for x in task.candidate_indices),
            estimated_group_cost=int(task.estimated_group_cost),
            chunk_candidate_cap=int(task.chunk_candidate_cap),
            chunk_start=int(task.chunk_start),
            chunk_end=int(task.chunk_end),
            first_candidate_id=str(getattr(task, "first_candidate_id", "")),
            module3_group_bytes_estimated=int(getattr(task, "module3_group_bytes_estimated", 0)),
        )
        for task in _group_order_execution_tasks(group_tasks, dispatch_policy=str(dispatch_policy))
    ]


def _equity_curve_payload(
    state: TensorState,
    candidate_id: str,
    split_id: str,
    scenario_id: str,
) -> dict[str, np.ndarray]:
    return _eval_equity_curve_payload(state, candidate_id, split_id, scenario_id)


def _trade_log_payload(
    state: TensorState,
    m4_out: _ExecutionView,
    candidate_id: str,
    split_id: str,
    scenario_id: str,
    eps: float = 1e-12,
) -> dict[str, np.ndarray]:
    return _eval_trade_log_payload(state, m4_out, candidate_id, split_id, scenario_id, eps=eps)


def _session_notional_by_trade_payload(trade_payload: dict[str, np.ndarray] | None) -> dict[int, float]:
    if not isinstance(trade_payload, dict):
        return {}
    session_id = np.asarray(trade_payload.get("session_id", np.zeros(0, dtype=np.int64)), dtype=np.int64)
    filled_qty = np.asarray(trade_payload.get("filled_qty", np.zeros(0, dtype=np.float64)), dtype=np.float64)
    exec_price = np.asarray(trade_payload.get("exec_price", np.zeros(0, dtype=np.float64)), dtype=np.float64)
    n = min(session_id.size, filled_qty.size, exec_price.size)
    if n <= 0:
        return {}
    out: dict[int, float] = {}
    for i in range(n):
        sid = int(session_id[i])
        out[sid] = out.get(sid, 0.0) + abs(float(filled_qty[i]) * float(exec_price[i]))
    return out


def _session_trade_count_by_payload(trade_payload: dict[str, np.ndarray] | None) -> dict[int, int]:
    if not isinstance(trade_payload, dict):
        return {}
    session_id = np.asarray(trade_payload.get("session_id", np.zeros(0, dtype=np.int64)), dtype=np.int64)
    filled_qty = np.asarray(trade_payload.get("filled_qty", np.zeros(0, dtype=np.float64)), dtype=np.float64)
    n = min(session_id.size, filled_qty.size)
    if n <= 0:
        return {}
    out: dict[int, int] = {}
    for i in range(n):
        if abs(float(filled_qty[i])) <= 1.0e-12:
            continue
        sid = int(session_id[i])
        out[sid] = out.get(sid, 0) + 1
    return out


def _session_gross_peak_by_equity_payload(equity_payload: dict[str, np.ndarray] | None) -> dict[int, float]:
    if not isinstance(equity_payload, dict):
        return {}
    session_id = np.asarray(equity_payload.get("session_id", np.zeros(0, dtype=np.int64)), dtype=np.int64)
    equity = np.asarray(equity_payload.get("equity", np.zeros(0, dtype=np.float64)), dtype=np.float64)
    margin_used = np.asarray(equity_payload.get("margin_used", np.zeros(0, dtype=np.float64)), dtype=np.float64)
    n = min(session_id.size, equity.size, margin_used.size)
    if n <= 0:
        return {}
    out: dict[int, float] = {}
    for i in range(n):
        sid = int(session_id[i])
        eq = max(abs(float(equity[i])), 1.0e-12)
        gross_peak = abs(float(margin_used[i])) / eq
        out[sid] = max(out.get(sid, 0.0), gross_peak)
    return out


def _availability_state_codes_from_worker_truth(
    *,
    session_ids_exec: np.ndarray,
    session_ids_raw: np.ndarray,
    trade_payload: dict[str, np.ndarray] | None,
    equity_payload: dict[str, np.ndarray] | None,
    dq_invalidated: bool,
) -> tuple[np.ndarray, np.ndarray]:
    observed = np.unique(
        np.r_[
            np.asarray(session_ids_exec, dtype=np.int64).reshape(-1),
            np.asarray(session_ids_raw, dtype=np.int64).reshape(-1),
        ]
    ).astype(np.int64)
    if observed.size <= 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int16)
    if bool(dq_invalidated):
        return observed, np.full(observed.shape[0], int(_MODULE6_AVAIL_INVALIDATED_BY_DQ), dtype=np.int16)
    trade_notional = _session_notional_by_trade_payload(trade_payload)
    trade_count = _session_trade_count_by_payload(trade_payload)
    gross_peak = _session_gross_peak_by_equity_payload(equity_payload)
    codes = np.zeros(observed.shape[0], dtype=np.int16)
    for i, sid in enumerate(observed.tolist()):
        has_activity = (
            float(trade_notional.get(int(sid), 0.0)) > 0.0
            or int(trade_count.get(int(sid), 0)) > 0
            or float(gross_peak.get(int(sid), 0.0)) > 0.0
        )
        codes[i] = np.int16(_MODULE6_AVAIL_OBSERVED_ACTIVE if has_activity else _MODULE6_AVAIL_OBSERVED_FLAT)
    return observed, codes


def _event_window_mask(T: int, event_idx: np.ndarray, pre: int, post: int) -> np.ndarray:
    return _eval_event_window_mask(T, event_idx, pre, post)


def _structural_weight_from_regime(regime_i8: np.ndarray) -> np.ndarray:
    return _eval_structural_weight_from_regime(regime_i8)


def _select_micro_rows(
    state: TensorState,
    split: SplitSpec,
    cfg: Module5HarnessConfig,
    m4_out: _ExecutionView,
    enabled_assets_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return _eval_select_micro_rows(state, split, cfg, m4_out, enabled_assets_mask)


def _collect_micro_diagnostics_payload(
    state: TensorState,
    m3: Module3Output,
    m4_out: _ExecutionView,
    candidate_id: str,
    split_id: str,
    scenario_id: str,
    split: SplitSpec,
    enabled_assets_mask: np.ndarray,
    cfg: Module5HarnessConfig,
) -> dict[str, np.ndarray] | None:
    return _eval_collect_micro_diagnostics_payload(
        state=state,
        m3=m3,
        m4_out=m4_out,
        candidate_id=candidate_id,
        split_id=split_id,
        scenario_id=scenario_id,
        split=split,
        enabled_assets_mask=enabled_assets_mask,
        cfg=cfg,
    )


def _collect_micro_profile_blocks_payload(
    state: TensorState,
    m3: Module3Output,
    candidate_id: str,
    split_id: str,
    scenario_id: str,
    enabled_assets_mask: np.ndarray,
    cfg: Module5HarnessConfig,
) -> dict[str, np.ndarray] | None:
    return _eval_collect_micro_profile_blocks_payload(
        state=state,
        m3=m3,
        candidate_id=candidate_id,
        split_id=split_id,
        scenario_id=scenario_id,
        enabled_assets_mask=enabled_assets_mask,
        cfg=cfg,
    )


def _collect_funnel_payload(
    state: TensorState,
    m4_out: _ExecutionView,
    candidate_id: str,
    split_id: str,
    scenario_id: str,
    enabled_assets_mask: np.ndarray,
    cfg: Module5HarnessConfig,
) -> dict[str, np.ndarray] | None:
    return _eval_collect_funnel_payload(
        state=state,
        m4_out=m4_out,
        candidate_id=candidate_id,
        split_id=split_id,
        scenario_id=scenario_id,
        enabled_assets_mask=enabled_assets_mask,
        cfg=cfg,
        require_pandas_fn=_require_pandas,
    )


def _extract_group_runtime_stats(rows: list[dict[str, Any]]) -> GroupExecutionRuntimeStats | None:
    if not rows:
        return None
    raw = rows[0].get("group_runtime_stats")
    if not isinstance(raw, dict):
        return None
    return GroupExecutionRuntimeStats(
        split_stress_sec=float(raw.get("split_stress_sec", 0.0)),
        module2_sec=float(raw.get("module2_sec", 0.0)),
        module3_sec=float(raw.get("module3_sec", 0.0)),
        candidate_loop_sec=float(raw.get("candidate_loop_sec", 0.0)),
        candidate_count=int(raw.get("candidate_count", len(rows))),
        market_overlay_bytes=int(raw.get("market_overlay_bytes", 0)),
        feature_overlay_bytes=int(raw.get("feature_overlay_bytes", 0)),
        module3_group_bytes_estimated=int(raw.get("module3_group_bytes_estimated", raw.get("module3_bytes", 0))),
        module3_group_bytes_realized=int(raw.get("module3_group_bytes_realized", raw.get("module3_bytes", 0))),
        module3_bytes=int(raw.get("module3_bytes", raw.get("module3_group_bytes_realized", 0))),
        candidate_scratch_bytes=int(raw.get("candidate_scratch_bytes", 0)),
        result_payload_bytes=int(raw.get("result_payload_bytes", 0)),
    )


def _rolling_p95_int(samples: list[int]) -> int | None:
    if not samples:
        return None
    arr = np.asarray(samples, dtype=np.int64)
    return int(np.quantile(arr, 0.95, method="higher"))


def _rolling_p95_float(samples: list[float]) -> float | None:
    if not samples:
        return None
    arr = np.asarray(samples, dtype=np.float64)
    return float(np.quantile(arr, 0.95, method="higher"))


def _run_group_task(
    group: _GroupTask,
    base_state: Any,
    candidates: list[CandidateSpec],
    splits: list[SplitSpec],
    scenarios: list[StressScenario],
    m2_configs: list[Module2Config],
    m3_configs: list[Module3Config],
    m4_configs: list[Module4Config],
    harness_cfg: Module5HarnessConfig,
) -> list[dict[str, Any]]:
    split = splits[group.split_idx]
    scenario = scenarios[group.scenario_idx]

    base = base_state if isinstance(base_state, BaseTensorState) else BaseTensorState.from_tensor_state(base_state)
    group_seed = _seed_for_task(
        harness_cfg.seed,
        split.split_id,
        scenario.scenario_id,
        str(group.m2_idx),
        str(group.m3_idx),
    )
    rng = np.random.default_rng(group_seed)
    outputs: list[dict[str, Any]] = []
    runtime_stats = GroupExecutionRuntimeStats()
    market_overlay = MarketOverlay.from_base(base)
    feature_overlay = FeatureOverlay.allocate(base, module3_output_mode=str(harness_cfg.module3_output_mode))
    scratch = CandidateScratch.allocate(base, _resolve_candidate_scratch_mode(harness_cfg))
    group_view = CombinedStateView(base, market_overlay, feature_overlay, CandidateScratch.template_from_base(base))

    t_group_build = time.perf_counter()
    active_t = _apply_split_domain_mask(group_view, split)
    _apply_missing_bursts(group_view, active_t, scenario, rng)
    _apply_jitter(group_view, active_t, scenario, rng)
    _recompute_bar_valid_inplace(group_view)
    _set_placeholders_from_bar_valid(group_view)
    _assert_placeholder_consistency(group_view)
    if harness_cfg.fail_on_non_finite:
        _assert_active_domain_ohlc(group_view, active_t)
        _validate_loaded_market_slice_active_domain(group_view, active_t)
    split_stress_sec = float(time.perf_counter() - t_group_build)

    t_module2 = time.perf_counter()
    _recompute_module2_on_stressed_state(group_view, m2_configs[group.m2_idx])
    post_m2_reasons = _apply_post_m2_invariants(group_view, active_t)
    module2_sec = float(time.perf_counter() - t_module2)

    t_module3 = time.perf_counter()
    m3_out_cached = run_module3_structural_aggregation(group_view, m3_configs[group.m3_idx])
    post_m3_reasons = _apply_post_m3_invariants(m3_out_cached)
    group_pre_m4_reasons = _apply_pre_m4_invariants(group_view, m3_out_cached)
    feature_overlay.module3_output = m3_out_cached
    feature_overlay.group_invariant_reason_codes = tuple(
        sorted(set(post_m2_reasons + post_m3_reasons + group_pre_m4_reasons))
    )
    feature_overlay.module3_group_bytes = int(measure_module3_output_bytes(m3_out_cached))
    feature_overlay.freeze_for_candidate_loop()
    module3_sec = float(time.perf_counter() - t_module3)

    dqs_src = getattr(base, "dqs_day_ta", None)
    if dqs_src is None:
        dqs_mat = np.ones((base.cfg.T, base.cfg.A), dtype=np.float64)
    else:
        dqs_mat = np.asarray(dqs_src, dtype=np.float64)
    if dqs_mat.shape != (base.cfg.T, base.cfg.A):
        raise RuntimeError(f"dqs_day_ta shape mismatch: got {dqs_mat.shape}, expected {(base.cfg.T, base.cfg.A)}")
    dqs_scope = dqs_mat[np.asarray(market_overlay.bar_valid, dtype=bool)]
    if dqs_scope.size <= 0:
        dqs_scope = np.asarray([1.0], dtype=np.float64)
    dqs_min = float(np.min(dqs_scope))
    dqs_median = float(np.median(dqs_scope))
    candidate_loop_t0 = time.perf_counter()
    total_payload_bytes = 0

    for ci in group.candidate_indices:
        c = candidates[ci]
        task_id = f"{c.candidate_id}|{split.split_id}|{scenario.scenario_id}"
        task_seed = _seed_for_task(group_seed, c.candidate_id, str(c.m4_idx))
        scratch.reset_from_base(base)
        st: CombinedStateView | None = CombinedStateView(
            base,
            market_overlay,
            feature_overlay,
            scratch,
            asset_enabled_mask=np.asarray(c.enabled_assets_mask, dtype=bool),
        )
        try:
            if task_id in set(harness_cfg.test_fail_task_ids):
                raise RuntimeError("InjectedTaskFailure: task_id match")
            if float(harness_cfg.test_fail_ratio) > 0.0:
                h = hashlib.sha256(f"{task_seed}|{task_id}".encode("utf-8")).digest()
                u = int.from_bytes(h[:8], "little", signed=False) / float(2**64 - 1)
                if u < float(harness_cfg.test_fail_ratio):
                    raise RuntimeError("InjectedTaskFailure: ratio")
            m3c = m3_out_cached
            quality_reason_codes = sorted(set(feature_overlay.group_invariant_reason_codes))
            if dqs_min < 1.0:
                quality_reason_codes = sorted(set(quality_reason_codes + ["DQ_DEGRADED_INPUT"]))
            if dqs_min <= 0.0:
                quality_reason_codes = sorted(set(quality_reason_codes + ["DQ_REJECTED_INPUT"]))

            m4_cfg = replace(
                m4_configs[c.m4_idx],
                stress_slippage_mult=float(m4_configs[c.m4_idx].stress_slippage_mult)
                * float(scenario.slippage_mult),
            )

            m4_sig: Module4SignalOutput = run_module4_signal_funnel(
                st,
                m3c,
                m4_cfg,
            )
            assert_float64("harness.module4_signal.target_qty_ta", m4_sig.target_qty_ta)
            close_px_safe = np.asarray(st.close_px, dtype=np.float64).copy()
            for a in range(int(st.cfg.A)):
                col = close_px_safe[:, a]
                finite = np.isfinite(col)
                if not np.any(finite):
                    col[:] = 1.0
                    continue
                first = int(np.flatnonzero(finite)[0])
                if first > 0:
                    col[:first] = col[first]
                for t_i in range(first + 1, int(st.cfg.T)):
                    if not np.isfinite(col[t_i]):
                        col[t_i] = col[t_i - 1]
            target_qty_raw = np.asarray(m4_sig.target_qty_ta, dtype=np.float64)
            target_qty_exec = _apply_latency_to_target_qty(
                target_qty_raw,
                latency_bars=int(harness_cfg.execution_latency_bars),
            )
            risk_res_raw = simulate_portfolio_from_signals(
                close_px_ta=close_px_safe,
                target_qty_ta=target_qty_raw,
                initial_cash=float(st.cfg.initial_cash),
                cost_cfg=CostConfig(
                    commission_per_share=0.0,
                    finra_taf_per_share_sell=0.0,
                    sec_fee_per_dollar_sell=0.0,
                    short_borrow_apr=0.0,
                    locate_fee_per_share_short_entry=0.0,
                    slippage_bps=0.0,
                ),
                risk_cfg=RiskConfig(),
                session_id_t=st.session_id,
                volume_ta=st.volume,
            )
            exec_slippage_bps = (
                float(m4_cfg.slippage_bps_mid_rvol)
                * float(scenario.slippage_mult)
                * float(harness_cfg.execution_slippage_mult)
                + float(harness_cfg.execution_extra_slippage_bps)
            )
            risk_res_exec = simulate_portfolio_from_signals(
                close_px_ta=close_px_safe,
                target_qty_ta=target_qty_exec,
                initial_cash=float(st.cfg.initial_cash),
                cost_cfg=CostConfig(
                    commission_per_share=float(harness_cfg.execution_transaction_cost_per_trade),
                    finra_taf_per_share_sell=0.0,
                    sec_fee_per_dollar_sell=0.0,
                    short_borrow_apr=0.0,
                    locate_fee_per_share_short_entry=0.0,
                    slippage_bps=max(0.0, float(exec_slippage_bps)),
                ),
                risk_cfg=RiskConfig(),
                session_id_t=st.session_id,
                volume_ta=st.volume,
            )
            m4_out = _materialize_risk_outputs_into_state(st, m4_sig, risk_res_exec)
            if _resolve_candidate_scratch_mode(harness_cfg) == "full":
                validate_state_hard(st)
            else:
                validate_candidate_execution_view(st)

            sess_ids, close_idx, daily_ret_exec = _candidate_daily_returns_close_to_close(
                st,
                split,
                initial_cash=float(st.cfg.initial_cash),
                equity_curve=risk_res_exec.equity_curve,
            )
            sess_ids_raw, _close_idx_raw, daily_ret_raw = _candidate_daily_returns_close_to_close(
                st,
                split,
                initial_cash=float(st.cfg.initial_cash),
                equity_curve=risk_res_raw.equity_curve,
            )
            if not np.array_equal(sess_ids_raw, sess_ids):
                map_raw = {int(s): float(v) for s, v in zip(sess_ids_raw.tolist(), daily_ret_raw.tolist())}
                daily_ret_raw = np.asarray(
                    [float(map_raw.get(int(s), 0.0)) for s in sess_ids.tolist()],
                    dtype=np.float64,
                )
            equity_payload = _equity_curve_payload(st, c.candidate_id, split.split_id, scenario.scenario_id)
            trade_payload = _trade_log_payload(st, m4_out, c.candidate_id, split.split_id, scenario.scenario_id)
            dq_invalidated = bool("DQ_REJECTED_INPUT" in quality_reason_codes)
            availability_state_session_ids, availability_state_codes = _availability_state_codes_from_worker_truth(
                session_ids_exec=sess_ids,
                session_ids_raw=sess_ids_raw,
                trade_payload=trade_payload,
                equity_payload=equity_payload,
                dq_invalidated=dq_invalidated,
            )

            micro_payload = _collect_micro_diagnostics_payload(
                state=st,
                m3=m3c,
                m4_out=m4_out,
                candidate_id=c.candidate_id,
                split_id=split.split_id,
                scenario_id=scenario.scenario_id,
                split=split,
                enabled_assets_mask=c.enabled_assets_mask,
                cfg=harness_cfg,
            )
            profile_payload = _collect_micro_profile_blocks_payload(
                state=st,
                m3=m3c,
                candidate_id=c.candidate_id,
                split_id=split.split_id,
                scenario_id=scenario.scenario_id,
                enabled_assets_mask=c.enabled_assets_mask,
                cfg=harness_cfg,
            )
            funnel_payload = _collect_funnel_payload(
                state=st,
                m4_out=m4_out,
                candidate_id=c.candidate_id,
                split_id=split.split_id,
                scenario_id=scenario.scenario_id,
                enabled_assets_mask=c.enabled_assets_mask,
                cfg=harness_cfg,
            )
            payload_bytes = (
                _tensor_nbytes_total(micro_payload)
                + _tensor_nbytes_total(profile_payload)
                + _tensor_nbytes_total(funnel_payload)
                + _tensor_nbytes_total(risk_res_exec.equity_curve)
                + _tensor_nbytes_total(risk_res_exec.filled_qty_ta)
                + _tensor_nbytes_total(risk_res_exec.exec_price_ta)
                + _tensor_nbytes_total(risk_res_exec.trade_cost_ta)
            )
            total_payload_bytes += int(payload_bytes)

            row = {
                "task_id": task_id,
                "candidate_id": c.candidate_id,
                "split_id": split.split_id,
                "scenario_id": scenario.scenario_id,
                "status": "ok",
                "error": "",
                "session_ids": sess_ids,
                "session_ids_exec": sess_ids,
                "session_ids_raw": sess_ids_raw,
                # Compatibility contract: daily_returns remains execution-adjusted.
                # The explicit execution-adjusted alias is daily_returns_exec.
                # The raw no-latency/no-extra-friction series is daily_returns_raw.
                "daily_returns": daily_ret_exec,
                "daily_returns_exec": daily_ret_exec,
                "daily_returns_raw": daily_ret_raw,
                "equity_payload": equity_payload,
                "trade_payload": trade_payload,
                "asset_pnl_by_symbol": _asset_pnl_by_symbol_from_state(st, split),
                "micro_payload": micro_payload,
                "profile_payload": profile_payload,
                "funnel_payload": funnel_payload,
                "m2_idx": int(c.m2_idx),
                "m3_idx": int(c.m3_idx),
                "m4_idx": int(c.m4_idx),
                "tags": list(c.tags),
                "test_days": int(daily_ret_exec.shape[0]),
                "task_seed": int(task_seed),
                "quality_reason_codes": quality_reason_codes,
                "dq_invalidated": bool(dq_invalidated),
                "availability_state_session_ids": availability_state_session_ids,
                "availability_state_codes": availability_state_codes,
                "asset_keys": [st.symbols[i] for i in np.flatnonzero(c.enabled_assets_mask).tolist()],
                "exception_signature": "",
                "risk_engine_metrics": {
                    "final_equity_raw": float(risk_res_raw.final_equity),
                    "max_drawdown_raw": float(risk_res_raw.max_drawdown),
                    "sharpe_raw": float(risk_res_raw.sharpe),
                    "sortino_raw": float(risk_res_raw.sortino),
                    "trades_raw": int(risk_res_raw.trades),
                    "final_equity_exec": float(risk_res_exec.final_equity),
                    "max_drawdown_exec": float(risk_res_exec.max_drawdown),
                    "sharpe_exec": float(risk_res_exec.sharpe),
                    "sortino_exec": float(risk_res_exec.sortino),
                    "trades_exec": int(risk_res_exec.trades),
                },
                "group_runtime_stats": {
                    "split_stress_sec": float(split_stress_sec),
                    "module2_sec": float(module2_sec),
                    "module3_sec": float(module3_sec),
                },
                "dqs_min": dqs_min,
                "dqs_median": dqs_median,
            }
            _eval_assert_artifact_dependency_contract(
                "strategy_results",
                candidate_row=row,
            )
            outputs.append(row)
        except Exception as exc:
            err_type = type(exc).__name__
            err_msg = str(exc)
            tb = traceback.format_exc()
            top_frame = _normalized_top_frame(tb)
            sig = f"{err_type}|{top_frame}"
            asset_keys = [base_state.symbols[i] for i in np.flatnonzero(c.enabled_assets_mask).tolist()]
            quality_reason_codes = sorted(set(post_m2_reasons + post_m3_reasons))
            state_dump: dict[str, Any] | None = None
            if _is_risk_constraint_breach(err_type, err_msg, top_frame, tb):
                quality_reason_codes = sorted(set(quality_reason_codes + ["RISK_CONSTRAINT_BREACH"]))
                if st is not None:
                    t_idx = _extract_breach_index(err_msg, st)
                    state_dump = _build_risk_constraint_state_dump(
                        st,
                        t_idx,
                        c.candidate_id,
                        split.split_id,
                        scenario.scenario_id,
                    )
            outputs.append(
                {
                    "task_id": task_id,
                    "candidate_id": c.candidate_id,
                    "split_id": split.split_id,
                    "scenario_id": scenario.scenario_id,
                    "status": "error",
                    "error_type": err_type,
                    "error_hash": _error_hash(err_type, err_msg),
                    "error": f"{err_type}: {err_msg}",
                    "traceback": tb,
                    "top_frame": top_frame,
                    "exception_signature": sig,
                    "session_ids": np.zeros(0, dtype=np.int64),
                    "session_ids_exec": np.zeros(0, dtype=np.int64),
                    "session_ids_raw": np.zeros(0, dtype=np.int64),
                    "daily_returns": np.zeros(0, dtype=np.float64),
                    "daily_returns_exec": np.zeros(0, dtype=np.float64),
                    "daily_returns_raw": np.zeros(0, dtype=np.float64),
                    "equity_payload": None,
                    "trade_payload": None,
                    "asset_pnl_by_symbol": {},
                    "micro_payload": None,
                    "profile_payload": None,
                    "funnel_payload": None,
                    "m2_idx": int(c.m2_idx),
                    "m3_idx": int(c.m3_idx),
                    "m4_idx": int(c.m4_idx),
                    "tags": list(c.tags),
                    "test_days": 0,
                    "task_seed": int(task_seed),
                    "quality_reason_codes": quality_reason_codes,
                    "dq_invalidated": bool("DQ_REJECTED_INPUT" in quality_reason_codes),
                    "availability_state_session_ids": np.zeros(0, dtype=np.int64),
                    "availability_state_codes": np.zeros(0, dtype=np.int16),
                    "asset_keys": asset_keys,
                    "state_dump": state_dump,
                    "group_runtime_stats": {
                        "split_stress_sec": float(split_stress_sec),
                        "module2_sec": float(module2_sec),
                        "module3_sec": float(module3_sec),
                    },
                    "dqs_min": 0.0,
                    "dqs_median": 0.0,
                }
            )
    candidate_loop_sec = float(time.perf_counter() - candidate_loop_t0)
    runtime_stats = GroupExecutionRuntimeStats(
        split_stress_sec=float(split_stress_sec),
        module2_sec=float(module2_sec),
        module3_sec=float(module3_sec),
        candidate_loop_sec=float(candidate_loop_sec),
        candidate_count=int(len(group.candidate_indices)),
        market_overlay_bytes=int(market_overlay.nbytes),
        feature_overlay_bytes=int(feature_overlay.feature_bytes),
        module3_group_bytes_estimated=int(max(1, getattr(group, "module3_group_bytes_estimated", 0))),
        module3_group_bytes_realized=int(feature_overlay.module3_group_bytes),
        module3_bytes=int(feature_overlay.module3_group_bytes),
        candidate_scratch_bytes=int(scratch.nbytes),
        result_payload_bytes=int(total_payload_bytes),
    )
    for row in outputs:
        row["group_runtime_stats"] = asdict(runtime_stats)
    return outputs


def _stack_payload_frames(payloads: list[dict[str, np.ndarray]]) -> Any:
    return _candidate_stack_payload_frames(payloads, _require_pandas)


def _write_json(path: Path, obj: Any) -> None:
    _artifact_write_json(path, obj)


def _write_frozen_module5_json(path: Path, obj: Any) -> None:
    _artifact_write_frozen_json(path, obj)


def _atomic_write_parquet(df: Any, path: Path) -> None:
    # Delegated helper preserves the atomic replace contract: os.replace(tmp, path)
    _artifact_atomic_write_parquet(df, path)


def _ledger_write(rows: list[dict[str, Any]], path: Path) -> None:
    global _WORKER_PROCESS
    if bool(_WORKER_PROCESS):
        raise RuntimeError("LEDGER_WRITE_FORBIDDEN_IN_WORKER")
    pdx = _require_pandas()
    if not rows:
        _atomic_write_parquet(pdx.DataFrame(), path)
        return
    df = pdx.DataFrame(rows)
    _atomic_write_parquet(df, path)


def _collect_ledger_rows_from_results(rows: list[dict[str, Any]], evaluation_timestamp: str) -> list[dict[str, Any]]:
    return _candidate_collect_ledger_rows_from_results(
        rows=rows,
        evaluation_timestamp=evaluation_timestamp,
        trade_count_from_payload_fn=_trade_count_from_payload,
        max_drawdown_from_returns_fn=_max_drawdown_from_returns,
        extract_final_equity_fn=_extract_final_equity,
    )


def _clip01(x: float) -> float:
    return _metrics_clip01(x)


def _apply_latency_to_target_qty(target_qty_ta: np.ndarray, latency_bars: int) -> np.ndarray:
    return _metrics_apply_latency_to_target_qty(target_qty_ta, latency_bars)


def _resample_returns_horizon(returns_1d: np.ndarray, horizon: int) -> np.ndarray:
    return _metrics_resample_returns_horizon(returns_1d, horizon)


def _slice_score_from_stats(dsr: dict[str, Any], pbo: dict[str, Any], spa: dict[str, Any]) -> float:
    return _metrics_slice_score_from_stats(dsr, pbo, spa)


def _effective_benchmark_for_horizon(benchmark: np.ndarray, horizon: int) -> np.ndarray:
    return _metrics_effective_benchmark_for_horizon(benchmark, horizon)


def _cum_return(ret_1d: np.ndarray) -> float:
    return _metrics_cum_return(ret_1d)


def _max_drawdown_from_returns(ret_1d: np.ndarray) -> float:
    return _metrics_max_drawdown_from_returns(ret_1d)


def _sharpe_daily(ret_1d: np.ndarray, eps: float = 1e-12) -> float:
    return _metrics_sharpe_daily(ret_1d, eps=eps)


def _turnover_from_trade_payload(trade_payload: dict[str, np.ndarray] | None, initial_cash: float) -> float:
    return _metrics_turnover_from_trade_payload(trade_payload, initial_cash)


def _trade_count_from_payload(trade_payload: dict[str, np.ndarray] | None) -> int:
    return _metrics_trade_count_from_payload(trade_payload)


def _extract_final_equity(row: dict[str, Any]) -> float:
    return _metrics_extract_final_equity(row)


def _margin_exposure_stats_from_equity_payloads(payloads: list[dict[str, np.ndarray]]) -> dict[str, float]:
    return _metrics_margin_exposure_stats_from_equity_payloads(payloads)


def _asset_notional_concentration_from_trade_payloads(payloads: list[dict[str, np.ndarray]]) -> float:
    return _metrics_asset_notional_concentration_from_trade_payloads(payloads)


def _asset_pnl_concentration_from_result_rows(rows: list[dict[str, Any]]) -> float:
    return _metrics_asset_pnl_concentration_from_result_rows(rows)


def _split_mode(split_id: str) -> str:
    s = str(split_id)
    if s.startswith("wf_"):
        return "wf"
    if s.startswith("cpcv_"):
        return "cpcv"
    return "other"


def _summarize_fold_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _candidate_summarize_fold_stats(rows)


def _plateau_key(feature: dict[str, float]) -> tuple[int, ...]:
    return _candidate_plateau_key(feature)


def _aggregate_candidate_baseline_matrices(
    results_ok: list[dict[str, Any]],
    bench_sessions: np.ndarray,
    bench_ret: np.ndarray,
    candidate_ids: list[str],
    min_days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, dict[str, dict[int, float]]]]:
    return _aggregation_aggregate_candidate_baseline_matrices(
        results_ok=results_ok,
        bench_sessions=bench_sessions,
        bench_ret=bench_ret,
        candidate_ids=candidate_ids,
        min_days=min_days,
    )


def _aggregate_candidate_baseline_matrix(
    results_ok: list[dict[str, Any]],
    bench_sessions: np.ndarray,
    bench_ret: np.ndarray,
    candidate_ids: list[str],
    min_days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, dict[str, dict[int, float]]]]:
    return _aggregation_aggregate_candidate_baseline_matrix(
        results_ok=results_ok,
        bench_sessions=bench_sessions,
        bench_ret=bench_ret,
        candidate_ids=candidate_ids,
        min_days=min_days,
    )


def _validate_institutional_harness_config(harness_cfg: Module5HarnessConfig) -> None:
    research_mode = str(harness_cfg.research_mode).strip().lower()
    if research_mode not in {"standard", "discovery"}:
        raise RuntimeError(f"research_mode must be 'standard' or 'discovery', got {harness_cfg.research_mode!r}")
    if not (0.0 <= float(harness_cfg.cluster_corr_threshold) <= 1.0):
        raise RuntimeError(f"cluster_corr_threshold must be in [0,1], got {harness_cfg.cluster_corr_threshold}")
    if int(harness_cfg.cluster_distance_block_size) < 1:
        raise RuntimeError(
            f"cluster_distance_block_size must be >=1, got {int(harness_cfg.cluster_distance_block_size)}"
        )
    if int(harness_cfg.cluster_distance_in_memory_max_n) < 1:
        raise RuntimeError(
            "cluster_distance_in_memory_max_n must be >=1, "
            f"got {int(harness_cfg.cluster_distance_in_memory_max_n)}"
        )
    if float(harness_cfg.execution_transaction_cost_per_trade) < 0.0:
        raise RuntimeError("execution_transaction_cost_per_trade must be >=0")
    if float(harness_cfg.execution_slippage_mult) < 0.0:
        raise RuntimeError("execution_slippage_mult must be >=0")
    if float(harness_cfg.execution_extra_slippage_bps) < 0.0:
        raise RuntimeError("execution_extra_slippage_bps must be >=0")
    if int(harness_cfg.execution_latency_bars) < 0:
        raise RuntimeError("execution_latency_bars must be >=0")
    if int(harness_cfg.regime_vol_window) < 2:
        raise RuntimeError("regime_vol_window must be >=2")
    if int(harness_cfg.regime_slope_window) < 2:
        raise RuntimeError("regime_slope_window must be >=2")
    if int(harness_cfg.regime_hurst_window) < 8:
        raise RuntimeError("regime_hurst_window must be >=8")
    if int(harness_cfg.regime_min_obs_per_mask) < 1:
        raise RuntimeError("regime_min_obs_per_mask must be >=1")
    if len(harness_cfg.horizon_minutes) == 0:
        raise RuntimeError("horizon_minutes must be non-empty")
    seen: set[int] = set()
    for value in harness_cfg.horizon_minutes:
        if isinstance(value, (bool, np.bool_)):
            raise RuntimeError("horizon_minutes entries must be positive integers")
        horizon = int(value)
        if horizon <= 0:
            raise RuntimeError(f"horizon_minutes entries must be >0, got {horizon}")
        if horizon in seen:
            raise RuntimeError(f"horizon_minutes must not contain duplicates, got {horizon}")
        seen.add(horizon)
    weights = {
        "robustness_weight_dsr": float(harness_cfg.robustness_weight_dsr),
        "robustness_weight_pbo": float(harness_cfg.robustness_weight_pbo),
        "robustness_weight_spa": float(harness_cfg.robustness_weight_spa),
        "robustness_weight_regime": float(harness_cfg.robustness_weight_regime),
        "robustness_weight_execution": float(harness_cfg.robustness_weight_execution),
        "robustness_weight_horizon": float(harness_cfg.robustness_weight_horizon),
    }
    for name, value in weights.items():
        if not (0.0 <= value <= 1.0):
            raise RuntimeError(f"{name} must be in [0,1], got {value}")
    if abs(sum(weights.values()) - 1.0) > 1e-12:
        raise RuntimeError("robustness weights must sum to 1.0 within tolerance 1e-12")
    if not (0.0 <= float(harness_cfg.robustness_reject_threshold) <= 1.0):
        raise RuntimeError("robustness_reject_threshold must be in [0,1]")
    if not (0.0 <= float(harness_cfg.execution_fragile_threshold) <= 1.0):
        raise RuntimeError("execution_fragile_threshold must be in [0,1]")


def _compute_stats_verdict(
    daily_returns_matrix_exec: np.ndarray,
    daily_returns_matrix_raw: np.ndarray,
    daily_benchmark_returns: np.ndarray,
    candidate_ids: list[str],
    harness_cfg: Module5HarnessConfig,
    report_root: Path | None = None,
) -> dict[str, Any]:
    return _robustness_compute_stats_verdict(
        daily_returns_matrix_exec=daily_returns_matrix_exec,
        daily_returns_matrix_raw=daily_returns_matrix_raw,
        daily_benchmark_returns=daily_benchmark_returns,
        candidate_ids=candidate_ids,
        harness_cfg=harness_cfg,
        report_root=report_root,
        clip01_fn=_clip01,
        cum_return_fn=_cum_return,
        resample_returns_horizon_fn=_resample_returns_horizon,
        seed_for_task_fn=_seed_for_task,
    )


def _build_candidate_artifacts(
    report_root: Path,
    run_id: str,
    run_started_utc: datetime,
    git_hash: str,
    candidates: list[CandidateSpec],
    all_results: list[dict[str, Any]],
    candidate_daily_mat: np.ndarray,
    daily_bmk: np.ndarray,
    common_sessions: np.ndarray,
    baseline_candidate_ids: list[str],
    candidate_scenario_series: dict[str, dict[str, dict[int, float]]],
    candidate_verdict: dict[str, dict[str, Any]],
    expected_baseline_tasks: int,
    scenarios: list[StressScenario],
    engine_cfg: EngineConfig,
    m2_configs: list[Module2Config],
    m3_configs: list[Module3Config],
    m4_configs: list[Module4Config],
    harness_cfg: Module5HarnessConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    return _candidate_build_candidate_artifacts(
        report_root=report_root,
        run_id=run_id,
        run_started_utc=run_started_utc,
        git_hash=git_hash,
        candidates=candidates,
        all_results=all_results,
        candidate_daily_mat=candidate_daily_mat,
        daily_bmk=daily_bmk,
        common_sessions=common_sessions,
        baseline_candidate_ids=baseline_candidate_ids,
        candidate_scenario_series=candidate_scenario_series,
        candidate_verdict=candidate_verdict,
        expected_baseline_tasks=expected_baseline_tasks,
        scenarios=scenarios,
        engine_cfg=engine_cfg,
        m2_configs=m2_configs,
        m3_configs=m3_configs,
        m4_configs=m4_configs,
        harness_cfg=harness_cfg,
        require_pandas_fn=_require_pandas,
        write_json_fn=_write_json,
        baseline_failure_reasons_fn=_baseline_failure_reasons,
        clip01_fn=_clip01,
        cum_return_fn=_cum_return,
        max_drawdown_from_returns_fn=_max_drawdown_from_returns,
        turnover_from_trade_payload_fn=_turnover_from_trade_payload,
        sharpe_daily_fn=_sharpe_daily,
        trade_count_from_payload_fn=_trade_count_from_payload,
        margin_exposure_stats_from_equity_payloads_fn=_margin_exposure_stats_from_equity_payloads,
        asset_pnl_concentration_from_result_rows_fn=_asset_pnl_concentration_from_result_rows,
        asset_notional_concentration_from_trade_payloads_fn=_asset_notional_concentration_from_trade_payloads,
        robustness_caps=ROBUSTNESS_CAPS,
    )


def _bounded_active_worker_count(*, pending_count: int, effective_workers: int) -> int:
    pending = int(max(0, pending_count))
    workers = int(max(1, effective_workers))
    return int(min(pending, workers))


def run_weightiz_harness(
    data_paths: list[str],
    symbols: list[str],
    engine_cfg: EngineConfig,
    m2_configs: list[Module2Config],
    m3_configs: list[Module3Config],
    m4_configs: list[Module4Config],
    harness_cfg: Module5HarnessConfig,
    candidate_specs: list[CandidateSpec] | None = None,
    data_loader_func: Callable[[str, str], Any] | None = None,
    stress_scenarios: list[StressScenario] | None = None,
    self_audit_report: dict[str, Any] | None = None,
) -> HarnessOutput:
    if not m2_configs or not m3_configs or not m4_configs:
        raise RuntimeError("m2_configs/m3_configs/m4_configs must be non-empty")
    if len(m2_configs) != 1:
        raise RuntimeError("ARCHITECTURE_CONSISTENCY_FAILURE: canonical path requires exactly one module2 config")

    _validate_institutional_harness_config(harness_cfg)

    run_started_utc = datetime.now(timezone.utc)
    run_id = run_started_utc.strftime("run_%Y%m%d_%H%M%S")
    report_root = Path(harness_cfg.report_dir).resolve() / run_id
    report_root.mkdir(parents=True, exist_ok=True)
    self_audit_report_path = report_root / "self_audit_report.json"
    _write_json(
        self_audit_report_path,
        self_audit_report if isinstance(self_audit_report, dict) else {"status": "missing"},
    )
    log_ctx = init_runtime_logger(run_id=run_id, run_dir=report_root, level="INFO")
    logger = get_logger("module5_harness", run_id=run_id)
    log_event(logger, "INFO", "module5_harness_start", event_type="harness_start")
    deadletter_path = report_root / "deadletter_tasks.jsonl"
    run_status_path = report_root / "run_status.json"

    base_state, keep_idx, keep_symbols, master_ts_ns, ingest_meta, tick_keep, dq_bundle = _ingest_master_aligned(
        data_paths=data_paths,
        symbols=symbols,
        engine_cfg=engine_cfg,
        harness_cfg=harness_cfg,
        data_loader_func=data_loader_func,
    )
    os.environ.pop("WEIGHTIZ_WORKER_PROCESS", None)
    run_weightiz_profile_engine(base_state, m2_configs[0])
    cleanup_orphan_shared_memory_segments()

    profile_windows = [15, 30, 60, 120, 240]
    dataset_hash = _stable_hash_obj(
        {
            "symbols": list(keep_symbols),
            "t": int(base_state.cfg.T),
            "a": int(base_state.cfg.A),
            "first_ts": int(master_ts_ns[0]) if master_ts_ns.size else 0,
            "last_ts": int(master_ts_ns[-1]) if master_ts_ns.size else 0,
        }
    )
    hash_inputs = {
        "data_hash": dataset_hash,
        "module2_config": [asdict(c) for c in m2_configs],
        "profile_windows": profile_windows,
        "schema_version": PROFILE_CACHE_SCHEMA_VERSION,
    }
    tensor_hash = compute_tensor_hash(hash_inputs)
    cache_dir = report_root.parent / "profile_cache"
    cleanup_stale_tmp_cache_files(cache_dir)
    tensor_npz_path, tensor_json_path = profile_cache_paths(cache_dir, tensor_hash)

    if tensor_npz_path.exists() and tensor_json_path.exists():
        feature_tensor, feature_manifest = load_tensor_cache(tensor_npz_path, tensor_json_path)
    else:
        feature_specs: list[FeatureSpec] = make_compat_feature_specs(profile_windows)
        feature_tensor, feature_map, window_map, _engine_meta = build_feature_tensor_from_state(
            base_state,
            feature_specs=feature_specs,
            engine_cfg=FeatureEngineConfig(
                tensor_backend="ram",
                compute_backend="numpy",
                parallel_backend="serial",
                seed=int(harness_cfg.seed),
                use_cache=False,
            ),
        )
        feature_manifest_obj = build_feature_manifest(
            feature_tensor,
            feature_map=feature_map,
            window_map=window_map,
            hash_inputs=hash_inputs,
            dataset_hash=dataset_hash,
            dataset_version="v1",
            asset_universe=list(keep_symbols),
            rows_per_asset=int(base_state.cfg.T),
            timestamp_start=str(int(master_ts_ns[0])) if master_ts_ns.size else "0",
            timestamp_end=str(int(master_ts_ns[-1])) if master_ts_ns.size else "0",
        )
        save_tensor_cache(tensor_npz_path, tensor_json_path, feature_tensor, feature_manifest_obj)
        feature_manifest = asdict(feature_manifest_obj)

    validate_feature_tensor_contract(feature_tensor, feature_manifest)
    compute_window_correlation_diagnostics(
        feature_tensor,
        feature_map={str(k): int(v) for k, v in feature_manifest.get("feature_map", {}).items()},
        window_map={str(k): int(v) for k, v in feature_manifest.get("window_map", {}).items()},
        run_dir=report_root,
    )
    tensor_bytes = estimate_tensor_bytes(*feature_tensor.shape)
    avail_ram = _available_memory_bytes()
    enforce_memory_safety(tensor_bytes, avail_ram)
    feature_registry, feature_handles_master = create_shared_feature_store(feature_tensor)
    atexit.register(
        lambda: close_shared_feature_store(
            feature_handles_master,
            is_master=True,
            owner_pid=feature_registry.owner_pid,
        )
    )

    A_filtered = base_state.cfg.A
    A_input = len(symbols)

    if candidate_specs is None:
        candidates = _build_candidate_specs_default(A_filtered, m2_configs, m3_configs, m4_configs)
    else:
        candidates = _normalize_candidate_specs(candidate_specs, keep_idx, A_filtered, A_input)

    candidates = sorted(candidates, key=lambda c: c.candidate_id)
    quick_settings = _quick_run_settings_from_env()

    wf_splits = _generate_wf_splits(base_state, harness_cfg)
    cpcv_splits = [] if quick_settings.disable_cpcv else _generate_cpcv_splits(base_state, harness_cfg)
    splits = wf_splits + cpcv_splits
    if quick_settings.enabled and wf_splits:
        splits = [wf_splits[0]]
    if quick_settings.enabled and not splits:
        splits = _generate_quick_fallback_split(base_state, harness_cfg)
    if not splits:
        raise RuntimeError("No WF/CPCV splits generated; adjust harness split parameters")

    for sp in splits:
        _validate_split(sp, enforce_guard=bool(harness_cfg.enforce_lookahead_guard))

    source_scenarios = stress_scenarios if stress_scenarios is not None else _default_stress_scenarios(harness_cfg)
    scenarios = [s for s in source_scenarios if s.enabled]
    if quick_settings.enabled and quick_settings.baseline_only:
        baseline = [s for s in scenarios if str(s.scenario_id) == "baseline"]
        if baseline:
            scenarios = [baseline[0]]
        elif scenarios:
            scenarios = [sorted(scenarios, key=lambda x: str(x.scenario_id))[0]]
    if not scenarios:
        raise RuntimeError("No enabled stress scenarios")

    _selected_artifacts = _eval_validate_canonical_artifact_dependencies(
        scratch_mode=_resolve_candidate_scratch_mode(harness_cfg),
        export_micro_diagnostics=bool(harness_cfg.export_micro_diagnostics),
        export_micro_profile_blocks=bool(harness_cfg.micro_diag_export_block_profiles),
        export_funnel_1545=bool(harness_cfg.micro_diag_export_funnel),
    )
    shared_base_state = BaseTensorState.from_tensor_state(base_state)
    avail = _available_memory_bytes()
    requested_workers = int(max(1, harness_cfg.parallel_workers))
    requested_process_pool = bool(harness_cfg.parallel_backend == "process_pool" and requested_workers > 1)
    if quick_settings.enabled:
        requested_process_pool = False
    mp_ctx_pre, resolved_mp_start_method = _resolve_mp_context()
    base_sharing_decision = _mem_resolve_base_sharing_mode(
        configured_mode=str(harness_cfg.base_sharing_mode),
        start_method=str(resolved_mp_start_method),
        platform_name=sys.platform,
    )
    market_overlay_bytes, feature_overlay_bytes, startup_module3_bytes, candidate_scratch_bytes = _estimate_group_runtime_bytes(
        shared_base_state,
        harness_cfg,
    )
    initial_memory_estimate = _mem_build_memory_accounting_estimate(
        available_bytes=int(avail),
        max_ram_utilization_frac=float(harness_cfg.max_ram_utilization_frac),
        safety_margin_frac=float(harness_cfg.safety_margin_frac),
        safety_margin_min_bytes=int(harness_cfg.safety_margin_min_bytes),
        base_bytes=int(shared_base_state.base_bytes),
        market_overlay_bytes=int(market_overlay_bytes),
        feature_overlay_bytes=int(feature_overlay_bytes),
        module3_bytes=int(startup_module3_bytes),
        candidate_scratch_bytes=int(candidate_scratch_bytes),
        queue_bytes=0,
        result_buffer_bytes=0,
        requested_workers=int(requested_workers),
    )
    process_pool_requested = bool(requested_process_pool)
    group_bound_execution_requested = bool(harness_cfg.group_bound_execution_enabled)
    process_pool_group_reuse_requested = bool(process_pool_requested and group_bound_execution_requested)
    initial_effective_workers = int(initial_memory_estimate.effective_workers)

    if group_bound_execution_requested:
        base_group_tasks = _order_group_execution_tasks(
            _build_base_group_execution_tasks(candidates, splits, scenarios),
            dispatch_policy=str(harness_cfg.group_dispatch_policy),
        )
        legacy_group_tasks: list[_GroupTask] = []
    else:
        base_group_tasks = []
        legacy_group_tasks = _build_group_tasks(candidates, splits, scenarios)
        if process_pool_requested:
            legacy_group_tasks = _split_group_tasks_by_candidate(
                legacy_group_tasks,
                chunk_size=int(max(1, harness_cfg.process_pool_candidate_chunk_size)),
            )
        legacy_group_tasks = _order_group_execution_tasks(
            legacy_group_tasks,
            dispatch_policy=str(harness_cfg.group_dispatch_policy),
        )
    if not base_group_tasks and not legacy_group_tasks:
        raise RuntimeError("No group tasks generated")

    all_results: list[dict[str, Any]] = []
    tasks_submitted = int(
        sum(len(g.candidate_indices) for g in (base_group_tasks if group_bound_execution_requested else legacy_group_tasks))
    )
    tasks_completed = 0
    groups_completed = 0
    failure_count = 0
    aborted = False
    abort_reason = ""
    first_exception_class = ""
    first_exception_message = ""
    first_exception_hash = ""
    failure_tracker: dict[tuple[str, str], dict[str, set[str] | bool]] = {}
    run_t0 = time.perf_counter()
    checkpoint_every_groups = 10
    pool_heartbeat_seconds = 5.0
    group_runtime_stats_path = report_root / "group_runtime_stats.jsonl"
    planned_group_task_count = int(len(legacy_group_tasks))
    next_base_group_idx = 0
    base_group_count_total = int(len(base_group_tasks))
    legacy_group_queue: deque[_GroupTask] = deque(legacy_group_tasks)
    active_group_batch: deque[_GroupTask] = deque()
    module3_realized_samples: list[int] = []
    candidate_loop_per_candidate_samples: list[float] = []
    result_payload_bytes_per_candidate_samples: list[int] = []
    candidate_incremental_bytes_samples: list[int] = []
    latest_group_runtime_stats: GroupExecutionRuntimeStats | None = None
    worker_overhead_bytes = int(initial_memory_estimate.worker_overhead_bytes)
    current_module3_group_bytes_estimated = int(max(1, startup_module3_bytes))
    dynamic_effective_workers = int(initial_effective_workers)
    max_in_flight = int(max(1, dynamic_effective_workers * max(1, int(harness_cfg.group_max_in_flight_factor))))
    payload_task_sample = base_group_tasks if group_bound_execution_requested else legacy_group_tasks
    payload_safe, payload_arg_max_bytes = _is_large_payload_safe(
        payload_task_sample,
        threshold_bytes=int(harness_cfg.payload_pickle_threshold_bytes),
    )
    queue_bytes = _mem_estimate_queue_bytes(
        in_flight_count=int(max_in_flight),
        sampled_task_bytes=int(payload_arg_max_bytes),
    )
    memory_estimate = _mem_build_memory_accounting_estimate(
        available_bytes=int(avail),
        max_ram_utilization_frac=float(harness_cfg.max_ram_utilization_frac),
        safety_margin_frac=float(harness_cfg.safety_margin_frac),
        safety_margin_min_bytes=int(harness_cfg.safety_margin_min_bytes),
        base_bytes=int(shared_base_state.base_bytes),
        market_overlay_bytes=int(market_overlay_bytes),
        feature_overlay_bytes=int(feature_overlay_bytes),
        module3_bytes=int(current_module3_group_bytes_estimated),
        candidate_scratch_bytes=int(candidate_scratch_bytes),
        queue_bytes=int(queue_bytes),
        result_buffer_bytes=0,
        requested_workers=int(requested_workers),
        worker_overhead_bytes=int(worker_overhead_bytes),
    )
    worker_pool_capacity = _mem_estimate_worker_pool_capacity(
        available_bytes=int(avail),
        max_ram_utilization_frac=float(harness_cfg.max_ram_utilization_frac),
        safety_margin_frac=float(harness_cfg.safety_margin_frac),
        safety_margin_min_bytes=int(harness_cfg.safety_margin_min_bytes),
        base_bytes=int(shared_base_state.base_bytes),
        market_overlay_bytes=int(market_overlay_bytes),
        feature_overlay_bytes=int(feature_overlay_bytes),
        queue_bytes=0,
        result_buffer_bytes=0,
        requested_workers=int(requested_workers),
        worker_overhead_bytes=int(worker_overhead_bytes),
    )
    max_workers = int(memory_estimate.effective_workers)
    if process_pool_requested and max_workers <= 1:
        execution_mode = "serial_forced_ram"
    elif process_pool_requested and (not payload_safe):
        execution_mode = "serial_forced_payload"
    elif quick_settings.enabled:
        execution_mode = "serial_quick_run"
    elif process_pool_requested:
        execution_mode = "process_pool"
    else:
        execution_mode = "serial"
    use_process_pool = execution_mode == "process_pool"
    dynamic_effective_workers = int(max_workers if use_process_pool else 1)
    executor_worker_count = int(max(1, worker_pool_capacity if use_process_pool else 1))
    max_in_flight = int(max(1, dynamic_effective_workers * max(1, int(harness_cfg.group_max_in_flight_factor))))
    process_pool_group_reuse_active = bool(group_bound_execution_requested)
    base_sharing_runtime = {
        "configured_mode": str(base_sharing_decision.configured_mode),
        "resolved_mode": str(base_sharing_decision.mode),
        "start_method": str(base_sharing_decision.start_method),
        "shared_base_state_active": bool(use_process_pool and base_sharing_decision.shared_base_state_active),
        "fallback_only": bool(base_sharing_decision.fallback_only),
        "cow_probe_required": bool(use_process_pool and base_sharing_decision.cow_probe_required),
        "reason": str(base_sharing_decision.reason),
    }
    large_payload_passing_avoided = bool((not use_process_pool) or payload_safe)
    strategy_ledger_path = report_root / "strategy_results.parquet"
    monitor = RuntimeMonitor(
        run_id=run_id,
        run_dir=report_root,
        expected_tensor_shape=tuple(int(x) for x in feature_tensor.shape),
        expected_worker_count=int(dynamic_effective_workers),
        health_check_interval=int(max(1, harness_cfg.health_check_interval)),
    )
    progress_interval_seconds = int(max(1, harness_cfg.progress_interval_seconds))
    last_progress_log = time.perf_counter()
    feature_tensor_role = _truth_build_feature_tensor_role(shared_memory_published=True)
    compute_authority = _truth_build_compute_authority()

    def _current_candidate_loop_sec_per_candidate_p95() -> float | None:
        return _rolling_p95_float(candidate_loop_per_candidate_samples)

    def _current_result_payload_bytes_per_candidate_p95() -> int | None:
        return _rolling_p95_int(result_payload_bytes_per_candidate_samples)

    def _current_candidate_incremental_bytes_p95() -> int | None:
        return _rolling_p95_int(candidate_incremental_bytes_samples)

    def _recompute_memory_estimate(*, queue_backlog_groups: int, result_backlog_rows: int) -> None:
        nonlocal current_module3_group_bytes_estimated, memory_estimate, max_workers, dynamic_effective_workers, max_in_flight
        realized_module3_p95 = _rolling_p95_int(module3_realized_samples)
        if realized_module3_p95 is not None:
            current_module3_group_bytes_estimated = int(max(1, realized_module3_p95))
        sampled_row_bytes = int(
            max(
                1,
                _current_result_payload_bytes_per_candidate_p95()
                or int(harness_cfg.startup_default_result_payload_bytes),
            )
        )
        queue_bytes_hint = _mem_estimate_queue_bytes(
            in_flight_count=int(max(0, queue_backlog_groups)),
            sampled_task_bytes=int(payload_arg_max_bytes),
        )
        result_buffer_bytes_hint = _mem_estimate_result_buffer_bytes(
            pending_rows=int(max(0, result_backlog_rows)),
            sampled_row_bytes=int(sampled_row_bytes),
        )
        memory_estimate = _mem_build_memory_accounting_estimate(
            available_bytes=int(avail),
            max_ram_utilization_frac=float(harness_cfg.max_ram_utilization_frac),
            safety_margin_frac=float(harness_cfg.safety_margin_frac),
            safety_margin_min_bytes=int(harness_cfg.safety_margin_min_bytes),
            base_bytes=int(shared_base_state.base_bytes),
            market_overlay_bytes=int(market_overlay_bytes),
            feature_overlay_bytes=int(feature_overlay_bytes),
            module3_bytes=int(current_module3_group_bytes_estimated),
            candidate_scratch_bytes=int(candidate_scratch_bytes),
            queue_bytes=int(queue_bytes_hint),
            result_buffer_bytes=int(result_buffer_bytes_hint),
            requested_workers=int(requested_workers),
            worker_overhead_bytes=int(worker_overhead_bytes),
        )
        max_workers = int(memory_estimate.effective_workers)
        if use_process_pool:
            dynamic_effective_workers = int(min(max_workers, executor_worker_count))
        else:
            dynamic_effective_workers = 1
        max_in_flight = int(max(1, dynamic_effective_workers * max(1, int(harness_cfg.group_max_in_flight_factor))))

    def _current_group_memory_cap(*, queue_backlog_groups: int, result_backlog_rows: int) -> int:
        if int(harness_cfg.group_max_memory_bytes) > 0:
            return int(harness_cfg.group_max_memory_bytes)
        sampled_row_bytes = int(
            max(
                1,
                _current_result_payload_bytes_per_candidate_p95()
                or int(harness_cfg.startup_default_result_payload_bytes),
            )
        )
        queue_bytes_hint = _mem_estimate_queue_bytes(
            in_flight_count=int(max(0, queue_backlog_groups)),
            sampled_task_bytes=int(payload_arg_max_bytes),
        )
        result_buffer_bytes_hint = _mem_estimate_result_buffer_bytes(
            pending_rows=int(max(0, result_backlog_rows)),
            sampled_row_bytes=int(sampled_row_bytes),
        )
        requested_for_cap = int(max(1, dynamic_effective_workers if use_process_pool else 1))
        return int(
            _mem_chunk_policy_memory_cap(
                budget_bytes=int(memory_estimate.budget_bytes),
                base_bytes=int(shared_base_state.base_bytes),
                requested_workers=requested_for_cap,
                queue_bytes=int(queue_bytes_hint),
                result_buffer_bytes=int(result_buffer_bytes_hint),
                safety_margin_bytes=int(memory_estimate.safety_margin_bytes),
            )
        )

    def _append_group_runtime_stats(stats: GroupExecutionRuntimeStats) -> None:
        row = asdict(stats)
        row["group_sequence"] = int(groups_completed)
        with group_runtime_stats_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _record_group_runtime_stats(rows: list[dict[str, Any]]) -> GroupExecutionRuntimeStats | None:
        nonlocal latest_group_runtime_stats
        stats = _extract_group_runtime_stats(rows)
        if stats is None:
            return None
        latest_group_runtime_stats = stats
        _append_group_runtime_stats(stats)
        if int(stats.module3_group_bytes_realized) > 0:
            module3_realized_samples.append(int(stats.module3_group_bytes_realized))
        if int(stats.candidate_count) > 0:
            cand_count = int(stats.candidate_count)
            candidate_loop_per_candidate_samples.append(float(stats.candidate_loop_sec) / float(cand_count))
            payload_per_candidate = int(max(1, int(stats.result_payload_bytes) // cand_count))
            result_payload_bytes_per_candidate_samples.append(payload_per_candidate)
            candidate_incremental_bytes_samples.append(
                int(max(1, int(stats.candidate_scratch_bytes) + payload_per_candidate))
            )
        return stats

    def _plan_next_group_wave() -> deque[_GroupTask]:
        nonlocal next_base_group_idx, planned_group_task_count
        if not group_bound_execution_requested:
            return deque()
        if next_base_group_idx >= base_group_count_total:
            return deque()
        queue_backlog_groups = max(0, base_group_count_total - next_base_group_idx)
        result_backlog_rows = max(0, len(all_results) - tasks_completed)
        _recompute_memory_estimate(
            queue_backlog_groups=queue_backlog_groups,
            result_backlog_rows=result_backlog_rows,
        )
        groups_per_wave = int(max(1, max_in_flight))
        base_batch = base_group_tasks[next_base_group_idx : next_base_group_idx + groups_per_wave]
        next_base_group_idx += len(base_batch)
        group_fixed_bytes = int(
            market_overlay_bytes
            + feature_overlay_bytes
            + current_module3_group_bytes_estimated
            + int(memory_estimate.worker_overhead_bytes)
        )
        group_memory_cap = _current_group_memory_cap(
            queue_backlog_groups=max(0, base_group_count_total - next_base_group_idx),
            result_backlog_rows=max(0, len(all_results) - tasks_completed),
        )
        planned = [
            task
            for base_group in base_batch
            for task in _chunk_group_execution_task(
                base_group,
                candidates,
                harness_cfg,
                group_fixed_bytes=group_fixed_bytes,
                module3_group_bytes_estimated=int(current_module3_group_bytes_estimated),
                max_group_memory_bytes=int(group_memory_cap),
                candidate_loop_sec_per_candidate_p95=_current_candidate_loop_sec_per_candidate_p95(),
                result_payload_bytes_per_candidate_p95=_current_result_payload_bytes_per_candidate_p95(),
                candidate_incremental_bytes_p95=_current_candidate_incremental_bytes_p95(),
            )
        ]
        planned = _order_group_execution_tasks(planned, dispatch_policy=str(harness_cfg.group_dispatch_policy))
        planned_group_task_count += int(len(planned))
        return deque(planned)

    def _groups_total_hint() -> int:
        if group_bound_execution_requested:
            return int(max(groups_completed, planned_group_task_count + max(0, base_group_count_total - next_base_group_idx)))
        return int(len(legacy_group_tasks))

    def _queue_backlog_hint(*, pending_count: int) -> int:
        if group_bound_execution_requested:
            return int(pending_count + len(active_group_batch) + max(0, base_group_count_total - next_base_group_idx))
        return int(pending_count + len(legacy_group_queue))

    def _write_run_status_checkpoint(phase: str, execution_mode_now: str) -> None:
        elapsed = float(time.perf_counter() - run_t0)
        _write_json(
            run_status_path,
            {
                "run_id": run_id,
                "phase": str(phase),
                "execution_mode": str(execution_mode_now),
                "groups_done": int(groups_completed),
                "groups_total": int(_groups_total_hint()),
                "tasks_done": int(tasks_completed),
                "tasks_total": int(tasks_submitted),
                "failures_so_far": int(failure_count),
                "elapsed_seconds": float(elapsed),
                "compute_authority": compute_authority,
                "feature_tensor_role": feature_tensor_role,
                "execution_topology": _truth_build_execution_topology(
                    execution_mode_now,
                    use_process_pool,
                    process_pool_group_reuse_active,
                    base_sharing=base_sharing_runtime,
                ),
                "artifact_dependency_matrix": _eval_canonical_artifact_dependency_matrix(),
                "canonical_artifacts_selected": list(_selected_artifacts),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
    _write_run_status_checkpoint("running", execution_mode)

    def _maybe_emit_progress(active_workers: int) -> None:
        nonlocal last_progress_log
        now = time.perf_counter()
        if (now - last_progress_log) < float(progress_interval_seconds):
            return
        elapsed = max(1e-9, now - run_t0)
        avg_strategy_time = float(elapsed / max(tasks_completed, 1))
        shm_mem_gb = float(feature_tensor.nbytes) / float(1024**3)
        log_event(
            logger,
            "INFO",
            "runtime_progress",
            event_type="runtime_progress",
            extra={
                "strategies_completed": int(tasks_completed),
                "workers_active": int(active_workers),
                "workers_effective": int(dynamic_effective_workers),
                "avg_strategy_time_sec": float(avg_strategy_time),
                "diagnostic_feature_tensor_memory_gb": float(shm_mem_gb),
                "feature_tensor_worker_role": str(feature_tensor_role["role"]),
                "base_sharing_mode": str(base_sharing_runtime["resolved_mode"]),
                "base_sharing_active": bool(base_sharing_runtime["shared_base_state_active"]),
                "base_sharing_fallback_only": bool(base_sharing_runtime["fallback_only"]),
                "grouped_post_m2_reuse_active": bool(group_bound_execution_requested),
                "grouped_post_m3_reuse_active": bool(group_bound_execution_requested),
                "elapsed_runtime_sec": float(elapsed),
            },
        )
        last_progress_log = now

    def _maybe_health_check(active_workers: int, queue_backlog: int, result_backlog_rows: int) -> None:
        if not monitor.should_check(int(tasks_completed)):
            return
        sampled_row_bytes = int(
            max(
                1,
                _current_result_payload_bytes_per_candidate_p95()
                or int(harness_cfg.startup_default_result_payload_bytes),
            )
        )
        result_buffer_bytes = _mem_estimate_result_buffer_bytes(
            pending_rows=int(result_backlog_rows),
            sampled_row_bytes=int(sampled_row_bytes),
        )
        realized_module3_p95 = _rolling_p95_int(module3_realized_samples)
        monitor.check_and_emit(
            strategies_completed=int(tasks_completed),
            tensor=feature_handles_master.array,
            worker_status={
                "active": int(active_workers),
                "expected": int(dynamic_effective_workers),
                "ok": bool(int(active_workers) <= int(dynamic_effective_workers)),
            },
            ledger_path=strategy_ledger_path,
            queue_backlog=int(queue_backlog),
            memory_status={
                "ok": bool(memory_estimate.total_projected_bytes <= memory_estimate.budget_bytes),
                "available_bytes": int(memory_estimate.available_bytes),
                "budget_bytes": int(memory_estimate.budget_bytes),
                "base_bytes": int(memory_estimate.base_bytes),
                "market_overlay_bytes": int(memory_estimate.market_overlay_bytes),
                "feature_overlay_bytes": int(memory_estimate.feature_overlay_bytes),
                "module3_bytes": int(memory_estimate.module3_bytes),
                "candidate_scratch_bytes": int(memory_estimate.candidate_scratch_bytes),
                "queue_bytes": int(memory_estimate.queue_bytes),
                "result_buffer_bytes": int(result_buffer_bytes),
                "safety_margin_bytes": int(memory_estimate.safety_margin_bytes),
            },
            extra_metrics={
                "requested_workers": int(requested_workers),
                "effective_workers": int(dynamic_effective_workers),
                "base_sharing_mode": str(base_sharing_runtime["resolved_mode"]),
                "base_sharing_active": bool(base_sharing_runtime["shared_base_state_active"]),
                "base_sharing_fallback_only": bool(base_sharing_runtime["fallback_only"]),
                "groups_completed": int(groups_completed),
                "groups_total_hint": int(_groups_total_hint()),
                "tasks_total": int(tasks_submitted),
                "result_backlog_rows": int(result_backlog_rows),
                "result_backlog_bytes": int(result_buffer_bytes),
                "module3_estimated_bytes": int(current_module3_group_bytes_estimated),
                "module3_realized_bytes_p95": int(realized_module3_p95 or 0),
                "latest_group_candidate_count": int(latest_group_runtime_stats.candidate_count) if latest_group_runtime_stats is not None else 0,
                "rolling_tasks_per_sec": float(tasks_completed / max(1e-9, time.perf_counter() - run_t0)),
            },
            require_ledger_exists=False,
        )

    def _capture_first_exception(row: dict[str, Any]) -> None:
        nonlocal first_exception_class, first_exception_message, first_exception_hash
        if first_exception_class:
            return
        et = str(row.get("error_type", "")).strip()
        em = str(row.get("error", "")).strip()
        eh = str(row.get("error_hash", "")).strip()
        if not et and not em:
            return
        first_exception_class = et or "RuntimeError"
        first_exception_message = em or et
        first_exception_hash = eh or _error_hash(first_exception_class, first_exception_message)

    def _process_group_rows(rows: list[dict[str, Any]], *, active_workers: int, queue_backlog: int) -> None:
        nonlocal tasks_completed, groups_completed, failure_count, aborted, abort_reason
        all_results.extend(rows)
        tasks_completed += int(len(rows))
        groups_completed += 1
        _record_group_runtime_stats(rows)
        err_rows = [r for r in rows if str(r.get("status", "")) != "ok"]
        failure_count += int(len(err_rows))
        for er in err_rows:
            _capture_first_exception(er)
            _update_failure_tracker(failure_tracker, er)
            _record_deadletter(deadletter_path, er)
            abort_now, reason_now = _should_abort_systemic(failure_tracker, er)
            if abort_now:
                aborted = True
                abort_reason = reason_now
                break
        if (groups_completed % int(checkpoint_every_groups) == 0) or (
            groups_completed >= int(_groups_total_hint())
        ):
            _write_run_status_checkpoint("running", execution_mode)
        _maybe_emit_progress(active_workers=active_workers)
        _maybe_health_check(
            active_workers=active_workers,
            queue_backlog=queue_backlog,
            result_backlog_rows=max(0, len(all_results) - tasks_completed),
        )

    mp_start_method = ""
    if use_process_pool:
        mp_ctx, mp_start_method = mp_ctx_pre, resolved_mp_start_method
        with ProcessPoolExecutor(
            max_workers=int(executor_worker_count),
            mp_context=mp_ctx,
            initializer=_init_worker_context,
            initargs=(
                shared_base_state,
                candidates,
                splits,
                scenarios,
                m2_configs,
                m3_configs,
                m4_configs,
                harness_cfg,
                feature_registry,
                log_ctx.queue,
                run_id,
            ),
        ) as ex:
            futs: dict[Any, _GroupTask] = {}
            pending: set[Any] = set()

            def _next_task() -> _GroupTask | None:
                nonlocal active_group_batch
                if group_bound_execution_requested:
                    if (not active_group_batch) and (not pending):
                        active_group_batch = _plan_next_group_wave()
                    if active_group_batch:
                        return active_group_batch.popleft()
                    return None
                if legacy_group_queue:
                    return legacy_group_queue.popleft()
                return None

            def _submit_more() -> None:
                submission_limit = int(max(1, dynamic_effective_workers))
                while len(pending) < submission_limit:
                    g = _next_task()
                    if g is None:
                        break
                    fut = ex.submit(_run_group_task_from_context, g)
                    futs[fut] = g
                    pending.add(fut)

            while True:
                _submit_more()
                if not pending:
                    break
                done, pending = wait(
                    pending,
                    timeout=float(pool_heartbeat_seconds),
                    return_when=FIRST_COMPLETED,
                )
                active_workers = _bounded_active_worker_count(
                    pending_count=len(pending),
                    effective_workers=int(executor_worker_count),
                )
                if not done:
                    _write_run_status_checkpoint("running", execution_mode)
                    _maybe_emit_progress(active_workers=active_workers)
                    _maybe_health_check(
                        active_workers=active_workers,
                        queue_backlog=_queue_backlog_hint(pending_count=len(pending)),
                        result_backlog_rows=max(0, len(all_results) - tasks_completed),
                    )
                    continue
                for fut in sorted(done, key=lambda f: str(futs[f].group_id)):
                    g = futs.pop(fut)
                    rows = _safe_execute_task(
                        g,
                        lambda _g, _f=fut: _f.result(),
                        candidates=candidates,
                        splits=splits,
                        scenarios=scenarios,
                        harness_cfg=harness_cfg,
                    )
                    _process_group_rows(
                        rows,
                        active_workers=active_workers,
                        queue_backlog=_queue_backlog_hint(pending_count=len(pending)),
                    )
                    if aborted:
                        for rem in list(pending):
                            rem.cancel()
                        pending.clear()
                        break
                if aborted:
                    break
    else:
        def _next_serial_group() -> _GroupTask | None:
            nonlocal active_group_batch
            if group_bound_execution_requested:
                if not active_group_batch:
                    active_group_batch = _plan_next_group_wave()
                if active_group_batch:
                    return active_group_batch.popleft()
                return None
            if legacy_group_queue:
                return legacy_group_queue.popleft()
            return None

        while True:
            g = _next_serial_group()
            if g is None:
                break
            timeout_sec = int(quick_settings.task_timeout_sec) if quick_settings.enabled else 0
            rows = _safe_execute_task(
                g,
                lambda _g: _run_with_timeout_alarm(
                    timeout_sec,
                    lambda: _run_group_task(
                        _g,
                        shared_base_state,
                        candidates,
                        splits,
                        scenarios,
                        m2_configs,
                        m3_configs,
                        m4_configs,
                        harness_cfg,
                    ),
                ),
                candidates=candidates,
                splits=splits,
                scenarios=scenarios,
                harness_cfg=harness_cfg,
            )
            _process_group_rows(
                rows,
                active_workers=1,
                queue_backlog=_queue_backlog_hint(pending_count=0),
            )
            if quick_settings.enabled and (
                (groups_completed % int(quick_settings.progress_every_groups) == 0)
                or (g is None)
            ):
                _maybe_emit_progress(active_workers=1)
            if aborted:
                break

    aborted_early = bool(aborted and tasks_completed < tasks_submitted)
    actual_group_task_count = int(planned_group_task_count if group_bound_execution_requested else len(legacy_group_tasks))

    # Deterministic collation order.
    all_results.sort(
        key=lambda r: (
            str(r.get("candidate_id", "")),
            str(r.get("split_id", "")),
            str(r.get("scenario_id", "")),
            str(r.get("task_id", "")),
        )
    )
    worker_result_queue: SimpleQueue = SimpleQueue()
    for row in all_results:
        worker_result_queue.put(row)

    ledger_rows: list[dict[str, Any]] = []
    while True:
        try:
            item = worker_result_queue.get_nowait()
        except Exception:
            break
        if isinstance(item, dict):
            ledger_rows.append(item)

    ok_results = [r for r in all_results if r.get("status") == "ok" and int(r.get("test_days", 0)) > 0]

    bench_sessions, bench_ret = _benchmark_daily_returns(base_state, harness_cfg.benchmark_symbol)
    try:
        (
            common_sessions,
            daily_mat_exec,
            daily_mat_raw,
            daily_bmk,
            baseline_candidate_ids,
            candidate_scenario_series,
        ) = _aggregate_candidate_baseline_matrices(
            ok_results,
            bench_sessions,
            bench_ret,
            candidate_ids=[str(c.candidate_id) for c in candidates],
            min_days=int(harness_cfg.daily_return_min_days),
        )
        stats_verdict = _compute_stats_verdict(
            daily_returns_matrix_exec=daily_mat_exec,
            daily_returns_matrix_raw=daily_mat_raw,
            daily_benchmark_returns=daily_bmk,
            candidate_ids=baseline_candidate_ids,
            harness_cfg=harness_cfg,
            report_root=report_root,
        )
    except Exception as exc:
        if quick_settings.enabled:
            common_sessions = np.zeros(0, dtype=np.int64)
            daily_mat_exec = np.zeros((0, 0), dtype=np.float64)
            daily_mat_raw = np.zeros((0, 0), dtype=np.float64)
            daily_bmk = np.zeros(0, dtype=np.float64)
            baseline_candidate_ids = []
            candidate_scenario_series = {}
            stats_verdict = {
                "leaderboard": [
                    {
                        "candidate_id": str(c.candidate_id),
                        "dsr": float("nan"),
                        "in_mcs": False,
                        "wrc_p": float("nan"),
                        "spa_p": float("nan"),
                        "pbo": None,
                        "pass": False,
                    }
                    for c in sorted(candidates, key=lambda x: str(x.candidate_id))
                ],
                "aborted": False,
                "abort_reason": "",
                "quick_run_stats_fallback": True,
                "quick_run_stats_error": f"{type(exc).__name__}: {exc}",
            }
        elif not aborted:
            if not first_exception_class:
                first_exception_class = type(exc).__name__
                first_exception_message = str(exc)
                first_exception_hash = _error_hash(first_exception_class, first_exception_message)
            _write_json(
                run_status_path,
                {
                    "run_id": run_id,
                    "phase": "failed_pre_aggregation",
                    "execution_mode": str(execution_mode),
                    "groups_done": int(groups_completed),
                    "groups_total": int(_groups_total_hint()),
                    "tasks_done": int(tasks_completed),
                    "tasks_total": int(tasks_submitted),
                    "failures_so_far": int(failure_count),
                    "compute_authority": compute_authority,
                    "feature_tensor_role": feature_tensor_role,
                    "execution_topology": _truth_build_execution_topology(
                        execution_mode,
                        use_process_pool,
                        process_pool_group_reuse_active,
                        base_sharing=base_sharing_runtime,
                    ),
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                    "first_exception": {
                        "class": first_exception_class,
                        "message": first_exception_message,
                        "error_hash": first_exception_hash,
                    },
                },
            )
            raise
        else:
            if not first_exception_class:
                first_exception_class = type(exc).__name__
                first_exception_message = str(exc)
                first_exception_hash = _error_hash(first_exception_class, first_exception_message)
            common_sessions = np.zeros(0, dtype=np.int64)
            daily_mat_exec = np.zeros((0, 0), dtype=np.float64)
            daily_mat_raw = np.zeros((0, 0), dtype=np.float64)
            daily_bmk = np.zeros(0, dtype=np.float64)
            baseline_candidate_ids = []
            candidate_scenario_series = {}
            stats_verdict = {
                "leaderboard": [
                    {
                        "candidate_id": str(c.candidate_id),
                        "dsr": float("nan"),
                        "in_mcs": False,
                        "wrc_p": float("nan"),
                        "spa_p": float("nan"),
                        "pbo": None,
                        "pass": False,
                    }
                    for c in sorted(candidates, key=lambda x: str(x.candidate_id))
                ],
                "aborted": True,
                "abort_reason": str(exc),
            }
    candidate_verdict = {
        str(x.get("candidate_id", "")): x
        for x in stats_verdict.get("leaderboard", [])
    }
    return _orchestrator_finalize_run_outputs(
        report_root=report_root,
        run_id=run_id,
        run_started_utc=run_started_utc,
        engine_cfg=engine_cfg,
        harness_cfg=harness_cfg,
        candidates=candidates,
        splits=splits,
        scenarios=scenarios,
        all_results=all_results,
        ok_results=ok_results,
        ledger_rows=ledger_rows,
        stats_verdict=stats_verdict,
        candidate_verdict=candidate_verdict,
        common_sessions=common_sessions,
        daily_mat_exec=daily_mat_exec,
        daily_mat_raw=daily_mat_raw,
        daily_bmk=daily_bmk,
        baseline_candidate_ids=baseline_candidate_ids,
        candidate_scenario_series=candidate_scenario_series,
        dq_bundle=dq_bundle,
        feature_tensor=feature_tensor,
        feature_handles_master=feature_handles_master,
        budget=int(memory_estimate.budget_bytes),
        avail=int(memory_estimate.available_bytes),
        tasks_completed=tasks_completed,
        tasks_submitted=tasks_submitted,
        groups_completed=groups_completed,
        failure_count=failure_count,
        failure_tracker=failure_tracker,
        aborted=aborted,
        aborted_early=aborted_early,
        abort_reason=abort_reason,
        execution_mode=execution_mode,
        use_process_pool=use_process_pool,
        effective_workers=int(dynamic_effective_workers if use_process_pool else 1),
        payload_safe=payload_safe,
        payload_arg_max_bytes=payload_arg_max_bytes,
        large_payload_passing_avoided=large_payload_passing_avoided,
        mp_start_method=str(mp_start_method if mp_start_method else mp.get_start_method(allow_none=True) or ""),
        runtime_seconds=float(time.perf_counter() - run_t0),
        dataset_hash=dataset_hash,
        keep_symbols=keep_symbols,
        symbols=symbols,
        keep_idx=keep_idx,
        ingest_meta=ingest_meta,
        n_group_tasks=int(actual_group_task_count),
        est_state=int(memory_estimate.per_worker_bytes),
        m2_configs=m2_configs,
        m3_configs=m3_configs,
        m4_configs=m4_configs,
        tensor_npz_path=tensor_npz_path,
        tensor_json_path=tensor_json_path,
        tensor_hash=tensor_hash,
        run_status_path=run_status_path,
        group_runtime_stats_path=group_runtime_stats_path,
        deadletter_path=deadletter_path,
        self_audit_report_path=self_audit_report_path,
        compute_authority=compute_authority,
        feature_tensor_role=feature_tensor_role,
        robustness_caps=ROBUSTNESS_CAPS,
        quick_settings=quick_settings,
        first_exception_class=first_exception_class,
        first_exception_message=first_exception_message,
        first_exception_hash=first_exception_hash,
        monitor=monitor,
        require_pandas_fn=_require_pandas,
        stack_payload_frames_fn=_stack_payload_frames,
        write_json_fn=_write_json,
        write_frozen_json_fn=_write_frozen_module5_json,
        build_candidate_artifacts_fn=_build_candidate_artifacts,
        collect_ledger_rows_fn=_collect_ledger_rows_from_results,
        ledger_write_fn=_ledger_write,
        git_hash_fn=_git_hash,
        stable_hash_obj_fn=_stable_hash_obj,
        execution_topology_fn=lambda execution_mode_now, use_process_pool_now: _truth_build_execution_topology(
            execution_mode_now,
            use_process_pool_now,
            process_pool_group_reuse_active,
            base_sharing=base_sharing_runtime,
        ),
        dq_accept=DQ_ACCEPT,
        dq_degrade=DQ_DEGRADE,
        dq_reject=DQ_REJECT,
        harness_output_cls=HarnessOutput,
    )


if __name__ == "__main__":
    log_event(get_logger("module5_harness"), "INFO", "module5_harness_ready", event_type="module5_ready")
