"""
Weightiz Institutional Engine - Module 5 Part 2 (Validation Harness)
=====================================================================

Validation harness and research orchestrator:
- Pandas IO boundary for minute OHLCV ingestion/alignment.
- Leakage-safe WF/CPCV split generation with purge+embargo.
- Adversarial stress perturbations on cloned tensor states.
- Deterministic orchestration of Module 2 -> Module 3 -> Module 4.
- Close-to-close daily return compression for candidate equity (overnight PnL preserved).
- Artifact export and statistical verdict wiring to Module 5 Part 1.

Architecture map:
1) Split construction: `_generate_wf_splits`, `_generate_cpcv_splits`.
2) Task dispatch: `_build_group_tasks`, `_safe_execute_task`, process pool/serial loop in `run_weightiz_harness`.
3) Candidate aggregation: `_aggregate_candidate_baseline_matrix` + `_build_candidate_artifacts`.
4) Artifact writing: `run_weightiz_harness` (run-level) + `_build_candidate_artifacts` (candidate-level).
5) Robustness scoring: `_build_candidate_artifacts` using `ROBUSTNESS_CAPS`.
6) Plateau clustering: `_plateau_key` + deterministic grouping in `_build_candidate_artifacts`.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
import copy
import hashlib
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import warnings
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover - runtime guard
    pd = None  # type: ignore[assignment]

from weightiz_module1_core import (
    EngineConfig,
    Phase,
    ProfileStatIdx,
    ScoreIdx,
    TensorState,
    preallocate_state,
    validate_loaded_market_slice,
    validate_state_hard,
)
from weightiz_module2_core import Module2Config, run_weightiz_profile_engine
from weightiz_module3_structure import (
    IB_MISSING_POLICY,
    IB_POLICY_NO_TRADE,
    ContextIdx,
    Module3Config,
    Module3Output,
    run_module3_structural_aggregation,
)
from weightiz_module4_strategy_funnel import Module4Config, Module4Output, RegimeIdx, run_module4_strategy_funnel
from weightiz_module5_stats import (
    deflated_sharpe_ratio,
    model_confidence_set,
    pbo_cscv,
    run_full_stats,
    spa_test,
    white_reality_check,
)
from weightiz_dq import DQ_ACCEPT, DQ_DEGRADE, DQ_REJECT, dq_apply, dq_validate
from weightiz_invariants import assert_or_flag_finite


ROBUSTNESS_CAPS: dict[str, float] = {
    "dd_cap": 0.35,
    "std_cap": 1.00,
    "conc_cap": 0.75,
}


@dataclass(frozen=True)
class Module5HarnessConfig:
    seed: int = 97
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
    # Test-only deterministic fault hooks.
    test_fail_task_ids: tuple[str, ...] = ()
    test_fail_ratio: float = 0.0


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


@dataclass(frozen=True)
class _GroupTask:
    group_id: str
    split_idx: int
    scenario_idx: int
    m2_idx: int
    m3_idx: int
    candidate_indices: tuple[int, ...]


_WORKER_CONTEXT: dict[str, Any] | None = None


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
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float64, np.float32, np.float16, np.float_)):
        return float(obj)
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8, np.int_)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


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
    raw = f"{error_type}|{error_msg}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _record_deadletter(path: Path, row: dict[str, Any]) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "candidate_id": str(row.get("candidate_id", "")),
        "split_id": str(row.get("split_id", "")),
        "scenario_id": str(row.get("scenario_id", "")),
        "seed": int(row.get("task_seed", 0)),
        "error_type": str(row.get("error_type", "")),
        "error_hash": str(row.get("error_hash", "")),
        "error_msg": str(row.get("error", "")),
        "traceback_preview": str(row.get("traceback", ""))[:800],
        "exception_signature": str(row.get("exception_signature", "")),
        "reason_codes": sorted([str(x) for x in row.get("quality_reason_codes", [])]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_to_jsonable(payload), ensure_ascii=False) + "\n")


def _should_abort_run(
    failure_count: int,
    total_tasks: int,
    failure_rate_threshold: float,
    failure_count_threshold: int,
) -> tuple[bool, str]:
    if total_tasks <= 0:
        return False, ""
    fail_rate = float(failure_count) / float(total_tasks)
    if int(failure_count) > int(failure_count_threshold):
        return True, f"failure_count>{int(failure_count_threshold)} ({int(failure_count)})"
    if fail_rate > float(failure_rate_threshold):
        return True, f"failure_rate>{float(failure_rate_threshold):.4f} ({fail_rate:.4f})"
    return False, ""


def _normalized_top_frame(traceback_text: str) -> str:
    tb = str(traceback_text or "")
    lines = [ln.strip() for ln in tb.splitlines() if ln.strip()]
    frame_lines = [ln for ln in lines if ln.startswith('File "') and ", line " in ln and ", in " in ln]
    if not frame_lines:
        return "unknown:0:unknown"
    ln = frame_lines[-1]
    # Format: File "...", line N, in FUNC
    try:
        p1 = ln.split('File "', 1)[1]
        path_part, rest = p1.split('", line ', 1)
        line_part, func_part = rest.split(", in ", 1)
        base = os.path.basename(path_part)
        lineno = int(line_part.strip())
        fn = func_part.strip()
        return f"{base}:{lineno}:{fn}"
    except Exception:
        return "unknown:0:unknown"


def _exception_signature(row: dict[str, Any]) -> tuple[str, str]:
    et = str(row.get("error_type", "")).strip() or "RuntimeError"
    top = str(row.get("top_frame", "")).strip()
    if not top:
        top = _normalized_top_frame(str(row.get("traceback", "")))
    return (et, top)


def _is_localized_reason_codes(reason_codes: list[str]) -> bool:
    rc = [str(x) for x in reason_codes]
    return any(
        c.startswith("DQ_")
        or c.startswith("INVARIANT_")
        or ("IB_MISSING" in c)
        or c == "TIMEOUT"
        for c in rc
    )


def _is_high_suspicion_exception(error_type: str) -> bool:
    return str(error_type) in {"ImportError", "TypeError", "KeyError", "IndexError"}


def _update_failure_tracker(
    tracker: dict[tuple[str, str], dict[str, set[str] | bool]],
    row: dict[str, Any],
) -> tuple[tuple[str, str], dict[str, set[str] | bool]]:
    sig = _exception_signature(row)
    rec = tracker.setdefault(
        sig,
        {
            "units": set(),
            "assets": set(),
            "candidates": set(),
            "high_suspicion": _is_high_suspicion_exception(sig[0]),
        },
    )
    rec["units"].add(str(row.get("task_id", "")))
    rec["candidates"].add(str(row.get("candidate_id", "")))
    assets = [str(x) for x in row.get("asset_keys", [])]
    for a in assets:
        rec["assets"].add(a)
    return sig, rec


def _should_abort_systemic(
    tracker: dict[tuple[str, str], dict[str, set[str] | bool]],
    row: dict[str, Any],
) -> tuple[bool, str]:
    if _is_localized_reason_codes([str(x) for x in row.get("quality_reason_codes", [])]):
        return False, ""
    sig = _exception_signature(row)
    rec = tracker.get(sig)
    if rec is None:
        return False, ""
    n_units = len(rec["units"])
    n_assets = len(rec["assets"])
    n_candidates = len(rec["candidates"])
    if n_units >= 3 and n_assets >= 2 and n_candidates >= 2:
        suspicion = "high" if bool(rec.get("high_suspicion", False)) else "standard"
        reason = (
            f"systemic_exception signature={sig[0]}|{sig[1]} "
            f"units={n_units} assets={n_assets} candidates={n_candidates} suspicion={suspicion}"
        )
        return True, reason
    return False, ""


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
    - otherwise use interpreter default
    """
    forced = str(os.environ.get("WEIGHTIZ_MP_START_METHOD", "")).strip().lower()
    if forced in {"spawn", "fork", "forkserver"}:
        return mp.get_context(forced), forced
    if sys.platform == "darwin":
        return mp.get_context("fork"), "fork"
    default_method = str(mp.get_start_method(allow_none=True) or "spawn")
    return mp.get_context(default_method), default_method


def _safe_execute_task(
    group: _GroupTask,
    executor_fn: Callable[[_GroupTask], list[dict[str, Any]]],
    candidates: list[CandidateSpec],
    splits: list[SplitSpec],
    scenarios: list[StressScenario],
    harness_cfg: Module5HarnessConfig,
) -> list[dict[str, Any]]:
    try:
        return executor_fn(group)
    except Exception as exc:
        split = splits[group.split_idx]
        scenario = scenarios[group.scenario_idx]
        err_type = type(exc).__name__
        err_msg = str(exc)
        err_hash = _error_hash(err_type, err_msg)
        tb = traceback.format_exc()
        top_frame = _normalized_top_frame(tb)
        sig = f"{err_type}|{top_frame}"
        reason_codes: list[str] = []
        if isinstance(exc, TimeoutError):
            reason_codes.append("TIMEOUT")
        out: list[dict[str, Any]] = []
        for ci in group.candidate_indices:
            c = candidates[int(ci)]
            task_id = f"{c.candidate_id}|{split.split_id}|{scenario.scenario_id}"
            t_seed = _seed_for_task(int(harness_cfg.seed), task_id)
            out.append(
                {
                    "task_id": task_id,
                    "candidate_id": c.candidate_id,
                    "split_id": split.split_id,
                    "scenario_id": scenario.scenario_id,
                    "status": "error",
                    "error_type": err_type,
                    "error_hash": err_hash,
                    "error": f"{err_type}: {err_msg}",
                    "traceback": tb,
                    "top_frame": top_frame,
                    "exception_signature": sig,
                    "session_ids": np.zeros(0, dtype=np.int64),
                    "daily_returns": np.zeros(0, dtype=np.float64),
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
                    "task_seed": int(t_seed),
                    "asset_keys": [],
                    "quality_reason_codes": sorted(reason_codes),
                    "dqs_min": 0.0,
                    "dqs_median": 0.0,
                }
            )
        return out


def _init_worker_context(
    base_state: TensorState,
    candidates: list[CandidateSpec],
    splits: list[SplitSpec],
    scenarios: list[StressScenario],
    m2_configs: list[Module2Config],
    m3_configs: list[Module3Config],
    m4_configs: list[Module4Config],
    harness_cfg: Module5HarnessConfig,
) -> None:
    global _WORKER_CONTEXT
    repo_root = str(Path(__file__).resolve().parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    # Keep worker BLAS/Arrow threading deterministic and avoid nested oversubscription.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("ARROW_NUM_THREADS", "1")
    _WORKER_CONTEXT = {
        "base_state": base_state,
        "candidates": candidates,
        "splits": splits,
        "scenarios": scenarios,
        "m2_configs": m2_configs,
        "m3_configs": m3_configs,
        "m4_configs": m4_configs,
        "harness_cfg": harness_cfg,
    }


def _run_group_task_from_context(group: _GroupTask) -> list[dict[str, Any]]:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Worker context not initialized")
    return _run_group_task(
        group=group,
        base_state=_WORKER_CONTEXT["base_state"],
        candidates=_WORKER_CONTEXT["candidates"],
        splits=_WORKER_CONTEXT["splits"],
        scenarios=_WORKER_CONTEXT["scenarios"],
        m2_configs=_WORKER_CONTEXT["m2_configs"],
        m3_configs=_WORKER_CONTEXT["m3_configs"],
        m4_configs=_WORKER_CONTEXT["m4_configs"],
        harness_cfg=_WORKER_CONTEXT["harness_cfg"],
    )


def _assert_finite(name: str, arr: np.ndarray) -> None:
    if not np.all(np.isfinite(arr)):
        bad = np.argwhere(~np.isfinite(arr))[:8]
        raise RuntimeError(f"{name} contains non-finite values at indices {bad.tolist()}")


def _clone_state(state: TensorState) -> TensorState:
    # Deep copy preserves strict immutability of base_state across workers/tasks.
    return copy.deepcopy(state)


def _clone_m3(m3: Module3Output) -> Module3Output:
    return Module3Output(
        block_id_t=m3.block_id_t.copy(),
        block_seq_t=m3.block_seq_t.copy(),
        block_end_flag_t=m3.block_end_flag_t.copy(),
        block_start_t_index_t=m3.block_start_t_index_t.copy(),
        block_end_t_index_t=m3.block_end_t_index_t.copy(),
        block_features_tak=m3.block_features_tak.copy(),
        block_valid_ta=m3.block_valid_ta.copy(),
        context_tac=m3.context_tac.copy(),
        context_valid_ta=m3.context_valid_ta.copy(),
        context_source_t_index_ta=m3.context_source_t_index_ta.copy(),
        ib_defined_ta=None if m3.ib_defined_ta is None else m3.ib_defined_ta.copy(),
    )


def _available_memory_bytes() -> int:
    # Unix/macOS fast path.
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        return page_size * pages
    except Exception:
        return 8 * 1024 * 1024 * 1024


def _estimate_state_bytes(T: int, A: int, B: int) -> int:
    # Conservative rough estimate for one full tensor state + outputs.
    core = T * A * B * 8 * 3  # vp, vp_delta, temp profile buffers
    ta = T * A * 8 * 24
    t = T * 8 * 16
    overhead = 128 * 1024 * 1024
    return int(core + ta + t + overhead)


def _find_col(df: Any, candidates: tuple[str, ...], name: str) -> str:
    cols = {str(c).strip().lower(): str(c) for c in df.columns}
    for cand in candidates:
        if cand in cols:
            return cols[cand]
    raise RuntimeError(f"Missing required column '{name}' in input file")


def _load_asset_frame(path: str, tz_name: str) -> Any:
    pdx = _require_pandas()
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"Data path does not exist: {path}")

    if p.suffix.lower() == ".parquet":
        df = pdx.read_parquet(p)
    else:
        df = pdx.read_csv(p)

    ts_col = None
    cols = {str(c).strip().lower(): str(c) for c in df.columns}
    for cand in ("timestamp", "ts", "datetime", "date", "time"):
        if cand in cols:
            ts_col = cols[cand]
            break
    o_col = _find_col(df, ("open", "o"), "open")
    h_col = _find_col(df, ("high", "h"), "high")
    l_col = _find_col(df, ("low", "l"), "low")
    c_col = _find_col(df, ("close", "c"), "close")
    v_col = _find_col(df, ("volume", "vol", "v"), "volume")

    if ts_col is not None:
        ts_raw = pdx.to_datetime(df[ts_col], utc=True, errors="coerce")
    elif isinstance(df.index, pdx.DatetimeIndex):
        ts_raw = pdx.to_datetime(df.index, utc=True, errors="coerce")
    else:
        raise RuntimeError(f"Missing required column 'timestamp' in input file and index is not DatetimeIndex: {path}")
    ts_idx = pdx.DatetimeIndex(ts_raw)
    keep = np.asarray(ts_idx.notna(), dtype=bool)
    if not np.any(keep):
        raise RuntimeError(f"No parseable timestamps in {path}")

    out = pdx.DataFrame(
        {
            # Keep canonical UTC time on the IO boundary.
            "timestamp": ts_idx[keep].floor("min"),
            "open": pdx.to_numeric(df.loc[keep, o_col], errors="coerce"),
            "high": pdx.to_numeric(df.loc[keep, h_col], errors="coerce"),
            "low": pdx.to_numeric(df.loc[keep, l_col], errors="coerce"),
            "close": pdx.to_numeric(df.loc[keep, c_col], errors="coerce"),
            "volume": pdx.to_numeric(df.loc[keep, v_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    out = out.drop_duplicates(subset=["timestamp"], keep="last")
    out = out.set_index("timestamp")
    return out


def _validate_utc_minute_index(idx: Any, label: str) -> Any:
    pdx = _require_pandas()
    if not isinstance(idx, pdx.DatetimeIndex):
        raise RuntimeError(f"{label} index must be pandas.DatetimeIndex, got {type(idx)!r}")
    if idx.tz is None:
        raise RuntimeError(f"{label} index must be timezone-aware")
    idx_utc = idx.tz_convert("UTC")

    if bool(idx_utc.has_duplicates):
        dup = idx_utc[idx_utc.duplicated(keep=False)][0]
        raise RuntimeError(f"{label} index has duplicate timestamp after UTC conversion: {dup!s}")

    ts_ns = idx_utc.asi8.astype(np.int64)
    if ts_ns.size > 1:
        d = np.diff(ts_ns)
        bad = np.flatnonzero(d <= 0)
        if bad.size > 0:
            i = int(bad[0])
            raise RuntimeError(
                f"{label} index must be strictly increasing in UTC: "
                f"t[{i}]={idx_utc[i]!s}, t[{i + 1}]={idx_utc[i + 1]!s}"
            )

    bad_minute = (
        (idx_utc.second.to_numpy(dtype=np.int64) != 0)
        | (idx_utc.microsecond.to_numpy(dtype=np.int64) != 0)
        | (idx_utc.nanosecond.to_numpy(dtype=np.int64) != 0)
    )
    if np.any(bad_minute):
        i = int(np.flatnonzero(bad_minute)[0])
        raise RuntimeError(
            f"{label} index must be aligned to UTC minute grid "
            f"(second/microsecond/nanosecond all zero); offending timestamp={idx_utc[i]!s}"
        )

    return idx_utc


def _build_clock_override_from_utc(
    ts_ns_utc: np.ndarray,
    cfg: EngineConfig,
    tz_name: str,
) -> dict[str, np.ndarray]:
    """
    Build deterministic clock arrays from UTC timestamps using zoneinfo.
    This keeps DST logic at the ingestion boundary (spec-true handoff).
    """
    T = int(ts_ns_utc.shape[0])
    if T <= 0:
        raise RuntimeError("Cannot build clock override for empty timestamps")

    tz_local = ZoneInfo(str(tz_name))
    tz_utc = ZoneInfo("UTC")

    minute_of_day = np.empty(T, dtype=np.int16)
    day_ord = np.empty(T, dtype=np.int64)
    for i in range(T):
        ns = int(ts_ns_utc[i])
        sec = ns // 1_000_000_000
        nsec = ns % 1_000_000_000
        dtu = datetime.fromtimestamp(sec, tz=tz_utc).replace(microsecond=int(nsec // 1000))
        dtl = dtu.astimezone(tz_local)
        minute_of_day[i] = np.int16(int(dtl.hour) * 60 + int(dtl.minute))
        day_ord[i] = np.int64(int(dtl.date().toordinal()))

    tod = (minute_of_day.astype(np.int32) - int(cfg.rth_open_minute)).astype(np.int16)
    session_change = np.empty(T, dtype=bool)
    session_change[0] = True
    session_change[1:] = day_ord[1:] != day_ord[:-1]
    session_id = np.cumsum(session_change, dtype=np.int64) - 1

    gap_min = np.zeros(T, dtype=np.float64)
    gap_min[1:] = (ts_ns_utc[1:] - ts_ns_utc[:-1]) / float(60 * 1_000_000_000)
    reset_flag = ((gap_min > float(cfg.gap_reset_minutes)) | session_change).astype(np.int8)
    reset_flag = np.where(reset_flag != 0, np.int8(1), np.int8(0)).astype(np.int8)
    reset_flag[0] = np.int8(1)
    gap_min[0] = 0.0

    valid_reset = np.isin(reset_flag, np.asarray([0, 1], dtype=np.int8))
    if not np.all(valid_reset):
        i = int(np.flatnonzero(~valid_reset)[0])
        raise RuntimeError(
            f"Clock override reset_flag must be binary {{0,1}}: t={i}, value={int(reset_flag[i])}"
        )
    if np.any(gap_min < 0.0):
        i = int(np.flatnonzero(gap_min < 0.0)[0])
        raise RuntimeError(
            f"Clock override gap_min must be non-negative: t={i}, value={float(gap_min[i]):.6f}"
        )

    phase = np.full(T, np.int8(Phase.WARMUP), dtype=np.int8)
    is_live = (tod >= int(cfg.warmup_minutes)) & (minute_of_day < int(cfg.flat_time_minute))
    phase[is_live] = np.int8(Phase.LIVE)
    is_select = (minute_of_day == int(cfg.flat_time_minute)) & (tod >= int(cfg.warmup_minutes))
    phase[is_select] = np.int8(Phase.OVERNIGHT_SELECT)
    is_flat = minute_of_day > int(cfg.flat_time_minute)
    phase[is_flat] = np.int8(Phase.FLATTEN)

    return {
        "minute_of_day": minute_of_day,
        "tod": tod,
        "session_id": session_id,
        "gap_min": gap_min,
        "reset_flag": reset_flag,
        "phase": phase,
    }


def _compute_bar_valid(
    open_px: np.ndarray,
    high_px: np.ndarray,
    low_px: np.ndarray,
    close_px: np.ndarray,
    volume: np.ndarray,
) -> np.ndarray:
    finite = (
        np.isfinite(open_px)
        & np.isfinite(high_px)
        & np.isfinite(low_px)
        & np.isfinite(close_px)
        & np.isfinite(volume)
    )
    phys = (
        (high_px >= low_px)
        & (high_px >= open_px)
        & (high_px >= close_px)
        & (low_px <= open_px)
        & (low_px <= close_px)
        & (volume >= 0.0)
    )
    return finite & phys


def _ingest_master_aligned(
    data_paths: list[str],
    symbols: list[str],
    engine_cfg: EngineConfig,
    harness_cfg: Module5HarnessConfig,
    data_loader_func: Callable[[str, str], Any] | None = None,
) -> tuple[TensorState, np.ndarray, list[str], np.ndarray, dict[str, Any], np.ndarray, dict[str, Any]]:
    if len(data_paths) != len(symbols):
        raise RuntimeError("data_paths and symbols lengths must match")

    pdx = _require_pandas()

    loader = data_loader_func if data_loader_func is not None else _load_asset_frame

    raw_frames: list[Any] = []
    dq_day_reports: list[dict[str, Any]] = []
    dq_bar_flags_rows: list[dict[str, Any]] = []
    for a, p in enumerate(data_paths):
        fr = loader(p, harness_cfg.timezone)
        dq_reports = dq_validate(
            df=fr,
            symbol=str(symbols[a]),
            tz_name=str(harness_cfg.timezone),
            session_open_minute=int(engine_cfg.rth_open_minute),
            session_close_minute=int(engine_cfg.flat_time_minute),
            timeframe_min=None,
        )
        fr_repaired, dq_reports_out, dq_bar_flags = dq_apply(
            df=fr,
            reports=dq_reports,
            tz_name=str(harness_cfg.timezone),
        )

        dq_day_reports.extend([r.to_row() for r in dq_reports_out])
        if dq_bar_flags.shape[0] > 0:
            dq_bar_flags_rows.extend(dq_bar_flags.to_dict(orient="records"))

        idx_utc = _validate_utc_minute_index(fr_repaired.index, f"asset={symbols[a]} path={p}")
        fr2 = fr_repaired.copy()
        fr2.index = idx_utc
        raw_frames.append(fr2)

    master_idx = raw_frames[0].index
    for fr in raw_frames[1:]:
        master_idx = master_idx.union(fr.index)
    master_idx = master_idx.sort_values()
    master_idx = _validate_utc_minute_index(master_idx, "master_idx (after union+sort)")

    T = int(master_idx.shape[0])
    A0 = int(len(symbols))
    open_ta = np.full((T, A0), np.nan, dtype=np.float64)
    high_ta = np.full((T, A0), np.nan, dtype=np.float64)
    low_ta = np.full((T, A0), np.nan, dtype=np.float64)
    close_ta = np.full((T, A0), np.nan, dtype=np.float64)
    vol_ta = np.full((T, A0), np.nan, dtype=np.float64)
    dqs_ta = np.full((T, A0), 1.0, dtype=np.float64)

    for a, fr in enumerate(raw_frames):
        re = fr.reindex(master_idx)
        open_ta[:, a] = re["open"].to_numpy(dtype=np.float64)
        high_ta[:, a] = re["high"].to_numpy(dtype=np.float64)
        low_ta[:, a] = re["low"].to_numpy(dtype=np.float64)
        close_ta[:, a] = re["close"].to_numpy(dtype=np.float64)
        vol_ta[:, a] = re["volume"].to_numpy(dtype=np.float64)
        if "dqs_day" in re.columns:
            dqs_col = pdx.to_numeric(re["dqs_day"], errors="coerce").to_numpy(dtype=np.float64)
            dqs_ta[:, a] = np.where(np.isfinite(dqs_col), dqs_col, 1.0)

    bar_valid_ta = _compute_bar_valid(open_ta, high_ta, low_ta, close_ta, vol_ta)
    coverage = np.mean(bar_valid_ta, axis=0)
    keep_assets = coverage >= float(harness_cfg.min_asset_coverage)

    if np.sum(keep_assets) < 2:
        raise RuntimeError(
            f"Coverage filter removed too many assets: kept={int(np.sum(keep_assets))}, required>=2"
        )

    keep_idx = np.where(keep_assets)[0]
    keep_symbols = [symbols[i] for i in keep_idx.tolist()]

    tick = np.asarray(engine_cfg.tick_size, dtype=np.float64)
    if tick.shape != (A0,):
        raise RuntimeError(
            f"engine_cfg.tick_size shape mismatch: got {tick.shape}, expected {(A0,)}"
        )

    open_keep = open_ta[:, keep_idx]
    high_keep = high_ta[:, keep_idx]
    low_keep = low_ta[:, keep_idx]
    close_keep = close_ta[:, keep_idx]
    vol_keep = vol_ta[:, keep_idx]
    dqs_keep = dqs_ta[:, keep_idx]
    bar_keep = bar_valid_ta[:, keep_idx]
    tick_keep = tick[keep_idx]

    ts_ns = master_idx.asi8.astype(np.int64)
    if ts_ns.size > 1 and np.any(np.diff(ts_ns) <= 0):
        i = int(np.flatnonzero(np.diff(ts_ns) <= 0)[0])
        raise RuntimeError(
            f"master_idx -> ts_ns must be strictly increasing int64: "
            f"i={i}, ts_ns[i]={int(ts_ns[i])}, ts_ns[i+1]={int(ts_ns[i + 1])}"
        )

    cfg = replace(engine_cfg, T=T, A=int(keep_idx.size), tick_size=tick_keep.copy())
    clk_override = _build_clock_override_from_utc(ts_ns, cfg, harness_cfg.timezone)
    state = preallocate_state(
        ts_ns=ts_ns,
        cfg=cfg,
        symbols=tuple(keep_symbols),
        clock_override=clk_override,
    )

    state.open_px[:, :] = open_keep
    state.high_px[:, :] = high_keep
    state.low_px[:, :] = low_keep
    state.close_px[:, :] = close_keep
    state.volume[:, :] = vol_keep
    state.bar_valid[:, :] = bar_keep
    state.dqs_day_ta = dqs_keep.copy()

    # Required finite placeholders for Module 1 slice validation; Module 2 overwrites causally.
    atr0 = np.maximum(4.0 * tick_keep[None, :], 1e-12)
    state.rvol[:, :] = np.where(bar_keep, 1.0, np.nan)
    state.atr_floor[:, :] = np.where(bar_keep, atr0, np.nan)

    validate_loaded_market_slice(state, 0, state.cfg.T)
    validate_state_hard(state)

    ingest_meta = {
        "master_rows": T,
        "assets_input": A0,
        "assets_kept": int(keep_idx.size),
        "coverage": coverage.tolist(),
        "symbols_kept": keep_symbols,
        "dq_counts": {
            "accept": int(sum(1 for r in dq_day_reports if str(r.get("decision")) == DQ_ACCEPT)),
            "degrade": int(sum(1 for r in dq_day_reports if str(r.get("decision")) == DQ_DEGRADE)),
            "reject": int(sum(1 for r in dq_day_reports if str(r.get("decision")) == DQ_REJECT)),
            "n_days_total": int(len(dq_day_reports)),
        },
    }

    dq_bundle = {
        "day_reports": dq_day_reports,
        "bar_flags_rows": dq_bar_flags_rows,
    }

    return state, keep_idx, keep_symbols, master_idx.asi8.astype(np.int64), ingest_meta, tick_keep, dq_bundle


def _session_bounds(session_id: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sid = np.asarray(session_id, dtype=np.int64)
    T = int(sid.shape[0])
    starts = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]]).astype(np.int64)
    ends = np.r_[starts[1:], T].astype(np.int64)
    sessions = sid[starts]
    return starts, ends, sessions


def _sessions_to_idx(session_id: np.ndarray, sessions: np.ndarray) -> np.ndarray:
    mask = np.isin(session_id, sessions)
    return np.flatnonzero(mask).astype(np.int64)


def _contiguous_segments(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if idx.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    cut = np.flatnonzero(np.diff(idx) > 1) + 1
    starts = np.r_[idx[0], idx[cut]]
    ends = np.r_[idx[cut - 1] + 1, idx[-1] + 1]
    return starts.astype(np.int64), ends.astype(np.int64)


def _apply_purge_embargo(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    T: int,
    purge_bars: int,
    embargo_bars: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tr = np.unique(np.asarray(train_idx, dtype=np.int64))
    te = np.unique(np.asarray(test_idx, dtype=np.int64))

    seg_starts, _seg_ends = _contiguous_segments(te)

    purge_mask = np.zeros(T, dtype=bool)
    for t0 in seg_starts.tolist():
        lo = max(0, int(t0) - int(purge_bars))
        hi = int(t0)
        if hi > lo:
            purge_mask[lo:hi] = True

    embargo_mask = np.zeros(T, dtype=bool)
    for t0 in seg_starts.tolist():
        lo = int(t0)
        hi = min(T, int(t0) + int(embargo_bars))
        if hi > lo:
            embargo_mask[lo:hi] = True

    purge_idx = np.flatnonzero(purge_mask & np.isin(np.arange(T, dtype=np.int64), tr)).astype(np.int64)
    embargo_idx = np.flatnonzero(embargo_mask & np.isin(np.arange(T, dtype=np.int64), te)).astype(np.int64)

    tr2 = tr[~np.isin(tr, purge_idx)]
    te2 = te[~np.isin(te, embargo_idx)]

    return tr2.astype(np.int64), te2.astype(np.int64), purge_idx, embargo_idx


def _generate_wf_splits(state: TensorState, cfg: Module5HarnessConfig) -> list[SplitSpec]:
    starts, ends, sessions = _session_bounds(state.session_id)
    n_s = int(sessions.size)

    train_n = int(cfg.wf_train_sessions)
    test_n = int(cfg.wf_test_sessions)
    step_n = int(cfg.wf_step_sessions)

    out: list[SplitSpec] = []
    if n_s < train_n + test_n:
        return out

    sid = state.session_id.astype(np.int64)
    fold = 0
    s0 = 0
    while s0 + train_n + test_n <= n_s:
        tr_s = sessions[s0 : s0 + train_n]
        te_s = sessions[s0 + train_n : s0 + train_n + test_n]

        tr_idx = _sessions_to_idx(sid, tr_s)
        te_idx = _sessions_to_idx(sid, te_s)

        tr_idx, te_idx, purge_idx, embargo_idx = _apply_purge_embargo(
            tr_idx,
            te_idx,
            state.cfg.T,
            cfg.purge_bars,
            cfg.embargo_bars,
        )

        out.append(
            SplitSpec(
                split_id=f"wf_{fold:03d}",
                mode="wf",
                train_idx=tr_idx,
                test_idx=te_idx,
                purge_idx=purge_idx,
                embargo_idx=embargo_idx,
                session_train_bounds=(int(tr_s[0]), int(tr_s[-1])),
                session_test_bounds=(int(te_s[0]), int(te_s[-1])),
                purge_bars=int(cfg.purge_bars),
                embargo_bars=int(cfg.embargo_bars),
                total_bars=int(state.cfg.T),
            )
        )
        fold += 1
        s0 += max(1, step_n)

    return out


def _generate_cpcv_splits(state: TensorState, cfg: Module5HarnessConfig) -> list[SplitSpec]:
    starts, ends, sessions = _session_bounds(state.session_id)
    n_s = int(sessions.size)
    S = int(cfg.cpcv_slices)
    k = int(cfg.cpcv_k_test)

    if S < 2 or k < 1 or k >= S:
        raise RuntimeError(f"Invalid CPCV params: slices={S}, k_test={k}")
    if n_s < S:
        return []

    # Deterministic contiguous session groups.
    groups = np.array_split(np.arange(n_s, dtype=np.int64), S)
    out: list[SplitSpec] = []
    sid = state.session_id.astype(np.int64)

    comb_iter = itertools.combinations(range(S), k)
    for i, test_grp_idx in enumerate(comb_iter):
        test_loc = np.concatenate([groups[g] for g in test_grp_idx if groups[g].size > 0]).astype(np.int64)
        if test_loc.size == 0:
            continue
        train_loc_mask = np.ones(n_s, dtype=bool)
        train_loc_mask[test_loc] = False
        train_loc = np.where(train_loc_mask)[0].astype(np.int64)
        if train_loc.size == 0:
            continue

        tr_s = sessions[train_loc]
        te_s = sessions[test_loc]

        tr_idx = _sessions_to_idx(sid, tr_s)
        te_idx = _sessions_to_idx(sid, te_s)

        tr_idx, te_idx, purge_idx, embargo_idx = _apply_purge_embargo(
            tr_idx,
            te_idx,
            state.cfg.T,
            cfg.purge_bars,
            cfg.embargo_bars,
        )

        out.append(
            SplitSpec(
                split_id=f"cpcv_{i:03d}",
                mode="cpcv",
                train_idx=tr_idx,
                test_idx=te_idx,
                purge_idx=purge_idx,
                embargo_idx=embargo_idx,
                session_train_bounds=(int(np.min(tr_s)), int(np.max(tr_s))),
                session_test_bounds=(int(np.min(te_s)), int(np.max(te_s))),
                purge_bars=int(cfg.purge_bars),
                embargo_bars=int(cfg.embargo_bars),
                total_bars=int(state.cfg.T),
            )
        )

    return out


def _generate_quick_fallback_split(state: TensorState, cfg: Module5HarnessConfig) -> list[SplitSpec]:
    T = int(state.cfg.T)
    if T < 2:
        return []
    cut = int(max(1, min(T - 1, T // 2)))
    tr_idx = np.arange(0, cut, dtype=np.int64)
    te_idx = np.arange(cut, T, dtype=np.int64)
    tr_idx, te_idx, purge_idx, embargo_idx = _apply_purge_embargo(
        tr_idx,
        te_idx,
        T,
        int(max(0, cfg.purge_bars)),
        int(max(0, cfg.embargo_bars)),
    )
    if tr_idx.size == 0 or te_idx.size == 0:
        return []
    sid = state.session_id.astype(np.int64)
    return [
        SplitSpec(
            split_id="wf_quick_000",
            mode="wf",
            train_idx=tr_idx,
            test_idx=te_idx,
            purge_idx=purge_idx,
            embargo_idx=embargo_idx,
            session_train_bounds=(int(np.min(sid[tr_idx])), int(np.max(sid[tr_idx]))),
            session_test_bounds=(int(np.min(sid[te_idx])), int(np.max(sid[te_idx]))),
            purge_bars=int(max(0, cfg.purge_bars)),
            embargo_bars=int(max(0, cfg.embargo_bars)),
            total_bars=T,
        )
    ]


def _validate_split(spec: SplitSpec, enforce_guard: bool) -> None:
    tr = np.asarray(spec.train_idx, dtype=np.int64)
    te = np.asarray(spec.test_idx, dtype=np.int64)
    purge = np.asarray(spec.purge_idx, dtype=np.int64)
    embargo = np.asarray(spec.embargo_idx, dtype=np.int64)

    if tr.size == 0 or te.size == 0:
        raise RuntimeError(f"Split {spec.split_id} has empty train or test index set")
    if np.any(np.diff(tr) < 0) or np.any(np.diff(te) < 0):
        raise RuntimeError(f"Split {spec.split_id} indices must be sorted")
    if tr.size != np.unique(tr).size or te.size != np.unique(te).size:
        raise RuntimeError(f"Split {spec.split_id} indices must be unique")
    inter = np.intersect1d(tr, te)
    if inter.size > 0:
        raise RuntimeError(f"Split {spec.split_id} leakage: train/test overlap exists")

    if enforce_guard:
        if tr.size != np.unique(tr).size:
            raise RuntimeError(f"Split {spec.split_id} train_idx must be unique")
        if te.size != np.unique(te).size:
            raise RuntimeError(f"Split {spec.split_id} test_idx must be unique")
        if purge.size != np.unique(purge).size:
            raise RuntimeError(f"Split {spec.split_id} purge_idx must be unique")
        if embargo.size != np.unique(embargo).size:
            raise RuntimeError(f"Split {spec.split_id} embargo_idx must be unique")

        all_idx = np.r_[tr, te, purge, embargo].astype(np.int64, copy=False)
        if all_idx.size == 0:
            raise RuntimeError(f"Split {spec.split_id} has no indices to validate")
        total_bars = int(spec.total_bars) if int(spec.total_bars) > 0 else int(np.max(all_idx) + 1)
        if np.any(all_idx < 0) or np.any(all_idx >= total_bars):
            bad = int(all_idx[(all_idx < 0) | (all_idx >= total_bars)][0])
            raise RuntimeError(
                f"Split {spec.split_id} has out-of-range index {bad} for total_bars={total_bars}"
            )

        leak_tp = np.intersect1d(tr, purge)
        if leak_tp.size > 0:
            i = int(leak_tp[0])
            raise RuntimeError(
                f"Split {spec.split_id} purge leakage: train_idx intersects purge_idx, "
                f"first_offending_index={i}"
            )

        leak_te = np.intersect1d(te, embargo)
        if leak_te.size > 0:
            i = int(leak_te[0])
            raise RuntimeError(
                f"Split {spec.split_id} embargo leakage: test_idx intersects embargo_idx, "
                f"first_offending_index={i}"
            )

        purge_bars = int(max(0, spec.purge_bars))
        embargo_bars = int(max(0, spec.embargo_bars))

        train_full = np.unique(np.r_[tr, purge]).astype(np.int64)
        test_full = np.unique(np.r_[te, embargo]).astype(np.int64)
        seg_starts, _seg_ends = _contiguous_segments(test_full)

        for t0 in seg_starts.tolist():
            t0_i = int(t0)

            lo_p = max(0, t0_i - purge_bars)
            hi_p = t0_i
            if hi_p > lo_p:
                bad_train = tr[(tr >= lo_p) & (tr < hi_p)]
                if bad_train.size > 0:
                    i = int(bad_train[0])
                    raise RuntimeError(
                        f"Split {spec.split_id} purge guard violated: "
                        f"forbidden_train_range=[{lo_p},{hi_p}) first_offending_index={i}"
                    )
                expected_purge = train_full[(train_full >= lo_p) & (train_full < hi_p)]
                if expected_purge.size > 0:
                    missing = expected_purge[~np.isin(expected_purge, purge)]
                    if missing.size > 0:
                        i = int(missing[0])
                        raise RuntimeError(
                            f"Split {spec.split_id} purge carve incomplete: "
                            f"required_range=[{lo_p},{hi_p}) first_offending_index={i}"
                        )

            lo_e = t0_i
            hi_e = min(total_bars, t0_i + embargo_bars)
            if hi_e > lo_e:
                bad_test = te[(te >= lo_e) & (te < hi_e)]
                if bad_test.size > 0:
                    i = int(bad_test[0])
                    raise RuntimeError(
                        f"Split {spec.split_id} embargo guard violated: "
                        f"forbidden_test_range=[{lo_e},{hi_e}) first_offending_index={i}"
                    )
                expected_embargo = test_full[(test_full >= lo_e) & (test_full < hi_e)]
                if expected_embargo.size > 0:
                    missing = expected_embargo[~np.isin(expected_embargo, embargo)]
                    if missing.size > 0:
                        i = int(missing[0])
                        raise RuntimeError(
                            f"Split {spec.split_id} embargo carve incomplete: "
                            f"required_range=[{lo_e},{hi_e}) first_offending_index={i}"
                        )


def _default_stress_scenarios(cfg: Module5HarnessConfig) -> list[StressScenario]:
    if cfg.stress_profile == "baseline_mild_severe":
        return [
            StressScenario(
                scenario_id="baseline",
                name="baseline",
                missing_burst_prob=0.0,
                missing_burst_min=0,
                missing_burst_max=0,
                jitter_sigma_bps=0.0,
                slippage_mult=1.0,
                enabled=True,
            ),
            StressScenario(
                scenario_id="mild",
                name="mild",
                missing_burst_prob=0.0005,
                missing_burst_min=2,
                missing_burst_max=5,
                jitter_sigma_bps=1.5,
                slippage_mult=1.5,
                enabled=True,
            ),
            StressScenario(
                scenario_id="severe",
                name="severe",
                missing_burst_prob=0.0020,
                missing_burst_min=5,
                missing_burst_max=20,
                jitter_sigma_bps=4.0,
                slippage_mult=3.0,
                enabled=True,
            ),
        ]
    raise RuntimeError(f"Unsupported stress_profile: {cfg.stress_profile}")


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
    if scenario.missing_burst_prob <= 0.0 or scenario.missing_burst_max <= 0:
        return

    T, A = state.bar_valid.shape
    start_mask = (rng.random((T, A)) < float(scenario.missing_burst_prob)) & active_t[:, None]
    starts = np.argwhere(start_mask)

    lo_len = int(max(1, scenario.missing_burst_min))
    hi_len = int(max(lo_len, scenario.missing_burst_max))

    for t0, a in starts.tolist():
        L = int(rng.integers(lo_len, hi_len + 1))
        t1 = min(T, int(t0) + L)
        state.open_px[t0:t1, a] = np.nan
        state.high_px[t0:t1, a] = np.nan
        state.low_px[t0:t1, a] = np.nan
        state.close_px[t0:t1, a] = np.nan
        state.volume[t0:t1, a] = np.nan
        state.rvol[t0:t1, a] = np.nan
        state.atr_floor[t0:t1, a] = np.nan
        state.bar_valid[t0:t1, a] = False


def _apply_jitter(
    state: TensorState,
    active_t: np.ndarray,
    scenario: StressScenario,
    rng: np.random.Generator,
) -> None:
    sigma_bps = float(scenario.jitter_sigma_bps)
    if sigma_bps <= 0.0:
        return

    eps = rng.normal(
        loc=0.0,
        scale=sigma_bps / 1e4,
        size=state.open_px.shape,
    ).astype(np.float64)
    mask = active_t[:, None] & state.bar_valid

    mult = 1.0 + eps
    state.open_px = np.where(mask, state.open_px * mult, state.open_px)
    state.high_px = np.where(mask, state.high_px * mult, state.high_px)
    state.low_px = np.where(mask, state.low_px * mult, state.low_px)
    state.close_px = np.where(mask, state.close_px * mult, state.close_px)

    # Re-enforce OHLC ordering after noise.
    stacked = np.stack([state.open_px, state.high_px, state.low_px, state.close_px], axis=2)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        hi = np.nanmax(stacked, axis=2)
        lo = np.nanmin(stacked, axis=2)
    state.high_px = np.where(mask, hi, state.high_px)
    state.low_px = np.where(mask, lo, state.low_px)


def _recompute_bar_valid_inplace(state: TensorState) -> None:
    state.bar_valid[:, :] = _compute_bar_valid(
        state.open_px,
        state.high_px,
        state.low_px,
        state.close_px,
        state.volume,
    )


def _set_placeholders_from_bar_valid(state: TensorState) -> None:
    tick = np.asarray(state.eps.eps_div, dtype=np.float64)[None, :]
    atr0 = np.maximum(4.0 * tick, 1e-12)
    state.rvol[:, :] = np.where(state.bar_valid, 1.0, np.nan)
    state.atr_floor[:, :] = np.where(state.bar_valid, atr0, np.nan)


def _assert_placeholder_consistency(state: TensorState) -> None:
    valid = np.asarray(state.bar_valid, dtype=bool)
    if state.rvol.shape != valid.shape or state.atr_floor.shape != valid.shape:
        raise RuntimeError(
            f"Placeholder shape mismatch: rvol={state.rvol.shape}, atr_floor={state.atr_floor.shape}, "
            f"bar_valid={valid.shape}"
        )
    if np.any(valid):
        if not np.all(np.isfinite(state.rvol[valid])):
            bad = np.argwhere(valid & (~np.isfinite(state.rvol)))[0]
            raise RuntimeError(
                f"rvol must be finite on valid bars; first_offending_index={[int(bad[0]), int(bad[1])]}"
            )
        if not np.all(np.isfinite(state.atr_floor[valid])):
            bad = np.argwhere(valid & (~np.isfinite(state.atr_floor)))[0]
            raise RuntimeError(
                f"atr_floor must be finite on valid bars; first_offending_index={[int(bad[0]), int(bad[1])]}"
            )
    invalid = ~valid
    if np.any(invalid & np.isfinite(state.rvol)):
        bad = np.argwhere(invalid & np.isfinite(state.rvol))[0]
        raise RuntimeError(
            f"rvol must be NaN on invalid bars; first_offending_index={[int(bad[0]), int(bad[1])]}"
        )
    if np.any(invalid & np.isfinite(state.atr_floor)):
        bad = np.argwhere(invalid & np.isfinite(state.atr_floor))[0]
        raise RuntimeError(
            f"atr_floor must be NaN on invalid bars; first_offending_index={[int(bad[0]), int(bad[1])]}"
        )


def _apply_post_m2_invariants(state: TensorState, active_t: np.ndarray) -> list[str]:
    scope = np.asarray(active_t, dtype=bool)[:, None] & np.asarray(state.bar_valid, dtype=bool)
    updated, flags = assert_or_flag_finite(
        features={
            "profile_stats": np.asarray(state.profile_stats, dtype=np.float64),
            "scores": np.asarray(state.scores, dtype=np.float64),
        },
        valid_mask=scope,
        context="post_m2",
    )
    state.bar_valid[:, :] = np.where(scope, updated, state.bar_valid)
    _set_placeholders_from_bar_valid(state)
    return ["INVARIANT_POST_M2_NONFINITE"] if int(flags.get("invalid_count", 0)) > 0 else []


def _apply_post_m3_invariants(m3: Module3Output) -> list[str]:
    reasons: list[str] = []
    block_updated, block_flags = assert_or_flag_finite(
        features={"block_features_tak": np.asarray(m3.block_features_tak, dtype=np.float64)},
        valid_mask=np.asarray(m3.block_valid_ta, dtype=bool),
        context="post_m3_block",
    )
    m3.block_valid_ta[:, :] = block_updated
    if int(block_flags.get("invalid_count", 0)) > 0:
        reasons.append("INVARIANT_POST_M3_BLOCK_NONFINITE")

    ctx_updated, ctx_flags = assert_or_flag_finite(
        features={"context_tac": np.asarray(m3.context_tac, dtype=np.float64)},
        valid_mask=np.asarray(m3.context_valid_ta, dtype=bool),
        context="post_m3_context",
    )
    m3.context_valid_ta[:, :] = ctx_updated
    m3.context_source_t_index_ta[:, :] = np.where(m3.context_valid_ta, m3.context_source_t_index_ta, -1)
    m3.context_tac[:, :, :] = np.where(m3.context_valid_ta[:, :, None], m3.context_tac, np.nan)
    if int(ctx_flags.get("invalid_count", 0)) > 0:
        reasons.append("INVARIANT_POST_M3_CONTEXT_NONFINITE")

    if (m3.ib_defined_ta is not None) and (str(IB_MISSING_POLICY).upper().strip() == str(IB_POLICY_NO_TRADE)):
        ib_ok = np.asarray(m3.ib_defined_ta, dtype=bool)
        before = np.asarray(m3.block_valid_ta, dtype=bool)
        after = before & ib_ok
        if int(np.sum(before & (~after))) > 0:
            reasons.append("IB_MISSING_NO_TRADE")
        m3.block_valid_ta[:, :] = after

    return sorted(set(reasons))


def _apply_pre_m4_invariants(state: TensorState, m3: Module3Output) -> list[str]:
    reasons: list[str] = []
    updated_bar, bar_flags = assert_or_flag_finite(
        features={
            "open_px": np.asarray(state.open_px, dtype=np.float64),
            "high_px": np.asarray(state.high_px, dtype=np.float64),
            "low_px": np.asarray(state.low_px, dtype=np.float64),
            "close_px": np.asarray(state.close_px, dtype=np.float64),
            "volume": np.asarray(state.volume, dtype=np.float64),
            "profile_stats": np.asarray(state.profile_stats, dtype=np.float64),
            "scores": np.asarray(state.scores, dtype=np.float64),
        },
        valid_mask=np.asarray(state.bar_valid, dtype=bool),
        context="pre_m4_state",
    )
    if int(bar_flags.get("invalid_count", 0)) > 0:
        reasons.append("INVARIANT_PRE_M4_STATE_NONFINITE")
    state.bar_valid[:, :] = updated_bar
    _set_placeholders_from_bar_valid(state)

    updated_ctx, ctx_flags = assert_or_flag_finite(
        features={"context_tac": np.asarray(m3.context_tac, dtype=np.float64)},
        valid_mask=np.asarray(m3.context_valid_ta, dtype=bool),
        context="pre_m4_context",
    )
    if int(ctx_flags.get("invalid_count", 0)) > 0:
        reasons.append("INVARIANT_PRE_M4_CONTEXT_NONFINITE")
    m3.context_valid_ta[:, :] = updated_ctx
    m3.context_source_t_index_ta[:, :] = np.where(m3.context_valid_ta, m3.context_source_t_index_ta, -1)
    m3.context_tac[:, :, :] = np.where(m3.context_valid_ta[:, :, None], m3.context_tac, np.nan)

    if (m3.ib_defined_ta is not None) and (str(IB_MISSING_POLICY).upper().strip() == str(IB_POLICY_NO_TRADE)):
        ib_ok = np.asarray(m3.ib_defined_ta, dtype=bool)
        before = np.asarray(m3.block_valid_ta, dtype=bool)
        after = before & ib_ok
        if int(np.sum(before & (~after))) > 0:
            reasons.append("IB_MISSING_NO_TRADE")
        m3.block_valid_ta[:, :] = after

    return sorted(set(reasons))


def _assert_active_domain_ohlc(state: TensorState, active_t: np.ndarray) -> None:
    mask = np.asarray(active_t, dtype=bool)[:, None] & np.asarray(state.bar_valid, dtype=bool)
    if not np.any(mask):
        return

    finite_bad = mask & (
        ~np.isfinite(state.open_px)
        | ~np.isfinite(state.high_px)
        | ~np.isfinite(state.low_px)
        | ~np.isfinite(state.close_px)
        | ~np.isfinite(state.volume)
    )
    if np.any(finite_bad):
        bad = np.argwhere(finite_bad)[0]
        raise RuntimeError(
            f"Active-domain OHLC contains non-finite values at "
            f"t={int(bad[0])}, a={int(bad[1])}"
        )

    hl_bad = mask & (state.high_px < state.low_px)
    if np.any(hl_bad):
        bad = np.argwhere(hl_bad)[0]
        raise RuntimeError(f"Active-domain OHLC violation high<low at t={int(bad[0])}, a={int(bad[1])}")

    open_bad = mask & ((state.open_px < state.low_px) | (state.open_px > state.high_px))
    if np.any(open_bad):
        bad = np.argwhere(open_bad)[0]
        raise RuntimeError(
            f"Active-domain OHLC violation open outside [low,high] at t={int(bad[0])}, a={int(bad[1])}"
        )

    close_bad = mask & ((state.close_px < state.low_px) | (state.close_px > state.high_px))
    if np.any(close_bad):
        bad = np.argwhere(close_bad)[0]
        raise RuntimeError(
            f"Active-domain OHLC violation close outside [low,high] at t={int(bad[0])}, a={int(bad[1])}"
        )

    vol_bad = mask & (state.volume < 0.0)
    if np.any(vol_bad):
        bad = np.argwhere(vol_bad)[0]
        raise RuntimeError(f"Active-domain volume violation (negative) at t={int(bad[0])}, a={int(bad[1])}")


def _validate_loaded_market_slice_active_domain(state: TensorState, active_t: np.ndarray) -> None:
    active_idx = np.flatnonzero(np.asarray(active_t, dtype=bool)).astype(np.int64)
    seg_starts, seg_ends = _contiguous_segments(active_idx)
    for s0, s1 in zip(seg_starts.tolist(), seg_ends.tolist()):
        validate_loaded_market_slice(state, int(s0), int(s1))


def _apply_enabled_assets(state: TensorState, m3: Module3Output, enabled_mask: np.ndarray) -> None:
    A = state.cfg.A
    mask = np.asarray(enabled_mask, dtype=bool)
    if mask.shape != (A,):
        raise RuntimeError(f"enabled_assets_mask shape mismatch: got {mask.shape}, expected {(A,)}")

    off = ~mask
    if not np.any(off):
        return

    state.open_px[:, off] = np.nan
    state.high_px[:, off] = np.nan
    state.low_px[:, off] = np.nan
    state.close_px[:, off] = np.nan
    state.volume[:, off] = np.nan
    state.rvol[:, off] = np.nan
    state.atr_floor[:, off] = np.nan
    state.bar_valid[:, off] = False

    m3.block_features_tak[:, off, :] = np.nan
    m3.block_valid_ta[:, off] = False
    m3.context_tac[:, off, :] = np.nan
    m3.context_valid_ta[:, off] = False
    m3.context_source_t_index_ta[:, off] = -1
    if m3.ib_defined_ta is not None:
        m3.ib_defined_ta[:, off] = False


def _candidate_daily_returns_close_to_close(
    state: TensorState,
    split: SplitSpec,
    initial_cash: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts, ends, sessions = _session_bounds(state.session_id)
    test_sessions = np.unique(state.session_id[split.test_idx].astype(np.int64))

    if test_sessions.size == 0:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
        )

    # Map session -> close row index.
    close_idx = ends - 1
    sess_close = sessions

    keep = np.isin(sess_close, test_sessions)
    sess_ids = sess_close[keep].astype(np.int64)
    idx = close_idx[keep].astype(np.int64)

    if idx.size == 0:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
        )

    eq_close = state.equity[idx].astype(np.float64)
    ret = np.empty(idx.size, dtype=np.float64)
    ret[0] = eq_close[0] / float(initial_cash) - 1.0
    if idx.size > 1:
        ret[1:] = eq_close[1:] / np.maximum(eq_close[:-1], 1e-12) - 1.0

    return sess_ids, idx, ret


def _asset_pnl_by_symbol_from_state(
    state: TensorState,
    split: SplitSpec,
) -> dict[str, float]:
    """
    Deterministic close-to-close mark-to-market PnL proxy by asset over test bars.
    """
    idx = np.unique(np.asarray(split.test_idx, dtype=np.int64))
    if idx.size == 0:
        return {}

    A = int(state.cfg.A)
    contrib = np.zeros(A, dtype=np.float64)

    close = np.asarray(state.close_px, dtype=np.float64)
    pos = np.asarray(state.position_qty, dtype=np.float64)
    valid = np.asarray(state.bar_valid, dtype=bool)

    for t in idx.tolist():
        t_i = int(t)
        if t_i <= 0:
            continue
        p_i = t_i - 1
        mask = (
            valid[t_i]
            & valid[p_i]
            & np.isfinite(pos[p_i])
            & np.isfinite(close[t_i])
            & np.isfinite(close[p_i])
        )
        if not np.any(mask):
            continue
        contrib[mask] += pos[p_i, mask] * (close[t_i, mask] - close[p_i, mask])

    out: dict[str, float] = {}
    for a in range(A):
        v = float(contrib[a])
        if np.isfinite(v) and abs(v) > 1e-18:
            out[str(state.symbols[a])] = v
    return out


def _benchmark_daily_returns(
    state: TensorState,
    benchmark_symbol: str,
) -> tuple[np.ndarray, np.ndarray]:
    starts, ends, sessions = _session_bounds(state.session_id)

    A = state.cfg.A
    sym_to_idx = {s: i for i, s in enumerate(state.symbols)}

    if benchmark_symbol in sym_to_idx:
        a = int(sym_to_idx[benchmark_symbol])
        sess_out: list[int] = []
        ret_out: list[float] = []
        prev_close: float | None = None

        for s0, s1, sid in zip(starts.tolist(), ends.tolist(), sessions.tolist()):
            v = state.bar_valid[s0:s1, a]
            if not np.any(v):
                continue
            local = np.flatnonzero(v)
            i_open = int(s0 + local[0])
            i_close = int(s0 + local[-1])
            p_open = float(state.open_px[i_open, a])
            p_close = float(state.close_px[i_close, a])
            if not np.isfinite(p_open) or not np.isfinite(p_close) or p_open <= 0.0 or p_close <= 0.0:
                continue
            if prev_close is None:
                r = p_close / p_open - 1.0
            else:
                r = p_close / max(prev_close, 1e-12) - 1.0
            prev_close = p_close
            sess_out.append(int(sid))
            ret_out.append(float(r))

        return np.asarray(sess_out, dtype=np.int64), np.asarray(ret_out, dtype=np.float64)

    # Fallback benchmark: equal-weight passive basket close-to-close.
    sess_out = []
    ret_out = []
    prev_close_basket: float | None = None

    for s0, s1, sid in zip(starts.tolist(), ends.tolist(), sessions.tolist()):
        close_seg = state.close_px[s0:s1]
        valid_seg = state.bar_valid[s0:s1]

        basket_close = np.nanmean(np.where(valid_seg, close_seg, np.nan), axis=1)
        finite_idx = np.flatnonzero(np.isfinite(basket_close))
        if finite_idx.size == 0:
            continue
        i_open = int(finite_idx[0])
        i_close = int(finite_idx[-1])
        p_open = float(basket_close[i_open])
        p_close = float(basket_close[i_close])
        if p_open <= 0.0 or p_close <= 0.0:
            continue

        if prev_close_basket is None:
            r = p_close / p_open - 1.0
        else:
            r = p_close / max(prev_close_basket, 1e-12) - 1.0
        prev_close_basket = p_close

        sess_out.append(int(sid))
        ret_out.append(float(r))

    return np.asarray(sess_out, dtype=np.int64), np.asarray(ret_out, dtype=np.float64)


def _build_candidate_specs_default(
    A: int,
    m2_configs: list[Module2Config],
    m3_configs: list[Module3Config],
    m4_configs: list[Module4Config],
) -> list[CandidateSpec]:
    all_on = np.ones(A, dtype=bool)
    out: list[CandidateSpec] = []
    cid = 0
    for i2 in range(len(m2_configs)):
        for i3 in range(len(m3_configs)):
            for i4 in range(len(m4_configs)):
                out.append(
                    CandidateSpec(
                        candidate_id=f"cand_{cid:04d}_m2{i2}_m3{i3}_m4{i4}",
                        m2_idx=i2,
                        m3_idx=i3,
                        m4_idx=i4,
                        enabled_assets_mask=all_on.copy(),
                        tags=(),
                    )
                )
                cid += 1
    return out


def _normalize_candidate_specs(
    specs: list[CandidateSpec],
    keep_idx: np.ndarray,
    A_filtered: int,
    A_input: int,
) -> list[CandidateSpec]:
    out: list[CandidateSpec] = []
    for spec in specs:
        m = np.asarray(spec.enabled_assets_mask, dtype=bool)
        if m.shape == (A_input,):
            m2 = m[keep_idx]
        elif m.shape == (A_filtered,):
            m2 = m.copy()
        else:
            raise RuntimeError(
                f"Candidate {spec.candidate_id} enabled_assets_mask has invalid shape {m.shape}; "
                f"expected {(A_input,)} or {(A_filtered,)}"
            )
        out.append(
            CandidateSpec(
                candidate_id=spec.candidate_id,
                m2_idx=int(spec.m2_idx),
                m3_idx=int(spec.m3_idx),
                m4_idx=int(spec.m4_idx),
                enabled_assets_mask=m2.astype(bool, copy=True),
                tags=tuple(spec.tags),
            )
        )
    return out


def _build_group_tasks(
    candidates: list[CandidateSpec],
    splits: list[SplitSpec],
    scenarios: list[StressScenario],
) -> list[_GroupTask]:
    groups: dict[tuple[int, int, int, int], list[int]] = {}
    for ci, c in enumerate(candidates):
        for si, _sp in enumerate(splits):
            for zi, sc in enumerate(scenarios):
                if not sc.enabled:
                    continue
                key = (si, zi, int(c.m2_idx), int(c.m3_idx))
                groups.setdefault(key, []).append(ci)

    out: list[_GroupTask] = []
    for key in sorted(groups.keys()):
        si, zi, m2i, m3i = key
        cand_idx = tuple(sorted(groups[key]))
        gid = f"g_s{si:03d}_z{zi:02d}_m2{m2i}_m3{m3i}"
        out.append(
            _GroupTask(
                group_id=gid,
                split_idx=si,
                scenario_idx=zi,
                m2_idx=m2i,
                m3_idx=m3i,
                candidate_indices=cand_idx,
            )
        )
    return out


def _split_group_tasks_by_candidate(
    group_tasks: list[_GroupTask],
    chunk_size: int = 1,
) -> list[_GroupTask]:
    """
    Deterministic process-pool chunking:
    split large grouped candidate batches into fixed-size chunks so
    progress/checkpoints advance earlier under heavy compute.
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


def _equity_curve_payload(
    state: TensorState,
    candidate_id: str,
    split_id: str,
    scenario_id: str,
) -> dict[str, np.ndarray]:
    eq = state.equity.astype(np.float64)
    peak = np.maximum.accumulate(eq)
    dd = np.where(peak > 0.0, eq / peak - 1.0, 0.0)
    T = state.cfg.T

    return {
        "ts_ns": state.ts_ns.copy(),
        "session_id": state.session_id.copy(),
        "candidate_id": np.full(T, candidate_id, dtype=object),
        "split_id": np.full(T, split_id, dtype=object),
        "scenario_id": np.full(T, scenario_id, dtype=object),
        "equity": eq.copy(),
        "drawdown": dd.astype(np.float64),
        "margin_used": state.margin_used.copy(),
        "buying_power": state.buying_power.copy(),
        "daily_loss": state.daily_loss.copy(),
    }


def _trade_log_payload(
    state: TensorState,
    m4_out: Module4Output,
    candidate_id: str,
    split_id: str,
    scenario_id: str,
    eps: float = 1e-12,
) -> dict[str, np.ndarray]:
    mask = np.isfinite(m4_out.exec_price_ta) & (np.abs(m4_out.filled_qty_ta) > float(eps))
    loc = np.argwhere(mask)
    if loc.size == 0:
        return {
            "ts_ns": np.zeros(0, dtype=np.int64),
            "candidate_id": np.zeros(0, dtype=object),
            "split_id": np.zeros(0, dtype=object),
            "scenario_id": np.zeros(0, dtype=object),
            "symbol": np.zeros(0, dtype=object),
            "filled_qty": np.zeros(0, dtype=np.float64),
            "exec_price": np.zeros(0, dtype=np.float64),
            "trade_cost": np.zeros(0, dtype=np.float64),
            "order_side": np.zeros(0, dtype=np.int8),
            "order_flags": np.zeros(0, dtype=np.uint16),
        }

    t_idx = loc[:, 0]
    a_idx = loc[:, 1]

    return {
        "ts_ns": state.ts_ns[t_idx].astype(np.int64),
        "candidate_id": np.full(t_idx.shape[0], candidate_id, dtype=object),
        "split_id": np.full(t_idx.shape[0], split_id, dtype=object),
        "scenario_id": np.full(t_idx.shape[0], scenario_id, dtype=object),
        "symbol": np.asarray([state.symbols[int(a)] for a in a_idx.tolist()], dtype=object),
        "filled_qty": m4_out.filled_qty_ta[t_idx, a_idx].astype(np.float64),
        "exec_price": m4_out.exec_price_ta[t_idx, a_idx].astype(np.float64),
        "trade_cost": m4_out.trade_cost_ta[t_idx, a_idx].astype(np.float64),
        "order_side": state.order_side[t_idx, a_idx].astype(np.int8),
        "order_flags": state.order_flags[t_idx, a_idx].astype(np.uint16),
    }


def _event_window_mask(T: int, event_idx: np.ndarray, pre: int, post: int) -> np.ndarray:
    mask = np.zeros(T, dtype=bool)
    if event_idx.size == 0:
        return mask
    lo_off = int(max(0, pre))
    hi_off = int(max(0, post))
    for i in event_idx.tolist():
        lo = max(0, int(i) - lo_off)
        hi = min(T, int(i) + hi_off + 1)
        mask[lo:hi] = True
    return mask


def _structural_weight_from_regime(regime_i8: np.ndarray) -> np.ndarray:
    r = np.asarray(regime_i8, dtype=np.int8)
    w = np.zeros(r.shape, dtype=np.float64)
    w[(r == np.int8(RegimeIdx.P_SHAPE)) | (r == np.int8(RegimeIdx.B_SHAPE))] = 1.5
    w[r == np.int8(RegimeIdx.TREND)] = 1.2
    return w


def _select_micro_rows(
    state: TensorState,
    split: SplitSpec,
    cfg: Module5HarnessConfig,
    m4_out: Module4Output,
    enabled_assets_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    T = state.cfg.T
    A = state.cfg.A
    mode = str(cfg.micro_diag_mode).strip().lower()

    a_mask = np.asarray(enabled_assets_mask, dtype=bool).copy()
    if a_mask.shape != (A,):
        raise RuntimeError(
            f"enabled_assets_mask shape mismatch for micro diagnostics: got {a_mask.shape}, expected {(A,)}"
        )

    if cfg.micro_diag_symbols:
        symbol_set = set(str(s) for s in cfg.micro_diag_symbols)
        a_mask &= np.asarray([s in symbol_set for s in state.symbols], dtype=bool)

    if mode == "off":
        return np.zeros(T, dtype=bool), a_mask

    if mode == "full_test":
        t_mask = np.zeros(T, dtype=bool)
        t_mask[split.test_idx] = True
    elif mode == "symbol_day":
        t_mask = np.ones(T, dtype=bool)
        if cfg.micro_diag_session_ids:
            sset = set(int(s) for s in cfg.micro_diag_session_ids)
            t_mask &= np.isin(state.session_id.astype(np.int64), np.asarray(sorted(sset), dtype=np.int64))
    elif mode == "events_only":
        fills_t = np.flatnonzero(np.any(np.abs(m4_out.filled_qty_ta) > 1e-12, axis=1)).astype(np.int64)
        select_t = np.flatnonzero(state.phase == np.int8(Phase.OVERNIGHT_SELECT)).astype(np.int64)
        event_idx = np.unique(np.r_[fills_t, select_t]).astype(np.int64)
        t_mask = _event_window_mask(
            T=T,
            event_idx=event_idx,
            pre=int(cfg.micro_diag_trade_window_pre),
            post=int(cfg.micro_diag_trade_window_post),
        )
    else:
        raise RuntimeError(f"Unsupported micro_diag_mode: {cfg.micro_diag_mode}")

    # Keep only rows with at least one valid enabled asset observation.
    valid_any = np.any(state.bar_valid[:, a_mask], axis=1) if np.any(a_mask) else np.zeros(T, dtype=bool)
    t_mask &= valid_any
    return t_mask, a_mask


def _collect_micro_diagnostics_payload(
    state: TensorState,
    m3: Module3Output,
    m4_out: Module4Output,
    candidate_id: str,
    split_id: str,
    scenario_id: str,
    split: SplitSpec,
    enabled_assets_mask: np.ndarray,
    cfg: Module5HarnessConfig,
) -> dict[str, np.ndarray] | None:
    if not bool(cfg.export_micro_diagnostics):
        return None

    t_mask, a_mask = _select_micro_rows(state, split, cfg, m4_out, enabled_assets_mask)
    if not np.any(t_mask) or not np.any(a_mask):
        return None

    loc = np.argwhere(t_mask[:, None] & a_mask[None, :])
    if loc.size == 0:
        return None

    if loc.shape[0] > int(cfg.micro_diag_max_rows):
        raise RuntimeError(
            f"micro_diagnostics row cap exceeded: rows={int(loc.shape[0])}, cap={int(cfg.micro_diag_max_rows)}"
        )

    t_idx = loc[:, 0].astype(np.int64)
    a_idx = loc[:, 1].astype(np.int64)

    winner_flag = (
        m4_out.overnight_winner_t[t_idx].astype(np.int64) == a_idx.astype(np.int64)
    ).astype(np.int8)

    return {
        "ts_ns": state.ts_ns[t_idx].astype(np.int64),
        "session_id": state.session_id[t_idx].astype(np.int64),
        "candidate_id": np.full(t_idx.shape[0], candidate_id, dtype=object),
        "split_id": np.full(t_idx.shape[0], split_id, dtype=object),
        "scenario_id": np.full(t_idx.shape[0], scenario_id, dtype=object),
        "symbol": np.asarray([state.symbols[int(a)] for a in a_idx.tolist()], dtype=object),
        "open": state.open_px[t_idx, a_idx].astype(np.float64),
        "high": state.high_px[t_idx, a_idx].astype(np.float64),
        "low": state.low_px[t_idx, a_idx].astype(np.float64),
        "close": state.close_px[t_idx, a_idx].astype(np.float64),
        "volume": state.volume[t_idx, a_idx].astype(np.float64),
        "bar_valid": state.bar_valid[t_idx, a_idx].astype(np.int8),
        "dclip": state.profile_stats[t_idx, a_idx, int(ProfileStatIdx.DCLIP)].astype(np.float64),
        "z_delta": state.profile_stats[t_idx, a_idx, int(ProfileStatIdx.Z_DELTA)].astype(np.float64),
        "gbreak": state.profile_stats[t_idx, a_idx, int(ProfileStatIdx.GBREAK)].astype(np.float64),
        "greject": state.profile_stats[t_idx, a_idx, int(ProfileStatIdx.GREJECT)].astype(np.float64),
        "score_bo_long": state.scores[t_idx, a_idx, int(ScoreIdx.SCORE_BO_LONG)].astype(np.float64),
        "score_bo_short": state.scores[t_idx, a_idx, int(ScoreIdx.SCORE_BO_SHORT)].astype(np.float64),
        "score_rej_long": state.scores[t_idx, a_idx, int(ScoreIdx.SCORE_REJ_LONG)].astype(np.float64),
        "score_rej_short": state.scores[t_idx, a_idx, int(ScoreIdx.SCORE_REJ_SHORT)].astype(np.float64),
        "ctx_x_poc": m3.context_tac[t_idx, a_idx, int(ContextIdx.CTX_X_POC)].astype(np.float64),
        "ctx_x_vah": m3.context_tac[t_idx, a_idx, int(ContextIdx.CTX_X_VAH)].astype(np.float64),
        "ctx_x_val": m3.context_tac[t_idx, a_idx, int(ContextIdx.CTX_X_VAL)].astype(np.float64),
        "ctx_trend_gate_spread_mean": m3.context_tac[t_idx, a_idx, int(ContextIdx.CTX_TREND_GATE_SPREAD_MEAN)].astype(np.float64),
        "ctx_poc_drift_x": m3.context_tac[t_idx, a_idx, int(ContextIdx.CTX_POC_DRIFT_X)].astype(np.float64),
        "ctx_poc_vs_prev_va": m3.context_tac[t_idx, a_idx, int(ContextIdx.CTX_POC_VS_PREV_VA)].astype(np.float64),
        "ctx_ib_high_x": m3.context_tac[t_idx, a_idx, int(ContextIdx.CTX_IB_HIGH_X)].astype(np.float64),
        "ctx_ib_low_x": m3.context_tac[t_idx, a_idx, int(ContextIdx.CTX_IB_LOW_X)].astype(np.float64),
        "regime_primary": m4_out.regime_primary_ta[t_idx, a_idx].astype(np.int8),
        "regime_confidence": m4_out.regime_confidence_ta[t_idx, a_idx].astype(np.float64),
        "intent_long": m4_out.intent_long_ta[t_idx, a_idx].astype(np.int8),
        "intent_short": m4_out.intent_short_ta[t_idx, a_idx].astype(np.int8),
        "target_qty": m4_out.target_qty_ta[t_idx, a_idx].astype(np.float64),
        "filled_qty": m4_out.filled_qty_ta[t_idx, a_idx].astype(np.float64),
        "exec_price": m4_out.exec_price_ta[t_idx, a_idx].astype(np.float64),
        "trade_cost": m4_out.trade_cost_ta[t_idx, a_idx].astype(np.float64),
        "position_qty": state.position_qty[t_idx, a_idx].astype(np.float64),
        "overnight_score": m4_out.overnight_score_ta[t_idx, a_idx].astype(np.float64),
        "overnight_winner_flag": winner_flag,
        "atr_eff": state.atr_floor[t_idx, a_idx].astype(np.float64),
        "rvol": state.rvol[t_idx, a_idx].astype(np.float64),
    }


def _collect_micro_profile_blocks_payload(
    state: TensorState,
    m3: Module3Output,
    candidate_id: str,
    split_id: str,
    scenario_id: str,
    enabled_assets_mask: np.ndarray,
    cfg: Module5HarnessConfig,
) -> dict[str, np.ndarray] | None:
    if not (bool(cfg.export_micro_diagnostics) and bool(cfg.micro_diag_export_block_profiles)):
        return None

    A = state.cfg.A
    a_mask = np.asarray(enabled_assets_mask, dtype=bool)
    if a_mask.shape != (A,):
        raise RuntimeError(f"enabled_assets_mask shape mismatch in profile blocks: {a_mask.shape}")
    if cfg.micro_diag_symbols:
        symbol_set = set(str(s) for s in cfg.micro_diag_symbols)
        a_mask &= np.asarray([s in symbol_set for s in state.symbols], dtype=bool)

    if not np.any(a_mask):
        return None

    block_rows = np.flatnonzero(m3.block_end_flag_t).astype(np.int64)
    if block_rows.size == 0:
        return None

    mask = m3.block_valid_ta[block_rows][:, a_mask]
    loc = np.argwhere(mask)
    if loc.size == 0:
        return None

    rr = block_rows[loc[:, 0].astype(np.int64)]
    aa_local = np.where(a_mask)[0].astype(np.int64)
    aa = aa_local[loc[:, 1].astype(np.int64)]

    if rr.shape[0] > int(cfg.micro_diag_max_rows):
        raise RuntimeError(
            f"micro_profile_blocks row cap exceeded: rows={int(rr.shape[0])}, cap={int(cfg.micro_diag_max_rows)}"
        )

    x_blob = state.x_grid.astype(np.float32).tobytes()
    return {
        "ts_ns": state.ts_ns[rr].astype(np.int64),
        "session_id": state.session_id[rr].astype(np.int64),
        "candidate_id": np.full(rr.shape[0], candidate_id, dtype=object),
        "split_id": np.full(rr.shape[0], split_id, dtype=object),
        "scenario_id": np.full(rr.shape[0], scenario_id, dtype=object),
        "symbol": np.asarray([state.symbols[int(a)] for a in aa.tolist()], dtype=object),
        "block_seq": m3.block_seq_t[rr].astype(np.int16),
        "n_bins": np.full(rr.shape[0], int(state.cfg.B), dtype=np.int32),
        "x_grid_blob": np.full(rr.shape[0], x_blob, dtype=object),
        "vp_block_blob": np.asarray([state.vp[int(t), int(a)].astype(np.float32).tobytes() for t, a in zip(rr.tolist(), aa.tolist())], dtype=object),
        "vp_delta_block_blob": np.asarray([state.vp_delta[int(t), int(a)].astype(np.float32).tobytes() for t, a in zip(rr.tolist(), aa.tolist())], dtype=object),
        "close_te": state.close_px[rr, aa].astype(np.float64),
        "atr_eff_te": state.atr_floor[rr, aa].astype(np.float64),
    }


def _collect_funnel_payload(
    state: TensorState,
    m4_out: Module4Output,
    candidate_id: str,
    split_id: str,
    scenario_id: str,
    enabled_assets_mask: np.ndarray,
    cfg: Module5HarnessConfig,
) -> dict[str, np.ndarray] | None:
    if not (bool(cfg.export_micro_diagnostics) and bool(cfg.micro_diag_export_funnel)):
        return None

    A = state.cfg.A
    a_mask = np.asarray(enabled_assets_mask, dtype=bool)
    if a_mask.shape != (A,):
        raise RuntimeError(f"enabled_assets_mask shape mismatch in funnel payload: {a_mask.shape}")
    if cfg.micro_diag_symbols:
        symbol_set = set(str(s) for s in cfg.micro_diag_symbols)
        a_mask &= np.asarray([s in symbol_set for s in state.symbols], dtype=bool)
    if not np.any(a_mask):
        return None

    t_sel = np.flatnonzero(state.phase == np.int8(Phase.OVERNIGHT_SELECT)).astype(np.int64)
    if t_sel.size == 0:
        return None

    out_rows: list[dict[str, Any]] = []
    for t in t_sel.tolist():
        winner = int(m4_out.overnight_winner_t[t])
        valid_assets = np.where(a_mask)[0].astype(np.int64)
        if valid_assets.size == 0:
            continue
        dclip = state.profile_stats[t, valid_assets, int(ProfileStatIdx.DCLIP)]
        zdel = state.profile_stats[t, valid_assets, int(ProfileStatIdx.Z_DELTA)]
        rvol = state.rvol[t, valid_assets]
        regime = m4_out.regime_primary_ta[t, valid_assets]
        sw = _structural_weight_from_regime(regime)
        ocs = sw * np.abs(dclip) * np.abs(zdel) * np.maximum(rvol, 0.0)
        cash_fallback = winner < 0
        for j, a in enumerate(valid_assets.tolist()):
            out_rows.append(
                {
                    "ts_ns": int(state.ts_ns[t]),
                    "session_id": int(state.session_id[t]),
                    "candidate_id": candidate_id,
                    "split_id": split_id,
                    "scenario_id": scenario_id,
                    "symbol": state.symbols[int(a)],
                    "dclip": float(dclip[j]),
                    "z_delta": float(zdel[j]),
                    "regime_primary": int(regime[j]),
                    "structural_weight": float(sw[j]),
                    "ocs": float(ocs[j]),
                    "is_winner": int(1 if int(a) == winner else 0),
                    "cash_fallback": int(1 if cash_fallback else 0),
                    "rvol": float(rvol[j]),
                }
            )

    if not out_rows:
        return None

    pdx = _require_pandas()
    df = pdx.DataFrame(out_rows)
    return {k: df[k].to_numpy() for k in df.columns.tolist()}


def _run_group_task(
    group: _GroupTask,
    base_state: TensorState,
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

    group_seed = _seed_for_task(
        harness_cfg.seed,
        group.group_id,
        split.split_id,
        scenario.scenario_id,
    )
    rng = np.random.default_rng(group_seed)

    # Build post-M3 cache for this (split, scenario, m2, m3) key.
    cached_state = _clone_state(base_state)
    active_t = _apply_split_domain_mask(cached_state, split)

    _apply_missing_bursts(cached_state, active_t, scenario, rng)
    _apply_jitter(cached_state, active_t, scenario, rng)
    _recompute_bar_valid_inplace(cached_state)
    _set_placeholders_from_bar_valid(cached_state)
    _assert_placeholder_consistency(cached_state)

    if harness_cfg.fail_on_non_finite:
        _assert_active_domain_ohlc(cached_state, active_t)
        _validate_loaded_market_slice_active_domain(cached_state, active_t)

    run_weightiz_profile_engine(cached_state, m2_configs[group.m2_idx])
    post_m2_reasons = _apply_post_m2_invariants(cached_state, active_t)
    m3_out_cached = run_module3_structural_aggregation(cached_state, m3_configs[group.m3_idx])
    post_m3_reasons = _apply_post_m3_invariants(m3_out_cached)

    outputs: list[dict[str, Any]] = []

    for ci in group.candidate_indices:
        c = candidates[ci]
        task_id = f"{c.candidate_id}|{split.split_id}|{scenario.scenario_id}"
        task_seed = _seed_for_task(group_seed, task_id)
        try:
            if task_id in set(harness_cfg.test_fail_task_ids):
                raise RuntimeError("InjectedTaskFailure: task_id match")
            if float(harness_cfg.test_fail_ratio) > 0.0:
                h = hashlib.sha256(f"{task_seed}|{task_id}".encode("utf-8")).digest()
                u = int.from_bytes(h[:8], "little", signed=False) / float(2**64 - 1)
                if u < float(harness_cfg.test_fail_ratio):
                    raise RuntimeError("InjectedTaskFailure: ratio")

            st = _clone_state(cached_state)
            m3c = _clone_m3(m3_out_cached)
            _apply_enabled_assets(st, m3c, c.enabled_assets_mask)
            pre_m4_reasons = _apply_pre_m4_invariants(st, m3c)
            quality_reason_codes = sorted(set(post_m2_reasons + post_m3_reasons + pre_m4_reasons))
            dqs_mat = np.asarray(getattr(st, "dqs_day_ta", np.ones((st.cfg.T, st.cfg.A), dtype=np.float64)), dtype=np.float64)
            if dqs_mat.shape != (st.cfg.T, st.cfg.A):
                raise RuntimeError(f"dqs_day_ta shape mismatch: got {dqs_mat.shape}, expected {(st.cfg.T, st.cfg.A)}")
            dqs_scope = dqs_mat[np.asarray(st.bar_valid, dtype=bool)]
            if dqs_scope.size <= 0:
                dqs_scope = np.asarray([1.0], dtype=np.float64)
            dqs_min = float(np.min(dqs_scope))
            dqs_median = float(np.median(dqs_scope))
            if dqs_min < 1.0:
                quality_reason_codes = sorted(set(quality_reason_codes + ["DQ_DEGRADED_INPUT"]))
            if dqs_min <= 0.0:
                quality_reason_codes = sorted(set(quality_reason_codes + ["DQ_REJECTED_INPUT"]))

            m4_cfg = replace(
                m4_configs[c.m4_idx],
                stress_slippage_mult=float(m4_configs[c.m4_idx].stress_slippage_mult)
                * float(scenario.slippage_mult),
            )

            m4_out = run_module4_strategy_funnel(st, m3c, m4_cfg)
            validate_state_hard(st)

            sess_ids, close_idx, daily_ret = _candidate_daily_returns_close_to_close(
                st,
                split,
                initial_cash=float(st.cfg.initial_cash),
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

            outputs.append(
                {
                    "task_id": task_id,
                    "candidate_id": c.candidate_id,
                    "split_id": split.split_id,
                    "scenario_id": scenario.scenario_id,
                    "status": "ok",
                    "error": "",
                    "session_ids": sess_ids,
                    "daily_returns": daily_ret,
                    "equity_payload": _equity_curve_payload(st, c.candidate_id, split.split_id, scenario.scenario_id),
                    "trade_payload": _trade_log_payload(st, m4_out, c.candidate_id, split.split_id, scenario.scenario_id),
                    "asset_pnl_by_symbol": _asset_pnl_by_symbol_from_state(st, split),
                    "micro_payload": micro_payload,
                    "profile_payload": profile_payload,
                    "funnel_payload": funnel_payload,
                    "m2_idx": int(c.m2_idx),
                    "m3_idx": int(c.m3_idx),
                    "m4_idx": int(c.m4_idx),
                    "tags": list(c.tags),
                    "test_days": int(daily_ret.shape[0]),
                    "task_seed": int(task_seed),
                    "quality_reason_codes": quality_reason_codes,
                    "asset_keys": [st.symbols[i] for i in np.flatnonzero(c.enabled_assets_mask).tolist()],
                    "exception_signature": "",
                    "dqs_min": dqs_min,
                    "dqs_median": dqs_median,
                }
            )
        except Exception as exc:
            err_type = type(exc).__name__
            err_msg = str(exc)
            tb = traceback.format_exc()
            top_frame = _normalized_top_frame(tb)
            sig = f"{err_type}|{top_frame}"
            asset_keys = [base_state.symbols[i] for i in np.flatnonzero(c.enabled_assets_mask).tolist()]
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
                    "daily_returns": np.zeros(0, dtype=np.float64),
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
                    "quality_reason_codes": sorted(set(post_m2_reasons + post_m3_reasons)),
                    "asset_keys": asset_keys,
                    "dqs_min": 0.0,
                    "dqs_median": 0.0,
                }
            )

    return outputs


def _stack_payload_frames(payloads: list[dict[str, np.ndarray]]) -> Any:
    pdx = _require_pandas()
    if not payloads:
        return pdx.DataFrame()
    frames = [pdx.DataFrame(p) for p in payloads]
    if not frames:
        return pdx.DataFrame()
    return pdx.concat(frames, axis=0, ignore_index=True)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(obj), f, ensure_ascii=False, indent=2)


def _clip01(x: float) -> float:
    if not np.isfinite(x):
        return 1.0
    return float(min(max(float(x), 0.0), 1.0))


def _cum_return(ret_1d: np.ndarray) -> float:
    r = np.asarray(ret_1d, dtype=np.float64)
    if r.size == 0:
        return 0.0
    return float(np.prod(1.0 + r) - 1.0)


def _max_drawdown_from_returns(ret_1d: np.ndarray) -> float:
    r = np.asarray(ret_1d, dtype=np.float64)
    if r.size == 0:
        return 0.0
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    dd = np.where(peak > 0.0, eq / peak - 1.0, 0.0)
    return float(abs(np.min(dd)))


def _sharpe_daily(ret_1d: np.ndarray, eps: float = 1e-12) -> float:
    r = np.asarray(ret_1d, dtype=np.float64)
    if r.size < 2:
        return 0.0
    mu = float(np.mean(r))
    sd = float(np.std(r, ddof=1))
    return float(mu / (sd + float(eps)))


def _turnover_from_trade_payload(trade_payload: dict[str, np.ndarray] | None, initial_cash: float) -> float:
    if not trade_payload:
        return 0.0
    qty = np.asarray(trade_payload.get("filled_qty", np.zeros(0, dtype=np.float64)), dtype=np.float64)
    px = np.asarray(trade_payload.get("exec_price", np.zeros(0, dtype=np.float64)), dtype=np.float64)
    if qty.size == 0 or px.size == 0 or qty.size != px.size:
        return 0.0
    notional = float(np.sum(np.abs(qty * px)))
    return float(notional / max(float(initial_cash), 1e-12))


def _trade_count_from_payload(trade_payload: dict[str, np.ndarray] | None) -> int:
    if not trade_payload:
        return 0
    qty = np.asarray(trade_payload.get("filled_qty", np.zeros(0, dtype=np.float64)), dtype=np.float64)
    if qty.size == 0:
        return 0
    return int(np.sum(np.abs(qty) > 1e-12))


def _margin_exposure_stats_from_equity_payloads(payloads: list[dict[str, np.ndarray]]) -> dict[str, float]:
    if not payloads:
        return {"avg_margin_used_frac": 0.0, "peak_margin_used_frac": 0.0}
    vals: list[np.ndarray] = []
    for p in payloads:
        eq = np.asarray(p.get("equity", np.zeros(0, dtype=np.float64)), dtype=np.float64)
        mg = np.asarray(p.get("margin_used", np.zeros(0, dtype=np.float64)), dtype=np.float64)
        if eq.size == 0 or mg.size == 0 or eq.size != mg.size:
            continue
        frac = np.abs(mg) / np.maximum(np.abs(eq), 1e-12)
        vals.append(frac.astype(np.float64))
    if not vals:
        return {"avg_margin_used_frac": 0.0, "peak_margin_used_frac": 0.0}
    allf = np.concatenate(vals, axis=0)
    return {
        "avg_margin_used_frac": float(np.mean(allf)),
        "peak_margin_used_frac": float(np.max(allf)),
    }


def _asset_notional_concentration_from_trade_payloads(payloads: list[dict[str, np.ndarray]]) -> float:
    if not payloads:
        return 0.0
    acc: dict[str, float] = {}
    for p in payloads:
        sym = np.asarray(p.get("symbol", np.zeros(0, dtype=object)), dtype=object)
        qty = np.asarray(p.get("filled_qty", np.zeros(0, dtype=np.float64)), dtype=np.float64)
        px = np.asarray(p.get("exec_price", np.zeros(0, dtype=np.float64)), dtype=np.float64)
        if sym.size == 0 or qty.size == 0 or px.size == 0:
            continue
        n = min(sym.size, qty.size, px.size)
        for i in range(n):
            s = str(sym[i])
            v = float(abs(float(qty[i]) * float(px[i])))
            acc[s] = acc.get(s, 0.0) + v
    if not acc:
        return 0.0
    total = float(sum(acc.values()))
    if total <= 0.0:
        return 0.0
    return float(max(acc.values()) / total)


def _asset_pnl_concentration_from_result_rows(rows: list[dict[str, Any]]) -> float:
    acc: dict[str, float] = {}
    for r in rows:
        payload = r.get("asset_pnl_by_symbol", {})
        if not isinstance(payload, dict):
            continue
        for k, v in payload.items():
            sym = str(k)
            vv = float(v)
            if not np.isfinite(vv):
                continue
            acc[sym] = acc.get(sym, 0.0) + vv
    if not acc:
        return 0.0
    abs_vals = np.asarray([abs(float(v)) for v in acc.values()], dtype=np.float64)
    total_abs = float(np.sum(abs_vals))
    if total_abs <= 1e-12:
        return 0.0
    return float(np.max(abs_vals) / total_abs)


def _split_mode(split_id: str) -> str:
    s = str(split_id)
    if s.startswith("wf_"):
        return "wf"
    if s.startswith("cpcv_"):
        return "cpcv"
    return "other"


def _summarize_fold_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "sharpe_daily_mean": 0.0,
            "sharpe_daily_median": 0.0,
            "sharpe_daily_worst": 0.0,
            "cum_return_mean": 0.0,
            "cum_return_median": 0.0,
            "max_drawdown_median": 0.0,
            "turnover_median": 0.0,
        }
    sh = np.asarray([float(r["sharpe_daily"]) for r in rows], dtype=np.float64)
    cr = np.asarray([float(r["cum_return"]) for r in rows], dtype=np.float64)
    dd = np.asarray([float(r["max_drawdown"]) for r in rows], dtype=np.float64)
    to = np.asarray([float(r["turnover"]) for r in rows], dtype=np.float64)
    return {
        "count": int(len(rows)),
        "sharpe_daily_mean": float(np.mean(sh)),
        "sharpe_daily_median": float(np.median(sh)),
        "sharpe_daily_worst": float(np.min(sh)),
        "cum_return_mean": float(np.mean(cr)),
        "cum_return_median": float(np.median(cr)),
        "max_drawdown_median": float(np.median(dd)),
        "turnover_median": float(np.median(to)),
    }


def _plateau_key(feature: dict[str, float]) -> tuple[int, ...]:
    return (
        int(np.rint(float(feature["entry_threshold"]) / 0.02)),
        int(np.rint(float(feature["exit_threshold"]) / 0.02)),
        int(np.rint(float(feature["top_k_intraday"]) / 1.0)),
        int(np.rint(float(feature["max_asset_cap_frac"]) / 0.05)),
        int(np.rint(float(feature["max_turnover_frac_per_bar"]) / 0.05)),
        int(np.rint(float(feature["block_minutes"]) / 5.0)),
        int(np.rint(float(feature["min_block_valid_ratio"]) / 0.05)),
    )


def _aggregate_candidate_baseline_matrix(
    results_ok: list[dict[str, Any]],
    bench_sessions: np.ndarray,
    bench_ret: np.ndarray,
    candidate_ids: list[str],
    min_days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, dict[str, dict[int, float]]]]:
    if not results_ok:
        raise RuntimeError("No successful candidate task results to assemble")

    bench_map = {int(s): float(r) for s, r in zip(bench_sessions.tolist(), bench_ret.tolist())}
    if not bench_map:
        raise RuntimeError("Benchmark session series is empty")

    cand_set = set(str(c) for c in candidate_ids)
    bucket: dict[str, dict[str, dict[int, list[float]]]] = {
        str(cid): {} for cid in sorted(cand_set)
    }

    for r in results_ok:
        cid = str(r.get("candidate_id", ""))
        if cid not in bucket:
            continue
        sid = str(r.get("scenario_id", "baseline"))
        sess = np.asarray(r.get("session_ids", np.zeros(0, dtype=np.int64)), dtype=np.int64)
        ret = np.asarray(r.get("daily_returns", np.zeros(0, dtype=np.float64)), dtype=np.float64)
        if sess.size == 0 or ret.size == 0 or sess.size != ret.size:
            continue
        by_sess = bucket[cid].setdefault(sid, {})
        for s, v in zip(sess.tolist(), ret.tolist()):
            vv = float(v)
            if not np.isfinite(vv):
                continue
            by_sess.setdefault(int(s), []).append(vv)

    scenario_series: dict[str, dict[str, dict[int, float]]] = {}
    for cid in sorted(bucket.keys()):
        per_scenario: dict[str, dict[int, float]] = {}
        for sid in sorted(bucket[cid].keys()):
            sess_map = bucket[cid][sid]
            if not sess_map:
                continue
            agg: dict[int, float] = {}
            for s in sorted(sess_map.keys()):
                vals = np.asarray(sess_map[s], dtype=np.float64)
                agg[int(s)] = float(np.median(vals))
            per_scenario[sid] = agg
        scenario_series[cid] = per_scenario

    baseline_candidate_ids = [
        cid
        for cid in sorted(candidate_ids)
        if "baseline" in scenario_series.get(str(cid), {})
        and len(scenario_series[str(cid)]["baseline"]) > 0
    ]
    if not baseline_candidate_ids:
        raise RuntimeError("No candidates have baseline daily returns for candidate-level stats")

    # Institutional deterministic policy:
    # build candidate daily matrix on the full post-DQ session domain (benchmark sessions),
    # and fill missing candidate session observations with neutral 0.0 daily return.
    # This prevents dropping valid no-trade days from the alignment matrix.
    common_sorted = np.asarray(sorted(bench_map.keys()), dtype=np.int64)
    if common_sorted.size <= 0:
        raise RuntimeError("No benchmark sessions available for candidate alignment")
    D = int(common_sorted.size)
    C = int(len(baseline_candidate_ids))
    if D < int(min_days):
        raise RuntimeError(f"Insufficient daily sample after candidate alignment: D={D}, required>={int(min_days)}")

    mat = np.empty((D, C), dtype=np.float64)
    for j, cid in enumerate(baseline_candidate_ids):
        mp = scenario_series[cid]["baseline"]
        mat[:, j] = np.asarray([float(mp.get(int(s), 0.0)) for s in common_sorted.tolist()], dtype=np.float64)
    bmk = np.asarray([bench_map[int(s)] for s in common_sorted.tolist()], dtype=np.float64)

    _assert_finite("candidate_daily_returns_matrix", mat)
    _assert_finite("daily_benchmark_returns", bmk)
    return common_sorted, mat, bmk, baseline_candidate_ids, scenario_series


def _compute_stats_verdict(
    daily_returns_matrix: np.ndarray,
    daily_benchmark_returns: np.ndarray,
    candidate_ids: list[str],
    harness_cfg: Module5HarnessConfig,
) -> dict[str, Any]:
    dsr = deflated_sharpe_ratio(daily_returns_matrix)
    pbo = pbo_cscv(
        daily_returns_matrix,
        S=int(harness_cfg.cpcv_slices),
        k=int(harness_cfg.cpcv_k_test),
    )
    wrc = white_reality_check(
        daily_returns_matrix,
        daily_benchmark_returns,
        seed=int(harness_cfg.seed + 101),
    )
    spa = spa_test(
        daily_returns_matrix,
        daily_benchmark_returns,
        seed=int(harness_cfg.seed + 202),
    )
    mcs = model_confidence_set(
        -daily_returns_matrix,
        alpha=0.10,
        seed=int(harness_cfg.seed + 303),
    )

    survivors = set(int(i) for i in np.asarray(mcs.get("survivors", np.array([], dtype=np.int64))).tolist())
    dsr_arr = np.asarray(dsr["dsr"], dtype=np.float64)

    leaderboard: list[dict[str, Any]] = []
    for j, cid in enumerate(candidate_ids):
        in_mcs = j in survivors
        dsr_j = float(dsr_arr[j]) if j < dsr_arr.size else float("nan")
        pass_flag = bool(in_mcs and (dsr_j >= 0.50))
        leaderboard.append(
            {
                "candidate_id": str(cid),
                "dsr": dsr_j,
                "in_mcs": in_mcs,
                "wrc_p": float(wrc["p_value"]),
                "spa_p": float(spa["p_value"]),
                "pbo": float(pbo["pbo"]) if np.isfinite(pbo["pbo"]) else None,
                "pass": pass_flag,
            }
        )

    return {
        "dsr": dsr,
        "pbo": pbo,
        "wrc": wrc,
        "spa": spa,
        "mcs": mcs,
        "leaderboard": leaderboard,
        "gate_defaults": {
            "dsr_min": 0.50,
            "mcs_membership_required": True,
        },
    }


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
    pdx = _require_pandas()

    candidates_dir = report_root / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    baseline_col = {str(cid): j for j, cid in enumerate(baseline_candidate_ids)}
    scenario_ids = [str(s.scenario_id) for s in scenarios]
    initial_cash = float(engine_cfg.initial_cash)

    candidate_rows: list[dict[str, Any]] = []

    for cand in sorted(candidates, key=lambda x: x.candidate_id):
        cdir = candidates_dir / str(cand.candidate_id)
        cdir.mkdir(parents=True, exist_ok=True)

        rows_all = [
            r
            for r in all_results
            if str(r.get("candidate_id", "")) == str(cand.candidate_id)
        ]
        rows_all = sorted(
            rows_all,
            key=lambda x: (
                str(x.get("scenario_id", "")),
                str(x.get("split_id", "")),
                str(x.get("task_id", "")),
            ),
        )
        rows = [r for r in rows_all if str(r.get("status", "")) == "ok"]

        cid = str(cand.candidate_id)
        aligned = cid in baseline_col
        if aligned:
            ret_series = candidate_daily_mat[:, int(baseline_col[cid])].astype(np.float64)
        else:
            ret_series = np.zeros(common_sessions.shape[0], dtype=np.float64)
        loss_series = -ret_series

        baseline_map = (
            candidate_scenario_series.get(cid, {}).get("baseline", {})
            if candidate_scenario_series is not None
            else {}
        )
        ret_df = pdx.DataFrame(
            {
                "session_id": common_sessions.astype(np.int64),
                "returns": ret_series,
                "is_observed_baseline": np.asarray(
                    [int(int(s) in baseline_map) for s in common_sessions.tolist()],
                    dtype=np.int8,
                ),
            }
        )
        loss_df = pdx.DataFrame(
            {
                "session_id": common_sessions.astype(np.int64),
                "losses": loss_series,
            }
        )
        ret_path = cdir / "candidate_returns.parquet"
        loss_path = cdir / "candidate_losses.parquet"
        ret_df.to_parquet(ret_path, index=False)
        loss_df.to_parquet(loss_path, index=False)

        fold_rows: list[dict[str, Any]] = []
        fold_sharpes: list[float] = []
        fold_dsrs: list[float] = []
        rows_base_all = [r for r in rows_all if str(r.get("scenario_id", "")) == "baseline"]
        rows_base = [r for r in rows_base_all if str(r.get("status", "")) == "ok"]
        baseline_fail_reasons: list[str] = []
        if int(expected_baseline_tasks) > 0 and len(rows_base) < int(expected_baseline_tasks):
            baseline_fail_reasons.append(
                f"baseline_ok_tasks={len(rows_base)} expected={int(expected_baseline_tasks)}"
            )
        baseline_err_rows = [r for r in rows_base_all if str(r.get("status", "")) != "ok"]
        for er in baseline_err_rows:
            baseline_fail_reasons.append(
                f"{str(er.get('split_id', ''))}:{str(er.get('error_type', 'error'))}"
            )
        dqs_row_median = np.asarray(
            [float(r.get("dqs_median", np.nan)) for r in rows_all],
            dtype=np.float64,
        )
        dqs_row_min = np.asarray(
            [float(r.get("dqs_min", np.nan)) for r in rows_all],
            dtype=np.float64,
        )
        dqs_vals_med = dqs_row_median[np.isfinite(dqs_row_median)]
        dqs_vals_min = dqs_row_min[np.isfinite(dqs_row_min)]
        dqs_median = float(np.median(dqs_vals_med)) if dqs_vals_med.size > 0 else 1.0
        dqs_min = float(np.min(dqs_vals_min)) if dqs_vals_min.size > 0 else 1.0
        reason_codes_flat: list[str] = []
        for rr in rows_all:
            reason_codes_flat.extend([str(x) for x in rr.get("quality_reason_codes", [])])
        dq_degrade_count = int(sum(1 for rr in rows_all if "DQ_DEGRADED_INPUT" in [str(x) for x in rr.get("quality_reason_codes", [])]))
        dq_reject_count = int(sum(1 for rr in rows_all if "DQ_REJECTED_INPUT" in [str(x) for x in rr.get("quality_reason_codes", [])]))
        if reason_codes_flat:
            uniq = sorted(set(reason_codes_flat))
            freq = {k: int(reason_codes_flat.count(k)) for k in uniq}
            dq_reason_top = sorted(freq.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[0][0]
        else:
            dq_reason_top = ""
        failed_candidate = len(baseline_fail_reasons) > 0
        for r in (rows_base if rows_base else rows):
            tr = np.asarray(r.get("daily_returns", np.zeros(0, dtype=np.float64)), dtype=np.float64)
            if tr.size == 0:
                continue
            sharpe_d = _sharpe_daily(tr)
            fold_sharpes.append(sharpe_d)
            if tr.size >= 3:
                dsr_one = deflated_sharpe_ratio(tr[:, None])
                fold_dsrs.append(float(np.asarray(dsr_one["dsr"], dtype=np.float64)[0]))
            else:
                fold_dsrs.append(0.0)
            fold_rows.append(
                {
                    "split_id": str(r.get("split_id", "")),
                    "mode": _split_mode(str(r.get("split_id", ""))),
                    "scenario_id": str(r.get("scenario_id", "")),
                    "cum_return": _cum_return(tr),
                    "sharpe_daily": sharpe_d,
                    "max_drawdown": _max_drawdown_from_returns(tr),
                    "turnover": _turnover_from_trade_payload(r.get("trade_payload"), initial_cash),
                    "test_days": int(r.get("test_days", 0)),
                }
            )

        wf_rows = [x for x in fold_rows if str(x["mode"]) == "wf"]
        cpcv_rows = [x for x in fold_rows if str(x["mode"]) == "cpcv"]

        per_stress: dict[str, Any] = {}
        for sid in scenario_ids:
            srows = [r for r in rows if str(r.get("scenario_id", "")) == sid]
            if not srows:
                per_stress[sid] = {
                    "n_tasks": 0,
                    "cum_return_median": 0.0,
                    "cum_return_mean": 0.0,
                    "max_drawdown_median": 0.0,
                    "turnover_median": 0.0,
                }
                continue
            ret_list = [
                np.asarray(r.get("daily_returns", np.zeros(0, dtype=np.float64)), dtype=np.float64)
                for r in srows
            ]
            ret_list = [x for x in ret_list if x.size > 0]
            if not ret_list:
                per_stress[sid] = {
                    "n_tasks": 0,
                    "cum_return_median": 0.0,
                    "cum_return_mean": 0.0,
                    "max_drawdown_median": 0.0,
                    "turnover_median": 0.0,
                }
                continue
            cum = np.asarray([_cum_return(x) for x in ret_list], dtype=np.float64)
            dd = np.asarray([_max_drawdown_from_returns(x) for x in ret_list], dtype=np.float64)
            to = np.asarray(
                [_turnover_from_trade_payload(r.get("trade_payload"), initial_cash) for r in srows],
                dtype=np.float64,
            )
            per_stress[sid] = {
                "n_tasks": int(len(ret_list)),
                "cum_return_median": float(np.median(cum)),
                "cum_return_mean": float(np.mean(cum)),
                "max_drawdown_median": float(np.median(dd)),
                "turnover_median": float(np.median(to)),
            }

        base_stress = per_stress.get("baseline", {"cum_return_median": 0.0, "max_drawdown_median": 0.0, "turnover_median": 0.0})
        for sid in scenario_ids:
            per_stress[sid]["delta_vs_baseline"] = {
                "pnl": float(per_stress[sid]["cum_return_median"] - base_stress["cum_return_median"]),
                "dd": float(per_stress[sid]["max_drawdown_median"] - base_stress["max_drawdown_median"]),
                "turnover": float(per_stress[sid]["turnover_median"] - base_stress["turnover_median"]),
            }

        rows_for_base_stats = rows_base if rows_base else rows
        trade_payloads = [r.get("trade_payload") for r in rows_for_base_stats if r.get("trade_payload") is not None]
        eq_payloads = [r.get("equity_payload") for r in rows_for_base_stats if r.get("equity_payload") is not None]
        n_trades = int(sum(_trade_count_from_payload(p) for p in trade_payloads))

        ret_non_zero = ret_series[np.abs(ret_series) > 1e-15]
        win_rate = float(np.mean(ret_series > 0.0)) if ret_series.size else 0.0
        avg_trade = float(np.mean(ret_non_zero)) if ret_non_zero.size else 0.0
        pos_sum = float(np.sum(ret_series[ret_series > 0.0])) if ret_series.size else 0.0
        neg_sum = float(np.sum(ret_series[ret_series < 0.0])) if ret_series.size else 0.0
        profit_factor = float(pos_sum / max(abs(neg_sum), 1e-12)) if ret_series.size else 0.0
        max_dd = _max_drawdown_from_returns(ret_series)
        cagr_ish = float(np.power(max(1.0 + _cum_return(ret_series), 1e-12), 252.0 / max(float(ret_series.size), 1.0)) - 1.0)
        exposure_stats = _margin_exposure_stats_from_equity_payloads(eq_payloads)
        conc = _asset_pnl_concentration_from_result_rows(rows_for_base_stats)
        if conc <= 0.0:
            conc = _asset_notional_concentration_from_trade_payloads(trade_payloads)

        if ret_series.size >= 3:
            full_stats = run_full_stats(
                returns_matrix=ret_series[:, None],
                benchmark=daily_bmk,
                losses=loss_series[:, None],
                bootstrap_spec={"B": 256, "avg_block_len": 20, "seed": int(harness_cfg.seed + 601)},
                cpcv_params={"S": int(harness_cfg.cpcv_slices), "k": int(harness_cfg.cpcv_k_test)},
            )
            dsr_full = float(np.asarray(full_stats["dsr"]["dsr"], dtype=np.float64)[0])
            pbo_val = float(full_stats["pbo"]["pbo"]) if np.isfinite(float(full_stats["pbo"]["pbo"])) else 1.0
        else:
            full_stats = {
                "skipped_due_to_failure": True,
                "reason": "insufficient_aligned_baseline_days",
                "bootstrap_spec": {"B": 256, "avg_block_len": 20, "seed": int(harness_cfg.seed + 601)},
                "cpcv_params": {"S": int(harness_cfg.cpcv_slices), "k": int(harness_cfg.cpcv_k_test)},
            }
            dsr_full = 0.0
            pbo_val = 1.0
            if not failed_candidate:
                failed_candidate = True
                baseline_fail_reasons.append("insufficient_aligned_baseline_days")
        dsr_median = float(np.median(np.asarray(fold_dsrs, dtype=np.float64))) if fold_dsrs else dsr_full
        fold_sharpe_std = float(np.std(np.asarray(fold_sharpes, dtype=np.float64), ddof=1)) if len(fold_sharpes) > 1 else 0.0
        dd_severe = float(per_stress.get("severe", base_stress)["max_drawdown_median"])
        verdict_row = candidate_verdict.get(cid, {})

        if failed_candidate:
            robustness_score = float("-inf")
        else:
            robustness_score = float(
                1.0 * _clip01(dsr_median)
                - 0.5 * _clip01(pbo_val)
                - 0.3 * _clip01(dd_severe / ROBUSTNESS_CAPS["dd_cap"])
                - 0.2 * _clip01(fold_sharpe_std / ROBUSTNESS_CAPS["std_cap"])
                - 0.2 * _clip01(conc / ROBUSTNESS_CAPS["conc_cap"])
            )

        m3cfg = m3_configs[int(cand.m3_idx)]
        m4cfg = m4_configs[int(cand.m4_idx)]
        feat = {
            "entry_threshold": float(m4cfg.entry_threshold),
            "exit_threshold": float(m4cfg.exit_threshold),
            "top_k_intraday": float(m4cfg.top_k_intraday),
            "max_asset_cap_frac": float(m4cfg.max_asset_cap_frac),
            "max_turnover_frac_per_bar": float(m4cfg.max_turnover_frac_per_bar),
            "block_minutes": float(m3cfg.block_minutes),
            "min_block_valid_ratio": float(m3cfg.min_block_valid_ratio),
        }

        candidate_config = {
            "run_id": run_id,
            "timestamp_utc": run_started_utc.isoformat(),
            "git_hash": git_hash,
            "candidate_id": str(cand.candidate_id),
            "m2_idx": int(cand.m2_idx),
            "m3_idx": int(cand.m3_idx),
            "m4_idx": int(cand.m4_idx),
            "enabled_assets_mask": np.asarray(cand.enabled_assets_mask, dtype=bool).tolist(),
            "engine_config": asdict(engine_cfg),
            "module2_config": asdict(m2_configs[int(cand.m2_idx)]),
            "module3_config": asdict(m3cfg),
            "module4_config": asdict(m4cfg),
        }
        _write_json(cdir / "candidate_config.json", candidate_config)
        _write_json(cdir / "candidate_stats.json", full_stats)

        candidate_metrics = {
            "candidate_id": str(cand.candidate_id),
            "base_metrics": {
                "n_days": int(ret_series.size),
                "n_trades": n_trades,
                "win_rate": win_rate,
                "avg_trade": avg_trade,
                "profit_factor": profit_factor,
                "max_drawdown": max_dd,
                "cagr_ish": cagr_ish,
                "avg_turnover": float(np.median([_turnover_from_trade_payload(p, initial_cash) for p in trade_payloads])) if trade_payloads else 0.0,
                "asset_pnl_concentration": conc,
                **exposure_stats,
            },
            "per_stress": per_stress,
            "per_fold": {
                "wf": {
                    "summary": _summarize_fold_stats(wf_rows),
                    "folds": wf_rows,
                },
                "cpcv": {
                    "summary": _summarize_fold_stats(cpcv_rows),
                    "folds": cpcv_rows,
                },
            },
            "robustness": {
                "score": robustness_score,
                "formula": "1*clip(dsr_median)-0.5*clip(pbo)-0.3*clip(dd_severe/dd_cap)-0.2*clip(fold_sharpe_std/std_cap)-0.2*clip(asset_concentration/conc_cap)",
                "dsr_source": "baseline_fold_median",
                "inputs": {
                    "dsr_median": dsr_median,
                    "pbo": pbo_val,
                    "dd_severe": dd_severe,
                    "fold_sharpe_std": fold_sharpe_std,
                    "asset_pnl_concentration": conc,
                },
                "caps": dict(ROBUSTNESS_CAPS),
            },
            "failed": bool(failed_candidate),
            "failure_reasons": sorted(set(baseline_fail_reasons)),
            "alignment": {
                "aligned_to_global_benchmark_sessions": bool(aligned),
                "global_session_count": int(common_sessions.shape[0]),
                "observed_baseline_session_count": int(len(baseline_map)),
            },
            "dq_summary": {
                "dq_min": float(dqs_min),
                "dq_median": float(dqs_median),
                "dq_degrade_count": int(dq_degrade_count),
                "dq_reject_count": int(dq_reject_count),
                "dq_reason_top": str(dq_reason_top),
            },
        }
        _write_json(cdir / "candidate_metrics.json", candidate_metrics)

        candidate_rows.append(
            {
                "candidate_id": str(cand.candidate_id),
                "m2_idx": int(cand.m2_idx),
                "m3_idx": int(cand.m3_idx),
                "m4_idx": int(cand.m4_idx),
                "n_tasks": int(len(rows)),
                "n_tasks_baseline": int(len(rows_base)),
                "n_days": int(ret_series.size),
                "n_days_observed_baseline": int(len(baseline_map)),
                "cum_return": _cum_return(ret_series),
                "max_drawdown": max_dd,
                "dsr_full": dsr_full,
                "dsr_median": dsr_median,
                "pbo": pbo_val,
                "fold_sharpe_std": fold_sharpe_std,
                "asset_pnl_concentration": conc,
                "robustness_score": robustness_score,
                "failed": bool(failed_candidate),
                "failure_reasons": "|".join(sorted(set(baseline_fail_reasons))),
                "dq_min": float(dqs_min),
                "dq_median": float(dqs_median),
                "dq_degrade_count": int(dq_degrade_count),
                "dq_reject_count": int(dq_reject_count),
                "dq_reason_top": str(dq_reason_top),
                "in_mcs": bool(verdict_row.get("in_mcs", False)),
                "pass": bool(verdict_row.get("pass", False)),
                "wrc_p": float(verdict_row.get("wrc_p", np.nan)) if verdict_row else np.nan,
                "spa_p": float(verdict_row.get("spa_p", np.nan)) if verdict_row else np.nan,
                **feat,
            }
        )

    # Plateau detection via deterministic grid bins.
    group_map: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    for row in candidate_rows:
        key = _plateau_key(row)
        group_map.setdefault(key, []).append(row)

    clusters: list[dict[str, Any]] = []
    cand_to_plateau: dict[str, str] = {}
    for i, key in enumerate(sorted(group_map.keys())):
        rows = sorted(group_map[key], key=lambda x: str(x["candidate_id"]))
        scores = np.asarray([float(r["robustness_score"]) for r in rows], dtype=np.float64)
        rep = sorted(rows, key=lambda x: (-float(x["robustness_score"]), str(x["candidate_id"])))[0]
        pid = f"plateau_{i:03d}"
        for r in rows:
            cand_to_plateau[str(r["candidate_id"])] = pid
        clusters.append(
            {
                "plateau_id": pid,
                "bin_key": list(key),
                "count": int(len(rows)),
                "median_score": float(np.median(scores)),
                "worst_score": float(np.min(scores)),
                "representative_candidate_id": str(rep["candidate_id"]),
                "candidate_ids": [str(r["candidate_id"]) for r in rows],
            }
        )

    robustness_rows = []
    for row in candidate_rows:
        out = dict(row)
        out["plateau_id"] = cand_to_plateau.get(str(row["candidate_id"]), "plateau_unk")
        robustness_rows.append(out)
    def _robust_sort_key(x: dict[str, Any]) -> tuple[int, float, str]:
        s = float(x["robustness_score"])
        failed = 1 if (not np.isfinite(s) and s < 0.0) else 0
        ord_score = -s if np.isfinite(s) else float("inf")
        return (failed, ord_score, str(x["candidate_id"]))
    robustness_rows = sorted(
        robustness_rows,
        key=_robust_sort_key,
    )

    return candidate_rows, robustness_rows, {
        "method": "grid_binning",
        "bin_spec": {
            "entry_threshold": 0.02,
            "exit_threshold": 0.02,
            "top_k_intraday": 1.0,
            "max_asset_cap_frac": 0.05,
            "max_turnover_frac_per_bar": 0.05,
            "block_minutes": 5.0,
            "min_block_valid_ratio": 0.05,
        },
        "clusters": clusters,
    }


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
) -> HarnessOutput:
    if not m2_configs or not m3_configs or not m4_configs:
        raise RuntimeError("m2_configs/m3_configs/m4_configs must be non-empty")

    run_started_utc = datetime.now(timezone.utc)
    run_id = run_started_utc.strftime("run_%Y%m%d_%H%M%S")
    report_root = Path(harness_cfg.report_dir).resolve() / run_id
    report_root.mkdir(parents=True, exist_ok=True)
    deadletter_path = report_root / "deadletter_tasks.jsonl"
    run_status_path = report_root / "run_status.json"

    base_state, keep_idx, keep_symbols, master_ts_ns, ingest_meta, tick_keep, dq_bundle = _ingest_master_aligned(
        data_paths=data_paths,
        symbols=symbols,
        engine_cfg=engine_cfg,
        harness_cfg=harness_cfg,
        data_loader_func=data_loader_func,
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

    # RAM policy: reduce worker count if projected footprint is too high.
    avail = _available_memory_bytes()
    est_state = _estimate_state_bytes(base_state.cfg.T, base_state.cfg.A, base_state.cfg.B)
    requested_workers = int(max(1, harness_cfg.parallel_workers))
    max_workers = int(requested_workers)
    budget = int(float(harness_cfg.max_ram_utilization_frac) * float(avail))

    if est_state * max_workers > budget:
        max_workers = max(1, budget // max(est_state, 1))
    max_workers = max(1, max_workers)
    process_pool_requested = bool(harness_cfg.parallel_backend == "process_pool" and requested_workers > 1)
    if quick_settings.enabled:
        process_pool_requested = False

    group_tasks = _build_group_tasks(candidates, splits, scenarios)
    if process_pool_requested:
        group_tasks = _split_group_tasks_by_candidate(group_tasks, chunk_size=1)
    if not group_tasks:
        raise RuntimeError("No group tasks generated")

    all_results: list[dict[str, Any]] = []
    tasks_submitted = int(sum(len(g.candidate_indices) for g in group_tasks))
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

    def _write_run_status_checkpoint(phase: str, execution_mode_now: str) -> None:
        elapsed = float(time.perf_counter() - run_t0)
        _write_json(
            run_status_path,
            {
                "run_id": run_id,
                "phase": str(phase),
                "execution_mode": str(execution_mode_now),
                "groups_done": int(groups_completed),
                "groups_total": int(len(group_tasks)),
                "tasks_done": int(tasks_completed),
                "tasks_total": int(tasks_submitted),
                "failures_so_far": int(failure_count),
                "elapsed_seconds": float(elapsed),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    payload_safe, payload_arg_max_bytes = _is_large_payload_safe(
        group_tasks,
        threshold_bytes=int(harness_cfg.payload_pickle_threshold_bytes),
    )
    ram_forces_serial = bool(process_pool_requested and max_workers <= 1)
    payload_forces_serial = bool(process_pool_requested and (not payload_safe))

    if quick_settings.enabled:
        execution_mode = "serial_quick_run"
    elif process_pool_requested and (not payload_forces_serial) and (not ram_forces_serial):
        execution_mode = "process_pool"
    elif payload_forces_serial:
        execution_mode = "serial_forced_payload"
    else:
        execution_mode = "serial_forced_ram"

    use_process_pool = execution_mode == "process_pool"
    effective_workers = int(max_workers if use_process_pool else 1)
    large_payload_passing_avoided = bool((not use_process_pool) or payload_safe)
    _write_run_status_checkpoint("running", execution_mode)

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

    mp_start_method = ""
    if use_process_pool:
        mp_ctx, mp_start_method = _resolve_mp_context()
        with ProcessPoolExecutor(
            max_workers=effective_workers,
            mp_context=mp_ctx,
            initializer=_init_worker_context,
            initargs=(
                base_state,
                candidates,
                splits,
                scenarios,
                m2_configs,
                m3_configs,
                m4_configs,
                harness_cfg,
            ),
        ) as ex:
            futs: dict[Any, _GroupTask] = {}
            for g in group_tasks:
                fut = ex.submit(_run_group_task_from_context, g)
                futs[fut] = g
            pending = set(futs.keys())
            while pending:
                done, pending = wait(
                    pending,
                    timeout=float(pool_heartbeat_seconds),
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    _write_run_status_checkpoint("running", execution_mode)
                    continue
                for fut in sorted(done, key=lambda f: str(futs[f].group_id)):
                    g = futs[fut]
                    rows = _safe_execute_task(
                        g,
                        lambda _g: fut.result(),
                        candidates=candidates,
                        splits=splits,
                        scenarios=scenarios,
                        harness_cfg=harness_cfg,
                    )
                    all_results.extend(rows)
                    tasks_completed += int(len(rows))
                    groups_completed += 1
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
                            for rem in pending:
                                rem.cancel()
                            break
                    if (groups_completed % int(checkpoint_every_groups) == 0) or (groups_completed == len(group_tasks)):
                        _write_run_status_checkpoint("running", execution_mode)
                    if aborted:
                        break
                if aborted:
                    break
    else:
        for g in group_tasks:
            timeout_sec = int(quick_settings.task_timeout_sec) if quick_settings.enabled else 0
            rows = _safe_execute_task(
                g,
                lambda _g: _run_with_timeout_alarm(
                    timeout_sec,
                    lambda: _run_group_task(
                        _g,
                        base_state,
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
            all_results.extend(rows)
            tasks_completed += int(len(rows))
            groups_completed += 1
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
            if quick_settings.enabled:
                if (groups_completed % int(quick_settings.progress_every_groups) == 0) or (groups_completed == len(group_tasks)):
                    elapsed = time.perf_counter() - run_t0
                    print(
                        "QUICK_RUN_PROGRESS "
                        f"completed_groups={groups_completed}/{len(group_tasks)} "
                        f"completed_tasks={tasks_completed}/{tasks_submitted} "
                        f"failures={failure_count} elapsed_sec={elapsed:.1f}",
                        flush=True,
                    )
            if (groups_completed % int(checkpoint_every_groups) == 0) or (groups_completed == len(group_tasks)):
                _write_run_status_checkpoint("running", execution_mode)
            if aborted:
                break

    aborted_early = bool(aborted and tasks_completed < tasks_submitted)

    # Deterministic collation order.
    all_results.sort(
        key=lambda r: (
            str(r.get("candidate_id", "")),
            str(r.get("split_id", "")),
            str(r.get("scenario_id", "")),
            str(r.get("task_id", "")),
        )
    )

    ok_results = [r for r in all_results if r.get("status") == "ok" and int(r.get("test_days", 0)) > 0]

    bench_sessions, bench_ret = _benchmark_daily_returns(base_state, harness_cfg.benchmark_symbol)
    try:
        common_sessions, daily_mat, daily_bmk, baseline_candidate_ids, candidate_scenario_series = _aggregate_candidate_baseline_matrix(
            ok_results,
            bench_sessions,
            bench_ret,
            candidate_ids=[str(c.candidate_id) for c in candidates],
            min_days=int(harness_cfg.daily_return_min_days),
        )
        stats_verdict = _compute_stats_verdict(daily_mat, daily_bmk, baseline_candidate_ids, harness_cfg)
    except Exception as exc:
        if quick_settings.enabled:
            common_sessions = np.zeros(0, dtype=np.int64)
            daily_mat = np.zeros((0, 0), dtype=np.float64)
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
            raise
        else:
            if not first_exception_class:
                first_exception_class = type(exc).__name__
                first_exception_message = str(exc)
                first_exception_hash = _error_hash(first_exception_class, first_exception_message)
            common_sessions = np.zeros(0, dtype=np.int64)
            daily_mat = np.zeros((0, 0), dtype=np.float64)
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

    # Attach per-task leaderboard metrics back into candidate_results.
    candidate_results: list[dict[str, object]] = []
    for r in all_results:
        out = {
            "task_id": r.get("task_id"),
            "candidate_id": r.get("candidate_id"),
            "split_id": r.get("split_id"),
            "scenario_id": r.get("scenario_id"),
            "status": r.get("status"),
            "error": r.get("error", ""),
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
                    "dsr": lb["dsr"],
                    "in_mcs": lb["in_mcs"],
                    "wrc_p": lb["wrc_p"],
                    "spa_p": lb["spa_p"],
                    "pbo": lb["pbo"],
                    "pass": lb["pass"],
                }
            )
        candidate_results.append(out)

    eq_payloads = [r["equity_payload"] for r in ok_results if r.get("equity_payload") is not None]
    tr_payloads = [r["trade_payload"] for r in ok_results if r.get("trade_payload") is not None]
    micro_payloads = [r["micro_payload"] for r in ok_results if r.get("micro_payload") is not None]
    profile_payloads = [r["profile_payload"] for r in ok_results if r.get("profile_payload") is not None]
    funnel_payloads = [r["funnel_payload"] for r in ok_results if r.get("funnel_payload") is not None]

    eq_df = _stack_payload_frames(eq_payloads)
    tr_df = _stack_payload_frames(tr_payloads)
    micro_df = _stack_payload_frames(micro_payloads)
    profile_df = _stack_payload_frames(profile_payloads)
    funnel_df = _stack_payload_frames(funnel_payloads)

    pdx = _require_pandas()

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
            daily_cols[cid] = daily_mat[:, int(baseline_col[cid])]
        else:
            daily_cols[cid] = np.zeros(int(common_sessions.shape[0]), dtype=np.float64)
    daily_df = pdx.DataFrame(daily_cols)
    daily_df.to_parquet(daily_path, index=False)

    _write_json(stats_raw_path, stats_verdict)

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

    candidate_rows, robustness_rows, plateaus_doc = _build_candidate_artifacts(
        report_root=report_root,
        run_id=run_id,
        run_started_utc=run_started_utc,
        git_hash=_git_hash(),
        candidates=candidates,
        all_results=all_results,
        candidate_daily_mat=daily_mat,
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

    pdx.DataFrame(sorted(candidate_rows, key=lambda x: str(x["candidate_id"]))).to_csv(leaderboard_csv_path, index=False)
    _write_json(leaderboard_json_path, sorted(candidate_rows, key=lambda x: str(x["candidate_id"])))
    pdx.DataFrame(robustness_rows).to_csv(robustness_csv_path, index=False)
    _write_json(plateaus_path, plateaus_doc)
    _write_json(
        verdict_path,
        {
            "leaderboard": sorted(candidate_rows, key=lambda x: str(x["candidate_id"])),
            "summary": {
                "n_candidates_with_baseline": int(daily_mat.shape[1]),
                "n_candidates_total": int(len(candidates)),
                "n_days": int(daily_mat.shape[0]),
                "benchmark_symbol": harness_cfg.benchmark_symbol,
            },
        },
    )

    run_status = {
        "run_id": run_id,
        "aborted": bool(aborted),
        "aborted_early": bool(aborted_early),
        "abort_reason": str(abort_reason),
        "execution_mode": str(execution_mode),
        "process_start_method": str(mp_start_method if mp_start_method else mp.get_start_method(allow_none=True) or ""),
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
    }
    _write_json(run_status_path, run_status)

    manifest = {
        "run_id": run_id,
        "run_started_utc": run_started_utc.isoformat(),
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": _git_hash(),
        "seed": int(harness_cfg.seed),
        "symbols_input": list(symbols),
        "symbols_kept": keep_symbols,
        "keep_idx": keep_idx.tolist(),
        "ingestion": ingest_meta,
        "engine_cfg_hash": _stable_hash_obj(asdict(engine_cfg)),
        "m2_hashes": [_stable_hash_obj(asdict(c)) for c in m2_configs],
        "m3_hashes": [_stable_hash_obj(asdict(c)) for c in m3_configs],
        "m4_hashes": [_stable_hash_obj(asdict(c)) for c in m4_configs],
        "harness_cfg_hash": _stable_hash_obj(asdict(harness_cfg)),
        "n_candidates": len(candidates),
        "n_splits": len(splits),
        "n_scenarios": len(scenarios),
        "n_group_tasks": len(group_tasks),
        "n_task_results": len(all_results),
        "n_ok_results": len(ok_results),
        "execution_mode": str(execution_mode),
        "process_start_method": str(mp_start_method if mp_start_method else mp.get_start_method(allow_none=True) or ""),
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
        "daily_matrix_shape": list(daily_mat.shape),
        "n_candidates_with_baseline": int(len(baseline_candidate_ids)),
        "parallel_backend": harness_cfg.parallel_backend,
        "parallel_workers_effective": int(effective_workers),
        "payload_safe": bool(payload_safe),
        "payload_arg_max_bytes": int(payload_arg_max_bytes),
        "large_payload_passing_avoided": bool(large_payload_passing_avoided),
        "robustness_score": {
            "formula": "1*clip(dsr_median)-0.5*clip(pbo)-0.3*clip(dd_severe/dd_cap)-0.2*clip(fold_sharpe_std/std_cap)-0.2*clip(asset_concentration/conc_cap)",
            "dsr_source": "baseline_fold_median",
            "caps": dict(ROBUSTNESS_CAPS),
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
            "accept_count": int((dq_day_df.get("decision", pdx.Series(dtype=str)) == DQ_ACCEPT).sum())
            if dq_day_df.shape[0] > 0
            else 0,
            "degrade_count": int((dq_day_df.get("decision", pdx.Series(dtype=str)) == DQ_DEGRADE).sum())
            if dq_day_df.shape[0] > 0
            else 0,
            "reject_count": int((dq_day_df.get("decision", pdx.Series(dtype=str)) == DQ_REJECT).sum())
            if dq_day_df.shape[0] > 0
            else 0,
        },
    }
    _write_json(manifest_path, manifest)

    artifact_paths = {
        "equity_curves": str(equity_path),
        "trade_log": str(trade_path),
        "daily_returns": str(daily_path),
        "verdict": str(verdict_path),
        "stats_raw": str(stats_raw_path),
        "run_manifest": str(manifest_path),
        "run_status": str(run_status_path),
        "leaderboard_csv": str(leaderboard_csv_path),
        "leaderboard_json": str(leaderboard_json_path),
        "robustness_leaderboard_csv": str(robustness_csv_path),
        "plateaus": str(plateaus_path),
        "deadletter_tasks": str(deadletter_path),
        "dq_report_csv": str(dq_report_path),
        "dq_bar_flags_parquet": str(dq_bar_flags_path),
    }
    if bool(harness_cfg.export_micro_diagnostics):
        artifact_paths["micro_diagnostics"] = str(micro_diag_path)
        if bool(harness_cfg.micro_diag_export_block_profiles):
            artifact_paths["micro_profile_blocks"] = str(micro_profile_blocks_path)
        if bool(harness_cfg.micro_diag_export_funnel):
            artifact_paths["funnel_1545"] = str(funnel_1545_path)

    return HarnessOutput(
        candidate_results=candidate_results,
        daily_returns_matrix=daily_mat,
        daily_benchmark_returns=daily_bmk,
        stats_verdict=stats_verdict,
        artifact_paths=artifact_paths,
        run_manifest=manifest,
    )


if __name__ == "__main__":
    print("MODULE5_HARNESS_READY")
