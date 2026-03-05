from __future__ import annotations

import atexit
from dataclasses import asdict
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import pickle
import multiprocessing as mp
import random
import resource
import signal
import traceback
from typing import Any

import numpy as np
import pandas as pd

from risk_engine import (
    CostConfig,
    REASON_RISK_CONSTRAINT_BREACH,
    REASON_WORKER_IO_VIOLATION,
    RiskConfig,
    SimulationResult,
    simulate_portfolio_task,
)
from strategy_engine import (
    EXPECTED_BASE_STRATEGY_COUNT,
    deterministic_jitter_seconds,
    family_counts,
    generate_sobol_strategy_specs,
    generate_strategy_specs,
    generate_swing_strategy_specs,
    strategy_payload,
    validate_grid_cardinality,
)
from weightiz_profile_engine import (
    MarketDataSharedHandle,
    SharedRegistry,
    attach_shared_buffers,
    cleanup_shared_buffers,
    close_attached_handles,
    load_and_align_market_data_once,
)
from weightiz_adaptive_search import adaptive_search


_WORKER_ARRAYS: dict[str, np.ndarray] = {}
_WORKER_HANDLES: dict[str, Any] = {}
_WORKER_SYMBOLS: tuple[str, ...] = ()
_WORKER_SPLIT_SESSIONS: dict[int, np.ndarray] = {}
_ACTIVE_SHM_REGISTRY: SharedRegistry | None = None
BATCH_SIZE = 50
CHECKPOINT_EVERY_BATCHES = 100


def _cleanup_active_shm_registry() -> None:
    global _ACTIVE_SHM_REGISTRY
    if _ACTIVE_SHM_REGISTRY is None:
        return
    try:
        cleanup_shared_buffers(_ACTIVE_SHM_REGISTRY)
    finally:
        _ACTIVE_SHM_REGISTRY = None


atexit.register(_cleanup_active_shm_registry)


def _cleanup_signal_handler(signum: int, frame: Any) -> None:
    del frame
    _cleanup_active_shm_registry()
    raise SystemExit(f"Received signal {int(signum)}")


def _install_randomness_guard() -> None:
    def _blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("RANDOMNESS_FORBIDDEN")

    random.random = _blocked  # type: ignore[assignment]
    random.randint = _blocked  # type: ignore[assignment]
    random.randrange = _blocked  # type: ignore[assignment]
    random.choice = _blocked  # type: ignore[assignment]
    random.choices = _blocked  # type: ignore[assignment]
    random.shuffle = _blocked  # type: ignore[assignment]
    random.uniform = _blocked  # type: ignore[assignment]

    np.random.seed = _blocked  # type: ignore[assignment]
    np.random.random = _blocked  # type: ignore[assignment]
    np.random.rand = _blocked  # type: ignore[assignment]
    np.random.randn = _blocked  # type: ignore[assignment]
    np.random.randint = _blocked  # type: ignore[assignment]
    np.random.default_rng = _blocked  # type: ignore[assignment]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_run_dir(project_root: Path, root_subdir: str = "artifacts") -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = (project_root / root_subdir / f"run_{ts}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _load_or_create_resume_run_dir(project_root: Path) -> tuple[Path, Path, Path]:
    root = (project_root / "artifacts" / "zimtra_sweep").resolve()
    root.mkdir(parents=True, exist_ok=True)
    resume_file = root / ".resume_run"
    incomplete_marker = root / ".run_incomplete"

    if resume_file.exists():
        resume_path = Path(resume_file.read_text(encoding="utf-8").strip()).expanduser()
        if resume_path.exists() and resume_path.is_dir():
            return resume_path.resolve(), resume_file, incomplete_marker

    run_dir = _make_run_dir(project_root, root_subdir="artifacts/zimtra_sweep")
    resume_file.write_text(str(run_dir) + "\n", encoding="utf-8")
    incomplete_marker.write_text(str(run_dir) + "\n", encoding="utf-8")
    return run_dir, resume_file, incomplete_marker


def _install_worker_io_guard() -> None:
    def _blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(f"{REASON_WORKER_IO_VIOLATION}: worker raw market data reads are forbidden")

    try:
        import pandas as _pd

        _pd.read_parquet = _blocked  # type: ignore[assignment]
        _pd.read_csv = _blocked  # type: ignore[assignment]
        _pd.read_pickle = _blocked  # type: ignore[assignment]
        _pd.read_json = _blocked  # type: ignore[assignment]
        _pd.read_hdf = _blocked  # type: ignore[assignment]
        _pd.read_feather = _blocked  # type: ignore[assignment]
    except Exception:
        pass

    try:
        import pyarrow.parquet as _pq

        _pq.read_table = _blocked  # type: ignore[assignment]
    except Exception:
        pass


def _worker_init(
    registry: SharedRegistry,
    symbols: tuple[str, ...],
    split_sessions_by_idx: dict[int, list[int]],
) -> None:
    global _WORKER_ARRAYS, _WORKER_HANDLES, _WORKER_SYMBOLS, _WORKER_SPLIT_SESSIONS
    _install_randomness_guard()
    _install_worker_io_guard()
    arrays, handles = attach_shared_buffers(registry)
    _WORKER_ARRAYS = arrays
    _WORKER_HANDLES = handles
    _WORKER_SYMBOLS = symbols
    _WORKER_SPLIT_SESSIONS = {
        int(k): np.asarray(v, dtype=np.int64) for k, v in split_sessions_by_idx.items()
    }


def _worker_shutdown() -> None:
    global _WORKER_HANDLES, _WORKER_SPLIT_SESSIONS
    close_attached_handles(_WORKER_HANDLES)
    _WORKER_HANDLES = {}
    _WORKER_SPLIT_SESSIONS = {}


def _build_wf_splits(
    ts_ns: np.ndarray,
    session_id: np.ndarray,
    train_months: int,
    test_months: int,
    n_splits: int,
    purge_days: int,
) -> list[dict[str, Any]]:
    ts = pd.to_datetime(ts_ns, utc=True)
    session_first_idx: dict[int, int] = {}
    session_date: dict[int, Any] = {}
    for i, sid in enumerate(session_id.tolist()):
        sid_i = int(sid)
        if sid_i not in session_first_idx:
            session_first_idx[sid_i] = int(i)
            session_date[sid_i] = ts[i].tz_convert("America/New_York").date()

    ordered_sessions = sorted(session_first_idx.keys())
    if len(ordered_sessions) < 40:
        raise RuntimeError("Not enough sessions for deterministic WF splits")

    session_dates = pd.Series([session_date[s] for s in ordered_sessions])
    first_date = pd.Timestamp(session_dates.iloc[0])

    splits: list[dict[str, Any]] = []
    for i in range(n_splits):
        train_start = first_date + pd.DateOffset(months=i * test_months)
        train_end = train_start + pd.DateOffset(months=train_months)
        test_start = train_end + pd.Timedelta(days=purge_days)
        test_end = test_start + pd.DateOffset(months=test_months)

        train_sids = [
            sid
            for sid in ordered_sessions
            if (pd.Timestamp(session_date[sid]) >= train_start) and (pd.Timestamp(session_date[sid]) < train_end)
        ]
        test_sids = [
            sid
            for sid in ordered_sessions
            if (pd.Timestamp(session_date[sid]) >= test_start) and (pd.Timestamp(session_date[sid]) < test_end)
        ]
        if not train_sids or not test_sids:
            continue

        mask = np.isin(session_id, np.asarray(test_sids, dtype=np.int64))
        splits.append(
            {
                "split_idx": len(splits),
                "train_sessions": train_sids,
                "test_sessions": test_sids,
                "test_mask": mask.astype(bool),
            }
        )
        if len(splits) >= n_splits:
            break

    if len(splits) != n_splits:
        raise RuntimeError(f"WF split construction failed: expected={n_splits} got={len(splits)}")
    return splits


def _build_cpcv_folds(n: int, k: int) -> list[tuple[int, ...]]:
    if n <= 0 or k <= 0 or k > n:
        raise RuntimeError(f"Invalid CPCV params n={n}, k={k}")
    folds = list(itertools.combinations(range(n), k))
    folds.sort()
    return folds


def _required_manifest_flags(grid_15120_locked: bool) -> dict[str, bool]:
    return {
        "WORKER_IO_GUARD": True,
        "INTEGER_SHARES": True,
        "DAY_START_EQUITY_PREV_CLOSE": True,
        "BUYING_POWER_CAP": True,
        "MISSING_BAR_HOLDING": True,
        "FIXED_ATR_STOP": True,
        "SHARED_MEMORY_DATA_TRANSPORT": True,
        "STRATEGY_GRID_15120_LOCKED": bool(grid_15120_locked),
    }


def _ensure_thread_caps() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"


def _build_task_signals(W: int) -> dict[str, np.ndarray]:
    key = f"W{int(W)}"
    return {
        "open": _WORKER_ARRAYS["open"],
        "high": _WORKER_ARRAYS["high"],
        "low": _WORKER_ARRAYS["low"],
        "close": _WORKER_ARRAYS["close"],
        "ATR": _WORKER_ARRAYS["atr"],
        "D": _WORKER_ARRAYS[f"D_{key}"],
        "A": _WORKER_ARRAYS[f"A_{key}"],
        "DELTA_EFF": _WORKER_ARRAYS[f"DELTA_EFF_{key}"],
        "S_BREAK": _WORKER_ARRAYS[f"S_BREAK_{key}"],
        "S_REJECT": _WORKER_ARRAYS[f"S_REJECT_{key}"],
        "RVOL": _WORKER_ARRAYS[f"RVOL_{key}"],
        "POC": _WORKER_ARRAYS[f"POC_{key}"],
        "VAH": _WORKER_ARRAYS[f"VAH_{key}"],
        "VAL": _WORKER_ARRAYS[f"VAL_{key}"],
    }


def _run_worker_task(task: dict[str, Any]) -> dict[str, Any]:
    try:
        strat = task["strategy"]
        wf_split_idx = int(task["wf_split_idx"])
        split_sessions = _WORKER_SPLIT_SESSIONS.get(wf_split_idx)
        if split_sessions is None:
            raise RuntimeError(f"Unknown wf_split_idx in worker: {wf_split_idx}")
        split_mask = np.isin(_WORKER_ARRAYS["session_id"], split_sessions)
        signals = _build_task_signals(int(strat["W"]))
        cost_cfg = CostConfig(**task["cost_cfg"])
        risk_cfg = RiskConfig(**task["risk_cfg"])

        res = simulate_portfolio_task(
            strategy=strat,
            signals=signals,
            symbols=_WORKER_SYMBOLS,
            split_mask=split_mask,
            cost_cfg=cost_cfg,
            risk_cfg=risk_cfg,
            initial_cash=float(task["initial_cash"]),
            ts_ns=_WORKER_ARRAYS["ts_ns"],
            minute_of_day=_WORKER_ARRAYS["minute_of_day"],
            session_id=_WORKER_ARRAYS["session_id"],
            bar_valid=_WORKER_ARRAYS["bar_valid"],
            last_valid_close=_WORKER_ARRAYS["last_valid_close"],
            wf_split_idx=int(task["wf_split_idx"]),
            cpcv_fold_idx=int(task["cpcv_fold_idx"]),
            scenario_id=str(task["scenario_id"]),
            active_asset_indices=task.get("active_assets"),
        )
        out = asdict(res)
        out["ok"] = True
        return out
    except Exception as exc:
        return {
            "ok": False,
            "strategy_id": str(task.get("strategy", {}).get("strategy_id", "")),
            "wf_split_idx": int(task.get("wf_split_idx", -1)),
            "cpcv_fold_idx": int(task.get("cpcv_fold_idx", -1)),
            "scenario_id": str(task.get("scenario_id", "")),
            "reason_code": REASON_RISK_CONSTRAINT_BREACH,
            "error_msg": f"{type(exc).__name__}: {exc}",
            "traceback_preview": traceback.format_exc(limit=8),
        }


def _run_worker_batch(batch: dict[str, Any]) -> dict[str, Any]:
    out_rows: list[dict[str, Any]] = []
    dead: list[dict[str, Any]] = []
    tasks = batch.get("tasks", [])
    for t in tasks:
        row = _run_worker_task(t)
        if bool(row.get("ok", False)):
            out_rows.append(row)
        else:
            dead.append(row)
    return {
        "batch_id": int(batch.get("batch_id", -1)),
        "rows": out_rows,
        "deadletters": dead,
        "task_count": int(len(tasks)),
    }


def chunk_strategies(strategies: list[Any], batch_size: int) -> list[list[Any]]:
    b = int(max(1, batch_size))
    return [strategies[i : i + b] for i in range(0, len(strategies), b)]


def _resolve_worker_count(requested: int) -> int:
    cpu_n = int(os.cpu_count() or 1)
    safe_cap = max(1, cpu_n - 2)
    return int(max(1, min(int(requested), 48, safe_cap)))


def _load_checkpoint_state(checkpoint_dir: Path) -> tuple[set[int], list[dict[str, Any]], list[dict[str, Any]]]:
    done_ids: set[int] = set()
    results: list[dict[str, Any]] = []
    deadletters: list[dict[str, Any]] = []

    done_file = checkpoint_dir / "completed_batches.txt"
    if done_file.exists():
        for line in done_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.isdigit():
                done_ids.add(int(line))

    rows_file = checkpoint_dir / "results.jsonl"
    if rows_file.exists():
        for line in rows_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except Exception:
                    continue

    dead_file = checkpoint_dir / "deadletters.jsonl"
    if dead_file.exists():
        for line in dead_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    deadletters.append(json.loads(line))
                except Exception:
                    continue

    return done_ids, results, deadletters


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _run_tasks_parallel(
    *,
    tasks: list[dict[str, Any]],
    workers: int,
    registry: SharedRegistry,
    symbols: tuple[str, ...],
    split_sessions_by_idx: dict[int, list[int]],
    run_dir: Path,
    phase_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    resolved_workers = _resolve_worker_count(int(workers))
    results: list[dict[str, Any]] = []
    deadletters: list[dict[str, Any]] = []

    if tasks:
        sample_task = tasks[0]
        payload_size = len(pickle.dumps(sample_task))
        MAX_PAYLOAD = 2 * 1024 * 1024
        if payload_size > MAX_PAYLOAD:
            raise RuntimeError(
                f"PAYLOAD_SIZE_GUARD_VIOLATION\n"
                f"Payload size: {payload_size} bytes\n"
                f"Maximum allowed: {MAX_PAYLOAD} bytes"
            )
        print(f"[PayloadGuard] task payload size verified: {payload_size} bytes")

    batches_raw = chunk_strategies(tasks, BATCH_SIZE)
    batches = [{"batch_id": i, "tasks": b} for i, b in enumerate(batches_raw)]

    checkpoint_dir = (run_dir / "checkpoints" / str(phase_name)).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    done_ids, ckpt_results, ckpt_dead = _load_checkpoint_state(checkpoint_dir)
    if ckpt_results:
        results.extend(ckpt_results)
    if ckpt_dead:
        deadletters.extend(ckpt_dead)

    pending = [b for b in batches if int(b["batch_id"]) not in done_ids]
    done_file = checkpoint_dir / "completed_batches.txt"
    rows_file = checkpoint_dir / "results.jsonl"
    dead_file = checkpoint_dir / "deadletters.jsonl"
    state_file = checkpoint_dir / "state.json"

    t0 = datetime.now(timezone.utc)
    batch_task_sizes = {int(b["batch_id"]): int(len(b["tasks"])) for b in batches}
    processed = int(sum(batch_task_sizes.get(bid, 0) for bid in done_ids))
    completed_batches = int(len(done_ids))
    total_batches = int(len(batches))
    total_tasks = int(len(tasks))

    if pending:
        chunksize = max(4, int(len(pending) / max(int(resolved_workers) * 8, 1)))
        with mp.Pool(
            processes=int(resolved_workers),
            initializer=_worker_init,
            initargs=(registry, symbols, split_sessions_by_idx),
        ) as pool:
            for batch_out in pool.imap_unordered(_run_worker_batch, pending, chunksize=int(chunksize)):
                bid = int(batch_out.get("batch_id", -1))
                rows = list(batch_out.get("rows", []))
                dead = list(batch_out.get("deadletters", []))
                task_count = int(batch_out.get("task_count", len(rows) + len(dead)))

                results.extend(rows)
                deadletters.extend(dead)
                _append_jsonl(rows_file, rows)
                _append_jsonl(dead_file, dead)
                with done_file.open("a", encoding="utf-8") as f:
                    f.write(f"{bid}\n")

                processed += task_count
                completed_batches += 1
                if (
                    completed_batches % CHECKPOINT_EVERY_BATCHES == 0
                    or completed_batches == total_batches
                ):
                    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
                    workers_alive = int(resolved_workers)
                    state_doc = {
                        "phase": str(phase_name),
                        "completed_batches": int(completed_batches),
                        "total_batches": int(total_batches),
                        "processed_tasks": int(processed),
                        "total_tasks": int(total_tasks),
                        "workers_requested": int(workers),
                        "workers_effective": int(resolved_workers),
                        "workers_alive": int(workers_alive),
                        "elapsed_seconds": float(elapsed),
                    }
                    state_file.write_text(json.dumps(state_doc, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(
                        f"[BatchProgress] phase={phase_name} "
                        f"completed_batches={completed_batches}/{total_batches} "
                        f"processed_tasks={processed}/{total_tasks} "
                        f"workers_alive={workers_alive} "
                        f"elapsed_sec={elapsed:.2f}"
                    )

    _worker_shutdown()
    dedup_results: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for r in results:
        k = (
            str(r.get("strategy_id", "")),
            int(r.get("wf_split_idx", -1)),
            int(r.get("cpcv_fold_idx", -1)),
            str(r.get("scenario_id", "")),
        )
        dedup_results[k] = r
    results = list(dedup_results.values())

    dedup_dead: dict[tuple[str, int, int, str, str], dict[str, Any]] = {}
    for r in deadletters:
        k = (
            str(r.get("strategy_id", "")),
            int(r.get("wf_split_idx", -1)),
            int(r.get("cpcv_fold_idx", -1)),
            str(r.get("scenario_id", "")),
            str(r.get("error_msg", "")),
        )
        dedup_dead[k] = r
    deadletters = list(dedup_dead.values())

    results.sort(
        key=lambda r: (
            str(r.get("strategy_id", "")),
            int(r.get("wf_split_idx", -1)),
            int(r.get("cpcv_fold_idx", -1)),
            str(r.get("scenario_id", "")),
        )
    )
    deadletters.sort(
        key=lambda r: (
            str(r.get("strategy_id", "")),
            int(r.get("wf_split_idx", -1)),
            int(r.get("cpcv_fold_idx", -1)),
            str(r.get("scenario_id", "")),
        )
    )
    return results, deadletters


def _screen_stage_a(
    *,
    strategies: list[dict[str, Any]],
    splits: list[dict[str, Any]],
    cfg: Any,
    registry: SharedRegistry,
    symbols: tuple[str, ...],
    initial_cash: float,
    split_sessions_by_idx: dict[int, list[int]],
    run_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    stage = cfg.zimtra_sweep.stage_a
    assets = [str(s).upper() for s in stage.screen_assets]
    selected = [i for i, s in enumerate(symbols) if s in set(assets)]
    if len(selected) != int(stage.gate_assets_total):
        raise RuntimeError(
            f"Stage-A screen assets mismatch: expected={stage.gate_assets_total}, got={len(selected)}"
        )

    subset_splits = splits[: int(stage.screen_wf_splits)]
    tasks: list[dict[str, Any]] = []
    for split in subset_splits:
        for spec in strategies:
            tasks.append(
                {
                    "strategy": spec,
                    "active_assets": selected,
                    "wf_split_idx": int(split["split_idx"]),
                    "cpcv_fold_idx": -1,
                    "scenario_id": "baseline",
                    "cost_cfg": asdict(CostConfig(**cfg.zimtra_sweep.cost.model_dump())),
                    "risk_cfg": asdict(RiskConfig(**cfg.zimtra_sweep.risk.model_dump())),
                    "initial_cash": float(initial_cash),
                }
            )

    results, dead = _run_tasks_parallel(
        tasks=tasks,
        workers=int(cfg.zimtra_sweep.stage_a.workers),
        registry=registry,
        symbols=symbols,
        split_sessions_by_idx=split_sessions_by_idx,
        run_dir=run_dir,
        phase_name="stage_a",
    )

    by_strategy: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        sid = str(r["strategy_id"])
        by_strategy.setdefault(sid, []).append(r)

    survivors: list[dict[str, Any]] = []
    by_id = {str(s["strategy_id"]): s for s in strategies}
    for sid in sorted(by_strategy.keys()):
        rows = by_strategy[sid]
        pf = float(np.mean([float(x.get("profit_factor", 0.0)) for x in rows]))
        mdd = float(np.max([float(x.get("max_drawdown", 1.0)) for x in rows]))
        breaches = int(np.sum([int(x.get("risk_breaches", 0)) for x in rows]))
        cumret_by_asset: dict[str, float] = {}
        for row in rows:
            d = row.get("per_asset_cumret", {})
            if isinstance(d, dict):
                for k, v in d.items():
                    cumret_by_asset[str(k)] = float(cumret_by_asset.get(str(k), 0.0) + float(v))
        positive_assets = int(np.sum([1 for _, v in cumret_by_asset.items() if float(v) > 0.0]))
        if (
            breaches == 0
            and pf > float(stage.gate_profit_factor_min)
            and mdd < float(stage.gate_max_drawdown_max)
            and positive_assets >= int(stage.gate_positive_assets_min)
        ):
            survivors.append(by_id[sid])

    survivors.sort(key=lambda x: str(x["strategy_id"]))
    return survivors, results, dead


def _stage_b_tasks(
    *,
    strategies: list[dict[str, Any]],
    splits: list[dict[str, Any]],
    cpcv_folds: list[tuple[int, ...]],
    scenarios: list[str],
    cfg: Any,
    initial_cash: float,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for split in splits:
        for fold_idx, _fold in enumerate(cpcv_folds):
            for scenario_id in scenarios:
                for spec in strategies:
                    tasks.append(
                        {
                            "strategy": spec,
                            "active_assets": None,
                            "wf_split_idx": int(split["split_idx"]),
                            "cpcv_fold_idx": int(fold_idx),
                            "scenario_id": str(scenario_id),
                            "cost_cfg": asdict(CostConfig(**cfg.zimtra_sweep.cost.model_dump())),
                            "risk_cfg": asdict(RiskConfig(**cfg.zimtra_sweep.risk.model_dump())),
                            "initial_cash": float(initial_cash),
                        }
                    )
    tasks.sort(
        key=lambda t: (
            str(t["strategy"]["strategy_id"]),
            int(t["wf_split_idx"]),
            int(t["cpcv_fold_idx"]),
            str(t["scenario_id"]),
        )
    )
    return tasks


def _aggregate_leaderboard(results: list[dict[str, Any]]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    agg = (
        df.groupby("strategy_id", sort=True)
        .agg(
            family=("strategy_id", lambda x: str(x.iloc[0]).split("_")[0]),
            tasks=("strategy_id", "count"),
            trades=("trades", "sum"),
            sharpe=("sharpe", "mean"),
            sortino=("sortino", "mean"),
            win_rate=("win_rate", "mean"),
            avg_ret=("avg_ret", "mean"),
            med_ret=("med_ret", "mean"),
            avg_holding_time_bars=("avg_holding_time_bars", "mean"),
            profit_factor=("profit_factor", "mean"),
            max_drawdown=("max_drawdown", "max"),
            risk_breaches=("risk_breaches", "sum"),
            daily_loss_breaches=("daily_loss_breaches", "sum"),
            final_equity=("final_equity", "mean"),
            exposure_utilization=("exposure_utilization", "mean"),
            reset_events=("reset_events", "sum"),
        )
        .reset_index()
    )
    agg["sharpe_adjusted_drawdown"] = agg["sharpe"] / (1.0 + agg["max_drawdown"])
    agg["avg_holding_time_days"] = agg["avg_holding_time_bars"] / 390.0
    agg = agg.sort_values(
        ["sharpe_adjusted_drawdown", "profit_factor", "max_drawdown", "strategy_id"],
        ascending=[False, False, True, True],
        kind="mergesort",
    )
    return agg


def _top100_per_asset(results: list[dict[str, Any]], leaderboard: pd.DataFrame) -> pd.DataFrame:
    if not results or leaderboard.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "rank",
                "strategy_id",
                "sharpe_adjusted_drawdown",
                "sharpe",
                "max_drawdown",
                "asset_pnl",
            ]
        )
    rows: list[dict[str, Any]] = []
    for r in results:
        sid = str(r.get("strategy_id", ""))
        per_asset = r.get("per_asset_cumret", {})
        if isinstance(per_asset, dict):
            for sym, pnl in per_asset.items():
                rows.append({"strategy_id": sid, "symbol": str(sym), "asset_pnl": float(pnl)})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    agg = (
        df.groupby(["symbol", "strategy_id"], sort=True)
        .agg(asset_pnl=("asset_pnl", "sum"))
        .reset_index()
    )
    merged = agg.merge(
        leaderboard[
            [
                "strategy_id",
                "sharpe_adjusted_drawdown",
                "sharpe",
                "max_drawdown",
            ]
        ],
        on="strategy_id",
        how="left",
    )
    merged = merged.sort_values(
        ["symbol", "sharpe_adjusted_drawdown", "asset_pnl", "strategy_id"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )
    merged["rank"] = merged.groupby("symbol").cumcount() + 1
    return merged[merged["rank"] <= 100].reset_index(drop=True)


def _write_deadletters(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_zimtra_sweep(
    *,
    cfg: Any,
    project_root: Path,
    config_path: Path,
    resolved_config_sha256: str,
) -> dict[str, Any]:
    _ensure_thread_caps()
    if bool(cfg.zimtra_sweep.forbid_randomness):
        _install_randomness_guard()
    signal.signal(signal.SIGTERM, _cleanup_signal_handler)
    signal.signal(signal.SIGINT, _cleanup_signal_handler)
    run_dir, resume_file, incomplete_marker = _load_or_create_resume_run_dir(project_root)

    workers = int(cfg.zimtra_sweep.workers)
    if workers > 50:
        raise RuntimeError(f"Worker cap exceeded: {workers} > 50")
    if str(cfg.zimtra_sweep.memory_transport) != "shared_memory":
        raise RuntimeError(
            f"Unsupported zimtra_sweep.memory_transport={cfg.zimtra_sweep.memory_transport!r}; "
            "only shared_memory is implemented (fail-closed)"
        )

    jitter = deterministic_jitter_seconds(str(cfg.run_name), int(cfg.harness.seed))
    time_doc = {
        "start_utc": _utc_now(),
        "jitter_seconds": int(jitter),
    }

    # deterministic jitter before first dataset load
    import time

    time.sleep(int(jitter))

    sampling = cfg.zimtra_sweep.sampling
    sampling_method = str(sampling.method).lower()
    if sampling_method == "sobol":
        required = (
            "profile_window_minutes",
            "profile_memory_sessions",
            "deltaeff_threshold",
            "distance_to_poc_atr",
            "acceptance_threshold",
            "rvol_filter",
            "holding_period_days",
        )
        ranges: dict[str, tuple[float, float]]
        lev_target = float(sampling.lev_target)
        if sampling.param_ranges:
            ranges = {
                str(k): (float(v[0]), float(v[1]))
                for k, v in sampling.param_ranges.items()
            }
        else:
            if cfg.zimtra_sweep.swing_grid is None:
                raise RuntimeError(
                    "Sobol sampling requires either zimtra_sweep.sampling.param_ranges "
                    "or zimtra_sweep.swing_grid to derive ranges"
                )
            g = cfg.zimtra_sweep.swing_grid.model_dump()
            ranges = {
                str(k): (float(min(g[k])), float(max(g[k])))
                for k in required
            }
            lev_target = float(g.get("lev_target", lev_target))

        missing = [k for k in required if k not in ranges]
        if missing:
            raise RuntimeError(
                "Sobol sampling requires explicit parameter ranges for all swing keys. "
                f"Missing: {missing}"
            )

        specs = generate_sobol_strategy_specs(
            n_samples=int(sampling.n_samples),
            param_ranges=ranges,
            seed=int(sampling.seed),
            lev_target=float(lev_target),
        )
    elif cfg.zimtra_sweep.swing_grid is not None:
        g = cfg.zimtra_sweep.swing_grid.model_dump()
        specs = generate_swing_strategy_specs(
            profile_window_minutes=[int(x) for x in g["profile_window_minutes"]],
            profile_memory_sessions=[int(x) for x in g["profile_memory_sessions"]],
            deltaeff_threshold=[float(x) for x in g["deltaeff_threshold"]],
            distance_to_poc_atr=[float(x) for x in g["distance_to_poc_atr"]],
            acceptance_threshold=[float(x) for x in g["acceptance_threshold"]],
            rvol_filter=[float(x) for x in g["rvol_filter"]],
            holding_period_days=[int(x) for x in g["holding_period_days"]],
            lev_target=float(g.get("lev_target", 1.5)),
        )
    else:
        specs = generate_strategy_specs()
        validate_grid_cardinality(specs)
        if len(specs) != EXPECTED_BASE_STRATEGY_COUNT:
            raise RuntimeError(
                f"STRATEGY_GRID_CARDINALITY_ERROR: expected={EXPECTED_BASE_STRATEGY_COUNT} got={len(specs)}"
            )
    strategy_rows = [strategy_payload(s) for s in specs]

    windows = sorted(set(int(s["W"]) for s in strategy_rows))
    handle: MarketDataSharedHandle = load_and_align_market_data_once(cfg, project_root, windows)
    global _ACTIVE_SHM_REGISTRY
    _ACTIVE_SHM_REGISTRY = handle.registry

    try:
        ts_ns = handle.local_arrays["ts_ns"]
        session_id = handle.local_arrays["session_id"]
        splits = _build_wf_splits(
            ts_ns=ts_ns,
            session_id=session_id,
            train_months=int(cfg.zimtra_sweep.cv.wf_train_months),
            test_months=int(cfg.zimtra_sweep.cv.wf_test_months),
            n_splits=int(cfg.zimtra_sweep.cv.wf_splits),
            purge_days=int(cfg.zimtra_sweep.cv.purge_trading_days),
        )
        cpcv_folds = _build_cpcv_folds(int(cfg.zimtra_sweep.cv.cpcv_n), int(cfg.zimtra_sweep.cv.cpcv_k))

        requested = [str(x).lower() for x in cfg.zimtra_sweep.scenarios.requested]
        available_hooks: list[str] = ["baseline"]
        if any(x not in available_hooks for x in requested):
            if bool(cfg.zimtra_sweep.scenarios.allow_baseline_only_without_hooks):
                scenarios = ["baseline"]
                scenario_mode = "baseline_only_missing_hooks"
            else:
                missing = sorted(set(requested) - set(available_hooks))
                raise RuntimeError(f"Scenario hooks missing for {missing}; fail-closed")
        else:
            scenarios = requested
            scenario_mode = "full_requested"

        stage_a_enabled = bool(cfg.zimtra_sweep.stage_a.enabled)
        stage_a_results: list[dict[str, Any]] = []
        stage_a_dead: list[dict[str, Any]] = []
        survivors = strategy_rows
        if stage_a_enabled:
            survivors, stage_a_results, stage_a_dead = _screen_stage_a(
                strategies=strategy_rows,
                splits=splits,
                cfg=cfg,
                registry=handle.registry,
                symbols=handle.symbols,
                initial_cash=float(cfg.engine.initial_cash),
                split_sessions_by_idx={
                    int(s["split_idx"]): [int(x) for x in s["test_sessions"]] for s in splits
                },
                run_dir=run_dir,
            )

        if len(survivors) == 0:
            raise RuntimeError("Stage-A produced zero survivors (fail-closed)")

        stage_b_tasks = _stage_b_tasks(
            strategies=survivors,
            splits=splits,
            cpcv_folds=cpcv_folds,
            scenarios=scenarios,
            cfg=cfg,
            initial_cash=float(cfg.engine.initial_cash),
        )

        rss_before = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        stage_b_results, stage_b_dead = _run_tasks_parallel(
            tasks=stage_b_tasks,
            workers=int(cfg.zimtra_sweep.stage_b.workers),
            registry=handle.registry,
            symbols=handle.symbols,
            split_sessions_by_idx={
                int(s["split_idx"]): [int(x) for x in s["test_sessions"]] for s in splits
            },
            run_dir=run_dir,
            phase_name="stage_b",
        )

        if bool(cfg.zimtra_sweep.adaptive.enabled):
            by_sid = {str(s["strategy_id"]): s for s in survivors}
            adaptive_input: list[dict[str, Any]] = []
            for row in stage_b_results:
                sid = str(row.get("strategy_id", ""))
                spec = by_sid.get(sid)
                if spec is None:
                    continue
                adaptive_input.append(
                    {
                        "strategy_id": sid,
                        "profit_factor": float(row.get("profit_factor", 0.0)),
                        "params": dict(spec),
                    }
                )
            adaptive_specs = adaptive_search(
                adaptive_input,
                int(cfg.zimtra_sweep.adaptive.new_samples),
                noise=float(cfg.zimtra_sweep.adaptive.noise),
                seed=int(cfg.zimtra_sweep.sampling.seed),
            )
            if adaptive_specs:
                for i, spec in enumerate(adaptive_specs):
                    spec["strategy_id"] = f"ADAPT_{i:06d}_{str(spec.get('strategy_id', ''))}"
                survivors.extend(adaptive_specs)
                stage_b_tasks_adapt = _stage_b_tasks(
                    strategies=adaptive_specs,
                    splits=splits,
                    cpcv_folds=cpcv_folds,
                    scenarios=scenarios,
                    cfg=cfg,
                    initial_cash=float(cfg.engine.initial_cash),
                )
                stage_b_results_adapt, stage_b_dead_adapt = _run_tasks_parallel(
                    tasks=stage_b_tasks_adapt,
                    workers=int(cfg.zimtra_sweep.stage_b.workers),
                    registry=handle.registry,
                    symbols=handle.symbols,
                    split_sessions_by_idx={
                        int(s["split_idx"]): [int(x) for x in s["test_sessions"]] for s in splits
                    },
                    run_dir=run_dir,
                    phase_name="stage_b_adaptive",
                )
                stage_b_results.extend(stage_b_results_adapt)
                stage_b_dead.extend(stage_b_dead_adapt)

        rss_after = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if rss_after > rss_before * 3 and rss_after > 0:
            raise RuntimeError("MEMORY_LEAK_GUARD_TRIGGERED")

        all_dead = stage_a_dead + stage_b_dead

        leaderboard_df = _aggregate_leaderboard(stage_b_results)
        leaderboard_path = run_dir / "leaderboard.csv"
        leaderboard_df.to_csv(leaderboard_path, index=False)
        (run_dir / "leaderboard.json").write_text(
            leaderboard_df.to_json(orient="records", indent=2), encoding="utf-8"
        )
        top_asset_df = _top100_per_asset(stage_b_results, leaderboard_df)
        top_asset_df.to_csv(run_dir / "top100_per_asset.csv", index=False)
        if not top_asset_df.empty:
            for sym in sorted(top_asset_df["symbol"].astype(str).unique().tolist()):
                block = top_asset_df[top_asset_df["symbol"].astype(str) == sym].head(100)
                print(f"\n[TOP_100_PER_ASSET] symbol={sym} rows={len(block)}")
                print(
                    block[
                        [
                            "rank",
                            "strategy_id",
                            "sharpe_adjusted_drawdown",
                            "sharpe",
                            "max_drawdown",
                            "asset_pnl",
                        ]
                    ].to_string(index=False)
                )

        # detailed artifacts
        eq_rows: list[dict[str, Any]] = []
        tr_rows: list[dict[str, Any]] = []
        dr_rows: list[dict[str, Any]] = []
        for row in stage_b_results:
            sid = str(row["strategy_id"])
            for e in row.get("equity_curve", []):
                eq_rows.append({"strategy_id": sid, **e})
            for tr in row.get("trade_log", []):
                tr_rows.append({"strategy_id": sid, **tr})
            for dr in row.get("daily_returns", []):
                dr_rows.append({"strategy_id": sid, **dr})

        pd.DataFrame(eq_rows).sort_values(["strategy_id", "ts_ns"], kind="mergesort").to_parquet(
            run_dir / "equity_curves.parquet", index=False
        )
        pd.DataFrame(tr_rows).sort_values(["strategy_id", "t", "symbol"], kind="mergesort").to_parquet(
            run_dir / "trade_log.parquet", index=False
        )
        pd.DataFrame(dr_rows).sort_values(["strategy_id", "day_key"], kind="mergesort").to_parquet(
            run_dir / "daily_returns.parquet", index=False
        )

        dq_report_df = pd.DataFrame(handle.dq_report).sort_values("symbol", kind="mergesort")
        dq_report_df.to_csv(run_dir / "dq_report.csv", index=False)

        bar_valid = handle.local_arrays["bar_valid"]
        invalid_count = np.sum(~bar_valid, axis=1)
        dq_flags = pd.DataFrame(
            {
                "ts_ns": handle.local_arrays["ts_ns"].astype(np.int64),
                "invalid_asset_count": invalid_count.astype(np.int64),
            }
        )
        dq_flags.to_parquet(run_dir / "dq_bar_flags.parquet", index=False)

        _write_deadletters(run_dir / "deadletter_tasks.jsonl", all_dead)

        manifest = {
            "run_id": run_dir.name,
            "run_name": str(cfg.run_name),
            "config_path": str(config_path),
            "resolved_config_sha256": str(resolved_config_sha256),
            "sampling_method": str(cfg.zimtra_sweep.sampling.method),
            "sampling_n_samples": int(cfg.zimtra_sweep.sampling.n_samples),
            "sampling_seed": int(cfg.zimtra_sweep.sampling.seed),
            "strategy_count_total": int(len(strategy_rows)),
            "strategy_count_stage_b": int(len(survivors)),
            "stage_b_reduction_ratio": float(len(survivors) / max(len(strategy_rows), 1)),
            "family_counts": family_counts(specs),
            "grid_mode": "swing_custom" if cfg.zimtra_sweep.swing_grid is not None else "default_15120",
            "wf_splits": int(len(splits)),
            "cpcv_folds": int(len(cpcv_folds)),
            "scenarios_requested": requested,
            "scenarios_effective": scenarios,
            "scenario_mode": scenario_mode,
            "workers": int(workers),
            "workers_stage_a": int(cfg.zimtra_sweep.stage_a.workers),
            "workers_stage_b": int(cfg.zimtra_sweep.stage_b.workers),
            "task_count_stage_b": int(len(stage_b_tasks)),
            "stage_a_results": int(len(stage_a_results)),
            "stage_b_results": int(len(stage_b_results)),
            "deadletter_count": int(len(all_dead)),
            "dataset_hash": str(handle.dataset_hash),
            "invariants": _required_manifest_flags(cfg.zimtra_sweep.swing_grid is None),
        }
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        pass_count = int(len(stage_b_results))
        failure_count = int(len(all_dead))
        total = int(pass_count + failure_count)
        summary = {
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "start_utc": time_doc["start_utc"],
            "end_utc": _utc_now(),
            "jitter_seconds": int(jitter),
            "strategy_count_total": int(len(strategy_rows)),
            "strategy_count_stage_b": int(len(survivors)),
            "stage_b_reduction_ratio": float(len(survivors) / max(len(strategy_rows), 1)),
            "task_count_total": int(total),
            "pass_count": int(pass_count),
            "failure_count": int(failure_count),
            "failure_rate": float(failure_count / max(total, 1)),
            "leaderboard_rows": int(leaderboard_df.shape[0]),
            "top100_per_asset_rows": int(top_asset_df.shape[0]),
        }
        (run_dir / "run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if resume_file.exists():
            resume_file.unlink()
        if incomplete_marker.exists():
            incomplete_marker.unlink()
        return summary
    finally:
        cleanup_shared_buffers(handle.registry)
        _ACTIVE_SHM_REGISTRY = None
