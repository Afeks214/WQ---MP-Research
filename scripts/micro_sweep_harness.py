# 1) imports
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from dataclasses import dataclass, field
import hashlib
import json
import multiprocessing
from pathlib import Path
import platform
import sys
import threading
import time
from typing import Any

import numpy as np
import pandas as pd
import psutil
from numba import njit, prange, set_num_threads


# 2) environment lock

def ensure_venv_lock() -> None:
    exe_raw = str(sys.executable)
    exe_resolved = str(Path(sys.executable).resolve())
    if (".venv" not in exe_raw) and (".venv" not in exe_resolved):
        raise RuntimeError("FATAL_ENVIRONMENT_ERROR: system python detected")


# 3) process cleanup

def cleanup_leftover_processes() -> None:
    """Terminate stale research processes safely (never self-kill)."""
    self_pid = os.getpid()
    markers = (
        "run_research",
        "sweep_runner",
        "benchmark",
        "micro_sweep",
    )

    victims: list[psutil.Process] = []
    for proc in psutil.process_iter(attrs=["pid", "cmdline", "name"]):
        try:
            pid = int(proc.info.get("pid") or -1)
            if pid <= 0 or pid == self_pid:
                continue
            cmdline = " ".join(proc.info.get("cmdline") or [])
            name = str(proc.info.get("name") or "")
            blob = f"{name} {cmdline}".lower()
            if any(m in blob for m in markers):
                victims.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    for proc in victims:
        try:
            proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    gone, alive = psutil.wait_procs(victims, timeout=2.0)
    del gone
    for proc in alive:
        try:
            proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


# 4) configuration constants
DEFAULT_BARS = 5000
DEFAULT_ASSETS = 4
DEFAULT_STRATEGIES = 64
DEFAULT_TASK_MULTIPLIER = 20
DEFAULT_TASKS = 1280
DEFAULT_WORKERS = 7
DEFAULT_BATCH_MULTIPLIER = 4

STALL_THRESHOLD_SEC = 10.0
CPU_UNDERSATURATION_THRESHOLD = 500.0
MEMORY_ALERT_MB = 3072.0

assert DEFAULT_TASKS == DEFAULT_STRATEGIES * DEFAULT_TASK_MULTIPLIER, "TASK_COUNT_MISMATCH"


# 5) dataset generation

def load_real_market_profile_dataset(parquet_path: str) -> dict[str, np.ndarray]:
    df = pd.read_parquet(parquet_path)

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"MISSING_REQUIRED_COLUMNS: {missing}")

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        if bool(ts.isna().any()):
            raise RuntimeError("INVALID_TIMESTAMP_VALUES")
        df = df.assign(timestamp=ts).sort_values("timestamp", kind="mergesort")
        df = df.drop_duplicates(subset=["timestamp"], keep="last")
    elif isinstance(df.index, pd.DatetimeIndex):
        idx = pd.to_datetime(df.index, utc=True, errors="coerce")
        if bool(np.any(pd.isna(idx))):
            raise RuntimeError("INVALID_TIMESTAMP_INDEX")
        df = df.copy()
        df.index = idx
        df = df.sort_index(kind="mergesort")
        df = df[~df.index.duplicated(keep="last")]
    else:
        raise RuntimeError("MISSING_TIMESTAMP_FOR_SORT")

    for col in required:
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        if np.any(~np.isfinite(vals)):
            raise RuntimeError(f"NON_FINITE_VALUES_IN_{col.upper()}")
        df[col] = vals

    open_ = df["open"].to_numpy(dtype=np.float64).reshape(-1, 1)
    high = df["high"].to_numpy(dtype=np.float64).reshape(-1, 1)
    low = df["low"].to_numpy(dtype=np.float64).reshape(-1, 1)
    close = df["close"].to_numpy(dtype=np.float64).reshape(-1, 1)
    volume = df["volume"].to_numpy(dtype=np.float64).reshape(-1, 1)

    bad_ohlc = (
        (high < low)
        | (high < open_)
        | (high < close)
        | (low > open_)
        | (low > close)
    )
    if bool(np.any(bad_ohlc)):
        raise RuntimeError("OHLC_CONSISTENCY_VIOLATION")

    bars = int(close.shape[0])
    if bars <= 5000:
        raise RuntimeError(f"INSUFFICIENT_BARS: bars={bars}")

    d = (close - np.mean(close, axis=0, keepdims=True)) / np.maximum(
        np.std(close, axis=0, keepdims=True), 1e-12
    )
    a_aff = np.exp(-np.abs(d))
    delta_eff = (close - open_) / np.maximum(np.abs(open_), 1e-12)
    s_break = 1.0 / (1.0 + np.exp(-(np.abs(d) + np.maximum(delta_eff, 0.0))))
    s_reject = 1.0 / (1.0 + np.exp(-(-np.abs(d) + a_aff)))

    return {
        "close": np.asarray(close, dtype=np.float64),
        "open": np.asarray(open_, dtype=np.float64),
        "high": np.asarray(high, dtype=np.float64),
        "low": np.asarray(low, dtype=np.float64),
        "volume": np.asarray(volume, dtype=np.float64),
        "D": np.asarray(d, dtype=np.float64),
        "A": np.asarray(a_aff, dtype=np.float64),
        "DELTA_EFF": np.asarray(delta_eff, dtype=np.float64),
        "S_BREAK": np.asarray(s_break, dtype=np.float64),
        "S_REJECT": np.asarray(s_reject, dtype=np.float64),
    }


# 6) simulated backtest task
WORKER_DATASET: dict[str, np.ndarray] | None = None
WORKER_HEARTBEAT: Any = None
WORKER_COUNT = 1


def simulated_backtest_task(task: dict[str, int]) -> dict[str, Any]:
    global WORKER_DATASET, WORKER_HEARTBEAT, WORKER_COUNT
    if WORKER_DATASET is None:
        raise RuntimeError("WORKER_DATASET_UNINITIALIZED")

    ident = multiprocessing.current_process()._identity
    worker_id = ((int(ident[0]) - 1) % max(1, int(WORKER_COUNT))) if ident else 0
    now = time.time()
    WORKER_HEARTBEAT[worker_id] = now

    task_id = int(task["task_id"])
    strategy_idx = int(task["strategy_idx"])

    close = WORKER_DATASET["close"]
    d_sig = WORKER_DATASET["D"]
    a_sig = WORKER_DATASET["A"]
    de_sig = WORKER_DATASET["DELTA_EFF"]
    sb_sig = WORKER_DATASET["S_BREAK"]
    sr_sig = WORKER_DATASET["S_REJECT"]

    bars, assets = close.shape
    asset_idx = strategy_idx % assets

    variation = 1.0 + ((task_id + strategy_idx) % 100) * 0.0001
    pos = 0
    cash = 0.0
    equity = 0.0
    trades = 0
    equity_points: list[float] = []

    for t in range(bars):
        if (t & 127) == 0:
            WORKER_HEARTBEAT[worker_id] = time.time()

        signal = (
            close[t, asset_idx] * 0.001 * variation
            + 0.2 * d_sig[t, asset_idx]
            + 0.1 * de_sig[t, asset_idx]
            + 0.1 * sb_sig[t, asset_idx]
            - 0.1 * sr_sig[t, asset_idx]
            + 0.05 * a_sig[t, asset_idx]
        )

        if signal > 0.08:
            if pos <= 0:
                trades += 1
            pos = 1
        elif signal < -0.08:
            if pos >= 0:
                trades += 1
            pos = -1

        pnl_step = pos * (close[t, asset_idx] - close[max(0, t - 1), asset_idx])
        cash += float(pnl_step)
        equity = cash

        if (t % 250) == 0:
            equity_points.append(float(equity))

    checksum_blob = f"{task_id}|{strategy_idx}|{equity:.12f}|{trades}|{sum(equity_points):.12f}"
    checksum = hashlib.sha256(checksum_blob.encode("utf-8")).hexdigest()[:16]

    return {
        "task_id": task_id,
        "strategy_idx": strategy_idx,
        "total_pnl": float(equity),
        "trade_count": int(trades),
        "checksum": checksum,
    }


# 7) watchdog system (init & thread)

def init_worker(dataset: dict[str, np.ndarray], heartbeat: Any, workers: int) -> None:
    global WORKER_DATASET, WORKER_HEARTBEAT, WORKER_COUNT
    WORKER_DATASET = dataset
    WORKER_HEARTBEAT = heartbeat
    WORKER_COUNT = int(workers)

    ident = multiprocessing.current_process()._identity
    if ident:
        worker_id = ((int(ident[0]) - 1) % max(1, int(WORKER_COUNT)))
        WORKER_HEARTBEAT[worker_id] = time.time()



def start_watchdog(
    heartbeat: Any,
    workers: int,
    stop_event: threading.Event,
    stall_threshold_sec: float,
) -> threading.Thread:
    def _loop() -> None:
        while not stop_event.is_set():
            now = time.time()
            for i in range(int(workers)):
                hb = float(heartbeat[i])
                if (now - hb) > float(stall_threshold_sec):
                    print(f"WORKER_STALL_DETECTED worker_id={i} lag_sec={now - hb:.2f}")
            time.sleep(1.0)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t


# 8) execution engines

def _checksum_results(rows: list[dict[str, Any]]) -> str:
    canonical = tuple(
        sorted(
            (
                int(r["task_id"]),
                int(r["strategy_idx"]),
                round(float(r["total_pnl"]), 10),
                int(r["trade_count"]),
                str(r["checksum"]),
            )
            for r in rows
        )
    )
    return hashlib.sha256(str(canonical).encode("utf-8")).hexdigest()


@njit(parallel=True, fastmath=True)
def numba_kernel_sim(close_col: np.ndarray, strategy_count: int, worker_count: int) -> tuple[float, np.int64]:
    S = int(strategy_count)
    W = int(max(1, worker_count))
    pnl_parts = np.zeros(W, dtype=np.float64)
    trade_parts = np.zeros(W, dtype=np.int64)

    chunk = S // W
    if chunk <= 0:
        chunk = 1

    for w in prange(W):
        start_strat = w * chunk
        end_strat = S if w == W - 1 else min((w + 1) * chunk, S)

        local_pnl = 0.0
        local_trades = 0

        for s in range(start_strat, end_strat):
            pos = 0
            variation = 1.0 + (s % 100) * 0.0001
            for t in range(close_col.shape[0]):
                signal = close_col[t] * 0.001 * variation
                if signal > 0.0:
                    local_pnl += signal
                    if pos <= 0:
                        local_trades += 1
                    pos = 1
                else:
                    local_pnl -= signal * 0.5
                    if pos >= 0:
                        local_trades += 1
                    pos = -1

        pnl_parts[w] = local_pnl
        trade_parts[w] = local_trades

    return float(np.sum(pnl_parts)), np.int64(np.sum(trade_parts))


def run_engine_current_mp(
    tasks: list[dict[str, int]],
    dataset: dict[str, np.ndarray],
    workers: int,
    heartbeat: Any,
    progress: "ProgressState",
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=int(workers),
        initializer=init_worker,
        initargs=(dataset, heartbeat, int(workers)),
    ) as ex:
        futures = [ex.submit(simulated_backtest_task, task) for task in tasks]
        for fut in as_completed(futures):
            row = fut.result()
            results.append(row)
            progress.increment()
    return results


def run_engine_streaming_mp(
    tasks: list[dict[str, int]],
    dataset: dict[str, np.ndarray],
    workers: int,
    heartbeat: Any,
    progress: "ProgressState",
    batch_size: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    idx = 0
    in_flight = set()

    with ProcessPoolExecutor(
        max_workers=int(workers),
        initializer=init_worker,
        initargs=(dataset, heartbeat, int(workers)),
    ) as ex:
        while idx < len(tasks) and len(in_flight) < int(batch_size):
            in_flight.add(ex.submit(simulated_backtest_task, tasks[idx]))
            idx += 1

        while in_flight:
            done, pending = wait(in_flight, return_when=FIRST_COMPLETED)
            for fut in done:
                row = fut.result()
                results.append(row)
                progress.increment()
                if idx < len(tasks):
                    pending.add(ex.submit(simulated_backtest_task, tasks[idx]))
                    idx += 1
            in_flight = pending

    return results


def run_engine_numpy_vector(dataset: dict[str, np.ndarray], strategies: int) -> list[dict[str, Any]]:
    close_col = dataset["close"][:, 0].astype(np.float64, copy=False)
    variations = (1.0 + (np.arange(strategies, dtype=np.float64) % 100.0) * 0.0001).reshape(-1, 1)

    signal = close_col.reshape(1, -1) * 0.001 * variations
    pnl = np.where(signal > 0.0, signal, -signal * 0.5)
    pnl_sum = np.sum(pnl, axis=1, dtype=np.float64)
    trade_count = np.sum(np.diff(np.sign(signal), axis=1) != 0, axis=1).astype(np.int64)

    rows: list[dict[str, Any]] = []
    for s in range(int(strategies)):
        rows.append(
            {
                "task_id": int(s),
                "strategy_idx": int(s),
                "total_pnl": float(pnl_sum[s]),
                "trade_count": int(trade_count[s]),
                "checksum": hashlib.sha256(f"{s}|{pnl_sum[s]:.12f}|{trade_count[s]}".encode("utf-8")).hexdigest()[:16],
                "note": "THROUGHPUT_PROXY_ONLY",
            }
        )
    return rows


def run_engine_numba_kernel(dataset: dict[str, np.ndarray], strategies: int, workers: int) -> list[dict[str, Any]]:
    close_col = dataset["close"][:, 0].astype(np.float64, copy=False)
    total_pnl, trade_count = numba_kernel_sim(close_col, int(strategies), int(max(1, workers - 1)))
    return [
        {
            "task_id": 0,
            "strategy_idx": -1,
            "total_pnl": float(total_pnl),
            "trade_count": int(trade_count),
            "checksum": hashlib.sha256(f"{total_pnl:.12f}|{trade_count}".encode("utf-8")).hexdigest()[:16],
            "note": "THROUGHPUT_PROXY_ONLY",
        }
    ]


# 9) telemetry system
@dataclass
class ProgressState:
    tasks_total: int
    engine_name: str
    started_at: float
    completed: int = 0
    cpu_samples: list[float] = field(default_factory=list)
    peak_ram_mb: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def increment(self) -> None:
        with self.lock:
            self.completed += 1

    def snapshot(self) -> tuple[int, int, float, float, float]:
        with self.lock:
            done = int(self.completed)
            total = int(self.tasks_total)
            peak = float(self.peak_ram_mb)
            elapsed = max(time.time() - self.started_at, 1e-9)
            tps = float(done / elapsed)
            eta = float((total - done) / max(tps, 1e-9)) if total > done else 0.0
            return done, total, tps, eta, peak


def start_telemetry(progress: ProgressState, stop_event: threading.Event) -> threading.Thread:
    proc = psutil.Process(os.getpid())
    psutil.cpu_percent(interval=None, percpu=True)

    def _loop() -> None:
        while not stop_event.is_set():
            cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
            cpu_total = float(sum(cpu_per_core))
            rss_mb = float(proc.memory_info().rss) / (1024.0 * 1024.0)

            with progress.lock:
                progress.cpu_samples.append(cpu_total)
                if rss_mb > progress.peak_ram_mb:
                    progress.peak_ram_mb = rss_mb

            done, total, tps, eta, peak = progress.snapshot()
            core_fmt = ",".join(f"{x:.1f}" for x in cpu_per_core)
            print(
                f"[{progress.engine_name}] {done}/{total} tps={tps:.2f} eta={eta:.1f}s "
                f"cpu_total={cpu_total:.1f}% cores=[{core_fmt}] rss={rss_mb:.1f}MB peak={peak:.1f}MB"
            )
            time.sleep(1.0)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t


# 10) benchmark runner & determinism test

def run_single_engine(
    engine_name: str,
    tasks: list[dict[str, int]],
    dataset: dict[str, np.ndarray],
    workers: int,
    batch_size: int,
    strategies: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tasks_total = len(tasks) if engine_name in {"current_mp", "streaming_mp"} else max(1, int(strategies))

    progress = ProgressState(tasks_total=tasks_total, engine_name=engine_name, started_at=time.time())

    heartbeat = multiprocessing.Array("d", int(workers))
    now = time.time()
    for i in range(int(workers)):
        heartbeat[i] = now

    stop_event = threading.Event()
    telemetry_thread = start_telemetry(progress, stop_event)
    watchdog_thread = start_watchdog(heartbeat, workers, stop_event, STALL_THRESHOLD_SEC)

    t0 = time.perf_counter()
    if engine_name == "current_mp":
        rows = run_engine_current_mp(tasks, dataset, workers, heartbeat, progress)
    elif engine_name == "streaming_mp":
        rows = run_engine_streaming_mp(tasks, dataset, workers, heartbeat, progress, batch_size)
    elif engine_name == "numpy_vector":
        rows = run_engine_numpy_vector(dataset, strategies)
        progress.completed = tasks_total
    elif engine_name == "numba_kernel":
        rows = run_engine_numba_kernel(dataset, strategies, workers)
        progress.completed = tasks_total
    else:
        stop_event.set()
        raise RuntimeError(f"Unknown engine: {engine_name}")

    elapsed = float(time.perf_counter() - t0)

    stop_event.set()
    telemetry_thread.join(timeout=2.0)
    watchdog_thread.join(timeout=2.0)

    avg_cpu = float(np.mean(progress.cpu_samples)) if progress.cpu_samples else 0.0
    peak_ram_mb = float(progress.peak_ram_mb)

    if avg_cpu < CPU_UNDERSATURATION_THRESHOLD:
        print(f"CPU_UNDERSATURATION_WARNING engine={engine_name} avg_cpu={avg_cpu:.1f}%")
    if peak_ram_mb > MEMORY_ALERT_MB:
        print(f"MEMORY_ALERT engine={engine_name} peak_ram_mb={peak_ram_mb:.1f}")

    total_pnl = float(sum(float(r.get("total_pnl", 0.0)) for r in rows))
    total_trades = int(sum(int(r.get("trade_count", 0)) for r in rows))

    summary = {
        "ENGINE": engine_name,
        "TIME_SEC": elapsed,
        "TASKS_PER_SEC": float(tasks_total / max(elapsed, 1e-9)),
        "CPU_UTIL_%": avg_cpu,
        "PEAK_RAM_MB": peak_ram_mb,
        "TOTAL_PNL": total_pnl,
        "TRADE_COUNT": total_trades,
        "CHECKSUM": _checksum_results(rows),
    }
    return summary, rows


def run_benchmark_suite(
    dataset: dict[str, np.ndarray],
    workers: int,
    bars: int,
    assets: int,
    strategies: int,
    task_multiplier: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tasks_total = int(strategies * task_multiplier)
    tasks = [{"task_id": i, "strategy_idx": i % strategies} for i in range(tasks_total)]
    batch_size = int(workers * DEFAULT_BATCH_MULTIPLIER)

    results: list[dict[str, Any]] = []

    current_1, rows_1 = run_single_engine("current_mp", tasks, dataset, workers, batch_size, strategies)
    current_2, rows_2 = run_single_engine("current_mp", tasks, dataset, workers, batch_size, strategies)

    if (
        not np.isclose(current_1["TOTAL_PNL"], current_2["TOTAL_PNL"], rtol=0.0, atol=1e-12)
        or int(current_1["TRADE_COUNT"]) != int(current_2["TRADE_COUNT"])
        or str(current_1["CHECKSUM"]) != str(current_2["CHECKSUM"])
    ):
        print("DETERMINISM_FAILURE")

    results.append(current_1)

    stream_r, _ = run_single_engine("streaming_mp", tasks, dataset, workers, batch_size, strategies)
    numpy_r, _ = run_single_engine("numpy_vector", tasks, dataset, workers, batch_size, strategies)
    numba_r, _ = run_single_engine("numba_kernel", tasks, dataset, workers, batch_size, strategies)

    results.extend([stream_r, numpy_r, numba_r])

    determinism_doc = {
        "run1_total_pnl": float(current_1["TOTAL_PNL"]),
        "run2_total_pnl": float(current_2["TOTAL_PNL"]),
        "run1_trade_count": int(current_1["TRADE_COUNT"]),
        "run2_trade_count": int(current_2["TRADE_COUNT"]),
        "run1_checksum": str(current_1["CHECKSUM"]),
        "run2_checksum": str(current_2["CHECKSUM"]),
        "match": bool(
            np.isclose(current_1["TOTAL_PNL"], current_2["TOTAL_PNL"], rtol=0.0, atol=1e-12)
            and int(current_1["TRADE_COUNT"]) == int(current_2["TRADE_COUNT"])
            and str(current_1["CHECKSUM"]) == str(current_2["CHECKSUM"])
        ),
    }

    workload_doc = {
        "bars": int(bars),
        "assets": int(assets),
        "strategies": int(strategies),
        "task_multiplier": int(task_multiplier),
        "tasks": int(tasks_total),
    }

    return results, {"determinism": determinism_doc, "workload": workload_doc}


# 11) result storage & Azure projection

def write_outputs(
    output_dir: Path,
    results: list[dict[str, Any]],
    meta: dict[str, Any],
    workers: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in results:
        rows.append(
            {
                "ENGINE": str(r["ENGINE"]),
                "TIME_SEC": float(r["TIME_SEC"]),
                "TASKS_PER_SEC": float(r["TASKS_PER_SEC"]),
                "CPU_UTIL_%": float(r["CPU_UTIL_%"]),
                "PEAK_RAM_MB": float(r["PEAK_RAM_MB"]),
                "AZURE_TPS_CONSERVATIVE": float(r["TASKS_PER_SEC"]) * (48.0 / float(workers)) * 0.60,
                "AZURE_TPS_EXPECTED": float(r["TASKS_PER_SEC"]) * (48.0 / float(workers)) * 0.75,
                "AZURE_TPS_OPTIMISTIC": float(r["TASKS_PER_SEC"]) * (48.0 / float(workers)) * 0.85,
            }
        )

    df = pd.DataFrame(rows, columns=[
        "ENGINE",
        "TIME_SEC",
        "TASKS_PER_SEC",
        "CPU_UTIL_%",
        "PEAK_RAM_MB",
        "AZURE_TPS_CONSERVATIVE",
        "AZURE_TPS_EXPECTED",
        "AZURE_TPS_OPTIMISTIC",
    ])
    df.to_csv(output_dir / "results.csv", index=False)

    lines = []
    lines.append("# Micro Sweep Benchmark Results")
    lines.append("")
    lines.append("ENGINE | TIME_SEC | TASKS_PER_SEC | CPU_UTIL_% | PEAK_RAM_MB")
    lines.append("---|---:|---:|---:|---:")
    for _, r in df.iterrows():
        lines.append(
            f"{r['ENGINE']} | {r['TIME_SEC']:.4f} | {r['TASKS_PER_SEC']:.4f} | {r['CPU_UTIL_%']:.2f} | {r['PEAK_RAM_MB']:.2f}"
        )

    lines.append("")
    lines.append("## Azure E64ads v7 Projection (48 workers)")
    lines.append("")
    lines.append("ENGINE | CONSERVATIVE_0.60 | EXPECTED_0.75 | OPTIMISTIC_0.85")
    lines.append("---|---:|---:|---:")
    for _, r in df.iterrows():
        lines.append(
            f"{r['ENGINE']} | {r['AZURE_TPS_CONSERVATIVE']:.4f} | {r['AZURE_TPS_EXPECTED']:.4f} | {r['AZURE_TPS_OPTIMISTIC']:.4f}"
        )

    lines.append("")
    lines.append("## Determinism")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(meta.get("determinism", {}), ensure_ascii=False, indent=2))
    lines.append("```")
    (output_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    vm = psutil.virtual_memory()
    system_profile = {
        "cpu_count_logical": int(psutil.cpu_count(logical=True) or 0),
        "cpu_count_physical": int(psutil.cpu_count(logical=False) or 0),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "numpy_version": np.__version__,
        "numba_version": getattr(sys.modules.get("numba"), "__version__", "unknown"),
        "pandas_version": pd.__version__,
        "total_ram_bytes": int(vm.total),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "workload": meta.get("workload", {}),
        "determinism": meta.get("determinism", {}),
    }
    (output_dir / "system_profile.json").write_text(
        json.dumps(system_profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# 12) main()

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deterministic micro sweep architecture benchmark")
    p.add_argument("--bars", type=int, default=DEFAULT_BARS)
    p.add_argument("--assets", type=int, default=DEFAULT_ASSETS)
    p.add_argument("--strategies", type=int, default=DEFAULT_STRATEGIES)
    p.add_argument("--task-multiplier", type=int, default=DEFAULT_TASK_MULTIPLIER)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--output-dir", type=str, default="artifacts/micro_sweep")
    return p.parse_args()


def main() -> None:
    multiprocessing.set_start_method("spawn", force=True)

    ensure_venv_lock()
    cleanup_leftover_processes()

    args = parse_args()

    bars = int(args.bars)
    assets = int(args.assets)
    strategies = int(args.strategies)
    task_multiplier = int(args.task_multiplier)
    workers = int(args.workers)

    if workers <= 0:
        raise RuntimeError("WORKER_COUNT_INVALID")

    tasks = int(strategies * task_multiplier)
    if tasks != int(strategies * task_multiplier):
        raise RuntimeError("TASK_COUNT_MISMATCH")

    # one-time numba thread pool setup after final worker resolution
    set_num_threads(max(1, workers - 1))

    dataset = load_real_market_profile_dataset("part-2023-1Min.parquet")
    bars = int(dataset["close"].shape[0])
    assets = int(dataset["close"].shape[1])

    results, meta = run_benchmark_suite(
        dataset=dataset,
        workers=workers,
        bars=bars,
        assets=assets,
        strategies=strategies,
        task_multiplier=task_multiplier,
    )

    output_dir = Path(args.output_dir).resolve()
    write_outputs(output_dir=output_dir, results=results, meta=meta, workers=workers)

    table_df = pd.DataFrame(
        [
            {
                "ENGINE": r["ENGINE"],
                "TIME_SEC": r["TIME_SEC"],
                "TASKS_PER_SEC": r["TASKS_PER_SEC"],
                "CPU_UTIL_%": r["CPU_UTIL_%"],
                "PEAK_RAM_MB": r["PEAK_RAM_MB"],
            }
            for r in results
        ]
    )

    print("ENGINE | TIME_SEC | TASKS_PER_SEC | CPU_UTIL_% | PEAK_RAM_MB")
    for _, row in table_df.iterrows():
        print(
            f"{row['ENGINE']} | {row['TIME_SEC']:.4f} | {row['TASKS_PER_SEC']:.4f} | "
            f"{row['CPU_UTIL_%']:.2f} | {row['PEAK_RAM_MB']:.2f}"
        )


if __name__ == "__main__":
    main()
