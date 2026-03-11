from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml
from module5.harness.artifact_writers import write_frozen_json

try:
    import psutil  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore[assignment]


MB = 1024.0 * 1024.0


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _resolve_report_dir(config_path: Path) -> Path:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    report_dir = Path(str(cfg["harness"]["report_dir"]))
    if not report_dir.is_absolute():
        report_dir = (REPO_ROOT / report_dir).resolve()
    return report_dir


def _resolve_run_dir(report_dir: Path) -> Path:
    latest_path = report_dir.parent / ".latest_run"
    if latest_path.exists():
        return Path(latest_path.read_text(encoding="utf-8").strip()).resolve()
    candidates = sorted((p for p in report_dir.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime_ns)
    if not candidates:
        raise RuntimeError(f"No run directory found under {report_dir}")
    return candidates[-1].resolve()


def _safe_p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    idx = min(len(ordered) - 1, int(math.ceil(0.95 * len(ordered))) - 1)
    return float(ordered[idx])


def _series_stats(values: list[float]) -> dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        return {"min": 0.0, "max": 0.0, "p95": 0.0, "last": 0.0}
    return {
        "min": float(min(vals)),
        "max": float(max(vals)),
        "p95": float(_safe_p95(vals)),
        "last": float(vals[-1]),
    }


def _sample_series(samples: list[dict[str, Any]], key: str) -> list[float]:
    return [float(sample[key]) for sample in samples if key in sample]


def _sample_process_tree_ps(root_pid: int, started_at: float) -> dict[str, Any]:
    parent_rss_mb = 0.0
    worker_rss_mb = 0.0
    worker_count = 0
    try:
        out = subprocess.check_output(["ps", "-axo", "pid=,ppid=,rss="], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        out = ""
    ppid_map: dict[int, int] = {}
    rss_map: dict[int, int] = {}
    for line in out.splitlines():
        parts = line.strip().split()
        if len(parts) != 3:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
            rss_kb = int(parts[2])
        except Exception:
            continue
        ppid_map[pid] = ppid
        rss_map[pid] = rss_kb
    if root_pid in rss_map:
        parent_rss_mb = float(rss_map[root_pid]) / 1024.0
    descendants: set[int] = set()
    frontier = [root_pid]
    while frontier:
        current = frontier.pop()
        children = [pid for pid, ppid in ppid_map.items() if ppid == current]
        for child in children:
            if child in descendants:
                continue
            descendants.add(child)
            frontier.append(child)
    for child in descendants:
        worker_rss_mb += float(rss_map.get(child, 0)) / 1024.0
        worker_count += 1
    return {
        "t_sec": float(time.perf_counter() - started_at),
        "parent_rss_mb": float(parent_rss_mb),
        "worker_rss_mb": float(worker_rss_mb),
        "worker_process_count": float(worker_count),
        "memory_sampler": "ps",
    }


def _memory_full_info_mb(proc: Any) -> dict[str, float]:
    try:
        full_info = proc.memory_full_info()
    except Exception:
        return {}
    metrics: dict[str, float] = {}
    for field in ("uss", "pss"):
        value = getattr(full_info, field, None)
        if value is None:
            continue
        metrics[f"{field}_mb"] = float(value) / MB
    return metrics


def _sample_process_tree(root_pid: int, started_at: float) -> dict[str, Any]:
    if psutil is None:
        return _sample_process_tree_ps(root_pid, started_at)
    proc = psutil.Process(root_pid)
    sample: dict[str, float | str] = {
        "t_sec": float(time.perf_counter() - started_at),
        "parent_rss_mb": 0.0,
        "worker_rss_mb": 0.0,
        "worker_process_count": 0.0,
        "memory_sampler": "psutil_rss_only",
    }
    try:
        sample["parent_rss_mb"] = float(proc.memory_info().rss) / MB
        parent_full = _memory_full_info_mb(proc)
        if parent_full:
            sample["memory_sampler"] = "psutil_full_info"
            if "uss_mb" in parent_full:
                sample["parent_uss_mb"] = float(parent_full["uss_mb"])
            if "pss_mb" in parent_full:
                sample["parent_pss_mb"] = float(parent_full["pss_mb"])
        children = proc.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        children = []
    for child in children:
        try:
            sample["worker_rss_mb"] = float(sample["worker_rss_mb"]) + (float(child.memory_info().rss) / MB)
            sample["worker_process_count"] = float(sample["worker_process_count"]) + 1.0
            child_full = _memory_full_info_mb(child)
            if child_full:
                sample["memory_sampler"] = "psutil_full_info"
                if "uss_mb" in child_full:
                    sample["worker_uss_mb"] = float(sample.get("worker_uss_mb", 0.0)) + float(child_full["uss_mb"])
                if "pss_mb" in child_full:
                    sample["worker_pss_mb"] = float(sample.get("worker_pss_mb", 0.0)) + float(child_full["pss_mb"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return {str(k): v for k, v in sample.items()}


def _resolve_memory_driver_metric(candidate: dict[str, Any], baseline: dict[str, Any]) -> tuple[str, float]:
    if bool(candidate.get("uss_supported", False)) and bool(baseline.get("uss_supported", False)):
        return (
            "worker_uss_mb",
            float(candidate.get("worker_uss_mb_stats", {}).get("max", 0.0))
            - float(baseline.get("worker_uss_mb_stats", {}).get("max", 0.0)),
        )
    return (
        "worker_rss_mb",
        float(candidate.get("worker_rss_mb_stats", {}).get("max", 0.0))
        - float(baseline.get("worker_rss_mb_stats", {}).get("max", 0.0)),
    )


def run_benchmark(*, config_path: Path, label: str, sample_interval_sec: float) -> Path:
    config_path = config_path.resolve()
    report_dir = _resolve_report_dir(config_path)
    report_dir.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = report_dir.parent / f"{label}_stdout.log"
    stderr_path = report_dir.parent / f"{label}_stderr.log"
    summary_path = report_dir.parent / f"{label}_benchmark_summary.json"

    cmd = [sys.executable, str(REPO_ROOT / "run_research.py"), "--config", str(config_path)]
    started_at = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout_fh, stderr_path.open("w", encoding="utf-8") as stderr_fh:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=stdout_fh,
            stderr=stderr_fh,
        )
        rss_samples: list[dict[str, Any]] = []
        while proc.poll() is None:
            rss_samples.append(_sample_process_tree(proc.pid, started_at))
            time.sleep(max(0.1, float(sample_interval_sec)))
        rss_samples.append(_sample_process_tree(proc.pid, started_at))
        exit_code = int(proc.wait())

    elapsed_wall_sec = float(time.perf_counter() - started_at)
    run_dir = _resolve_run_dir(report_dir)
    manifest = _load_json(run_dir / "run_manifest.json")
    run_status = _load_json(run_dir / "run_status.json")
    run_summary_path = run_dir / "run_summary.json"
    run_summary = _load_json(run_summary_path) if run_summary_path.exists() else {}
    health_rows = _load_jsonl(run_dir / "runtime_health_checks.jsonl")
    group_rows = _load_jsonl(run_dir / "group_runtime_stats.jsonl")
    worker_uss_series = _sample_series(rss_samples, "worker_uss_mb")
    parent_uss_series = _sample_series(rss_samples, "parent_uss_mb")
    worker_pss_series = _sample_series(rss_samples, "worker_pss_mb")
    parent_pss_series = _sample_series(rss_samples, "parent_pss_mb")

    tasks_completed = int(run_status.get("tasks_completed", manifest.get("tasks_completed", 0)))
    groups_completed = int(run_status.get("groups_completed", manifest.get("groups_completed", 0)))
    tasks_per_sec = float(tasks_completed / max(elapsed_wall_sec, 1e-9))
    summary = {
        "label": str(label),
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "exit_code": int(exit_code),
        "elapsed_wall_sec": float(elapsed_wall_sec),
        "tasks_completed": int(tasks_completed),
        "tasks_per_sec": float(tasks_per_sec),
        "tasks_per_hour": float(tasks_per_sec * 3600.0),
        "groups_completed": int(groups_completed),
        "groups_per_hour": float(groups_completed / max(elapsed_wall_sec, 1e-9) * 3600.0),
        "requested_workers": int(health_rows[-1].get("requested_workers", manifest.get("parallel_workers_effective", 0)) if health_rows else manifest.get("parallel_workers_effective", 0)),
        "effective_workers": int(health_rows[-1].get("effective_workers", manifest.get("parallel_workers_effective", 0)) if health_rows else manifest.get("parallel_workers_effective", 0)),
        "candidate_rows_processed": int(manifest.get("n_candidate_rows", 0)),
        "failures": int(run_summary.get("failure_count", manifest.get("failure_count", 0))),
        "final_artifact_generation_sec": float(manifest.get("final_artifact_generation_sec", 0.0)),
        "execution_topology": dict(run_status.get("execution_topology", {})),
        "health_checks_count": int(len(health_rows)),
        "group_stats_count": int(len(group_rows)),
        "run_summary_present": bool(run_summary_path.exists()),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "rss_samples": rss_samples,
        "memory_sampler": str(rss_samples[-1].get("memory_sampler", "unknown")) if rss_samples else "unknown",
        "rss_interpretation_caveat": (
            "RSS is per-process resident memory and can double-count shared pages; "
            "prefer USS/PSS when interpreting Linux copy-on-write runs."
        ),
        "worker_rss_mb_stats": _series_stats([float(x["worker_rss_mb"]) for x in rss_samples]),
        "parent_rss_mb_stats": _series_stats([float(x["parent_rss_mb"]) for x in rss_samples]),
        "uss_supported": bool(worker_uss_series or parent_uss_series),
        "pss_supported": bool(worker_pss_series or parent_pss_series),
        "worker_uss_mb_stats": _series_stats(worker_uss_series),
        "parent_uss_mb_stats": _series_stats(parent_uss_series),
        "worker_pss_mb_stats": _series_stats(worker_pss_series),
        "parent_pss_mb_stats": _series_stats(parent_pss_series),
        "queue_backlog_stats": _series_stats([float(row.get("queue_backlog", 0.0)) for row in health_rows]),
        "result_backlog_bytes_stats": _series_stats([float(row.get("result_backlog_bytes", 0.0)) for row in health_rows]),
        "module3_estimated_bytes_stats": _series_stats(
            [float(row.get("module3_estimated_bytes", 0.0)) for row in health_rows]
            or [float(row.get("module3_group_bytes_estimated", 0.0)) for row in group_rows]
        ),
        "module3_realized_bytes_stats": _series_stats(
            [float(row.get("module3_realized_bytes_p95", 0.0)) for row in health_rows]
            or [float(row.get("module3_group_bytes_realized", 0.0)) for row in group_rows]
        ),
        "active_effective_ratio_stats": _series_stats(
            [
                float(row.get("worker_status", {}).get("active", 0)) / max(1.0, float(row.get("worker_status", {}).get("expected", 1)))
                for row in health_rows
            ]
        ),
        "chunk_size_distribution": {},
        "chunk_size_series": [int(row.get("candidate_count", 0)) for row in group_rows],
        "module3_estimated_bytes_series": [int(row.get("module3_group_bytes_estimated", 0)) for row in group_rows],
        "module3_realized_bytes_series": [int(row.get("module3_group_bytes_realized", 0)) for row in group_rows],
        "health_series": health_rows,
    }

    chunk_dist: dict[str, int] = {}
    for row in group_rows:
        key = str(int(row.get("candidate_count", 0)))
        chunk_dist[key] = int(chunk_dist.get(key, 0) + 1)
    summary["chunk_size_distribution"] = chunk_dist

    write_frozen_json(summary_path, summary)
    print(json.dumps({"label": label, "summary_path": str(summary_path), "elapsed_wall_sec": elapsed_wall_sec, "tasks_per_sec": tasks_per_sec}, indent=2))
    return summary_path


def compare_benchmarks(*, candidate_path: Path, baseline_path: Path, output_path: Path | None) -> Path:
    candidate = _load_json(candidate_path)
    baseline = _load_json(baseline_path)
    speedup = float(baseline["elapsed_wall_sec"]) / max(1e-9, float(candidate["elapsed_wall_sec"]))
    throughput_gain = float(candidate["tasks_per_sec"]) / max(1e-9, float(baseline["tasks_per_sec"]))
    queue_delta = float(candidate["queue_backlog_stats"]["p95"] - baseline["queue_backlog_stats"]["p95"])
    result_backlog_delta = float(candidate["result_backlog_bytes_stats"]["p95"] - baseline["result_backlog_bytes_stats"]["p95"])
    memory_driver_metric, worker_memory_delta = _resolve_memory_driver_metric(candidate, baseline)
    worker_rss_delta = float(candidate["worker_rss_mb_stats"]["max"] - baseline["worker_rss_mb_stats"]["max"])
    parent_rss_delta = float(candidate["parent_rss_mb_stats"]["max"] - baseline["parent_rss_mb_stats"]["max"])

    drivers: list[str] = []
    if worker_memory_delta < -64.0:
        drivers.append("memory")
    if queue_delta < 0.0 or result_backlog_delta < 0.0:
        drivers.append("scheduling")
    if not drivers:
        drivers.append("compute")

    comparison = {
        "candidate_summary": str(candidate_path),
        "baseline_summary": str(baseline_path),
        "speedup_vs_baseline": float(speedup),
        "throughput_gain_vs_baseline": float(throughput_gain),
        "memory_driver_metric": str(memory_driver_metric),
        "worker_memory_delta_mb": float(worker_memory_delta),
        "worker_rss_delta_mb": float(worker_rss_delta),
        "parent_rss_delta_mb": float(parent_rss_delta),
        "queue_backlog_p95_delta": float(queue_delta),
        "result_backlog_bytes_p95_delta": float(result_backlog_delta),
        "primary_driver": "+".join(drivers),
    }
    resolved_output = output_path or candidate_path.parent / "benchmark_comparison.json"
    write_frozen_json(resolved_output, comparison)
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return resolved_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or compare local runtime benchmarks")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run")
    run_p.add_argument("--config", required=True, type=Path)
    run_p.add_argument("--label", required=True)
    run_p.add_argument("--sample-interval-sec", type=float, default=1.0)

    cmp_p = sub.add_parser("compare")
    cmp_p.add_argument("--candidate", required=True, type=Path)
    cmp_p.add_argument("--baseline", required=True, type=Path)
    cmp_p.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()
    if args.cmd == "run":
        run_benchmark(
            config_path=args.config,
            label=str(args.label),
            sample_interval_sec=float(args.sample_interval_sec),
        )
        return
    compare_benchmarks(
        candidate_path=args.candidate.resolve(),
        baseline_path=args.baseline.resolve(),
        output_path=args.output.resolve() if args.output is not None else None,
    )


if __name__ == "__main__":
    main()
