from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_runtime_benchmark_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "runtime_benchmark.py"
    spec = importlib.util.spec_from_file_location("runtime_benchmark", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compare_benchmarks_prefers_uss_for_memory_driver(tmp_path: Path) -> None:
    benchmark = _load_runtime_benchmark_module()
    candidate_path = tmp_path / "candidate.json"
    baseline_path = tmp_path / "baseline.json"
    output_path = tmp_path / "comparison.json"

    candidate_path.write_text(
        json.dumps(
            {
                "elapsed_wall_sec": 10.0,
                "tasks_per_sec": 20.0,
                "queue_backlog_stats": {"p95": 50.0},
                "result_backlog_bytes_stats": {"p95": 0.0},
                "worker_rss_mb_stats": {"max": 800.0},
                "parent_rss_mb_stats": {"max": 300.0},
                "worker_uss_mb_stats": {"max": 200.0},
                "uss_supported": True,
            }
        ),
        encoding="utf-8",
    )
    baseline_path.write_text(
        json.dumps(
            {
                "elapsed_wall_sec": 20.0,
                "tasks_per_sec": 10.0,
                "queue_backlog_stats": {"p95": 50.0},
                "result_backlog_bytes_stats": {"p95": 0.0},
                "worker_rss_mb_stats": {"max": 700.0},
                "parent_rss_mb_stats": {"max": 250.0},
                "worker_uss_mb_stats": {"max": 400.0},
                "uss_supported": True,
            }
        ),
        encoding="utf-8",
    )

    benchmark.compare_benchmarks(
        candidate_path=candidate_path,
        baseline_path=baseline_path,
        output_path=output_path,
    )

    comparison = json.loads(output_path.read_text(encoding="utf-8"))
    assert comparison["memory_driver_metric"] == "worker_uss_mb"
    assert comparison["worker_memory_delta_mb"] == -200.0
    assert comparison["primary_driver"] == "memory"
