from __future__ import annotations

import copy
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs" / "local_adaptive_discovery_7core.yaml"
OUTPUT_DIR = REPO_ROOT / "configs" / "_generated"
V2_OUTPUT = OUTPUT_DIR / "local_runtime_benchmark_7core_1536_v2.yaml"
BASELINE_OUTPUT = OUTPUT_DIR / "local_runtime_benchmark_7core_1536_baseline.yaml"


def _build_module3_matrix(seed_cfg: dict[str, object]) -> list[dict[str, object]]:
    windows = (15, 20, 30, 40)
    valid_ratios = (0.62, 0.68, 0.74, 0.80)
    out: list[dict[str, object]] = []
    for i, window in enumerate(windows):
        for j, ratio in enumerate(valid_ratios):
            cfg = copy.deepcopy(seed_cfg)
            cfg["block_minutes"] = int(window)
            cfg["min_block_valid_ratio"] = float(ratio)
            cfg["ib_pop_frac"] = float(round(0.008 + 0.0005 * i + 0.001 * j, 4))
            cfg["include_partial_last_block"] = bool(j < 2)
            out.append(cfg)
    return out


def _prepare_runtime_config(*, group_bound: bool, report_root_name: str) -> dict[str, object]:
    cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise RuntimeError("Base benchmark config is not a mapping")

    cfg["run_name"] = f"local_runtime_benchmark_7core_1536_{'v2' if group_bound else 'baseline'}"
    cfg["module3_configs"] = _build_module3_matrix(copy.deepcopy(cfg["module3_configs"][0]))

    harness = dict(cfg["harness"])
    harness.update(
        {
            "research_mode": "standard",
            "report_dir": f"./artifacts/benchmarks/{report_root_name}/module5_harness",
            "parallel_backend": "process_pool",
            "parallel_workers": 7,
            "group_bound_execution_enabled": bool(group_bound),
            "group_dispatch_policy": "largest_first_stable",
            "group_max_in_flight_factor": 2,
            "group_target_wall_time_sec": 20.0,
            "group_max_result_payload_bytes": 2 * 1024 * 1024,
            "group_max_memory_bytes": 0,
            "group_min_candidates_per_chunk": 1,
            "group_max_candidates_per_chunk_hard": 32,
            "startup_default_candidate_loop_sec": 0.50,
            "startup_default_result_payload_bytes": 16 * 1024,
            "startup_default_candidate_incremental_bytes": 512 * 1024,
            "startup_default_module3_bytes": 64 * 1024 * 1024,
            "scratch_mode": "compact",
            "strict_candidate_state_validation": "compact_execution_view",
            "risk_breach_state_dump_enabled": False,
            "debug_full_state_payloads": False,
            "module3_output_mode": "full_legacy",
            "base_sharing_mode": "auto",
            "process_pool_candidate_chunk_size": 1,
            "export_micro_diagnostics": False,
            "health_check_interval": 128,
            "progress_interval_seconds": 5,
        }
    )
    cfg["harness"] = harness
    return cfg


def _candidate_count(cfg: dict[str, object]) -> int:
    return int(len(cfg["module2_configs"]) * len(cfg["module3_configs"]) * len(cfg["module4_configs"]))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    v2_cfg = _prepare_runtime_config(group_bound=True, report_root_name="local_runtime_v2")
    baseline_cfg = _prepare_runtime_config(group_bound=False, report_root_name="local_runtime_baseline")

    V2_OUTPUT.write_text(yaml.safe_dump(v2_cfg, sort_keys=False), encoding="utf-8")
    BASELINE_OUTPUT.write_text(yaml.safe_dump(baseline_cfg, sort_keys=False), encoding="utf-8")

    print(f"wrote {V2_OUTPUT.relative_to(REPO_ROOT)} candidates={_candidate_count(v2_cfg)}")
    print(f"wrote {BASELINE_OUTPUT.relative_to(REPO_ROOT)} candidates={_candidate_count(baseline_cfg)}")


if __name__ == "__main__":
    main()
