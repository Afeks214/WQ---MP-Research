# Linux Proof Run Stage 1

## Scope

This runbook defines the first real Linux proof run for the grouped runtime hardening branch. It is a proof-run readiness gate only. It does not declare Linux production readiness and it does not waive the known final-artifact parent-memory accumulation risk.

## Required Environment

- Linux host
- Python `3.9.6` from the repository root `.python-version`
- Repository root as current working directory
- Existing virtual environment at `./.venv`
- `WEIGHTIZ_MP_START_METHOD` unset for the primary proof run

## Exact Command Block

```bash
python -V
uname -a
env | grep WEIGHTIZ_MP_START_METHOD
./.venv/bin/python scripts/build_local_runtime_benchmark_configs.py
./.venv/bin/python scripts/runtime_benchmark.py run --label v2_linux_proof --config configs/_generated/local_runtime_benchmark_7core_1536_v2.yaml --sample-interval-sec 1.0
./.venv/bin/python scripts/runtime_benchmark.py run --label baseline_linux_proof --config configs/_generated/local_runtime_benchmark_7core_1536_baseline.yaml --sample-interval-sec 1.0
./.venv/bin/python scripts/runtime_benchmark.py compare --candidate artifacts/benchmarks/local_runtime_v2/v2_linux_proof_benchmark_summary.json --baseline artifacts/benchmarks/local_runtime_baseline/baseline_linux_proof_benchmark_summary.json --output artifacts/benchmarks/linux_proof_run_stage1_comparison.json
```

`env | grep WEIGHTIZ_MP_START_METHOD` is expected to produce no output for the primary proof run. If it prints a value, the proof run is not valid.

## Required Artifacts To Inspect

V2 run:

- `artifacts/benchmarks/local_runtime_v2/module5_harness/<run_id>/run_status.json`
- `artifacts/benchmarks/local_runtime_v2/module5_harness/<run_id>/run_manifest.json`
- `artifacts/benchmarks/local_runtime_v2/module5_harness/<run_id>/run_summary.json`
- `artifacts/benchmarks/local_runtime_v2/module5_harness/<run_id>/runtime_health_checks.jsonl`
- `artifacts/benchmarks/local_runtime_v2/module5_harness/<run_id>/group_runtime_stats.jsonl`
- `artifacts/benchmarks/local_runtime_v2/v2_linux_proof_benchmark_summary.json`

Baseline run:

- `artifacts/benchmarks/local_runtime_baseline/module5_harness/<run_id>/run_status.json`
- `artifacts/benchmarks/local_runtime_baseline/module5_harness/<run_id>/run_manifest.json`
- `artifacts/benchmarks/local_runtime_baseline/module5_harness/<run_id>/run_summary.json`
- `artifacts/benchmarks/local_runtime_baseline/module5_harness/<run_id>/runtime_health_checks.jsonl`
- `artifacts/benchmarks/local_runtime_baseline/module5_harness/<run_id>/group_runtime_stats.jsonl`
- `artifacts/benchmarks/local_runtime_baseline/baseline_linux_proof_benchmark_summary.json`

Comparison:

- `artifacts/benchmarks/linux_proof_run_stage1_comparison.json`

## Hard Acceptance Gates

### A. Start-Method Truth

- `run_status.process_start_method` must be non-empty for both runs.
- For the primary proof run, `WEIGHTIZ_MP_START_METHOD` must be unset.
- If `run_status.process_start_method == "fork"` and `run_status.execution_topology.base_sharing.configured_mode == "auto"`, then `run_status.execution_topology.base_sharing.resolved_mode` must be `fork_cow`.
- If `run_status.execution_topology.base_sharing.resolved_mode == "fork_cow"`, then `run_status.execution_topology.base_sharing.shared_base_state_active` must be `true`.
- If `run_status.execution_topology.base_sharing.resolved_mode == "serialized_copy"`, then `run_status.execution_topology.base_sharing.shared_base_state_active` must be `false` and `fallback_only` must be `true`.

Exact fail-closed rule:

```text
if run_status.execution_topology.base_sharing.resolved_mode == "fork_cow" and shared_base_state_active != true:
    FAIL("INCONSISTENT_BASE_SHARING_TRUTH")
```

### B. Benchmark Health

- `benchmark_summary.exit_code == 0`
- `benchmark_summary.failures == 0`
- `run_summary.failure_count == 0`
- `run_manifest.failure_count == 0`
- `run_manifest.aborted == false`
- `runtime_health_checks.jsonl` must exist and contain at least 1 row.
- `group_runtime_stats.jsonl` must exist and contain at least 1 row for the V2 run.
- `benchmark_summary.active_effective_ratio_stats.p95 >= 0.85`
- `benchmark_summary.queue_backlog_stats.max <= 4 * benchmark_summary.requested_workers * 32`
- `benchmark_summary.result_backlog_bytes_stats.max <= 256 * 1024 * 1024`

Fail conditions:

- any runtime artifact contract error
- any non-zero failure count
- any aborted run
- queue backlog above the hard bound
- result backlog bytes above the hard bound
- active/effective ratio below the hard bound

### C. Memory Evidence

- `benchmark_summary.parent_rss_mb_stats` and `benchmark_summary.worker_rss_mb_stats` must exist for both runs.
- If `benchmark_summary.uss_supported == true`, then `benchmark_summary.worker_uss_mb_stats` and `benchmark_summary.parent_uss_mb_stats` must both be present and used for interpretation before RSS.
- If `benchmark_summary.pss_supported == true`, then `benchmark_summary.worker_pss_mb_stats` and `benchmark_summary.parent_pss_mb_stats` must both be present and reviewed.
- `benchmark_summary.final_artifact_generation_sec` must be present and greater than or equal to `0.0`.
- `proof_run_parent_peak_limit_mb = 6144`

Exact larger-scale block:

```text
if parent_peak_memory_mb > proof_run_parent_peak_limit_mb:
    BLOCK_LARGER_SCALE("FINALIZATION_PARENT_MEMORY_TOO_HIGH")
```

### D. Telemetry Completeness

The V2 benchmark summary must expose all of the following fields:

- `execution_topology`
- `requested_workers`
- `effective_workers`
- `queue_backlog_stats`
- `result_backlog_bytes_stats`
- `module3_estimated_bytes_stats`
- `module3_realized_bytes_stats`
- `final_artifact_generation_sec`
- `parent_rss_mb_stats`
- `worker_rss_mb_stats`
- `uss_supported`
- `pss_supported`

The underlying runtime artifacts must expose:

- `run_status.process_start_method`
- `run_status.execution_topology.base_sharing.configured_mode`
- `run_status.execution_topology.base_sharing.resolved_mode`
- `run_status.execution_topology.base_sharing.shared_base_state_active`
- `run_status.execution_topology.base_sharing.fallback_only`

### E. Comparison Integrity

- Candidate and baseline runs must use the committed generated configs without manual edits.
- `linux_proof_run_stage1_comparison.json` must exist.
- `comparison.primary_driver` must not imply a memory win unless `comparison.memory_driver_metric` shows a negative worker-memory delta.
- If `comparison.memory_driver_metric == "worker_rss_mb"` and `comparison.worker_memory_delta_mb > 0`, the proof run may still pass as scheduling-driven, but it does not count as a memory improvement.

### F. Gate To Larger Server Trial

Larger server trial is allowed only if all of the following are true:

- all start-method truth gates pass
- all benchmark health gates pass
- telemetry completeness gates pass
- no new correctness regression appears in `run_status`, `run_manifest`, or `runtime_health_checks.jsonl`
- parent peak memory does not exceed `6144` MiB
- the comparison remains honest about the speedup driver

The larger server trial remains blocked if the parent peak memory exceeds `6144` MiB even when the proof run otherwise succeeds.

## Residual Risk Kept Explicit

- Final artifact generation still accumulates parent-memory state and is not redesigned by this stage.
- Linux proof-run success does not imply Linux production readiness.
