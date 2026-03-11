# ARCHITECTURE AUDIT REPORT

## Canonical Pipeline Status

Canonical path enforced:

`CLI -> run_research.py -> Module5 harness -> Module2 (master only) -> Module3 -> Module4 signal funnel -> risk_engine -> metrics/ledger`

## Architecture Map (Static Call Graph Summary)

- Entrypoint: `main()` in `run_research.py:772`
  - `run_architecture_consistency_check()` at `run_research.py:787`
  - `run_preflight_validation_suite(...)` at `run_research.py:789`
  - `run_weightiz_harness(...)` at `run_research.py:814`
- Harness orchestration: `run_weightiz_harness(...)` in `weightiz_module5_harness.py:3877`
  - Module2 master run: `run_weightiz_profile_engine(...)` at `weightiz_module5_harness.py:3912`
  - Shared-memory create (master): `create_shared_feature_store(...)` at `weightiz_module5_harness.py:3986`
  - Worker shared-memory attach: `attach_shared_feature_store(...)` at `weightiz_module5_harness.py:734`
  - Module3 worker call: `run_module3_structural_aggregation(...)` at `weightiz_module5_harness.py:2499`
  - Module4 signal-only worker call: `run_module4_signal_funnel(...)` at `weightiz_module5_harness.py:2542`
  - Execution authority: `simulate_portfolio_from_signals(...)` at `weightiz_module5_harness.py:2566` and `:2586`

## Runtime Entry Points Analysis

- `weightiz.cli.run_research` is active research entrypoint.
- `weightiz.module5.worker_io_guard` is fatal stub only:
  - `run_zimtra_sweep(...)` raises `PARALLEL_ENGINE_FORBIDDEN`.

## Detected Violations (This Pass)

1. Module4 execution API remained callable in `weightiz_module4_strategy_funnel.py` (`run_module4_strategy_funnel`).
2. Missing consolidated architecture gate test file `tests/test_architecture_pipeline.py`.
3. Missing architecture audit report artifact in repo root.

## Fixes Applied

1. Enforced Module4 execution ban in canonical system:
   - Updated `weightiz_module4_strategy_funnel.py:455`
   - `run_module4_strategy_funnel(...)` now raises:
     - `MODULE4_EXECUTION_FORBIDDEN_IN_CANONICAL_PATH`
2. Tightened architecture guard:
   - Updated `weightiz_architecture_guard.py`
   - Added `_assert_module4_signal_only(...)`
   - `run_architecture_consistency_check()` now validates Module4 execution API is forbidden in canonical path.
3. Added architecture enforcement tests:
   - New `tests/test_architecture_pipeline.py`
   - Verifies:
     - single runtime dispatch + sweep fatal stub
     - worker pipeline uses SHM + Module3 + Module4 signal funnel + risk_engine
     - Module2 worker guard exists
     - Module4 execution API forbidden
     - runtime modules are print-free
4. Added this report:
   - `ARCHITECTURE_AUDIT_REPORT.md`

## Guards Inserted / Verified

- `weightiz.cli.run_research` startup gate order includes:
  - architecture check
  - preflight validation
  - canonical harness dispatch
- Worker guard:
  - `worker_feature_source == "shared_memory"`
- Module2 worker execution ban:
  - `MODULE2_WORKER_EXECUTION_FORBIDDEN`
- Shared memory ownership model:
  - master creates/unlinks
  - workers attach read-only
- Float64 guard utility:
  - `assert_float64(...)` in `weightiz_dtype_guard.py`

## Validation Executed

Command:

`PYTHONPATH=. .venv/bin/pytest -q tests/test_architecture_pipeline.py tests/test_architecture_consistency_guard.py tests/test_canonical_single_path.py tests/test_worker_no_profile_recompute.py tests/test_risk_engine_execution_authority.py tests/test_no_runtime_prints.py tests/test_module2_worker_forbidden.py tests/test_startup_gate_order.py`

Result:

- `12 passed`

## Remaining Risks

1. `weightiz_module4_strategy_funnel.py` still contains legacy execution implementation body after the early fatal guard; it is now non-executable via canonical runtime but still present as dead legacy code.
2. Repository contains additional non-architecture modified/untracked files outside this pass; they were not altered by this enforcement commit.
