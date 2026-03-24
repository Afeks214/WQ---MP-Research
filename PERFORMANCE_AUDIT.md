# Performance Audit

## Scope
- Mode: PERFORMANCE_RECOVERY_MODE
- Objective: Recover runtime/memory contract without changing math semantics, reset/warmup semantics, lookahead rules, or public output identities.
- Code revision base: `96e4cbb` + local remediation/performance patches.
- Authoritative runtime path preserved:
  - `src/weightiz/cli/run_module5.py`
  - `src/weightiz/cli/run_research.py`
  - `src/weightiz/module5/orchestrator.py`
  - `src/weightiz/module2/core.py`
  - `src/weightiz/module2/market_profile_engine.py`
  - `src/weightiz/module2/market_profile_kernels.py`
  - `src/weightiz/module2/tensor_builder.py`

## Benchmark Scenario (Frozen)
- Sealed mode
- `T=390`, `A=4`, `B=240`, `W=60`
- Deterministic synthetic OHLCV fixture (fixed RNG seed)
- Warm once, then 4 measured runs
- End-to-end call: `run_weightiz_profile_engine`
- Measurement:
  - Runtime: average seconds per run
  - Memory: `tracemalloc` peak bytes across measured loop

## Baseline vs Final
- Baseline (HEAD pre-recovery):
  - `avg_sec=0.3234`
  - `peak_bytes=13,347,672`
- Final (post-recovery):
  - `avg_sec=0.4792`
  - `peak_bytes=14,789,462`
- Relative:
  - Runtime ratio: `1.4818x` (`+48.18%`)
  - Peak memory ratio: `1.1080x` (`+10.80%`)

## Contract Evaluation
- Runtime gate: PASS (`1.4818x < 2.0x`)
- Memory gate: PASS (`+10.80% < +20%`)
- Correctness gate: PASS
  - `tests/test_spec_alignment.py`: `39 passed`
  - canonical/architecture/startup guards: `3 passed`
  - module2 guard suites: `25 passed`
- Determinism gate: PASS (no fastmath/parallel/prange, float64 preserved)
- Repo safety gate: PASS (single-engine canonical path preserved; no push)

## Hotspot Forensics Summary
Initial dominant hotspots (current remediated-correct state, before recovery patches):
1. `run_streaming_profile_engine` self time in sealed branch
2. `numpy.nanmedian` stack via `_rolling_median_mad_causal` and sealed cap recomputation
3. `compute_value_area_greedy` bounded loop

Key diagnosis:
- Main bottleneck was implementation waste (NaN/masked-array median machinery), not paper-required arithmetic alone.
- Secondary memory blowup came from persistent `vp_buy`/`vp_sell` tensors plus sealed-path temporaries.

## Optimization Decision Record
Options evaluated:
- Option A (NumPy cleanup + buffer/flow optimization, no JIT): chosen
- Option B (cleanup + targeted Numba): not needed after Option A met contract
- Option C (broader Numba): rejected due complexity/risk and unnecessary once gate satisfied

Rationale:
- Option A achieved both runtime and memory gates with minimal invasive changes and zero semantic drift.
- JIT was avoided because required gates were met without altering compilation model.

## Accepted Optimizations
1. Replaced sealed cap window `nanmedian` path with explicit fast median/MAD helper.
2. Replaced NaN-shifted past-only delta-noise input with shifted rolling outputs (equivalent semantics, lower overhead).
3. Replaced persistent `vp_buy`/`vp_sell` storage with derived lazy public properties from `vp`/`vp_delta`.

## Rejected Optimizations
- Numba conversion (`njit`) for hot kernels: rejected as unnecessary after structural recovery.

## Residual Risks
- Lazy `vp_buy`/`vp_sell` properties allocate temporaries when accessed frequently by downstream consumers. Current test and runtime workloads do not show gate breaches, but heavy repeated consumer-side access should be monitored.
- Sealed reprojection remains computationally heavier than pre-remediation behavior by design; current implementation is inside contract but should retain benchmark checks in CI.
