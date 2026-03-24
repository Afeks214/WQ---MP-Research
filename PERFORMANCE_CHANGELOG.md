# Performance Changelog

## PERF-001
- Change: Replaced sealed decision-time cap recomputation `np.nanmedian`/masked-array path with `_window_median_mad_volume` fast path in `src/weightiz/module2/market_profile_engine.py`.
- Why: cProfile showed repeated `np.nanmedian` in sealed loop as a dominant hotspot (~0.40s cumulative in one benchmarked run).
- Correctness impact: None observed.
- Validation: `tests/test_spec_alignment.py` full green; canonical/architecture guards green; module2 guard suites green.
- Measured effect: Runtime improved but remained above contract; memory unchanged materially.
- Decision: Accepted as intermediate optimization.

## PERF-002
- Change: Reworked delta-noise past-only computation to shift rolling MAD outputs instead of injecting NaN-shifted input windows in `src/weightiz/module2/market_profile_engine.py`.
- Why: Reduce NaN-heavy rolling median overhead while preserving exact past-only semantics.
- Correctness impact: None observed after test adjustment to equivalent observable assertion in reset spy case.
- Validation: `tests/test_spec_alignment.py` full green; canonical/architecture guards green; module2 guard suites green.
- Measured effect: Runtime improved significantly; memory still above contract due persistent tensor footprint.
- Decision: Accepted.

## PERF-003
- Change: Converted `vp_buy` and `vp_sell` from persistent preallocated tensors to derived lazy public properties on `TensorState` in `src/weightiz/module1/core.py`; removed redundant mutable/finiteness/writeability handling for non-persistent channels in `src/weightiz/module2/core.py` and `src/weightiz/module2/market_profile_engine.py`.
- Why: Persistent `(T,A,B)` buy/sell tensors dominated peak memory budget; API can remain soft-expansion compatible with additive derived outputs.
- Correctness impact: None observed; output identities `VP_buy+VP_sell=VP` and `VP_buy-VP_sell=VP_delta` remain true by definition and verified by tests.
- Validation: `tests/test_spec_alignment.py` full green; canonical/architecture guards green; module2 guard suites green.
- Measured effect: Peak memory returned inside contract; runtime remained inside contract.
- Decision: Accepted.

## PERF-REJECTED-001
- Candidate: Numba conversion of sealed reprojection kernel.
- Why rejected now: Contract recovered with lower-risk NumPy/structure cleanup; no JIT needed.
- Decision: Rejected.
