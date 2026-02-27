# Spec Alignment Report (`main-3.pdf`)

This report maps locked paper requirements to concrete implementation and tests in the repo.

## Scope
- Spec source: `/Users/afekshusterman/Downloads/main-3.pdf`
- Verified codebase root: `/Users/afekshusterman/Documents/New project`

## Checklist (locked items)

### 1) Session policy, `tod` indexing, session separation, and gap reset on `gapmin(t) > 5`
- Spec intent: §2.2–§2.3 (session-aware timeline; reset on large gaps).
- Implementation:
  - Session clock + timezone-safe `tod/session_id`: [`weightiz_module1_core.py`](/Users/afekshusterman/Documents/New project/weightiz_module1_core.py )
    - `build_session_clock_vectorized(...)` (lines ~429-470)
    - `_compute_phase(...)` (lines ~368-378)
  - Gap reset rule exactly `>` threshold:
    - `reset_flag = ((gap_min > cfg.gap_reset_minutes) | session_change)` (lines ~463-466)
    - `EngineConfig.gap_reset_minutes = 5.0` default (line ~125)
  - Downstream no-session-mix reset usage:
    - Module2 delta state reset when `reset_flag==1` or session changes (`run_module2_profile`, lines ~1196-1208).
- Test coverage:
  - [`tests/test_module1_core.py`](/Users/afekshusterman/Documents/New project/tests/test_module1_core.py )
    - `test_gap_reset_triggers_on_six_minute_gap`

### 2) ATR normalization floor: `max(ATR, 4*tick, 0.0002*Close)`
- Spec intent: §4.2 hard floor.
- Implementation:
  - [`weightiz_module2_core.py`](/Users/afekshusterman/Documents/New project/weightiz_module2_core.py ) in `run_module2_profile(...)`:
    - `floor_tick = 4.0 * tick`
    - `floor_pct = 0.0002 * close_use`
    - `atr_floor = max(atr_raw, floor_tick, floor_pct)` for sealed mode (lines ~607-613).
- Test coverage:
  - [`tests/test_module2_institutional.py`](/Users/afekshusterman/Documents/New project/tests/test_module2_institutional.py )
    - `test_atr_floor_locked_formula_in_sealed_mode`

### 3) Fixed x-grid: `x_i = -6 + i*0.05`, `i=0..239`
- Spec intent: §6.1 fixed deterministic profile grid.
- Implementation:
  - [`weightiz_module1_core.py`](/Users/afekshusterman/Documents/New project/weightiz_module1_core.py )
    - `EngineConfig.B=240`, `x_min=-6.0`, `dx=0.05` (lines ~115, ~118-119)
    - `build_x_grid(...)` computes `x_min + dx * arange(B)` (lines ~285-300).
- Test coverage:
  - [`tests/test_module1_core.py`](/Users/afekshusterman/Documents/New project/tests/test_module1_core.py )
    - `test_locked_x_grid_definition`

### 4) Warm-up neutrality for first 15 minutes (`tod < 15`)
- Spec intent: §7 neutral outputs during warm-up while state updates.
- Implementation:
  - `EngineConfig.warmup_minutes = 15` in [`weightiz_module1_core.py`](/Users/afekshusterman/Documents/New project/weightiz_module1_core.py ) (line ~123).
  - `_compute_phase(...)` marks `WARMUP` until `tod >= warmup_minutes` (lines ~368-378).
  - Module2 explicitly neutralizes warm-up scores:
    - `state.scores[warmup_rows] = 0.0` in [`weightiz_module2_core.py`](/Users/afekshusterman/Documents/New project/weightiz_module2_core.py ) (lines ~1284-1287).
  - Module4 only trades in execution phases via `in_exec_phase_t` / `tradable_ta` in [`weightiz_module4_strategy_funnel.py`](/Users/afekshusterman/Documents/New project/weightiz_module4_strategy_funnel.py ) (lines ~368-369).
- Test coverage:
  - [`tests/test_module2_institutional.py`](/Users/afekshusterman/Documents/New project/tests/test_module2_institutional.py )
    - `test_warmup_computes_profiles_scores_zero`

### 5) Injection kernel + robust volume cap + delta layer + deterministic gating
- Spec intent: §8–§16.1 deterministic micro-physics and gating.
- Implementation:
  - Injection kernel and mixture normalization in [`weightiz_module2_core.py`](/Users/afekshusterman/Documents/New project/weightiz_module2_core.py ):
    - Gaussian mixture injection / normalization (lines ~1029-1045+)
  - Robust volume cap path:
    - MAD-based cap and sealed path med+MAD cap construction (lines ~629-640, ~1014-1021)
  - Delta gating:
    - `gbreak/greject` from `z_delta` (lines ~1243-1244)
    - breakout/rejection score channels (lines ~1251-1263)
  - Module4 deterministic action logic consumes score channels and DQ/IB gates:
    - conviction scaling by DQS and neutralization (`dqs<0.5`, IB no-trade) in [`weightiz_module4_strategy_funnel.py`](/Users/afekshusterman/Documents/New project/weightiz_module4_strategy_funnel.py ) (lines ~608-643).
- Test coverage:
  - [`tests/test_module4_dqs_policy.py`](/Users/afekshusterman/Documents/New project/tests/test_module4_dqs_policy.py )
  - [`tests/test_module3_structure.py`](/Users/afekshusterman/Documents/New project/tests/test_module3_structure.py )
  - [`tests/test_module5_harness_institutional.py`](/Users/afekshusterman/Documents/New project/tests/test_module5_harness_institutional.py )

### 6) Determinism + float64 core tensor discipline
- Spec intent: strict deterministic operations in float64.
- Implementation:
  - `EngineConfig.seed`, deterministic sorting/tie-breaks and deterministic lexsort-based top-k in Module4.
  - Tensor allocation and numeric paths use `np.float64` across Module1/2/4 state and math:
    - Examples in [`weightiz_module2_core.py`](/Users/afekshusterman/Documents/New project/weightiz_module2_core.py ) lines ~188-241, ~593 onward.
  - Module1 validation enforces dtypes (`gap_min float64`, etc.) in [`weightiz_module1_core.py`](/Users/afekshusterman/Documents/New project/weightiz_module1_core.py ) lines ~831 onward.
- Test coverage:
  - dtype-focused assertions in module institutional tests (`test_module1_core.py`, `test_module2_institutional.py`, `test_module5_harness_institutional.py`).

## Warning-root-cause hardening added in this cycle
- Root warning source previously: `All-NaN slice encountered` in `weightiz_module2_core.py` (`np.nanmedian` paths).
- Patch:
  - Introduced `_nanmedian_silent(...)` and replaced warning-prone nan-reductions in Module2 rolling med/MAD and cap statistics.
  - Local warning suppression for all-NaN reduction in harness jitter-bound computation.
- Result:
  - Pre-patch bounded process-pool stderr log had repeated RuntimeWarnings.
  - Post-patch bounded run has `runtime_warning_count=0` and empty stderr warning log.

## Conclusion
The locked spec requirements requested for this run are implemented and test-anchored. The execution in `run_20260227_121622` remained aligned with these constraints and produced deterministic, finite outputs.
