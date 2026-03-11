# Sweep Code Map (Repo Truth, Code-Anchored)

## 1) Config Contract + Strict Validation
- Loader: `weightiz.cli.run_research::_load_config`
- Pydantic root model: `weightiz.cli.run_research::RunConfigModel` (`extra="forbid"`)
- Strict component models:
  - `weightiz.cli.run_research::Module2ConfigModel` (`extra="forbid"`)
  - `weightiz.cli.run_research::Module3ConfigModel` (`extra="forbid"`)
  - `weightiz.cli.run_research::Module4ConfigModel` (`extra="forbid"`)
  - `weightiz.cli.run_research::HarnessConfigModel` (`extra="forbid"`)

### W/T/A/B schema keys (used for family sweeps)
- `W` -> `module2_configs[*].profile_window_bars`
  - Anchor: `weightiz.cli.run_research` (`Module2ConfigModel.profile_window_bars`)
- `T` -> `module4_configs[*].entry_threshold`
  - Anchor: `weightiz.cli.run_research` (`Module4ConfigModel.entry_threshold`)
- `A` -> `module4_configs[*].trend_poc_drift_min_abs`
  - Anchor: `weightiz.cli.run_research` (`Module4ConfigModel.trend_poc_drift_min_abs`)
- `B` -> `module4_configs[*].neutral_poc_drift_max_abs`
  - Anchor: `weightiz.cli.run_research` (`Module4ConfigModel.neutral_poc_drift_max_abs`)

### Cheap scaling axis confirmed
- `module4_configs[*].top_k_intraday`
  - Schema anchor: `weightiz.cli.run_research` (`Module4ConfigModel.top_k_intraday`)
  - Strategy usage anchor: `weightiz_module4_strategy_funnel.py` (Top-K selection and sizing block)

## 2) Session Clock, Gap Reset, Warmup Neutralization

### Gap reset (5-minute rule)
- Clock build: `weightiz_module1_core.py::build_session_clock_vectorized`
- Rule:
  - `gap_min[1:] = (ts_ns[1:] - ts_ns[:-1]) / NS_PER_MIN`
  - `reset_flag = ((gap_min > cfg.gap_reset_minutes) | session_change).astype(int8)`
  - `reset_flag[0] = 1`

### Warmup phase (time-based)
- Phase classification: `weightiz_module1_core.py::_compute_phase`
  - WARMUP/LIVE/OVERNIGHT_SELECT/FLATTEN from `tod`, `warmup_minutes`, `flat_time_minute`
- Warmup neutral scores:
  - `weightiz_module2_core.py` sets `state.scores[warmup_rows] = 0.0`
  - `warmup_rows = state.phase == Phase.WARMUP`

## 3) Module 2 Value-Area/POC
- VA algorithm: `weightiz_module2_core.py::compute_value_area_greedy`
  - Start from POC bin and expand left/right by higher adjacent mass
  - Deterministic tie-break: smaller `|x|`, then left
- Runtime call site: `weightiz_module2_core.py` near `compute_value_area_greedy(...)`
  - `va_threshold` is passed directly from `Module2Config`

## 4) Candidate Expansion + Cost Model

### Candidate enumeration
- Default auto-grid enumeration:
  - `weightiz_module5_harness.py::_build_candidate_specs_default`
  - Nested loops over `m2_idx x m3_idx x m4_idx`

### Grouping key (expensive recompute boundary)
- Group task build:
  - `weightiz_module5_harness.py::_build_group_tasks`
  - Group key: `(split_idx, scenario_idx, m2_idx, m3_idx)`
- Implication:
  - Increasing M2/M3 increases expensive group count (profile + structural recompute)
  - Increasing M4 increases candidates inside group (cheaper relative to M2/M3 expansion)

## 5) Strategy Logic Anchors (Module 4)
- Regime + intent gating:
  - `weightiz_module4_strategy_funnel.py` block around trend/neutral/shape definitions
  - Entry conditions use `entry_threshold`, `trend_poc_drift_min_abs`, `neutral_poc_drift_max_abs`
- Intraday top-k sizing:
  - `weightiz_module4_strategy_funnel.py` block around `k = min(max(top_k_intraday,...))`

## 6) Metrics + Output Artifacts
- Candidate metrics computation and write:
  - `weightiz_module5_harness.py` candidate metrics block (`candidate_metrics.json`)
- Leaderboards:
  - `weightiz_module5_harness.py` writes `leaderboard.csv` and `robustness_leaderboard.csv`
- Run-level metadata:
  - `weightiz_module5_harness.py` writes `run_manifest.json` and `run_status.json`

## 7) Family Mode Packaging (CLI-compatible extension)
- Trigger: `weightiz.cli.run_research::_family_mode_enabled` for `run_name.startswith("sweep_family_")`
- Family outputs in `artifacts/<run_name>/`:
  - `pid`, `run.log`, `results.parquet`, `summary.json`, `audit_bundle/*`
- Deterministic jitter:
  - `weightiz.cli.run_research::_deterministic_jitter_seconds`
  - `jitter = 10 + (sha256(run_name + seed) % 21)`
- Strict worker cap:
  - `weightiz.cli.run_research::main` raises fatal error if `harness.parallel_workers > 14`
