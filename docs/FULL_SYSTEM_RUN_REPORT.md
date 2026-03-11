# Full System Run Report (Spec-Faithful, Bounded, Non-Quick)

## 1) Run identifiers
- Config used: [`configs/_generated/spec_mp_breakout_rejection_eval_small.yaml`](/Users/afekshusterman/Documents/New project/configs/_generated/spec_mp_breakout_rejection_eval_small.yaml)
- Harness run dir: [`artifacts/module5_harness/run_20260227_121622`](/Users/afekshusterman/Documents/New project/artifacts/module5_harness/run_20260227_121622)
- Sweep-v2 manifest: not used for this final run (direct `weightiz.cli.run_research` execution).
- Command:
```bash
./.venv/bin/python -m weightiz.cli.run_research --config ./configs/_generated/spec_mp_breakout_rejection_eval_small.yaml
```

## 2) DQ summary
Source: [`dq_report.csv`](/Users/afekshusterman/Documents/New project/artifacts/module5_harness/run_20260227_121622/dq_report.csv)
- ACCEPT: 556
- DEGRADE: 84
- REJECT: 0
- DQS min / median: 0.75 / 1.0

## 3) Execution summary
Source: [`run_status.json`](/Users/afekshusterman/Documents/New project/artifacts/module5_harness/run_20260227_121622/run_status.json), [`run_manifest.json`](/Users/afekshusterman/Documents/New project/artifacts/module5_harness/run_20260227_121622/run_manifest.json)
- execution_mode: `process_pool`
- process_start_method: `fork`
- tasks_submitted/completed: `36 / 36`
- groups_completed: `36`
- failure_rate: `0.0`
- systemic breaker fired: `false`
- abort_reason: `""` (none)
- runtime_warning_count: `0`
- deadletter file: configured path exists in manifest, deadletter count = `0`

## 4) Strategy evaluation summary
Source: [`robustness_leaderboard.csv`](/Users/afekshusterman/Documents/New project/artifacts/module5_harness/run_20260227_121622/robustness_leaderboard.csv)
- candidates evaluated: 4
- failed candidates: 0
- pass candidates: 1
- finite robustness scores: 4/4

Top candidates (available rows = 4):

| candidate_id | robustness_score | pass | failed | dq_median |
|---|---:|---:|---:|---:|
| mp_brk_rej_c01 | -0.144755 | False | False | 1.0 |
| mp_brk_rej_c02 | -0.146433 | False | False | 1.0 |
| mp_brk_rej_c03 | -0.146711 | True | False | 1.0 |
| mp_brk_rej_c00 | -0.162102 | False | False | 1.0 |

## 5) Deep dive: top passing candidate
Candidate selected for deep dive: `mp_brk_rej_c03`
- File: [`candidates/mp_brk_rej_c03/candidate_metrics.json`](/Users/afekshusterman/Documents/New project/artifacts/module5_harness/run_20260227_121622/candidates/mp_brk_rej_c03/candidate_metrics.json)

Key fields:
- `failed`: `false`
- `failure_reasons`: `[]`
- `dq_summary`:
  - `dq_min`: `0.75`
  - `dq_median`: `1.0`
  - `dq_degrade_count`: `9`
  - `dq_reject_count`: `0`
  - `dq_reason_top`: `DQ_DEGRADED_INPUT`
- `per_stress.baseline`:
  - `n_tasks`: `9`
  - `cum_return_median`: `-0.013324322566288571`
  - `max_drawdown_median`: `0.0006734367546343822`
  - `turnover_median`: `46.329828202748836`
- `per_fold.wf.summary`:
  - `count`: `9`
  - `sharpe_daily_median`: `-0.4601887734994137`
  - `cum_return_median`: `-0.013324322566288571`

## 6) Monte Carlo / MCS / stats path confirmation
Source: [`stats_raw.json`](/Users/afekshusterman/Documents/New project/artifacts/module5_harness/run_20260227_121622/stats_raw.json)
- Present keys include: `dsr`, `pbo`, `wrc`, `spa`, `mcs`, `leaderboard`.
- `mcs` object exists with survivor/elimination/p-value structures.
- This confirms the statistical validation stack executed in this run.

## 7) Spec alignment confirmation
This run used sealed deterministic settings and respects the locked paper constraints documented in:
- [`docs/SPEC_ALIGNMENT_REPORT.md`](/Users/afekshusterman/Documents/New project/docs/SPEC_ALIGNMENT_REPORT.md)

Critical points preserved during run:
- Session-aware clock, gap reset on `gapmin > 5`
- ATR floor formula in sealed mode
- Fixed x-grid `[-6, 6)` at `dx=0.05`
- Warm-up neutrality
- Deterministic DQ/DQS + IB no-trade + Module4 neutralization gates

## 8) What worked / what failed / why
Worked:
- Full pipeline executed end-to-end (`DQ -> M2 -> M3 -> invariants -> M4 -> M5 stats`).
- Process-pool run completed with no deadletters and zero runtime warnings.
- All candidates produced finite robustness values.

Failed/limitations:
- Strategy performance is weak in this bounded window (all robustness scores negative), though this is a valid computed outcome, not a system failure.

Why:
- Baseline fold returns in this short 2024Q4 bounded sample are mostly negative; robustness penalties (PBO, concentration, variance terms) dominate.

## 9) Scaling notes
- Recommended worker count for current machine: `4` (validated stable in process pool).
- Remaining bottleneck: Module2 compute time dominates wall-clock; chunked group execution improves observability but not underlying per-task compute cost.

Readiness status:
- `wiring_ok`: **TRUE** (all required artifacts + completed execution)
- `evaluation_ready`: **TRUE** (at least one non-failed candidate and finite robustness scores)
