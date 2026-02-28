# Stage A+B Strict Run Report

## A) Test Status Before Run (proof)
- Command executed before strict run:
  - `./.venv/bin/python -m unittest discover -s tests`
- Result summary:
  - `Ran 127 tests in 11.981s`
  - `OK (skipped=4)`

## B) Run Identifier
- Config used (immutable):
  - `/Users/afekshusterman/Documents/New project/configs/_generated/stage_ab_breakout_STRICT.yaml`
- Config SHA256:
  - `6820173977be89cc8e3b96fb79956f53de05251d506820764656843db9360c3e`
- Harness run directory:
  - `/Users/afekshusterman/Documents/New project/artifacts/module5_harness/run_20260228_193204`

## C) DQ Summary
- Source: `dq_report.csv`
- Decision counts:
  - `ACCEPT=350`
  - `DEGRADE=60`
  - `REJECT=0`
- DQS stats (`dqs_final`):
  - `min=0.75`
  - `median=1.0`
  - `max=1.0`

## D) Execution Summary
- Source: `run_status.json`
- `aborted=true`
- `failure_rate=1.0`
- `runtime_warning_count=0`
- `systemic_breaker.triggered=true`
- `abort_reason=systemic_exception signature=RuntimeError|weightiz_module4_strategy_funnel.py:234:_execute_to_target units=3 assets=10 candidates=3 suspicion=standard`

## E) Leaderboard Summary
- Source: `robustness_leaderboard.csv`
- Candidate count: `4`
- Failed count: `4`
- Non-failed count: `0`
- Finite robustness value count: `4` (all are `-inf`)

Top rows:

| candidate_id | robustness_score | pass | failed | dq_median | dq_min | failure_reasons |
|---|---:|---:|---:|---:|---:|---|
| stageA_c00 | -inf | False | True | 0.0 | 0.0 | baseline_ok_tasks=0 expected=5\|wf_000:RuntimeError |
| stageA_c01 | -inf | False | True | 0.0 | 0.0 | baseline_ok_tasks=0 expected=5\|wf_000:RuntimeError |
| stageA_c02 | -inf | False | True | 0.0 | 0.0 | baseline_ok_tasks=0 expected=5\|wf_000:RuntimeError |
| stageA_c03 | -inf | False | True | 1.0 | 1.0 | baseline_ok_tasks=0 expected=5 |

## F) RISK_CONSTRAINT_BREACH Forensics
- Deadletter source: `deadletter_tasks.jsonl`
- Observed deadletter rows: `3`
- Observed reason codes: empty for these rows.
- Error type in first rows:
  - `RuntimeError: Non-finite/non-positive execution price at a=8: nan`
  - `RuntimeError: Non-finite/non-positive execution price at a=8: nan`
  - `RuntimeError: Non-finite/non-positive execution price at a=6: nan`

Interpretation:
- This strict run did **not** fail on `RISK_CONSTRAINT_BREACH`.
- It failed earlier on strict Module4 execution-price validity (`NaN` at execution point), so no `STATE_DUMP` risk payloads were expected in this run.
- The `RISK_CONSTRAINT_BREACH` isolation + `STATE_DUMP` path is validated by regression test:
  - `tests/test_option_a_risk_breach_state_dump.py`.

## G) Breakout Firing Confirmation
- `trade_log.parquet` exists but has `0` rows.
- `micro_diagnostics.parquet` exists but has `0` rows.
- Conclusion:
  - Breakout entries did not fire in this strict run because candidate tasks failed at early execution due strict non-finite execution price checks.

## Final Assessment
- The Option A reset goals for strict accounting and test integrity are met.
- The strict Stage A+B run is currently blocked by a deterministic data/feature-to-execution NaN path in Module4 (`_execute_to_target`).
- Next engineering action should target root-cause elimination of non-finite execution prices while preserving strict fail-closed behavior.
