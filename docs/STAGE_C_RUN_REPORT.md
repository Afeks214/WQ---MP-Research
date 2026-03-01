# Stage C Run Report

## A) Pre-flight Proof

- Unit tests command: `./.venv/bin/python -m unittest discover -s tests`
- Result: `Ran 136 tests ... OK (skipped=4)`
- Git status (summary at launch): modified `weightiz_module5_stats.py`, untracked operational docs/config scripts present.

## B) Data Window & Universe

- Source: [/Users/afekshusterman/Documents/New project/docs/STAGE_C_DATA_WINDOW.md](/Users/afekshusterman/Documents/New project/docs/STAGE_C_DATA_WINDOW.md)
- Final symbols: `EEM, GLD, HYG, IWM, QQQ, SPY, TLT, XLE, XLK, XLU`
- Dropped symbols: `none`
- target_start: `2024-01-02T14:31:00Z`
- target_end: `2024-12-31T21:00:00Z`
- window_days: `364`
- CPCV mode: `enabled`

## C) Run Status

- run_dir: `/Users/afekshusterman/Documents/New project/artifacts/module5_harness/run_20260228_231125`
- run_id: `run_20260228_231125`
- phase: `running`
- aborted: `False`
- abort_reason: ``
- failure_rate: `n/a`
- systemic_breaker.triggered: `False`
- runtime_warning_count: `n/a`
- tasks_done/tasks_total: `6/1980`

## D) Deadletter & Scale Rate

- deadletter file: `/Users/afekshusterman/Documents/New project/artifacts/module5_harness/run_20260228_231125/deadletter_tasks.jsonl`
- total_deadletters: `6`
- reason_counts: `{"NEXT_OPEN_UNAVAILABLE": 6}`
- top_symbols: `{"HYG": 6}`
- class_A_count: `6`
- class_B_count: `0`
- tasks_completed (for KPI formula): `6`
- trading_sessions_available: `252`
- approx_total_bars_evaluated: `5896800`
- deadletters_per_1M_bars: `1.0175010175010175`

## E) Leaderboard + Export

- robustness_leaderboard.csv present: `False`
- candidate_count: `0`
- finite_robustness_score_count: `0`
- top10 export: `/Users/afekshusterman/Documents/New project/artifacts/debug/stage_c_top10.csv`
- Top10 table unavailable because run did not reach leaderboard finalization before bounded stop.

## F) Decision

- decision: `PROCEED_WITH_MONITORING`
- Rule path: no Class B and no breaker; evaluated by deadletters_per_1M_bars + finite candidate count.

## Notes

- This report reflects a single Stage C execution started once and then operationally bounded due projected multi-hour runtime (`1980` tasks).
- Because the run did not finalize, end-of-run artifacts such as `robustness_leaderboard.csv` were not emitted in this run directory.