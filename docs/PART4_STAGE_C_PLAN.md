# PART4_STAGE_C_PLAN (PLANNING ONLY)

## 1) Expansion matrix
- Scenarios: baseline + mild + severe.
- Window: expand to 6-12 months.
- Candidates: deterministic grid expansion from 4 to 30.
- Splits: WF retained; CPCV added if supported.

## 2) Acceptance gate
- Stage C execution starts only if CLASS_B count == 0 in Part 4 policy decision.

## 3) KPI monitoring checklist
- Watch `run_status.json` for `aborted`, `systemic_breaker.triggered`, `failure_rate`.
- Watch `dq_report.csv` for DQ mix and DQS quantiles.
- Watch `deadletter_tasks.jsonl` for reason-code drift and Class B emergence.
- Watch `robustness_leaderboard.csv` for finite-score coverage and failed/pass mix.

## 4) Stop conditions
- Stop on systemic breaker trigger.
- Stop if CLASS_B_DANGEROUS_MISMATCH > 0.
- Stop if deadletters per 1M bars breaches guardrail persistently.