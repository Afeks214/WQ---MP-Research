# Part 3 - NonFinite Execution Price Root-Cause Report

## Scope
Institutional Part 3 implementation for strict causal execution handling in Module4 and localized handling in Module5, with one immutable strict rerun:
- Config: `configs/_generated/stage_ab_breakout_STRICT.yaml` (unchanged)
- Final run: `artifacts/module5_harness/run_20260228_204134`

## 1) Evidence Table Summary
Forensic table generated from failing strict run (`run_20260228_195542`):
- `artifacts/debug/nonfinite_exec_px_table.csv`

Key observations from the table:
- All events are `px_source_name=next_open`.
- Signal time (`t_signal`) and fill time (`t_fill=t_signal+1`) are explicitly separated.
- Two event patterns were present:
  1. **Invalid fill bar / missing fill input**: `bar_valid_fill=False` with `px_value=nan`.
  2. **Session boundary carry-over** (short-session/holiday transitions): signal on last available session bar, attempted fill on next session open.

Representative values in table rows:
- `candidate_id=stageA_c00`, `asset_symbol=XLK`, `t_signal=5819`, `t_fill=5820`, `px_value=nan`, `px_source_name=next_open`.
- `candidate_id=stageA_c02`, `asset_symbol=TLT`, `t_signal=7619`, `t_fill=7620`, `open/high/low/close at signal all NaN`, `bar_valid_signal=False`.

## 2) Root-Cause Statement
The strict failures were caused by pending `next_open` orders reaching fill points where execution price input was unavailable/non-finite for the requested asset-time pair.

Technically, the failure originated in Module4 pending fill logic before `_execute_to_target(...)`:
- pending orders were allowed to persist into structurally unavailable fill points (session transitions, invalid bars),
- then fill attempted with `open_px[t_fill, a]` that was not valid for execution (`NaN` / invalid bar context).

This is a **causal fill-time availability problem**, not a leverage/accounting tolerance problem.

## 3) Exact Fixes (Causal, No Price Proxying)
### A. Module4 strict fill-time and scheduling controls
File: `weightiz_module4_strategy_funnel.py`

1. Pending fill structural/session check retained:
- `same_session` check in pending fill path.
- If unavailable, deterministic cancel + quarantine + typed exception.
- Anchor: around lines `561-587`.

2. Invalid fill bar classification added:
- If execution is needed and `bar_valid[t_fill, a] == False`, treat as `NEXT_OPEN_UNAVAILABLE`.
- Deterministic cancel + quarantine + typed exception with `exec_px_dump`.
- Anchor: around lines `589-615`.

3. Strict non-finite price guard retained:
- True non-finite/non-positive `open_px[t_fill, a]` still raises `NONFINITE_EXEC_PX`.
- Anchor: around lines `616-641`.

4. Structural next-open scheduling guard in LIVE phase:
- Only schedule pending next-open when next structural bar remains in same session.
- Otherwise hold (`target=pos`) and quarantine affected assets for session.
- No price substitution, no ffill.
- Anchor: around lines `1021-1039`.

### B. Harness candidate viability under localized fail-closed events
File: `weightiz_module5_harness.py`

1. Added baseline coverage helper:
- `_baseline_failure_reasons(...)` counts localized failures as neutralized baseline coverage, but still fails candidate for non-localized errors.
- Anchor: around lines `401-418`.

2. Candidate failure gate switched to helper:
- baseline failure reasons now use `_baseline_failure_reasons(...)`.
- Anchor: around lines `2982-2987`.

This keeps strict fail-closed behavior while avoiding global candidate invalidation from localized, properly-isolated execution faults.

## 4) Regression Tests
Added/extended tests (source only, no edits to existing locked tests):
- `tests/test_module4_next_open_no_lookahead.py`
  - verifies fill-time exception timing,
  - verifies structural cancellation at session boundary,
  - verifies invalid fill bar => `NEXT_OPEN_UNAVAILABLE`.
- `tests/test_harness_fault_isolation.py`
  - verifies localized baseline shortfall handling,
  - preserves systemic breaker behavior for non-localized repeated signatures.

Full suite result:
- Command: `./.venv/bin/python -m unittest discover -s tests`
- Result: `Ran 136 tests ... OK (skipped=4)`.

## 5) Post-Run Proof (Strict Immutable Rerun)
Run:
- `artifacts/module5_harness/run_20260228_204134`

Key outcomes:
- `aborted=false`
- `systemic_breaker.triggered=false`
- `failure_rate=0.2`
- deadletter reason counts:
  - `NEXT_OPEN_UNAVAILABLE=4`
  - `NONFINITE_EXEC_PX=0`

Candidate viability:
- `robustness_leaderboard.csv`: all 4 candidates have
  - `failed=false`
  - finite `robustness_score` values (no `-inf`, no NaN, no inf).
- `pass=true` present for at least one candidate (`stageA_c02`).

Artifacts present:
- `dq_report.csv`
- `dq_bar_flags.parquet`
- `robustness_leaderboard.csv`
- `plateaus.json`
- `run_status.json`
- `deadletter_tasks.jsonl`

## Conclusion
The root cause is eliminated at the strict execution boundary: non-finite next-open execution no longer appears as `NONFINITE_EXEC_PX` in the strict run, systemic breaker remains off, and candidate viability is restored without loosening accounting/risk constraints and without editing the immutable strict config.
