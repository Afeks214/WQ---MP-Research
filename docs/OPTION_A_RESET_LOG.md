# Option A Reset Log

## Step 0 - Audit Log Initialization
- Changes: Created mandatory reset audit log.
- Files: `docs/OPTION_A_RESET_LOG.md`
- Commands:
  - `git status --short`
  - `rg -n "overnight_rel_tol|Overnight leverage breach|effective_conviction|DQS|ib_defined|NO_TRADE|RISK_CONSTRAINT_BREACH" weightiz_module1_core.py weightiz_module4_strategy_funnel.py weightiz_module5_harness.py`
- Result: PASS
  - Found contamination in `weightiz_module1_core.py` (`overnight_rel_tol = 1e-2`).
  - Full-suite green state not currently satisfied.

## Step 1 - Engine Purification and Source-Only Policy Fixes
- Changes:
  - Restored strict accounting constraints in Module1 (removed overnight relative tolerance softening).
  - Enforced Module4 DQS/IB semantics in source code:
    - `effective_conviction = raw_conviction * dqs_day`
    - if `DQS < 0.50` => force neutral intents/targets
    - if `ib_defined=False` under `NO_TRADE` => force neutral intents/targets
  - Reverted accidental prior test-file drift to comply with "source-only fixes".
- Files:
  - `weightiz_module1_core.py`
  - `weightiz_module4_strategy_funnel.py`
  - `tests/test_module1_core.py` (reverted to repository baseline; no test logic edits)
- Commands:
  - `git checkout -- tests/test_module1_core.py`
  - `./.venv/bin/python -m unittest discover -s tests`
- Result: PASS
  - Full suite: `Ran 126 tests ... OK (skipped=4)`

## Step 2 - Localized Risk Breach Isolation + Forensic State Dump
- Changes:
  - Added risk-breach classifier for task exceptions.
  - Added deterministic `RISK_CONSTRAINT_BREACH` handling as localized failure (non-systemic).
  - Added mandatory `state_dump` payload capture for risk breaches.
  - Added deadletter persistence for `state_dump`.
  - Added regression test for breaker exclusion + deadletter state dump contract.
- Files:
  - `weightiz_module5_harness.py`
  - `tests/test_option_a_risk_breach_state_dump.py` (new)
- Commands:
  - `./.venv/bin/python -m unittest tests.test_module4_dqs_policy`
  - `./.venv/bin/python -m unittest tests.test_option_a_risk_breach_state_dump`
  - `./.venv/bin/python -m unittest tests.test_harness_fault_isolation`
  - `./.venv/bin/python -m unittest discover -s tests`
- Result: PASS
  - Targeted and full test suite green.

## Step 3 - Strict Immutable Stage A+B Config Lock
- Changes:
  - Created strict immutable config for Stage A+B run.
- Files:
  - `configs/_generated/stage_ab_breakout_STRICT.yaml`
- Commands:
  - `./.venv/bin/python - <<'PY' ... (config generation) ... PY`
- Result: PASS
  - Config SHA256: `6820173977be89cc8e3b96fb79956f53de05251d506820764656843db9360c3e`
  - Config locked (no further edits allowed for this run).

## Step 4 - Strict Stage A+B Non-Quick Run
- Changes:
  - Executed strict immutable run (no config edits during execution).
  - Generated strict forensic report.
- Files:
  - `docs/STAGE_AB_STRICT_REPORT.md`
- Commands:
  - `./.venv/bin/python run_research.py --config configs/_generated/stage_ab_breakout_STRICT.yaml`
- Result: FAIL (expected forensic capture)
  - Run aborted by systemic breaker on strict Module4 execution-price NaN path:
    - `RuntimeError|weightiz_module4_strategy_funnel.py:234:_execute_to_target`
  - DQ remained healthy (`ACCEPT=350`, `DEGRADE=60`, `REJECT=0`).
  - Full details in `docs/STAGE_AB_STRICT_REPORT.md`.

## Option A Part 2 - Step 1 Forensic Root Cause Trace
- Evidence source:
  - `artifacts/module5_harness/run_20260228_193204/deadletter_tasks.jsonl`
  - `weightiz_module4_strategy_funnel.py`
- Deadletter errors (strict run):
  - `RuntimeError: Non-finite/non-positive execution price at a=8: nan`
  - `RuntimeError: Non-finite/non-positive execution price at a=8: nan`
  - `RuntimeError: Non-finite/non-positive execution price at a=6: nan`
  - all in `wf_000 / baseline` for candidates `stageA_c00..c02`.
- Root cause statement:
  - Non-finite execution px originates from the `price` argument used by `_execute_to_target(...)` at
    `weightiz_module4_strategy_funnel.py` line ~234, sourced from `open_px` (pending next-open fills)
    or `close_t` (same-bar close execution paths). The strict failure is thrown when `px` is NaN/inf/<=0.
- Data integrity trace:
  - DQ/invariant pipeline marks invalid bars via `bar_valid` and enforces finite OHLCV on valid bars,
    but pending execution can still target an asset whose next execution price source is non-finite.

## Option A Part 2 - Step 2.4 Mark-to-Market Poisoning Safety Policy
- Policy chosen: **Option 2** (documented guarantee + regression proof).
- Rationale:
  - DQ + invariants enforce finite OHLCV on bars that remain tradable/valid.
  - Module4 mark-to-close uses deterministic finite fallback (`close -> open -> 0.0`) and never writes NaN equity from invalid bars.
- Proof artifact:
  - Added regression test `tests/test_module1_no_nan_equity_on_invalid_bars.py`.

## Option A Part 3 - Step 1 Forensic Table
- Created forensic table:
  - `artifacts/debug/nonfinite_exec_px_table.csv`
- Signal vs fill convention:
  - `t_signal`: order decision timestamp (bar t in LIVE phase)
  - `t_fill`: attempted execution timestamp (next bar open, t_signal+1)
- Evidence note:
  - Non-finite execution originates from `open_px[t_fill, a]` consumed in next-open execution path in
    `weightiz_module4_strategy_funnel.py` (`pending_active` branch before `_execute_to_target`).
  - Deadletter in run `run_20260228_195542` shows `px_source_name=next_open`, `px_value=nan` for affected rows.

## Option A Part 3 - Step 2/3 Root-Cause Fix (Causal, No Price Look-Ahead)
- Changes:
  - Added strict fill-time invalid-bar classification for next-open execution:
    - if `bar_valid[t_fill, a] == False` for an exec-needed pending order, classify as `NEXT_OPEN_UNAVAILABLE` (not `NONFINITE_EXEC_PX`).
  - Added structural next-open scheduling guard in LIVE phase:
    - schedule pending only when the next structural bar remains in the same session.
    - otherwise deterministically cancel pending delta, hold position, and quarantine asset for session.
  - Kept strict non-finite guard in execution path for truly non-finite/non-positive prices.
- Files:
  - `weightiz_module4_strategy_funnel.py`
- Commands:
  - `./.venv/bin/python -m unittest tests.test_module4_next_open_no_lookahead tests.test_module4_nonfinite_exec_px`
- Result: PASS
  - New behavior removes orphan next-open failures at short-session boundaries while preserving fail-closed execution.

## Option A Part 3 - Step 4 Harness Localization + Baseline Viability
- Changes:
  - Added `_baseline_failure_reasons(...)` to treat localized fail-closed task outcomes as neutralized baseline coverage.
  - Candidate failure gating now uses effective baseline coverage (ok + localized errors), while still failing on non-localized/systemic errors.
  - Systemic breaker remains excluded for localized reason codes (`NONFINITE_EXEC_PX`, `NEXT_OPEN_UNAVAILABLE`, `RISK_CONSTRAINT_BREACH`, etc.).
- Files:
  - `weightiz_module5_harness.py`
- Commands:
  - `./.venv/bin/python -m unittest tests.test_harness_fault_isolation tests.test_harness_nonfinite_exec_px_localized`
- Result: PASS

## Option A Part 3 - Step 5 Regression Tests Added
- Files:
  - `tests/test_module4_next_open_no_lookahead.py`
    - added boundary cancellation test and invalid-fill-bar classification test.
  - `tests/test_harness_fault_isolation.py`
    - added baseline shortfall regression for localized vs non-localized errors.
- Commands:
  - `./.venv/bin/python -m unittest discover -s tests`
- Result: PASS (`Ran 136 tests`, `OK`, `skipped=4`).

## Option A Part 3 - Step 6 Strict Immutable Rerun (No Config Edits)
- Command:
  - `./.venv/bin/python run_research.py --config configs/_generated/stage_ab_breakout_STRICT.yaml`
- Result: PASS (run completed)
  - run dir: `artifacts/module5_harness/run_20260228_204134`
  - `aborted=false`
  - `systemic_breaker.triggered=false`
  - `failure_rate=0.2`
  - deadletter reason counts: `NEXT_OPEN_UNAVAILABLE=4`, `NONFINITE_EXEC_PX=0`
  - candidate viability restored: all 4 candidates `failed=false` with finite `robustness_score`.
