# PART4 Git Pre-Flight Report

## Scope
Read-only pre-flight only.  
No commit/push/merge/rebase/fetch/pull executed.

## Repo Snapshot
- pwd: `/Users/afekshusterman/Documents/New project`
- git rev-parse --show-toplevel: `/Users/afekshusterman/Documents/New project`
- branch: `main`
- HEAD: `880034c0fda396b9a9af9d8909e342dfccabddf6`
- remotes:
  - `origin https://github.com/Afeks214/WQ---MP-Research.git (fetch)`
  - `origin https://github.com/Afeks214/WQ---MP-Research.git (push)`

## Git Status (porcelain v1)
### Modified
- tests/test_harness_fault_isolation.py
- weightiz_module1_core.py
- weightiz_module4_strategy_funnel.py
- weightiz_module5_harness.py
- weightiz_module5_stats.py

### Untracked
- configs/_generated/stage_ab_breakout_STRICT.yaml
- configs/_generated/stage_ab_family1_breakout_v1.yaml
- docs/OPTION_A_RESET_LOG.md
- docs/PART3_NONFINITE_ROOT_CAUSE_REPORT.md
- docs/PART4_DEADLETTER_POLICY_DECISION.md
- docs/PART4_SCALE_GATE_KPIS.md
- docs/PART4_STAGE_C_PLAN.md
- docs/STAGE_AB_STRICT_REPORT.md
- tests/test_harness_nonfinite_exec_px_localized.py
- tests/test_module1_no_nan_equity_on_invalid_bars.py
- tests/test_module4_next_open_no_lookahead.py
- tests/test_module4_nonfinite_exec_px.py
- tests/test_option_a_risk_breach_state_dump.py

## Inventory (path | bytes | category)
| path | bytes | category |
|---|---:|---|
| configs/_generated/stage_ab_breakout_STRICT.yaml | 3614 | configs |
| configs/_generated/stage_ab_family1_breakout_v1.yaml | 3948 | configs (review) |
| docs/OPTION_A_RESET_LOG.md | 8092 | docs |
| docs/PART3_NONFINITE_ROOT_CAUSE_REPORT.md | 5147 | docs |
| docs/PART4_DEADLETTER_POLICY_DECISION.md | 1499 | docs |
| docs/PART4_SCALE_GATE_KPIS.md | 924 | docs |
| docs/PART4_STAGE_C_PLAN.md | 844 | docs |
| docs/STAGE_AB_STRICT_REPORT.md | 3276 | docs |
| tests/test_harness_fault_isolation.py | 4657 | tests |
| tests/test_harness_nonfinite_exec_px_localized.py | 5134 | tests |
| tests/test_module1_no_nan_equity_on_invalid_bars.py | 3203 | tests |
| tests/test_module4_next_open_no_lookahead.py | 6647 | tests |
| tests/test_module4_nonfinite_exec_px.py | 4697 | tests |
| tests/test_option_a_risk_breach_state_dump.py | 4253 | tests |
| weightiz_module1_core.py | 38796 | core_source |
| weightiz_module4_strategy_funnel.py | 47861 | core_source |
| weightiz_module5_harness.py | 158044 | core_source |
| weightiz_module5_stats.py | 22981 | core_source (review) |

## Security Checks
- Secret scan (`git diff | grep -iE ...`): PASS
- Secret scan staged (`git diff --cached | grep -iE ...`): PASS

## Large File Check (>1MB)
- PASS: no modified/untracked files > 1MB.

## Ignore Policy Validation
- `.gitignore` contains:
  - `artifacts/`
  - `data/`
- `git check-ignore` confirms:
  - `artifacts/` ignored
  - `artifacts/module5_harness/` ignored
  - `data/alpaca/` ignored
- `configs/_generated/stage_ab_breakout_STRICT.yaml` is not ignored.

## Final Staging Decision
### Allowlist
- weightiz_module1_core.py
- weightiz_module4_strategy_funnel.py
- weightiz_module5_harness.py
- tests/test_harness_fault_isolation.py
- tests/test_harness_nonfinite_exec_px_localized.py
- tests/test_module1_no_nan_equity_on_invalid_bars.py
- tests/test_module4_next_open_no_lookahead.py
- tests/test_module4_nonfinite_exec_px.py
- tests/test_option_a_risk_breach_state_dump.py
- configs/_generated/stage_ab_breakout_STRICT.yaml
- docs/OPTION_A_RESET_LOG.md
- docs/STAGE_AB_STRICT_REPORT.md
- docs/PART3_NONFINITE_ROOT_CAUSE_REPORT.md
- docs/PART4_DEADLETTER_POLICY_DECISION.md
- docs/PART4_SCALE_GATE_KPIS.md
- docs/PART4_STAGE_C_PLAN.md

### Review (excluded by default)
- weightiz_module5_stats.py
- configs/_generated/stage_ab_family1_breakout_v1.yaml

### Denylist
- artifacts/**
- data/**
- **/*.parquet
- **/*.jsonl
- **/*.csv
- .venv/**
- **/__pycache__/**

## Statement
No commit/push executed in this task.
