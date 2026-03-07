#!/usr/bin/env bash
set -euo pipefail

echo "== Freeze staging preflight =="
git status --short

echo "== Unstage everything =="
git reset

stage_file() {
  local p="$1"
  if [[ ! -f "$p" ]]; then
    echo "ERROR: missing allowlist file: $p"
    exit 1
  fi
  git add "$p"
}

echo "== Stage explicit allowlist =="
stage_file "weightiz_module1_core.py"
stage_file "weightiz_module4_strategy_funnel.py"
stage_file "weightiz_module5_harness.py"

stage_file "tests/test_harness_fault_isolation.py"
stage_file "tests/test_harness_nonfinite_exec_px_localized.py"
stage_file "tests/test_module1_no_nan_equity_on_invalid_bars.py"
stage_file "tests/test_module4_next_open_no_lookahead.py"
stage_file "tests/test_module4_nonfinite_exec_px.py"
stage_file "tests/test_option_a_risk_breach_state_dump.py"

if git check-ignore -q "configs/_generated/stage_ab_breakout_STRICT.yaml"; then
  if [[ ! -f "configs/_generated/stage_ab_breakout_STRICT.yaml" ]]; then
    echo "ERROR: missing strict config: configs/_generated/stage_ab_breakout_STRICT.yaml"
    exit 1
  fi
  git add -f "configs/_generated/stage_ab_breakout_STRICT.yaml"
else
  stage_file "configs/_generated/stage_ab_breakout_STRICT.yaml"
fi

stage_file "docs/OPTION_A_RESET_LOG.md"
stage_file "docs/STAGE_AB_STRICT_REPORT.md"
stage_file "docs/PART3_NONFINITE_ROOT_CAUSE_REPORT.md"
stage_file "docs/PART4_DEADLETTER_POLICY_DECISION.md"
stage_file "docs/PART4_SCALE_GATE_KPIS.md"
stage_file "docs/PART4_STAGE_C_PLAN.md"

echo "== Staged summary =="
git status --short
git diff --cached --name-only
git diff --cached --stat

echo "== Denylist enforcement =="
if git diff --cached --name-only | grep -E "^(artifacts/|data/)|\\.parquet$|\\.jsonl$|\\.csv$|^\\.venv/|/__pycache__/"; then
  echo "ERROR: forbidden files staged"
  exit 1
fi

echo "== Review-list enforcement =="
if git diff --cached --name-only | grep -E "^(weightiz_module5_stats\\.py|configs/_generated/stage_ab_family1_breakout_v1\\.yaml)$"; then
  echo "ERROR: review-list file staged (not allowed by default)"
  exit 1
fi

echo "== Freeze staging complete (no commit, no push) =="
