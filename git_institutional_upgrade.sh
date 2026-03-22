#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

EXPECTED_BRANCH="main-mp"
CURRENT_BRANCH="$(git branch --show-current)"

if [[ "$CURRENT_BRANCH" != "$EXPECTED_BRANCH" ]]; then
  echo "Refusing to run on branch '$CURRENT_BRANCH' (expected '$EXPECTED_BRANCH')." >&2
  exit 1
fi

git add \
  src/weightiz/module6/config.py \
  src/weightiz/module6/scoring.py \
  src/weightiz/module6/frontier.py \
  tests/module6_testkit.py \
  tests/test_module6_scoring.py \
  tests/test_module6_frontier.py \
  tests/test_module6_cross_universe_comparability.py
git commit -m "feat(m6): implement minimum risk budget and target gross exposure"

git add \
  src/weightiz/module6/reduction.py \
  src/weightiz/module6/utils.py \
  tests/test_module6_reduction.py
git commit -m "feat(m6): introduce pre-optimizer break-even friction sweep"

git add \
  scripts/query_module6_selection_proof.py \
  src/weightiz/module6/generators/cluster_balanced.py \
  src/weightiz/module6/ledger.py \
  src/weightiz/module6/matrices.py \
  src/weightiz/module6/simulator/minute_refine.py \
  src/weightiz/module6/simulator/session_path.py \
  tests/test_module6_matrices.py \
  tests/test_module6_session_simulator.py
git commit -m "fix(m6): eradicate cheap cost heuristics and fillna band-aids"

git push origin HEAD:main-mp
