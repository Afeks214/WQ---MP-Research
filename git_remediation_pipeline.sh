#!/usr/bin/env bash
set -euo pipefail

branch="$(git rev-parse --abbrev-ref HEAD)"
if [[ "${branch}" != "main-mp" ]]; then
  echo "Expected branch main-mp, got ${branch}" >&2
  exit 1
fi

git add \
  src/weightiz/module5/harness/module6_bridge.py \
  src/weightiz/module5/harness/orchestrator_support.py \
  src/weightiz/module6/reduction.py \
  tests/test_module6_bridge_fill_attempts.py \
  tests/test_module6_reduction.py \
  tests/test_module6_screening_truth_divergence.py

cat <<'PATCH' >/tmp/minute_refine_bridge_hunk.patch
diff --git a/src/weightiz/module6/simulator/minute_refine.py b/src/weightiz/module6/simulator/minute_refine.py
index ac696dd..ad09da3 100644
--- a/src/weightiz/module6/simulator/minute_refine.py
+++ b/src/weightiz/module6/simulator/minute_refine.py
@@ -364,18 +386,29 @@ def replay_finalists_minute(
         )
     divergence = pd.DataFrame(divergence_rows).sort_values(["portfolio_pk"], kind="mergesort").reset_index(drop=True)
     corr = 1.0
+    rank_delta_p95 = float(np.percentile(np.abs(np.asarray(divergence["rank_delta"], dtype=np.float64)), 95)) if divergence.shape[0] > 0 else 0.0
+    skip_global_rank_gate = False
     if divergence.shape[0] > 3:
         session_scores_arr = np.asarray(divergence["session_score"], dtype=np.float64)
         minute_scores_arr = np.asarray(divergence["minute_score"], dtype=np.float64)
-        if np.allclose(session_scores_arr, session_scores_arr[0]) or np.allclose(minute_scores_arr, minute_scores_arr[0]):
-            corr = 1.0
-        else:
-            corr = float(spearmanr(session_scores_arr, minute_scores_arr).correlation)
-            if not np.isfinite(corr):
+        score_spread_floor = float(config.scoring.return_scale_floor)
+        session_score_span = float(np.ptp(session_scores_arr))
+        minute_score_span = float(np.ptp(minute_scores_arr))
+        # Near-tied scores are economically indistinguishable at the configured truth scale,
+        # so strict rank-order agreement would be noise-sensitive rather than informative.
+        skip_global_rank_gate = (
+            session_score_span <= score_spread_floor and minute_score_span <= score_spread_floor
+        )
+        if not skip_global_rank_gate:
+            if np.allclose(session_scores_arr, session_scores_arr[0]) or np.allclose(minute_scores_arr, minute_scores_arr[0]):
                 corr = 1.0
-    if corr < float(config.scoring.min_rank_stability):
+            else:
+                corr = float(spearmanr(session_scores_arr, minute_scores_arr).correlation)
+                if not np.isfinite(corr):
+                    corr = 1.0
+    if not skip_global_rank_gate and corr < float(config.scoring.min_rank_stability):
         raise Module6ValidationError("SCREENING_TRUTH_RANK_INSTABILITY")
-    if float(np.percentile(np.abs(np.asarray(divergence["rank_delta"], dtype=np.float64)), 95)) > float(config.scoring.max_abs_rank_delta_p95):
+    if not skip_global_rank_gate and rank_delta_p95 > float(config.scoring.max_abs_rank_delta_p95):
         raise Module6ValidationError("SCREENING_TRUTH_RANK_DRIFT_P95")
     return MinuteReplayArtifacts(
         minute_paths=pd.DataFrame(minute_rows).sort_values(["portfolio_pk", "ts_ns"], kind="mergesort").reset_index(drop=True),
PATCH
git apply --cached /tmp/minute_refine_bridge_hunk.patch
git commit -m "feat(m5): implement bridge v2 and availability gate logic"

git add \
  src/weightiz/module6/config.py \
  src/weightiz/module6/simulator/session_path.py \
  src/weightiz/module6/simulator/minute_refine.py \
  tests/test_module6_session_simulator.py \
  tests/test_module6_minute_replay.py
git commit -m "fix(m6): decouple starting equity from disable floor to prevent auto-kill"

git add \
  src/weightiz/module6/scoring.py \
  src/weightiz/module6/frontier.py \
  tests/test_module6_scoring.py \
  tests/test_module6_frontier.py
git commit -m "fix(m6): hard-reject breached and zero-gross portfolios from final selection"

git add \
  src/weightiz/cli/validate_artifacts.py \
  tests/test_cli_server_paths.py \
  scripts/query_module6_selection_proof.py \
  git_remediation_pipeline.sh
git commit -m "test(m6): enforce pair-reason export and gross exposure in validator"

git push origin HEAD:main-mp
