# Dynamic Gates Forensic Audit (2026-03-17)

## 1. Scope and locked context
- Scope: dynamic-gates seam only (`gate_calibrator`, config builders/models, `run_research` wiring, seam tests).
- Locked architecture preserved: `Module2 -> Module3 -> Module4 (signal-only) -> Module5/risk/execution -> Module6`.
- Non-goals respected: no Module4 redesign, no risk engine authority change, no schema contract drift, no alternate runtime path.

## 2. Known defects and closure status
- Defect 1 (builder-time live harness calibration): **Closed**.
  - `build_harness_config` no longer instantiates `GateCalibrator` and does not call `calibrate()`.
- Defect 2 (missing explicit regression test for builder passivity): **Closed**.
  - Added `TestHarnessBuilderPassivity.test_build_harness_config_keeps_thresholds_static_except_explicit_override`.
- Defect 3 (duplicate kwargs concern): **Not real in live code**.
  - `py_compile` passes and only one assignment each for `robustness_reject_threshold` and `execution_fragile_threshold` in `Module5HarnessConfig(...)` construction.

## 3. Live-code findings
- Pre-run dynamic calibration is limited to Module6 intake geometry gates and applied before Module6 execution (`run_research.py`).
- Post-run empirical calibration remains report-only and is written into `run_manifest`/`run_status` metadata only.
- Harness builder now applies only configured values + explicit overrides for harness thresholds.
- Policy/safety gates listed in `GateCalibrator` stay static.

## 4. Gate inventory

| GATE_ID | LOCATION | CURRENT_VALUE | OWNER | WHAT_IT_GUARDS | UPSTREAM_INPUTS | DOWNSTREAM_EFFECT | FAILURE_MODE_IF_TOO_HIGH | FAILURE_MODE_IF_TOO_LOW | SEMANTIC_TYPE | STATIC_OR_DYNAMIC_DECISION | RATIONALE |
|---|---|---:|---|---|---|---|---|---|---|---|---|
| G01 min_availability_ratio | module6 intake + calibrator | static baseline 0.95 (standard), dynamic bounded | Module6 intake | minimum observed support ratio | run geometry (`wf_test/common`, CPCV share) | admission into reduction/scoring | false rejects | weak-support admissions | GEOMETRY | DYNAMIC_GEOMETRY | depends on run split geometry |
| G02 required_comparison_support | module6 scoring + calibrator | static baseline 0.85, dynamic bounded | Module6 scoring | cross-universe comparability support | run geometry | cross-universe rejection | over-pruning finalists | cross-universe instability | GEOMETRY | DYNAMIC_GEOMETRY | burden scales with support calendar geometry |
| G03 robustness_reject_threshold | module5 robustness support + calibrator | baseline 0.40, post-run report-only dynamic | Module5 harness | weak candidate rejection | robustness score distribution | diagnostics/calibration metadata | over-reject exploration | low-quality candidates pass | DISTRIBUTION | REPORT_ONLY_DYNAMIC | empirical tail separation gate |
| G04 execution_fragile_threshold | module5 robustness support + calibrator | baseline 0.50, post-run report-only dynamic | Module5 harness | execution fragility labeling | execution robustness/fill-failure distribution | diagnostics/calibration metadata | over-label fragile | execution-fragile candidates survive | DISTRIBUTION | REPORT_ONLY_DYNAMIC | empirical execution distribution gate |
| G05 fill_failure_control_limit | calibrator report-only | baseline 0.25, dynamic control-limit report | Module5 diagnostics | abnormal fill-failure process drift | fill-failure history (past-only) | report-only anomaly limit | noisy alerts | drift missed | CONTROL_LIMIT | REPORT_ONLY_DYNAMIC | process monitoring, not admission policy |
| G06 run_policy_class | module6 config/ledger/reduction | standard/representative_discovery | Module6 policy contract | admission contract semantics | config + artifact class tags | intake/ledger acceptance logic | wrong policy branch | silent policy mismatch | POLICY | KEEP_STATIC | defines system semantics |
| G07 canonical_selection_stage | module6 intake | module5_bridge_canonical_baseline_v1 | Module6 policy contract | canonical artifact identity | config literal | bridge artifact validation | false contract mismatch | accepts wrong artifact stage | POLICY | KEEP_STATIC | identity contract |
| G08 require_bridge_artifacts | module6 intake | true | Module6 safety contract | artifact presence | config bool | fail-closed before intake | unnecessary aborts | hidden missing-source admission | POLICY | KEEP_STATIC | schema/lineage boundary |
| G09 require_zero_filled_daily_returns_non_authoritative | module6 intake | true | Module6 safety contract | replay truth convention | config bool | replay consistency checks | strictness false positives | replay inconsistency acceptance | SAFETY | KEEP_STATIC | fail-closed truth boundary |
| G10 support_policy_version | module6 simulator/orchestrator/scoring | constant tag | Module6 policy contract | support semantics versioning | config constant | comparability checks | forced mismatch | silent semantic drift | POLICY | KEEP_STATIC | explicit policy version contract |
| G11 ranking_policy_version | module6 scoring | constant tag | Module6 policy contract | ranking semantics versioning | config constant | ranking validation | forced mismatch | silent ranking drift | POLICY | KEEP_STATIC | explicit policy version contract |
| G12 constraint_policy_version | module6 simulator | constant tag | Module6 policy contract | portfolio constraint semantics | config constant | simulator/validation consistency | forced mismatch | silent constraint drift | POLICY | KEEP_STATIC | explicit policy version contract |
| G13 overnight_policy_version | module6 simulator | constant tag | Module6 policy contract | overnight semantics versioning | config constant | overnight rule checks | forced mismatch | silent overnight drift | POLICY | KEEP_STATIC | explicit policy version contract |
| G14 friction_policy_version | module6 simulator | constant tag | Module6 policy contract | friction semantics versioning | config constant | replay/scoring consistency | forced mismatch | silent friction drift | POLICY | KEEP_STATIC | explicit policy version contract |
| G15 module6_policy_class_match | module6 ledger/reduction | exact equality enforced | Module6 ledger | candidate-policy consistency | candidate artifacts + config | fail-closed intake | reject valid mixed sets | accept mixed-policy artifacts | POLICY | KEEP_STATIC | identity/compliance boundary |
| G16 calendar_version_singleton | module6 scoring | exact singleton required | Module6 scoring | single calendar contract | session score metadata | cross-universe score validity | false reject | mixed-calendar contamination | POLICY | KEEP_STATIC | identity consistency boundary |
| G17 comparison_support_recomputed | module6 scoring/orchestrator | must be true for cross-universe | Module6 scoring | true comparable-support semantics | orchestrator recomputation flag | comparable truth-score gating | false reject | non-comparable scoring accepted | POLICY | KEEP_STATIC | prevents semantic shortcut |
| G18 duplicate_corr_threshold | module6 reduction | 0.85 | Module6 reduction policy | duplicate elimination policy | pairwise return corr | duplicate pruning | removes diversification | duplicate leak-through | POLICY | KEEP_STATIC | portfolio construction policy |
| G19 drawdown_concurrence_threshold | module6 reduction | 0.60 | Module6 reduction policy | duplicate drawdown overlap policy | drawdown concurrence | duplicate pruning | over-pruning | duplicate leak-through | POLICY | KEEP_STATIC | policy-level de-dup rule |
| G20 min_observed_sessions | module6 intake | 126 (standard default) | Module6 intake | minimum sample sufficiency | observed session count | admission gate | over-reject short histories | unstable admissions | SAFETY | KEEP_BOUNDED_STATIC | hard sample floor |
| G21 min_cross_universe_support | module6 scoring | 0.85 | Module6 scoring | minimum support floor | config static | cross-universe required support | over-reject | weak comparability accepted | SAFETY | KEEP_BOUNDED_STATIC | hard floor inside dynamic blend |
| G22 min_truth_score_ratio | module6 scoring | 0.80 | Module6 scoring | truth replay quality floor | replay session/minute scores | final acceptance | rejects too many | accepts drifted truth | SAFETY | KEEP_BOUNDED_STATIC | hard integrity floor |
| G23 max_allowed_return_drift_frac | module6 scoring | 0.20 | Module6 scoring | return drift cap | replay deltas | reject inconsistent portfolios | false rejects | drift tolerance too loose | SAFETY | KEEP_BOUNDED_STATIC | fail-closed drift cap |
| G24 max_allowed_drawdown_drift | module6 scoring | 0.02 | Module6 scoring | drawdown drift cap | replay deltas | reject inconsistent portfolios | false rejects | allows drawdown drift | SAFETY | KEEP_BOUNDED_STATIC | fail-closed drift cap |
| G25 max_allowed_turnover_drift_frac | module6 scoring | 0.25 | Module6 scoring | turnover drift cap | replay deltas | reject inconsistent portfolios | false rejects | allows turnover drift | SAFETY | KEEP_BOUNDED_STATIC | fail-closed drift cap |
| G26 max_allowed_gross_exposure_drift_frac | module6 scoring | 0.15 | Module6 scoring | gross exposure drift cap | replay deltas | reject inconsistent portfolios | false rejects | allows gross drift | SAFETY | KEEP_BOUNDED_STATIC | fail-closed drift cap |
| G27 max_allowed_breach_count_delta | module6 scoring | 0 | Module6 scoring | risk breach count invariance | replay breach counts | reject inconsistent portfolios | false rejects | allows hidden breach inflation | SAFETY | KEEP_BOUNDED_STATIC | hard invariant |
| G28 min_rank_stability | module6 scoring | 0.90 | Module6 scoring | ranking stability floor | replay rank correlation | final acceptance | over-reject | unstable ranking accepted | SAFETY | KEEP_BOUNDED_STATIC | truth-replay stability floor |
| G29 max_abs_rank_delta_p95 | module6 scoring | 16 | Module6 scoring | tail rank drift cap | replay rank delta distribution | final acceptance | over-reject | unstable rank tails accepted | SAFETY | KEEP_BOUNDED_STATIC | tail drift cap |
| G30 shortlist_session_keep | module6 scoring | 1024 | Module6 scoring | session shortlist capacity | candidate count | compute/memory/control | too expensive/noisy | over-aggressive pruning | CAPACITY | KEEP_BOUNDED_STATIC | institutional capacity ceiling |
| G31 shortlist_minute_keep | module6 scoring | 256 | Module6 scoring | minute shortlist capacity | candidate count | replay workload | too expensive/noisy | over-pruning | CAPACITY | KEEP_BOUNDED_STATIC | bounded runtime capacity |
| G32 final_scalar_keep | module6 scoring | 64 | Module6 scoring | final scalar ranking capacity | shortlist output | final candidate set size | excess downstream load | diversity loss | CAPACITY | KEEP_BOUNDED_STATIC | bounded final-stage capacity |
| G33 final_primary_count | module6 scoring | 6 | Module6 scoring | production sleeve count | ranked finalists | final portfolio composition | concentration pressure | under-diversification by underfill | CAPACITY | KEEP_BOUNDED_STATIC | explicit portfolio capacity policy |
| G34 final_alternate_count | module6 scoring | 6 | Module6 scoring | alternate sleeve count | ranked finalists | contingency set | oversized alternates | insufficient fallback | CAPACITY | KEEP_BOUNDED_STATIC | explicit contingency capacity policy |
| G35 execution_max_volume_participation | module4 risk_engine + harness | 1.0 default (bounded (0,1]) | risk_engine realism | participation realism cap | volume, target qty | fill truncation/rejection | excess rejected fills | unrealistic fills | SAFETY | KEEP_BOUNDED_STATIC | hard realism bound |
| G36 daily_loss_limit_frac | module6 simulator/risk replay | 0.10 | simulator | kill-switch risk cap | equity path | forced de-risk/disable | premature disable | insufficient capital protection | SAFETY | KEEP_BOUNDED_STATIC | emergency risk invariant |

## 5. Gate type decisions
- Dynamic now (implemented): `G01`, `G02`, `G03`, `G04`, `G05`.
- Static policy (intentionally static): `G06`-`G19`.
- Static safety/capacity (bounded static): `G20`-`G36`.
- Summary: dynamic gates are selective and bounded; policy/safety contracts were not converted to empirical heuristics.

## 6. Dynamic formulas that are allowed

### G01 `min_availability_ratio` (pre-run, active)
- STATIC_BASELINE: configured intake value (standard default resolves to 0.95).
- CURRENT_FAILURE_IF_STATIC: over/under strictness when split geometry changes.
- DYNAMIC_DRIVER: `wf_test/common`, `cpcv_k_test/cpcv_slices`, common-session granularity.
- FORMULA: `max(static, clip(max(wf_test/common, cpcv_test_share) + 1/common, 0.05, 0.98))`.
- FLOOR: `0.05`.
- CAP: `0.98`.
- WARMUP_REQUIREMENT: `common_sessions > 0`.
- FALLBACK_IF_INSUFFICIENT_DATA: static baseline.
- LOG_MESSAGE_TEMPLATE: `dynamic_geometry ... bounded=... static=... final=...`.

### G02 `required_comparison_support` (pre-run, active)
- STATIC_BASELINE: configured value (default 0.85).
- CURRENT_FAILURE_IF_STATIC: mismatch between required support and actual calendar burden.
- DYNAMIC_DRIVER: same run-geometry shares + stricter margin.
- FORMULA: `max(static, max(min_availability_ratio_dyn, clip(max(wf_test/common, cpcv_test_share) + 2/common, 0.10, 0.99)))`.
- FLOOR: `0.10`.
- CAP: `0.99`.
- WARMUP_REQUIREMENT: `common_sessions > 0`.
- FALLBACK_IF_INSUFFICIENT_DATA: static baseline.
- LOG_MESSAGE_TEMPLATE: `dynamic_geometry ... bounded=... static=... final=...`.

### G03 `robustness_reject_threshold` (post-run report-only)
- STATIC_BASELINE: configured harness value (default 0.40).
- CURRENT_FAILURE_IF_STATIC: fixed cutoff ignores campaign-specific score shape.
- DYNAMIC_DRIVER: empirical robustness distribution.
- FORMULA: `max(static, clip(max(q30, median - 1.4826*MAD), 0.0, 0.99))`.
- FLOOR: `0.0`.
- CAP: `0.99`.
- WARMUP_REQUIREMENT: `n >= min_distribution_samples` (default 20).
- FALLBACK_IF_INSUFFICIENT_DATA: static baseline.
- LOG_MESSAGE_TEMPLATE: `dynamic_distribution ... n=... final=...`.

### G04 `execution_fragile_threshold` (post-run report-only)
- STATIC_BASELINE: configured harness value (default 0.50).
- CURRENT_FAILURE_IF_STATIC: fixed fragility cutoff ignores execution regime variation.
- DYNAMIC_DRIVER: empirical execution robustness (or `1 - fill_failure_rate` fallback).
- FORMULA: `max(static, clip(max(q25, median - 0.8*1.4826*MAD), 0.0, 0.99))`.
- FLOOR: `0.0`.
- CAP: `0.99`.
- WARMUP_REQUIREMENT: `n >= min_distribution_samples`.
- FALLBACK_IF_INSUFFICIENT_DATA: static baseline.
- LOG_MESSAGE_TEMPLATE: `dynamic_distribution ... n=... final=...`.

### G05 `fill_failure_control_limit` (post-run report-only)
- STATIC_BASELINE: 0.25.
- CURRENT_FAILURE_IF_STATIC: insensitive to changing fill-failure process noise.
- DYNAMIC_DRIVER: historical fill-failure process (past-only history).
- FORMULA: `max(static, clip(median(history) + k*1.4826*MAD(history), 0.05, 0.95))`, default `k=3`.
- FLOOR: `0.05`.
- CAP: `0.95`.
- WARMUP_REQUIREMENT: sample size `> warmup` and history size `>= warmup`.
- FALLBACK_IF_INSUFFICIENT_DATA: static baseline.
- LOG_MESSAGE_TEMPLATE: `dynamic_control_limit ... warmup=... n=... final=...`.

## 7. Static gates intentionally left static
- Policy contracts (`run_policy_class`, policy versions, canonical stage, bridge requirements, class/calendar consistency) remain static to preserve explicit semantics and fail-closed identity checks.
- Safety caps/floors (observed sessions, truth replay drift bounds, rank stability, loss limits, participation limits) remain static or bounded-static to prevent adaptive weakening under noisy samples.
- Capacity gates (shortlists and final counts) remain bounded-static because they encode explicit compute and portfolio cardinality contracts, not empirical tail separation.

## 8. Builder passivity conclusion
- `build_harness_config` is a passive config-construction step and does not perform live dynamic mutation of Harness runtime thresholds.
- Explicit overrides remain honored as explicit configuration semantics only (`gate_overrides.*.value`).
- No empirical stats are consumed in builder-time harness threshold construction.

## 9. Report-only recalibration conclusion
- Post-run empirical recalibration remains report-only in this pass.
- `run_research.py` computes `harness_post_run_report_only` calibration and stores it in manifest/status metadata.
- Already-executed run decisions are not retroactively mutated.

## 10. Blast-radius audit
- Runtime architecture unchanged: canonical single path preserved.
- Module4 remains signal-only; no execution authority changes in `risk_engine`.
- Module6 policy contracts unchanged and still explicit.
- No artifact schema breakage observed on touched run-manifest/status metadata keys.
- Harness builder mutation path removed; calibrator wiring remains narrow (`builders` + `run_research`).
- Unrelated local edits were isolated in stash and excluded from push-set.

## 11. Validation evidence
- Focused seam:
  - `PYTHONPATH=src python3 -m pytest -q tests/test_gate_calibrator.py tests/test_module6_policy_contract.py tests/test_module6_e2e.py::test_run_research_fails_closed_when_module6_blocks tests/test_e2e_config_integrity.py`
  - Result: `22 passed`.
- Canonical architecture:
  - `PYTHONPATH=src python3 -m pytest -q tests/test_stage_a_cloud_campaign.py tests/test_canonical_single_path.py tests/test_architecture_pipeline.py tests/test_cli_server_paths.py`
  - Result: `14 passed`.
- Config compatibility:
  - `RunConfigModel.model_validate` over representative canonical configs (`e2e_proving`, `e2e_test`, `local_canonical_proving`, `local_discovery_short_7core`, `local_adaptive_discovery_7core`, `local_trade_density_micro_7core_2025`, `template`, `server/compute-small`).
  - Result: all validated.

## 12. Push gate recommendation
- Recommendation: **READY_TO_PUSH**.
- Basis:
  - known defects closed,
  - builder passivity verified,
  - report-only recalibration preserved,
  - no duplicate kwargs blocker in live code,
  - focused + architecture + compatibility validation green,
  - unrelated local files isolated and excluded.
