# FAMILY SWEEP AUDIT REPORT (RIGOROUS CLOSURE / CITADEL-STYLE)

## Red Flags Acknowledgement + Mandatory Fixes Applied
- `RED FLAG 1 (Warmup=W fallacy)`: generator mutates only `module2.profile_window_bars`; it does **not** overwrite `profile_warmup_bars`.
  - Evidence: `scripts/build_family_sweeps.py` line 226-232.
- `RED FLAG 2 (hardcoded thresholds)`: generator pool extraction reads source config values dynamically; no hardcoded T/A/B lists.
  - Evidence: `scripts/build_family_sweeps.py` line 68-114 + regex probes (Phase 2).
- `RED FLAG 3 (canonical hash double-rounding)`: canonical hash now sorts by `(config_id, seed)`, casts numeric to float64, rounds to 8 decimals, serializes CSV bytes, hashes sha256.
  - Evidence: `run_research.py` line 900-914.
- `RED FLAG 4 (silent clamp)`: hard fail with explicit message if `parallel_workers > 14`.
  - Evidence: `run_research.py` line 1210-1213.
- `RED FLAG 5 (smoke timeout)`: smoke run used temporary shortened date range + reduced split params + single manual candidate; post-run bounds checks included.
  - Evidence: Phase 5 commands/output.
- `RED FLAG 6 (stale aggregator merge)`: aggregator now enforces required families, selects deterministic latest candidate for duplicates, and fail-closes on summary/file hash mismatch.
  - Evidence: `scripts/aggregate_family_leaderboard.py` line 55-106, 109-147.

---

## Phase 0 — Clean Room Prep (verbatim outputs)

### 0A. Environment
```bash
=== PHASE 0A ===
/usr/bin/python3
Python 3.9.6
pip 21.2.4 from /Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/site-packages/pip (python 3.9)
Darwin fa:71:08:a4:3f:78 25.3.0 Darwin Kernel Version 25.3.0: Wed Jan 28 20:49:24 PST 2026; root:xnu-12377.81.4~5/RELEASE_ARM64_T8132 arm64
```

### 0B. Repo identity
```bash
=== PHASE 0B ===
/Users/afekshusterman/Weightiz-Research/WQ---MP-Research
5dfb9be5c3a88d09d3d7cde3a73919110335f50a
 M run_research.py
?? configs/_generated/run_20260302_101215_53d7fee.yaml
?? configs/_generated/run_20260302_101215_53d7fee_local.yaml
?? configs/_generated/run_family_d_orch_20260302_230039_chunk_00.yaml
?? configs/_generated/run_family_d_orch_20260302_230039_chunk_01.yaml
?? configs/_generated/run_family_d_orch_20260302_230039_chunk_02.yaml
?? configs/_generated/run_family_d_orch_20260302_230039_chunk_03.yaml
?? configs/_generated/run_family_d_orch_20260302_230039_chunk_04.yaml
?? configs/_generated/run_family_d_orch_20260302_230039_chunk_05.yaml
?? configs/_generated/run_family_d_orch_20260302_230039_chunk_06.yaml
?? configs/_generated/run_family_d_orch_20260302_230039_chunk_07.yaml
?? configs/_generated/run_family_d_orch_20260303_013728_chunk_00.yaml
?? configs/_generated/run_family_d_orch_20260303_013728_chunk_01.yaml
?? configs/_generated/run_family_d_orch_20260303_013728_chunk_02.yaml
?? configs/_generated/run_family_d_orch_20260303_013728_chunk_03.yaml
?? configs/_generated/run_family_d_orch_20260303_013728_chunk_04.yaml
?? configs/_generated/run_family_d_orch_20260303_013728_chunk_05.yaml
?? configs/_generated/run_family_d_orch_20260303_013728_chunk_06.yaml
?? configs/_generated/run_family_d_orch_20260303_013728_chunk_07.yaml
?? configs/_generated/run_family_d_rejection.yaml
?? configs/_generated/sweep_family_marathoners.yaml
?? configs/_generated/sweep_family_snipers.yaml
?? configs/_generated/sweep_family_sprinters.yaml
?? configs/_generated/sweep_family_surfers.yaml
?? configs/sweep_20x432.yaml
?? docs/SWEEP_CODE_MAP.md
?? docs/SWEEP_MATH_SPEC.md
?? launch_all_families.sh
?? scripts/aggregate_family_leaderboard.py
?? scripts/analyze_results.py
?? scripts/build_family_sweeps.py
?? scripts/validate_config.py
```

### 0C. Untracked generated configs listed (no deletion)
```bash
=== PHASE 0C ===
build_sweep_20x432.py
run_20260302_101215_53d7fee.yaml
run_20260302_101215_53d7fee_local.yaml
run_family_d_orch_20260302_230039_chunk_00.yaml
run_family_d_orch_20260302_230039_chunk_01.yaml
run_family_d_orch_20260302_230039_chunk_02.yaml
run_family_d_orch_20260302_230039_chunk_03.yaml
run_family_d_orch_20260302_230039_chunk_04.yaml
run_family_d_orch_20260302_230039_chunk_05.yaml
run_family_d_orch_20260302_230039_chunk_06.yaml
run_family_d_orch_20260302_230039_chunk_07.yaml
run_family_d_orch_20260303_013728_chunk_00.yaml
run_family_d_orch_20260303_013728_chunk_01.yaml
run_family_d_orch_20260303_013728_chunk_02.yaml
run_family_d_orch_20260303_013728_chunk_03.yaml
run_family_d_orch_20260303_013728_chunk_04.yaml
run_family_d_orch_20260303_013728_chunk_05.yaml
run_family_d_orch_20260303_013728_chunk_06.yaml
run_family_d_orch_20260303_013728_chunk_07.yaml
run_family_d_rejection.yaml
stage_ab_breakout_STRICT.yaml
stage_c_breakout_scale_30.yaml
sweep_20x432.yaml
sweep_family_marathoners.yaml
sweep_family_snipers.yaml
sweep_family_sprinters.yaml
sweep_family_surfers.yaml
```

---

## Phase 1 — Static Compliance Checklist

| Requirement | PASS/FAIL | Evidence (path:function:line) | Notes |
|---|---|---|---|
| Family trigger `run_name.startswith("sweep_family_")` | PASS | `run_research.py::_family_mode_enabled` line 799-800 | Exact prefix gate present |
| Strict YAML schema (`extra="forbid"`) | PASS | `run_research.py` models at lines 55,72,98,157,179,230,269,282,293,306 | Root + nested strict schemas |
| Worker cap fail-closed >14 | PASS | `run_research.py::main` line 1210-1213 | Raises `ValueError("Strict cap exceeded: ...")` |
| Deterministic jitter from hash, bounded [10,30] | PASS | `run_research.py::_deterministic_jitter_seconds` line 803-806 | `10 + (sha256(...) % 21)` |
| Results integrity checks pre-parquet | PASS | `run_research.py::_assert_results_integrity` line 877-897 | required cols + unique `(config_id,seed)` + finite numeric |
| Canonical reproducibility hash | PASS | `run_research.py::_canonical_results_sha256` line 900-914 | deterministic sort + float64 cast + round(8) + stable CSV bytes |
| Canonical hash excludes operational timestamps | PASS | `run_research.py::_canonical_results_sha256` line 900-914 + `summary.json` timestamps written separately at line 1167-1174 | Hash computed only from `results_df` |
| Non-family regression guard | PASS | `run_research.py::main` line 1205-1213 and 1228-1235 | family-only branch gates; non-family path bypasses family controls |

Diff summary target files:
```bash
=== DIFF STAT TARGET FILES ===
 run_research.py | 478 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 1 file changed, 478 insertions(+)
```
(Other target files are currently untracked new files; content audited directly.)

---

## Phase 2 — Pool Provenance Report (HARD GATE)

### Authorized source confirmation
Source-of-truth used by default generator:
- `configs/sweep_20x432.yaml`
- Anchor: `scripts/build_family_sweeps.py` line 17 (`DEFAULT_SOURCE_CFG`) and line 314-317 (`--source-config` default)

### Extracted pools (verbatim)
```bash
SOURCE /Users/afekshusterman/Weightiz-Research/WQ---MP-Research/configs/sweep_20x432.yaml
W [60] COUNT 1
T [0.5, 0.55, 0.6, 0.65] COUNT 4
A [0.25, 0.4, 0.55] COUNT 3
B [0.1, 0.15] COUNT 2
K [2, 4, 6] COUNT 3
```

### No hardcoded threshold list evidence (verbatim)
```bash
RG_THRESHOLDS
RG_W60
```
No matches returned for:
- `0.5|0.52|...|0.68`
- `[60]` or `\b60\b`
in `scripts/build_family_sweeps.py`.

### Pool extraction anchors
- `scripts/build_family_sweeps.py::_extract_source_pools` line 68-114
- `scripts/build_family_sweeps.py::_resolve_source_cfg_paths` line 42-58

---

## Phase 3 — Family Grid Proof Report

### Generator run output (verbatim)
```bash
SOURCE_CONFIGS=/Users/afekshusterman/Weightiz-Research/WQ---MP-Research/configs/sweep_20x432.yaml SOURCE_POOLS W=[60] T=[0.5, 0.55, 0.6, 0.65] A=[0.25, 0.4, 0.55] B=[0.1, 0.15] TOPK=[2, 4, 6] M3_BASELINE_POOL=10
sprinters: W=[60] T=[0.5, 0.55, 0.6] A=[0.25, 0.4] B=[0.1, 0.15] TOPK=[2, 4, 6] M3=10 M2=1 M3=10 M4=36 TOTAL=360 M4_UNIQUE=36 PATH=/Users/afekshusterman/Weightiz-Research/WQ---MP-Research/configs/_generated/sweep_family_sprinters.yaml
surfers: W=[60] T=[0.55, 0.6, 0.65] A=[0.25, 0.4, 0.55] B=[0.1, 0.15] TOPK=[2, 4, 6] M3=8 M2=1 M3=8 M4=54 TOTAL=432 M4_UNIQUE=54 PATH=/Users/afekshusterman/Weightiz-Research/WQ---MP-Research/configs/_generated/sweep_family_surfers.yaml
snipers: W=[60] T=[0.55, 0.6, 0.65] A=[0.25, 0.4, 0.55] B=[0.1, 0.15] TOPK=[2, 4, 6] M3=8 M2=1 M3=8 M4=54 TOTAL=432 M4_UNIQUE=54 PATH=/Users/afekshusterman/Weightiz-Research/WQ---MP-Research/configs/_generated/sweep_family_snipers.yaml
marathoners: W=[60] T=[0.55, 0.6, 0.65] A=[0.4, 0.55] B=[0.1, 0.15] TOPK=[2, 4, 6] M3=10 M2=1 M3=10 M4=36 TOTAL=360 M4_UNIQUE=36 PATH=/Users/afekshusterman/Weightiz-Research/WQ---MP-Research/configs/_generated/sweep_family_marathoners.yaml
```

### Family sets/counts proof (verbatim)
```bash
SOURCE_T_POOL [0.5, 0.55, 0.6, 0.65]
FAMILY=sprinters M2=1 M3=10 M4=36 TOTAL=360 W=[60] T=[0.5, 0.55, 0.6] A=[0.25, 0.4] B=[0.1, 0.15] K=[2, 4, 6]
FAMILY=surfers M2=1 M3=8 M4=54 TOTAL=432 W=[60] T=[0.55, 0.6, 0.65] A=[0.25, 0.4, 0.55] B=[0.1, 0.15] K=[2, 4, 6]
FAMILY=snipers M2=1 M3=8 M4=54 TOTAL=432 W=[60] T=[0.55, 0.6, 0.65] A=[0.25, 0.4, 0.55] B=[0.1, 0.15] K=[2, 4, 6]
FAMILY=marathoners M2=1 M3=10 M4=36 TOTAL=360 W=[60] T=[0.55, 0.6, 0.65] A=[0.4, 0.55] B=[0.1, 0.15] K=[2, 4, 6]
SNIPERS_TOP_TAIL_RULE True SNIPERS_T [0.55, 0.6, 0.65] TOP3 [0.55, 0.6, 0.65]
SPRINTERS_LOW_PERCENTILE_RULE True SPRINTERS_T [0.5, 0.55, 0.6] FIRST3 [0.5, 0.55, 0.6]
FAMILY DIVERSITY LIMITATION: W cannot differentiate families
IMPLICATION: families differ only via T/A/B/K and module3 breadth
REMEDIATION: Expand configs/sweep_20x432.yaml with additional profile_window_bars values; do not auto-inject.
```

### Slicing semantics (as implemented)
- Sprinters: lower percentile T slice (`_q_slice(T, p(0), p(66))`)
- Snipers: top-tail T slice (`_q_slice(T, p(34), p(100))`)
- Anchor: `scripts/build_family_sweeps.py` lines 160-193

---

## Phase 4 — Fail-Closed Evidence Log

| Probe | Command/Snippet | Expected | Actual | PASS/FAIL | Error excerpt |
|---|---|---|---|---|---|
| Unknown top-level field | `_load_config` on temp YAML with `__bogus_key__` | Validation crash | `ValidationError` | PASS | `1 validation error for RunConfigModel` |
| Unknown nested field | `_load_config` on temp YAML with `module4_configs[0].__bogus_nested__` | Validation crash | `ValidationError` | PASS | `1 validation error for RunConfigModel` |
| Worker cap = 15 | `python run_research.py --config /tmp/sweep_family_workers15_probe.yaml` | Hard fail | ValueError raised | PASS | `Strict cap exceeded: 15 > 14. Fail-closed.` |
| Jitter determinism | `_deterministic_jitter_seconds` seeds [1..5] twice | In range, stable, diverse | `[29,10,30,10,23]` twice, 4 distinct | PASS | `IN_RANGE True, STABLE_REPEAT True, DISTINCT_COUNT 4` |

Verbatim outputs:
```bash
probe_top_unknown EXPECTED_FAIL ValidationError 1 validation error for RunConfigModel
probe_nested_unknown EXPECTED_FAIL ValidationError 1 validation error for RunConfigModel
```
```bash
WROTE /tmp/sweep_family_workers15_probe.yaml
Traceback (most recent call last):
...
ValueError: Strict cap exceeded: 15 > 14. Fail-closed.
EXIT_CODE=1
```
```bash
RUN_NAME sweep_family_sprinters
SEEDS [1, 2, 3, 4, 5]
JITTER_PASS1 [29, 10, 30, 10, 23]
JITTER_PASS2 [29, 10, 30, 10, 23]
IN_RANGE True
STABLE_REPEAT True
DISTINCT_COUNT 4
```

---

## Phase 5 — Runtime Packaging Verification Report (single smoke)

### Smoke profile used (minimal cost)
- Family: `sweep_family_sprinters`
- Temporary smoke config: `/tmp/sweep_family_sprinters_smoke.yaml`
- Smoke-only reductions:
  - short date range: `2024-01-02` to `2024-01-15`
  - single manual candidate
  - reduced split params (`wf_train/test/step=1`, `cpcv_slices=2`, `cpcv_k_test=1`)
  - baseline-only stress scenario

### Successful run output (verbatim)
```bash
MODULE2_OK elapsed_sec=2.685201 T=3510 A=10 B=240 W=60
...
RUN_COMPLETE
{
  "run_id": "run_20260303_210045",
  "run_dir": "/Users/afekshusterman/Weightiz-Research/WQ---MP-Research/artifacts/module5_harness/run_20260303_210045",
  "n_candidate_results": 1,
  "pass_count": 1,
  "aborted": false,
  ...
}
```

### Required folder structure (verbatim)
```bash
artifacts/sweep_family_sprinters
artifacts/sweep_family_sprinters/audit_bundle
artifacts/sweep_family_sprinters/audit_bundle/config.yaml
artifacts/sweep_family_sprinters/audit_bundle/env_vars.txt
artifacts/sweep_family_sprinters/audit_bundle/git_commit.txt
artifacts/sweep_family_sprinters/audit_bundle/machine_info.json
artifacts/sweep_family_sprinters/audit_bundle/pip_freeze.txt
artifacts/sweep_family_sprinters/audit_bundle/python_version.txt
artifacts/sweep_family_sprinters/audit_bundle/results_parquet_sha256_canonical.txt
artifacts/sweep_family_sprinters/audit_bundle/results_parquet_sha256_file.txt
artifacts/sweep_family_sprinters/audit_bundle/run_manifest.json
artifacts/sweep_family_sprinters/audit_bundle/run_status.json
artifacts/sweep_family_sprinters/audit_bundle/timestamps.json
artifacts/sweep_family_sprinters/pid
artifacts/sweep_family_sprinters/results.parquet
artifacts/sweep_family_sprinters/run.log
artifacts/sweep_family_sprinters/summary.json
```

### Numeric sanity checks (verbatim)
```bash
ROWS 1
UNIQUE_CONFIG_SEED 1
MISSING_REQUIRED []
CHECK_WIN_RATE_RANGE True
CHECK_TRADES_NONNEG True
CHECK_TRADES_INTEGERLIKE True
CHECK_PROFIT_FACTOR_NONNEG True
```

### Canonical hash stability proof (rerun same config)
```bash
HASH1=44e5d39bf4e50ea92c009cbf8dc43285ac8ef0fdbebee146ff5dc926c991635e
FILEHASH1=cef8695d62468d5b297957b5a4cae35bcb1bb3f0d4eb764f5f5a74eb701441e6
...
HASH2=44e5d39bf4e50ea92c009cbf8dc43285ac8ef0fdbebee146ff5dc926c991635e
FILEHASH2=cef8695d62468d5b297957b5a4cae35bcb1bb3f0d4eb764f5f5a74eb701441e6
CANONICAL_HASH_MATCH=TRUE
FILE_HASH_MATCH=TRUE
```

Summary hash fields:
- `results_sha256_canonical`: `44e5d39bf4e50ea92c009cbf8dc43285ac8ef0fdbebee146ff5dc926c991635e`
- `results_sha256_file`: `cef8695d62468d5b297957b5a4cae35bcb1bb3f0d4eb764f5f5a74eb701441e6`

---

## Phase 6 — Aggregator Verification Report

### 6A/6C Fail-closed on missing required families (real artifacts root)
```bash
RuntimeError: Missing required family artifacts: ['sweep_family_surfers', 'sweep_family_snipers', 'sweep_family_marathoners']
AGG_REAL_EXIT=1
```
PASS (required-family hard gate works).

### 6A stale artifact defense (synthetic probe)
- Probe setup: copied sprinters folder to all required family names under `/tmp/agg_probe`, then injected bad `results_sha256_file` in surfers summary.
- Output:
```bash
RuntimeError: Stale artifacts detected in folder. Manual cleanup required. File hash mismatch for /private/tmp/agg_probe/sweep_family_surfers: summary=0000BADHASH actual=cef8695d62468d5b297957b5a4cae35bcb1bb3f0d4eb764f5f5a74eb701441e6
AGG_PROBE_STALE_EXIT=1
```
PASS.

### 6A duplicate handling + 6D success path (synthetic probe)
- Probe setup: `/tmp/agg_probe_ok` includes duplicate `sweep_family_sprinters_older` with older summary timestamp.
- Aggregator selected deterministic latest canonical family folders and succeeded.
- Output:
```bash
AGGREGATION_OK rows=4 out=/private/tmp/agg_probe_ok/leaderboard.parquet
TOP50_MD=/private/tmp/agg_probe_ok/leaderboard_top50.md
FAMILY_SELECTIONS
sweep_family_sprinters => /private/tmp/agg_probe_ok/sweep_family_sprinters
sweep_family_surfers => /private/tmp/agg_probe_ok/sweep_family_surfers
sweep_family_snipers => /private/tmp/agg_probe_ok/sweep_family_snipers
sweep_family_marathoners => /private/tmp/agg_probe_ok/sweep_family_marathoners
```
Top-10 preview printed from `/tmp/agg_probe_ok/leaderboard.parquet` (4 rows total).

---

## Final GO/NO-GO Decision

**Decision: GO (with one explicit data-coverage limitation acknowledged).**

All required audit gates passed after surgical fixes, with one non-fatal limitation:
- `W` pool in authorized source (`configs/sweep_20x432.yaml`) has only one value (`60`), so family differentiation cannot occur on W today.
- Remediation is config-only (not auto-injected): expand `configs/sweep_20x432.yaml` `module2_configs[*].profile_window_bars` values.

No strategy math (Modules 2/3/4/5) was modified by this audit closure.
