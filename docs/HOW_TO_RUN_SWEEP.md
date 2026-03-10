# How To Run 20x100 Sweep

## 1) Preconditions

1. Data exists under `./data/alpaca/clean` for all 20 symbols in `configs/sweep_20x100.yaml`.
2. Python env is active and dependencies are installed.

## 2) Scout first (recommended)

```bash
./.venv/bin/python run_research.py --config ./configs/sweep_scout_20x12.yaml
```

Use scout results (`robustness_leaderboard.csv` + `plateaus.json`) to identify stable regions.
Then launch the full 20x100 sweep constrained to winning plateaus.

## 3) Launch the full 20x100 sweep

```bash
./.venv/bin/python run_research.py --config ./configs/sweep_20x100.yaml
```

## 4) Expected artifacts

Primary run directory:
- `artifacts/module5_harness/run_YYYYMMDD_HHMMSS/`

Global handoff + registry:
- `artifacts/.latest_run`
- `artifacts/run_index.jsonl`

Run outputs:
- `run_manifest.json`
- `run_summary.json`
- `verdict.json` (candidate-level leaderboard)
- `leaderboard.csv`
- `leaderboard.json`
- `robustness_leaderboard.csv`
- `plateaus.json`
- `candidates/<candidate_id>/candidate_config.json`
- `candidates/<candidate_id>/candidate_metrics.json`
- `candidates/<candidate_id>/candidate_stats.json`
- `candidates/<candidate_id>/candidate_returns.parquet`
- `candidates/<candidate_id>/candidate_losses.parquet`

## 4) Quick validation checks

```bash
LATEST_RUN=$(cat artifacts/.latest_run)
ls "$LATEST_RUN"/candidates | wc -l
python - <<'PY'
import pandas as pd
import json
from pathlib import Path
run = Path(open('artifacts/.latest_run').read().strip())
lb = pd.read_csv(run/'leaderboard.csv')
rb = pd.read_csv(run/'robustness_leaderboard.csv')
vd = json.loads((run/'verdict.json').read_text())
print('leaderboard_rows', len(lb))
print('robust_rows', len(rb))
print('is_sorted_desc', rb['robustness_score'].is_monotonic_decreasing)
print('verdict_rows', len(vd['leaderboard']))
print('verdict_is_candidate_level', all('candidate_id' in x and 'task_id' not in x for x in vd['leaderboard']))
print('plateaus_exists', (run/'plateaus.json').exists())
PY
```

## 5) Determinism check (same config, same seeds)

Run twice and compare:
- `resolved_config_sha256` from `run_summary.json`
- `robustness_leaderboard.csv` exact byte equality

```bash
# after two runs, compare file hashes
sha256sum <run1>/robustness_leaderboard.csv <run2>/robustness_leaderboard.csv
```

## 6) Sweep v2 Auto-Resolve (one command)

Use the v2 orchestrator to auto-resolve symbols from `data/alpaca/clean`, generate immutable derived configs under `configs/_generated/`, run scout, derive focused candidates from scout plateaus, then run focused full:

```bash
./.venv/bin/python ./scripts/run_sweep_auto.py \
  --base-config ./configs/sweep_20x100.yaml \
  --scout-config ./configs/sweep_scout_20x12.yaml \
  --clean-dir ./data/alpaca/clean \
  --mode scout_then_focused_full \
  --target-symbols 20 \
  --min-symbols 8 \
  --top-plateaus 3 \
  --max-focused-candidates 30
```

Expected v2 outputs:
- Inventory + manifest: `artifacts/sweep_v2/<UTC_TS>/data_inventory.csv` and `manifest.json`
- Derived configs: `configs/_generated/sweep_auto_full_<UTC_TS>.yaml`, `sweep_auto_scout_<UTC_TS>.yaml`, `sweep_auto_focused_<UTC_TS>.yaml`
- Run artifacts from `run_research.py` under `artifacts/module5_harness/run_...`

## 7) DQ artifacts (robustness hardening)

Each harness run now writes deterministic Data Quality artifacts under the run directory:
- `dq_report.csv`: one row per (symbol, session_date) with decision (`ACCEPT|DEGRADE|REJECT`), reason codes, inferred timeframe, cadence-aware session metrics (`cadence_day_min`, `cadence_day_stable`, delta count/CV), expected/observed/missing bars, bad-tick diagnostics, and DQS components.
- `dq_bar_flags.parquet`: per-bar diagnostics (`timestamp`, `symbol`, `dq_filled_bar`, `dq_issue_flags`, `dqs_day`).

Repair behavior is fail-closed and deterministic:
- Micro-gaps are repaired strictly intra-session only.
- Filled bars use `O=H=L=C=prev_close` and `volume=0`.
- No overnight forward fill is allowed.

## 8) Quick-Run Health Check

Use this when you want a fast end-to-end wiring proof (DQ -> M2 -> M3 -> M4 -> M5 artifacts) without running full CPCV/stress workload:

```bash
./.venv/bin/python ./scripts/run_sweep_auto.py --mode scout_only --quick-run
```

Quick-run behavior is deterministic and schema-safe:
- symbols reduced to first 2 alphabetically from the selected universe,
- candidates reduced to ~2-3 total,
- CPCV disabled and WF reduced to a single split at runtime,
- baseline stress scenario only,
- per-group timeout and progress logging enabled.

Quick-run verifies these artifacts in the final run directory:
- `dq_report.csv`
- `dq_bar_flags.parquet`
- `robustness_leaderboard.csv`
- `run_status.json`

Sweep-v2 logs for quick-run and non-quick subprocess execution are stored under:
- `artifacts/sweep_v2/<UTC_TS>/logs/`

For observability on long non-quick runs:
- `artifacts/sweep_v2/<UTC_TS>/manifest.json` is written immediately after config generation (`status=running`) and updated at run end.
- `artifacts/module5_harness/run_<...>/run_status.json` is checkpointed periodically during execution.

## 9) Robustness & Data Quality

- DQ decisions are per `(symbol, session_date)`: `ACCEPT | DEGRADE | REJECT`.
- `DQS` is propagated at runtime as `dqs_day_ta` and scales Module4 conviction deterministically: `effective_conviction = raw_conviction * DQS`.
- Safety gates:
  - if `DQS < 0.50`, Module4 blocks new entries (neutral).
  - if IB is undefined and policy is `NO_TRADE` (default), Module4 blocks new entries for those rows.
- Invariant guards run post-Module2, post-Module3, and pre-Module4; non-finite rows are masked out (not traded), not silently passed.
- Circuit breaker aborts only on systemic repeated exception signatures:
  - same signature across `>=3` units, `>=2` assets, and `>=2` candidates.
  - localized DQ/invariant reason-coded issues are deadlettered but do not trigger systemic abort by themselves.

## 10) Build Clean Cache From MarketData Bundle

Use the atomic builder to ingest a bundle and replace `data/alpaca/clean` safely:

```bash
./.venv/bin/python scripts/build_clean_cache_from_bundle.py \
  --bundle-zip /mnt/data/MarketData-20260225T215136Z-1-001.zip \
  --extract-dir /mnt/data/MarketData_unzipped \
  --target-year 2024
```

What it does:
- extracts the bundle,
- chooses preferred partitions per symbol (2024 + `1Min` first),
- normalizes to canonical columns with UTC `timestamp`,
- builds `data/alpaca/clean_build_tmp_<ts>/`,
- swaps atomically to `data/alpaca/clean/` and keeps `data/alpaca/clean_backup_<ts>/`.

Build manifest:
- `artifacts/clean_cache_build/<ts>/build_manifest.json`

## 11) DQ-Aware Symbol Selection

`scripts/run_sweep_auto.py` now ranks symbols by DQ reliability before raw row count:
1. `coverage_ratio = (ACCEPT + DEGRADE) / total_days` (desc)
2. `median_dqs` (desc)
3. `quality_score` (desc)
4. symbol (asc)

Selection evidence is written to:
- `artifacts/sweep_v2/<ts>/selection_dq_table.csv`
