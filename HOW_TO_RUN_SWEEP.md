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
