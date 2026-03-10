# Audit README (Sealed-Spec Validation + Factory Checks)

This repo includes deterministic audit tests for sealed kernel behavior.

## What the tests prove

1. **Value Area Rule (greedy expansion)**
   - Confirms value-area expansion follows the sealed deterministic side-selection rule.
   - Confirms tie-break order is deterministic.

2. **Sigma Floor Invariant in sealed mode**
   - Confirms `sigma1 >= dx` and `sigma2 >= dx` at required bars.

3. **Determinism replay**
   - Running Module 2 twice with identical inputs/seeds yields identical state digest.

4. **Prefix-invariance / no-lookahead**
   - If bars after `t0` are modified, all Module 2 outputs up to `t0` remain identical.
   - This directly tests causal computation guarantees.

5. **DST boundary sanity (zoneinfo path)**
   - Confirms ET open alignment across DST start/end transitions.
   - Confirms session IDs increment on ET local-date changes (no clock drift behavior).

6. **Strategy factory acceptance**
   - Confirms a sweep config with `1x10x10=100` variants writes exactly 100 candidate folders.
   - Confirms `verdict.json` leaderboard is candidate-level (not task-level).
   - Confirms deterministic replay produces identical `resolved_config_sha256` and identical
     `robustness_leaderboard.csv` bytes.

## What the tests do not prove

- They do not prove the strategy has positive expected alpha.
- They do not guarantee production market data quality (missing/corporate-action issues still depend on ingestion QA).
- They do not validate execution microstructure assumptions beyond modeled slippage/cost terms.

## How to run

```bash
./.venv/bin/python -m unittest tests.test_module2_core tests.test_timezone_dst
```

For the full suite (if environment is fully configured):

```bash
./.venv/bin/python -m pytest -q
```
