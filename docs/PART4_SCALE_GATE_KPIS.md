# PART4_SCALE_GATE_KPIS

- RUN_DIR: `/Users/afekshusterman/Documents/New project/artifacts/module5_harness/run_20260228_204134`
- total_deadletters: 4
- total_bars_used_for_rate: 195462
- deadletters_per_1M_bars: 20.464336

## Percent by reason_code
- NEXT_OPEN_UNAVAILABLE: 100.0000%

## Per-symbol deadletter rate
```
              deadletters   bars  deadletters_per_1M_bars
asset_symbol                                             
TLT                     2  97740                20.462451
XLK                     2  97722                20.466221
```

## Join coverage
- pct_dq_join_ok_signal: 100.0000%
- pct_dq_join_ok_fill: 100.0000%
- pct_cache_join_ok_signal: 100.0000%
- pct_cache_join_ok_fill: 100.0000%

## Decision rule
- If any CLASS_B -> STOP
- Else if deadletters per 1M bars <= 50 -> OK to proceed
- Else -> OK with monitoring

## Recommendation
- recommendation: **PROCEED**
- rationale: 20.464336 <= 50.0