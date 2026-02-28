# PART4_DEADLETTER_POLICY_DECISION

- RUN_DIR: `/Users/afekshusterman/Documents/New project/artifacts/module5_harness/run_20260228_204134`
- deadletter_rows: 4

## Counts and percentages
- CLASS_A_EXPECTED_HOLE: 4 (100.0000%)

## Per symbol counts
```
policy_class  CLASS_A_EXPECTED_HOLE
asset_symbol                       
TLT                               2
XLK                               2
```

## Example rows CLASS_A_EXPECTED_HOLE (up to 5)
```
          reason_code asset_symbol               ts_fill_utc  dq_issue_flags_fill  dq_filled_bar_fill  dqs_day_fill  cache_open_fill  bar_valid_fill  open_px_fill          policy_class
NEXT_OPEN_UNAVAILABLE          XLK 2024-11-21 20:31:00+00:00                    0               False           1.0       234.460007           False           NaN CLASS_A_EXPECTED_HOLE
NEXT_OPEN_UNAVAILABLE          XLK 2024-11-21 20:31:00+00:00                    0               False           1.0       234.460007           False           NaN CLASS_A_EXPECTED_HOLE
NEXT_OPEN_UNAVAILABLE          TLT 2024-12-02 14:46:00+00:00                    0               False           1.0        93.139999           False           NaN CLASS_A_EXPECTED_HOLE
NEXT_OPEN_UNAVAILABLE          TLT 2024-12-02 14:46:00+00:00                    0               False           1.0        93.139999           False           NaN CLASS_A_EXPECTED_HOLE
```

## Example rows CLASS_B_DANGEROUS_MISMATCH (up to 5)
- none

## Decision
Only CLASS A observed. Proceed to KPI gate.