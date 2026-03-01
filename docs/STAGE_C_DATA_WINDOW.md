# Stage C Data Window

## Source Universe (from strict baseline)

EEM, GLD, HYG, IWM, QQQ, SPY, TLT, XLE, XLK, XLU

## Per-Symbol Coverage Table

```text
symbol                                                           cache_path_used      ts_col_mode   method_used       first_timestamp_utc        last_timestamp_utc  coverage_days  approx_row_count
   EEM /Users/afekshusterman/Documents/New project/data/alpaca/clean/EEM.parquet column:timestamp pyarrow_stats 2024-01-02 14:31:00+00:00 2024-12-31 21:00:00+00:00            364             97580
   GLD /Users/afekshusterman/Documents/New project/data/alpaca/clean/GLD.parquet column:timestamp pyarrow_stats 2024-01-02 14:31:00+00:00 2024-12-31 21:00:00+00:00            364             97703
   HYG /Users/afekshusterman/Documents/New project/data/alpaca/clean/HYG.parquet column:timestamp pyarrow_stats 2024-01-02 14:31:00+00:00 2024-12-31 21:00:00+00:00            364             97689
   IWM /Users/afekshusterman/Documents/New project/data/alpaca/clean/IWM.parquet column:timestamp pyarrow_stats 2024-01-02 14:31:00+00:00 2024-12-31 21:00:00+00:00            364             97740
   QQQ /Users/afekshusterman/Documents/New project/data/alpaca/clean/QQQ.parquet column:timestamp pyarrow_stats 2024-01-02 14:31:00+00:00 2024-12-31 21:00:00+00:00            364             97740
   SPY /Users/afekshusterman/Documents/New project/data/alpaca/clean/SPY.parquet column:timestamp pyarrow_stats 2024-01-02 14:31:00+00:00 2024-12-31 21:00:00+00:00            364             97740
   TLT /Users/afekshusterman/Documents/New project/data/alpaca/clean/TLT.parquet column:timestamp pyarrow_stats 2024-01-02 14:31:00+00:00 2024-12-31 21:00:00+00:00            364             97740
   XLE /Users/afekshusterman/Documents/New project/data/alpaca/clean/XLE.parquet column:timestamp pyarrow_stats 2024-01-02 14:31:00+00:00 2024-12-31 21:00:00+00:00            364             97740
   XLK /Users/afekshusterman/Documents/New project/data/alpaca/clean/XLK.parquet column:timestamp pyarrow_stats 2024-01-02 14:31:00+00:00 2024-12-31 21:00:00+00:00            364             97722
   XLU /Users/afekshusterman/Documents/New project/data/alpaca/clean/XLU.parquet column:timestamp pyarrow_stats 2024-01-02 14:31:00+00:00 2024-12-31 21:00:00+00:00            364             97739
```

## Bottleneck Protection

No symbols dropped by Gate A/B bottleneck rule.

## Final Stage C Universe

EEM, GLD, HYG, IWM, QQQ, SPY, TLT, XLE, XLK, XLU

## Final Window

- target_start: `2024-01-02T14:31:00+00:00`
- target_end: `2024-12-31T21:00:00+00:00`
- window_days: `364`
- trading_sessions_available (common NY sessions): `252`
- minimum_required_sessions: `60`
- sufficiency: `PASS`