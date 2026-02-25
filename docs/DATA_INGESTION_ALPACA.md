# Data Ingestion: Alpaca Minute Bars

This repository includes an institutional-grade minute-bar ingestion pipeline for Alpaca Market Data API v2.

## 1) Environment Variables

Set Alpaca credentials in your shell (never commit credentials):

```bash
export ALPACA_API_KEY="<your_key>"
export ALPACA_SECRET_KEY="<your_secret>"
```

The names are configurable in `configs/data_alpaca.yaml` via:
- `alpaca.api_key_env`
- `alpaca.secret_key_env`

Runtime enforcement is fail-closed on fixed env vars only:
- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`

## 2) Run the Fetch Pipeline

```bash
python scripts/fetch_alpaca_data.py --config ./configs/data_alpaca.yaml
```

The pipeline will:
1. Call Alpaca `/v2/stocks/bars` with pagination and symbol chunking.
2. Run a pinned-feed preflight check on the first symbol in sorted symbol order.
3. Retry HTTP 429 (rate-limit) with deterministic exponential backoff.
4. Use deterministic safe-limit fallback (`10000 -> 1000`) only on narrow invalid-limit errors.
5. Canonicalize and clean minute bars deterministically into staging.
6. Promote to official clean cache only for symbols that pass configured QA policy.

## 2.2) Historical Entitlement Preflight

Before the symbol loop, ingestion probes two tiny windows for the same symbol:
- recent window (last ~45 minutes, UTC)
- historical window (`start` to `min(start+1 day, end)`)

If recent returns bars but historical returns zero bars, ingestion aborts early with:
- `historical_data_not_available_for_requested_range`
- `preflight_entitlement_class = historical_denied_or_limited`

This usually indicates account plan limits on historical data access. Actionable fix:
- upgrade Alpaca plan (e.g. Algo Trader Plus) or switch provider/feed entitlement.

If both windows return zero bars, ingestion aborts with:
- `preflight_no_data_recent_or_historical`

All preflight diagnostics are written into QA/manifest:
- `preflight_recent_start/end`
- `preflight_historical_start/end`
- `preflight_recent_bar_count`
- `preflight_historical_bar_count`
- `preflight_entitlement_class`
- `preflight_actionable_fix`

## 2.1) Operator Runbook (Canary -> Universe10 3Y)

```bash
export ALPACA_API_KEY="<your_key>"
export ALPACA_SECRET_KEY="<your_secret>"

# DST canary first
python scripts/fetch_alpaca_data.py --config ./configs/data_alpaca_canary_dst_both.yaml

# Full 10-asset / 3-year pull
python scripts/fetch_alpaca_data.py --config ./configs/data_alpaca_universe10_3y.yaml
```

Review reports at:
- `data/alpaca/reports/<RUN_ID>_qa.json`
- `data/alpaca/reports/<RUN_ID>_manifest.json`

Acceptance thresholds:
- `coverage_pct >= 95`
- `missing_minutes_pct <= 5`
- `invariants_ok == true`

## 3) Output Layout

Given `storage.root: ./data/alpaca`:

- Raw fetched payloads (per symbol / request window):
  - `data/alpaca/raw/<SYMBOL>/<START>_<END>_<FEED>_<ADJUSTMENT>.parquet`
- Staging clean outputs (always written when cleaning succeeds):
  - `data/alpaca/clean_staging/<SYMBOL>.parquet`
- Clean canonical cache (for `run_research.py`):
  - `data/alpaca/clean/<SYMBOL>.parquet` (written only after QA pass)
- Per-run QA report:
  - `data/alpaca/reports/<RUN_ID>_qa.json`
- Per-run manifest report:
  - `data/alpaca/reports/<RUN_ID>_manifest.json`

`RUN_ID` format is hash-anchored: `run_<YYYYMMDD>_<config_sha256_prefix>`.

## 4) Canonical Clean Schema

All clean files use:
- `timestamp` (UTC tz-aware)
- `open`
- `high`
- `low`
- `close`
- `volume`

## 5) Cleaning Rules (Deterministic)

1. Parse timestamps as UTC and preserve raw intraminute timestamp (`ts_raw`) before flooring.
2. Build canonical minute key as `timestamp = floor(ts_raw, "min")`.
3. Deterministically sort by `(timestamp, ts_raw)` using stable sort.
4. Deduplicate by minute with aggregation:
   - `open=first`
   - `close=last`
   - `high=max`
   - `low=min`
   - `volume=sum`
5. Drop transient `ts_raw` after minute aggregation (clean output remains canonical OHLCV only).
   - `ts_raw` is required to deterministically order intraminute duplicates for `open=first` / `close=last`.
   - If duplicate minute timestamps are present without `ts_raw`, cleaning fails closed with a runtime error.
6. Enforce OHLC invariants and `volume >= 0`.
   - Invalid rows are dropped (no heuristic repairs).
7. Enforce strictly increasing unique timestamp index.
8. Convert to exchange time (`America/New_York`) for session filtering and QA fields.

## 6) Session Policy

Configured under `alpaca.session_policy`:
- `RTH`: keep bars between `rth_open` and `rth_close` in exchange time.
- `ETH`: keep all minutes.

`rth_close_inclusive` controls whether the close minute is included:
- `false` (default): minute bars treat close time as bar-end boundary.
  - Example: `rth_close: "16:00"` includes through `15:59`, excludes `16:00`.
- `true`: include `16:00` itself.

Why default `false`: most 1-minute market datasets model the last regular bar as the minute ending at 16:00 (timestamped 15:59), and this avoids accidentally admitting auction/post-market prints.

DST handling is zoneinfo/pandas tz-conversion based and tested.

## 7) Coverage QA Modes

Configure `alpaca.calendar_mode`:
- `naive` (default behavior in code): expected minutes are fixed-width by policy and observed sessions.
- `nyse` (recommended): expected minutes are computed from NYSE trading sessions, including half-days and holidays.
  - In NYSE mode, session close is treated as a boundary; expected-minute counting uses `close - 1 minute` as the last tradable minute.
  - `rth_close_inclusive` is retained in metadata for traceability, but does not expand NYSE expected minutes.

`nyse` mode dependency:

```bash
pip install exchange-calendars
```

If `calendar_mode: "nyse"` is set and dependency is missing, ingestion fails closed.

## 7.1) QA Policy Modes

Configure:
- `alpaca.qa_policy: "coverage_threshold" | "strict_no_holes"`
- `alpaca.coverage_min_pct` (default `99.0`)

Default (`coverage_threshold`) requirements:
- `invariants_ok == true`
- `coverage_pct >= coverage_min_pct`
- `missing_minutes_pct <= (100 - coverage_min_pct)`

Strict mode (`strict_no_holes`) requirements:
- `invariants_ok == true`
- `missing_minutes_total == 0`

Notes:
- Missing minutes are **not fabricated** in clean output.
- Downstream harness alignment should mark unavailable bars with validity flags.
- QA reports include bounded missing-minute previews (`missing_minutes_preview`, first 200).

## 8) QA Report Fields

Top-level report fields include (both QA and manifest):
- `source` (`alpaca`)
- `base_url`
- `feed`
- `adjustment`
- `timeframe`
- `sort`
- `limit_requested`
- `limit_effective`
- `calendar_expectations_available`

Each symbol section includes:
- `raw_rows`
- `parsed_rows`
- `duplicate_rows_collapsed`
- `rows_after_session_policy`
- `rows_dropped_invalid`
- `rows_final`
- `invariants_ok`
- session subfields:
  - `qa_mode` (`naive` or `nyse`)
  - `expected_minutes_total`
  - `observed_minutes_total`
  - `missing_minutes_total`
  - `coverage_pct`
  - `missing_minutes_pct`
  - `coverage_pct_naive` / `missing_minutes_pct_naive` (present in naive mode)
  - `missing_minutes`
  - `n_sessions`
  - `expected_minutes`
  - `post_clean` strict checks:
    - no NaNs in canonical columns
    - UTC tz-aware strict monotonic unique timestamps
    - RTH boundary correctness (15:59 last minute for `rth_close=16:00`, exclusive mode)
    - policy-driven coverage gate (`coverage_threshold` or `strict_no_holes`)

Preflight diagnostics are recorded in both reports:
- `preflight_symbol`
- `preflight_start`
- `preflight_end`
- `preflight_status_code`
- `preflight_error_class`
- `preflight_error_msg`

Recommended thresholds for review:
- `invariants_ok == true`
- `coverage_pct >= 95` for stable universe symbols
- `missing_minutes_pct <= 5`

## 9) Integration with run_research.py

Point your research config to the clean cache:

```yaml
data:
  root: "./data/alpaca/clean"
  format: "parquet"
  path_by_symbol:
    SPY: "SPY.parquet"
    QQQ: "QQQ.parquet"
```

The existing runner will perform in-memory date filtering and pass normalized bars into the harness.
