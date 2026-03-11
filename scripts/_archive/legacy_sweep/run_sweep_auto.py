#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    raise RuntimeError("pandas is required. Install with: pip install pandas") from exc

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise RuntimeError("pyyaml is required. Install with: pip install pyyaml") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = "configs/sweep_20x100.yaml"
DEFAULT_SCOUT_CONFIG = "configs/sweep_scout_20x12.yaml"
DEFAULT_CLEAN_DIR = "data/alpaca/clean"
QUICK_RUN_SYMBOLS = 2
QUICK_RUN_MAX_CANDIDATES = 3

TS_ALIASES = ("timestamp", "ts", "datetime", "date", "time")
OPEN_ALIASES = ("open", "o")
HIGH_ALIASES = ("high", "h")
LOW_ALIASES = ("low", "l")
CLOSE_ALIASES = ("close", "c")
VOLUME_ALIASES = ("volume", "vol", "v")


@dataclass(frozen=True)
class SymbolInventory:
    symbol: str
    path: str
    file_size_bytes: int
    row_count: int
    detected_columns: str
    has_required_aliases: bool
    nan_rate_ohlcv: float
    duplicate_timestamp_count: int
    monotonic_timestamp_ok: bool
    quality_score: float
    excluded: bool
    exclusion_reason: str


@dataclass(frozen=True)
class SymbolDQProbe:
    symbol: str
    coverage_ratio: float
    median_dqs: float
    reject_ratio: float
    total_days: int
    accept_days: int
    degrade_days: int
    reject_days: int
    probe_error: str


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_yaml_mapping(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise RuntimeError(f"Config root must be an object/mapping: {path}")
    return raw


def _dump_yaml_deterministic(payload: Dict[str, Any]) -> str:
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_inventory_csv(path: Path, rows: Sequence[SymbolInventory]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "symbol",
        "path",
        "file_size_bytes",
        "row_count",
        "detected_columns",
        "has_required_aliases",
        "nan_rate_ohlcv",
        "duplicate_timestamp_count",
        "monotonic_timestamp_ok",
        "quality_score",
        "excluded",
        "exclusion_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(rows, key=lambda x: x.symbol):
            w.writerow(
                {
                    "symbol": r.symbol,
                    "path": r.path,
                    "file_size_bytes": int(r.file_size_bytes),
                    "row_count": int(r.row_count),
                    "detected_columns": r.detected_columns,
                    "has_required_aliases": bool(r.has_required_aliases),
                    "nan_rate_ohlcv": float(r.nan_rate_ohlcv),
                    "duplicate_timestamp_count": int(r.duplicate_timestamp_count),
                    "monotonic_timestamp_ok": bool(r.monotonic_timestamp_ok),
                    "quality_score": float(r.quality_score),
                    "excluded": bool(r.excluded),
                    "exclusion_reason": r.exclusion_reason,
                }
            )


def _write_selection_dq_csv(path: Path, rows: Sequence[SymbolDQProbe]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "symbol",
        "coverage_ratio",
        "median_dqs",
        "reject_ratio",
        "total_days",
        "accept_days",
        "degrade_days",
        "reject_days",
        "probe_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(rows, key=lambda x: x.symbol):
            w.writerow(
                {
                    "symbol": r.symbol,
                    "coverage_ratio": float(r.coverage_ratio),
                    "median_dqs": float(r.median_dqs),
                    "reject_ratio": float(r.reject_ratio),
                    "total_days": int(r.total_days),
                    "accept_days": int(r.accept_days),
                    "degrade_days": int(r.degrade_days),
                    "reject_days": int(r.reject_days),
                    "probe_error": str(r.probe_error),
                }
            )


def _safe_num_rows_and_columns(path: Path) -> Tuple[int, List[str]]:
    try:
        import pyarrow.parquet as pq  # type: ignore

        pf = pq.ParquetFile(path)
        n_rows = int(pf.metadata.num_rows)
        cols = [str(x) for x in pf.schema.names]
        return n_rows, cols
    except Exception:
        df = pd.read_parquet(path)
        return int(len(df)), [str(c) for c in df.columns]


def _read_sample(path: Path, columns: Sequence[str], sample_rows: int) -> Any:
    unique_cols = list(dict.fromkeys([str(c) for c in columns]))
    try:
        import pyarrow.parquet as pq  # type: ignore

        pf = pq.ParquetFile(path)
        batches = pf.iter_batches(batch_size=max(1, int(sample_rows)), columns=unique_cols)
        first = next(batches, None)
        if first is None:
            return pd.DataFrame(columns=unique_cols)
        return first.to_pandas()
    except Exception:
        df = pd.read_parquet(path, columns=unique_cols)
        if len(df) > int(sample_rows):
            return df.iloc[: int(sample_rows)].copy()
        return df


def _find_col_name(columns: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    lookup: Dict[str, str] = {}
    for raw in columns:
        key = str(raw).strip().lower()
        if key not in lookup:
            lookup[key] = str(raw)
    for alias in aliases:
        if alias in lookup:
            return lookup[alias]
    return None


def _resolve_required_mapping(columns: Sequence[str]) -> Tuple[Dict[str, str], List[str]]:
    mapping = {
        "timestamp": _find_col_name(columns, TS_ALIASES),
        "open": _find_col_name(columns, OPEN_ALIASES),
        "high": _find_col_name(columns, HIGH_ALIASES),
        "low": _find_col_name(columns, LOW_ALIASES),
        "close": _find_col_name(columns, CLOSE_ALIASES),
        "volume": _find_col_name(columns, VOLUME_ALIASES),
    }
    missing = sorted([k for k, v in mapping.items() if v is None])
    return {k: str(v) for k, v in mapping.items() if v is not None}, missing


def compute_quality_score(
    row_count: int,
    nan_rate_ohlcv: float,
    duplicate_timestamp_count: int,
    monotonic_timestamp_ok: bool,
) -> float:
    # Deterministic rule tuned to prefer completeness and clean canonical shape.
    # Higher is better.
    return (
        float(max(int(row_count), 0))
        - 100000.0 * float(max(nan_rate_ohlcv, 0.0))
        - 100.0 * float(max(int(duplicate_timestamp_count), 0))
        - (0.0 if bool(monotonic_timestamp_ok) else 50000.0)
    )


def inspect_symbol_file(path: Path, sample_rows: int = 20000) -> SymbolInventory:
    symbol = path.stem.strip().upper()
    size_bytes = int(path.stat().st_size)

    try:
        row_count, columns = _safe_num_rows_and_columns(path)
    except Exception as exc:
        return SymbolInventory(
            symbol=symbol,
            path=str(path.resolve()),
            file_size_bytes=size_bytes,
            row_count=0,
            detected_columns="",
            has_required_aliases=False,
            nan_rate_ohlcv=1.0,
            duplicate_timestamp_count=0,
            monotonic_timestamp_ok=False,
            quality_score=float("-inf"),
            excluded=True,
            exclusion_reason=f"read_failure:{type(exc).__name__}",
        )

    mapping, missing = _resolve_required_mapping(columns)
    detected_columns = json.dumps(mapping, sort_keys=True)
    if missing:
        return SymbolInventory(
            symbol=symbol,
            path=str(path.resolve()),
            file_size_bytes=size_bytes,
            row_count=int(row_count),
            detected_columns=detected_columns,
            has_required_aliases=False,
            nan_rate_ohlcv=1.0,
            duplicate_timestamp_count=0,
            monotonic_timestamp_ok=False,
            quality_score=float("-inf"),
            excluded=True,
            exclusion_reason=f"missing_required_columns:{','.join(missing)}",
        )

    try:
        sample_cols = [
            mapping["timestamp"],
            mapping["open"],
            mapping["high"],
            mapping["low"],
            mapping["close"],
            mapping["volume"],
        ]
        sample_df = _read_sample(path, sample_cols, sample_rows=max(1, int(sample_rows)))
    except Exception as exc:
        return SymbolInventory(
            symbol=symbol,
            path=str(path.resolve()),
            file_size_bytes=size_bytes,
            row_count=int(row_count),
            detected_columns=detected_columns,
            has_required_aliases=True,
            nan_rate_ohlcv=1.0,
            duplicate_timestamp_count=0,
            monotonic_timestamp_ok=False,
            quality_score=float("-inf"),
            excluded=True,
            exclusion_reason=f"sample_read_failure:{type(exc).__name__}",
        )

    ts = pd.to_datetime(sample_df[mapping["timestamp"]], utc=True, errors="coerce")
    valid_ts = ts[ts.notna()]
    if valid_ts.shape[0] >= 2:
        ts_ns = valid_ts.astype("int64").to_numpy()
        diffs = ts_ns[1:] - ts_ns[:-1]
        monotonic_ok = bool((diffs > 0).all())
    else:
        monotonic_ok = True

    dup_count = int(valid_ts.duplicated(keep=False).sum())

    ohlcv_na = 0
    cells = 0
    for key in ("open", "high", "low", "close", "volume"):
        vals = pd.to_numeric(sample_df[mapping[key]], errors="coerce")
        ohlcv_na += int(vals.isna().sum())
        cells += int(vals.shape[0])
    nan_rate = float(ohlcv_na / cells) if cells > 0 else 1.0

    excluded = bool(row_count <= 0)
    reason = "empty_file" if excluded else ""
    score = compute_quality_score(
        row_count=int(row_count),
        nan_rate_ohlcv=nan_rate,
        duplicate_timestamp_count=dup_count,
        monotonic_timestamp_ok=monotonic_ok,
    )

    return SymbolInventory(
        symbol=symbol,
        path=str(path.resolve()),
        file_size_bytes=size_bytes,
        row_count=int(row_count),
        detected_columns=detected_columns,
        has_required_aliases=True,
        nan_rate_ohlcv=nan_rate,
        duplicate_timestamp_count=dup_count,
        monotonic_timestamp_ok=monotonic_ok,
        quality_score=score,
        excluded=excluded,
        exclusion_reason=reason,
    )


def build_inventory(clean_dir: Path) -> List[SymbolInventory]:
    files = sorted(clean_dir.glob("*.parquet"), key=lambda p: p.stem.strip().upper())
    return [inspect_symbol_file(p) for p in files]


def _coerce_frame_for_dq(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if isinstance(df.index, pd.DatetimeIndex):
        work = df.copy()
        work.index = pd.to_datetime(work.index, utc=True, errors="coerce")
        work = work[~work.index.isna()].copy()
        return work

    mapping, missing = _resolve_required_mapping([str(c) for c in df.columns])
    if missing:
        raise RuntimeError(f"missing_required_columns:{','.join(missing)}")
    work = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df[mapping["timestamp"]], utc=True, errors="coerce"),
            "open": pd.to_numeric(df[mapping["open"]], errors="coerce"),
            "high": pd.to_numeric(df[mapping["high"]], errors="coerce"),
            "low": pd.to_numeric(df[mapping["low"]], errors="coerce"),
            "close": pd.to_numeric(df[mapping["close"]], errors="coerce"),
            "volume": pd.to_numeric(df[mapping["volume"]], errors="coerce"),
        }
    )
    work = work.dropna(subset=["timestamp"]).copy()
    work = work.set_index("timestamp")
    return work


def probe_symbol_dq_metrics(
    symbol: str,
    path: Path,
    *,
    tz_name: str,
    session_open_minute: int,
    session_close_minute: int,
    probe_year_fallback: int = 2024,
    probe_start_utc: Optional[pd.Timestamp] = None,
    probe_end_utc: Optional[pd.Timestamp] = None,
) -> SymbolDQProbe:
    try:
        repo_str = str(REPO_ROOT)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
        from weightiz.shared.validation.dq import DQ_ACCEPT, DQ_DEGRADE, DQ_REJECT, dq_validate
    except Exception as exc:
        return SymbolDQProbe(
            symbol=str(symbol),
            coverage_ratio=-1.0,
            median_dqs=-1.0,
            reject_ratio=1.0,
            total_days=0,
            accept_days=0,
            degrade_days=0,
            reject_days=0,
            probe_error=f"dq_import_failure:{type(exc).__name__}",
        )

    try:
        frame = _coerce_frame_for_dq(path)
        if frame.empty:
            raise RuntimeError("empty_frame")
        if probe_start_utc is not None:
            frame = frame.loc[frame.index >= probe_start_utc].copy()
        if probe_end_utc is not None:
            frame = frame.loc[frame.index <= probe_end_utc].copy()
        if (probe_start_utc is not None) or (probe_end_utc is not None):
            if frame.empty:
                start_s = str(probe_start_utc) if probe_start_utc is not None else "None"
                end_s = str(probe_end_utc) if probe_end_utc is not None else "None"
                raise RuntimeError(f"no_rows_for_range:{start_s}..{end_s}")
        else:
            ts_local = pd.DatetimeIndex(frame.index).tz_convert(tz_name)
            mask_year = ts_local.year == int(probe_year_fallback)
            frame = frame.loc[mask_year].copy()
            if frame.empty:
                raise RuntimeError(f"no_rows_for_year:{probe_year_fallback}")

        reports = dq_validate(
            df=frame,
            symbol=str(symbol),
            tz_name=str(tz_name),
            session_open_minute=int(session_open_minute),
            session_close_minute=int(session_close_minute),
            timeframe_min=None,
        )
        total = int(len(reports))
        if total <= 0:
            raise RuntimeError("dq_reports_empty")
        accept_days = int(sum(1 for r in reports if str(r.decision) == DQ_ACCEPT))
        degrade_days = int(sum(1 for r in reports if str(r.decision) == DQ_DEGRADE))
        reject_days = int(sum(1 for r in reports if str(r.decision) == DQ_REJECT))
        coverage = float((accept_days + degrade_days) / max(total, 1))
        reject_ratio = float(reject_days / max(total, 1))
        dqs_vals = [float(r.dqs_final) for r in reports if np.isfinite(float(r.dqs_final))]
        median_dqs = float(np.median(np.asarray(dqs_vals, dtype=np.float64))) if dqs_vals else 0.0
        return SymbolDQProbe(
            symbol=str(symbol),
            coverage_ratio=float(coverage),
            median_dqs=float(median_dqs),
            reject_ratio=float(reject_ratio),
            total_days=int(total),
            accept_days=int(accept_days),
            degrade_days=int(degrade_days),
            reject_days=int(reject_days),
            probe_error="",
        )
    except Exception as exc:
        return SymbolDQProbe(
            symbol=str(symbol),
            coverage_ratio=-1.0,
            median_dqs=-1.0,
            reject_ratio=1.0,
            total_days=0,
            accept_days=0,
            degrade_days=0,
            reject_days=0,
            probe_error=f"probe_failure:{type(exc).__name__}",
        )


def _parse_optional_utc_timestamp(value: Any) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def _selection_probe_window(base_cfg: Dict[str, Any]) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], int]:
    data = base_cfg.get("data", {})
    start_ts: Optional[pd.Timestamp] = None
    end_ts: Optional[pd.Timestamp] = None
    if isinstance(data, dict):
        start_ts = _parse_optional_utc_timestamp(data.get("start"))
        end_ts = _parse_optional_utc_timestamp(data.get("end"))

    if (start_ts is not None) and (end_ts is not None) and (start_ts > end_ts):
        start_ts, end_ts = None, None

    if start_ts is not None:
        probe_year = int(start_ts.year)
    elif end_ts is not None:
        probe_year = int(end_ts.year)
    else:
        probe_year = 2024
    return start_ts, end_ts, probe_year


def select_symbols_deterministic(
    inventory_rows: Sequence[SymbolInventory],
    target_symbols: int,
    min_symbols: int,
    dq_metrics_by_symbol: Optional[Dict[str, SymbolDQProbe]] = None,
) -> Tuple[List[str], List[SymbolInventory], Dict[str, str]]:
    valid = [r for r in inventory_rows if not r.excluded and r.has_required_aliases]
    excluded = {r.symbol: r.exclusion_reason for r in inventory_rows if r.excluded}

    if len(valid) < int(min_symbols):
        raise RuntimeError(
            "Insufficient valid symbols in clean cache: "
            f"valid={len(valid)} < min_symbols={int(min_symbols)}. "
            f"Populate {DEFAULT_CLEAN_DIR} with more canonical parquet files."
        )

    dq_map = dq_metrics_by_symbol or {}

    def _rank_key(r: SymbolInventory) -> Tuple[float, float, float, str]:
        dq = dq_map.get(r.symbol)
        if dq is None:
            coverage = -1.0
            median_dqs = -1.0
        else:
            coverage = float(dq.coverage_ratio)
            median_dqs = float(dq.median_dqs)
        return (-coverage, -median_dqs, -float(r.quality_score), str(r.symbol))

    ranked = sorted(valid, key=_rank_key)
    if len(ranked) >= int(target_symbols):
        selected_ranked = ranked[: int(target_symbols)]
    else:
        selected_ranked = ranked

    chosen_sorted = sorted([r.symbol for r in selected_ranked])
    return chosen_sorted, ranked, excluded


def _replace_symbols_in_config(base_cfg: Dict[str, Any], symbols_sorted: Sequence[str]) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    if "symbols" not in cfg:
        raise RuntimeError("Config is missing required 'symbols' key")
    cfg["symbols"] = [str(s) for s in symbols_sorted]
    return cfg


def _apply_sweep_abort_guardrails(cfg: Dict[str, Any], *, force_serial: bool = False) -> Dict[str, Any]:
    """
    Prevent scout/full runs from aborting after the first failing group.
    This keeps Sweep v2 exploration deterministic even when one m3 slice is unhealthy.
    """
    out = copy.deepcopy(cfg)
    harness = out.get("harness")
    if not isinstance(harness, dict):
        return out

    rate_prev = harness.get("failure_rate_abort_threshold")
    count_prev = harness.get("failure_count_abort_threshold")
    harness["failure_rate_abort_threshold"] = 1.0
    harness["failure_count_abort_threshold"] = max(int(count_prev or 0), 1_000_000)
    if bool(force_serial):
        # Quick-run guardrail: keep execution strictly serial and bounded.
        harness["parallel_backend"] = "serial"
        harness["parallel_workers"] = 1
    out["harness"] = harness
    return out


def _reduce_auto_grid_for_quick_run(cfg: Dict[str, Any], max_candidates: int) -> int:
    m2 = cfg.get("module2_configs")
    m3 = cfg.get("module3_configs")
    m4 = cfg.get("module4_configs")
    if not isinstance(m2, list) or not isinstance(m3, list) or not isinstance(m4, list):
        raise RuntimeError("Quick-run requires module2_configs/module3_configs/module4_configs lists")
    if not m2 or not m3 or not m4:
        raise RuntimeError("Quick-run cannot operate with empty module config lists")

    n2 = min(len(m2), 1)
    n3 = min(len(m3), 2)
    n4 = min(len(m4), 2)
    while n2 * n3 * n4 > int(max_candidates):
        if n4 > 1:
            n4 -= 1
        elif n3 > 1:
            n3 -= 1
        elif n2 > 1:
            n2 -= 1
        else:
            break

    cfg["module2_configs"] = copy.deepcopy(m2[:n2])
    cfg["module3_configs"] = copy.deepcopy(m3[:n3])
    cfg["module4_configs"] = copy.deepcopy(m4[:n4])
    return int(n2 * n3 * n4)


def _apply_quick_run_reduction(
    cfg: Dict[str, Any],
    chosen_symbols: Sequence[str],
    quick_symbol_count: int = QUICK_RUN_SYMBOLS,
    max_candidates: int = QUICK_RUN_MAX_CANDIDATES,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if int(quick_symbol_count) < 2:
        raise RuntimeError("quick_symbol_count must be >= 2 due to run config schema")
    chosen_sorted = sorted([str(s).strip().upper() for s in chosen_symbols])
    if len(chosen_sorted) < int(quick_symbol_count):
        raise RuntimeError(
            f"Quick-run requires at least {int(quick_symbol_count)} symbols; got {len(chosen_sorted)}"
        )
    quick_symbols = chosen_sorted[: int(quick_symbol_count)]
    out = _replace_symbols_in_config(cfg, quick_symbols)

    policy: Dict[str, Any] = {
        "quick_symbols": quick_symbols,
        "candidate_cap": int(max_candidates),
    }

    candidates = out.get("candidates")
    mode = "auto_grid"
    if isinstance(candidates, dict):
        mode = str(candidates.get("mode", "auto_grid")).strip().lower()
    if mode == "manual" and isinstance(candidates, dict):
        specs = candidates.get("specs", [])
        if isinstance(specs, list):
            specs_sorted = sorted(specs, key=lambda x: str((x or {}).get("candidate_id", "")))
            candidates["specs"] = copy.deepcopy(specs_sorted[:2])
            policy["quick_candidate_mode"] = "manual"
            policy["quick_manual_specs_kept"] = int(len(candidates["specs"]))
        else:
            candidates["specs"] = []
            policy["quick_candidate_mode"] = "manual"
            policy["quick_manual_specs_kept"] = 0
        out["candidates"] = candidates
    else:
        est = _reduce_auto_grid_for_quick_run(out, max_candidates=int(max_candidates))
        cobj = out.get("candidates")
        if isinstance(cobj, dict):
            cobj["mode"] = "auto_grid"
            cobj["specs"] = []
            out["candidates"] = cobj
        else:
            out["candidates"] = {"mode": "auto_grid", "specs": []}
        policy["quick_candidate_mode"] = "auto_grid"
        policy["quick_auto_grid_estimated_candidates"] = int(est)

    harness = out.get("harness")
    if isinstance(harness, dict):
        if "parallel_backend" in harness:
            harness["parallel_backend"] = "serial"
        if "parallel_workers" in harness:
            harness["parallel_workers"] = 1
        if "wf_train_sessions" in harness:
            harness["wf_train_sessions"] = 1
        if "wf_test_sessions" in harness:
            harness["wf_test_sessions"] = 1
        if "wf_step_sessions" in harness:
            harness["wf_step_sessions"] = 1
        if "purge_bars" in harness:
            harness["purge_bars"] = 0
        if "embargo_bars" in harness:
            harness["embargo_bars"] = 0
        if "min_asset_coverage" in harness:
            harness["min_asset_coverage"] = 0.0
        if "daily_return_min_days" in harness:
            harness["daily_return_min_days"] = 1
        if "cpcv_slices" in harness:
            harness["cpcv_slices"] = 2
        if "cpcv_k_test" in harness:
            harness["cpcv_k_test"] = 1
        out["harness"] = harness

    baseline_enabled: List[str] = []
    scenarios = out.get("stress_scenarios")
    if isinstance(scenarios, list) and scenarios:
        baseline_found = False
        for s in scenarios:
            if not isinstance(s, dict):
                continue
            sid = str(s.get("scenario_id", "")).strip().lower()
            is_baseline = sid == "baseline"
            s["enabled"] = bool(is_baseline)
            if is_baseline:
                baseline_found = True
                baseline_enabled.append(str(s.get("scenario_id", "baseline")))
        if not baseline_found:
            ranked = sorted(
                [s for s in scenarios if isinstance(s, dict)],
                key=lambda x: str(x.get("scenario_id", "")),
            )
            if ranked:
                keep_sid = str(ranked[0].get("scenario_id", ""))
                for s in scenarios:
                    if isinstance(s, dict):
                        s["enabled"] = str(s.get("scenario_id", "")) == keep_sid
                baseline_enabled = [keep_sid]
        out["stress_scenarios"] = scenarios
    policy["quick_enabled_scenarios"] = baseline_enabled
    return out, policy


def _write_derived_config(path: Path, cfg: Dict[str, Any]) -> Tuple[Path, str]:
    text = _dump_yaml_deterministic(cfg)
    _write_text(path, text)
    return path.resolve(), _sha256_text(text)


def _read_latest_run_dir() -> Path:
    latest_path = REPO_ROOT / "artifacts" / ".latest_run"
    if not latest_path.exists():
        raise RuntimeError("Missing artifacts/.latest_run after run_research execution")
    run_dir = Path(latest_path.read_text(encoding="utf-8").strip())
    if not run_dir.exists():
        raise RuntimeError(f"Latest run directory does not exist: {run_dir}")
    return run_dir.resolve()


def _run_research(
    config_path: Path,
    *,
    quick_run: bool = False,
    log_dir: Optional[Path] = None,
) -> Path:
    cmd = [sys.executable, "-m", "weightiz.cli.run_research", "--config", str(config_path.resolve())]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    if quick_run:
        env["QUICK_RUN"] = "1"
        env.setdefault("QUICK_RUN_TASK_TIMEOUT_SEC", "120")
        env.setdefault("QUICK_RUN_PROGRESS_EVERY", "1")
    if log_dir is None:
        log_dir = (REPO_ROOT / "artifacts" / "sweep_v2" / "_logs").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{config_path.stem}_stdout.log"
    stderr_path = log_dir / f"{config_path.stem}_stderr.log"
    with stdout_path.open("w", encoding="utf-8") as f_out, stderr_path.open("w", encoding="utf-8") as f_err:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            text=True,
            stdout=f_out,
            stderr=f_err,
            check=False,
            env=env,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"run_research failed (rc={proc.returncode}) for config={config_path}. "
            f"See logs: {stdout_path} and {stderr_path}"
        )
    return _read_latest_run_dir()


def _verify_quick_run_artifacts(run_dir: Path) -> Dict[str, Any]:
    rq = run_dir.resolve()
    dq_report = rq / "dq_report.csv"
    dq_flags = rq / "dq_bar_flags.parquet"
    leaderboard = rq / "robustness_leaderboard.csv"
    run_status = rq / "run_status.json"
    plateaus = rq / "plateaus.json"

    for required in (dq_report, dq_flags, leaderboard, run_status):
        if not required.exists():
            raise RuntimeError(f"Quick-run artifact missing: {required}")

    dq_df = pd.read_csv(dq_report)
    dq_flags_df = pd.read_parquet(dq_flags)
    rb_df = pd.read_csv(leaderboard)
    status_doc = json.loads(run_status.read_text(encoding="utf-8"))
    stats_raw_doc: Dict[str, Any] = {}
    stats_raw_path = rq / "stats_raw.json"
    if stats_raw_path.exists():
        try:
            stats_raw_doc = json.loads(stats_raw_path.read_text(encoding="utf-8"))
        except Exception:
            stats_raw_doc = {}

    if int(dq_df.shape[0]) <= 0:
        raise RuntimeError(f"Quick-run artifact empty: {dq_report}")
    if int(dq_flags_df.shape[0]) <= 0:
        raise RuntimeError(f"Quick-run artifact empty: {dq_flags}")
    if int(rb_df.shape[0]) <= 0:
        raise RuntimeError(f"Quick-run artifact empty: {leaderboard}")

    failure_rate = _to_float(status_doc.get("failure_rate", float("nan")), default=float("nan"))
    wiring_ok = bool(np.isfinite(failure_rate) and float(failure_rate) < 1.0)
    if not wiring_ok:
        raise RuntimeError(
            f"Quick-run verification failed: failure_rate={failure_rate:.4f} in {run_status}. "
            "All candidate tasks failed."
        )

    failed_mask = rb_df["failed"].map(_to_bool).to_numpy(dtype=bool) if "failed" in rb_df.columns else np.zeros(int(rb_df.shape[0]), dtype=bool)
    robust_series = (
        rb_df["robustness_score"].apply(lambda x: _to_float(x, default=float("nan")))
        if "robustness_score" in rb_df.columns
        else pd.Series(np.full(int(rb_df.shape[0]), np.nan), dtype=np.float64)
    )
    finite_robust = robust_series.map(np.isfinite).to_numpy(dtype=bool)
    non_failed = ~failed_mask
    evaluation_ready = bool(np.any(non_failed & finite_robust))
    eval_reason = ""
    if not evaluation_ready:
        if int(rb_df.shape[0]) <= 0:
            eval_reason = "leaderboard_empty"
        elif bool(np.all(failed_mask)):
            eval_reason = "all_candidates_failed"
        elif not bool(np.any(finite_robust)):
            eval_reason = "no_finite_robustness_scores"
        else:
            eval_reason = "no_non_failed_finite_candidate"
        stats_err = str(stats_raw_doc.get("quick_run_stats_error", "")).strip()
        if stats_err:
            eval_reason = f"{eval_reason};{stats_err}"

    rb = rb_df.copy()
    if "robustness_score" in rb.columns:
        rb["_r"] = rb["robustness_score"].apply(lambda x: _to_float(x, default=float("-inf")))
        rb = rb.sort_values(["_r", "candidate_id"], ascending=[False, True], kind="mergesort")
    elif "candidate_id" in rb.columns:
        rb = rb.sort_values(["candidate_id"], ascending=[True], kind="mergesort")
    top5: List[Dict[str, Any]] = []
    for _, row in rb.head(5).iterrows():
        top5.append(
            {
                "candidate_id": str(row.get("candidate_id", "")),
                "robustness_score": _to_float(row.get("robustness_score", float("nan")), default=float("nan")),
                "pass": _to_bool(row.get("pass", False)),
                "failed": _to_bool(row.get("failed", False)),
            }
        )

    return {
        "run_dir": str(rq),
        "dq_report_csv": str(dq_report),
        "dq_bar_flags_parquet": str(dq_flags),
        "robustness_leaderboard_csv": str(leaderboard),
        "run_status_json": str(run_status),
        "plateaus_json_exists": bool(plateaus.exists()),
        "dq_report_rows": int(dq_df.shape[0]),
        "dq_bar_flags_rows": int(dq_flags_df.shape[0]),
        "leaderboard_rows": int(rb_df.shape[0]),
        "failure_rate": float(failure_rate),
        "wiring_ok": bool(wiring_ok),
        "evaluation_ready": bool(evaluation_ready),
        "evaluation_ready_reason": str(eval_reason),
        "stats_raw_json": str(stats_raw_path) if stats_raw_path.exists() else None,
        "top5": top5,
    }


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(int(value))
    txt = str(value).strip().lower()
    return txt in {"1", "true", "yes", "y", "t"}


def _to_float(value: Any, default: float = float("-inf")) -> float:
    try:
        out = float(value)
        if out != out:
            return default
        return out
    except Exception:
        return default


def choose_focused_candidates(
    robustness_csv: Path,
    plateaus_json: Path,
    top_plateaus: int,
    max_candidates: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if int(max_candidates) <= 0:
        raise RuntimeError("--max-focused-candidates must be > 0")
    if not robustness_csv.exists():
        raise RuntimeError(f"Scout artifact missing: {robustness_csv}")
    if not plateaus_json.exists():
        raise RuntimeError(f"Scout artifact missing: {plateaus_json}")

    rb = pd.read_csv(robustness_csv)
    if "candidate_id" not in rb.columns:
        raise RuntimeError("robustness_leaderboard.csv missing candidate_id")

    rows = rb.copy()
    if "failed" in rows.columns:
        rows = rows[~rows["failed"].apply(_to_bool)]
    if "pass" in rows.columns:
        rows = rows[rows["pass"].apply(_to_bool)]

    if rows.empty:
        # Fallback: preserve determinism while not dropping all candidates.
        rows = rb.copy()
        if "failed" in rows.columns:
            rows = rows[~rows["failed"].apply(_to_bool)]
        if rows.empty:
            rows = rb.copy()

    rows = rows.copy()
    rows["_robustness"] = rows.get("robustness_score", pd.Series([float("-inf")] * len(rows))).apply(
        lambda x: _to_float(x, default=float("-inf"))
    )
    rows = rows.sort_values(["_robustness", "candidate_id"], ascending=[False, True], kind="mergesort")

    with plateaus_json.open("r", encoding="utf-8") as f:
        plateaus_doc = json.load(f)
    clusters = list(plateaus_doc.get("clusters", []))

    clusters_sorted = sorted(
        clusters,
        key=lambda c: (
            -_to_float(c.get("median_score"), default=float("-inf")),
            str(c.get("plateau_id", "")),
        ),
    )

    selected_ids: List[str] = []
    selected_set: set[str] = set()
    selected_plateaus: List[str] = []

    for cluster in clusters_sorted[: max(0, int(top_plateaus))]:
        plateau_id = str(cluster.get("plateau_id", ""))
        if not plateau_id:
            continue
        if "plateau_id" in rows.columns:
            plateau_rows = rows[rows["plateau_id"].astype(str) == plateau_id]
        else:
            plateau_rows = rows.iloc[0:0]
        if plateau_rows.empty:
            cand_ids = [str(x) for x in cluster.get("candidate_ids", [])]
            plateau_rows = rows[rows["candidate_id"].astype(str).isin(cand_ids)]
        if plateau_rows.empty:
            continue

        selected_plateaus.append(plateau_id)
        for _, rr in plateau_rows.sort_values(["_robustness", "candidate_id"], ascending=[False, True], kind="mergesort").iterrows():
            cid = str(rr["candidate_id"])
            if cid in selected_set:
                continue
            selected_ids.append(cid)
            selected_set.add(cid)
            if len(selected_ids) >= int(max_candidates):
                break
        if len(selected_ids) >= int(max_candidates):
            break

    if len(selected_ids) < int(max_candidates):
        for _, rr in rows.iterrows():
            cid = str(rr["candidate_id"])
            if cid in selected_set:
                continue
            selected_ids.append(cid)
            selected_set.add(cid)
            if len(selected_ids) >= int(max_candidates):
                break

    if not selected_ids:
        raise RuntimeError("Focused candidate selection produced an empty set")

    filtered = rows[rows["candidate_id"].astype(str).isin(selected_ids)].copy()
    filtered = filtered.sort_values(["_robustness", "candidate_id"], ascending=[False, True], kind="mergesort")

    selected_rows: List[Dict[str, Any]] = []
    for cid in selected_ids:
        rr = filtered[filtered["candidate_id"].astype(str) == cid]
        if rr.empty:
            continue
        row = rr.iloc[0].to_dict()
        row["candidate_id"] = str(row["candidate_id"])
        selected_rows.append(row)

    meta = {
        "top_plateaus_requested": int(top_plateaus),
        "selected_plateaus": selected_plateaus,
        "max_focused_candidates": int(max_candidates),
        "selected_candidate_count": int(len(selected_rows)),
        "filter_policy": "failed=false and pass=true when available; deterministic fallback to nonfailed/all",
    }
    return selected_rows, meta


def _supports_manual_candidates() -> bool:
    # The current run_research schema includes candidates.mode/manual.
    # Guard remains explicit for forward compatibility.
    try:
        repo_str = str(REPO_ROOT)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
        from weightiz.cli.run_research import CandidatesModel, RunConfigModel  # type: ignore

        return bool(
            "candidates" in RunConfigModel.model_fields
            and "mode" in CandidatesModel.model_fields
        )
    except Exception:
        return False


def build_focused_config(
    base_cfg: Dict[str, Any],
    symbols_sorted: Sequence[str],
    selected_candidates: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Any], str]:
    cfg = _replace_symbols_in_config(base_cfg, symbols_sorted)

    can_manual = _supports_manual_candidates()
    if can_manual:
        specs: List[Dict[str, Any]] = []
        for row in selected_candidates:
            cid = str(row.get("candidate_id", "")).strip()
            if not cid:
                continue
            try:
                m2_idx = int(row["m2_idx"])
                m3_idx = int(row["m3_idx"])
                m4_idx = int(row["m4_idx"])
            except Exception as exc:
                raise RuntimeError(f"Focused candidate row missing m2/m3/m4 index for candidate_id={cid}") from exc
            specs.append(
                {
                    "candidate_id": cid,
                    "m2_idx": m2_idx,
                    "m3_idx": m3_idx,
                    "m4_idx": m4_idx,
                    "enabled_assets": "all",
                    "tags": ["auto_focus"],
                }
            )
        if not specs:
            raise RuntimeError("No valid candidate specs could be built for focused config")
        cfg["candidates"] = {
            "mode": "manual",
            "specs": specs,
        }
        return cfg, "manual_candidates"

    # Schema-safe fallback: reduce auto-grid dimensions to selected index sets.
    m2 = cfg.get("module2_configs")
    m3 = cfg.get("module3_configs")
    m4 = cfg.get("module4_configs")
    if not isinstance(m2, list) or not isinstance(m3, list) or not isinstance(m4, list):
        raise RuntimeError("Base config is missing module2_configs/module3_configs/module4_configs lists")

    used_m2 = sorted({int(r["m2_idx"]) for r in selected_candidates})
    used_m3 = sorted({int(r["m3_idx"]) for r in selected_candidates})
    used_m4 = sorted({int(r["m4_idx"]) for r in selected_candidates})

    cfg["module2_configs"] = [copy.deepcopy(m2[i]) for i in used_m2]
    cfg["module3_configs"] = [copy.deepcopy(m3[i]) for i in used_m3]
    cfg["module4_configs"] = [copy.deepcopy(m4[i]) for i in used_m4]
    cfg["candidates"] = {"mode": "auto_grid", "specs": []}
    return cfg, "auto_grid_reduced"


def _top_candidates_from_run(run_dir: Path, n: int = 5) -> List[Dict[str, Any]]:
    rb_path = run_dir / "robustness_leaderboard.csv"
    if not rb_path.exists():
        return []
    df = pd.read_csv(rb_path)
    if "robustness_score" in df.columns:
        df["_r"] = df["robustness_score"].apply(lambda x: _to_float(x, default=float("-inf")))
        df = df.sort_values(["_r", "candidate_id"], ascending=[False, True], kind="mergesort")
    elif "candidate_id" in df.columns:
        df = df.sort_values(["candidate_id"], ascending=[True], kind="mergesort")
    out: List[Dict[str, Any]] = []
    for _, row in df.head(max(0, int(n))).iterrows():
        out.append(
            {
                "candidate_id": str(row.get("candidate_id", "")),
                "robustness_score": _to_float(row.get("robustness_score", float("nan")), default=float("nan")),
                "pass": _to_bool(row.get("pass", False)),
                "failed": _to_bool(row.get("failed", False)),
            }
        )
    return out


def _selection_session_policy(base_cfg: Dict[str, Any]) -> Tuple[str, int, int]:
    engine = base_cfg.get("engine", {})
    harness = base_cfg.get("harness", {})
    tz_name = "America/New_York"
    if isinstance(harness, dict):
        tz_name = str(harness.get("timezone", tz_name))
    open_min = 570
    close_min = 945
    if isinstance(engine, dict):
        open_min = int(engine.get("rth_open_minute", open_min))
        close_min = int(engine.get("flat_time_minute", close_min))
    return tz_name, open_min, close_min


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sweep v2 auto-resolve + scout + focused full")
    p.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    p.add_argument("--scout-config", default=DEFAULT_SCOUT_CONFIG)
    p.add_argument("--clean-dir", default=DEFAULT_CLEAN_DIR)
    p.add_argument("--target-symbols", type=int, default=20)
    p.add_argument("--min-symbols", type=int, default=8)
    p.add_argument(
        "--mode",
        choices=("scout_only", "full_only", "scout_then_focused_full"),
        default="scout_then_focused_full",
    )
    p.add_argument("--top-plateaus", type=int, default=3)
    p.add_argument("--max-focused-candidates", type=int, default=30)
    p.add_argument("--quick-run", action="store_true", help="Run deterministic reduced workload verification mode")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    base_cfg_path = (REPO_ROOT / args.base_config).resolve()
    scout_cfg_path = (REPO_ROOT / args.scout_config).resolve()
    clean_dir = (REPO_ROOT / args.clean_dir).resolve()
    if not clean_dir.exists():
        raise RuntimeError(f"Clean cache directory not found: {clean_dir}")

    ts = _utc_stamp()
    artifacts_dir = (REPO_ROOT / "artifacts" / "sweep_v2" / ts).resolve()
    logs_dir = (artifacts_dir / "logs").resolve()
    gen_dir = (REPO_ROOT / "configs" / "_generated").resolve()
    gen_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Step 1+2: inventory + sanity metrics.
    inventory_rows = build_inventory(clean_dir)
    inventory_csv_path = artifacts_dir / "data_inventory.csv"
    _write_inventory_csv(inventory_csv_path, inventory_rows)

    existing_symbols = sorted([r.symbol for r in inventory_rows])

    base_cfg = _load_yaml_mapping(base_cfg_path)
    probe_cfg = base_cfg
    if args.mode in {"scout_only", "scout_then_focused_full"} and scout_cfg_path.exists():
        probe_cfg = _load_yaml_mapping(scout_cfg_path)
    tz_name, session_open_minute, session_close_minute = _selection_session_policy(probe_cfg)
    probe_start_utc, probe_end_utc, probe_year_fallback = _selection_probe_window(probe_cfg)

    # Step 3: deterministic DQ-aware selection.
    dq_probe_rows: List[SymbolDQProbe] = []
    for r in sorted(inventory_rows, key=lambda x: x.symbol):
        if r.excluded:
            continue
        dq_probe_rows.append(
            probe_symbol_dq_metrics(
                symbol=r.symbol,
                path=Path(r.path),
                tz_name=tz_name,
                session_open_minute=int(session_open_minute),
                session_close_minute=int(session_close_minute),
                probe_year_fallback=int(probe_year_fallback),
                probe_start_utc=probe_start_utc,
                probe_end_utc=probe_end_utc,
            )
        )
    dq_probe_by_symbol = {r.symbol: r for r in dq_probe_rows}
    selection_dq_csv_path = artifacts_dir / "selection_dq_table.csv"
    _write_selection_dq_csv(selection_dq_csv_path, dq_probe_rows)

    chosen_symbols, ranked_rows, excluded_symbols = select_symbols_deterministic(
        inventory_rows,
        target_symbols=int(args.target_symbols),
        min_symbols=int(args.min_symbols),
        dq_metrics_by_symbol=dq_probe_by_symbol,
    )
    chosen_quick_symbols: List[str] = []
    if bool(args.quick_run):
        if len(chosen_symbols) < QUICK_RUN_SYMBOLS:
            raise RuntimeError(
                f"--quick-run requires at least {QUICK_RUN_SYMBOLS} selected symbols; got {len(chosen_symbols)}"
            )
        chosen_quick_symbols = chosen_symbols[:QUICK_RUN_SYMBOLS]

    # Step 4: write derived configs, immutable base config.
    full_quick_policy: Dict[str, Any] = {}
    if bool(args.quick_run):
        quick_full_cfg, full_quick_policy = _apply_quick_run_reduction(
            base_cfg,
            chosen_symbols=chosen_symbols,
            quick_symbol_count=QUICK_RUN_SYMBOLS,
            max_candidates=QUICK_RUN_MAX_CANDIDATES,
        )
        derived_full_cfg = _apply_sweep_abort_guardrails(quick_full_cfg, force_serial=True)
        full_cfg_path = gen_dir / f"sweep_auto_full_quick_{ts}.yaml"
    else:
        derived_full_cfg = _apply_sweep_abort_guardrails(
            _replace_symbols_in_config(base_cfg, chosen_symbols),
            force_serial=False,
        )
        full_cfg_path = gen_dir / f"sweep_auto_full_{ts}.yaml"
    full_cfg_path_abs, full_cfg_sha = _write_derived_config(full_cfg_path, derived_full_cfg)

    derived_scout_cfg_path_abs: Optional[Path] = None
    derived_scout_cfg_sha: Optional[str] = None
    scout_cfg_for_run: Optional[Path] = None
    if args.mode in {"scout_only", "scout_then_focused_full"}:
        if not scout_cfg_path.exists():
            raise RuntimeError(
                f"Scout config required for mode={args.mode} but missing: {scout_cfg_path}. "
                "Provide --scout-config or use --mode full_only."
            )
        scout_cfg = _load_yaml_mapping(scout_cfg_path)
        if bool(args.quick_run):
            quick_scout_cfg, _ = _apply_quick_run_reduction(
                scout_cfg,
                chosen_symbols=chosen_symbols,
                quick_symbol_count=QUICK_RUN_SYMBOLS,
                max_candidates=QUICK_RUN_MAX_CANDIDATES,
            )
            derived_scout_cfg = _apply_sweep_abort_guardrails(quick_scout_cfg, force_serial=True)
            out_scout_path = gen_dir / f"sweep_auto_scout_quick_{ts}.yaml"
        else:
            derived_scout_cfg = _apply_sweep_abort_guardrails(
                _replace_symbols_in_config(scout_cfg, chosen_symbols),
                force_serial=False,
            )
            out_scout_path = gen_dir / f"sweep_auto_scout_{ts}.yaml"
        derived_scout_cfg_path_abs, derived_scout_cfg_sha = _write_derived_config(out_scout_path, derived_scout_cfg)
        scout_cfg_for_run = derived_scout_cfg_path_abs

    manifest: Dict[str, Any] = {
        "timestamp": ts,
        "start_time_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "mode": args.mode,
        "quick_run": bool(args.quick_run),
        "clean_dir": str(clean_dir),
        "logs_dir": str(logs_dir),
        "base_config": str(base_cfg_path),
        "scout_config": str(scout_cfg_path) if scout_cfg_path.exists() else None,
        "existing_symbols": existing_symbols,
        "excluded_symbols": excluded_symbols,
        "chosen_symbols": chosen_symbols,
        "chosen_quick_symbols": chosen_quick_symbols if bool(args.quick_run) else [],
        "selection_rule": (
            "DQ-aware rank: (1) coverage_ratio=(ACCEPT+DEGRADE)/total_days desc, "
            "(2) median_dqs desc, (3) quality_score desc, (4) symbol asc; "
            "quality_score = row_count - 100000*nan_rate_ohlcv - 100*duplicate_timestamp_count - "
            "(50000 if monotonic_timestamp_ok=false else 0)"
        ),
        "selection_dq_table_csv": str(selection_dq_csv_path),
        "selection_policy": {
            "timezone": tz_name,
            "session_open_minute": int(session_open_minute),
            "session_close_minute": int(session_close_minute),
            "probe_year_fallback": int(probe_year_fallback),
            "probe_start_utc": str(probe_start_utc) if probe_start_utc is not None else None,
            "probe_end_utc": str(probe_end_utc) if probe_end_utc is not None else None,
        },
        "derived_full_config": {
            "path": str(full_cfg_path_abs),
            "sha256": full_cfg_sha,
        },
        "quick_run_policy": full_quick_policy if bool(args.quick_run) else {},
        "derived_scout_config": (
            {
                "path": str(derived_scout_cfg_path_abs),
                "sha256": str(derived_scout_cfg_sha),
            }
            if derived_scout_cfg_path_abs is not None
            else None
        ),
        "inventory_csv": str(inventory_csv_path),
        "ranked_symbols": [
            {
                "symbol": r.symbol,
                "coverage_ratio": float(dq_probe_by_symbol.get(r.symbol).coverage_ratio) if r.symbol in dq_probe_by_symbol else -1.0,
                "median_dqs": float(dq_probe_by_symbol.get(r.symbol).median_dqs) if r.symbol in dq_probe_by_symbol else -1.0,
                "reject_ratio": float(dq_probe_by_symbol.get(r.symbol).reject_ratio) if r.symbol in dq_probe_by_symbol else 1.0,
                "dq_total_days": int(dq_probe_by_symbol.get(r.symbol).total_days) if r.symbol in dq_probe_by_symbol else 0,
                "dq_probe_error": str(dq_probe_by_symbol.get(r.symbol).probe_error) if r.symbol in dq_probe_by_symbol else "not_probed",
                "quality_score": float(r.quality_score),
                "row_count": int(r.row_count),
                "nan_rate_ohlcv": float(r.nan_rate_ohlcv),
                "duplicate_timestamp_count": int(r.duplicate_timestamp_count),
                "monotonic_timestamp_ok": bool(r.monotonic_timestamp_ok),
            }
            for r in ranked_rows
        ],
        "chosen_symbols_ranked": [
            {
                "symbol": s,
                "coverage_ratio": float(dq_probe_by_symbol.get(s).coverage_ratio) if s in dq_probe_by_symbol else -1.0,
                "median_dqs": float(dq_probe_by_symbol.get(s).median_dqs) if s in dq_probe_by_symbol else -1.0,
                "reject_ratio": float(dq_probe_by_symbol.get(s).reject_ratio) if s in dq_probe_by_symbol else 1.0,
            }
            for s in chosen_symbols
        ],
    }
    manifest_path = artifacts_dir / "manifest.json"
    _write_json(manifest_path, manifest)

    scout_run_dir: Optional[Path] = None
    full_run_dir: Optional[Path] = None
    focused_run_dir: Optional[Path] = None
    focused_cfg_path_abs: Optional[Path] = None
    focused_cfg_sha: Optional[str] = None

    # Step 5: scout run.
    if args.mode in {"scout_only", "scout_then_focused_full"}:
        if scout_cfg_for_run is None:
            raise RuntimeError("Internal error: scout config path not set")
        scout_run_dir = _run_research(scout_cfg_for_run, quick_run=bool(args.quick_run), log_dir=logs_dir)
        manifest["scout_run_dir"] = str(scout_run_dir)

    # Full only mode executes derived full directly.
    if args.mode == "full_only":
        full_run_dir = _run_research(full_cfg_path_abs, quick_run=bool(args.quick_run), log_dir=logs_dir)
        manifest["full_run_dir"] = str(full_run_dir)

    # Step 6+7: focused full from scout plateaus.
    if args.mode == "scout_then_focused_full":
        if scout_run_dir is None:
            raise RuntimeError("Internal error: scout run directory not available")

        selected_rows, focus_meta = choose_focused_candidates(
            robustness_csv=scout_run_dir / "robustness_leaderboard.csv",
            plateaus_json=scout_run_dir / "plateaus.json",
            top_plateaus=int(args.top_plateaus),
            max_candidates=int(args.max_focused_candidates),
        )
        focused_cfg, focus_mode = build_focused_config(
            base_cfg=base_cfg,
            symbols_sorted=chosen_quick_symbols if bool(args.quick_run) else chosen_symbols,
            selected_candidates=selected_rows,
        )
        quick_focus_policy: Dict[str, Any] = {}
        if bool(args.quick_run):
            focused_cfg, quick_focus_policy = _apply_quick_run_reduction(
                focused_cfg,
                chosen_symbols=chosen_quick_symbols if chosen_quick_symbols else chosen_symbols,
                quick_symbol_count=QUICK_RUN_SYMBOLS,
                max_candidates=QUICK_RUN_MAX_CANDIDATES,
            )
        focused_cfg = _apply_sweep_abort_guardrails(focused_cfg, force_serial=bool(args.quick_run))
        focused_cfg_path = (
            gen_dir / f"sweep_auto_focused_quick_{ts}.yaml"
            if bool(args.quick_run)
            else gen_dir / f"sweep_auto_focused_{ts}.yaml"
        )
        focused_cfg_path_abs, focused_cfg_sha = _write_derived_config(focused_cfg_path, focused_cfg)
        manifest["focused_config"] = {
            "path": str(focused_cfg_path_abs),
            "sha256": focused_cfg_sha,
            "focus_mode": focus_mode,
            "focus_meta": focus_meta,
            "quick_run_policy": quick_focus_policy if bool(args.quick_run) else {},
            "selected_candidates": [
                {
                    "candidate_id": str(r.get("candidate_id", "")),
                    "m2_idx": int(r.get("m2_idx", -1)),
                    "m3_idx": int(r.get("m3_idx", -1)),
                    "m4_idx": int(r.get("m4_idx", -1)),
                    "plateau_id": str(r.get("plateau_id", "")),
                    "robustness_score": _to_float(r.get("_robustness", r.get("robustness_score", float("nan"))), default=float("nan")),
                }
                for r in selected_rows
            ],
        }

        focused_run_dir = _run_research(
            focused_cfg_path_abs,
            quick_run=bool(args.quick_run),
            log_dir=logs_dir,
        )
        manifest["focused_run_dir"] = str(focused_run_dir)

    final_run_dir = focused_run_dir or full_run_dir or scout_run_dir
    quick_check: Optional[Dict[str, Any]] = None
    if bool(args.quick_run):
        if final_run_dir is None:
            raise RuntimeError("Quick-run finished without a final run directory")
        quick_check = _verify_quick_run_artifacts(final_run_dir)
        manifest["quick_run_artifact_check"] = quick_check
        manifest["quick_run_logs_dir"] = str(logs_dir)
        manifest["wiring_ok"] = bool(quick_check.get("wiring_ok", False))
        manifest["evaluation_ready"] = bool(quick_check.get("evaluation_ready", False))
        manifest["evaluation_ready_reason"] = str(quick_check.get("evaluation_ready_reason", ""))
        if not bool(quick_check.get("evaluation_ready", False)):
            print(
                "QUICK_RUN_EVALUATION_NOT_READY "
                f"reason={manifest['evaluation_ready_reason']}",
                flush=True,
            )
    manifest["status"] = "completed"
    manifest["end_time_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(manifest_path, manifest)

    top5 = _top_candidates_from_run(final_run_dir, n=5) if final_run_dir is not None else []

    print("SWEEP_V2_COMPLETE")
    print(json.dumps(
        {
            "quick_run": bool(args.quick_run),
            "chosen_symbols_count": len(chosen_symbols),
            "chosen_symbols": chosen_symbols,
            "chosen_quick_symbols": chosen_quick_symbols if bool(args.quick_run) else [],
            "derived_full_config": str(full_cfg_path_abs),
            "derived_scout_config": str(derived_scout_cfg_path_abs) if derived_scout_cfg_path_abs else None,
            "derived_focused_config": str(focused_cfg_path_abs) if focused_cfg_path_abs else None,
            "manifest": str(manifest_path),
            "quick_run_logs_dir": str(logs_dir) if bool(args.quick_run) else None,
            "quick_run_artifact_check": quick_check,
            "scout_run_dir": str(scout_run_dir) if scout_run_dir else None,
            "full_run_dir": str(full_run_dir) if full_run_dir else None,
            "focused_run_dir": str(focused_run_dir) if focused_run_dir else None,
            "top5_final": top5,
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
