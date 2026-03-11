#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sys
from datetime import datetime, timedelta, timezone
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

try:
    import yaml
except Exception as exc:  # pragma: no cover - import guard
    raise RuntimeError("pyyaml is required. Install with: pip install pyyaml") from exc

try:
    from pydantic import BaseModel, ConfigDict, Field, model_validator
except Exception as exc:  # pragma: no cover - import guard
    raise RuntimeError("pydantic>=2 is required. Install with: pip install 'pydantic>=2'") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from weightiz.shared.io.market_data.alpaca_client import AlpacaAPIError, AlpacaClient
from weightiz.shared.io.market_data.cleaning import (
    CANONICAL_COLUMNS,
    canonicalize_alpaca_bars,
    merge_canonical_frames,
    run_post_clean_qa_or_raise,
    summarize_session_meta_for_clean_frame,
)


LOGGER = logging.getLogger("weightiz.alpaca.fetch")
REQUIRED_KEY_ENV = "ALPACA_API_KEY"
REQUIRED_SECRET_ENV = "ALPACA_SECRET_KEY"
PERMISSION_KEYWORDS = ("forbidden", "not authorized", "insufficient", "permission")


class AlpacaSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Kept for backwards config compatibility. Runtime enforcement uses fixed env names only.
    api_key_env: str = REQUIRED_KEY_ENV
    secret_key_env: str = REQUIRED_SECRET_ENV

    base_url: str = "https://data.alpaca.markets"
    feed: Literal["sip", "iex"] = "iex"
    adjustment: Literal["raw", "split", "dividend", "all"] = "raw"
    timeframe: str = "1Min"
    sort: Literal["asc"] = "asc"
    limit_per_page: int = 10_000
    start: str
    end: str
    symbols: List[str] = Field(default_factory=list)
    session_policy: Literal["RTH", "ETH"] = "RTH"
    timezone: str = "America/New_York"
    rth_open: str = "09:30"
    rth_close: str = "16:00"
    rth_close_inclusive: bool = False
    calendar_mode: Literal["naive", "nyse"] = "naive"
    rate_limit_sleep_sec: float = 0.25
    max_symbols_per_request: int = 100
    max_retries_429: int = 6
    backoff_base_sec: float = 0.5
    backoff_max_sec: float = 8.0
    failure_rate_threshold: float = 0.2
    min_ok_symbols: int = 2
    qa_policy: Literal["strict_no_holes", "coverage_threshold"] = "coverage_threshold"
    coverage_min_pct: float = 99.0

    @model_validator(mode="after")
    def validate_symbols(self) -> "AlpacaSection":
        clean = [str(s).strip().upper() for s in self.symbols if str(s).strip()]
        if len(clean) == 0:
            raise ValueError("alpaca.symbols must contain at least one symbol")
        if len(set(clean)) != len(clean):
            raise ValueError("alpaca.symbols must be unique")
        self.symbols = clean
        if self.max_symbols_per_request <= 0:
            raise ValueError("alpaca.max_symbols_per_request must be > 0")
        if self.rate_limit_sleep_sec < 0:
            raise ValueError("alpaca.rate_limit_sleep_sec must be >= 0")
        if self.max_retries_429 < 0:
            raise ValueError("alpaca.max_retries_429 must be >= 0")
        if self.backoff_base_sec < 0:
            raise ValueError("alpaca.backoff_base_sec must be >= 0")
        if self.backoff_max_sec < 0:
            raise ValueError("alpaca.backoff_max_sec must be >= 0")
        if self.limit_per_page <= 0:
            raise ValueError("alpaca.limit_per_page must be > 0")
        if self.failure_rate_threshold < 0 or self.failure_rate_threshold > 1:
            raise ValueError("alpaca.failure_rate_threshold must be in [0,1]")
        if self.min_ok_symbols < 0:
            raise ValueError("alpaca.min_ok_symbols must be >= 0")
        if self.coverage_min_pct <= 0 or self.coverage_min_pct > 100:
            raise ValueError("alpaca.coverage_min_pct must be in (0,100]")
        return self


class StorageSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root: str = "./data/alpaca"
    write_format: Literal["parquet", "csv"] = "parquet"
    overwrite_clean: bool = False


class FetchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alpaca: AlpacaSection
    storage: StorageSection


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )


def _load_config(path: Path) -> FetchConfig:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise RuntimeError("Config root must be a YAML object")
    return FetchConfig.model_validate(raw)


def _env_required_fixed(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v


def _parse_iso_utc(text: str) -> datetime:
    s = str(text).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _stamp_utc(text: str) -> str:
    return _parse_iso_utc(text).strftime("%Y%m%dT%H%M%SZ")


def _run_id(config_sha256: str) -> str:
    return f"run_{datetime.now(timezone.utc).strftime('%Y%m%d')}_{str(config_sha256)[:10]}"


def _git_commit_hash() -> Optional[str]:
    try:
        import subprocess

        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out or None
    except Exception:
        return None


def _write_frame(df: Any, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    elif fmt == "csv":
        df.to_csv(path, index=False)
    else:
        raise RuntimeError(f"Unsupported write_format={fmt!r}")


def _read_frame(path: Path) -> Any:
    import pandas as pd

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise RuntimeError(f"Unsupported clean cache file extension: {path}")


def _resolved_config_sha256(cfg: FetchConfig) -> str:
    payload = cfg.model_dump(mode="json")
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _request_params_sha256(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _raw_filename(start: str, end: str, feed: str, adjustment: str, fmt: str) -> str:
    ext = "parquet" if fmt == "parquet" else "csv"
    return f"{_stamp_utc(start)}_{_stamp_utc(end)}_{str(feed).lower()}_{str(adjustment).lower()}.{ext}"


def _resolve_versioned_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for i in range(1, 10_000):
        candidate = path.with_name(f"{stem}__v{i:03d}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Unable to find deterministic version suffix for path={path}")


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _extract_status_code(exc: Exception) -> Optional[int]:
    raw = getattr(exc, "status_code", None)
    if raw is not None:
        try:
            return int(raw)
        except Exception:
            pass
    m = re.search(r"status=(\d{3})", str(exc))
    if m:
        return int(m.group(1))
    return None


def _is_permission_denied_error(status_code: Optional[int], message: str) -> bool:
    if status_code in (401, 403):
        return True
    low = str(message).lower()
    return any(k in low for k in PERMISSION_KEYWORDS)


def _calendar_expectations_available() -> bool:
    return find_spec("exchange_calendars") is not None


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _build_preflight_window(start_iso: str, end_iso: str) -> Tuple[str, str]:
    start_dt = _parse_iso_utc(start_iso)
    end_dt = _parse_iso_utc(end_iso)
    pre_end = min(start_dt + timedelta(days=1), end_dt)
    return start_dt.isoformat().replace("+00:00", "Z"), pre_end.isoformat().replace("+00:00", "Z")


def _build_recent_preflight_window(now_utc: datetime) -> Tuple[str, str]:
    end_dt = now_utc.replace(second=0, microsecond=0)
    start_dt = end_dt - timedelta(minutes=45)
    return start_dt.isoformat().replace("+00:00", "Z"), end_dt.isoformat().replace("+00:00", "Z")


def _compute_invariants_ok(df: Any) -> bool:
    import numpy as np

    if df is None or int(df.shape[0]) <= 0:
        return True
    open_v = df["open"].to_numpy(dtype=np.float64)
    high_v = df["high"].to_numpy(dtype=np.float64)
    low_v = df["low"].to_numpy(dtype=np.float64)
    close_v = df["close"].to_numpy(dtype=np.float64)
    vol_v = df["volume"].to_numpy(dtype=np.float64)
    return bool(
        np.all(np.isfinite(open_v))
        and np.all(np.isfinite(high_v))
        and np.all(np.isfinite(low_v))
        and np.all(np.isfinite(close_v))
        and np.all(np.isfinite(vol_v))
        and np.all(high_v >= np.maximum(open_v, close_v))
        and np.all(low_v <= np.minimum(open_v, close_v))
        and np.all(vol_v >= 0.0)
    )


def _evaluate_qa_policy(
    *,
    qa_policy: str,
    coverage_min_pct: float,
    invariants_ok: bool,
    session_meta: Dict[str, Any],
) -> Tuple[bool, str]:
    if not bool(invariants_ok):
        return False, "invariants_ok_false"

    missing_total = int(session_meta.get("missing_minutes_total", 0))
    coverage_pct = float(session_meta.get("coverage_pct", 0.0))
    missing_pct = float(session_meta.get("missing_minutes_pct", 0.0))
    policy = str(qa_policy).strip().lower()

    if policy == "strict_no_holes":
        if missing_total != 0:
            return False, f"strict_no_holes_missing_minutes_total={missing_total}"
        return True, ""

    min_cov = float(coverage_min_pct)
    max_missing = float(100.0 - min_cov)
    if coverage_pct < min_cov:
        return False, f"coverage_pct_below_min({coverage_pct:.6f}<{min_cov:.6f})"
    if missing_pct > max_missing:
        return False, f"missing_minutes_pct_above_max({missing_pct:.6f}>{max_missing:.6f})"
    return True, ""


def _promote_staging_to_clean(staging_path: Path, clean_path: Path) -> None:
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = clean_path.with_suffix(clean_path.suffix + ".tmp")
    shutil.copy2(staging_path, temp_path)
    os.replace(temp_path, clean_path)


def _write_abort_reports(
    reports_root: Path,
    run_id: str,
    qa_doc: Dict[str, Any],
    manifest_doc: Dict[str, Any],
) -> None:
    qa_path = reports_root / f"{run_id}_qa.json"
    manifest_path = reports_root / f"{run_id}_manifest.json"
    _write_json(qa_path, qa_doc)
    _write_json(manifest_path, manifest_doc)


def run_fetch(config_path: Path, symbol_filter: Optional[str] = None) -> Dict[str, Any]:
    _setup_logging()
    cfg = _load_config(config_path)
    config_sha256 = _resolved_config_sha256(cfg)
    run_id = _run_id(config_sha256)

    # Strict fixed env var enforcement.
    api_key = _env_required_fixed(REQUIRED_KEY_ENV)
    secret_key = _env_required_fixed(REQUIRED_SECRET_ENV)

    client = AlpacaClient(
        api_key=api_key,
        secret_key=secret_key,
        base_url=cfg.alpaca.base_url,
        rate_limit_sleep_sec=cfg.alpaca.rate_limit_sleep_sec,
        max_retries_429=cfg.alpaca.max_retries_429,
        backoff_base_sec=cfg.alpaca.backoff_base_sec,
        backoff_max_sec=cfg.alpaca.backoff_max_sec,
    )

    root = Path(cfg.storage.root)
    raw_root = root / "raw"
    clean_staging_root = root / "clean_staging"
    clean_root = root / "clean"
    reports_root = root / "reports"
    raw_root.mkdir(parents=True, exist_ok=True)
    clean_staging_root.mkdir(parents=True, exist_ok=True)
    clean_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    symbols_sorted = sorted(cfg.alpaca.symbols)
    if symbol_filter:
        sym = str(symbol_filter).strip().upper()
        if sym not in set(symbols_sorted):
            raise RuntimeError(f"Requested --symbol {sym!r} not present in config symbols")
        symbols_sorted = [sym]
    calendar_available = _calendar_expectations_available()
    git_hash = _git_commit_hash()

    base_params = {
        "source": "alpaca",
        "base_url": cfg.alpaca.base_url,
        "feed": cfg.alpaca.feed,
        "adjustment": cfg.alpaca.adjustment,
        "timeframe": cfg.alpaca.timeframe,
        "sort": cfg.alpaca.sort,
        "limit_requested": int(cfg.alpaca.limit_per_page),
        "limit_effective": int(cfg.alpaca.limit_per_page),
        "calendar_expectations_available": bool(calendar_available),
        "qa_policy": str(cfg.alpaca.qa_policy),
        "coverage_min_pct": float(cfg.alpaca.coverage_min_pct),
    }

    preflight_recent_start, preflight_recent_end = _build_recent_preflight_window(datetime.now(timezone.utc))
    preflight_historical_start, preflight_historical_end = _build_preflight_window(cfg.alpaca.start, cfg.alpaca.end)
    preflight_symbol = symbols_sorted[0]
    preflight_diag: Dict[str, Any] = {
        "preflight_symbol": preflight_symbol,
        "preflight_start": preflight_historical_start,
        "preflight_end": preflight_historical_end,
        "preflight_recent_start": preflight_recent_start,
        "preflight_recent_end": preflight_recent_end,
        "preflight_historical_start": preflight_historical_start,
        "preflight_historical_end": preflight_historical_end,
        "preflight_recent_bar_count": 0,
        "preflight_historical_bar_count": 0,
        "preflight_entitlement_class": "unknown",
        "preflight_actionable_fix": "",
        "preflight_status_code": None,
        "preflight_error_class": "ok",
        "preflight_error_msg": "",
    }

    qa_symbols: Dict[str, Dict[str, Any]] = {}
    manifest_symbols: Dict[str, Dict[str, Any]] = {}
    limit_effective_candidates: List[int] = [int(cfg.alpaca.limit_per_page)]

    # Preflight entitlement + permission validation on pinned feed.
    try:
        recent_bars, _, _, recent_meta = client.fetch_bars_multi_with_meta(
            symbols=[preflight_symbol],
            timeframe=cfg.alpaca.timeframe,
            start=preflight_recent_start,
            end=preflight_recent_end,
            feed=cfg.alpaca.feed,
            adjustment=cfg.alpaca.adjustment,
            sort=cfg.alpaca.sort,
            max_symbols_per_request=1,
            limit=cfg.alpaca.limit_per_page,
        )
        recent_count = int(len(list(recent_bars.get(preflight_symbol, []))))
        preflight_diag["preflight_recent_bar_count"] = int(recent_count)
        pf_recent = dict(recent_meta.get(preflight_symbol, {}))
        if pf_recent:
            limit_effective_candidates.append(int(pf_recent.get("limit_effective", cfg.alpaca.limit_per_page)))

        hist_bars, _, _, hist_meta = client.fetch_bars_multi_with_meta(
            symbols=[preflight_symbol],
            timeframe=cfg.alpaca.timeframe,
            start=preflight_historical_start,
            end=preflight_historical_end,
            feed=cfg.alpaca.feed,
            adjustment=cfg.alpaca.adjustment,
            sort=cfg.alpaca.sort,
            max_symbols_per_request=1,
            limit=cfg.alpaca.limit_per_page,
        )
        historical_count = int(len(list(hist_bars.get(preflight_symbol, []))))
        preflight_diag["preflight_historical_bar_count"] = int(historical_count)
        pf_hist = dict(hist_meta.get(preflight_symbol, {}))
        if pf_hist:
            limit_effective_candidates.append(int(pf_hist.get("limit_effective", cfg.alpaca.limit_per_page)))

        if recent_count > 0 and historical_count == 0:
            preflight_diag["preflight_entitlement_class"] = "historical_denied_or_limited"
            preflight_diag["preflight_actionable_fix"] = (
                "Account plan likely limits historical market data (e.g. Basic plan may only allow latest 15 minutes). "
                "Upgrade to Algo Trader Plus or use another provider."
            )

            for symbol in symbols_sorted:
                qa_symbols[symbol] = {
                    "status": "failed",
                    "error_type": "RuntimeError",
                    "error_msg": "historical_data_not_available_for_requested_range",
                    "limit_fallback_used": False,
                }
                manifest_symbols[symbol] = {
                    "status": "failed",
                    "error_type": "RuntimeError",
                    "error_msg": "historical_data_not_available_for_requested_range",
                    "limit_fallback_used": False,
                }

            limit_effective = min(limit_effective_candidates) if limit_effective_candidates else int(cfg.alpaca.limit_per_page)
            qa_doc = {
                "run_id": run_id,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": git_hash,
                "config_sha256": config_sha256,
                **base_params,
                "limit_effective": int(limit_effective),
                **preflight_diag,
                "symbols": {s: qa_symbols[s] for s in symbols_sorted},
                "summary": {
                    "ok_count": 0,
                    "failed_count": len(symbols_sorted),
                    "failure_rate": 1.0,
                    "all_failed": True,
                    "min_ok_symbols": int(cfg.alpaca.min_ok_symbols),
                    "exit_reason": "historical_data_not_available_for_requested_range",
                    "abort_reason": "historical_data_not_available_for_requested_range",
                },
            }
            manifest_doc = {
                "run_id": run_id,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": git_hash,
                "config_sha256": config_sha256,
                **base_params,
                "limit_effective": int(limit_effective),
                **preflight_diag,
                "symbols": {s: manifest_symbols[s] for s in symbols_sorted},
                "summary": {
                    "ok_count": 0,
                    "failed_count": len(symbols_sorted),
                    "failure_rate": 1.0,
                    "all_failed": True,
                    "min_ok_symbols": int(cfg.alpaca.min_ok_symbols),
                    "exit_reason": "historical_data_not_available_for_requested_range",
                    "abort_reason": "historical_data_not_available_for_requested_range",
                },
            }
            _write_abort_reports(
                reports_root=reports_root,
                run_id=run_id,
                qa_doc=qa_doc,
                manifest_doc=manifest_doc,
            )
            raise RuntimeError("historical_data_not_available_for_requested_range")

        if recent_count == 0 and historical_count == 0:
            preflight_diag["preflight_entitlement_class"] = "no_data_recent_or_historical"
            preflight_diag["preflight_actionable_fix"] = (
                "No bars returned for recent and historical probes; validate symbol, timeframe, session policy, and market state."
            )
            for symbol in symbols_sorted:
                qa_symbols[symbol] = {
                    "status": "failed",
                    "error_type": "RuntimeError",
                    "error_msg": "preflight_no_data_recent_or_historical",
                    "limit_fallback_used": False,
                }
                manifest_symbols[symbol] = {
                    "status": "failed",
                    "error_type": "RuntimeError",
                    "error_msg": "preflight_no_data_recent_or_historical",
                    "limit_fallback_used": False,
                }

            limit_effective = min(limit_effective_candidates) if limit_effective_candidates else int(cfg.alpaca.limit_per_page)
            qa_doc = {
                "run_id": run_id,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": git_hash,
                "config_sha256": config_sha256,
                **base_params,
                "limit_effective": int(limit_effective),
                **preflight_diag,
                "symbols": {s: qa_symbols[s] for s in symbols_sorted},
                "summary": {
                    "ok_count": 0,
                    "failed_count": len(symbols_sorted),
                    "failure_rate": 1.0,
                    "all_failed": True,
                    "min_ok_symbols": int(cfg.alpaca.min_ok_symbols),
                    "exit_reason": "preflight_no_data_recent_or_historical",
                    "abort_reason": "preflight_no_data_recent_or_historical",
                },
            }
            manifest_doc = {
                "run_id": run_id,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": git_hash,
                "config_sha256": config_sha256,
                **base_params,
                "limit_effective": int(limit_effective),
                **preflight_diag,
                "symbols": {s: manifest_symbols[s] for s in symbols_sorted},
                "summary": {
                    "ok_count": 0,
                    "failed_count": len(symbols_sorted),
                    "failure_rate": 1.0,
                    "all_failed": True,
                    "min_ok_symbols": int(cfg.alpaca.min_ok_symbols),
                    "exit_reason": "preflight_no_data_recent_or_historical",
                    "abort_reason": "preflight_no_data_recent_or_historical",
                },
            }
            _write_abort_reports(
                reports_root=reports_root,
                run_id=run_id,
                qa_doc=qa_doc,
                manifest_doc=manifest_doc,
            )
            raise RuntimeError("preflight_no_data_recent_or_historical")

        preflight_diag["preflight_entitlement_class"] = "ok"
        preflight_diag["preflight_actionable_fix"] = ""
    except Exception as exc:
        status_code = _extract_status_code(exc)
        msg = str(exc)
        if msg in (
            "historical_data_not_available_for_requested_range",
            "preflight_no_data_recent_or_historical",
        ):
            raise
        preflight_diag["preflight_status_code"] = status_code
        preflight_diag["preflight_error_msg"] = msg[:500]
        denied = _is_permission_denied_error(status_code=status_code, message=msg)
        preflight_diag["preflight_error_class"] = "permission_denied_feed" if denied else "preflight_failed"
        if denied:
            for symbol in symbols_sorted:
                qa_symbols[symbol] = {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_msg": msg[:500],
                    "limit_fallback_used": False,
                }
                manifest_symbols[symbol] = {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_msg": msg[:500],
                    "limit_fallback_used": False,
                }

            limit_effective = min(limit_effective_candidates) if limit_effective_candidates else int(cfg.alpaca.limit_per_page)
            qa_doc: Dict[str, Any] = {
                "run_id": run_id,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": git_hash,
                "config_sha256": config_sha256,
                **base_params,
                "limit_effective": int(limit_effective),
                **preflight_diag,
                "symbols": {s: qa_symbols[s] for s in symbols_sorted},
                "summary": {
                    "ok_count": 0,
                    "failed_count": len(symbols_sorted),
                    "failure_rate": 1.0,
                    "all_failed": True,
                    "min_ok_symbols": int(cfg.alpaca.min_ok_symbols),
                    "exit_reason": "permission_denied_feed",
                    "abort_reason": "permission_denied_feed",
                },
            }
            manifest_doc: Dict[str, Any] = {
                "run_id": run_id,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": git_hash,
                "config_sha256": config_sha256,
                **base_params,
                "limit_effective": int(limit_effective),
                **preflight_diag,
                "symbols": {s: manifest_symbols[s] for s in symbols_sorted},
                "summary": {
                    "ok_count": 0,
                    "failed_count": len(symbols_sorted),
                    "failure_rate": 1.0,
                    "all_failed": True,
                    "min_ok_symbols": int(cfg.alpaca.min_ok_symbols),
                    "exit_reason": "permission_denied_feed",
                    "abort_reason": "permission_denied_feed",
                },
            }
            _write_abort_reports(
                reports_root=reports_root,
                run_id=run_id,
                qa_doc=qa_doc,
                manifest_doc=manifest_doc,
            )
            raise RuntimeError("permission_denied_feed")
        raise

    for symbol in symbols_sorted:
        try:
            bars_by_symbol, feed_by_symbol, warnings, meta_by_symbol = client.fetch_bars_multi_with_meta(
                symbols=[symbol],
                timeframe=cfg.alpaca.timeframe,
                start=cfg.alpaca.start,
                end=cfg.alpaca.end,
                feed=cfg.alpaca.feed,
                adjustment=cfg.alpaca.adjustment,
                sort=cfg.alpaca.sort,
                max_symbols_per_request=1,
                limit=cfg.alpaca.limit_per_page,
            )

            records = list(bars_by_symbol.get(symbol, []))
            if len(records) == 0:
                raise RuntimeError(f"No bars returned for symbol={symbol}")

            import pandas as pd

            sym_meta = dict(meta_by_symbol.get(symbol, {}))
            limit_effective = int(sym_meta.get("limit_effective", cfg.alpaca.limit_per_page))
            limit_effective_candidates.append(limit_effective)

            raw_fmt = cfg.storage.write_format
            raw_base = raw_root / symbol / _raw_filename(
                cfg.alpaca.start,
                cfg.alpaca.end,
                feed_by_symbol.get(symbol, cfg.alpaca.feed),
                cfg.alpaca.adjustment,
                raw_fmt,
            )
            raw_file = _resolve_versioned_path(raw_base)
            raw_df = pd.DataFrame(records)
            _write_frame(raw_df, raw_file, raw_fmt)
            raw_sha = _file_sha256(raw_file)

            clean_df, qa_clean = canonicalize_alpaca_bars(
                records=records,
                symbol=symbol,
                timezone=cfg.alpaca.timezone,
                session_policy=cfg.alpaca.session_policy,
                rth_open=cfg.alpaca.rth_open,
                rth_close=cfg.alpaca.rth_close,
                rth_close_inclusive=cfg.alpaca.rth_close_inclusive,
                calendar_mode=cfg.alpaca.calendar_mode,
            )
            if clean_df.empty:
                raise RuntimeError(f"Clean output is empty for symbol={symbol}; fail-closed")

            out_ext = "parquet" if cfg.storage.write_format == "parquet" else "csv"
            clean_path = clean_root / f"{symbol}.{out_ext}"
            staging_path = clean_staging_root / f"{symbol}.{out_ext}"

            if clean_path.exists() and not cfg.storage.overwrite_clean:
                existing = _read_frame(clean_path)
                for c in CANONICAL_COLUMNS:
                    if c not in existing.columns:
                        raise RuntimeError(f"Existing clean cache missing column '{c}' for symbol={symbol}")
                target_clean = merge_canonical_frames(existing.loc[:, list(CANONICAL_COLUMNS)], clean_df)
                clean_write_mode = "merge"
            else:
                target_clean = clean_df.loc[:, list(CANONICAL_COLUMNS)].copy()
                clean_write_mode = "overwrite" if cfg.storage.overwrite_clean else "create"

            # Always write staging, regardless of later QA pass/fail.
            _write_frame(target_clean, staging_path, cfg.storage.write_format)

            session_meta_final = summarize_session_meta_for_clean_frame(
                clean=target_clean,
                timezone=cfg.alpaca.timezone,
                session_policy=cfg.alpaca.session_policy,
                rth_open=cfg.alpaca.rth_open,
                rth_close=cfg.alpaca.rth_close,
                rth_close_inclusive=cfg.alpaca.rth_close_inclusive,
                calendar_mode=cfg.alpaca.calendar_mode,
            )
            post_clean_qa = run_post_clean_qa_or_raise(
                clean=target_clean,
                session_meta=dict(session_meta_final),
                timezone=cfg.alpaca.timezone,
                session_policy=cfg.alpaca.session_policy,
                rth_open=cfg.alpaca.rth_open,
                rth_close=cfg.alpaca.rth_close,
                rth_close_inclusive=cfg.alpaca.rth_close_inclusive,
                calendar_mode=cfg.alpaca.calendar_mode,
            )
            invariants_ok_final = _compute_invariants_ok(target_clean)
            qa_pass, qa_fail_reason = _evaluate_qa_policy(
                qa_policy=cfg.alpaca.qa_policy,
                coverage_min_pct=cfg.alpaca.coverage_min_pct,
                invariants_ok=invariants_ok_final,
                session_meta=session_meta_final,
            )

            qa_clean["session"] = dict(session_meta_final)
            qa_clean["post_clean"] = dict(post_clean_qa)
            qa_clean["invariants_ok"] = bool(invariants_ok_final)

            if not qa_pass:
                qa_symbols[symbol] = {
                    "status": "failed_qa",
                    "qa_fail_reason": str(qa_fail_reason),
                    "coverage_pct": float(session_meta_final.get("coverage_pct", 0.0)),
                    "missing_minutes_total": int(session_meta_final.get("missing_minutes_total", 0)),
                    "missing_minutes_pct": float(session_meta_final.get("missing_minutes_pct", 0.0)),
                    "missing_minutes_preview": list(session_meta_final.get("missing_minutes_preview", [])),
                    "qa": qa_clean,
                    "staging_file": str(staging_path),
                    "limit_fallback_used": bool(sym_meta.get("limit_fallback_used", False)),
                }
                manifest_symbols[symbol] = {
                    "status": "failed_qa",
                    "qa_fail_reason": str(qa_fail_reason),
                    "coverage_pct": float(session_meta_final.get("coverage_pct", 0.0)),
                    "missing_minutes_total": int(session_meta_final.get("missing_minutes_total", 0)),
                    "missing_minutes_pct": float(session_meta_final.get("missing_minutes_pct", 0.0)),
                    "missing_minutes_preview": list(session_meta_final.get("missing_minutes_preview", [])),
                    "staging_file": str(staging_path),
                    "raw_files": [str(raw_file)],
                    "raw_file_sha256": [str(raw_sha)],
                    "limit_requested": int(sym_meta.get("limit_requested", cfg.alpaca.limit_per_page)),
                    "limit_effective": int(sym_meta.get("limit_effective", cfg.alpaca.limit_per_page)),
                    "limit_fallback_used": bool(sym_meta.get("limit_fallback_used", False)),
                    "limit_fallback_reason": str(sym_meta.get("limit_fallback_reason", "")),
                }
                continue

            _promote_staging_to_clean(staging_path=staging_path, clean_path=clean_path)
            final_rows = int(target_clean.shape[0])

            request_params = {
                "symbol": symbol,
                "feed": cfg.alpaca.feed,
                "adjustment": cfg.alpaca.adjustment,
                "timeframe": cfg.alpaca.timeframe,
                "sort": cfg.alpaca.sort,
                "limit_requested": int(cfg.alpaca.limit_per_page),
                "start": cfg.alpaca.start,
                "end": cfg.alpaca.end,
            }

            qa_symbols[symbol] = {
                "status": "ok",
                "qa": qa_clean,
                "feed_used": feed_by_symbol.get(symbol, cfg.alpaca.feed),
                "raw_file": str(raw_file),
                "staging_file": str(staging_path),
                "clean_file": str(clean_path),
                "rows_clean_file": int(final_rows),
                "clean_write_mode": clean_write_mode,
                "request_params_sha256": _request_params_sha256(request_params),
                "retry_429_count": int(sym_meta.get("retry_429_count", 0)),
                "total_sleep_seconds": float(sym_meta.get("total_sleep_seconds", 0.0)),
                "backoff_schedule_seconds": list(sym_meta.get("backoff_schedule_seconds", [])),
                "limit_requested": int(sym_meta.get("limit_requested", cfg.alpaca.limit_per_page)),
                "limit_effective": int(sym_meta.get("limit_effective", cfg.alpaca.limit_per_page)),
                "limit_fallback_used": bool(sym_meta.get("limit_fallback_used", False)),
                "limit_fallback_reason": str(sym_meta.get("limit_fallback_reason", "")),
                "warnings": sorted(set(warnings)),
            }
            manifest_symbols[symbol] = {
                "status": "ok",
                "raw_files": [str(raw_file)],
                "raw_file_sha256": [str(raw_sha)],
                "staging_file": str(staging_path),
                "clean_file": str(clean_path),
                "fetch_params": request_params,
                "request_params_sha256": _request_params_sha256(request_params),
                "retry_429_count": int(sym_meta.get("retry_429_count", 0)),
                "total_sleep_seconds": float(sym_meta.get("total_sleep_seconds", 0.0)),
                "backoff_schedule_seconds": list(sym_meta.get("backoff_schedule_seconds", [])),
                "limit_requested": int(sym_meta.get("limit_requested", cfg.alpaca.limit_per_page)),
                "limit_effective": int(sym_meta.get("limit_effective", cfg.alpaca.limit_per_page)),
                "limit_fallback_used": bool(sym_meta.get("limit_fallback_used", False)),
                "limit_fallback_reason": str(sym_meta.get("limit_fallback_reason", "")),
            }
        except Exception as exc:
            qa_symbols[symbol] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_msg": str(exc)[:500],
                "limit_fallback_used": False,
            }
            manifest_symbols[symbol] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_msg": str(exc)[:500],
                "limit_requested": int(cfg.alpaca.limit_per_page),
                "limit_effective": int(cfg.alpaca.limit_per_page),
                "limit_fallback_used": False,
                "limit_fallback_reason": "",
            }

    ok_count = int(sum(1 for s in symbols_sorted if qa_symbols.get(s, {}).get("status") == "ok"))
    failed_count = int(len(symbols_sorted) - ok_count)
    failure_rate = float(failed_count / len(symbols_sorted)) if symbols_sorted else 1.0
    all_failed = bool(ok_count == 0)

    exit_reason = ""
    if all_failed:
        exit_reason = "all_symbols_failed"
    elif failure_rate > float(cfg.alpaca.failure_rate_threshold):
        exit_reason = "failure_rate_threshold_exceeded"
    elif ok_count < int(cfg.alpaca.min_ok_symbols):
        exit_reason = "min_ok_symbols_not_met"

    run_limit_effective = min(limit_effective_candidates) if limit_effective_candidates else int(cfg.alpaca.limit_per_page)

    qa_doc: Dict[str, Any] = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_hash,
        "config_sha256": config_sha256,
        **base_params,
        "limit_effective": int(run_limit_effective),
        **preflight_diag,
        "symbols": {s: qa_symbols[s] for s in symbols_sorted},
        "summary": {
            "ok_count": int(ok_count),
            "failed_count": int(failed_count),
            "failure_rate": float(failure_rate),
            "all_failed": bool(all_failed),
            "min_ok_symbols": int(cfg.alpaca.min_ok_symbols),
            "exit_reason": str(exit_reason),
        },
    }
    manifest_doc: Dict[str, Any] = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_hash,
        "config_sha256": config_sha256,
        **base_params,
        "limit_effective": int(run_limit_effective),
        **preflight_diag,
        "symbols": {s: manifest_symbols[s] for s in symbols_sorted},
        "summary": {
            "ok_count": int(ok_count),
            "failed_count": int(failed_count),
            "failure_rate": float(failure_rate),
            "all_failed": bool(all_failed),
            "min_ok_symbols": int(cfg.alpaca.min_ok_symbols),
            "exit_reason": str(exit_reason),
        },
    }

    qa_path = reports_root / f"{run_id}_qa.json"
    manifest_path = reports_root / f"{run_id}_manifest.json"
    _write_json(qa_path, qa_doc)
    _write_json(manifest_path, manifest_doc)

    summary = {
        "run_id": run_id,
        "qa_report": str(qa_path),
        "manifest_report": str(manifest_path),
        "n_symbols": len(symbols_sorted),
        "config_sha256": config_sha256,
        "exit_reason": str(exit_reason),
    }
    LOGGER.info("ALPACA_FETCH_COMPLETE %s", json.dumps(summary, sort_keys=True))
    if exit_reason:
        raise RuntimeError(exit_reason)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and clean Alpaca minute bars into local cache.")
    parser.add_argument("--config", required=True, help="Path to YAML config (e.g. ./configs/data_alpaca.yaml)")
    parser.add_argument("--symbol", required=False, help="Optional single symbol override for diagnostic runs")
    args = parser.parse_args()

    try:
        run_fetch(Path(args.config).resolve(), symbol_filter=args.symbol)
    except AlpacaAPIError as exc:
        raise SystemExit(f"Alpaca API failure: {exc}")
    except Exception as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
