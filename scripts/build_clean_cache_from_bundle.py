#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_ZIP = "/mnt/data/MarketData-20260225T215136Z-1-001.zip"
DEFAULT_EXTRACT_DIR = "/mnt/data/MarketData_unzipped"
TARGET_START_UTC = "2024-01-01T00:00:00Z"
TARGET_END_UTC = "2024-12-31T23:59:59Z"

TS_ALIASES = ("timestamp", "ts", "datetime", "date", "time")
PART_RE = re.compile(r"part-(\d{4})-([A-Za-z0-9]+)\.parquet$")


@dataclass(frozen=True)
class BuildRow:
    symbol: str
    source_file: str
    selected_year: int
    selected_timeframe: str
    rows_in: int
    rows_out: int
    dropped_nan_ohlcv: int
    dropped_invalid_ohlc: int
    dropped_negative_volume: int
    dropped_duplicates: int
    min_timestamp_utc: str
    max_timestamp_utc: str


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_write(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _find_col_name(columns: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    lookup = {str(c).strip().lower(): str(c) for c in columns}
    for a in aliases:
        if a in lookup:
            return lookup[a]
    return None


def _extract_bundle(bundle_zip: Path, extract_dir: Path) -> None:
    if not bundle_zip.exists():
        raise RuntimeError(f"Bundle zip not found: {bundle_zip}")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_zip, "r") as zf:
        zf.extractall(extract_dir)


def _print_tree(root: Path, max_depth: int = 3, max_lines: int = 300) -> None:
    n = 0
    for p in sorted(root.rglob("*"), key=lambda x: str(x)):
        rel = p.relative_to(root)
        depth = len(rel.parts)
        if depth > max_depth:
            continue
        print(str(root / rel))
        n += 1
        if n >= max_lines:
            break


def _parse_part_file(path: Path) -> Tuple[int, str]:
    m = PART_RE.match(path.name)
    if m is None:
        raise RuntimeError(f"Unrecognized part filename: {path.name}")
    return int(m.group(1)), str(m.group(2))


def _timeframe_priority(tf: str) -> int:
    t = str(tf).lower()
    if t == "1min":
        return 0
    if t == "5min":
        return 1
    return 9


def choose_preferred_part(symbol_dir: Path, target_year: int = 2024) -> Path:
    parts = sorted(symbol_dir.glob("part-*.parquet"), key=lambda p: p.name)
    if not parts:
        raise RuntimeError(f"No partition parquet files under {symbol_dir}")

    ranked: List[Tuple[Tuple[int, int, int, str], Path]] = []
    for p in parts:
        year, tf = _parse_part_file(p)
        key = (
            0 if int(year) == int(target_year) else 1,
            _timeframe_priority(tf),
            abs(int(year) - int(target_year)),
            p.name,
        )
        ranked.append((key, p))
    ranked.sort(key=lambda x: x[0])
    return ranked[0][1]


def _normalize_symbol_frame(
    src_path: Path,
    symbol: str,
    target_start: pd.Timestamp,
    target_end: pd.Timestamp,
) -> Tuple[pd.DataFrame, Dict[str, int], str, str]:
    raw = pd.read_parquet(src_path)

    ts_col = _find_col_name([str(c) for c in raw.columns], TS_ALIASES)
    if ts_col is not None:
        ts = pd.to_datetime(raw[ts_col], utc=True, errors="coerce")
    elif isinstance(raw.index, pd.DatetimeIndex):
        ts = pd.to_datetime(raw.index, utc=True, errors="coerce")
    else:
        raise RuntimeError(f"{symbol}: missing timestamp column and DatetimeIndex in {src_path}")

    required = {"open", "high", "low", "close", "volume"}
    missing = sorted([c for c in required if c not in raw.columns])
    if missing:
        raise RuntimeError(f"{symbol}: missing required columns {missing} in {src_path}")

    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": pd.to_numeric(raw["open"], errors="coerce").astype(np.float64),
            "high": pd.to_numeric(raw["high"], errors="coerce").astype(np.float64),
            "low": pd.to_numeric(raw["low"], errors="coerce").astype(np.float64),
            "close": pd.to_numeric(raw["close"], errors="coerce").astype(np.float64),
            "volume": pd.to_numeric(raw["volume"], errors="coerce").astype(np.float64),
        }
    )

    rows_in = int(df.shape[0])
    df = df.dropna(subset=["timestamp"]).copy()
    df = df[(df["timestamp"] >= target_start) & (df["timestamp"] <= target_end)].copy()
    df = df.sort_values(["timestamp"], kind="mergesort")

    before_dup = int(df.shape[0])
    df = df.drop_duplicates(subset=["timestamp"], keep="first")
    dropped_duplicates = int(before_dup - df.shape[0])

    nan_mask = ~(np.isfinite(df["open"]) & np.isfinite(df["high"]) & np.isfinite(df["low"]) & np.isfinite(df["close"]) & np.isfinite(df["volume"]))
    dropped_nan = int(nan_mask.sum())
    df = df.loc[~nan_mask].copy()

    invalid_ohlc = (
        (df["high"] < df["low"])
        | (df["open"] < df["low"])
        | (df["open"] > df["high"])
        | (df["close"] < df["low"])
        | (df["close"] > df["high"])
    )
    dropped_invalid = int(invalid_ohlc.sum())
    df = df.loc[~invalid_ohlc].copy()

    negative_volume = df["volume"] < 0.0
    dropped_negative = int(negative_volume.sum())
    df = df.loc[~negative_volume].copy()

    df = df.sort_values(["timestamp"], kind="mergesort").reset_index(drop=True)

    if df.empty:
        min_ts = ""
        max_ts = ""
    else:
        min_ts = str(df["timestamp"].iloc[0])
        max_ts = str(df["timestamp"].iloc[-1])

    stats = {
        "rows_in": int(rows_in),
        "rows_out": int(df.shape[0]),
        "dropped_nan_ohlcv": int(dropped_nan),
        "dropped_invalid_ohlc": int(dropped_invalid),
        "dropped_negative_volume": int(dropped_negative),
        "dropped_duplicates": int(dropped_duplicates),
    }
    return df, stats, min_ts, max_ts


def build_clean_cache_from_bundle(
    bundle_zip: Path,
    extract_dir: Path,
    repo_root: Path,
    target_year: int = 2024,
) -> Dict[str, Any]:
    ts = _utc_stamp()
    target_start = pd.Timestamp(TARGET_START_UTC, tz="UTC")
    target_end = pd.Timestamp(TARGET_END_UTC, tz="UTC")

    _extract_bundle(bundle_zip, extract_dir)

    source_root = extract_dir / "MarketData"
    if not source_root.exists():
        raise RuntimeError(f"Extracted bundle missing MarketData root: {source_root}")

    print("EXTRACTED_TREE_TOP3")
    print(str(extract_dir))
    _print_tree(extract_dir, max_depth=3)

    symbol_dirs = sorted([p for p in source_root.iterdir() if p.is_dir()], key=lambda p: p.name.upper())
    if not symbol_dirs:
        raise RuntimeError(f"No symbol folders found under {source_root}")

    clean_dir = repo_root / "data" / "alpaca" / "clean"
    build_tmp = repo_root / "data" / "alpaca" / f"clean_build_tmp_{ts}"
    backup_dir = repo_root / "data" / "alpaca" / f"clean_backup_{ts}"
    build_tmp.mkdir(parents=True, exist_ok=False)

    rows: List[BuildRow] = []
    for sym_dir in symbol_dirs:
        symbol = sym_dir.name.strip().upper()
        chosen = choose_preferred_part(sym_dir, target_year=target_year)
        year, tf = _parse_part_file(chosen)
        norm, st, min_ts, max_ts = _normalize_symbol_frame(
            src_path=chosen,
            symbol=symbol,
            target_start=target_start,
            target_end=target_end,
        )
        out_path = build_tmp / f"{symbol}.parquet"
        norm.to_parquet(out_path, index=False)
        rows.append(
            BuildRow(
                symbol=symbol,
                source_file=str(chosen),
                selected_year=int(year),
                selected_timeframe=str(tf),
                rows_in=int(st["rows_in"]),
                rows_out=int(st["rows_out"]),
                dropped_nan_ohlcv=int(st["dropped_nan_ohlcv"]),
                dropped_invalid_ohlc=int(st["dropped_invalid_ohlc"]),
                dropped_negative_volume=int(st["dropped_negative_volume"]),
                dropped_duplicates=int(st["dropped_duplicates"]),
                min_timestamp_utc=min_ts,
                max_timestamp_utc=max_ts,
            )
        )

    build_art_dir = repo_root / "artifacts" / "clean_cache_build" / ts
    build_manifest_path = build_art_dir / "build_manifest.json"

    build_manifest: Dict[str, Any] = {
        "timestamp": ts,
        "bundle_zip": str(bundle_zip.resolve()),
        "bundle_zip_sha256": _sha256_file(bundle_zip),
        "extracted_path": str(extract_dir.resolve()),
        "target_window_utc": {"start": TARGET_START_UTC, "end": TARGET_END_UTC},
        "symbols_built": [r.symbol for r in rows],
        "rows": [r.__dict__ for r in rows],
        "clean_build_tmp": str(build_tmp.resolve()),
        "clean_dir_final": str(clean_dir.resolve()),
        "clean_backup_dir": str(backup_dir.resolve()),
    }
    _json_write(build_manifest_path, build_manifest)

    moved_old = False
    try:
        if clean_dir.exists():
            clean_dir.rename(backup_dir)
            moved_old = True
        build_tmp.rename(clean_dir)
    except Exception:
        if (not clean_dir.exists()) and moved_old and backup_dir.exists():
            backup_dir.rename(clean_dir)
        raise

    final_symbols = sorted([p.stem.upper() for p in clean_dir.glob("*.parquet")])
    print("NEW_CLEAN_CACHE_SYMBOLS", len(final_symbols), final_symbols)
    for sym in final_symbols[:2]:
        p = clean_dir / f"{sym}.parquet"
        d = pd.read_parquet(p)
        ts_col = d["timestamp"] if "timestamp" in d.columns else pd.Series([], dtype="datetime64[ns, UTC]")
        ts_parsed = pd.to_datetime(ts_col, utc=True, errors="coerce")
        print(
            "SANITY",
            sym,
            "rows=",
            int(d.shape[0]),
            "head_ts=",
            [str(x) for x in ts_parsed.head(3).tolist()],
            "tail_ts=",
            [str(x) for x in ts_parsed.tail(3).tolist()],
        )

    build_manifest["final_clean_symbols"] = final_symbols
    _json_write(build_manifest_path, build_manifest)
    return build_manifest


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build atomic clean cache from MarketData bundle")
    p.add_argument("--bundle-zip", default=DEFAULT_BUNDLE_ZIP)
    p.add_argument("--extract-dir", default=DEFAULT_EXTRACT_DIR)
    p.add_argument("--target-year", type=int, default=2024)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    manifest = build_clean_cache_from_bundle(
        bundle_zip=Path(args.bundle_zip).resolve(),
        extract_dir=Path(args.extract_dir).resolve(),
        repo_root=REPO_ROOT,
        target_year=int(args.target_year),
    )
    print("CLEAN_CACHE_BUILD_COMPLETE")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
