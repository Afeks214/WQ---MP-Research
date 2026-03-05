#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_FAMILIES = (
    "sweep_family_sprinters",
    "sweep_family_surfers",
    "sweep_family_snipers",
    "sweep_family_marathoners",
)


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if np.isfinite(x):
            return float(x)
    except Exception:
        pass
    return float(default)


def _read_summary(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(doc, dict):
            return doc
    except Exception:
        pass
    return {}


def _parse_iso_utc(ts: str) -> datetime:
    try:
        x = str(ts).strip()
        if x.endswith("Z"):
            x = x[:-1] + "+00:00"
        dt = datetime.fromisoformat(x)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception as exc:
        raise RuntimeError(f"Invalid ISO timestamp in summary: {ts!r}") from exc


def _detect_stale_artifacts(family_dir: Path, summary: dict[str, Any]) -> None:
    results_path = family_dir / "results.parquet"
    expected_file_hash = str(summary.get("results_sha256_file", "")).strip()
    if not expected_file_hash:
        raise RuntimeError(f"Stale artifacts detected in folder. Manual cleanup required. Missing results_sha256_file in {family_dir}/summary.json")
    actual_file_hash = hashlib.sha256(results_path.read_bytes()).hexdigest()
    if actual_file_hash != expected_file_hash:
        raise RuntimeError(
            "Stale artifacts detected in folder. Manual cleanup required. "
            f"File hash mismatch for {family_dir}: summary={expected_file_hash} actual={actual_file_hash}"
        )
    finished_s = summary.get("family_finished_utc")
    if not finished_s:
        raise RuntimeError(f"Stale artifacts detected in folder. Manual cleanup required. Missing family_finished_utc in {family_dir}/summary.json")
    _parse_iso_utc(str(finished_s))


def _resolve_family_candidates(artifacts_root: Path) -> dict[str, Path]:
    dirs = sorted([p for p in artifacts_root.glob("sweep_family_*") if p.is_dir()])
    buckets: dict[str, list[tuple[datetime, Path]]] = {k: [] for k in REQUIRED_FAMILIES}
    for d in dirs:
        name = d.name
        fam_match = None
        for fam in REQUIRED_FAMILIES:
            if name == fam or name.startswith(fam + "_") or name.startswith(fam + "-"):
                fam_match = fam
                break
        if fam_match is None:
            continue
        summary = _read_summary(d / "summary.json")
        ts = summary.get("family_finished_utc") if isinstance(summary, dict) else None
        dt = _parse_iso_utc(str(ts)) if ts else datetime.fromtimestamp(d.stat().st_mtime, tz=timezone.utc)
        buckets[fam_match].append((dt, d))

    missing = [k for k in REQUIRED_FAMILIES if len(buckets[k]) == 0]
    if missing:
        raise RuntimeError(f"Missing required family artifacts: {missing}")

    chosen: dict[str, Path] = {}
    for fam, items in buckets.items():
        items_sorted = sorted(items, key=lambda x: (x[0], str(x[1])), reverse=True)
        chosen[fam] = items_sorted[0][1]
    return chosen


def _load_family_tables(artifacts_root: Path) -> tuple[pd.DataFrame, dict[str, float], dict[str, str]]:
    frames: list[pd.DataFrame] = []
    family_penalty: dict[str, float] = {}
    chosen_paths: dict[str, str] = {}

    family_dirs = _resolve_family_candidates(artifacts_root)
    for family_name in REQUIRED_FAMILIES:
        family_dir = family_dirs[family_name]
        results_path = family_dir / "results.parquet"
        summary_path = family_dir / "summary.json"
        if not results_path.exists():
            raise RuntimeError(f"Missing results.parquet for required family: {family_dir}")
        if not summary_path.exists():
            raise RuntimeError(f"Missing summary.json for required family: {family_dir}")

        df = pd.read_parquet(results_path)
        if df.empty:
            raise RuntimeError(f"Empty results.parquet for required family: {family_dir}")

        summary = _read_summary(summary_path)
        if not summary:
            raise RuntimeError(f"Invalid summary.json for required family: {family_dir}")
        _detect_stale_artifacts(family_dir, summary)
        gap_rate = _safe_float(summary.get("gap_reset_stats", {}).get("gap_reset_rate", 0.0), default=0.0)
        # Deterministic stability penalty:
        # no penalty up to 0.5% reset-rate, linear to cap 1.0 at +2% above threshold.
        penalty = float(np.clip(max(0.0, gap_rate - 0.005) / 0.02, 0.0, 1.0))
        family_penalty[str(family_dir.name)] = penalty
        chosen_paths[family_name] = str(family_dir)

        df = df.copy()
        if "family" not in df.columns:
            df["family"] = str(family_name)
        frames.append(df)

    if not frames:
        return pd.DataFrame(), family_penalty, chosen_paths
    merged = pd.concat(frames, axis=0, ignore_index=True)
    return merged, family_penalty, chosen_paths


def _select_base_score(df: pd.DataFrame) -> pd.Series:
    for col in ("robustness_score", "cum_return", "avg_ret", "profit_factor"):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.zeros(len(df), dtype=np.float64), index=df.index, dtype=np.float64)


def _finite_filter(df: pd.DataFrame, required_numeric: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in required_numeric:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    mask = np.ones(len(out), dtype=bool)
    for col in required_numeric:
        mask &= np.isfinite(out[col].to_numpy(dtype=np.float64))
    return out.loc[mask].copy()


def _render_top50_markdown(df: pd.DataFrame, out_path: Path) -> None:
    cols = [
        c
        for c in (
            "rank",
            "family",
            "config_id",
            "seed",
            "composite_score",
            "base_score",
            "stability_penalty",
            "trades",
            "win_rate",
            "avg_ret",
            "max_drawdown",
        )
        if c in df.columns
    ]
    top = df.head(50).copy()
    if len(top) > 0:
        try:
            table = top.loc[:, cols].to_markdown(index=False)
        except Exception:
            table = "```\n" + top.loc[:, cols].to_string(index=False) + "\n```"
    else:
        table = "_No rows_"
    lines = [
        "# Family Leaderboard Top 50",
        "",
        f"Rows: {len(top)}",
        "",
        table,
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate family sweep results into one leaderboard")
    parser.add_argument("--artifacts-root", default="artifacts", help="Artifacts root directory")
    parser.add_argument("--min-trades", type=int, default=10, help="Minimum trades required")
    args = parser.parse_args()

    artifacts_root = Path(args.artifacts_root).expanduser().resolve()
    artifacts_root.mkdir(parents=True, exist_ok=True)

    merged, family_penalty, chosen_paths = _load_family_tables(artifacts_root)
    if merged.empty:
        raise SystemExit("No family results.parquet files found under artifacts/sweep_family_*")

    required_cols = ["family", "config_id", "seed"]
    missing = [c for c in required_cols if c not in merged.columns]
    if missing:
        raise SystemExit(f"Required columns missing from merged family results: {missing}")

    required_numeric = ["trades", "win_rate", "avg_ret", "med_ret", "profit_factor", "max_drawdown"]
    clean = _finite_filter(merged, required_numeric=required_numeric)
    clean = clean.loc[pd.to_numeric(clean["trades"], errors="coerce") >= int(args.min_trades)].copy()
    if clean.empty:
        raise SystemExit("No rows remain after finite filter + min-trades gate")

    clean["base_score"] = _select_base_score(clean)
    base_vals = pd.to_numeric(clean["base_score"], errors="coerce").to_numpy(dtype=np.float64)
    clean = clean.loc[np.isfinite(base_vals)].copy()
    if clean.empty:
        raise SystemExit("No rows remain after finite base_score gate")

    clean["family"] = clean["family"].astype(str)
    clean["stability_penalty"] = clean["family"].map(lambda x: float(family_penalty.get(str(x), 0.0))).astype(np.float64)
    clean["composite_score"] = clean["base_score"].astype(np.float64) - clean["stability_penalty"].astype(np.float64)
    comp = clean["composite_score"].to_numpy(dtype=np.float64)
    clean = clean.loc[np.isfinite(comp)].copy()
    if clean.empty:
        raise SystemExit("No rows remain after finite composite_score gate")

    clean = clean.sort_values(
        ["composite_score", "base_score", "family", "config_id", "seed"],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    clean["rank"] = np.arange(1, len(clean) + 1, dtype=np.int64)

    out_parquet = artifacts_root / "leaderboard.parquet"
    out_md = artifacts_root / "leaderboard_top50.md"
    clean.to_parquet(out_parquet, index=False)
    _render_top50_markdown(clean, out_md)

    print(f"AGGREGATION_OK rows={len(clean)} out={out_parquet}")
    print(f"TOP50_MD={out_md}")
    print("FAMILY_SELECTIONS")
    for fam in REQUIRED_FAMILIES:
        print(f"{fam} => {chosen_paths.get(fam, 'N/A')}")


if __name__ == "__main__":
    main()
