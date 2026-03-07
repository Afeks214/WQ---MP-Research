#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _to_float(v: Any) -> float:
    try:
        x = float(v)
        return x
    except Exception:
        return float('nan')


def _find_newest_run(artifacts_root: Path) -> Path | None:
    runs = [p for p in artifacts_root.iterdir() if p.is_dir() and p.name != "_orchestrator_tmp"]
    if not runs:
        return None
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _print_tree(root: Path, max_depth: int = 4) -> None:
    print(f"Artifact tree (max_depth={max_depth}):")
    print(root.name + "/")
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        depth = len(rel.parts)
        if depth > max_depth:
            continue
        indent = "  " * depth
        suffix = "/" if p.is_dir() else ""
        print(f"{indent}{rel.name}{suffix}")


def _read_run_status(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "run_status.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _pick_primary_csv(run_dir: Path) -> Path | None:
    preferred = [
        "robustness_leaderboard.csv",
        "leaderboard.csv",
        "metrics.csv",
        "candidate_metrics.csv",
    ]
    for name in preferred:
        p = run_dir / name
        if p.exists() and p.is_file():
            return p
    # Fallback: first CSV found in run root.
    csvs = sorted([p for p in run_dir.glob("*.csv") if p.is_file()])
    return csvs[0] if csvs else None


def _pick_perf_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "robustness_score",
        "dsr_median",
        "sharpe",
        "Sharpe",
        "total_pnl",
        "Total PnL",
        "cum_return",
        "pnl",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return numeric_cols[0] if numeric_cols else None


def _load_candidate_m4_map(run_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    croot = run_dir / "candidates"
    if not croot.exists() or not croot.is_dir():
        return out
    for cand_dir in sorted([p for p in croot.iterdir() if p.is_dir()]):
        cand_id = cand_dir.name
        cfg_path = cand_dir / "candidate_config.json"
        if not cfg_path.exists():
            continue
        try:
            doc = json.loads(cfg_path.read_text(encoding="utf-8"))
            m4 = doc.get("module4_config", {}) if isinstance(doc, dict) else {}
            if not isinstance(m4, dict):
                m4 = {}
            out[cand_id] = m4
        except Exception:
            continue
    return out


def _format_table(df: pd.DataFrame, cols: list[str]) -> str:
    view = df.loc[:, cols].copy()
    # deterministic formatting
    for c in view.columns:
        if pd.api.types.is_float_dtype(view[c]):
            view[c] = view[c].map(lambda x: "" if pd.isna(x) else f"{float(x):.6g}")
    return view.to_string(index=False)


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    artifacts = repo / "artifacts"

    if not artifacts.exists():
        print("FAIL: artifacts directory not found")
        return 1

    run_dir = _find_newest_run(artifacts)
    if run_dir is None:
        print("FAIL: no run directories found under artifacts")
        return 1

    print(f"Newest run directory: {run_dir}")
    _print_tree(run_dir, max_depth=4)

    status = _read_run_status(run_dir)
    tasks_done = int(status.get("tasks_done", 0) or 0)
    tasks_total = int(status.get("tasks_total", 0) or 0)
    elapsed = float(status.get("elapsed_seconds", 0.0) or 0.0)
    print(f"Run progress: {tasks_done}/{tasks_total} tasks, elapsed={elapsed:.2f}s")

    csv_path = _pick_primary_csv(run_dir)
    if csv_path is None:
        print(f"Results CSV not generated yet, currently at {tasks_done}/{tasks_total} tasks.")
        return 0

    print(f"Primary results CSV: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"FAIL: unable to read CSV {csv_path}: {exc}")
        return 1

    if df.empty:
        print(f"Results CSV is empty, currently at {tasks_done}/{tasks_total} tasks.")
        return 0

    perf_col = _pick_perf_column(df)
    if perf_col is None:
        print("FAIL: no numeric performance column found in results CSV.")
        return 1

    df["__perf"] = pd.to_numeric(df[perf_col], errors="coerce")
    df = df.sort_values(["__perf", "candidate_id" if "candidate_id" in df.columns else df.columns[0]], ascending=[False, True], kind="mergesort")
    top = df.head(10).copy()

    # Candidate ID extraction
    if "candidate_id" not in top.columns:
        top["candidate_id"] = top.index.astype(str)

    m4_map = _load_candidate_m4_map(run_dir)

    def m4_val(cid: str, key: str) -> Any:
        m = m4_map.get(str(cid), {})
        if key in m:
            return m.get(key)
        return np.nan

    # Fill strategy columns from CSV if present, else from candidate_config.json mapping
    for key in ["strategy_type", "delta_th", "dev_th", "score_gate"]:
        if key not in top.columns:
            top[key] = top["candidate_id"].map(lambda cid: m4_val(str(cid), key))

    # Derive display metrics
    if "win_rate" not in top.columns:
        top["win_rate"] = np.nan
    if "max_drawdown" not in top.columns:
        top["max_drawdown"] = np.nan

    if "sharpe" in top.columns:
        top["sharpe_display"] = top["sharpe"]
    elif "Sharpe" in top.columns:
        top["sharpe_display"] = top["Sharpe"]
    elif "dsr_median" in top.columns:
        top["sharpe_display"] = top["dsr_median"]
    else:
        top["sharpe_display"] = top["__perf"]

    show_cols = [
        "candidate_id",
        "strategy_type",
        "delta_th",
        "dev_th",
        "score_gate",
        "sharpe_display",
        "win_rate",
        "max_drawdown",
    ]
    show_cols = [c for c in show_cols if c in top.columns]

    print("\nTop 10 strategies:")
    print(_format_table(top, show_cols))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
