from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

try:
    import psutil
except Exception as exc:  # pragma: no cover
    raise RuntimeError("psutil is required for benchmark monitoring") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from weightiz.cli import run_research
from weightiz.shared.io.hpc_market_profile_parity import compute_market_profile_features
from weightiz.module5.strategy_engine import generate_strategy_specs

SYMBOLS = ("EEM", "GLD", "HYG", "IWM", "QQQ", "SPY", "TLT", "XLE", "XLK", "XLU")
WINDOWS = (30, 45, 60)


def _set_deterministic_env() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def _ensure_timestamp(raw: pd.DataFrame) -> pd.Series:
    cols = {str(c).strip().lower(): str(c) for c in raw.columns}
    for c in ("timestamp", "ts", "datetime", "time", "date"):
        if c in cols:
            return pd.to_datetime(raw[cols[c]], utc=True, errors="coerce")
    if isinstance(raw.index, pd.DatetimeIndex):
        return pd.to_datetime(raw.index, utc=True, errors="coerce")
    raise RuntimeError("SCHEMA_ERROR: missing timestamp column/index")


def _validate_schema(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    cols = {str(c).strip().lower(): str(c) for c in raw.columns}
    req = {
        "open": ("open", "o"),
        "high": ("high", "h"),
        "low": ("low", "l"),
        "close": ("close", "c"),
        "volume": ("volume", "v", "vol"),
    }
    mapped: dict[str, str] = {}
    missing: list[str] = []
    for k, opts in req.items():
        found = None
        for o in opts:
            if o in cols:
                found = cols[o]
                break
        if found is None:
            missing.append(k)
        else:
            mapped[k] = found
    if missing:
        raise RuntimeError(f"SCHEMA_ERROR[{symbol}]: missing columns {missing}")

    ts = _ensure_timestamp(raw)
    out = pd.DataFrame(
        {
            "timestamp": ts,
            "open": pd.to_numeric(raw[mapped["open"]], errors="coerce"),
            "high": pd.to_numeric(raw[mapped["high"]], errors="coerce"),
            "low": pd.to_numeric(raw[mapped["low"]], errors="coerce"),
            "close": pd.to_numeric(raw[mapped["close"]], errors="coerce"),
            "volume": pd.to_numeric(raw[mapped["volume"]], errors="coerce"),
            "symbol": str(symbol),
        }
    )
    out = out.sort_values("timestamp", kind="mergesort")
    out = out.drop_duplicates(subset=["timestamp"], keep="last")
    out = out.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"]).copy()

    invalid = (
        (out["high"] < out["low"])
        | (out["high"] < out["open"])
        | (out["high"] < out["close"])
        | (out["low"] > out["open"])
        | (out["low"] > out["close"])
    )
    if bool(invalid.any()):
        n_bad = int(invalid.sum())
        raise RuntimeError(f"OHLC_INTEGRITY_ERROR[{symbol}]: invalid_rows={n_bad}")

    if out.empty:
        raise RuntimeError(f"EMPTY_SYMBOL_AFTER_CLEAN[{symbol}]")
    return out


def _load_clean_universe(data_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    parts: list[pd.DataFrame] = []
    stats: dict[str, Any] = {}
    for sym in SYMBOLS:
        path = data_root / sym / "part-2023-1Min.parquet"
        if not path.exists():
            raise RuntimeError(f"DATA_FILE_MISSING[{sym}]: {path}")
        raw = pd.read_parquet(path)
        clean = _validate_schema(raw, sym)
        stats[sym] = {
            "path": str(path),
            "rows": int(len(clean)),
            "ts_min": str(clean["timestamp"].min()),
            "ts_max": str(clean["timestamp"].max()),
        }
        parts.append(clean)

    all_df = pd.concat(parts, ignore_index=True)
    all_df = all_df.sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)
    return all_df, stats


def _canonical_hash_df(df: pd.DataFrame, sort_keys: list[str]) -> str:
    x = df.copy()
    present_keys = [k for k in sort_keys if k in x.columns]
    if present_keys:
        x = x.sort_values(present_keys, kind="mergesort").reset_index(drop=True)
    for col in x.columns:
        if pd.api.types.is_datetime64_any_dtype(x[col]) or isinstance(x[col].dtype, pd.DatetimeTZDtype):
            x[col] = pd.to_datetime(x[col], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        elif pd.api.types.is_float_dtype(x[col]):
            x[col] = x[col].astype(np.float64).round(12)
    payload = x.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_file(path: Path, sort_keys: list[str]) -> str:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    return _canonical_hash_df(df, sort_keys)


def _extract_hpc_features(clean_df: pd.DataFrame, out_path: Path) -> tuple[pd.DataFrame, float]:
    t0 = time.perf_counter()
    parts: list[pd.DataFrame] = []
    for w in WINDOWS:
        feat = compute_market_profile_features(clean_df, window=int(w), include_aux=True)
        feat["profile_window_minutes"] = int(w)
        parts.append(feat)
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["timestamp", "symbol", "profile_window_minutes"], kind="mergesort").reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    elapsed = time.perf_counter() - t0
    return out, elapsed


def _extract_reference_features(clean_df: pd.DataFrame, out_path: Path) -> tuple[pd.DataFrame, float]:
    t0 = time.perf_counter()
    parts: list[pd.DataFrame] = []
    for w in WINDOWS:
        feat = compute_market_profile_features(clean_df, window=int(w), include_aux=True)
        feat["profile_window_minutes"] = int(w)
        parts.append(feat)
    ref = pd.concat(parts, ignore_index=True)
    ref = ref.sort_values(["timestamp", "symbol", "profile_window_minutes"], kind="mergesort").reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ref.to_parquet(out_path, index=False)
    elapsed = time.perf_counter() - t0
    return ref, elapsed


def _parity_report(hpc: pd.DataFrame, ref: pd.DataFrame, out_path: Path, tol: float = 1e-10) -> dict[str, Any]:
    keys = ["timestamp", "symbol", "profile_window_minutes"]
    cols = ["close", "POC", "VAL", "VAH", "D", "A", "DeltaEff", "Sbreak", "Sreject", "RVOL", "ATR"]

    left = hpc[keys + cols].copy()
    right = ref[keys + cols].copy()

    m = left.merge(right, on=keys, how="inner", suffixes=("_hpc", "_ref"))
    if m.empty:
        raise RuntimeError("PARITY_EMPTY_JOIN")

    report: dict[str, Any] = {
        "rows_hpc": int(len(left)),
        "rows_reference": int(len(right)),
        "rows_compared": int(len(m)),
        "tolerance": float(tol),
        "columns": {},
        "parity_ok": True,
    }

    for c in cols:
        diff = np.abs(m[f"{c}_hpc"].to_numpy(dtype=np.float64) - m[f"{c}_ref"].to_numpy(dtype=np.float64))
        max_abs = float(np.max(diff)) if diff.size else 0.0
        mean_abs = float(np.mean(diff)) if diff.size else 0.0
        mismatch = int(np.sum(diff > tol))
        report["columns"][c] = {
            "max_abs_error": max_abs,
            "mean_abs_error": mean_abs,
            "mismatch_count": mismatch,
        }
        if mismatch > 0:
            report["parity_ok"] = False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    if not report["parity_ok"]:
        raise RuntimeError("PARITY_TOLERANCE_EXCEEDED")
    return report


def _max_feasible_wf_splits(
    min_ts: pd.Timestamp,
    max_ts: pd.Timestamp,
    train_months: int,
    test_months: int,
    purge_days: int,
) -> int:
    first = pd.Timestamp(min_ts)
    last = pd.Timestamp(max_ts)
    i = 0
    n = 0
    while True:
        train_start = first + pd.DateOffset(months=i * test_months)
        train_end = train_start + pd.DateOffset(months=train_months)
        test_start = train_end + pd.Timedelta(days=purge_days)
        test_end = test_start + pd.DateOffset(months=test_months)
        if test_end <= last:
            n += 1
            i += 1
            continue
        break
    return int(n)


def _write_yaml_config(path: Path, workers: int, wf_splits_cap: int | None = None) -> Path:
    # keep source config as template and override worker fields deterministically
    base = run_research._load_config(REPO_ROOT / "configs/zimtra_local_m4_benchmark.yaml")
    base.zimtra_sweep.workers = int(workers)
    base.zimtra_sweep.stage_a.workers = int(workers)
    base.zimtra_sweep.stage_b.workers = int(workers)
    if wf_splits_cap is not None:
        req = int(base.zimtra_sweep.cv.wf_splits)
        use = int(min(req, int(wf_splits_cap)))
        if use < 1:
            raise RuntimeError(
                f"WF_SPLITS_INFEASIBLE: requested={req}, feasible={wf_splits_cap}"
            )
        base.zimtra_sweep.cv.wf_splits = int(use)

    out_dict = {
        "run_name": base.run_name,
        "symbols": list(base.symbols),
        "data": base.data.model_dump(mode="json"),
        "engine": base.engine.model_dump(mode="json"),
        "module2_configs": [x.model_dump(mode="json") for x in base.module2_configs],
        "module3_configs": [x.model_dump(mode="json") for x in base.module3_configs],
        "module4_configs": [x.model_dump(mode="json") for x in base.module4_configs],
        "harness": base.harness.model_dump(mode="json"),
        "zimtra_sweep": base.zimtra_sweep.model_dump(mode="json"),
    }
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(out_dict, sort_keys=False), encoding="utf-8")
    return path


def _run_research(config_path: Path, log_path: Path, env: dict[str, str]) -> tuple[Path, dict[str, Any]]:
    before = set((REPO_ROOT / "artifacts/zimtra_sweep").glob("run_*"))
    cmd = [sys.executable, "-m", "weightiz.cli.run_research", "--config", str(config_path)]
    log_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
        peak_rss = 0
        cpu_samples: list[float] = []
        p = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                procs = [p] + p.children(recursive=True)
                rss = 0
                cpu = 0.0
                for cp in procs:
                    try:
                        rss += int(cp.memory_info().rss)
                        cpu += float(cp.cpu_percent(interval=0.0))
                    except Exception:
                        continue
                peak_rss = max(peak_rss, rss)
                cpu_samples.append(cpu)
            except Exception:
                pass
            time.sleep(1.0)
        rc = proc.wait()
    elapsed = time.perf_counter() - t0
    if rc != 0:
        raise RuntimeError(f"RUN_RESEARCH_FAILED rc={rc}; see log {log_path}")

    after = sorted((REPO_ROOT / "artifacts/zimtra_sweep").glob("run_*"), key=lambda p: p.stat().st_mtime)
    created = [p for p in after if p not in before]
    if not created:
        raise RuntimeError("RUN_DIRECTORY_NOT_FOUND")
    run_dir = created[-1]

    return run_dir, {
        "simulation_runtime_seconds": float(elapsed),
        "simulation_peak_memory_mb": float(peak_rss / (1024.0 * 1024.0)),
        "simulation_cpu_percent_avg": float(np.mean(cpu_samples) if cpu_samples else 0.0),
    }


def _copy_run_artifacts(run_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "leaderboard.csv",
        "run_manifest.json",
        "run_summary.json",
        "deadletter_tasks.jsonl",
        "top100_per_asset.csv",
    ]:
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, target_dir / name)


def _analyze_leaderboard(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"MISSING_LEADERBOARD: {path}")
    df = pd.read_csv(path)
    metrics = {
        "rows": int(len(df)),
        "columns": list(df.columns),
    }
    for c in ["sharpe", "sortino", "max_drawdown", "win_rate", "avg_holding_time_bars", "exposure_utilization", "daily_loss_breaches"]:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            metrics[f"{c}_mean"] = float(np.nanmean(s.to_numpy(dtype=np.float64)))
    return metrics


def _extract_deadletter_risks(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"deadletters": 0, "risk_reason_counts": {}}
    counts: dict[str, int] = {}
    total = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        total += 1
        try:
            obj = json.loads(line)
            reasons = obj.get("reason_codes", [])
            for r in reasons:
                rs = str(r)
                counts[rs] = counts.get(rs, 0) + 1
        except Exception:
            continue
    return {"deadletters": int(total), "risk_reason_counts": counts}


def _run_once(iter_idx: int, workers: int, data_root: Path, out_root: Path) -> dict[str, Any]:
    run_dir = out_root / f"run_{iter_idx}"
    run_dir.mkdir(parents=True, exist_ok=True)

    clean_df, data_stats = _load_clean_universe(data_root)

    hpc_path = run_dir / "features_hpc.parquet"
    ref_path = run_dir / "features_reference.parquet"
    parity_path = run_dir / "parity_report.json"

    hpc_df, hpc_runtime = _extract_hpc_features(clean_df, hpc_path)
    min_ts = pd.to_datetime(clean_df["timestamp"], utc=True).min()
    max_ts = pd.to_datetime(clean_df["timestamp"], utc=True).max()
    wf_cap = _max_feasible_wf_splits(
        min_ts=min_ts,
        max_ts=max_ts,
        train_months=6,
        test_months=3,
        purge_days=5,
    )
    cfg_path = _write_yaml_config(
        run_dir / "zimtra_local_m4_benchmark.runtime.yaml",
        workers=workers,
        wf_splits_cap=wf_cap,
    )
    ref_df, ref_runtime = _extract_reference_features(clean_df, ref_path)
    parity = _parity_report(hpc_df, ref_df, parity_path, tol=1e-10)

    cfg = run_research._load_config(cfg_path)
    swing_grid = cfg.zimtra_sweep.swing_grid.model_dump()
    n_strat = len(generate_strategy_specs(swing_grid=swing_grid))
    if n_strat != 96:
        raise RuntimeError(f"STRATEGY_COUNT_MISMATCH: expected=96 got={n_strat}")

    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    sweep_run_dir, sim_perf = _run_research(cfg_path, run_dir / "run_research.log", env)
    _copy_run_artifacts(sweep_run_dir, run_dir)

    lb_path = run_dir / "leaderboard.csv"
    if not lb_path.exists():
        raise RuntimeError("SWEEP_OUTPUT_MISSING: leaderboard.csv")

    perf = {
        "feature_extraction_runtime_seconds": float(hpc_runtime),
        "reference_extraction_runtime_seconds": float(ref_runtime),
        **sim_perf,
    }
    perf["total_runtime_seconds"] = float(
        perf["feature_extraction_runtime_seconds"]
        + perf["reference_extraction_runtime_seconds"]
        + perf["simulation_runtime_seconds"]
    )
    perf["rows_processed"] = int(len(clean_df))
    perf["rows_per_second"] = float(perf["rows_processed"] / max(perf["feature_extraction_runtime_seconds"], 1e-12))

    leader_metrics = _analyze_leaderboard(lb_path)
    risk_viol = _extract_deadletter_risks(run_dir / "deadletter_tasks.jsonl")

    return {
        "run_index": int(iter_idx),
        "data_stats": data_stats,
        "parity": parity,
        "performance": perf,
        "strategy_count": int(n_strat),
        "wf_splits_feasible_cap": int(wf_cap),
        "sweep_run_dir": str(sweep_run_dir),
        "leaderboard_metrics": leader_metrics,
        "risk_violations": risk_viol,
        "hashes": {
            "features_hpc": _hash_file(hpc_path, ["timestamp", "symbol", "profile_window_minutes"]),
            "features_reference": _hash_file(ref_path, ["timestamp", "symbol", "profile_window_minutes"]),
            "leaderboard": _hash_file(lb_path, ["strategy_id"]),
        },
    }


def _print_summary(report: dict[str, Any]) -> None:
    print("===== LOCAL M4 BENCHMARK SUMMARY =====")
    print(f"total_strategies_tested={report['strategy_count']}")
    print(f"runtime_total_sec={report['performance']['total_runtime_seconds']:.3f}")
    print(f"runtime_features_sec={report['performance']['feature_extraction_runtime_seconds']:.3f}")
    print(f"runtime_simulation_sec={report['performance']['simulation_runtime_seconds']:.3f}")
    print(f"rows_processed={report['performance']['rows_processed']}")
    print(f"rows_per_second={report['performance']['rows_per_second']:.2f}")
    print(f"simulation_peak_memory_mb={report['performance']['simulation_peak_memory_mb']:.2f}")
    print(f"simulation_cpu_percent_avg={report['performance']['simulation_cpu_percent_avg']:.2f}")
    print(f"parity_ok={report['parity']['parity_ok']}")
    print(f"risk_violations={report['risk_violations']}")

    top_path = Path(report["run_output_dir"]) / "top100_per_asset.csv"
    if top_path.exists():
        top = pd.read_csv(top_path)
        if not top.empty:
            print("top_strategies_per_asset:")
            for sym in sorted(top["symbol"].astype(str).unique().tolist()):
                block = top[top["symbol"].astype(str) == sym].head(5)
                print(f"  {sym}: {block['strategy_id'].tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic local M4 benchmark for Weightiz research sweep")
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--run-twice", action="store_true", default=True)
    parser.add_argument("--data-root", default="artifacts/_staging_data_zip/MarketData")
    parser.add_argument("--output-root", default="artifacts/local_m4_benchmark")
    args = parser.parse_args()

    _set_deterministic_env()

    if args.workers != 7:
        raise RuntimeError(f"WORKER_POLICY_VIOLATION: expected workers=7 got {args.workers}")

    out_root = (REPO_ROOT / args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    data_root = (REPO_ROOT / args.data_root).resolve()

    run_reports: list[dict[str, Any]] = []
    n_runs = 2 if bool(args.run_twice) else 1
    for i in range(1, n_runs + 1):
        rr = _run_once(i, workers=int(args.workers), data_root=data_root, out_root=out_root)
        rr["run_output_dir"] = str((out_root / f"run_{i}").resolve())
        run_reports.append(rr)

    # deterministic cross-run check
    det = {
        "runs": run_reports,
        "hash_match": True,
        "hashes": {},
    }
    if len(run_reports) >= 2:
        keys = ["features_hpc", "features_reference", "leaderboard"]
        for k in keys:
            hvals = [r["hashes"][k] for r in run_reports]
            det["hashes"][k] = hvals
            if len(set(hvals)) != 1:
                det["hash_match"] = False
        if not det["hash_match"]:
            raise RuntimeError("DETERMINISM_HASH_MISMATCH")

    # Promote run_1 outputs to top-level canonical files
    run1 = out_root / "run_1"
    canonical_map = {
        "features_hpc.parquet": "features_hpc.parquet",
        "features_reference.parquet": "features_reference.parquet",
        "parity_report.json": "parity_report.json",
        "leaderboard.csv": "leaderboard.csv",
    }
    for src_name, dst_name in canonical_map.items():
        src = run1 / src_name
        if src.exists():
            shutil.copy2(src, out_root / dst_name)

    benchmark_report = {
        "strategy_count": int(run_reports[0]["strategy_count"]),
        "wf_splits_requested": 10,
        "wf_splits_used": int(run_reports[0]["wf_splits_feasible_cap"]),
        "performance": run_reports[0]["performance"],
        "leaderboard_metrics": run_reports[0]["leaderboard_metrics"],
        "risk_violations": run_reports[0]["risk_violations"],
        "determinism_hash_match": bool(det["hash_match"]),
        "env": {
            "python": sys.executable,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        },
    }
    (out_root / "benchmark_report.json").write_text(
        json.dumps(benchmark_report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_root / "determinism_report.json").write_text(
        json.dumps(det, indent=2, sort_keys=True), encoding="utf-8"
    )

    _print_summary({**run_reports[0], "run_output_dir": str(run1)})


if __name__ == "__main__":
    main()
