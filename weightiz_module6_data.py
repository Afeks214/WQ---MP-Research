"""
Weightiz Module 6 - Data Layer
==============================

Pure data/analytics utilities for the Deep-Quant Visualizer.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore[assignment]


REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "equity_curves.parquet",
    "trade_log.parquet",
    "daily_returns.parquet",
    "verdict.json",
    "stats_raw.json",
    "run_manifest.json",
)

REQUIRED_EQUITY_COLS: tuple[str, ...] = (
    "ts_ns",
    "session_id",
    "candidate_id",
    "split_id",
    "scenario_id",
    "equity",
    "drawdown",
    "margin_used",
    "buying_power",
    "daily_loss",
)

REQUIRED_TRADE_COLS: tuple[str, ...] = (
    "ts_ns",
    "candidate_id",
    "split_id",
    "scenario_id",
    "symbol",
    "filled_qty",
    "exec_price",
    "trade_cost",
    "order_side",
    "order_flags",
)


@dataclass(frozen=True)
class Module6Config:
    artifacts_root: str = "./artifacts/module5_harness"
    default_run_id: str | None = None
    rolling_window_days: int = 63
    calmar_window_days: int = 252
    risk_free_daily: float = 0.0
    intraday_leverage_ref: float = 10.0
    overnight_leverage_ref: float = 2.0
    use_manifest_leverage_when_available: bool = True
    max_profile_blocks_render: int = 26
    max_days_in_memory: int = 1500
    enable_micro_diagnostics_required_views: bool = True
    fail_on_schema_mismatch: bool = True
    timezone: str = "America/New_York"


def _require_pandas() -> Any:
    if pd is None:
        raise RuntimeError("pandas is required for Module 6 data layer")
    return pd


def _require_cols(df: Any, cols: tuple[str, ...], name: str, fail: bool) -> list[str]:
    missing = [c for c in cols if c not in df.columns]
    if missing and fail:
        raise RuntimeError(f"{name} missing required columns: {missing}")
    return missing


def to_et_datetime(ts_ns: np.ndarray | Any, timezone: str = "America/New_York") -> Any:
    pdx = _require_pandas()
    dt = pdx.to_datetime(np.asarray(ts_ns, dtype=np.int64), unit="ns", utc=True)
    return dt.tz_convert(timezone)


def list_run_ids(artifacts_root: str) -> list[str]:
    root = Path(artifacts_root).resolve()
    if not root.exists():
        return []
    run_ids = [p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    return sorted(run_ids)


def resolve_run_dir(cfg: Module6Config, run_id: str | None = None) -> Path:
    root = Path(cfg.artifacts_root).resolve()
    rid = run_id or cfg.default_run_id
    if rid is None:
        runs = list_run_ids(str(root))
        if not runs:
            raise RuntimeError(f"No run directories found under {root}")
        rid = runs[-1]
    run_dir = root / rid
    if not run_dir.exists() or not run_dir.is_dir():
        raise RuntimeError(f"Run directory not found: {run_dir}")
    return run_dir


def load_run_bundle(cfg: Module6Config, run_id: str | None = None) -> dict[str, Any]:
    pdx = _require_pandas()
    run_dir = resolve_run_dir(cfg, run_id)

    missing = [f for f in REQUIRED_ARTIFACTS if not (run_dir / f).exists()]
    if missing:
        raise RuntimeError(f"Run {run_dir.name} missing required artifacts: {missing}")

    eq = pdx.read_parquet(run_dir / "equity_curves.parquet")
    tr = pdx.read_parquet(run_dir / "trade_log.parquet")
    dr = pdx.read_parquet(run_dir / "daily_returns.parquet")

    _require_cols(eq, REQUIRED_EQUITY_COLS, "equity_curves", cfg.fail_on_schema_mismatch)
    _require_cols(tr, REQUIRED_TRADE_COLS, "trade_log", cfg.fail_on_schema_mismatch)

    with (run_dir / "verdict.json").open("r", encoding="utf-8") as f:
        verdict = json.load(f)
    with (run_dir / "stats_raw.json").open("r", encoding="utf-8") as f:
        stats_raw = json.load(f)
    with (run_dir / "run_manifest.json").open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    micro_path = run_dir / "micro_diagnostics.parquet"
    profile_path = run_dir / "micro_profile_blocks.parquet"
    funnel_path = run_dir / "funnel_1545.parquet"

    micro = pdx.read_parquet(micro_path) if micro_path.exists() else None
    profile = pdx.read_parquet(profile_path) if profile_path.exists() else None
    funnel = pdx.read_parquet(funnel_path) if funnel_path.exists() else None

    # Deterministic ordering.
    eq = eq.sort_values(["candidate_id", "split_id", "scenario_id", "ts_ns"], kind="mergesort")
    tr = tr.sort_values(["candidate_id", "split_id", "scenario_id", "ts_ns"], kind="mergesort")
    dr = dr.sort_values(["session_id"], kind="mergesort")

    return {
        "run_dir": str(run_dir),
        "run_id": run_dir.name,
        "equity": eq.reset_index(drop=True),
        "trade": tr.reset_index(drop=True),
        "daily": dr.reset_index(drop=True),
        "verdict": verdict,
        "stats_raw": stats_raw,
        "manifest": manifest,
        "micro": micro.reset_index(drop=True) if micro is not None else None,
        "profile": profile.reset_index(drop=True) if profile is not None else None,
        "funnel": funnel.reset_index(drop=True) if funnel is not None else None,
    }


def candidate_filter(df: Any, candidate_id: str, split_id: str, scenario_id: str) -> Any:
    out = df[(df["candidate_id"] == candidate_id) & (df["split_id"] == split_id) & (df["scenario_id"] == scenario_id)]
    return out.sort_values(["ts_ns"], kind="mergesort").reset_index(drop=True)


def extract_daily_candidate_returns(daily_df: Any, task_id: str, max_days: int) -> np.ndarray:
    if task_id not in daily_df.columns:
        raise RuntimeError(f"Task column not found in daily_returns: {task_id}")
    x = daily_df[task_id].to_numpy(dtype=np.float64)
    if max_days > 0 and x.shape[0] > max_days:
        x = x[-int(max_days) :]
    return x


def rolling_sharpe(returns_d: np.ndarray, window: int, risk_free_daily: float = 0.0, eps: float = 1e-12) -> np.ndarray:
    r = np.asarray(returns_d, dtype=np.float64)
    W = int(max(2, window))
    out = np.full(r.shape, np.nan, dtype=np.float64)
    if r.size < W:
        return out
    ex = r - float(risk_free_daily)
    for i in range(W - 1, r.size):
        seg = ex[i - W + 1 : i + 1]
        mu = float(np.mean(seg))
        sd = float(np.std(seg, ddof=1))
        out[i] = mu / (sd + float(eps)) * float(np.sqrt(252.0))
    return out


def rolling_calmar(returns_d: np.ndarray, window: int, eps: float = 1e-12) -> np.ndarray:
    r = np.asarray(returns_d, dtype=np.float64)
    W = int(max(2, window))
    out = np.full(r.shape, np.nan, dtype=np.float64)
    if r.size < W:
        return out
    for i in range(W - 1, r.size):
        seg = r[i - W + 1 : i + 1]
        eq = np.cumprod(1.0 + seg)
        cagr = float(eq[-1] ** (252.0 / max(float(W), 1.0)) - 1.0)
        peak = np.maximum.accumulate(eq)
        dd = eq / np.maximum(peak, eps) - 1.0
        max_dd_abs = float(np.max(np.abs(dd)))
        out[i] = cagr / (max_dd_abs + float(eps))
    return out


def leverage_utilization(
    equity: np.ndarray,
    margin_used: np.ndarray,
    buying_power: np.ndarray,
    leverage_ref: float,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    eq = np.asarray(equity, dtype=np.float64)
    mu = np.asarray(margin_used, dtype=np.float64)
    bp = np.asarray(buying_power, dtype=np.float64)
    L = float(leverage_ref)
    denom = np.maximum(eq * L, float(eps))
    u_margin = mu / denom
    u_bp = 1.0 - bp / denom
    return u_margin, u_bp


def x_to_price(close_px: np.ndarray, x_coord: np.ndarray, atr_eff: np.ndarray) -> np.ndarray:
    c = np.asarray(close_px, dtype=np.float64)
    x = np.asarray(x_coord, dtype=np.float64)
    a = np.asarray(atr_eff, dtype=np.float64)
    return c + x * a


def compute_episode_mfe_mae(micro_df: Any, trade_df: Any, eps: float = 1e-12) -> Any:
    pdx = _require_pandas()
    if micro_df is None or len(micro_df) == 0:
        return pdx.DataFrame(
            columns=["symbol", "entry_ts_ns", "exit_ts_ns", "side", "mfe", "mae", "win", "notional"]
        )

    req_cols = ["symbol", "ts_ns", "close", "filled_qty"]
    for c in req_cols:
        if c not in micro_df.columns:
            raise RuntimeError(f"micro_diagnostics missing required column for MFE/MAE: {c}")

    out_rows: list[dict[str, Any]] = []

    for sym, g in micro_df.groupby("symbol", sort=True):
        g2 = g.sort_values("ts_ns", kind="mergesort").reset_index(drop=True)
        qty_fill = g2["filled_qty"].to_numpy(dtype=np.float64)
        px = g2["close"].to_numpy(dtype=np.float64)
        ts = g2["ts_ns"].to_numpy(dtype=np.int64)

        pos = np.cumsum(qty_fill)
        prev = np.r_[0.0, pos[:-1]]
        starts = np.flatnonzero((np.abs(prev) <= eps) & (np.abs(pos) > eps))
        ends = np.flatnonzero((np.abs(prev) > eps) & (np.abs(pos) <= eps))

        if starts.size == 0:
            continue

        end_ptr = 0
        for s in starts.tolist():
            while end_ptr < ends.size and int(ends[end_ptr]) < int(s):
                end_ptr += 1
            if end_ptr < ends.size:
                e = int(ends[end_ptr])
                end_ptr += 1
            else:
                e = int(g2.shape[0] - 1)
            if e <= s:
                continue

            side = 1.0 if pos[s] > 0 else -1.0
            p0 = float(px[s])
            pseg = px[s : e + 1]
            if side > 0:
                ret = pseg / max(p0, eps) - 1.0
                mfe = float(np.max(ret))
                mae = float(np.min(ret))
                pnl = float((px[e] - p0) * abs(pos[s]))
            else:
                ret = p0 / np.maximum(pseg, eps) - 1.0
                mfe = float(np.max(ret))
                mae = float(np.min(ret))
                pnl = float((p0 - px[e]) * abs(pos[s]))

            out_rows.append(
                {
                    "symbol": str(sym),
                    "entry_ts_ns": int(ts[s]),
                    "exit_ts_ns": int(ts[e]),
                    "side": "LONG" if side > 0 else "SHORT",
                    "mfe": mfe,
                    "mae": mae,
                    "win": int(1 if pnl > 0 else 0),
                    "notional": float(abs(pos[s]) * p0),
                }
            )

    if not out_rows:
        return pdx.DataFrame(
            columns=["symbol", "entry_ts_ns", "exit_ts_ns", "side", "mfe", "mae", "win", "notional"]
        )
    return pdx.DataFrame(out_rows)


def build_funnel_table(
    funnel_df: Any | None,
    micro_df: Any | None,
    selected_session: int | None = None,
    selected_ts_ns: int | None = None,
) -> Any:
    pdx = _require_pandas()
    if funnel_df is not None and len(funnel_df) > 0:
        dff = funnel_df.copy()
        if selected_session is not None:
            dff = dff[dff["session_id"].astype(np.int64) == int(selected_session)]
        if selected_ts_ns is not None:
            dff = dff[dff["ts_ns"].astype(np.int64) == int(selected_ts_ns)]
        return dff.sort_values(["ocs", "symbol"], ascending=[False, True], kind="mergesort").reset_index(drop=True)

    if micro_df is None or len(micro_df) == 0:
        return pdx.DataFrame(columns=["symbol", "dclip", "z_delta", "regime_primary", "structural_weight", "ocs", "is_winner", "cash_fallback"])

    req = ["ts_ns", "session_id", "symbol", "dclip", "z_delta", "regime_primary", "rvol", "overnight_winner_flag"]
    for c in req:
        if c not in micro_df.columns:
            raise RuntimeError(f"Cannot build funnel table from micro diagnostics. Missing column: {c}")

    dff = micro_df.copy()
    if selected_session is not None:
        dff = dff[dff["session_id"].astype(np.int64) == int(selected_session)]

    if selected_ts_ns is None and len(dff) > 0:
        selected_ts_ns = int(np.max(dff["ts_ns"].to_numpy(dtype=np.int64)))
    if selected_ts_ns is None:
        return pdx.DataFrame(columns=["symbol", "dclip", "z_delta", "regime_primary", "structural_weight", "ocs", "is_winner", "cash_fallback"])

    dff = dff[dff["ts_ns"].astype(np.int64) == int(selected_ts_ns)]
    if len(dff) == 0:
        return dff

    reg = dff["regime_primary"].to_numpy(dtype=np.int8)
    sw = np.zeros(reg.shape, dtype=np.float64)
    sw[(reg == 3) | (reg == 4)] = 1.5
    sw[reg == 2] = 1.2
    ocs = sw * np.abs(dff["dclip"].to_numpy(dtype=np.float64)) * np.abs(dff["z_delta"].to_numpy(dtype=np.float64)) * np.maximum(
        dff["rvol"].to_numpy(dtype=np.float64),
        0.0,
    )

    out = pdx.DataFrame(
        {
            "ts_ns": dff["ts_ns"].to_numpy(dtype=np.int64),
            "session_id": dff["session_id"].to_numpy(dtype=np.int64),
            "symbol": dff["symbol"].astype(str).to_numpy(),
            "dclip": dff["dclip"].to_numpy(dtype=np.float64),
            "z_delta": dff["z_delta"].to_numpy(dtype=np.float64),
            "regime_primary": reg.astype(np.int8),
            "structural_weight": sw,
            "ocs": ocs,
            "is_winner": dff["overnight_winner_flag"].to_numpy(dtype=np.int8),
        }
    )

    winner_exists = bool(np.any(out["is_winner"].to_numpy(dtype=np.int8) == 1))
    out["cash_fallback"] = np.int8(0 if winner_exists else 1)

    return out.sort_values(["ocs", "symbol"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
