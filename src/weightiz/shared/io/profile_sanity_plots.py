from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_CANONICAL = (
    "timestamp",
    "close",
    "POC",
    "VAL",
    "VAH",
    "DeltaEff",
    "Sbreak",
)


def _resolve_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lowered = {str(c).strip().lower(): str(c) for c in df.columns}
    for name in candidates:
        hit = lowered.get(name.lower())
        if hit is not None:
            return hit
    return None


def validate_required_columns(df: pd.DataFrame) -> dict[str, str]:
    """Resolve required columns with case/alias matching; fail closed if missing."""
    alias_map: dict[str, tuple[str, ...]] = {
        "timestamp": ("timestamp", "ts", "datetime", "time", "date"),
        "close": ("close", "c"),
        "POC": ("POC", "poc"),
        "VAL": ("VAL", "val"),
        "VAH": ("VAH", "vah"),
        "DeltaEff": ("DeltaEff", "deltaeff", "delta_eff", "DELTA_EFF"),
        "Sbreak": ("Sbreak", "sbreak", "s_break", "S_BREAK"),
    }

    resolved: dict[str, str] = {}
    missing: list[str] = []
    for canonical, candidates in alias_map.items():
        col = _resolve_column(df, candidates)
        if col is None:
            missing.append(canonical)
        else:
            resolved[canonical] = col

    if missing:
        raise RuntimeError(f"DIAGNOSTICS_MISSING_COLUMNS: {missing}")
    return resolved


def validate_numeric_finite(df: pd.DataFrame) -> None:
    checks = ("close", "POC", "VAH", "VAL")
    bad_fields: list[str] = []
    for col in checks:
        arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(arr).all():
            bad_fields.append(col)
    if bad_fields:
        raise RuntimeError(f"DIAGNOSTICS_NONFINITE_VALUES: {bad_fields}")


def validate_price_units(df: pd.DataFrame) -> tuple[float, float]:
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=np.float64)
    poc = pd.to_numeric(df["POC"], errors="coerce").to_numpy(dtype=np.float64)
    vah = pd.to_numeric(df["VAH"], errors="coerce").to_numpy(dtype=np.float64)
    val = pd.to_numeric(df["VAL"], errors="coerce").to_numpy(dtype=np.float64)

    median_abs_diff = float(abs(np.median(poc) - np.median(close)))
    true_range_proxy = np.abs(vah - val)
    median_true_range = float(np.median(true_range_proxy))

    if not np.isfinite(median_abs_diff) or not np.isfinite(median_true_range):
        raise RuntimeError("DIAGNOSTICS_NONFINITE_VALUES: unit validation produced non-finite medians")

    if median_abs_diff > 5.0 * median_true_range:
        raise RuntimeError("DIAGNOSTICS_UNIT_MISMATCH: POC/VAL/VAH likely not in price units")

    return median_abs_diff, median_true_range


def maybe_downsample(df: pd.DataFrame, max_rows: int = 200_000, step: int = 5) -> tuple[pd.DataFrame, int]:
    if len(df) > int(max_rows):
        out = df.iloc[:: int(step)].copy()
        if out.empty:
            raise RuntimeError("DIAGNOSTICS_EMPTY_AFTER_FILTER: deterministic downsample produced empty frame")
        return out, int(step)
    return df, 1


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    if x.size != y.size:
        raise RuntimeError("DIAGNOSTICS_EMPTY_AFTER_FILTER: correlation arrays size mismatch")
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx == 0.0 or sy == 0.0:
        return 0.0
    c = np.corrcoef(x, y)[0, 1]
    return float(c if np.isfinite(c) else 0.0)


def compute_fast_diagnostics(df: pd.DataFrame) -> dict[str, Any]:
    median_abs_diff, median_true_range = validate_price_units(df)
    close = df["close"].to_numpy(dtype=np.float64)
    poc = df["POC"].to_numpy(dtype=np.float64)
    delta_eff = df["DeltaEff"].to_numpy(dtype=np.float64)
    sbreak = df["Sbreak"].to_numpy(dtype=np.float64)

    return {
        "median_abs_diff_poc_close": float(median_abs_diff),
        "median_true_range_proxy": float(median_true_range),
        "corr_deltaeff_sbreak": _safe_corr(delta_eff, sbreak),
        "corr_close_poc": _safe_corr(close, poc),
    }


def generate_profile_sanity_plots(features_path: Path, output_dir: Path) -> dict[str, Any]:
    """
    Generate deterministic profile diagnostics charts.

    Soft skip rule: matplotlib import failure returns non-fatal status.
    All other validation errors are fail-closed RuntimeError.
    """
    df = pd.read_parquet(features_path)
    resolved = validate_required_columns(df)

    plot_df = pd.DataFrame({
        "timestamp": pd.to_datetime(df[resolved["timestamp"]], errors="coerce", utc=True),
        "close": pd.to_numeric(df[resolved["close"]], errors="coerce"),
        "POC": pd.to_numeric(df[resolved["POC"]], errors="coerce"),
        "VAL": pd.to_numeric(df[resolved["VAL"]], errors="coerce"),
        "VAH": pd.to_numeric(df[resolved["VAH"]], errors="coerce"),
        "DeltaEff": pd.to_numeric(df[resolved["DeltaEff"]], errors="coerce"),
        "Sbreak": pd.to_numeric(df[resolved["Sbreak"]], errors="coerce"),
    }).sort_values("timestamp", kind="mergesort")

    if plot_df.empty:
        raise RuntimeError("DIAGNOSTICS_EMPTY_AFTER_FILTER: no rows available for plotting")

    if plot_df["timestamp"].isna().all():
        raise RuntimeError("DIAGNOSTICS_EMPTY_AFTER_FILTER: timestamp parse produced all-NaT")

    validate_numeric_finite(plot_df)
    metrics = compute_fast_diagnostics(plot_df)

    plot_df, downsample_factor = maybe_downsample(plot_df, max_rows=200_000, step=5)
    rows_plotted = int(len(plot_df))
    if rows_plotted == 0:
        raise RuntimeError("DIAGNOSTICS_EMPTY_AFTER_FILTER: empty frame after downsample")

    summary: dict[str, Any] = {
        "rows_input": int(len(df)),
        "rows_plotted": rows_plotted,
        "downsample_factor": int(downsample_factor),
        "median_abs_diff_poc_close": float(metrics["median_abs_diff_poc_close"]),
        "median_true_range_proxy": float(metrics["median_true_range_proxy"]),
        "corr_deltaeff_sbreak": float(metrics["corr_deltaeff_sbreak"]),
        "corr_close_poc": float(metrics["corr_close_poc"]),
        "plots_generated": False,
        "png_determinism_note": "PNG files excluded from deterministic hashing",
    }

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        summary["reason"] = "matplotlib_missing"
        return summary

    output_dir.mkdir(parents=True, exist_ok=True)

    t = plot_df["timestamp"]
    close = plot_df["close"].to_numpy(dtype=np.float64)
    poc = plot_df["POC"].to_numpy(dtype=np.float64)
    vah = plot_df["VAH"].to_numpy(dtype=np.float64)
    val = plot_df["VAL"].to_numpy(dtype=np.float64)
    va_width = vah - val
    delta_eff = plot_df["DeltaEff"].to_numpy(dtype=np.float64)
    sbreak = plot_df["Sbreak"].to_numpy(dtype=np.float64)

    fig1, ax1 = plt.subplots(figsize=(12, 4), dpi=120)
    ax1.plot(t, close, label="close", linewidth=0.8)
    ax1.plot(t, poc, label="POC", linewidth=0.8)
    ax1.set_title("Price vs POC")
    ax1.legend(loc="best")
    fig1.tight_layout()
    fig1.savefig(output_dir / "poc_price.png")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(12, 4), dpi=120)
    ax2.plot(t, va_width, label="VA width (VAH-VAL)", linewidth=0.8)
    ax2.set_title("Value Area Width")
    ax2.legend(loc="best")
    fig2.tight_layout()
    fig2.savefig(output_dir / "value_area_width.png")
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(12, 4), dpi=120)
    ax3.plot(t, delta_eff, label="DeltaEff", linewidth=0.8)
    ax3.plot(t, sbreak, label="Sbreak", linewidth=0.8)
    ax3.set_title("Delta Confirmation")
    ax3.legend(loc="best")
    fig3.tight_layout()
    fig3.savefig(output_dir / "delta_breakout.png")
    plt.close(fig3)

    summary["plots_generated"] = True
    return summary
