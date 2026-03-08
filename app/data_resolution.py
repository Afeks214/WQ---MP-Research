from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.config_models import DataConfigModel, RunConfigModel

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore[assignment]


def require_pandas() -> Any:
    if pd is None:
        raise RuntimeError("pandas is required. Install with: pip install pandas")
    return pd


def resolve_data_paths(cfg: RunConfigModel, project_root: Path) -> list[str]:
    syms = [s.strip().upper() for s in cfg.symbols]
    d = cfg.data

    root = Path(d.root)
    if not root.is_absolute():
        root = (project_root / root).resolve()

    out: list[str] = []
    missing: list[str] = []

    for s in syms:
        mapped = d.path_by_symbol.get(s, d.path_by_symbol.get(s.lower(), d.path_by_symbol.get(s.upper())))
        if mapped is None:
            p = root / f"{s}.{d.format}"
        else:
            p0 = Path(mapped)
            p = p0 if p0.is_absolute() else (root / p0)
        p = p.resolve()
        if not p.exists():
            missing.append(f"{s}: {p}")
        else:
            out.append(str(p))

    if missing:
        raise RuntimeError("Missing data files:\n" + "\n".join(missing))
    return out


def find_col(df: Any, candidates: tuple[str, ...], name: str) -> str:
    cols = {str(c).strip().lower(): str(c) for c in df.columns}
    for c in candidates:
        if c in cols:
            return cols[c]
    raise RuntimeError(f"Missing required column '{name}'")


def in_memory_date_filter_loader(data_cfg: DataConfigModel) -> Callable[[str, str], Any]:
    pdx = require_pandas()

    start_utc = pdx.to_datetime(data_cfg.start, utc=True) if data_cfg.start is not None else None
    end_utc = pdx.to_datetime(data_cfg.end, utc=True) if data_cfg.end is not None else None

    def _load(path: str, tz_name: str) -> Any:
        del tz_name
        p = Path(path)
        if not p.exists():
            raise RuntimeError(f"Data path does not exist: {path}")

        suffix = p.suffix.lower()
        if suffix == ".parquet":
            df = pdx.read_parquet(p)
        else:
            df = pdx.read_csv(p)

        if data_cfg.timestamp_column is not None:
            ts_col = find_col(df, (data_cfg.timestamp_column.strip().lower(),), "timestamp")
        else:
            ts_col = find_col(df, ("timestamp", "ts", "datetime", "date", "time"), "timestamp")

        o_col = find_col(df, ("open", "o"), "open")
        h_col = find_col(df, ("high", "h"), "high")
        l_col = find_col(df, ("low", "l"), "low")
        c_col = find_col(df, ("close", "c"), "close")
        v_col = find_col(df, ("volume", "vol", "v"), "volume")

        ts = pdx.to_datetime(df[ts_col], utc=True, errors="coerce")
        keep = ts.notna().to_numpy(dtype=bool)

        if start_utc is not None:
            keep &= (ts >= start_utc).to_numpy(dtype=bool)
        if end_utc is not None:
            keep &= (ts <= end_utc).to_numpy(dtype=bool)

        if not np.any(keep):
            raise RuntimeError(f"No rows after timestamp/date filtering for {path}")

        out = pdx.DataFrame(
            {
                "timestamp": ts[keep].dt.floor("min"),
                "open": pdx.to_numeric(df.loc[keep, o_col], errors="coerce"),
                "high": pdx.to_numeric(df.loc[keep, h_col], errors="coerce"),
                "low": pdx.to_numeric(df.loc[keep, l_col], errors="coerce"),
                "close": pdx.to_numeric(df.loc[keep, c_col], errors="coerce"),
                "volume": pdx.to_numeric(df.loc[keep, v_col], errors="coerce"),
            }
        )

        out = out.dropna(subset=["timestamp"]).sort_values("timestamp", kind="mergesort")
        out = out.drop_duplicates(subset=["timestamp"], keep="last")
        out = out.set_index("timestamp")
        return out

    return _load
