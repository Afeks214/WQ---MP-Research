from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None


_XMIN = -6.0
_XMAX = 6.0
_DX = 0.05
_SIGMA = 0.3
_WINDOW = 60
_ATR_FLOOR = 1e-6
_EPS_DELTA = 1e-9
_VA_THRESHOLD = 0.70


def _require_columns(df: pd.DataFrame) -> dict[str, str]:
    cols = {str(c).strip().lower(): str(c) for c in df.columns}
    req = {
        "timestamp": ("timestamp", "ts", "datetime", "time", "date"),
        "open": ("open", "o"),
        "high": ("high", "h"),
        "low": ("low", "l"),
        "close": ("close", "c"),
        "volume": ("volume", "vol", "v"),
    }
    out: dict[str, str] = {}
    missing: list[str] = []
    for key, opts in req.items():
        hit = None
        for o in opts:
            if o in cols:
                hit = cols[o]
                break
        if hit is None:
            missing.append(key)
        else:
            out[key] = hit
    if missing:
        raise RuntimeError(f"HPC_MISSING_COLUMNS: {missing}")
    return out


def _rolling_mean_1d(arr: np.ndarray, window: int) -> np.ndarray:
    n = arr.shape[0]
    csum = np.concatenate(([0.0], np.cumsum(arr, dtype=np.float64)))
    end = np.arange(1, n + 1, dtype=np.int64)
    start = np.maximum(0, end - int(window))
    sums = csum[end] - csum[start]
    cnt = (end - start).astype(np.float64)
    return sums / cnt


def _rolling_sum_2d(arr: np.ndarray, window: int) -> np.ndarray:
    # arr shape: (T, B)
    t = arr.shape[0]
    csum = np.vstack((np.zeros((1, arr.shape[1]), dtype=np.float64), np.cumsum(arr, axis=0, dtype=np.float64)))
    end = np.arange(1, t + 1, dtype=np.int64)
    start = np.maximum(0, end - int(window))
    return csum[end] - csum[start]


def _value_area_indices_py(vp_grid: np.ndarray, poc_idx: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    t_len, b_len = vp_grid.shape
    ival = np.empty(t_len, dtype=np.int64)
    ivah = np.empty(t_len, dtype=np.int64)

    for t in range(t_len):
        vp = vp_grid[t]
        p = int(poc_idx[t])
        total = float(np.sum(vp))
        if total <= 0.0:
            ival[t] = p
            ivah[t] = p
            continue
        target = float(threshold * total)
        l = p
        r = p
        cum = float(vp[p])

        while cum < target and (l > 0 or r < b_len - 1):
            left = float(vp[l - 1]) if l > 0 else -1.0
            right = float(vp[r + 1]) if r < b_len - 1 else -1.0

            if right > left:
                r += 1
                cum += right
            elif left > right:
                l -= 1
                cum += left
            else:
                if l > 0:
                    l -= 1
                    cum += left
                elif r < b_len - 1:
                    r += 1
                    cum += right
                else:
                    break

        ival[t] = l
        ivah[t] = r

    return ival, ivah


if njit is not None:

    @njit(cache=True)
    def _value_area_indices_nb(vp_grid: np.ndarray, poc_idx: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
        t_len, b_len = vp_grid.shape
        ival = np.empty(t_len, dtype=np.int64)
        ivah = np.empty(t_len, dtype=np.int64)

        for t in range(t_len):
            p = int(poc_idx[t])
            total = 0.0
            for b in range(b_len):
                total += vp_grid[t, b]

            if total <= 0.0:
                ival[t] = p
                ivah[t] = p
                continue

            target = threshold * total
            l = p
            r = p
            cum = vp_grid[t, p]

            while cum < target and (l > 0 or r < b_len - 1):
                left = vp_grid[t, l - 1] if l > 0 else -1.0
                right = vp_grid[t, r + 1] if r < b_len - 1 else -1.0

                if right > left:
                    r += 1
                    cum += right
                elif left > right:
                    l -= 1
                    cum += left
                else:
                    if l > 0:
                        l -= 1
                        cum += left
                    elif r < b_len - 1:
                        r += 1
                        cum += right
                    else:
                        break

            ival[t] = l
            ivah[t] = r

        return ival, ivah


else:
    _value_area_indices_nb = None


def _value_area_indices(vp_grid: np.ndarray, poc_idx: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    if _value_area_indices_nb is not None:
        return _value_area_indices_nb(vp_grid, poc_idx, float(threshold))
    return _value_area_indices_py(vp_grid, poc_idx, float(threshold))


def _compute_one_symbol(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    cols = _require_columns(df)

    work = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df[cols["timestamp"]], utc=True, errors="coerce"),
            "open": pd.to_numeric(df[cols["open"]], errors="coerce"),
            "high": pd.to_numeric(df[cols["high"]], errors="coerce"),
            "low": pd.to_numeric(df[cols["low"]], errors="coerce"),
            "close": pd.to_numeric(df[cols["close"]], errors="coerce"),
            "volume": pd.to_numeric(df[cols["volume"]], errors="coerce"),
        }
    )

    work = work.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"]).sort_values(
        "timestamp", kind="mergesort"
    )
    work = work.drop_duplicates(subset=["timestamp"], keep="last")

    if work.empty:
        raise RuntimeError(f"HPC_EMPTY_SYMBOL_DATA: symbol={symbol}")

    if (
        (work["high"] < work["low"]).any()
        or (work["high"] < work["open"]).any()
        or (work["high"] < work["close"]).any()
        or (work["low"] > work["open"]).any()
        or (work["low"] > work["close"]).any()
    ):
        raise RuntimeError(f"HPC_OHLC_INTEGRITY_FAIL: symbol={symbol}")

    ts = work["timestamp"].reset_index(drop=True)
    close = work["close"].to_numpy(dtype=np.float64)
    high = work["high"].to_numpy(dtype=np.float64)
    low = work["low"].to_numpy(dtype=np.float64)
    volume = work["volume"].to_numpy(dtype=np.float64)

    t_len = close.shape[0]

    returns = np.empty(t_len, dtype=np.float64)
    returns[0] = 0.0
    returns[1:] = np.diff(close)

    tr = high - low
    atr = _rolling_mean_1d(tr, _WINDOW)
    atr = np.maximum(atr, _ATR_FLOOR)

    close_ma = _rolling_mean_1d(close, _WINDOW)
    x = (close - close_ma) / atr

    bins = np.arange(_XMIN, _XMAX, _DX, dtype=np.float64)
    # (T, B) gaussian injection matrix
    z = (x[:, None] - bins[None, :]) / _SIGMA
    kernel = np.exp(-0.5 * (z * z), dtype=np.float64)
    contrib = kernel * volume[:, None]

    vp_grid = _rolling_sum_2d(contrib, _WINDOW)

    direction = np.sign(returns)
    pos = (direction > 0.0).astype(np.float64)
    neg = (direction < 0.0).astype(np.float64)

    vpbuy_grid = vp_grid * pos[:, None]
    vpsell_grid = vp_grid * neg[:, None]
    vpdelta_grid = vpbuy_grid - vpsell_grid

    vp = np.sum(vp_grid, axis=1)
    vpbuy = np.sum(vpbuy_grid, axis=1)
    vpsell = np.sum(vpsell_grid, axis=1)
    vpdelta = np.sum(vpdelta_grid, axis=1)

    poc_idx = np.argmax(vp_grid, axis=1).astype(np.int64)
    ival_idx, ivah_idx = _value_area_indices(vp_grid, poc_idx, _VA_THRESHOLD)

    poc_x = bins[poc_idx]
    val_x = bins[ival_idx]
    vah_x = bins[ivah_idx]

    poc = close_ma + poc_x * atr
    val = close_ma + val_x * atr
    vah = close_ma + vah_x * atr

    d = close - poc
    a_width = vah - val
    delta_eff = vpdelta / (vp + _EPS_DELTA)
    sbreak = ((close > vah) | (close < val)).astype(np.float64)
    sreject = (np.sign(d) != np.sign(returns)).astype(np.float64)

    out = pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": str(symbol),
            "close": close,
            "POC": poc,
            "VAL": val,
            "VAH": vah,
            "VP": vp,
            "VPbuy": vpbuy,
            "VPsell": vpsell,
            "VPdelta": vpdelta,
            "D": d,
            "A": a_width,
            "DeltaEff": delta_eff,
            "Sbreak": sbreak,
            "Sreject": sreject,
        }
    )

    numeric_cols = [
        "close",
        "POC",
        "VAL",
        "VAH",
        "VP",
        "VPbuy",
        "VPsell",
        "VPdelta",
        "D",
        "A",
        "DeltaEff",
        "Sbreak",
        "Sreject",
    ]
    bad = ~np.isfinite(out[numeric_cols].to_numpy(dtype=np.float64))
    if np.any(bad):
        raise RuntimeError(f"HPC_NONFINITE_OUTPUT: symbol={symbol}")

    return out


def compute_market_profile_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute deterministic Market Profile features from real OHLCV input.

    Input required columns:
      timestamp, open, high, low, close, volume
    Optional:
      symbol (for multi-symbol batches)
    """
    if df.empty:
        raise RuntimeError("HPC_EMPTY_INPUT")

    symbol_col = None
    for cand in ("symbol", "Symbol", "ticker", "Ticker"):
        if cand in df.columns:
            symbol_col = cand
            break

    results: list[pd.DataFrame] = []
    if symbol_col is None:
        results.append(_compute_one_symbol(df, symbol="UNKNOWN"))
    else:
        # Stable deterministic symbol order
        symbols = sorted(str(s) for s in pd.unique(df[symbol_col].astype(str)))
        for sym in symbols:
            part = df.loc[df[symbol_col].astype(str) == sym]
            results.append(_compute_one_symbol(part, symbol=sym))

    out = pd.concat(results, ignore_index=True)
    out = out.sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)

    required_out = [
        "timestamp",
        "symbol",
        "close",
        "POC",
        "VAL",
        "VAH",
        "VP",
        "VPbuy",
        "VPsell",
        "VPdelta",
        "D",
        "A",
        "DeltaEff",
        "Sbreak",
        "Sreject",
    ]
    missing = [c for c in required_out if c not in out.columns]
    if missing:
        raise RuntimeError(f"HPC_OUTPUT_SCHEMA_MISMATCH: missing={missing}")

    return out[required_out]
