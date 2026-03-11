from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None

EPS_PDF = 1e-12
EPS_VOL = 1e-12


@dataclass(frozen=True)
class ParityConfig:
    va_threshold: float = 0.70
    x_min: float = -6.0
    x_max: float = 6.0
    dx: float = 0.05


CFG = ParityConfig()
X_GRID = np.arange(CFG.x_min, CFG.x_max, CFG.dx, dtype=np.float64)


def clip(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def sigmoid(u: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-u))


def mad_scalar(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    m = np.median(x)
    return float(np.median(np.abs(x - m)))


def _require_ohlcv_columns(df: pd.DataFrame) -> dict[str, str]:
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
        found = None
        for o in opts:
            if o in cols:
                found = cols[o]
                break
        if found is None:
            missing.append(key)
        else:
            out[key] = found
    if missing:
        raise RuntimeError(f"PARITY_MISSING_COLUMNS: {missing}")
    return out


def _prepare_df(raw: pd.DataFrame) -> pd.DataFrame:
    cols = _require_ohlcv_columns(raw)
    ts = pd.to_datetime(raw[cols["timestamp"]], errors="coerce", utc=True)
    out = pd.DataFrame(
        {
            "timestamp": ts,
            "open": pd.to_numeric(raw[cols["open"]], errors="coerce"),
            "high": pd.to_numeric(raw[cols["high"]], errors="coerce"),
            "low": pd.to_numeric(raw[cols["low"]], errors="coerce"),
            "close": pd.to_numeric(raw[cols["close"]], errors="coerce"),
            "volume": pd.to_numeric(raw[cols["volume"]], errors="coerce"),
        }
    )
    out = out.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"]).copy()
    out = out.sort_values("timestamp", kind="mergesort")
    out = out.drop_duplicates(subset=["timestamp"], keep="last")
    if out.empty:
        raise RuntimeError("PARITY_EMPTY_INPUT")

    bad = (
        (out["high"] < out["low"])
        | (out["high"] < out["open"])
        | (out["high"] < out["close"])
        | (out["low"] > out["open"])
        | (out["low"] > out["close"])
    )
    if bool(bad.any()):
        idx = int(np.flatnonzero(bad.to_numpy())[0])
        raise RuntimeError(f"PARITY_OHLC_INTEGRITY_FAIL at row={idx}")

    ts_utc = pd.to_datetime(out["timestamp"], utc=True)
    ts_et = ts_utc.dt.tz_convert("US/Eastern")
    out["ts"] = ts_utc
    out["ts_et"] = ts_et
    out["session_date"] = ts_et.dt.date
    out["minute_of_day"] = ts_et.dt.hour * 60 + ts_et.dt.minute
    out["tod"] = out["minute_of_day"] - (9 * 60 + 30)

    dt_min = out["ts"].diff().dt.total_seconds() / 60.0
    out["gap_min"] = dt_min.fillna(0.0)
    out["reset_flag_gap_gt_5"] = (out["gap_min"] > 5.0).astype(np.int8)

    return out


def compute_atr_floor(df: pd.DataFrame, atr_period: int = 14) -> pd.Series:
    high = df["high"].astype(np.float64).to_numpy()
    low = df["low"].astype(np.float64).to_numpy()
    close = df["close"].astype(np.float64).to_numpy()

    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))

    atr = pd.Series(tr, index=df.index).ewm(span=atr_period, adjust=False).mean().astype(np.float64)
    tick = 0.01
    atr_floor = np.maximum(atr.to_numpy(), np.maximum(4.0 * tick, 0.0002 * close))
    return pd.Series(atr_floor, index=df.index, dtype=np.float64)


def compute_rvol_diurnal_causal(df: pd.DataFrame, vol_col: str = "volume") -> pd.Series:
    if "tod" not in df.columns:
        raise RuntimeError("compute_rvol_diurnal_causal requires 'tod'")

    vol = df[vol_col].astype(np.float64)
    baseline = (
        df.groupby("tod", sort=False)[vol_col]
        .transform(lambda s: s.expanding().median().shift(1))
        .astype(np.float64)
    )
    global_base = vol.expanding().median().shift(1)
    baseline = baseline.fillna(global_base).bfill()
    return vol / (baseline + EPS_VOL)


def make_va_offsets(n_bins: int) -> np.ndarray:
    offs = [0]
    for k in range(1, n_bins):
        offs.append(+k)
        offs.append(-k)
        if len(offs) >= n_bins:
            break
    return np.asarray(offs[:n_bins], dtype=np.int64)


def compute_value_area_idx_offsetscan(
    vp: np.ndarray, poc_idx: np.ndarray, va_threshold: float = 0.70
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vp = np.asarray(vp, dtype=np.float64)
    poc_idx = np.asarray(poc_idx, dtype=np.int64)
    t_len, n_bins = vp.shape

    offsets = make_va_offsets(n_bins)
    ipoc = poc_idx.copy()

    total = vp.sum(axis=1)
    target = total * float(va_threshold)

    idx_mat = ipoc[:, None] + offsets[None, :]
    valid = (idx_mat >= 0) & (idx_mat < n_bins)

    vp_scan = np.zeros((t_len, n_bins), dtype=np.float64)
    rr, cc = np.where(valid)
    vp_scan[rr, cc] = vp[rr, idx_mat[rr, cc]]

    csum = np.cumsum(vp_scan, axis=1)
    k = np.argmax(csum >= target[:, None], axis=1).astype(np.int64)

    empty = total <= 0.0
    k = np.where(empty, 0, k)

    j = np.arange(n_bins, dtype=np.int64)[None, :]
    included = j <= k[:, None]
    idx_included = np.where(included & valid, idx_mat, -1)

    ivah = idx_included.max(axis=1)
    big = np.int64(n_bins + 10)
    ival_tmp = np.where(idx_included >= 0, idx_included, big)
    ival = ival_tmp.min(axis=1)

    ivah = np.where(ivah < 0, ipoc, ivah)
    ival = np.where((ival >= big) | (ival < 0), ipoc, ival)
    return ipoc, ivah, ival


if njit is not None:

    @njit(cache=True)
    def _build_vp_numba(
        close: np.ndarray,
        p: np.ndarray,
        v: np.ndarray,
        v_delta: np.ndarray,
        atrf: np.ndarray,
        w: int,
        x_min: float,
        dx: float,
        n_bins: int,
        eps_pdf: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        m = len(close)
        start_idx = w - 1
        t_len = m - start_idx
        vp = np.zeros((t_len, n_bins), dtype=np.float64)
        vpd = np.zeros((t_len, n_bins), dtype=np.float64)

        for t_out in range(t_len):
            t_end = start_idx + t_out
            w0 = t_end - (w - 1)
            w1 = t_end + 1

            atr_end = atrf[t_end]
            if (not np.isfinite(atr_end)) or (atr_end <= 0.0):
                atr_end = max(1e-4, 0.0002 * close[t_end])

            ref_close = close[t_end]
            denom = atr_end + eps_pdf

            for j in range(w0, w1):
                xw = (p[j] - ref_close) / denom
                idx = int(np.rint((xw - x_min) / dx))
                if idx < 0:
                    idx = 0
                elif idx >= n_bins:
                    idx = n_bins - 1
                vp[t_out, idx] += v[j]
                vpd[t_out, idx] += v_delta[j]

        return vp, vpd


def build_daily_vp_tensors(df_day: pd.DataFrame, x_grid: np.ndarray, w: int = 60) -> dict[str, Any]:
    df = df_day.copy()
    close = df["close"].astype(np.float64).to_numpy()
    p = close
    v = df["volume"].astype(np.float64).to_numpy()

    dP = np.zeros_like(close, dtype=np.float64)
    dP[1:] = close[1:] - close[:-1]
    sgn = np.sign(dP)
    v_delta = v * sgn

    if "ATR_floor" in df.columns:
        atrf = df["ATR_floor"].astype(np.float64).to_numpy()
    else:
        atrf = compute_atr_floor(df, atr_period=14).to_numpy(dtype=np.float64)

    n_bins = int(len(x_grid))
    if len(df) < w:
        return {
            "VP": np.zeros((0, n_bins), dtype=np.float64),
            "VP_delta": np.zeros((0, n_bins), dtype=np.float64),
            "POC_idx": np.zeros((0,), dtype=np.int64),
            "df_mp": df.iloc[0:0].copy(),
            "ts": np.array([], dtype="datetime64[ns]"),
        }

    start_idx = w - 1
    df_mp = df.iloc[start_idx:].copy().reset_index(drop=True)
    ts_out = pd.to_datetime(df_mp["ts"], utc=True).to_numpy()

    if njit is not None:
        VP, VPd = _build_vp_numba(
            close,
            p,
            v,
            v_delta,
            atrf,
            int(w),
            float(x_grid[0]),
            float(x_grid[1] - x_grid[0]),
            int(n_bins),
            float(EPS_PDF),
        )
    else:
        VP = np.zeros((len(df) - start_idx, n_bins), dtype=np.float64)
        VPd = np.zeros((len(df) - start_idx, n_bins), dtype=np.float64)
        DX = float(x_grid[1] - x_grid[0])
        XMIN = float(x_grid[0])
        for t_out in range(VP.shape[0]):
            t_end = start_idx + t_out
            w0 = t_end - (w - 1)
            w1 = t_end + 1
            atr_end = float(atrf[t_end])
            if (not np.isfinite(atr_end)) or atr_end <= 0:
                atr_end = max(1e-4, 0.0002 * close[t_end])
            xw = (p[w0:w1] - close[t_end]) / (atr_end + EPS_PDF)
            idx = np.rint((xw - XMIN) / DX).astype(np.int64)
            idx = np.clip(idx, 0, n_bins - 1)
            np.add.at(VP[t_out], idx, v[w0:w1])
            np.add.at(VPd[t_out], idx, v_delta[w0:w1])

    POC_idx = np.argmax(VP, axis=1).astype(np.int64)
    return {
        "VP": VP,
        "VP_delta": VPd,
        "POC_idx": POC_idx,
        "df_mp": df_mp,
        "ts": ts_out,
    }


def compute_r6_spec_from_tensors(
    df_mp: pd.DataFrame,
    VP: np.ndarray,
    VP_delta: np.ndarray,
    poc_idx: np.ndarray,
    x_grid: np.ndarray,
    w: int = 60,
) -> pd.DataFrame:
    x = np.asarray(x_grid, dtype=np.float64)
    VP = np.asarray(VP, dtype=np.float64)
    VP_delta = np.asarray(VP_delta, dtype=np.float64)
    poc_idx = np.asarray(poc_idx, dtype=np.int64)

    T, N = VP.shape
    IDX_ZERO = int(np.argmin(np.abs(x)))
    DX = float(np.round(x[1] - x[0], 12))
    LN9 = float(np.log(9.0))
    scale = 1.4826

    RVOL = df_mp["rvol"].astype(np.float64).to_numpy()
    body_pct = df_mp["body_pct"].astype(np.float64).to_numpy()
    tod = df_mp["tod"].astype(np.int64).to_numpy()
    reset_flag = df_mp["reset_flag_gap_gt_5"].astype(bool).to_numpy()

    VP_sum = VP.sum(axis=1)
    mu_prof = (VP @ x) / (VP_sum + EPS_VOL)
    Ex2 = (VP @ (x**2)) / (VP_sum + EPS_VOL)
    sigma_prof = np.sqrt(np.maximum(Ex2 - mu_prof**2, 0.0))
    sigma_eff = np.maximum(sigma_prof, 2.0 * DX)

    D = (-mu_prof) / (sigma_eff + EPS_PDF)
    Dclip = clip(D, -6.0, 6.0)

    VP_max = VP.max(axis=1)
    A_aff = VP[:, IDX_ZERO] / (VP_max + EPS_VOL)

    row = np.arange(T, dtype=np.int64)
    vp_i0 = VP[:, IDX_ZERO]
    vpd_i0 = VP_delta[:, IDX_ZERO]
    delta0 = vpd_i0 / (vp_i0 + EPS_VOL)

    vp_poc = VP[row, poc_idx]
    vpd_poc = VP_delta[row, poc_idx]
    delta_poc = vpd_poc / (vp_poc + EPS_VOL)

    wpoc = 1.0 - A_aff
    delta_eff = wpoc * delta_poc + (1.0 - wpoc) * delta0

    Sbase_break_long = sigmoid(Dclip * (+1.0) - 1.0) * RVOL
    Sbase_break_short = sigmoid(Dclip * (-1.0) - 1.0) * RVOL

    RVOLtrend = ((RVOL > 2.0) & (body_pct > 0.60)).astype(np.float64)
    Sbase_reject = sigmoid(2.0 - np.abs(Dclip)) * A_aff * (1.0 - RVOLtrend)

    d_delta_eff = np.zeros(T, dtype=np.float64)
    d_delta_eff[1:] = delta_eff[1:] - delta_eff[:-1]
    if T >= 2:
        d_delta_eff[1:] = np.where(reset_flag[:-1], 0.0, d_delta_eff[1:])

    sigma_level = np.zeros(T, dtype=np.float64)
    sigma_chg = np.zeros(T, dtype=np.float64)
    for t in range(T):
        mt = int(tod[t])
        L = min(3 * int(w), mt)
        if L <= 0 or t == 0:
            continue
        start = max(0, t - L)
        end = t
        sigma_level[t] = scale * mad_scalar(delta_eff[start:end])
        sigma_chg[t] = scale * mad_scalar(d_delta_eff[start:end])

    sigma_delta = np.maximum(np.maximum(sigma_level, sigma_chg), 0.05)
    z_delta = delta_eff / (sigma_delta + EPS_PDF)

    Gbreak = sigmoid(LN9 * (z_delta - 1.0))
    Greject = sigmoid(LN9 * (-z_delta - 1.0))

    Score_BO_Long = Sbase_break_long * Gbreak
    Score_BO_Short = Sbase_break_short * Gbreak
    Score_Reject = Sbase_reject * Greject

    ipoc, ivah, ival = compute_value_area_idx_offsetscan(VP, poc_idx, va_threshold=CFG.va_threshold)

    out = df_mp.copy()
    out["D"] = D
    out["Dclip"] = Dclip
    out["A_affinity"] = A_aff
    out["delta_eff"] = delta_eff
    out["z_delta"] = z_delta
    out["Gbreak"] = Gbreak
    out["Greject"] = Greject
    out["Score_BO_Long"] = Score_BO_Long
    out["Score_BO_Short"] = Score_BO_Short
    out["Score_Reject"] = Score_Reject
    out["ipoc"] = ipoc
    out["ivah"] = ivah
    out["ival"] = ival
    out["x_poc"] = x[ipoc]
    out["x_vah"] = x[ivah]
    out["x_val"] = x[ival]

    return out


def inject_price_levels(out: pd.DataFrame) -> pd.DataFrame:
    df = out.copy()
    if "ATR_floor" not in df.columns:
        df["ATR_floor"] = compute_atr_floor(df, atr_period=14).astype(np.float64)

    close = df["close"].astype(np.float64).to_numpy()
    atrf = df["ATR_floor"].astype(np.float64).to_numpy()

    df["POC_price"] = close + df["x_poc"].to_numpy(dtype=np.float64) * atrf
    df["VAH_price"] = close + df["x_vah"].to_numpy(dtype=np.float64) * atrf
    df["VAL_price"] = close + df["x_val"].to_numpy(dtype=np.float64) * atrf
    return df


def _compute_one_symbol_parity(
    df_raw: pd.DataFrame,
    symbol: str,
    window: int,
    include_aux: bool = False,
) -> pd.DataFrame:
    df0 = _prepare_df(df_raw)
    if df0.empty:
        raise RuntimeError(
            f"PARITY_EMPTY_INPUT_BEFORE_SESSION_PROCESSING: symbol={symbol}, window={int(window)}"
        )
    df0["ATR_floor"] = compute_atr_floor(df0, atr_period=14).astype(np.float64)
    df0["rvol"] = compute_rvol_diurnal_causal(df0, vol_col="volume").astype(np.float64)

    body = np.abs(df0["close"].to_numpy(dtype=np.float64) - df0["open"].to_numpy(dtype=np.float64))
    rng = np.maximum(df0["high"].to_numpy(dtype=np.float64) - df0["low"].to_numpy(dtype=np.float64), 1e-12)
    df0["body_pct"] = np.clip(body / rng, 0.0, 1.0)

    chunks: list[pd.DataFrame] = []

    def _emit_chunk(frame: pd.DataFrame) -> pd.DataFrame | None:
        built = build_daily_vp_tensors(frame, x_grid=X_GRID, w=int(window))
        VP = built["VP"]
        VP_delta = built["VP_delta"]
        poc_idx = built["POC_idx"]
        df_mp = built["df_mp"]
        if VP.shape[0] == 0:
            return None

        day_out = compute_r6_spec_from_tensors(
            df_mp=df_mp,
            VP=VP,
            VP_delta=VP_delta,
            poc_idx=poc_idx,
            x_grid=X_GRID,
            w=int(window),
        )
        day_out = inject_price_levels(day_out)

        vp = VP.sum(axis=1)
        vpdelta = VP_delta.sum(axis=1)
        vpbuy = 0.5 * (vp + vpdelta)
        vpsell = 0.5 * (vp - vpdelta)

        sbreak = np.maximum(
            day_out["Score_BO_Long"].to_numpy(dtype=np.float64),
            day_out["Score_BO_Short"].to_numpy(dtype=np.float64),
        )
        sreject = day_out["Score_Reject"].to_numpy(dtype=np.float64)

        feat_map: dict[str, Any] = {
            "timestamp": pd.to_datetime(day_out["ts"], utc=True),
            "symbol": str(symbol),
            "close": day_out["close"].to_numpy(dtype=np.float64),
            "POC": day_out["POC_price"].to_numpy(dtype=np.float64),
            "VAL": day_out["VAL_price"].to_numpy(dtype=np.float64),
            "VAH": day_out["VAH_price"].to_numpy(dtype=np.float64),
            "VP": vp.astype(np.float64),
            "VPbuy": vpbuy.astype(np.float64),
            "VPsell": vpsell.astype(np.float64),
            "VPdelta": vpdelta.astype(np.float64),
            "D": day_out["Dclip"].to_numpy(dtype=np.float64),
            "A": day_out["A_affinity"].to_numpy(dtype=np.float64),
            "DeltaEff": day_out["delta_eff"].to_numpy(dtype=np.float64),
            "Sbreak": sbreak.astype(np.float64),
            "Sreject": sreject.astype(np.float64),
        }
        if include_aux:
            feat_map["RVOL"] = day_out["rvol"].to_numpy(dtype=np.float64)
            feat_map["ATR"] = day_out["ATR_floor"].to_numpy(dtype=np.float64)
        return pd.DataFrame(feat_map)

    # Preserve session-bucket processing for normal windows.
    for _, day_df in df0.groupby("session_date", sort=True):
        feat = _emit_chunk(day_df)
        if feat is not None:
            chunks.append(feat)

    # Fail over to continuous multi-session processing for long windows (e.g. week-sized W).
    if not chunks:
        feat_all = _emit_chunk(df0)
        if feat_all is not None:
            chunks.append(feat_all)

    if not chunks:
        day_sizes = df0.groupby("session_date", sort=True).size()
        max_session_rows = int(day_sizes.max()) if len(day_sizes) else 0
        raise RuntimeError(
            "PARITY_EMPTY_AFTER_SESSION_PROCESSING: "
            f"symbol={symbol}, window={int(window)}, rows={int(len(df0))}, "
            f"max_session_rows={max_session_rows}"
        )

    out = pd.concat(chunks, ignore_index=True)
    out = out.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return out


def compute_market_profile_features(
    df: pd.DataFrame,
    window: int = 60,
    include_aux: bool = False,
) -> pd.DataFrame:
    if df.empty:
        raise RuntimeError("PARITY_EMPTY_INPUT")

    symbol_col = None
    for c in ("symbol", "Symbol", "ticker", "Ticker"):
        if c in df.columns:
            symbol_col = c
            break

    parts: list[pd.DataFrame] = []
    if symbol_col is None:
        parts.append(
            _compute_one_symbol_parity(
                df,
                symbol="UNKNOWN",
                window=int(window),
                include_aux=bool(include_aux),
            )
        )
    else:
        for sym in sorted(pd.unique(df[symbol_col].astype(str)).tolist()):
            parts.append(
                _compute_one_symbol_parity(
                    df[df[symbol_col].astype(str) == sym],
                    symbol=str(sym),
                    window=int(window),
                    include_aux=bool(include_aux),
                )
            )

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)

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
    if include_aux:
        numeric_cols.extend(["RVOL", "ATR"])
    arr = out[numeric_cols].to_numpy(dtype=np.float64)
    if not np.isfinite(arr).all():
        raise RuntimeError("PARITY_NONFINITE_OUTPUT")
    out_cols = [
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
    if include_aux:
        out_cols.extend(["RVOL", "ATR"])
    return out[out_cols]
