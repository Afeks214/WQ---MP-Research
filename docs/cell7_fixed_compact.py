# ==============================================================================

# CELL R7-ARMOR (SEALED, SELF-CONTAINED) — Master Batch Processor + RUN + PLOTS

# ==============================================================================

# WHAT THIS CELL DOES (END-TO-END):

#   1) Preprocess raw OHLCV (session columns, gaps, ATR_floor, RVOL, body_pct)

#   2) Build per-day rolling VP tensors (VP, VP_delta, POC_idx) — self-contained

#   3) Compute SPEC Section 13–14 outputs (mu_prof, sigma_prof, Dclip, A_affinity,

#      delta_eff, z_delta, Gbreak, Greject, trading scores)

#   4) Map lattice levels to price space (POC/VAH/VAL prices)

#   5) Persist previous-day levels and normalized distances (clipped)

#   6) Produce visible OUTPUT:

#        - df_enriched head + describe

#        - 3 plots (price+levels, state+gating, scores)

#

# Dependencies: numpy, pandas, matplotlib only.

# ==============================================================================

# SEALED upgrades:

#   [SEAL-1] Self-contained VP_BUILDER (no external cell dependency).

#   [SEAL-2] RVOL computed causally across sessions by same TOD (diurnal baseline).

#   [SEAL-3] Value Area computed via deterministic offset-scan (no while loops).

#   [SEAL-4] Prev-day normalized distances clipped to prevent outlier blowups.

#   [SEAL-5] Reduced merges/copies in long loops (uses reindex maps).

#   [SEAL-6] TS normalization to UTC tz-aware for stable reindexing.

#   [SEAL-7] ATR_floor mapping fallback (compute locally if mapping fails).

# ==============================================================================

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from datetime import datetime

from typing import Optional, Dict, Any

# ------------------------------------------------------------------------------

# 1) Locked numerical helpers (deterministic)

# ------------------------------------------------------------------------------

EPS_PDF   = 1e-12

EPS_VOL   = 1e-12

EPS_RANGE = 1e-12

def sigmoid(u: np.ndarray) -> np.ndarray:

    u = np.asarray(u, dtype=np.float64)

    return 1.0 / (1.0 + np.exp(-u))

def clip(x: np.ndarray, a: float, b: float) -> np.ndarray:

    x = np.asarray(x, dtype=np.float64)

    return np.minimum(np.maximum(x, a), b)

def mad_scalar(x: np.ndarray) -> float:

    x = np.asarray(x, dtype=np.float64)

    if x.size == 0:

        return 0.0

    m = np.median(x)

    return float(np.median(np.abs(x - m)))

def assert_finite(name: str, arr: np.ndarray):

    arr = np.asarray(arr)

    if not np.all(np.isfinite(arr)):

        bad = np.where(~np.isfinite(arr))[0][:10]

        raise RuntimeError(f"[FAIL] Non-finite in {name}. First bad indices: {bad}")

# ------------------------------------------------------------------------------

# 2) Session columns (strictly deterministic, US/Eastern RTH)

# ------------------------------------------------------------------------------

def add_session_columns(df_raw: pd.DataFrame) -> pd.DataFrame:

    df = df_raw.copy()

    # Ensure datetime index or ts column

    if "ts" in df.columns:

        ts = pd.to_datetime(df["ts"], errors="coerce")

    else:

        ts = pd.to_datetime(df.index, errors="coerce")

    if ts.isna().any():

        raise RuntimeError("[R7] ts contains NaT after parsing. Check your raw index/ts formatting.")

    # Force tz-aware to UTC then convert to US/Eastern for session logic

    if ts.dt.tz is None:

        ts = ts.dt.tz_localize("UTC")

    else:

        ts = ts.dt.tz_convert("UTC")

    ts_et = ts.dt.tz_convert("US/Eastern")

    df["ts"] = ts                      # tz-aware UTC

    df["ts_et"] = ts_et                # tz-aware ET

    df["session_date"] = ts_et.dt.date

    df["minute_of_day"] = ts_et.dt.hour * 60 + ts_et.dt.minute

    df["tod"] = df["minute_of_day"] - (9 * 60 + 30)  # minutes since 09:30 ET

    return df

def add_gap_flags(df: pd.DataFrame) -> pd.DataFrame:

    ts = pd.to_datetime(df["ts"], utc=True)

    dt_min = ts.diff().dt.total_seconds() / 60.0

    df["gap_min"] = dt_min.fillna(0.0)

    df["reset_flag_gap_gt_5"] = (df["gap_min"] > 5.0).astype(np.int8)

    return df

# ------------------------------------------------------------------------------

# 3) ATR_floor (used for price-space mapping + prev-day normalized distances)

# ------------------------------------------------------------------------------

def compute_atr_floor(df: pd.DataFrame, atr_period: int = 14) -> pd.Series:

    high  = df["high"].astype(np.float64).values

    low   = df["low"].astype(np.float64).values

    close = df["close"].astype(np.float64).values

    prev_close = np.roll(close, 1)

    prev_close[0] = close[0]

    tr1 = high - low

    tr2 = np.abs(high - prev_close)

    tr3 = np.abs(low - prev_close)

    tr = np.maximum(tr1, np.maximum(tr2, tr3))

    atr = pd.Series(tr, index=df.index).ewm(span=atr_period, adjust=False).mean().astype(np.float64)

    tick = float(globals().get("tick_size") or globals().get("TICK_SIZE") or 1e-4)

    atr_floor = np.maximum(atr.values, np.maximum(4.0 * tick, 0.0002 * close))

    return pd.Series(atr_floor, index=df.index, dtype=np.float64)

# ------------------------------------------------------------------------------

# 4) RVOL (SEALED): diurnal baseline across sessions, causal, same TOD

# ------------------------------------------------------------------------------

def compute_rvol_diurnal_causal(df: pd.DataFrame, vol_col: str = "volume") -> pd.Series:

    if "tod" not in df.columns:

        raise RuntimeError("compute_rvol_diurnal_causal requires 'tod'. Call add_session_columns first.")

    vol = df[vol_col].astype(np.float64)

    baseline = (

        df.groupby("tod", sort=False)[vol_col]

          .transform(lambda s: s.expanding().median().shift(1))

          .astype(np.float64)

    )

    global_base = vol.expanding().median().shift(1)

    baseline = baseline.fillna(global_base).fillna(method="bfill")

    return vol / (baseline + EPS_VOL)

# ------------------------------------------------------------------------------

# 5) Value Area boundaries (SEALED): deterministic offset-scan (no while loops)

# ------------------------------------------------------------------------------

def make_va_offsets(N: int) -> np.ndarray:

    offs = [0]

    for k in range(1, N):

        offs.append(+k)

        offs.append(-k)

        if len(offs) >= N:

            break

    return np.array(offs[:N], dtype=np.int64)

def compute_value_area_idx_offsetscan(

    VP: np.ndarray,

    poc_idx: np.ndarray,

    va_threshold: float = 0.70

) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    VP = np.asarray(VP, dtype=np.float64)

    poc_idx = np.asarray(poc_idx, dtype=np.int64)

    T, N = VP.shape

    offsets = make_va_offsets(N)

    ipoc = poc_idx.copy()

    total = VP.sum(axis=1)

    target = total * float(va_threshold)

    idx_mat = ipoc[:, None] + offsets[None, :]

    valid = (idx_mat >= 0) & (idx_mat < N)

    vp_scan = np.zeros((T, N), dtype=np.float64)

    rr, cc = np.where(valid)

    vp_scan[rr, cc] = VP[rr, idx_mat[rr, cc]]

    csum = np.cumsum(vp_scan, axis=1)

    k = np.argmax(csum >= target[:, None], axis=1).astype(np.int64)

    empty = (total <= 0.0)

    k = np.where(empty, 0, k)

    j = np.arange(N, dtype=np.int64)[None, :]

    included = j <= k[:, None]

    idx_included = np.where(included & valid, idx_mat, -1)

    ivah = idx_included.max(axis=1)

    big = np.int64(N + 10)

    ivaltmp = np.where(idx_included >= 0, idx_included, big)

    ival = ivaltmp.min(axis=1)

    ivah = np.where(ivah < 0, ipoc, ivah)

    ival = np.where((ival >= big) | (ival < 0), ipoc, ival)

    return ipoc, ivah, ival

# ------------------------------------------------------------------------------

# 6) SELF-CONTAINED VP_BUILDER: per-day rolling VP tensors aligned to df_mp

# ------------------------------------------------------------------------------

def build_daily_vp_tensors(

    df_day: pd.DataFrame,

    X_GRID: np.ndarray,

    W: int = 60,

    price_mode: str = "close",  # "close" or "hlc3"

    vol_col: str = "volume",

) -> Dict[str, Any]:

    df = df_day.copy()

    df.columns = [c.lower() for c in df.columns]

    # Normalize ts to UTC tz-aware (SEALED)

    if "ts" not in df.columns:

        df["ts"] = pd.to_datetime(df.index, utc=True)

    else:

        df["ts"] = pd.to_datetime(df["ts"], utc=True)

    df = df.sort_values("ts")

    for c in ["open", "high", "low", "close", vol_col]:

        if c not in df.columns:

            raise RuntimeError(f"build_daily_vp_tensors missing column: {c}")

    close = df["close"].astype(np.float64).values

    if price_mode.lower() == "hlc3":

        p = (df["high"].astype(np.float64).values + df["low"].astype(np.float64).values + close) / 3.0

    else:

        p = close

    v = df[vol_col].astype(np.float64).values

    # Signed Return Volume proxy (robust, deterministic)

    dP = np.zeros_like(close, dtype=np.float64)

    dP[1:] = close[1:] - close[:-1]

    sgn = np.sign(dP)  # -1,0,+1

    v_delta = v * sgn

    x = np.asarray(X_GRID, dtype=np.float64)

    N = int(len(x))

    if N < 3:

        raise RuntimeError("X_GRID too short.")

    DX = float(x[1] - x[0])

    x_min = float(x[0])

    # Need ATR_floor to map x<->price. Prefer precomputed column from df_day_full.

    if "ATR_floor" in df.columns:

        atrf = df["ATR_floor"].astype(np.float64).values

    elif "atr_floor" in df.columns:

        atrf = df["atr_floor"].astype(np.float64).values

    else:

        tmp = df.copy()

        tmp["ATR_floor"] = compute_atr_floor(tmp, atr_period=14).astype(np.float64)

        atrf = tmp["ATR_floor"].values

    M = len(df)

    if M < W:

        return {

            "VP": np.zeros((0, N), dtype=np.float64),

            "VP_delta": np.zeros((0, N), dtype=np.float64),

            "POC_idx": np.zeros((0,), dtype=np.int64),

            "df_mp": df.iloc[0:0].copy(),

            "ts": np.array([], dtype="datetime64[ns]"),

        }

    start_idx = W - 1

    T = M - start_idx

    df_mp = df.iloc[start_idx:].copy().reset_index(drop=True)

    ts_out = pd.to_datetime(df_mp["ts"], utc=True).values

    VP = np.zeros((T, N), dtype=np.float64)

    VPd = np.zeros((T, N), dtype=np.float64)

    for t_out in range(T):

        t_end = start_idx + t_out

        w0 = t_end - (W - 1)

        w1 = t_end + 1

        atr_end = float(atrf[t_end])

        if (not np.isfinite(atr_end)) or (atr_end <= 0.0):

            atr_end = float(np.maximum(1e-4, 0.0002 * close[t_end]))

        pw = p[w0:w1]

        vw = v[w0:w1]

        vdw = v_delta[w0:w1]

        xw = (pw - close[t_end]) / (atr_end + EPS_PDF)

        idx = np.rint((xw - x_min) / DX).astype(np.int64)

        idx = np.clip(idx, 0, N - 1)

        np.add.at(VP[t_out], idx, vw)

        np.add.at(VPd[t_out], idx, vdw)

    POC_idx = np.argmax(VP, axis=1).astype(np.int64)

    assert_finite("VP(day)", VP)

    assert_finite("VP_delta(day)", VPd)

    return {

        "VP": VP,

        "VP_delta": VPd,

        "POC_idx": POC_idx,

        "df_mp": df_mp,

        "ts": ts_out,

    }

# ------------------------------------------------------------------------------

# 7) VP_BUILDER binding (self-contained)

# ------------------------------------------------------------------------------

def VP_BUILDER(df_day_full: pd.DataFrame) -> Dict[str, Any]:

    return build_daily_vp_tensors(df_day_full, X_GRID=X_GRID, W=60, price_mode="close", vol_col="volume")

# ------------------------------------------------------------------------------

# 8) SPEC Sections 13–14 computation

# ------------------------------------------------------------------------------

def compute_r6_spec_from_tensors(

    df_mp: pd.DataFrame,

    VP: np.ndarray,

    VP_delta: np.ndarray,

    poc_idx: np.ndarray,

    X_GRID: np.ndarray,

    W: int = 60,

) -> pd.DataFrame:

    x = np.asarray(X_GRID, dtype=np.float64)

    VP = np.asarray(VP, dtype=np.float64)

    VP_delta = np.asarray(VP_delta, dtype=np.float64)

    poc_idx = np.asarray(poc_idx, dtype=np.int64)

    T, N = VP.shape

    if VP_delta.shape != (T, N):

        raise RuntimeError(f"VP_delta shape mismatch: {VP_delta.shape} vs {(T,N)}")

    if poc_idx.shape[0] != T:

        raise RuntimeError(f"POC_idx length mismatch: {poc_idx.shape[0]} vs T={T}")

    if len(x) != N:

        raise RuntimeError(f"X_GRID length mismatch: {len(x)} vs N={N}")

    # Required scalars

    RVOL = df_mp["rvol"].astype(np.float64).values

    body_pct = df_mp["body_pct"].astype(np.float64).values

    tod = df_mp["tod"].astype(np.int64).values

    reset_flag = df_mp["reset_flag_gap_gt_5"].astype(bool).values

    IDX_ZERO = int(np.argmin(np.abs(x)))

    DX = float(np.round(x[1] - x[0], 12))

    LN9 = float(np.log(9.0))

    scale = 1.4826

    # --- Section 13 ---

    VP_sum = VP.sum(axis=1)

    mu_prof = (VP @ x) / (VP_sum + EPS_VOL)

    Ex2 = (VP @ (x ** 2)) / (VP_sum + EPS_VOL)

    sigma_prof = np.sqrt(np.maximum(Ex2 - mu_prof**2, 0.0))

    sigma_eff = np.maximum(sigma_prof, 2.0 * DX)

    D = (-mu_prof) / (sigma_eff + EPS_PDF)

    Dclip = clip(D, -6.0, 6.0)

    VP_max = VP.max(axis=1)

    A_aff = VP[:, IDX_ZERO] / (VP_max + EPS_VOL)

    # --- Section 13.2 ---

    row = np.arange(T, dtype=np.int64)

    vp_i0 = VP[:, IDX_ZERO]

    vpd_i0 = VP_delta[:, IDX_ZERO]

    delta0 = vpd_i0 / (vp_i0 + EPS_VOL)

    vp_poc = VP[row, poc_idx]

    vpd_poc = VP_delta[row, poc_idx]

    delta_poc = vpd_poc / (vp_poc + EPS_VOL)

    wpoc = 1.0 - A_aff

    delta_eff = wpoc * delta_poc + (1.0 - wpoc) * delta0

    # --- Section 14: base scores ---

    Sbase_break_long  = sigmoid(Dclip * (+1.0) - 1.0) * RVOL

    Sbase_break_short = sigmoid(Dclip * (-1.0) - 1.0) * RVOL

    RVOLtrend = ((RVOL > 2.0) & (body_pct > 0.60)).astype(np.float64)

    Sbase_reject = sigmoid(2.0 - np.abs(Dclip)) * A_aff * (1.0 - RVOLtrend)

    # --- Section 14.1: delta gating ---

    d_delta_eff = np.zeros(T, dtype=np.float64)

    d_delta_eff[1:] = delta_eff[1:] - delta_eff[:-1]

    if T >= 2:

        d_delta_eff[1:] = np.where(reset_flag[:-1], 0.0, d_delta_eff[1:])

    sigma_level = np.zeros(T, dtype=np.float64)

    sigma_chg   = np.zeros(T, dtype=np.float64)

    for t in range(T):

        mt = int(tod[t])

        L = min(3 * int(W), mt)

        if L <= 0 or t == 0:

            continue

        start = max(0, t - L)

        end = t

        sigma_level[t] = scale * mad_scalar(delta_eff[start:end])

        sigma_chg[t]   = scale * mad_scalar(d_delta_eff[start:end])

    sigma_delta = np.maximum(np.maximum(sigma_level, sigma_chg), 0.05)

    z_delta = delta_eff / (sigma_delta + EPS_PDF)

    Gbreak  = sigmoid(LN9 * (z_delta - 1.0))

    Greject = sigmoid(LN9 * (-z_delta - 1.0))

    Score_BO_Long  = Sbase_break_long  * Gbreak

    Score_BO_Short = Sbase_break_short * Gbreak

    Score_Reject   = Sbase_reject      * Greject

    Score_Rej_Long  = Score_Reject * sigmoid(-Dclip)

    Score_Rej_Short = Score_Reject * sigmoid(+Dclip)

    # --- Value Area (SEALED) ---

    ipoc, ivah, ival = compute_value_area_idx_offsetscan(VP, poc_idx, va_threshold=0.70)

    out = df_mp.copy()

    out["mu_prof"] = mu_prof

    out["sigma_prof"] = sigma_prof

    out["sigma_eff_prof"] = sigma_eff

    out["D"] = D

    out["Dclip"] = Dclip

    out["A_affinity"] = A_aff

    out["delta0"] = delta0

    out["delta_poc"] = delta_poc

    out["delta_eff"] = delta_eff

    out["z_delta"] = z_delta

    out["Gbreak"] = Gbreak

    out["Greject"] = Greject

    out["Score_BO_Long"] = Score_BO_Long

    out["Score_BO_Short"] = Score_BO_Short

    out["Score_Reject"] = Score_Reject

    out["Score_Rej_Long"] = Score_Rej_Long

    out["Score_Rej_Short"] = Score_Rej_Short

    out["ipoc"] = ipoc

    out["ivah"] = ivah

    out["ival"] = ival

    out["x_poc"] = x[ipoc]

    out["x_vah"] = x[ivah]

    out["x_val"] = x[ival]

    for nm in ["mu_prof","sigma_prof","Dclip","A_affinity","delta_eff","Score_BO_Long","Score_Reject"]:

        assert_finite(nm, out[nm].values)

    return out

# ------------------------------------------------------------------------------

# 9) Price-space mapping + previous day persistence (SEALED with clipping)

# ------------------------------------------------------------------------------

def inject_price_levels_and_prev_day(out: pd.DataFrame, dist_clip: float = 10.0) -> pd.DataFrame:

    df = out.copy()

    # If ATR_floor missing, compute locally (SEALED fallback)

    if "ATR_floor" not in df.columns:

        df["ATR_floor"] = compute_atr_floor(df, atr_period=14).astype(np.float64)

    if df["ATR_floor"].isna().all():

        raise RuntimeError("[R7] ATR_floor is all NaN even after fallback. Check input data.")

    close = df["close"].astype(np.float64).values

    atrf  = df["ATR_floor"].astype(np.float64).values

    df["POC_price"] = close + df["x_poc"].values * atrf

    df["VAH_price"] = close + df["x_vah"].values * atrf

    df["VAL_price"] = close + df["x_val"].values * atrf

    df["prev_day_POC"] = np.nan

    df["prev_day_VAH"] = np.nan

    df["prev_day_VAL"] = np.nan

    prev_poc = np.nan

    prev_vah = np.nan

    prev_val = np.nan

    last_session = None

    sdates = df["session_date"].values

    col_p = df.columns.get_loc("prev_day_POC")

    col_h = df.columns.get_loc("prev_day_VAH")

    col_l = df.columns.get_loc("prev_day_VAL")

    for i in range(len(df)):

        sd = sdates[i]

        if last_session is None:

            last_session = sd

        if sd != last_session:

            mask = (sdates == last_session)

            prev_poc = df.loc[mask, "POC_price"].iloc[-1]

            prev_vah = df.loc[mask, "VAH_price"].iloc[-1]

            prev_val = df.loc[mask, "VAL_price"].iloc[-1]

            last_session = sd

        df.iat[i, col_p] = prev_poc

        df.iat[i, col_h] = prev_vah

        df.iat[i, col_l] = prev_val

    denom = df["ATR_floor"].astype(np.float64).values + EPS_PDF

    d_poc = (df["close"].astype(np.float64).values - df["prev_day_POC"].astype(np.float64).values) / denom

    d_vah = (df["close"].astype(np.float64).values - df["prev_day_VAH"].astype(np.float64).values) / denom

    d_val = (df["close"].astype(np.float64).values - df["prev_day_VAL"].astype(np.float64).values) / denom

    df["D_prev_POC"] = clip(d_poc, -dist_clip, +dist_clip)

    df["D_prev_VAH"] = clip(d_vah, -dist_clip, +dist_clip)

    df["D_prev_VAL"] = clip(d_val, -dist_clip, +dist_clip)

    return df

# ------------------------------------------------------------------------------

# 10) R7 Factory runner (SEALED): includes ts alignment and ATR_floor mapping

# ------------------------------------------------------------------------------

def run_R7_factory(

    df_raw: pd.DataFrame,

    X_GRID: np.ndarray,

    output_parquet_path: Optional[str] = None,

    write_daily_chunks: bool = False,

    daily_chunk_dir: str = "r7_chunks",

    max_days: Optional[int] = None,

    dist_clip: float = 10.0,

) -> pd.DataFrame:

    df0 = df_raw.copy()

    df0.columns = [c.lower() for c in df0.columns]

    for col in ["open","high","low","close","volume"]:

        if col not in df0.columns:

            raise RuntimeError(f"Missing required column '{col}' in df_raw.")

    # Deterministic sort + sessionize + gaps

    df0 = df0.sort_index()

    df0 = add_session_columns(df0)

    df0 = add_gap_flags(df0)

    # Normalize ts to UTC tz-aware explicitly (SEALED)

    df0["ts"] = pd.to_datetime(df0["ts"], utc=True)

    # Float64 OHLCV

    for col in ["open","high","low","close","volume"]:

        df0[col] = df0[col].astype(np.float64)

    # ATR_floor once for full series

    df0["ATR_floor"] = compute_atr_floor(df0, atr_period=14).astype(np.float64)

    # RVOL once for full series

    if "rvol" not in df0.columns:

        df0["rvol"] = compute_rvol_diurnal_causal(df0, vol_col="volume").astype(np.float64)

    # body_pct once

    if "body_pct" not in df0.columns:

        rng = (df0["high"] - df0["low"]).astype(np.float64).values

        body = np.abs((df0["close"] - df0["open"]).astype(np.float64).values)

        body_pct = body / (rng + EPS_RANGE)

        body_pct = np.where(np.isfinite(body_pct), body_pct, 0.0)

        df0["body_pct"] = body_pct.astype(np.float64)

    # Session list

    sessions = pd.Series(df0["session_date"].values).dropna().unique().tolist()

    sessions = sorted(sessions)

    if max_days is not None:

        sessions = sessions[:int(max_days)]

    if write_daily_chunks:

        import os

        os.makedirs(daily_chunk_dir, exist_ok=True)

    enriched_days = []

    total_days = len(sessions)

    print(f"[R7] Starting factory on {total_days} sessions...")

    t0 = datetime.utcnow()

    # Build UTC tz-aware ts index maps (SEALED)

    df0_indexed = df0.set_index("ts", drop=False)

    map_reset = df0_indexed["reset_flag_gap_gt_5"]

    map_atrf  = df0_indexed["ATR_floor"]

    map_rvol  = df0_indexed["rvol"]

    map_body  = df0_indexed["body_pct"]

    map_tod   = df0_indexed["tod"]

    map_sdate = df0_indexed["session_date"]

    for di, sd in enumerate(sessions, start=1):

        df_day_full = df0[df0["session_date"] == sd]

        if df_day_full.empty:

            continue

        # Fail-closed: drop corrupted OHLCV rows

        bad = df_day_full[["open","high","low","close","volume"]].isna().any(axis=1)

        if bad.any():

            df_day_full = df_day_full.loc[~bad]

        built = VP_BUILDER(df_day_full)

        VP = built.get("VP", built.get("VP_day", None))

        VP_delta = built.get("VP_delta", built.get("DeltaP_day", built.get("VPDelta", None)))

        poc_idx = built.get("POC_idx", built.get("poc_idx", None))

        if VP is None or VP_delta is None or poc_idx is None:

            raise RuntimeError(

                f"[R7] VP_BUILDER output missing required keys on session {sd}.\n"

                f"Keys returned: {list(built.keys())}\n"

                f"Required: VP (or VP_day), VP_delta (or DeltaP_day), POC_idx."

            )

        VP = np.asarray(VP, dtype=np.float64)

        VP_delta = np.asarray(VP_delta, dtype=np.float64)

        poc_idx = np.asarray(poc_idx, dtype=np.int64)

        T = VP.shape[0]

        if T == 0:

            continue

        if VP.shape[1] != len(X_GRID):

            raise RuntimeError(f"[R7] Grid mismatch on {sd}: VP N={VP.shape[1]} but X_GRID N={len(X_GRID)}.")

        if VP_delta.shape != VP.shape:

            raise RuntimeError(f"[R7] VP_delta shape mismatch on {sd}: {VP_delta.shape} vs {VP.shape}.")

        if poc_idx.shape[0] != T:

            raise RuntimeError(f"[R7] POC_idx length mismatch on {sd}: {poc_idx.shape[0]} vs T={T}.")

        # Canonical df_mp aligned to tensors

        df_mp = built["df_mp"].copy() if ("df_mp" in built and isinstance(built["df_mp"], pd.DataFrame)) else df_day_full.tail(T).copy()

        # Ensure df_mp has ts and normalize to UTC tz-aware (SEALED)

        if "ts" not in df_mp.columns:

            df_mp["ts"] = pd.to_datetime(df_mp.index, utc=True)

        else:

            df_mp["ts"] = pd.to_datetime(df_mp["ts"], utc=True)

        df_mp = df_mp.sort_values("ts").copy()

        # Align scalars via reindex on tz-aware UTC ts (SEALED)

        ts_idx = pd.DatetimeIndex(pd.to_datetime(df_mp["ts"], utc=True))

        # Reindex can miss values when ts precision/tz metadata drifts; fill before integer casting.

        ts_et = ts_idx.tz_convert("US/Eastern")

        tod_fallback = (ts_et.hour * 60 + ts_et.minute - (9 * 60 + 30)).astype(np.int64)

        sdate_fallback = np.array(ts_et.date, dtype=object)

        reset_aligned = pd.to_numeric(map_reset.reindex(ts_idx), errors="coerce")

        if "reset_flag_gap_gt_5" in df_mp.columns:

            reset_aligned = reset_aligned.fillna(pd.to_numeric(df_mp["reset_flag_gap_gt_5"], errors="coerce"))

        df_mp["reset_flag_gap_gt_5"] = reset_aligned.fillna(0).astype(np.int8).values

        rvol_aligned = pd.to_numeric(map_rvol.reindex(ts_idx), errors="coerce")

        if "rvol" in df_mp.columns:

            rvol_aligned = rvol_aligned.fillna(pd.to_numeric(df_mp["rvol"], errors="coerce"))

        df_mp["rvol"] = rvol_aligned.fillna(1.0).astype(np.float64).values

        body_aligned = pd.to_numeric(map_body.reindex(ts_idx), errors="coerce")

        if "body_pct" in df_mp.columns:

            body_aligned = body_aligned.fillna(pd.to_numeric(df_mp["body_pct"], errors="coerce"))

        df_mp["body_pct"] = body_aligned.fillna(0.0).astype(np.float64).values

        tod_aligned = pd.to_numeric(map_tod.reindex(ts_idx), errors="coerce")

        if "tod" in df_mp.columns:

            tod_aligned = tod_aligned.fillna(pd.to_numeric(df_mp["tod"], errors="coerce"))

        tod_aligned = tod_aligned.fillna(pd.Series(tod_fallback, index=ts_idx, dtype=np.float64))

        df_mp["tod"] = tod_aligned.astype(np.int64).values

        sdate_aligned = map_sdate.reindex(ts_idx)

        if "session_date" in df_mp.columns:

            sdate_aligned = sdate_aligned.fillna(pd.Series(df_mp["session_date"].values, index=ts_idx))

        df_mp["session_date"] = sdate_aligned.fillna(pd.Series(sdate_fallback, index=ts_idx)).values

        # ATR_floor mapping + fallback

        atrf_aligned = pd.to_numeric(map_atrf.reindex(ts_idx), errors="coerce")

        if "ATR_floor" in df_mp.columns:

            atrf_aligned = atrf_aligned.fillna(pd.to_numeric(df_mp["ATR_floor"], errors="coerce"))

        df_mp["ATR_floor"] = atrf_aligned.astype(np.float64).values

        if (df_mp["ATR_floor"].isna().all()) or (not np.isfinite(df_mp["ATR_floor"].values).any()):

            df_mp["ATR_floor"] = compute_atr_floor(df_mp, atr_period=14).astype(np.float64)

        else:

            df_mp["ATR_floor"] = df_mp["ATR_floor"].fillna(compute_atr_floor(df_mp, atr_period=14)).astype(np.float64)

        # Compute spec outputs

        df_day_out = compute_r6_spec_from_tensors(

            df_mp=df_mp,

            VP=VP,

            VP_delta=VP_delta,

            poc_idx=poc_idx,

            X_GRID=X_GRID,

            W=60,

        )

        df_day_out = inject_price_levels_and_prev_day(df_day_out, dist_clip=dist_clip)

        enriched_days.append(df_day_out)

        if write_daily_chunks:

            chunk_path = f"{daily_chunk_dir}/df_enriched_{sd}.parquet"

            df_day_out.to_parquet(chunk_path, index=False)

        if di == 1 or di % 10 == 0 or di == total_days:

            elapsed = (datetime.utcnow() - t0).total_seconds()

            rate = di / max(elapsed, 1e-9)

            print(f"[R7] Day {di:>3}/{total_days} | session={sd} | T={len(df_day_out):>4} | days/sec={rate:.3f}")

    if not enriched_days:

        raise RuntimeError("[R7] No enriched output produced (check session filtering).")

    df_enriched = pd.concat(enriched_days, axis=0, ignore_index=True)

    print("\n======================== R7 QUALITY REPORT ========================")

    print(f"Rows processed (canonical): {len(df_enriched):,}")

    core_cols = ["Score_BO_Long","Score_BO_Short","Score_Reject","Dclip","A_affinity","delta_eff","Gbreak","Greject"]

    nan_counts = df_enriched[core_cols].isna().sum()

    print("NaN counts (core outputs):")

    print(nan_counts.to_string())

    if nan_counts.sum() != 0:

        raise RuntimeError("[R7 FAIL] NaNs detected in core outputs — fix upstream alignment/data guard.")

    bad_va = ((df_enriched["ival"] > df_enriched["ipoc"]) | (df_enriched["ipoc"] > df_enriched["ivah"])).sum()

    print(f"VA index ordering violations (ival<=ipoc<=ivah): {int(bad_va)}")

    if bad_va > 0:

        raise RuntimeError("[R7 FAIL] Value Area index ordering violated — check VP builder alignment.")

    max_abs_prev = np.nanmax(np.abs(df_enriched[["D_prev_POC","D_prev_VAH","D_prev_VAL"]].values))

    print(f"Max |D_prev_*| after clipping: {float(max_abs_prev):.4f} (clip={float(dist_clip):.2f})")

    if output_parquet_path is not None:

        df_enriched.to_parquet(output_parquet_path, index=False)

        print(f"[R7] Exported df_enriched to: {output_parquet_path}")

    return df_enriched

# ==============================================================================

# 12) RUN + OUTPUTS (VISIBLE DATA + PLOTS)

# ==============================================================================

# You MUST set:

#   - DF_RAW_NAME to the name of your raw dataframe variable (e.g., "df" or "df_raw")

#   - max_days for quick validation, then set None for full run

# ==============================================================================

DF_RAW_NAME = "df_raw"   # change to "df" if your dataframe is named df

MAX_DAYS_TO_RUN = 3      # set None for full run after validation

if "X_GRID" not in globals():

    raise RuntimeError("X_GRID not found. Define X_GRID before running this cell.")

if DF_RAW_NAME not in globals():

    raise RuntimeError(f"Raw dataframe '{DF_RAW_NAME}' not found. Change DF_RAW_NAME to your actual dataframe name.")

df_raw = globals()[DF_RAW_NAME]

df_enriched = run_R7_factory(

    df_raw=df_raw,

    X_GRID=X_GRID,

    output_parquet_path=None,

    write_daily_chunks=False,

    max_days=MAX_DAYS_TO_RUN,

    dist_clip=10.0,

)

# --------- Table output (so you SEE data) ----------

print("\n[R7] df_enriched.head(12):")

try:

    display(df_enriched.head(12))

except Exception:

    print(df_enriched.head(12).to_string())

print("\n[R7] Core columns describe():")

core_cols = ["Score_BO_Long","Score_BO_Short","Score_Reject","Dclip","A_affinity","delta_eff","z_delta","Gbreak","Greject"]

try:

    display(df_enriched[core_cols].describe(percentiles=[0.01,0.05,0.5,0.95,0.99]))

except Exception:

    print(df_enriched[core_cols].describe().to_string())

# --------- Plot prep ----------

ts_plot = pd.to_datetime(df_enriched["ts"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)

df_enriched["_ts_plot"] = ts_plot

# --------- Plot 1: Price + levels ----------

plt.figure(figsize=(14, 6))

plt.plot(df_enriched["_ts_plot"], df_enriched["close"], label="Close", linewidth=1.2)

plt.plot(df_enriched["_ts_plot"], df_enriched["POC_price"], label="POC (today)", linewidth=1.0)

plt.plot(df_enriched["_ts_plot"], df_enriched["VAH_price"], label="VAH (today)", linewidth=1.0)

plt.plot(df_enriched["_ts_plot"], df_enriched["VAL_price"], label="VAL (today)", linewidth=1.0)

plt.plot(df_enriched["_ts_plot"], df_enriched["prev_day_POC"], label="Prev POC", linewidth=1.0, linestyle="--")

plt.plot(df_enriched["_ts_plot"], df_enriched["prev_day_VAH"], label="Prev VAH", linewidth=1.0, linestyle="--")

plt.plot(df_enriched["_ts_plot"], df_enriched["prev_day_VAL"], label="Prev VAL", linewidth=1.0, linestyle="--")

plt.title("R7 — Price + Market Profile Levels (Today + Previous Day)")

plt.xlabel("Time")

plt.ylabel("Price")

plt.legend()

plt.grid(True)

plt.show()

# --------- Plot 2: State + gating ----------

plt.figure(figsize=(14, 6))

plt.plot(df_enriched["_ts_plot"], df_enriched["Dclip"], label="Dclip", linewidth=1.0)

plt.plot(df_enriched["_ts_plot"], df_enriched["delta_eff"], label="delta_eff", linewidth=1.0)

plt.plot(df_enriched["_ts_plot"], df_enriched["z_delta"], label="z_delta", linewidth=1.0)

plt.plot(df_enriched["_ts_plot"], df_enriched["Gbreak"], label="Gbreak", linewidth=1.0, linestyle="--")

plt.plot(df_enriched["_ts_plot"], df_enriched["Greject"], label="Greject", linewidth=1.0, linestyle="--")

plt.title("R7 — State + Delta Gating")

plt.xlabel("Time")

plt.legend()

plt.grid(True)

plt.show()

# --------- Plot 3: Trading scores ----------

plt.figure(figsize=(14, 6))

plt.plot(df_enriched["_ts_plot"], df_enriched["Score_BO_Long"], label="Score_BO_Long", linewidth=1.0)

plt.plot(df_enriched["_ts_plot"], df_enriched["Score_BO_Short"], label="Score_BO_Short", linewidth=1.0)

plt.plot(df_enriched["_ts_plot"], df_enriched["Score_Reject"], label="Score_Reject", linewidth=1.0)

plt.title("R7 — Trading Scores (Breakout Long/Short, Reject)")

plt.xlabel("Time")

plt.legend()

plt.grid(True)

plt.show()

print("\n[R7] DONE — If results look sane, set MAX_DAYS_TO_RUN=None for full-year run.")
