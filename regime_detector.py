from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RegimeConfig:
    vol_window: int = 60
    vol_low_pct: float = 0.33
    vol_high_pct: float = 0.67
    slope_window: int = 60
    slope_z_threshold: float = 0.25
    hurst_window: int = 120
    range_hurst_threshold: float = 0.45
    min_obs_per_mask: int = 20
    eps: float = 1e-12


def _validate_returns_1d(x: np.ndarray, name: str = "benchmark_returns") -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    if arr.ndim != 1:
        raise RuntimeError(f"{name} must be 1D, got ndim={arr.ndim}")
    if arr.shape[0] < 3:
        raise RuntimeError(f"{name} must have T>=3, got T={arr.shape[0]}")
    if not np.all(np.isfinite(arr)):
        bad = np.argwhere(~np.isfinite(arr))[:8]
        raise RuntimeError(f"{name} contains non-finite values at indices {bad.tolist()}")
    return arr


def _rolling_mean_std(x: np.ndarray, window: int, eps: float) -> tuple[np.ndarray, np.ndarray]:
    n = int(x.shape[0])
    w = int(max(1, window))
    out_m = np.full(n, np.nan, dtype=np.float64)
    out_s = np.full(n, np.nan, dtype=np.float64)
    if n < w:
        return out_m, out_s
    cs = np.cumsum(x, dtype=np.float64)
    cs2 = np.cumsum(x * x, dtype=np.float64)
    for t in range(w - 1, n):
        lo = t - w + 1
        s1 = cs[t] - (cs[lo - 1] if lo > 0 else 0.0)
        s2 = cs2[t] - (cs2[lo - 1] if lo > 0 else 0.0)
        mu = s1 / float(w)
        var = max((s2 / float(w)) - mu * mu, 0.0)
        out_m[t] = mu
        out_s[t] = np.sqrt(max(var, float(eps)))
    return out_m, out_s


def _rolling_hurst_rs(x: np.ndarray, window: int, eps: float) -> np.ndarray:
    n = int(x.shape[0])
    w = int(max(8, window))
    out = np.full(n, np.nan, dtype=np.float64)
    if n < w:
        return out
    logw = np.log(float(w))
    for t in range(w - 1, n):
        seg = x[t - w + 1 : t + 1]
        cen = seg - np.mean(seg)
        y = np.cumsum(cen)
        r = float(np.max(y) - np.min(y))
        s = float(np.std(seg, ddof=1))
        rs = r / max(s, float(eps))
        if rs <= 0.0:
            out[t] = 0.5
            continue
        h = np.log(rs) / max(logw, float(eps))
        out[t] = float(np.clip(h, 0.0, 1.0))
    return out


def detect_regimes(
    benchmark_returns: np.ndarray,
    cfg: RegimeConfig | None = None,
) -> dict[str, Any]:
    c = cfg or RegimeConfig()
    r = _validate_returns_1d(benchmark_returns)
    n = int(r.shape[0])

    _, vol = _rolling_mean_std(r, window=int(c.vol_window), eps=float(c.eps))
    _, ma_sd = _rolling_mean_std(r, window=int(c.slope_window), eps=float(c.eps))
    ma, _ = _rolling_mean_std(r, window=int(c.slope_window), eps=float(c.eps))
    slope = np.full(n, np.nan, dtype=np.float64)
    w = int(max(1, c.slope_window))
    if n > w:
        slope[w:] = ma[w:] - ma[:-w]
    slope_z = slope / np.maximum(ma_sd, float(c.eps))
    hurst = _rolling_hurst_rs(r, window=int(c.hurst_window), eps=float(c.eps))

    vol_ok = np.isfinite(vol)
    if np.any(vol_ok):
        v_low = float(np.quantile(vol[vol_ok], float(c.vol_low_pct)))
        v_high = float(np.quantile(vol[vol_ok], float(c.vol_high_pct)))
    else:
        v_low = 0.0
        v_high = 0.0
    vol_reg = np.zeros(n, dtype=np.int8)
    vol_reg[vol_ok & (vol <= v_low)] = np.int8(-1)
    vol_reg[vol_ok & (vol >= v_high)] = np.int8(1)

    tr_reg = np.zeros(n, dtype=np.int8)
    tr_reg[np.isfinite(slope_z) & (slope_z >= float(c.slope_z_threshold))] = np.int8(1)
    tr_reg[np.isfinite(slope_z) & (slope_z <= -float(c.slope_z_threshold))] = np.int8(-1)

    range_reg = np.zeros(n, dtype=np.int8)
    range_reg[np.isfinite(hurst) & (hurst <= float(c.range_hurst_threshold))] = np.int8(1)

    return {
        "volatility_regime": vol_reg,
        "trend_regime": tr_reg,
        "range_regime": range_reg,
        "volatility": vol,
        "slope_z": slope_z,
        "hurst": hurst,
        "config": c,
    }


def build_regime_masks(regime_doc: dict[str, Any], min_obs: int = 20) -> dict[str, np.ndarray]:
    vol = np.asarray(regime_doc.get("volatility_regime"), dtype=np.int8)
    trn = np.asarray(regime_doc.get("trend_regime"), dtype=np.int8)
    rng = np.asarray(regime_doc.get("range_regime"), dtype=np.int8)
    if vol.shape != trn.shape or vol.shape != rng.shape:
        raise RuntimeError("regime arrays must have matching shapes")
    masks = {
        "vol_low": vol == np.int8(-1),
        "vol_mid": vol == np.int8(0),
        "vol_high": vol == np.int8(1),
        "trend_down": trn == np.int8(-1),
        "trend_flat": trn == np.int8(0),
        "trend_up": trn == np.int8(1),
        "range_true": rng == np.int8(1),
        "range_false": rng == np.int8(0),
    }
    out: dict[str, np.ndarray] = {}
    for k, m in masks.items():
        mm = np.asarray(m, dtype=bool)
        if int(np.sum(mm)) >= int(min_obs):
            out[k] = mm
    return out


def regime_sample_counts(masks: dict[str, np.ndarray]) -> dict[str, int]:
    return {str(k): int(np.sum(np.asarray(v, dtype=bool))) for k, v in masks.items()}
