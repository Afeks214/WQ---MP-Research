from __future__ import annotations

from typing import Any

import numpy as np


def clip01(x: float) -> float:
    if not np.isfinite(x):
        return 1.0
    return float(min(max(float(x), 0.0), 1.0))


def apply_latency_to_target_qty(target_qty_ta: np.ndarray, latency_bars: int) -> np.ndarray:
    tgt = np.asarray(target_qty_ta, dtype=np.float64)
    if tgt.ndim != 2:
        raise RuntimeError(f"target_qty_ta must be 2D, got ndim={tgt.ndim}")
    lag = int(max(0, latency_bars))
    if lag <= 0:
        return tgt.copy()
    t, a = tgt.shape
    out = np.zeros((t, a), dtype=np.float64)
    if lag < t:
        out[lag:, :] = tgt[:-lag, :]
    return out


def resample_returns_horizon(returns_1d: np.ndarray, horizon: int) -> np.ndarray:
    r = np.asarray(returns_1d, dtype=np.float64)
    h = int(horizon)
    if h <= 0:
        raise RuntimeError(f"horizon must be >=1, got {h}")
    if r.size == 0:
        return np.zeros(0, dtype=np.float64)
    if h == 1:
        return r.copy()
    n = int(r.size // h)
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    x = r[: n * h].reshape(n, h)
    return np.prod(1.0 + x, axis=1) - 1.0


def slice_score_from_stats(dsr: dict[str, Any], pbo: dict[str, Any], spa: dict[str, Any]) -> float:
    dsr_arr = np.asarray(dsr.get("dsr", np.zeros(0, dtype=np.float64)), dtype=np.float64)
    dsr_score = clip01(float(np.mean(dsr_arr))) if dsr_arr.size > 0 else 0.5
    pbo_val = float(pbo.get("pbo", np.nan))
    pbo_score = clip01(1.0 - pbo_val) if np.isfinite(pbo_val) else 0.5
    spa_p = float(spa.get("p_value", np.nan))
    spa_score = clip01(1.0 - spa_p) if np.isfinite(spa_p) else 0.5
    return float((dsr_score + pbo_score + spa_score) / 3.0)


def effective_benchmark_for_horizon(benchmark: np.ndarray, horizon: int) -> np.ndarray:
    return resample_returns_horizon(np.asarray(benchmark, dtype=np.float64), horizon=int(horizon))


def cum_return(ret_1d: np.ndarray) -> float:
    r = np.asarray(ret_1d, dtype=np.float64)
    if r.size == 0:
        return 0.0
    return float(np.prod(1.0 + r) - 1.0)


def max_drawdown_from_returns(ret_1d: np.ndarray) -> float:
    r = np.asarray(ret_1d, dtype=np.float64)
    if r.size == 0:
        return 0.0
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    dd = np.where(peak > 0.0, eq / peak - 1.0, 0.0)
    return float(abs(np.min(dd)))


def sharpe_daily(ret_1d: np.ndarray, eps: float = 1e-12) -> float:
    r = np.asarray(ret_1d, dtype=np.float64)
    if r.size < 2:
        return 0.0
    mu = float(np.mean(r))
    sd = float(np.std(r, ddof=1))
    return float(mu / (sd + float(eps)))


def turnover_from_trade_payload(trade_payload: dict[str, np.ndarray] | None, initial_cash: float) -> float:
    if not trade_payload:
        return 0.0
    qty = np.asarray(trade_payload.get("filled_qty", np.zeros(0, dtype=np.float64)), dtype=np.float64)
    px = np.asarray(trade_payload.get("exec_price", np.zeros(0, dtype=np.float64)), dtype=np.float64)
    if qty.size == 0 or px.size == 0 or qty.size != px.size:
        return 0.0
    notional = float(np.sum(np.abs(qty * px)))
    return float(notional / max(float(initial_cash), 1e-12))


def trade_count_from_payload(trade_payload: dict[str, np.ndarray] | None) -> int:
    if not trade_payload:
        return 0
    qty = np.asarray(trade_payload.get("filled_qty", np.zeros(0, dtype=np.float64)), dtype=np.float64)
    if qty.size == 0:
        return 0
    return int(np.sum(np.abs(qty) > 1e-12))


def extract_final_equity(row: dict[str, Any]) -> float:
    payload = row.get("equity_payload")
    if not isinstance(payload, dict):
        return float("nan")
    eq = np.asarray(payload.get("equity", np.zeros(0, dtype=np.float64)), dtype=np.float64)
    if eq.size == 0:
        return float("nan")
    return float(eq[-1])


def margin_exposure_stats_from_equity_payloads(payloads: list[dict[str, np.ndarray]]) -> dict[str, float]:
    if not payloads:
        return {"avg_margin_used_frac": 0.0, "peak_margin_used_frac": 0.0}
    vals: list[np.ndarray] = []
    for p in payloads:
        eq = np.asarray(p.get("equity", np.zeros(0, dtype=np.float64)), dtype=np.float64)
        mg = np.asarray(p.get("margin_used", np.zeros(0, dtype=np.float64)), dtype=np.float64)
        if eq.size == 0 or mg.size == 0 or eq.size != mg.size:
            continue
        frac = np.abs(mg) / np.maximum(np.abs(eq), 1e-12)
        vals.append(frac.astype(np.float64))
    if not vals:
        return {"avg_margin_used_frac": 0.0, "peak_margin_used_frac": 0.0}
    allf = np.concatenate(vals, axis=0)
    return {
        "avg_margin_used_frac": float(np.mean(allf)),
        "peak_margin_used_frac": float(np.max(allf)),
    }


def asset_notional_concentration_from_trade_payloads(payloads: list[dict[str, np.ndarray]]) -> float:
    if not payloads:
        return 0.0
    acc: dict[str, float] = {}
    for p in payloads:
        sym = np.asarray(p.get("symbol", np.zeros(0, dtype=object)), dtype=object)
        qty = np.asarray(p.get("filled_qty", np.zeros(0, dtype=np.float64)), dtype=np.float64)
        px = np.asarray(p.get("exec_price", np.zeros(0, dtype=np.float64)), dtype=np.float64)
        if sym.size == 0 or qty.size == 0 or px.size == 0:
            continue
        n = min(sym.size, qty.size, px.size)
        for i in range(n):
            s = str(sym[i])
            v = float(abs(float(qty[i]) * float(px[i])))
            acc[s] = acc.get(s, 0.0) + v
    if not acc:
        return 0.0
    total = float(sum(acc.values()))
    if total <= 0.0:
        return 0.0
    return float(max(acc.values()) / total)


def asset_pnl_concentration_from_result_rows(rows: list[dict[str, Any]]) -> float:
    acc: dict[str, float] = {}
    for r in rows:
        payload = r.get("asset_pnl_by_symbol", {})
        if not isinstance(payload, dict):
            continue
        for k, v in payload.items():
            sym = str(k)
            vv = float(v)
            if not np.isfinite(vv):
                continue
            acc[sym] = acc.get(sym, 0.0) + vv
    if not acc:
        return 0.0
    abs_vals = np.asarray([abs(float(v)) for v in acc.values()], dtype=np.float64)
    total_abs = float(np.sum(abs_vals))
    if total_abs <= 1e-12:
        return 0.0
    return float(np.max(abs_vals) / total_abs)
