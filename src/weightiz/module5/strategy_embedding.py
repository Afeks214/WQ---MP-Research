from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StrategyEmbeddingConfig:
    corr_threshold: float = 0.90
    distance_block_size: int = 256
    distance_in_memory_max_n: int = 2500
    autocorr_lag: int = 1
    seed: int = 47
    eps: float = 1e-12


def _validate_returns_2d(x: np.ndarray, name: str = "returns_matrix") -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    if arr.ndim != 2:
        raise RuntimeError(f"{name} must be 2D, got ndim={arr.ndim}")
    t, n = arr.shape
    if t < 3:
        raise RuntimeError(f"{name} must have T>=3, got T={t}")
    if n < 1:
        raise RuntimeError(f"{name} must have N>=1, got N={n}")
    if not np.all(np.isfinite(arr)):
        bad = np.argwhere(~np.isfinite(arr))[:8]
        raise RuntimeError(f"{name} contains non-finite values at indices {bad.tolist()}")
    return arr


def _autocorr_lag(arr_tn: np.ndarray, lag: int, eps: float) -> np.ndarray:
    t, n = arr_tn.shape
    l = int(max(1, lag))
    if t <= l + 1:
        return np.zeros(n, dtype=np.float64)
    x = arr_tn[l:, :]
    y = arr_tn[:-l, :]
    mx = np.mean(x, axis=0)
    my = np.mean(y, axis=0)
    xc = x - mx[None, :]
    yc = y - my[None, :]
    num = np.sum(xc * yc, axis=0)
    den = np.sqrt(np.sum(xc * xc, axis=0) * np.sum(yc * yc, axis=0))
    return np.clip(num / np.maximum(den, float(eps)), -1.0, 1.0)


def _drawdown_profile(arr_tn: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eq = np.cumprod(np.clip(1.0 + arr_tn, float(eps), np.inf), axis=0)
    peak = np.maximum.accumulate(eq, axis=0)
    dd = eq / np.maximum(peak, float(eps)) - 1.0
    dd_mean = np.mean(dd, axis=0)
    dd_q50 = np.quantile(dd, 0.50, axis=0)
    dd_q10 = np.quantile(dd, 0.10, axis=0)
    max_dd = np.abs(np.min(dd, axis=0))
    return dd_mean, dd_q50, dd_q10, max_dd


def _turnover_proxy(arr_tn: np.ndarray, turnover_proxy: np.ndarray | None) -> np.ndarray:
    _, n = arr_tn.shape
    if turnover_proxy is None:
        if arr_tn.shape[0] <= 1:
            return np.zeros(n, dtype=np.float64)
        return np.mean(np.abs(np.diff(arr_tn, axis=0)), axis=0)

    tp = np.asarray(turnover_proxy, dtype=np.float64)
    if tp.ndim == 1:
        if tp.shape[0] != n:
            raise RuntimeError(f"turnover_proxy 1D length mismatch: got {tp.shape[0]}, expected {n}")
        return tp
    if tp.ndim == 2:
        if tp.shape != arr_tn.shape:
            raise RuntimeError(f"turnover_proxy 2D shape mismatch: got {tp.shape}, expected {arr_tn.shape}")
        return np.mean(np.abs(tp), axis=0)
    raise RuntimeError(f"turnover_proxy must be 1D or 2D, got ndim={tp.ndim}")


def build_strategy_embeddings(
    returns_matrix: np.ndarray,
    turnover_proxy: np.ndarray | None = None,
    seed: int = 47,
    autocorr_lag: int = 1,
    eps: float = 1e-12,
) -> dict[str, Any]:
    _ = int(seed)  # Deterministic API contract; retained for stable signature.
    r = _validate_returns_2d(returns_matrix)
    mu = np.mean(r, axis=0)
    vol = np.std(r, axis=0, ddof=1)
    vol_safe = np.maximum(vol, float(eps))
    z = (r - mu[None, :]) / vol_safe[None, :]
    skew = np.mean(z**3, axis=0)
    kurt_excess = np.mean(z**4, axis=0) - 3.0
    ac1 = _autocorr_lag(r, lag=autocorr_lag, eps=eps)
    dd_mean, dd_q50, dd_q10, max_dd = _drawdown_profile(r, eps=eps)
    to_proxy = _turnover_proxy(r, turnover_proxy)
    downside = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2, axis=0))
    hit_rate = np.mean(r > 0.0, axis=0)

    feature_names = [
        "mean_return",
        "volatility",
        "skew",
        "kurtosis_excess",
        "autocorr_lag1",
        "drawdown_mean",
        "drawdown_q50",
        "drawdown_q10",
        "max_drawdown",
        "turnover_proxy",
        "downside_vol",
        "hit_rate",
    ]
    emb = np.column_stack(
        [
            mu,
            vol,
            skew,
            kurt_excess,
            ac1,
            dd_mean,
            dd_q50,
            dd_q10,
            max_dd,
            to_proxy,
            downside,
            hit_rate,
        ]
    ).astype(np.float64, copy=False)
    if not np.all(np.isfinite(emb)):
        bad = np.argwhere(~np.isfinite(emb))[:8]
        raise RuntimeError(f"strategy embeddings contain non-finite values at indices {bad.tolist()}")
    return {
        "embeddings": emb,
        "feature_names": feature_names,
    }


def compute_correlation_distance(
    returns_matrix: np.ndarray,
    block_size: int = 256,
    out_path: str | None = None,
    in_memory_max_n: int = 2500,
    eps: float = 1e-12,
) -> np.ndarray:
    r = _validate_returns_2d(returns_matrix)
    t, n = r.shape
    bs = int(max(1, block_size))
    mu = np.mean(r, axis=0)
    sd = np.std(r, axis=0, ddof=1)
    sd_safe = np.where(sd > float(eps), sd, 1.0)
    z = (r - mu[None, :]) / sd_safe[None, :]
    z[:, sd <= float(eps)] = 0.0
    denom = float(max(t - 1, 1))

    use_memmap = (out_path is not None) or (n > int(in_memory_max_n))
    if use_memmap:
        if out_path is None:
            raise RuntimeError("out_path must be provided when memmap distance is required")
        p = Path(out_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        dist = np.memmap(p, mode="w+", dtype=np.float64, shape=(n, n))
    else:
        dist = np.empty((n, n), dtype=np.float64)

    for i0 in range(0, n, bs):
        i1 = min(n, i0 + bs)
        zi = z[:, i0:i1]
        for j0 in range(i0, n, bs):
            j1 = min(n, j0 + bs)
            zj = z[:, j0:j1]
            corr = (zi.T @ zj) / denom
            blk = np.clip(1.0 - corr, 0.0, 2.0)
            dist[i0:i1, j0:j1] = blk
            if i0 != j0:
                dist[j0:j1, i0:i1] = blk.T
    np.fill_diagonal(dist, 0.0)
    if isinstance(dist, np.memmap):
        dist.flush()
    return dist


def cluster_strategies_hierarchical_threshold(
    returns_matrix: np.ndarray,
    corr_threshold: float = 0.90,
    block_size: int = 256,
    distance_out_path: str | None = None,
    in_memory_max_n: int = 2500,
    turnover_proxy: np.ndarray | None = None,
    seed: int = 47,
    eps: float = 1e-12,
) -> dict[str, Any]:
    r = _validate_returns_2d(returns_matrix)
    _, n = r.shape
    if not (0.0 <= float(corr_threshold) <= 1.0):
        raise RuntimeError(f"corr_threshold must be in [0,1], got {corr_threshold}")
    distance_threshold = float(1.0 - float(corr_threshold))
    emb_doc = build_strategy_embeddings(
        r,
        turnover_proxy=turnover_proxy,
        seed=seed,
        autocorr_lag=1,
        eps=eps,
    )
    dist = compute_correlation_distance(
        r,
        block_size=block_size,
        out_path=distance_out_path,
        in_memory_max_n=in_memory_max_n,
        eps=eps,
    )

    parent = np.arange(n, dtype=np.int64)

    def find(x: int) -> int:
        y = int(x)
        while int(parent[y]) != y:
            parent[y] = parent[int(parent[y])]
            y = int(parent[y])
        return y

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    bs = int(max(1, block_size))
    for i0 in range(0, n, bs):
        i1 = min(n, i0 + bs)
        for j0 in range(i0, n, bs):
            j1 = min(n, j0 + bs)
            blk = np.asarray(dist[i0:i1, j0:j1], dtype=np.float64)
            if i0 == j0:
                ii, jj = np.where(np.triu(blk <= distance_threshold, k=1))
            else:
                ii, jj = np.where(blk <= distance_threshold)
            for k in range(int(ii.shape[0])):
                union(i0 + int(ii[k]), j0 + int(jj[k]))

    roots = np.asarray([find(i) for i in range(n)], dtype=np.int64)
    uniq = np.unique(roots)
    root_to_label = {int(rt): int(i) for i, rt in enumerate(uniq.tolist())}
    labels = np.asarray([root_to_label[int(rt)] for rt in roots.tolist()], dtype=np.int64)

    mu = np.mean(r, axis=0)
    vol = np.std(r, axis=0, ddof=1)
    reps: list[int] = []
    for cid in range(int(uniq.shape[0])):
        idx = np.flatnonzero(labels == cid).astype(np.int64)
        if idx.size <= 0:
            continue
        order = np.lexsort((idx, vol[idx], -mu[idx]))
        reps.append(int(idx[int(order[0])]))
    reps_arr = np.asarray(reps, dtype=np.int64)

    dist_path = ""
    if isinstance(dist, np.memmap):
        dist_path = str(Path(dist.filename).resolve()) if getattr(dist, "filename", None) else ""

    return {
        "embeddings": np.asarray(emb_doc["embeddings"], dtype=np.float64),
        "feature_names": list(emb_doc["feature_names"]),
        "strategy_distance_matrix": dist,
        "cluster_labels": labels,
        "cluster_representatives": reps_arr,
        "n_eff": int(reps_arr.shape[0]),
        "distance_is_memmap": bool(isinstance(dist, np.memmap)),
        "distance_path": dist_path,
        "corr_threshold": float(corr_threshold),
    }
