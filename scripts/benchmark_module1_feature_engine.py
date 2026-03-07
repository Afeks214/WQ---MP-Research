#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from weightiz_module1_core import FeatureEngineConfig, build_feature_tensor, make_compat_feature_specs


def _build_data(T: int, A: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    base = 100.0 + np.cumsum(rng.normal(0.0, 0.01, size=(T, A)).astype(np.float64), axis=0)
    close = np.maximum(base, 1e-6)
    open_ = close * (1.0 + rng.normal(0.0, 5e-4, size=(T, A)).astype(np.float64))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 4e-4, size=(T, A)).astype(np.float64)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 4e-4, size=(T, A)).astype(np.float64)))
    volume = np.maximum(1.0, rng.lognormal(mean=7.0, sigma=0.4, size=(T, A)).astype(np.float64))
    ts0 = np.datetime64("2024-01-02T14:30:00", "ns").astype(np.int64)
    ts = ts0 + np.arange(T, dtype=np.int64) * np.int64(60 * 1_000_000_000)
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "bar_valid": np.ones((T, A), dtype=bool),
        "ts_ns": ts,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark Module 1 feature engine")
    ap.add_argument("--assets", type=int, default=32)
    ap.add_argument("--bars", type=int, default=100_000)
    ap.add_argument("--backend", type=str, default="ram", choices=["ram", "memmap"])
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--cache-dir", type=str, default="artifacts/feature_cache")
    ap.add_argument("--memmap-path", type=str, default="artifacts/feature_tensor.memmap")
    args = ap.parse_args()

    data = _build_data(T=int(args.bars), A=int(args.assets), seed=int(args.seed))
    specs = make_compat_feature_specs([15, 30, 60, 120, 240])
    cfg = FeatureEngineConfig(
        tensor_backend=str(args.backend),
        compute_backend="numpy",
        parallel_backend="serial",
        seed=int(args.seed),
        cache_dir=str(Path(args.cache_dir).resolve()),
        memmap_path=str(Path(args.memmap_path).resolve()),
        use_cache=False,
    )

    t0 = time.perf_counter()
    tensor, feature_map, window_map, meta = build_feature_tensor(data, specs, engine_cfg=cfg, ts_ns=data["ts_ns"])
    dt = time.perf_counter() - t0

    nbytes = int(np.asarray(tensor).nbytes)
    print(
        {
            "shape": list(tensor.shape),
            "features": int(len(feature_map)),
            "windows": int(len(window_map)),
            "seconds": float(dt),
            "tensor_gb": float(nbytes) / float(1024**3),
            "cache_hit": bool(meta.get("cache_hit", False)),
            "backend": str(args.backend),
        }
    )


if __name__ == "__main__":
    main()
