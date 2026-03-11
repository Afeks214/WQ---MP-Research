from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from weightiz.module1.core import (
    FeatureEngineConfig,
    FeatureSpec,
    build_feature_dag,
    build_feature_tensor,
    load_feature_registry,
    resolve_feature_execution_order,
    rolling_view,
    sanitize_market_data,
)


def _toy_data(T: int = 16, A: int = 3) -> dict[str, np.ndarray]:
    base = np.linspace(100.0, 101.5, T, dtype=np.float64)[:, None]
    off = np.arange(A, dtype=np.float64)[None, :] * 0.1
    close = base + off
    open_ = close - 0.02
    high = close + 0.05
    low = close - 0.05
    vol = np.full((T, A), 1000.0, dtype=np.float64)
    ts = np.datetime64("2024-01-02T14:30:00", "ns").astype(np.int64) + np.arange(T, dtype=np.int64) * 60 * 1_000_000_000
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": vol,
        "bar_valid": np.ones((T, A), dtype=bool),
        "ts_ns": ts,
    }


def test_load_feature_registry_and_normalization(tmp_path: Path):
    p = tmp_path / "f.yaml"
    p.write_text(
        """
features:
  - name: momentum
    window: [50, 10, 10]
    input: price
  - name: normalized_momentum
    window: [10, 50]
    input: price
    depends_on: momentum
""".strip(),
        encoding="utf-8",
    )
    specs = load_feature_registry(p)
    assert [s.name for s in specs] == ["momentum", "normalized_momentum"]
    assert specs[0].windows == (10, 50)
    assert specs[1].dependencies == ("momentum",)


def test_dag_topological_and_cycle_detection():
    specs = [
        FeatureSpec(name="a", windows=(5,), input_fields=("close",)),
        FeatureSpec(name="b", windows=(5,), input_fields=("close",), dependencies=("a",)),
    ]
    dag = build_feature_dag(specs)
    assert resolve_feature_execution_order(dag) == ["a", "b"]

    dag_cycle = {"a": {"b"}, "b": {"a"}}
    with pytest.raises(RuntimeError, match="Cyclic"):
        resolve_feature_execution_order(dag_cycle)


def test_rolling_view_shape_and_values():
    x = np.arange(8, dtype=np.float64).reshape(2, 4)
    rv = rolling_view(x, 3)
    assert rv.shape == (2, 2, 3)
    np.testing.assert_allclose(rv[0, 0], np.array([0.0, 1.0, 2.0]))


def test_sanitize_market_data_detects_and_fixes():
    d = _toy_data(T=10, A=2)
    d["close"][3, 0] = np.nan
    d["volume"][5, 1] = np.inf
    d["high"][6, 1] = d["close"][6, 1] + 1000.0  # outlier
    clean, logs = sanitize_market_data(d, FeatureEngineConfig(ffill_gap_limit=2, mad_clip_k=6.0))
    assert clean["close"].shape == d["close"].shape
    assert clean["bar_valid"].shape == d["bar_valid"].shape
    assert len(logs) > 0


def test_vectorized_feature_matches_loop_baseline():
    d = _toy_data(T=12, A=2)
    specs = [FeatureSpec(name="roll_mean_close", windows=(3,), input_fields=("close",))]
    tensor, fmap, wmap, _meta = build_feature_tensor(d, specs, engine_cfg=FeatureEngineConfig(use_cache=False))
    assert tensor.shape == (2, 12, 1, 1)
    assert fmap == {"roll_mean_close": 0}
    assert wmap == {"0": 3}

    close = d["close"]
    ref = np.zeros_like(close)
    for t in range(close.shape[0]):
        s = max(0, t - 2)
        ref[t] = np.mean(close[s : t + 1], axis=0)
    np.testing.assert_allclose(tensor[:, :, 0, 0], ref.T, rtol=0.0, atol=1e-12)


def test_deterministic_replay_serial_and_process_pool(tmp_path: Path):
    d = _toy_data(T=24, A=3)
    specs = [
        FeatureSpec(name="roll_mean_close", windows=(5, 10), input_fields=("close",)),
        FeatureSpec(name="roll_std_close", windows=(5, 10), input_fields=("close",)),
    ]

    cfg_serial = FeatureEngineConfig(seed=123, cache_dir=str(tmp_path / "cache1"), parallel_backend="serial")
    t1, _f1, _w1, _m1 = build_feature_tensor(d, specs, engine_cfg=cfg_serial)
    t2, _f2, _w2, _m2 = build_feature_tensor(d, specs, engine_cfg=cfg_serial)
    np.testing.assert_allclose(np.asarray(t1), np.asarray(t2), rtol=0.0, atol=0.0)

    cfg_pool = FeatureEngineConfig(
        seed=123,
        cache_dir=str(tmp_path / "cache2"),
        parallel_backend="process_pool",
        max_workers=2,
        use_cache=False,
    )
    tp, _fp, _wp, _mp = build_feature_tensor(d, specs, engine_cfg=cfg_pool)
    np.testing.assert_allclose(np.asarray(t1), np.asarray(tp), rtol=0.0, atol=1e-12)


def test_memmap_backend_and_cache_hit(tmp_path: Path):
    d = _toy_data(T=20, A=2)
    specs = [FeatureSpec(name="roll_mean_vol", windows=(4,), input_fields=("volume",))]
    cfg = FeatureEngineConfig(
        tensor_backend="memmap",
        cache_dir=str(tmp_path / "cache"),
        memmap_path=str(tmp_path / "artifacts" / "feature_tensor.memmap"),
    )
    t1, _f1, _w1, m1 = build_feature_tensor(d, specs, engine_cfg=cfg)
    assert Path(cfg.memmap_path).exists()
    assert t1.flags.writeable is False
    assert bool(m1["cache_hit"]) is False

    t2, _f2, _w2, m2 = build_feature_tensor(d, specs, engine_cfg=cfg)
    assert bool(m2["cache_hit"]) is True
    np.testing.assert_allclose(np.asarray(t1), np.asarray(t2), rtol=0.0, atol=1e-12)


def test_cupy_backend_fallback_to_numpy(tmp_path: Path):
    d = _toy_data(T=8, A=2)
    specs = [FeatureSpec(name="roll_mean_close", windows=(3,), input_fields=("close",))]
    cfg = FeatureEngineConfig(compute_backend="cupy", cache_dir=str(tmp_path / "cache"), use_cache=False)
    _t, _f, _w, meta = build_feature_tensor(d, specs, engine_cfg=cfg)
    assert meta["compute_backend_effective"] in {"numpy", "cupy"}
