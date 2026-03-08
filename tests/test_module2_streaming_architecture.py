from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from module2.feature_engine_shared_memory import SharedMemoryFeatureEngine
from module2.market_profile_engine import (
    GoldenManifest,
    benchmark_memory_layout,
    verify_golden_manifest,
)
from module2.tensor_builder import RollingMoments, apply_rolling_update, init_rolling_profile_state
from weightiz_module1_core import EngineConfig, preallocate_state
from weightiz_module2_core import Module2Config, run_weightiz_profile_engine


def _state(T: int = 80, A: int = 2):
    ts_ns = np.arange(T, dtype=np.int64) * np.int64(60_000_000_000)
    cfg = EngineConfig(T=T, A=A, B=240, tick_size=np.full(A, 0.01, dtype=np.float64), mode="sealed")
    st = preallocate_state(ts_ns=ts_ns, cfg=cfg, symbols=tuple(f"S{i}" for i in range(A)))
    close = np.full((T, A), 100.0, dtype=np.float64)
    st.open_px[:] = close
    st.high_px[:] = close * 1.001
    st.low_px[:] = close * 0.999
    st.close_px[:] = close
    st.volume[:] = 1_000_000.0
    st.bar_valid[:] = True
    return st


def test_ring_update_matches_naive_window_sum():
    W, A, N = 4, 2, 3
    rolling = init_rolling_profile_state(window=W, assets=A, bins=N)

    history_total: list[np.ndarray] = []
    history_delta: list[np.ndarray] = []
    history_m0: list[np.ndarray] = []
    history_m1: list[np.ndarray] = []
    history_m2: list[np.ndarray] = []

    rng = np.random.default_rng(3)
    for t in range(9):
        total = rng.normal(size=(A, N)).astype(np.float64)
        delta = rng.normal(size=(A, N)).astype(np.float64)
        m0 = rng.normal(size=A).astype(np.float64)
        m1 = rng.normal(size=A).astype(np.float64)
        m2 = rng.normal(size=A).astype(np.float64)

        apply_rolling_update(
            rolling,
            t_index=t,
            inj_total_an=total,
            inj_delta_an=delta,
            moments=RollingMoments(m0=m0, m1=m1, m2=m2),
        )

        history_total.append(total)
        history_delta.append(delta)
        history_m0.append(m0)
        history_m1.append(m1)
        history_m2.append(m2)

        lo = max(0, t - W + 1)
        np.testing.assert_allclose(rolling.vp_total_an, np.sum(history_total[lo : t + 1], axis=0), atol=1e-12)
        np.testing.assert_allclose(rolling.vp_delta_an, np.sum(history_delta[lo : t + 1], axis=0), atol=1e-12)
        np.testing.assert_allclose(rolling.agg_m0_a, np.sum(history_m0[lo : t + 1], axis=0), atol=1e-12)
        np.testing.assert_allclose(rolling.agg_m1_a, np.sum(history_m1[lo : t + 1], axis=0), atol=1e-12)
        np.testing.assert_allclose(rolling.agg_m2_a, np.sum(history_m2[lo : t + 1], axis=0), atol=1e-12)


def test_streaming_engine_freezes_output_tensors():
    st = _state(T=90, A=2)
    run_weightiz_profile_engine(st, Module2Config(profile_window_bars=20, profile_warmup_bars=20))

    assert st.vp.flags.writeable is False
    assert st.vp_delta.flags.writeable is False
    assert st.profile_stats.flags.writeable is False
    assert st.scores.flags.writeable is False


def test_shared_memory_engine_attach_readonly():
    eng = SharedMemoryFeatureEngine()
    arrays = {
        "open": np.arange(12, dtype=np.float64).reshape(3, 4),
        "close": np.ones((3, 4), dtype=np.float64),
    }
    handles = eng.publish_arrays(arrays)
    views, refs = SharedMemoryFeatureEngine.attach_readonly(handles)
    try:
        assert set(views.keys()) == {"open", "close"}
        assert views["open"].flags.writeable is False
        with pytest.raises(ValueError):
            views["open"][0, 0] = 0.0
    finally:
        for shm in refs:
            shm.close()
        eng.close(unlink=True)


def test_golden_manifest_required_hash_fields_must_match(tmp_path: Path):
    manifest = GoldenManifest(
        dataset_hash="d1",
        code_hash="c1",
        spec_version="s1",
        config_signature="k1",
        python_version="3",
        numpy_version="2",
        platform="x",
        timezone="UTC",
        seed=17,
        schema={"vp": "float64"},
        generated_at_utc="2026-03-08T00:00:00+00:00",
    )

    verify_golden_manifest(
        manifest=manifest,
        dataset_hash="d1",
        code_hash="c1",
        spec_version="s1",
        config_signature="k1",
    )

    with pytest.raises(RuntimeError, match="GOLDEN_REPLAY_HASH_MISMATCH"):
        verify_golden_manifest(
            manifest=manifest,
            dataset_hash="d2",
            code_hash="c1",
            spec_version="s1",
            config_signature="k1",
        )


def test_layout_benchmark_returns_c_and_f():
    out = benchmark_memory_layout(steps=40, assets=32, bins=64, window=8)
    assert set(out.keys()) == {"C", "F"}
    assert out["C"] >= 0.0
    assert out["F"] >= 0.0


def test_module2_golden_required_precheck(tmp_path: Path):
    st = _state(T=70, A=1)
    manifest_path = tmp_path / "golden_manifest.json"
    payload = {
        "dataset_hash": "wrong",
        "code_hash": "wrong",
        "spec_version": "wrong",
        "config_signature": "wrong",
        "python_version": "3",
        "numpy_version": "2",
        "platform": "x",
        "timezone": "UTC",
        "seed": 17,
        "schema": {"vp": "float64"},
        "generated_at_utc": "2026-03-08T00:00:00+00:00",
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="GOLDEN_REPLAY_HASH_MISMATCH"):
        run_weightiz_profile_engine(
            st,
            Module2Config(
                profile_window_bars=20,
                profile_warmup_bars=20,
                golden_required=True,
                golden_manifest_path=str(manifest_path),
            ),
        )
