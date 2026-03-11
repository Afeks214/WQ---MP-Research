from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import numpy as np

pd = pytest.importorskip("pandas")
from pandas.testing import assert_frame_equal

import weightiz_module5_harness as h
from weightiz_module1_core import EngineConfig
from weightiz_module2_core import Module2Config
from weightiz_module3_structure import Module3Config
from weightiz_module4_strategy_funnel import Module4Config


def _cfg(T: int, A: int) -> EngineConfig:
    return EngineConfig(
        T=T,
        A=A,
        B=64,
        tick_size=np.full(A, 0.01, dtype=np.float64),
        mode="sealed",
        timezone="America/New_York",
    )


def _frame(idx: "pd.DatetimeIndex", bias: float = 0.0) -> "pd.DataFrame":
    n = int(idx.shape[0])
    t = np.arange(n, dtype=np.float64)
    close = 100.0 + bias + 0.003 * t + 0.12 * np.sin(t / 3.0) + 0.04 * np.cos(t / 7.0)
    open_px = close - 0.01
    high = np.maximum(open_px, close) + 0.02
    low = np.minimum(open_px, close) - 0.02
    volume = 1000.0 + (t % 7.0)
    return pd.DataFrame(
        {
            "open": open_px,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


def _small_market_frames(*, sessions: int = 16, bars_per_session: int = 24) -> dict[str, "pd.DataFrame"]:
    frames: dict[str, pd.DataFrame] = {}
    for sym, bias in [("S1", 0.0), ("S2", 0.2)]:
        chunks = []
        days = pd.date_range("2024-01-02", periods=sessions, freq="B", tz="America/New_York")
        for d_i, d in enumerate(days):
            start = d.replace(hour=9, minute=30, second=0)
            idx = pd.date_range(start=start, periods=bars_per_session, freq="1min", tz="America/New_York")
            chunks.append(_frame(idx.tz_convert("UTC"), bias=bias + 0.05 * d_i))
        frames[sym] = pd.concat(chunks, axis=0)
    return frames


def _normalize_obj(obj):
    if isinstance(obj, dict) and "group_runtime_stats" in obj:
        obj = {k: v for k, v in obj.items() if str(k) != "group_runtime_stats"}
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass
    if hasattr(obj, "tolist") and callable(obj.tolist):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _normalize_obj(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_normalize_obj(v) for v in obj]
    return obj


def _stable_result_hash(out: h.HarnessOutput) -> str:
    payload = {
        "candidate_results": _normalize_obj(out.candidate_results),
        "stats_verdict": _normalize_obj(out.stats_verdict),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _sort_frame(df):
    volatile = {"run_id", "generated_utc", "updated_utc", "evaluation_timestamp", "report_dir"}
    out = df.drop(columns=[c for c in volatile if c in df.columns], errors="ignore").copy()
    sort_priority = [
        "candidate_id",
        "split_id",
        "scenario_id",
        "session_id",
        "timestamp",
        "symbol",
        "asset_idx",
        "t_index",
        "bar_index",
    ]
    sort_cols = [c for c in sort_priority if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="mergesort")
    return out.reset_index(drop=True).reindex(sorted(out.columns), axis=1)


def _run_small_group_bound_harness(
    report_dir: Path,
    *,
    chunk_hard_cap: int,
    scratch_mode: str,
    strict_validation: str,
    export_micro: bool = False,
) -> h.HarnessOutput:
    frames = _small_market_frames(sessions=16, bars_per_session=24)

    def loader(path: str, _tz_name: str):
        return frames[path]

    engine_cfg = _cfg(T=1, A=2)
    m2_configs = [Module2Config(profile_window_bars=3, profile_warmup_bars=3, rvol_lookback_sessions=2)]
    m3_configs = [Module3Config(block_minutes=5, min_block_valid_bars=1, min_block_valid_ratio=0.0)]
    m4_configs = [Module4Config()]
    candidate_specs = [
        h.CandidateSpec(
            candidate_id=f"cand_{i}",
            m2_idx=0,
            m3_idx=0,
            m4_idx=0,
            enabled_assets_mask=np.asarray([True, True] if (i % 2 == 0) else [True, False], dtype=bool),
            tags=(),
        )
        for i in range(3)
    ]
    stress_scenarios = [
        h.StressScenario(
            scenario_id="baseline",
            name="baseline",
            missing_burst_prob=0.0,
            missing_burst_min=0,
            missing_burst_max=0,
            jitter_sigma_bps=0.0,
            slippage_mult=1.0,
            enabled=True,
        )
    ]
    harness_cfg = h.Module5HarnessConfig(
        report_dir=str(report_dir),
        parallel_backend="serial",
        parallel_workers=1,
        purge_bars=0,
        embargo_bars=0,
        wf_train_sessions=2,
        wf_test_sessions=1,
        wf_step_sessions=1,
        cpcv_slices=3,
        cpcv_k_test=1,
        daily_return_min_days=1,
        benchmark_symbol="S1",
        min_asset_coverage=1.0,
        fail_on_non_finite=True,
        seed=19,
        group_bound_execution_enabled=True,
        group_min_candidates_per_chunk=1,
        group_max_candidates_per_chunk_hard=int(chunk_hard_cap),
        group_target_wall_time_sec=60.0,
        group_max_result_payload_bytes=4 * 1024 * 1024,
        group_max_memory_bytes=0,
        scratch_mode=str(scratch_mode),
        strict_candidate_state_validation=str(strict_validation),
        export_micro_diagnostics=bool(export_micro),
        micro_diag_export_block_profiles=bool(export_micro),
        micro_diag_export_funnel=bool(export_micro),
    )
    return h.run_weightiz_harness(
        data_paths=["S1", "S2"],
        symbols=["S1", "S2"],
        engine_cfg=engine_cfg,
        m2_configs=m2_configs,
        m3_configs=m3_configs,
        m4_configs=m4_configs,
        harness_cfg=harness_cfg,
        candidate_specs=candidate_specs,
        data_loader_func=loader,
        stress_scenarios=stress_scenarios,
    )


def test_group_executor_chunk_invariance_same_seed_different_chunk_sizes(tmp_path: Path) -> None:
    out_chunk_1 = _run_small_group_bound_harness(
        tmp_path / "chunk1",
        chunk_hard_cap=1,
        scratch_mode="compact",
        strict_validation="compact_execution_view",
    )
    out_chunk_2 = _run_small_group_bound_harness(
        tmp_path / "chunk2",
        chunk_hard_cap=2,
        scratch_mode="compact",
        strict_validation="compact_execution_view",
    )

    assert _stable_result_hash(out_chunk_1) == _stable_result_hash(out_chunk_2)


def test_group_executor_repeatability_same_seed_same_config_three_times(tmp_path: Path) -> None:
    hashes = [
        _stable_result_hash(
            _run_small_group_bound_harness(
                tmp_path / f"repeat_{i}",
                chunk_hard_cap=2,
                scratch_mode="compact",
                strict_validation="compact_execution_view",
            )
        )
        for i in range(3)
    ]

    assert hashes[0] == hashes[1] == hashes[2]


def test_compact_and_full_scratch_produce_equivalent_canonical_artifacts(tmp_path: Path) -> None:
    compact = _run_small_group_bound_harness(
        tmp_path / "compact",
        chunk_hard_cap=2,
        scratch_mode="compact",
        strict_validation="compact_execution_view",
        export_micro=True,
    )
    full = _run_small_group_bound_harness(
        tmp_path / "full",
        chunk_hard_cap=2,
        scratch_mode="full",
        strict_validation="full_tensorstate",
        export_micro=True,
    )

    required = [
        "strategy_results",
        "trade_log",
        "equity_curves",
        "micro_diagnostics",
        "micro_profile_blocks",
        "funnel_1545",
    ]
    for key in required:
        assert key in compact.artifact_paths
        assert key in full.artifact_paths
        left = _sort_frame(pd.read_parquet(compact.artifact_paths[key]))
        right = _sort_frame(pd.read_parquet(full.artifact_paths[key]))
        assert_frame_equal(left, right, check_dtype=False, rtol=0.0, atol=1e-12)
