from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

import weightiz.module5.orchestrator as h
from weightiz.module1.core import EngineConfig
from weightiz.module2.core import Module2Config
from weightiz.module3.bridge import Module3Config
from weightiz.module4.strategy_funnel import Module4Config

pd = pytest.importorskip("pandas")


def _frame(idx: "pd.DatetimeIndex", bias: float = 0.0) -> "pd.DataFrame":
    n = int(idx.shape[0])
    t = np.arange(n, dtype=np.float64)
    close = 100.0 + bias + 0.01 * t
    open_px = close - 0.01
    high = np.maximum(open_px, close) + 0.02
    low = np.minimum(open_px, close) - 0.02
    volume = 1_000.0 + t
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


def _market_frames(*, sessions: int = 5, bars_per_session: int = 8) -> dict[str, "pd.DataFrame"]:
    frames: dict[str, pd.DataFrame] = {}
    for sym, bias in [("S1", 0.0), ("S2", 0.3)]:
        chunks = []
        days = pd.date_range("2024-01-02", periods=sessions, freq="B", tz="America/New_York")
        for day_idx, day in enumerate(days):
            start = day.replace(hour=9, minute=30, second=0)
            idx = pd.date_range(start=start, periods=bars_per_session, freq="1min", tz="America/New_York")
            chunks.append(_frame(idx.tz_convert("UTC"), bias=bias + 0.05 * day_idx))
        frames[sym] = pd.concat(chunks, axis=0)
    return frames


def _run_feedback_probe(report_dir: Path) -> list[int]:
    frames = _market_frames()

    def loader(path: str, _tz_name: str):
        return frames[path]

    engine_cfg = EngineConfig(
        T=1,
        A=2,
        B=32,
        tick_size=np.full(2, 0.01, dtype=np.float64),
        mode="sealed",
        timezone="America/New_York",
    )
    m2_cfgs = [Module2Config(profile_window_bars=3, profile_warmup_bars=3, rvol_lookback_sessions=2)]
    m3_cfgs = [
        Module3Config(block_minutes=5, min_block_valid_bars=1, min_block_valid_ratio=0.0),
        Module3Config(block_minutes=10, min_block_valid_bars=1, min_block_valid_ratio=0.0),
    ]
    m4_cfgs = [Module4Config()]
    candidate_specs = [
        h.CandidateSpec(
            candidate_id="cand_a",
            m2_idx=0,
            m3_idx=0,
            m4_idx=0,
            enabled_assets_mask=np.asarray([True, True], dtype=bool),
            tags=(),
        ),
        h.CandidateSpec(
            candidate_id="cand_b",
            m2_idx=0,
            m3_idx=1,
            m4_idx=0,
            enabled_assets_mask=np.asarray([True, True], dtype=bool),
            tags=(),
        ),
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
        seed=23,
        group_bound_execution_enabled=True,
        group_max_in_flight_factor=1,
        group_min_candidates_per_chunk=1,
        group_max_candidates_per_chunk_hard=16,
        group_target_wall_time_sec=60.0,
        group_max_result_payload_bytes=1024 * 1024,
        group_max_memory_bytes=0,
        startup_default_module3_bytes=111,
    )
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

    seen_estimates: list[int] = []
    orig_chunk = h._chunk_group_execution_task

    def wrap_chunk(*args, **kwargs):
        seen_estimates.append(int(kwargs["module3_group_bytes_estimated"]))
        return orig_chunk(*args, **kwargs)

    def fake_run_group_task(group, *_args, **_kwargs):
        split_id = f"wf_{int(group.split_idx):03d}"
        scenario_id = "baseline"
        realized = 4096 if int(group.m3_idx) == 0 else 8192
        candidate_results: list[dict[str, object]] = []
        for ci in group.candidate_indices:
            candidate_id = candidate_specs[int(ci)].candidate_id
            candidate_results.append(
                {
                    "task_id": f"{candidate_id}|{split_id}|{scenario_id}",
                    "candidate_id": candidate_id,
                    "split_id": split_id,
                    "scenario_id": scenario_id,
                    "status": "ok",
                    "error": "",
                    "session_ids": np.asarray([1], dtype=np.int64),
                    "session_ids_exec": np.asarray([1], dtype=np.int64),
                    "session_ids_raw": np.asarray([1], dtype=np.int64),
                    "daily_returns": np.asarray([0.0], dtype=np.float64),
                    "daily_returns_exec": np.asarray([0.0], dtype=np.float64),
                    "daily_returns_raw": np.asarray([0.0], dtype=np.float64),
                    "test_days": 1,
                    "quality_reason_codes": [],
                    "m2_idx": 0,
                    "m3_idx": int(group.m3_idx),
                    "m4_idx": 0,
                    "tags": [],
                    "dqs_min": 1.0,
                    "dqs_median": 1.0,
                    "group_runtime_stats": {
                        "split_stress_sec": 0.01,
                        "module2_sec": 0.01,
                        "module3_sec": 0.01,
                        "candidate_loop_sec": 0.02,
                        "candidate_count": int(len(group.candidate_indices)),
                        "market_overlay_bytes": 128,
                        "feature_overlay_bytes": 256,
                        "module3_group_bytes_estimated": int(getattr(group, "module3_group_bytes_estimated", 0)),
                        "module3_group_bytes_realized": realized,
                        "module3_bytes": realized,
                        "candidate_scratch_bytes": 64,
                        "result_payload_bytes": 32,
                    },
                }
            )
        return candidate_results

    with (
        mock.patch.object(h, "_chunk_group_execution_task", side_effect=wrap_chunk),
        mock.patch.object(h, "_run_group_task", side_effect=fake_run_group_task),
        mock.patch.object(h, "compute_window_correlation_diagnostics", return_value=([], [])),
        mock.patch.object(
            h,
            "_aggregate_candidate_baseline_matrices",
            return_value=(
                np.asarray([1], dtype=np.int64),
                np.zeros((1, 0), dtype=np.float64),
                np.zeros((1, 0), dtype=np.float64),
                np.zeros(1, dtype=np.float64),
                [],
                {},
            ),
        ),
        mock.patch.object(h, "_compute_stats_verdict", return_value={"leaderboard": []}),
        mock.patch.object(h, "_orchestrator_finalize_run_outputs", return_value=SimpleNamespace(seen_estimates=seen_estimates)),
    ):
        out = h.run_weightiz_harness(
            data_paths=["S1", "S2"],
            symbols=["S1", "S2"],
            engine_cfg=engine_cfg,
            m2_configs=m2_cfgs,
            m3_configs=m3_cfgs,
            m4_configs=m4_cfgs,
            harness_cfg=harness_cfg,
            candidate_specs=candidate_specs,
            data_loader_func=loader,
            stress_scenarios=stress_scenarios,
        )
    return list(out.seen_estimates)


def test_module3_realized_bytes_feedback_updates_later_wave_planning_deterministically(tmp_path: Path) -> None:
    first = _run_feedback_probe(tmp_path / "first")
    second = _run_feedback_probe(tmp_path / "second")

    assert first == second
    assert len(first) >= 2
    assert first[0] == 111
    assert first[1] == 4096
