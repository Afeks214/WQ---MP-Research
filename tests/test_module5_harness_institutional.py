from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore[assignment]

from weightiz_module1_core import EngineConfig, deterministic_digest_sha256, preallocate_state, validate_state_hard
from weightiz_module2_core import Module2Config
from weightiz_module3_structure import ContextIdx, Module3Config, Module3Output, Struct30mIdx
from weightiz_module4_strategy_funnel import Module4Config, Module4Output
import weightiz_module5_harness as h


@unittest.skipIf(pd is None, "pandas not available")
class TestModule5HarnessInstitutional(unittest.TestCase):
    def _cfg(self, T: int, A: int) -> EngineConfig:
        return EngineConfig(
            T=T,
            A=A,
            B=64,
            tick_size=np.full(A, 0.01, dtype=np.float64),
            mode="sealed",
            timezone="America/New_York",
        )

    def _utc_minute_ns(self, start: str, periods: int) -> np.ndarray:
        idx = pd.date_range(start=start, periods=periods, freq="1min", tz="UTC")
        return idx.asi8.astype(np.int64)

    def _frame(self, idx: "pd.DatetimeIndex", bias: float = 0.0) -> "pd.DataFrame":
        n = int(idx.shape[0])
        t = np.arange(n, dtype=np.float64)
        close = 100.0 + bias + 0.01 * t
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

    def _small_market_frames(self, sessions: int = 5, bars_per_session: int = 12) -> dict[str, "pd.DataFrame"]:
        frames: dict[str, pd.DataFrame] = {}
        for sym, bias in [("S1", 0.0), ("S2", 0.2)]:
            chunks = []
            days = pd.date_range("2024-01-02", periods=sessions, freq="B", tz="America/New_York")
            for d_i, d in enumerate(days):
                start = d.replace(hour=9, minute=30, second=0)
                idx = pd.date_range(start=start, periods=bars_per_session, freq="1min", tz="America/New_York")
                idx = idx.tz_convert("UTC")
                chunks.append(self._frame(idx, bias=bias + 0.05 * d_i))
            frames[sym] = pd.concat(chunks, axis=0)
        return frames

    def _dummy_m3(self, state: h.TensorState) -> Module3Output:
        T, A = state.cfg.T, state.cfg.A
        return Module3Output(
            block_id_t=np.zeros(T, dtype=np.int64),
            block_seq_t=np.zeros(T, dtype=np.int16),
            block_end_flag_t=np.zeros(T, dtype=bool),
            block_start_t_index_t=np.zeros(T, dtype=np.int64),
            block_end_t_index_t=np.zeros(T, dtype=np.int64),
            block_features_tak=np.zeros((T, A, int(Struct30mIdx.N_FIELDS)), dtype=np.float64),
            block_valid_ta=np.asarray(state.bar_valid, dtype=bool).copy(),
            context_tac=np.zeros((T, A, int(ContextIdx.N_FIELDS)), dtype=np.float64),
            context_valid_ta=np.asarray(state.bar_valid, dtype=bool).copy(),
            context_source_t_index_ta=np.full((T, A), -1, dtype=np.int64),
        )

    def _dummy_m4(self, state: h.TensorState) -> Module4Output:
        T, A = state.cfg.T, state.cfg.A
        return Module4Output(
            regime_primary_ta=np.zeros((T, A), dtype=np.int8),
            regime_confidence_ta=np.ones((T, A), dtype=np.float64),
            intent_long_ta=np.zeros((T, A), dtype=np.int8),
            intent_short_ta=np.zeros((T, A), dtype=np.int8),
            target_qty_ta=np.zeros((T, A), dtype=np.float64),
            filled_qty_ta=np.zeros((T, A), dtype=np.float64),
            exec_price_ta=np.zeros((T, A), dtype=np.float64),
            trade_cost_ta=np.zeros((T, A), dtype=np.float64),
            overnight_score_ta=np.zeros((T, A), dtype=np.float64),
            overnight_winner_t=np.full(T, -1, dtype=np.int64),
            kill_switch_t=np.zeros(T, dtype=np.int8),
        )

    def _run_minimal_harness(self, report_dir: Path, harness_overrides: dict[str, object] | None = None) -> h.HarnessOutput:
        frames = self._small_market_frames(sessions=5, bars_per_session=12)

        def loader(path: str, _tz_name: str) -> "pd.DataFrame":
            return frames[path]

        engine_cfg = self._cfg(T=1, A=2)
        m2_configs = [Module2Config(profile_window_bars=3, profile_warmup_bars=3, rvol_lookback_sessions=2)]
        m3_configs = [Module3Config(block_minutes=5, min_block_valid_bars=1, min_block_valid_ratio=0.0)]
        m4_configs = [Module4Config()]
        candidate_specs = [
            h.CandidateSpec(
                candidate_id="cand_single",
                m2_idx=0,
                m3_idx=0,
                m4_idx=0,
                enabled_assets_mask=np.ones(2, dtype=bool),
                tags=(),
            )
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
            cpcv_slices=10,
            cpcv_k_test=5,
            daily_return_min_days=1,
            benchmark_symbol="S1",
            min_asset_coverage=1.0,
            fail_on_non_finite=True,
            seed=13,
        )
        if harness_overrides:
            harness_cfg = dataclasses.replace(harness_cfg, **harness_overrides)

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

    def test_clock_override_first_row_invariants(self) -> None:
        ts_ns = self._utc_minute_ns("2024-01-03T14:30:00", periods=16)
        cfg = self._cfg(T=ts_ns.shape[0], A=2)
        clk = h._build_clock_override_from_utc(ts_ns, cfg, "America/New_York")

        self.assertEqual(int(clk["reset_flag"][0]), 1)
        self.assertEqual(float(clk["gap_min"][0]), 0.0)
        self.assertTrue(np.all(np.isin(clk["reset_flag"], np.array([0, 1], dtype=np.int8))))
        self.assertTrue(np.all(clk["gap_min"] >= 0.0))

        st = preallocate_state(ts_ns=ts_ns, cfg=cfg, symbols=("S1", "S2"), clock_override=clk)
        validate_state_hard(st)

    def test_split_purge_embargo_no_leakage(self) -> None:
        T = 500
        t0 = 200
        test_len = 50
        purge = 10
        embargo = 7

        all_idx = np.arange(T, dtype=np.int64)
        test_idx = np.arange(t0, t0 + test_len, dtype=np.int64)
        train_idx = np.setdiff1d(all_idx, test_idx)

        tr2, te2, purge_idx, embargo_idx = h._apply_purge_embargo(
            train_idx=train_idx,
            test_idx=test_idx,
            T=T,
            purge_bars=purge,
            embargo_bars=embargo,
        )

        spec = h.SplitSpec(
            split_id="wf_000",
            mode="wf",
            train_idx=tr2,
            test_idx=te2,
            purge_idx=purge_idx,
            embargo_idx=embargo_idx,
            session_train_bounds=(0, 0),
            session_test_bounds=(0, 0),
            purge_bars=purge,
            embargo_bars=embargo,
            total_bars=T,
        )
        h._validate_split(spec, enforce_guard=True)

        self.assertEqual(np.intersect1d(spec.train_idx, np.arange(t0 - purge, t0, dtype=np.int64)).size, 0)
        self.assertEqual(np.intersect1d(spec.test_idx, np.arange(t0, t0 + embargo, dtype=np.int64)).size, 0)

        bad_train = np.unique(np.r_[spec.train_idx, np.int64(t0 - 1)])
        bad_spec = dataclasses.replace(spec, train_idx=bad_train)
        with self.assertRaisesRegex(RuntimeError, "first_offending_index"):
            h._validate_split(bad_spec, enforce_guard=True)

    def test_ingest_master_index_monotonic_and_unique(self) -> None:
        engine_cfg = self._cfg(T=1, A=2)
        harness_cfg = h.Module5HarnessConfig(min_asset_coverage=0.5, fail_on_non_finite=False)

        idx_dup = pd.DatetimeIndex(
            [
                "2024-01-03 14:30:00+00:00",
                "2024-01-03 14:31:00+00:00",
                "2024-01-03 14:31:00+00:00",
            ]
        )
        idx_ok = pd.DatetimeIndex(
            [
                "2024-01-03 14:30:00+00:00",
                "2024-01-03 14:31:00+00:00",
                "2024-01-03 14:32:00+00:00",
            ]
        )
        dup_frames = {"S1": self._frame(idx_dup, bias=0.0), "S2": self._frame(idx_ok, bias=0.1)}

        def dup_loader(path: str, _tz_name: str) -> "pd.DataFrame":
            return dup_frames[path]

        _state_d, _keep_idx_d, _keep_symbols_d, master_ts_ns_d, _ingest_meta_d, _tick_d, dq_d = h._ingest_master_aligned(
            data_paths=["S1", "S2"],
            symbols=["S1", "S2"],
            engine_cfg=engine_cfg,
            harness_cfg=harness_cfg,
            data_loader_func=dup_loader,
        )
        self.assertTrue(np.all(np.diff(master_ts_ns_d) > 0))
        reasons_d = "|".join(str(r.get("reason_codes", "")) for r in dq_d.get("day_reports", []))
        self.assertIn("DUPLICATE_TIMESTAMP", reasons_d)

        idx_ooo = pd.DatetimeIndex(
            [
                "2024-01-03 14:30:00+00:00",
                "2024-01-03 14:32:00+00:00",
                "2024-01-03 14:31:00+00:00",
            ]
        )
        ooo_frames = {"S1": self._frame(idx_ooo, bias=0.0), "S2": self._frame(idx_ok, bias=0.1)}

        def ooo_loader(path: str, _tz_name: str) -> "pd.DataFrame":
            return ooo_frames[path]

        _state_o, _keep_idx_o, _keep_symbols_o, master_ts_ns_o, _ingest_meta_o, _tick_o, dq_o = h._ingest_master_aligned(
            data_paths=["S1", "S2"],
            symbols=["S1", "S2"],
            engine_cfg=engine_cfg,
            harness_cfg=harness_cfg,
            data_loader_func=ooo_loader,
        )
        self.assertTrue(np.all(np.diff(master_ts_ns_o) > 0))
        reasons_o = "|".join(str(r.get("reason_codes", "")) for r in dq_o.get("day_reports", []))
        self.assertIn("NON_MONOTONIC_TIMESTAMP", reasons_o)

        idx_local = pd.date_range(
            "2024-01-03 09:30:00",
            periods=4,
            freq="1min",
            tz="America/New_York",
        )
        valid_frames = {"S1": self._frame(idx_local, bias=0.0), "S2": self._frame(idx_local, bias=0.1)}

        def ok_loader(path: str, _tz_name: str) -> "pd.DataFrame":
            return valid_frames[path]

        state, _keep_idx, _keep_symbols, master_ts_ns, _ingest_meta, _tick, _dq_bundle = h._ingest_master_aligned(
            data_paths=["S1", "S2"],
            symbols=["S1", "S2"],
            engine_cfg=engine_cfg,
            harness_cfg=harness_cfg,
            data_loader_func=ok_loader,
        )
        self.assertTrue(np.all(np.diff(master_ts_ns) > 0))
        self.assertTrue(np.all(np.diff(state.ts_ns) > 0))

    def test_load_asset_frame_accepts_datetime_index_without_timestamp_column(self) -> None:
        idx = pd.date_range(
            "2024-01-03 09:31:00",
            periods=4,
            freq="1min",
            tz="America/New_York",
        ).tz_convert("UTC")
        df = pd.DataFrame(
            {
                "open": [100.0, 100.1, 100.2, 100.3],
                "high": [100.2, 100.3, 100.4, 100.5],
                "low": [99.9, 100.0, 100.1, 100.2],
                "close": [100.1, 100.2, 100.3, 100.4],
                "volume": [1000.0, 1001.0, 1002.0, 1003.0],
            },
            index=idx,
        )

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "asset.parquet"
            df.to_parquet(p)
            out = h._load_asset_frame(str(p), "America/New_York")

        self.assertIsInstance(out.index, pd.DatetimeIndex)
        self.assertIsNotNone(out.index.tz)
        self.assertEqual(int(out.shape[0]), 4)
        self.assertEqual(list(out.columns), ["open", "high", "low", "close", "volume"])

    def test_orchestration_order_placeholders_after_perturbations(self) -> None:
        T, A = 16, 2
        ts_ns = self._utc_minute_ns("2024-01-03T14:30:00", periods=T)
        cfg = self._cfg(T=T, A=A)
        base_state = preallocate_state(ts_ns=ts_ns, cfg=cfg, symbols=("S1", "S2"))

        t_grid = np.arange(T, dtype=np.float64)[:, None]
        a_grid = np.arange(A, dtype=np.float64)[None, :]
        close = 100.0 + 0.01 * t_grid + 0.05 * a_grid
        open_px = close - 0.01
        base_state.open_px[:, :] = open_px
        base_state.close_px[:, :] = close
        base_state.high_px[:, :] = np.maximum(open_px, close) + 0.02
        base_state.low_px[:, :] = np.minimum(open_px, close) - 0.02
        base_state.volume[:, :] = 1000.0
        base_state.bar_valid[:, :] = True
        h._set_placeholders_from_bar_valid(base_state)

        split = h.SplitSpec(
            split_id="wf_000",
            mode="wf",
            train_idx=np.arange(0, 10, dtype=np.int64),
            test_idx=np.arange(10, 16, dtype=np.int64),
            purge_idx=np.zeros(0, dtype=np.int64),
            embargo_idx=np.zeros(0, dtype=np.int64),
            session_train_bounds=(0, 0),
            session_test_bounds=(0, 0),
            purge_bars=0,
            embargo_bars=0,
            total_bars=T,
        )
        scenario = h.StressScenario(
            scenario_id="stress",
            name="stress",
            missing_burst_prob=0.25,
            missing_burst_min=1,
            missing_burst_max=2,
            jitter_sigma_bps=2.0,
            slippage_mult=1.0,
            enabled=True,
        )
        candidate = h.CandidateSpec(
            candidate_id="cand0",
            m2_idx=0,
            m3_idx=0,
            m4_idx=0,
            enabled_assets_mask=np.ones(A, dtype=bool),
            tags=(),
        )
        group = h._GroupTask(
            group_id="g0",
            split_idx=0,
            scenario_idx=0,
            m2_idx=0,
            m3_idx=0,
            candidate_indices=(0,),
        )
        harness_cfg = h.Module5HarnessConfig(seed=41, fail_on_non_finite=True)
        m2_cfgs = [Module2Config(profile_window_bars=3, profile_warmup_bars=3)]
        m3_cfgs = [Module3Config(block_minutes=5, min_block_valid_bars=1, min_block_valid_ratio=0.0)]
        m4_cfgs = [Module4Config()]

        call_order: list[str] = []
        orig_split = h._apply_split_domain_mask
        orig_missing = h._apply_missing_bursts
        orig_jitter = h._apply_jitter
        orig_recompute = h._recompute_bar_valid_inplace
        orig_validate = h.validate_loaded_market_slice

        def wrap_split(*args, **kwargs):
            call_order.append("split_mask")
            return orig_split(*args, **kwargs)

        def wrap_missing(*args, **kwargs):
            call_order.append("missing")
            return orig_missing(*args, **kwargs)

        def wrap_jitter(*args, **kwargs):
            call_order.append("jitter")
            return orig_jitter(*args, **kwargs)

        def wrap_recompute(*args, **kwargs):
            call_order.append("recompute")
            return orig_recompute(*args, **kwargs)

        def wrap_validate(*args, **kwargs):
            call_order.append("validate")
            state = args[0]
            valid = np.asarray(state.bar_valid, dtype=bool)
            self.assertTrue(np.all(np.isfinite(state.rvol[valid])))
            self.assertTrue(np.all(np.isfinite(state.atr_floor[valid])))
            invalid = ~valid
            if np.any(invalid):
                self.assertTrue(np.all(~np.isfinite(state.rvol[invalid])))
                self.assertTrue(np.all(~np.isfinite(state.atr_floor[invalid])))
            return orig_validate(*args, **kwargs)

        def wrap_m2(state: h.TensorState, _cfg: Module2Config) -> None:
            call_order.append("module2")
            return None

        def wrap_m3(state: h.TensorState, _cfg: Module3Config) -> Module3Output:
            call_order.append("module3")
            return self._dummy_m3(state)

        def wrap_m4(state: h.TensorState, _m3: Module3Output, _cfg: Module4Config) -> Module4Output:
            return self._dummy_m4(state)

        with (
            mock.patch.object(h, "_apply_split_domain_mask", side_effect=wrap_split),
            mock.patch.object(h, "_apply_missing_bursts", side_effect=wrap_missing),
            mock.patch.object(h, "_apply_jitter", side_effect=wrap_jitter),
            mock.patch.object(h, "_recompute_bar_valid_inplace", side_effect=wrap_recompute),
            mock.patch.object(h, "validate_loaded_market_slice", side_effect=wrap_validate),
            mock.patch.object(h, "run_weightiz_profile_engine", side_effect=wrap_m2),
            mock.patch.object(h, "run_module3_structural_aggregation", side_effect=wrap_m3),
            mock.patch.object(h, "run_module4_strategy_funnel", side_effect=wrap_m4),
        ):
            rows = h._run_group_task(
                group=group,
                base_state=base_state,
                candidates=[candidate],
                splits=[split],
                scenarios=[scenario],
                m2_configs=m2_cfgs,
                m3_configs=m3_cfgs,
                m4_configs=m4_cfgs,
                harness_cfg=harness_cfg,
            )

        self.assertEqual(len(rows), 1)
        expected = ["split_mask", "missing", "jitter", "recompute", "validate", "module2", "module3"]
        self.assertEqual(call_order[: len(expected)], expected)

    def test_base_state_immutability(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m5_base_state_immut_") as td:
            report_dir = Path(td) / "artifacts"
            captured: dict[str, object] = {}
            orig_ingest = h._ingest_master_aligned

            def wrap_ingest(*args, **kwargs):
                out = orig_ingest(*args, **kwargs)
                st = out[0]
                captured["state"] = st
                captured["digest_before"] = deterministic_digest_sha256(st)
                return out

            with mock.patch.object(h, "_ingest_master_aligned", side_effect=wrap_ingest):
                _ = self._run_minimal_harness(report_dir=report_dir)

            self.assertIn("state", captured)
            before = str(captured["digest_before"])
            after = deterministic_digest_sha256(captured["state"])  # type: ignore[arg-type]
            self.assertEqual(before, after)

    def test_process_pool_fallback_determinism(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m5_payload_fallback_") as td:
            report_dir = Path(td) / "artifacts"
            out = self._run_minimal_harness(
                report_dir=report_dir,
                harness_overrides={
                    "parallel_backend": "process_pool",
                    "parallel_workers": 2,
                    "payload_pickle_threshold_bytes": 1,
                },
            )

            self.assertEqual(str(out.run_manifest.get("execution_mode")), "serial_forced_payload")
            self.assertFalse(bool(out.run_manifest.get("payload_safe", True)))
            self.assertEqual(int(out.run_manifest.get("parallel_workers_effective", 0)), 1)

            self.assertIn("run_status", out.artifact_paths)
            run_status_path = Path(str(out.artifact_paths["run_status"]))
            self.assertTrue(run_status_path.exists())
            run_status = json.loads(run_status_path.read_text(encoding="utf-8"))
            self.assertIn("first_exception", run_status)
            self.assertIn("class", run_status["first_exception"])
            self.assertIn("message", run_status["first_exception"])
            self.assertIn("error_hash", run_status["first_exception"])

    @unittest.skipIf(os.name == "nt", "process_pool fork context test is POSIX-only")
    def test_process_pool_executes_tasks_and_updates_status(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m5_process_pool_progress_") as td:
            report_dir = Path(td) / "artifacts"
            os.environ["WEIGHTIZ_MP_START_METHOD"] = "fork"
            try:
                out = self._run_minimal_harness(
                    report_dir=report_dir,
                    harness_overrides={
                        "parallel_backend": "process_pool",
                        "parallel_workers": 2,
                        "payload_pickle_threshold_bytes": 1024 * 1024,
                    },
                )
            finally:
                os.environ.pop("WEIGHTIZ_MP_START_METHOD", None)

            self.assertEqual(str(out.run_manifest.get("execution_mode")), "process_pool")
            run_status_path = Path(str(out.artifact_paths["run_status"]))
            run_status = json.loads(run_status_path.read_text(encoding="utf-8"))
            self.assertGreater(int(run_status.get("tasks_completed", 0)), 0)
            self.assertEqual(
                int(run_status.get("tasks_completed", 0)),
                int(run_status.get("tasks_submitted", 0)),
            )
            self.assertIn(
                str(run_status.get("process_start_method", "")),
                {"fork", "spawn", "forkserver"},
            )

    def test_aggregate_candidate_baseline_matrix_keeps_zero_return_days(self) -> None:
        bench_sessions = np.asarray([101, 102, 103, 104, 105], dtype=np.int64)
        bench_ret = np.asarray([0.0, 0.001, -0.002, 0.0, 0.003], dtype=np.float64)
        results_ok = [
            {
                "candidate_id": "cand_a",
                "scenario_id": "baseline",
                "session_ids": np.asarray([102, 104], dtype=np.int64),
                "daily_returns": np.asarray([0.01, -0.02], dtype=np.float64),
                "status": "ok",
                "test_days": 2,
            }
        ]
        common, mat, _bmk, baseline_ids, _series = h._aggregate_candidate_baseline_matrix(
            results_ok=results_ok,
            bench_sessions=bench_sessions,
            bench_ret=bench_ret,
            candidate_ids=["cand_a"],
            min_days=3,
        )
        self.assertEqual(common.tolist(), [101, 102, 103, 104, 105])
        self.assertEqual(baseline_ids, ["cand_a"])
        self.assertEqual(mat.shape, (5, 1))
        self.assertAlmostEqual(float(mat[0, 0]), 0.0, places=12)
        self.assertAlmostEqual(float(mat[1, 0]), 0.01, places=12)
        self.assertAlmostEqual(float(mat[2, 0]), 0.0, places=12)
        self.assertAlmostEqual(float(mat[3, 0]), -0.02, places=12)
        self.assertAlmostEqual(float(mat[4, 0]), 0.0, places=12)

    def test_split_group_tasks_by_candidate_deterministic(self) -> None:
        candidates = [
            h.CandidateSpec(
                candidate_id=f"c{i}",
                m2_idx=0,
                m3_idx=0,
                m4_idx=i,
                enabled_assets_mask=np.ones(2, dtype=bool),
                tags=(),
            )
            for i in range(4)
        ]
        splits = [
            h.SplitSpec(
                split_id="wf_000",
                mode="wf",
                train_idx=np.arange(0, 10, dtype=np.int64),
                test_idx=np.arange(10, 15, dtype=np.int64),
                purge_idx=np.zeros(0, dtype=np.int64),
                embargo_idx=np.zeros(0, dtype=np.int64),
                session_train_bounds=(0, 0),
                session_test_bounds=(0, 0),
                purge_bars=0,
                embargo_bars=0,
                total_bars=20,
            )
        ]
        scenarios = [
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
        grouped = h._build_group_tasks(candidates, splits, scenarios)
        self.assertEqual(len(grouped), 1)
        self.assertEqual(len(grouped[0].candidate_indices), 4)

        split_groups = h._split_group_tasks_by_candidate(grouped, chunk_size=1)
        self.assertEqual(len(split_groups), 4)
        self.assertEqual([g.candidate_indices for g in split_groups], [(0,), (1,), (2,), (3,)])
        self.assertEqual([g.group_id for g in split_groups], [f"{grouped[0].group_id}_p{i:03d}" for i in range(4)])

    def test_minimal_harness_has_finite_robustness_for_non_pathological_candidate(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m5_finite_robustness_") as td:
            report_dir = Path(td) / "artifacts"
            out = self._run_minimal_harness(
                report_dir=report_dir,
                harness_overrides={"daily_return_min_days": 3},
            )
            rb_path = Path(str(out.artifact_paths["robustness_leaderboard_csv"]))
            self.assertTrue(rb_path.exists())
            rb = pd.read_csv(rb_path)
            self.assertGreaterEqual(int(rb.shape[0]), 1)
            scores = pd.to_numeric(rb["robustness_score"], errors="coerce")
            self.assertTrue(bool(np.isfinite(scores).any()))

    def test_validation_report_artifact_and_extended_stats_schema(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m5_validation_report_") as td:
            report_dir = Path(td) / "artifacts"
            with mock.patch.object(h, "compute_window_correlation_diagnostics", return_value=([], [])):
                out = self._run_minimal_harness(
                    report_dir=report_dir,
                    harness_overrides={"daily_return_min_days": 3, "execution_latency_bars": 0},
                )
            self.assertIn("validation_report", out.artifact_paths)
            p = Path(str(out.artifact_paths["validation_report"]))
            self.assertTrue(p.exists())
            rows = json.loads(p.read_text(encoding="utf-8"))
            self.assertIsInstance(rows, list)
            self.assertGreaterEqual(len(rows), 1)
            row = dict(rows[0])
            for k in (
                "strategy_id",
                "cluster_id",
                "cluster_representative",
                "dsr",
                "pbo",
                "spa_p",
                "mcs_inclusion",
                "regime_robustness",
                "execution_robustness",
                "horizon_robustness",
                "robustness_score",
                "reject",
                "fragile",
            ):
                self.assertIn(k, row)

            stats_raw = json.loads(Path(str(out.artifact_paths["stats_raw"])).read_text(encoding="utf-8"))
            self.assertIn("cluster", stats_raw)
            self.assertIn("regime_validation", stats_raw)
            self.assertIn("horizon_validation", stats_raw)
            self.assertIn("execution_validation", stats_raw)

    def test_validation_report_is_deterministic_for_same_seed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m5_validation_det_1_") as td1, tempfile.TemporaryDirectory(
            prefix="m5_validation_det_2_"
        ) as td2:
            with mock.patch.object(h, "compute_window_correlation_diagnostics", return_value=([], [])):
                out1 = self._run_minimal_harness(
                    report_dir=Path(td1) / "artifacts",
                    harness_overrides={"daily_return_min_days": 3, "seed": 17, "execution_latency_bars": 0},
                )
            with mock.patch.object(h, "compute_window_correlation_diagnostics", return_value=([], [])):
                out2 = self._run_minimal_harness(
                    report_dir=Path(td2) / "artifacts",
                    harness_overrides={"daily_return_min_days": 3, "seed": 17, "execution_latency_bars": 0},
                )
            b1 = Path(str(out1.artifact_paths["validation_report"])).read_bytes()
            b2 = Path(str(out2.artifact_paths["validation_report"])).read_bytes()
            self.assertEqual(b1, b2)

    def test_clean_fixture_emits_no_runtime_warnings(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m5_warning_regression_") as td:
            report_dir = Path(td) / "artifacts"
            with warnings.catch_warnings(record=True) as wrn:
                warnings.simplefilter("always", RuntimeWarning)
                _ = self._run_minimal_harness(
                    report_dir=report_dir,
                    harness_overrides={"daily_return_min_days": 3},
                )
            runtime_warnings = [w for w in wrn if issubclass(w.category, RuntimeWarning)]
            self.assertEqual(
                len(runtime_warnings),
                0,
                msg=f"unexpected runtime warnings: {[str(w.message) for w in runtime_warnings[:5]]}",
            )


if __name__ == "__main__":
    unittest.main()
