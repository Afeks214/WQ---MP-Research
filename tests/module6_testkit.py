from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from weightiz.module6.config import GeneratorConfig, IntakeConfig, Module6Config, ScoringConfig
from weightiz.module6.orchestrator import run_module6_portfolio_research
from weightiz.module5.harness.module6_bridge import (
    AVAIL_OBSERVED_ACTIVE,
    AVAIL_OBSERVED_FLAT,
    AVAIL_STRUCTURALLY_MISSING,
    build_module6_bridge_artifacts,
)


class DummyEngineConfig:
    def __init__(self, initial_cash: float = 1_000_000.0) -> None:
        self.initial_cash = float(initial_cash)


class DummyCandidate:
    def __init__(self, candidate_id: str, m2_idx: int, m3_idx: int, m4_idx: int) -> None:
        self.candidate_id = candidate_id
        self.m2_idx = int(m2_idx)
        self.m3_idx = int(m3_idx)
        self.m4_idx = int(m4_idx)
        self.enabled_assets_mask = [True, True]
        self.tags = [f"family:{m4_idx}"]


def _candidate_specs() -> list[dict[str, Any]]:
    return [
        {"candidate_id": "cand_000", "family_id": "dup", "hypothesis_id": "h0", "base": 0.0010, "missing_mod": None},
        {"candidate_id": "cand_001", "family_id": "dup", "hypothesis_id": "h0", "base": 0.0010, "missing_mod": None},
        {"candidate_id": "cand_002", "family_id": "hedge", "hypothesis_id": "h1", "base": -0.0007, "missing_mod": None},
        {"candidate_id": "cand_003", "family_id": "alt", "hypothesis_id": "h2", "base": 0.0006, "missing_mod": None},
        {"candidate_id": "cand_004", "family_id": "fragile", "hypothesis_id": "h3", "base": 0.0002, "missing_mod": 5},
        {"candidate_id": "cand_005", "family_id": "alt", "hypothesis_id": "h4", "base": 0.0015, "missing_mod": None},
    ]


def make_test_config() -> Module6Config:
    return Module6Config(
        intake=IntakeConfig(min_availability_ratio=0.75, min_observed_sessions=20, require_bridge_artifacts=True, required_comparison_support=0.70),
        generator=GeneratorConfig(
            random_sparse_quota=16,
            cluster_balanced_quota=8,
            hrp_variant_quota=9,
            mv_variant_quota=1,
            random_sparse_batch_size=8,
            active_cardinality_choices=(2, 3, 4),
            enable_mv_diagnostic=False,
        ),
        scoring=ScoringConfig(
            shortlist_session_keep=12,
            shortlist_minute_keep=6,
            final_scalar_keep=6,
            final_primary_count=2,
            final_alternate_count=2,
            min_cross_universe_support=0.70,
        ),
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_synthetic_module5_run(tmp_path: Path, *, n_sessions: int = 64) -> Path:
    run_dir = tmp_path / "synthetic_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = run_dir / "candidates"
    candidates_dir.mkdir(exist_ok=True)
    sessions = np.arange(int(n_sessions), dtype=np.int64)
    session_dates = pd.bdate_range("2024-01-01", periods=int(n_sessions), tz="UTC")
    benchmark = 0.0004 * np.sin(np.linspace(0.0, 8.0, int(n_sessions)))
    candidate_rows: list[dict[str, Any]] = []
    robustness_rows: list[dict[str, Any]] = []
    strategy_rows: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    daily_df = pd.DataFrame({"session_id": sessions, "benchmark": benchmark})
    candidates = [DummyCandidate(spec["candidate_id"], idx, idx % 2, idx % 3) for idx, spec in enumerate(_candidate_specs())]
    for idx, spec in enumerate(_candidate_specs()):
        candidate_id = spec["candidate_id"]
        base = float(spec["base"])
        split_series: dict[str, np.ndarray] = {}
        for split_idx, split_id in enumerate(("wf_000", "wf_001")):
            ret = base + 0.0003 * np.sin(np.linspace(0.0, 6.0, int(n_sessions)) + split_idx)
            if candidate_id == "cand_001":
                ret = ret + 1.0e-5
            if candidate_id == "cand_002":
                ret = -0.8 * (0.0010 + 0.0003 * np.sin(np.linspace(0.0, 6.0, int(n_sessions)) + split_idx))
            if candidate_id == "cand_004":
                ret = ret.copy()
                mask = (sessions % int(spec["missing_mod"])) == 0
                ret[mask] = 0.0
            raw = ret + 5.0e-5
            observed_mask = np.ones(int(n_sessions), dtype=bool)
            if candidate_id == "cand_004":
                observed_mask[(sessions % int(spec["missing_mod"])) == 0] = False
            availability_codes = np.where(
                observed_mask,
                np.where(np.abs(ret) > 1.0e-12, AVAIL_OBSERVED_ACTIVE, AVAIL_OBSERVED_FLAT),
                AVAIL_STRUCTURALLY_MISSING,
            )
            trade_payload = {
                "ts_ns": np.asarray(
                    [
                        int((session_dates[int(s)] + pd.Timedelta(hours=20)).value)
                        for s in sessions[observed_mask]
                    ],
                    dtype=np.int64,
                ),
                "session_id": sessions[observed_mask].astype(np.int64),
                "filled_qty": np.asarray([10.0] * int(np.sum(observed_mask)), dtype=np.float64),
                "exec_price": np.asarray([100.0 + idx] * int(np.sum(observed_mask)), dtype=np.float64),
            }
            ts_ns: list[int] = []
            eq_vals: list[float] = []
            margin_vals: list[float] = []
            buying_power_vals: list[float] = []
            daily_loss_vals: list[float] = []
            session_ids: list[int] = []
            micro_rows: list[dict[str, Any]] = []
            equity = 1_000_000.0
            for session_id, session_ret, observed in zip(sessions.tolist(), ret.tolist(), observed_mask.tolist()):
                if observed:
                    step = np.power(1.0 + float(session_ret), 0.25)
                    for minute in range(4):
                        ts = int((session_dates[int(session_id)] + pd.Timedelta(hours=14, minutes=30 + minute)).value)
                        equity *= step
                        ts_ns.append(ts)
                        session_ids.append(int(session_id))
                        eq_vals.append(float(equity))
                        margin_vals.append(float(abs(session_ret) * 1000000.0))
                        buying_power_vals.append(float(max(0.0, equity - abs(session_ret) * 1000000.0)))
                        daily_loss_vals.append(float(max(0.0, 1_000_000.0 - equity)))
                        micro_rows.append(
                            {
                                "ts_ns": ts,
                                "session_id": int(session_id),
                                "candidate_id": candidate_id,
                                "split_id": split_id,
                                "scenario_id": "baseline",
                                "symbol": "SYM0",
                                "filled_qty": 10.0 if minute == 0 else 0.0,
                                "exec_price": float(100.0 + idx),
                                "trade_cost": 0.0,
                                "overnight_winner_flag": int(session_id % 7 == 0),
                            }
                        )
            equity_payload = {
                "ts_ns": np.asarray(ts_ns, dtype=np.int64),
                "session_id": np.asarray(session_ids, dtype=np.int64),
                "equity": np.asarray(eq_vals, dtype=np.float64),
                "margin_used": np.asarray(margin_vals, dtype=np.float64),
                "buying_power": np.asarray(buying_power_vals, dtype=np.float64),
                "daily_loss": np.asarray(daily_loss_vals, dtype=np.float64),
            }
            row = {
                "task_id": f"{candidate_id}_{split_id}_baseline",
                "candidate_id": candidate_id,
                "split_id": split_id,
                "scenario_id": "baseline",
                "status": "ok",
                "session_ids_exec": sessions[observed_mask].copy(),
                "session_ids_raw": sessions[observed_mask].copy(),
                "daily_returns_exec": ret[observed_mask].copy(),
                "daily_returns_raw": raw[observed_mask].copy(),
                "daily_returns": ret[observed_mask].copy(),
                "test_days": int(np.sum(observed_mask)),
                "m2_idx": idx,
                "m3_idx": idx % 2,
                "m4_idx": idx % 3,
                "tags": [f"family:{spec['family_id']}"],
                "quality_reason_codes": [],
                "dq_invalidated": False,
                "availability_state_session_ids": sessions[observed_mask].copy(),
                "availability_state_codes": availability_codes[observed_mask].astype(np.int16),
                "equity_payload": equity_payload,
                "trade_payload": trade_payload,
                "micro_payload": {
                    "session_id": np.asarray([row["session_id"] for row in micro_rows], dtype=np.int64),
                    "overnight_winner_flag": np.asarray([row["overnight_winner_flag"] for row in micro_rows], dtype=np.int8),
                },
            }
            all_results.append(row)
            split_series[split_id] = ret.copy()
            eq_df = pd.DataFrame(
                {
                    "ts_ns": equity_payload["ts_ns"],
                    "session_id": equity_payload["session_id"],
                    "candidate_id": candidate_id,
                    "split_id": split_id,
                    "scenario_id": "baseline",
                    "equity": equity_payload["equity"],
                    "drawdown": 0.0,
                    "margin_used": equity_payload["margin_used"],
                    "buying_power": equity_payload["buying_power"],
                    "daily_loss": equity_payload["daily_loss"],
                }
            )
            trade_df = pd.DataFrame(
                {
                    "ts_ns": trade_payload["ts_ns"],
                    "session_id": trade_payload["session_id"],
                    "candidate_id": candidate_id,
                    "split_id": split_id,
                    "scenario_id": "baseline",
                    "symbol": "SYM0",
                    "filled_qty": trade_payload["filled_qty"],
                    "exec_price": trade_payload["exec_price"],
                    "trade_cost": np.zeros(trade_payload["ts_ns"].shape[0], dtype=np.float64),
                    "order_side": np.ones(trade_payload["ts_ns"].shape[0], dtype=np.int8),
                    "order_flags": np.zeros(trade_payload["ts_ns"].shape[0], dtype=np.uint16),
                }
            )
            if (run_dir / "equity_curves.parquet").exists():
                pd.concat([pd.read_parquet(run_dir / "equity_curves.parquet"), eq_df], ignore_index=True).to_parquet(run_dir / "equity_curves.parquet", index=False)
                pd.concat([pd.read_parquet(run_dir / "trade_log.parquet"), trade_df], ignore_index=True).to_parquet(run_dir / "trade_log.parquet", index=False)
                pd.concat([pd.read_parquet(run_dir / "micro_diagnostics.parquet"), pd.DataFrame(micro_rows)], ignore_index=True).to_parquet(run_dir / "micro_diagnostics.parquet", index=False)
            else:
                eq_df.to_parquet(run_dir / "equity_curves.parquet", index=False)
                trade_df.to_parquet(run_dir / "trade_log.parquet", index=False)
                pd.DataFrame(micro_rows).to_parquet(run_dir / "micro_diagnostics.parquet", index=False)
        aggregate = np.median(np.vstack([split_series["wf_000"], split_series["wf_001"]]), axis=0)
        daily_df[candidate_id] = aggregate
        cdir = candidates_dir / candidate_id
        cdir.mkdir(exist_ok=True)
        _write_json(
            cdir / "candidate_config.json",
            {
                "candidate_id": candidate_id,
                "run_id": "synthetic_run",
                "m2_idx": idx,
                "m3_idx": idx % 2,
                "m4_idx": idx % 3,
                "enabled_assets_mask": [True, True],
                "stage_a_metadata": {
                    "campaign_id": "cmp0",
                    "family_id": spec["family_id"],
                    "family_name": spec["family_id"],
                    "hypothesis_id": spec["hypothesis_id"],
                    "evaluation_role": "multi_window_live",
                    "evaluation_window": 30,
                    "window_set": "15,30",
                    "window_set_size": 2,
                    "parameter_hash": f"param_{candidate_id}",
                    "tags_serialized": "",
                },
                "engine_config": {"initial_cash": 1_000_000.0, "intraday_leverage_max": 6.0, "overnight_leverage": 2.0},
            },
        )
        _write_json(
            cdir / "candidate_metrics.json",
            {
                "candidate_id": candidate_id,
                "base_metrics": {
                    "n_days": int(n_sessions),
                    "n_trades": int(n_sessions * 2),
                    "avg_turnover": 0.02 if candidate_id != "cand_004" else 0.005,
                    "avg_margin_used_frac": 0.10,
                    "peak_margin_used_frac": 0.50,
                    "max_drawdown": 0.05,
                    "asset_pnl_concentration": 0.30,
                },
                "robustness": {"score": 0.7 - 0.05 * idx},
                "failed": False,
                "failure_reasons": [],
                "alignment": {"aligned_to_global_benchmark_sessions": True, "global_session_count": int(n_sessions), "observed_baseline_session_count": int(n_sessions)},
                "dq_summary": {"dq_min": 1.0, "dq_median": 1.0, "dq_degrade_count": 0, "dq_reject_count": 0},
                "per_fold": {
                    "wf": {
                        "summary": {"count": 2},
                        "folds": [
                            {"split_id": "wf_000", "scenario_id": "baseline", "cum_return": float(np.sum(split_series["wf_000"])), "turnover": 0.02, "max_drawdown": 0.05, "sharpe_daily": 1.0, "test_days": int(n_sessions)},
                            {"split_id": "wf_001", "scenario_id": "baseline", "cum_return": float(np.sum(split_series["wf_001"])), "turnover": 0.02, "max_drawdown": 0.05, "sharpe_daily": 1.0, "test_days": int(n_sessions)},
                        ],
                    },
                    "cpcv": {"summary": {"count": 0}, "folds": []},
                },
                "per_stress": {"baseline": {"n_tasks": 2, "cum_return_mean": float(np.sum(aggregate)), "cum_return_median": float(np.sum(aggregate)), "max_drawdown_median": 0.05, "turnover_median": 0.02}},
            },
        )
        _write_json(cdir / "candidate_stats.json", {"dsr": {"dsr": 0.6 - 0.03 * idx}, "pbo": {"pbo": 0.2 + 0.05 * idx}})
        pd.DataFrame({"session_id": sessions, "returns": aggregate, "is_observed_baseline": np.ones(int(n_sessions), dtype=np.int8)}).to_parquet(cdir / "candidate_returns.parquet", index=False)
        pd.DataFrame({"session_id": sessions, "losses": -aggregate}).to_parquet(cdir / "candidate_losses.parquet", index=False)
        candidate_rows.append(
            {
                "candidate_id": candidate_id,
                "m2_idx": idx,
                "m3_idx": idx % 2,
                "m4_idx": idx % 3,
                "n_tasks": 2,
                "n_tasks_baseline": 2,
                "n_days": int(n_sessions),
                "n_days_observed_baseline": int(n_sessions if spec["missing_mod"] is None else n_sessions - n_sessions // spec["missing_mod"]),
                "cum_return": float(np.sum(aggregate)),
                "max_drawdown": 0.05,
                "dsr_full": 0.6 - 0.03 * idx,
                "dsr_median": 0.6 - 0.03 * idx,
                "pbo": 0.2 + 0.05 * idx,
                "fold_sharpe_std": 0.1,
                "asset_pnl_concentration": 0.30,
                "robustness_score": 0.7 - 0.05 * idx,
                "cluster_id": 0 if idx < 2 else idx,
                "cluster_representative": "cand_000" if idx < 2 else candidate_id,
                "regime_robustness": 0.5,
                "execution_robustness": 0.5,
                "horizon_robustness": 0.5,
                "research_mode": "standard",
                "standard_reject": False,
                "standard_pass": True,
                "discovery_included": False,
                "fragile": candidate_id == "cand_004",
                "reject": False,
                "failed": False,
                "failure_reasons": "",
                "dq_min": 1.0,
                "dq_median": 1.0,
                "dq_degrade_count": 0,
                "dq_reject_count": 0,
                "dq_reason_top": "",
                "cost_adjusted_expectancy": float(np.mean(aggregate)),
                "overnight_suitability_score": 0.2,
                "zimtra_compliance_flags": "",
                "in_mcs": True,
                "pass": True,
                "wrc_p": 0.05,
                "spa_p": 0.05,
                "campaign_id": "cmp0",
                "family_id": spec["family_id"],
                "family_name": spec["family_id"],
                "hypothesis_id": spec["hypothesis_id"],
                "evaluation_role": "multi_window_live",
                "evaluation_window": 30,
                "window_set": "15,30",
                "window_set_size": 2,
                "parameter_hash": f"param_{candidate_id}",
                "tags_serialized": "",
            }
        )
        robustness_rows.append({**candidate_rows[-1], "plateau_id": f"plateau_{idx:03d}"})
        strategy_rows.append(
            {
                "strategy_id": candidate_id,
                "strategy_hash": f"hash_{candidate_id}",
                "parameter_values": json.dumps({"candidate_id": candidate_id}),
                "parameter_hash": f"param_{candidate_id}",
                "asset_count": 2,
                "total_trades": int(n_sessions * 2),
                "sharpe": 1.0,
                "sortino": 1.0,
                "max_drawdown": 0.05,
                "final_equity": 1_010_000.0,
                "family_id": spec["family_id"],
                "family_name": spec["family_id"],
                "hypothesis_id": spec["hypothesis_id"],
                "evaluation_role": "multi_window_live",
                "evaluation_window": 30,
                "window_set": "15,30",
                "overnight_suitability_score": 0.2,
                "zimtra_compliance_flags": "",
                "tags": "",
                "evaluation_timestamp": "2026-03-11T00:00:00+00:00",
            }
        )
    pd.DataFrame(candidate_rows).sort_values("candidate_id").to_csv(run_dir / "leaderboard.csv", index=False)
    pd.DataFrame(robustness_rows).sort_values("candidate_id").to_csv(run_dir / "robustness_leaderboard.csv", index=False)
    pd.DataFrame(strategy_rows).sort_values("strategy_id").to_parquet(run_dir / "strategy_results.parquet", index=False)
    daily_df.to_parquet(run_dir / "daily_returns.parquet", index=False)
    bridge_paths, bridge_summary = build_module6_bridge_artifacts(
        report_root=run_dir,
        run_id="synthetic_run",
        execution_mode="process_pool",
        common_sessions=sessions,
        canonical_reference_split_id="wf_000",
        canonical_reference_scenario_id="baseline",
        canonical_reference_policy="synthetic_test_fixture_v1",
        baseline_candidate_ids=[spec["candidate_id"] for spec in _candidate_specs()],
        candidate_daily_mat=np.column_stack([daily_df[spec["candidate_id"]].to_numpy(dtype=np.float64) for spec in _candidate_specs()]),
        candidates=candidates,
        candidate_rows=candidate_rows,
        all_results=all_results,
        engine_cfg=DummyEngineConfig(),
        require_pandas_fn=lambda: pd,
    )
    _write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": "synthetic_run",
            "dataset_hash": "dataset_synth_hash",
            "daily_matrix_shape": [int(n_sessions), len(_candidate_specs())],
            "daily_matrix_shape_raw": [int(n_sessions), len(_candidate_specs())],
            "execution_mode": "process_pool",
            "module6_bridge": bridge_summary,
        },
    )
    return run_dir


def run_module6_on_synthetic(tmp_path: Path, *, config: Module6Config | None = None) -> Any:
    cfg = config if config is not None else make_test_config()
    run_dir = build_synthetic_module5_run(tmp_path)
    return run_module6_portfolio_research(run_dir, output_dir=run_dir / "module6_out", config=cfg)
