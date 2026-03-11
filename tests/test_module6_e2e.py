from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_research
from tests.module6_testkit import make_test_config, run_module6_on_synthetic


def test_module6_e2e_smoke(tmp_path):
    report = run_module6_on_synthetic(tmp_path, config=make_test_config())
    assert len(report.selected_portfolio_pks) > 0
    assert (report.output_dir / "portfolio_candidates.parquet").exists()
    assert (report.output_dir / "portfolio_scores.parquet").exists()
    assert (report.output_dir / "portfolio_weight_history.parquet").exists()
    assert (report.output_dir / "comparison_support_calendar.parquet").exists()
    assert (report.output_dir / "overlap" / "execution_overlap_proxy_composite.npz").exists()
    assert (report.output_dir / "dependence" / "reduced_universe_000" / "covariance.npy").exists()


def test_run_research_fails_closed_when_module6_blocks(tmp_path, monkeypatch):
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("{}", encoding="utf-8")
    run_dir = tmp_path / "synthetic_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    status_path = run_dir / "run_status.json"
    manifest_path.write_text(json.dumps({"run_id": "synthetic_run"}), encoding="utf-8")
    status_path.write_text(json.dumps({}), encoding="utf-8")
    dummy_cfg = SimpleNamespace(
        search=SimpleNamespace(seed=1),
        symbols=["SPY"],
        data=SimpleNamespace(),
        harness=SimpleNamespace(report_dir=str(tmp_path / "artifacts")),
    )
    harness_cfg = SimpleNamespace(
        report_dir=str(tmp_path / "artifacts"),
        research_mode="standard",
        parallel_backend="serial",
    )
    dummy_out = SimpleNamespace(
        artifact_paths={"run_manifest": str(manifest_path), "run_status": str(status_path)},
        run_manifest={
            "run_id": "synthetic_run",
            "n_candidates": 1,
            "failure_count": 0,
            "failure_rate": 0.0,
            "parallel_backend": "serial",
            "parallel_workers_effective": 1,
            "payload_safe": True,
            "large_payload_passing_avoided": True,
            "aborted": False,
            "abort_reason": "",
        },
        stats_verdict={"leaderboard": [{"pass": True}]},
        candidate_results=[],
    )
    monkeypatch.setattr(run_research, "_load_config", lambda path: dummy_cfg)
    monkeypatch.setattr(run_research, "_map_legacy_zimtra_aliases", lambda cfg: cfg)
    monkeypatch.setattr(run_research, "_enforce_canonical_runtime_path", lambda cfg: None)
    monkeypatch.setattr(run_research, "_configure_deterministic_runtime", lambda seed: None)
    monkeypatch.setattr(run_research, "run_full_self_audit", lambda **kwargs: {})
    monkeypatch.setattr(run_research, "run_architecture_consistency_check", lambda: None)
    monkeypatch.setattr(run_research, "run_preflight_validation_suite", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_research, "_resolved_config_sha256", lambda cfg: "sha256")
    monkeypatch.setattr(run_research, "_resolve_data_paths", lambda *args, **kwargs: [])
    monkeypatch.setattr(run_research, "_build_engine_config", lambda cfg: object())
    monkeypatch.setattr(run_research, "_build_module2_configs", lambda cfg: [])
    monkeypatch.setattr(run_research, "_build_module3_configs", lambda cfg: [])
    monkeypatch.setattr(run_research, "_build_module4_configs", lambda cfg: [])
    monkeypatch.setattr(run_research, "_build_harness_config", lambda cfg, project_root: harness_cfg)
    monkeypatch.setattr(run_research, "in_memory_date_filter_loader", lambda data_cfg: (lambda *_args, **_kwargs: None))
    monkeypatch.setattr(run_research, "_build_stress_scenarios", lambda cfg: None)
    monkeypatch.setattr(run_research, "_build_candidates", lambda cfg: [])
    monkeypatch.setattr(run_research, "run_weightiz_harness", lambda **kwargs: dummy_out)
    monkeypatch.setattr(run_research, "_ensure_run_artifact_link", lambda *args, **kwargs: Path(tmp_path))
    monkeypatch.setattr(run_research, "_append_run_registry", lambda **kwargs: None)
    monkeypatch.setattr(run_research, "_artifact_write_json", lambda path, payload: Path(path).write_text(json.dumps(payload), encoding="utf-8"))
    monkeypatch.setattr(run_research, "get_logger", lambda name: logging.getLogger(name))
    monkeypatch.setattr(run_research, "log_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_research, "run_module6_portfolio_research", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(sys, "argv", ["run_research.py", "--config", str(config_path)])
    with pytest.raises(RuntimeError, match="MODULE6_SUPPORTED_FLOW_BLOCKED"):
        run_research.main()
