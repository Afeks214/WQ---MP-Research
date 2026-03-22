from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from weightiz.cli import run_module6, validate_artifacts
from weightiz.shared.config.paths import ProjectPaths, resolve_repo_path


def test_project_paths_discover_repo_shape() -> None:
    paths = ProjectPaths.discover()
    assert paths.repo_root.name == "New project"
    assert paths.src_root == paths.repo_root / "src"
    assert paths.configs_root == paths.repo_root / "configs"


def test_resolve_repo_path_is_repo_relative() -> None:
    paths = ProjectPaths.discover()
    resolved = resolve_repo_path("configs/server/compute-small.yaml", project_root=paths.repo_root)
    assert resolved == paths.repo_root / "configs" / "server" / "compute-small.yaml"


def test_validate_artifacts_accepts_minimal_run_dir(tmp_path: Path) -> None:
    (tmp_path / "run_manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "run_status.json").write_text("{}", encoding="utf-8")
    (tmp_path / "run_summary.json").write_text(
        json.dumps({"module6_status": "disabled", "module6_output_dir": None}),
        encoding="utf-8",
    )
    validate_artifacts.main(["--run-dir", str(tmp_path)])


def test_validate_artifacts_requires_module6_status(tmp_path: Path) -> None:
    (tmp_path / "run_manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "run_status.json").write_text("{}", encoding="utf-8")
    (tmp_path / "run_summary.json").write_text(json.dumps({"module6_output_dir": None}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="RUN_SUMMARY_FIELD_MISSING"):
        validate_artifacts.main(["--run-dir", str(tmp_path)])


def test_validate_artifacts_enforces_module6_contract(tmp_path: Path) -> None:
    module6_dir = tmp_path / "module6"
    dep_dir = module6_dir / "dependence" / "ru_000"
    frontiers_dir = module6_dir / "frontiers"
    dep_dir.mkdir(parents=True, exist_ok=True)
    frontiers_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "run_manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "run_status.json").write_text("{}", encoding="utf-8")
    (tmp_path / "run_summary.json").write_text(
        json.dumps(
            {
                "module6_status": "completed",
                "module6_output_dir": str(module6_dir),
                "module6_selected_count": 1,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame({"portfolio_pk": ["p0"], "final_score": [0.9], "starting_equity": [2000.0]}).to_parquet(
        module6_dir / "portfolio_scores.parquet",
        index=False,
    )
    pd.DataFrame({"portfolio_pk": ["p0"]}).to_parquet(module6_dir / "portfolio_candidates.parquet", index=False)
    pd.DataFrame({"portfolio_pk": ["p0"], "strategy_instance_pk": ["s0"], "target_weight": [1.0]}).to_parquet(
        module6_dir / "portfolio_weight_history.parquet",
        index=False,
    )
    pd.DataFrame(
        {
            "portfolio_pk": ["p0"],
            "session_id": [1],
            "gross_exposure_mult": [1.25],
            "equity": [1995.0],
        }
    ).to_parquet(module6_dir / "portfolio_paths_session.parquet", index=False)
    pd.DataFrame({"portfolio_pk": ["p0"]}).to_parquet(frontiers_dir / "selected_frontier.parquet", index=False)
    np.save(dep_dir / "asset_column_indices.npy", np.asarray([1, 2], dtype=np.int64))
    np.save(dep_dir / "pair_completion_reason_codes.npy", np.asarray([["", "PAIR_OK"], ["PAIR_OK", ""]], dtype="<U24"))
    validate_artifacts.main(["--run-dir", str(tmp_path), "--require-module6"])


def test_validate_artifacts_rejects_zero_gross_selected_portfolios(tmp_path: Path) -> None:
    module6_dir = tmp_path / "module6"
    dep_dir = module6_dir / "dependence" / "ru_000"
    frontiers_dir = module6_dir / "frontiers"
    dep_dir.mkdir(parents=True, exist_ok=True)
    frontiers_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "run_manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "run_status.json").write_text("{}", encoding="utf-8")
    (tmp_path / "run_summary.json").write_text(
        json.dumps(
            {
                "module6_status": "completed",
                "module6_output_dir": str(module6_dir),
                "module6_selected_count": 1,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame({"portfolio_pk": ["p0"], "final_score": [0.9]}).to_parquet(module6_dir / "portfolio_scores.parquet", index=False)
    pd.DataFrame({"portfolio_pk": ["p0"]}).to_parquet(module6_dir / "portfolio_candidates.parquet", index=False)
    pd.DataFrame({"portfolio_pk": ["p0"], "strategy_instance_pk": ["s0"], "target_weight": [1.0]}).to_parquet(
        module6_dir / "portfolio_weight_history.parquet",
        index=False,
    )
    pd.DataFrame(
        {
            "portfolio_pk": ["p0"],
            "session_id": [1],
            "gross_exposure_mult": [0.0],
            "equity": [1000.0],
        }
    ).to_parquet(module6_dir / "portfolio_paths_session.parquet", index=False)
    pd.DataFrame({"portfolio_pk": ["p0"]}).to_parquet(frontiers_dir / "selected_frontier.parquet", index=False)
    np.save(dep_dir / "asset_column_indices.npy", np.asarray([1], dtype=np.int64))
    np.save(dep_dir / "pair_completion_reason_codes.npy", np.asarray([[""]], dtype="<U24"))
    with pytest.raises(RuntimeError, match="ZERO_GROSS_EXPOSURE"):
        validate_artifacts.main(["--run-dir", str(tmp_path), "--require-module6"])


def test_run_module6_requires_run_dir(tmp_path: Path) -> None:
    config_path = tmp_path / "module6.yaml"
    config_path.write_text(yaml.safe_dump({"module6": {}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="MODULE6_RUN_DIR_REQUIRED"):
        run_module6.main(["--config", str(config_path)])
