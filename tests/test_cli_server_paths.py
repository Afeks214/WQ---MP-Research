from __future__ import annotations

import json
from pathlib import Path

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
    (tmp_path / "run_summary.json").write_text(json.dumps({"module6_output_dir": None}), encoding="utf-8")
    validate_artifacts.main(["--run-dir", str(tmp_path)])


def test_run_module6_requires_run_dir(tmp_path: Path) -> None:
    config_path = tmp_path / "module6.yaml"
    config_path.write_text(yaml.safe_dump({"module6": {}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="MODULE6_RUN_DIR_REQUIRED"):
        run_module6.main(["--config", str(config_path)])
