from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


_PROJECT_ROOT_ENV = "WEIGHTIZ_PROJECT_ROOT"
_ARTIFACTS_ROOT_ENV = "WEIGHTIZ_ARTIFACTS_ROOT"
_LOGS_ROOT_ENV = "WEIGHTIZ_LOGS_ROOT"
_TMP_ROOT_ENV = "WEIGHTIZ_TMP_ROOT"


def resolve_project_root(project_root: str | Path | None = None) -> Path:
    if project_root is not None:
        return Path(project_root).expanduser().resolve()
    env_root = os.environ.get(_PROJECT_ROOT_ENV)
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[4]


def resolve_repo_path(path_value: str | Path, *, project_root: str | Path | None = None) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (resolve_project_root(project_root) / path).resolve()


@dataclass(frozen=True)
class ProjectPaths:
    repo_root: Path
    src_root: Path
    configs_root: Path
    artifacts_root: Path
    logs_root: Path
    tmp_root: Path
    infra_root: Path

    @classmethod
    def discover(
        cls,
        *,
        project_root: str | Path | None = None,
        artifacts_root: str | Path | None = None,
        logs_root: str | Path | None = None,
        tmp_root: str | Path | None = None,
    ) -> "ProjectPaths":
        root = resolve_project_root(project_root)
        return cls(
            repo_root=root,
            src_root=root / "src",
            configs_root=root / "configs",
            artifacts_root=resolve_repo_path(artifacts_root or "artifacts", project_root=root),
            logs_root=resolve_repo_path(logs_root or "logs", project_root=root),
            tmp_root=resolve_repo_path(tmp_root or "tmp", project_root=root),
            infra_root=root / "infra",
        )

    def export_environment(self) -> None:
        os.environ[_PROJECT_ROOT_ENV] = str(self.repo_root)
        os.environ[_ARTIFACTS_ROOT_ENV] = str(self.artifacts_root)
        os.environ[_LOGS_ROOT_ENV] = str(self.logs_root)
        os.environ[_TMP_ROOT_ENV] = str(self.tmp_root)

    def ensure_runtime_dirs(self) -> None:
        self.artifacts_root.mkdir(parents=True, exist_ok=True)
        self.logs_root.mkdir(parents=True, exist_ok=True)
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        self.infra_root.mkdir(parents=True, exist_ok=True)
