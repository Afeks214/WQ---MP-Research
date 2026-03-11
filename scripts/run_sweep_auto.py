#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

from scripts._archive.legacy_sweep.run_sweep_auto import (
    REPO_ROOT,
    _apply_quick_run_reduction,
    _read_latest_run_dir,
    _verify_quick_run_artifacts,
    subprocess,
)

__all__ = [
    "_apply_quick_run_reduction",
    "_read_latest_run_dir",
    "_run_research",
    "_verify_quick_run_artifacts",
    "subprocess",
]


def _run_research(
    config_path: Path,
    *,
    quick_run: bool = False,
    log_dir: Path | None = None,
) -> Path:
    cmd = [sys.executable, "-m", "weightiz.cli.run_research", "--config", str(config_path.resolve())]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    if quick_run:
        env["QUICK_RUN"] = "1"
        env.setdefault("QUICK_RUN_TASK_TIMEOUT_SEC", "120")
        env.setdefault("QUICK_RUN_PROGRESS_EVERY", "1")
    if log_dir is None:
        log_dir = (REPO_ROOT / "artifacts" / "sweep_v2" / "_logs").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{config_path.stem}_stdout.log"
    stderr_path = log_dir / f"{config_path.stem}_stderr.log"
    with stdout_path.open("w", encoding="utf-8") as f_out, stderr_path.open("w", encoding="utf-8") as f_err:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            text=True,
            stdout=f_out,
            stderr=f_err,
            check=False,
            env=env,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"run_research failed (rc={proc.returncode}) for config={config_path}. "
            f"See logs: {stdout_path} and {stderr_path}"
        )
    return _read_latest_run_dir()


if __name__ == "__main__":
    raise RuntimeError(
        "DEPRECATED_SWEEP_PIPELINE: scripts/run_sweep_auto.py is a compatibility shim only. "
        "Use the unified pipeline: ./.venv/bin/python -m weightiz.cli.run_research --config <yaml>. "
        "Legacy implementation lives under scripts/_archive/legacy_sweep/run_sweep_auto.py"
    )
