from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from weightiz.shared.config.models import RunConfigModel


def resolved_config_sha256(cfg: RunConfigModel) -> str:
    payload = cfg.model_dump(mode="json")
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def append_run_registry(
    artifacts_root: Path,
    run_id: str,
    run_dir: Path,
    symbols: list[str],
    n_candidates: int,
    pass_count: int,
    resolved_config_sha256: str,
) -> None:
    artifacts_root.mkdir(parents=True, exist_ok=True)

    entry = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "path": str(run_dir.resolve()),
        "symbols": symbols,
        "n_candidates": int(n_candidates),
        "pass_count": int(pass_count),
        "resolved_config_sha256": str(resolved_config_sha256),
    }

    index_path = artifacts_root / "run_index.jsonl"
    with index_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    latest_path = artifacts_root / ".latest_run"
    latest_path.write_text(str(run_dir.resolve()) + "\n", encoding="utf-8")


def ensure_run_artifact_link(artifacts_root: Path, run_dir: Path) -> Path:
    module5_root = artifacts_root / "module5_harness"
    module5_root.mkdir(parents=True, exist_ok=True)
    target = module5_root / run_dir.name
    if target.resolve() == run_dir.resolve():
        return target
    if target.exists() or target.is_symlink():
        return target
    try:
        target.symlink_to(run_dir.resolve(), target_is_directory=True)
    except Exception:
        # Fallback: create directory marker with absolute pointer.
        target.mkdir(parents=True, exist_ok=True)
        (target / ".run_path").write_text(str(run_dir.resolve()) + "\n", encoding="utf-8")
    return target
