from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any


def _to_dict(cfg: Any) -> dict[str, Any]:
    if isinstance(cfg, dict):
        return cfg
    if hasattr(cfg, "model_dump"):
        return dict(cfg.model_dump())
    if is_dataclass(cfg):
        return asdict(cfg)
    return dict(cfg.__dict__)


def run_preflight_validation_suite(config: Any, context: dict[str, Any] | None = None) -> None:
    cfg = _to_dict(config)
    ctx = context or {}

    # Deterministic seed requirement
    search = cfg.get("search", {}) if isinstance(cfg.get("search", {}), dict) else {}
    harness = cfg.get("harness", {}) if isinstance(cfg.get("harness", {}), dict) else {}
    seed = search.get("seed", harness.get("seed"))
    if seed is None:
        raise RuntimeError("PREFLIGHT_VALIDATION_FAILED: config.search.seed must not be None")

    config_hash = str(ctx.get("config_hash", "")).strip()
    if config_hash and len(config_hash) < 16:
        raise RuntimeError("PREFLIGHT_VALIDATION_FAILED: invalid config hash")

    # Canonical runtime path only guard.
    if bool(ctx.get("parallel_runtime_enabled", False)):
        raise RuntimeError("PREFLIGHT_VALIDATION_FAILED: parallel runtime path is forbidden")

    report_dir = ctx.get("report_dir")
    if report_dir:
        p = Path(str(report_dir)).resolve()
        p.mkdir(parents=True, exist_ok=True)
        probe = p / ".preflight_write_probe"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink(missing_ok=True)

    # Shared-memory availability probe.
    try:
        shm = shared_memory.SharedMemory(create=True, size=8)
        shm.close()
        shm.unlink()
    except Exception as exc:
        raise RuntimeError("PREFLIGHT_VALIDATION_FAILED: shared memory unavailable") from exc

    # If a manifest reference exists, ensure JSON parseable (sanity check).
    manifest_path = ctx.get("manifest_path")
    if manifest_path:
        p = Path(str(manifest_path))
        if p.exists():
            json.loads(p.read_text(encoding="utf-8"))
