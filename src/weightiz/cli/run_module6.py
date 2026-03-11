from __future__ import annotations

import argparse
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import yaml

from weightiz.module5.harness.artifact_writers import write_json as _write_json
from weightiz.module6 import Module6Config, run_module6_portfolio_research
from weightiz.shared.config.paths import ProjectPaths, resolve_repo_path


def _apply_dataclass_overrides(obj: Any, payload: dict[str, Any] | None) -> Any:
    if not isinstance(payload, dict):
        return obj
    valid = {field.name: payload[field.name] for field in fields(obj) if field.name in payload}
    return replace(obj, **valid)


def build_module6_config(config_block: dict[str, Any] | None) -> Module6Config:
    if config_block is None:
        config_block = {}
    if not isinstance(config_block, dict):
        raise RuntimeError("module6 config payload must be a mapping")
    cfg = Module6Config()
    return replace(
        cfg,
        intake=_apply_dataclass_overrides(cfg.intake, config_block.get("intake")),
        reduction=_apply_dataclass_overrides(cfg.reduction, config_block.get("reduction")),
        dependence=_apply_dataclass_overrides(cfg.dependence, config_block.get("dependence")),
        generator=_apply_dataclass_overrides(cfg.generator, config_block.get("generator")),
        simulator=_apply_dataclass_overrides(cfg.simulator, config_block.get("simulator")),
        scoring=_apply_dataclass_overrides(cfg.scoring, config_block.get("scoring")),
        export=_apply_dataclass_overrides(cfg.export, config_block.get("export")),
        runtime=_apply_dataclass_overrides(cfg.runtime, config_block.get("runtime")),
    )


def _load_module6_payload(path: Path) -> tuple[dict[str, Any], Module6Config]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise RuntimeError("Module 6 CLI config root must be an object/mapping")
    config_block = raw.get("module6", raw.get("config", {}))
    return raw, build_module6_config(config_block)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Weightiz Module 6 runner")
    parser.add_argument("--config", required=True, help="Path to Module 6 YAML config")
    parser.add_argument("--run-dir", default=None, help="Override Module 5 run directory")
    parser.add_argument("--output-dir", default=None, help="Override Module 6 output directory")
    parser.add_argument("--project-root", default=None, help="Override repository root for path resolution")
    parser.add_argument("--logs-root", default=None, help="Override CLI log root for this run")
    parser.add_argument("--tmp-root", default=None, help="Override temporary root for this run")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    paths = ProjectPaths.discover(
        project_root=args.project_root,
        logs_root=args.logs_root,
        tmp_root=args.tmp_root,
    )
    paths.ensure_runtime_dirs()
    paths.export_environment()

    config_path = resolve_repo_path(args.config, project_root=paths.repo_root)
    raw, cfg = _load_module6_payload(config_path)
    run_dir_value = args.run_dir or raw.get("run_dir")
    if not run_dir_value:
        raise RuntimeError("MODULE6_RUN_DIR_REQUIRED")
    run_dir = resolve_repo_path(str(run_dir_value), project_root=paths.repo_root)
    output_dir = None
    if args.output_dir:
        output_dir = resolve_repo_path(args.output_dir, project_root=paths.repo_root)
    elif raw.get("output_dir"):
        output_dir = resolve_repo_path(str(raw["output_dir"]), project_root=paths.repo_root)

    report = run_module6_portfolio_research(run_dir=run_dir, output_dir=output_dir, config=cfg)
    record = {
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "output_dir": str(report.output_dir),
        "selected_count": int(len(report.selected_portfolio_pks)),
    }
    _write_json(paths.logs_root / "module6_latest_run.json", record)


if __name__ == "__main__":
    main()
