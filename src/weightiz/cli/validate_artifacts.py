from __future__ import annotations

import argparse
import json

from weightiz.shared.config.paths import ProjectPaths, resolve_repo_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Weightiz run artifacts")
    parser.add_argument("--run-dir", required=True, help="Run directory to validate")
    parser.add_argument("--project-root", default=None, help="Override repository root for path resolution")
    parser.add_argument("--require-module6", action="store_true", help="Require Module 6 outputs to exist")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    paths = ProjectPaths.discover(project_root=args.project_root)
    run_dir = resolve_repo_path(args.run_dir, project_root=paths.repo_root)
    required = [
        run_dir / "run_manifest.json",
        run_dir / "run_status.json",
        run_dir / "run_summary.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"ARTIFACTS_MISSING: {missing}")

    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    module6_dir = summary.get("module6_output_dir")
    if args.require_module6:
        if not module6_dir:
            raise RuntimeError("MODULE6_OUTPUT_DIR_MISSING")
        module6_root = resolve_repo_path(str(module6_dir), project_root=paths.repo_root)
        module6_required = [
            module6_root / "portfolio_scores.parquet",
            module6_root / "portfolio_candidates.parquet",
            module6_root / "portfolio_weight_history.parquet",
        ]
        module6_missing = [str(path) for path in module6_required if not path.exists()]
        if module6_missing:
            raise RuntimeError(f"MODULE6_ARTIFACTS_MISSING: {module6_missing}")


if __name__ == "__main__":
    main()
