from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from weightiz.shared.config.paths import ProjectPaths, resolve_repo_path


def _module6_dependence_dirs(module6_root: Path) -> list[Path]:
    dependence_root = module6_root / "dependence"
    if not dependence_root.exists():
        return []
    return sorted(path for path in dependence_root.iterdir() if path.is_dir())


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
    if "module6_status" not in summary:
        raise RuntimeError("RUN_SUMMARY_FIELD_MISSING: module6_status")
    module6_dir = summary.get("module6_output_dir")
    if args.require_module6:
        if not module6_dir:
            raise RuntimeError("MODULE6_OUTPUT_DIR_MISSING")
        if str(summary.get("module6_status")) != "completed":
            raise RuntimeError(f"MODULE6_STATUS_INVALID: {summary.get('module6_status')}")
        module6_root = resolve_repo_path(str(module6_dir), project_root=paths.repo_root)
        module6_required = [
            module6_root / "portfolio_scores.parquet",
            module6_root / "portfolio_candidates.parquet",
            module6_root / "portfolio_weight_history.parquet",
            module6_root / "portfolio_paths_session.parquet",
            module6_root / "frontiers" / "selected_frontier.parquet",
        ]
        module6_missing = [str(path) for path in module6_required if not path.exists()]
        if module6_missing:
            raise RuntimeError(f"MODULE6_ARTIFACTS_MISSING: {module6_missing}")
        dependence_dirs = _module6_dependence_dirs(module6_root)
        if not dependence_dirs:
            raise RuntimeError("MODULE6_DEPENDENCE_ARTIFACTS_MISSING: no dependence directories found")
        dependence_missing: list[str] = []
        for dep_dir in dependence_dirs:
            for artifact_name in ("asset_column_indices.npy", "pair_completion_reason_codes.npy"):
                artifact_path = dep_dir / artifact_name
                if not artifact_path.exists():
                    dependence_missing.append(str(artifact_path))
        if dependence_missing:
            raise RuntimeError(f"MODULE6_DEPENDENCE_ARTIFACTS_MISSING: {dependence_missing}")
        selected_frontier = pd.read_parquet(module6_root / "frontiers" / "selected_frontier.parquet")
        selected_count = int(summary.get("module6_selected_count", 0) or 0)
        selected_pks = selected_frontier["portfolio_pk"].astype(str).head(selected_count).tolist()
        if selected_count > 0 and len(selected_pks) < selected_count:
            raise RuntimeError("MODULE6_SELECTED_FRONTIER_SHORT")
        if selected_pks:
            session_paths = pd.read_parquet(module6_root / "portfolio_paths_session.parquet")
            session_paths["portfolio_pk"] = session_paths["portfolio_pk"].astype(str)
            gross_peak = (
                session_paths.groupby("portfolio_pk", dropna=False)["gross_exposure_mult"]
                .max()
                .rename("gross_exposure_peak")
            )
            selected_gross = gross_peak.reindex(selected_pks).fillna(0.0)
            zero_gross = selected_gross.loc[selected_gross <= 0.0]
            if not zero_gross.empty:
                raise RuntimeError(
                    "MODULE6_SELECTED_PORTFOLIOS_ZERO_GROSS_EXPOSURE: "
                    + ", ".join(f"{pk}={float(value):.6f}" for pk, value in zero_gross.items())
                )


if __name__ == "__main__":
    main()
