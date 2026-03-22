#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _load_run_summary(run_dir: Path) -> dict:
    return json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))


def _load_intake_diagnostics(run_dir: Path, summary: dict) -> dict:
    diagnostics_ref = summary.get("module6_intake_diagnostics")
    if not diagnostics_ref:
        return {}
    diagnostics_path = Path(str(diagnostics_ref))
    if not diagnostics_path.is_absolute():
        diagnostics_path = run_dir / diagnostics_path
    if not diagnostics_path.exists():
        return {}
    return json.loads(diagnostics_path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only proof query for fresh Module 6 selections")
    parser.add_argument("--run-dir", required=True, help="Run directory that contains run_summary.json and module6/")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text table")
    return parser


def _load_selected_portfolios(run_dir: Path, summary: dict) -> tuple[Path | None, list[str]]:
    module6_output_dir = summary.get("module6_output_dir")
    if not module6_output_dir:
        return None, []
    module6_dir = Path(str(module6_output_dir))
    frontier_path = module6_dir / "frontiers" / "selected_frontier.parquet"
    if not frontier_path.exists():
        return module6_dir, []
    selected_frontier = pd.read_parquet(frontier_path)
    selected_count = int(summary.get("module6_selected_count", 0) or 0)
    selected_pks = selected_frontier["portfolio_pk"].astype(str).head(selected_count).tolist()
    return module6_dir, selected_pks


def _proof_rows(run_dir: Path) -> list[dict[str, float | int | str]]:
    summary = _load_run_summary(run_dir)
    intake_diagnostics = _load_intake_diagnostics(run_dir, summary)
    module6_dir, selected_pks = _load_selected_portfolios(run_dir, summary)
    if module6_dir is None or not selected_pks:
        return [
            {
                "run_id": str(summary.get("run_id", "")),
                "module6_status": str(summary.get("module6_status", "unknown")),
                "module6_failure_stage": str(summary.get("module6_failure_stage", "")),
                "module6_error_message": str(summary.get("module6_error_message", "")),
                "selected_portfolio_count": int(summary.get("module6_selected_count", 0) or 0),
                "friction_gate_rejected_strategy_count": int(
                    intake_diagnostics.get("metric_summary", {}).get("friction_gate_reject_count", 0)
                ),
                "first_zero_gate": str(intake_diagnostics.get("first_zero_gate", "")),
            }
        ]
    scores = pd.read_parquet(module6_dir / "portfolio_scores.parquet")
    session_paths = pd.read_parquet(module6_dir / "portfolio_paths_session.parquet")
    reduction_diagnostics_path = module6_dir / "diagnostics" / "module6_reduction_diagnostics.json"
    reduction_diagnostics = (
        json.loads(reduction_diagnostics_path.read_text(encoding="utf-8"))
        if reduction_diagnostics_path.exists()
        else {}
    )
    score_rows = scores.copy()
    score_rows["portfolio_pk"] = score_rows["portfolio_pk"].astype(str)
    path_rows = session_paths.copy()
    path_rows["portfolio_pk"] = path_rows["portfolio_pk"].astype(str)
    rows: list[dict[str, float | int | str]] = []
    for portfolio_pk in selected_pks:
        score = score_rows.loc[score_rows["portfolio_pk"] == portfolio_pk]
        path = path_rows.loc[path_rows["portfolio_pk"] == portfolio_pk].sort_values(
            ["session_id"],
            kind="mergesort",
        )
        if score.empty or path.empty:
            continue
        score_row = score.iloc[0]
        rows.append(
            {
                "portfolio_pk": portfolio_pk,
                "starting_equity_score": float(score_row.get("starting_equity", float("nan"))),
                "starting_equity_first_session": float(path["session_start_equity"].iloc[0]),
                "first_session_equity": float(path["equity"].iloc[0]),
                "final_equity": float(path["equity"].iloc[-1]),
                "max_gross_exposure": float(path["gross_exposure_mult"].max()),
                "average_gross_exposure": float(path["gross_exposure_mult"].mean()),
                "disable_flag": int(score_row.get("session_disable_flag", score_row.get("disable_flag", 0))),
                "breach_count": int(score_row.get("session_breach_count", score_row.get("breach_count", 0))),
                "realized_volatility": float(score_row.get("minute_realized_volatility", score_row.get("session_realized_volatility", float("nan")))),
                "session_count": int(path.shape[0]),
                "friction_gate_rejected_strategy_count": int(
                    reduction_diagnostics.get("friction_gate_rejected_strategy_count", 0)
                ),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    rows = _proof_rows(run_dir)
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return
    if not rows:
        print("No selected portfolios were found.")
        return
    if "portfolio_pk" not in rows[0]:
        print(
            "run_id module6_status module6_failure_stage selected_portfolio_count "
            "friction_gate_rejected_strategy_count first_zero_gate module6_error_message"
        )
        for row in rows:
            print(
                f"{row['run_id']} "
                f"{row['module6_status']} "
                f"{row['module6_failure_stage']} "
                f"{row['selected_portfolio_count']} "
                f"{row['friction_gate_rejected_strategy_count']} "
                f"{row['first_zero_gate']} "
                f"{row['module6_error_message']}"
            )
        return
    print(
        "portfolio_pk starting_equity_score starting_equity_first_session "
        "first_session_equity final_equity max_gross_exposure average_gross_exposure "
        "realized_volatility disable_flag breach_count friction_gate_rejected_strategy_count session_count"
    )
    for row in rows:
        print(
            f"{row['portfolio_pk']} "
            f"{row['starting_equity_score']:.6f} "
            f"{row['starting_equity_first_session']:.6f} "
            f"{row['first_session_equity']:.6f} "
            f"{row['final_equity']:.6f} "
            f"{row['max_gross_exposure']:.6f} "
            f"{row['average_gross_exposure']:.6f} "
            f"{row['realized_volatility']:.6f} "
            f"{row['disable_flag']} "
            f"{row['breach_count']} "
            f"{row['friction_gate_rejected_strategy_count']} "
            f"{row['session_count']}"
        )


if __name__ == "__main__":
    main()
