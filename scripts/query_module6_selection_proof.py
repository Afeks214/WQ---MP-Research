#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only proof query for fresh Module 6 selections")
    parser.add_argument("--run-dir", required=True, help="Run directory that contains run_summary.json and module6/")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text table")
    return parser


def _load_selected_portfolios(run_dir: Path) -> tuple[Path, list[str]]:
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    module6_dir = Path(str(summary["module6_output_dir"]))
    selected_frontier = pd.read_parquet(module6_dir / "frontiers" / "selected_frontier.parquet")
    selected_count = int(summary.get("module6_selected_count", 0) or 0)
    selected_pks = selected_frontier["portfolio_pk"].astype(str).head(selected_count).tolist()
    return module6_dir, selected_pks


def _proof_rows(run_dir: Path) -> list[dict[str, float | int | str]]:
    module6_dir, selected_pks = _load_selected_portfolios(run_dir)
    scores = pd.read_parquet(module6_dir / "portfolio_scores.parquet")
    session_paths = pd.read_parquet(module6_dir / "portfolio_paths_session.parquet")
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
                "disable_flag": int(score_row.get("session_disable_flag", score_row.get("disable_flag", 0))),
                "breach_count": int(score_row.get("session_breach_count", score_row.get("breach_count", 0))),
                "session_count": int(path.shape[0]),
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
    print(
        "portfolio_pk starting_equity_score starting_equity_first_session "
        "first_session_equity final_equity max_gross_exposure disable_flag breach_count session_count"
    )
    for row in rows:
        print(
            f"{row['portfolio_pk']} "
            f"{row['starting_equity_score']:.6f} "
            f"{row['starting_equity_first_session']:.6f} "
            f"{row['first_session_equity']:.6f} "
            f"{row['final_equity']:.6f} "
            f"{row['max_gross_exposure']:.6f} "
            f"{row['disable_flag']} "
            f"{row['breach_count']} "
            f"{row['session_count']}"
        )


if __name__ == "__main__":
    main()
