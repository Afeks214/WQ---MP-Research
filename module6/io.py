from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from module6.config import Module6Config
from module6.utils import Module6ValidationError, require_columns


@dataclass(frozen=True)
class Module5ArtifactPaths:
    run_dir: Path
    run_manifest: Path
    leaderboard: Path
    robustness_leaderboard: Path
    strategy_results: Path
    strategy_instance_selection: Path
    strategy_instance_session_returns: Path
    equity_curves: Path
    trade_log: Path
    micro_diagnostics: Path | None
    funnel_1545: Path | None
    daily_returns: Path | None


@dataclass
class LoadedModule5Run:
    paths: Module5ArtifactPaths
    run_manifest: dict[str, Any]
    leaderboard: pd.DataFrame
    robustness_leaderboard: pd.DataFrame
    strategy_results: pd.DataFrame
    strategy_instance_selection: pd.DataFrame
    strategy_instance_session_returns: pd.DataFrame
    equity_curves: pd.DataFrame
    trade_log: pd.DataFrame
    micro_diagnostics: pd.DataFrame | None
    funnel_1545: pd.DataFrame | None


def _resolve_optional(path: Path) -> Path | None:
    return path if path.exists() else None


def discover_module5_run_artifacts(run_dir: str | Path, config: Module6Config) -> Module5ArtifactPaths:
    root = Path(run_dir).resolve()
    if not root.exists():
        raise Module6ValidationError(f"module5 run directory does not exist: {root}")
    required = {
        "run_manifest.json": root / "run_manifest.json",
        "leaderboard.csv": root / "leaderboard.csv",
        "robustness_leaderboard.csv": root / "robustness_leaderboard.csv",
        "strategy_results.parquet": root / "strategy_results.parquet",
        "equity_curves.parquet": root / "equity_curves.parquet",
        "trade_log.parquet": root / "trade_log.parquet",
    }
    bridge_required = {
        "strategy_instance_selection.parquet": root / "strategy_instance_selection.parquet",
        "strategy_instance_session_returns.parquet": root / "strategy_instance_session_returns.parquet",
    }
    for name, path in required.items():
        if not path.exists():
            raise Module6ValidationError(f"required Module 5 artifact missing: {name}")
    if config.intake.require_bridge_artifacts:
        for name, path in bridge_required.items():
            if not path.exists():
                raise Module6ValidationError(
                    f"required Module 6 bridge artifact missing: {name}"
                )
    return Module5ArtifactPaths(
        run_dir=root,
        run_manifest=required["run_manifest.json"],
        leaderboard=required["leaderboard.csv"],
        robustness_leaderboard=required["robustness_leaderboard.csv"],
        strategy_results=required["strategy_results.parquet"],
        strategy_instance_selection=bridge_required["strategy_instance_selection.parquet"],
        strategy_instance_session_returns=bridge_required["strategy_instance_session_returns.parquet"],
        equity_curves=required["equity_curves.parquet"],
        trade_log=required["trade_log.parquet"],
        micro_diagnostics=_resolve_optional(root / "micro_diagnostics.parquet"),
        funnel_1545=_resolve_optional(root / "funnel_1545.parquet"),
        daily_returns=_resolve_optional(root / "daily_returns.parquet"),
    )


def load_module5_run(run_dir: str | Path, config: Module6Config) -> LoadedModule5Run:
    paths = discover_module5_run_artifacts(run_dir=run_dir, config=config)
    run_manifest = json.loads(paths.run_manifest.read_text(encoding="utf-8"))
    leaderboard = pd.read_csv(paths.leaderboard)
    robustness = pd.read_csv(paths.robustness_leaderboard)
    strategy_results = pd.read_parquet(paths.strategy_results)
    selection = pd.read_parquet(paths.strategy_instance_selection)
    session_returns = pd.read_parquet(paths.strategy_instance_session_returns)
    equity_curves = pd.read_parquet(paths.equity_curves)
    trade_log = pd.read_parquet(paths.trade_log)
    micro = pd.read_parquet(paths.micro_diagnostics) if paths.micro_diagnostics else None
    funnel = pd.read_parquet(paths.funnel_1545) if paths.funnel_1545 else None
    validate_source_contracts(
        run_manifest=run_manifest,
        leaderboard=leaderboard,
        robustness=robustness,
        strategy_results=strategy_results,
        selection=selection,
        session_returns=session_returns,
        equity_curves=equity_curves,
        trade_log=trade_log,
        micro_diagnostics=micro,
    )
    return LoadedModule5Run(
        paths=paths,
        run_manifest=run_manifest,
        leaderboard=leaderboard,
        robustness_leaderboard=robustness,
        strategy_results=strategy_results,
        strategy_instance_selection=selection,
        strategy_instance_session_returns=session_returns,
        equity_curves=equity_curves,
        trade_log=trade_log,
        micro_diagnostics=micro,
        funnel_1545=funnel,
    )


def validate_source_contracts(
    *,
    run_manifest: dict[str, Any],
    leaderboard: pd.DataFrame,
    robustness: pd.DataFrame,
    strategy_results: pd.DataFrame,
    selection: pd.DataFrame,
    session_returns: pd.DataFrame,
    equity_curves: pd.DataFrame,
    trade_log: pd.DataFrame,
    micro_diagnostics: pd.DataFrame | None,
) -> None:
    if str(run_manifest.get("dataset_hash", "")).strip() == "":
        raise Module6ValidationError("run_manifest missing dataset_hash")
    if str(run_manifest.get("run_id", "")).strip() == "":
        raise Module6ValidationError("run_manifest missing run_id")
    require_columns(
        leaderboard,
        [
            "candidate_id",
            "parameter_hash",
            "family_id",
            "hypothesis_id",
            "cost_adjusted_expectancy",
            "failed",
            "reject",
            "dq_min",
            "dq_median",
            "dq_degrade_count",
            "dq_reject_count",
        ],
        "leaderboard.csv",
    )
    require_columns(robustness, ["candidate_id"], "robustness_leaderboard.csv")
    require_columns(strategy_results, ["strategy_id", "parameter_hash"], "strategy_results.parquet")
    require_columns(
        selection,
        [
            "strategy_id",
            "candidate_id",
            "split_id",
            "scenario_id",
            "execution_mode",
            "selection_stage",
            "calendar_version",
            "portfolio_instance_role",
            "canonical_reference_split_id",
            "canonical_reference_scenario_id",
            "canonical_reference_policy",
        ],
        "strategy_instance_selection.parquet",
    )
    require_columns(
        session_returns,
        [
            "strategy_id",
            "candidate_id",
            "split_id",
            "scenario_id",
            "execution_mode",
            "selection_stage",
            "calendar_version",
            "session_id",
            "return_exec",
            "return_raw",
            "availability_state_code",
            "availability_state_source",
            "observed_exec",
            "observed_raw",
            "session_turnover",
            "session_trade_count",
            "gross_mult_mean",
            "gross_mult_peak",
            "buying_power_min",
            "buying_power_min_frac",
            "daily_loss_max",
        ],
        "strategy_instance_session_returns.parquet",
    )
    require_columns(
        equity_curves,
        ["ts_ns", "session_id", "candidate_id", "split_id", "scenario_id", "equity", "margin_used"],
        "equity_curves.parquet",
    )
    require_columns(
        trade_log,
        ["ts_ns", "session_id", "candidate_id", "split_id", "scenario_id", "symbol", "filled_qty", "exec_price", "trade_cost"],
        "trade_log.parquet",
    )
    if micro_diagnostics is None:
        raise Module6ValidationError("micro_diagnostics.parquet is required for Module 6 truth replay")
    require_columns(
        micro_diagnostics,
        ["ts_ns", "session_id", "candidate_id", "split_id", "scenario_id", "symbol", "filled_qty", "exec_price", "trade_cost"],
        "micro_diagnostics.parquet",
    )
    if "module6_bridge" not in run_manifest:
        raise Module6ValidationError("run_manifest missing module6_bridge summary")
    if not bool(run_manifest.get("daily_matrix_shape_raw")):
        raise Module6ValidationError("run_manifest missing daily_matrix_shape_raw")
    bridge = dict(run_manifest.get("module6_bridge", {}))
    for key in ("canonical_reference_split_id", "canonical_reference_scenario_id", "canonical_reference_policy", "calendar_version"):
        if str(bridge.get(key, "")).strip() == "":
            raise Module6ValidationError(f"run_manifest.module6_bridge missing {key}")
