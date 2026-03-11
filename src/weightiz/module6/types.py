from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StrategyMasterRow:
    strategy_pk: str
    strategy_id: str
    source_run_id: str
    dataset_hash: str
    parameter_hash: str
    enabled_assets_hash: str
    family_id: str
    hypothesis_id: str
    portfolio_admit_flag: bool


@dataclass(frozen=True)
class StrategySessionRow:
    strategy_instance_pk: str
    strategy_pk: str
    session_id: int
    return_exec: float
    return_raw: float
    availability_state_code: int


@dataclass(frozen=True)
class ReducedUniverseSpec:
    reduced_universe_id: str
    strategy_instance_pks: tuple[str, ...]
    representative_strategy_instance_pks: tuple[str, ...]
    retained_hedge_strategy_instance_pks: tuple[str, ...]
    cluster_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PortfolioCandidateSpec:
    portfolio_pk: str
    reduced_universe_id: str
    generator_family: str
    rebalance_policy: str
    target_weights: dict[str, float]
    cash_weight: float
    seed: int
    batch_id: str


@dataclass(frozen=True)
class PortfolioPathSummary:
    portfolio_pk: str
    reduced_universe_id: str
    final_equity: float
    annualized_return: float
    max_drawdown: float
    turnover: float
    first_pass_score: float
    compliance_flags: tuple[str, ...]


@dataclass(frozen=True)
class PortfolioSelectionReport:
    run_id: str
    output_dir: Path
    selected_portfolio_pks: tuple[str, ...]
    alternate_portfolio_pks: tuple[str, ...]
    summary: dict[str, Any]

