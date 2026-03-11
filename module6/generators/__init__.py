from __future__ import annotations

import pandas as pd

from module6.config import Module6Config
from module6.generators.cluster_balanced import generate_cluster_balanced_batch
from module6.generators.hrp import generate_hrp_variants
from module6.generators.mv_shrinkage import generate_mv_variants
from module6.generators.random_sparse import generate_random_sparse_batch
from module6.types import ReducedUniverseSpec


def generate_all_portfolios(
    *,
    reduced_universe: ReducedUniverseSpec,
    strategy_frame: pd.DataFrame,
    covariance_bundle,
    returns_exec,
    column_indices,
    config: Module6Config,
    calendar_version: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    weight_frames = []
    for generator in (
        lambda: generate_random_sparse_batch(
            reduced_universe=reduced_universe,
            strategy_frame=strategy_frame,
            config=config,
            calendar_version=calendar_version,
        ),
        lambda: generate_cluster_balanced_batch(
            reduced_universe=reduced_universe,
            strategy_frame=strategy_frame,
            config=config,
            calendar_version=calendar_version,
        ),
        lambda: generate_hrp_variants(
            reduced_universe=reduced_universe,
            strategy_frame=strategy_frame,
            covariance_bundle=covariance_bundle,
            config=config,
            calendar_version=calendar_version,
        ),
        lambda: generate_mv_variants(
            reduced_universe=reduced_universe,
            strategy_frame=strategy_frame,
            covariance_bundle=covariance_bundle,
            returns_exec=returns_exec,
            column_indices=column_indices,
            config=config,
            calendar_version=calendar_version,
        ),
    ):
        frame, weight_frame = generator()
        if frame.shape[0] > 0:
            frames.append(frame)
            weight_frames.append(weight_frame)
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    candidates = pd.concat(frames, axis=0, ignore_index=True).drop_duplicates("portfolio_pk", keep="first")
    weights = pd.concat(weight_frames, axis=0, ignore_index=True)
    return (
        candidates.sort_values(["generator_family", "portfolio_pk"], kind="mergesort").reset_index(drop=True),
        weights.sort_values(["portfolio_pk", "strategy_instance_pk"], kind="mergesort").reset_index(drop=True),
    )
