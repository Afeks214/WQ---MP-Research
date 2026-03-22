from __future__ import annotations

import numpy as np
import pandas as pd

from weightiz.module6.config import Module6Config
from weightiz.module6.utils import Module6ValidationError, require_columns, require_numeric_series


def _false_series(df: pd.DataFrame) -> pd.Series:
    return pd.Series(False, index=df.index, dtype=bool)


def _required_numeric(df: pd.DataFrame, column: str, *, integer: bool = False) -> pd.Series:
    if column not in df.columns:
        raise Module6ValidationError(f"frontier selection missing required column: {column}")
    return require_numeric_series(df[column], label=f"frontier.{column}", integer=integer)


def _int_flag(df: pd.DataFrame, column: str) -> pd.Series:
    return _required_numeric(df, column, integer=True).astype(bool)


def _nonpositive_float(df: pd.DataFrame, column: str) -> pd.Series:
    return _required_numeric(df, column) <= 0.0


def _strict_live_mask(df: pd.DataFrame, config: Module6Config) -> pd.Series:
    dead = _false_series(df)
    for column in ("session_disable_flag", "minute_disable_flag", "disable_flag"):
        if column in df.columns:
            dead = dead | _int_flag(df, column)
    for column in ("session_breach_count", "minute_breach_count", "breach_count"):
        if column in df.columns:
            dead = dead | _int_flag(df, column)
    for column in ("session_gross_exposure_peak", "minute_gross_exposure_peak", "gross_exposure_peak"):
        if column in df.columns:
            dead = dead | _nonpositive_float(df, column)
    gross_columns = [
        column
        for column in ("minute_average_gross_exposure", "session_average_gross_exposure", "average_gross_exposure")
        if column in df.columns
    ]
    vol_columns = [
        column
        for column in ("minute_realized_volatility", "session_realized_volatility", "realized_volatility")
        if column in df.columns
    ]
    if not gross_columns:
        raise Module6ValidationError("frontier selection requires average gross exposure metrics")
    if not vol_columns:
        raise Module6ValidationError("frontier selection requires realized volatility metrics")
    for column in gross_columns:
        dead = dead | (
            _required_numeric(df, column) < float(config.scoring.min_average_gross_exposure)
        )
    for column in vol_columns:
        dead = dead | (
            _required_numeric(df, column) < float(config.scoring.min_realized_volatility)
        )
    return ~dead


def pareto_frontier(df: pd.DataFrame, maximize: list[str], minimize: list[str]) -> pd.DataFrame:
    if df.shape[0] <= 0:
        return df.copy()
    keep = np.ones(df.shape[0], dtype=bool)
    vals_max = np.asarray(df[maximize], dtype=np.float64) if maximize else np.zeros((df.shape[0], 0), dtype=np.float64)
    vals_min = np.asarray(df[minimize], dtype=np.float64) if minimize else np.zeros((df.shape[0], 0), dtype=np.float64)
    for i in range(df.shape[0]):
        if not keep[i]:
            continue
        dom_max = np.all(vals_max >= vals_max[i], axis=1) if vals_max.shape[1] > 0 else np.ones(df.shape[0], dtype=bool)
        dom_min = np.all(vals_min <= vals_min[i], axis=1) if vals_min.shape[1] > 0 else np.ones(df.shape[0], dtype=bool)
        strict = (
            np.any(vals_max > vals_max[i], axis=1) if vals_max.shape[1] > 0 else np.zeros(df.shape[0], dtype=bool)
        ) | (
            np.any(vals_min < vals_min[i], axis=1) if vals_min.shape[1] > 0 else np.zeros(df.shape[0], dtype=bool)
        )
        dominated = dom_max & dom_min & strict
        dominated[i] = False
        if np.any(dominated):
            keep[i] = False
    return df.loc[keep].copy()


def select_diverse_finalists(
    *,
    scores: pd.DataFrame,
    portfolio_weights: pd.DataFrame,
    strategy_frame: pd.DataFrame,
    config: Module6Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if scores.shape[0] <= 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    require_columns(
        scores,
        [
            "portfolio_pk",
            "cross_universe_reject",
            "comparable_truth_score",
            "final_score",
            "minute_annualized_return",
            "minute_max_drawdown",
            "minute_turnover",
            "availability_burden",
            "minute_average_gross_exposure",
            "minute_realized_volatility",
            "session_average_gross_exposure",
            "session_realized_volatility",
        ],
        "frontier_scores",
    )
    live_mask = _strict_live_mask(scores, config)
    if scores["cross_universe_reject"].isna().any():
        raise Module6ValidationError("frontier selection cross_universe_reject contains missing values")
    eligible_scores = scores.loc[
        (~scores["cross_universe_reject"].astype(bool)) & live_mask
    ].copy()
    if eligible_scores.shape[0] <= 0:
        raise Module6ValidationError("NO_COMPARABLE_FINALISTS_SURVIVED")
    risk_return = pareto_frontier(eligible_scores, maximize=["minute_annualized_return"], minimize=["minute_max_drawdown"])
    operational = pareto_frontier(
        eligible_scores,
        maximize=["minute_average_gross_exposure"],
        minimize=["minute_turnover", "availability_burden"],
    )
    global_frontier = pareto_frontier(
        eligible_scores,
        maximize=["final_score", "minute_annualized_return", "minute_average_gross_exposure"],
        minimize=["minute_max_drawdown", "minute_turnover", "availability_burden"],
    )
    weight_map = {
        pk: grp.set_index("strategy_instance_pk")["target_weight"].sort_index()
        for pk, grp in portfolio_weights.groupby("portfolio_pk", dropna=False, sort=True)
    }
    cluster_map = dict(strategy_frame[["strategy_instance_pk", "cluster_id"]].itertuples(index=False, name=None))
    selected_rows: list[pd.Series] = []
    for row in eligible_scores.itertuples(index=False):
        series = weight_map.get(str(row.portfolio_pk))
        if series is None:
            continue
        clusters = series.groupby(series.index.map(lambda x: cluster_map.get(x, -1))).sum()
        accept = True
        for prev in selected_rows:
            prev_series = weight_map[str(prev.portfolio_pk)]
            union = set(series.index.tolist()) | set(prev_series.index.tolist())
            inter = set(series.index.tolist()) & set(prev_series.index.tolist())
            jaccard = float(len(inter) / max(len(union), 1))
            all_idx = sorted(union)
            a = series.reindex(all_idx, fill_value=0.0).to_numpy(dtype=np.float64)
            b = prev_series.reindex(all_idx, fill_value=0.0).to_numpy(dtype=np.float64)
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            cosine = float(np.dot(a, b) / denom) if denom > 0.0 else 0.0
            prev_clusters = prev_series.groupby(prev_series.index.map(lambda x: cluster_map.get(x, -1))).sum()
            c_union = sorted(set(clusters.index.tolist()) | set(prev_clusters.index.tolist()))
            ca = clusters.reindex(c_union, fill_value=0.0).to_numpy(dtype=np.float64)
            cb = prev_clusters.reindex(c_union, fill_value=0.0).to_numpy(dtype=np.float64)
            c_denom = float(np.linalg.norm(ca) * np.linalg.norm(cb))
            cluster_cos = float(np.dot(ca, cb) / c_denom) if c_denom > 0.0 else 0.0
            if jaccard >= 0.80 or cosine >= 0.95 or cluster_cos >= 0.90:
                accept = False
                break
        if accept:
            selected_rows.append(eligible_scores.loc[eligible_scores["portfolio_pk"] == str(row.portfolio_pk)].iloc[0])
        if len(selected_rows) >= int(config.scoring.final_scalar_keep):
            break
    selected = pd.DataFrame(selected_rows)
    return (
        global_frontier.sort_values(["final_score", "portfolio_pk"], ascending=[False, True], kind="mergesort").reset_index(drop=True),
        risk_return.sort_values(["minute_annualized_return", "portfolio_pk"], ascending=[False, True], kind="mergesort").reset_index(drop=True),
        operational.sort_values(["portfolio_pk"], kind="mergesort").reset_index(drop=True),
        selected.reset_index(drop=True),
    )
