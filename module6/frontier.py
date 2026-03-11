from __future__ import annotations

import numpy as np
import pandas as pd

from module6.config import Module6Config


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
    risk_return = pareto_frontier(scores, maximize=["minute_annualized_return"], minimize=["minute_max_drawdown"])
    operational = pareto_frontier(scores, maximize=["headroom"] if "headroom" in scores.columns else [], minimize=["minute_turnover", "availability_burden"])
    global_frontier = pareto_frontier(
        scores,
        maximize=["final_score", "minute_annualized_return"],
        minimize=["minute_max_drawdown", "minute_turnover", "availability_burden"],
    )
    weight_map = {
        pk: grp.set_index("strategy_instance_pk")["target_weight"].sort_index()
        for pk, grp in portfolio_weights.groupby("portfolio_pk", dropna=False, sort=True)
    }
    cluster_map = dict(strategy_frame[["strategy_instance_pk", "cluster_id"]].itertuples(index=False, name=None))
    selected_rows: list[pd.Series] = []
    for row in scores.itertuples(index=False):
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
            selected_rows.append(scores.loc[scores["portfolio_pk"] == str(row.portfolio_pk)].iloc[0])
        if len(selected_rows) >= int(config.scoring.final_scalar_keep):
            break
    selected = pd.DataFrame(selected_rows)
    return (
        global_frontier.sort_values(["final_score", "portfolio_pk"], ascending=[False, True], kind="mergesort").reset_index(drop=True),
        risk_return.sort_values(["minute_annualized_return", "portfolio_pk"], ascending=[False, True], kind="mergesort").reset_index(drop=True),
        operational.sort_values(["portfolio_pk"], kind="mergesort").reset_index(drop=True),
        selected.reset_index(drop=True),
    )
