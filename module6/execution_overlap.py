from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse

from module6.config import DependenceConfig
from module6.utils import Module6ValidationError, safe_divide


@dataclass(frozen=True)
class ExecutionOverlapComponents:
    symbol_support: sparse.csr_matrix
    activity_concurrence: sparse.csr_matrix
    gross_exposure_concurrence: sparse.csr_matrix
    rebalance_collision: sparse.csr_matrix
    composite: sparse.csr_matrix


def bucketize_activity_signature(turnover_matrix: np.ndarray, buckets: int) -> np.ndarray:
    x = np.asarray(turnover_matrix, dtype=np.float64)
    if x.ndim != 2:
        raise Module6ValidationError("turnover_matrix must be 2D for activity signatures")
    if buckets <= 1:
        raise Module6ValidationError("activity signature buckets must be > 1")
    out = np.zeros(x.shape, dtype=np.uint8)
    for col in range(x.shape[1]):
        vals = x[:, col]
        pos = vals[vals > 0.0]
        if pos.size <= 0:
            continue
        cuts = np.quantile(pos, np.linspace(0.0, 1.0, buckets + 1))
        out[:, col] = np.asarray(np.digitize(vals, cuts[1:-1], right=True), dtype=np.uint8)
    return out


def rebalance_signature(turnover_matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(turnover_matrix, dtype=np.float64)
    sig = np.zeros(x.shape, dtype=np.int8)
    for col in range(x.shape[1]):
        vals = x[:, col]
        pos = vals[vals > 0.0]
        if pos.size <= 0:
            continue
        threshold = float(np.quantile(pos, 0.90))
        sig[:, col] = (vals >= threshold).astype(np.int8)
    return sig


def _pairwise_sparse_from_pairs(
    n_count: int,
    pairs: Iterable[tuple[int, int, float]],
) -> sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for i, j, score in pairs:
        if i == j or score <= 0.0:
            continue
        rows.extend([i, j])
        cols.extend([j, i])
        vals.extend([float(score), float(score)])
    return sparse.csr_matrix((np.asarray(vals, dtype=np.float64), (rows, cols)), shape=(n_count, n_count))


def build_execution_overlap_proxy(
    *,
    instance_rows: pd.DataFrame,
    trade_log: pd.DataFrame,
    turnover_matrix: np.ndarray,
    gross_peak_matrix: np.ndarray,
    config: DependenceConfig,
    candidate_pairs: list[tuple[int, int]] | None = None,
) -> ExecutionOverlapComponents:
    required_instance_cols = {"strategy_instance_pk", "candidate_id", "split_id", "scenario_id"}
    if not required_instance_cols.issubset(set(instance_rows.columns)):
        raise Module6ValidationError("instance_rows missing required columns for execution overlap proxy")
    ordered = instance_rows.reset_index(drop=True).copy()
    strategy_instance_pks = ordered["strategy_instance_pk"].astype(str).tolist()
    instance_keys = (
        ordered["candidate_id"].astype(str)
        + "|"
        + ordered["split_id"].astype(str)
        + "|"
        + ordered["scenario_id"].astype(str)
    ).tolist()
    n_count = len(strategy_instance_pks)
    pk_to_instance_key = {str(pk): str(key) for pk, key in zip(strategy_instance_pks, instance_keys)}
    if candidate_pairs is None:
        candidate_pairs = [(i, j) for i in range(n_count) for j in range(i + 1, n_count)]

    required_cols = {"candidate_id", "split_id", "scenario_id", "symbol"}
    if not required_cols.issubset(set(trade_log.columns)):
        raise Module6ValidationError("trade_log missing required columns for execution overlap proxy")

    trade_log = trade_log.copy()
    trade_log["strategy_instance_key"] = (
        trade_log["candidate_id"].astype(str)
        + "|"
        + trade_log["split_id"].astype(str)
        + "|"
        + trade_log["scenario_id"].astype(str)
    )
    symbol_support_map = (
        trade_log.groupby("strategy_instance_key", dropna=False)["symbol"]
        .agg(lambda s: tuple(sorted(pd.unique(s.astype(str)).tolist())))
        .to_dict()
    )
    if not symbol_support_map:
        raise Module6ValidationError("execution overlap proxy requires symbol support map")
    activity = (np.asarray(turnover_matrix, dtype=np.float64) > 0.0).astype(np.int8)
    gross = np.asarray(gross_peak_matrix, dtype=np.float64)
    rb = rebalance_signature(np.asarray(turnover_matrix, dtype=np.float64))
    pairs_symbol: list[tuple[int, int, float]] = []
    pairs_activity: list[tuple[int, int, float]] = []
    pairs_gross: list[tuple[int, int, float]] = []
    pairs_rb: list[tuple[int, int, float]] = []
    pairs_comp: list[tuple[int, int, float]] = []

    for i, j in candidate_pairs:
        key_i = str(strategy_instance_pks[i])
        key_j = str(strategy_instance_pks[j])
        sym_i = set(symbol_support_map.get(pk_to_instance_key[key_i], ()))
        sym_j = set(symbol_support_map.get(pk_to_instance_key[key_j], ()))
        if not sym_i or not sym_j:
            raise Module6ValidationError("execution overlap proxy input missing symbol support")
        score_symbol = float(len(sym_i & sym_j) / max(len(sym_i | sym_j), 1))
        act_i = activity[:, i].astype(bool)
        act_j = activity[:, j].astype(bool)
        union = int(np.sum(act_i | act_j))
        score_activity = float(np.sum(act_i & act_j) / union) if union > 0 else 0.0
        gross_i = gross[:, i]
        gross_j = gross[:, j]
        denom = float(np.linalg.norm(gross_i) * np.linalg.norm(gross_j))
        score_gross = float(np.dot(gross_i, gross_j) / denom) if denom > 0.0 else 0.0
        rb_i = rb[:, i].astype(bool)
        rb_j = rb[:, j].astype(bool)
        rb_union = int(np.sum(rb_i | rb_j))
        score_rb = float(np.sum(rb_i & rb_j) / rb_union) if rb_union > 0 else 0.0
        score_comp = (
            float(config.overlap_weight_symbol_support) * score_symbol
            + float(config.overlap_weight_activity) * score_activity
            + float(config.overlap_weight_gross) * score_gross
            + float(config.overlap_weight_rebalance) * score_rb
        )
        pairs_symbol.append((i, j, score_symbol))
        pairs_activity.append((i, j, score_activity))
        pairs_gross.append((i, j, score_gross))
        pairs_rb.append((i, j, score_rb))
        pairs_comp.append((i, j, score_comp))

    return ExecutionOverlapComponents(
        symbol_support=_pairwise_sparse_from_pairs(n_count, pairs_symbol),
        activity_concurrence=_pairwise_sparse_from_pairs(n_count, pairs_activity),
        gross_exposure_concurrence=_pairwise_sparse_from_pairs(n_count, pairs_gross),
        rebalance_collision=_pairwise_sparse_from_pairs(n_count, pairs_rb),
        composite=_pairwise_sparse_from_pairs(n_count, pairs_comp),
    )


def finalist_exact_overlap(
    trade_log: pd.DataFrame,
    instance_rows: pd.DataFrame,
) -> pd.DataFrame:
    if trade_log.shape[0] <= 0 or instance_rows.shape[0] <= 1:
        return pd.DataFrame(columns=["strategy_instance_pk_i", "strategy_instance_pk_j", "exact_overlap_notional"])
    rows = instance_rows.copy()
    rows["strategy_instance_key"] = rows["candidate_id"].astype(str) + "|" + rows["split_id"].astype(str) + "|" + rows["scenario_id"].astype(str)
    key_to_pk = dict(rows[["strategy_instance_key", "strategy_instance_pk"]].itertuples(index=False, name=None))
    trade_df = trade_log.copy()
    trade_df["strategy_instance_key"] = (
        trade_df["candidate_id"].astype(str) + "|" + trade_df["split_id"].astype(str) + "|" + trade_df["scenario_id"].astype(str)
    )
    trade_df = trade_df.loc[trade_df["strategy_instance_key"].isin(key_to_pk.keys())].copy()
    trade_df["strategy_instance_pk"] = trade_df["strategy_instance_key"].map(key_to_pk)
    trade_df["notional"] = np.abs(np.asarray(trade_df["filled_qty"], dtype=np.float64) * np.asarray(trade_df["exec_price"], dtype=np.float64))
    grouped = trade_df.groupby(["ts_ns", "symbol", "strategy_instance_pk"], dropna=False)["notional"].sum().reset_index()
    out_rows: list[dict[str, object]] = []
    for (_, _), grp in grouped.groupby(["ts_ns", "symbol"], dropna=False):
        pairs = grp[["strategy_instance_pk", "notional"]].to_dict("records")
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                out_rows.append(
                    {
                        "strategy_instance_pk_i": str(pairs[i]["strategy_instance_pk"]),
                        "strategy_instance_pk_j": str(pairs[j]["strategy_instance_pk"]),
                        "exact_overlap_notional": float(min(float(pairs[i]["notional"]), float(pairs[j]["notional"]))),
                    }
                )
    if not out_rows:
        return pd.DataFrame(columns=["strategy_instance_pk_i", "strategy_instance_pk_j", "exact_overlap_notional"])
    return pd.DataFrame(out_rows).groupby(
        ["strategy_instance_pk_i", "strategy_instance_pk_j"], dropna=False
    )["exact_overlap_notional"].sum().reset_index()
