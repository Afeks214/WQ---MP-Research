from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from module6.config import Module6Config
from module6.execution_overlap import build_execution_overlap_proxy
from module6.types import ReducedUniverseSpec
from module6.utils import Module6ValidationError, stable_sha256_parts


@dataclass(frozen=True)
class ReductionArtifacts:
    admitted_instances: pd.DataFrame
    cluster_membership: pd.DataFrame
    reduced_universes: list[ReducedUniverseSpec]
    overlap_proxy: object


def _standardize_columns(matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    mu = np.mean(x, axis=0, keepdims=True)
    sd = np.std(x, axis=0, ddof=1, keepdims=True)
    sd = np.where(sd > 1.0e-12, sd, 1.0)
    return (x - mu) / sd


def _projection_neighbors(z: np.ndarray, width: int, seed: int, top_k: int) -> list[tuple[int, int]]:
    rng = np.random.default_rng(int(seed))
    proj = rng.standard_normal((z.shape[0], int(width)))
    emb = np.asarray(z.T @ proj, dtype=np.float64)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.maximum(norms, 1.0e-12)
    pairs: set[tuple[int, int]] = set()
    block = 128
    for start in range(0, emb.shape[0], block):
        sim = emb[start : start + block] @ emb.T
        for local_idx in range(sim.shape[0]):
            row_idx = start + local_idx
            row = sim[local_idx]
            nn = np.argsort(row, kind="mergesort")[::-1]
            kept = 0
            for idx in nn:
                if int(idx) == int(row_idx):
                    continue
                i, j = sorted((int(row_idx), int(idx)))
                pairs.add((i, j))
                kept += 1
                if kept >= int(top_k):
                    break
    return sorted(pairs)


def reduce_universe(
    *,
    ledgers: dict[str, pd.DataFrame],
    matrices: dict[str, np.ndarray | object],
    run,
    output_dir,
    config: Module6Config,
) -> ReductionArtifacts:
    strategy_master = ledgers["strategy_master"].copy()
    instance_master = ledgers["strategy_instance_master"].copy()
    session_ledger = ledgers["strategy_session_ledger"].copy()
    admitted = strategy_master.loc[
        strategy_master["portfolio_admit_flag"].astype(bool)
        & (~strategy_master["failed"].fillna(False).astype(bool))
        & (~strategy_master["reject"].fillna(False).astype(bool))
        & (pd.to_numeric(strategy_master["availability_ratio"], errors="coerce").fillna(0.0) >= float(config.intake.min_availability_ratio))
        & (pd.to_numeric(strategy_master["observed_session_count"], errors="coerce").fillna(0).astype(int) >= int(config.intake.min_observed_sessions))
        & (pd.to_numeric(strategy_master["avg_turnover_metrics"], errors="coerce").fillna(0.0) >= 0.0)
    ].copy()
    if admitted.shape[0] <= 0:
        raise Module6ValidationError("no admitted strategies survived pre-reduction intake gates")
    canonical_instances = instance_master.loc[
        instance_master["portfolio_instance_role"] == "canonical_portfolio"
    ].copy()
    canonical_instances = canonical_instances.loc[
        canonical_instances["strategy_pk"].isin(admitted["strategy_pk"])
    ].sort_values(["strategy_instance_pk"], kind="mergesort")
    column_index = pd.DataFrame(matrices["column_index"]).copy() if "column_index" in matrices else None
    if column_index is None:
        raise Module6ValidationError("matrix column_index missing from reduction inputs")
    canonical_instances = canonical_instances.merge(
        column_index[["column_idx", "strategy_instance_pk"]],
        on="strategy_instance_pk",
        how="inner",
    )
    if canonical_instances.shape[0] <= 0:
        raise Module6ValidationError("no canonical instances matched matrix columns")
    r_exec = np.asarray(matrices["R_exec"], dtype=np.float64)
    a = np.asarray(matrices["A"], dtype=bool)
    u = np.asarray(matrices["U"], dtype=np.float64)
    gross_peak = np.asarray(matrices["gross_peak"], dtype=np.float64)
    column_idx = canonical_instances["column_idx"].to_numpy(dtype=np.int64)
    r_sub = r_exec[:, column_idx]
    a_sub = a[:, column_idx]
    support_ratio = np.mean(a_sub, axis=0)
    keep_mask = support_ratio >= float(config.intake.min_availability_ratio)
    canonical_instances = canonical_instances.loc[keep_mask].reset_index(drop=True)
    column_idx = canonical_instances["column_idx"].to_numpy(dtype=np.int64)
    r_sub = r_exec[:, column_idx]
    if r_sub.shape[1] <= 0:
        raise Module6ValidationError("all strategies removed by availability gate during reduction")
    signature = []
    for j in range(r_sub.shape[1]):
        signature.append(stable_sha256_parts(*np.round(r_sub[:, j], 12).tolist()))
    canonical_instances["return_signature"] = signature
    canonical_instances = canonical_instances.sort_values(
        ["parameter_hash", "return_signature", "strategy_instance_pk"], kind="mergesort"
    ).drop_duplicates(["parameter_hash", "return_signature"], keep="first").reset_index(drop=True)
    column_idx = canonical_instances["column_idx"].to_numpy(dtype=np.int64)
    r_sub = r_exec[:, column_idx]
    z = _standardize_columns(r_sub)
    pairs = _projection_neighbors(
        z,
        width=int(config.reduction.projection_width),
        seed=int(config.runtime.random_projection_seed),
        top_k=int(config.reduction.ann_top_k),
    )
    overlap = build_execution_overlap_proxy(
        instance_rows=canonical_instances[["strategy_instance_pk", "candidate_id", "split_id", "scenario_id"]],
        trade_log=run.trade_log,
        turnover_matrix=u[:, column_idx],
        gross_peak_matrix=gross_peak[:, column_idx],
        config=config.dependence,
        candidate_pairs=pairs,
    )
    parent = list(range(len(column_idx)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(i: int, j: int) -> None:
        ri = find(i)
        rj = find(j)
        if ri != rj:
            parent[rj] = ri

    for i, j in pairs:
        xi = r_sub[:, i]
        xj = r_sub[:, j]
        corr = float(np.corrcoef(xi, xj)[0, 1]) if np.std(xi) > 0.0 and np.std(xj) > 0.0 else 0.0
        equity = np.cumprod(1.0 + np.vstack([xi, xj]).T, axis=0)
        dd = (equity - np.maximum.accumulate(equity, axis=0)) / np.maximum(np.maximum.accumulate(equity, axis=0), 1.0e-12)
        tail_i = dd[:, 0] <= np.quantile(dd[:, 0], 0.05)
        tail_j = dd[:, 1] <= np.quantile(dd[:, 1], 0.05)
        union_ct = int(np.sum(tail_i | tail_j))
        concurrence = float(np.sum(tail_i & tail_j) / union_ct) if union_ct > 0 else 0.0
        if corr >= float(config.reduction.duplicate_corr_threshold) and concurrence >= float(config.reduction.drawdown_concurrence_threshold):
            union(i, j)

    cluster_ids = np.asarray([find(i) for i in range(len(column_idx))], dtype=np.int64)
    cluster_map = {cid: rank for rank, cid in enumerate(sorted(pd.unique(cluster_ids).tolist()))}
    canonical_instances["cluster_id"] = np.asarray([cluster_map[int(x)] for x in cluster_ids], dtype=np.int64)
    rank_tuple = canonical_instances.apply(
        lambda row: (
            int(bool(row["portfolio_admit_flag"])),
            float(row["robustness_score"]),
            float(row["availability_ratio"]),
            -float(row["avg_turnover_metrics"]),
            -float(row["constraint_flag_count"]),
            str(row["strategy_instance_pk"]),
        ),
        axis=1,
    )
    canonical_instances["representative_rank_tuple"] = rank_tuple
    membership_rows: list[dict[str, object]] = []
    representative_pks: list[str] = []
    for cluster_id, grp in canonical_instances.groupby("cluster_id", dropna=False, sort=True):
        grp = grp.sort_values(["representative_rank_tuple"], ascending=False, kind="mergesort")
        representative = grp.iloc[0]
        representative_pks.append(str(representative["strategy_instance_pk"]))
        erank = 1
        if grp.shape[0] > 1:
            local = np.asarray(r_exec[:, grp["column_idx"].to_numpy(dtype=np.int64)], dtype=np.float64)
            cov = np.cov(local, rowvar=False)
            if cov.ndim == 0:
                erank = 1
            else:
                eig = np.linalg.eigvalsh(np.asarray(cov, dtype=np.float64))
                eig = eig[eig > 1.0e-18]
                if eig.size > 0:
                    p = eig / np.sum(eig)
                    erank = int(max(1, np.floor(np.exp(-np.sum(p * np.log(np.maximum(p, 1.0e-18)))))))
        cluster_cap = int(min(3, max(1, erank)))
        keep_grp = grp.head(cluster_cap)
        for row in grp.itertuples(index=False):
            membership_rows.append(
                {
                    "reduced_universe_id": "reduced_universe_000",
                    "cluster_id": int(cluster_id),
                    "strategy_pk": str(row.strategy_pk),
                    "strategy_instance_pk": str(row.strategy_instance_pk),
                    "candidate_id": str(row.candidate_id),
                    "family_id": str(row.family_id),
                    "representative_strategy_instance_pk": str(representative["strategy_instance_pk"]),
                    "representative_candidate_id": str(representative["candidate_id"]),
                    "cluster_cap": int(cluster_cap),
                    "retained_in_reduced_universe": bool(str(row.strategy_instance_pk) in set(keep_grp["strategy_instance_pk"].astype(str).tolist())),
                }
            )
    membership_df = pd.DataFrame(membership_rows).sort_values(
        ["cluster_id", "strategy_instance_pk"], kind="mergesort"
    ).reset_index(drop=True)
    reduced_keep = membership_df.loc[membership_df["retained_in_reduced_universe"].astype(bool), "strategy_instance_pk"].astype(str).tolist()
    excluded = canonical_instances.loc[~canonical_instances["strategy_instance_pk"].isin(reduced_keep)].copy()
    rep_idx = canonical_instances.loc[canonical_instances["strategy_instance_pk"].isin(representative_pks), ["strategy_instance_pk", "column_idx"]]
    retained_hedges: list[str] = []
    if excluded.shape[0] > 0 and rep_idx.shape[0] > 0:
        rep_cols = rep_idx["column_idx"].to_numpy(dtype=np.int64)
        rep_ret = np.asarray(r_exec[:, rep_cols], dtype=np.float64)
        hedge_scores: list[tuple[float, str]] = []
        for row in excluded.itertuples(index=False):
            col = int(row.column_idx)
            vals = np.asarray(r_exec[:, col], dtype=np.float64)
            corr = np.asarray([
                np.corrcoef(vals, rep_ret[:, j])[0, 1] if np.std(vals) > 0.0 and np.std(rep_ret[:, j]) > 0.0 else 0.0
                for j in range(rep_ret.shape[1])
            ], dtype=np.float64)
            hedge_scores.append((float(np.min(corr)), str(row.strategy_instance_pk)))
        hedge_scores = sorted(hedge_scores, key=lambda x: (x[0], x[1]))
        retained_hedges = [pk for _, pk in hedge_scores[: int(config.reduction.hedge_keep_count)]]
    reduced_final = tuple(sorted(set(reduced_keep + retained_hedges)))
    spec = ReducedUniverseSpec(
        reduced_universe_id="reduced_universe_000",
        strategy_instance_pks=reduced_final,
        representative_strategy_instance_pks=tuple(sorted(set(representative_pks))),
        retained_hedge_strategy_instance_pks=tuple(sorted(set(retained_hedges))),
        cluster_count=int(membership_df["cluster_id"].nunique()),
        metadata={"n_admitted": int(canonical_instances.shape[0])},
    )
    out_dir = output_dir / "reduced_universes"
    out_dir.mkdir(parents=True, exist_ok=True)
    membership_df.to_parquet(out_dir / "reduced_universe_000_membership.parquet", index=False)
    pd.DataFrame(
        {
            "reduced_universe_id": [spec.reduced_universe_id],
            "strategy_instance_pks": [list(spec.strategy_instance_pks)],
            "representative_strategy_instance_pks": [list(spec.representative_strategy_instance_pks)],
            "retained_hedge_strategy_instance_pks": [list(spec.retained_hedge_strategy_instance_pks)],
            "cluster_count": [spec.cluster_count],
            "metadata": [spec.metadata],
        }
    ).to_parquet(out_dir / "reduced_universe_000.parquet", index=False)
    return ReductionArtifacts(
        admitted_instances=canonical_instances,
        cluster_membership=membership_df,
        reduced_universes=[spec],
        overlap_proxy=overlap,
    )
