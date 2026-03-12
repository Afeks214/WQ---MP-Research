from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from weightiz.module6.config import (
    MODULE6_RUN_POLICY_REPRESENTATIVE_DISCOVERY,
    Module6Config,
    resolve_intake_gate_thresholds,
)
from weightiz.module6.execution_overlap import build_execution_overlap_proxy
from weightiz.module6.types import ReducedUniverseSpec
from weightiz.module6.utils import Module6ValidationError, stable_sha256_parts


@dataclass(frozen=True)
class ReductionArtifacts:
    admitted_instances: pd.DataFrame
    cluster_membership: pd.DataFrame
    reduced_universes: list[ReducedUniverseSpec]
    overlap_proxy: object


def _standardize_columns(matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    if not np.isfinite(x).all():
        raise Module6ValidationError("reduction received non-finite returns matrix")
    mu = np.mean(x, axis=0, keepdims=True)
    sd = np.std(x, axis=0, ddof=1, keepdims=True)
    sd = np.where(sd > 1.0e-12, sd, 1.0)
    z = (x - mu) / sd
    z = np.clip(z, -1.0e6, 1.0e6)
    z = z / np.maximum(np.linalg.norm(z, axis=0, keepdims=True), 1.0)
    if not np.isfinite(z).all():
        raise Module6ValidationError("reduction standardization produced non-finite values")
    return z


def _projection_neighbors(z: np.ndarray, width: int, seed: int, top_k: int) -> list[tuple[int, int]]:
    rng = np.random.default_rng(int(seed))
    proj = rng.standard_normal((z.shape[0], int(width))) / max(np.sqrt(float(max(z.shape[0], 1))), 1.0)
    emb = np.zeros((z.shape[1], int(width)), dtype=np.float64)
    for w_idx in range(int(width)):
        emb[:, w_idx] = np.sum(z * proj[:, w_idx][:, None], axis=0, dtype=np.float64)
    if not np.isfinite(emb).all():
        raise Module6ValidationError("reduction projection embedding produced non-finite values")
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


def _build_overlap_trade_log(
    *,
    instance_rows: pd.DataFrame,
    trade_log: pd.DataFrame,
    intake_policy_class: str,
) -> pd.DataFrame:
    overlap_trade_log = trade_log.copy()
    if str(intake_policy_class) != MODULE6_RUN_POLICY_REPRESENTATIVE_DISCOVERY:
        return overlap_trade_log
    required_cols = ["candidate_id", "split_id", "scenario_id", "symbol"]
    if not set(required_cols).issubset(set(overlap_trade_log.columns)):
        return overlap_trade_log
    overlap_trade_log = overlap_trade_log[required_cols].copy()
    overlap_trade_log["instance_key"] = overlap_trade_log[required_cols[:3]].astype(str).agg("|".join, axis=1)
    instance_rows = instance_rows[required_cols[:3]].drop_duplicates().copy()
    instance_rows["instance_key"] = instance_rows.astype(str).agg("|".join, axis=1)
    direct_support_keys = set(overlap_trade_log["instance_key"].astype(str).tolist())
    missing = instance_rows.loc[~instance_rows["instance_key"].isin(direct_support_keys)].copy()
    if missing.shape[0] <= 0:
        return overlap_trade_log.drop(columns=["instance_key"])
    candidate_symbol_map = (
        overlap_trade_log.groupby("candidate_id", dropna=False)["symbol"]
        .agg(lambda s: tuple(sorted(pd.unique(s.astype(str)).tolist())))
        .to_dict()
    )
    alias_rows: list[dict[str, object]] = []
    unresolved_candidates: list[str] = []
    for row in missing.itertuples(index=False):
        candidate_id = str(row.candidate_id)
        symbols = tuple(candidate_symbol_map.get(candidate_id, ()))
        if not symbols:
            unresolved_candidates.append(candidate_id)
            continue
        for symbol in symbols:
            alias_rows.append(
                {
                    "candidate_id": candidate_id,
                    "split_id": str(row.split_id),
                    "scenario_id": str(row.scenario_id),
                    "symbol": str(symbol),
                }
            )
    if unresolved_candidates:
        unresolved_unique = sorted(set(unresolved_candidates))
        raise Module6ValidationError(
            "representative discovery overlap proxy missing candidate-level symbol support: "
            + ",".join(unresolved_unique[:10])
        )
    if alias_rows:
        overlap_trade_log = pd.concat(
            [overlap_trade_log.drop(columns=["instance_key"]), pd.DataFrame(alias_rows)],
            ignore_index=True,
        ).drop_duplicates(required_cols, keep="first")
    else:
        overlap_trade_log = overlap_trade_log.drop(columns=["instance_key"])
    return overlap_trade_log


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
    try:
        intake_policy_class, min_availability_ratio, min_observed_sessions = resolve_intake_gate_thresholds(
            config.intake
        )
    except ValueError as exc:
        raise Module6ValidationError(str(exc)) from exc
    if "module6_policy_class" not in strategy_master.columns:
        raise Module6ValidationError("strategy_master missing module6_policy_class")
    observed_policy_classes = {
        str(value).strip().lower()
        for value in pd.unique(strategy_master["module6_policy_class"]).tolist()
    }
    if observed_policy_classes != {str(intake_policy_class)}:
        raise Module6ValidationError(
            "strategy_master policy class mismatch for reduction intake: "
            f"configured={intake_policy_class} observed={sorted(observed_policy_classes)}"
        )
    reject_gate = (
        ~strategy_master["reject"].fillna(False).astype(bool)
        if str(intake_policy_class) == "standard"
        else pd.Series(True, index=strategy_master.index, dtype=bool)
    )
    admitted = strategy_master.loc[
        strategy_master["portfolio_admit_flag"].astype(bool)
        & (~strategy_master["failed"].fillna(False).astype(bool))
        & reject_gate
        & (pd.to_numeric(strategy_master["availability_ratio"], errors="coerce").fillna(0.0) >= float(min_availability_ratio))
        & (pd.to_numeric(strategy_master["observed_session_count"], errors="coerce").fillna(0).astype(int) >= int(min_observed_sessions))
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
    keep_mask = support_ratio >= float(min_availability_ratio)
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
    canonical_instances["pre_reduction_rank"] = canonical_instances.apply(
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
    if canonical_instances.shape[0] > int(config.reduction.pre_reduction_cap):
        general_keep = canonical_instances.sort_values(
            ["pre_reduction_rank"], ascending=False, kind="mergesort"
        ).head(max(int(config.reduction.pre_reduction_cap) - int(config.reduction.hedge_keep_count), 1))
        hedge_keep = canonical_instances.sort_values(
            ["cost_adjusted_expectancy", "strategy_instance_pk"],
            ascending=[True, True],
            kind="mergesort",
        ).head(int(config.reduction.hedge_keep_count))
        keep_ids = sorted(set(general_keep["strategy_instance_pk"].astype(str).tolist()) | set(hedge_keep["strategy_instance_pk"].astype(str).tolist()))
        canonical_instances = canonical_instances.loc[canonical_instances["strategy_instance_pk"].isin(keep_ids)].reset_index(drop=True)
    column_idx = canonical_instances["column_idx"].to_numpy(dtype=np.int64)
    r_sub = r_exec[:, column_idx]
    z = _standardize_columns(r_sub)
    pairs = _projection_neighbors(
        z,
        width=int(config.reduction.projection_width),
        seed=int(config.runtime.random_projection_seed),
        top_k=int(config.reduction.ann_top_k),
    )
    overlap_trade_log = _build_overlap_trade_log(
        instance_rows=canonical_instances[["candidate_id", "split_id", "scenario_id"]],
        trade_log=run.trade_log,
        intake_policy_class=str(intake_policy_class),
    )
    overlap = build_execution_overlap_proxy(
        instance_rows=canonical_instances[["strategy_instance_pk", "candidate_id", "split_id", "scenario_id"]],
        trade_log=overlap_trade_log,
        turnover_matrix=u[:, column_idx],
        gross_peak_matrix=gross_peak[:, column_idx],
        config=config.dependence,
        candidate_pairs=pairs,
    )
    canonical_instances["overlap_proxy_idx"] = np.arange(canonical_instances.shape[0], dtype=np.int64)
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
    ranked_final = (
        canonical_instances.loc[
            canonical_instances["strategy_instance_pk"].isin(sorted(set(reduced_keep + retained_hedges)))
        ]
        .sort_values(["representative_rank_tuple"], ascending=False, kind="mergesort")
        ["strategy_instance_pk"]
        .astype(str)
        .tolist()
    )
    reduced_final = tuple(ranked_final[: int(config.reduction.reduced_universe_cap)])
    mv_final = tuple(ranked_final[: min(int(config.reduction.mv_universe_cap), len(ranked_final))])
    membership_df["retained_in_reduced_universe"] = membership_df["strategy_instance_pk"].astype(str).isin(set(reduced_final))
    spec = ReducedUniverseSpec(
        reduced_universe_id="reduced_universe_000",
        strategy_instance_pks=reduced_final,
        representative_strategy_instance_pks=tuple(sorted(set(representative_pks))),
        retained_hedge_strategy_instance_pks=tuple(sorted(set(pk for pk in retained_hedges if pk in reduced_final))),
        cluster_count=int(membership_df["cluster_id"].nunique()),
        metadata={"n_admitted": int(canonical_instances.shape[0]), "pre_reduction_cap": int(config.reduction.pre_reduction_cap), "reduced_universe_cap": int(config.reduction.reduced_universe_cap)},
    )
    mv_spec = ReducedUniverseSpec(
        reduced_universe_id="reduced_universe_mv_000",
        strategy_instance_pks=mv_final,
        representative_strategy_instance_pks=tuple(sorted(set(pk for pk in representative_pks if pk in mv_final))),
        retained_hedge_strategy_instance_pks=tuple(sorted(set(pk for pk in retained_hedges if pk in mv_final))),
        cluster_count=int(membership_df["cluster_id"].nunique()),
        metadata={"n_admitted": int(canonical_instances.shape[0]), "mv_universe_cap": int(config.reduction.mv_universe_cap)},
    )
    out_dir = output_dir / "reduced_universes"
    out_dir.mkdir(parents=True, exist_ok=True)
    membership_df.to_parquet(out_dir / "reduced_universe_000_membership.parquet", index=False)
    membership_mv_df = membership_df.copy()
    membership_mv_df["reduced_universe_id"] = "reduced_universe_mv_000"
    membership_mv_df["retained_in_reduced_universe"] = membership_mv_df["strategy_instance_pk"].astype(str).isin(set(mv_final))
    membership_mv_df.to_parquet(out_dir / "reduced_universe_mv_000_membership.parquet", index=False)
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
    pd.DataFrame(
        {
            "reduced_universe_id": [mv_spec.reduced_universe_id],
            "strategy_instance_pks": [list(mv_spec.strategy_instance_pks)],
            "representative_strategy_instance_pks": [list(mv_spec.representative_strategy_instance_pks)],
            "retained_hedge_strategy_instance_pks": [list(mv_spec.retained_hedge_strategy_instance_pks)],
            "cluster_count": [mv_spec.cluster_count],
            "metadata": [mv_spec.metadata],
        }
    ).to_parquet(out_dir / "reduced_universe_mv_000.parquet", index=False)
    return ReductionArtifacts(
        admitted_instances=canonical_instances,
        cluster_membership=membership_df,
        reduced_universes=[spec, mv_spec],
        overlap_proxy=overlap,
    )
