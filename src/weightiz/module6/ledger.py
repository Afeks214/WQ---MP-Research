from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from weightiz.module6.config import Module6Config
from weightiz.module6.constants import BASE_AVAIL_ALLOWED_CODES
from weightiz.module6.io import LoadedModule5Run
from weightiz.module6.utils import Module6ValidationError, assert_no_duplicates, count_flag_tokens, ensure_directory, stable_sha256_parts


def _enabled_assets_hash(enabled_assets_mask: list[bool] | tuple[bool, ...]) -> str:
    payload = "|".join("1" if bool(x) else "0" for x in enabled_assets_mask).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _metric_scalar(payload: Any, key: str, default: float) -> float:
    if not isinstance(payload, dict):
        return float(default)
    raw = payload.get(key, default)
    if isinstance(raw, (list, tuple)):
        arr = np.asarray(raw, dtype=np.float64).reshape(-1)
        finite = arr[np.isfinite(arr)]
        return float(finite[0]) if finite.size > 0 else float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def build_strategy_instance_master(run: LoadedModule5Run, config: Module6Config) -> pd.DataFrame:
    run_id = str(run.run_manifest["run_id"])
    dataset_hash = str(run.run_manifest["dataset_hash"])
    selection = run.strategy_instance_selection.copy()
    leaderboard = run.leaderboard.copy()
    candidate_cfg_rows: list[dict[str, Any]] = []
    candidates_root = run.paths.run_dir / "candidates"
    for cdir in sorted(candidates_root.iterdir()):
        if not cdir.is_dir():
            continue
        cfg_path = cdir / "candidate_config.json"
        metrics_path = cdir / "candidate_metrics.json"
        stats_path = cdir / "candidate_stats.json"
        if not cfg_path.exists() or not metrics_path.exists() or not stats_path.exists():
            raise Module6ValidationError(f"candidate artifact set incomplete for {cdir.name}")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        stage_meta = dict(cfg.get("stage_a_metadata", {}))
        base_metrics = dict(metrics.get("base_metrics", {}))
        robustness = dict(metrics.get("robustness", {}))
        dq_summary = dict(metrics.get("dq_summary", {}))
        candidate_cfg_rows.append(
            {
                "candidate_id": str(cfg["candidate_id"]),
                "enabled_assets_hash": _enabled_assets_hash(list(cfg.get("enabled_assets_mask", []))),
                "parameter_hash_cfg": str(stage_meta.get("parameter_hash", "")),
                "campaign_id": str(stage_meta.get("campaign_id", "")),
                "family_id_cfg": str(stage_meta.get("family_id", "")),
                "family_name_cfg": str(stage_meta.get("family_name", "")),
                "hypothesis_id_cfg": str(stage_meta.get("hypothesis_id", "")),
                "evaluation_role_cfg": str(stage_meta.get("evaluation_role", "")),
                "evaluation_window_cfg": stage_meta.get("evaluation_window"),
                "window_set_cfg": str(stage_meta.get("window_set", "")),
                "window_set_size_cfg": int(stage_meta.get("window_set_size", 0)),
                "tags_serialized_cfg": str(stage_meta.get("tags_serialized", "")),
                "n_days_metrics": int(base_metrics.get("n_days", 0)),
                "avg_turnover_metrics": float(base_metrics.get("avg_turnover", 0.0)),
                "avg_margin_used_frac_metrics": float(base_metrics.get("avg_margin_used_frac", 0.0)),
                "peak_margin_used_frac_metrics": float(base_metrics.get("peak_margin_used_frac", 0.0)),
                "candidate_metrics_failed": bool(metrics.get("failed", False)),
                "candidate_failure_reasons": "|".join(str(x) for x in metrics.get("failure_reasons", [])),
                "dq_min_metrics": float(dq_summary.get("dq_min", 1.0)),
                "dq_median_metrics": float(dq_summary.get("dq_median", 1.0)),
                "dq_degrade_count_metrics": int(dq_summary.get("dq_degrade_count", 0)),
                "dq_reject_count_metrics": int(dq_summary.get("dq_reject_count", 0)),
                "stats_dsr": _metric_scalar(stats.get("dsr", {}), "dsr", float("nan")),
                "stats_pbo": _metric_scalar(stats.get("pbo", {}), "pbo", float("nan")),
                "engine_config_json": json.dumps(cfg.get("engine_config", {}), sort_keys=True),
            }
        )
    cfg_df = pd.DataFrame(candidate_cfg_rows)
    if cfg_df.shape[0] <= 0:
        raise Module6ValidationError("no candidate configuration rows found for strategy_instance_master")

    merged = (
        selection.merge(leaderboard, on="candidate_id", how="left", suffixes=("", "_lb"))
        .merge(cfg_df, on="candidate_id", how="left", suffixes=("", "_cfg"))
    )
    if merged["parameter_hash"].isna().any():
        merged["parameter_hash"] = merged["parameter_hash"].fillna(merged["parameter_hash_cfg"])
    if merged["family_id"].isna().any():
        merged["family_id"] = merged["family_id"].fillna(merged["family_id_cfg"])
    if merged["hypothesis_id"].isna().any():
        merged["hypothesis_id"] = merged["hypothesis_id"].fillna(merged["hypothesis_id_cfg"])
    if merged["evaluation_role"].isna().any():
        merged["evaluation_role"] = merged["evaluation_role"].fillna(merged["evaluation_role_cfg"])
    merged["evaluation_window"] = merged["evaluation_window"].where(
        merged["evaluation_window"].notna(),
        merged["evaluation_window_cfg"],
    )
    merged["window_set"] = merged["window_set"].where(merged["window_set"].notna(), merged["window_set_cfg"])
    merged["window_set_size"] = merged["window_set_size"].where(
        merged["window_set_size"].notna(), merged["window_set_size_cfg"]
    )
    merged["tags_serialized"] = merged["tags_serialized"].where(
        merged["tags_serialized"].notna(), merged["tags_serialized_cfg"]
    )
    merged["enabled_assets_hash"] = merged["enabled_assets_hash"].fillna("")
    required = ["candidate_id", "parameter_hash", "enabled_assets_hash", "calendar_version", "selection_stage", "execution_mode"]
    for col in required:
        if merged[col].isna().any() or (merged[col].astype(str).str.strip() == "").any():
            raise Module6ValidationError(f"strategy_instance_master missing required identity column: {col}")

    merged["strategy_id"] = merged["candidate_id"].astype(str)
    merged["source_run_id"] = run_id
    merged["dataset_hash"] = dataset_hash
    merged["strategy_pk"] = merged.apply(
        lambda row: stable_sha256_parts(
            dataset_hash,
            run_id,
            str(row["strategy_id"]),
            str(row["parameter_hash"]),
            str(row["enabled_assets_hash"]),
        ),
        axis=1,
    )
    merged["strategy_instance_pk"] = merged.apply(
        lambda row: stable_sha256_parts(
            str(row["strategy_pk"]),
            str(row["split_id"]),
            str(row["scenario_id"]),
            str(row["execution_mode"]),
            str(row["selection_stage"]),
            str(row["calendar_version"]),
        ),
        axis=1,
    )
    merged["portfolio_admit_flag"] = ~(
        merged["failed"].fillna(False).astype(bool)
        | merged["reject"].fillna(False).astype(bool)
        | merged["candidate_metrics_failed"].fillna(False).astype(bool)
    )
    merged["constraint_flag_count"] = merged["zimtra_compliance_flags"].fillna("").astype(str).map(count_flag_tokens)
    merged["availability_ratio"] = 0.0
    merged["observed_session_count"] = 0
    merged["first_session_id"] = -1
    merged["last_session_id"] = -1
    merged["max_gap_sessions"] = 0
    merged["contiguous_support_ok"] = False
    assert_no_duplicates(merged, ["strategy_instance_pk"], "strategy_instance_master")

    canonical_counts = (
        merged.loc[merged["portfolio_instance_role"] == "canonical_portfolio"]
        .groupby("strategy_pk", dropna=False)
        .size()
    )
    if canonical_counts.empty:
        raise Module6ValidationError("no canonical portfolio instances found in selection artifact")
    if bool((canonical_counts != 1).any()):
        raise Module6ValidationError(
            f"canonical portfolio instance cardinality failure: {canonical_counts[canonical_counts != 1].to_dict()}"
        )
    for col in ("canonical_reference_split_id", "canonical_reference_scenario_id", "canonical_reference_policy"):
        if merged[col].isna().any() or (merged[col].astype(str).str.strip() == "").any():
            raise Module6ValidationError(f"strategy_instance_master missing canonical reference field: {col}")
    return merged


def build_strategy_session_ledger(run: LoadedModule5Run, instance_master: pd.DataFrame, config: Module6Config) -> pd.DataFrame:
    rows = run.strategy_instance_session_returns.copy()
    master_cols = [
        "strategy_instance_pk",
        "strategy_pk",
        "strategy_id",
        "candidate_id",
        "split_id",
        "scenario_id",
        "execution_mode",
        "selection_stage",
        "calendar_version",
        "portfolio_instance_role",
    ]
    merged = rows.merge(
        instance_master[master_cols],
        on=[
            "strategy_id",
            "candidate_id",
            "split_id",
            "scenario_id",
            "execution_mode",
            "selection_stage",
            "calendar_version",
        ],
        how="left",
    )
    if merged["strategy_instance_pk"].isna().any():
        sample = merged.loc[merged["strategy_instance_pk"].isna()].head(5).to_dict("records")
        raise Module6ValidationError(f"unresolved strategy instances in session ledger join: sample={sample}")
    assert_no_duplicates(
        merged,
        ["strategy_instance_pk", "session_id"],
        "strategy_session_ledger",
    )
    if merged["availability_state_code"].isna().any():
        raise Module6ValidationError("strategy_session_ledger missing availability_state_code")
    observed_codes = sorted({int(x) for x in pd.unique(merged["availability_state_code"]).tolist()})
    invalid_codes = [code for code in observed_codes if int(code) not in set(BASE_AVAIL_ALLOWED_CODES)]
    if invalid_codes:
        raise Module6ValidationError(f"strategy_session_ledger contains invalid base availability codes: {invalid_codes}")
    if not np.isfinite(np.asarray(merged["return_exec"], dtype=np.float64)).all():
        raise Module6ValidationError("strategy_session_ledger contains non-finite return_exec")
    if not np.isfinite(np.asarray(merged["return_raw"], dtype=np.float64)).all():
        raise Module6ValidationError("strategy_session_ledger contains non-finite return_raw")

    stats = (
        merged.assign(is_observed=merged["availability_state_code"].isin([1, 2]).astype(np.int8))
        .groupby("strategy_instance_pk", dropna=False)
        .agg(
            observed_session_count=("is_observed", "sum"),
            first_session_id=("session_id", "min"),
            last_session_id=("session_id", "max"),
        )
        .reset_index()
    )
    gaps: list[dict[str, Any]] = []
    for pk, grp in merged.groupby("strategy_instance_pk", dropna=False, sort=False):
        observed = np.sort(
            np.asarray(
                grp.loc[grp["availability_state_code"].isin([1, 2]), "session_id"],
                dtype=np.int64,
            )
        )
        if observed.size <= 1:
            gaps.append(
                {
                    "strategy_instance_pk": pk,
                    "availability_ratio": float(observed.size / max(grp.shape[0], 1)),
                    "max_gap_sessions": 0,
                    "contiguous_support_ok": bool(observed.size == grp.shape[0]),
                }
            )
            continue
        diffs = np.diff(observed)
        gaps.append(
            {
                "strategy_instance_pk": pk,
                "availability_ratio": float(observed.size / max(grp.shape[0], 1)),
                "max_gap_sessions": int(np.max(diffs) - 1),
                "contiguous_support_ok": bool(np.all(diffs == 1)),
            }
        )
    gap_df = pd.DataFrame(gaps)
    merged = merged.merge(stats, on="strategy_instance_pk", how="left").merge(gap_df, on="strategy_instance_pk", how="left")
    return merged.sort_values(["strategy_instance_pk", "session_id"], kind="mergesort").reset_index(drop=True)


def build_strategy_master(
    run: LoadedModule5Run,
    instance_master: pd.DataFrame,
    session_ledger: pd.DataFrame,
    config: Module6Config,
) -> pd.DataFrame:
    grouped = (
        session_ledger.groupby(["strategy_pk", "strategy_id"], dropna=False)
        .agg(
            availability_ratio=("availability_ratio", "max"),
            observed_session_count=("observed_session_count", "max"),
            first_session_id=("first_session_id", "min"),
            last_session_id=("last_session_id", "max"),
        )
        .reset_index()
    )
    canonical = instance_master.loc[
        instance_master["portfolio_instance_role"] == "canonical_portfolio"
    ].copy()
    canonical = canonical.drop_duplicates("strategy_pk", keep="first")
    master = canonical.merge(grouped, on=["strategy_pk", "strategy_id"], how="left", suffixes=("", "_agg"))
    keep_cols = [
        "strategy_pk",
        "strategy_id",
        "source_run_id",
        "dataset_hash",
        "parameter_hash",
        "enabled_assets_hash",
        "campaign_id",
        "family_id",
        "family_name",
        "hypothesis_id",
        "evaluation_role",
        "evaluation_window",
        "window_set",
        "window_set_size",
        "tags_serialized",
        "portfolio_admit_flag",
        "failed",
        "reject",
        "pass",
        "fragile",
        "robustness_score",
        "regime_robustness",
        "execution_robustness",
        "horizon_robustness",
        "cost_adjusted_expectancy",
        "overnight_suitability_score",
        "zimtra_compliance_flags",
        "dq_min",
        "dq_median",
        "dq_degrade_count",
        "dq_reject_count",
        "availability_ratio",
        "observed_session_count",
        "first_session_id",
        "last_session_id",
        "avg_turnover_metrics",
        "avg_margin_used_frac_metrics",
        "peak_margin_used_frac_metrics",
        "constraint_flag_count",
    ]
    master = master[keep_cols].copy()
    assert_no_duplicates(master, ["strategy_pk"], "strategy_master")
    return master.sort_values(["strategy_pk"], kind="mergesort").reset_index(drop=True)


def enrich_strategy_instance_master_from_session_ledger(
    instance_master: pd.DataFrame,
    session_ledger: pd.DataFrame,
) -> pd.DataFrame:
    stats = (
        session_ledger.groupby("strategy_instance_pk", dropna=False)
        .agg(
            availability_ratio=("availability_ratio", "max"),
            observed_session_count=("observed_session_count", "max"),
            first_session_id=("first_session_id", "min"),
            last_session_id=("last_session_id", "max"),
            max_gap_sessions=("max_gap_sessions", "max"),
            contiguous_support_ok=("contiguous_support_ok", "max"),
        )
        .reset_index()
    )
    enriched = instance_master.drop(
        columns=[
            "availability_ratio",
            "observed_session_count",
            "first_session_id",
            "last_session_id",
            "max_gap_sessions",
            "contiguous_support_ok",
        ],
        errors="ignore",
    ).merge(
        stats,
        on="strategy_instance_pk",
        how="left",
    )
    fill_defaults: dict[str, object] = {
        "availability_ratio": 0.0,
        "observed_session_count": 0,
        "first_session_id": -1,
        "last_session_id": -1,
        "max_gap_sessions": 0,
        "contiguous_support_ok": False,
    }
    for col, default in fill_defaults.items():
        enriched[col] = enriched[col].fillna(default)
    return enriched.sort_values(["strategy_instance_pk"], kind="mergesort").reset_index(drop=True)


def materialize_canonical_ledgers(
    run: LoadedModule5Run,
    output_dir: Path,
    config: Module6Config,
) -> dict[str, pd.DataFrame]:
    out_dir = ensure_directory(output_dir)
    instance_master = build_strategy_instance_master(run=run, config=config)
    session_ledger = build_strategy_session_ledger(run=run, instance_master=instance_master, config=config)
    instance_master = enrich_strategy_instance_master_from_session_ledger(
        instance_master=instance_master,
        session_ledger=session_ledger,
    )
    strategy_master = build_strategy_master(run=run, instance_master=instance_master, session_ledger=session_ledger, config=config)

    instance_master.to_parquet(out_dir / "strategy_instance_master.parquet", index=False)
    session_ledger.to_parquet(out_dir / "strategy_session_ledger.parquet", index=False)
    strategy_master.to_parquet(out_dir / "strategy_master.parquet", index=False)
    return {
        "strategy_master": strategy_master,
        "strategy_instance_master": instance_master,
        "strategy_session_ledger": session_ledger,
    }
