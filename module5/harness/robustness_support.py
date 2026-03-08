from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from regime_detector import RegimeConfig, build_regime_masks, detect_regimes, regime_sample_counts
from strategy_embedding import cluster_strategies_hierarchical_threshold
from weightiz_module5_stats import (
    deflated_sharpe_ratio,
    model_confidence_set,
    pbo_cscv,
    spa_test,
    white_reality_check,
)


def _slice_score_from_stats(
    dsr: dict[str, Any],
    pbo: dict[str, Any],
    spa: dict[str, Any],
    *,
    clip01_fn: Callable[[float], float],
) -> float:
    dsr_arr = np.asarray(dsr.get("dsr", np.zeros(0, dtype=np.float64)), dtype=np.float64)
    dsr_score = clip01_fn(float(np.mean(dsr_arr))) if dsr_arr.size > 0 else 0.5
    pbo_val = float(pbo.get("pbo", np.nan))
    pbo_score = clip01_fn(1.0 - pbo_val) if np.isfinite(pbo_val) else 0.5
    spa_p = float(spa.get("p_value", np.nan))
    spa_score = clip01_fn(1.0 - spa_p) if np.isfinite(spa_p) else 0.5
    return float((dsr_score + pbo_score + spa_score) / 3.0)


def _effective_benchmark_for_horizon(
    benchmark: np.ndarray,
    horizon: int,
    *,
    resample_returns_horizon_fn: Callable[[np.ndarray, int], np.ndarray],
) -> np.ndarray:
    return resample_returns_horizon_fn(np.asarray(benchmark, dtype=np.float64), int(horizon))


def compute_stats_verdict(
    daily_returns_matrix_exec: np.ndarray,
    daily_returns_matrix_raw: np.ndarray,
    daily_benchmark_returns: np.ndarray,
    candidate_ids: list[str],
    harness_cfg: Any,
    report_root: Path | None = None,
    *,
    clip01_fn: Callable[[float], float],
    cum_return_fn: Callable[[np.ndarray], float],
    resample_returns_horizon_fn: Callable[[np.ndarray, int], np.ndarray],
    seed_for_task_fn: Callable[[int, str, int], int],
) -> dict[str, Any]:
    ret_exec = np.asarray(daily_returns_matrix_exec, dtype=np.float64)
    ret_raw = np.asarray(daily_returns_matrix_raw, dtype=np.float64)
    bmk = np.asarray(daily_benchmark_returns, dtype=np.float64)
    if ret_exec.shape != ret_raw.shape:
        raise RuntimeError(f"raw/exec daily matrix shape mismatch: exec={ret_exec.shape}, raw={ret_raw.shape}")
    if ret_exec.ndim != 2:
        raise RuntimeError(f"daily_returns_matrix_exec must be 2D, got ndim={ret_exec.ndim}")
    if bmk.ndim != 1 or bmk.shape[0] != ret_exec.shape[0]:
        raise RuntimeError("daily_benchmark_returns shape mismatch with daily returns matrix")

    distance_path: str | None = None
    if int(ret_exec.shape[1]) > int(harness_cfg.cluster_distance_in_memory_max_n) and report_root is not None:
        distance_path = str((Path(report_root) / "cluster_distance_matrix.dat").resolve())

    cluster_doc = cluster_strategies_hierarchical_threshold(
        ret_exec,
        corr_threshold=float(harness_cfg.cluster_corr_threshold),
        block_size=int(harness_cfg.cluster_distance_block_size),
        distance_out_path=distance_path,
        in_memory_max_n=int(harness_cfg.cluster_distance_in_memory_max_n),
        seed=int(harness_cfg.seed),
    )
    cluster_labels = np.asarray(cluster_doc["cluster_labels"], dtype=np.int64)
    cluster_reps = np.asarray(cluster_doc["cluster_representatives"], dtype=np.int64)
    if cluster_reps.size <= 0:
        cluster_reps = np.arange(ret_exec.shape[1], dtype=np.int64)
        cluster_labels = np.arange(ret_exec.shape[1], dtype=np.int64)
    n_eff = int(cluster_reps.shape[0])

    ret_exec_rep = ret_exec[:, cluster_reps]
    ret_raw_rep = ret_raw[:, cluster_reps]

    dsr = deflated_sharpe_ratio(ret_exec_rep, n_trials=n_eff)
    pbo = pbo_cscv(
        ret_exec_rep,
        S=int(harness_cfg.cpcv_slices),
        k=int(harness_cfg.cpcv_k_test),
        n_trials_effective=n_eff,
    )
    wrc = white_reality_check(
        ret_exec_rep,
        bmk,
        seed=int(harness_cfg.seed + 101),
    )
    spa = spa_test(
        ret_exec_rep,
        bmk,
        seed=int(harness_cfg.seed + 202),
    )
    mcs = model_confidence_set(
        -ret_exec_rep,
        alpha=0.10,
        seed=int(harness_cfg.seed + 303),
    )

    regime_cfg = RegimeConfig(
        vol_window=int(harness_cfg.regime_vol_window),
        slope_window=int(harness_cfg.regime_slope_window),
        hurst_window=int(harness_cfg.regime_hurst_window),
        min_obs_per_mask=int(harness_cfg.regime_min_obs_per_mask),
    )
    regime_doc = detect_regimes(bmk, cfg=regime_cfg)
    regime_masks = build_regime_masks(regime_doc, min_obs=int(harness_cfg.regime_min_obs_per_mask))
    regime_details: dict[str, Any] = {}
    regime_scores: list[float] = []
    for rid in sorted(regime_masks.keys()):
        m = np.asarray(regime_masks[rid], dtype=bool)
        idx = np.flatnonzero(m).astype(np.int64)
        if idx.size < 3:
            continue
        r_slice = ret_exec_rep[idx, :]
        b_slice = bmk[idx]
        dsr_r = deflated_sharpe_ratio(r_slice, n_trials=n_eff)
        pbo_r = pbo_cscv(
            r_slice,
            S=int(harness_cfg.cpcv_slices),
            k=int(harness_cfg.cpcv_k_test),
            n_trials_effective=n_eff,
        )
        spa_r = spa_test(
            r_slice,
            b_slice,
            seed=int(seed_for_task_fn(harness_cfg.seed, "regime", rid)),
        )
        score = _slice_score_from_stats(dsr_r, pbo_r, spa_r, clip01_fn=clip01_fn)
        regime_scores.append(score)
        regime_details[rid] = {
            "obs": int(idx.size),
            "score": float(score),
            "dsr_mean": float(np.mean(np.asarray(dsr_r.get("dsr", np.zeros(0, dtype=np.float64)), dtype=np.float64))),
            "pbo": float(pbo_r.get("pbo", np.nan)),
            "spa_p": float(spa_r.get("p_value", np.nan)),
        }
    regime_robustness = float(np.mean(np.asarray(regime_scores, dtype=np.float64))) if regime_scores else 0.5

    horizon_list = [int(h) for h in harness_cfg.horizon_minutes]
    horizon_details: dict[str, Any] = {}
    horizon_scores: list[float] = []
    for h in horizon_list:
        cols = [resample_returns_horizon_fn(ret_exec_rep[:, j], h) for j in range(ret_exec_rep.shape[1])]
        b_h = _effective_benchmark_for_horizon(bmk, h, resample_returns_horizon_fn=resample_returns_horizon_fn)
        if not cols:
            horizon_details[str(h)] = {"insufficient": True, "score": 0.5, "obs": 0}
            horizon_scores.append(0.5)
            continue
        r_h = np.column_stack(cols).astype(np.float64)
        n_obs = min(int(r_h.shape[0]), int(b_h.shape[0]))
        if n_obs < 3:
            horizon_details[str(h)] = {"insufficient": True, "score": 0.5, "obs": int(n_obs)}
            horizon_scores.append(0.5)
            continue
        r_h = r_h[:n_obs, :]
        dsr_h = deflated_sharpe_ratio(r_h, n_trials=n_eff)
        pbo_h = pbo_cscv(
            r_h,
            S=int(harness_cfg.cpcv_slices),
            k=int(harness_cfg.cpcv_k_test),
            n_trials_effective=n_eff,
        )
        dsr_h_score = clip01_fn(float(np.mean(np.asarray(dsr_h.get("dsr", np.zeros(0, dtype=np.float64)), dtype=np.float64))))
        pbo_h_val = float(pbo_h.get("pbo", np.nan))
        pbo_h_score = clip01_fn(1.0 - pbo_h_val) if np.isfinite(pbo_h_val) else 0.5
        h_score = float(0.5 * dsr_h_score + 0.5 * pbo_h_score)
        horizon_scores.append(h_score)
        horizon_details[str(h)] = {
            "insufficient": False,
            "score": float(h_score),
            "obs": int(n_obs),
            "dsr_mean": float(np.mean(np.asarray(dsr_h.get("dsr", np.zeros(0, dtype=np.float64)), dtype=np.float64))),
            "pbo": pbo_h_val,
        }
    horizon_robustness = float(np.mean(np.asarray(horizon_scores, dtype=np.float64))) if horizon_scores else 0.5

    cum_raw_rep = np.asarray([cum_return_fn(ret_raw_rep[:, j]) for j in range(ret_raw_rep.shape[1])], dtype=np.float64)
    cum_exec_rep = np.asarray([cum_return_fn(ret_exec_rep[:, j]) for j in range(ret_exec_rep.shape[1])], dtype=np.float64)
    den = np.maximum(np.abs(cum_raw_rep), 1e-6)
    pen = np.maximum(cum_raw_rep - cum_exec_rep, 0.0) / den
    execution_score_rep = 1.0 - np.clip(pen, 0.0, 1.0)
    execution_robustness = float(np.mean(execution_score_rep)) if execution_score_rep.size > 0 else 0.5

    dsr_arr_rep = np.asarray(dsr["dsr"], dtype=np.float64)
    pbo_val = float(pbo["pbo"]) if np.isfinite(float(pbo["pbo"])) else float("nan")
    spa_p = float(spa["p_value"]) if np.isfinite(float(spa["p_value"])) else float("nan")
    pbo_score = clip01_fn(1.0 - pbo_val) if np.isfinite(pbo_val) else 0.5
    spa_score = clip01_fn(1.0 - spa_p) if np.isfinite(spa_p) else 0.5
    survivors = set(int(i) for i in np.asarray(mcs.get("survivors", np.array([], dtype=np.int64))).tolist())
    rep_pos_by_col = {int(col): int(i) for i, col in enumerate(cluster_reps.tolist())}

    leaderboard: list[dict[str, Any]] = []
    for j, cid in enumerate(candidate_ids):
        cidx = int(j)
        cl_id = int(cluster_labels[cidx]) if cidx < cluster_labels.size else int(cidx)
        rep_col = int(cluster_reps[cl_id]) if cl_id < cluster_reps.size else int(cidx)
        rep_pos = int(rep_pos_by_col.get(rep_col, 0))
        in_mcs = bool(rep_pos in survivors)
        dsr_j = float(dsr_arr_rep[rep_pos]) if rep_pos < dsr_arr_rep.size else float("nan")
        dsr_score_j = clip01_fn(dsr_j)
        exec_j = float(execution_score_rep[rep_pos]) if rep_pos < execution_score_rep.size else 0.5
        score = float(
            float(harness_cfg.robustness_weight_dsr) * dsr_score_j
            + float(harness_cfg.robustness_weight_pbo) * pbo_score
            + float(harness_cfg.robustness_weight_spa) * spa_score
            + float(harness_cfg.robustness_weight_regime) * regime_robustness
            + float(harness_cfg.robustness_weight_execution) * exec_j
            + float(harness_cfg.robustness_weight_horizon) * horizon_robustness
        )
        reject = bool(score < float(harness_cfg.robustness_reject_threshold))
        fragile = bool(exec_j < float(harness_cfg.execution_fragile_threshold))
        pass_flag = bool((not reject) and in_mcs)
        leaderboard.append(
            {
                "candidate_id": str(cid),
                "cluster_id": cl_id,
                "cluster_representative": str(candidate_ids[rep_col]) if rep_col < len(candidate_ids) else str(cid),
                "dsr": dsr_j,
                "in_mcs": in_mcs,
                "wrc_p": float(wrc["p_value"]),
                "spa_p": spa_p,
                "pbo": pbo_val if np.isfinite(pbo_val) else None,
                "regime_robustness": float(regime_robustness),
                "execution_robustness": float(exec_j),
                "horizon_robustness": float(horizon_robustness),
                "robustness_score": float(score),
                "fragile": fragile,
                "reject": reject,
                "pass": pass_flag,
            }
        )
    leaderboard = sorted(leaderboard, key=lambda x: str(x["candidate_id"]))

    return {
        "dsr": dsr,
        "pbo": pbo,
        "wrc": wrc,
        "spa": spa,
        "mcs": mcs,
        "cluster": {
            "n_clusters": int(n_eff),
            "n_candidates": int(ret_exec.shape[1]),
            "corr_threshold": float(harness_cfg.cluster_corr_threshold),
            "cluster_labels": cluster_labels,
            "cluster_representatives": cluster_reps,
            "distance_is_memmap": bool(cluster_doc.get("distance_is_memmap", False)),
            "distance_path": str(cluster_doc.get("distance_path", "")),
        },
        "regime_validation": {
            "config": asdict(regime_cfg),
            "mask_counts": regime_sample_counts(regime_masks),
            "details": regime_details,
            "regime_robustness": float(regime_robustness),
        },
        "horizon_validation": {
            "horizons": horizon_list,
            "details": horizon_details,
            "horizon_robustness": float(horizon_robustness),
        },
        "execution_validation": {
            "execution_robustness": float(execution_robustness),
            "cum_return_raw_rep": cum_raw_rep,
            "cum_return_exec_rep": cum_exec_rep,
            "score_rep": execution_score_rep,
        },
        "leaderboard": leaderboard,
        "gate_defaults": {
            "robustness_reject_threshold": float(harness_cfg.robustness_reject_threshold),
            "execution_fragile_threshold": float(harness_cfg.execution_fragile_threshold),
            "mcs_membership_required": True,
        },
    }
