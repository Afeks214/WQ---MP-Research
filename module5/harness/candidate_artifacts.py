from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from weightiz_module5_stats import deflated_sharpe_ratio, run_full_stats


def stack_payload_frames(payloads: list[dict[str, np.ndarray]], require_pandas_fn: Callable[[], Any]) -> Any:
    pdx = require_pandas_fn()
    if not payloads:
        return pdx.DataFrame()
    frames = [pdx.DataFrame(p) for p in payloads]
    if not frames:
        return pdx.DataFrame()
    return pdx.concat(frames, axis=0, ignore_index=True)


def collect_ledger_rows_from_results(
    rows: list[dict[str, Any]],
    evaluation_timestamp: str,
    trade_count_from_payload_fn: Callable[[dict[str, np.ndarray] | None], int],
    max_drawdown_from_returns_fn: Callable[[np.ndarray], float],
    extract_final_equity_fn: Callable[[dict[str, Any]], float],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        if str(r.get("status", "")) != "ok":
            continue
        cid = str(r.get("candidate_id", ""))
        spec = {
            "m2_idx": int(r.get("m2_idx", -1)),
            "m3_idx": int(r.get("m3_idx", -1)),
            "m4_idx": int(r.get("m4_idx", -1)),
            "tags": list(r.get("tags", [])),
        }
        spec_blob = json.dumps(spec, sort_keys=True, separators=(",", ":"))
        sh = hashlib.sha256(spec_blob.encode("utf-8")).hexdigest()
        daily_ret = np.asarray(r.get("daily_returns", np.zeros(0, dtype=np.float64)), dtype=np.float64)
        sharpe = float(np.mean(daily_ret) / np.std(daily_ret, ddof=1) * np.sqrt(252.0)) if daily_ret.size >= 2 and float(np.std(daily_ret, ddof=1)) > 0 else 0.0
        down = daily_ret[daily_ret < 0.0]
        sortino = float(np.mean(daily_ret) / np.std(down, ddof=1) * np.sqrt(252.0)) if down.size >= 2 and float(np.std(down, ddof=1)) > 0 else 0.0
        out.append(
            {
                "strategy_id": cid,
                "strategy_hash": sh,
                "parameter_values": spec_blob,
                "asset_count": int(len(r.get("asset_keys", []))),
                "total_trades": int(trade_count_from_payload_fn(r.get("trade_payload"))),
                "sharpe": float(sharpe),
                "sortino": float(sortino),
                "max_drawdown": float(max_drawdown_from_returns_fn(daily_ret)),
                "final_equity": float(extract_final_equity_fn(r)),
                "evaluation_timestamp": str(evaluation_timestamp),
            }
        )
    return out


def summarize_fold_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "sharpe_daily_mean": 0.0,
            "sharpe_daily_median": 0.0,
            "sharpe_daily_worst": 0.0,
            "cum_return_mean": 0.0,
            "cum_return_median": 0.0,
            "max_drawdown_median": 0.0,
            "turnover_median": 0.0,
        }
    sh = np.asarray([float(r["sharpe_daily"]) for r in rows], dtype=np.float64)
    cr = np.asarray([float(r["cum_return"]) for r in rows], dtype=np.float64)
    dd = np.asarray([float(r["max_drawdown"]) for r in rows], dtype=np.float64)
    to = np.asarray([float(r["turnover"]) for r in rows], dtype=np.float64)
    return {
        "count": int(len(rows)),
        "sharpe_daily_mean": float(np.mean(sh)),
        "sharpe_daily_median": float(np.median(sh)),
        "sharpe_daily_worst": float(np.min(sh)),
        "cum_return_mean": float(np.mean(cr)),
        "cum_return_median": float(np.median(cr)),
        "max_drawdown_median": float(np.median(dd)),
        "turnover_median": float(np.median(to)),
    }


def plateau_key(feature: dict[str, float]) -> tuple[int, ...]:
    return (
        int(np.rint(float(feature["entry_threshold"]) / 0.02)),
        int(np.rint(float(feature["exit_threshold"]) / 0.02)),
        int(np.rint(float(feature["top_k_intraday"]) / 1.0)),
        int(np.rint(float(feature["max_asset_cap_frac"]) / 0.05)),
        int(np.rint(float(feature["max_turnover_frac_per_bar"]) / 0.05)),
        int(np.rint(float(feature["block_minutes"]) / 5.0)),
        int(np.rint(float(feature["min_block_valid_ratio"]) / 0.05)),
    )


def split_mode(split_id: str) -> str:
    s = str(split_id)
    if s.startswith("wf_"):
        return "wf"
    if s.startswith("cpcv_"):
        return "cpcv"
    return "other"


def build_candidate_artifacts(
    report_root: Path,
    run_id: str,
    run_started_utc: Any,
    git_hash: str,
    candidates: list[Any],
    all_results: list[dict[str, Any]],
    candidate_daily_mat: np.ndarray,
    daily_bmk: np.ndarray,
    common_sessions: np.ndarray,
    baseline_candidate_ids: list[str],
    candidate_scenario_series: dict[str, dict[str, dict[int, float]]],
    candidate_verdict: dict[str, dict[str, Any]],
    expected_baseline_tasks: int,
    scenarios: list[Any],
    engine_cfg: Any,
    m2_configs: list[Any],
    m3_configs: list[Any],
    m4_configs: list[Any],
    harness_cfg: Any,
    *,
    require_pandas_fn: Callable[[], Any],
    write_json_fn: Callable[[Path, Any], None],
    baseline_failure_reasons_fn: Callable[[list[dict[str, Any]], int], list[str]],
    clip01_fn: Callable[[float], float],
    cum_return_fn: Callable[[np.ndarray], float],
    max_drawdown_from_returns_fn: Callable[[np.ndarray], float],
    turnover_from_trade_payload_fn: Callable[[dict[str, np.ndarray] | None, float], float],
    sharpe_daily_fn: Callable[[np.ndarray], float],
    trade_count_from_payload_fn: Callable[[dict[str, np.ndarray] | None], int],
    margin_exposure_stats_from_equity_payloads_fn: Callable[[list[dict[str, np.ndarray]]], dict[str, float]],
    asset_pnl_concentration_from_result_rows_fn: Callable[[list[dict[str, Any]]], float],
    asset_notional_concentration_from_trade_payloads_fn: Callable[[list[dict[str, np.ndarray]]], float],
    robustness_caps: dict[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pdx = require_pandas_fn()

    candidates_dir = report_root / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    baseline_col = {str(cid): j for j, cid in enumerate(baseline_candidate_ids)}
    scenario_ids = [str(s.scenario_id) for s in scenarios]
    initial_cash = float(engine_cfg.initial_cash)

    candidate_rows: list[dict[str, Any]] = []

    for cand in sorted(candidates, key=lambda x: x.candidate_id):
        cdir = candidates_dir / str(cand.candidate_id)
        cdir.mkdir(parents=True, exist_ok=True)

        rows_all = [
            r
            for r in all_results
            if str(r.get("candidate_id", "")) == str(cand.candidate_id)
        ]
        rows_all = sorted(
            rows_all,
            key=lambda x: (
                str(x.get("scenario_id", "")),
                str(x.get("split_id", "")),
                str(x.get("task_id", "")),
            ),
        )
        rows = [r for r in rows_all if str(r.get("status", "")) == "ok"]

        cid = str(cand.candidate_id)
        aligned = cid in baseline_col
        if aligned:
            ret_series = candidate_daily_mat[:, int(baseline_col[cid])].astype(np.float64)
        else:
            ret_series = np.zeros(common_sessions.shape[0], dtype=np.float64)
        loss_series = -ret_series

        baseline_map = (
            candidate_scenario_series.get(cid, {}).get("baseline", {})
            if candidate_scenario_series is not None
            else {}
        )
        ret_df = pdx.DataFrame(
            {
                "session_id": common_sessions.astype(np.int64),
                "returns": ret_series,
                "is_observed_baseline": np.asarray(
                    [int(int(s) in baseline_map) for s in common_sessions.tolist()],
                    dtype=np.int8,
                ),
            }
        )
        loss_df = pdx.DataFrame(
            {
                "session_id": common_sessions.astype(np.int64),
                "losses": loss_series,
            }
        )
        ret_path = cdir / "candidate_returns.parquet"
        loss_path = cdir / "candidate_losses.parquet"
        ret_df.to_parquet(ret_path, index=False)
        loss_df.to_parquet(loss_path, index=False)

        fold_rows: list[dict[str, Any]] = []
        fold_sharpes: list[float] = []
        fold_dsrs: list[float] = []
        rows_base_all = [r for r in rows_all if str(r.get("scenario_id", "")) == "baseline"]
        rows_base = [r for r in rows_base_all if str(r.get("status", "")) == "ok"]
        baseline_fail_reasons = baseline_failure_reasons_fn(
            rows_base_all=rows_base_all,
            expected_baseline_tasks=int(expected_baseline_tasks),
        )
        dqs_row_median = np.asarray(
            [float(r.get("dqs_median", np.nan)) for r in rows_all],
            dtype=np.float64,
        )
        dqs_row_min = np.asarray(
            [float(r.get("dqs_min", np.nan)) for r in rows_all],
            dtype=np.float64,
        )
        dqs_vals_med = dqs_row_median[np.isfinite(dqs_row_median)]
        dqs_vals_min = dqs_row_min[np.isfinite(dqs_row_min)]
        dqs_median = float(np.median(dqs_vals_med)) if dqs_vals_med.size > 0 else 1.0
        dqs_min = float(np.min(dqs_vals_min)) if dqs_vals_min.size > 0 else 1.0
        reason_codes_flat: list[str] = []
        for rr in rows_all:
            reason_codes_flat.extend([str(x) for x in rr.get("quality_reason_codes", [])])
        dq_degrade_count = int(sum(1 for rr in rows_all if "DQ_DEGRADED_INPUT" in [str(x) for x in rr.get("quality_reason_codes", [])]))
        dq_reject_count = int(sum(1 for rr in rows_all if "DQ_REJECTED_INPUT" in [str(x) for x in rr.get("quality_reason_codes", [])]))
        if reason_codes_flat:
            uniq = sorted(set(reason_codes_flat))
            freq = {k: int(reason_codes_flat.count(k)) for k in uniq}
            dq_reason_top = sorted(freq.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[0][0]
        else:
            dq_reason_top = ""
        failed_candidate = len(baseline_fail_reasons) > 0
        for r in (rows_base if rows_base else rows):
            tr = np.asarray(r.get("daily_returns", np.zeros(0, dtype=np.float64)), dtype=np.float64)
            if tr.size == 0:
                continue
            sharpe_d = sharpe_daily_fn(tr)
            fold_sharpes.append(sharpe_d)
            if tr.size >= 3:
                dsr_one = deflated_sharpe_ratio(tr[:, None])
                fold_dsrs.append(float(np.asarray(dsr_one["dsr"], dtype=np.float64)[0]))
            else:
                fold_dsrs.append(0.0)
            fold_rows.append(
                {
                    "split_id": str(r.get("split_id", "")),
                    "mode": split_mode(str(r.get("split_id", ""))),
                    "scenario_id": str(r.get("scenario_id", "")),
                    "cum_return": cum_return_fn(tr),
                    "sharpe_daily": sharpe_d,
                    "max_drawdown": max_drawdown_from_returns_fn(tr),
                    "turnover": turnover_from_trade_payload_fn(r.get("trade_payload"), initial_cash),
                    "test_days": int(r.get("test_days", 0)),
                }
            )

        wf_rows = [x for x in fold_rows if str(x["mode"]) == "wf"]
        cpcv_rows = [x for x in fold_rows if str(x["mode"]) == "cpcv"]

        per_stress: dict[str, Any] = {}
        for sid in scenario_ids:
            srows = [r for r in rows if str(r.get("scenario_id", "")) == sid]
            if not srows:
                per_stress[sid] = {
                    "n_tasks": 0,
                    "cum_return_median": 0.0,
                    "cum_return_mean": 0.0,
                    "max_drawdown_median": 0.0,
                    "turnover_median": 0.0,
                }
                continue
            ret_list = [
                np.asarray(r.get("daily_returns", np.zeros(0, dtype=np.float64)), dtype=np.float64)
                for r in srows
            ]
            ret_list = [x for x in ret_list if x.size > 0]
            if not ret_list:
                per_stress[sid] = {
                    "n_tasks": 0,
                    "cum_return_median": 0.0,
                    "cum_return_mean": 0.0,
                    "max_drawdown_median": 0.0,
                    "turnover_median": 0.0,
                }
                continue
            cum = np.asarray([cum_return_fn(x) for x in ret_list], dtype=np.float64)
            dd = np.asarray([max_drawdown_from_returns_fn(x) for x in ret_list], dtype=np.float64)
            to = np.asarray(
                [turnover_from_trade_payload_fn(r.get("trade_payload"), initial_cash) for r in srows],
                dtype=np.float64,
            )
            per_stress[sid] = {
                "n_tasks": int(len(ret_list)),
                "cum_return_median": float(np.median(cum)),
                "cum_return_mean": float(np.mean(cum)),
                "max_drawdown_median": float(np.median(dd)),
                "turnover_median": float(np.median(to)),
            }

        base_stress = per_stress.get("baseline", {"cum_return_median": 0.0, "max_drawdown_median": 0.0, "turnover_median": 0.0})
        for sid in scenario_ids:
            per_stress[sid]["delta_vs_baseline"] = {
                "pnl": float(per_stress[sid]["cum_return_median"] - base_stress["cum_return_median"]),
                "dd": float(per_stress[sid]["max_drawdown_median"] - base_stress["max_drawdown_median"]),
                "turnover": float(per_stress[sid]["turnover_median"] - base_stress["turnover_median"]),
            }

        rows_for_base_stats = rows_base if rows_base else rows
        trade_payloads = [r.get("trade_payload") for r in rows_for_base_stats if r.get("trade_payload") is not None]
        eq_payloads = [r.get("equity_payload") for r in rows_for_base_stats if r.get("equity_payload") is not None]
        n_trades = int(sum(trade_count_from_payload_fn(p) for p in trade_payloads))

        ret_non_zero = ret_series[np.abs(ret_series) > 1e-15]
        win_rate = float(np.mean(ret_series > 0.0)) if ret_series.size else 0.0
        avg_trade = float(np.mean(ret_non_zero)) if ret_non_zero.size else 0.0
        pos_sum = float(np.sum(ret_series[ret_series > 0.0])) if ret_series.size else 0.0
        neg_sum = float(np.sum(ret_series[ret_series < 0.0])) if ret_series.size else 0.0
        profit_factor = float(pos_sum / max(abs(neg_sum), 1e-12)) if ret_series.size else 0.0
        max_dd = max_drawdown_from_returns_fn(ret_series)
        cagr_ish = float(np.power(max(1.0 + cum_return_fn(ret_series), 1e-12), 252.0 / max(float(ret_series.size), 1.0)) - 1.0)
        exposure_stats = margin_exposure_stats_from_equity_payloads_fn(eq_payloads)
        conc = asset_pnl_concentration_from_result_rows_fn(rows_for_base_stats)
        if conc <= 0.0:
            conc = asset_notional_concentration_from_trade_payloads_fn(trade_payloads)

        if ret_series.size >= 3:
            full_stats = run_full_stats(
                returns_matrix=ret_series[:, None],
                benchmark=daily_bmk,
                losses=loss_series[:, None],
                bootstrap_spec={"B": 256, "avg_block_len": 20, "seed": int(harness_cfg.seed + 601)},
                cpcv_params={"S": int(harness_cfg.cpcv_slices), "k": int(harness_cfg.cpcv_k_test)},
            )
            dsr_full = float(np.asarray(full_stats["dsr"]["dsr"], dtype=np.float64)[0])
            pbo_val = float(full_stats["pbo"]["pbo"]) if np.isfinite(float(full_stats["pbo"]["pbo"])) else 1.0
        else:
            full_stats = {
                "skipped_due_to_failure": True,
                "reason": "insufficient_aligned_baseline_days",
                "bootstrap_spec": {"B": 256, "avg_block_len": 20, "seed": int(harness_cfg.seed + 601)},
                "cpcv_params": {"S": int(harness_cfg.cpcv_slices), "k": int(harness_cfg.cpcv_k_test)},
            }
            dsr_full = 0.0
            pbo_val = 1.0
            if not failed_candidate:
                failed_candidate = True
                baseline_fail_reasons.append("insufficient_aligned_baseline_days")
        dsr_median = float(np.median(np.asarray(fold_dsrs, dtype=np.float64))) if fold_dsrs else dsr_full
        fold_sharpe_std = float(np.std(np.asarray(fold_sharpes, dtype=np.float64), ddof=1)) if len(fold_sharpes) > 1 else 0.0
        dd_severe = float(per_stress.get("severe", base_stress)["max_drawdown_median"])
        verdict_row = candidate_verdict.get(cid, {})
        verdict_score = float(verdict_row.get("robustness_score", np.nan)) if isinstance(verdict_row, dict) else float("nan")
        has_verdict_score = bool(np.isfinite(verdict_score))

        if failed_candidate:
            robustness_score = float("-inf")
        elif has_verdict_score:
            robustness_score = float(verdict_score)
        else:
            robustness_score = float(
                1.0 * clip01_fn(dsr_median)
                - 0.5 * clip01_fn(pbo_val)
                - 0.3 * clip01_fn(dd_severe / robustness_caps["dd_cap"])
                - 0.2 * clip01_fn(fold_sharpe_std / robustness_caps["std_cap"])
                - 0.2 * clip01_fn(conc / robustness_caps["conc_cap"])
            )

        m3cfg = m3_configs[int(cand.m3_idx)]
        m4cfg = m4_configs[int(cand.m4_idx)]
        feat = {
            "entry_threshold": float(m4cfg.entry_threshold),
            "exit_threshold": float(m4cfg.exit_threshold),
            "top_k_intraday": float(m4cfg.top_k_intraday),
            "max_asset_cap_frac": float(m4cfg.max_asset_cap_frac),
            "max_turnover_frac_per_bar": float(m4cfg.max_turnover_frac_per_bar),
            "block_minutes": float(m3cfg.block_minutes),
            "min_block_valid_ratio": float(m3cfg.min_block_valid_ratio),
        }

        candidate_config = {
            "run_id": run_id,
            "timestamp_utc": run_started_utc.isoformat(),
            "git_hash": git_hash,
            "candidate_id": str(cand.candidate_id),
            "m2_idx": int(cand.m2_idx),
            "m3_idx": int(cand.m3_idx),
            "m4_idx": int(cand.m4_idx),
            "enabled_assets_mask": np.asarray(cand.enabled_assets_mask, dtype=bool).tolist(),
            "engine_config": asdict(engine_cfg),
            "module2_config": asdict(m2_configs[int(cand.m2_idx)]),
            "module3_config": asdict(m3cfg),
            "module4_config": asdict(m4cfg),
        }
        write_json_fn(cdir / "candidate_config.json", candidate_config)
        write_json_fn(cdir / "candidate_stats.json", full_stats)

        candidate_metrics = {
            "candidate_id": str(cand.candidate_id),
            "base_metrics": {
                "n_days": int(ret_series.size),
                "n_trades": n_trades,
                "win_rate": win_rate,
                "avg_trade": avg_trade,
                "profit_factor": profit_factor,
                "max_drawdown": max_dd,
                "cagr_ish": cagr_ish,
                "avg_turnover": float(np.median([turnover_from_trade_payload_fn(p, initial_cash) for p in trade_payloads])) if trade_payloads else 0.0,
                "asset_pnl_concentration": conc,
                **exposure_stats,
            },
            "per_stress": per_stress,
            "per_fold": {
                "wf": {
                    "summary": summarize_fold_stats(wf_rows),
                    "folds": wf_rows,
                },
                "cpcv": {
                    "summary": summarize_fold_stats(cpcv_rows),
                    "folds": cpcv_rows,
                },
            },
            "robustness": {
                "score": robustness_score,
                "formula": (
                    "0.20*dsr+0.15*(1-pbo)+0.10*(1-spa_p)+0.20*regime+0.20*execution+0.15*horizon"
                    if has_verdict_score
                    else "1*clip(dsr_median)-0.5*clip(pbo)-0.3*clip(dd_severe/dd_cap)-0.2*clip(fold_sharpe_std/std_cap)-0.2*clip(asset_concentration/conc_cap)"
                ),
                "dsr_source": "cluster_representative_exec" if has_verdict_score else "baseline_fold_median",
                "inputs": {
                    "dsr_median": dsr_median,
                    "pbo": pbo_val,
                    "dd_severe": dd_severe,
                    "fold_sharpe_std": fold_sharpe_std,
                    "asset_pnl_concentration": conc,
                    "verdict_robustness_score": verdict_score if has_verdict_score else None,
                },
                "caps": dict(robustness_caps),
            },
            "failed": bool(failed_candidate),
            "failure_reasons": sorted(set(baseline_fail_reasons)),
            "alignment": {
                "aligned_to_global_benchmark_sessions": bool(aligned),
                "global_session_count": int(common_sessions.shape[0]),
                "observed_baseline_session_count": int(len(baseline_map)),
            },
            "dq_summary": {
                "dq_min": float(dqs_min),
                "dq_median": float(dqs_median),
                "dq_degrade_count": int(dq_degrade_count),
                "dq_reject_count": int(dq_reject_count),
                "dq_reason_top": str(dq_reason_top),
            },
        }
        write_json_fn(cdir / "candidate_metrics.json", candidate_metrics)

        candidate_rows.append(
            {
                "candidate_id": str(cand.candidate_id),
                "m2_idx": int(cand.m2_idx),
                "m3_idx": int(cand.m3_idx),
                "m4_idx": int(cand.m4_idx),
                "n_tasks": int(len(rows)),
                "n_tasks_baseline": int(len(rows_base)),
                "n_days": int(ret_series.size),
                "n_days_observed_baseline": int(len(baseline_map)),
                "cum_return": cum_return_fn(ret_series),
                "max_drawdown": max_dd,
                "dsr_full": dsr_full,
                "dsr_median": dsr_median,
                "pbo": pbo_val,
                "fold_sharpe_std": fold_sharpe_std,
                "asset_pnl_concentration": conc,
                "robustness_score": robustness_score,
                "cluster_id": (
                    int(verdict_row.get("cluster_id"))
                    if isinstance(verdict_row.get("cluster_id"), (int, np.integer))
                    or (isinstance(verdict_row.get("cluster_id"), str) and str(verdict_row.get("cluster_id")).strip().lstrip("-").isdigit())
                    else -1
                ),
                "cluster_representative": str(verdict_row.get("cluster_representative", "")),
                "regime_robustness": float(verdict_row.get("regime_robustness", np.nan)),
                "execution_robustness": float(verdict_row.get("execution_robustness", np.nan)),
                "horizon_robustness": float(verdict_row.get("horizon_robustness", np.nan)),
                "research_mode": str(verdict_row.get("research_mode", "standard")),
                "standard_reject": bool(verdict_row.get("standard_reject", verdict_row.get("reject", False))),
                "standard_pass": bool(verdict_row.get("standard_pass", verdict_row.get("pass", False))),
                "discovery_included": bool(verdict_row.get("discovery_included", False)),
                "fragile": bool(verdict_row.get("fragile", False)),
                "reject": bool(verdict_row.get("reject", False)),
                "failed": bool(failed_candidate),
                "failure_reasons": "|".join(sorted(set(baseline_fail_reasons))),
                "dq_min": float(dqs_min),
                "dq_median": float(dqs_median),
                "dq_degrade_count": int(dq_degrade_count),
                "dq_reject_count": int(dq_reject_count),
                "dq_reason_top": str(dq_reason_top),
                "in_mcs": bool(verdict_row.get("in_mcs", False)),
                "pass": bool(verdict_row.get("pass", False)),
                "wrc_p": float(verdict_row.get("wrc_p", np.nan)) if verdict_row else np.nan,
                "spa_p": float(verdict_row.get("spa_p", np.nan)) if verdict_row else np.nan,
                **feat,
            }
        )

    group_map: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    for row in candidate_rows:
        key = plateau_key(row)
        group_map.setdefault(key, []).append(row)

    clusters: list[dict[str, Any]] = []
    cand_to_plateau: dict[str, str] = {}
    for i, key in enumerate(sorted(group_map.keys())):
        rows = sorted(group_map[key], key=lambda x: str(x["candidate_id"]))
        scores = np.asarray([float(r["robustness_score"]) for r in rows], dtype=np.float64)
        rep = sorted(rows, key=lambda x: (-float(x["robustness_score"]), str(x["candidate_id"]))) [0]
        pid = f"plateau_{i:03d}"
        for r in rows:
            cand_to_plateau[str(r["candidate_id"])] = pid
        clusters.append(
            {
                "plateau_id": pid,
                "bin_key": list(key),
                "count": int(len(rows)),
                "median_score": float(np.median(scores)),
                "worst_score": float(np.min(scores)),
                "representative_candidate_id": str(rep["candidate_id"]),
                "candidate_ids": [str(r["candidate_id"]) for r in rows],
            }
        )

    robustness_rows = []
    for row in candidate_rows:
        out = dict(row)
        out["plateau_id"] = cand_to_plateau.get(str(row["candidate_id"]), "plateau_unk")
        robustness_rows.append(out)

    def _robust_sort_key(x: dict[str, Any]) -> tuple[int, float, str]:
        s = float(x["robustness_score"])
        failed = 1 if (not np.isfinite(s) and s < 0.0) else 0
        ord_score = -s if np.isfinite(s) else float("inf")
        return (failed, ord_score, str(x["candidate_id"]))

    robustness_rows = sorted(robustness_rows, key=_robust_sort_key)

    return candidate_rows, robustness_rows, {
        "method": "grid_binning",
        "bin_spec": {
            "entry_threshold": 0.02,
            "exit_threshold": 0.02,
            "top_k_intraday": 1.0,
            "max_asset_cap_frac": 0.05,
            "max_turnover_frac_per_bar": 0.05,
            "block_minutes": 5.0,
            "min_block_valid_ratio": 0.05,
        },
        "clusters": clusters,
    }
