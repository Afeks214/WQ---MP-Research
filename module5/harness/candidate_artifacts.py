from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.stage_a_discovery import parse_stage_a_tags, parse_stage_a_window_set, stable_stage_a_hash
from weightiz_module5_stats import deflated_sharpe_ratio, run_full_stats


def _serialize_tags(tags: list[str] | tuple[str, ...]) -> str:
    vals = sorted(str(x) for x in tags if str(x).strip())
    return "|".join(vals)


def _serialize_flags(flags: list[str] | set[str] | tuple[str, ...]) -> str:
    vals = sorted(str(x) for x in flags if str(x).strip())
    return "|".join(vals)


def _safe_mean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return float("nan")
    return float(np.mean(arr))


def _stage_a_metadata(
    *,
    tags: list[str] | tuple[str, ...],
    m2_idx: int,
    m3_idx: int,
    m4_idx: int,
) -> dict[str, Any]:
    meta = parse_stage_a_tags(tags)
    window_set = parse_stage_a_window_set(meta.get("window_set"))
    evaluation_window = meta.get("evaluation_window")
    out: dict[str, Any] = {
        "campaign_id": str(meta.get("campaign_id", "")),
        "family_id": str(meta.get("family_id", "")),
        "family_name": str(meta.get("family_name", "")),
        "hypothesis_id": str(meta.get("hypothesis_id", "")),
        "evaluation_role": str(meta.get("evaluation_role", "")),
        "window_set": ",".join(str(int(x)) for x in window_set),
        "window_set_size": int(len(window_set)),
        "parameter_hash": str(
            meta.get(
                "parameter_hash",
                stable_stage_a_hash(
                    {
                        "m2_idx": int(m2_idx),
                        "m3_idx": int(m3_idx),
                        "m4_idx": int(m4_idx),
                        "tags": list(tags),
                    }
                ),
            )
        ),
        "tags_serialized": _serialize_tags(tags),
    }
    if evaluation_window is not None and str(evaluation_window).strip():
        out["evaluation_window"] = int(evaluation_window)
    else:
        out["evaluation_window"] = None
    return out


def _overnight_suitability_score(rows_all: list[dict[str, Any]]) -> float:
    collected: list[float] = []
    for row in rows_all:
        payload = row.get("micro_payload")
        if not isinstance(payload, dict):
            continue
        scores = np.asarray(payload.get("overnight_score", np.zeros(0, dtype=np.float64)), dtype=np.float64)
        if scores.size <= 0:
            continue
        winner = np.asarray(payload.get("overnight_winner_flag", np.zeros(scores.shape[0], dtype=np.int8)), dtype=np.int8)
        finite_scores = scores[np.isfinite(scores)]
        if winner.shape == scores.shape and np.any(winner > 0):
            chosen = scores[(winner > 0) & np.isfinite(scores)]
            if chosen.size > 0:
                collected.extend(float(x) for x in chosen.tolist())
                continue
        collected.extend(float(x) for x in finite_scores.tolist())
    return _safe_mean(collected)


def _zimtra_compliance_flags(
    *,
    rows_all: list[dict[str, Any]],
    baseline_fail_reasons: list[str],
    verdict_row: dict[str, Any],
) -> str:
    flags: set[str] = set()
    codes = [str(x) for row in rows_all for x in row.get("quality_reason_codes", [])]
    if any(code == "RISK_CONSTRAINT_BREACH" for code in codes):
        flags.add("risk_constraint_breach")
    if any(code == "DQ_DEGRADED_INPUT" for code in codes):
        flags.add("dq_degraded_input")
    if any(code == "DQ_REJECTED_INPUT" for code in codes):
        flags.add("dq_rejected_input")
    if any(code.startswith("INVARIANT_") for code in codes):
        flags.add("localized_invariant_warning")
    if bool(verdict_row.get("fragile", False)):
        flags.add("fragile_execution")
    if baseline_fail_reasons:
        flags.add("baseline_failure")
    return _serialize_flags(flags)


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
        tags = [str(x) for x in r.get("tags", [])]
        spec = {
            "m2_idx": int(r.get("m2_idx", -1)),
            "m3_idx": int(r.get("m3_idx", -1)),
            "m4_idx": int(r.get("m4_idx", -1)),
            "tags": tags,
        }
        spec_blob = json.dumps(spec, sort_keys=True, separators=(",", ":"))
        sh = hashlib.sha256(spec_blob.encode("utf-8")).hexdigest()
        daily_ret = np.asarray(r.get("daily_returns", np.zeros(0, dtype=np.float64)), dtype=np.float64)
        meta = _stage_a_metadata(
            tags=tags,
            m2_idx=int(r.get("m2_idx", -1)),
            m3_idx=int(r.get("m3_idx", -1)),
            m4_idx=int(r.get("m4_idx", -1)),
        )
        overnight_score = _overnight_suitability_score([r])
        compliance_flags = _zimtra_compliance_flags(rows_all=[r], baseline_fail_reasons=[], verdict_row={})
        sharpe = float(np.mean(daily_ret) / np.std(daily_ret, ddof=1) * np.sqrt(252.0)) if daily_ret.size >= 2 and float(np.std(daily_ret, ddof=1)) > 0 else 0.0
        down = daily_ret[daily_ret < 0.0]
        sortino = float(np.mean(daily_ret) / np.std(down, ddof=1) * np.sqrt(252.0)) if down.size >= 2 and float(np.std(down, ddof=1)) > 0 else 0.0
        out.append(
            {
                "strategy_id": cid,
                "strategy_hash": sh,
                "parameter_values": spec_blob,
                "parameter_hash": str(meta["parameter_hash"]),
                "asset_count": int(len(r.get("asset_keys", []))),
                "total_trades": int(trade_count_from_payload_fn(r.get("trade_payload"))),
                "cost_adjusted_expectancy": float(np.mean(daily_ret)) if daily_ret.size > 0 else 0.0,
                "sharpe": float(sharpe),
                "sortino": float(sortino),
                "max_drawdown": float(max_drawdown_from_returns_fn(daily_ret)),
                "final_equity": float(extract_final_equity_fn(r)),
                "family_id": str(meta["family_id"]),
                "family_name": str(meta["family_name"]),
                "hypothesis_id": str(meta["hypothesis_id"]),
                "evaluation_role": str(meta["evaluation_role"]),
                "evaluation_window": meta["evaluation_window"],
                "window_set": str(meta["window_set"]),
                "overnight_suitability_score": float(overnight_score) if np.isfinite(overnight_score) else None,
                "zimtra_compliance_flags": compliance_flags,
                "tags": str(meta["tags_serialized"]),
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
        stage_a_meta = _stage_a_metadata(
            tags=list(cand.tags),
            m2_idx=int(cand.m2_idx),
            m3_idx=int(cand.m3_idx),
            m4_idx=int(cand.m4_idx),
        )
        overnight_score = _overnight_suitability_score(rows_all)
        compliance_flags = _zimtra_compliance_flags(
            rows_all=rows_all,
            baseline_fail_reasons=baseline_fail_reasons,
            verdict_row=verdict_row if isinstance(verdict_row, dict) else {},
        )
        evaluation_window = (
            int(stage_a_meta["evaluation_window"])
            if stage_a_meta.get("evaluation_window") is not None
            else int(getattr(m3_configs[int(cand.m3_idx)], "block_minutes", -1))
        )
        cost_adjusted_expectancy = float(np.mean(ret_series)) if ret_series.size > 0 else 0.0

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
            "block_minutes": float(evaluation_window),
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
            "tags": list(cand.tags),
            "stage_a_metadata": stage_a_meta,
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
            "stage_a_metadata": {
                **stage_a_meta,
                "cost_adjusted_expectancy": float(cost_adjusted_expectancy),
                "overnight_suitability_score": float(overnight_score) if np.isfinite(overnight_score) else None,
                "zimtra_compliance_flags": compliance_flags,
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
                "cost_adjusted_expectancy": float(cost_adjusted_expectancy),
                "overnight_suitability_score": float(overnight_score) if np.isfinite(overnight_score) else np.nan,
                "zimtra_compliance_flags": compliance_flags,
                "in_mcs": bool(verdict_row.get("in_mcs", False)),
                "pass": bool(verdict_row.get("pass", False)),
                "wrc_p": float(verdict_row.get("wrc_p", np.nan)) if verdict_row else np.nan,
                "spa_p": float(verdict_row.get("spa_p", np.nan)) if verdict_row else np.nan,
                "campaign_id": str(stage_a_meta["campaign_id"]),
                "family_id": str(stage_a_meta["family_id"]),
                "family_name": str(stage_a_meta["family_name"]),
                "hypothesis_id": str(stage_a_meta["hypothesis_id"]),
                "evaluation_role": str(stage_a_meta["evaluation_role"]),
                "evaluation_window": stage_a_meta["evaluation_window"],
                "window_set": str(stage_a_meta["window_set"]),
                "window_set_size": int(stage_a_meta["window_set_size"]),
                "parameter_hash": str(stage_a_meta["parameter_hash"]),
                "tags_serialized": str(stage_a_meta["tags_serialized"]),
                **feat,
            }
        )

    hypothesis_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in candidate_rows:
        family_id = str(row.get("family_id", "")).strip()
        hypothesis_id = str(row.get("hypothesis_id", "")).strip()
        if not family_id or not hypothesis_id:
            continue
        hypothesis_groups.setdefault((family_id, hypothesis_id), []).append(row)

    for group_rows in hypothesis_groups.values():
        probe_rows = sorted(
            [row for row in group_rows if str(row.get("evaluation_role", "")) == "window_probe"],
            key=lambda x: (int(x.get("evaluation_window") or -1), str(x.get("candidate_id", ""))),
        )
        live_rows = [
            row
            for row in group_rows
            if str(row.get("evaluation_role", "")) == "multi_window_live"
        ]
        expectancies = np.asarray(
            [float(row.get("cost_adjusted_expectancy", 0.0)) for row in probe_rows],
            dtype=np.float64,
        )
        finite_expectancies = expectancies[np.isfinite(expectancies)]
        if finite_expectancies.size > 0:
            pos = float(np.mean(finite_expectancies > 0.0))
            neg = float(np.mean(finite_expectancies < 0.0))
            dominant_sign_share = max(pos, neg, 0.0)
            pair_total = 0
            pair_conflict = 0
            for i in range(finite_expectancies.size):
                for j in range(i + 1, finite_expectancies.size):
                    si = float(np.sign(finite_expectancies[i]))
                    sj = float(np.sign(finite_expectancies[j]))
                    if si == 0.0 or sj == 0.0:
                        continue
                    pair_total += 1
                    if si != sj:
                        pair_conflict += 1
            conflict_score = float(pair_conflict / pair_total) if pair_total > 0 else 0.0
            scale = float(np.mean(np.abs(finite_expectancies)))
            stability = 1.0 - min(1.0, float(np.std(finite_expectancies)) / max(scale, 1.0e-12))
            consistency = 0.45 * dominant_sign_share + 0.25 * (1.0 - conflict_score) + 0.30 * stability
        else:
            conflict_score = 1.0
            stability = 0.0
            consistency = 0.0

        primary_rows = live_rows if live_rows else probe_rows
        hypothesis_expectancy = _safe_mean([float(row.get("cost_adjusted_expectancy", np.nan)) for row in primary_rows])
        window_set_size = int(group_rows[0].get("window_set_size", 0)) if group_rows else 0
        window_probe_completion = float(len(probe_rows) / max(window_set_size, 1)) if window_set_size > 0 else 0.0
        standard_reject_count = int(sum(bool(row.get("standard_reject", False)) for row in group_rows))
        standard_pass_count = int(sum(bool(row.get("standard_pass", False)) for row in group_rows))
        overnight_hypothesis = _safe_mean(
            [float(row.get("overnight_suitability_score", np.nan)) for row in group_rows]
        )

        for row in group_rows:
            row["cross_window_consistency_score"] = float(consistency)
            row["cross_window_conflict_score"] = float(conflict_score)
            row["multi_scale_stability_score"] = float(stability)
            row["hypothesis_cost_adjusted_expectancy"] = float(hypothesis_expectancy) if np.isfinite(hypothesis_expectancy) else np.nan
            row["hypothesis_window_probe_count"] = int(len(probe_rows))
            row["hypothesis_window_probe_completion"] = float(window_probe_completion)
            row["hypothesis_standard_reject_count"] = int(standard_reject_count)
            row["hypothesis_standard_pass_count"] = int(standard_pass_count)
            row["hypothesis_overnight_suitability_score"] = float(overnight_hypothesis) if np.isfinite(overnight_hypothesis) else np.nan

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
