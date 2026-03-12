from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from weightiz.module4.telemetry import DecisionReasonCode

EPS = 1.0e-12
MEANINGFUL_ALLOCATION_SCORE_FLOOR = 0.05


def _abs_stats(values: Any) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.abs(arr[np.isfinite(arr)])
    if finite.size <= 0:
        return {
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    return {
        "mean": float(np.mean(finite)),
        "p50": float(np.percentile(finite, 50.0)),
        "p95": float(np.percentile(finite, 95.0)),
        "max": float(np.max(finite)),
    }


def _count_reason_codes(reason_code_ta: np.ndarray, code: DecisionReasonCode) -> int:
    return int(np.sum(np.asarray(reason_code_ta, dtype=np.int16) == np.int16(code)))


def _risk_breach_flags(error_type: str, error_message: str) -> dict[str, int]:
    txt = f"{error_type}|{error_message}".upper()
    return {
        "daily_loss_breach_count": int("RISK_DAILY_LOSS_BREACH" in txt),
        "account_disable_breach_count": int("RISK_ACCOUNT_DISABLE_THRESHOLD" in txt),
        "overnight_exposure_breach_count": int("RISK_OVERNIGHT_EXPOSURE_BREACH" in txt),
    }


def build_candidate_path_diagnostics(
    *,
    task_id: str,
    candidate_id: str,
    split_id: str,
    scenario_id: str,
    status: str,
    m2_idx: int,
    m3_idx: int,
    m4_idx: int,
    enabled_assets_mask: np.ndarray,
    quality_reason_codes: list[str],
    m4_sig: Any,
    target_qty_exec: np.ndarray,
    risk_res_exec: Any,
    trade_payload: dict[str, np.ndarray] | None,
) -> dict[str, Any]:
    enabled = np.asarray(enabled_assets_mask, dtype=bool)
    if enabled.ndim != 1:
        raise RuntimeError(f"enabled_assets_mask must be 1D, got shape={enabled.shape}")
    active_assets = int(np.sum(enabled))
    target_weight_raw_ta = getattr(m4_sig, "target_weight_ta", None)
    if target_weight_raw_ta is None:
        target_weight_raw_ta = np.asarray(m4_sig.target_qty_ta, dtype=np.float64)
    else:
        target_weight_raw_ta = np.asarray(target_weight_raw_ta, dtype=np.float64)
    if target_weight_raw_ta.ndim != 2:
        raise RuntimeError(f"module4 target weight telemetry must be 2D, got shape={target_weight_raw_ta.shape}")
    if enabled.shape[0] != target_weight_raw_ta.shape[1]:
        raise RuntimeError(
            f"enabled_assets_mask length mismatch: got {enabled.shape[0]}, expected {target_weight_raw_ta.shape[1]}"
        )

    target_qty_exec_ta = np.asarray(target_qty_exec, dtype=np.float64)
    regime_confidence_ta = np.asarray(m4_sig.regime_confidence_ta, dtype=np.float64)
    intent_long_ta = np.asarray(m4_sig.intent_long_ta, dtype=bool)
    intent_short_ta = np.asarray(m4_sig.intent_short_ta, dtype=bool)
    conviction_net_ta = getattr(m4_sig, "conviction_net_ta", None)
    allocation_score_ta = getattr(m4_sig, "allocation_score_ta", None)
    decision_reason_code_ta = getattr(m4_sig, "decision_reason_code_ta", None)

    if conviction_net_ta is None:
        conviction_net_ta = np.zeros_like(target_weight_raw_ta, dtype=np.float64)
    else:
        conviction_net_ta = np.asarray(conviction_net_ta, dtype=np.float64)
    if allocation_score_ta is None:
        allocation_score_ta = np.zeros_like(target_weight_raw_ta, dtype=np.float64)
    else:
        allocation_score_ta = np.asarray(allocation_score_ta, dtype=np.float64)
    if decision_reason_code_ta is None:
        decision_reason_code_ta = np.zeros_like(target_weight_raw_ta, dtype=np.int16)
    else:
        decision_reason_code_ta = np.asarray(decision_reason_code_ta, dtype=np.int16)

    target_weight_raw = target_weight_raw_ta[:, enabled]
    target_qty_exec = target_qty_exec_ta[:, enabled]
    regime_confidence = regime_confidence_ta[:, enabled]
    intent_long = intent_long_ta[:, enabled]
    intent_short = intent_short_ta[:, enabled]
    conviction_net = conviction_net_ta[:, enabled]
    allocation_score = allocation_score_ta[:, enabled]
    reason_code = decision_reason_code_ta[:, enabled]

    signal_mask = intent_long | intent_short | (np.abs(conviction_net) > EPS)
    signal_bars_any = np.any(signal_mask, axis=1)
    confidence_bars_any = np.any(regime_confidence > EPS, axis=1)
    allocation_raw_bars_any = np.any(np.abs(target_weight_raw) > EPS, axis=1)
    allocation_exec_bars_any = np.any(np.abs(target_qty_exec) > EPS, axis=1)
    meaningful_signal_mask = np.abs(allocation_score) >= float(MEANINGFUL_ALLOCATION_SCORE_FLOOR)
    meaningful_signal_bars_any = np.any(meaningful_signal_mask, axis=1)

    trade_payload = trade_payload or {}
    trade_rows = int(np.asarray(trade_payload.get("filled_qty", np.zeros(0, dtype=np.float64)), dtype=np.float64).size)
    risk_diag = dict(getattr(risk_res_exec, "execution_diagnostics", {}) or {})
    desired_fill_attempt_count = int(risk_diag.get("desired_fill_attempt_count", 0))
    filled_trade_count = int(risk_diag.get("filled_trade_count", 0))

    score_stats = _abs_stats(allocation_score)
    conviction_stats = _abs_stats(conviction_net)

    row = {
        "task_id": str(task_id),
        "candidate_id": str(candidate_id),
        "split_id": str(split_id),
        "scenario_id": str(scenario_id),
        "status": str(status),
        "m2_idx": int(m2_idx),
        "m3_idx": int(m3_idx),
        "m4_idx": int(m4_idx),
        "enabled_asset_count": int(active_assets),
        "quality_reason_codes": list(str(x) for x in quality_reason_codes),
        "signal_bars_any": int(np.sum(signal_bars_any)),
        "confidence_bars_any": int(np.sum(confidence_bars_any)),
        "allocation_raw_bars_any": int(np.sum(allocation_raw_bars_any)),
        "allocation_exec_bars_any": int(np.sum(allocation_exec_bars_any)),
        "conviction_nonzero_cells": int(np.sum(np.abs(conviction_net) > EPS)),
        "confidence_nonzero_cells": int(np.sum(regime_confidence > EPS)),
        "allocation_score_nonzero_cells": int(np.sum(np.abs(allocation_score) > EPS)),
        "allocation_score_abs_mean": float(score_stats["mean"]),
        "allocation_score_abs_p50": float(score_stats["p50"]),
        "allocation_score_abs_p95": float(score_stats["p95"]),
        "allocation_score_abs_max": float(score_stats["max"]),
        "conviction_abs_max": float(conviction_stats["max"]),
        "meaningful_allocation_score_floor": float(MEANINGFUL_ALLOCATION_SCORE_FLOOR),
        "meaningful_signal_bars_any": int(np.sum(meaningful_signal_bars_any)),
        "never_crosses_meaningful_signal": bool(not np.any(meaningful_signal_mask)),
        "reason_low_regime_confidence_cells": _count_reason_codes(reason_code, DecisionReasonCode.LOW_REGIME_CONFIDENCE),
        "reason_zero_score_cells": _count_reason_codes(reason_code, DecisionReasonCode.ZERO_SCORE_AFTER_MASK),
        "reason_zero_conviction_cells": _count_reason_codes(reason_code, DecisionReasonCode.ZERO_CONVICTION),
        "reason_risk_filter_block_cells": _count_reason_codes(reason_code, DecisionReasonCode.RISK_FILTER_BLOCK),
        "reason_masked_not_tradable_cells": _count_reason_codes(reason_code, DecisionReasonCode.MASKED_NOT_TRADABLE),
        "reason_invalid_input_cells": _count_reason_codes(reason_code, DecisionReasonCode.INVALID_INPUT),
        "desired_fill_attempt_count": desired_fill_attempt_count,
        "desired_fill_qty_abs_sum": float(risk_diag.get("desired_fill_qty_abs_sum", 0.0)),
        "filled_trade_count": filled_trade_count,
        "filled_qty_abs_sum": float(risk_diag.get("filled_qty_abs_sum", 0.0)),
        "volume_cap_hit_count": int(risk_diag.get("volume_cap_hit_count", 0)),
        "volume_cap_rejected_count": int(risk_diag.get("volume_cap_rejected_count", 0)),
        "volume_cap_desired_qty_abs_sum": float(risk_diag.get("volume_cap_desired_qty_abs_sum", 0.0)),
        "volume_cap_filled_qty_abs_sum": float(risk_diag.get("volume_cap_filled_qty_abs_sum", 0.0)),
        "volume_cap_clipped_qty_abs_sum": float(risk_diag.get("volume_cap_clipped_qty_abs_sum", 0.0)),
        "buying_power_cap_hit_count": int(risk_diag.get("buying_power_cap_hit_count", 0)),
        "buying_power_cap_desired_qty_abs_sum": float(risk_diag.get("buying_power_cap_desired_qty_abs_sum", 0.0)),
        "buying_power_cap_filled_qty_abs_sum": float(risk_diag.get("buying_power_cap_filled_qty_abs_sum", 0.0)),
        "buying_power_cap_clipped_qty_abs_sum": float(risk_diag.get("buying_power_cap_clipped_qty_abs_sum", 0.0)),
        "daily_loss_breach_count": 0,
        "account_disable_breach_count": 0,
        "overnight_exposure_breach_count": 0,
        "trade_log_rows": int(trade_rows),
        "candidate_generated": 1,
        "has_signal_opportunity": bool(np.any(signal_bars_any)),
        "has_allocation_raw": bool(np.any(allocation_raw_bars_any)),
        "has_allocation_exec": bool(np.any(allocation_exec_bars_any)),
        "has_fill_attempt": bool(desired_fill_attempt_count > 0),
        "has_filled_trade": bool(filled_trade_count > 0),
        "has_trade_log": bool(trade_rows > 0),
    }
    row["no_setup_candidate"] = bool(not row["has_signal_opportunity"])
    row["weak_signal_candidate"] = bool(row["never_crosses_meaningful_signal"])
    row["zero_allocation_candidate"] = bool(row["has_signal_opportunity"] and (not row["has_allocation_exec"]))
    row["risk_choke_candidate"] = bool(
        row["has_fill_attempt"]
        and (not row["has_filled_trade"])
        and (
            int(row["buying_power_cap_hit_count"]) > 0
            or int(row["daily_loss_breach_count"]) > 0
            or int(row["account_disable_breach_count"]) > 0
            or int(row["overnight_exposure_breach_count"]) > 0
        )
    )
    row["volume_cap_choke_candidate"] = bool(
        row["has_fill_attempt"]
        and (not row["has_filled_trade"])
        and int(row["volume_cap_hit_count"]) > 0
    )
    row["other_blocker_candidate"] = bool(
        row["has_fill_attempt"]
        and (not row["has_filled_trade"])
        and (not row["risk_choke_candidate"])
        and (not row["volume_cap_choke_candidate"])
    )
    return row


def build_error_candidate_path_diagnostics(
    *,
    task_id: str,
    candidate_id: str,
    split_id: str,
    scenario_id: str,
    status: str,
    m2_idx: int,
    m3_idx: int,
    m4_idx: int,
    enabled_asset_count: int,
    quality_reason_codes: list[str],
    error_type: str,
    error_message: str,
) -> dict[str, Any]:
    breach = _risk_breach_flags(error_type, error_message)
    row = {
        "task_id": str(task_id),
        "candidate_id": str(candidate_id),
        "split_id": str(split_id),
        "scenario_id": str(scenario_id),
        "status": str(status),
        "m2_idx": int(m2_idx),
        "m3_idx": int(m3_idx),
        "m4_idx": int(m4_idx),
        "enabled_asset_count": int(enabled_asset_count),
        "quality_reason_codes": list(str(x) for x in quality_reason_codes),
        "signal_bars_any": 0,
        "confidence_bars_any": 0,
        "allocation_raw_bars_any": 0,
        "allocation_exec_bars_any": 0,
        "conviction_nonzero_cells": 0,
        "confidence_nonzero_cells": 0,
        "allocation_score_nonzero_cells": 0,
        "allocation_score_abs_mean": 0.0,
        "allocation_score_abs_p50": 0.0,
        "allocation_score_abs_p95": 0.0,
        "allocation_score_abs_max": 0.0,
        "conviction_abs_max": 0.0,
        "meaningful_allocation_score_floor": float(MEANINGFUL_ALLOCATION_SCORE_FLOOR),
        "meaningful_signal_bars_any": 0,
        "never_crosses_meaningful_signal": True,
        "reason_low_regime_confidence_cells": 0,
        "reason_zero_score_cells": 0,
        "reason_zero_conviction_cells": 0,
        "reason_risk_filter_block_cells": 0,
        "reason_masked_not_tradable_cells": 0,
        "reason_invalid_input_cells": 0,
        "desired_fill_attempt_count": 0,
        "desired_fill_qty_abs_sum": 0.0,
        "filled_trade_count": 0,
        "filled_qty_abs_sum": 0.0,
        "volume_cap_hit_count": 0,
        "volume_cap_rejected_count": 0,
        "volume_cap_desired_qty_abs_sum": 0.0,
        "volume_cap_filled_qty_abs_sum": 0.0,
        "volume_cap_clipped_qty_abs_sum": 0.0,
        "buying_power_cap_hit_count": 0,
        "buying_power_cap_desired_qty_abs_sum": 0.0,
        "buying_power_cap_filled_qty_abs_sum": 0.0,
        "buying_power_cap_clipped_qty_abs_sum": 0.0,
        "trade_log_rows": 0,
        "candidate_generated": 1,
        "has_signal_opportunity": False,
        "has_allocation_raw": False,
        "has_allocation_exec": False,
        "has_fill_attempt": False,
        "has_filled_trade": False,
        "has_trade_log": False,
        "error_type": str(error_type),
        "error_message": str(error_message),
        **breach,
    }
    row["no_setup_candidate"] = True
    row["weak_signal_candidate"] = True
    row["zero_allocation_candidate"] = True
    row["risk_choke_candidate"] = bool(
        int(row["daily_loss_breach_count"]) > 0
        or int(row["account_disable_breach_count"]) > 0
        or int(row["overnight_exposure_breach_count"]) > 0
    )
    row["volume_cap_choke_candidate"] = False
    row["other_blocker_candidate"] = bool(not row["risk_choke_candidate"])
    return row


def _fraction(numer: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return float(numer) / float(denom)


def _clip_ratio_stats(candidate_df: Any) -> dict[str, float]:
    desired = np.asarray(candidate_df["desired_fill_qty_abs_sum"], dtype=np.float64)
    filled = np.asarray(candidate_df["filled_qty_abs_sum"], dtype=np.float64)
    mask = desired > EPS
    if not np.any(mask):
        return {"mean": 0.0, "median": 0.0, "max": 0.0}
    ratio = 1.0 - np.clip(filled[mask] / np.maximum(desired[mask], EPS), 0.0, 1.0)
    return {
        "mean": float(np.mean(ratio)),
        "median": float(np.median(ratio)),
        "max": float(np.max(ratio)),
    }


def write_trade_blocker_artifacts(
    *,
    all_results: list[dict[str, Any]],
    report_root: Path,
    require_pandas_fn: Callable[[], Any],
    write_json_fn: Callable[[Path, Any], None],
) -> tuple[dict[str, str], dict[str, Any]]:
    pdx = require_pandas_fn()
    rows: list[dict[str, Any]] = []
    for r in all_results:
        diag = r.get("execution_path_diagnostics")
        if isinstance(diag, dict):
            rows.append(dict(diag))
            continue
        rows.append(
            build_error_candidate_path_diagnostics(
                task_id=str(r.get("task_id", "")),
                candidate_id=str(r.get("candidate_id", "")),
                split_id=str(r.get("split_id", "")),
                scenario_id=str(r.get("scenario_id", "")),
                status=str(r.get("status", "error")),
                m2_idx=int(r.get("m2_idx", 0)),
                m3_idx=int(r.get("m3_idx", 0)),
                m4_idx=int(r.get("m4_idx", 0)),
                enabled_asset_count=len(r.get("asset_keys", [])),
                quality_reason_codes=[str(x) for x in r.get("quality_reason_codes", [])],
                error_type=str(r.get("error_type", "")),
                error_message=str(r.get("error", "")),
            )
        )

    candidate_df = pdx.DataFrame(rows)
    if candidate_df.shape[0] > 0:
        candidate_df = candidate_df.sort_values(
            ["candidate_id", "split_id", "scenario_id", "task_id"],
            kind="mergesort",
        ).reset_index(drop=True)

    total_paths = int(candidate_df.shape[0])
    no_setup_paths = int(candidate_df["no_setup_candidate"].sum()) if total_paths else 0
    weak_signal_paths = int(candidate_df["weak_signal_candidate"].sum()) if total_paths else 0
    zero_allocation_paths = int(candidate_df["zero_allocation_candidate"].sum()) if total_paths else 0
    risk_choke_paths = int(candidate_df["risk_choke_candidate"].sum()) if total_paths else 0
    volume_cap_choke_paths = int(candidate_df["volume_cap_choke_candidate"].sum()) if total_paths else 0
    other_paths = int(candidate_df["other_blocker_candidate"].sum()) if total_paths else 0

    funnel_rows: list[dict[str, Any]] = []
    if total_paths:
        group_cols = [("overall", None), ("split_scenario", ["split_id", "scenario_id"])]
        for level, cols in group_cols:
            if cols is None:
                grouped = [(("overall",), candidate_df)]
            else:
                grouped = list(candidate_df.groupby(cols, dropna=False, sort=True))
            for key, frame in grouped:
                paths = int(frame.shape[0])
                row = {
                    "level": str(level),
                    "paths_total": paths,
                    "paths_with_signal": int(frame["has_signal_opportunity"].sum()),
                    "paths_with_allocation_exec": int(frame["has_allocation_exec"].sum()),
                    "paths_with_fill_attempt": int(frame["has_fill_attempt"].sum()),
                    "paths_with_filled_trade": int(frame["has_filled_trade"].sum()),
                    "paths_with_trade_log": int(frame["has_trade_log"].sum()),
                    "paths_no_setup": int(frame["no_setup_candidate"].sum()),
                    "paths_weak_signal": int(frame["weak_signal_candidate"].sum()),
                    "paths_zero_allocation": int(frame["zero_allocation_candidate"].sum()),
                    "paths_risk_choke": int(frame["risk_choke_candidate"].sum()),
                    "paths_volume_cap_choke": int(frame["volume_cap_choke_candidate"].sum()),
                    "paths_other_blocker": int(frame["other_blocker_candidate"].sum()),
                }
                row["signal_fraction"] = _fraction(int(row["paths_with_signal"]), paths)
                row["allocation_fraction"] = _fraction(int(row["paths_with_allocation_exec"]), paths)
                row["fill_attempt_fraction"] = _fraction(int(row["paths_with_fill_attempt"]), paths)
                row["filled_fraction"] = _fraction(int(row["paths_with_filled_trade"]), paths)
                row["trade_log_fraction"] = _fraction(int(row["paths_with_trade_log"]), paths)
                if cols is None:
                    row["split_id"] = "ALL"
                    row["scenario_id"] = "ALL"
                else:
                    split_id, scenario_id = key
                    row["split_id"] = str(split_id)
                    row["scenario_id"] = str(scenario_id)
                funnel_rows.append(row)
    funnel_df = pdx.DataFrame(funnel_rows)
    if funnel_df.shape[0] > 0:
        funnel_df = funnel_df.sort_values(["level", "split_id", "scenario_id"], kind="mergesort").reset_index(drop=True)

    clip_stats = _clip_ratio_stats(candidate_df) if total_paths else {"mean": 0.0, "median": 0.0, "max": 0.0}
    ranking = [
        {
            "blocker": name,
            "affected_paths": int(count),
            "fraction_of_paths": _fraction(int(count), total_paths),
        }
        for name, count in sorted(
            {
                "WEAK_SIGNALS": weak_signal_paths,
                "ZERO_ALLOCATION": zero_allocation_paths,
                "RISK_CHOKE": risk_choke_paths,
                "VOLUME_CAP_CHOKE": volume_cap_choke_paths,
                "NO_SETUPS": no_setup_paths,
                "OTHER": other_paths,
            }.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )
    ]

    summary = {
        "paths_total": total_paths,
        "thresholds": {
            "meaningful_allocation_score_floor": float(MEANINGFUL_ALLOCATION_SCORE_FLOOR),
            "eps": float(EPS),
        },
        "candidate_survival_funnel": {
            "generated": total_paths,
            "with_signal": int(candidate_df["has_signal_opportunity"].sum()) if total_paths else 0,
            "with_allocation_exec": int(candidate_df["has_allocation_exec"].sum()) if total_paths else 0,
            "with_fill_attempt": int(candidate_df["has_fill_attempt"].sum()) if total_paths else 0,
            "with_filled_trade": int(candidate_df["has_filled_trade"].sum()) if total_paths else 0,
            "with_trade_log": int(candidate_df["has_trade_log"].sum()) if total_paths else 0,
        },
        "signal_analysis": {
            "candidate_paths_with_signal": int(candidate_df["has_signal_opportunity"].sum()) if total_paths else 0,
            "candidate_paths_without_signal": no_setup_paths,
            "candidate_paths_never_cross_meaningful_signal": weak_signal_paths,
            "allocation_score_abs_mean": float(candidate_df["allocation_score_abs_mean"].mean()) if total_paths else 0.0,
            "allocation_score_abs_p95_mean": float(candidate_df["allocation_score_abs_p95"].mean()) if total_paths else 0.0,
            "allocation_score_abs_max_max": float(candidate_df["allocation_score_abs_max"].max()) if total_paths else 0.0,
            "conviction_abs_max_max": float(candidate_df["conviction_abs_max"].max()) if total_paths else 0.0,
        },
        "allocation_analysis": {
            "candidate_paths_zero_allocation": zero_allocation_paths,
            "candidate_paths_with_allocation_raw": int(candidate_df["has_allocation_raw"].sum()) if total_paths else 0,
            "candidate_paths_with_allocation_exec": int(candidate_df["has_allocation_exec"].sum()) if total_paths else 0,
            "allocation_exec_bar_fraction_mean": float(
                np.mean(
                    np.asarray(candidate_df["allocation_exec_bars_any"], dtype=np.float64)
                    / np.maximum(np.asarray(candidate_df["signal_bars_any"], dtype=np.float64), 1.0)
                )
            ) if total_paths else 0.0,
        },
        "risk_choke_analysis": {
            "candidate_paths_risk_choke": risk_choke_paths,
            "buying_power_cap_hit_total": int(candidate_df["buying_power_cap_hit_count"].sum()) if total_paths else 0,
            "daily_loss_breach_paths": int(candidate_df["daily_loss_breach_count"].sum()) if total_paths else 0,
            "account_disable_paths": int(candidate_df["account_disable_breach_count"].sum()) if total_paths else 0,
            "overnight_exposure_breach_paths": int(candidate_df["overnight_exposure_breach_count"].sum()) if total_paths else 0,
            "buying_power_clipped_qty_abs_sum": float(candidate_df["buying_power_cap_clipped_qty_abs_sum"].sum()) if total_paths else 0.0,
        },
        "volume_cap_analysis": {
            "candidate_paths_volume_cap_choke": volume_cap_choke_paths,
            "volume_cap_hit_total": int(candidate_df["volume_cap_hit_count"].sum()) if total_paths else 0,
            "volume_cap_rejected_total": int(candidate_df["volume_cap_rejected_count"].sum()) if total_paths else 0,
            "desired_fill_qty_abs_sum_total": float(candidate_df["desired_fill_qty_abs_sum"].sum()) if total_paths else 0.0,
            "filled_qty_abs_sum_total": float(candidate_df["filled_qty_abs_sum"].sum()) if total_paths else 0.0,
            "volume_cap_clipped_qty_abs_sum_total": float(candidate_df["volume_cap_clipped_qty_abs_sum"].sum()) if total_paths else 0.0,
            "fill_clipping_ratio_mean": float(clip_stats["mean"]),
            "fill_clipping_ratio_median": float(clip_stats["median"]),
            "fill_clipping_ratio_max": float(clip_stats["max"]),
        },
        "setup_universe_analysis": {
            "candidate_paths_no_setup": no_setup_paths,
            "candidate_paths_no_fill_attempt": int((~candidate_df["has_fill_attempt"]).sum()) if total_paths else 0,
            "candidate_paths_no_filled_trade": int((~candidate_df["has_filled_trade"]).sum()) if total_paths else 0,
            "candidate_paths_filled_but_no_trade_log": int(
                ((candidate_df["has_filled_trade"]) & (~candidate_df["has_trade_log"])).sum()
            ) if total_paths else 0,
        },
        "dominant_blocker_ranking": ranking,
    }

    candidate_path = report_root / "candidate_path_diagnostics.parquet"
    funnel_path = report_root / "execution_funnel_diagnostics.parquet"
    summary_path = report_root / "trade_blocker_summary.json"
    candidate_df.to_parquet(candidate_path, index=False)
    funnel_df.to_parquet(funnel_path, index=False)
    write_json_fn(summary_path, summary)
    return (
        {
            "candidate_path_diagnostics": str(candidate_path),
            "execution_funnel_diagnostics": str(funnel_path),
            "trade_blocker_summary": str(summary_path),
        },
        summary,
    )
