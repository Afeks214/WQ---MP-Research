from __future__ import annotations

import traceback
from typing import Any, Callable

import numpy as np

_REQUIRED_WORKER_CONTEXT_KEYS = (
    "base_state",
    "candidates",
    "splits",
    "scenarios",
    "m2_configs",
    "m3_configs",
    "m4_configs",
    "harness_cfg",
)


def safe_execute_task(
    group: Any,
    executor_fn: Callable[[Any], list[dict[str, Any]]],
    candidates: list[Any],
    splits: list[Any],
    scenarios: list[Any],
    harness_cfg: Any,
    *,
    error_hash_fn: Callable[[str, str], str],
    normalized_top_frame_fn: Callable[[str], str],
    seed_for_task_fn: Callable[..., int],
) -> list[dict[str, Any]]:
    try:
        return executor_fn(group)
    except Exception as exc:
        split = splits[group.split_idx]
        scenario = scenarios[group.scenario_idx]
        err_type = type(exc).__name__
        err_msg = str(exc)
        err_hash = error_hash_fn(err_type, err_msg)
        tb = traceback.format_exc()
        top_frame = normalized_top_frame_fn(tb)
        sig = f"{err_type}|{top_frame}"
        reason_codes: list[str] = []
        if isinstance(exc, TimeoutError):
            reason_codes.append("TIMEOUT")
        out: list[dict[str, Any]] = []
        group_seed = seed_for_task_fn(
            int(harness_cfg.seed),
            str(split.split_id),
            str(scenario.scenario_id),
            str(group.m2_idx),
            str(group.m3_idx),
        )
        for ci in group.candidate_indices:
            c = candidates[int(ci)]
            task_id = f"{c.candidate_id}|{split.split_id}|{scenario.scenario_id}"
            t_seed = seed_for_task_fn(int(group_seed), str(c.candidate_id), str(c.m4_idx))
            out.append(
                {
                    "task_id": task_id,
                    "candidate_id": c.candidate_id,
                    "split_id": split.split_id,
                    "scenario_id": scenario.scenario_id,
                    "status": "error",
                    "error_type": err_type,
                    "error_hash": err_hash,
                    "error": f"{err_type}: {err_msg}",
                    "traceback": tb,
                    "top_frame": top_frame,
                    "exception_signature": sig,
                    "session_ids": np.zeros(0, dtype=np.int64),
                    "session_ids_exec": np.zeros(0, dtype=np.int64),
                    "session_ids_raw": np.zeros(0, dtype=np.int64),
                    "daily_returns": np.zeros(0, dtype=np.float64),
                    "daily_returns_exec": np.zeros(0, dtype=np.float64),
                    "daily_returns_raw": np.zeros(0, dtype=np.float64),
                    "equity_payload": None,
                    "trade_payload": None,
                    "asset_pnl_by_symbol": {},
                    "micro_payload": None,
                    "profile_payload": None,
                    "funnel_payload": None,
                    "m2_idx": int(c.m2_idx),
                    "m3_idx": int(c.m3_idx),
                    "m4_idx": int(c.m4_idx),
                    "tags": list(c.tags),
                    "test_days": 0,
                    "task_seed": int(t_seed),
                    "asset_keys": [],
                    "quality_reason_codes": sorted(reason_codes),
                    "dqs_min": 0.0,
                    "dqs_median": 0.0,
                }
            )
        return out


def run_group_task_from_context(
    group: Any,
    *,
    worker_context: dict[str, Any] | None,
    run_group_task_fn: Callable[..., list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if worker_context is None:
        raise RuntimeError("Worker context not initialized")
    missing = [key for key in _REQUIRED_WORKER_CONTEXT_KEYS if key not in worker_context]
    if missing:
        missing_keys = ", ".join(missing)
        raise RuntimeError(f"Worker context missing required keys: {missing_keys}")
    return run_group_task_fn(
        group=group,
        base_state=worker_context["base_state"],
        candidates=worker_context["candidates"],
        splits=worker_context["splits"],
        scenarios=worker_context["scenarios"],
        m2_configs=worker_context["m2_configs"],
        m3_configs=worker_context["m3_configs"],
        m4_configs=worker_context["m4_configs"],
        harness_cfg=worker_context["harness_cfg"],
    )
