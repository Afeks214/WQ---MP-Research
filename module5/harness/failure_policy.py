from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from module5.harness.artifact_writers import to_jsonable
from weightiz_module1_core import TensorState


def error_hash(error_type: str, error_msg: str) -> str:
    raw = f"{error_type}|{error_msg}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def record_deadletter(path: Path, row: dict[str, Any]) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "candidate_id": str(row.get("candidate_id", "")),
        "split_id": str(row.get("split_id", "")),
        "scenario_id": str(row.get("scenario_id", "")),
        "seed": int(row.get("task_seed", 0)),
        "error_type": str(row.get("error_type", "")),
        "error_hash": str(row.get("error_hash", "")),
        "error_msg": str(row.get("error", "")),
        "traceback_preview": str(row.get("traceback", ""))[:800],
        "exception_signature": str(row.get("exception_signature", "")),
        "reason_codes": sorted([str(x) for x in row.get("quality_reason_codes", [])]),
    }
    state_dump = row.get("state_dump")
    if isinstance(state_dump, dict):
        payload["state_dump"] = to_jsonable(state_dump)
    exec_px_dump = row.get("exec_px_dump")
    if isinstance(exec_px_dump, dict):
        payload["exec_px_dump"] = to_jsonable(exec_px_dump)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(payload), ensure_ascii=False) + "\n")


def should_abort_run(
    failure_count: int,
    total_tasks: int,
    failure_rate_threshold: float,
    failure_count_threshold: int,
) -> tuple[bool, str]:
    if total_tasks <= 0:
        return False, ""
    fail_rate = float(failure_count) / float(total_tasks)
    if int(failure_count) > int(failure_count_threshold):
        return True, f"failure_count>{int(failure_count_threshold)} ({int(failure_count)})"
    if fail_rate > float(failure_rate_threshold):
        return True, f"failure_rate>{float(failure_rate_threshold):.4f} ({fail_rate:.4f})"
    return False, ""


def normalized_top_frame(traceback_text: str) -> str:
    tb = str(traceback_text or "")
    lines = [ln.strip() for ln in tb.splitlines() if ln.strip()]
    frame_lines = [ln for ln in lines if ln.startswith('File "') and ", line " in ln and ", in " in ln]
    if not frame_lines:
        return "unknown:0:unknown"
    ln = frame_lines[-1]
    try:
        p1 = ln.split('File "', 1)[1]
        path_part, rest = p1.split('", line ', 1)
        line_part, func_part = rest.split(", in ", 1)
        base = os.path.basename(path_part)
        lineno = int(line_part.strip())
        fn = func_part.strip()
        return f"{base}:{lineno}:{fn}"
    except Exception:
        return "unknown:0:unknown"


def exception_signature(row: dict[str, Any]) -> tuple[str, str]:
    et = str(row.get("error_type", "")).strip() or "RuntimeError"
    top = str(row.get("top_frame", "")).strip()
    if not top:
        top = normalized_top_frame(str(row.get("traceback", "")))
    return (et, top)


def is_localized_reason_codes(reason_codes: list[str]) -> bool:
    rc = [str(x) for x in reason_codes]
    return any(
        c.startswith("DQ_")
        or c.startswith("INVARIANT_")
        or ("IB_MISSING" in c)
        or c == "TIMEOUT"
        or c == "RISK_CONSTRAINT_BREACH"
        or c == "NONFINITE_EXEC_PX"
        or c == "NEXT_OPEN_UNAVAILABLE"
        for c in rc
    )


def is_risk_constraint_breach(error_type: str, error_msg: str, top_frame: str, traceback_text: str = "") -> bool:
    if str(error_type).strip() not in {"RuntimeError", "ValueError", "AssertionError"}:
        return False
    top = str(top_frame).strip() or normalized_top_frame(str(traceback_text))
    top_low = top.lower()
    msg_low = str(error_msg).lower()
    if ("weightiz_module1_core.py" in top_low) and ("_validate_portfolio_constraints" in top_low):
        return True
    risk_terms = (
        "leverage breach",
        "portfolio constraints",
        "margin_used",
        "buying_power",
        "equity",
    )
    return any(term in msg_low for term in risk_terms)


def baseline_failure_reasons(
    rows_base_all: list[dict[str, Any]],
    expected_baseline_tasks: int,
) -> list[str]:
    rows_base_ok = [r for r in rows_base_all if str(r.get("status", "")) == "ok"]
    rows_base_err = [r for r in rows_base_all if str(r.get("status", "")) != "ok"]
    localized_err_rows = [
        r for r in rows_base_err if is_localized_reason_codes([str(x) for x in r.get("quality_reason_codes", [])])
    ]
    hard_err_rows = [r for r in rows_base_err if r not in localized_err_rows]
    effective_ok = int(len(rows_base_ok) + len(localized_err_rows))

    reasons: list[str] = []
    if int(expected_baseline_tasks) > 0 and effective_ok < int(expected_baseline_tasks):
        reasons.append(f"baseline_ok_tasks={effective_ok} expected={int(expected_baseline_tasks)}")
    for er in hard_err_rows:
        reasons.append(f"{str(er.get('split_id', ''))}:{str(er.get('error_type', 'error'))}")
    return reasons


def extract_breach_index(error_msg: str, state: TensorState | None) -> int:
    m = re.search(r"\bt\s*=\s*(-?\d+)", str(error_msg))
    if m is not None:
        idx = int(m.group(1))
        if state is None:
            return max(idx, 0)
        return int(min(max(idx, 0), state.cfg.T - 1))
    if state is None:
        return 0
    eq = np.asarray(state.equity, dtype=np.float64)
    mg = np.asarray(state.margin_used, dtype=np.float64)
    lev = np.asarray(state.leverage_limit, dtype=np.float64)
    valid = np.isfinite(eq) & np.isfinite(mg) & np.isfinite(lev)
    breach = valid & (mg > (eq * lev))
    if np.any(breach):
        return int(np.where(breach)[0][0])
    return int(max(state.cfg.T - 1, 0))


def build_risk_constraint_state_dump(
    state: TensorState,
    t: int,
    candidate_id: str,
    split_id: str,
    scenario_id: str,
) -> dict[str, Any]:
    idx = int(min(max(int(t), 0), state.cfg.T - 1))
    ts_ns = int(state.ts_ns[idx])
    ts_utc = datetime.fromtimestamp(float(ts_ns) / 1_000_000_000.0, tz=timezone.utc).isoformat()
    close_row = np.asarray(state.close_px[idx], dtype=np.float64)
    pos_row = np.asarray(state.position_qty[idx], dtype=np.float64)
    position_value = pos_row * close_row
    per_asset = []
    for a, sym in enumerate(state.symbols):
        per_asset.append(
            {
                "a": int(a),
                "symbol": str(sym),
                "position_qty": float(pos_row[a]),
                "position_value": float(position_value[a]),
                "close_px": float(close_row[a]),
            }
        )
    equity_t = float(state.equity[idx])
    leverage_limit_t = float(state.leverage_limit[idx])
    return {
        "t": int(idx),
        "ts_utc": ts_utc,
        "candidate_id": str(candidate_id),
        "split_id": str(split_id),
        "scenario_id": str(scenario_id),
        "equity_t": equity_t,
        "margin_used_t": float(state.margin_used[idx]),
        "leverage_limit_t": leverage_limit_t,
        "buying_power_t": float(state.buying_power[idx]),
        "cash_t": float(state.available_cash[idx]),
        "realized_pnl_t": float(state.realized_pnl[idx]),
        "max_margin_allowed_t": float(equity_t * leverage_limit_t),
        "assets": per_asset,
    }


def is_high_suspicion_exception(error_type: str) -> bool:
    return str(error_type) in {"ImportError", "TypeError", "KeyError", "IndexError"}


def update_failure_tracker(
    tracker: dict[tuple[str, str], dict[str, set[str] | bool]],
    row: dict[str, Any],
) -> tuple[tuple[str, str], dict[str, set[str] | bool]]:
    sig = exception_signature(row)
    rec = tracker.setdefault(
        sig,
        {
            "units": set(),
            "assets": set(),
            "candidates": set(),
            "high_suspicion": is_high_suspicion_exception(sig[0]),
        },
    )
    rec["units"].add(str(row.get("task_id", "")))
    rec["candidates"].add(str(row.get("candidate_id", "")))
    assets = [str(x) for x in row.get("asset_keys", [])]
    for a in assets:
        rec["assets"].add(a)
    return sig, rec


def should_abort_systemic(
    tracker: dict[tuple[str, str], dict[str, set[str] | bool]],
    row: dict[str, Any],
) -> tuple[bool, str]:
    if is_localized_reason_codes([str(x) for x in row.get("quality_reason_codes", [])]):
        return False, ""
    sig = exception_signature(row)
    rec = tracker.get(sig)
    if rec is None:
        return False, ""
    n_units = len(rec["units"])
    n_assets = len(rec["assets"])
    n_candidates = len(rec["candidates"])
    if n_units >= 3 and n_assets >= 2 and n_candidates >= 2:
        suspicion = "high" if bool(rec.get("high_suspicion", False)) else "standard"
        reason = (
            f"systemic_exception signature={sig[0]}|{sig[1]} "
            f"units={n_units} assets={n_assets} candidates={n_candidates} suspicion={suspicion}"
        )
        return True, reason
    return False, ""
