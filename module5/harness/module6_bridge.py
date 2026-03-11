from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

import numpy as np


AVAIL_OBSERVED_ACTIVE = 1
AVAIL_OBSERVED_FLAT = 2
AVAIL_STRUCTURALLY_MISSING = 3
AVAIL_FORCED_ZERO_BY_PORTFOLIO = 4
AVAIL_FORCED_CASH_BY_RISK = 5
AVAIL_INVALIDATED_BY_DQ = 6

_SELECTION_STAGE = "module5_bridge_canonical_baseline_v1"


def _calendar_version(common_sessions: np.ndarray) -> str:
    sess = np.asarray(common_sessions, dtype=np.int64).reshape(-1)
    if sess.size <= 0:
        raise RuntimeError("common_sessions must be non-empty for module6 bridge persistence")
    blob = "|".join(str(int(x)) for x in sess.tolist()).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _series_map(
    session_ids: np.ndarray | list[int] | None,
    values: np.ndarray | list[float] | None,
) -> dict[int, float]:
    sess = np.asarray(session_ids if session_ids is not None else np.zeros(0, dtype=np.int64), dtype=np.int64)
    vals = np.asarray(values if values is not None else np.zeros(0, dtype=np.float64), dtype=np.float64)
    if sess.size != vals.size:
        raise RuntimeError(
            f"session/value size mismatch in module6 bridge; session_count={int(sess.size)} value_count={int(vals.size)}"
        )
    out: dict[int, float] = {}
    for s, v in zip(sess.tolist(), vals.tolist()):
        vv = float(v)
        if not np.isfinite(vv):
            raise RuntimeError(f"non-finite session return in module6 bridge; session_id={int(s)} value={vv}")
        out[int(s)] = vv
    return out


def _turnover_by_session(trade_payload: dict[str, np.ndarray] | None, ts_to_session: dict[int, int], initial_cash: float) -> dict[int, float]:
    if not trade_payload:
        return {}
    ts = np.asarray(trade_payload.get("ts_ns", np.zeros(0, dtype=np.int64)), dtype=np.int64)
    qty = np.asarray(trade_payload.get("filled_qty", np.zeros(0, dtype=np.float64)), dtype=np.float64)
    px = np.asarray(trade_payload.get("exec_price", np.zeros(0, dtype=np.float64)), dtype=np.float64)
    if ts.size == 0 or qty.size == 0 or px.size == 0:
        return {}
    n = min(ts.size, qty.size, px.size)
    if n <= 0:
        return {}
    out: dict[int, float] = {}
    for i in range(n):
        session_id = ts_to_session.get(int(ts[i]))
        if session_id is None:
            continue
        notional = float(abs(float(qty[i]) * float(px[i])))
        out[session_id] = out.get(session_id, 0.0) + notional / max(float(initial_cash), 1.0e-12)
    return out


def _trade_count_by_session(trade_payload: dict[str, np.ndarray] | None, ts_to_session: dict[int, int]) -> dict[int, int]:
    if not trade_payload:
        return {}
    ts = np.asarray(trade_payload.get("ts_ns", np.zeros(0, dtype=np.int64)), dtype=np.int64)
    qty = np.asarray(trade_payload.get("filled_qty", np.zeros(0, dtype=np.float64)), dtype=np.float64)
    if ts.size == 0 or qty.size == 0:
        return {}
    n = min(ts.size, qty.size)
    out: dict[int, int] = {}
    for i in range(n):
        if abs(float(qty[i])) <= 1.0e-12:
            continue
        session_id = ts_to_session.get(int(ts[i]))
        if session_id is None:
            continue
        out[session_id] = out.get(session_id, 0) + 1
    return out


def _equity_stats_by_session(equity_payload: dict[str, np.ndarray] | None) -> tuple[dict[int, float], dict[int, float], dict[int, float], dict[int, float]]:
    if not equity_payload:
        return {}, {}, {}, {}
    session_id = np.asarray(equity_payload.get("session_id", np.zeros(0, dtype=np.int64)), dtype=np.int64)
    equity = np.asarray(equity_payload.get("equity", np.zeros(0, dtype=np.float64)), dtype=np.float64)
    margin_used = np.asarray(equity_payload.get("margin_used", np.zeros(0, dtype=np.float64)), dtype=np.float64)
    buying_power = np.asarray(equity_payload.get("buying_power", np.zeros(0, dtype=np.float64)), dtype=np.float64)
    daily_loss = np.asarray(equity_payload.get("daily_loss", np.zeros(0, dtype=np.float64)), dtype=np.float64)
    n = min(session_id.size, equity.size, margin_used.size, buying_power.size, daily_loss.size)
    if n <= 0:
        return {}, {}, {}, {}
    gross_frac: dict[int, list[float]] = {}
    buying_power_min: dict[int, float] = {}
    daily_loss_max: dict[int, float] = {}
    for i in range(n):
        s = int(session_id[i])
        eq = max(abs(float(equity[i])), 1.0e-12)
        gm = abs(float(margin_used[i])) / eq
        gross_frac.setdefault(s, []).append(gm)
        bp = float(buying_power[i])
        dl = float(daily_loss[i])
        buying_power_min[s] = bp if s not in buying_power_min else min(buying_power_min[s], bp)
        daily_loss_max[s] = dl if s not in daily_loss_max else max(daily_loss_max[s], dl)
    gross_mean = {k: float(np.mean(np.asarray(v, dtype=np.float64))) for k, v in gross_frac.items()}
    gross_peak = {k: float(np.max(np.asarray(v, dtype=np.float64))) for k, v in gross_frac.items()}
    return gross_mean, gross_peak, buying_power_min, daily_loss_max


def _overnight_by_session(micro_payload: dict[str, np.ndarray] | None) -> dict[int, int]:
    if not micro_payload:
        return {}
    session_id = np.asarray(micro_payload.get("session_id", np.zeros(0, dtype=np.int64)), dtype=np.int64)
    overnight = np.asarray(
        micro_payload.get("overnight_winner_flag", np.zeros(session_id.shape[0], dtype=np.int8)),
        dtype=np.int8,
    )
    n = min(session_id.size, overnight.size)
    out: dict[int, int] = {}
    for i in range(n):
        s = int(session_id[i])
        if int(overnight[i]) > 0:
            out[s] = 1
        else:
            out.setdefault(s, 0)
    return out


def _select_canonical_row(
    baseline_rows: list[dict[str, Any]],
    aggregate_ret: np.ndarray,
    common_sessions: np.ndarray,
    initial_cash: float,
) -> dict[str, Any]:
    if not baseline_rows:
        raise RuntimeError("cannot select canonical module6 bridge instance without baseline rows")
    common = np.asarray(common_sessions, dtype=np.int64)
    target = np.asarray(aggregate_ret, dtype=np.float64)
    if common.size != target.size:
        raise RuntimeError(
            f"common session / aggregate return size mismatch in module6 bridge; sessions={int(common.size)} target={int(target.size)}"
        )
    best_key: tuple[float, float, float, str] | None = None
    best_row: dict[str, Any] | None = None
    for row in baseline_rows:
        exec_map = _series_map(row.get("session_ids_exec"), row.get("daily_returns_exec"))
        vec = np.asarray([float(exec_map.get(int(s), 0.0)) for s in common.tolist()], dtype=np.float64)
        support = float(np.mean(np.asarray([int(int(s) in exec_map) for s in common.tolist()], dtype=np.float64)))
        turnover = _turnover_by_session(
            row.get("trade_payload"),
            {int(ts): int(sess) for ts, sess in zip(
                np.asarray(row.get("equity_payload", {}).get("ts_ns", np.zeros(0, dtype=np.int64)), dtype=np.int64).tolist(),
                np.asarray(row.get("equity_payload", {}).get("session_id", np.zeros(0, dtype=np.int64)), dtype=np.int64).tolist(),
            )},
            initial_cash,
        )
        total_turnover = float(sum(turnover.values()))
        key = (
            float(np.sum(np.abs(vec - target))),
            -support,
            total_turnover,
            str(row.get("split_id", "")),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_row = row
    if best_row is None:
        raise RuntimeError("failed to resolve canonical module6 bridge instance")
    return best_row


def build_module6_bridge_artifacts(
    *,
    report_root: Path,
    run_id: str,
    execution_mode: str,
    common_sessions: np.ndarray,
    baseline_candidate_ids: list[str],
    candidate_daily_mat: np.ndarray,
    candidates: list[Any],
    candidate_rows: list[dict[str, Any]],
    all_results: list[dict[str, Any]],
    engine_cfg: Any,
    require_pandas_fn: Callable[[], Any],
) -> tuple[dict[str, str], dict[str, Any]]:
    pdx = require_pandas_fn()
    common = np.asarray(common_sessions, dtype=np.int64).reshape(-1)
    if common.size <= 0:
        raise RuntimeError("module6 bridge requires a non-empty canonical session calendar")
    baseline_col = {str(cid): j for j, cid in enumerate(baseline_candidate_ids)}
    candidate_meta = {str(row["candidate_id"]): row for row in candidate_rows}
    calendar_version = _calendar_version(common)
    selection_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []
    initial_cash = float(getattr(engine_cfg, "initial_cash", 1_000_000.0))

    for cand in sorted(candidates, key=lambda x: str(x.candidate_id)):
        candidate_id = str(cand.candidate_id)
        meta = candidate_meta.get(candidate_id, {})
        strategy_id = candidate_id
        candidate_results = sorted(
            [r for r in all_results if str(r.get("candidate_id", "")) == candidate_id],
            key=lambda x: (str(x.get("scenario_id", "")), str(x.get("split_id", "")), str(x.get("task_id", ""))),
        )
        if not candidate_results:
            raise RuntimeError(f"module6 bridge requires at least one result row per candidate; candidate_id={candidate_id}")
        baseline_ok_rows = [
            r for r in candidate_results
            if str(r.get("status", "")) == "ok" and str(r.get("scenario_id", "")) == "baseline"
        ]
        baseline_any_rows = [
            r for r in candidate_results
            if str(r.get("scenario_id", "")) == "baseline"
        ]
        aggregate_ret = (
            np.asarray(candidate_daily_mat[:, int(baseline_col[candidate_id])], dtype=np.float64)
            if candidate_id in baseline_col
            else np.zeros(common.shape[0], dtype=np.float64)
        )
        if baseline_ok_rows:
            canonical_row = _select_canonical_row(
                baseline_rows=baseline_ok_rows,
                aggregate_ret=aggregate_ret,
                common_sessions=common,
                initial_cash=initial_cash,
            )
        elif baseline_any_rows:
            canonical_row = sorted(
                baseline_any_rows,
                key=lambda x: (str(x.get("status", "")) != "ok", str(x.get("split_id", "")), str(x.get("task_id", ""))),
            )[0]
        else:
            canonical_row = candidate_results[0]

        for row in candidate_results:
            split_id = str(row.get("split_id", ""))
            scenario_id = str(row.get("scenario_id", ""))
            status = str(row.get("status", ""))
            dq_codes = {str(x) for x in row.get("quality_reason_codes", [])}
            role = "excluded"
            if status == "ok":
                role = "robustness_only"
            if canonical_row is row:
                role = "canonical_portfolio"

            exec_map = _series_map(row.get("session_ids_exec"), row.get("daily_returns_exec"))
            raw_map = _series_map(row.get("session_ids_raw"), row.get("daily_returns_raw"))
            equity_payload = row.get("equity_payload")
            trade_payload = row.get("trade_payload")
            micro_payload = row.get("micro_payload")
            eq_ts = np.asarray(
                equity_payload.get("ts_ns", np.zeros(0, dtype=np.int64)) if isinstance(equity_payload, dict) else np.zeros(0, dtype=np.int64),
                dtype=np.int64,
            )
            eq_session = np.asarray(
                equity_payload.get("session_id", np.zeros(0, dtype=np.int64)) if isinstance(equity_payload, dict) else np.zeros(0, dtype=np.int64),
                dtype=np.int64,
            )
            ts_to_session = {int(ts): int(sess) for ts, sess in zip(eq_ts.tolist(), eq_session.tolist())}
            session_turnover = _turnover_by_session(trade_payload if isinstance(trade_payload, dict) else None, ts_to_session, initial_cash)
            session_trade_count = _trade_count_by_session(trade_payload if isinstance(trade_payload, dict) else None, ts_to_session)
            gross_mean, gross_peak, buying_power_min, daily_loss_max = _equity_stats_by_session(
                equity_payload if isinstance(equity_payload, dict) else None
            )
            overnight_flag = _overnight_by_session(micro_payload if isinstance(micro_payload, dict) else None)

            candidate_distance = float("nan")
            if candidate_id in baseline_col:
                aligned = np.asarray([float(exec_map.get(int(s), 0.0)) for s in common.tolist()], dtype=np.float64)
                candidate_distance = float(np.sum(np.abs(aligned - aggregate_ret)))
            selection_rows.append(
                {
                    "strategy_id": strategy_id,
                    "candidate_id": candidate_id,
                    "split_id": split_id,
                    "scenario_id": scenario_id,
                    "execution_mode": str(execution_mode),
                    "selection_stage": _SELECTION_STAGE,
                    "calendar_version": calendar_version,
                    "portfolio_instance_role": role,
                    "status": status,
                    "n_sessions_exec": int(len(exec_map)),
                    "n_sessions_raw": int(len(raw_map)),
                    "support_coverage_exec": float(len(exec_map) / max(common.size, 1)),
                    "support_coverage_raw": float(len(raw_map) / max(common.size, 1)),
                    "candidate_distance_to_baseline_aggregate": candidate_distance,
                    "parameter_hash": str(meta.get("parameter_hash", "")),
                    "family_id": str(meta.get("family_id", "")),
                    "hypothesis_id": str(meta.get("hypothesis_id", "")),
                    "failed": bool(status != "ok"),
                }
            )

            for session_id in common.tolist():
                s = int(session_id)
                observed_exec = s in exec_map
                observed_raw = s in raw_map
                if "DQ_REJECTED_INPUT" in dq_codes:
                    availability_state_code = AVAIL_INVALIDATED_BY_DQ
                elif not (observed_exec or observed_raw):
                    availability_state_code = AVAIL_STRUCTURALLY_MISSING
                else:
                    has_activity = (
                        float(session_turnover.get(s, 0.0)) > 0.0
                        or float(gross_peak.get(s, 0.0)) > 0.0
                        or int(session_trade_count.get(s, 0)) > 0
                    )
                    availability_state_code = AVAIL_OBSERVED_ACTIVE if has_activity else AVAIL_OBSERVED_FLAT
                session_rows.append(
                    {
                        "strategy_id": strategy_id,
                        "candidate_id": candidate_id,
                        "split_id": split_id,
                        "scenario_id": scenario_id,
                        "execution_mode": str(execution_mode),
                        "selection_stage": _SELECTION_STAGE,
                        "calendar_version": calendar_version,
                        "session_id": s,
                        "return_exec": float(exec_map.get(s, 0.0)),
                        "return_raw": float(raw_map.get(s, 0.0)),
                        "availability_state_code": int(availability_state_code),
                        "observed_exec": int(observed_exec),
                        "observed_raw": int(observed_raw),
                        "session_turnover": float(session_turnover.get(s, 0.0)),
                        "session_trade_count": int(session_trade_count.get(s, 0)),
                        "gross_mult_mean": float(gross_mean.get(s, 0.0)),
                        "gross_mult_peak": float(gross_peak.get(s, 0.0)),
                        "buying_power_min": float(buying_power_min.get(s, 0.0)),
                        "daily_loss_max": float(daily_loss_max.get(s, 0.0)),
                        "overnight_flag": int(overnight_flag.get(s, 0)),
                    }
                )

    selection_df = pdx.DataFrame(selection_rows)
    session_df = pdx.DataFrame(session_rows)
    if selection_df.shape[0] <= 0:
        raise RuntimeError("module6 bridge selection artifact would be empty")
    if session_df.shape[0] <= 0:
        raise RuntimeError("module6 bridge session-return artifact would be empty")

    selection_df = selection_df.sort_values(
        ["candidate_id", "scenario_id", "split_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    session_df = session_df.sort_values(
        ["candidate_id", "scenario_id", "split_id", "session_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    canonical_counts = (
        selection_df[selection_df["portfolio_instance_role"] == "canonical_portfolio"]
        .groupby("candidate_id", dropna=False)
        .size()
        .reindex(sorted(str(c.candidate_id) for c in candidates), fill_value=0)
    )
    bad = canonical_counts[canonical_counts != 1]
    if bad.shape[0] > 0:
        raise RuntimeError(
            "module6 bridge canonical selection cardinality failure; "
            + ",".join(f"{str(idx)}={int(val)}" for idx, val in bad.items())
        )

    selection_path = report_root / "strategy_instance_selection.parquet"
    session_path = report_root / "strategy_instance_session_returns.parquet"
    selection_df.to_parquet(selection_path, index=False)
    session_df.to_parquet(session_path, index=False)

    summary = {
        "selection_path": str(selection_path),
        "session_returns_path": str(session_path),
        "calendar_version": calendar_version,
        "selection_stage": _SELECTION_STAGE,
        "n_strategy_instances": int(selection_df.shape[0]),
        "n_session_rows": int(session_df.shape[0]),
        "n_canonical_instances": int((selection_df["portfolio_instance_role"] == "canonical_portfolio").sum()),
    }
    return {
        "strategy_instance_selection": str(selection_path),
        "strategy_instance_session_returns": str(session_path),
    }, summary
