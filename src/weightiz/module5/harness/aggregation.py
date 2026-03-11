from __future__ import annotations

from typing import Any

import numpy as np


def _assert_finite(name: str, arr: np.ndarray) -> None:
    x = np.asarray(arr)
    if not np.all(np.isfinite(x)):
        bad = np.argwhere(~np.isfinite(x))[0].tolist()
        raise RuntimeError(f"{name} contains non-finite values; first_bad_index={bad}")


def aggregate_candidate_baseline_matrices(
    results_ok: list[dict[str, Any]],
    bench_sessions: np.ndarray,
    bench_ret: np.ndarray,
    candidate_ids: list[str],
    min_days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, dict[str, dict[int, float]]]]:
    if not results_ok:
        raise RuntimeError("No successful candidate task results to assemble")

    bench_map = {int(s): float(r) for s, r in zip(bench_sessions.tolist(), bench_ret.tolist())}
    if not bench_map:
        raise RuntimeError("Benchmark session series is empty")

    cand_set = set(str(c) for c in candidate_ids)
    bucket_exec: dict[str, dict[str, dict[int, list[float]]]] = {
        str(cid): {} for cid in sorted(cand_set)
    }
    bucket_raw: dict[str, dict[str, dict[int, list[float]]]] = {
        str(cid): {} for cid in sorted(cand_set)
    }

    for r in results_ok:
        cid = str(r.get("candidate_id", ""))
        if cid not in bucket_exec:
            continue
        sid = str(r.get("scenario_id", "baseline"))
        sess_exec = np.asarray(
            r.get("session_ids_exec", r.get("session_ids", np.zeros(0, dtype=np.int64))),
            dtype=np.int64,
        )
        ret_exec = np.asarray(
            r.get("daily_returns_exec", r.get("daily_returns", np.zeros(0, dtype=np.float64))),
            dtype=np.float64,
        )
        sess_raw = np.asarray(
            r.get("session_ids_raw", sess_exec),
            dtype=np.int64,
        )
        ret_raw = np.asarray(
            r.get("daily_returns_raw", ret_exec),
            dtype=np.float64,
        )

        if sess_exec.size > 0 and ret_exec.size > 0 and sess_exec.size == ret_exec.size:
            by_sess_exec = bucket_exec[cid].setdefault(sid, {})
            for s, v in zip(sess_exec.tolist(), ret_exec.tolist()):
                vv = float(v)
                if not np.isfinite(vv):
                    continue
                by_sess_exec.setdefault(int(s), []).append(vv)
        if sess_raw.size > 0 and ret_raw.size > 0 and sess_raw.size == ret_raw.size:
            by_sess_raw = bucket_raw[cid].setdefault(sid, {})
            for s, v in zip(sess_raw.tolist(), ret_raw.tolist()):
                vv = float(v)
                if not np.isfinite(vv):
                    continue
                by_sess_raw.setdefault(int(s), []).append(vv)

    scenario_series_exec: dict[str, dict[str, dict[int, float]]] = {}
    scenario_series_raw: dict[str, dict[str, dict[int, float]]] = {}
    for cid in sorted(bucket_exec.keys()):
        per_scenario: dict[str, dict[int, float]] = {}
        for sid in sorted(bucket_exec[cid].keys()):
            sess_map = bucket_exec[cid][sid]
            if not sess_map:
                continue
            agg: dict[int, float] = {}
            for s in sorted(sess_map.keys()):
                vals = np.asarray(sess_map[s], dtype=np.float64)
                agg[int(s)] = float(np.median(vals))
            per_scenario[sid] = agg
        scenario_series_exec[cid] = per_scenario
    for cid in sorted(bucket_raw.keys()):
        per_scenario: dict[str, dict[int, float]] = {}
        for sid in sorted(bucket_raw[cid].keys()):
            sess_map = bucket_raw[cid][sid]
            if not sess_map:
                continue
            agg: dict[int, float] = {}
            for s in sorted(sess_map.keys()):
                vals = np.asarray(sess_map[s], dtype=np.float64)
                agg[int(s)] = float(np.median(vals))
            per_scenario[sid] = agg
        scenario_series_raw[cid] = per_scenario

    baseline_candidate_ids = [
        cid
        for cid in sorted(candidate_ids)
        if "baseline" in scenario_series_exec.get(str(cid), {})
        and len(scenario_series_exec[str(cid)]["baseline"]) > 0
    ]
    if not baseline_candidate_ids:
        raise RuntimeError("No candidates have baseline daily returns for candidate-level stats")

    common_sorted = np.asarray(sorted(bench_map.keys()), dtype=np.int64)
    if common_sorted.size <= 0:
        raise RuntimeError("No benchmark sessions available for candidate alignment")
    d_count = int(common_sorted.size)
    c_count = int(len(baseline_candidate_ids))
    if d_count < int(min_days):
        raise RuntimeError(f"Insufficient daily sample after candidate alignment: D={d_count}, required>={int(min_days)}")

    mat_exec = np.empty((d_count, c_count), dtype=np.float64)
    mat_raw = np.empty((d_count, c_count), dtype=np.float64)
    for j, cid in enumerate(baseline_candidate_ids):
        mp_exec = scenario_series_exec.get(cid, {}).get("baseline", {})
        mp_raw = scenario_series_raw.get(cid, {}).get("baseline", {})
        mat_exec[:, j] = np.asarray(
            [float(mp_exec.get(int(s), 0.0)) for s in common_sorted.tolist()],
            dtype=np.float64,
        )
        mat_raw[:, j] = np.asarray(
            [float(mp_raw.get(int(s), 0.0)) for s in common_sorted.tolist()],
            dtype=np.float64,
        )
    bmk = np.asarray([bench_map[int(s)] for s in common_sorted.tolist()], dtype=np.float64)

    _assert_finite("candidate_daily_returns_matrix_exec", mat_exec)
    _assert_finite("candidate_daily_returns_matrix_raw", mat_raw)
    _assert_finite("daily_benchmark_returns", bmk)
    return common_sorted, mat_exec, mat_raw, bmk, baseline_candidate_ids, scenario_series_exec


def aggregate_candidate_baseline_matrix(
    results_ok: list[dict[str, Any]],
    bench_sessions: np.ndarray,
    bench_ret: np.ndarray,
    candidate_ids: list[str],
    min_days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, dict[str, dict[int, float]]]]:
    common, mat_exec, _mat_raw, bmk, baseline_ids, series_exec = aggregate_candidate_baseline_matrices(
        results_ok=results_ok,
        bench_sessions=bench_sessions,
        bench_ret=bench_ret,
        candidate_ids=candidate_ids,
        min_days=min_days,
    )
    return common, mat_exec, bmk, baseline_ids, series_exec
