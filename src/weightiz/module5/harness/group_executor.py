from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GroupExecutionKey:
    split_idx: int
    scenario_idx: int
    m2_idx: int
    m3_idx: int


@dataclass(frozen=True)
class GroupExecutionTask:
    group_id: str
    split_idx: int
    scenario_idx: int
    m2_idx: int
    m3_idx: int
    candidate_indices: tuple[int, ...]
    estimated_group_cost: int = 0
    chunk_candidate_cap: int = 0
    chunk_start: int = 0
    chunk_end: int = 0
    first_candidate_id: str = ""
    module3_group_bytes_estimated: int = 0

    @property
    def key(self) -> GroupExecutionKey:
        return GroupExecutionKey(
            split_idx=int(self.split_idx),
            scenario_idx=int(self.scenario_idx),
            m2_idx=int(self.m2_idx),
            m3_idx=int(self.m3_idx),
        )


@dataclass(frozen=True)
class GroupExecutionRuntimeStats:
    split_stress_sec: float = 0.0
    module2_sec: float = 0.0
    module3_sec: float = 0.0
    candidate_loop_sec: float = 0.0
    candidate_count: int = 0
    market_overlay_bytes: int = 0
    feature_overlay_bytes: int = 0
    module3_group_bytes_estimated: int = 0
    module3_group_bytes_realized: int = 0
    module3_bytes: int = 0
    candidate_scratch_bytes: int = 0
    result_payload_bytes: int = 0


@dataclass(frozen=True)
class GroupExecutionResult:
    group_id: str
    key: GroupExecutionKey
    candidate_rows: tuple[dict[str, Any], ...]
    runtime_stats: GroupExecutionRuntimeStats


def _base_group_id(*, split_idx: int, scenario_idx: int, m2_idx: int, m3_idx: int) -> str:
    return f"g_s{int(split_idx):04d}_x{int(scenario_idx):04d}_m2_{int(m2_idx):03d}_m3_{int(m3_idx):03d}"


def _sorted_candidate_pairs(candidates: list[Any], indices: list[int]) -> list[tuple[str, int]]:
    return sorted(
        ((str(candidates[int(idx)].candidate_id), int(idx)) for idx in indices),
        key=lambda item: (item[0], item[1]),
    )


def build_base_group_execution_tasks(
    candidates: list[Any],
    splits: list[Any],
    scenarios: list[Any],
) -> list[GroupExecutionTask]:
    groups: dict[tuple[int, int, int, int], list[int]] = {}
    for ci, c in enumerate(candidates):
        for si, _sp in enumerate(splits):
            for xi, _sc in enumerate(scenarios):
                key = (int(si), int(xi), int(c.m2_idx), int(c.m3_idx))
                groups.setdefault(key, []).append(int(ci))
    out: list[GroupExecutionTask] = []
    for (si, xi, i2, i3), cand_idx in sorted(groups.items(), key=lambda kv: kv[0]):
        ordered_pairs = _sorted_candidate_pairs(candidates, cand_idx)
        ordered_idx = tuple(idx for _, idx in ordered_pairs)
        first_candidate_id = ordered_pairs[0][0] if ordered_pairs else ""
        out.append(
            GroupExecutionTask(
                group_id=_base_group_id(split_idx=si, scenario_idx=xi, m2_idx=i2, m3_idx=i3),
                split_idx=si,
                scenario_idx=xi,
                m2_idx=i2,
                m3_idx=i3,
                candidate_indices=ordered_idx,
                estimated_group_cost=len(ordered_idx),
                chunk_candidate_cap=len(ordered_idx),
                chunk_start=0,
                chunk_end=len(ordered_idx),
                first_candidate_id=first_candidate_id,
                module3_group_bytes_estimated=0,
            )
        )
    return out


def _resolve_chunk_cap(
    *,
    candidate_count: int,
    target_group_wall_time_sec: float,
    max_result_payload_bytes: int,
    max_group_memory_bytes: int,
    min_candidates_per_chunk: int,
    max_candidates_per_chunk_hard: int,
    startup_default_candidate_loop_sec: float,
    startup_default_result_payload_bytes: int,
    startup_default_candidate_incremental_bytes: int,
    group_fixed_bytes: int,
    candidate_loop_sec_per_candidate_p95: float | None,
    result_payload_bytes_per_candidate_p95: int | None,
    candidate_incremental_bytes_p95: int | None,
) -> int:
    loop_sec = float(candidate_loop_sec_per_candidate_p95 or startup_default_candidate_loop_sec)
    payload_bytes = int(result_payload_bytes_per_candidate_p95 or startup_default_result_payload_bytes)
    incremental_bytes = int(candidate_incremental_bytes_p95 or startup_default_candidate_incremental_bytes)
    max_candidates_hard = int(max(1, max_candidates_per_chunk_hard))
    min_candidates = int(max(1, min_candidates_per_chunk))

    chunk_cap_by_time = max(1, int(float(target_group_wall_time_sec) // max(loop_sec, 1e-9)))
    chunk_cap_by_payload = max(1, int(int(max_result_payload_bytes) // max(payload_bytes, 1)))
    remaining_group_memory = max(1, int(max_group_memory_bytes) - int(group_fixed_bytes))
    chunk_cap_by_memory = max(1, int(remaining_group_memory // max(incremental_bytes, 1)))

    resolved = min(
        int(candidate_count),
        max_candidates_hard,
        chunk_cap_by_time,
        chunk_cap_by_payload,
        chunk_cap_by_memory,
    )
    return int(max(min_candidates, resolved))


def chunk_group_execution_task(
    base_group: GroupExecutionTask,
    candidates: list[Any],
    *,
    module3_group_bytes_estimated: int,
    target_group_wall_time_sec: float,
    max_result_payload_bytes: int,
    max_group_memory_bytes: int,
    min_candidates_per_chunk: int,
    max_candidates_per_chunk_hard: int,
    startup_default_candidate_loop_sec: float,
    startup_default_result_payload_bytes: int,
    startup_default_candidate_incremental_bytes: int,
    group_fixed_bytes: int,
    candidate_loop_sec_per_candidate_p95: float | None = None,
    result_payload_bytes_per_candidate_p95: int | None = None,
    candidate_incremental_bytes_p95: int | None = None,
) -> list[GroupExecutionTask]:
    candidate_count = int(len(base_group.candidate_indices))
    chunk_cap = _resolve_chunk_cap(
        candidate_count=candidate_count,
        target_group_wall_time_sec=target_group_wall_time_sec,
        max_result_payload_bytes=max_result_payload_bytes,
        max_group_memory_bytes=max_group_memory_bytes,
        min_candidates_per_chunk=min_candidates_per_chunk,
        max_candidates_per_chunk_hard=max_candidates_per_chunk_hard,
        startup_default_candidate_loop_sec=startup_default_candidate_loop_sec,
        startup_default_result_payload_bytes=startup_default_result_payload_bytes,
        startup_default_candidate_incremental_bytes=startup_default_candidate_incremental_bytes,
        group_fixed_bytes=group_fixed_bytes,
        candidate_loop_sec_per_candidate_p95=candidate_loop_sec_per_candidate_p95,
        result_payload_bytes_per_candidate_p95=result_payload_bytes_per_candidate_p95,
        candidate_incremental_bytes_p95=candidate_incremental_bytes_p95,
    )
    if candidate_count <= chunk_cap:
        return [
            GroupExecutionTask(
                group_id=str(base_group.group_id),
                split_idx=int(base_group.split_idx),
                scenario_idx=int(base_group.scenario_idx),
                m2_idx=int(base_group.m2_idx),
                m3_idx=int(base_group.m3_idx),
                candidate_indices=tuple(int(x) for x in base_group.candidate_indices),
                estimated_group_cost=int(base_group.estimated_group_cost),
                chunk_candidate_cap=int(chunk_cap),
                chunk_start=0,
                chunk_end=int(candidate_count),
                first_candidate_id=str(base_group.first_candidate_id),
                module3_group_bytes_estimated=int(module3_group_bytes_estimated),
            )
        ]

    tasks: list[GroupExecutionTask] = []
    for offset in range(0, candidate_count, chunk_cap):
        part = tuple(base_group.candidate_indices[offset : offset + chunk_cap])
        first_candidate_id = str(candidates[int(part[0])].candidate_id) if part else ""
        tasks.append(
            GroupExecutionTask(
                group_id=f"{base_group.group_id}_p{offset // chunk_cap:03d}",
                split_idx=int(base_group.split_idx),
                scenario_idx=int(base_group.scenario_idx),
                m2_idx=int(base_group.m2_idx),
                m3_idx=int(base_group.m3_idx),
                candidate_indices=part,
                estimated_group_cost=int(base_group.estimated_group_cost),
                chunk_candidate_cap=int(chunk_cap),
                chunk_start=int(offset),
                chunk_end=int(offset + len(part)),
                first_candidate_id=first_candidate_id,
                module3_group_bytes_estimated=int(module3_group_bytes_estimated),
            )
        )
    return tasks


def order_group_execution_tasks(
    tasks: list[GroupExecutionTask],
    *,
    dispatch_policy: str,
) -> list[GroupExecutionTask]:
    out = list(tasks)
    if str(dispatch_policy).strip().lower() == "largest_first_stable":
        out.sort(
            key=lambda g: (
                -int(g.estimated_group_cost),
                int(g.split_idx),
                int(g.scenario_idx),
                int(g.m2_idx),
                int(g.m3_idx),
                str(g.first_candidate_id),
                int(g.chunk_start),
            )
        )
    else:
        out.sort(
            key=lambda g: (
                int(g.split_idx),
                int(g.scenario_idx),
                int(g.m2_idx),
                int(g.m3_idx),
                str(g.first_candidate_id),
                int(g.chunk_start),
            )
        )
    return out


def build_group_execution_tasks(
    candidates: list[Any],
    splits: list[Any],
    scenarios: list[Any],
    *,
    dispatch_policy: str,
    target_group_wall_time_sec: float,
    max_result_payload_bytes: int,
    max_group_memory_bytes: int,
    min_candidates_per_chunk: int,
    max_candidates_per_chunk_hard: int,
    startup_default_candidate_loop_sec: float,
    startup_default_result_payload_bytes: int,
    startup_default_candidate_incremental_bytes: int,
    group_fixed_bytes: int,
    module3_group_bytes_estimated: int | None = None,
    candidate_loop_sec_per_candidate_p95: float | None = None,
    result_payload_bytes_per_candidate_p95: int | None = None,
    candidate_incremental_bytes_p95: int | None = None,
) -> list[GroupExecutionTask]:
    tasks: list[GroupExecutionTask] = []
    for base_group in build_base_group_execution_tasks(candidates, splits, scenarios):
        tasks.extend(
            chunk_group_execution_task(
                base_group,
                candidates,
                module3_group_bytes_estimated=int(max(1, int(module3_group_bytes_estimated or 1))),
                target_group_wall_time_sec=target_group_wall_time_sec,
                max_result_payload_bytes=max_result_payload_bytes,
                max_group_memory_bytes=max_group_memory_bytes,
                min_candidates_per_chunk=min_candidates_per_chunk,
                max_candidates_per_chunk_hard=max_candidates_per_chunk_hard,
                startup_default_candidate_loop_sec=startup_default_candidate_loop_sec,
                startup_default_result_payload_bytes=startup_default_result_payload_bytes,
                startup_default_candidate_incremental_bytes=startup_default_candidate_incremental_bytes,
                group_fixed_bytes=group_fixed_bytes,
                candidate_loop_sec_per_candidate_p95=candidate_loop_sec_per_candidate_p95,
                result_payload_bytes_per_candidate_p95=result_payload_bytes_per_candidate_p95,
                candidate_incremental_bytes_p95=candidate_incremental_bytes_p95,
            )
        )
    return order_group_execution_tasks(tasks, dispatch_policy=dispatch_policy)
