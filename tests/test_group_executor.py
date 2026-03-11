from __future__ import annotations

from dataclasses import dataclass

from module5.harness.group_executor import GroupExecutionTask, build_group_execution_tasks


@dataclass(frozen=True)
class _Candidate:
    candidate_id: str
    m2_idx: int
    m3_idx: int


@dataclass(frozen=True)
class _Split:
    split_id: str


@dataclass(frozen=True)
class _Scenario:
    scenario_id: str


def test_build_group_execution_tasks_keeps_group_intact_when_under_caps() -> None:
    tasks = build_group_execution_tasks(
        candidates=[_Candidate("c0", 0, 0), _Candidate("c1", 0, 0)],
        splits=[_Split("s0")],
        scenarios=[_Scenario("x0")],
        dispatch_policy="largest_first_stable",
        target_group_wall_time_sec=60.0,
        max_result_payload_bytes=1_000_000,
        max_group_memory_bytes=1_000_000,
        min_candidates_per_chunk=1,
        max_candidates_per_chunk_hard=32,
        startup_default_candidate_loop_sec=1.0,
        startup_default_result_payload_bytes=128,
        startup_default_candidate_incremental_bytes=128,
        group_fixed_bytes=1024,
    )

    assert tasks == [
        GroupExecutionTask(
            group_id="g_s0000_x0000_m2_000_m3_000",
            split_idx=0,
            scenario_idx=0,
            m2_idx=0,
            m3_idx=0,
            candidate_indices=(0, 1),
            estimated_group_cost=2,
            chunk_candidate_cap=2,
            chunk_start=0,
            chunk_end=2,
            first_candidate_id="c0",
            module3_group_bytes_estimated=1,
        )
    ]


def test_build_group_execution_tasks_splits_deterministically_by_memory_cap() -> None:
    tasks = build_group_execution_tasks(
        candidates=[
            _Candidate("c0", 0, 0),
            _Candidate("c1", 0, 0),
            _Candidate("c2", 0, 0),
            _Candidate("c3", 0, 0),
            _Candidate("c4", 0, 0),
        ],
        splits=[_Split("s0")],
        scenarios=[_Scenario("x0")],
        dispatch_policy="largest_first_stable",
        target_group_wall_time_sec=60.0,
        max_result_payload_bytes=1_000_000,
        max_group_memory_bytes=1_600,
        min_candidates_per_chunk=1,
        max_candidates_per_chunk_hard=32,
        startup_default_candidate_loop_sec=1.0,
        startup_default_result_payload_bytes=128,
        startup_default_candidate_incremental_bytes=256,
        group_fixed_bytes=1088,
    )

    assert [t.group_id for t in tasks] == [
        "g_s0000_x0000_m2_000_m3_000_p000",
        "g_s0000_x0000_m2_000_m3_000_p001",
        "g_s0000_x0000_m2_000_m3_000_p002",
    ]
    assert [t.candidate_indices for t in tasks] == [(0, 1), (2, 3), (4,)]
    assert [t.chunk_start for t in tasks] == [0, 2, 4]
    assert [t.chunk_end for t in tasks] == [2, 4, 5]


def test_build_group_execution_tasks_dispatches_largest_first_stable() -> None:
    tasks = build_group_execution_tasks(
        candidates=[
            _Candidate("c0", 0, 0),
            _Candidate("c1", 0, 0),
            _Candidate("c2", 1, 1),
        ],
        splits=[_Split("s0"), _Split("s1")],
        scenarios=[_Scenario("x0")],
        dispatch_policy="largest_first_stable",
        target_group_wall_time_sec=60.0,
        max_result_payload_bytes=1_000_000,
        max_group_memory_bytes=1_000_000,
        min_candidates_per_chunk=1,
        max_candidates_per_chunk_hard=32,
        startup_default_candidate_loop_sec=1.0,
        startup_default_result_payload_bytes=128,
        startup_default_candidate_incremental_bytes=128,
        group_fixed_bytes=1024,
    )

    assert [t.estimated_group_cost for t in tasks] == [2, 2, 1, 1]
    assert [t.group_id for t in tasks] == [
        "g_s0000_x0000_m2_000_m3_000",
        "g_s0001_x0000_m2_000_m3_000",
        "g_s0000_x0000_m2_001_m3_001",
        "g_s0001_x0000_m2_001_m3_001",
    ]


def test_build_group_execution_tasks_orders_candidates_by_candidate_id_not_input_position() -> None:
    tasks = build_group_execution_tasks(
        candidates=[
            _Candidate("c9", 0, 0),
            _Candidate("c1", 0, 0),
            _Candidate("c5", 0, 0),
        ],
        splits=[_Split("s0")],
        scenarios=[_Scenario("x0")],
        dispatch_policy="largest_first_stable",
        target_group_wall_time_sec=60.0,
        max_result_payload_bytes=1_000_000,
        max_group_memory_bytes=1_000_000,
        min_candidates_per_chunk=1,
        max_candidates_per_chunk_hard=32,
        startup_default_candidate_loop_sec=1.0,
        startup_default_result_payload_bytes=128,
        startup_default_candidate_incremental_bytes=128,
        group_fixed_bytes=1024,
    )

    assert len(tasks) == 1
    assert tasks[0].candidate_indices == (1, 2, 0)
    assert tasks[0].first_candidate_id == "c1"
