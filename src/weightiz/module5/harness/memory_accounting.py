from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MemoryAccountingEstimate:
    available_bytes: int
    budget_bytes: int
    base_bytes: int
    market_overlay_bytes: int
    feature_overlay_bytes: int
    module3_bytes: int
    candidate_scratch_bytes: int
    worker_overhead_bytes: int
    queue_bytes: int
    result_buffer_bytes: int
    safety_margin_bytes: int
    requested_workers: int
    effective_workers: int
    total_projected_bytes: int
    per_worker_bytes: int


@dataclass(frozen=True)
class BaseSharingDecision:
    configured_mode: str
    mode: str
    start_method: str
    cow_probe_required: bool
    fallback_only: bool
    shared_base_state_active: bool
    reason: str


def resolve_base_sharing_mode(
    *,
    configured_mode: str,
    start_method: str,
    platform_name: str,
) -> BaseSharingDecision:
    selected = str(configured_mode).strip().lower()
    if selected == "explicit_shm":
        raise RuntimeError("EXPLICIT_SHM_NOT_IMPLEMENTED")
    if selected == "fork_cow":
        return BaseSharingDecision(
            configured_mode=str(selected),
            mode="fork_cow",
            start_method=str(start_method),
            cow_probe_required=True,
            fallback_only=False,
            shared_base_state_active=True,
            reason="configured_fork_cow",
        )
    linux_fork = str(platform_name).lower().startswith("linux") and str(start_method).strip().lower() == "fork"
    if linux_fork:
        return BaseSharingDecision(
            configured_mode=str(selected),
            mode="fork_cow",
            start_method=str(start_method),
            cow_probe_required=True,
            fallback_only=False,
            shared_base_state_active=True,
            reason="auto_linux_fork",
        )
    return BaseSharingDecision(
        configured_mode=str(selected),
        mode="serialized_copy",
        start_method=str(start_method),
        cow_probe_required=False,
        fallback_only=True,
        shared_base_state_active=False,
        reason="auto_fallback_non_linux_or_non_fork",
    )


def estimate_worker_overhead_bytes(
    *,
    market_overlay_bytes: int,
    feature_overlay_bytes: int,
    module3_bytes: int,
    candidate_scratch_bytes: int,
) -> int:
    explicit = int(market_overlay_bytes) + int(feature_overlay_bytes) + int(module3_bytes) + int(candidate_scratch_bytes)
    return int(max(128 * 1024**2, explicit // 8))


def estimate_queue_bytes(*, in_flight_count: int, sampled_task_bytes: int) -> int:
    return int(max(0, in_flight_count) * max(0, sampled_task_bytes))


def estimate_result_buffer_bytes(*, pending_rows: int, sampled_row_bytes: int) -> int:
    return int(max(0, pending_rows) * max(0, sampled_row_bytes))


def build_memory_accounting_estimate(
    *,
    available_bytes: int,
    max_ram_utilization_frac: float,
    safety_margin_frac: float,
    safety_margin_min_bytes: int,
    base_bytes: int,
    market_overlay_bytes: int,
    feature_overlay_bytes: int,
    module3_bytes: int,
    candidate_scratch_bytes: int,
    queue_bytes: int,
    result_buffer_bytes: int,
    requested_workers: int,
    worker_overhead_bytes: int | None = None,
) -> MemoryAccountingEstimate:
    avail = int(max(0, available_bytes))
    budget = int(max(0.0, float(max_ram_utilization_frac)) * float(avail))
    safety_margin_bytes = int(max(int(safety_margin_min_bytes), float(safety_margin_frac) * float(avail)))
    per_worker_overhead = int(
        estimate_worker_overhead_bytes(
            market_overlay_bytes=market_overlay_bytes,
            feature_overlay_bytes=feature_overlay_bytes,
            module3_bytes=module3_bytes,
            candidate_scratch_bytes=candidate_scratch_bytes,
        )
        if worker_overhead_bytes is None
        else max(0, int(worker_overhead_bytes))
    )
    per_worker_bytes = int(
        max(1, int(market_overlay_bytes) + int(feature_overlay_bytes) + int(module3_bytes) + int(candidate_scratch_bytes) + per_worker_overhead)
    )
    usable_budget = int(
        max(
            0,
            budget
            - int(base_bytes)
            - int(queue_bytes)
            - int(result_buffer_bytes)
            - int(safety_margin_bytes),
        )
    )
    effective_workers = int(max(1, usable_budget // max(per_worker_bytes, 1)))
    effective_workers = int(min(max(1, int(requested_workers)), effective_workers))
    total_projected_bytes = int(
        int(base_bytes)
        + int(effective_workers) * int(per_worker_bytes)
        + int(queue_bytes)
        + int(result_buffer_bytes)
        + int(safety_margin_bytes)
    )
    return MemoryAccountingEstimate(
        available_bytes=avail,
        budget_bytes=budget,
        base_bytes=int(base_bytes),
        market_overlay_bytes=int(market_overlay_bytes),
        feature_overlay_bytes=int(feature_overlay_bytes),
        module3_bytes=int(module3_bytes),
        candidate_scratch_bytes=int(candidate_scratch_bytes),
        worker_overhead_bytes=int(per_worker_overhead),
        queue_bytes=int(queue_bytes),
        result_buffer_bytes=int(result_buffer_bytes),
        safety_margin_bytes=int(safety_margin_bytes),
        requested_workers=int(max(1, requested_workers)),
        effective_workers=int(max(1, effective_workers)),
        total_projected_bytes=int(total_projected_bytes),
        per_worker_bytes=int(per_worker_bytes),
    )


def chunk_policy_memory_cap(
    *,
    budget_bytes: int,
    base_bytes: int,
    requested_workers: int,
    queue_bytes: int,
    result_buffer_bytes: int,
    safety_margin_bytes: int,
) -> int:
    usable = int(
        max(
            1,
            int(budget_bytes)
            - int(base_bytes)
            - int(queue_bytes)
            - int(result_buffer_bytes)
            - int(safety_margin_bytes),
        )
    )
    return int(max(1, usable // max(1, int(requested_workers))))
