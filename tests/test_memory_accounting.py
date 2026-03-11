from __future__ import annotations

import pytest

from weightiz.module5.harness.memory_accounting import (
    build_memory_accounting_estimate,
    chunk_policy_memory_cap,
    resolve_base_sharing_mode,
)


def test_build_memory_accounting_estimate_caps_effective_workers() -> None:
    estimate = build_memory_accounting_estimate(
        available_bytes=10_000,
        max_ram_utilization_frac=0.8,
        safety_margin_frac=0.1,
        safety_margin_min_bytes=500,
        base_bytes=1_000,
        market_overlay_bytes=1_000,
        feature_overlay_bytes=1_000,
        module3_bytes=500,
        candidate_scratch_bytes=500,
        queue_bytes=200,
        result_buffer_bytes=300,
        requested_workers=8,
        worker_overhead_bytes=0,
    )

    assert estimate.budget_bytes == 8_000
    assert estimate.safety_margin_bytes == 1_000
    assert estimate.per_worker_bytes == 3_000
    assert estimate.effective_workers == 1
    assert estimate.total_projected_bytes == 5_500


def test_chunk_policy_memory_cap_uses_requested_workers() -> None:
    assert chunk_policy_memory_cap(
        budget_bytes=10_000,
        base_bytes=2_000,
        requested_workers=4,
        queue_bytes=1_000,
        result_buffer_bytes=1_000,
        safety_margin_bytes=1_000,
    ) == 1_250


def test_resolve_base_sharing_mode_prefers_fork_cow_only_on_linux_fork() -> None:
    linux_fork = resolve_base_sharing_mode(
        configured_mode="auto",
        start_method="fork",
        platform_name="linux",
    )
    assert linux_fork.configured_mode == "auto"
    assert linux_fork.mode == "fork_cow"
    assert linux_fork.cow_probe_required is True
    assert linux_fork.fallback_only is False
    assert linux_fork.shared_base_state_active is True
    assert linux_fork.reason == "auto_linux_fork"

    mac_spawn = resolve_base_sharing_mode(
        configured_mode="auto",
        start_method="spawn",
        platform_name="darwin",
    )
    assert mac_spawn.configured_mode == "auto"
    assert mac_spawn.mode == "serialized_copy"
    assert mac_spawn.cow_probe_required is False
    assert mac_spawn.fallback_only is True
    assert mac_spawn.shared_base_state_active is False
    assert mac_spawn.reason == "auto_fallback_non_linux_or_non_fork"


def test_resolve_base_sharing_mode_explicit_shm_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="EXPLICIT_SHM_NOT_IMPLEMENTED"):
        resolve_base_sharing_mode(
            configured_mode="explicit_shm",
            start_method="spawn",
            platform_name="linux",
        )
