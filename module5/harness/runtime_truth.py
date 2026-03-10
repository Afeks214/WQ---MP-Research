from __future__ import annotations


def build_feature_tensor_role(shared_memory_published: bool) -> dict[str, object]:
    return {
        "role": "diagnostics_cache_only",
        "shared_memory_published": bool(shared_memory_published),
        "used_in_worker_compute": False,
    }


def build_compute_authority() -> dict[str, str]:
    return {
        "candidate_execution_authority": "stressed_tensor_state",
        "module2_authority": "recomputed_on_stressed_state",
        "module3_authority": "stressed_state_post_module2",
        "module4_authority": "stressed_state_plus_module3_output",
        "risk_engine_signal_authority": "module4_signal_output.target_qty_ta",
    }


def build_execution_topology(
    execution_mode: str,
    use_process_pool: bool,
    process_pool_group_reuse_active: bool = False,
) -> dict[str, object]:
    return {
        "mode": str(execution_mode),
        "process_pool_candidate_split": bool(use_process_pool),
        "grouped_post_m2_reuse_active": bool((not use_process_pool) or process_pool_group_reuse_active),
        "grouped_post_m3_reuse_active": bool((not use_process_pool) or process_pool_group_reuse_active),
    }
