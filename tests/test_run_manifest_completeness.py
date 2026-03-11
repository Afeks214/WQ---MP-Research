from __future__ import annotations

import inspect

import weightiz.module5.orchestrator as h


def test_manifest_required_fields_present_in_harness_source():
    src = inspect.getsource(h)
    required = [
        "git_commit",
        "config_hash",
        "dataset_hash",
        "search_seed",
        "asset_count",
        "strategy_count",
        "runtime_seconds",
        "start_time",
        "end_time",
    ]
    for k in required:
        assert f'"{k}"' in src
