from __future__ import annotations

from pathlib import Path


def test_manifest_required_fields_present_in_harness_source():
    src = Path("weightiz_module5_harness.py").read_text(encoding="utf-8")
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
