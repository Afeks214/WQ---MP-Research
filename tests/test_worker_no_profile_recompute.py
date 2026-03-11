from __future__ import annotations

import pytest

import weightiz.module5.orchestrator as h


def test_worker_context_must_include_required_runtime_keys():
    h._WORKER_CONTEXT = {}
    with pytest.raises(RuntimeError, match="base_state"):
        h._run_group_task_from_context(object())
