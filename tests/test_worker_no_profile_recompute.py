from __future__ import annotations

import pytest

import weightiz_module5_harness as h


def test_worker_feature_source_must_be_shared_memory():
    h._WORKER_CONTEXT = {"worker_feature_source": "disk"}
    with pytest.raises(RuntimeError, match="worker_feature_source"):
        h._run_group_task_from_context(object())
