from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from weightiz_runtime_monitor import RuntimeMonitor


def test_runtime_monitor_emits_health_rows(tmp_path: Path) -> None:
    mon = RuntimeMonitor(
        run_id="run_test",
        run_dir=tmp_path,
        expected_tensor_shape=(2, 3, 4, 1),
        expected_worker_count=2,
        health_check_interval=1,
    )
    tensor = np.zeros((2, 3, 4, 1), dtype=np.float64)
    mon.check_and_emit(
        strategies_completed=1,
        tensor=tensor,
        worker_status={"active": 1},
        ledger_path=tmp_path / "strategy_results.parquet",
        queue_backlog=0,
        memory_status={"ok": True},
        require_ledger_exists=False,
    )
    p = tmp_path / "runtime_health_checks.jsonl"
    assert p.exists()
    txt = p.read_text(encoding="utf-8")
    assert "tensor_valid" in txt


def test_runtime_monitor_raises_on_shape_mismatch(tmp_path: Path) -> None:
    mon = RuntimeMonitor(
        run_id="run_test",
        run_dir=tmp_path,
        expected_tensor_shape=(2, 3, 4, 1),
        expected_worker_count=1,
        health_check_interval=1,
    )
    bad_tensor = np.zeros((2, 3, 5, 1), dtype=np.float64)
    with pytest.raises(RuntimeError, match="RUNTIME_HEALTH_TEST_FAILED"):
        mon.check_and_emit(
            strategies_completed=1,
            tensor=bad_tensor,
            worker_status={"active": 1},
            ledger_path=tmp_path / "strategy_results.parquet",
            queue_backlog=0,
            memory_status={"ok": True},
            require_ledger_exists=False,
        )
