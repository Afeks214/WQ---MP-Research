from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from weightiz_dtype_guard import assert_float64


@dataclass
class RuntimeMonitor:
    run_id: str
    run_dir: Path
    # Diagnostics/cache tensor only; this monitor does not imply worker compute authority.
    expected_tensor_shape: tuple[int, int, int, int]
    expected_worker_count: int
    health_check_interval: int = 50

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.health_path = self.run_dir / "runtime_health_checks.jsonl"
        self._last_checked = -1

    def should_check(self, strategies_completed: int) -> bool:
        n = int(max(1, self.health_check_interval))
        if strategies_completed <= 0:
            return False
        if strategies_completed == self._last_checked:
            return False
        return (strategies_completed % n) == 0

    def check_and_emit(
        self,
        *,
        strategies_completed: int,
        tensor: np.ndarray,
        worker_status: dict[str, Any],
        ledger_path: Path,
        queue_backlog: int,
        memory_status: dict[str, Any],
        extra_metrics: dict[str, Any] | None = None,
        require_ledger_exists: bool = False,
    ) -> None:
        assert_float64("runtime_monitor.tensor", np.asarray(tensor))
        t = np.asarray(tensor)
        if tuple(t.shape) != tuple(self.expected_tensor_shape):
            self._emit(
                strategies_completed=strategies_completed,
                tensor_valid=False,
                worker_status=worker_status,
                ledger_status=False,
                memory_status=memory_status,
                queue_backlog=int(queue_backlog),
                extra_metrics=extra_metrics,
            )
            raise RuntimeError("RUNTIME_HEALTH_TEST_FAILED")

        has_nan = bool(np.isnan(t).any())
        tensor_valid = (not has_nan)
        workers_active = int(worker_status.get("active", 0))
        workers_expected = int(worker_status.get("expected", self.expected_worker_count))
        worker_ok = 0 <= workers_active <= max(1, workers_expected)
        ledger_exists = bool(Path(ledger_path).exists())
        ledger_ok = ledger_exists if require_ledger_exists else True
        memory_ok = bool(memory_status.get("ok", True))

        self._emit(
            strategies_completed=strategies_completed,
            tensor_valid=bool(tensor_valid),
            worker_status={
                "active": workers_active,
                "expected": workers_expected,
                "ok": bool(worker_ok),
            },
            ledger_status={
                "path": str(Path(ledger_path)),
                "exists": bool(ledger_exists),
                "ok": bool(ledger_ok),
            },
            memory_status=memory_status,
            queue_backlog=int(queue_backlog),
            extra_metrics=extra_metrics,
        )
        self._last_checked = int(strategies_completed)

        if (not tensor_valid) or (not worker_ok) or (not ledger_ok) or (not memory_ok):
            raise RuntimeError("RUNTIME_HEALTH_TEST_FAILED")

    def _emit(
        self,
        *,
        strategies_completed: int,
        tensor_valid: bool,
        worker_status: dict[str, Any],
        ledger_status: dict[str, Any] | bool,
        memory_status: dict[str, Any],
        queue_backlog: int,
        extra_metrics: dict[str, Any] | None,
    ) -> None:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": str(self.run_id),
            "strategies_completed": int(strategies_completed),
            "diagnostic_feature_tensor_valid": bool(tensor_valid),
            "diagnostic_feature_tensor_role": "diagnostics_cache_only",
            "worker_status": worker_status,
            "ledger_status": ledger_status,
            "memory_status": memory_status,
            "queue_backlog": int(queue_backlog),
        }
        if extra_metrics:
            row.update(dict(extra_metrics))
        with self.health_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
