from __future__ import annotations

import inspect
from pathlib import Path

import weightiz.shared.logging.system_logger as system_logger
from weightiz.shared.logging.system_logger import get_logger, init_runtime_logger, log_event, shutdown_runtime_logger


def test_runtime_logger_queue_writes_files(tmp_path: Path) -> None:
    ctx = init_runtime_logger(run_id="run_test", run_dir=tmp_path, level="INFO")
    logger = get_logger("test", run_id="run_test")
    log_event(logger, "INFO", "hello", event_type="unit_test", strategy_id="s1")
    shutdown_runtime_logger()

    runtime_log = tmp_path / "runtime.log"
    events = tmp_path / "system_events.jsonl"
    assert runtime_log.exists()
    assert events.exists()
    assert "hello" in runtime_log.read_text(encoding="utf-8")
    assert "unit_test" in events.read_text(encoding="utf-8")


def test_logging_module_uses_queue_handler_listener() -> None:
    src = inspect.getsource(system_logger)
    assert "QueueHandler" in src
    assert "QueueListener" in src
