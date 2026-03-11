from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import atexit
import json
import logging
from logging.handlers import QueueHandler, QueueListener
import multiprocessing as mp
from pathlib import Path
from typing import Any


_RUNTIME_CONTEXT: "RuntimeLoggerContext | None" = None


class _JsonlEventHandler(logging.Handler):
    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - simple I/O
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": str(record.levelname),
            "message": str(record.getMessage()),
            "run_id": str(getattr(record, "run_id", "")),
            "module": str(getattr(record, "module_name", "")),
            "worker_id": str(getattr(record, "worker_id", "")),
            "strategy_id": str(getattr(record, "strategy_id", "")),
            "event_type": str(getattr(record, "event_type", "log")),
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            return


@dataclass
class RuntimeLoggerContext:
    run_id: str
    queue: mp.Queue
    listener: QueueListener
    runtime_log_path: Path
    events_jsonl_path: Path


def _level_from_name(level: str) -> int:
    s = str(level).strip().upper()
    return {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }.get(s, logging.INFO)


def init_runtime_logger(
    *,
    run_id: str,
    run_dir: Path,
    level: str = "INFO",
) -> RuntimeLoggerContext:
    global _RUNTIME_CONTEXT
    if _RUNTIME_CONTEXT is not None and str(_RUNTIME_CONTEXT.run_id) == str(run_id):
        return _RUNTIME_CONTEXT
    if _RUNTIME_CONTEXT is not None:
        shutdown_runtime_logger()

    log_level = _level_from_name(level)
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime_log_path = run_dir / "runtime.log"
    events_jsonl_path = run_dir / "system_events.jsonl"

    queue: mp.Queue = mp.Queue()
    file_handler = logging.FileHandler(runtime_log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s run_id=%(run_id)s module=%(module_name)s worker=%(worker_id)s strategy=%(strategy_id)s event=%(event_type)s msg=%(message)s")
    )
    jsonl_handler = _JsonlEventHandler(events_jsonl_path)
    jsonl_handler.setLevel(log_level)

    listener = QueueListener(queue, file_handler, jsonl_handler, respect_handler_level=True)
    listener.start()

    root = logging.getLogger("weightiz")
    root.setLevel(log_level)
    root.handlers = []
    root.propagate = False
    root.addHandler(QueueHandler(queue))

    ctx = RuntimeLoggerContext(
        run_id=str(run_id),
        queue=queue,
        listener=listener,
        runtime_log_path=runtime_log_path,
        events_jsonl_path=events_jsonl_path,
    )
    _RUNTIME_CONTEXT = ctx
    atexit.register(shutdown_runtime_logger)
    return ctx


def configure_worker_logging(queue: mp.Queue, level: str = "INFO") -> None:
    log_level = _level_from_name(level)
    root = logging.getLogger("weightiz")
    root.setLevel(log_level)
    root.handlers = []
    root.propagate = False
    root.addHandler(QueueHandler(queue))


def get_logger(
    module: str,
    *,
    run_id: str = "",
    worker_id: str = "",
    strategy_id: str = "",
) -> logging.LoggerAdapter:
    base = logging.getLogger("weightiz")
    return logging.LoggerAdapter(
        base,
        {
            "run_id": str(run_id),
            "module_name": str(module),
            "worker_id": str(worker_id),
            "strategy_id": str(strategy_id),
        },
    )


def log_event(
    logger: logging.LoggerAdapter,
    level: str,
    message: str,
    *,
    event_type: str,
    strategy_id: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    payload = dict(logger.extra)
    payload["event_type"] = str(event_type)
    if str(strategy_id):
        payload["strategy_id"] = str(strategy_id)
    if extra:
        payload.update({str(k): v for k, v in extra.items()})
    lvl = _level_from_name(level)
    logger.logger.log(lvl, str(message), extra=payload)


def shutdown_runtime_logger() -> None:
    global _RUNTIME_CONTEXT
    if _RUNTIME_CONTEXT is None:
        return
    try:
        _RUNTIME_CONTEXT.listener.stop()
    except Exception:
        pass
    try:
        _RUNTIME_CONTEXT.queue.close()
    except Exception:
        pass
    _RUNTIME_CONTEXT = None
