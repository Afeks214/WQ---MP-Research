from __future__ import annotations

from typing import Any, Callable


class _WorkerIOGuardInstaller:
    def __call__(self) -> None:
        try:
            import pandas as pd
        except Exception:
            return

        marker = "_weightiz_worker_io_guard_installed"
        if bool(getattr(pd, marker, False)):
            return

        def _guarded(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
            def _wrapped(*args: Any, **kwargs: Any) -> Any:
                raise RuntimeError(f"WORKER_IO_VIOLATION: {name} is forbidden in worker context")

            setattr(_wrapped, "__wrapped__", fn)
            return _wrapped

        for attr in ("read_parquet", "read_csv", "read_feather", "read_pickle"):
            original = getattr(pd, attr, None)
            if callable(original):
                setattr(pd, attr, _guarded(attr, original))

        setattr(pd, marker, True)


_install_worker_io_guard = _WorkerIOGuardInstaller()


def run_zimtra_sweep(*_args, **_kwargs):
    raise RuntimeError("PARALLEL_ENGINE_FORBIDDEN: use canonical Module5 pipeline")
