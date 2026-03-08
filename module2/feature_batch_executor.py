from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, TypeVar
import numpy as np

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class BatchExecutorConfig:
    backend: str = "serial"  # serial | process_pool
    max_workers: int | None = None
    seed: int = 17


def deterministic_asset_blocks(n_assets: int, n_blocks: int) -> list[np.ndarray]:
    if n_assets <= 0:
        return []
    n_blocks = max(1, int(n_blocks))
    edges = np.linspace(0, n_assets, n_blocks + 1, dtype=np.int64)
    return [np.arange(edges[i], edges[i + 1], dtype=np.int64) for i in range(n_blocks) if edges[i] < edges[i + 1]]


def run_batches(
    *,
    config: BatchExecutorConfig,
    items: Sequence[T],
    fn: Callable[[T], R],
) -> list[R]:
    if config.backend == "serial" or len(items) <= 1:
        return [fn(it) for it in items]
    if config.backend != "process_pool":
        raise RuntimeError(f"Unsupported BatchExecutor backend={config.backend!r}")

    # Deterministic submit and collect order.
    with ProcessPoolExecutor(max_workers=config.max_workers) as ex:
        futs = [ex.submit(fn, it) for it in items]
        return [f.result() for f in futs]


def make_worker_seed(base_seed: int, worker_index: int) -> int:
    return int(base_seed) + int(worker_index)
