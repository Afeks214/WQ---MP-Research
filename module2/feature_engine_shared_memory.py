from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Mapping
import numpy as np


@dataclass(frozen=True)
class SharedArraySpec:
    name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass
class SharedArrayHandle:
    shm_name: str
    shape: tuple[int, ...]
    dtype: np.dtype


class SharedMemoryFeatureEngine:
    """Deterministic shared-memory transport for OHLCV tensors."""

    def __init__(self) -> None:
        self._owners: dict[str, shared_memory.SharedMemory] = {}

    def publish_arrays(self, arrays: Mapping[str, np.ndarray]) -> dict[str, SharedArrayHandle]:
        out: dict[str, SharedArrayHandle] = {}
        for key in sorted(arrays.keys()):
            arr = np.ascontiguousarray(np.asarray(arrays[key], dtype=np.float64))
            shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
            buf = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
            buf[:] = arr
            buf.flags.writeable = False
            self._owners[key] = shm
            out[key] = SharedArrayHandle(shm_name=shm.name, shape=arr.shape, dtype=arr.dtype)
        return out

    @staticmethod
    def attach_readonly(handles: Mapping[str, SharedArrayHandle]) -> tuple[dict[str, np.ndarray], list[shared_memory.SharedMemory]]:
        arrays: dict[str, np.ndarray] = {}
        refs: list[shared_memory.SharedMemory] = []
        for key in sorted(handles.keys()):
            h = handles[key]
            shm = shared_memory.SharedMemory(name=h.shm_name, create=False)
            arr = np.ndarray(h.shape, dtype=np.dtype(h.dtype), buffer=shm.buf)
            arr.flags.writeable = False
            arrays[key] = arr
            refs.append(shm)
        return arrays, refs

    def close(self, unlink: bool = True) -> None:
        for shm in self._owners.values():
            try:
                shm.close()
            finally:
                if unlink:
                    try:
                        shm.unlink()
                    except FileNotFoundError:
                        pass
        self._owners.clear()
