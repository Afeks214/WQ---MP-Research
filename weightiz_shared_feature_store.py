from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import signal
from typing import Any

import numpy as np
from multiprocessing import shared_memory

from weightiz_dtype_guard import assert_float64


_SHM_PREFIX = "weightiz_tensor_"
_OWNER_RE = re.compile(r"^weightiz_tensor_(\d+)_")


@dataclass(frozen=True)
class SharedFeatureRegistry:
    name: str
    shape: tuple[int, ...]
    dtype: str
    owner_pid: int


@dataclass
class SharedFeatureHandles:
    shm: shared_memory.SharedMemory
    array: np.ndarray


def estimate_tensor_bytes(A: int, T: int, F: int, W: int) -> int:
    return int(A) * int(T) * int(F) * int(W) * 8


def enforce_memory_safety(tensor_bytes: int, available_ram_bytes: int) -> None:
    if int(tensor_bytes) > int(0.8 * float(available_ram_bytes)):
        raise RuntimeError("Feature tensor exceeds safe memory limit")


def _list_weightiz_segments() -> list[str]:
    dev_shm = Path("/dev/shm")
    if not dev_shm.exists():
        return []
    out: list[str] = []
    for p in dev_shm.glob(f"{_SHM_PREFIX}*"):
        out.append(p.name)
    return sorted(set(out))


def _owner_pid_from_name(name: str) -> int | None:
    m = _OWNER_RE.match(str(name))
    if m is None:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def cleanup_orphan_shared_memory_segments() -> int:
    removed = 0
    for seg in _list_weightiz_segments():
        owner = _owner_pid_from_name(seg)
        if owner is None:
            continue
        if _pid_exists(owner):
            continue
        try:
            shm = shared_memory.SharedMemory(name=seg, create=False)
        except FileNotFoundError:
            continue
        try:
            shm.unlink()
            removed += 1
        finally:
            shm.close()
    return removed


def create_shared_feature_store(tensor: np.ndarray) -> tuple[SharedFeatureRegistry, SharedFeatureHandles]:
    assert_float64("shared_feature_store_create", tensor)
    owner_pid = int(os.getpid())
    shm_name = f"{_SHM_PREFIX}{owner_pid}_{os.urandom(4).hex()}"
    shm = shared_memory.SharedMemory(name=shm_name, create=True, size=int(tensor.nbytes))
    arr = np.ndarray(tensor.shape, dtype=np.float64, buffer=shm.buf)
    arr[:] = tensor
    registry = SharedFeatureRegistry(
        name=shm_name,
        shape=tuple(int(x) for x in tensor.shape),
        dtype="float64",
        owner_pid=owner_pid,
    )
    return registry, SharedFeatureHandles(shm=shm, array=arr)


def attach_shared_feature_store(registry: SharedFeatureRegistry) -> SharedFeatureHandles:
    shm = shared_memory.SharedMemory(name=registry.name, create=False)
    arr = np.ndarray(tuple(registry.shape), dtype=np.float64, buffer=shm.buf)
    arr.setflags(write=False)
    assert_float64("shared_feature_store_attach", arr)
    return SharedFeatureHandles(shm=shm, array=arr)


def close_shared_feature_store(handles: SharedFeatureHandles, is_master: bool, owner_pid: int | None = None) -> None:
    try:
        handles.shm.close()
    finally:
        if is_master:
            if owner_pid is not None and int(owner_pid) != int(os.getpid()):
                return
            try:
                handles.shm.unlink()
            except FileNotFoundError:
                pass


def install_shm_cleanup_handlers(cleanup_fn: Any) -> None:
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, cleanup_fn)
        except Exception:
            continue
