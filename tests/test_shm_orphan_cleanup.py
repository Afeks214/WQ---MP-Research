from __future__ import annotations

from types import SimpleNamespace

import weightiz_shared_feature_store as shm_mod


class _FakeShm:
    unlinked: list[str] = []

    def __init__(self, name: str, create: bool = False):
        self.name = name

    def unlink(self):
        _FakeShm.unlinked.append(self.name)

    def close(self):
        return None


def test_orphan_shared_memory_cleanup(monkeypatch):
    monkeypatch.setattr(shm_mod, "_list_weightiz_segments", lambda: ["weightiz_tensor_111_aaaa", "weightiz_tensor_222_bbbb"])
    monkeypatch.setattr(shm_mod, "_pid_exists", lambda pid: pid == 222)
    monkeypatch.setattr(shm_mod, "shared_memory", SimpleNamespace(SharedMemory=_FakeShm))

    removed = shm_mod.cleanup_orphan_shared_memory_segments()
    assert removed == 1
    assert _FakeShm.unlinked == ["weightiz_tensor_111_aaaa"]
