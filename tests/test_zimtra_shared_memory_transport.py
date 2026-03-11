import numpy as np

from weightiz.shared.io.profile_engine import attach_shared_buffers, cleanup_shared_buffers, write_shared_buffers


def test_shared_memory_attach_readonly_views() -> None:
    arr = np.arange(12, dtype=np.float64).reshape(3, 4)
    reg = write_shared_buffers({"x": arr})
    try:
        arrays, handles = attach_shared_buffers(reg)
        try:
            x = arrays["x"]
            assert x.flags.writeable is False
            assert np.allclose(x, arr)
        finally:
            for h in handles.values():
                h.close()
    finally:
        cleanup_shared_buffers(reg)
