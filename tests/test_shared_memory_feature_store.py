from __future__ import annotations

import numpy as np

from weightiz_shared_feature_store import attach_shared_feature_store, close_shared_feature_store, create_shared_feature_store


def test_shared_feature_store_attach_readonly():
    t = np.random.default_rng(1).normal(size=(2, 10, 3, 2)).astype(np.float64)
    reg, h = create_shared_feature_store(t)
    try:
        wh = attach_shared_feature_store(reg)
        try:
            assert wh.array.shape == t.shape
            assert wh.array.dtype == np.float64
            assert wh.array.flags.writeable is False
        finally:
            close_shared_feature_store(wh, is_master=False)
    finally:
        close_shared_feature_store(h, is_master=True, owner_pid=reg.owner_pid)
