from __future__ import annotations

import numpy as np

from weightiz.shared.io.shared_feature_store import close_shared_feature_store, create_shared_feature_store


def test_master_close_unlink_cleanup():
    t = np.ones((1, 4, 2, 2), dtype=np.float64)
    reg, h = create_shared_feature_store(t)
    close_shared_feature_store(h, is_master=True, owner_pid=reg.owner_pid)
