from __future__ import annotations

import numpy as np
import pytest

from weightiz.module6.dependence import build_covariance_bundle
from weightiz.module6.psd import enforce_psd
from weightiz.module6.config import DependenceConfig
from weightiz.module6.utils import Module6ValidationError


def test_enforce_psd_rejects_large_negative_mass():
    with pytest.raises(Module6ValidationError):
        enforce_psd(np.asarray([[1.0, 3.0], [3.0, -10.0]], dtype=np.float64), DependenceConfig())


def test_build_covariance_bundle_returns_psd_outputs():
    r = np.asarray([[0.01, 0.011, -0.01], [0.02, 0.021, -0.02], [0.0, 0.001, 0.002], [0.01, 0.009, -0.008]], dtype=np.float64)
    a = np.ones_like(r, dtype=bool)
    g = np.asarray([[0.1, 0.2, 0.0, 0.0], [0.1, 0.19, 0.0, 0.0], [-0.1, 0.0, 0.1, 0.2]], dtype=np.float64)
    bundle = build_covariance_bundle(r, a, g, np.asarray([0, 1, 2], dtype=np.int64), DependenceConfig())
    eig = np.linalg.eigvalsh(bundle.covariance)
    assert np.all(eig >= -1.0e-9)
    assert bundle.drawdown_concurrence.shape == (3, 3)

