from __future__ import annotations

import numpy as np
import pytest

from weightiz.cli.run_module6 import build_module6_config
from weightiz.module6.dependence import _repair_covariance, build_covariance_bundle
from weightiz.module6.psd import enforce_psd
from weightiz.module6.config import DependenceConfig
from weightiz.module6.utils import Module6ValidationError


def test_enforce_psd_rejects_large_negative_mass():
    with pytest.raises(Module6ValidationError):
        enforce_psd(np.asarray([[1.0, 3.0], [3.0, -10.0]], dtype=np.float64), DependenceConfig())


def test_build_covariance_bundle_returns_psd_outputs():
    r = np.asarray([[0.01, 0.011, -0.01], [0.02, 0.021, -0.02], [0.0, 0.001, 0.002], [0.01, 0.009, -0.008]], dtype=np.float64)
    a = np.ones_like(r, dtype=bool)
    g = np.asarray([[0.1, 0.2, 0.0, 0.0], [0.1, 0.19, 0.0, 0.0], [0.1, 0.0, 0.1, 0.2]], dtype=np.float64)
    cfg = DependenceConfig(pair_support_min_sessions=2, pair_support_max_sessions=2, pair_support_full_min_sessions=4, pair_support_full_max_sessions=4)
    bundle = build_covariance_bundle(r, a, g, np.asarray([0, 1, 2], dtype=np.int64), cfg, asset_support_minimum=1)
    eig = np.linalg.eigvalsh(bundle.covariance)
    assert np.all(eig >= -1.0e-9)
    assert bundle.drawdown_concurrence.shape == (3, 3)
    assert bundle.pair_overlap_counts.shape == (3, 3)
    assert bundle.covariance_pre_psd.shape == (3, 3)


def test_build_covariance_bundle_zero_overlap_pairs_use_prior_completion():
    r = np.asarray(
        [
            [0.02, 0.03, 0.00],
            [0.01, 0.02, 0.00],
            [0.00, 0.01, 0.04],
            [0.00, 0.00, 0.05],
        ],
        dtype=np.float64,
    )
    a = np.asarray(
        [
            [True, True, False],
            [True, True, False],
            [False, True, True],
            [False, False, True],
        ],
        dtype=bool,
    )
    g = np.asarray([[0.4, 0.0, 0.0, 0.0], [0.3, 0.1, 0.0, 0.0], [0.0, 0.2, 0.4, 0.0]], dtype=np.float64)
    cfg = DependenceConfig(pair_support_min_sessions=1, pair_support_max_sessions=1, pair_support_full_min_sessions=2, pair_support_full_max_sessions=2)
    bundle = build_covariance_bundle(r, a, g, np.asarray([0, 1, 2], dtype=np.int64), cfg, asset_support_minimum=1)
    assert int(bundle.pair_overlap_counts[0, 2]) == 0
    assert float(bundle.pair_reliability[0, 2]) == 0.0
    assert bundle.pair_completion_reason_codes[0, 2] == "PAIR_OVERLAP_ZERO"
    assert bundle.pair_completion_reason_codes[0, 1] == ""
    assert bool(bundle.completion_mask[0, 2])
    assert np.isfinite(bundle.covariance).all()
    assert bundle.completion_prior_used is True
    assert "PAIR_OVERLAP_ZERO" in bundle.completion_reason_codes


def test_build_covariance_bundle_submin_overlap_pairs_emit_reason_code():
    r = np.asarray(
        [
            [0.02, 0.03],
            [0.01, 0.00],
            [0.00, 0.04],
            [0.00, 0.05],
        ],
        dtype=np.float64,
    )
    a = np.asarray(
        [
            [True, True],
            [True, False],
            [False, True],
            [False, True],
        ],
        dtype=bool,
    )
    g = np.asarray([[0.4, 0.0, 0.0, 0.0], [0.3, 0.1, 0.0, 0.0]], dtype=np.float64)
    cfg = DependenceConfig(pair_support_min_sessions=2, pair_support_max_sessions=2, pair_support_full_min_sessions=4, pair_support_full_max_sessions=4)
    bundle = build_covariance_bundle(r, a, g, np.asarray([0, 1], dtype=np.int64), cfg, asset_support_minimum=1)
    assert int(bundle.pair_overlap_counts[0, 1]) == 1
    assert float(bundle.pair_reliability[0, 1]) == 0.0
    assert bundle.pair_completion_reason_codes[0, 1] == "PAIR_OVERLAP_SUBMIN"
    assert bundle.completion_prior_used is True
    assert "PAIR_OVERLAP_SUBMIN" in bundle.completion_reason_codes


def test_enforce_psd_rejects_large_projection_distortion():
    with pytest.raises(Module6ValidationError, match="PSD_DISTORTION_EXCESSIVE"):
        _repair_covariance(covariance=np.asarray([[1.0, 5.0], [5.0, 1.0]], dtype=np.float64), config=DependenceConfig())


def test_build_covariance_bundle_rejects_excessive_regime_mismatch():
    r = np.asarray([[0.01, 0.02], [0.02, 0.03], [0.01, 0.01], [0.0, 0.02]], dtype=np.float64)
    a = np.ones_like(r, dtype=bool)
    g = np.asarray([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    cfg = DependenceConfig(
        pair_support_min_sessions=2,
        pair_support_max_sessions=2,
        pair_support_full_min_sessions=4,
        pair_support_full_max_sessions=4,
    )
    with pytest.raises(Module6ValidationError, match="REGIME_MISMATCH_EXCESSIVE"):
        build_covariance_bundle(r, a, g, np.asarray([0, 1], dtype=np.int64), cfg, asset_support_minimum=1)


def test_build_covariance_bundle_enforces_asset_support_floor_for_direct_callers():
    r = np.random.default_rng(0).normal(size=(10, 2)).astype(np.float64)
    a = np.ones((10, 2), dtype=bool)
    g = np.ones((2, 4), dtype=np.float64)
    with pytest.raises(Module6ValidationError, match="DEPENDENCE_ASSET_SUPPORT_TOO_SHORT"):
        build_covariance_bundle(r, a, g, np.asarray([0, 1], dtype=np.int64), DependenceConfig())


def test_completion_reason_summary_matches_pair_level_reason_surface():
    r = np.asarray(
        [
            [0.02, 0.03, 0.00],
            [0.01, 0.02, 0.00],
            [0.00, 0.01, 0.04],
            [0.00, 0.00, 0.05],
        ],
        dtype=np.float64,
    )
    a = np.asarray(
        [
            [True, True, False],
            [True, True, False],
            [False, True, True],
            [False, False, True],
        ],
        dtype=bool,
    )
    g = np.asarray([[0.4, 0.0, 0.0, 0.0], [0.3, 0.1, 0.0, 0.0], [0.0, 0.2, 0.4, 0.0]], dtype=np.float64)
    cfg = DependenceConfig(pair_support_min_sessions=1, pair_support_max_sessions=1, pair_support_full_min_sessions=2, pair_support_full_max_sessions=2)
    bundle = build_covariance_bundle(r, a, g, np.asarray([0, 1, 2], dtype=np.int64), cfg, asset_support_minimum=1)
    pair_level_codes = {
        str(code)
        for code in np.unique(bundle.pair_completion_reason_codes)
        if str(code)
    }
    assert pair_level_codes == set(bundle.completion_reason_codes)


def test_build_module6_config_is_backward_compatible_without_sparse_fields():
    cfg = build_module6_config({"intake": {"min_availability_ratio": 0.5, "min_observed_sessions": 4}})
    assert cfg.dependence.pair_support_min_sessions == 20
    assert cfg.dependence.psd_max_distortion == pytest.approx(0.30)
