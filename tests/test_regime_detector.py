from __future__ import annotations

import unittest

import numpy as np

from regime_detector import RegimeConfig, build_regime_masks, detect_regimes, regime_sample_counts


class TestRegimeDetector(unittest.TestCase):
    def test_detect_regimes_is_deterministic(self) -> None:
        rng = np.random.default_rng(101)
        r = rng.normal(0.0, 0.01, size=600).astype(np.float64)
        cfg = RegimeConfig(vol_window=40, slope_window=40, hurst_window=80, min_obs_per_mask=10)
        a = detect_regimes(r, cfg=cfg)
        b = detect_regimes(r, cfg=cfg)
        self.assertTrue(np.array_equal(a["volatility_regime"], b["volatility_regime"]))
        self.assertTrue(np.array_equal(a["trend_regime"], b["trend_regime"]))
        self.assertTrue(np.array_equal(a["range_regime"], b["range_regime"]))

    def test_build_masks_and_counts(self) -> None:
        rng = np.random.default_rng(202)
        low_vol = rng.normal(0.0, 0.002, size=200)
        high_vol = rng.normal(0.0, 0.03, size=200)
        trend = np.linspace(-0.01, 0.01, 200, dtype=np.float64)
        r = np.concatenate([low_vol, trend, high_vol]).astype(np.float64)

        doc = detect_regimes(
            r,
            cfg=RegimeConfig(
                vol_window=30,
                slope_window=30,
                hurst_window=60,
                min_obs_per_mask=15,
            ),
        )
        masks = build_regime_masks(doc, min_obs=15)
        counts = regime_sample_counts(masks)

        self.assertGreaterEqual(len(masks), 3)
        self.assertTrue(all(int(v) >= 15 for v in counts.values()))
        self.assertIn("vol_high", masks)
        self.assertIn("vol_low", masks)

    def test_minimum_obs_filter(self) -> None:
        r = np.linspace(-0.002, 0.002, 40, dtype=np.float64)
        doc = detect_regimes(r, cfg=RegimeConfig(vol_window=20, slope_window=20, hurst_window=20))
        masks = build_regime_masks(doc, min_obs=100)
        self.assertEqual(masks, {})

    def test_zero_and_constant_returns_are_deterministic(self) -> None:
        zero = np.zeros(120, dtype=np.float64)
        const = np.full(120, 0.001, dtype=np.float64)
        cfg = RegimeConfig(vol_window=20, slope_window=20, hurst_window=32, min_obs_per_mask=5)
        doc_zero = detect_regimes(zero, cfg=cfg)
        doc_const = detect_regimes(const, cfg=cfg)

        for doc in (doc_zero, doc_const):
            self.assertTrue(np.all(np.isfinite(np.nan_to_num(doc["volatility"], nan=0.0))))
            self.assertTrue(np.all(np.isfinite(np.nan_to_num(doc["slope_z"], nan=0.0))))
            self.assertTrue(np.all(np.isfinite(np.nan_to_num(doc["hurst"], nan=0.5))))
            masks = build_regime_masks(doc, min_obs=5)
            self.assertIsInstance(masks, dict)

    def test_short_series_and_quantile_collapse_are_handled(self) -> None:
        short = np.asarray([0.0, 0.001, -0.001, 0.0, 0.001], dtype=np.float64)
        doc_short = detect_regimes(short, cfg=RegimeConfig(vol_window=20, slope_window=20, hurst_window=32))
        self.assertEqual(doc_short["volatility_regime"].shape[0], short.shape[0])

        flat_vol = np.tile(np.asarray([0.001, -0.001], dtype=np.float64), 80)
        doc_flat = detect_regimes(flat_vol, cfg=RegimeConfig(vol_window=16, slope_window=16, hurst_window=32))
        masks = build_regime_masks(doc_flat, min_obs=10)
        counts = regime_sample_counts(masks)
        self.assertTrue(all(int(v) >= 10 for v in counts.values()))


if __name__ == "__main__":
    unittest.main()
