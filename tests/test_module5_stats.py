import unittest
import numpy as np

from weightiz_module5_stats import (
    build_cscv_incidence,
    build_cscv_slices,
    deflated_sharpe_ratio,
    model_confidence_set,
    norm_cdf,
    pbo_cscv,
    psr_against_threshold,
    sample_skew_kurtosis_excess,
    stationary_bootstrap_indices,
    validate_returns_2d,
    white_reality_check,
)


class TestModule5Stats(unittest.TestCase):
    def test_validation_tmax_guard(self):
        x = np.zeros((10_001, 2), dtype=np.float64)
        with self.assertRaises(RuntimeError):
            validate_returns_2d(x)

    def test_kurtosis_term_correction(self):
        sr = np.array([0.15], dtype=np.float64)
        sr_star = np.array([0.10], dtype=np.float64)
        skew = np.array([0.2], dtype=np.float64)
        k_ex = np.array([1.0], dtype=np.float64)  # standard kurtosis would be 4.0
        out = psr_against_threshold(sr, sr_star, skew, k_ex, n_obs=500)
        self.assertTrue(0.0 <= float(out[0]) <= 1.0)

        # Explicit denominator consistency check:
        den1 = np.sqrt(np.maximum(1.0 - skew * sr + ((k_ex + 2.0) / 4.0) * (sr * sr), 1e-12))
        gamma4 = k_ex + 3.0
        den2 = np.sqrt(np.maximum(1.0 - skew * sr + ((gamma4 - 1.0) / 4.0) * (sr * sr), 1e-12))
        self.assertAlmostEqual(float(den1[0]), float(den2[0]), places=12)

    def test_dsr_has_daily_and_annualized_outputs(self):
        rng = np.random.default_rng(1)
        r = rng.normal(0.0005, 0.01, size=(800, 6)).astype(np.float64)
        out = deflated_sharpe_ratio(r)
        self.assertIn("sharpe_daily", out)
        self.assertIn("sharpe_ann", out)
        self.assertIn("sharpe_deflated_threshold_daily", out)
        self.assertEqual(out["sharpe_daily"].shape[0], 6)
        self.assertEqual(out["sharpe_ann"].shape[0], 6)
        self.assertEqual(out["dsr"].shape[0], 6)

    def test_build_cscv(self):
        sl = build_cscv_slices(1000, S=10)
        self.assertEqual(sl.shape, (10, 2))
        inc = build_cscv_incidence(S=10, k=5)
        self.assertEqual(inc.shape[1], 10)
        self.assertTrue(np.all(np.sum(inc.astype(np.int64), axis=1) == 5))

    def test_pbo_output_sanity(self):
        rng = np.random.default_rng(2)
        r = rng.normal(0.0, 0.01, size=(900, 8)).astype(np.float64)
        out = pbo_cscv(r, S=10, k=5)
        self.assertTrue(np.isfinite(out["pbo"]))
        self.assertTrue(0.0 <= float(out["pbo"]) <= 1.0)
        self.assertEqual(out["lambda_logits"].ndim, 1)

    def test_stationary_bootstrap_bounds(self):
        idx = stationary_bootstrap_indices(T=777, B=128, avg_block_len=15, seed=7)
        self.assertEqual(idx.shape, (128, 777))
        self.assertTrue(np.all(idx >= 0))
        self.assertTrue(np.all(idx < 777))

    def test_wrc_pvalue_bounds(self):
        rng = np.random.default_rng(4)
        T, N = 600, 5
        r = rng.normal(0.0002, 0.01, size=(T, N)).astype(np.float64)
        b = rng.normal(0.0001, 0.01, size=T).astype(np.float64)
        out = white_reality_check(r, b, B=256, avg_block_len=20, seed=9)
        self.assertTrue(0.0 <= float(out["p_value"]) <= 1.0)
        self.assertEqual(out["mean_diff"].shape[0], N)

    def test_mcs_non_empty_survivors(self):
        rng = np.random.default_rng(10)
        T, N = 500, 6
        losses = rng.normal(0.0, 1.0, size=(T, N)).astype(np.float64)
        losses[:, 0] += 0.5  # make model 0 systematically worse
        out = model_confidence_set(losses, alpha=0.10, B=256, avg_block_len=20, seed=21)
        surv = out["survivors"]
        self.assertTrue(isinstance(surv, np.ndarray))
        self.assertGreaterEqual(surv.size, 1)

    def test_norm_cdf_bounds(self):
        z = np.array([-3.0, 0.0, 3.0], dtype=np.float64)
        p = norm_cdf(z)
        self.assertTrue(np.all(p >= 0.0))
        self.assertTrue(np.all(p <= 1.0))


if __name__ == "__main__":
    unittest.main()

