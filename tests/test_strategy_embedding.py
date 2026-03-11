from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from weightiz.module5.strategy_embedding import (
    build_strategy_embeddings,
    cluster_strategies_hierarchical_threshold,
    compute_correlation_distance,
)


class TestStrategyEmbedding(unittest.TestCase):
    def test_embedding_shape_and_determinism(self) -> None:
        rng = np.random.default_rng(11)
        r = rng.normal(0.0, 0.01, size=(400, 12)).astype(np.float64)
        a = build_strategy_embeddings(r, seed=123)
        b = build_strategy_embeddings(r, seed=123)
        self.assertEqual(a["embeddings"].shape, (12, 12))
        self.assertEqual(len(a["feature_names"]), 12)
        self.assertTrue(np.array_equal(a["embeddings"], b["embeddings"]))

    def test_cluster_reduces_highly_correlated_strategies(self) -> None:
        rng = np.random.default_rng(22)
        base = rng.normal(0.0, 0.01, size=500).astype(np.float64)
        g1 = np.column_stack([base + rng.normal(0.0, 0.0005, size=500) for _ in range(6)]).astype(np.float64)
        g2_raw = rng.normal(0.0, 0.01, size=500).astype(np.float64)
        g2 = np.column_stack([g2_raw + rng.normal(0.0, 0.0005, size=500) for _ in range(6)]).astype(np.float64)
        r = np.concatenate([g1, g2], axis=1)

        out = cluster_strategies_hierarchical_threshold(
            r,
            corr_threshold=0.90,
            block_size=64,
            in_memory_max_n=1000,
            seed=77,
        )
        labels = np.asarray(out["cluster_labels"], dtype=np.int64)
        reps = np.asarray(out["cluster_representatives"], dtype=np.int64)

        self.assertEqual(labels.shape[0], 12)
        self.assertGreaterEqual(reps.shape[0], 2)
        self.assertLess(reps.shape[0], 12)
        self.assertEqual(int(out["n_eff"]), int(reps.shape[0]))

    def test_representative_tie_break_is_deterministic(self) -> None:
        rng = np.random.default_rng(33)
        r = rng.normal(0.0, 0.01, size=(300, 4)).astype(np.float64)
        r[:, 1] = r[:, 0]
        out_a = cluster_strategies_hierarchical_threshold(r, corr_threshold=0.99, seed=1)
        out_b = cluster_strategies_hierarchical_threshold(r, corr_threshold=0.99, seed=1)
        self.assertTrue(np.array_equal(out_a["cluster_labels"], out_b["cluster_labels"]))
        self.assertTrue(np.array_equal(out_a["cluster_representatives"], out_b["cluster_representatives"]))

    def test_distance_memmap_path_for_large_n(self) -> None:
        rng = np.random.default_rng(44)
        r = rng.normal(0.0, 0.01, size=(120, 64)).astype(np.float64)
        with tempfile.TemporaryDirectory(prefix="strategy_embedding_dist_") as td:
            p = Path(td) / "dist.dat"
            d = compute_correlation_distance(
                r,
                block_size=16,
                out_path=str(p),
                in_memory_max_n=8,
            )
            self.assertEqual(d.shape, (64, 64))
            self.assertTrue(p.exists())
            self.assertTrue(np.all(np.isfinite(np.asarray(d, dtype=np.float64))))

    def test_single_strategy_and_zero_variance_column_are_supported(self) -> None:
        one = np.zeros((30, 1), dtype=np.float64)
        out = cluster_strategies_hierarchical_threshold(one, corr_threshold=0.90, seed=7)
        self.assertEqual(np.asarray(out["cluster_labels"]).tolist(), [0])
        self.assertEqual(np.asarray(out["cluster_representatives"]).tolist(), [0])

        r = np.column_stack(
            [
                np.zeros(80, dtype=np.float64),
                np.linspace(-0.01, 0.01, 80, dtype=np.float64),
            ]
        )
        d = np.asarray(compute_correlation_distance(r, block_size=8, in_memory_max_n=1000), dtype=np.float64)
        self.assertTrue(np.all(np.isfinite(d)))
        self.assertEqual(float(d[0, 0]), 0.0)

    def test_isolated_strategies_remain_separate_and_order_is_stable(self) -> None:
        t = np.linspace(0.0, 8.0 * np.pi, 320, dtype=np.float64)
        r = np.column_stack(
            [
                0.01 * np.sin(t),
                0.01 * np.cos(t),
                0.01 * np.sin(2.0 * t + 0.2),
            ]
        ).astype(np.float64)
        out_a = cluster_strategies_hierarchical_threshold(r, corr_threshold=0.99, seed=5)
        out_b = cluster_strategies_hierarchical_threshold(r[:, [2, 0, 1]], corr_threshold=0.99, seed=5)
        self.assertEqual(int(out_a["n_eff"]), 3)
        self.assertEqual(int(out_b["n_eff"]), 3)
        self.assertEqual(sorted(np.asarray(out_a["cluster_representatives"]).tolist()), [0, 1, 2])
        self.assertEqual(sorted(np.asarray(out_b["cluster_representatives"]).tolist()), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
