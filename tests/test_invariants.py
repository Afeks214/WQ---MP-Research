from __future__ import annotations

import unittest

import numpy as np

from weightiz_invariants import assert_or_flag_finite


class TestInvariants(unittest.TestCase):
    def test_masks_nonfinite_without_raising(self) -> None:
        valid = np.array([[True, True], [True, False]], dtype=bool)
        feat2 = np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float64)
        feat3 = np.zeros((2, 2, 3), dtype=np.float64)
        feat3[0, 0, 1] = np.inf

        updated, flags = assert_or_flag_finite(
            features={"feat2": feat2, "feat3": feat3},
            valid_mask=valid,
            context="unit",
        )

        self.assertFalse(bool(updated[0, 1]))
        self.assertFalse(bool(updated[0, 0]))
        self.assertTrue(bool(updated[1, 0]))
        self.assertEqual(int(flags["invalid_count"]), 2)
        self.assertIn("feat2", flags["offending_features"])
        self.assertIn("feat3", flags["offending_features"])

    def test_programmer_misuse_raises(self) -> None:
        valid = np.ones((2, 2), dtype=bool)
        bad = np.zeros((2, 3), dtype=np.float64)
        with self.assertRaisesRegex(RuntimeError, "shape prefix mismatch"):
            assert_or_flag_finite(features={"bad": bad}, valid_mask=valid, context="unit")


if __name__ == "__main__":
    unittest.main()
