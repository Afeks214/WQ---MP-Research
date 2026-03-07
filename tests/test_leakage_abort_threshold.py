from __future__ import annotations

import numpy as np
import pytest

from weightiz_module2_core import compute_window_correlation_diagnostics


def test_window_leakage_abort_threshold(tmp_path):
    # Perfectly correlated windows -> must abort at threshold 0.995
    base = np.random.default_rng(7).normal(size=(2, 20, 3)).astype(np.float64)
    t = np.stack([base, base], axis=-1)  # [A,T,F,W=2]

    with pytest.raises(RuntimeError, match="WINDOW_STATISTICAL_LEAKAGE_ABORT"):
        compute_window_correlation_diagnostics(
            t,
            feature_map={"f0": 0, "f1": 1, "f2": 2},
            window_map={"0": 15, "1": 30},
            warning_threshold=0.98,
            abort_threshold=0.995,
            run_dir=tmp_path,
        )

    assert (tmp_path / "window_correlation_diagnostics.parquet").exists()
    assert (tmp_path / "window_leakage_warnings.jsonl").exists()
