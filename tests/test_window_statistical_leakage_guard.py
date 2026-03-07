from __future__ import annotations

import numpy as np

from weightiz_module2_core import compute_window_correlation_diagnostics


def test_window_statistical_leakage_warning_output(tmp_path):
    rng = np.random.default_rng(9)
    a = rng.normal(size=(2, 50, 2)).astype(np.float64)
    b = a * 0.99 + 0.01 * rng.normal(size=(2, 50, 2))
    t = np.stack([a, b], axis=-1)

    rows, warns = compute_window_correlation_diagnostics(
        t,
        feature_map={"f0": 0, "f1": 1},
        window_map={"0": 15, "1": 30},
        warning_threshold=0.5,
        abort_threshold=1.1,
        run_dir=tmp_path,
    )
    assert len(rows) > 0
    assert len(warns) > 0
    assert (tmp_path / "window_correlation_diagnostics.parquet").exists()
    assert (tmp_path / "window_leakage_warnings.jsonl").exists()
