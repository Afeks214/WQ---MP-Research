from __future__ import annotations

from tests.module6_testkit import make_test_config, run_module6_on_synthetic


def test_module6_e2e_smoke(tmp_path):
    report = run_module6_on_synthetic(tmp_path, config=make_test_config())
    assert len(report.selected_portfolio_pks) > 0
    assert (report.output_dir / "portfolio_candidates.parquet").exists()
    assert (report.output_dir / "portfolio_scores.parquet").exists()
