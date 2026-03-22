from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import sparse

from weightiz.module6.export import write_module6_outputs
from weightiz.module6.types import PortfolioSelectionReport


def test_write_module6_outputs_sanitizes_nonfinite_json(tmp_path):
    output_dir = tmp_path / "module6_out"
    overlap_proxy = SimpleNamespace(
        symbol_support=sparse.eye(1, format="csr"),
        activity_concurrence=sparse.eye(1, format="csr"),
        gross_exposure_concurrence=sparse.eye(1, format="csr"),
        rebalance_collision=sparse.eye(1, format="csr"),
        composite=sparse.eye(1, format="csr"),
    )
    dependence_bundle = SimpleNamespace(
        covariance=np.eye(1, dtype=np.float64),
        covariance_pre_psd=np.eye(1, dtype=np.float64),
        correlation=np.eye(1, dtype=np.float64),
        downside_covariance=np.eye(1, dtype=np.float64),
        regime_overlap=np.eye(1, dtype=np.float64),
        asset_column_indices=np.asarray([7], dtype=np.int64),
        pair_overlap_counts=np.ones((1, 1), dtype=np.int64),
        pair_reliability=np.ones((1, 1), dtype=np.float64),
        pair_completion_reason_codes=np.asarray([[""]], dtype="<U24"),
        completion_mask=np.zeros((1, 1), dtype=bool),
        asset_observed_counts=np.asarray([4], dtype=np.int64),
        asset_support_minimum=2,
        pair_support_minimum=2,
        pair_support_full=3,
        completion_prior_used=True,
        completion_reason_codes=("PAIR_OVERLAP_ZERO",),
        drawdown_concurrence=sparse.eye(1, format="csr"),
        shrinkage=np.inf,
        negative_mass=np.nan,
        negative_eigen_mass_ratio=np.nan,
        condition_number=np.inf,
        psd_projection_distortion=np.inf,
        min_eigenvalue_pre=-1.0,
        min_eigenvalue_post=0.0,
        off_diagonal_sign_flip_rate=np.nan,
        spurious_extreme_correlation_rate=np.nan,
        regime_mismatch_rate=np.nan,
        effective_pair_count=0,
        effective_pair_reliability=np.nan,
        prior_only_pair_count=0,
        zero_overlap_pair_count=0,
        submin_overlap_pair_count=0,
        repair_status="warn",
    )
    report = PortfolioSelectionReport(
        run_id="run_x",
        output_dir=output_dir,
        selected_portfolio_pks=("p1",),
        alternate_portfolio_pks=(),
        summary={"quality": np.nan, "rank": np.inf},
    )

    write_module6_outputs(
        output_dir=output_dir,
        candidates=pd.DataFrame({"portfolio_pk": ["p1"], "reduced_universe_id": ["u0"], "cash_weight": [0.0]}),
        portfolio_weights=pd.DataFrame({"portfolio_pk": ["p1"], "strategy_instance_pk": ["s1"], "target_weight": [1.0]}),
        session_paths=pd.DataFrame({"portfolio_pk": ["p1"], "session_id": [1], "equity": [1000.0]}),
        comparison_support_session_paths=pd.DataFrame({"portfolio_pk": ["p1"], "session_id": [1], "equity": [1000.0]}),
        minute_paths=pd.DataFrame({"portfolio_pk": ["p1"], "ts_ns": [1], "equity": [1000.0]}),
        minute_component_diagnostics=pd.DataFrame({"portfolio_pk": ["p1"], "micro_trade_notional": [0.0]}),
        session_scores=pd.DataFrame({"portfolio_pk": ["p1"], "first_pass_score": [0.5]}),
        comparison_support_session_scores=pd.DataFrame({"portfolio_pk": ["p1"], "first_pass_score": [0.5]}),
        finalist_scores=pd.DataFrame({"portfolio_pk": ["p1"], "final_score": [0.4]}),
        comparable_scores=pd.DataFrame({"portfolio_pk": ["p1"], "comparable_truth_score": [np.inf], "x": [np.nan]}),
        divergence=pd.DataFrame({"portfolio_pk": ["p1"], "rank_delta": [0]}),
        weight_history=pd.DataFrame({"portfolio_pk": ["p1"], "session_id": [1], "strategy_instance_pk": ["s1"]}),
        comparison_support_calendar=pd.DataFrame({"session_id": [1]}),
        dependence_artifacts={"u0": dependence_bundle},
        overlap_proxy=overlap_proxy,
        overlap_proxy_index=pd.DataFrame({"strategy_instance_pk": ["s1"], "overlap_proxy_idx": [0]}),
        exact_overlap=pd.DataFrame({"portfolio_pk_a": ["p1"], "portfolio_pk_b": ["p1"], "overlap": [1.0]}),
        global_frontier=pd.DataFrame({"portfolio_pk": ["p1"], "final_score": [0.4]}),
        risk_return_frontier=pd.DataFrame({"portfolio_pk": ["p1"], "minute_annualized_return": [0.1]}),
        operational_frontier=pd.DataFrame({"portfolio_pk": ["p1"], "availability_burden": [0.0]}),
        selected_frontier=pd.DataFrame({"portfolio_pk": ["p1"], "final_score": [0.4]}),
        selection_report=report,
    )

    selection_text = (output_dir / "portfolio_selection_report.json").read_text(encoding="utf-8")
    metadata_text = (output_dir / "dependence" / "u0" / "metadata.json").read_text(encoding="utf-8")

    assert "NaN" not in selection_text
    assert "Infinity" not in selection_text
    assert "NaN" not in metadata_text
    assert "Infinity" not in metadata_text

    json.loads(selection_text)
    metadata = json.loads(metadata_text)
    assert metadata["asset_column_indices"] == [7]
    assert metadata["completion_prior_used"] is True
    assert metadata["completion_reason_codes"] == ["PAIR_OVERLAP_ZERO"]
    assert metadata["min_eigenvalue_pre"] == -1.0
    assert metadata["min_eigenvalue_post"] == 0.0


def test_write_module6_outputs_writes_pair_reason_artifact_with_asset_ordering(tmp_path):
    output_dir = tmp_path / "module6_out_pair_reasons"
    overlap_proxy = SimpleNamespace(
        symbol_support=sparse.eye(2, format="csr"),
        activity_concurrence=sparse.eye(2, format="csr"),
        gross_exposure_concurrence=sparse.eye(2, format="csr"),
        rebalance_collision=sparse.eye(2, format="csr"),
        composite=sparse.eye(2, format="csr"),
    )
    dependence_bundle = SimpleNamespace(
        covariance=np.eye(2, dtype=np.float64),
        covariance_pre_psd=np.eye(2, dtype=np.float64),
        correlation=np.eye(2, dtype=np.float64),
        downside_covariance=np.eye(2, dtype=np.float64),
        regime_overlap=np.eye(2, dtype=np.float64),
        asset_column_indices=np.asarray([11, 17], dtype=np.int64),
        pair_overlap_counts=np.asarray([[4, 0], [0, 4]], dtype=np.int64),
        pair_reliability=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        pair_completion_reason_codes=np.asarray([["", "PAIR_OVERLAP_ZERO"], ["PAIR_OVERLAP_ZERO", ""]], dtype="<U24"),
        completion_mask=np.asarray([[False, True], [True, False]], dtype=bool),
        asset_observed_counts=np.asarray([4, 4], dtype=np.int64),
        asset_support_minimum=2,
        pair_support_minimum=2,
        pair_support_full=3,
        completion_prior_used=True,
        completion_reason_codes=("PAIR_OVERLAP_ZERO",),
        drawdown_concurrence=sparse.eye(2, format="csr"),
        shrinkage=0.5,
        negative_mass=0.0,
        negative_eigen_mass_ratio=0.0,
        condition_number=1.0,
        psd_projection_distortion=0.0,
        min_eigenvalue_pre=1.0,
        min_eigenvalue_post=1.0,
        off_diagonal_sign_flip_rate=0.0,
        spurious_extreme_correlation_rate=0.0,
        regime_mismatch_rate=0.0,
        effective_pair_count=0,
        effective_pair_reliability=0.0,
        prior_only_pair_count=1,
        zero_overlap_pair_count=1,
        submin_overlap_pair_count=0,
        repair_status="clean",
    )
    report = PortfolioSelectionReport(
        run_id="run_pair_reason",
        output_dir=output_dir,
        selected_portfolio_pks=("p1",),
        alternate_portfolio_pks=(),
        summary={},
    )

    write_module6_outputs(
        output_dir=output_dir,
        candidates=pd.DataFrame({"portfolio_pk": ["p1"], "reduced_universe_id": ["u0"], "cash_weight": [0.0]}),
        portfolio_weights=pd.DataFrame({"portfolio_pk": ["p1"], "strategy_instance_pk": ["s1"], "target_weight": [1.0]}),
        session_paths=pd.DataFrame({"portfolio_pk": ["p1"], "session_id": [1], "equity": [1000.0]}),
        comparison_support_session_paths=pd.DataFrame({"portfolio_pk": ["p1"], "session_id": [1], "equity": [1000.0]}),
        minute_paths=pd.DataFrame({"portfolio_pk": ["p1"], "ts_ns": [1], "equity": [1000.0]}),
        minute_component_diagnostics=pd.DataFrame({"portfolio_pk": ["p1"], "micro_trade_notional": [0.0]}),
        session_scores=pd.DataFrame({"portfolio_pk": ["p1"], "first_pass_score": [0.5]}),
        comparison_support_session_scores=pd.DataFrame({"portfolio_pk": ["p1"], "first_pass_score": [0.5]}),
        finalist_scores=pd.DataFrame({"portfolio_pk": ["p1"], "final_score": [0.4]}),
        comparable_scores=pd.DataFrame({"portfolio_pk": ["p1"], "comparable_truth_score": [0.4], "cross_universe_reject": [False]}),
        divergence=pd.DataFrame({"portfolio_pk": ["p1"], "rank_delta": [0]}),
        weight_history=pd.DataFrame({"portfolio_pk": ["p1"], "session_id": [1], "strategy_instance_pk": ["s1"]}),
        comparison_support_calendar=pd.DataFrame({"session_id": [1]}),
        dependence_artifacts={"u0": dependence_bundle},
        overlap_proxy=overlap_proxy,
        overlap_proxy_index=pd.DataFrame({"strategy_instance_pk": ["s1"], "overlap_proxy_idx": [0]}),
        exact_overlap=pd.DataFrame({"portfolio_pk_a": ["p1"], "portfolio_pk_b": ["p1"], "overlap": [1.0]}),
        global_frontier=pd.DataFrame({"portfolio_pk": ["p1"], "final_score": [0.4]}),
        risk_return_frontier=pd.DataFrame({"portfolio_pk": ["p1"], "minute_annualized_return": [0.1]}),
        operational_frontier=pd.DataFrame({"portfolio_pk": ["p1"], "availability_burden": [0.0]}),
        selected_frontier=pd.DataFrame({"portfolio_pk": ["p1"], "final_score": [0.4]}),
        selection_report=report,
    )

    dep_dir = output_dir / "dependence" / "u0"
    asset_order = np.load(dep_dir / "asset_column_indices.npy")
    pair_reason_codes = np.load(dep_dir / "pair_completion_reason_codes.npy")
    metadata = json.loads((dep_dir / "metadata.json").read_text(encoding="utf-8"))

    assert dep_dir.joinpath("asset_column_indices.npy").exists()
    assert dep_dir.joinpath("pair_completion_reason_codes.npy").exists()
    assert asset_order.tolist() == [11, 17]
    assert pair_reason_codes.shape == (2, 2)
    assert pair_reason_codes[0, 1] == "PAIR_OVERLAP_ZERO"
    assert metadata["asset_column_indices"] == [11, 17]
    assert metadata["pair_completion_reason_codes_artifact"] == "pair_completion_reason_codes.npy"
