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
        correlation=np.eye(1, dtype=np.float64),
        downside_covariance=np.eye(1, dtype=np.float64),
        regime_overlap=np.eye(1, dtype=np.float64),
        common_support=np.ones((1, 1), dtype=bool),
        drawdown_concurrence=sparse.eye(1, format="csr"),
        shrinkage=np.inf,
        negative_mass=np.nan,
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
    json.loads(metadata_text)
