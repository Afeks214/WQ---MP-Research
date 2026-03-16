from __future__ import annotations

import textwrap

import pytest
import yaml

from weightiz.module5.stage_a_discovery import stage_a_family_specs_by_id
from weightiz.module5.strategy_registry import load_strategy_registry, strategy_registry_hash, validate_strategy_registry


def test_live_strategy_registry_covers_all_active_stage_a_family_ids() -> None:
    path = __file__
    registry_path = (
        __import__("pathlib").Path(path).resolve().parents[1] / "configs" / "strategy_registry.yaml"
    )
    registry = load_strategy_registry(registry_path)

    validate_strategy_registry(registry, required_family_ids=set(stage_a_family_specs_by_id().keys()))
    assert set(stage_a_family_specs_by_id().keys()).issubset(set(registry.keys()))
    assert len(strategy_registry_hash(registry)) == 16


def test_live_strategy_registry_declares_metadata_only_semantics_for_priors_and_kill_conditions() -> None:
    registry_path = __import__("pathlib").Path(__file__).resolve().parents[1] / "configs" / "strategy_registry.yaml"
    raw = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    assert raw["registry_contract"]["confidence_prior_semantics"] == "expert_metadata_prior_not_empirical"
    assert raw["registry_contract"]["kill_conditions_semantics"] == "metadata_only_not_runtime_bound"


def test_strategy_registry_fails_closed_when_required_family_is_missing(tmp_path) -> None:
    path = tmp_path / "strategy_registry.yaml"
    path.write_text(
        textwrap.dedent(
            """
            strategies:
              - strategy_family_id: F1
                economic_mechanism: one
                expected_edge_source: two
                expected_market_conditions: three
                expected_kill_conditions: four
                liquidity_sensitivity: medium
                cost_sensitivity: medium
                confidence_prior: 0.5
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    registry = load_strategy_registry(path)
    with pytest.raises(RuntimeError, match="missing required family ids"):
        validate_strategy_registry(registry, required_family_ids={"F1", "F2"})


def test_strategy_registry_rejects_invalid_enums_and_confidence(tmp_path) -> None:
    path = tmp_path / "strategy_registry.yaml"
    path.write_text(
        textwrap.dedent(
            """
            strategies:
              - strategy_family_id: F1
                economic_mechanism: one
                expected_edge_source: two
                expected_market_conditions: three
                expected_kill_conditions: four
                liquidity_sensitivity: broken
                cost_sensitivity: medium
                confidence_prior: 1.5
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="liquidity_sensitivity"):
        load_strategy_registry(path)
