from __future__ import annotations

from pathlib import Path

import yaml

from weightiz.shared.config.models import RunConfigModel


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise RuntimeError(f"duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def _load_unique_yaml(path: Path) -> dict[str, object]:
    raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    assert isinstance(raw, dict)
    return raw


def test_e2e_smoke_config_is_duplicate_free_and_explicitly_smoke_only() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "e2e_test.yaml"
    raw = _load_unique_yaml(path)
    RunConfigModel.model_validate(raw)

    assert str(raw["run_name"]).startswith("e2e_smoke")
    assert float(raw["harness"]["robustness_reject_threshold"]) == 0.0
    scenarios = list(raw["stress_scenarios"])
    assert len(scenarios) == 1
    assert str(scenarios[0]["scenario_id"]) == "baseline"


def test_e2e_proving_config_is_duplicate_free_and_keeps_reject_gate_enabled() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "e2e_proving.yaml"
    raw = _load_unique_yaml(path)
    RunConfigModel.model_validate(raw)

    assert str(raw["run_name"]).startswith("e2e_proving")
    assert float(raw["harness"]["robustness_reject_threshold"]) > 0.0
    assert "stress_scenarios" not in raw
