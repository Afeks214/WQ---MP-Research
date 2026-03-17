from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError
import yaml

from weightiz.cli.run_research import _load_config
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


def test_run_config_accepts_recovered_harness_fields_but_still_rejects_unknown_extras() -> None:
    cfg = RunConfigModel.model_validate(
        {
            "symbols": ["SPY", "QQQ"],
            "harness": {
                "disable_cpcv_splits": True,
                "execution_finra_taf_per_share_sell": 0.000195,
                "execution_sec31_rate_per_million": 15.0,
            },
        }
    )
    assert bool(cfg.harness.disable_cpcv_splits) is True
    assert float(cfg.harness.execution_finra_taf_per_share_sell) == 0.000195
    assert float(cfg.harness.execution_sec31_rate_per_million) == 15.0

    try:
        RunConfigModel.model_validate(
            {
                "symbols": ["SPY", "QQQ"],
                "harness": {
                    "disable_cpcv_splits": True,
                    "totally_unknown_field": 1,
                },
            }
        )
    except ValidationError as exc:
        assert "Extra inputs are not permitted" in str(exc)
    else:
        raise AssertionError("unknown harness extras must remain forbidden")


def test_run_research_loader_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    path = tmp_path / "dup.yaml"
    path.write_text(
        "symbols: [SPY, QQQ]\n"
        "harness:\n"
        "  parallel_workers: 1\n"
        "harness:\n"
        "  parallel_workers: 2\n",
        encoding="utf-8",
    )

    try:
        _load_config(path)
    except RuntimeError as exc:
        assert "duplicate YAML key: harness" == str(exc)
    else:
        raise AssertionError("duplicate YAML keys must fail closed")


def test_recovered_minimal_mac_m4_config_is_duplicate_free_and_validates() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "run_minimal_e2e_mac_m4.yaml"
    raw = _load_unique_yaml(path)
    cfg = RunConfigModel.model_validate(raw)

    assert cfg.symbols == ["SPY", "QQQ"]
    assert len(cfg.module4_configs) == 14
    assert bool(cfg.harness.disable_cpcv_splits) is True
    assert int(cfg.harness.parallel_workers) == 3
    assert float(cfg.module6["intake"]["min_availability_ratio"]) == 0.20
    assert int(cfg.module6["intake"]["min_observed_sessions"]) == 30
    assert float(cfg.module6["intake"]["required_comparison_support"]) == 0.80
    assert float(cfg.module6["scoring"]["min_cross_universe_support"]) == 0.80
