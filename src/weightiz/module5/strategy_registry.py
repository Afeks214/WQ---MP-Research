from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from weightiz.module5.stage_a_discovery import stage_a_family_specs_by_id

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise RuntimeError("pyyaml is required for strategy registry loading") from exc


_VALID_SENSITIVITY = {"low", "medium", "high"}


@dataclass(frozen=True)
class StrategyWhyRecord:
    strategy_family_id: str
    economic_mechanism: str
    expected_edge_source: str
    expected_market_conditions: str
    # Narrative metadata only. These strings are not bound to a runtime kill switch.
    expected_kill_conditions: str
    liquidity_sensitivity: str
    cost_sensitivity: str
    # Expert metadata prior used for ranking/context, not an empirical posterior estimate.
    confidence_prior: float

    def __post_init__(self) -> None:
        for field_name in (
            "strategy_family_id",
            "economic_mechanism",
            "expected_edge_source",
            "expected_market_conditions",
            "expected_kill_conditions",
        ):
            if str(getattr(self, field_name)).strip() == "":
                raise RuntimeError(f"strategy registry field must be non-empty: {field_name}")
        if str(self.liquidity_sensitivity).strip() not in _VALID_SENSITIVITY:
            raise RuntimeError("strategy registry liquidity_sensitivity must be one of: low, medium, high")
        if str(self.cost_sensitivity).strip() not in _VALID_SENSITIVITY:
            raise RuntimeError("strategy registry cost_sensitivity must be one of: low, medium, high")
        if not (0.0 <= float(self.confidence_prior) <= 1.0):
            raise RuntimeError("strategy registry confidence_prior must be in [0,1]")


def load_strategy_registry(path: str | Path) -> dict[str, StrategyWhyRecord]:
    registry_path = Path(path).resolve()
    if not registry_path.exists():
        raise RuntimeError(f"strategy registry file missing: {registry_path}")
    raw = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("strategy registry root must be a mapping")
    rows = raw.get("strategies")
    if not isinstance(rows, list):
        raise RuntimeError("strategy registry must contain a 'strategies' list")
    out: dict[str, StrategyWhyRecord] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("strategy registry entries must be mappings")
        record = StrategyWhyRecord(
            strategy_family_id=str(row.get("strategy_family_id", "")),
            economic_mechanism=str(row.get("economic_mechanism", "")),
            expected_edge_source=str(row.get("expected_edge_source", "")),
            expected_market_conditions=str(row.get("expected_market_conditions", "")),
            expected_kill_conditions=str(row.get("expected_kill_conditions", "")),
            liquidity_sensitivity=str(row.get("liquidity_sensitivity", "")),
            cost_sensitivity=str(row.get("cost_sensitivity", "")),
            confidence_prior=float(row.get("confidence_prior", 0.0)),
        )
        family_id = str(record.strategy_family_id)
        if family_id in out:
            raise RuntimeError(f"duplicate strategy registry family id: {family_id}")
        out[family_id] = record
    return out


def validate_strategy_registry(
    registry: dict[str, StrategyWhyRecord],
    *,
    required_family_ids: set[str] | None = None,
) -> None:
    if not isinstance(registry, dict) or not registry:
        raise RuntimeError("strategy registry must be a non-empty mapping")
    for family_id, record in registry.items():
        if str(family_id).strip() == "":
            raise RuntimeError("strategy registry family id must be non-empty")
        if str(record.strategy_family_id) != str(family_id):
            raise RuntimeError(
                f"strategy registry key mismatch: key={family_id!r} record={record.strategy_family_id!r}"
            )
    required = set(required_family_ids or set(stage_a_family_specs_by_id().keys()))
    missing = sorted(required - set(registry.keys()))
    if missing:
        raise RuntimeError(f"strategy registry missing required family ids: {missing}")


def strategy_registry_hash(registry: dict[str, StrategyWhyRecord]) -> str:
    payload = {
        key: asdict(record)
        for key, record in sorted(registry.items(), key=lambda item: str(item[0]))
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    import hashlib

    return hashlib.sha256(blob.encode("ascii")).hexdigest()[:16]


def strategy_why_dict(record: StrategyWhyRecord | None) -> dict[str, Any] | None:
    return asdict(record) if record is not None else None
