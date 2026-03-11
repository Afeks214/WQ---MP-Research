from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Iterable


STAGE_A_CAMPAIGN_ID = "stage_a_cloud_20260310"
STAGE_A_RUN_NAME = "cloud_stage_a_discovery_5000"
STAGE_A_RESEARCH_MODE = "discovery"
STAGE_A_RESEARCH_THRESHOLD = 0.40
STAGE_A_LIVE_ENTRY_THRESHOLD = 0.30
STAGE_A_DATA_START_UTC = "2024-09-10T00:00:00Z"
STAGE_A_DATA_END_UTC = "2026-03-10T23:59:59Z"
STAGE_A_SYMBOLS: tuple[str, ...] = (
    "SPY",
    "QQQ",
    "GLD",
    "EEM",
    "HYG",
    "IWM",
    "XLU",
    "XLK",
    "XLE",
    "TLT",
)
STAGE_A_WINDOW_SET: tuple[int, ...] = (15, 20, 30, 40, 60, 90, 240)
STAGE_A_F6_WINDOW_SET: tuple[int, ...] = (30, 60, 90, 240)
STAGE_A_WINDOW_INDEX: dict[int, int] = {int(w): int(i) for i, w in enumerate(STAGE_A_WINDOW_SET)}
STAGE_A_PROCESS_BACKEND = "process_pool"
STAGE_A_PROCESS_WORKERS = 48
STAGE_A_TOTAL_CANDIDATES = 5000


@dataclass(frozen=True)
class StageAFamilySpec:
    family_id: str
    family_name: str
    hypothesis: str
    candidate_budget: int
    hypothesis_count: int
    window_set: tuple[int, ...]
    live_axes: tuple[str, ...]
    live_role: str
    restricted_window_subset_justification: str = ""


STAGE_A_FAMILY_SPECS: tuple[StageAFamilySpec, ...] = (
    StageAFamilySpec(
        family_id="F1",
        family_name="acceptance_rejection_geometry",
        hypothesis="Acceptance / rejection geometry around POC, VAH, VAL and value width.",
        candidate_budget=840,
        hypothesis_count=120,
        window_set=STAGE_A_WINDOW_SET,
        live_axes=(
            "module4.exit_threshold",
            "module4.trend_spread_min",
            "module4.trend_poc_drift_min_abs",
        ),
        live_role="window_probe",
    ),
    StageAFamilySpec(
        family_id="F2",
        family_name="delta_confirmation_aggression",
        hypothesis="Deviation only matters when aggression confirms or denies the move.",
        candidate_budget=840,
        hypothesis_count=120,
        window_set=STAGE_A_WINDOW_SET,
        live_axes=(
            "module4.exit_threshold",
            "module4.regime_confidence_min",
            "module4.conviction_scale",
        ),
        live_role="window_probe",
    ),
    StageAFamilySpec(
        family_id="F3",
        family_name="participation_rvol_conviction",
        hypothesis="The same geometry means something different under low and high participation.",
        candidate_budget=840,
        hypothesis_count=120,
        window_set=STAGE_A_WINDOW_SET,
        live_axes=(
            "module4.exit_threshold",
            "module4.regime_confidence_min",
            "module4.conviction_scale",
        ),
        live_role="window_probe",
    ),
    StageAFamilySpec(
        family_id="F4",
        family_name="session_state_initial_balance",
        hypothesis="Profile behavior changes around warmup, IB definition and late-session timing.",
        candidate_budget=840,
        hypothesis_count=120,
        window_set=STAGE_A_WINDOW_SET,
        live_axes=("module3.ib_pop_frac", "module3.min_block_valid_ratio", "module3.rolling_context_period"),
        live_role="window_probe",
    ),
    StageAFamilySpec(
        family_id="F5",
        family_name="multi_scale_alignment",
        hypothesis="Alignment or conflict across short, medium and long profile states carries alpha.",
        candidate_budget=840,
        hypothesis_count=105,
        window_set=STAGE_A_WINDOW_SET,
        live_axes=("module4.anchor_window_index", "module4.regime_confidence_min", "module4.conviction_scale"),
        live_role="probe_plus_multi_window_live",
    ),
    StageAFamilySpec(
        family_id="F6",
        family_name="shape_fingerprint_regimes",
        hypothesis="Profile shape acts as a state machine beyond raw price location.",
        candidate_budget=800,
        hypothesis_count=200,
        window_set=STAGE_A_F6_WINDOW_SET,
        live_axes=(
            "module4.shape_skew_min_abs",
            "module4.double_dist_sep_x",
            "module4.double_dist_valley_frac",
            "module4.regime_confidence_min",
        ),
        live_role="window_probe",
        restricted_window_subset_justification=(
            "F6 is intentionally restricted to 30/60/90/240 minutes because the shape hypothesis "
            "needs medium-to-long profile structure rather than the shortest 15/20/40 probes."
        ),
    ),
)


def stable_stage_a_hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(blob.encode("ascii")).hexdigest()


def encode_stage_a_tags(meta: dict[str, Any]) -> tuple[str, ...]:
    out: list[str] = []
    for key in sorted(meta.keys()):
        value = meta[key]
        if isinstance(value, (list, tuple)):
            rendered = ",".join(str(x) for x in value)
        elif isinstance(value, bool):
            rendered = "true" if value else "false"
        else:
            rendered = str(value)
        out.append(f"{key}={rendered}")
    return tuple(out)


def parse_stage_a_tags(tags: Iterable[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in tags:
        tag = str(raw).strip()
        if not tag:
            continue
        if "=" not in tag:
            out[tag] = "true"
            continue
        key, value = tag.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def parse_stage_a_window_set(raw: str | None) -> tuple[int, ...]:
    if raw is None:
        return ()
    out: list[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return tuple(out)


def stage_a_family_specs_by_id() -> dict[str, StageAFamilySpec]:
    return {spec.family_id: spec for spec in STAGE_A_FAMILY_SPECS}
