#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CFG = REPO_ROOT / "configs" / "sweep_20x100.yaml"
DEFAULT_SOURCE_CFG = REPO_ROOT / "configs" / "sweep_20x432.yaml"
OUT_DIR = REPO_ROOT / "configs" / "_generated"
TARGET_TOTAL = 432
MIN_TOTAL = 300
MAX_TOTAL = 600

FAMILY_W_TARGETS: dict[str, set[int]] = {
    "sprinters": {5, 10, 15, 20},
    "surfers": {30, 60, 90, 120},
    "snipers": {10, 30, 60},
    "marathoners": {240, 360, 480},
}

FAMILY_MIN_W_COUNT: dict[str, int] = {
    "sprinters": 2,
    "surfers": 2,
    "snipers": 1,
    "marathoners": 2,
}

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import run_research


def _require_mapping(x: Any, name: str) -> dict[str, Any]:
    if not isinstance(x, dict):
        raise RuntimeError(f"{name} must be a mapping/object")
    return x


def _require_list(x: Any, name: str) -> list[Any]:
    if not isinstance(x, list) or len(x) == 0:
        raise RuntimeError(f"{name} must be a non-empty list")
    return x


def _stable_sort_values(values: set[Any]) -> list[Any]:
    return sorted(values, key=lambda x: (float(x), str(x)))


def _load_yaml(path: Path) -> dict[str, Any]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise RuntimeError(f"Invalid YAML root in {path}: expected mapping")
    return doc


def _extract_source_pools(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        raise RuntimeError(f"Source config does not exist: {path}")

    doc = _load_yaml(path)
    w_pool: set[int] = set()
    t_pool: set[float] = set()
    a_pool: set[float] = set()
    b_pool: set[float] = set()
    topk_pool: set[int] = set()

    m2 = doc.get("module2_configs", [])
    if isinstance(m2, list):
        for row in m2:
            if isinstance(row, dict) and "profile_window_bars" in row:
                w_pool.add(int(row["profile_window_bars"]))

    m4 = doc.get("module4_configs", [])
    if isinstance(m4, list):
        for row in m4:
            if not isinstance(row, dict):
                continue
            if "entry_threshold" in row:
                t_pool.add(float(row["entry_threshold"]))
            if "trend_poc_drift_min_abs" in row:
                a_pool.add(float(row["trend_poc_drift_min_abs"]))
            if "neutral_poc_drift_max_abs" in row:
                b_pool.add(float(row["neutral_poc_drift_max_abs"]))
            if "top_k_intraday" in row:
                topk_pool.add(int(row["top_k_intraday"]))

    if not w_pool:
        raise RuntimeError("No W pool found (module2.profile_window_bars) in source config")
    if not t_pool:
        raise RuntimeError("No T pool found (module4.entry_threshold) in source config")
    if not a_pool:
        raise RuntimeError("No A pool found (module4.trend_poc_drift_min_abs) in source config")
    if not b_pool:
        raise RuntimeError("No B pool found (module4.neutral_poc_drift_max_abs) in source config")
    if not topk_pool:
        raise RuntimeError("No K pool found (module4.top_k_intraday) in source config")

    return {
        "W": _stable_sort_values(w_pool),
        "T": _stable_sort_values(t_pool),
        "A": _stable_sort_values(a_pool),
        "B": _stable_sort_values(b_pool),
        "TOPK": _stable_sort_values(topk_pool),
    }


def _dedupe_lists(items: list[list[Any]]) -> list[list[Any]]:
    out: list[list[Any]] = []
    seen: set[str] = set()
    for item in items:
        key = json.dumps(item, sort_keys=False, separators=(",", ":"))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _prefix_candidates(values: list[Any]) -> list[list[Any]]:
    if not values:
        return []
    return _dedupe_lists([list(values[:k]) for k in range(1, len(values) + 1)])


def _suffix_candidates(values: list[Any]) -> list[list[Any]]:
    if not values:
        return []
    return _dedupe_lists([list(values[-k:]) for k in range(1, len(values) + 1)])


def _semantic_t_candidates(family: str, t_pool: list[float]) -> list[list[float]]:
    n = len(t_pool)
    if n == 0:
        raise RuntimeError(f"{family}: empty T pool")

    if family == "sprinters":
        # Lower-percentile only: prefix subsets up to lower 75% of the pool.
        max_k = max(1, int(math.ceil(0.75 * n)))
        return _dedupe_lists([list(t_pool[:k]) for k in range(1, max_k + 1)])

    if family == "surfers":
        # Mid/high only: suffix subsets beginning from the mid/high index.
        start = max(0, (n // 2) - 1)
        return _dedupe_lists([list(t_pool[s:]) for s in range(start, n)])

    if family == "snipers":
        # Strict top-tail only: suffix subsets limited to upper 50% tail.
        max_k = max(1, int(math.ceil(0.5 * n)))
        return _dedupe_lists([list(t_pool[-k:]) for k in range(1, max_k + 1)])

    if family == "marathoners":
        # High-threshold only: suffix subsets in upper 75% zone.
        max_k = max(1, int(math.ceil(0.75 * n)))
        return _dedupe_lists([list(t_pool[-k:]) for k in range(1, max_k + 1)])

    raise RuntimeError(f"Unsupported family for T semantics: {family}")


def _build_initial_family_axes(pools: dict[str, Any], m3_baseline_pool: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    w_pool = sorted(int(x) for x in pools["W"])
    t_pool = [float(x) for x in pools["T"]]
    a_pool = [float(x) for x in pools["A"]]
    b_pool = [float(x) for x in pools["B"]]
    k_pool = [int(x) for x in pools["TOPK"]]

    out: dict[str, dict[str, Any]] = {}
    for family in ("sprinters", "surfers", "snipers", "marathoners"):
        routed_w = [w for w in w_pool if w in FAMILY_W_TARGETS[family]]
        if len(routed_w) == 0:
            raise RuntimeError(
                f"{family}: routed W set is empty. pool_W={w_pool}, target_W={sorted(FAMILY_W_TARGETS[family])}"
            )

        t_cands = _semantic_t_candidates(family, t_pool)
        if len(t_cands) == 0:
            raise RuntimeError(f"{family}: no semantic T candidates generated")

        out[family] = {
            "W": list(routed_w),
            "T": copy.deepcopy(t_cands[-1]),
            "A": list(a_pool),
            "B": list(b_pool),
            "TOPK": list(k_pool),
            "M3_POOL": [copy.deepcopy(x) for x in m3_baseline_pool],
            "T_CANDS": [copy.deepcopy(x) for x in t_cands],
            "A_CANDS": _prefix_candidates(list(a_pool)),
            "B_CANDS": _prefix_candidates(list(b_pool)),
            "K_CANDS": _prefix_candidates(list(k_pool)),
            "W_TARGET": sorted(FAMILY_W_TARGETS[family]),
            "W_MIN": int(FAMILY_MIN_W_COUNT[family]),
        }
    return out


def _count_m4(t_values: list[float], a_values: list[float], b_values: list[float], k_values: list[int]) -> int:
    return int(len(t_values) * len(a_values) * len(b_values) * len(k_values))


def _best_total_choice(candidates: list[dict[str, Any]], *, target_total: int, min_total: int, max_total: int) -> dict[str, Any]:
    if not candidates:
        raise RuntimeError("No total candidates provided")

    in_band = [c for c in candidates if min_total <= int(c["total"]) <= max_total]
    if in_band:
        in_band.sort(key=lambda c: (abs(int(c["total"]) - target_total), int(c["total"]), str(c["key"])))
        return in_band[0]

    candidates.sort(key=lambda c: (abs(int(c["total"]) - target_total), int(c["total"]), str(c["key"])))
    return candidates[0]


def _tune_family(
    *,
    family: str,
    spec: dict[str, Any],
    target_total: int,
    min_total: int,
    max_total: int,
) -> dict[str, Any]:
    tuned = {
        "W": list(int(x) for x in spec["W"]),
        "T": list(float(x) for x in spec["T"]),
        "A": list(float(x) for x in spec["A"]),
        "B": list(float(x) for x in spec["B"]),
        "TOPK": list(int(x) for x in spec["TOPK"]),
        "M3_POOL": [copy.deepcopy(x) for x in spec["M3_POOL"]],
    }
    logs: list[str] = []
    reachable_totals: set[int] = set()

    # Step 1: tune M3 count first (project totals before applying).
    m4_count = _count_m4(tuned["T"], tuned["A"], tuned["B"], tuned["TOPK"])
    step1_candidates: list[dict[str, Any]] = []
    for n3 in range(1, len(tuned["M3_POOL"]) + 1):
        total = int(len(tuned["W"]) * n3 * m4_count)
        reachable_totals.add(total)
        step1_candidates.append({"key": f"M3={n3}", "m3": n3, "total": total})
    chosen_step1 = _best_total_choice(step1_candidates, target_total=target_total, min_total=min_total, max_total=max_total)
    m3_count = int(chosen_step1["m3"])
    total = int(chosen_step1["total"])
    logs.append(
        f"STEP1_M3_PROJECT family={family} candidates={[(c['m3'], c['total']) for c in step1_candidates]} chosen_m3={m3_count} total={total}"
    )
    fallback_triggered = False
    fallback_reason = ""

    if min_total <= total <= max_total:
        m3_values = [copy.deepcopy(tuned["M3_POOL"][i]) for i in range(m3_count)]
        return {
            "W": tuned["W"],
            "T": tuned["T"],
            "A": tuned["A"],
            "B": tuned["B"],
            "TOPK": tuned["TOPK"],
            "M3": m3_values,
            "TOTAL": total,
            "FALLBACK": fallback_triggered,
            "FALLBACK_REASON": fallback_reason,
            "LOGS": logs,
        }

    # Step 2: W reduction only (respect minimum W count).
    step2_candidates: list[dict[str, Any]] = []
    min_w = int(spec["W_MIN"])
    w_values = list(int(x) for x in tuned["W"])
    for keep_n in range(len(w_values), min_w - 1, -1):
        w_subset = list(w_values[:keep_n])
        total_w = int(len(w_subset) * m3_count * m4_count)
        reachable_totals.add(total_w)
        step2_candidates.append({"key": f"W={w_subset}", "w": w_subset, "total": total_w})
    chosen_step2 = _best_total_choice(step2_candidates, target_total=target_total, min_total=min_total, max_total=max_total)
    if chosen_step2["w"] != tuned["W"]:
        logs.append(
            f"STEP2_W_REDUCTION family={family} from_W={tuned['W']} to_W={chosen_step2['w']} projected_total={chosen_step2['total']}"
        )
        tuned["W"] = list(int(x) for x in chosen_step2["w"])
    total = int(len(tuned["W"]) * m3_count * m4_count)
    logs.append(f"STEP2_W_PROJECT family={family} total={total}")

    if min_total <= total <= max_total:
        m3_values = [copy.deepcopy(tuned["M3_POOL"][i]) for i in range(m3_count)]
        return {
            "W": tuned["W"],
            "T": tuned["T"],
            "A": tuned["A"],
            "B": tuned["B"],
            "TOPK": tuned["TOPK"],
            "M3": m3_values,
            "TOTAL": total,
            "FALLBACK": fallback_triggered,
            "FALLBACK_REASON": fallback_reason,
            "LOGS": logs,
        }

    # Step 3: M4 size control by deterministic slicing of existing pools.
    # T candidates are family-semantic constrained; A/B/K candidates are deterministic prefixes.
    step3_candidates: list[dict[str, Any]] = []
    for t_vals in [copy.deepcopy(x) for x in spec["T_CANDS"]]:
        for a_vals in [list(float(v) for v in x) for x in spec["A_CANDS"]]:
            for b_vals in [list(float(v) for v in x) for x in spec["B_CANDS"]]:
                for k_vals in [list(int(v) for v in x) for x in spec["K_CANDS"]]:
                    m4_n = _count_m4(
                        [float(v) for v in t_vals],
                        [float(v) for v in a_vals],
                        [float(v) for v in b_vals],
                        [int(v) for v in k_vals],
                    )
                    tot = int(len(tuned["W"]) * m3_count * m4_n)
                    reachable_totals.add(tot)
                    step3_candidates.append(
                        {
                            "key": (
                                tuple(t_vals),
                                tuple(a_vals),
                                tuple(b_vals),
                                tuple(k_vals),
                            ),
                            "T": [float(v) for v in t_vals],
                            "A": [float(v) for v in a_vals],
                            "B": [float(v) for v in b_vals],
                            "TOPK": [int(v) for v in k_vals],
                            "total": tot,
                            "m4": m4_n,
                        }
                    )

    chosen_step3 = _best_total_choice(step3_candidates, target_total=target_total, min_total=min_total, max_total=max_total)
    tuned["T"] = [float(v) for v in chosen_step3["T"]]
    tuned["A"] = [float(v) for v in chosen_step3["A"]]
    tuned["B"] = [float(v) for v in chosen_step3["B"]]
    tuned["TOPK"] = [int(v) for v in chosen_step3["TOPK"]]
    total = int(chosen_step3["total"])
    logs.append(
        f"STEP3_M4_SLICE family={family} T={tuned['T']} A={tuned['A']} B={tuned['B']} K={tuned['TOPK']} total={total}"
    )

    if not (min_total <= total <= max_total):
        fallback_triggered = True
        rmin = min(reachable_totals) if reachable_totals else total
        rmax = max(reachable_totals) if reachable_totals else total
        fallback_reason = (
            f"FALLBACK TRIGGERED: reachable TOTAL range under constraints is [{rmin}, {rmax}], "
            f"selected closest TOTAL={total} to target={target_total} without inventing values."
        )
        logs.append(fallback_reason)

    m3_values = [copy.deepcopy(tuned["M3_POOL"][i]) for i in range(m3_count)]
    return {
        "W": tuned["W"],
        "T": tuned["T"],
        "A": tuned["A"],
        "B": tuned["B"],
        "TOPK": tuned["TOPK"],
        "M3": m3_values,
        "TOTAL": total,
        "FALLBACK": fallback_triggered,
        "FALLBACK_REASON": fallback_reason,
        "LOGS": logs,
    }


def _build_m2_list(m2_base: dict[str, Any], w_values: list[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for w in w_values:
        row = copy.deepcopy(m2_base)
        row["profile_window_bars"] = int(w)
        out.append(row)
    return out


def _build_m4_list(
    m4_base: dict[str, Any],
    t_values: list[float],
    a_values: list[float],
    b_values: list[float],
    topk_values: list[int],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t, a, b, topk in itertools.product(t_values, a_values, b_values, topk_values):
        row = copy.deepcopy(m4_base)
        row["entry_threshold"] = float(t)
        row["trend_poc_drift_min_abs"] = float(a)
        row["neutral_poc_drift_max_abs"] = float(b)
        row["top_k_intraday"] = int(topk)
        out.append(row)
    return out


def _dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _proof_unique(rows: list[dict[str, Any]]) -> int:
    hashes = {json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=False) for r in rows}
    return int(len(hashes))


def _generate_family_cfg(
    base: dict[str, Any],
    family_name: str,
    tuned: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    out = copy.deepcopy(base)
    out["run_name"] = f"sweep_family_{family_name}"

    harness = _require_mapping(out.get("harness", {}), "harness")
    harness["parallel_backend"] = "process_pool"
    harness["parallel_workers"] = 14
    out["harness"] = harness

    m2_base = copy.deepcopy(
        _require_mapping(_require_list(base.get("module2_configs"), "module2_configs")[0], "module2_configs[0]")
    )
    m4_base = copy.deepcopy(
        _require_mapping(_require_list(base.get("module4_configs"), "module4_configs")[0], "module4_configs[0]")
    )

    m2_list = _build_m2_list(m2_base=m2_base, w_values=[int(x) for x in tuned["W"]])
    m3_list = [copy.deepcopy(x) for x in tuned["M3"]]
    m4_list = _build_m4_list(
        m4_base=m4_base,
        t_values=[float(x) for x in tuned["T"]],
        a_values=[float(x) for x in tuned["A"]],
        b_values=[float(x) for x in tuned["B"]],
        topk_values=[int(x) for x in tuned["TOPK"]],
    )

    out["module2_configs"] = m2_list
    out["module3_configs"] = m3_list
    out["module4_configs"] = m4_list

    counts = {
        "M2": len(m2_list),
        "M3": len(m3_list),
        "M4": len(m4_list),
        "TOTAL": int(len(m2_list) * len(m3_list) * len(m4_list)),
        "M4_UNIQUE": _proof_unique(m4_list),
    }
    return out, counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build family sweep YAMLs from authorized source pools")
    parser.add_argument(
        "--source-config",
        default=str(DEFAULT_SOURCE_CFG),
        help="Authorized source-of-truth config path for W/T/A/B/K pools",
    )
    args = parser.parse_args()

    if not BASELINE_CFG.exists():
        raise SystemExit(f"Missing baseline config: {BASELINE_CFG}")

    source_cfg = Path(args.source_config).expanduser().resolve()
    if source_cfg != DEFAULT_SOURCE_CFG.resolve():
        raise SystemExit(
            f"Fail-closed: source-config must be exactly {DEFAULT_SOURCE_CFG.resolve()}, got {source_cfg}"
        )

    base = _require_mapping(_load_yaml(BASELINE_CFG), "baseline root")
    baseline_m3_pool = [
        copy.deepcopy(_require_mapping(x, f"module3_configs[{i}]"))
        for i, x in enumerate(_require_list(base.get("module3_configs"), "module3_configs"))
    ]

    pools = _extract_source_pools(source_cfg)
    families = _build_initial_family_axes(pools, baseline_m3_pool)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(
        "SOURCE_CONFIG="
        f"{source_cfg} "
        "SOURCE_POOLS "
        f"W={pools['W']} "
        f"T={pools['T']} "
        f"A={pools['A']} "
        f"B={pools['B']} "
        f"TOPK={pools['TOPK']} "
        f"M3_BASELINE_POOL={len(baseline_m3_pool)}"
    )

    for family in ("sprinters", "surfers", "snipers", "marathoners"):
        spec = families[family]
        print(
            f"{family}: ROUTED_W={sorted(int(x) for x in spec['W'])} "
            f"TARGET_W={spec['W_TARGET']} "
            f"T_SEMANTIC_RULE="
            + (
                "lower-percentile prefix"
                if family == "sprinters"
                else "mid/high suffix"
                if family == "surfers"
                else "strict top-tail suffix"
                if family == "snipers"
                else "high-threshold suffix"
            )
        )

        tuned = _tune_family(
            family=family,
            spec=spec,
            target_total=TARGET_TOTAL,
            min_total=MIN_TOTAL,
            max_total=MAX_TOTAL,
        )

        cfg, counts = _generate_family_cfg(base=base, family_name=family, tuned=tuned)
        out_path = OUT_DIR / f"sweep_family_{family}.yaml"
        _dump_yaml(out_path, cfg)
        run_research._load_config(out_path)

        for line in tuned["LOGS"]:
            print(line)
        print(
            f"{family}: FINAL "
            f"W={sorted(int(x) for x in tuned['W'])} "
            f"T={list(float(x) for x in tuned['T'])} "
            f"A={list(float(x) for x in tuned['A'])} "
            f"B={list(float(x) for x in tuned['B'])} "
            f"K={list(int(x) for x in tuned['TOPK'])} "
            f"M2={counts['M2']} M3={counts['M3']} M4={counts['M4']} TOTAL={counts['TOTAL']} "
            f"M4_UNIQUE={counts['M4_UNIQUE']} FALLBACK={bool(tuned['FALLBACK'])} "
            f"PATH={out_path}"
        )
        if tuned["FALLBACK_REASON"]:
            print(tuned["FALLBACK_REASON"])


if __name__ == "__main__":
    main()
