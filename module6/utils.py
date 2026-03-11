from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


class Module6ValidationError(RuntimeError):
    pass


def stable_sha256_parts(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_json_hash(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_columns(df: Any, required: Iterable[str], artifact_name: str) -> None:
    missing = [str(col) for col in required if str(col) not in df.columns]
    if missing:
        raise Module6ValidationError(
            f"{artifact_name} missing required columns: {', '.join(sorted(missing))}"
        )


def assert_no_duplicates(df: Any, cols: list[str], artifact_name: str) -> None:
    dup_mask = df.duplicated(cols, keep=False)
    if bool(np.any(np.asarray(dup_mask, dtype=bool))):
        sample = df.loc[dup_mask, cols].head(5).to_dict("records")
        raise Module6ValidationError(
            f"{artifact_name} contains duplicate keys for {cols}; sample={sample}"
        )


def count_flag_tokens(serialized_flags: str) -> int:
    txt = str(serialized_flags).strip()
    if not txt:
        return 0
    return int(len([tok for tok in txt.split("|") if tok.strip()]))


def normalized_rank(values: np.ndarray, ascending: bool = True) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size <= 0:
        return np.zeros(0, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    if not ascending:
        order = order[::-1]
    ranks = np.empty(arr.size, dtype=np.float64)
    ranks[order] = np.arange(arr.size, dtype=np.float64)
    if arr.size == 1:
        return np.ones(1, dtype=np.float64)
    return 1.0 - ranks / float(arr.size - 1)


def safe_divide(numer: np.ndarray | float, denom: np.ndarray | float, eps: float = 1.0e-12) -> np.ndarray:
    n = np.asarray(numer, dtype=np.float64)
    d = np.asarray(denom, dtype=np.float64)
    return n / np.maximum(d, float(eps))


def state_code_to_bool(code: int) -> bool:
    return int(code) in (1, 2)


def target_weights_hash(weights: dict[str, float]) -> str:
    ordered = sorted((str(k), round(float(v), 12)) for k, v in weights.items() if abs(float(v)) > 0.0)
    return stable_sha256_parts(*[f"{k}:{v:.12f}" for k, v in ordered])


def portfolio_pk(
    *,
    reduced_universe_id: str,
    generator_family: str,
    rebalance_policy: str,
    target_weights_hash_value: str,
    cash_policy: str,
    constraint_policy_version: str,
    ranking_policy_version: str,
    overnight_policy_version: str,
    friction_policy_version: str,
    support_policy_version: str,
    calendar_version: str,
) -> str:
    return stable_sha256_parts(
        reduced_universe_id,
        generator_family,
        rebalance_policy,
        target_weights_hash_value,
        cash_policy,
        constraint_policy_version,
        ranking_policy_version,
        overnight_policy_version,
        friction_policy_version,
        support_policy_version,
        calendar_version,
    )


def normalize_long_only_weights(weights: dict[str, float], min_cash_weight: float) -> tuple[dict[str, float], float]:
    clean = {str(k): max(0.0, float(v)) for k, v in weights.items() if float(v) > 0.0}
    total = float(sum(clean.values()))
    if total <= 0.0:
        return {}, 1.0
    scale = min(1.0 - float(min_cash_weight), 1.0) / total
    scaled = {k: float(v * scale) for k, v in clean.items() if v * scale > 0.0}
    cash_weight = 1.0 - float(sum(scaled.values()))
    return scaled, float(cash_weight)
