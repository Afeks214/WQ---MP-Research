from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from module6.config import Module6Config


@dataclass(frozen=True)
class ConstraintProjectionResult:
    weights: np.ndarray
    cash_weight: float
    projected: bool
    infeasible: bool
    flags: tuple[str, ...]


def project_to_feasible_weights(
    *,
    target_weights: np.ndarray,
    gross_mult: np.ndarray,
    overnight_flags: np.ndarray,
    cluster_ids: np.ndarray,
    family_ids: np.ndarray,
    priority_scores: np.ndarray,
    config: Module6Config,
) -> ConstraintProjectionResult:
    w = np.maximum(np.asarray(target_weights, dtype=np.float64), 0.0)
    gross_mult = np.asarray(gross_mult, dtype=np.float64)
    overnight = np.asarray(overnight_flags, dtype=np.int8)
    cluster_ids = np.asarray(cluster_ids, dtype=np.int64)
    family_ids = np.asarray(family_ids, dtype=object)
    priority = np.asarray(priority_scores, dtype=np.float64)
    flags: list[str] = []
    projected = False
    for _ in range(16):
        changed = False
        clipped = np.minimum(w, float(config.generator.per_sleeve_cap))
        if not np.allclose(clipped, w):
            flags.append("per_sleeve_cap_projected")
            w = clipped
            projected = True
            changed = True
        for cluster_id in sorted(pd_unique_int(cluster_ids)):
            mask = cluster_ids == cluster_id
            total = float(np.sum(w[mask]))
            if total > float(config.generator.per_cluster_cap) + 1.0e-12:
                w[mask] *= float(config.generator.per_cluster_cap) / total
                flags.append("per_cluster_cap_projected")
                projected = True
                changed = True
        for family_id in pd_unique_obj(family_ids):
            mask = family_ids == family_id
            total = float(np.sum(w[mask]))
            if total > float(config.generator.per_family_cap) + 1.0e-12:
                w[mask] *= float(config.generator.per_family_cap) / total
                flags.append("per_family_cap_projected")
                projected = True
                changed = True
        overnight_active = np.where((overnight > 0) & (w > 0.0))[0]
        if overnight_active.size > int(config.simulator.max_overnight_sleeves):
            ranked = sorted(
                overnight_active.tolist(),
                key=lambda idx: (float(priority[idx]), str(idx)),
            )
            for idx in ranked[: overnight_active.size - int(config.simulator.max_overnight_sleeves)]:
                w[int(idx)] = 0.0
            flags.append("overnight_limit_projected")
            projected = True
            changed = True
        gross = float(np.sum(w * np.maximum(gross_mult, 0.0)))
        leverage_cap = float(config.simulator.overnight_leverage if np.any((overnight > 0) & (w > 0.0)) else config.simulator.intraday_leverage_max)
        if gross > leverage_cap + 1.0e-12:
            w *= leverage_cap / gross
            flags.append("gross_limit_projected")
            projected = True
            changed = True
        total = float(np.sum(w))
        max_risk = 1.0 - float(config.generator.minimum_cash_weight)
        if total > max_risk + 1.0e-12:
            w *= max_risk / total
            flags.append("cash_floor_projected")
            projected = True
            changed = True
        if not changed:
            break
    cash_weight = 1.0 - float(np.sum(w))
    infeasible = bool(
        np.any(w < -1.0e-12)
        or float(np.sum(w)) > 1.0 + 1.0e-12
        or cash_weight < float(config.generator.minimum_cash_weight) - 1.0e-12
        or np.any(w > float(config.generator.per_sleeve_cap) + 1.0e-12)
    )
    return ConstraintProjectionResult(
        weights=np.asarray(w, dtype=np.float64),
        cash_weight=float(cash_weight),
        projected=bool(projected),
        infeasible=bool(infeasible),
        flags=tuple(sorted(set(flags))),
    )


def check_path_constraints(
    *,
    equity: float,
    day_start_equity: float,
    gross_exposure_mult: float,
    cash_weight: float,
    config: Module6Config,
    overnight_active_count: int,
) -> tuple[str, ...]:
    flags: list[str] = []
    if float(equity) < float(config.simulator.account_disable_equity):
        flags.append("capital_floor_hit")
    if float(day_start_equity - equity) > float(config.simulator.daily_loss_limit_frac) * max(float(day_start_equity), 1.0e-12):
        flags.append("daily_loss_breach")
    limit = float(config.simulator.overnight_leverage if overnight_active_count > 0 else config.simulator.intraday_leverage_max)
    if float(gross_exposure_mult) > limit + 1.0e-12:
        flags.append("gross_limit_breach")
    if overnight_active_count > int(config.simulator.max_overnight_sleeves):
        flags.append("overnight_limit_breach")
    if float(cash_weight) < float(config.generator.minimum_cash_weight) - 1.0e-12:
        flags.append("cash_floor_breach")
    return tuple(sorted(set(flags)))


def pd_unique_int(values: np.ndarray) -> list[int]:
    return sorted({int(x) for x in np.asarray(values, dtype=np.int64).tolist()})


def pd_unique_obj(values: np.ndarray) -> list[object]:
    return sorted({x for x in np.asarray(values, dtype=object).tolist()}, key=lambda x: str(x))

