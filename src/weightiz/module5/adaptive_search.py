from __future__ import annotations

from typing import Any

import numpy as np


def adaptive_search(
    base_results: list[dict[str, Any]],
    n_new: int,
    noise: float = 0.15,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Generate new strategy parameter dictionaries around top-performing strategies.

    Deterministic policy:
    - stable sort by (-profit_factor, strategy_id)
    - deterministic RNG seeded by `seed`
    """
    if int(n_new) <= 0 or len(base_results) == 0:
        return []

    rows = sorted(
        list(base_results),
        key=lambda x: (-float(x.get("profit_factor", 0.0)), str(x.get("strategy_id", ""))),
    )
    top_n = max(5, int(len(rows) * 0.05))
    top = rows[:top_n]
    if len(top) == 0:
        return []

    rng = np.random.default_rng(int(seed))
    per_top = max(1, int(np.ceil(int(n_new) / max(len(top), 1))))
    out: list[dict[str, Any]] = []
    for strat in top:
        params = strat.get("params", {})
        if not isinstance(params, dict):
            continue
        for _ in range(per_top):
            mutated: dict[str, Any] = {}
            for k in sorted(params.keys()):
                v = params[k]
                if isinstance(v, bool):
                    mutated[k] = bool(v)
                elif isinstance(v, (int, float)):
                    vv = float(v)
                    delta = vv * float(noise) * float(rng.uniform(-1.0, 1.0))
                    mutated[k] = max(0.0, vv + delta)
                else:
                    mutated[k] = v
            if mutated:
                out.append(mutated)
            if len(out) >= int(n_new):
                return out
    return out[: int(n_new)]

