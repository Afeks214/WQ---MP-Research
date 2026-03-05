from __future__ import annotations

from strategy_engine import generate_sobol_strategy_specs, strategy_payload
from weightiz_adaptive_search import adaptive_search


RANGES = {
    "profile_window_minutes": (30.0, 60.0),
    "profile_memory_sessions": (2.0, 4.0),
    "deltaeff_threshold": (0.15, 0.45),
    "distance_to_poc_atr": (0.25, 1.0),
    "acceptance_threshold": (0.3, 0.7),
    "rvol_filter": (1.0, 2.0),
    "holding_period_days": (1.0, 4.0),
}


def test_sobol_determinism() -> None:
    s1 = generate_sobol_strategy_specs(
        n_samples=1024,
        param_ranges=RANGES,
        seed=42,
    )
    s2 = generate_sobol_strategy_specs(
        n_samples=1024,
        param_ranges=RANGES,
        seed=42,
    )

    p1 = [strategy_payload(x) for x in s1]
    p2 = [strategy_payload(x) for x in s2]
    assert p1 == p2


def test_adaptive_generation() -> None:
    fake_results = [
        {
            "strategy_id": "S0",
            "profit_factor": 1.3,
            "params": {"deltaeff_threshold": 0.3},
        }
    ]
    out = adaptive_search(fake_results, 100, noise=0.15, seed=42)
    assert len(out) > 0

