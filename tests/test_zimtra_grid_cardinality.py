from strategy_engine import EXPECTED_BASE_STRATEGY_COUNT, family_counts, generate_strategy_specs


def test_grid_cardinality_exact_15120() -> None:
    specs = generate_strategy_specs()
    assert len(specs) == EXPECTED_BASE_STRATEGY_COUNT
    counts = family_counts(specs)
    assert counts["F1"] == 2400
    assert counts["F2"] == 1800
    assert counts["F3"] == 1800
    assert counts["F4"] == 3240
    assert counts["F5"] == 1920
    assert counts["F6"] == 3960
