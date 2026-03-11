from __future__ import annotations

from weightiz.cli import run_research as rr


def test_legacy_workers_alias_maps_to_harness():
    cfg = rr.RunConfigModel.model_validate(
        {
            "symbols": ["AAA", "BBB"],
            "zimtra_sweep": {"enabled": False, "workers": 7, "seed": 13},
        }
    )
    cfg = rr._map_legacy_zimtra_aliases(cfg)
    assert cfg.harness.parallel_workers == 7
    assert cfg.search.seed == 13
