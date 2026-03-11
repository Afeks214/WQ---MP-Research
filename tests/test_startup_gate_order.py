from __future__ import annotations

import inspect

from weightiz.cli import run_research


def test_startup_gate_order_in_run_research() -> None:
    src = inspect.getsource(run_research)
    i_seed = src.index("_configure_deterministic_runtime(int(cfg.search.seed))")
    i_self = src.index("run_full_self_audit(")
    i_arch = src.index("run_architecture_consistency_check()")
    i_pre = src.index("run_preflight_validation_suite(")
    i_harness = src.index("out = run_weightiz_harness(")
    assert i_seed < i_self < i_arch < i_pre < i_harness
