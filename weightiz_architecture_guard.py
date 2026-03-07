from __future__ import annotations

import ast
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _load(path: Path) -> str:
    if not path.exists():
        raise RuntimeError("ARCHITECTURE_CONSISTENCY_FAILURE")
    return path.read_text(encoding="utf-8")


def _assert_sweep_stub(root: Path) -> None:
    path = root / "sweep_runner.py"
    src = _load(path)
    if "PARALLEL_ENGINE_FORBIDDEN" not in src:
        raise RuntimeError("ARCHITECTURE_CONSISTENCY_FAILURE")
    tree = ast.parse(src)
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(fn) != 1 or fn[0].name != "run_zimtra_sweep":
        raise RuntimeError("ARCHITECTURE_CONSISTENCY_FAILURE")


def _assert_run_research_dispatch(root: Path) -> None:
    path = root / "run_research.py"
    src = _load(path)
    if "run_weightiz_harness(" not in src:
        raise RuntimeError("ARCHITECTURE_CONSISTENCY_FAILURE")
    if "run_zimtra_sweep(" in src:
        raise RuntimeError("ARCHITECTURE_CONSISTENCY_FAILURE")


def _assert_worker_pipeline(root: Path) -> None:
    path = root / "weightiz_module5_harness.py"
    src = _load(path)
    if "run_module4_signal_funnel(" not in src:
        raise RuntimeError("ARCHITECTURE_CONSISTENCY_FAILURE")
    if "simulate_portfolio_from_signals(" not in src:
        raise RuntimeError("ARCHITECTURE_CONSISTENCY_FAILURE")
    if "attach_shared_feature_store(" not in src:
        raise RuntimeError("ARCHITECTURE_CONSISTENCY_FAILURE")
    tree = ast.parse(src)
    run_group_src = ""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_run_group_task":
            run_group_src = ast.get_source_segment(src, node) or ""
            break
    if "run_weightiz_profile_engine(" in run_group_src:
        raise RuntimeError("ARCHITECTURE_CONSISTENCY_FAILURE")
    if "run_module4_strategy_funnel(" in run_group_src:
        raise RuntimeError("ARCHITECTURE_CONSISTENCY_FAILURE")


def _assert_module4_signal_only(root: Path) -> None:
    path = root / "weightiz_module4_strategy_funnel.py"
    src = _load(path)
    if "def run_module4_signal_funnel(" not in src:
        raise RuntimeError("ARCHITECTURE_CONSISTENCY_FAILURE")
    if "MODULE4_EXECUTION_FORBIDDEN_IN_CANONICAL_PATH" not in src:
        raise RuntimeError("ARCHITECTURE_CONSISTENCY_FAILURE")


def _assert_risk_engine(root: Path) -> None:
    path = root / "risk_engine.py"
    src = _load(path)
    if "def simulate_portfolio_from_signals(" not in src:
        raise RuntimeError("ARCHITECTURE_CONSISTENCY_FAILURE")


def _assert_single_writer(root: Path) -> None:
    path = root / "weightiz_module5_harness.py"
    src = _load(path)
    if "LEDGER_WRITE_FORBIDDEN_IN_WORKER" not in src:
        raise RuntimeError("ARCHITECTURE_CONSISTENCY_FAILURE")


def run_architecture_consistency_check() -> None:
    root = _repo_root()
    _assert_run_research_dispatch(root)
    _assert_sweep_stub(root)
    _assert_worker_pipeline(root)
    _assert_module4_signal_only(root)
    _assert_risk_engine(root)
    _assert_single_writer(root)
