from __future__ import annotations

import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


_SOURCE_FILES = {
    "run_research.py": "src/weightiz/cli/run_research.py",
    "sweep_runner.py": "sweep_runner.py",
    "weightiz_module2_core.py": "src/weightiz/module2/core.py",
    "weightiz_module3_structure.py": "src/weightiz/module3/bridge.py",
    "weightiz_module4_strategy_funnel.py": "src/weightiz/module4/strategy_funnel.py",
    "weightiz_module5_harness.py": "src/weightiz/module5/orchestrator.py",
    "weightiz_shared_feature_store.py": "src/weightiz/shared/io/shared_feature_store.py",
    "weightiz_feature_tensor_cache.py": "src/weightiz/shared/io/feature_tensor_cache.py",
    "weightiz_dtype_guard.py": "src/weightiz/shared/validation/dtype_guard.py",
    "weightiz_system_logger.py": "src/weightiz/shared/logging/system_logger.py",
    "risk_engine.py": "src/weightiz/module4/risk_engine.py",
}

_RUNTIME_FILES = (
    "run_research.py",
    "weightiz_module2_core.py",
    "weightiz_module3_structure.py",
    "weightiz_module4_strategy_funnel.py",
    "weightiz_module5_harness.py",
    "risk_engine.py",
)


def _repo_root(project_root: Path | None) -> Path:
    if project_root is not None:
        return Path(project_root).resolve()
    return Path(__file__).resolve().parents[4]


def _read_sources(root: Path, overrides: dict[str, str] | None = None) -> dict[str, str]:
    overrides = overrides or {}
    out: dict[str, str] = {}
    for logical_name, rel_path in _SOURCE_FILES.items():
        if logical_name in overrides:
            out[logical_name] = str(overrides[logical_name])
            continue
        p = root / rel_path
        if not p.exists():
            raise RuntimeError(f"SELF_AUDIT_SOURCE_MISSING:{logical_name}")
        out[logical_name] = p.read_text(encoding="utf-8")
    return out


def _function_source(module_src: str, fn_name: str) -> str:
    tree = ast.parse(module_src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            return ast.get_source_segment(module_src, node) or ""
    return ""


def _call_sites(sources: dict[str, str], fn_name: str) -> list[str]:
    callers: list[str] = []
    for rel, src in sources.items():
        if not rel.endswith(".py"):
            continue
        try:
            tree = ast.parse(src)
        except Exception:
            continue
        hit = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                if isinstance(f, ast.Name) and f.id == fn_name:
                    hit = True
                    break
                if isinstance(f, ast.Attribute) and f.attr == fn_name:
                    hit = True
                    break
        if hit:
            callers.append(rel)
    return sorted(callers)


def _extract_seed(cfg: Any, explicit_seed: int | None) -> int | None:
    if explicit_seed is not None:
        return int(explicit_seed)
    if cfg is None:
        return None
    search = getattr(cfg, "search", None)
    if search is None:
        return None
    seed = getattr(search, "seed", None)
    if seed is None and isinstance(search, dict):
        seed = search.get("seed")
    if seed is None:
        return None
    return int(seed)


def _extract_report_dir(cfg: Any, root: Path) -> Path | None:
    if cfg is None:
        return None
    harness = getattr(cfg, "harness", None)
    if harness is None:
        return None
    report_dir = getattr(harness, "report_dir", None)
    if report_dir is None and isinstance(harness, dict):
        report_dir = harness.get("report_dir")
    if report_dir is None:
        return None
    return (root / str(report_dir)).resolve() if not Path(str(report_dir)).is_absolute() else Path(str(report_dir)).resolve()


def _set_category(report: dict[str, Any], name: str, violations: list[str], details: dict[str, Any]) -> None:
    report["categories"][name] = {
        "status": "fail" if violations else "pass",
        "violations": list(violations),
        "details": details,
    }


def _validate_cached_tensor(cache_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "cache_dir": str(cache_dir),
        "cache_found": False,
        "validated_file": "",
        "tensor_rank_ok": False,
        "dtype_ok": False,
        "nan_ok": False,
        "inf_ok": False,
    }
    if not cache_dir.exists():
        return result
    npz_files = sorted(cache_dir.glob("profile_tensor_*.npz"), key=lambda p: p.stat().st_mtime)
    if not npz_files:
        return result
    path = npz_files[-1]
    result["cache_found"] = True
    result["validated_file"] = str(path)
    with np.load(path, allow_pickle=False) as payload:
        if "tensor" not in payload:
            raise RuntimeError("SELF_AUDIT_FAILURE: cached tensor missing 'tensor' field")
        tensor = np.asarray(payload["tensor"])
    result["tensor_rank_ok"] = bool(tensor.ndim == 4)
    result["dtype_ok"] = bool(tensor.dtype == np.float64)
    result["nan_ok"] = bool(not np.isnan(tensor).any())
    result["inf_ok"] = bool(not np.isinf(tensor).any())
    return result


def _write_report(run_dir: Path | None, report: dict[str, Any]) -> None:
    if run_dir is None:
        return
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "self_audit_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def run_full_self_audit(
    *,
    cfg: Any | None = None,
    project_root: Path | None = None,
    run_dir: Path | None = None,
    source_overrides: dict[str, str] | None = None,
    env_overrides: dict[str, str] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    root = _repo_root(project_root)
    sources = _read_sources(root, source_overrides)
    env = dict(os.environ)
    if env_overrides:
        env.update({str(k): str(v) for k, v in env_overrides.items()})

    report: dict[str, Any] = {
        "audit_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "violations": [],
        "environment": {
            "PYTHONHASHSEED": str(env.get("PYTHONHASHSEED", "")),
            "OMP_NUM_THREADS": str(env.get("OMP_NUM_THREADS", "")),
            "MKL_NUM_THREADS": str(env.get("MKL_NUM_THREADS", "")),
            "OPENBLAS_NUM_THREADS": str(env.get("OPENBLAS_NUM_THREADS", "")),
            "NUMEXPR_NUM_THREADS": str(env.get("NUMEXPR_NUM_THREADS", "")),
            "VECLIB_MAXIMUM_THREADS": str(env.get("VECLIB_MAXIMUM_THREADS", "")),
        },
        "categories": {},
        "call_graph": {},
    }
    all_violations: list[str] = []

    # Category 1: architecture graph and canonical pipeline
    c1: list[str] = []
    run_src = sources["run_research.py"]
    sweep_src = sources["sweep_runner.py"]
    harness_src = sources["weightiz_module5_harness.py"]
    m4_src = sources["weightiz_module4_strategy_funnel.py"]
    run_group_src = _function_source(harness_src, "_run_group_task")
    init_worker_src = _function_source(harness_src, "_init_worker_context")
    if "run_weightiz_harness(" not in run_src:
        c1.append("run_research missing harness dispatch")
    if "run_zimtra_sweep(" in run_src:
        c1.append("parallel sweep dispatch reachable from run_research")
    if "PARALLEL_ENGINE_FORBIDDEN" not in sweep_src:
        c1.append("sweep_runner is not fatal stub")
    if "run_module4_signal_funnel(" not in run_group_src:
        c1.append("worker path missing run_module4_signal_funnel")
    if "simulate_portfolio_from_signals(" not in run_group_src:
        c1.append("worker path missing risk_engine simulation call")
    if "run_weightiz_profile_engine(" in run_group_src:
        c1.append("Module2 called in worker path")
    if "build_feature_tensor_multiaxis(" in run_group_src:
        c1.append("feature tensor construction appears in worker path")
    if "MODULE4_EXECUTION_FORBIDDEN_IN_CANONICAL_PATH" not in m4_src:
        c1.append("Module4 execution API not forbidden")
    harness_callers = _call_sites(sources, "run_weightiz_harness")
    if harness_callers != ["run_research.py"]:
        c1.append(f"multiple runtime entrypoints detected: {harness_callers}")
    report["call_graph"]["run_weightiz_harness_callers"] = harness_callers
    _set_category(
        report,
        "architecture_graph",
        c1,
        {
            "worker_required_calls": ["run_module4_signal_funnel", "simulate_portfolio_from_signals"],
            "worker_forbidden_calls": ["run_weightiz_profile_engine", "build_feature_tensor_multiaxis"],
        },
    )
    all_violations.extend(c1)

    # Category 2: execution authority
    c2: list[str] = []
    risk_src = sources["risk_engine.py"]
    if "def simulate_portfolio_from_signals(" not in risk_src:
        c2.append("risk_engine missing simulate_portfolio_from_signals")
    if "MODULE4_EXECUTION_FORBIDDEN_IN_CANONICAL_PATH" not in m4_src:
        c2.append("Module4 execution path still enabled")
    if "_execute_to_target(" in run_group_src:
        c2.append("execution helper referenced in worker path")
    _set_category(report, "execution_authority", c2, {})
    all_violations.extend(c2)

    # Category 3: shared memory safety
    c3: list[str] = []
    shm_src = sources["weightiz_shared_feature_store.py"]
    run_harness_src = _function_source(harness_src, "run_weightiz_harness")
    if "def create_shared_feature_store(" not in shm_src:
        c3.append("shared memory create API missing")
    if "def attach_shared_feature_store(" not in shm_src:
        c3.append("shared memory attach API missing")
    if "arr.setflags(write=False)" not in shm_src:
        c3.append("worker shared memory is not read-only")
    if "if is_master:" not in shm_src or "handles.shm.unlink()" not in shm_src:
        c3.append("master-only unlink guard missing")
    if "create_shared_feature_store(" in run_group_src:
        c3.append("worker path allocates shared memory")
    if "attach_shared_feature_store(" not in init_worker_src:
        c3.append("worker path does not attach shared memory")
    if "create_shared_feature_store(" not in run_harness_src:
        c3.append("master path does not create shared memory")
    _set_category(report, "shared_memory_safety", c3, {})
    all_violations.extend(c3)

    # Category 4: float precision
    c4: list[str] = []
    dtype_src = sources["weightiz_dtype_guard.py"]
    if "def assert_float64(" not in dtype_src or "FLOAT64_ENFORCEMENT_FAILURE" not in dtype_src:
        c4.append("assert_float64 guard missing")
    boundary_tokens = {
        "weightiz_module2_core.py": "assert_float64(",
        "weightiz_feature_tensor_cache.py": "assert_float64(",
        "weightiz_shared_feature_store.py": "assert_float64(",
        "weightiz_module3_structure.py": "assert_float64(",
        "weightiz_module4_strategy_funnel.py": "assert_float64(",
        "risk_engine.py": "assert_float64(",
        "weightiz_module5_harness.py": "assert_float64(",
    }
    for rel, token in boundary_tokens.items():
        if token not in sources.get(rel, ""):
            c4.append(f"float64 enforcement missing in {rel}")
    for rel in _RUNTIME_FILES:
        if "np.float32" in sources[rel] or "dtype=float32" in sources[rel]:
            c4.append(f"float32 token detected in runtime module {rel}")
    _set_category(report, "float_precision", c4, {})
    all_violations.extend(c4)

    # Category 5: data sanity
    c5: list[str] = []
    m2_src = sources["weightiz_module2_core.py"]
    if "FEATURE_TENSOR_CONTAINS_NAN" not in m2_src:
        c5.append("NaN guard missing for feature tensor")
    if "FEATURE_TENSOR_CONTAINS_INF" not in m2_src:
        c5.append("Inf guard missing for feature tensor")
    if "arr.ndim != 4" not in m2_src and "tensor.ndim != 4" not in m2_src:
        c5.append("feature tensor rank guard missing")
    cache_validation: dict[str, Any] = {}
    report_dir = _extract_report_dir(cfg, root)
    if report_dir is not None:
        cache_dir = report_dir.parent / "profile_cache"
        try:
            cache_validation = _validate_cached_tensor(cache_dir)
            if cache_validation.get("cache_found"):
                if not cache_validation.get("tensor_rank_ok", False):
                    c5.append("cached feature tensor rank mismatch")
                if not cache_validation.get("dtype_ok", False):
                    c5.append("cached feature tensor dtype mismatch")
                if not cache_validation.get("nan_ok", False):
                    c5.append("cached feature tensor contains NaN")
                if not cache_validation.get("inf_ok", False):
                    c5.append("cached feature tensor contains Inf")
        except Exception as exc:
            c5.append(f"cached feature tensor validation failed: {type(exc).__name__}")
            cache_validation = {"error": f"{type(exc).__name__}: {exc}"}
    _set_category(report, "data_sanity", c5, {"cache_validation": cache_validation})
    all_violations.extend(c5)

    # Category 6: deterministic runtime
    c6: list[str] = []
    actual_seed = _extract_seed(cfg, seed)
    if actual_seed is None:
        c6.append("config.search.seed is missing")
    required_env = {
        "PYTHONHASHSEED": str(actual_seed) if actual_seed is not None else "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    for k, expected in required_env.items():
        got = str(env.get(k, ""))
        if got != expected:
            c6.append(f"deterministic env mismatch: {k}={got} expected={expected}")
    _set_category(report, "runtime_determinism", c6, {"seed": actual_seed})
    all_violations.extend(c6)

    # Category 7: logging integrity
    c7: list[str] = []
    for rel in _RUNTIME_FILES:
        if "print(" in sources[rel]:
            c7.append(f"print() found in runtime module {rel}")
    logger_src = sources["weightiz_system_logger.py"]
    if "QueueHandler" not in logger_src or "QueueListener" not in logger_src:
        c7.append("centralized queue logger missing QueueHandler/QueueListener")
    _set_category(report, "logging_integrity", c7, {})
    all_violations.extend(c7)

    # Category 8: ledger single writer
    c8: list[str] = []
    if "LEDGER_WRITE_FORBIDDEN_IN_WORKER" not in harness_src:
        c8.append("ledger single-writer guard missing")
    if "os.replace(tmp, path)" not in harness_src:
        c8.append("atomic ledger replace missing")
    forbidden_io_tokens = ("to_parquet(", "to_csv(", "_write_json(", "_ledger_write(")
    for tok in forbidden_io_tokens:
        if tok in run_group_src:
            c8.append(f"worker path performs artifact IO: {tok}")
    _set_category(report, "ledger_integrity", c8, {})
    all_violations.extend(c8)

    # Category 9: worker isolation
    c9: list[str] = []
    required_worker = (
        "run_module3_structural_aggregation(",
        "run_module4_signal_funnel(",
        "simulate_portfolio_from_signals(",
    )
    forbidden_worker = (
        "run_weightiz_profile_engine(",
        "build_feature_tensor_multiaxis(",
        "build_feature_tensor_from_state(",
        "read_parquet(",
        "read_csv(",
        "run_module4_strategy_funnel(",
    )
    for tok in required_worker:
        if tok not in run_group_src:
            c9.append(f"worker pipeline missing required call {tok}")
    for tok in forbidden_worker:
        if tok in run_group_src:
            c9.append(f"worker pipeline contains forbidden call {tok}")
    _set_category(report, "worker_isolation", c9, {})
    all_violations.extend(c9)

    # Category 10: module contracts
    c10: list[str] = []
    if "MODULE2_WORKER_EXECUTION_FORBIDDEN" not in m2_src:
        c10.append("Module2 worker execution guard missing")
    if "def run_module3_structural_aggregation(" not in sources["weightiz_module3_structure.py"]:
        c10.append("Module3 aggregation entrypoint missing")
    if "def run_module4_signal_funnel(" not in m4_src:
        c10.append("Module4 signal funnel entrypoint missing")
    if "def simulate_portfolio_from_signals(" not in risk_src:
        c10.append("risk_engine execution entrypoint missing")
    _set_category(report, "module_contracts", c10, {})
    all_violations.extend(c10)

    report["violations"] = all_violations
    if all_violations:
        report["status"] = "fail"

    _write_report(run_dir, report)

    if all_violations:
        raise RuntimeError("SELF_AUDIT_FAILURE")
    return report
