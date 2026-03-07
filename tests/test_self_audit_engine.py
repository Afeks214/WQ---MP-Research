from __future__ import annotations

from pathlib import Path

import pytest

from weightiz_self_audit import run_full_self_audit


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _src(path: str) -> str:
    return (_root() / path).read_text(encoding="utf-8")


def test_self_audit_passes_and_writes_report(tmp_path: Path) -> None:
    report = run_full_self_audit(
        project_root=_root(),
        seed=17,
        run_dir=tmp_path,
        env_overrides={
            "PYTHONHASHSEED": "17",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        },
    )
    assert report["status"] == "pass"
    assert (tmp_path / "self_audit_report.json").exists()


def test_self_audit_fails_on_illegal_module4_execution() -> None:
    bad_m4 = _src("weightiz_module4_strategy_funnel.py").replace(
        "MODULE4_EXECUTION_FORBIDDEN_IN_CANONICAL_PATH",
        "MODULE4_EXECUTION_ALLOWED",
        1,
    )
    with pytest.raises(RuntimeError, match="SELF_AUDIT_FAILURE"):
        run_full_self_audit(
            project_root=_root(),
            seed=17,
            source_overrides={"weightiz_module4_strategy_funnel.py": bad_m4},
        )


def test_self_audit_fails_when_module2_appears_in_worker() -> None:
    bad_harness = _src("weightiz_module5_harness.py").replace(
        "st = _clone_state(cached_state)",
        "st = _clone_state(cached_state)\n            run_weightiz_profile_engine(st, m2_configs[group.m2_idx])",
        1,
    )
    with pytest.raises(RuntimeError, match="SELF_AUDIT_FAILURE"):
        run_full_self_audit(
            project_root=_root(),
            seed=17,
            source_overrides={"weightiz_module5_harness.py": bad_harness},
        )


def test_self_audit_fails_on_float32_usage() -> None:
    bad_risk = _src("risk_engine.py") + "\nX_FLOAT32 = np.array([1.0], dtype=np.float32)\n"
    with pytest.raises(RuntimeError, match="SELF_AUDIT_FAILURE"):
        run_full_self_audit(
            project_root=_root(),
            seed=17,
            source_overrides={"risk_engine.py": bad_risk},
        )


def test_self_audit_fails_on_multiple_runtime_entrypoints() -> None:
    bad_sweep = "from weightiz_module5_harness import run_weightiz_harness\nrun_weightiz_harness()\n"
    with pytest.raises(RuntimeError, match="SELF_AUDIT_FAILURE"):
        run_full_self_audit(
            project_root=_root(),
            seed=17,
            source_overrides={"sweep_runner.py": bad_sweep},
        )


def test_self_audit_fails_on_missing_seed() -> None:
    with pytest.raises(RuntimeError, match="SELF_AUDIT_FAILURE"):
        run_full_self_audit(project_root=_root(), seed=None)


def test_self_audit_fails_on_shared_memory_misuse() -> None:
    bad_shm = _src("weightiz_shared_feature_store.py").replace("arr.setflags(write=False)", "arr.setflags(write=True)", 1)
    with pytest.raises(RuntimeError, match="SELF_AUDIT_FAILURE"):
        run_full_self_audit(
            project_root=_root(),
            seed=17,
            source_overrides={"weightiz_shared_feature_store.py": bad_shm},
        )
