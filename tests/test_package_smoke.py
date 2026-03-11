from __future__ import annotations

import importlib

import weightiz


def test_weightiz_package_importable() -> None:
    assert weightiz.__name__ == "weightiz"


def test_weightiz_wave1_skeleton_importable() -> None:
    modules = (
        "weightiz.cli",
        "weightiz.shared",
        "weightiz.shared.io",
        "weightiz.shared.validation",
        "weightiz.shared.hashing",
        "weightiz.shared.math",
        "weightiz.shared.calendar",
        "weightiz.shared.config",
        "weightiz.shared.logging",
        "weightiz.shared.utils",
        "weightiz.module1",
        "weightiz.module2",
        "weightiz.module3",
        "weightiz.module4",
        "weightiz.module5",
        "weightiz.module6",
    )
    for module_name in modules:
        imported = importlib.import_module(module_name)
        assert imported is not None
