from __future__ import annotations

import json
from pathlib import Path

from module5.harness.artifact_writers import write_frozen_json, write_json


def test_write_json_replaces_file_without_leaking_tempfiles(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text("{\"stale\": true}", encoding="utf-8")

    write_json(path, {"alpha": 1, "beta": [2, 3]})

    assert json.loads(path.read_text(encoding="utf-8")) == {"alpha": 1, "beta": [2, 3]}
    assert list(tmp_path.glob("*.tmp")) == []


def test_write_frozen_json_sorts_keys(tmp_path: Path) -> None:
    path = tmp_path / "artifact_frozen.json"
    write_frozen_json(path, {"beta": 2, "alpha": 1})
    text = path.read_text(encoding="utf-8")
    assert text.index('"alpha"') < text.index('"beta"')
