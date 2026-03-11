from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def to_jsonable(obj: Any) -> Any:
    np_floating = getattr(np, "floating", ())
    np_integer = getattr(np, "integer", ())
    np_bool_ = getattr(np, "bool_", ())
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (float,)) or (np_floating and isinstance(obj, np_floating)):
        return float(obj)
    if isinstance(obj, (bool,)) or (np_bool_ and isinstance(obj, np_bool_)):
        return bool(obj)
    if isinstance(obj, (int,)) or (np_integer and isinstance(obj, np_integer)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def write_json(path: Path, obj: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(to_jsonable(obj), ensure_ascii=False, indent=2),
    )


def write_frozen_json(path: Path, obj: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(to_jsonable(obj), ensure_ascii=False, indent=2, sort_keys=True),
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def atomic_write_parquet(df: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
