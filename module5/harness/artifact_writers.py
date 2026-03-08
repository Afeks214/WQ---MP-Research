from __future__ import annotations

import json
import os
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, ensure_ascii=False, indent=2)


def write_frozen_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, ensure_ascii=False, indent=2, sort_keys=True)


def atomic_write_parquet(df: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
