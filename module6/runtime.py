from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from module6.matrices import MatrixStore


@dataclass(frozen=True)
class BatchPlan:
    batch_id: str
    start: int
    end: int


def plan_batches(total_count: int, batch_size: int, prefix: str) -> list[BatchPlan]:
    out: list[BatchPlan] = []
    for start in range(0, int(total_count), int(max(batch_size, 1))):
        end = min(int(total_count), start + int(max(batch_size, 1)))
        out.append(BatchPlan(batch_id=f"{prefix}_{len(out):04d}", start=start, end=end))
    return out


def open_matrix_store(store: MatrixStore) -> dict[str, np.ndarray | sparse.csr_matrix]:
    return {
        "R_exec": np.load(store.returns_exec_path, mmap_mode="r"),
        "R_raw": np.load(store.returns_raw_path, mmap_mode="r"),
        "A": np.load(store.availability_path, mmap_mode="r"),
        "U": np.load(store.turnover_path, mmap_mode="r"),
        "state_codes": np.load(store.state_code_path, mmap_mode="r"),
        "gross_peak": np.load(store.gross_peak_path, mmap_mode="r"),
        "gross_mean": np.load(store.gross_mean_path, mmap_mode="r"),
        "buying_power_min": np.load(store.buying_power_min_path, mmap_mode="r"),
        "overnight_flag": np.load(store.overnight_flag_path, mmap_mode="r"),
        "F": sparse.load_npz(store.family_incidence_path).tocsr(),
        "G": np.load(store.regime_exposure_path, mmap_mode="r"),
    }


def emit_runtime_metrics(**kwargs: object) -> dict[str, object]:
    return dict(sorted(kwargs.items(), key=lambda item: str(item[0])))
