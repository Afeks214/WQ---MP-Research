from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import numpy as np


@dataclass(frozen=True)
class ValidationMismatch:
    level: int
    metric: str
    t: int
    a: int
    module_value: float
    reference_value: float
    abs_error: float


def _first_mismatch(lhs: np.ndarray, rhs: np.ndarray, tol: float) -> tuple[int, int] | None:
    if lhs.shape != rhs.shape:
        raise RuntimeError(f"Shape mismatch in forensic validation: {lhs.shape} vs {rhs.shape}")
    diff = np.abs(lhs - rhs)
    bad = np.argwhere(diff > float(tol))
    if bad.size == 0:
        return None
    i0 = bad[0]
    if i0.shape[0] >= 2:
        return int(i0[0]), int(i0[1])
    return int(i0[0]), 0


def _dump_failure(
    *,
    artifacts_dir: Path,
    prefix: str,
    module_arr: np.ndarray,
    reference_arr: np.ndarray,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / f"{prefix}_forensic_mismatch.npz"
    np.savez_compressed(path, module=module_arr, reference=reference_arr)


def validate_three_levels(
    *,
    timestamps_ns: np.ndarray,
    module_outputs: Mapping[str, np.ndarray],
    reference_outputs: Mapping[str, np.ndarray],
    tolerance: float = 1e-12,
    artifacts_dir: str | Path | None = None,
) -> None:
    art_dir = Path(artifacts_dir) if artifacts_dir is not None else Path("artifacts")

    levels: list[tuple[int, list[str]]] = [
        (1, ["mu", "sigma1", "sigma2", "w1", "w2", "vprof"]),
        (2, ["vp_total", "vp_delta"]),
        (
            3,
            ["ipoc", "ival", "ivah", "d", "affinity", "delta_eff", "breakout", "rejection"],
        ),
    ]

    for level, keys in levels:
        for key in keys:
            if key not in module_outputs or key not in reference_outputs:
                continue
            mod = np.asarray(module_outputs[key])
            ref = np.asarray(reference_outputs[key])
            loc = _first_mismatch(mod, ref, tolerance)
            if loc is None:
                continue
            t, a = loc
            mv = float(mod[t, a]) if mod.ndim >= 2 else float(mod[t])
            rv = float(ref[t, a]) if ref.ndim >= 2 else float(ref[t])
            err = abs(mv - rv)
            ts = int(timestamps_ns[t]) if t < timestamps_ns.shape[0] else -1
            _dump_failure(
                artifacts_dir=art_dir,
                prefix=f"level{level}_{key}_t{t}_a{a}",
                module_arr=mod,
                reference_arr=ref,
            )
            raise RuntimeError(
                "FORENSIC_VALIDATION_MISMATCH: "
                f"level={level} metric={key} t={t} ts_ns={ts} a={a} "
                f"module={mv:.16e} reference={rv:.16e} abs_err={err:.16e}"
            )
