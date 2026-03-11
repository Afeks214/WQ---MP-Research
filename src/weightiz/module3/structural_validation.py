from __future__ import annotations

import hashlib
import time
import tracemalloc

import numpy as np

from .schema import FeatureIdx
from .structural_prefix_sums import build_prefix_count, build_prefix_sum, rolling_mean_from_prefix
from .types import Module3Output


def validate_shared_tensor_contract(shared_feature_tensor: np.ndarray) -> None:
    x = np.asarray(shared_feature_tensor)
    if x.ndim != 4:
        raise RuntimeError(f"shared_feature_tensor must be [A,T,F,W], got shape={x.shape}")
    if x.dtype != np.float64:
        raise RuntimeError(f"shared_feature_tensor dtype must be float64, got {x.dtype}")
    if int(x.shape[2]) < int(FeatureIdx.N_FIELDS):
        raise RuntimeError(
            f"shared_feature_tensor F dimension too small: F={x.shape[2]}, expected>={int(FeatureIdx.N_FIELDS)}"
        )


def validate_output_contract(output: Module3Output) -> None:
    output.assert_float64()
    s = np.asarray(output.structure_tensor)
    c = np.asarray(output.context_tensor)
    fp = np.asarray(output.profile_fingerprint_tensor)
    rg = np.asarray(output.profile_regime_tensor)

    if s.ndim != 4:
        raise RuntimeError(f"structure_tensor must be 4D, got shape={s.shape}")
    if c.ndim != 4:
        raise RuntimeError(f"context_tensor must be 4D, got shape={c.shape}")
    if fp.ndim != 4:
        raise RuntimeError(f"profile_fingerprint_tensor must be 4D, got shape={fp.shape}")
    if rg.ndim != 4 or rg.shape[2] != 1:
        raise RuntimeError(f"profile_regime_tensor must be [A,T,1,W], got shape={rg.shape}")

    A, T, _, W = s.shape
    for name, arr in [
        ("context_tensor", c),
        ("profile_fingerprint_tensor", fp),
        ("profile_regime_tensor", rg),
    ]:
        if arr.shape[0] != A or arr.shape[1] != T or arr.shape[3] != W:
            raise RuntimeError(
                f"{name} shape mismatch with structure_tensor: got {arr.shape}, expected [A={A},T={T},*,W={W}]"
            )


def validate_prefix_sum_parity(series_atw: np.ndarray, window: int, *, atol: float = 1e-12) -> None:
    x = np.asarray(series_atw, dtype=np.float64)
    if x.ndim != 3:
        raise RuntimeError(f"series_atw must be [A,T,W], got shape={x.shape}")
    if int(window) <= 0:
        raise RuntimeError("window must be > 0")

    valid = np.isfinite(x)
    s = np.where(valid, x, 0.0)

    ps = build_prefix_sum(s)
    pc = build_prefix_count(valid)
    pref = rolling_mean_from_prefix(ps, pc, int(window))

    A, T, W = x.shape
    naive = np.full((A, T, W), np.nan, dtype=np.float64)
    w = int(window)
    for t in range(T):
        lo = t - w + 1
        if lo < 0:
            continue
        seg = x[:, lo : t + 1, :]
        naive[:, t, :] = np.nanmean(seg, axis=1)

    diff = np.abs(pref - naive)
    mask = np.isfinite(diff)
    max_abs = float(np.max(diff[mask])) if np.any(mask) else 0.0
    if max_abs > float(atol):
        raise RuntimeError(
            f"Prefix parity validation failed: max_abs_error={max_abs:.3e}, atol={float(atol):.3e}"
        )


def validate_window_alignment(structural_windows: tuple[int, ...], T: int) -> None:
    if int(T) <= 0:
        raise RuntimeError("T must be > 0")
    for w in structural_windows:
        ww = int(w)
        if ww <= 0:
            raise RuntimeError(f"Invalid window {ww}")
        if ww > int(T):
            # allowed for warmup-heavy runs; this check only enforces causality bounds
            continue


def validate_context_causality(context_source_index_atw: np.ndarray, session_id_t: np.ndarray) -> None:
    src = np.asarray(context_source_index_atw, dtype=np.int64)
    sid = np.asarray(session_id_t, dtype=np.int64)
    if src.ndim != 3:
        raise RuntimeError(f"context_source_index_atw must be [A,T,W], got shape={src.shape}")
    if sid.ndim != 1 or sid.shape[0] != src.shape[1]:
        raise RuntimeError(
            f"session_id_t shape mismatch: got {sid.shape}, expected {(src.shape[1],)}"
        )
    A, T, W = src.shape

    t_idx = np.arange(T, dtype=np.int64)[None, :, None]
    bad_future = (src >= 0) & (src > t_idx)
    if np.any(bad_future):
        loc = np.argwhere(bad_future)[0]
        raise RuntimeError(
            f"Context source is in the future at a={int(loc[0])}, t={int(loc[1])}, w={int(loc[2])}"
        )

    for a in range(A):
        for w in range(W):
            idx = src[a, :, w]
            m = idx >= 0
            if not np.any(m):
                continue
            src_sid = sid[np.where(m, idx, 0)]
            dst_sid = sid
            bad_sid = m & (src_sid != dst_sid)
            if np.any(bad_sid):
                t_bad = int(np.flatnonzero(bad_sid)[0])
                raise RuntimeError(
                    f"Context source crosses session boundary at a={a}, t={t_bad}, w={w}"
                )


def validate_fingerprint_stability(
    fp_ref: np.ndarray,
    fp_new: np.ndarray,
    *,
    atol: float = 1e-12,
) -> None:
    a = np.asarray(fp_ref, dtype=np.float64)
    b = np.asarray(fp_new, dtype=np.float64)
    if a.shape != b.shape:
        raise RuntimeError(f"Fingerprint shape mismatch: {a.shape} vs {b.shape}")
    diff = np.abs(a - b)
    mask = np.isfinite(diff)
    max_abs = float(np.max(diff[mask])) if np.any(mask) else 0.0
    if max_abs > float(atol):
        raise RuntimeError(
            f"Fingerprint stability validation failed: max_abs_error={max_abs:.3e}, atol={float(atol):.3e}"
        )


def deterministic_digest_sha256_module3(output: Module3Output) -> str:
    h = hashlib.sha256()
    arrs = [
        np.asarray(output.structure_tensor, dtype=np.float64),
        np.asarray(output.context_tensor, dtype=np.float64),
        np.asarray(output.profile_fingerprint_tensor, dtype=np.float64),
        np.asarray(output.profile_regime_tensor, dtype=np.float64),
    ]
    if output.context_source_index_atw is not None:
        arrs.append(np.asarray(output.context_source_index_atw, dtype=np.int64))

    for a in arrs:
        h.update(np.ascontiguousarray(a).view(np.uint8))
    return h.hexdigest()


def run_forensic_validation(
    shared_feature_tensor: np.ndarray,
    output: Module3Output,
    *,
    session_id_t: np.ndarray,
    structural_windows: tuple[int, ...],
    atol: float = 1e-12,
) -> None:
    validate_shared_tensor_contract(shared_feature_tensor)
    validate_output_contract(output)
    validate_window_alignment(structural_windows, int(np.asarray(shared_feature_tensor).shape[1]))

    dclip = np.asarray(shared_feature_tensor, dtype=np.float64)[:, :, int(FeatureIdx.DCLIP), :]
    for w in structural_windows:
        validate_prefix_sum_parity(dclip, int(w), atol=float(atol))

    if output.context_source_index_atw is not None:
        validate_context_causality(output.context_source_index_atw, session_id_t)

    validate_fingerprint_stability(
        np.asarray(output.profile_fingerprint_tensor, dtype=np.float64),
        np.asarray(output.profile_fingerprint_tensor, dtype=np.float64).copy(),
        atol=float(atol),
    )


def benchmark_naive_vs_prefix(
    series_atw: np.ndarray,
    window: int,
    *,
    repeats: int = 3,
) -> dict[str, float | str]:
    """Deterministic benchmark of naive vs prefix rolling mean.

    Returns runtime, memory peak, and cache-efficiency proxy metrics.
    """
    x = np.asarray(series_atw, dtype=np.float64)
    if x.ndim != 3:
        raise RuntimeError(f"series_atw must be [A,T,W], got shape={x.shape}")
    if int(window) <= 0:
        raise RuntimeError("window must be > 0")
    if int(repeats) <= 0:
        raise RuntimeError("repeats must be > 0")

    A, T, W = x.shape
    w = int(window)

    def _naive() -> np.ndarray:
        out = np.full((A, T, W), np.nan, dtype=np.float64)
        for t in range(T):
            lo = t - w + 1
            if lo < 0:
                continue
            out[:, t, :] = np.mean(x[:, lo : t + 1, :], axis=1, dtype=np.float64)
        return out

    def _prefix() -> np.ndarray:
        ps = build_prefix_sum(x)
        pc = build_prefix_count(np.ones(x.shape, dtype=bool))
        return rolling_mean_from_prefix(ps, pc, w)

    def _time_and_peak(fn) -> tuple[float, int]:
        best = float("inf")
        peak = 0
        for _ in range(int(repeats)):
            tracemalloc.start()
            t0 = time.perf_counter()
            _ = fn()
            dt = time.perf_counter() - t0
            _, p = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            best = min(best, float(dt))
            peak = max(peak, int(p))
        return best, peak

    naive_runtime, naive_peak = _time_and_peak(_naive)
    prefix_runtime, prefix_peak = _time_and_peak(_prefix)

    elems = float(max(1, A * T * W))
    # Cache efficiency proxy (no hardware counters required).
    naive_eps = elems / max(naive_runtime, 1e-12)
    prefix_eps = elems / max(prefix_runtime, 1e-12)

    return {
        "naive_runtime_sec": float(naive_runtime),
        "prefix_runtime_sec": float(prefix_runtime),
        "speedup_x": float(naive_runtime / max(prefix_runtime, 1e-12)),
        "naive_memory_peak_bytes": float(naive_peak),
        "prefix_memory_peak_bytes": float(prefix_peak),
        "naive_cache_efficiency_proxy_eps": float(naive_eps),
        "prefix_cache_efficiency_proxy_eps": float(prefix_eps),
        "cache_efficiency_metric": "elements_per_second_proxy",
    }
