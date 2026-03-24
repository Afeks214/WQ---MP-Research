from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np

from weightiz.module1.core import EngineConfig, Phase, preallocate_state


REPO_ROOT = Path(__file__).resolve().parents[2]
M1_CORE = str(REPO_ROOT / "src/weightiz/module1/core.py")
M2_CORE = str(REPO_ROOT / "src/weightiz/module2/core.py")
M2_ENGINE = str(REPO_ROOT / "src/weightiz/module2/market_profile_engine.py")
M2_KERNELS = str(REPO_ROOT / "src/weightiz/module2/market_profile_kernels.py")
M2_TENSOR = str(REPO_ROOT / "src/weightiz/module2/tensor_builder.py")
M2_REFERENCE = str(REPO_ROOT / "src/weightiz/module2/reference_pipeline.py")
M5_ORCH = str(REPO_ROOT / "src/weightiz/module5/orchestrator.py")
CLI_MODULE5 = str(REPO_ROOT / "src/weightiz/cli/run_module5.py")
CLI_RESEARCH = str(REPO_ROOT / "src/weightiz/cli/run_research.py")
CFG_MODELS = str(REPO_ROOT / "src/weightiz/shared/config/models.py")
CFG_BUILDERS = str(REPO_ROOT / "src/weightiz/shared/config/builders.py")
PROFILE_ENGINE = str(REPO_ROOT / "src/weightiz/shared/io/profile_engine.py")
PARITY_ENGINE = str(REPO_ROOT / "src/weightiz/shared/io/hpc_market_profile_parity.py")
LEGACY_HPC = str(REPO_ROOT / "engine/hpc_market_profile.py")

EXECUTED_PATH = (
    "src/weightiz/cli/run_module5.py:6-9 -> "
    "src/weightiz/cli/run_research.py:747-824 -> "
    "src/weightiz/module5/orchestrator.py:2529-2538 -> "
    "src/weightiz/module2/core.py:1796-1877 -> "
    "src/weightiz/module2/market_profile_engine.py:287-529 -> "
    "src/weightiz/module2/market_profile_kernels.py:49-160 -> "
    "src/weightiz/module2/tensor_builder.py:50-79"
)

SQRT_2PI = float(np.sqrt(2.0 * np.pi))


def make_clock_override(
    *,
    minute_of_day: Iterable[int],
    tod: Iterable[int] | None = None,
    session_id: Iterable[int] | None = None,
    gap_min: Iterable[float] | None = None,
    reset_flag: Iterable[int] | None = None,
    phase: Iterable[int] | None = None,
) -> dict[str, np.ndarray]:
    minute_arr = np.asarray(list(minute_of_day), dtype=np.int16)
    T = int(minute_arr.shape[0])
    if tod is None:
        tod_arr = (minute_arr.astype(np.int32) - (9 * 60 + 30)).astype(np.int16)
    else:
        tod_arr = np.asarray(list(tod), dtype=np.int16)
    if session_id is None:
        sid_arr = np.zeros(T, dtype=np.int64)
    else:
        sid_arr = np.asarray(list(session_id), dtype=np.int64)
    if gap_min is None:
        gap_arr = np.zeros(T, dtype=np.float64)
        if T > 1:
            gap_arr[1:] = 1.0
    else:
        gap_arr = np.asarray(list(gap_min), dtype=np.float64)
    if reset_flag is None:
        rst_arr = np.zeros(T, dtype=np.int8)
        rst_arr[0] = np.int8(1)
        rst_arr[1:] = (sid_arr[1:] != sid_arr[:-1]).astype(np.int8)
    else:
        rst_arr = np.asarray(list(reset_flag), dtype=np.int8)
    if phase is None:
        phase_arr = np.where(tod_arr < 15, np.int8(Phase.WARMUP), np.int8(Phase.LIVE)).astype(np.int8)
    else:
        phase_arr = np.asarray(list(phase), dtype=np.int8)
    return {
        "minute_of_day": minute_arr,
        "tod": tod_arr,
        "session_id": sid_arr,
        "gap_min": gap_arr,
        "reset_flag": rst_arr,
        "phase": phase_arr,
    }


def make_state(
    *,
    T: int,
    tick_size: float = 0.01,
    mode: str = "sealed",
    clock_override: dict[str, np.ndarray] | None = None,
) -> object:
    start_ns = np.datetime64("2025-01-06T14:30:00", "ns").astype(np.int64)
    ts_ns = start_ns + np.arange(T, dtype=np.int64) * np.int64(60_000_000_000)
    cfg = EngineConfig(
        T=T,
        A=1,
        B=240,
        tick_size=np.array([tick_size], dtype=np.float64),
        mode=mode,
    )
    return preallocate_state(ts_ns=ts_ns, cfg=cfg, symbols=("A0",), clock_override=clock_override)


def fill_bars(state: object, rows: list[tuple[float, float, float, float, float]]) -> None:
    bars = np.asarray(rows, dtype=np.float64)
    state.open_px[:, 0] = bars[:, 0]
    state.high_px[:, 0] = bars[:, 1]
    state.low_px[:, 0] = bars[:, 2]
    state.close_px[:, 0] = bars[:, 3]
    state.volume[:, 0] = bars[:, 4]
    state.bar_valid[:, 0] = True


def make_fixed_physics(
    *,
    T: int,
    atr_eff: float = 1.0,
    rvol: float = 1.0,
    clv: float = 0.0,
    body_pct: float = 0.0,
    sigma1: float = 0.1,
    sigma2: float = 0.1,
    w1: float = 1.0,
    w2: float = 0.0,
    cap_v_eff: float = 1.0e9,
    ret_norm: float = 0.0,
    s_r: float = 0.05,
) -> SimpleNamespace:
    shape = (T, 1)
    return SimpleNamespace(
        atr_eff=np.full(shape, atr_eff, dtype=np.float64),
        rvol=np.full(shape, rvol, dtype=np.float64),
        clv=np.full(shape, clv, dtype=np.float64),
        body_pct=np.full(shape, body_pct, dtype=np.float64),
        sigma1=np.full(shape, sigma1, dtype=np.float64),
        sigma2=np.full(shape, sigma2, dtype=np.float64),
        w1=np.full(shape, w1, dtype=np.float64),
        w2=np.full(shape, w2, dtype=np.float64),
        cap_v_eff=np.full(shape, cap_v_eff, dtype=np.float64),
        ret_norm=np.full(shape, ret_norm, dtype=np.float64),
        s_r=np.full(shape, s_r, dtype=np.float64),
        rvol_eligible=np.ones(shape, dtype=bool),
    )


def paper_gaussian(x_grid: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    z = (x_grid - float(mu)) / float(sigma)
    return np.exp(-0.5 * z * z) / (float(sigma) * SQRT_2PI)


def paper_bar_profile(
    *,
    open_px: float,
    high_px: float,
    low_px: float,
    close_px: float,
    volume: float,
    current_close: float,
    atr_floor_t: float,
    rvol_t: float,
    cap_eff_t: float,
    tick_size: float,
    x_grid: np.ndarray,
    dx: float,
    eps_pdf: float = 1.0e-12,
) -> np.ndarray:
    range_k = max(float(high_px) - float(low_px), float(tick_size))
    body_k = abs(float(close_px) - float(open_px))
    body_pct = body_k / range_k
    mean_k = 0.5 * (float(open_px) + float(close_px))
    mu = (mean_k - float(current_close)) / (float(atr_floor_t) + float(tick_size))
    w_rvol = float(rvol_t) / (1.0 + float(rvol_t))
    range_eff = w_rvol * range_k + (1.0 - w_rvol) * float(atr_floor_t)
    sigma_base = range_eff / (4.0 * (float(atr_floor_t) + float(tick_size)))
    sigma1 = max(sigma_base / (1.0 + np.log1p(float(rvol_t))), float(dx))
    sigma2 = max(sigma_base, float(dx))
    w1 = float(np.clip(body_pct, 0.0, 1.0))
    w2 = 1.0 - w1
    pdf = w1 * paper_gaussian(x_grid, mu, sigma1) + w2 * paper_gaussian(x_grid, mu, sigma2)
    pdf /= float(np.sum(pdf) + eps_pdf)
    vprof = min(float(volume), float(cap_eff_t))
    return vprof * pdf


def paper_profile_stats_from_vp(vp_row: np.ndarray, x_grid: np.ndarray, eps_vol: float) -> tuple[float, float]:
    total = float(np.sum(vp_row))
    mu = float(np.dot(vp_row, x_grid) / (total + float(eps_vol)))
    sigma = float(np.sqrt(np.dot(vp_row, (x_grid - mu) ** 2) / (total + float(eps_vol))))
    return mu, sigma
