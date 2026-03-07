from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
from typing import Any

import numpy as np


W_GRID = (60, 120, 390, 1950)
S_BREAK_THR_GRID = (0.50, 0.55, 0.60, 0.65, 0.70)
S_REJECT_THR_GRID = (0.50, 0.55, 0.60, 0.65, 0.70)
D_THR_GRID = (1.0, 1.5, 2.0, 2.5, 3.0)
D_LOW_GRID = (0.5, 1.0, 1.5)
D_HIGH_GRID = (1.5, 2.0, 2.5, 3.0)
A_THR_GRID = (0.4, 0.5, 0.6)
A_ACCEPT_THR_GRID = (0.6, 0.7, 0.8)
RVOL_THR_GRID = (1.0, 1.5, 2.0, 3.0)
DE_THR_GRID = (0.00, 0.05, 0.10, 0.20)
VA_DIST_GRID = (0.2, 0.5, 1.0)
S_BREAK_MID_LOW_GRID = (0.35, 0.40, 0.45)
S_BREAK_MID_HIGH_GRID = (0.50, 0.55, 0.60)
LEV_TARGET_GRID = (1.0, 1.5, 2.0)
EXIT_MODEL_GRID = ("E1", "E2", "E3", "E4", "E5")
E1_ATR_STOP_MULT_GRID = (0.5, 1.0, 1.5, 2.0)
E5_TIME_EXIT_BARS_GRID = (390, 780, 1170)

EXPECTED_BASE_STRATEGY_COUNT = 15120
_SOBOL_SWING_REQUIRED_KEYS = (
    "profile_window_minutes",
    "profile_memory_sessions",
    "deltaeff_threshold",
    "distance_to_poc_atr",
    "acceptance_threshold",
    "rvol_filter",
    "holding_period_days",
)


@dataclass(frozen=True)
class StrategySpec:
    family: str
    W: int
    profile_window_minutes: int | None
    profile_memory_sessions: int | None
    deltaeff_threshold: float | None
    distance_to_poc_atr: float | None
    acceptance_threshold: float | None
    rvol_filter: float | None
    holding_period_days: int | None
    lev_target: float
    exit_model: str
    atr_stop_mult: float | None
    time_exit_bars: int | None
    s_break_thr: float | None
    s_reject_thr: float | None
    d_thr: float | None
    d_low: float | None
    d_high: float | None
    a_thr: float | None
    a_accept_thr: float | None
    rvol_thr: float | None
    de_thr: float | None
    va_dist: float | None
    s_break_mid_low: float | None
    s_break_mid_high: float | None


def _fmt_num(value: float | int | None) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.8f}".rstrip("0").rstrip(".")


def strategy_id(spec: StrategySpec) -> str:
    ordered = [
        ("family", spec.family),
        ("W", spec.W),
        ("profile_window_minutes", spec.profile_window_minutes),
        ("profile_memory_sessions", spec.profile_memory_sessions),
        ("deltaeff_threshold", spec.deltaeff_threshold),
        ("distance_to_poc_atr", spec.distance_to_poc_atr),
        ("acceptance_threshold", spec.acceptance_threshold),
        ("rvol_filter", spec.rvol_filter),
        ("holding_period_days", spec.holding_period_days),
        ("lev", spec.lev_target),
        ("exit", spec.exit_model),
        ("atr_stop_mult", spec.atr_stop_mult),
        ("time_exit_bars", spec.time_exit_bars),
        ("s_break_thr", spec.s_break_thr),
        ("s_reject_thr", spec.s_reject_thr),
        ("d_thr", spec.d_thr),
        ("d_low", spec.d_low),
        ("d_high", spec.d_high),
        ("a_thr", spec.a_thr),
        ("a_accept_thr", spec.a_accept_thr),
        ("rvol_thr", spec.rvol_thr),
        ("de_thr", spec.de_thr),
        ("va_dist", spec.va_dist),
        ("s_break_mid_low", spec.s_break_mid_low),
        ("s_break_mid_high", spec.s_break_mid_high),
    ]
    canonical = "|".join(f"{k}={_fmt_num(v) if k != 'family' and k != 'exit' else v}" for k, v in ordered)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{spec.family}_{digest}"


def _exit_variants() -> list[tuple[str, float | None, int | None]]:
    variants: list[tuple[str, float | None, int | None]] = []
    for m in EXIT_MODEL_GRID:
        if m == "E1":
            for atr_mult in E1_ATR_STOP_MULT_GRID:
                variants.append((m, float(atr_mult), None))
        elif m == "E5":
            for n_bars in E5_TIME_EXIT_BARS_GRID:
                variants.append((m, None, int(n_bars)))
        else:
            variants.append((m, None, None))
    return variants


def generate_strategy_specs(swing_grid: dict[str, Any] | None = None) -> list[StrategySpec]:
    if swing_grid is not None:
        return generate_swing_strategy_specs(
            profile_window_minutes=[int(x) for x in swing_grid["profile_window_minutes"]],
            profile_memory_sessions=[int(x) for x in swing_grid["profile_memory_sessions"]],
            deltaeff_threshold=[float(x) for x in swing_grid["deltaeff_threshold"]],
            distance_to_poc_atr=[float(x) for x in swing_grid["distance_to_poc_atr"]],
            acceptance_threshold=[float(x) for x in swing_grid["acceptance_threshold"]],
            rvol_filter=[float(x) for x in swing_grid["rvol_filter"]],
            holding_period_days=[int(x) for x in swing_grid["holding_period_days"]],
            lev_target=float(swing_grid.get("lev_target", 1.5)),
        )
    specs: list[StrategySpec] = []
    exits = _exit_variants()

    # F1 Breakout continuation
    for W, s_break_thr, rvol_thr, lev, (exit_model, atr_mult, time_bars) in itertools.product(
        W_GRID,
        S_BREAK_THR_GRID,
        RVOL_THR_GRID,
        LEV_TARGET_GRID,
        exits,
    ):
        specs.append(
            StrategySpec(
                family="F1",
                W=int(W),
                profile_window_minutes=None,
                profile_memory_sessions=None,
                deltaeff_threshold=None,
                distance_to_poc_atr=None,
                acceptance_threshold=None,
                rvol_filter=None,
                holding_period_days=None,
                lev_target=float(lev),
                exit_model=exit_model,
                atr_stop_mult=atr_mult,
                time_exit_bars=time_bars,
                s_break_thr=float(s_break_thr),
                s_reject_thr=None,
                d_thr=None,
                d_low=None,
                d_high=None,
                a_thr=None,
                a_accept_thr=None,
                rvol_thr=float(rvol_thr),
                de_thr=None,
                va_dist=None,
                s_break_mid_low=None,
                s_break_mid_high=None,
            )
        )

    # F2 Rejection
    for W, s_reject_thr, a_thr, lev, (exit_model, atr_mult, time_bars) in itertools.product(
        W_GRID,
        S_REJECT_THR_GRID,
        A_THR_GRID,
        LEV_TARGET_GRID,
        exits,
    ):
        specs.append(
            StrategySpec(
                family="F2",
                W=int(W),
                profile_window_minutes=None,
                profile_memory_sessions=None,
                deltaeff_threshold=None,
                distance_to_poc_atr=None,
                acceptance_threshold=None,
                rvol_filter=None,
                holding_period_days=None,
                lev_target=float(lev),
                exit_model=exit_model,
                atr_stop_mult=atr_mult,
                time_exit_bars=time_bars,
                s_break_thr=None,
                s_reject_thr=float(s_reject_thr),
                d_thr=None,
                d_low=None,
                d_high=None,
                a_thr=float(a_thr),
                a_accept_thr=None,
                rvol_thr=None,
                de_thr=None,
                va_dist=None,
                s_break_mid_low=None,
                s_break_mid_high=None,
            )
        )

    # F3 Deviation mean reversion
    for W, d_thr, a_accept, lev, (exit_model, atr_mult, time_bars) in itertools.product(
        W_GRID,
        D_THR_GRID,
        A_ACCEPT_THR_GRID,
        LEV_TARGET_GRID,
        exits,
    ):
        specs.append(
            StrategySpec(
                family="F3",
                W=int(W),
                profile_window_minutes=None,
                profile_memory_sessions=None,
                deltaeff_threshold=None,
                distance_to_poc_atr=None,
                acceptance_threshold=None,
                rvol_filter=None,
                holding_period_days=None,
                lev_target=float(lev),
                exit_model=exit_model,
                atr_stop_mult=atr_mult,
                time_exit_bars=time_bars,
                s_break_thr=None,
                s_reject_thr=None,
                d_thr=float(d_thr),
                d_low=None,
                d_high=None,
                a_thr=None,
                a_accept_thr=float(a_accept),
                rvol_thr=None,
                de_thr=None,
                va_dist=None,
                s_break_mid_low=None,
                s_break_mid_high=None,
            )
        )

    # F4 Acceptance breakout
    for W, va_dist, sb_low, sb_high, lev, (exit_model, atr_mult, time_bars) in itertools.product(
        W_GRID,
        VA_DIST_GRID,
        S_BREAK_MID_LOW_GRID,
        S_BREAK_MID_HIGH_GRID,
        LEV_TARGET_GRID,
        exits,
    ):
        if float(sb_high) <= float(sb_low):
            continue
        specs.append(
            StrategySpec(
                family="F4",
                W=int(W),
                profile_window_minutes=None,
                profile_memory_sessions=None,
                deltaeff_threshold=None,
                distance_to_poc_atr=None,
                acceptance_threshold=None,
                rvol_filter=None,
                holding_period_days=None,
                lev_target=float(lev),
                exit_model=exit_model,
                atr_stop_mult=atr_mult,
                time_exit_bars=time_bars,
                s_break_thr=None,
                s_reject_thr=None,
                d_thr=None,
                d_low=None,
                d_high=None,
                a_thr=None,
                a_accept_thr=None,
                rvol_thr=None,
                de_thr=None,
                va_dist=float(va_dist),
                s_break_mid_low=float(sb_low),
                s_break_mid_high=float(sb_high),
            )
        )

    # F5 Delta continuation
    for W, de_thr, rvol_thr, lev, (exit_model, atr_mult, time_bars) in itertools.product(
        W_GRID,
        DE_THR_GRID,
        RVOL_THR_GRID,
        LEV_TARGET_GRID,
        exits,
    ):
        specs.append(
            StrategySpec(
                family="F5",
                W=int(W),
                profile_window_minutes=None,
                profile_memory_sessions=None,
                deltaeff_threshold=None,
                distance_to_poc_atr=None,
                acceptance_threshold=None,
                rvol_filter=None,
                holding_period_days=None,
                lev_target=float(lev),
                exit_model=exit_model,
                atr_stop_mult=atr_mult,
                time_exit_bars=time_bars,
                s_break_thr=None,
                s_reject_thr=None,
                d_thr=None,
                d_low=None,
                d_high=None,
                a_thr=None,
                a_accept_thr=None,
                rvol_thr=float(rvol_thr),
                de_thr=float(de_thr),
                va_dist=None,
                s_break_mid_low=None,
                s_break_mid_high=None,
            )
        )

    # F6 Profile rotation
    for W, a_thr, d_low, d_high, lev, (exit_model, atr_mult, time_bars) in itertools.product(
        W_GRID,
        A_THR_GRID,
        D_LOW_GRID,
        D_HIGH_GRID,
        LEV_TARGET_GRID,
        exits,
    ):
        if float(d_high) <= float(d_low):
            continue
        specs.append(
            StrategySpec(
                family="F6",
                W=int(W),
                profile_window_minutes=None,
                profile_memory_sessions=None,
                deltaeff_threshold=None,
                distance_to_poc_atr=None,
                acceptance_threshold=None,
                rvol_filter=None,
                holding_period_days=None,
                lev_target=float(lev),
                exit_model=exit_model,
                atr_stop_mult=atr_mult,
                time_exit_bars=time_bars,
                s_break_thr=None,
                s_reject_thr=None,
                d_thr=None,
                d_low=float(d_low),
                d_high=float(d_high),
                a_thr=float(a_thr),
                a_accept_thr=None,
                rvol_thr=None,
                de_thr=None,
                va_dist=None,
                s_break_mid_low=None,
                s_break_mid_high=None,
            )
        )

    specs.sort(key=lambda s: (s.family, s.W, strategy_id(s)))
    validate_grid_cardinality(specs)
    return specs


def generate_swing_strategy_specs(
    *,
    profile_window_minutes: list[int],
    profile_memory_sessions: list[int],
    deltaeff_threshold: list[float],
    distance_to_poc_atr: list[float],
    acceptance_threshold: list[float],
    rvol_filter: list[float],
    holding_period_days: list[int],
    lev_target: float = 1.5,
) -> list[StrategySpec]:
    specs: list[StrategySpec] = []
    for w_min, mem, de_th, dist, acc, rvf, hold_days in itertools.product(
        sorted(int(x) for x in profile_window_minutes),
        sorted(int(x) for x in profile_memory_sessions),
        sorted(float(x) for x in deltaeff_threshold),
        sorted(float(x) for x in distance_to_poc_atr),
        sorted(float(x) for x in acceptance_threshold),
        sorted(float(x) for x in rvol_filter),
        sorted(int(x) for x in holding_period_days),
    ):
        w_eff = int(w_min * mem)
        specs.append(
            StrategySpec(
                family="SWING",
                W=w_eff,
                profile_window_minutes=int(w_min),
                profile_memory_sessions=int(mem),
                deltaeff_threshold=float(de_th),
                distance_to_poc_atr=float(dist),
                acceptance_threshold=float(acc),
                rvol_filter=float(rvf),
                holding_period_days=int(hold_days),
                lev_target=float(lev_target),
                exit_model="E5",
                atr_stop_mult=None,
                time_exit_bars=int(hold_days * 390),
                s_break_thr=None,
                s_reject_thr=None,
                d_thr=None,
                d_low=None,
                d_high=None,
                a_thr=None,
                a_accept_thr=None,
                rvol_thr=None,
                de_thr=None,
                va_dist=None,
                s_break_mid_low=None,
                s_break_mid_high=None,
            )
        )

    specs.sort(
        key=lambda s: (
            int(s.profile_window_minutes or 0),
            int(s.profile_memory_sessions or 0),
            float(s.deltaeff_threshold or 0.0),
            float(s.distance_to_poc_atr or 0.0),
            float(s.acceptance_threshold or 0.0),
            float(s.rvol_filter or 0.0),
            int(s.holding_period_days or 0),
            strategy_id(s),
        )
    )
    return specs


def _is_power_of_two(n: int) -> bool:
    return int(n) > 0 and (int(n) & (int(n) - 1)) == 0


def _map_sobol_value(u01: float, low: float, high: float) -> float:
    uu = float(np.clip(float(u01), 0.0, 1.0))
    return float(low + uu * (high - low))


def _map_sobol_int(u01: float, low: int, high: int) -> int:
    if int(high) < int(low):
        raise RuntimeError(f"Invalid integer Sobol range: low={low}, high={high}")
    width = int(high) - int(low) + 1
    idx = int(np.floor(np.clip(float(u01), 0.0, 1.0 - np.finfo(np.float64).eps) * width))
    return int(low + min(max(idx, 0), width - 1))


def generate_sobol_strategy_specs(
    *,
    n_samples: int,
    param_ranges: dict[str, tuple[float, float]],
    seed: int = 42,
    lev_target: float = 1.5,
) -> list[StrategySpec]:
    assert _is_power_of_two(int(n_samples))
    if not _is_power_of_two(int(n_samples)):
        raise RuntimeError(
            f"Sobol n_samples must be power-of-two for random_base2; got {int(n_samples)}"
        )

    missing = [k for k in _SOBOL_SWING_REQUIRED_KEYS if k not in param_ranges]
    if missing:
        raise RuntimeError(
            f"Sobol param_ranges missing required keys: {missing}"
        )

    for k in _SOBOL_SWING_REQUIRED_KEYS:
        lo, hi = param_ranges[k]
        if not (np.isfinite(float(lo)) and np.isfinite(float(hi))):
            raise RuntimeError(f"Sobol param_ranges[{k}] contains non-finite bounds: {(lo, hi)}")
        if float(hi) < float(lo):
            raise RuntimeError(f"Sobol param_ranges[{k}] has high < low: {(lo, hi)}")

    try:
        from scipy.stats import qmc  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Sobol sampling requires scipy. Install with: ./.venv/bin/python -m pip install scipy"
        ) from exc

    dims = len(_SOBOL_SWING_REQUIRED_KEYS)
    if int(dims) > 20:
        raise RuntimeError("Sobol dimension >20 not supported in this engine")
    sampler = qmc.Sobol(d=dims, scramble=True, seed=int(seed))
    m = int(np.log2(int(n_samples)))
    points = np.asarray(sampler.random_base2(m=m), dtype=np.float64)
    if points.shape != (int(n_samples), dims):
        raise RuntimeError(
            f"Unexpected Sobol point shape: got={points.shape}, expected={(int(n_samples), dims)}"
        )

    specs: list[StrategySpec] = []
    for i in range(points.shape[0]):
        u = points[i]
        w_min = _map_sobol_int(
            u[0],
            int(param_ranges["profile_window_minutes"][0]),
            int(param_ranges["profile_window_minutes"][1]),
        )
        mem = _map_sobol_int(
            u[1],
            int(param_ranges["profile_memory_sessions"][0]),
            int(param_ranges["profile_memory_sessions"][1]),
        )
        de_th = _map_sobol_value(
            u[2],
            float(param_ranges["deltaeff_threshold"][0]),
            float(param_ranges["deltaeff_threshold"][1]),
        )
        dist = _map_sobol_value(
            u[3],
            float(param_ranges["distance_to_poc_atr"][0]),
            float(param_ranges["distance_to_poc_atr"][1]),
        )
        acc = _map_sobol_value(
            u[4],
            float(param_ranges["acceptance_threshold"][0]),
            float(param_ranges["acceptance_threshold"][1]),
        )
        rvf = _map_sobol_value(
            u[5],
            float(param_ranges["rvol_filter"][0]),
            float(param_ranges["rvol_filter"][1]),
        )
        hold_days = _map_sobol_int(
            u[6],
            int(param_ranges["holding_period_days"][0]),
            int(param_ranges["holding_period_days"][1]),
        )

        w_eff = int(w_min * mem)
        specs.append(
            StrategySpec(
                family="SWING",
                W=w_eff,
                profile_window_minutes=int(w_min),
                profile_memory_sessions=int(mem),
                deltaeff_threshold=float(de_th),
                distance_to_poc_atr=float(dist),
                acceptance_threshold=float(acc),
                rvol_filter=float(rvf),
                holding_period_days=int(hold_days),
                lev_target=float(lev_target),
                exit_model="E5",
                atr_stop_mult=None,
                time_exit_bars=int(hold_days * 390),
                s_break_thr=None,
                s_reject_thr=None,
                d_thr=None,
                d_low=None,
                d_high=None,
                a_thr=None,
                a_accept_thr=None,
                rvol_thr=None,
                de_thr=None,
                va_dist=None,
                s_break_mid_low=None,
                s_break_mid_high=None,
            )
        )

    return specs


def validate_grid_cardinality(specs: list[StrategySpec]) -> None:
    n = int(len(specs))
    if n != EXPECTED_BASE_STRATEGY_COUNT:
        raise RuntimeError(
            f"STRATEGY_GRID_CARDINALITY_ERROR: expected={EXPECTED_BASE_STRATEGY_COUNT} got={n}"
        )


def strategy_payload(spec: StrategySpec) -> dict[str, Any]:
    return {
        "strategy_id": strategy_id(spec),
        "family": spec.family,
        "W": int(spec.W),
        "profile_window_minutes": None
        if spec.profile_window_minutes is None
        else int(spec.profile_window_minutes),
        "profile_memory_sessions": None
        if spec.profile_memory_sessions is None
        else int(spec.profile_memory_sessions),
        "deltaeff_threshold": None
        if spec.deltaeff_threshold is None
        else float(spec.deltaeff_threshold),
        "distance_to_poc_atr": None
        if spec.distance_to_poc_atr is None
        else float(spec.distance_to_poc_atr),
        "acceptance_threshold": None
        if spec.acceptance_threshold is None
        else float(spec.acceptance_threshold),
        "rvol_filter": None if spec.rvol_filter is None else float(spec.rvol_filter),
        "holding_period_days": None
        if spec.holding_period_days is None
        else int(spec.holding_period_days),
        "lev_target": float(spec.lev_target),
        "exit_model": str(spec.exit_model),
        "atr_stop_mult": None if spec.atr_stop_mult is None else float(spec.atr_stop_mult),
        "time_exit_bars": None if spec.time_exit_bars is None else int(spec.time_exit_bars),
        "s_break_thr": None if spec.s_break_thr is None else float(spec.s_break_thr),
        "s_reject_thr": None if spec.s_reject_thr is None else float(spec.s_reject_thr),
        "d_thr": None if spec.d_thr is None else float(spec.d_thr),
        "d_low": None if spec.d_low is None else float(spec.d_low),
        "d_high": None if spec.d_high is None else float(spec.d_high),
        "a_thr": None if spec.a_thr is None else float(spec.a_thr),
        "a_accept_thr": None if spec.a_accept_thr is None else float(spec.a_accept_thr),
        "rvol_thr": None if spec.rvol_thr is None else float(spec.rvol_thr),
        "de_thr": None if spec.de_thr is None else float(spec.de_thr),
        "va_dist": None if spec.va_dist is None else float(spec.va_dist),
        "s_break_mid_low": None if spec.s_break_mid_low is None else float(spec.s_break_mid_low),
        "s_break_mid_high": None if spec.s_break_mid_high is None else float(spec.s_break_mid_high),
    }


def family_counts(specs: list[StrategySpec]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in specs:
        out[s.family] = int(out.get(s.family, 0) + 1)
    return out


def deterministic_jitter_seconds(run_name: str, seed: int) -> int:
    token = f"{str(run_name)}{int(seed)}".encode("utf-8")
    h = int.from_bytes(hashlib.sha256(token).digest()[:8], byteorder="big", signed=False)
    return int(10 + (h % 21))


def sigmoid(x: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x64))
