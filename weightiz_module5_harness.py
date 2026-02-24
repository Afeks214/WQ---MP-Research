"""
Weightiz Institutional Engine - Module 5 Part 2 (Validation Harness)
=====================================================================

Validation harness and research orchestrator:
- Pandas IO boundary for minute OHLCV ingestion/alignment.
- Leakage-safe WF/CPCV split generation with purge+embargo.
- Adversarial stress perturbations on cloned tensor states.
- Deterministic orchestration of Module 2 -> Module 3 -> Module 4.
- Close-to-close daily return compression for candidate equity (overnight PnL preserved).
- Artifact export and statistical verdict wiring to Module 5 Part 1.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import traceback
from typing import Any, Callable

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover - runtime guard
    pd = None  # type: ignore[assignment]

from weightiz_module1_core import (
    EngineConfig,
    Phase,
    ProfileStatIdx,
    ScoreIdx,
    TensorState,
    preallocate_state,
    validate_loaded_market_slice,
    validate_state_hard,
)
from weightiz_module2_core import Module2Config, run_weightiz_profile_engine
from weightiz_module3_structure import ContextIdx, Module3Config, Module3Output, run_module3_structural_aggregation
from weightiz_module4_strategy_funnel import Module4Config, Module4Output, RegimeIdx, run_module4_strategy_funnel
from weightiz_module5_stats import (
    deflated_sharpe_ratio,
    model_confidence_set,
    pbo_cscv,
    spa_test,
    white_reality_check,
)


@dataclass(frozen=True)
class Module5HarnessConfig:
    seed: int = 97
    timezone: str = "America/New_York"
    freq: str = "1min"
    min_asset_coverage: float = 0.80
    purge_bars: int = 60
    embargo_bars: int = 30
    wf_train_sessions: int = 60
    wf_test_sessions: int = 20
    wf_step_sessions: int = 20
    cpcv_slices: int = 10
    cpcv_k_test: int = 5
    parallel_backend: str = "process_pool"
    parallel_workers: int = max(1, (os.cpu_count() or 2) - 1)
    stress_profile: str = "baseline_mild_severe"
    max_ram_utilization_frac: float = 0.70
    enforce_lookahead_guard: bool = True
    report_dir: str = "./artifacts/module5_harness"
    fail_on_non_finite: bool = True
    daily_return_min_days: int = 60
    benchmark_symbol: str = "SPY"
    export_micro_diagnostics: bool = False
    micro_diag_mode: str = "events_only"
    micro_diag_symbols: tuple[str, ...] = ()
    micro_diag_session_ids: tuple[int, ...] = ()
    micro_diag_trade_window_pre: int = 90
    micro_diag_trade_window_post: int = 180
    micro_diag_export_block_profiles: bool = True
    micro_diag_export_funnel: bool = True
    micro_diag_max_rows: int = 5_000_000


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    m2_idx: int
    m3_idx: int
    m4_idx: int
    enabled_assets_mask: np.ndarray
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class SplitSpec:
    split_id: str
    mode: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    purge_idx: np.ndarray
    embargo_idx: np.ndarray
    session_train_bounds: tuple[int, int]
    session_test_bounds: tuple[int, int]


@dataclass(frozen=True)
class StressScenario:
    scenario_id: str
    name: str
    missing_burst_prob: float
    missing_burst_min: int
    missing_burst_max: int
    jitter_sigma_bps: float
    slippage_mult: float
    enabled: bool = True


@dataclass
class HarnessOutput:
    candidate_results: list[dict[str, object]]
    daily_returns_matrix: np.ndarray
    daily_benchmark_returns: np.ndarray
    stats_verdict: dict[str, object]
    artifact_paths: dict[str, str]
    run_manifest: dict[str, object]


@dataclass(frozen=True)
class _GroupTask:
    group_id: str
    split_idx: int
    scenario_idx: int
    m2_idx: int
    m3_idx: int
    candidate_indices: tuple[int, ...]


def _require_pandas() -> Any:
    if pd is None:
        raise RuntimeError("pandas is required for Module 5 harness IO/export boundary")
    return pd


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float64, np.float32, np.float16, np.float_)):
        return float(obj)
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8, np.int_)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _stable_hash_obj(obj: Any) -> str:
    payload = json.dumps(_to_jsonable(obj), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _git_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def _seed_for_task(base_seed: int, *parts: str) -> int:
    h = hashlib.sha256()
    h.update(str(base_seed).encode("utf-8"))
    for p in parts:
        h.update(b"|")
        h.update(p.encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "little", signed=False) % (2**32 - 1)


def _assert_finite(name: str, arr: np.ndarray) -> None:
    if not np.all(np.isfinite(arr)):
        bad = np.argwhere(~np.isfinite(arr))[:8]
        raise RuntimeError(f"{name} contains non-finite values at indices {bad.tolist()}")


def _clone_state(state: TensorState) -> TensorState:
    # Deep copy preserves strict immutability of base_state across workers/tasks.
    return copy.deepcopy(state)


def _clone_m3(m3: Module3Output) -> Module3Output:
    return Module3Output(
        block_id_t=m3.block_id_t.copy(),
        block_seq_t=m3.block_seq_t.copy(),
        block_end_flag_t=m3.block_end_flag_t.copy(),
        block_start_t_index_t=m3.block_start_t_index_t.copy(),
        block_end_t_index_t=m3.block_end_t_index_t.copy(),
        block_features_tak=m3.block_features_tak.copy(),
        block_valid_ta=m3.block_valid_ta.copy(),
        context_tac=m3.context_tac.copy(),
        context_valid_ta=m3.context_valid_ta.copy(),
        context_source_t_index_ta=m3.context_source_t_index_ta.copy(),
    )


def _available_memory_bytes() -> int:
    # Unix/macOS fast path.
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        return page_size * pages
    except Exception:
        return 8 * 1024 * 1024 * 1024


def _estimate_state_bytes(T: int, A: int, B: int) -> int:
    # Conservative rough estimate for one full tensor state + outputs.
    core = T * A * B * 8 * 3  # vp, vp_delta, temp profile buffers
    ta = T * A * 8 * 24
    t = T * 8 * 16
    overhead = 128 * 1024 * 1024
    return int(core + ta + t + overhead)


def _find_col(df: Any, candidates: tuple[str, ...], name: str) -> str:
    cols = {str(c).strip().lower(): str(c) for c in df.columns}
    for cand in candidates:
        if cand in cols:
            return cols[cand]
    raise RuntimeError(f"Missing required column '{name}' in input file")


def _load_asset_frame(path: str, tz_name: str) -> Any:
    pdx = _require_pandas()
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"Data path does not exist: {path}")

    if p.suffix.lower() == ".parquet":
        df = pdx.read_parquet(p)
    else:
        df = pdx.read_csv(p)

    ts_col = _find_col(df, ("timestamp", "ts", "datetime", "date", "time"), "timestamp")
    o_col = _find_col(df, ("open", "o"), "open")
    h_col = _find_col(df, ("high", "h"), "high")
    l_col = _find_col(df, ("low", "l"), "low")
    c_col = _find_col(df, ("close", "c"), "close")
    v_col = _find_col(df, ("volume", "vol", "v"), "volume")

    ts = pdx.to_datetime(df[ts_col], utc=True, errors="coerce")
    keep = ts.notna().to_numpy(dtype=bool)
    if not np.any(keep):
        raise RuntimeError(f"No parseable timestamps in {path}")

    out = pdx.DataFrame(
        {
            "timestamp": ts[keep].dt.tz_convert(tz_name).dt.floor("min"),
            "open": pdx.to_numeric(df.loc[keep, o_col], errors="coerce"),
            "high": pdx.to_numeric(df.loc[keep, h_col], errors="coerce"),
            "low": pdx.to_numeric(df.loc[keep, l_col], errors="coerce"),
            "close": pdx.to_numeric(df.loc[keep, c_col], errors="coerce"),
            "volume": pdx.to_numeric(df.loc[keep, v_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    out = out.drop_duplicates(subset=["timestamp"], keep="last")
    out = out.set_index("timestamp")
    return out


def _compute_bar_valid(
    open_px: np.ndarray,
    high_px: np.ndarray,
    low_px: np.ndarray,
    close_px: np.ndarray,
    volume: np.ndarray,
) -> np.ndarray:
    finite = (
        np.isfinite(open_px)
        & np.isfinite(high_px)
        & np.isfinite(low_px)
        & np.isfinite(close_px)
        & np.isfinite(volume)
    )
    phys = (
        (high_px >= low_px)
        & (high_px >= open_px)
        & (high_px >= close_px)
        & (low_px <= open_px)
        & (low_px <= close_px)
        & (volume >= 0.0)
    )
    return finite & phys


def _ingest_master_aligned(
    data_paths: list[str],
    symbols: list[str],
    engine_cfg: EngineConfig,
    harness_cfg: Module5HarnessConfig,
    data_loader_func: Callable[[str, str], Any] | None = None,
) -> tuple[TensorState, np.ndarray, list[str], np.ndarray, dict[str, Any], np.ndarray]:
    if len(data_paths) != len(symbols):
        raise RuntimeError("data_paths and symbols lengths must match")

    pdx = _require_pandas()

    loader = data_loader_func if data_loader_func is not None else _load_asset_frame

    raw_frames: list[Any] = []
    for p in data_paths:
        raw_frames.append(loader(p, harness_cfg.timezone))

    master_idx = raw_frames[0].index
    for fr in raw_frames[1:]:
        master_idx = master_idx.union(fr.index)
    master_idx = master_idx.sort_values()

    T = int(master_idx.shape[0])
    A0 = int(len(symbols))
    open_ta = np.full((T, A0), np.nan, dtype=np.float64)
    high_ta = np.full((T, A0), np.nan, dtype=np.float64)
    low_ta = np.full((T, A0), np.nan, dtype=np.float64)
    close_ta = np.full((T, A0), np.nan, dtype=np.float64)
    vol_ta = np.full((T, A0), np.nan, dtype=np.float64)

    for a, fr in enumerate(raw_frames):
        re = fr.reindex(master_idx)
        open_ta[:, a] = re["open"].to_numpy(dtype=np.float64)
        high_ta[:, a] = re["high"].to_numpy(dtype=np.float64)
        low_ta[:, a] = re["low"].to_numpy(dtype=np.float64)
        close_ta[:, a] = re["close"].to_numpy(dtype=np.float64)
        vol_ta[:, a] = re["volume"].to_numpy(dtype=np.float64)

    bar_valid_ta = _compute_bar_valid(open_ta, high_ta, low_ta, close_ta, vol_ta)
    coverage = np.mean(bar_valid_ta, axis=0)
    keep_assets = coverage >= float(harness_cfg.min_asset_coverage)

    if np.sum(keep_assets) < 2:
        raise RuntimeError(
            f"Coverage filter removed too many assets: kept={int(np.sum(keep_assets))}, required>=2"
        )

    keep_idx = np.where(keep_assets)[0]
    keep_symbols = [symbols[i] for i in keep_idx.tolist()]

    tick = np.asarray(engine_cfg.tick_size, dtype=np.float64)
    if tick.shape != (A0,):
        raise RuntimeError(
            f"engine_cfg.tick_size shape mismatch: got {tick.shape}, expected {(A0,)}"
        )

    open_keep = open_ta[:, keep_idx]
    high_keep = high_ta[:, keep_idx]
    low_keep = low_ta[:, keep_idx]
    close_keep = close_ta[:, keep_idx]
    vol_keep = vol_ta[:, keep_idx]
    bar_keep = bar_valid_ta[:, keep_idx]
    tick_keep = tick[keep_idx]

    ts_ns = master_idx.tz_convert("UTC").asi8.astype(np.int64)

    cfg = replace(engine_cfg, T=T, A=int(keep_idx.size), tick_size=tick_keep.copy())
    state = preallocate_state(ts_ns=ts_ns, cfg=cfg, symbols=tuple(keep_symbols))

    state.open_px[:, :] = open_keep
    state.high_px[:, :] = high_keep
    state.low_px[:, :] = low_keep
    state.close_px[:, :] = close_keep
    state.volume[:, :] = vol_keep
    state.bar_valid[:, :] = bar_keep

    # Required finite placeholders for Module 1 slice validation; Module 2 overwrites causally.
    atr0 = np.maximum(4.0 * tick_keep[None, :], 1e-12)
    state.rvol[:, :] = np.where(bar_keep, 1.0, np.nan)
    state.atr_floor[:, :] = np.where(bar_keep, atr0, np.nan)

    validate_loaded_market_slice(state, 0, state.cfg.T)
    validate_state_hard(state)

    ingest_meta = {
        "master_rows": T,
        "assets_input": A0,
        "assets_kept": int(keep_idx.size),
        "coverage": coverage.tolist(),
        "symbols_kept": keep_symbols,
    }

    return state, keep_idx, keep_symbols, master_idx.asi8.astype(np.int64), ingest_meta, tick_keep


def _session_bounds(session_id: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sid = np.asarray(session_id, dtype=np.int64)
    T = int(sid.shape[0])
    starts = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]]).astype(np.int64)
    ends = np.r_[starts[1:], T].astype(np.int64)
    sessions = sid[starts]
    return starts, ends, sessions


def _sessions_to_idx(session_id: np.ndarray, sessions: np.ndarray) -> np.ndarray:
    mask = np.isin(session_id, sessions)
    return np.flatnonzero(mask).astype(np.int64)


def _contiguous_segments(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if idx.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    cut = np.flatnonzero(np.diff(idx) > 1) + 1
    starts = np.r_[idx[0], idx[cut]]
    ends = np.r_[idx[cut - 1] + 1, idx[-1] + 1]
    return starts.astype(np.int64), ends.astype(np.int64)


def _apply_purge_embargo(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    T: int,
    purge_bars: int,
    embargo_bars: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tr = np.unique(np.asarray(train_idx, dtype=np.int64))
    te = np.unique(np.asarray(test_idx, dtype=np.int64))

    seg_starts, _seg_ends = _contiguous_segments(te)

    purge_mask = np.zeros(T, dtype=bool)
    for t0 in seg_starts.tolist():
        lo = max(0, int(t0) - int(purge_bars))
        hi = int(t0)
        if hi > lo:
            purge_mask[lo:hi] = True

    embargo_mask = np.zeros(T, dtype=bool)
    for t0 in seg_starts.tolist():
        lo = int(t0)
        hi = min(T, int(t0) + int(embargo_bars))
        if hi > lo:
            embargo_mask[lo:hi] = True

    purge_idx = np.flatnonzero(purge_mask & np.isin(np.arange(T, dtype=np.int64), tr)).astype(np.int64)
    embargo_idx = np.flatnonzero(embargo_mask & np.isin(np.arange(T, dtype=np.int64), te)).astype(np.int64)

    tr2 = tr[~np.isin(tr, purge_idx)]
    te2 = te[~np.isin(te, embargo_idx)]

    return tr2.astype(np.int64), te2.astype(np.int64), purge_idx, embargo_idx


def _generate_wf_splits(state: TensorState, cfg: Module5HarnessConfig) -> list[SplitSpec]:
    starts, ends, sessions = _session_bounds(state.session_id)
    n_s = int(sessions.size)

    train_n = int(cfg.wf_train_sessions)
    test_n = int(cfg.wf_test_sessions)
    step_n = int(cfg.wf_step_sessions)

    out: list[SplitSpec] = []
    if n_s < train_n + test_n:
        return out

    sid = state.session_id.astype(np.int64)
    fold = 0
    s0 = 0
    while s0 + train_n + test_n <= n_s:
        tr_s = sessions[s0 : s0 + train_n]
        te_s = sessions[s0 + train_n : s0 + train_n + test_n]

        tr_idx = _sessions_to_idx(sid, tr_s)
        te_idx = _sessions_to_idx(sid, te_s)

        tr_idx, te_idx, purge_idx, embargo_idx = _apply_purge_embargo(
            tr_idx,
            te_idx,
            state.cfg.T,
            cfg.purge_bars,
            cfg.embargo_bars,
        )

        out.append(
            SplitSpec(
                split_id=f"wf_{fold:03d}",
                mode="wf",
                train_idx=tr_idx,
                test_idx=te_idx,
                purge_idx=purge_idx,
                embargo_idx=embargo_idx,
                session_train_bounds=(int(tr_s[0]), int(tr_s[-1])),
                session_test_bounds=(int(te_s[0]), int(te_s[-1])),
            )
        )
        fold += 1
        s0 += max(1, step_n)

    return out


def _generate_cpcv_splits(state: TensorState, cfg: Module5HarnessConfig) -> list[SplitSpec]:
    starts, ends, sessions = _session_bounds(state.session_id)
    n_s = int(sessions.size)
    S = int(cfg.cpcv_slices)
    k = int(cfg.cpcv_k_test)

    if S < 2 or k < 1 or k >= S:
        raise RuntimeError(f"Invalid CPCV params: slices={S}, k_test={k}")
    if n_s < S:
        return []

    # Deterministic contiguous session groups.
    groups = np.array_split(np.arange(n_s, dtype=np.int64), S)
    out: list[SplitSpec] = []
    sid = state.session_id.astype(np.int64)

    comb_iter = itertools.combinations(range(S), k)
    for i, test_grp_idx in enumerate(comb_iter):
        test_loc = np.concatenate([groups[g] for g in test_grp_idx if groups[g].size > 0]).astype(np.int64)
        if test_loc.size == 0:
            continue
        train_loc_mask = np.ones(n_s, dtype=bool)
        train_loc_mask[test_loc] = False
        train_loc = np.where(train_loc_mask)[0].astype(np.int64)
        if train_loc.size == 0:
            continue

        tr_s = sessions[train_loc]
        te_s = sessions[test_loc]

        tr_idx = _sessions_to_idx(sid, tr_s)
        te_idx = _sessions_to_idx(sid, te_s)

        tr_idx, te_idx, purge_idx, embargo_idx = _apply_purge_embargo(
            tr_idx,
            te_idx,
            state.cfg.T,
            cfg.purge_bars,
            cfg.embargo_bars,
        )

        out.append(
            SplitSpec(
                split_id=f"cpcv_{i:03d}",
                mode="cpcv",
                train_idx=tr_idx,
                test_idx=te_idx,
                purge_idx=purge_idx,
                embargo_idx=embargo_idx,
                session_train_bounds=(int(np.min(tr_s)), int(np.max(tr_s))),
                session_test_bounds=(int(np.min(te_s)), int(np.max(te_s))),
            )
        )

    return out


def _validate_split(spec: SplitSpec, enforce_guard: bool) -> None:
    tr = np.asarray(spec.train_idx, dtype=np.int64)
    te = np.asarray(spec.test_idx, dtype=np.int64)

    if tr.size == 0 or te.size == 0:
        raise RuntimeError(f"Split {spec.split_id} has empty train or test index set")
    if np.any(np.diff(tr) < 0) or np.any(np.diff(te) < 0):
        raise RuntimeError(f"Split {spec.split_id} indices must be sorted")
    if tr.size != np.unique(tr).size or te.size != np.unique(te).size:
        raise RuntimeError(f"Split {spec.split_id} indices must be unique")
    inter = np.intersect1d(tr, te)
    if inter.size > 0:
        raise RuntimeError(f"Split {spec.split_id} leakage: train/test overlap exists")

    if enforce_guard:
        # Bar-level guard; modules are causal, but index-level leakage must still be absent.
        if np.min(te) <= np.max(np.intersect1d(tr, te, assume_unique=False), initial=-1):
            raise RuntimeError(f"Split {spec.split_id} look-ahead guard violated")


def _default_stress_scenarios(cfg: Module5HarnessConfig) -> list[StressScenario]:
    if cfg.stress_profile == "baseline_mild_severe":
        return [
            StressScenario(
                scenario_id="baseline",
                name="baseline",
                missing_burst_prob=0.0,
                missing_burst_min=0,
                missing_burst_max=0,
                jitter_sigma_bps=0.0,
                slippage_mult=1.0,
                enabled=True,
            ),
            StressScenario(
                scenario_id="mild",
                name="mild",
                missing_burst_prob=0.0005,
                missing_burst_min=2,
                missing_burst_max=5,
                jitter_sigma_bps=1.5,
                slippage_mult=1.5,
                enabled=True,
            ),
            StressScenario(
                scenario_id="severe",
                name="severe",
                missing_burst_prob=0.0020,
                missing_burst_min=5,
                missing_burst_max=20,
                jitter_sigma_bps=4.0,
                slippage_mult=3.0,
                enabled=True,
            ),
        ]
    raise RuntimeError(f"Unsupported stress_profile: {cfg.stress_profile}")


def _apply_split_domain_mask(state: TensorState, split: SplitSpec) -> np.ndarray:
    T = state.cfg.T
    active_t = np.zeros(T, dtype=bool)
    active_t[split.train_idx] = True
    active_t[split.test_idx] = True

    inactive = ~active_t
    if np.any(inactive):
        state.open_px[inactive] = np.nan
        state.high_px[inactive] = np.nan
        state.low_px[inactive] = np.nan
        state.close_px[inactive] = np.nan
        state.volume[inactive] = np.nan
        state.rvol[inactive] = np.nan
        state.atr_floor[inactive] = np.nan
        state.bar_valid[inactive] = False

    return active_t


def _apply_missing_bursts(
    state: TensorState,
    active_t: np.ndarray,
    scenario: StressScenario,
    rng: np.random.Generator,
) -> None:
    if scenario.missing_burst_prob <= 0.0 or scenario.missing_burst_max <= 0:
        return

    T, A = state.bar_valid.shape
    start_mask = (rng.random((T, A)) < float(scenario.missing_burst_prob)) & active_t[:, None]
    starts = np.argwhere(start_mask)

    lo_len = int(max(1, scenario.missing_burst_min))
    hi_len = int(max(lo_len, scenario.missing_burst_max))

    for t0, a in starts.tolist():
        L = int(rng.integers(lo_len, hi_len + 1))
        t1 = min(T, int(t0) + L)
        state.open_px[t0:t1, a] = np.nan
        state.high_px[t0:t1, a] = np.nan
        state.low_px[t0:t1, a] = np.nan
        state.close_px[t0:t1, a] = np.nan
        state.volume[t0:t1, a] = np.nan
        state.rvol[t0:t1, a] = np.nan
        state.atr_floor[t0:t1, a] = np.nan
        state.bar_valid[t0:t1, a] = False


def _apply_jitter(
    state: TensorState,
    active_t: np.ndarray,
    scenario: StressScenario,
    rng: np.random.Generator,
) -> None:
    sigma_bps = float(scenario.jitter_sigma_bps)
    if sigma_bps <= 0.0:
        return

    eps = rng.normal(
        loc=0.0,
        scale=sigma_bps / 1e4,
        size=state.open_px.shape,
    ).astype(np.float64)
    mask = active_t[:, None] & state.bar_valid

    mult = 1.0 + eps
    state.open_px = np.where(mask, state.open_px * mult, state.open_px)
    state.high_px = np.where(mask, state.high_px * mult, state.high_px)
    state.low_px = np.where(mask, state.low_px * mult, state.low_px)
    state.close_px = np.where(mask, state.close_px * mult, state.close_px)

    # Re-enforce OHLC ordering after noise.
    stacked = np.stack([state.open_px, state.high_px, state.low_px, state.close_px], axis=2)
    hi = np.nanmax(stacked, axis=2)
    lo = np.nanmin(stacked, axis=2)
    state.high_px = np.where(mask, hi, state.high_px)
    state.low_px = np.where(mask, lo, state.low_px)


def _recompute_bar_valid_inplace(state: TensorState) -> None:
    state.bar_valid[:, :] = _compute_bar_valid(
        state.open_px,
        state.high_px,
        state.low_px,
        state.close_px,
        state.volume,
    )


def _apply_enabled_assets(state: TensorState, m3: Module3Output, enabled_mask: np.ndarray) -> None:
    A = state.cfg.A
    mask = np.asarray(enabled_mask, dtype=bool)
    if mask.shape != (A,):
        raise RuntimeError(f"enabled_assets_mask shape mismatch: got {mask.shape}, expected {(A,)}")

    off = ~mask
    if not np.any(off):
        return

    state.open_px[:, off] = np.nan
    state.high_px[:, off] = np.nan
    state.low_px[:, off] = np.nan
    state.close_px[:, off] = np.nan
    state.volume[:, off] = np.nan
    state.rvol[:, off] = np.nan
    state.atr_floor[:, off] = np.nan
    state.bar_valid[:, off] = False

    m3.block_features_tak[:, off, :] = np.nan
    m3.block_valid_ta[:, off] = False
    m3.context_tac[:, off, :] = np.nan
    m3.context_valid_ta[:, off] = False
    m3.context_source_t_index_ta[:, off] = -1


def _candidate_daily_returns_close_to_close(
    state: TensorState,
    split: SplitSpec,
    initial_cash: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts, ends, sessions = _session_bounds(state.session_id)
    test_sessions = np.unique(state.session_id[split.test_idx].astype(np.int64))

    if test_sessions.size == 0:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
        )

    # Map session -> close row index.
    close_idx = ends - 1
    sess_close = sessions

    keep = np.isin(sess_close, test_sessions)
    sess_ids = sess_close[keep].astype(np.int64)
    idx = close_idx[keep].astype(np.int64)

    if idx.size == 0:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
        )

    eq_close = state.equity[idx].astype(np.float64)
    ret = np.empty(idx.size, dtype=np.float64)
    ret[0] = eq_close[0] / float(initial_cash) - 1.0
    if idx.size > 1:
        ret[1:] = eq_close[1:] / np.maximum(eq_close[:-1], 1e-12) - 1.0

    return sess_ids, idx, ret


def _benchmark_daily_returns(
    state: TensorState,
    benchmark_symbol: str,
) -> tuple[np.ndarray, np.ndarray]:
    starts, ends, sessions = _session_bounds(state.session_id)

    A = state.cfg.A
    sym_to_idx = {s: i for i, s in enumerate(state.symbols)}

    if benchmark_symbol in sym_to_idx:
        a = int(sym_to_idx[benchmark_symbol])
        sess_out: list[int] = []
        ret_out: list[float] = []
        prev_close: float | None = None

        for s0, s1, sid in zip(starts.tolist(), ends.tolist(), sessions.tolist()):
            v = state.bar_valid[s0:s1, a]
            if not np.any(v):
                continue
            local = np.flatnonzero(v)
            i_open = int(s0 + local[0])
            i_close = int(s0 + local[-1])
            p_open = float(state.open_px[i_open, a])
            p_close = float(state.close_px[i_close, a])
            if not np.isfinite(p_open) or not np.isfinite(p_close) or p_open <= 0.0 or p_close <= 0.0:
                continue
            if prev_close is None:
                r = p_close / p_open - 1.0
            else:
                r = p_close / max(prev_close, 1e-12) - 1.0
            prev_close = p_close
            sess_out.append(int(sid))
            ret_out.append(float(r))

        return np.asarray(sess_out, dtype=np.int64), np.asarray(ret_out, dtype=np.float64)

    # Fallback benchmark: equal-weight passive basket close-to-close.
    sess_out = []
    ret_out = []
    prev_close_basket: float | None = None

    for s0, s1, sid in zip(starts.tolist(), ends.tolist(), sessions.tolist()):
        close_seg = state.close_px[s0:s1]
        valid_seg = state.bar_valid[s0:s1]

        basket_close = np.nanmean(np.where(valid_seg, close_seg, np.nan), axis=1)
        finite_idx = np.flatnonzero(np.isfinite(basket_close))
        if finite_idx.size == 0:
            continue
        i_open = int(finite_idx[0])
        i_close = int(finite_idx[-1])
        p_open = float(basket_close[i_open])
        p_close = float(basket_close[i_close])
        if p_open <= 0.0 or p_close <= 0.0:
            continue

        if prev_close_basket is None:
            r = p_close / p_open - 1.0
        else:
            r = p_close / max(prev_close_basket, 1e-12) - 1.0
        prev_close_basket = p_close

        sess_out.append(int(sid))
        ret_out.append(float(r))

    return np.asarray(sess_out, dtype=np.int64), np.asarray(ret_out, dtype=np.float64)


def _build_candidate_specs_default(
    A: int,
    m2_configs: list[Module2Config],
    m3_configs: list[Module3Config],
    m4_configs: list[Module4Config],
) -> list[CandidateSpec]:
    all_on = np.ones(A, dtype=bool)
    out: list[CandidateSpec] = []
    cid = 0
    for i2 in range(len(m2_configs)):
        for i3 in range(len(m3_configs)):
            for i4 in range(len(m4_configs)):
                out.append(
                    CandidateSpec(
                        candidate_id=f"cand_{cid:04d}_m2{i2}_m3{i3}_m4{i4}",
                        m2_idx=i2,
                        m3_idx=i3,
                        m4_idx=i4,
                        enabled_assets_mask=all_on.copy(),
                        tags=(),
                    )
                )
                cid += 1
    return out


def _normalize_candidate_specs(
    specs: list[CandidateSpec],
    keep_idx: np.ndarray,
    A_filtered: int,
    A_input: int,
) -> list[CandidateSpec]:
    out: list[CandidateSpec] = []
    for spec in specs:
        m = np.asarray(spec.enabled_assets_mask, dtype=bool)
        if m.shape == (A_input,):
            m2 = m[keep_idx]
        elif m.shape == (A_filtered,):
            m2 = m.copy()
        else:
            raise RuntimeError(
                f"Candidate {spec.candidate_id} enabled_assets_mask has invalid shape {m.shape}; "
                f"expected {(A_input,)} or {(A_filtered,)}"
            )
        out.append(
            CandidateSpec(
                candidate_id=spec.candidate_id,
                m2_idx=int(spec.m2_idx),
                m3_idx=int(spec.m3_idx),
                m4_idx=int(spec.m4_idx),
                enabled_assets_mask=m2.astype(bool, copy=True),
                tags=tuple(spec.tags),
            )
        )
    return out


def _build_group_tasks(
    candidates: list[CandidateSpec],
    splits: list[SplitSpec],
    scenarios: list[StressScenario],
) -> list[_GroupTask]:
    groups: dict[tuple[int, int, int, int], list[int]] = {}
    for ci, c in enumerate(candidates):
        for si, _sp in enumerate(splits):
            for zi, sc in enumerate(scenarios):
                if not sc.enabled:
                    continue
                key = (si, zi, int(c.m2_idx), int(c.m3_idx))
                groups.setdefault(key, []).append(ci)

    out: list[_GroupTask] = []
    for key in sorted(groups.keys()):
        si, zi, m2i, m3i = key
        cand_idx = tuple(sorted(groups[key]))
        gid = f"g_s{si:03d}_z{zi:02d}_m2{m2i}_m3{m3i}"
        out.append(
            _GroupTask(
                group_id=gid,
                split_idx=si,
                scenario_idx=zi,
                m2_idx=m2i,
                m3_idx=m3i,
                candidate_indices=cand_idx,
            )
        )
    return out


def _equity_curve_payload(
    state: TensorState,
    candidate_id: str,
    split_id: str,
    scenario_id: str,
) -> dict[str, np.ndarray]:
    eq = state.equity.astype(np.float64)
    peak = np.maximum.accumulate(eq)
    dd = np.where(peak > 0.0, eq / peak - 1.0, 0.0)
    T = state.cfg.T

    return {
        "ts_ns": state.ts_ns.copy(),
        "session_id": state.session_id.copy(),
        "candidate_id": np.full(T, candidate_id, dtype=object),
        "split_id": np.full(T, split_id, dtype=object),
        "scenario_id": np.full(T, scenario_id, dtype=object),
        "equity": eq.copy(),
        "drawdown": dd.astype(np.float64),
        "margin_used": state.margin_used.copy(),
        "buying_power": state.buying_power.copy(),
        "daily_loss": state.daily_loss.copy(),
    }


def _trade_log_payload(
    state: TensorState,
    m4_out: Module4Output,
    candidate_id: str,
    split_id: str,
    scenario_id: str,
    eps: float = 1e-12,
) -> dict[str, np.ndarray]:
    mask = np.isfinite(m4_out.exec_price_ta) & (np.abs(m4_out.filled_qty_ta) > float(eps))
    loc = np.argwhere(mask)
    if loc.size == 0:
        return {
            "ts_ns": np.zeros(0, dtype=np.int64),
            "candidate_id": np.zeros(0, dtype=object),
            "split_id": np.zeros(0, dtype=object),
            "scenario_id": np.zeros(0, dtype=object),
            "symbol": np.zeros(0, dtype=object),
            "filled_qty": np.zeros(0, dtype=np.float64),
            "exec_price": np.zeros(0, dtype=np.float64),
            "trade_cost": np.zeros(0, dtype=np.float64),
            "order_side": np.zeros(0, dtype=np.int8),
            "order_flags": np.zeros(0, dtype=np.uint16),
        }

    t_idx = loc[:, 0]
    a_idx = loc[:, 1]

    return {
        "ts_ns": state.ts_ns[t_idx].astype(np.int64),
        "candidate_id": np.full(t_idx.shape[0], candidate_id, dtype=object),
        "split_id": np.full(t_idx.shape[0], split_id, dtype=object),
        "scenario_id": np.full(t_idx.shape[0], scenario_id, dtype=object),
        "symbol": np.asarray([state.symbols[int(a)] for a in a_idx.tolist()], dtype=object),
        "filled_qty": m4_out.filled_qty_ta[t_idx, a_idx].astype(np.float64),
        "exec_price": m4_out.exec_price_ta[t_idx, a_idx].astype(np.float64),
        "trade_cost": m4_out.trade_cost_ta[t_idx, a_idx].astype(np.float64),
        "order_side": state.order_side[t_idx, a_idx].astype(np.int8),
        "order_flags": state.order_flags[t_idx, a_idx].astype(np.uint16),
    }


def _event_window_mask(T: int, event_idx: np.ndarray, pre: int, post: int) -> np.ndarray:
    mask = np.zeros(T, dtype=bool)
    if event_idx.size == 0:
        return mask
    lo_off = int(max(0, pre))
    hi_off = int(max(0, post))
    for i in event_idx.tolist():
        lo = max(0, int(i) - lo_off)
        hi = min(T, int(i) + hi_off + 1)
        mask[lo:hi] = True
    return mask


def _structural_weight_from_regime(regime_i8: np.ndarray) -> np.ndarray:
    r = np.asarray(regime_i8, dtype=np.int8)
    w = np.zeros(r.shape, dtype=np.float64)
    w[(r == np.int8(RegimeIdx.P_SHAPE)) | (r == np.int8(RegimeIdx.B_SHAPE))] = 1.5
    w[r == np.int8(RegimeIdx.TREND)] = 1.2
    return w


def _select_micro_rows(
    state: TensorState,
    split: SplitSpec,
    cfg: Module5HarnessConfig,
    m4_out: Module4Output,
    enabled_assets_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    T = state.cfg.T
    A = state.cfg.A
    mode = str(cfg.micro_diag_mode).strip().lower()

    a_mask = np.asarray(enabled_assets_mask, dtype=bool).copy()
    if a_mask.shape != (A,):
        raise RuntimeError(
            f"enabled_assets_mask shape mismatch for micro diagnostics: got {a_mask.shape}, expected {(A,)}"
        )

    if cfg.micro_diag_symbols:
        symbol_set = set(str(s) for s in cfg.micro_diag_symbols)
        a_mask &= np.asarray([s in symbol_set for s in state.symbols], dtype=bool)

    if mode == "off":
        return np.zeros(T, dtype=bool), a_mask

    if mode == "full_test":
        t_mask = np.zeros(T, dtype=bool)
        t_mask[split.test_idx] = True
    elif mode == "symbol_day":
        t_mask = np.ones(T, dtype=bool)
        if cfg.micro_diag_session_ids:
            sset = set(int(s) for s in cfg.micro_diag_session_ids)
            t_mask &= np.isin(state.session_id.astype(np.int64), np.asarray(sorted(sset), dtype=np.int64))
    elif mode == "events_only":
        fills_t = np.flatnonzero(np.any(np.abs(m4_out.filled_qty_ta) > 1e-12, axis=1)).astype(np.int64)
        select_t = np.flatnonzero(state.phase == np.int8(Phase.OVERNIGHT_SELECT)).astype(np.int64)
        event_idx = np.unique(np.r_[fills_t, select_t]).astype(np.int64)
        t_mask = _event_window_mask(
            T=T,
            event_idx=event_idx,
            pre=int(cfg.micro_diag_trade_window_pre),
            post=int(cfg.micro_diag_trade_window_post),
        )
    else:
        raise RuntimeError(f"Unsupported micro_diag_mode: {cfg.micro_diag_mode}")

    # Keep only rows with at least one valid enabled asset observation.
    valid_any = np.any(state.bar_valid[:, a_mask], axis=1) if np.any(a_mask) else np.zeros(T, dtype=bool)
    t_mask &= valid_any
    return t_mask, a_mask


def _collect_micro_diagnostics_payload(
    state: TensorState,
    m3: Module3Output,
    m4_out: Module4Output,
    candidate_id: str,
    split_id: str,
    scenario_id: str,
    split: SplitSpec,
    enabled_assets_mask: np.ndarray,
    cfg: Module5HarnessConfig,
) -> dict[str, np.ndarray] | None:
    if not bool(cfg.export_micro_diagnostics):
        return None

    t_mask, a_mask = _select_micro_rows(state, split, cfg, m4_out, enabled_assets_mask)
    if not np.any(t_mask) or not np.any(a_mask):
        return None

    loc = np.argwhere(t_mask[:, None] & a_mask[None, :])
    if loc.size == 0:
        return None

    if loc.shape[0] > int(cfg.micro_diag_max_rows):
        raise RuntimeError(
            f"micro_diagnostics row cap exceeded: rows={int(loc.shape[0])}, cap={int(cfg.micro_diag_max_rows)}"
        )

    t_idx = loc[:, 0].astype(np.int64)
    a_idx = loc[:, 1].astype(np.int64)

    winner_flag = (
        m4_out.overnight_winner_t[t_idx].astype(np.int64) == a_idx.astype(np.int64)
    ).astype(np.int8)

    return {
        "ts_ns": state.ts_ns[t_idx].astype(np.int64),
        "session_id": state.session_id[t_idx].astype(np.int64),
        "candidate_id": np.full(t_idx.shape[0], candidate_id, dtype=object),
        "split_id": np.full(t_idx.shape[0], split_id, dtype=object),
        "scenario_id": np.full(t_idx.shape[0], scenario_id, dtype=object),
        "symbol": np.asarray([state.symbols[int(a)] for a in a_idx.tolist()], dtype=object),
        "open": state.open_px[t_idx, a_idx].astype(np.float64),
        "high": state.high_px[t_idx, a_idx].astype(np.float64),
        "low": state.low_px[t_idx, a_idx].astype(np.float64),
        "close": state.close_px[t_idx, a_idx].astype(np.float64),
        "volume": state.volume[t_idx, a_idx].astype(np.float64),
        "bar_valid": state.bar_valid[t_idx, a_idx].astype(np.int8),
        "dclip": state.profile_stats[t_idx, a_idx, int(ProfileStatIdx.DCLIP)].astype(np.float64),
        "z_delta": state.profile_stats[t_idx, a_idx, int(ProfileStatIdx.Z_DELTA)].astype(np.float64),
        "gbreak": state.profile_stats[t_idx, a_idx, int(ProfileStatIdx.GBREAK)].astype(np.float64),
        "greject": state.profile_stats[t_idx, a_idx, int(ProfileStatIdx.GREJECT)].astype(np.float64),
        "score_bo_long": state.scores[t_idx, a_idx, int(ScoreIdx.SCORE_BO_LONG)].astype(np.float64),
        "score_bo_short": state.scores[t_idx, a_idx, int(ScoreIdx.SCORE_BO_SHORT)].astype(np.float64),
        "score_rej_long": state.scores[t_idx, a_idx, int(ScoreIdx.SCORE_REJ_LONG)].astype(np.float64),
        "score_rej_short": state.scores[t_idx, a_idx, int(ScoreIdx.SCORE_REJ_SHORT)].astype(np.float64),
        "ctx_x_poc": m3.context_tac[t_idx, a_idx, int(ContextIdx.CTX_X_POC)].astype(np.float64),
        "ctx_x_vah": m3.context_tac[t_idx, a_idx, int(ContextIdx.CTX_X_VAH)].astype(np.float64),
        "ctx_x_val": m3.context_tac[t_idx, a_idx, int(ContextIdx.CTX_X_VAL)].astype(np.float64),
        "ctx_trend_gate_spread_mean": m3.context_tac[t_idx, a_idx, int(ContextIdx.CTX_TREND_GATE_SPREAD_MEAN)].astype(np.float64),
        "ctx_poc_drift_x": m3.context_tac[t_idx, a_idx, int(ContextIdx.CTX_POC_DRIFT_X)].astype(np.float64),
        "ctx_poc_vs_prev_va": m3.context_tac[t_idx, a_idx, int(ContextIdx.CTX_POC_VS_PREV_VA)].astype(np.float64),
        "ctx_ib_high_x": m3.context_tac[t_idx, a_idx, int(ContextIdx.CTX_IB_HIGH_X)].astype(np.float64),
        "ctx_ib_low_x": m3.context_tac[t_idx, a_idx, int(ContextIdx.CTX_IB_LOW_X)].astype(np.float64),
        "regime_primary": m4_out.regime_primary_ta[t_idx, a_idx].astype(np.int8),
        "regime_confidence": m4_out.regime_confidence_ta[t_idx, a_idx].astype(np.float64),
        "intent_long": m4_out.intent_long_ta[t_idx, a_idx].astype(np.int8),
        "intent_short": m4_out.intent_short_ta[t_idx, a_idx].astype(np.int8),
        "target_qty": m4_out.target_qty_ta[t_idx, a_idx].astype(np.float64),
        "filled_qty": m4_out.filled_qty_ta[t_idx, a_idx].astype(np.float64),
        "exec_price": m4_out.exec_price_ta[t_idx, a_idx].astype(np.float64),
        "trade_cost": m4_out.trade_cost_ta[t_idx, a_idx].astype(np.float64),
        "position_qty": state.position_qty[t_idx, a_idx].astype(np.float64),
        "overnight_score": m4_out.overnight_score_ta[t_idx, a_idx].astype(np.float64),
        "overnight_winner_flag": winner_flag,
        "atr_eff": state.atr_floor[t_idx, a_idx].astype(np.float64),
        "rvol": state.rvol[t_idx, a_idx].astype(np.float64),
    }


def _collect_micro_profile_blocks_payload(
    state: TensorState,
    m3: Module3Output,
    candidate_id: str,
    split_id: str,
    scenario_id: str,
    enabled_assets_mask: np.ndarray,
    cfg: Module5HarnessConfig,
) -> dict[str, np.ndarray] | None:
    if not (bool(cfg.export_micro_diagnostics) and bool(cfg.micro_diag_export_block_profiles)):
        return None

    A = state.cfg.A
    a_mask = np.asarray(enabled_assets_mask, dtype=bool)
    if a_mask.shape != (A,):
        raise RuntimeError(f"enabled_assets_mask shape mismatch in profile blocks: {a_mask.shape}")
    if cfg.micro_diag_symbols:
        symbol_set = set(str(s) for s in cfg.micro_diag_symbols)
        a_mask &= np.asarray([s in symbol_set for s in state.symbols], dtype=bool)

    if not np.any(a_mask):
        return None

    block_rows = np.flatnonzero(m3.block_end_flag_t).astype(np.int64)
    if block_rows.size == 0:
        return None

    mask = m3.block_valid_ta[block_rows][:, a_mask]
    loc = np.argwhere(mask)
    if loc.size == 0:
        return None

    rr = block_rows[loc[:, 0].astype(np.int64)]
    aa_local = np.where(a_mask)[0].astype(np.int64)
    aa = aa_local[loc[:, 1].astype(np.int64)]

    if rr.shape[0] > int(cfg.micro_diag_max_rows):
        raise RuntimeError(
            f"micro_profile_blocks row cap exceeded: rows={int(rr.shape[0])}, cap={int(cfg.micro_diag_max_rows)}"
        )

    x_blob = state.x_grid.astype(np.float32).tobytes()
    return {
        "ts_ns": state.ts_ns[rr].astype(np.int64),
        "session_id": state.session_id[rr].astype(np.int64),
        "candidate_id": np.full(rr.shape[0], candidate_id, dtype=object),
        "split_id": np.full(rr.shape[0], split_id, dtype=object),
        "scenario_id": np.full(rr.shape[0], scenario_id, dtype=object),
        "symbol": np.asarray([state.symbols[int(a)] for a in aa.tolist()], dtype=object),
        "block_seq": m3.block_seq_t[rr].astype(np.int16),
        "n_bins": np.full(rr.shape[0], int(state.cfg.B), dtype=np.int32),
        "x_grid_blob": np.full(rr.shape[0], x_blob, dtype=object),
        "vp_block_blob": np.asarray([state.vp[int(t), int(a)].astype(np.float32).tobytes() for t, a in zip(rr.tolist(), aa.tolist())], dtype=object),
        "vp_delta_block_blob": np.asarray([state.vp_delta[int(t), int(a)].astype(np.float32).tobytes() for t, a in zip(rr.tolist(), aa.tolist())], dtype=object),
        "close_te": state.close_px[rr, aa].astype(np.float64),
        "atr_eff_te": state.atr_floor[rr, aa].astype(np.float64),
    }


def _collect_funnel_payload(
    state: TensorState,
    m4_out: Module4Output,
    candidate_id: str,
    split_id: str,
    scenario_id: str,
    enabled_assets_mask: np.ndarray,
    cfg: Module5HarnessConfig,
) -> dict[str, np.ndarray] | None:
    if not (bool(cfg.export_micro_diagnostics) and bool(cfg.micro_diag_export_funnel)):
        return None

    A = state.cfg.A
    a_mask = np.asarray(enabled_assets_mask, dtype=bool)
    if a_mask.shape != (A,):
        raise RuntimeError(f"enabled_assets_mask shape mismatch in funnel payload: {a_mask.shape}")
    if cfg.micro_diag_symbols:
        symbol_set = set(str(s) for s in cfg.micro_diag_symbols)
        a_mask &= np.asarray([s in symbol_set for s in state.symbols], dtype=bool)
    if not np.any(a_mask):
        return None

    t_sel = np.flatnonzero(state.phase == np.int8(Phase.OVERNIGHT_SELECT)).astype(np.int64)
    if t_sel.size == 0:
        return None

    out_rows: list[dict[str, Any]] = []
    for t in t_sel.tolist():
        winner = int(m4_out.overnight_winner_t[t])
        valid_assets = np.where(a_mask)[0].astype(np.int64)
        if valid_assets.size == 0:
            continue
        dclip = state.profile_stats[t, valid_assets, int(ProfileStatIdx.DCLIP)]
        zdel = state.profile_stats[t, valid_assets, int(ProfileStatIdx.Z_DELTA)]
        rvol = state.rvol[t, valid_assets]
        regime = m4_out.regime_primary_ta[t, valid_assets]
        sw = _structural_weight_from_regime(regime)
        ocs = sw * np.abs(dclip) * np.abs(zdel) * np.maximum(rvol, 0.0)
        cash_fallback = winner < 0
        for j, a in enumerate(valid_assets.tolist()):
            out_rows.append(
                {
                    "ts_ns": int(state.ts_ns[t]),
                    "session_id": int(state.session_id[t]),
                    "candidate_id": candidate_id,
                    "split_id": split_id,
                    "scenario_id": scenario_id,
                    "symbol": state.symbols[int(a)],
                    "dclip": float(dclip[j]),
                    "z_delta": float(zdel[j]),
                    "regime_primary": int(regime[j]),
                    "structural_weight": float(sw[j]),
                    "ocs": float(ocs[j]),
                    "is_winner": int(1 if int(a) == winner else 0),
                    "cash_fallback": int(1 if cash_fallback else 0),
                    "rvol": float(rvol[j]),
                }
            )

    if not out_rows:
        return None

    pdx = _require_pandas()
    df = pdx.DataFrame(out_rows)
    return {k: df[k].to_numpy() for k in df.columns.tolist()}


def _run_group_task(
    group: _GroupTask,
    base_state: TensorState,
    candidates: list[CandidateSpec],
    splits: list[SplitSpec],
    scenarios: list[StressScenario],
    m2_configs: list[Module2Config],
    m3_configs: list[Module3Config],
    m4_configs: list[Module4Config],
    harness_cfg: Module5HarnessConfig,
) -> list[dict[str, Any]]:
    split = splits[group.split_idx]
    scenario = scenarios[group.scenario_idx]

    group_seed = _seed_for_task(
        harness_cfg.seed,
        group.group_id,
        split.split_id,
        scenario.scenario_id,
    )
    rng = np.random.default_rng(group_seed)

    # Build post-M3 cache for this (split, scenario, m2, m3) key.
    cached_state = _clone_state(base_state)
    active_t = _apply_split_domain_mask(cached_state, split)

    _apply_missing_bursts(cached_state, active_t, scenario, rng)
    _apply_jitter(cached_state, active_t, scenario, rng)
    _recompute_bar_valid_inplace(cached_state)

    # Refresh placeholders required by Module 1 slice checks.
    tick = cached_state.eps.eps_div[None, :]
    cached_state.rvol[:, :] = np.where(cached_state.bar_valid, 1.0, np.nan)
    cached_state.atr_floor[:, :] = np.where(cached_state.bar_valid, np.maximum(4.0 * tick, 1e-12), np.nan)

    if harness_cfg.fail_on_non_finite:
        validate_loaded_market_slice(cached_state, 0, cached_state.cfg.T)

    run_weightiz_profile_engine(cached_state, m2_configs[group.m2_idx])
    m3_out_cached = run_module3_structural_aggregation(cached_state, m3_configs[group.m3_idx])

    outputs: list[dict[str, Any]] = []

    for ci in group.candidate_indices:
        c = candidates[ci]
        task_id = f"{c.candidate_id}|{split.split_id}|{scenario.scenario_id}"
        try:
            st = _clone_state(cached_state)
            m3c = _clone_m3(m3_out_cached)
            _apply_enabled_assets(st, m3c, c.enabled_assets_mask)

            m4_cfg = replace(
                m4_configs[c.m4_idx],
                stress_slippage_mult=float(m4_configs[c.m4_idx].stress_slippage_mult)
                * float(scenario.slippage_mult),
            )

            m4_out = run_module4_strategy_funnel(st, m3c, m4_cfg)
            validate_state_hard(st)

            sess_ids, close_idx, daily_ret = _candidate_daily_returns_close_to_close(
                st,
                split,
                initial_cash=float(st.cfg.initial_cash),
            )

            micro_payload = _collect_micro_diagnostics_payload(
                state=st,
                m3=m3c,
                m4_out=m4_out,
                candidate_id=c.candidate_id,
                split_id=split.split_id,
                scenario_id=scenario.scenario_id,
                split=split,
                enabled_assets_mask=c.enabled_assets_mask,
                cfg=harness_cfg,
            )
            profile_payload = _collect_micro_profile_blocks_payload(
                state=st,
                m3=m3c,
                candidate_id=c.candidate_id,
                split_id=split.split_id,
                scenario_id=scenario.scenario_id,
                enabled_assets_mask=c.enabled_assets_mask,
                cfg=harness_cfg,
            )
            funnel_payload = _collect_funnel_payload(
                state=st,
                m4_out=m4_out,
                candidate_id=c.candidate_id,
                split_id=split.split_id,
                scenario_id=scenario.scenario_id,
                enabled_assets_mask=c.enabled_assets_mask,
                cfg=harness_cfg,
            )

            outputs.append(
                {
                    "task_id": task_id,
                    "candidate_id": c.candidate_id,
                    "split_id": split.split_id,
                    "scenario_id": scenario.scenario_id,
                    "status": "ok",
                    "error": "",
                    "session_ids": sess_ids,
                    "daily_returns": daily_ret,
                    "equity_payload": _equity_curve_payload(st, c.candidate_id, split.split_id, scenario.scenario_id),
                    "trade_payload": _trade_log_payload(st, m4_out, c.candidate_id, split.split_id, scenario.scenario_id),
                    "micro_payload": micro_payload,
                    "profile_payload": profile_payload,
                    "funnel_payload": funnel_payload,
                    "m2_idx": int(c.m2_idx),
                    "m3_idx": int(c.m3_idx),
                    "m4_idx": int(c.m4_idx),
                    "tags": list(c.tags),
                    "test_days": int(daily_ret.shape[0]),
                }
            )
        except Exception as exc:
            outputs.append(
                {
                    "task_id": task_id,
                    "candidate_id": c.candidate_id,
                    "split_id": split.split_id,
                    "scenario_id": scenario.scenario_id,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "session_ids": np.zeros(0, dtype=np.int64),
                    "daily_returns": np.zeros(0, dtype=np.float64),
                    "equity_payload": None,
                    "trade_payload": None,
                    "micro_payload": None,
                    "profile_payload": None,
                    "funnel_payload": None,
                    "m2_idx": int(c.m2_idx),
                    "m3_idx": int(c.m3_idx),
                    "m4_idx": int(c.m4_idx),
                    "tags": list(c.tags),
                    "test_days": 0,
                }
            )

    return outputs


def _stack_payload_frames(payloads: list[dict[str, np.ndarray]]) -> Any:
    pdx = _require_pandas()
    if not payloads:
        return pdx.DataFrame()
    frames = [pdx.DataFrame(p) for p in payloads]
    if not frames:
        return pdx.DataFrame()
    return pdx.concat(frames, axis=0, ignore_index=True)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(obj), f, ensure_ascii=False, indent=2)


def _assemble_daily_matrix(
    results_ok: list[dict[str, Any]],
    bench_sessions: np.ndarray,
    bench_ret: np.ndarray,
    min_days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if not results_ok:
        raise RuntimeError("No successful candidate task results to assemble")

    bench_map = {int(s): float(r) for s, r in zip(bench_sessions.tolist(), bench_ret.tolist())}

    sess_sets: list[set[int]] = []
    series_maps: list[dict[int, float]] = []
    task_ids: list[str] = []

    for r in results_ok:
        sid = np.asarray(r["session_ids"], dtype=np.int64)
        ret = np.asarray(r["daily_returns"], dtype=np.float64)
        if sid.size == 0 or ret.size == 0:
            continue
        m = {int(s): float(v) for s, v in zip(sid.tolist(), ret.tolist())}
        task_ids.append(str(r["task_id"]))
        series_maps.append(m)
        sess_sets.append(set(m.keys()))

    if not series_maps:
        raise RuntimeError("All candidate results are empty after filtering")

    common = set(bench_map.keys())
    for s in sess_sets:
        common &= s

    if not common:
        raise RuntimeError("No common sessions across candidates and benchmark")

    common_sorted = np.asarray(sorted(common), dtype=np.int64)
    D = int(common_sorted.size)
    C = int(len(series_maps))

    if D < int(min_days):
        raise RuntimeError(f"Insufficient daily sample after alignment: D={D}, required>={int(min_days)}")

    mat = np.empty((D, C), dtype=np.float64)
    for j, mp in enumerate(series_maps):
        mat[:, j] = np.asarray([mp[int(s)] for s in common_sorted.tolist()], dtype=np.float64)

    bmk = np.asarray([bench_map[int(s)] for s in common_sorted.tolist()], dtype=np.float64)

    _assert_finite("daily_returns_matrix", mat)
    _assert_finite("daily_benchmark_returns", bmk)

    return common_sorted, mat, bmk, task_ids


def _compute_stats_verdict(
    daily_returns_matrix: np.ndarray,
    daily_benchmark_returns: np.ndarray,
    task_ids: list[str],
    harness_cfg: Module5HarnessConfig,
) -> dict[str, Any]:
    dsr = deflated_sharpe_ratio(daily_returns_matrix)
    pbo = pbo_cscv(
        daily_returns_matrix,
        S=int(harness_cfg.cpcv_slices),
        k=int(harness_cfg.cpcv_k_test),
    )
    wrc = white_reality_check(
        daily_returns_matrix,
        daily_benchmark_returns,
        seed=int(harness_cfg.seed + 101),
    )
    spa = spa_test(
        daily_returns_matrix,
        daily_benchmark_returns,
        seed=int(harness_cfg.seed + 202),
    )
    mcs = model_confidence_set(
        -daily_returns_matrix,
        alpha=0.10,
        seed=int(harness_cfg.seed + 303),
    )

    survivors = set(int(i) for i in np.asarray(mcs.get("survivors", np.array([], dtype=np.int64))).tolist())
    dsr_arr = np.asarray(dsr["dsr"], dtype=np.float64)

    leaderboard: list[dict[str, Any]] = []
    for j, tid in enumerate(task_ids):
        in_mcs = j in survivors
        dsr_j = float(dsr_arr[j]) if j < dsr_arr.size else float("nan")
        pass_flag = bool(in_mcs and (dsr_j >= 0.50))
        leaderboard.append(
            {
                "task_id": tid,
                "dsr": dsr_j,
                "in_mcs": in_mcs,
                "wrc_p": float(wrc["p_value"]),
                "spa_p": float(spa["p_value"]),
                "pbo": float(pbo["pbo"]) if np.isfinite(pbo["pbo"]) else None,
                "pass": pass_flag,
            }
        )

    return {
        "dsr": dsr,
        "pbo": pbo,
        "wrc": wrc,
        "spa": spa,
        "mcs": mcs,
        "leaderboard": leaderboard,
        "gate_defaults": {
            "dsr_min": 0.50,
            "mcs_membership_required": True,
        },
    }


def run_weightiz_harness(
    data_paths: list[str],
    symbols: list[str],
    engine_cfg: EngineConfig,
    m2_configs: list[Module2Config],
    m3_configs: list[Module3Config],
    m4_configs: list[Module4Config],
    harness_cfg: Module5HarnessConfig,
    candidate_specs: list[CandidateSpec] | None = None,
    data_loader_func: Callable[[str, str], Any] | None = None,
    stress_scenarios: list[StressScenario] | None = None,
) -> HarnessOutput:
    if not m2_configs or not m3_configs or not m4_configs:
        raise RuntimeError("m2_configs/m3_configs/m4_configs must be non-empty")

    run_started_utc = datetime.now(timezone.utc)

    base_state, keep_idx, keep_symbols, master_ts_ns, ingest_meta, tick_keep = _ingest_master_aligned(
        data_paths=data_paths,
        symbols=symbols,
        engine_cfg=engine_cfg,
        harness_cfg=harness_cfg,
        data_loader_func=data_loader_func,
    )

    A_filtered = base_state.cfg.A
    A_input = len(symbols)

    if candidate_specs is None:
        candidates = _build_candidate_specs_default(A_filtered, m2_configs, m3_configs, m4_configs)
    else:
        candidates = _normalize_candidate_specs(candidate_specs, keep_idx, A_filtered, A_input)

    candidates = sorted(candidates, key=lambda c: c.candidate_id)

    splits = _generate_wf_splits(base_state, harness_cfg) + _generate_cpcv_splits(base_state, harness_cfg)
    if not splits:
        raise RuntimeError("No WF/CPCV splits generated; adjust harness split parameters")

    for sp in splits:
        _validate_split(sp, enforce_guard=bool(harness_cfg.enforce_lookahead_guard))

    source_scenarios = stress_scenarios if stress_scenarios is not None else _default_stress_scenarios(harness_cfg)
    scenarios = [s for s in source_scenarios if s.enabled]
    if not scenarios:
        raise RuntimeError("No enabled stress scenarios")

    group_tasks = _build_group_tasks(candidates, splits, scenarios)
    if not group_tasks:
        raise RuntimeError("No group tasks generated")

    # RAM policy: reduce worker count if projected footprint is too high.
    avail = _available_memory_bytes()
    est_state = _estimate_state_bytes(base_state.cfg.T, base_state.cfg.A, base_state.cfg.B)
    max_workers = int(max(1, harness_cfg.parallel_workers))
    budget = int(float(harness_cfg.max_ram_utilization_frac) * float(avail))

    if est_state * max_workers > budget:
        max_workers = max(1, budget // max(est_state, 1))
    max_workers = max(1, max_workers)

    all_results: list[dict[str, Any]] = []

    use_process_pool = harness_cfg.parallel_backend == "process_pool" and max_workers > 1

    if use_process_pool:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = []
            for g in group_tasks:
                futs.append(
                    ex.submit(
                        _run_group_task,
                        g,
                        base_state,
                        candidates,
                        splits,
                        scenarios,
                        m2_configs,
                        m3_configs,
                        m4_configs,
                        harness_cfg,
                    )
                )
            for fut in as_completed(futs):
                all_results.extend(fut.result())
    else:
        for g in group_tasks:
            all_results.extend(
                _run_group_task(
                    g,
                    base_state,
                    candidates,
                    splits,
                    scenarios,
                    m2_configs,
                    m3_configs,
                    m4_configs,
                    harness_cfg,
                )
            )

    # Deterministic collation order.
    all_results.sort(key=lambda r: str(r.get("task_id", "")))

    ok_results = [r for r in all_results if r.get("status") == "ok" and int(r.get("test_days", 0)) > 0]

    bench_sessions, bench_ret = _benchmark_daily_returns(base_state, harness_cfg.benchmark_symbol)
    common_sessions, daily_mat, daily_bmk, task_ids = _assemble_daily_matrix(
        ok_results,
        bench_sessions,
        bench_ret,
        min_days=int(harness_cfg.daily_return_min_days),
    )

    stats_verdict = _compute_stats_verdict(daily_mat, daily_bmk, task_ids, harness_cfg)

    # Attach per-task leaderboard metrics back into candidate_results.
    lb_map = {str(x["task_id"]): x for x in stats_verdict.get("leaderboard", [])}
    candidate_results: list[dict[str, object]] = []
    for r in all_results:
        out = {
            "task_id": r.get("task_id"),
            "candidate_id": r.get("candidate_id"),
            "split_id": r.get("split_id"),
            "scenario_id": r.get("scenario_id"),
            "status": r.get("status"),
            "error": r.get("error", ""),
            "m2_idx": r.get("m2_idx"),
            "m3_idx": r.get("m3_idx"),
            "m4_idx": r.get("m4_idx"),
            "tags": r.get("tags", []),
            "test_days": int(r.get("test_days", 0)),
        }
        lb = lb_map.get(str(r.get("task_id")))
        if lb is not None:
            out.update(
                {
                    "dsr": lb["dsr"],
                    "in_mcs": lb["in_mcs"],
                    "wrc_p": lb["wrc_p"],
                    "spa_p": lb["spa_p"],
                    "pbo": lb["pbo"],
                    "pass": lb["pass"],
                }
            )
        candidate_results.append(out)

    # Artifact export
    run_id = run_started_utc.strftime("run_%Y%m%d_%H%M%S")
    report_root = Path(harness_cfg.report_dir).resolve() / run_id
    report_root.mkdir(parents=True, exist_ok=True)

    eq_payloads = [r["equity_payload"] for r in ok_results if r.get("equity_payload") is not None]
    tr_payloads = [r["trade_payload"] for r in ok_results if r.get("trade_payload") is not None]
    micro_payloads = [r["micro_payload"] for r in ok_results if r.get("micro_payload") is not None]
    profile_payloads = [r["profile_payload"] for r in ok_results if r.get("profile_payload") is not None]
    funnel_payloads = [r["funnel_payload"] for r in ok_results if r.get("funnel_payload") is not None]

    eq_df = _stack_payload_frames(eq_payloads)
    tr_df = _stack_payload_frames(tr_payloads)
    micro_df = _stack_payload_frames(micro_payloads)
    profile_df = _stack_payload_frames(profile_payloads)
    funnel_df = _stack_payload_frames(funnel_payloads)

    pdx = _require_pandas()

    equity_path = report_root / "equity_curves.parquet"
    trade_path = report_root / "trade_log.parquet"
    daily_path = report_root / "daily_returns.parquet"
    verdict_path = report_root / "verdict.json"
    stats_raw_path = report_root / "stats_raw.json"
    manifest_path = report_root / "run_manifest.json"
    micro_diag_path = report_root / "micro_diagnostics.parquet"
    micro_profile_blocks_path = report_root / "micro_profile_blocks.parquet"
    funnel_1545_path = report_root / "funnel_1545.parquet"

    eq_df.to_parquet(equity_path, index=False)
    tr_df.to_parquet(trade_path, index=False)

    daily_df = pdx.DataFrame({"session_id": common_sessions.astype(np.int64), "benchmark": daily_bmk})
    for j, tid in enumerate(task_ids):
        daily_df[tid] = daily_mat[:, j]
    daily_df.to_parquet(daily_path, index=False)

    _write_json(verdict_path, {"leaderboard": stats_verdict.get("leaderboard", []), "summary": {
        "n_candidates": int(daily_mat.shape[1]),
        "n_days": int(daily_mat.shape[0]),
        "benchmark_symbol": harness_cfg.benchmark_symbol,
    }})
    _write_json(stats_raw_path, stats_verdict)

    if bool(harness_cfg.export_micro_diagnostics):
        micro_df.to_parquet(micro_diag_path, index=False)
        if bool(harness_cfg.micro_diag_export_block_profiles):
            profile_df.to_parquet(micro_profile_blocks_path, index=False)
        if bool(harness_cfg.micro_diag_export_funnel):
            funnel_df.to_parquet(funnel_1545_path, index=False)

    manifest = {
        "run_id": run_id,
        "run_started_utc": run_started_utc.isoformat(),
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": _git_hash(),
        "seed": int(harness_cfg.seed),
        "symbols_input": list(symbols),
        "symbols_kept": keep_symbols,
        "keep_idx": keep_idx.tolist(),
        "ingestion": ingest_meta,
        "engine_cfg_hash": _stable_hash_obj(asdict(engine_cfg)),
        "m2_hashes": [_stable_hash_obj(asdict(c)) for c in m2_configs],
        "m3_hashes": [_stable_hash_obj(asdict(c)) for c in m3_configs],
        "m4_hashes": [_stable_hash_obj(asdict(c)) for c in m4_configs],
        "harness_cfg_hash": _stable_hash_obj(asdict(harness_cfg)),
        "n_candidates": len(candidates),
        "n_splits": len(splits),
        "n_scenarios": len(scenarios),
        "n_group_tasks": len(group_tasks),
        "n_task_results": len(all_results),
        "n_ok_results": len(ok_results),
        "daily_matrix_shape": list(daily_mat.shape),
        "parallel_backend": harness_cfg.parallel_backend,
        "parallel_workers_effective": int(max_workers),
        "memory": {
            "available_bytes": int(avail),
            "estimated_state_bytes": int(est_state),
            "budget_bytes": int(budget),
        },
        "micro_diagnostics": {
            "enabled": bool(harness_cfg.export_micro_diagnostics),
            "mode": str(harness_cfg.micro_diag_mode),
            "symbols_filter": list(harness_cfg.micro_diag_symbols),
            "session_ids_filter": [int(x) for x in harness_cfg.micro_diag_session_ids],
            "max_rows": int(harness_cfg.micro_diag_max_rows),
            "rows_exported": int(len(micro_df)),
            "profile_rows_exported": int(len(profile_df)),
            "funnel_rows_exported": int(len(funnel_df)),
        },
    }
    _write_json(manifest_path, manifest)

    artifact_paths = {
        "equity_curves": str(equity_path),
        "trade_log": str(trade_path),
        "daily_returns": str(daily_path),
        "verdict": str(verdict_path),
        "stats_raw": str(stats_raw_path),
        "run_manifest": str(manifest_path),
    }
    if bool(harness_cfg.export_micro_diagnostics):
        artifact_paths["micro_diagnostics"] = str(micro_diag_path)
        if bool(harness_cfg.micro_diag_export_block_profiles):
            artifact_paths["micro_profile_blocks"] = str(micro_profile_blocks_path)
        if bool(harness_cfg.micro_diag_export_funnel):
            artifact_paths["funnel_1545"] = str(funnel_1545_path)

    return HarnessOutput(
        candidate_results=candidate_results,
        daily_returns_matrix=daily_mat,
        daily_benchmark_returns=daily_bmk,
        stats_verdict=stats_verdict,
        artifact_paths=artifact_paths,
        run_manifest=manifest,
    )


if __name__ == "__main__":
    print("MODULE5_HARNESS_READY")
