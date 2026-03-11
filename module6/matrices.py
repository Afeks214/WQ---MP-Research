from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from numpy.lib.format import open_memmap

from module6.calendar import build_portfolio_calendar_frame
from module6.config import Module6Config
from module6.constants import BASE_AVAIL_ACTIVE_CODES
from module6.utils import Module6ValidationError, ensure_directory


@dataclass(frozen=True)
class MatrixStore:
    calendar: np.ndarray
    calendar_index_path: Path
    column_index: pd.DataFrame
    returns_exec_path: Path
    returns_raw_path: Path
    availability_path: Path
    turnover_path: Path
    state_code_path: Path
    gross_peak_path: Path
    gross_mean_path: Path
    buying_power_min_path: Path
    overnight_flag_path: Path
    family_incidence_path: Path
    regime_exposure_path: Path


def _write_dense_matrix(path: Path, shape: tuple[int, int], dtype: Any) -> np.memmap:
    return open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def build_matrix_store(
    *,
    ledgers: dict[str, pd.DataFrame],
    run,
    output_dir: Path,
    config: Module6Config,
) -> MatrixStore:
    out_dir = ensure_directory(output_dir / "matrix_store")
    strategy_master = ledgers["strategy_master"].copy()
    instance_master = ledgers["strategy_instance_master"].copy()
    session_ledger = ledgers["strategy_session_ledger"].copy()

    canonical_instances = instance_master.loc[
        instance_master["portfolio_instance_role"] == "canonical_portfolio"
    ].copy()
    canonical_instances = canonical_instances.loc[
        canonical_instances["portfolio_admit_flag"].astype(bool)
    ].copy()
    canonical_instances = canonical_instances.sort_values(["strategy_instance_pk"], kind="mergesort").reset_index(drop=True)
    if canonical_instances.shape[0] <= 0:
        raise Module6ValidationError("no admitted canonical portfolio instances available for matrix build")
    calendar_frame = build_portfolio_calendar_frame(strategy_session_returns=session_ledger, equity_curves=run.equity_curves)
    calendar = np.asarray(calendar_frame.sessions, dtype=np.int64)
    T = int(calendar.shape[0])
    N = int(canonical_instances.shape[0])
    calendar_frame.frame.to_parquet(out_dir / "calendar_index.parquet", index=False)
    column_index = canonical_instances[
        ["strategy_instance_pk", "strategy_pk", "candidate_id", "family_id", "hypothesis_id"]
    ].copy()
    column_index.insert(0, "column_idx", np.arange(N, dtype=np.int64))
    if column_index["strategy_instance_pk"].duplicated().any():
        raise Module6ValidationError("duplicate strategy_instance_pk in matrix column index")
    column_index.to_parquet(out_dir / "column_index.parquet", index=False)

    returns_exec = _write_dense_matrix(out_dir / "returns_exec.npy", (T, N), np.float64)
    returns_raw = _write_dense_matrix(out_dir / "returns_raw.npy", (T, N), np.float64)
    availability = _write_dense_matrix(out_dir / "availability.npy", (T, N), np.bool_)
    turnover = _write_dense_matrix(out_dir / "activity_turnover.npy", (T, N), np.float32)
    state_codes = _write_dense_matrix(out_dir / "availability_state_codes.npy", (T, N), np.int16)
    gross_peak = _write_dense_matrix(out_dir / "gross_mult_peak.npy", (T, N), np.float32)
    gross_mean = _write_dense_matrix(out_dir / "gross_mult_mean.npy", (T, N), np.float32)
    buying_power_min = _write_dense_matrix(out_dir / "buying_power_min.npy", (T, N), np.float32)
    overnight_flag = _write_dense_matrix(out_dir / "overnight_flag.npy", (T, N), np.int8)

    session_index = {int(s): i for i, s in enumerate(calendar.tolist())}
    column_lookup = {str(pk): int(idx) for idx, pk in enumerate(column_index["strategy_instance_pk"].tolist())}
    returns_exec[:, :] = 0.0
    returns_raw[:, :] = 0.0
    availability[:, :] = False
    turnover[:, :] = 0.0
    state_codes[:, :] = 0
    gross_peak[:, :] = 0.0
    gross_mean[:, :] = 0.0
    buying_power_min[:, :] = 0.0
    overnight_flag[:, :] = 0

    canonical_sessions = session_ledger.loc[
        session_ledger["strategy_instance_pk"].isin(column_lookup.keys())
    ].copy()
    for row in canonical_sessions.itertuples(index=False):
        t = session_index[int(row.session_id)]
        n = column_lookup[str(row.strategy_instance_pk)]
        returns_exec[t, n] = float(row.return_exec)
        returns_raw[t, n] = float(row.return_raw)
        turnover[t, n] = np.float32(float(row.session_turnover))
        state_codes[t, n] = np.int16(int(row.availability_state_code))
        availability[t, n] = bool(int(row.availability_state_code) in BASE_AVAIL_ACTIVE_CODES)
        gross_peak[t, n] = np.float32(float(row.gross_mult_peak))
        gross_mean[t, n] = np.float32(float(row.gross_mult_mean))
        buying_power_min[t, n] = np.float32(float(getattr(row, "buying_power_min_frac", row.buying_power_min)))
        overnight_flag[t, n] = np.int8(int(row.overnight_flag))
    for name, arr in {
        "returns_exec": np.asarray(returns_exec),
        "returns_raw": np.asarray(returns_raw),
        "turnover": np.asarray(turnover),
        "gross_peak": np.asarray(gross_peak),
        "gross_mean": np.asarray(gross_mean),
    }.items():
        if not np.isfinite(arr).all():
            raise Module6ValidationError(f"{name} matrix contains non-finite values")

    family_values = sorted(strategy_master["family_id"].fillna("").astype(str).unique().tolist())
    family_index = {family: idx for idx, family in enumerate(family_values)}
    family_rows = column_index["family_id"].fillna("").astype(str).map(family_index).to_numpy(dtype=np.int64)
    family_incidence = sparse.csr_matrix(
        (
            np.ones(N, dtype=np.int8),
            (np.arange(N, dtype=np.int64), family_rows),
        ),
        shape=(N, len(family_values)),
        dtype=np.int8,
    )
    sparse.save_npz(out_dir / "family_incidence.npz", family_incidence)

    benchmark_daily = None
    if run.paths.daily_returns and run.paths.daily_returns.exists():
        daily_df = pd.read_parquet(run.paths.daily_returns)
        if "benchmark" in daily_df.columns:
            benchmark_daily = (
                daily_df[["session_id", "benchmark"]]
                .drop_duplicates("session_id", keep="last")
                .sort_values("session_id", kind="mergesort")
            )
    if benchmark_daily is None:
        benchmark_daily = pd.DataFrame({"session_id": calendar, "benchmark": np.zeros(T, dtype=np.float64)})
    benchmark_map = {int(s): float(v) for s, v in benchmark_daily[["session_id", "benchmark"]].itertuples(index=False, name=None)}
    benchmark_series = np.asarray([benchmark_map.get(int(s), 0.0) for s in calendar.tolist()], dtype=np.float64)
    rolling_vol = pd.Series(benchmark_series).rolling(5, min_periods=1).std(ddof=0).fillna(0.0).to_numpy(dtype=np.float64)
    vol_cut = float(np.median(rolling_vol))
    regime_idx = (
        (benchmark_series > 0.0).astype(np.int64) * 2
        + (rolling_vol > vol_cut).astype(np.int64)
    )
    K = 4
    regime_exposure = np.zeros((N, K), dtype=np.float32)
    for n in range(N):
        for k in range(K):
            mask = regime_idx == k
            if not np.any(mask):
                continue
            vals = np.asarray(returns_exec[mask, n], dtype=np.float64)
            obs = np.asarray(availability[mask, n], dtype=bool)
            if not np.any(obs):
                continue
            regime_exposure[n, k] = np.float32(float(np.mean(vals[obs])))
    np.save(out_dir / "regime_exposure.npy", regime_exposure)

    return MatrixStore(
        calendar=calendar,
        column_index=column_index,
        returns_exec_path=out_dir / "returns_exec.npy",
        returns_raw_path=out_dir / "returns_raw.npy",
        availability_path=out_dir / "availability.npy",
        turnover_path=out_dir / "activity_turnover.npy",
        state_code_path=out_dir / "availability_state_codes.npy",
        gross_peak_path=out_dir / "gross_mult_peak.npy",
        gross_mean_path=out_dir / "gross_mult_mean.npy",
        buying_power_min_path=out_dir / "buying_power_min.npy",
        overnight_flag_path=out_dir / "overnight_flag.npy",
        calendar_index_path=out_dir / "calendar_index.parquet",
        family_incidence_path=out_dir / "family_incidence.npz",
        regime_exposure_path=out_dir / "regime_exposure.npy",
    )
