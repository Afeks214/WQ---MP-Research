from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from weightiz.module6.utils import Module6ValidationError, stable_sha256_parts


@dataclass(frozen=True)
class PortfolioCalendarFrame:
    sessions: np.ndarray
    frame: pd.DataFrame


def calendar_version_from_sessions(session_ids: np.ndarray | list[int]) -> str:
    sess = np.asarray(session_ids, dtype=np.int64).reshape(-1)
    if sess.size <= 0:
        raise Module6ValidationError("calendar_version cannot be computed from an empty session set")
    return stable_sha256_parts(*[int(x) for x in sess.tolist()])[:16]


def build_portfolio_calendar_frame(
    strategy_session_returns: pd.DataFrame,
    equity_curves: pd.DataFrame,
) -> PortfolioCalendarFrame:
    if strategy_session_returns.shape[0] <= 0:
        raise Module6ValidationError("strategy_session_returns is empty")
    sessions = pd.DataFrame(
        {"session_id": np.asarray(sorted(pd.unique(strategy_session_returns["session_id"]).tolist()), dtype=np.int64)}
    )
    if sessions.shape[0] <= 0:
        raise Module6ValidationError("no session_ids found for portfolio calendar")
    eq = equity_curves[["session_id", "ts_ns"]].copy()
    eq["session_id"] = eq["session_id"].astype(np.int64)
    eq["ts_ns"] = eq["ts_ns"].astype(np.int64)
    session_ts = (
        eq.groupby("session_id", dropna=False)["ts_ns"]
        .min()
        .rename("session_open_ts_ns")
        .reset_index()
    )
    frame = sessions.merge(session_ts, on="session_id", how="left")
    if frame["session_open_ts_ns"].isna().any():
        missing = frame.loc[frame["session_open_ts_ns"].isna(), "session_id"].astype(int).tolist()
        raise Module6ValidationError(f"calendar frame missing session timestamps: {missing[:10]}")
    frame["session_open_ts_ns"] = frame["session_open_ts_ns"].astype(np.int64)
    frame["session_open_utc"] = pd.to_datetime(frame["session_open_ts_ns"], utc=True, errors="coerce")
    if frame["session_open_utc"].isna().any():
        raise Module6ValidationError("calendar frame contains non-convertible session timestamps")
    frame["session_date"] = frame["session_open_utc"].dt.strftime("%Y-%m-%d")
    frame["weekday"] = frame["session_open_utc"].dt.dayofweek.astype(np.int8)
    frame["week_token"] = frame["session_open_utc"].dt.strftime("%G-%V")
    frame["week_index"] = pd.factorize(frame["week_token"], sort=True)[0].astype(np.int32)
    frame["is_monday_close"] = (frame["weekday"] == 0).astype(np.int8)
    return PortfolioCalendarFrame(
        sessions=frame["session_id"].to_numpy(dtype=np.int64),
        frame=frame.sort_values("session_id", kind="mergesort").reset_index(drop=True),
    )


def build_portfolio_calendar(strategy_session_returns: pd.DataFrame, equity_curves: pd.DataFrame) -> np.ndarray:
    return build_portfolio_calendar_frame(strategy_session_returns=strategy_session_returns, equity_curves=equity_curves).sessions


def align_series_to_calendar(
    session_ids: np.ndarray | list[int],
    values: np.ndarray | list[float],
    calendar: np.ndarray,
    fill_value: float,
) -> np.ndarray:
    sess = np.asarray(session_ids, dtype=np.int64).reshape(-1)
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    if sess.size != vals.size:
        raise Module6ValidationError(
            f"session/value length mismatch during calendar alignment; sessions={int(sess.size)} values={int(vals.size)}"
        )
    mapping = {int(s): float(v) for s, v in zip(sess.tolist(), vals.tolist())}
    return np.asarray([float(mapping.get(int(s), fill_value)) for s in np.asarray(calendar, dtype=np.int64).tolist()], dtype=np.float64)


def common_support_mask(availability: np.ndarray, column_indices: np.ndarray) -> np.ndarray:
    arr = np.asarray(availability, dtype=bool)
    idx = np.asarray(column_indices, dtype=np.int64).reshape(-1)
    if arr.ndim != 2:
        raise Module6ValidationError(f"availability matrix must be 2D; ndim={arr.ndim}")
    if idx.size <= 0:
        raise Module6ValidationError("common_support_mask requires at least one column index")
    return np.all(arr[:, idx], axis=1)
