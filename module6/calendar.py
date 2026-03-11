from __future__ import annotations

import numpy as np
import pandas as pd

from module6.utils import Module6ValidationError, stable_sha256_parts


def calendar_version_from_sessions(session_ids: np.ndarray | list[int]) -> str:
    sess = np.asarray(session_ids, dtype=np.int64).reshape(-1)
    if sess.size <= 0:
        raise Module6ValidationError("calendar_version cannot be computed from an empty session set")
    return stable_sha256_parts(*[int(x) for x in sess.tolist()])[:16]


def build_portfolio_calendar(strategy_session_returns: pd.DataFrame) -> np.ndarray:
    if strategy_session_returns.shape[0] <= 0:
        raise Module6ValidationError("strategy_session_returns is empty")
    sessions = np.asarray(
        sorted(pd.unique(strategy_session_returns["session_id"]).tolist()),
        dtype=np.int64,
    )
    if sessions.size <= 0:
        raise Module6ValidationError("no session_ids found for portfolio calendar")
    return sessions


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

