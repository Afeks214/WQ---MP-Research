from __future__ import annotations

import numpy as np

from .schema import ContextIdx, StructIdx


def _session_spans(session_id_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(session_id_t, dtype=np.int64)
    if s.ndim != 1:
        raise RuntimeError(f"session_id_t must be 1D, got shape={s.shape}")
    T = int(s.shape[0])
    if T <= 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]]).astype(np.int64)
    ends = np.r_[starts[1:], T].astype(np.int64)
    return starts, ends


def _base_context_from_structure(structure_tensor: np.ndarray) -> np.ndarray:
    st = np.asarray(structure_tensor, dtype=np.float64)
    if st.ndim != 4:
        raise RuntimeError(f"structure_tensor must be [A,T,F_struct,W], got shape={st.shape}")
    A, T, _, W = st.shape
    out = np.full((A, T, int(ContextIdx.N_FIELDS), W), np.nan, dtype=np.float64)

    mapping = [
        (ContextIdx.CTX_X_POC, StructIdx.X_POC),
        (ContextIdx.CTX_X_VAH, StructIdx.X_VAH),
        (ContextIdx.CTX_X_VAL, StructIdx.X_VAL),
        (ContextIdx.CTX_VA_WIDTH_X, StructIdx.VA_WIDTH_X),
        (ContextIdx.CTX_DCLIP_MEAN, StructIdx.DCLIP_MEAN),
        (ContextIdx.CTX_AFFINITY_MEAN, StructIdx.AFFINITY_MEAN),
        (ContextIdx.CTX_ZDELTA_MEAN, StructIdx.ZDELTA_MEAN),
        (ContextIdx.CTX_DELTA_EFF_MEAN, StructIdx.DELTA_EFF_MEAN),
        (ContextIdx.CTX_TREND_GATE_SPREAD_MEAN, StructIdx.TREND_GATE_SPREAD_MEAN),
        (ContextIdx.CTX_POC_DRIFT_X, StructIdx.POC_DRIFT_X),
        (ContextIdx.CTX_VALID_RATIO, StructIdx.VALID_RATIO),
        (ContextIdx.CTX_IB_HIGH_X, StructIdx.IB_HIGH_X),
        (ContextIdx.CTX_IB_LOW_X, StructIdx.IB_LOW_X),
        (ContextIdx.CTX_POC_VS_PREV_VA, StructIdx.POC_VS_PREV_VA),
    ]
    for c_idx, s_idx in mapping:
        out[:, :, int(c_idx), :] = st[:, :, int(s_idx), :]
    return out


def _required_context_finite_mask(base_context: np.ndarray) -> np.ndarray:
    req = np.asarray(base_context, dtype=np.float64)[
        :, :, : int(ContextIdx.CTX_POC_VS_PREV_VA) + 1, :
    ]
    return np.all(np.isfinite(req), axis=2)


def build_context_tensor(
    structure_tensor: np.ndarray,
    profile_regime_tensor: np.ndarray,
    session_id_t: np.ndarray,
    *,
    mode: str = "ffill_last_complete",
    rolling_period: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build context tensor with causal, session-safe propagation.

    Returns:
        context_tensor [A,T,C,W], context_valid_atw [A,T,W], context_source_index_atw [A,T,W]
    """
    base = _base_context_from_structure(structure_tensor)
    regime = np.asarray(profile_regime_tensor, dtype=np.float64)
    if regime.ndim != 4 or regime.shape[2] != 1:
        raise RuntimeError(f"profile_regime_tensor must be [A,T,1,W], got shape={regime.shape}")
    A, T, C, W = base.shape

    ctx = np.full(base.shape, np.nan, dtype=np.float64)
    valid = np.zeros((A, T, W), dtype=bool)
    src_idx = np.full((A, T, W), -1, dtype=np.int64)
    base_row_ok = _required_context_finite_mask(base)

    starts, ends = _session_spans(session_id_t)
    mode_s = str(mode)
    rp = int(rolling_period)

    for a in range(A):
        for w in range(W):
            for s0, s1 in zip(starts.tolist(), ends.tolist()):
                last_idx = -1
                roll_sum = np.zeros(C, dtype=np.float64)
                roll_queue: list[np.ndarray] = []

                for t in range(int(s0), int(s1)):
                    row = base[a, t, :, w]
                    row_ok = bool(base_row_ok[a, t, w])

                    if mode_s == "ffill_last_complete":
                        if row_ok:
                            last_idx = t
                        if last_idx >= 0:
                            ctx[a, t, :, w] = base[a, last_idx, :, w]
                            valid[a, t, w] = True
                            src_idx[a, t, w] = int(last_idx)

                    elif mode_s == "rolling_context":
                        if row_ok:
                            roll_queue.append(row.copy())
                            roll_sum += row
                            if len(roll_queue) > rp:
                                roll_sum -= roll_queue.pop(0)
                            last_idx = t
                        if roll_queue:
                            ctx[a, t, :, w] = roll_sum / float(len(roll_queue))
                            valid[a, t, w] = True
                            src_idx[a, t, w] = int(last_idx)

                    elif mode_s == "regime_context":
                        if row_ok:
                            last_idx = t
                        if last_idx >= 0:
                            ctx[a, t, :, w] = base[a, last_idx, :, w]
                            valid[a, t, w] = True
                            src_idx[a, t, w] = int(last_idx)

                            r_now = float(regime[a, t, 0, w])
                            run = 1
                            k = t - 1
                            while k >= int(s0) and float(regime[a, k, 0, w]) == r_now:
                                run += 1
                                k -= 1
                            span = float(max(1, t - int(s0) + 1))
                            ctx[a, t, int(ContextIdx.CTX_REGIME_CODE), w] = r_now
                            ctx[a, t, int(ContextIdx.CTX_REGIME_PERSISTENCE), w] = float(run) / span
                    else:
                        raise RuntimeError(
                            "Unsupported context mode. Expected one of: "
                            "ffill_last_complete, rolling_context, regime_context"
                        )

    if mode_s != "regime_context":
        ctx[:, :, int(ContextIdx.CTX_REGIME_CODE), :] = 0.0
        ctx[:, :, int(ContextIdx.CTX_REGIME_PERSISTENCE), :] = 0.0

    return ctx, valid, src_idx
