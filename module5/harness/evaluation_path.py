from __future__ import annotations

from typing import Any, Callable

import numpy as np

from weightiz_module1_core import Phase, ProfileStatIdx, ScoreIdx, TensorState
from weightiz_module3_structure import ContextIdx, Module3Output
from weightiz_module4_strategy_funnel import Module4SignalOutput, RegimeIdx


def materialize_risk_outputs_into_state(
    state: TensorState,
    m4_sig: Module4SignalOutput,
    risk_res: Any,
    execution_view_cls: type,
) -> Any:
    t_count = state.cfg.T
    a_count = state.cfg.A
    filled = np.asarray(risk_res.filled_qty_ta, dtype=np.float64)
    exec_px = np.asarray(risk_res.exec_price_ta, dtype=np.float64)
    tcost = np.asarray(risk_res.trade_cost_ta, dtype=np.float64)
    pos = np.asarray(risk_res.position_qty_ta, dtype=np.float64)
    if filled.shape != (t_count, a_count) or exec_px.shape != (t_count, a_count) or tcost.shape != (t_count, a_count) or pos.shape != (t_count, a_count):
        raise RuntimeError("risk_engine output shape mismatch")

    state.equity[:] = np.asarray(risk_res.equity_curve, dtype=np.float64)
    state.position_qty[:, :] = pos
    state.margin_used[:] = np.asarray(risk_res.margin_used_t, dtype=np.float64)
    state.buying_power[:] = np.maximum(
        0.0,
        float(state.cfg.intraday_leverage_max) * state.equity - state.margin_used,
    )
    state.daily_loss[:] = np.asarray(risk_res.daily_loss_t, dtype=np.float64)
    side = np.zeros((t_count, a_count), dtype=np.int8)
    side[filled > 0.0] = 1
    side[filled < 0.0] = -1
    state.order_side[:, :] = side
    state.order_flags[:, :] = np.uint16(0)

    return execution_view_cls(
        regime_primary_ta=np.asarray(m4_sig.regime_primary_ta, dtype=np.int8),
        regime_confidence_ta=np.asarray(m4_sig.regime_confidence_ta, dtype=np.float64),
        intent_long_ta=np.asarray(m4_sig.intent_long_ta, dtype=bool),
        intent_short_ta=np.asarray(m4_sig.intent_short_ta, dtype=bool),
        target_qty_ta=np.asarray(m4_sig.target_qty_ta, dtype=np.float64),
        filled_qty_ta=filled,
        exec_price_ta=exec_px,
        trade_cost_ta=tcost,
        overnight_score_ta=np.zeros((t_count, a_count), dtype=np.float64),
        overnight_winner_t=np.full(t_count, -1, dtype=np.int16),
        kill_switch_t=np.zeros(t_count, dtype=bool),
    )


def candidate_daily_returns_close_to_close(
    state: TensorState,
    split: Any,
    initial_cash: float,
    session_bounds_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
    equity_curve: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _starts, ends, sessions = session_bounds_fn(state.session_id)
    test_sessions = np.unique(state.session_id[split.test_idx].astype(np.int64))

    if test_sessions.size == 0:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
        )

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

    eq_src = np.asarray(state.equity if equity_curve is None else equity_curve, dtype=np.float64)
    if eq_src.ndim != 1 or eq_src.shape[0] != int(state.cfg.T):
        raise RuntimeError(
            f"equity_curve shape mismatch: got {eq_src.shape}, expected {(int(state.cfg.T),)}"
        )
    eq_close = eq_src[idx].astype(np.float64)
    ret = np.empty(idx.size, dtype=np.float64)
    ret[0] = eq_close[0] / float(initial_cash) - 1.0
    if idx.size > 1:
        ret[1:] = eq_close[1:] / np.maximum(eq_close[:-1], 1e-12) - 1.0

    return sess_ids, idx, ret


def asset_pnl_by_symbol_from_state(
    state: TensorState,
    split: Any,
) -> dict[str, float]:
    idx = np.unique(np.asarray(split.test_idx, dtype=np.int64))
    if idx.size == 0:
        return {}

    a_count = int(state.cfg.A)
    contrib = np.zeros(a_count, dtype=np.float64)

    close = np.asarray(state.close_px, dtype=np.float64)
    pos = np.asarray(state.position_qty, dtype=np.float64)
    valid = np.asarray(state.bar_valid, dtype=bool)

    for t in idx.tolist():
        t_i = int(t)
        if t_i <= 0:
            continue
        p_i = t_i - 1
        mask = (
            valid[t_i]
            & valid[p_i]
            & np.isfinite(pos[p_i])
            & np.isfinite(close[t_i])
            & np.isfinite(close[p_i])
        )
        if not np.any(mask):
            continue
        contrib[mask] += pos[p_i, mask] * (close[t_i, mask] - close[p_i, mask])

    out: dict[str, float] = {}
    for a in range(a_count):
        v = float(contrib[a])
        if np.isfinite(v) and abs(v) > 1e-18:
            out[str(state.symbols[a])] = v
    return out


def benchmark_daily_returns(
    state: TensorState,
    benchmark_symbol: str,
    session_bounds_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    starts, ends, sessions = session_bounds_fn(state.session_id)

    a_count = state.cfg.A
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


def equity_curve_payload(
    state: TensorState,
    candidate_id: str,
    split_id: str,
    scenario_id: str,
) -> dict[str, np.ndarray]:
    eq = state.equity.astype(np.float64)
    peak = np.maximum.accumulate(eq)
    dd = np.where(peak > 0.0, eq / peak - 1.0, 0.0)
    t_count = state.cfg.T

    return {
        "ts_ns": state.ts_ns.copy(),
        "session_id": state.session_id.copy(),
        "candidate_id": np.full(t_count, candidate_id, dtype=object),
        "split_id": np.full(t_count, split_id, dtype=object),
        "scenario_id": np.full(t_count, scenario_id, dtype=object),
        "equity": eq.copy(),
        "drawdown": dd.astype(np.float64),
        "margin_used": state.margin_used.copy(),
        "buying_power": state.buying_power.copy(),
        "daily_loss": state.daily_loss.copy(),
    }


def trade_log_payload(
    state: TensorState,
    m4_out: Any,
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


def event_window_mask(t_count: int, event_idx: np.ndarray, pre: int, post: int) -> np.ndarray:
    mask = np.zeros(t_count, dtype=bool)
    if event_idx.size == 0:
        return mask
    lo_off = int(max(0, pre))
    hi_off = int(max(0, post))
    for i in event_idx.tolist():
        lo = max(0, int(i) - lo_off)
        hi = min(t_count, int(i) + hi_off + 1)
        mask[lo:hi] = True
    return mask


def structural_weight_from_regime(regime_i8: np.ndarray) -> np.ndarray:
    r = np.asarray(regime_i8, dtype=np.int8)
    w = np.zeros(r.shape, dtype=np.float64)
    w[(r == np.int8(RegimeIdx.P_SHAPE)) | (r == np.int8(RegimeIdx.B_SHAPE))] = 1.5
    w[r == np.int8(RegimeIdx.TREND)] = 1.2
    return w


def select_micro_rows(
    state: TensorState,
    split: Any,
    cfg: Any,
    m4_out: Any,
    enabled_assets_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    t_count = state.cfg.T
    a_count = state.cfg.A
    mode = str(cfg.micro_diag_mode).strip().lower()

    a_mask = np.asarray(enabled_assets_mask, dtype=bool).copy()
    if a_mask.shape != (a_count,):
        raise RuntimeError(
            f"enabled_assets_mask shape mismatch for micro diagnostics: got {a_mask.shape}, expected {(a_count,)}"
        )

    if cfg.micro_diag_symbols:
        symbol_set = set(str(s) for s in cfg.micro_diag_symbols)
        a_mask &= np.asarray([s in symbol_set for s in state.symbols], dtype=bool)

    if mode == "off":
        return np.zeros(t_count, dtype=bool), a_mask

    if mode == "full_test":
        t_mask = np.zeros(t_count, dtype=bool)
        t_mask[split.test_idx] = True
    elif mode == "symbol_day":
        t_mask = np.ones(t_count, dtype=bool)
        if cfg.micro_diag_session_ids:
            sset = set(int(s) for s in cfg.micro_diag_session_ids)
            t_mask &= np.isin(state.session_id.astype(np.int64), np.asarray(sorted(sset), dtype=np.int64))
    elif mode == "events_only":
        fills_t = np.flatnonzero(np.any(np.abs(m4_out.filled_qty_ta) > 1e-12, axis=1)).astype(np.int64)
        select_t = np.flatnonzero(state.phase == np.int8(Phase.OVERNIGHT_SELECT)).astype(np.int64)
        event_idx = np.unique(np.r_[fills_t, select_t]).astype(np.int64)
        t_mask = event_window_mask(
            t_count=t_count,
            event_idx=event_idx,
            pre=int(cfg.micro_diag_trade_window_pre),
            post=int(cfg.micro_diag_trade_window_post),
        )
    else:
        raise RuntimeError(f"Unsupported micro_diag_mode: {cfg.micro_diag_mode}")

    valid_any = np.any(state.bar_valid[:, a_mask], axis=1) if np.any(a_mask) else np.zeros(t_count, dtype=bool)
    t_mask &= valid_any
    return t_mask, a_mask


def collect_micro_diagnostics_payload(
    state: TensorState,
    m3: Module3Output,
    m4_out: Any,
    candidate_id: str,
    split_id: str,
    scenario_id: str,
    split: Any,
    enabled_assets_mask: np.ndarray,
    cfg: Any,
) -> dict[str, np.ndarray] | None:
    if not bool(cfg.export_micro_diagnostics):
        return None

    t_mask, a_mask = select_micro_rows(state, split, cfg, m4_out, enabled_assets_mask)
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
    context_valid_ta = m3.context_valid_ta[t_idx, a_idx].astype(np.int8)
    context_source_t_index_ta = m3.context_source_t_index_ta[t_idx, a_idx].astype(np.int64)

    context_valid_any_window = None
    context_valid_all_windows = None
    context_source_index_first_valid_window = None
    context_source_index_last_valid_window = None
    if getattr(m3, "context_valid_atw", None) is not None:
        context_valid_rows = np.asarray(m3.context_valid_atw[a_idx, t_idx, :], dtype=bool)
        context_valid_any_window = np.any(context_valid_rows, axis=1).astype(np.int8)
        context_valid_all_windows = np.all(context_valid_rows, axis=1).astype(np.int8)
        if getattr(m3, "context_source_index_atw", None) is not None:
            context_source_rows = np.asarray(m3.context_source_index_atw[a_idx, t_idx, :], dtype=np.int64)
            first_valid = np.full(t_idx.shape[0], -1, dtype=np.int64)
            last_valid = np.full(t_idx.shape[0], -1, dtype=np.int64)
            for row_idx in range(t_idx.shape[0]):
                valid_windows = np.flatnonzero(context_valid_rows[row_idx])
                if valid_windows.size == 0:
                    continue
                first_valid[row_idx] = context_source_rows[row_idx, int(valid_windows[0])]
                last_valid[row_idx] = context_source_rows[row_idx, int(valid_windows[-1])]
            context_source_index_first_valid_window = first_valid
            context_source_index_last_valid_window = last_valid

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
        "context_valid_ta": context_valid_ta,
        "context_source_t_index_ta": context_source_t_index_ta,
        "context_valid_any_window": context_valid_any_window,
        "context_valid_all_windows": context_valid_all_windows,
        "context_source_index_first_valid_window": context_source_index_first_valid_window,
        "context_source_index_last_valid_window": context_source_index_last_valid_window,
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


def collect_micro_profile_blocks_payload(
    state: TensorState,
    m3: Module3Output,
    candidate_id: str,
    split_id: str,
    scenario_id: str,
    enabled_assets_mask: np.ndarray,
    cfg: Any,
) -> dict[str, np.ndarray] | None:
    if not (bool(cfg.export_micro_diagnostics) and bool(cfg.micro_diag_export_block_profiles)):
        return None

    a_count = state.cfg.A
    a_mask = np.asarray(enabled_assets_mask, dtype=bool)
    if a_mask.shape != (a_count,):
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

    x_blob = state.x_grid.astype(np.float64).tobytes()
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
        "vp_block_blob": np.asarray([state.vp[int(t), int(a)].astype(np.float64).tobytes() for t, a in zip(rr.tolist(), aa.tolist())], dtype=object),
        "vp_delta_block_blob": np.asarray([state.vp_delta[int(t), int(a)].astype(np.float64).tobytes() for t, a in zip(rr.tolist(), aa.tolist())], dtype=object),
        "close_te": state.close_px[rr, aa].astype(np.float64),
        "atr_eff_te": state.atr_floor[rr, aa].astype(np.float64),
    }


def collect_funnel_payload(
    state: TensorState,
    m4_out: Any,
    candidate_id: str,
    split_id: str,
    scenario_id: str,
    enabled_assets_mask: np.ndarray,
    cfg: Any,
    require_pandas_fn: Callable[[], Any],
) -> dict[str, np.ndarray] | None:
    if not (bool(cfg.export_micro_diagnostics) and bool(cfg.micro_diag_export_funnel)):
        return None

    a_count = state.cfg.A
    a_mask = np.asarray(enabled_assets_mask, dtype=bool)
    if a_mask.shape != (a_count,):
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
        sw = structural_weight_from_regime(regime)
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

    pdx = require_pandas_fn()
    df = pdx.DataFrame(out_rows)
    return {k: df[k].to_numpy() for k in df.columns.tolist()}
