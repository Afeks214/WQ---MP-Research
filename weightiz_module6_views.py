"""
Weightiz Module 6 - Plotly View Builders
========================================
"""

from __future__ import annotations

from typing import Any

import numpy as np

from weightiz_module6_data import rolling_calmar, rolling_sharpe, to_et_datetime, x_to_price


def _require_plotly() -> tuple[Any, Any]:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("plotly is required for Module 6 visualizations") from exc
    return go, make_subplots


def build_macro_figure(
    equity_df: Any,
    daily_df: Any,
    task_id: str,
    rolling_window_days: int,
    calmar_window_days: int,
    risk_free_daily: float,
    leverage_ref: float,
    timezone: str,
) -> Any:
    go, make_subplots = _require_plotly()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Equity & Drawdown", "Rolling Sharpe / Calmar", "Margin & Buying Power Utilization"),
    )

    if len(equity_df) == 0:
        fig.add_annotation(text="No equity data", x=0.5, y=0.5, showarrow=False)
        return fig

    ts = to_et_datetime(equity_df["ts_ns"].to_numpy(dtype=np.int64), timezone=timezone)
    eq = equity_df["equity"].to_numpy(dtype=np.float64)
    dd = equity_df["drawdown"].to_numpy(dtype=np.float64)
    margin = equity_df["margin_used"].to_numpy(dtype=np.float64)
    bp = equity_df["buying_power"].to_numpy(dtype=np.float64)

    fig.add_trace(go.Scattergl(x=ts, y=eq, name="Equity", line=dict(color="#0f766e", width=2)), row=1, col=1)
    fig.add_trace(go.Scattergl(x=ts, y=dd, name="Drawdown", line=dict(color="#b91c1c", width=1.5)), row=1, col=1)

    if task_id in daily_df.columns:
        r_d = daily_df[task_id].to_numpy(dtype=np.float64)
        r_ts = daily_df["session_id"].to_numpy(dtype=np.int64)
        rs = rolling_sharpe(r_d, rolling_window_days, risk_free_daily=risk_free_daily)
        rc = rolling_calmar(r_d, calmar_window_days)
        fig.add_trace(go.Scatter(x=r_ts, y=rs, name="Rolling Sharpe", line=dict(color="#1d4ed8", width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=r_ts, y=rc, name="Rolling Calmar", line=dict(color="#f59e0b", width=2)), row=2, col=1)

    denom = np.maximum(eq * float(leverage_ref), 1e-12)
    u_margin = margin / denom
    u_bp = 1.0 - bp / denom
    fig.add_trace(go.Scattergl(x=ts, y=u_margin, name="Margin Util", line=dict(color="#7c3aed", width=1.5)), row=3, col=1)
    fig.add_trace(go.Scattergl(x=ts, y=u_bp, name="BP Util", line=dict(color="#ea580c", width=1.5)), row=3, col=1)
    fig.add_hline(y=1.0, line_dash="dash", line_color="#ef4444", row=3, col=1)

    fig.update_layout(height=900, template="plotly_white", hovermode="x unified")
    return fig


def build_mfe_mae_figure(episodes_df: Any, timezone: str) -> Any:
    go, _ = _require_plotly()
    fig = go.Figure()
    if episodes_df is None or len(episodes_df) == 0:
        fig.add_annotation(text="No MFE/MAE episodes available", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_white")
        return fig

    mae = np.abs(episodes_df["mae"].to_numpy(dtype=np.float64))
    mfe = episodes_df["mfe"].to_numpy(dtype=np.float64)
    win = episodes_df["win"].to_numpy(dtype=np.int8)
    color = np.where(win > 0, "#16a34a", "#dc2626")
    size = np.clip(np.sqrt(np.maximum(episodes_df["notional"].to_numpy(dtype=np.float64), 0.0)) / 20.0, 6.0, 22.0)

    fig.add_trace(
        go.Scattergl(
            x=mae,
            y=mfe,
            mode="markers",
            marker=dict(color=color, size=size, opacity=0.75, line=dict(width=0.4, color="#111827")),
            text=episodes_df["symbol"].astype(str),
            hovertemplate="symbol=%{text}<br>|MAE|=%{x:.4f}<br>MFE=%{y:.4f}<extra></extra>",
            name="Episodes",
        )
    )
    fig.update_layout(template="plotly_white", title="MFE vs MAE", xaxis_title="|MAE|", yaxis_title="MFE")
    return fig


def _decode_blob_f32(blob: bytes) -> np.ndarray:
    if blob is None:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(blob, dtype=np.float32)


def build_micro_matrix_figure(
    micro_day_df: Any,
    profile_blocks_df: Any | None,
    max_profile_blocks_render: int,
    timezone: str,
) -> Any:
    go, make_subplots = _require_plotly()
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
        subplot_titles=("Candles + Dynamic VA + Anchored Profiles", "Hybrid Delta z_delta"),
    )

    if micro_day_df is None or len(micro_day_df) == 0:
        fig.add_annotation(text="No micro diagnostics available", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_white", height=900)
        return fig

    d = micro_day_df.sort_values("ts_ns", kind="mergesort").reset_index(drop=True)
    ts = to_et_datetime(d["ts_ns"].to_numpy(dtype=np.int64), timezone=timezone)
    ts_list = list(pd.to_datetime(ts, errors="coerce"))

    fig.add_trace(
        go.Candlestick(
            x=ts,
            open=d["open"],
            high=d["high"],
            low=d["low"],
            close=d["close"],
            name="OHLC",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    if all(c in d.columns for c in ("ctx_x_poc", "ctx_x_vah", "ctx_x_val", "atr_eff", "close")):
        poc_px = x_to_price(d["close"].to_numpy(dtype=np.float64), d["ctx_x_poc"].to_numpy(dtype=np.float64), d["atr_eff"].to_numpy(dtype=np.float64))
        vah_px = x_to_price(d["close"].to_numpy(dtype=np.float64), d["ctx_x_vah"].to_numpy(dtype=np.float64), d["atr_eff"].to_numpy(dtype=np.float64))
        val_px = x_to_price(d["close"].to_numpy(dtype=np.float64), d["ctx_x_val"].to_numpy(dtype=np.float64), d["atr_eff"].to_numpy(dtype=np.float64))
        fig.add_trace(go.Scattergl(x=ts, y=poc_px, name="POC", line=dict(color="#111827", width=1.6)), row=1, col=1)
        fig.add_trace(go.Scattergl(x=ts, y=vah_px, name="VAH", line=dict(color="#2563eb", width=1.2)), row=1, col=1)
        fig.add_trace(go.Scattergl(x=ts, y=val_px, name="VAL", line=dict(color="#2563eb", width=1.2)), row=1, col=1)

    if profile_blocks_df is not None and len(profile_blocks_df) > 0:
        pb = profile_blocks_df.sort_values("ts_ns", kind="mergesort").tail(int(max_profile_blocks_render)).reset_index(drop=True)
        for _, r in pb.iterrows():
            x_blob = r.get("x_grid_blob", None)
            vp_blob = r.get("vp_block_blob", None)
            if not isinstance(x_blob, (bytes, bytearray)) or not isinstance(vp_blob, (bytes, bytearray)):
                continue
            x_grid = _decode_blob_f32(bytes(x_blob)).astype(np.float64)
            vp = _decode_blob_f32(bytes(vp_blob)).astype(np.float64)
            if x_grid.size == 0 or vp.size == 0 or x_grid.size != vp.size:
                continue

            p_te = float(r.get("close_te", np.nan))
            a_te = float(r.get("atr_eff_te", np.nan))
            if (not np.isfinite(p_te)) or (not np.isfinite(a_te)):
                continue
            p_bins = p_te + x_grid * a_te
            mass = vp / max(float(np.nansum(vp)), 1e-12)

            te = to_et_datetime(np.array([int(r["ts_ns"])], dtype=np.int64), timezone=timezone)[0]
            sec_scale = 30.0 * 60.0 * 0.45
            dt_sec = np.clip(mass, 0.0, None) * sec_scale
            left = [te - np.timedelta64(int(s), "s") for s in dt_sec]
            right = [te + np.timedelta64(int(s), "s") for s in dt_sec]

            x_poly = np.array(right.tolist() + left[::-1].tolist(), dtype="datetime64[ns]")
            y_poly = np.r_[p_bins, p_bins[::-1]]

            fig.add_trace(
                go.Scatter(
                    x=x_poly,
                    y=y_poly,
                    mode="lines",
                    line=dict(width=0.0),
                    fill="toself",
                    fillcolor="rgba(30,136,229,0.14)",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )

    if "z_delta" in d.columns:
        fig.add_trace(
            go.Bar(x=ts, y=d["z_delta"], name="z_delta", marker_color="#0ea5e9", opacity=0.85),
            row=2,
            col=1,
        )
        fig.add_hline(y=0.0, line_color="#111827", line_width=1, row=2, col=1)

    fig.update_layout(height=950, template="plotly_white", hovermode="x unified", xaxis_rangeslider_visible=False)
    return fig


def build_brain_figure(
    micro_day_df: Any,
    entry_threshold: float,
    exit_threshold: float | None,
    timezone: str,
) -> Any:
    go, make_subplots = _require_plotly()
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

    if micro_day_df is None or len(micro_day_df) == 0:
        fig.add_annotation(text="No micro diagnostics available", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_white")
        return fig

    d = micro_day_df.sort_values("ts_ns", kind="mergesort").reset_index(drop=True)
    ts = to_et_datetime(d["ts_ns"].to_numpy(dtype=np.int64), timezone=timezone)

    bo = np.maximum(
        d.get("score_bo_long", 0.0).to_numpy(dtype=np.float64),
        d.get("score_bo_short", 0.0).to_numpy(dtype=np.float64),
    )
    rej = np.maximum(
        d.get("score_rej_long", 0.0).to_numpy(dtype=np.float64),
        d.get("score_rej_short", 0.0).to_numpy(dtype=np.float64),
    )

    fig.add_trace(go.Scattergl(x=ts, y=bo, name="SCORE_BO", line=dict(color="#1d4ed8", width=2)))
    fig.add_trace(go.Scattergl(x=ts, y=rej, name="SCORE_REJ", line=dict(color="#059669", width=2)))

    fig.add_hline(y=float(entry_threshold), line_dash="dash", line_color="#dc2626")
    if exit_threshold is not None:
        fig.add_hline(y=float(exit_threshold), line_dash="dot", line_color="#f59e0b")

    if "regime_primary" in d.columns:
        reg = d["regime_primary"].to_numpy(dtype=np.int8)
        cmap = {
            2: "rgba(59,130,246,0.12)",
            3: "rgba(22,163,74,0.12)",
            4: "rgba(220,38,38,0.12)",
            5: "rgba(245,158,11,0.14)",
            1: "rgba(107,114,128,0.10)",
            0: "rgba(0,0,0,0.0)",
        }
        # Contiguous spans.
        start = 0
        for i in range(1, len(reg) + 1):
            if i == len(reg) or reg[i] != reg[start]:
                color = cmap.get(int(reg[start]), "rgba(0,0,0,0.0)")
                if color != "rgba(0,0,0,0.0)":
                    if start < 0 or (i - 1) < 0 or start >= len(ts_list) or (i - 1) >= len(ts_list):
                        continue
                    if pd.isna(ts_list[start]) or pd.isna(ts_list[i - 1]):
                        continue
                    fig.add_vrect(x0=ts_list[start], x1=ts_list[i - 1], fillcolor=color, line_width=0, layer="below")
                start = i

    if "intent_long" in d.columns:
        il = d["intent_long"].to_numpy(dtype=np.int8) > 0
        if np.any(il):
            fig.add_trace(
                go.Scattergl(
                    x=np.asarray(ts)[il],
                    y=bo[il],
                    mode="markers",
                    marker=dict(color="#16a34a", symbol="triangle-up", size=7),
                    name="Intent Long",
                )
            )
    if "intent_short" in d.columns:
        is_ = d["intent_short"].to_numpy(dtype=np.int8) > 0
        if np.any(is_):
            fig.add_trace(
                go.Scattergl(
                    x=np.asarray(ts)[is_],
                    y=bo[is_],
                    mode="markers",
                    marker=dict(color="#dc2626", symbol="triangle-down", size=7),
                    name="Intent Short",
                )
            )

    fig.update_layout(template="plotly_white", hovermode="x unified", height=520)
    return fig


def build_funnel_bar_figure(funnel_table: Any, timezone: str) -> Any:
    go, _ = _require_plotly()
    fig = go.Figure()
    if funnel_table is None or len(funnel_table) == 0:
        fig.add_annotation(text="No 15:45 funnel data", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_white")
        return fig

    t = funnel_table.sort_values(["ocs", "symbol"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    winner = t["is_winner"].to_numpy(dtype=np.int8)
    color = np.where(winner > 0, "#16a34a", "#64748b")

    fig.add_trace(
        go.Bar(
            x=t["symbol"].astype(str),
            y=t["ocs"].to_numpy(dtype=np.float64),
            marker_color=color,
            text=np.round(t["ocs"].to_numpy(dtype=np.float64), 4),
            textposition="outside",
            name="OCS",
        )
    )
    fig.update_layout(template="plotly_white", title="15:45 Overnight Conviction Score", yaxis_title="OCS")
    return fig
