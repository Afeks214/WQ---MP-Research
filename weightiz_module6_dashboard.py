"""
Weightiz Module 6 - Deep-Quant Visualizer (Streamlit)
======================================================

Run:
    streamlit run /Users/afekshusterman/Documents/New\ project/weightiz_module6_dashboard.py
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from weightiz_module6_data import (
    Module6Config,
    build_funnel_table,
    candidate_filter,
    compute_episode_mfe_mae,
    list_run_ids,
    load_run_bundle,
)
from weightiz_module6_views import (
    build_brain_figure,
    build_funnel_bar_figure,
    build_macro_figure,
    build_mfe_mae_figure,
    build_micro_matrix_figure,
)


def _require_streamlit() -> Any:
    try:
        import streamlit as st
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("streamlit is required to run Module 6 dashboard") from exc
    return st


def _infer_leverage_ref(cfg: Module6Config, manifest: dict[str, Any]) -> float:
    if not bool(cfg.use_manifest_leverage_when_available):
        return float(cfg.intraday_leverage_ref)
    # Current run manifest does not store explicit leverage numbers yet.
    # Keep deterministic fallback until manifest fields are extended.
    _ = manifest
    return float(cfg.intraday_leverage_ref)


def _render_data_qa(st: Any, bundle: dict[str, Any]) -> None:
    st.subheader("Artifact QA")
    eq = bundle["equity"]
    tr = bundle["trade"]
    dr = bundle["daily"]
    st.write({
        "run_id": bundle["run_id"],
        "equity_rows": int(len(eq)),
        "trade_rows": int(len(tr)),
        "daily_rows": int(len(dr)),
        "micro_rows": int(len(bundle["micro"])) if bundle["micro"] is not None else 0,
        "profile_block_rows": int(len(bundle["profile"])) if bundle["profile"] is not None else 0,
        "funnel_rows": int(len(bundle["funnel"])) if bundle["funnel"] is not None else 0,
    })

    with st.expander("Schema Preview", expanded=False):
        st.write("equity columns", list(eq.columns))
        st.write("trade columns", list(tr.columns))
        st.write("daily columns", list(dr.columns))

    with st.expander("Missing Value Ratios", expanded=False):
        miss_eq = (eq.isna().sum() / max(len(eq), 1)).sort_values(ascending=False)
        miss_tr = (tr.isna().sum() / max(len(tr), 1)).sort_values(ascending=False)
        st.write("equity missing ratios")
        st.dataframe(miss_eq.to_frame("missing_ratio"), use_container_width=True)
        st.write("trade missing ratios")
        st.dataframe(miss_tr.to_frame("missing_ratio"), use_container_width=True)


def _main() -> None:
    st = _require_streamlit()

    st.set_page_config(
        page_title="Weightiz Deep-Quant Visualizer",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    cfg = Module6Config()

    st.title("Weightiz Deep-Quant Visualizer")
    st.caption("Institutional forensic dashboard over Module 5 artifacts")

    @st.cache_data(show_spinner=False)
    def _load(run_id: str, cfg_dict: dict[str, Any]) -> dict[str, Any]:
        c = Module6Config(**cfg_dict)
        return load_run_bundle(c, run_id=run_id)

    run_ids = list_run_ids(cfg.artifacts_root)
    if not run_ids:
        st.error(f"No runs found under {cfg.artifacts_root}")
        st.stop()

    default_run = cfg.default_run_id if cfg.default_run_id in run_ids else run_ids[-1]
    run_id = st.sidebar.selectbox("run_id", options=run_ids, index=run_ids.index(default_run))

    bundle = _load(run_id, asdict(cfg))

    eq = bundle["equity"]
    tr = bundle["trade"]
    daily = bundle["daily"]
    micro = bundle["micro"]
    profile = bundle["profile"]
    funnel = bundle["funnel"]
    manifest = bundle["manifest"]

    candidates = sorted(eq["candidate_id"].astype(str).unique().tolist())
    candidate_id = st.sidebar.selectbox("candidate_id", options=candidates)

    eq_c = eq[eq["candidate_id"].astype(str) == candidate_id]
    splits = sorted(eq_c["split_id"].astype(str).unique().tolist())
    split_id = st.sidebar.selectbox("split_id", options=splits)

    eq_cs = eq_c[eq_c["split_id"].astype(str) == split_id]
    scenarios = sorted(eq_cs["scenario_id"].astype(str).unique().tolist())
    scenario_id = st.sidebar.selectbox("scenario_id", options=scenarios)

    task_id = f"{candidate_id}|{split_id}|{scenario_id}"
    eq_sel = candidate_filter(eq, candidate_id, split_id, scenario_id)
    tr_sel = candidate_filter(tr, candidate_id, split_id, scenario_id)

    lev_ref = _infer_leverage_ref(cfg, manifest)

    st.sidebar.markdown("---")
    st.sidebar.write("task_id", task_id)
    st.sidebar.write("leverage_ref", lev_ref)

    tabs = st.tabs(["Macro & Risk", "Physics & Structure", "Decision Brain", "15:45 Funnel", "Data QA"])

    with tabs[0]:
        st.subheader("Macro Portfolio & Risk")
        fig_macro = build_macro_figure(
            equity_df=eq_sel,
            daily_df=daily,
            task_id=task_id,
            rolling_window_days=int(cfg.rolling_window_days),
            calmar_window_days=int(cfg.calmar_window_days),
            risk_free_daily=float(cfg.risk_free_daily),
            leverage_ref=float(lev_ref),
            timezone=cfg.timezone,
        )
        st.plotly_chart(fig_macro, use_container_width=True)

        mfe_df = None
        if micro is not None:
            micro_sel = candidate_filter(micro, candidate_id, split_id, scenario_id)
            mfe_df = compute_episode_mfe_mae(micro_sel, tr_sel)
        fig_mfe = build_mfe_mae_figure(mfe_df, timezone=cfg.timezone)
        st.plotly_chart(fig_mfe, use_container_width=True)

    with tabs[1]:
        st.subheader("Physics & Structure Engine")
        if micro is None:
            st.warning("micro_diagnostics.parquet not exported. Physics layer unavailable.")
        else:
            micro_sel = candidate_filter(micro, candidate_id, split_id, scenario_id)
            symbols = sorted(micro_sel["symbol"].astype(str).unique().tolist())
            if not symbols:
                st.info("No symbol data in selected slice")
            else:
                symbol = st.selectbox("symbol", options=symbols, key="physics_symbol")
                day_sessions = sorted(micro_sel[micro_sel["symbol"].astype(str) == symbol]["session_id"].astype(np.int64).unique().tolist())
                session_id = st.selectbox("session_id", options=day_sessions, key="physics_session")

                md = micro_sel[(micro_sel["symbol"].astype(str) == symbol) & (micro_sel["session_id"].astype(np.int64) == int(session_id))]
                pb = None
                if profile is not None:
                    pb_sel = candidate_filter(profile, candidate_id, split_id, scenario_id)
                    pb = pb_sel[(pb_sel["symbol"].astype(str) == symbol) & (pb_sel["session_id"].astype(np.int64) == int(session_id))]

                fig_micro = build_micro_matrix_figure(
                    micro_day_df=md,
                    profile_blocks_df=pb,
                    max_profile_blocks_render=int(cfg.max_profile_blocks_render),
                    timezone=cfg.timezone,
                )
                st.plotly_chart(fig_micro, use_container_width=True)

    with tabs[2]:
        st.subheader("Brain - Strategy Decision Logic")
        if micro is None:
            st.warning("micro_diagnostics.parquet not exported. Brain layer unavailable.")
        else:
            micro_sel = candidate_filter(micro, candidate_id, split_id, scenario_id)
            symbols = sorted(micro_sel["symbol"].astype(str).unique().tolist())
            if symbols:
                symbol = st.selectbox("symbol", options=symbols, key="brain_symbol")
                day_sessions = sorted(micro_sel[micro_sel["symbol"].astype(str) == symbol]["session_id"].astype(np.int64).unique().tolist())
                session_id = st.selectbox("session_id", options=day_sessions, key="brain_session")
                md = micro_sel[(micro_sel["symbol"].astype(str) == symbol) & (micro_sel["session_id"].astype(np.int64) == int(session_id))]
                fig_brain = build_brain_figure(
                    micro_day_df=md,
                    entry_threshold=0.55,
                    exit_threshold=0.25,
                    timezone=cfg.timezone,
                )
                st.plotly_chart(fig_brain, use_container_width=True)

    with tabs[3]:
        st.subheader("15:45 Zimtra Funnel Diagnostic")
        micro_sel = candidate_filter(micro, candidate_id, split_id, scenario_id) if micro is not None else None
        funnel_sel = candidate_filter(funnel, candidate_id, split_id, scenario_id) if funnel is not None else None

        session_opt = None
        if micro_sel is not None and len(micro_sel) > 0:
            sess = sorted(micro_sel["session_id"].astype(np.int64).unique().tolist())
            if sess:
                session_opt = st.selectbox("session_id", options=sess, key="funnel_session")

        ft = build_funnel_table(funnel_sel, micro_sel, selected_session=session_opt)
        if len(ft) == 0:
            st.info("No funnel data for current selection")
        else:
            cash_fallback = bool(int(np.max(ft["cash_fallback"].to_numpy(dtype=np.int8))) == 1)
            if cash_fallback:
                st.warning("Cash Fallback Triggered (no overnight winner)")
            else:
                winners = ft[ft["is_winner"].astype(np.int8) == 1]
                if len(winners) > 0:
                    st.success(f"Winner: {winners.iloc[0]['symbol']}")

            st.dataframe(ft, use_container_width=True)
            fig_funnel = build_funnel_bar_figure(ft, timezone=cfg.timezone)
            st.plotly_chart(fig_funnel, use_container_width=True)

    with tabs[4]:
        _render_data_qa(st, bundle)


if __name__ == "__main__":
    _main()
