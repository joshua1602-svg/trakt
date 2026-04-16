"""Streamlit renderer for the Pipeline tab.

UI-only module to keep streamlit_app_erm.py from growing into a monolith.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.express as px

from pipeline_prep import normalize_pipeline_snapshot, PipelinePrepConfig
from pipeline_reconciliation import reconcile_completed_pipeline_to_funded, summarize_reconciliation


def _load_pipeline_csv(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    return normalize_pipeline_snapshot(raw, PipelinePrepConfig())


def render_pipeline_tab(funded_df: pd.DataFrame) -> None:
    st.markdown("### Pipeline Flow")
    st.caption(
        "Snapshot-based view: stage timing is inferred from weekly snapshots unless an event log is supplied. "
        "Pipeline COMPLETED is not funded truth; funded tape remains the source of truth."
    )

    with st.expander("Pipeline data source", expanded=True):
        pipeline_path = st.text_input(
            "Path to weekly pipeline CSV",
            value=st.session_state.get("pipeline_file_path", ""),
            key="pipeline_file_path",
            help="Local path to the weekly pipeline snapshot CSV.",
        )

    if not pipeline_path:
        st.info("Add a pipeline CSV path to enable flow and reconciliation analytics.")
        return

    try:
        pipeline_df = _load_pipeline_csv(pipeline_path)
    except Exception as e:
        st.error(f"Could not load pipeline file: {e}")
        return

    if pipeline_df.empty:
        st.warning("Pipeline file loaded but no rows were available after normalization.")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Snapshot Date", str(pd.to_datetime(pipeline_df["snapshot_date"].iloc[0]).date()))
    with col2:
        st.metric("Pipeline Rows", f"{len(pipeline_df):,}")
    with col3:
        live_rows = int(pipeline_df["is_live_stage"].sum()) if "is_live_stage" in pipeline_df.columns else 0
        st.metric("Live Rows", f"{live_rows:,}")
    with col4:
        completed_rows = int((pipeline_df["pipeline_stage"] == "COMPLETED").sum())
        st.metric("Completed Stage Rows", f"{completed_rows:,}")

    stage_counts = (
        pipeline_df.groupby("pipeline_stage", observed=True)
        .size()
        .reset_index(name="count")
    )

    if not stage_counts.empty:
        fig = px.bar(stage_counts, x="pipeline_stage", y="count", title="Pipeline Stage Count")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Completed vs funded reconciliation")
    recon_df = reconcile_completed_pipeline_to_funded(pipeline_df, funded_df)

    if recon_df.empty:
        st.info("No COMPLETED pipeline rows found for reconciliation.")
        return

    summary = summarize_reconciliation(recon_df, funded_df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Completed (Pipeline)", f"{summary.completed_pipeline_rows:,}")
    c2.metric("Matched", f"{summary.matched_pipeline_rows:,}")
    c3.metric("Unmatched", f"{summary.unmatched_pipeline_rows:,}")
    c4.metric("Funded not in Completed set", f"{summary.funded_rows_not_seen_in_pipeline_completed:,}")

    match_breakdown = recon_df.groupby("match_status", observed=True).size().reset_index(name="count")
    if not match_breakdown.empty:
        fig2 = px.pie(match_breakdown, names="match_status", values="count", title="Completed reconciliation status")
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(recon_df, use_container_width=True, hide_index=True)

    st.download_button(
        "Download reconciliation CSV",
        data=recon_df.to_csv(index=False).encode("utf-8"),
        file_name="pipeline_completed_reconciliation.csv",
        mime="text/csv",
    )
