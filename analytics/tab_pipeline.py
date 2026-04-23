"""Streamlit renderer for the Pipeline tab.

UI-only module to keep streamlit_app_erm.py from growing into a monolith.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

from pipeline_expected_funding import build_expected_funding_dataset, load_expected_funding_config
from pipeline_forward_risk import aggregate_forward_region_exposure, ForwardRiskSchemaConfig
from pipeline_prep import normalize_pipeline_snapshot, PipelinePrepConfig
from pipeline_reconciliation import reconcile_completed_pipeline_to_funded, summarize_reconciliation


def _load_pipeline_csv(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    return normalize_pipeline_snapshot(raw, PipelinePrepConfig())


def _load_expected_funding_config(path: str) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Expected funding config file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _currency(v: float) -> str:
    return f"£{float(v):,.0f}"


def render_pipeline_tab(funded_df: pd.DataFrame) -> None:
    st.markdown("### Pipeline Flow")
    st.caption(
        "Pipeline snapshot data is not funded truth. Reconciliation ties COMPLETED pipeline to funded records. "
        "Expected funding is an assumption-driven planning layer."
    )

    with st.expander("Pipeline data source", expanded=True):
        pipeline_path = st.text_input(
            "Path to weekly pipeline CSV",
            value=st.session_state.get("pipeline_file_path", ""),
            key="pipeline_file_path",
            help="Local path to the weekly pipeline snapshot CSV.",
        )
        config_path = st.text_input(
            "Path to expected-funding config YAML (optional)",
            value=st.session_state.get("pipeline_expected_funding_config_path", "config/client/pipeline_expected_funding.yaml"),
            key="pipeline_expected_funding_config_path",
            help="Model assumptions for expected funding / forward exposure.",
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

    # A. Pipeline overview
    st.markdown("#### A. Pipeline overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Snapshot Date", str(pd.to_datetime(pipeline_df["snapshot_date"].iloc[0]).date()))
    with col2:
        st.metric("Pipeline Rows", f"{len(pipeline_df):,}")
    with col3:
        live_rows = int(pipeline_df["is_live_stage"].sum()) if "is_live_stage" in pipeline_df.columns else 0
        st.metric("Live Rows", f"{live_rows:,}")
    with col4:
        completed_rows = int((pipeline_df["stage"] == "COMPLETED").sum())
        st.metric("Completed Stage Rows", f"{completed_rows:,}")

    stage_counts = pipeline_df.groupby("stage", observed=True).size().reset_index(name="count")
    if not stage_counts.empty:
        fig = px.bar(stage_counts, x="stage", y="count", title="Pipeline stage counts")
        st.plotly_chart(fig, use_container_width=True)

    stage_amount = (
        pipeline_df.groupby("stage", observed=True)["loan_amount"]
        .sum(min_count=1)
        .fillna(0)
        .reset_index(name="pipeline_amount")
    )
    if not stage_amount.empty:
        fig_amt = px.bar(stage_amount, x="stage", y="pipeline_amount", title="Pipeline amount by stage")
        st.plotly_chart(fig_amt, use_container_width=True)

    # B. Completed vs funded reconciliation
    st.markdown("#### B. Completed vs funded reconciliation")
    recon_df = reconcile_completed_pipeline_to_funded(pipeline_df, funded_df)

    if recon_df.empty:
        st.info("No COMPLETED pipeline rows found for reconciliation.")
        return

    summary = summarize_reconciliation(recon_df, funded_df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Completed (Pipeline)", f"{summary.pipeline_rows:,}")
    c2.metric("Matched", f"{summary.matched_pipeline_rows:,}")
    c3.metric("Unmatched", f"{summary.unmatched_pipeline_rows:,}")
    c4.metric("Funded not seen", f"{summary.funded_rows_not_seen_in_pipeline:,}")

    match_breakdown = (
        recon_df.groupby("reconciliation_match_status", observed=True).size().reset_index(name="count")
    )
    if not match_breakdown.empty:
        fig2 = px.pie(
            match_breakdown,
            names="reconciliation_match_status",
            values="count",
            title="Completed reconciliation status",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(recon_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download reconciliation CSV",
        data=recon_df.to_csv(index=False).encode("utf-8"),
        file_name="pipeline_completed_reconciliation.csv",
        mime="text/csv",
    )

    # C. Expected funding
    st.markdown("#### C. Expected funding (modelled)")
    try:
        ef_config_raw = _load_expected_funding_config(config_path)
    except Exception as e:
        st.warning(f"Expected funding config not loaded ({e}); using defaults.")
        ef_config_raw = {}

    ef_config = load_expected_funding_config(ef_config_raw)
    expected_df = build_expected_funding_dataset(pipeline_df, recon_df, ef_config)

    if expected_df.empty:
        st.info("No eligible rows for expected funding after stage/reconciliation filters.")
    else:
        e1, e2, e3 = st.columns(3)
        e1.metric("Model Version", ef_config.model_version)
        e2.metric("Expected Funded Amount", _currency(expected_df["expected_funded_amount"].sum()))
        e3.metric("High-Confidence Expected", _currency(expected_df["high_confidence_expected_amount"].sum()))

        exp_by_stage = (
            expected_df.groupby("stage", observed=True)["expected_funded_amount"]
            .sum(min_count=1)
            .fillna(0)
            .reset_index()
        )
        fig3 = px.bar(exp_by_stage, x="stage", y="expected_funded_amount", title="Expected funded amount by stage")
        st.plotly_chart(fig3, use_container_width=True)

        bucketed = expected_df.copy()
        bucketed["expected_funded_month"] = pd.to_datetime(bucketed["expected_funded_date"]).dt.to_period("M").dt.to_timestamp()
        exp_by_month = (
            bucketed.groupby("expected_funded_month", observed=True)["expected_funded_amount"]
            .sum(min_count=1)
            .fillna(0)
            .reset_index()
            .sort_values("expected_funded_month")
        )
        fig4 = px.bar(
            exp_by_month,
            x="expected_funded_month",
            y="expected_funded_amount",
            title="Expected funding by expected funded date bucket (month)",
        )
        st.plotly_chart(fig4, use_container_width=True)

        st.dataframe(expected_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download expected funding CSV",
            data=expected_df.to_csv(index=False).encode("utf-8"),
            file_name="pipeline_expected_funding.csv",
            mime="text/csv",
        )

    # D. Forward region concentration
    st.markdown("#### D. Forward region concentration")
    st.caption("Forward concentration combines funded current exposure with modelled expected pipeline exposure.")

    forward_schema_raw = ef_config_raw.get("forward_risk", {}) if isinstance(ef_config_raw, dict) else {}
    forward_schema = ForwardRiskSchemaConfig(
        funded_region_column=forward_schema_raw.get("funded_region_column"),
        funded_exposure_column=forward_schema_raw.get("funded_exposure_column"),
    )

    forward_region_df = aggregate_forward_region_exposure(
        funded_df=funded_df,
        expected_df=expected_df,
        region_limit_df=None,
        schema_config=forward_schema,
    )

    if forward_region_df.empty:
        st.info("No forward region concentration rows available.")
        return

    st.dataframe(forward_region_df, use_container_width=True, hide_index=True)
    fig5 = px.bar(
        forward_region_df,
        x="property_region",
        y=["funded_current_exposure", "expected_pipeline_exposure", "combined_forward_exposure"],
        title="Forward region exposure",
        barmode="group",
    )
    st.plotly_chart(fig5, use_container_width=True)

    fig6 = px.pie(
        forward_region_df,
        names="property_region",
        values="combined_forward_exposure",
        title="Combined forward exposure % by region",
    )
    st.plotly_chart(fig6, use_container_width=True)
