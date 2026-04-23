"""Streamlit renderers for Pipeline + Forward Exposure tabs."""

from __future__ import annotations

import logging
from datetime import timezone

import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from blob_storage import (
        download_pipeline_snapshot_to_tempfile,
        get_latest_pipeline_snapshot,
        is_azure_configured,
        list_pipeline_snapshots,
    )
    from charts_plotly import apply_chart_theme, strat_bar_chart_pure
    from pipeline_expected_funding import build_expected_funding_dataset, load_expected_funding_config
    from pipeline_forward_risk import ForwardRiskSchemaConfig, aggregate_forward_region_exposure
    from pipeline_prep import PipelinePrepConfig, normalize_pipeline_snapshot
    from pipeline_reconciliation import reconcile_completed_pipeline_to_funded, summarize_reconciliation
    from pipeline_snapshot_selector import resolve_pipeline_snapshot_selection
except ImportError:
    from analytics.blob_storage import (
        download_pipeline_snapshot_to_tempfile,
        get_latest_pipeline_snapshot,
        is_azure_configured,
        list_pipeline_snapshots,
    )
    from analytics.charts_plotly import apply_chart_theme, strat_bar_chart_pure
    from analytics.pipeline_expected_funding import build_expected_funding_dataset, load_expected_funding_config
    from analytics.pipeline_forward_risk import ForwardRiskSchemaConfig, aggregate_forward_region_exposure
    from analytics.pipeline_prep import PipelinePrepConfig, normalize_pipeline_snapshot
    from analytics.pipeline_reconciliation import reconcile_completed_pipeline_to_funded, summarize_reconciliation
    from analytics.pipeline_snapshot_selector import resolve_pipeline_snapshot_selection

logger = logging.getLogger(__name__)

try:
    from pipeline_tab_helpers import (
        DEFAULT_EXPECTED_CONFIG_RELATIVE,
        add_pipeline_stratification_buckets,
        load_expected_funding_config_yaml,
    )
except ImportError:
    from analytics.pipeline_tab_helpers import (
        DEFAULT_EXPECTED_CONFIG_RELATIVE,
        add_pipeline_stratification_buckets,
        load_expected_funding_config_yaml,
    )


def _currency(v: float) -> str:
    return f"£{float(v):,.0f}"


def _load_expected_funding_config(path: str | None) -> tuple[dict, str | None]:
    return load_expected_funding_config_yaml(path)


def _load_pipeline_csv(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    return normalize_pipeline_snapshot(raw, PipelinePrepConfig())


def _render_pipeline_stratification(df: pd.DataFrame, title: str, group_col: str):
    st.markdown(f"##### {title}")
    if group_col not in df.columns:
        st.info(f"{title} is not available in this snapshot.")
        return

    fig_bal, msg_bal, _ = strat_bar_chart_pure(df, group_col, value_col="loan_amount", agg="sum", title=f"{title} – Pipeline Amount")
    if fig_bal is not None:
        st.plotly_chart(fig_bal, use_container_width=True)
    else:
        st.info(msg_bal or f"No data for {title}.")

    fig_cnt, msg_cnt, _ = strat_bar_chart_pure(df, group_col, value_col="loan_amount", agg="count", title=f"{title} – Opportunity Count")
    if fig_cnt is not None:
        st.plotly_chart(fig_cnt, use_container_width=True)
    else:
        st.info(msg_cnt or f"No data for {title} count.")


def _resolve_pipeline_snapshot_inputs(show_controls: bool) -> dict:
    """Resolve selected pipeline snapshot and return loaded dataframe state."""
    state: dict = {
        "pipeline_path": "",
        "selected_snapshot": None,
        "snapshot_modified": None,
        "pipeline_df": pd.DataFrame(),
    }

    if not is_azure_configured():
        if show_controls:
            st.error("Azure Blob Storage is not configured. Configure DATA_STORAGE_CONNECTION or AZURE_STORAGE_ACCOUNT.")
        return state

    try:
        latest_snapshot = get_latest_pipeline_snapshot()
        snapshots = list_pipeline_snapshots()
        snapshot_names = [s.blob_name for s in snapshots]
        if not snapshot_names:
            if show_controls:
                st.info("No pipeline snapshot blobs found. Upload a CSV to inbound/pipeline/.")
            return state

        selected_snapshot = st.session_state.get("pipeline_snapshot_blob")
        default_idx = resolve_pipeline_snapshot_selection(
            snapshot_names=snapshot_names,
            latest_blob_name=latest_snapshot.blob_name if latest_snapshot else None,
            prior_selection=selected_snapshot,
        )

        if show_controls:
            selected_snapshot = st.selectbox(
                "Weekly pipeline snapshot blob",
                options=snapshot_names,
                index=default_idx,
                key="pipeline_snapshot_blob",
                help="Snapshots are sourced from Azure Blob Storage.",
            )
        else:
            selected_snapshot = selected_snapshot if selected_snapshot in snapshot_names else snapshot_names[default_idx]

        selected_obj = next((s for s in snapshots if s.blob_name == selected_snapshot), None)
        pipeline_path = download_pipeline_snapshot_to_tempfile(selected_snapshot)
        pipeline_df = _load_pipeline_csv(pipeline_path)

        state.update(
            {
                "pipeline_path": pipeline_path,
                "selected_snapshot": selected_snapshot,
                "snapshot_modified": selected_obj.last_modified if selected_obj else None,
                "pipeline_df": pipeline_df,
            }
        )
        return state
    except Exception as e:
        logger.exception("Failed to resolve pipeline snapshots")
        if show_controls:
            st.error(f"Could not load pipeline snapshots from Azure Blob Storage: {e}")
        return state


def _render_snapshot_status(selected_snapshot: str | None, snapshot_modified, funded_df: pd.DataFrame):
    st.markdown("#### Snapshot status")
    s1, s2, s3 = st.columns(3)
    s1.metric("Selected Snapshot Blob", selected_snapshot or "None")
    s2.metric(
        "Last Modified (UTC)",
        snapshot_modified.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S") if snapshot_modified else "n/a",
    )
    s3.metric("Reconciliation dataset available", "Yes" if funded_df is not None and not funded_df.empty else "No")


def render_pipeline_tab(funded_df: pd.DataFrame) -> None:
    st.markdown("### Pipeline")
    st.caption("Pipeline tab shows snapshot/funnel composition analytics. Funded tape remains truth.")

    with st.expander("Pipeline data source", expanded=True):
        snapshot_state = _resolve_pipeline_snapshot_inputs(show_controls=True)

    _render_snapshot_status(snapshot_state["selected_snapshot"], snapshot_state["snapshot_modified"], funded_df)

    pipeline_df = snapshot_state["pipeline_df"]
    if pipeline_df is None or pipeline_df.empty:
        st.info("Pipeline analytics will appear once a valid snapshot blob is available.")
        return

    # A. Funnel overview
    st.markdown("#### A. Funnel overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Snapshot Date", str(pd.to_datetime(pipeline_df["snapshot_date"].iloc[0]).date()))
    col2.metric("Pipeline Rows", f"{len(pipeline_df):,}")
    col3.metric("Live Rows", f"{int(pipeline_df.get('is_live_stage', pd.Series(dtype=bool)).sum()):,}")
    col4.metric("Completed Stage Rows", f"{int((pipeline_df['stage'] == 'COMPLETED').sum()):,}")

    stage_counts = pipeline_df.groupby("stage", observed=True).size().reset_index(name="count")
    fig = px.bar(stage_counts, x="stage", y="count", title="Pipeline stage counts")
    st.plotly_chart(apply_chart_theme(fig, "Pipeline stage counts"), use_container_width=True)

    stage_amount = (
        pipeline_df.groupby("stage", observed=True)["loan_amount"].sum(min_count=1).fillna(0).reset_index(name="pipeline_amount")
    )
    fig_amt = px.bar(stage_amount, x="stage", y="pipeline_amount", title="Pipeline amount by stage")
    st.plotly_chart(apply_chart_theme(fig_amt, "Pipeline amount by stage"), use_container_width=True)

    # B. Reconciliation control
    st.markdown("#### B. Completed-stage reconciliation control")
    recon_df = reconcile_completed_pipeline_to_funded(pipeline_df, funded_df)
    if recon_df.empty:
        st.info("No COMPLETED pipeline rows found for reconciliation.")
    else:
        summary = summarize_reconciliation(recon_df, funded_df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Completed (Pipeline)", f"{summary.pipeline_rows:,}")
        c2.metric("Matched", f"{summary.matched_pipeline_rows:,}")
        c3.metric("Unmatched", f"{summary.unmatched_pipeline_rows:,}")
        c4.metric("Funded not seen", f"{summary.funded_rows_not_seen_in_pipeline:,}")
        st.dataframe(recon_df, use_container_width=True, hide_index=True)

    # C. Pipeline MI stratifications
    st.markdown("#### C. Pipeline composition stratifications")
    strat_df = add_pipeline_stratification_buckets(pipeline_df)
    _render_pipeline_stratification(strat_df, "Geographic Distribution", "property_region")
    _render_pipeline_stratification(strat_df, "Broker", "broker")
    _render_pipeline_stratification(strat_df, "LTV", "ltv_bucket")
    _render_pipeline_stratification(strat_df, "Borrower Age", "age_bucket")
    _render_pipeline_stratification(strat_df, "Ticket Size", "ticket_bucket")


def render_forward_exposure_tab(funded_df: pd.DataFrame) -> None:
    st.markdown("### Forward Exposure")
    st.caption("Forward Exposure is modelled and assumption-driven; funded tape remains truth.")

    snapshot_state = _resolve_pipeline_snapshot_inputs(show_controls=False)
    _render_snapshot_status(snapshot_state["selected_snapshot"], snapshot_state["snapshot_modified"], funded_df)

    pipeline_df = snapshot_state["pipeline_df"]
    if pipeline_df is None or pipeline_df.empty:
        st.info("Select/ingest a pipeline snapshot in the Pipeline tab to enable projection analytics.")
        return

    config_input = st.text_input(
        "Expected-funding config YAML",
        value=st.session_state.get("pipeline_expected_funding_config_path", DEFAULT_EXPECTED_CONFIG_RELATIVE),
        key="pipeline_expected_funding_config_path",
        help="Assumption model for expected funding and forward exposure.",
    )

    ef_config_raw: dict
    config_loaded_from = None
    try:
        ef_config_raw, config_loaded_from = _load_expected_funding_config(config_input)
        st.success(f"Loaded expected-funding config: {config_loaded_from}")
    except Exception as e:
        st.warning(f"Expected funding config not loaded ({e}). Using defaults.")
        ef_config_raw = {}

    recon_df = reconcile_completed_pipeline_to_funded(pipeline_df, funded_df)
    ef_config = load_expected_funding_config(ef_config_raw)
    expected_df = build_expected_funding_dataset(pipeline_df, recon_df, ef_config)

    st.markdown("#### A. Expected funding (modelled)")
    if expected_df.empty:
        st.info("No eligible rows for expected funding after stage/reconciliation filters.")
    else:
        e1, e2, e3 = st.columns(3)
        e1.metric("Model Version", ef_config.model_version)
        e2.metric("Expected Funded Amount", _currency(expected_df["expected_funded_amount"].sum()))
        e3.metric("High-Confidence Expected", _currency(expected_df["high_confidence_expected_amount"].sum()))

        exp_by_stage = expected_df.groupby("stage", observed=True)["expected_funded_amount"].sum(min_count=1).fillna(0).reset_index()
        fig3 = px.bar(exp_by_stage, x="stage", y="expected_funded_amount", title="Expected funded amount by stage")
        st.plotly_chart(apply_chart_theme(fig3, "Expected funded amount by stage"), use_container_width=True)

    st.markdown("#### B. Forward region concentration")
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
    st.plotly_chart(apply_chart_theme(fig5, "Forward region exposure"), use_container_width=True)
