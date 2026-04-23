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
    from charts_plotly import PRIMARY_COLOR, SECONDARY_COLOR, apply_chart_theme, strat_bar_chart_pure
    from mi_prep import format_currency
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
    from analytics.charts_plotly import PRIMARY_COLOR, SECONDARY_COLOR, apply_chart_theme, strat_bar_chart_pure
    from analytics.mi_prep import format_currency
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


# ---------------------------------------------------------------------------
# KPI tile helper — matches the .kpi-box CSS class defined in streamlit_app_erm.py
# ---------------------------------------------------------------------------

def _kpi_tile(label: str, value: str, subtitle: str = "") -> str:
    return (
        f'<div class="kpi-box">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-subtitle">{subtitle}</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _load_expected_funding_config(path: str | None) -> tuple[dict, str | None]:
    return load_expected_funding_config_yaml(path)


def _load_pipeline_csv(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    return normalize_pipeline_snapshot(raw, PipelinePrepConfig())


def _clean_blob_label(blob_name: str) -> str:
    """Strip blob storage path prefix — return filename only."""
    if not blob_name:
        return "—"
    return blob_name.rsplit("/", 1)[-1]


def _fmt_date(dt, fmt: str = "%d %b %Y") -> str:
    if dt is None:
        return "—"
    try:
        return dt.astimezone(timezone.utc).strftime(fmt)
    except Exception:
        return "—"


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _treemap_pair(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    section_title: str,
    title_amount: str,
    title_count: str,
    top_n: int = 15,
) -> None:
    """Side-by-side treemaps (amount + count) — same gradient/template as Stratifications."""
    st.markdown(f"#### {section_title}")
    if group_col not in df.columns:
        st.info(f"{section_title} data is not available in this snapshot.")
        return

    col1, col2 = st.columns(2)

    with col1:
        amt = (
            df.groupby(group_col, observed=True)[value_col]
            .sum()
            .reset_index(name="amount")
            .sort_values("amount", ascending=False)
            .head(top_n)
        )
        amt = amt[amt["amount"] > 0]
        if not amt.empty:
            fig = px.treemap(
                amt,
                path=[px.Constant("All"), group_col],
                values="amount",
                color="amount",
                color_continuous_scale=[(0, "#F0F2F6"), (1, PRIMARY_COLOR)],
            )
            fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), title_text=title_amount, title_x=0)
            fig.update_traces(
                texttemplate="<b>%{label}</b><br>£%{value:,.0f}<br>(%{percentEntry:.1%})",
            )
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No amount data available for {section_title.lower()}.")

    with col2:
        cnt = (
            df.groupby(group_col, observed=True)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(top_n)
        )
        cnt = cnt[cnt["count"] > 0]
        if not cnt.empty:
            fig = px.treemap(
                cnt,
                path=[px.Constant("All"), group_col],
                values="count",
                color="count",
                color_continuous_scale=[(0, "#F0F2F6"), (1, PRIMARY_COLOR)],
            )
            fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), title_text=title_count, title_x=0)
            fig.update_traces(
                texttemplate="<b>%{label}</b><br>%{value:,} apps<br>(%{percentEntry:.1%})",
            )
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No count data available for {section_title.lower()}.")


def _bar_strat_pair(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    section_title: str,
    title_amount: str,
    title_count: str,
) -> None:
    """Side-by-side bar charts (amount + count) for a bucketed dimension."""
    st.markdown(f"#### {section_title}")
    if group_col not in df.columns:
        st.info(f"{section_title} data is not available in this snapshot.")
        return

    col1, col2 = st.columns(2)
    with col1:
        fig, msg, _ = strat_bar_chart_pure(df, group_col, value_col=value_col, agg="sum", title=title_amount)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(msg or f"No amount data for {section_title.lower()}.")

    with col2:
        fig, msg, _ = strat_bar_chart_pure(df, group_col, value_col=value_col, agg="count", title=title_count)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(msg or f"No count data for {section_title.lower()}.")


# ---------------------------------------------------------------------------
# Snapshot resolution
# ---------------------------------------------------------------------------

def _resolve_pipeline_snapshot_inputs(show_controls: bool) -> dict:
    """Resolve the selected pipeline snapshot; return loaded state dict."""
    state: dict = {
        "pipeline_path": "",
        "selected_snapshot": None,
        "snapshot_modified": None,
        "pipeline_df": pd.DataFrame(),
    }

    if not is_azure_configured():
        if show_controls:
            st.error("Pipeline data source is not configured. Please contact your administrator.")
        return state

    try:
        latest_snapshot = get_latest_pipeline_snapshot()
        snapshots = list_pipeline_snapshots()
        snapshot_names = [s.blob_name for s in snapshots]
        if not snapshot_names:
            if show_controls:
                st.info("No pipeline snapshots are currently available.")
            return state

        prior_selection = st.session_state.get("pipeline_snapshot_blob")
        default_idx = resolve_pipeline_snapshot_selection(
            snapshot_names=snapshot_names,
            latest_blob_name=latest_snapshot.blob_name if latest_snapshot else None,
            prior_selection=prior_selection,
        )

        if show_controls:
            # Display clean filenames; store full blob name in session state
            display_labels = [_clean_blob_label(n) for n in snapshot_names]
            selected_label = st.selectbox(
                "Snapshot",
                options=display_labels,
                index=default_idx,
                key="pipeline_snapshot_display",
                help="Select a weekly pipeline snapshot to analyse.",
            )
            label_idx = display_labels.index(selected_label)
            selected_snapshot = snapshot_names[label_idx]
            st.session_state["pipeline_snapshot_blob"] = selected_snapshot
        else:
            selected_snapshot = (
                prior_selection
                if prior_selection in snapshot_names
                else snapshot_names[default_idx]
            )

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
    except Exception:
        logger.exception("Failed to resolve pipeline snapshots")
        if show_controls:
            st.error("Unable to load pipeline data. Please try again or contact support.")
        return state


# ---------------------------------------------------------------------------
# Public tab renderers
# ---------------------------------------------------------------------------

def render_pipeline_tab(funded_df: pd.DataFrame) -> None:
    st.markdown("### Pipeline Overview")
    st.caption(
        "Pipeline metrics reflect the most recent weekly snapshot. "
        "The funded portfolio tape remains the source of truth for all completed loans."
    )

    with st.expander("Data Source", expanded=True):
        snapshot_state = _resolve_pipeline_snapshot_inputs(show_controls=True)

    pipeline_df = snapshot_state["pipeline_df"]

    if pipeline_df is None or pipeline_df.empty:
        st.info("Pipeline analytics will appear once a valid snapshot is available.")
        return

    # ── KPI strip ──────────────────────────────────────────────────────────
    st.markdown("#### Pipeline Metrics")

    snapshot_date = "—"
    if "snapshot_date" in pipeline_df.columns:
        try:
            snapshot_date = str(pd.to_datetime(pipeline_df["snapshot_date"].iloc[0]).date())
        except Exception:
            pass

    last_updated = _fmt_date(snapshot_state["snapshot_modified"])

    live_series = pipeline_df.get("is_live_stage", pd.Series(dtype=bool))
    live_count = int(live_series.sum()) if hasattr(live_series, "sum") else len(pipeline_df)
    completed_count = (
        int((pipeline_df["stage"] == "COMPLETED").sum()) if "stage" in pipeline_df.columns else 0
    )
    pipeline_amount = float(pipeline_df["loan_amount"].sum()) if "loan_amount" in pipeline_df.columns else 0.0

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(_kpi_tile("Snapshot Date", snapshot_date, "Reporting period"), unsafe_allow_html=True)
    with col2:
        st.markdown(_kpi_tile("Last Updated", last_updated, "Data refreshed"), unsafe_allow_html=True)
    with col3:
        st.markdown(_kpi_tile("Active Applications", f"{live_count:,}", "Live pipeline"), unsafe_allow_html=True)
    with col4:
        st.markdown(
            _kpi_tile("Pipeline Amount", format_currency(pipeline_amount), "Indicative loan value"),
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            _kpi_tile("Completions", f"{completed_count:,}", "Completed applications"),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Funnel by stage ────────────────────────────────────────────────────
    st.markdown("### Pipeline Funnel")

    if "stage" in pipeline_df.columns:
        col_a, col_b = st.columns(2)

        with col_a:
            stage_counts = (
                pipeline_df.groupby("stage", observed=True)
                .size()
                .reset_index(name="Applications")
            )
            stage_counts.columns = ["Stage", "Applications"]
            fig_cnt = px.bar(stage_counts, x="Stage", y="Applications")
            fig_cnt.update_traces(
                marker_color=PRIMARY_COLOR,
                texttemplate="%{y:,}",
                textposition="outside",
            )
            st.plotly_chart(apply_chart_theme(fig_cnt, "Applications by Stage"), use_container_width=True)

        with col_b:
            if "loan_amount" in pipeline_df.columns:
                stage_amount = (
                    pipeline_df.groupby("stage", observed=True)["loan_amount"]
                    .sum(min_count=1)
                    .fillna(0)
                    .reset_index()
                )
                stage_amount.columns = ["Stage", "Amount"]
                fig_amt = px.bar(stage_amount, x="Stage", y="Amount")
                fig_amt.update_traces(
                    marker_color=PRIMARY_COLOR,
                    texttemplate="£%{y:,.0f}",
                    textposition="outside",
                )
                fig_amt.update_yaxes(title_text="Indicative Amount (£)")
                st.plotly_chart(
                    apply_chart_theme(fig_amt, "Indicative Amount by Stage"),
                    use_container_width=True,
                )
            else:
                st.info("Loan amount data is not available in this snapshot.")
    else:
        st.info("Stage information is not available in this snapshot.")

    # ── Completions reconciliation (collapsed by default) ──────────────────
    with st.expander("Completions Reconciliation", expanded=False):
        st.caption(
            "Reconciles completed pipeline applications against the funded portfolio. "
            "Unmatched items may be in-flight or pending confirmation."
        )
        recon_df = reconcile_completed_pipeline_to_funded(pipeline_df, funded_df)
        if recon_df.empty:
            st.info("No completions found in this snapshot for reconciliation.")
        else:
            summary = summarize_reconciliation(recon_df, funded_df)
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.markdown(
                    _kpi_tile("Completions", f"{summary.pipeline_rows:,}", "In this snapshot"),
                    unsafe_allow_html=True,
                )
            with r2:
                st.markdown(
                    _kpi_tile("Matched", f"{summary.matched_pipeline_rows:,}", "Confirmed in funded tape"),
                    unsafe_allow_html=True,
                )
            with r3:
                st.markdown(
                    _kpi_tile("Unmatched", f"{summary.unmatched_pipeline_rows:,}", "Pending confirmation"),
                    unsafe_allow_html=True,
                )
            with r4:
                st.markdown(
                    _kpi_tile("Funded Only", f"{summary.funded_rows_not_seen_in_pipeline:,}", "Not in pipeline"),
                    unsafe_allow_html=True,
                )

    # ── Pipeline composition stratifications ───────────────────────────────
    st.markdown("### Pipeline Composition")

    strat_df = add_pipeline_stratification_buckets(pipeline_df)

    # Treemaps for geographic and broker — matching Stratifications tab convention
    _treemap_pair(
        strat_df, "property_region", "loan_amount",
        "Geographic Distribution",
        "Pipeline Amount by Region (Top 15)",
        "Applications by Region (Top 15)",
    )

    _treemap_pair(
        strat_df, "broker", "loan_amount",
        "Broker Distribution",
        "Pipeline Amount by Broker (Top 15)",
        "Applications by Broker (Top 15)",
    )

    # Bar charts for bucketed dimensions — matching Stratifications tab convention
    _bar_strat_pair(
        strat_df, "ltv_bucket", "loan_amount",
        "LTV Distribution",
        "Pipeline Amount by LTV",
        "Applications by LTV",
    )

    _bar_strat_pair(
        strat_df, "age_bucket", "loan_amount",
        "Borrower Age Distribution",
        "Pipeline Amount by Borrower Age",
        "Applications by Borrower Age",
    )

    _bar_strat_pair(
        strat_df, "ticket_bucket", "loan_amount",
        "Ticket Size Distribution",
        "Pipeline Amount by Ticket Size",
        "Applications by Ticket Size",
    )


def render_forward_exposure_tab(funded_df: pd.DataFrame) -> None:
    st.markdown("### Forward Exposure")
    st.caption(
        "Forward Exposure is a modelled projection of expected pipeline completions. "
        "All figures are assumption-driven estimates; the funded portfolio tape remains the source of truth."
    )

    # Re-use the snapshot resolved in the Pipeline tab (no controls shown here)
    snapshot_state = _resolve_pipeline_snapshot_inputs(show_controls=False)

    pipeline_df = snapshot_state["pipeline_df"]
    if pipeline_df is None or pipeline_df.empty:
        st.info(
            "Forward Exposure analytics require a pipeline snapshot. "
            "Please ensure the Pipeline tab has loaded successfully."
        )
        return

    # Load expected-funding config silently — path not exposed to the user
    ef_config_raw: dict = {}
    config_ok = False
    try:
        ef_config_raw, _ = _load_expected_funding_config(DEFAULT_EXPECTED_CONFIG_RELATIVE)
        config_ok = True
    except Exception:
        ef_config_raw = {}

    recon_df = reconcile_completed_pipeline_to_funded(pipeline_df, funded_df)
    ef_config = load_expected_funding_config(ef_config_raw)
    expected_df = build_expected_funding_dataset(pipeline_df, recon_df, ef_config)

    snapshot_date = "—"
    if "snapshot_date" in pipeline_df.columns:
        try:
            snapshot_date = str(pd.to_datetime(pipeline_df["snapshot_date"].iloc[0]).date())
        except Exception:
            pass

    last_updated = _fmt_date(snapshot_state["snapshot_modified"])
    expected_amount = float(expected_df["expected_funded_amount"].sum()) if not expected_df.empty else 0.0
    high_conf_amount = (
        float(expected_df["high_confidence_expected_amount"].sum()) if not expected_df.empty else 0.0
    )
    model_status = "Active" if config_ok else "Defaults Applied"
    model_version = getattr(ef_config, "model_version", None) or "Standard"

    # ── KPI strip ──────────────────────────────────────────────────────────
    st.markdown("#### Forward Exposure Metrics")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(_kpi_tile("Snapshot Date", snapshot_date, "Pipeline basis"), unsafe_allow_html=True)
    with col2:
        st.markdown(_kpi_tile("Last Updated", last_updated, "Data refreshed"), unsafe_allow_html=True)
    with col3:
        st.markdown(
            _kpi_tile("Expected Funding", format_currency(expected_amount), "Modelled conversion"),
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            _kpi_tile("High Confidence", format_currency(high_conf_amount), "Strong likelihood"),
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(_kpi_tile("Model Status", model_status, model_version), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Expected funding by stage ──────────────────────────────────────────
    st.markdown("### Expected Funding")

    if expected_df.empty:
        st.info(
            "No eligible applications remain for expected funding modelling "
            "after stage and reconciliation filters."
        )
    else:
        exp_by_stage = (
            expected_df.groupby("stage", observed=True)["expected_funded_amount"]
            .sum(min_count=1)
            .fillna(0)
            .reset_index()
        )
        exp_by_stage.columns = ["Stage", "Expected Amount"]

        fig_exp = px.bar(exp_by_stage, x="Stage", y="Expected Amount")
        fig_exp.update_traces(
            marker_color=PRIMARY_COLOR,
            texttemplate="£%{y:,.0f}",
            textposition="outside",
        )
        fig_exp.update_yaxes(title_text="Expected Amount (£)")
        st.plotly_chart(apply_chart_theme(fig_exp, "Expected Funding by Stage"), use_container_width=True)

    # ── Regional forward concentration ─────────────────────────────────────
    st.markdown("### Regional Concentration")

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
        st.info("Regional concentration data is not available.")
        return

    # Identify region column (fall back to first column if standard name is absent)
    region_col = "property_region" if "property_region" in forward_region_df.columns else forward_region_df.columns[0]

    col_r1, col_r2 = st.columns(2)

    with col_r1:
        if "combined_forward_exposure" in forward_region_df.columns:
            top_fwd = (
                forward_region_df[[region_col, "combined_forward_exposure"]]
                .sort_values("combined_forward_exposure", ascending=False)
                .head(15)
            )
            top_fwd = top_fwd[top_fwd["combined_forward_exposure"] > 0]
            if not top_fwd.empty:
                fig_fwd = px.treemap(
                    top_fwd,
                    path=[px.Constant("All Regions"), region_col],
                    values="combined_forward_exposure",
                    color="combined_forward_exposure",
                    color_continuous_scale=[(0, "#F0F2F6"), (1, PRIMARY_COLOR)],
                )
                fig_fwd.update_layout(
                    margin=dict(l=0, r=0, t=40, b=0),
                    title_text="Forward Exposure by Region",
                    title_x=0,
                )
                fig_fwd.update_traces(
                    texttemplate="<b>%{label}</b><br>£%{value:,.0f}<br>(%{percentEntry:.1%})",
                )
                fig_fwd.update_coloraxes(showscale=False)
                st.plotly_chart(fig_fwd, use_container_width=True)
            else:
                st.info("No forward exposure data available by region.")
        else:
            st.info("Combined forward exposure data is not available.")

    with col_r2:
        funded_col = "funded_current_exposure"
        pipeline_col = "expected_pipeline_exposure"
        if funded_col in forward_region_df.columns and pipeline_col in forward_region_df.columns:
            display_df = forward_region_df[[region_col, funded_col, pipeline_col]].copy()
            display_df = display_df.rename(
                columns={
                    region_col: "Region",
                    funded_col: "Funded (Current)",
                    pipeline_col: "Expected Pipeline",
                }
            )
            fig_comp = px.bar(
                display_df,
                x="Region",
                y=["Funded (Current)", "Expected Pipeline"],
                barmode="stack",
                color_discrete_sequence=[PRIMARY_COLOR, SECONDARY_COLOR],
            )
            fig_comp.update_layout(legend_title_text="")
            st.plotly_chart(
                apply_chart_theme(fig_comp, "Funded vs Expected Pipeline by Region"),
                use_container_width=True,
            )
        else:
            st.info("Exposure breakdown by region is not available.")
