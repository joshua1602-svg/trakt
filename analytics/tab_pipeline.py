"""Streamlit renderers for Pipeline + Forward Exposure tabs."""

from __future__ import annotations

import logging
from datetime import timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from blob_storage import (
        download_pipeline_snapshot_to_tempfile,
        get_latest_pipeline_snapshot,
        is_azure_configured,
        list_pipeline_snapshots,
    )
    from charts_plotly import PRIMARY_COLOR, SECONDARY_COLOR, apply_chart_theme, strat_bar_chart_pure
    from mi_prep import format_currency, weighted_average
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
    from analytics.mi_prep import format_currency, weighted_average
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
        prepare_weekly_trend_dataset,
    )
except ImportError:
    from analytics.pipeline_tab_helpers import (
        DEFAULT_EXPECTED_CONFIG_RELATIVE,
        add_pipeline_stratification_buckets,
        load_expected_funding_config_yaml,
        prepare_weekly_trend_dataset,
    )

FLOW_STAGES = ["KFI", "APPLICATION", "OFFER", "COMPLETED"]
STAGE_LABELS = {
    "KFI": "KFIs",
    "APPLICATION": "Applications",
    "OFFER": "Offers",
    "COMPLETED": "Completions",
}
STAGE_COLORS = [PRIMARY_COLOR, SECONDARY_COLOR, "#7EBAB5", "#6B6B6B"]


def _kpi_tile(label: str, value: str, subtitle: str = "") -> str:
    return (
        f'<div class="kpi-box">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-subtitle">{subtitle}</div>'
        f"</div>"
    )


def _load_expected_funding_config(path: str | None) -> tuple[dict, str | None]:
    return load_expected_funding_config_yaml(path)


def _load_pipeline_csv(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    return normalize_pipeline_snapshot(raw, PipelinePrepConfig())


def _clean_blob_label(blob_name: str) -> str:
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


def _pick_first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _fmt_delta(curr: float, prev: float, is_count: bool = False) -> str:
    delta = curr - prev
    if is_count:
        return f"{delta:+,.0f}"
    return f"{delta:+,.0f}"


@st.cache_data(show_spinner=False)
def _build_weekly_snapshot_history(selected_snapshot: str, max_weeks: int = 16) -> pd.DataFrame:
    snapshots = list_pipeline_snapshots()
    names = [s.blob_name for s in snapshots]
    if selected_snapshot not in names:
        return pd.DataFrame()

    selected_idx = names.index(selected_snapshot)
    chosen = snapshots[selected_idx:selected_idx + max_weeks]
    rows: list[dict] = []

    for snap in reversed(chosen):
        try:
            path = download_pipeline_snapshot_to_tempfile(snap.blob_name)
            snap_df = _load_pipeline_csv(path)
            if snap_df.empty:
                continue
            snap_date = pd.to_datetime(snap_df.get("snapshot_date"), errors="coerce").max()
            for stage in FLOW_STAGES:
                part = snap_df[snap_df["stage"].eq(stage)] if "stage" in snap_df.columns else pd.DataFrame()
                rows.append(
                    {
                        "snapshot_blob": snap.blob_name,
                        "snapshot_date": snap_date,
                        "stage": stage,
                        "count": int(len(part)),
                        "amount": float(pd.to_numeric(part.get("loan_amount", 0), errors="coerce").fillna(0).sum()),
                    }
                )
        except Exception:
            logger.exception("Failed to read pipeline snapshot history blob=%s", snap.blob_name)
            continue

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["snapshot_date"] = pd.to_datetime(out["snapshot_date"], errors="coerce")
    out = out.sort_values(["snapshot_date", "stage"])
    return out


def _render_weekly_flow_summary(history_df: pd.DataFrame) -> None:
    st.markdown("### Weekly Flow Summary")
    st.caption("Weekly origination MI across key funnel stages, including week-on-week change and rolling averages.")

    if history_df.empty:
        st.info("Weekly flow history is not yet available. Load additional snapshots to unlock trend reporting.")
        return

    cols = st.columns(4)
    for idx, stage in enumerate(FLOW_STAGES):
        stage_hist = history_df[history_df["stage"].eq(stage)].sort_values("snapshot_date")
        if stage_hist.empty:
            cols[idx].info(f"No {STAGE_LABELS[stage].lower()} data available.")
            continue

        current = stage_hist.iloc[-1]
        previous = stage_hist.iloc[-2] if len(stage_hist) >= 2 else None

        amt = float(current["amount"])
        cnt = float(current["count"])
        prev_amt = float(previous["amount"]) if previous is not None else 0.0
        prev_cnt = float(previous["count"]) if previous is not None else 0.0

        rolling = stage_hist.tail(5)
        roll_amt = float(rolling["amount"].mean()) if not rolling.empty else 0.0
        roll_cnt = float(rolling["count"].mean()) if not rolling.empty else 0.0

        subtitle = (
            f"Count {cnt:,.0f} | WoW {_fmt_delta(cnt, prev_cnt, is_count=True)} "
            f"| 5W avg {roll_cnt:,.1f}"
        )
        cols[idx].markdown(
            _kpi_tile(
                f"{STAGE_LABELS[stage]} This Week",
                format_currency(amt),
                subtitle,
            ),
            unsafe_allow_html=True,
        )

        cols[idx].caption(
            f"Last week: {format_currency(prev_amt)} | Δ amount {_fmt_delta(amt, prev_amt)} | "
            f"5W avg amount {format_currency(roll_amt)}"
        )


def _render_weekly_trend_charts(history_df: pd.DataFrame) -> None:
    st.markdown("### Weekly Origination Trends")
    if history_df.empty:
        st.info("Trend charts will populate as additional weekly snapshots become available.")
        return

    trend = prepare_weekly_trend_dataset(history_df)
    stage_options = [STAGE_LABELS[s] for s in FLOW_STAGES if STAGE_LABELS[s] in set(trend["stage_label"].astype(str))]
    selected_stage_labels = st.multiselect(
        "Filter stages",
        options=stage_options,
        default=stage_options,
        key="weekly_stage_filter",
    )
    trend = trend[trend["stage_label"].isin(selected_stage_labels)].copy()
    if trend.empty:
        st.info("No weekly data available for the selected stages.")
        return

    col1, col2 = st.columns(2)
    with col1:
        fig_cnt = px.bar(
            trend,
            x="week",
            y="count",
            color="stage_label",
            barmode="stack",
            category_orders={"week": list(trend["week"].cat.categories), "stage_label": stage_options},
            color_discrete_sequence=STAGE_COLORS,
        )
        fig_cnt.update_yaxes(title_text="Applications")
        fig_cnt.update_xaxes(title_text="Week")
        st.plotly_chart(apply_chart_theme(fig_cnt, "Weekly Volumes by Stage"), use_container_width=True)

    with col2:
        fig_amt = px.bar(
            trend,
            x="week",
            y="amount",
            color="stage_label",
            barmode="stack",
            category_orders={"week": list(trend["week"].cat.categories), "stage_label": stage_options},
            color_discrete_sequence=STAGE_COLORS,
        )
        fig_amt.update_yaxes(title_text="Amount (£)")
        fig_amt.update_xaxes(title_text="Week")
        st.plotly_chart(apply_chart_theme(fig_amt, "Weekly Amounts by Stage"), use_container_width=True)


def _render_ltv_age_bubble(df: pd.DataFrame, exposure_col: str, color_col: str | None, title: str, key_prefix: str) -> None:
    bubble_df = df.dropna(subset=["youngest_borrower_age", "current_loan_to_value", exposure_col]).copy()
    bubble_df = bubble_df[pd.to_numeric(bubble_df[exposure_col], errors="coerce").fillna(0) > 0]
    if bubble_df.empty:
        st.info("Insufficient data for bubble chart.")
        return

    if len(bubble_df) > 2000:
        bubble_df = bubble_df.sample(2000, random_state=42)
        st.caption("Showing 2,000 random loans for performance")

    bubble_df["current_loan_to_value_pct"] = pd.to_numeric(bubble_df["current_loan_to_value"], errors="coerce")
    fig = px.scatter(
        bubble_df,
        x="youngest_borrower_age",
        y="current_loan_to_value_pct",
        size=exposure_col,
        color=color_col if color_col in bubble_df.columns else None,
        hover_data={
            "youngest_borrower_age": ":.0f",
            "current_loan_to_value_pct": ":.1f",
            exposure_col: ":,.0f",
        },
        color_discrete_sequence=[PRIMARY_COLOR, SECONDARY_COLOR, "#7EBAB5", "#6B6B6B"],
    )
    fig = apply_chart_theme(fig, title)
    fig.update_layout(legend_title_text="")
    fig.update_traces(marker=dict(opacity=0.7, line=dict(width=0.5, color="white")))
    fig.update_xaxes(title_text="Youngest Borrower Age (years)")
    fig.update_yaxes(title_text="Current LTV (%)")
    st.plotly_chart(fig, use_container_width=True)


def _render_portfolio_concentration_matrix(df: pd.DataFrame, exposure_col: str, key_prefix: str) -> None:
    st.markdown("### Portfolio Concentration Matrix")
    st.caption("Deep dive into the intersection of any two risk dimensions.")
    if key_prefix == "pipeline":
        st.caption("Pipeline exposure, not funded exposure.")

    col_x, col_y, col_m = st.columns(3)
    dim_map = {
        "Region": "geographic_region",
        "Product Type": "erm_product_type",
        "LTV Bucket": "ltv_bucket",
        "Age Bucket": "age_bucket",
        "Ticket Size": "ticket_bucket",
        "Vintage": "origination_year",
        "Broker Channel": "broker_channel",
    }
    avail = [k for k, v in dim_map.items() if v in df.columns]
    if not avail:
        st.info("No concentration dimensions are available.")
        return

    with col_x:
        row_label = st.selectbox("Rows (Y-Axis)", avail, index=0, key=f"{key_prefix}_rows")
    with col_y:
        c_idx = 2 if len(avail) > 2 else 0
        col_label = st.selectbox("Columns (X-Axis)", avail, index=c_idx, key=f"{key_prefix}_cols")
    with col_m:
        metric_choice = st.radio("Metric", options=["Balance", "Count"], horizontal=True, key=f"{key_prefix}_metric")

    row_col = dim_map[row_label]
    col_col = dim_map[col_label]
    if metric_choice == "Balance":
        mat = df.groupby([row_col, col_col], observed=True)[exposure_col].sum().unstack(fill_value=0)
        txt = mat.applymap(lambda x: f"£{x/1_000_000:.1f}M" if x > 0 else "")
        hovertemplate = "%{y}, %{x}<br><b>£%{z:,.0f}</b><extra></extra>"
    else:
        mat = df.groupby([row_col, col_col], observed=True).size().unstack(fill_value=0)
        txt = mat.applymap(lambda x: f"{int(x):,}" if x > 0 else "")
        hovertemplate = "%{y}, %{x}<br><b>%{z:,} Loans</b><extra></extra>"

    fig_mx = go.Figure(data=go.Heatmap(
        z=mat.values,
        x=[str(c).replace("_", " ").title() for c in mat.columns],
        y=[str(i).replace("_", " ").title() for i in mat.index],
        text=txt.values,
        texttemplate="%{text}",
        textfont={"size": 11},
        colorscale=[[0, "#F0F2F6"], [1, PRIMARY_COLOR]],
        showscale=True,
        hovertemplate=hovertemplate,
        xgap=2,
        ygap=2,
    ))
    fig_mx.update_layout(
        title=dict(text=f"<b>{metric_choice}</b>: {row_label} vs {col_label}"),
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=500,
    )
    st.plotly_chart(fig_mx, use_container_width=True)


def _treemap_pair(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    section_title: str,
    title_amount: str,
    title_count: str,
    top_n: int = 15,
) -> None:
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
            fig.update_traces(texttemplate="<b>%{label}</b><br>£%{value:,.0f}<br>(%{percentEntry:.1%})")
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
            fig.update_traces(texttemplate="<b>%{label}</b><br>%{value:,} apps<br>(%{percentEntry:.1%})")
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


def _resolve_pipeline_snapshot_inputs(show_controls: bool) -> dict:
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
            display_labels = [_clean_blob_label(n) for n in snapshot_names]
            selected_label = st.selectbox(
                "Snapshot",
                options=display_labels,
                index=default_idx,
                key="pipeline_snapshot_display",
                help="Select the reporting snapshot for weekly MI.",
            )
            label_idx = display_labels.index(selected_label)
            selected_snapshot = snapshot_names[label_idx]
            st.session_state["pipeline_snapshot_blob"] = selected_snapshot
        else:
            selected_snapshot = prior_selection if prior_selection in snapshot_names else snapshot_names[default_idx]

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


def _aggregate_dimension(df: pd.DataFrame, dim_col: str, value_col: str) -> pd.DataFrame:
    if dim_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame(columns=[dim_col, "amount"])
    out = (
        df.groupby(dim_col, observed=True)[value_col]
        .sum(min_count=1)
        .fillna(0)
        .reset_index(name="amount")
    )
    return out[out["amount"] > 0]


def render_pipeline_tab(funded_df: pd.DataFrame) -> None:
    st.markdown("### Pipeline")
    st.caption("Executive weekly origination MI covering flow, funnel snapshot, and pipeline composition.")

    with st.expander("Snapshot & Source", expanded=True):
        snapshot_state = _resolve_pipeline_snapshot_inputs(show_controls=True)

    pipeline_df = snapshot_state["pipeline_df"]
    if pipeline_df is None or pipeline_df.empty:
        st.info("Pipeline MI will appear once a valid weekly snapshot is available.")
        return

    history_df = _build_weekly_snapshot_history(snapshot_state["selected_snapshot"], max_weeks=16)

    snapshot_date = "—"
    if "snapshot_date" in pipeline_df.columns:
        snapshot_dt = pd.to_datetime(pipeline_df["snapshot_date"], errors="coerce").max()
        if pd.notna(snapshot_dt):
            snapshot_date = snapshot_dt.strftime("%d %b %Y")

    overview_cols = st.columns(3)
    overview_cols[0].markdown(_kpi_tile("Snapshot", snapshot_date, "Reporting week"), unsafe_allow_html=True)
    overview_cols[1].markdown(
        _kpi_tile("Last Updated", _fmt_date(snapshot_state["snapshot_modified"]), "Source status: available"),
        unsafe_allow_html=True,
    )
    history_weeks = int(history_df["snapshot_blob"].nunique()) if not history_df.empty else 1
    overview_cols[2].markdown(_kpi_tile("Data Coverage", f"{history_weeks} weeks", "Weekly history loaded"), unsafe_allow_html=True)

    _render_weekly_flow_summary(history_df)
    _render_weekly_trend_charts(history_df)

    st.markdown("### Pipeline Stage MI")
    st.caption("Outstanding stock metrics are point-in-time; weekly flow metrics are based on current vs prior snapshot.")
    live_series = pipeline_df.get("is_live_stage", pd.Series(dtype=bool))
    live_df = pipeline_df[live_series] if len(live_series) == len(pipeline_df) else pipeline_df.copy()
    current_week = history_df["snapshot_date"].max() if not history_df.empty else pd.NaT
    prev_week = history_df[history_df["snapshot_date"] < current_week]["snapshot_date"].max() if not history_df.empty else pd.NaT

    def _stage_tile(stage: str, label: str, flow: bool = False):
        if flow:
            curr = history_df[(history_df["stage"] == stage) & (history_df["snapshot_date"] == current_week)] if pd.notna(current_week) else pd.DataFrame()
            prev = history_df[(history_df["stage"] == stage) & (history_df["snapshot_date"] == prev_week)] if pd.notna(prev_week) else pd.DataFrame()
            amount = float(curr["amount"].sum()) if not curr.empty else 0.0
            count = int(curr["count"].sum()) if not curr.empty else 0
            prev_amt = float(prev["amount"].sum()) if not prev.empty else 0.0
            basis_label = "New this week"
        else:
            subset = live_df[live_df.get("stage", pd.Series(dtype="string")).eq(stage)]
            amount = float(pd.to_numeric(subset.get("loan_amount", 0), errors="coerce").fillna(0).sum())
            count = int(len(subset))
            prev = history_df[(history_df["stage"] == stage) & (history_df["snapshot_date"] == prev_week)] if pd.notna(prev_week) else pd.DataFrame()
            prev_amt = float(prev["amount"].sum()) if not prev.empty else 0.0
            basis_label = "Outstanding"
        delta_txt = f"Δ vs prior week: {format_currency(amount - prev_amt)}" if pd.notna(prev_week) else "Δ vs prior week: N/A"
        return _kpi_tile(label, format_currency(amount), f"{basis_label} | {count:,} loans | {delta_txt}")

    st.markdown("#### Outstanding Stock")
    o1, o2, o3 = st.columns(3)
    o1.markdown(_stage_tile("KFI", "Outstanding KFIs"), unsafe_allow_html=True)
    o2.markdown(_stage_tile("APPLICATION", "Outstanding Applications"), unsafe_allow_html=True)
    o3.markdown(_stage_tile("OFFER", "Outstanding Offers"), unsafe_allow_html=True)

    st.markdown("#### Weekly Flow")
    f1, f2, f3, f4 = st.columns(4)
    f1.markdown(_stage_tile("KFI", "Weekly new KFIs", flow=True), unsafe_allow_html=True)
    f2.markdown(_stage_tile("APPLICATION", "Weekly new Applications", flow=True), unsafe_allow_html=True)
    f3.markdown(_stage_tile("OFFER", "Weekly new Offers", flow=True), unsafe_allow_html=True)
    f4.markdown(_stage_tile("COMPLETED", "Weekly completions", flow=True), unsafe_allow_html=True)

    if "stage" in pipeline_df.columns:
        funnel_counts = (
            pipeline_df[pipeline_df["stage"].isin(FLOW_STAGES)]
            .groupby("stage", observed=True)
            .size()
            .reindex(FLOW_STAGES, fill_value=0)
            .reset_index(name="count")
        )
        fig_funnel = go.Figure(go.Funnel(y=funnel_counts["stage"], x=funnel_counts["count"], textinfo="value+percent previous"))
        st.plotly_chart(apply_chart_theme(fig_funnel, "Pipeline Conversion Funnel (Aggregate Snapshot)"), use_container_width=True)
        kfi = float(funnel_counts.loc[funnel_counts["stage"] == "KFI", "count"].sum())
        app = float(funnel_counts.loc[funnel_counts["stage"] == "APPLICATION", "count"].sum())
        offer = float(funnel_counts.loc[funnel_counts["stage"] == "OFFER", "count"].sum())
        comp = float(funnel_counts.loc[funnel_counts["stage"] == "COMPLETED", "count"].sum())
        kfi_app = (app / kfi * 100) if kfi > 0 else 0.0
        app_offer = (offer / app * 100) if app > 0 else 0.0
        offer_comp = (comp / offer * 100) if offer > 0 else 0.0
        st.caption(
            f"KFI→Application: {kfi_app:.1f}% | Application→Offer: {app_offer:.1f}% | "
            f"Offer→Completion: {offer_comp:.1f}%"
        )
        st.caption("Based on snapshot stage counts — not loan-level transitions.")

    st.markdown("### Pipeline Composition")
    strat_df = add_pipeline_stratification_buckets(live_df)

    _treemap_pair(
        strat_df,
        "broker",
        "loan_amount",
        "Broker Distribution",
        "Pipeline Amount by Broker (Top 15)",
        "Applications by Broker (Top 15)",
    )
    _treemap_pair(
        strat_df,
        "property_region",
        "loan_amount",
        "Geographic Distribution",
        "Pipeline Amount by Region (Top 15)",
        "Applications by Region (Top 15)",
    )
    _bar_strat_pair(strat_df, "ltv_bucket", "loan_amount", "LTV Distribution", "Pipeline Amount by LTV", "Applications by LTV")
    _bar_strat_pair(strat_df, "ticket_bucket", "loan_amount", "Ticket Size Distribution", "Pipeline Amount by Ticket Size", "Applications by Ticket Size")
    _bar_strat_pair(strat_df, "age_bucket", "loan_amount", "Borrower Age Distribution", "Pipeline Amount by Borrower Age", "Applications by Borrower Age")
    _render_portfolio_concentration_matrix(
        strat_df.rename(columns={"property_region": "geographic_region", "broker": "broker_channel", "product": "erm_product_type"}),
        exposure_col="loan_amount",
        key_prefix="pipeline",
    )

    st.markdown("### Reconciliation & Control")
    st.caption("Control check: completed pipeline applications reconciled against funded portfolio records.")
    recon_df = reconcile_completed_pipeline_to_funded(pipeline_df, funded_df)
    if recon_df.empty:
        st.info("No completions available in this snapshot for reconciliation checks.")
    else:
        summary = summarize_reconciliation(recon_df, funded_df)
        r1, r2, r3, r4 = st.columns(4)
        r1.markdown(_kpi_tile("Completions", f"{summary.pipeline_rows:,}", "Pipeline completed records"), unsafe_allow_html=True)
        r2.markdown(_kpi_tile("Matched", f"{summary.matched_pipeline_rows:,}", "Confirmed funded matches"), unsafe_allow_html=True)
        r3.markdown(_kpi_tile("Pending", f"{summary.unmatched_pipeline_rows:,}", "Awaiting confirmation"), unsafe_allow_html=True)
        r4.markdown(_kpi_tile("Funded Only", f"{summary.funded_rows_not_seen_in_pipeline:,}", "In funded tape only"), unsafe_allow_html=True)


def render_forward_exposure_tab(funded_df: pd.DataFrame) -> None:
    st.markdown("### Forward Exposure")
    st.caption("Management view of current funded book, run-rate trajectory, and forward expected exposure.")

    snapshot_state = _resolve_pipeline_snapshot_inputs(show_controls=False)
    pipeline_df = snapshot_state["pipeline_df"]
    if pipeline_df is None or pipeline_df.empty:
        st.info("Forward Exposure requires a valid pipeline snapshot from the Pipeline tab.")
        return

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

    st.markdown("### Forward Forecast KPI Strip")
    if expected_df.empty:
        st.info("Forecast KPIs are unavailable because no expected forward records were generated.")
    else:
        w = pd.to_numeric(expected_df["expected_funded_amount"], errors="coerce").fillna(0)
        forecast_balance = float(w.sum())
        wa_ltv_fwd = float(weighted_average(pd.to_numeric(expected_df.get("current_ltv"), errors="coerce"), w)) if w.sum() > 0 else 0.0
        wa_rate_subtitle = "Balance-weighted using expected forward balance"
        if "product_rate" in expected_df.columns and w.sum() > 0:
            wa_rate_fwd = float(weighted_average(pd.to_numeric(expected_df.get("product_rate"), errors="coerce"), w))
        else:
            rate_series = pd.to_numeric(expected_df.get("product_rate"), errors="coerce")
            wa_rate_fwd = float(rate_series.mean()) if rate_series.notna().any() else 0.0
            wa_rate_subtitle = "Simple average fallback (forward balance unavailable)"
        wa_age_fwd = float(weighted_average(pd.to_numeric(expected_df.get("youngest_borrower_age"), errors="coerce"), w)) if ("youngest_borrower_age" in expected_df.columns and w.sum() > 0) else 0.0
        avg_loan_size = float(forecast_balance / max(len(expected_df), 1))
        top_region = (
            expected_df.groupby("property_region", observed=True)["expected_funded_amount"].sum().sort_values(ascending=False).head(1)
            if "property_region" in expected_df.columns else pd.Series(dtype=float)
        )
        top_broker = (
            expected_df.groupby("broker", observed=True)["expected_funded_amount"].sum().sort_values(ascending=False).head(1)
            if "broker" in expected_df.columns else pd.Series(dtype=float)
        )
        monthly_run_rate = 0.0
        if "expected_funded_date" in expected_df.columns:
            run = expected_df.copy()
            run["exp_month"] = pd.to_datetime(run["expected_funded_date"], errors="coerce").dt.to_period("M")
            monthly_series = run.groupby("exp_month", observed=True)["expected_funded_amount"].sum()
            monthly_run_rate = float(monthly_series.mean()) if not monthly_series.empty else 0.0
        annualised_run_rate = monthly_run_rate * 12
        weekly_run_rate = monthly_run_rate / 4.345 if monthly_run_rate > 0 else 0.0
        if forecast_balance >= 100_000_000:
            weeks_to_100mm = 0.0
        elif weekly_run_rate > 0:
            weeks_to_100mm = (100_000_000 - forecast_balance) / weekly_run_rate
        else:
            weeks_to_100mm = None

        tiles = st.columns(5)
        tiles[0].markdown(_kpi_tile("Forecast balance", format_currency(forecast_balance), "Forward expected exposure"), unsafe_allow_html=True)
        tiles[1].markdown(_kpi_tile("WA LTV", f"{wa_ltv_fwd:.1f}%", "Forecast weighted"), unsafe_allow_html=True)
        tiles[2].markdown(_kpi_tile("WA interest rate", f"{wa_rate_fwd:.2%}", wa_rate_subtitle), unsafe_allow_html=True)
        tiles[3].markdown(_kpi_tile("WA borrower age", f"{wa_age_fwd:.0f}", "Forecast weighted"), unsafe_allow_html=True)
        tiles[4].markdown(_kpi_tile("Average loan size", format_currency(avg_loan_size), "Forecast weighted"), unsafe_allow_html=True)
        tiles2 = st.columns(5)
        if not top_region.empty and forecast_balance > 0:
            region_amt = float(top_region.iloc[0])
            region_name = str(top_region.index[0])
            region_pct = region_amt / forecast_balance * 100
            region_value = f"{region_pct:.1f}%"
            region_sub = f"{region_name} · {format_currency(region_amt)} (forward only)"
        else:
            region_value = "N/A"
            region_sub = "Forward-only concentration unavailable"
        if not top_broker.empty and forecast_balance > 0:
            broker_amt = float(top_broker.iloc[0])
            broker_name = str(top_broker.index[0])
            broker_pct = broker_amt / forecast_balance * 100
            broker_value = f"{broker_pct:.1f}%"
            broker_sub = f"{broker_name} · {format_currency(broker_amt)} (forward only)"
        else:
            broker_value = "N/A"
            broker_sub = "Forward-only concentration unavailable"
        tiles2[0].markdown(_kpi_tile("Largest geographic region exposure", region_value, region_sub), unsafe_allow_html=True)
        tiles2[1].markdown(_kpi_tile("Largest broker exposure", broker_value, broker_sub), unsafe_allow_html=True)
        tiles2[2].markdown(_kpi_tile("Annualised run-rate", format_currency(annualised_run_rate), "Forecast completions pace"), unsafe_allow_html=True)
        tiles2[3].markdown(_kpi_tile("Monthly run-rate completions", format_currency(monthly_run_rate), "Forecast completions pace"), unsafe_allow_html=True)
        weeks_text = f"{weeks_to_100mm:,.1f} weeks" if weeks_to_100mm is not None else "N/A"
        tiles2[4].markdown(_kpi_tile("Weeks to £100MM completions", weeks_text, "Based on forward monthly completion run-rate"), unsafe_allow_html=True)

    # Top summary: funded book truth
    st.markdown("### Funded Book Summary")
    balance_col = _pick_first_col(funded_df, ["total_balance", "current_outstanding_balance", "current_principal_balance"])
    ltv_col = _pick_first_col(funded_df, ["current_loan_to_value", "current_ltv"])
    rate_col = _pick_first_col(funded_df, ["current_interest_rate", "interest_rate"])
    age_col = _pick_first_col(funded_df, ["youngest_borrower_age", "borrower_age"])

    total_funded = float(pd.to_numeric(funded_df.get(balance_col, 0), errors="coerce").fillna(0).sum()) if balance_col else 0.0
    funded_count = int(len(funded_df))
    wa_ltv = float(weighted_average(funded_df.get(ltv_col, pd.Series(dtype=float)), funded_df.get(balance_col, pd.Series(dtype=float)))) if (ltv_col and balance_col and total_funded > 0) else 0.0
    wa_rate = float(weighted_average(funded_df.get(rate_col, pd.Series(dtype=float)), funded_df.get(balance_col, pd.Series(dtype=float)))) if (rate_col and balance_col and total_funded > 0) else 0.0
    wa_age = float(weighted_average(funded_df.get(age_col, pd.Series(dtype=float)), funded_df.get(balance_col, pd.Series(dtype=float)))) if (age_col and balance_col and total_funded > 0) else 0.0

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.markdown(_kpi_tile("Total Funded Exposure", format_currency(total_funded), "Funded tape truth"), unsafe_allow_html=True)
    s2.markdown(_kpi_tile("Loan Count", f"{funded_count:,}", "Funded portfolio"), unsafe_allow_html=True)
    s3.markdown(_kpi_tile("WA LTV", f"{wa_ltv:.1f}%", "Balance-weighted"), unsafe_allow_html=True)
    s4.markdown(_kpi_tile("WA Rate", f"{wa_rate:.2%}", "Balance-weighted"), unsafe_allow_html=True)
    s5.markdown(_kpi_tile("WA Borrower Age", f"{wa_age:.0f}", "Balance-weighted"), unsafe_allow_html=True)

    st.markdown("### Run Rate & Milestones")
    history_df = _build_weekly_snapshot_history(snapshot_state["selected_snapshot"], max_weeks=16)
    completion_hist = history_df[history_df["stage"].eq("COMPLETED")].sort_values("snapshot_date") if not history_df.empty else pd.DataFrame()
    time_to_100mm = "Unavailable"
    if total_funded >= 100_000_000:
        time_to_100mm = "Reached"

    if completion_hist.empty:
        st.info("Run-rate analytics will appear once weekly completions history is available.")
    else:
        recent = completion_hist.tail(5)
        avg_weekly = float(recent["amount"].mean()) if not recent.empty else 0.0
        annualised = avg_weekly * 52
        combined_forward = total_funded + float(expected_df["expected_funded_amount"].sum()) if not expected_df.empty else total_funded
        runway_weeks = (1_000_000_000 - total_funded) / avg_weekly if avg_weekly > 0 and total_funded < 1_000_000_000 else None
        if total_funded < 100_000_000 and avg_weekly > 0:
            time_to_100mm = f"{((100_000_000 - total_funded) / avg_weekly):,.1f} weeks"

        rr1, rr2, rr3, rr4 = st.columns(4)
        rr1.markdown(_kpi_tile("5W Avg Completions", format_currency(avg_weekly), "Recent weekly average"), unsafe_allow_html=True)
        rr2.markdown(_kpi_tile("Annualised Run Rate", format_currency(annualised), "5W pace annualised"), unsafe_allow_html=True)
        rr3.markdown(_kpi_tile("Projected Forward", format_currency(combined_forward), "Funded + expected pipeline"), unsafe_allow_html=True)
        milestone_text = f"{runway_weeks:,.1f} weeks to £1bn" if runway_weeks is not None else "Milestone reached / not set"
        rr4.markdown(_kpi_tile("Milestone Trajectory", milestone_text, "Indicative pacing"), unsafe_allow_html=True)

        fig_run = go.Figure()
        fig_run.add_trace(go.Bar(x=completion_hist["snapshot_date"], y=completion_hist["amount"], marker_color=PRIMARY_COLOR, name="Weekly Completions"))
        fig_run.add_trace(go.Scatter(x=recent["snapshot_date"], y=[avg_weekly] * len(recent), mode="lines", line=dict(color=SECONDARY_COLOR, dash="dash"), name="5W average"))
        fig_run.update_yaxes(title_text="Amount (£)")
        fig_run.update_xaxes(title_text="Week")
        st.plotly_chart(apply_chart_theme(fig_run, "Completions Run-Rate Trend"), use_container_width=True)
    st.markdown(_kpi_tile("Time to £100MM", time_to_100mm, "Mandatory milestone KPI"), unsafe_allow_html=True)

    st.markdown("### Forward Exposure")
    expected_amount = float(expected_df["expected_funded_amount"].sum()) if not expected_df.empty else 0.0
    high_conf_amount = float(expected_df["high_confidence_expected_amount"].sum()) if not expected_df.empty else 0.0
    combined_amount = total_funded + expected_amount

    f1, f2, f3 = st.columns(3)
    f1.markdown(_kpi_tile("Funded Truth", format_currency(total_funded), "Current funded exposure"), unsafe_allow_html=True)
    f2.markdown(_kpi_tile("Expected Pipeline Funding", format_currency(expected_amount), "Modelled expected completion"), unsafe_allow_html=True)
    f3.markdown(_kpi_tile("Combined Forward Exposure", format_currency(combined_amount), f"High-confidence: {format_currency(high_conf_amount)}"), unsafe_allow_html=True)

    st.markdown("### Concentration Risk")
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

    if not forward_region_df.empty:
        region_col = "property_region" if "property_region" in forward_region_df.columns else forward_region_df.columns[0]
        c1, c2 = st.columns(2)
        with c1:
            if "combined_forward_exposure" in forward_region_df.columns:
                top = forward_region_df[[region_col, "combined_forward_exposure"]].sort_values("combined_forward_exposure", ascending=False).head(15)
                fig = px.treemap(
                    top,
                    path=[px.Constant("All Regions"), region_col],
                    values="combined_forward_exposure",
                    color="combined_forward_exposure",
                    color_continuous_scale=[(0, "#F0F2F6"), (1, PRIMARY_COLOR)],
                )
                fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), title_text="Forward Exposure by Region", title_x=0)
                fig.update_traces(texttemplate="<b>%{label}</b><br>£%{value:,.0f}<br>(%{percentEntry:.1%})")
                fig.update_coloraxes(showscale=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Region forward exposure is not available.")

        with c2:
            if {"funded_current_exposure", "expected_pipeline_exposure"}.issubset(forward_region_df.columns):
                comp = forward_region_df[[region_col, "funded_current_exposure", "expected_pipeline_exposure"]].rename(
                    columns={region_col: "Region", "funded_current_exposure": "Funded", "expected_pipeline_exposure": "Expected"}
                )
                fig_comp = px.bar(comp, x="Region", y=["Funded", "Expected"], barmode="stack", color_discrete_sequence=[PRIMARY_COLOR, SECONDARY_COLOR])
                fig_comp.update_layout(legend_title_text="")
                st.plotly_chart(apply_chart_theme(fig_comp, "Funded vs Forward by Region"), use_container_width=True)
            else:
                st.info("Region concentration comparison is not available.")
    else:
        st.info("Regional concentration data is not available.")

    # Broker concentration framing
    broker_col = _pick_first_col(funded_df, ["broker_channel", "broker"])
    expected_broker_col = "broker" if "broker" in expected_df.columns else None
    if broker_col or expected_broker_col:
        st.markdown("#### Broker Concentration")
        funded_broker = _aggregate_dimension(funded_df.rename(columns={broker_col: "broker"}) if broker_col else pd.DataFrame(), "broker", balance_col) if (broker_col and balance_col) else pd.DataFrame(columns=["broker", "amount"])
        expected_broker = _aggregate_dimension(expected_df, "broker", "expected_funded_amount") if expected_broker_col else pd.DataFrame(columns=["broker", "amount"])
        merged = funded_broker.merge(expected_broker, on="broker", how="outer", suffixes=("_funded", "_expected")).fillna(0)
        if not merged.empty:
            merged["combined"] = merged.get("amount_funded", 0) + merged.get("amount_expected", 0)
            top = merged.sort_values("combined", ascending=False).head(15)
            bc1, bc2 = st.columns(2)
            with bc1:
                fig = px.treemap(
                    top,
                    path=[px.Constant("All Brokers"), "broker"],
                    values="combined",
                    color="combined",
                    color_continuous_scale=[(0, "#F0F2F6"), (1, PRIMARY_COLOR)],
                )
                fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), title_text="Forward Exposure by Broker", title_x=0)
                fig.update_traces(texttemplate="<b>%{label}</b><br>£%{value:,.0f}<br>(%{percentEntry:.1%})")
                fig.update_coloraxes(showscale=False)
                st.plotly_chart(fig, use_container_width=True)
            with bc2:
                bar_df = top.rename(columns={"amount_funded": "Funded", "amount_expected": "Expected", "broker": "Broker"})
                fig_bar = px.bar(bar_df, x="Broker", y=["Funded", "Expected"], barmode="stack", color_discrete_sequence=[PRIMARY_COLOR, SECONDARY_COLOR])
                fig_bar.update_layout(legend_title_text="")
                st.plotly_chart(apply_chart_theme(fig_bar, "Funded vs Forward by Broker"), use_container_width=True)

    st.markdown("### Relationship Analysis")
    forward_visual_df = expected_df.rename(
        columns={
            "expected_funded_amount": "total_balance",
            "property_region": "geographic_region",
            "broker": "broker_channel",
            "product": "erm_product_type",
        }
    ).copy()
    if "youngest_borrower_age" not in forward_visual_df.columns and "borrower_age" in forward_visual_df.columns:
        forward_visual_df["youngest_borrower_age"] = forward_visual_df["borrower_age"]
    forward_visual_df = add_pipeline_stratification_buckets(forward_visual_df)
    _render_ltv_age_bubble(
        forward_visual_df,
        exposure_col="total_balance",
        color_col="erm_product_type",
        title="LTV vs Youngest Borrower Age",
        key_prefix="forward",
    )
    _render_portfolio_concentration_matrix(forward_visual_df, exposure_col="total_balance", key_prefix="forward")

    st.markdown("### Model & Calibration Status")
    model_mode = "Empirical" if config_ok else "Hybrid Defaults"
    calibration_window = f"{min(12, int(history_df['snapshot_blob'].nunique())) if not history_df.empty else 0} weeks"
    match_rate = 0.0
    if not recon_df.empty and getattr(recon_df, "shape", (0, 0))[0] > 0:
        match_rate = float((recon_df.get("is_reconciled_to_funded", False) == True).mean() * 100)  # noqa: E712

    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(_kpi_tile("Model Mode", model_mode, "Expected funding engine"), unsafe_allow_html=True)
    m2.markdown(_kpi_tile("Calibration Window", calibration_window, "Weekly history basis"), unsafe_allow_html=True)
    m3.markdown(_kpi_tile("Match Rate", f"{match_rate:.1f}%", "Completions reconciled"), unsafe_allow_html=True)
    m4.markdown(_kpi_tile("Model Version", ef_config.model_version, "Production configuration"), unsafe_allow_html=True)
