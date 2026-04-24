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
    )
except ImportError:
    from analytics.pipeline_tab_helpers import (
        DEFAULT_EXPECTED_CONFIG_RELATIVE,
        add_pipeline_stratification_buckets,
        load_expected_funding_config_yaml,
    )

FLOW_STAGES = ["KFI", "APPLICATION", "OFFER", "COMPLETED"]
STAGE_LABELS = {
    "KFI": "KFIs",
    "APPLICATION": "Applications",
    "OFFER": "Offers",
    "COMPLETED": "Completions",
}


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

    trend = history_df.copy()
    trend["stage_label"] = trend["stage"].map(STAGE_LABELS)
    trend["week"] = pd.to_datetime(trend["snapshot_date"], errors="coerce").dt.strftime("%d %b %Y")

    col1, col2 = st.columns(2)
    with col1:
        fig_cnt = px.line(
            trend,
            x="week",
            y="count",
            color="stage_label",
            markers=True,
            color_discrete_sequence=[PRIMARY_COLOR, SECONDARY_COLOR, "#7EBAB5", "#6B6B6B"],
        )
        fig_cnt.update_yaxes(title_text="Applications")
        fig_cnt.update_xaxes(title_text="Week")
        st.plotly_chart(apply_chart_theme(fig_cnt, "Weekly Volumes by Stage"), use_container_width=True)

    with col2:
        fig_amt = px.line(
            trend,
            x="week",
            y="amount",
            color="stage_label",
            markers=True,
            color_discrete_sequence=[PRIMARY_COLOR, SECONDARY_COLOR, "#7EBAB5", "#6B6B6B"],
        )
        fig_amt.update_yaxes(title_text="Amount (£)")
        fig_amt.update_xaxes(title_text="Week")
        st.plotly_chart(apply_chart_theme(fig_amt, "Weekly Amounts by Stage"), use_container_width=True)


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

    st.markdown("### Pipeline Snapshot")
    live_series = pipeline_df.get("is_live_stage", pd.Series(dtype=bool))
    live_df = pipeline_df[live_series] if len(live_series) == len(pipeline_df) else pipeline_df.copy()

    active_count = int(len(live_df))
    active_amount = float(pd.to_numeric(live_df.get("loan_amount", 0), errors="coerce").fillna(0).sum())
    completed_count = int((pipeline_df.get("stage", pd.Series(dtype="string")) == "COMPLETED").sum())

    large_threshold = 250_000
    large_mask = pd.to_numeric(live_df.get("loan_amount", 0), errors="coerce").fillna(0) >= large_threshold
    large_count = int(large_mask.sum())
    large_amount = float(pd.to_numeric(live_df.loc[large_mask, "loan_amount"], errors="coerce").fillna(0).sum()) if "loan_amount" in live_df.columns else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(_kpi_tile("Total Active Pipeline", f"{active_count:,}", "Live opportunities"), unsafe_allow_html=True)
    k2.markdown(_kpi_tile("Active Pipeline Amount", format_currency(active_amount), "Indicative value"), unsafe_allow_html=True)
    k3.markdown(_kpi_tile("Completions (Snapshot)", f"{completed_count:,}", "Completed this snapshot"), unsafe_allow_html=True)
    k4.markdown(_kpi_tile("Large Loans ≥ £250k", f"{large_count:,}", f"{format_currency(large_amount)} exposure"), unsafe_allow_html=True)

    if "stage" in pipeline_df.columns:
        col_a, col_b = st.columns(2)
        with col_a:
            stage_counts = pipeline_df.groupby("stage", observed=True).size().reset_index(name="Applications")
            fig_cnt = px.bar(stage_counts, x="stage", y="Applications")
            fig_cnt.update_traces(marker_color=PRIMARY_COLOR, texttemplate="%{y:,}", textposition="outside")
            fig_cnt.update_xaxes(title_text="Stage")
            st.plotly_chart(apply_chart_theme(fig_cnt, "Applications by Stage"), use_container_width=True)

        with col_b:
            stage_amount = (
                pipeline_df.groupby("stage", observed=True)["loan_amount"]
                .sum(min_count=1)
                .fillna(0)
                .reset_index(name="Amount")
            ) if "loan_amount" in pipeline_df.columns else pd.DataFrame()
            if stage_amount.empty:
                st.info("Loan amount data is not available for stage balance reporting.")
            else:
                fig_amt = px.bar(stage_amount, x="stage", y="Amount")
                fig_amt.update_traces(marker_color=PRIMARY_COLOR, texttemplate="£%{y:,.0f}", textposition="outside")
                fig_amt.update_xaxes(title_text="Stage")
                fig_amt.update_yaxes(title_text="Indicative amount (£)")
                st.plotly_chart(apply_chart_theme(fig_amt, "Indicative Stage Balances"), use_container_width=True)

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

    if completion_hist.empty:
        st.info("Run-rate analytics will appear once weekly completions history is available.")
    else:
        recent = completion_hist.tail(5)
        avg_weekly = float(recent["amount"].mean()) if not recent.empty else 0.0
        annualised = avg_weekly * 52
        combined_forward = total_funded + float(expected_df["expected_funded_amount"].sum()) if not expected_df.empty else total_funded
        runway_weeks = (1_000_000_000 - total_funded) / avg_weekly if avg_weekly > 0 and total_funded < 1_000_000_000 else None

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
