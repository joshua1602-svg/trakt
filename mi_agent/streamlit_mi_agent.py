#!/usr/bin/env python3
"""
streamlit_mi_agent.py — Trakt MI Agent v1 (Streamlit workbench).

A simple, controlled Management Information interface: ask a natural-language
question, which is translated into a governed ``MIQuerySpec`` (deterministic
parser, or an optional cheap LLM), validated, executed against canonical data,
and rendered as an enterprise-ready Plotly chart with the result table,
interpreted query, warnings and exports.

Run:
    streamlit run mi_agent/streamlit_mi_agent.py

All heavy lifting lives in ``mi_agent_workflow`` / the rest of the MI Agent
stack; this file is presentation only. The LLM (when enabled) only ever
proposes MIQuerySpec JSON from the data-free semantic catalogue — it never sees
raw data, executes code, or bypasses the validator.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `streamlit run mi_agent/streamlit_mi_agent.py` from the repo root: make
# the package importable via absolute imports regardless of how it is launched.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from mi_agent.mi_agent_config import get_llm_config  # noqa: E402
from mi_agent.mi_agent_workflow import (  # noqa: E402
    EXAMPLE_QUESTIONS,
    chart_html_str,
    metadata_json_str,
    result_csv_bytes,
    run_mi_agent_query,
    spec_json_str,
)

DEFAULT_SEMANTICS = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"
PRIMARY = "#232D55"   # Trakt navy (matches analytics/charts_plotly.py)
MUTED = "#5A6275"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _find_demo_csv() -> str | None:
    candidates = sorted(_REPO_ROOT.glob("synthetic_demo/**/*canonical_typed.csv"))
    return str(candidates[0]) if candidates else None


def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("df", None)
    ss.setdefault("df_source", None)
    ss.setdefault("question", "")
    ss.setdefault("last_result", None)
    ss.setdefault("history", [])


def _load_dataframe(sidebar) -> None:
    """Resolve the active dataframe from uploader / path / demo into session."""
    ss = st.session_state
    uploaded = sidebar.file_uploader("Upload canonical CSV", type=["csv"])
    if uploaded is not None:
        try:
            ss.df = pd.read_csv(uploaded)
            ss.df_source = f"upload: {uploaded.name}"
        except Exception as exc:  # noqa: BLE001
            sidebar.error(f"Could not read uploaded CSV: {exc}")

    with sidebar.expander("Developer: local path / demo data"):
        path = st.text_input("Local CSV path", value="")
        if st.button("Load from path") and path.strip():
            try:
                ss.df = pd.read_csv(path.strip())
                ss.df_source = f"path: {path.strip()}"
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not read CSV at path: {exc}")
        demo = _find_demo_csv()
        if demo and st.button("Load synthetic demo CSV"):
            ss.df = pd.read_csv(demo)
            ss.df_source = f"demo: {Path(demo).name}"
        if demo:
            st.caption(f"Demo available: `{Path(demo).name}`")


def _render_interpreted(result: dict) -> None:
    st.markdown("##### Interpreted as")
    interp = result.get("interpreted") or {}
    rows = "".join(
        f"<tr><td style='color:{MUTED};padding:2px 16px 2px 0'>{k}</td>"
        f"<td style='font-weight:600'>{v}</td></tr>"
        for k, v in interp.items()
    )
    st.markdown(f"<table>{rows}</table>", unsafe_allow_html=True)


def _render_validation(result: dict) -> None:
    validation = result.get("validation") or {}
    parse_meta = result.get("parse_metadata") or {}
    if validation.get("ok"):
        st.success("Validation passed.")
        if parse_meta.get("parser_mode") == "llm" and parse_meta.get("repair_attempts"):
            st.info(
                f"LLM output was repaired: repair_attempts="
                f"{parse_meta.get('repair_attempts')}, original error count="
                f"{parse_meta.get('original_error_count')}, final status: "
                f"{parse_meta.get('status')}."
            )
    else:
        errs = validation.get("errors", []) or parse_meta.get("validation_errors", [])
        st.error("The agent understood the request, but the proposed query "
                 "failed validation.")
        if errs:
            st.markdown("**Validation errors:**")
            for e in errs:
                st.markdown(f"- {e}")
        st.markdown("**Suggested action:** rephrase the question, or pick one of "
                    "the suggested examples in the sidebar.")
        if parse_meta.get("parser_mode") == "llm":
            st.caption(f"Parser: LLM · repair_attempts="
                       f"{parse_meta.get('repair_attempts')} · "
                       f"status: {parse_meta.get('status')}")


def _render_run_details(result: dict) -> None:
    """Parser mode + LLM call/cost observability (kept in an expander)."""
    pm = result.get("parse_metadata") or {}
    detail = result.get("parser_mode_detail") or pm.get("parser_mode_detail") \
        or result.get("parser_mode")
    llm = pm.get("llm") or {}
    cost_status = llm.get("cost_estimate_status")
    cost = llm.get("estimated_total_cost", 0.0)
    cost_str = (f"${cost:.4f}" if cost_status == "estimated"
                else ("$0.0000" if (llm.get("calls", 0) == 0) else "unknown"))
    with st.expander("Run details (parser & cost)"):
        st.markdown(f"- **Parser mode:** {detail}")
        st.markdown(f"- **Model:** {pm.get('model') or '—'}")
        st.markdown(f"- **LLM calls:** {llm.get('calls', 0)}")
        st.markdown(f"- **Repair attempts:** {pm.get('repair_attempts', 0)}")
        if pm.get("repair_skipped_reason"):
            st.markdown(f"- **Repair skipped:** {pm.get('repair_skipped_reason')}")
        st.markdown(f"- **Estimated cost:** {cost_str}")
        if llm.get("calls"):
            st.markdown(
                f"- **Tokens:** in={llm.get('input_tokens', 0)}, "
                f"out={llm.get('output_tokens', 0)}, "
                f"total={llm.get('total_tokens', 0)} · "
                f"cache_supported={llm.get('prompt_cache_supported')}"
            )
        conf = pm.get("parser_confidence")
        if conf:
            st.markdown(f"- **Parser confidence:** {conf}")


def _render_outputs(result: dict) -> None:
    # Chart
    chart = result.get("chart_result")
    st.markdown("#### Chart")
    if chart is not None:
        st.plotly_chart(chart.fig, use_container_width=True)
    else:
        st.info("No chart for this result (table/summary-only output).")

    # Result table
    qres = result.get("query_result")
    st.markdown("#### Result table")
    if qres is not None and qres.data is not None:
        st.dataframe(qres.data, use_container_width=True, hide_index=True)
        st.caption(f"{qres.result_type} · {qres.row_count} row(s)")

    # Interpreted query + validation
    col1, col2 = st.columns(2)
    with col1:
        _render_interpreted(result)
    with col2:
        st.markdown("##### Validation")
        _render_validation(result)

    # Warnings
    warnings = result.get("warnings") or []
    with st.expander(f"Warnings and assumptions ({len(warnings)})",
                     expanded=bool(warnings)):
        if warnings:
            for w in warnings:
                st.markdown(f"- {w}")
        else:
            st.caption("No warnings.")

    # Run details (parser mode + LLM cost/token observability)
    _render_run_details(result)

    # Interpreted spec (raw)
    with st.expander("Interpreted query (MIQuerySpec JSON)"):
        st.json(result.get("spec") or {})

    # Exports
    st.markdown("#### Export")
    c1, c2, c3, c4 = st.columns(4)
    if qres is not None:
        c1.download_button("Result CSV", data=result_csv_bytes(qres),
                           file_name="mi_result.csv", mime="text/csv")
    html = chart_html_str(chart)
    if html:
        c2.download_button("Chart HTML", data=html.encode("utf-8"),
                           file_name="mi_chart.html", mime="text/html")
    c3.download_button("MIQuerySpec JSON",
                       data=spec_json_str(result.get("spec_obj") or result.get("spec")),
                       file_name="mi_query_spec.json", mime="application/json")
    c4.download_button("Metadata JSON", data=metadata_json_str(result),
                       file_name="mi_metadata.json", mime="application/json")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    st.set_page_config(page_title="Trakt MI Agent", layout="wide")
    _init_state()
    ss = st.session_state
    cfg = get_llm_config()

    st.markdown(
        f"<h2 style='color:{PRIMARY};margin-bottom:0'>Trakt MI Agent</h2>"
        f"<p style='color:{MUTED};margin-top:4px'>Ask a Management Information "
        "question in plain English. It is translated into a governed, validated "
        "query — never free-form code — then executed and charted.</p>",
        unsafe_allow_html=True,
    )

    # ---- Sidebar ----------------------------------------------------------
    sb = st.sidebar
    sb.markdown("### Data")
    _load_dataframe(sb)
    if ss.df is not None:
        sb.success(f"Loaded {len(ss.df):,} rows ({ss.df_source}).")

    sb.markdown("### Parser")
    llm_choice = sb.radio(
        "Mode", ["Deterministic", "LLM"],
        index=1 if cfg.available else 0,
        help="The LLM only proposes a governed MIQuerySpec; the validator and "
             "executor remain the control layer.",
    )
    if llm_choice == "LLM" and not cfg.available:
        sb.warning("LLM requested but unavailable — using deterministic parser.")
    sb.caption(f"LLM provider: **{cfg.provider}** · model: **{cfg.model}**")
    sb.caption(cfg.status)
    for w in cfg.warnings:
        sb.caption(f"⚠️ {w}")

    with sb.expander("Advanced"):
        semantics_path = st.text_input("Semantics registry path",
                                       value=str(DEFAULT_SEMANTICS))

    sb.markdown("### Example questions")
    for ex in EXAMPLE_QUESTIONS:
        if sb.button(ex, use_container_width=True):
            ss.question = ex

    # ---- Question input ---------------------------------------------------
    typed = st.chat_input("Ask an MI question, e.g. 'Show balance by region'")
    if typed:
        ss.question = typed
    question = ss.question

    if not question:
        st.info("Enter a question above or pick an example from the sidebar.")
        return
    if ss.df is None:
        st.error("No data loaded. Upload a canonical CSV (or load the demo) "
                 "from the sidebar first.")
        return

    st.markdown(f"**Question:** {question}")

    use_llm = (llm_choice == "LLM") and cfg.available
    parser_mode = "llm" if use_llm else "deterministic"
    try:
        result = run_mi_agent_query(
            question, ss.df, semantics_path,
            llm_enabled=use_llm, model=cfg.model, parser_mode=parser_mode,
            max_repair_attempts=cfg.max_repair_attempts,
            catalog_mode=cfg.catalog_mode, zero_cost_first=cfg.zero_cost_first,
            provider=cfg.provider,
        )
    except Exception as exc:  # noqa: BLE001 - never crash the app
        st.error(f"Unexpected error while running the query: {exc}")
        return

    ss.last_result = result
    ss.history.append({"question": question, "ok": result.get("ok")})

    if result.get("error") and not result.get("ok"):
        # Validation failures get the dedicated panel; other errors shown plainly.
        if result.get("validation") and not result["validation"].get("ok"):
            _render_interpreted(result)
            _render_validation(result)
        else:
            st.error(result["error"])
            if result.get("interpreted"):
                _render_interpreted(result)
        _render_run_details(result)
        return

    _render_outputs(result)


main()
