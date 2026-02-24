#!/usr/bin/env python3
"""
exception_queue.py

Streamlit Exception Triage UI – Phase 1.

Standalone Streamlit page for reviewing validation findings, applying
overrides / accepts / escalations, and maintaining an immutable remediation
ledger with hash-chain integrity.

Run with:
    streamlit run exception_queue.py
"""

import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from exception_db import (
    init_db,
    get_snapshots,
    get_findings,
    get_remediations,
    create_remediation,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_PATH = "trakt_exceptions.db"

MATERIALITY_COLOURS = {
    "BLOCKING": "#FF4B4B",
    "REVIEW": "#FFA500",
    "INFO": "#888888",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _colour_materiality(val: str) -> str:
    colour = MATERIALITY_COLOURS.get(val, "#888888")
    return f"color: {colour}; font-weight: bold"


def _load_canonical_value(
    canonical_path: str | None,
    row_index: int | None,
    field_name: str | None,
) -> str | None:
    """Try to read the original value from the canonical CSV."""
    if not canonical_path or row_index is None or not field_name:
        return None
    p = Path(canonical_path)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, low_memory=False)
        if field_name in df.columns and 0 <= row_index < len(df):
            val = df.iloc[row_index][field_name]
            return str(val) if pd.notna(val) else None
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Exception Queue", layout="wide")
st.title("Exception Queue — Validation Triage")

conn = init_db(DB_PATH)
snapshots = get_snapshots(conn)

if not snapshots:
    st.warning("No snapshots found. Run ingest_violations.py first.")
    st.stop()


# ---------------------------------------------------------------------------
# Snapshot selector
# ---------------------------------------------------------------------------

snapshot_options = {
    f"{s['id'][:8]}… | {s['input_file']} | {s['created_at'][:19]} | {s['status']}": s["id"]
    for s in snapshots
}
selected_label = st.selectbox("Select snapshot", list(snapshot_options.keys()))
snapshot_id = snapshot_options[selected_label]

# Retrieve current snapshot metadata
snapshot_meta = next(s for s in snapshots if s["id"] == snapshot_id)


# ---------------------------------------------------------------------------
# Sidebar stats
# ---------------------------------------------------------------------------

all_findings = get_findings(conn, snapshot_id)

with st.sidebar:
    st.header("Snapshot Stats")
    st.markdown(f"**File:** {snapshot_meta['input_file']}")
    st.markdown(f"**Rows:** {snapshot_meta['row_count']}")
    st.markdown(f"**Status:** {snapshot_meta['status']}")
    st.divider()

    status_counts = {"open": 0, "accepted": 0, "overridden": 0, "escalated": 0}
    for f in all_findings:
        s = f["status"]
        if s in status_counts:
            status_counts[s] += 1

    for label, count in status_counts.items():
        st.metric(label.capitalize(), count)


# ---------------------------------------------------------------------------
# Findings table with filters
# ---------------------------------------------------------------------------

st.subheader("Findings")

col1, col2, col3, col4 = st.columns(4)
with col1:
    filter_status = st.multiselect(
        "Status",
        ["open", "accepted", "overridden", "escalated"],
        default=["open"],
    )
with col2:
    filter_severity = st.multiselect("Severity", ["error", "warn"])
with col3:
    filter_materiality = st.multiselect("Materiality", ["BLOCKING", "REVIEW", "INFO"])
with col4:
    unique_fields = sorted({f["field_name"] for f in all_findings if f["field_name"]})
    filter_field = st.multiselect("Field", unique_fields)

# Apply filters
filtered = all_findings
if filter_status:
    filtered = [f for f in filtered if f["status"] in filter_status]
if filter_severity:
    filtered = [f for f in filtered if f["severity"] in filter_severity]
if filter_materiality:
    filtered = [f for f in filtered if f["materiality"] in filter_materiality]
if filter_field:
    filtered = [f for f in filtered if f["field_name"] in filter_field]

if not filtered:
    st.info("No findings match the selected filters.")
    st.stop()

# Display as dataframe
display_cols = [
    "rule_id", "severity", "field_name", "row_index",
    "message", "materiality", "status",
]
df_display = pd.DataFrame(filtered)[display_cols + ["id"]]

st.dataframe(
    df_display[display_cols].style.applymap(
        _colour_materiality, subset=["materiality"]
    ),
    use_container_width=True,
    hide_index=True,
)


# ---------------------------------------------------------------------------
# Override action panel
# ---------------------------------------------------------------------------

st.subheader("Remediation Action")

finding_options = {
    f"{f['rule_id']} | row {f['row_index']} | {f['field_name']} | {f['message'][:60]}": f["id"]
    for f in filtered
}
selected_finding_label = st.selectbox("Select finding to remediate", list(finding_options.keys()))
selected_finding_id = finding_options[selected_finding_label]
selected_finding = next(f for f in filtered if f["id"] == selected_finding_id)

# Show finding detail
st.markdown("---")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown(f"**Rule:** `{selected_finding['rule_id']}`")
    st.markdown(f"**Field:** `{selected_finding['field_name'] or 'N/A'}`")
    st.markdown(f"**Row:** `{selected_finding['row_index']}`")
    st.markdown(f"**Severity:** `{selected_finding['severity']}`")
    st.markdown(f"**Materiality:** `{selected_finding['materiality']}`")
with col_b:
    st.markdown(f"**Message:** {selected_finding['message']}")
    st.markdown(f"**Classification:** `{selected_finding['classification']}`")
    st.markdown(f"**Status:** `{selected_finding['status']}`")

    # Try to show current value from canonical CSV
    original_value = _load_canonical_value(
        snapshot_meta.get("canonical_path"),
        selected_finding.get("row_index"),
        selected_finding.get("field_name"),
    )
    if original_value is not None:
        st.markdown(f"**Current value in CSV:** `{original_value}`")
    else:
        original_value = ""

# Remediation form
with st.form("remediation_form", clear_on_submit=True):
    action = st.selectbox("Action", ["accept", "override", "escalate"])
    override_value = st.text_input(
        "Override value (required if action is 'override')",
        disabled=(action != "override") if False else False,
    )
    justification = st.text_area("Justification (mandatory)", height=100)
    user_name = st.text_input("Your name")

    submitted = st.form_submit_button("Submit Remediation")

    if submitted:
        # Validation
        if not justification.strip():
            st.error("Justification is mandatory.")
        elif not user_name.strip():
            st.error("Please enter your name.")
        elif action == "override" and not override_value.strip():
            st.error("Override value is required when action is 'override'.")
        else:
            user_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, user_name.strip().lower()))
            create_remediation(
                conn=conn,
                finding_id=selected_finding_id,
                snapshot_id=snapshot_id,
                action=action,
                field_name=selected_finding.get("field_name"),
                row_index=selected_finding.get("row_index"),
                original_value=original_value,
                override_value=override_value if action == "override" else None,
                rule_id=selected_finding["rule_id"],
                justification=justification.strip(),
                user_id=user_id,
                user_name=user_name.strip(),
            )
            st.success(
                f"Remediation recorded: {action} for finding "
                f"{selected_finding['rule_id']} (row {selected_finding['row_index']})"
            )
            st.rerun()


# ---------------------------------------------------------------------------
# Remediation ledger
# ---------------------------------------------------------------------------

st.subheader("Remediation Ledger")
remediations = get_remediations(conn, snapshot_id)

if remediations:
    ledger_cols = [
        "created_at", "action", "rule_id", "field_name", "row_index",
        "original_value", "override_value", "justification",
        "user_name", "record_hash", "prev_hash",
    ]
    df_ledger = pd.DataFrame(remediations)
    st.dataframe(
        df_ledger[[c for c in ledger_cols if c in df_ledger.columns]],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No remediations recorded for this snapshot yet.")
