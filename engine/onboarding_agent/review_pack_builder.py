"""
review_pack_builder.py
=====================

PART 9 — static HTML onboarding review pack.

Renders the accumulated :class:`OnboardingProject` state into a single
self-contained HTML file styled to match the existing Trakt demo aesthetic
(``synthetic_demo/demo_overview_v3.html``). No Streamlit, no external assets.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import List

from .onboarding_models import OnboardingProject

_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',Arial,sans-serif;font-size:13px;background:#f0f3f8;color:#1b2132;line-height:1.5}
.page{max-width:1100px;margin:28px auto;padding:0 16px}
.hero{background:linear-gradient(135deg,#1b3872 0%,#2f5d9f 100%);color:#fff;border-radius:10px 10px 0 0;padding:28px 32px 22px}
.hero h1{font-size:22px;font-weight:700;letter-spacing:-.3px}
.hero .sub{font-size:12px;opacity:.85;margin-top:4px}
.card{background:#fff;border:1px solid #d8e0ec;padding:22px 28px;margin-bottom:14px}
.card:last-child{border-radius:0 0 10px 10px}
h2{font-size:15px;font-weight:700;color:#1b3872;margin-bottom:12px;padding-bottom:6px;border-bottom:2px solid #e3eaf7}
table{width:100%;border-collapse:collapse;font-size:12px;margin:8px 0 6px}
th{background:#e8eef7;color:#1b3872;font-weight:600;padding:7px 8px;text-align:left;border:1px solid #d0d9ea;vertical-align:bottom}
td{padding:6px 8px;border:1px solid #e3e8f0;vertical-align:top;word-break:break-word}
tr:nth-child(even) td{background:#f8fafd}
.badge{display:inline-block;border-radius:4px;padding:2px 7px;font-size:11px;font-weight:600}
.b-ok{background:#edf7ee;color:#1f6f3a}
.b-warn{background:#fff4d6;color:#8a5b00}
.b-block{background:#fbe4e4;color:#9b1c1c}
.b-info{background:#e7eefc;color:#274a9c}
.kpi-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;margin-bottom:14px}
.kpi-card{border:1px solid #d8e0ec;border-radius:7px;padding:12px 14px;background:#f9fbfe}
.kpi-label{font-size:11px;color:#5f6b7a;font-weight:500;margin-bottom:4px}
.kpi-value{font-size:20px;font-weight:700;color:#1b3872}
.callout{border-radius:6px;padding:10px 14px;font-size:12px;margin-top:8px;line-height:1.6}
.callout.warn{background:#fff8e1;border-left:4px solid #f59e0b;color:#7c5700}
.callout.pass{background:#edf7ee;border-left:4px solid #1f8f4a;color:#1f4f2f}
.callout.block{background:#fdecec;border-left:4px solid #d23b3b;color:#7a1d1d}
ul{margin:6px 0 6px 18px}
li{margin:3px 0}
.meta{font-size:12px;color:#5f6b7a;margin-bottom:8px}
"""


def _esc(v) -> str:
    return html.escape(str(v))


def _sev_badge(sev: str) -> str:
    cls = {"blocking": "b-block", "high": "b-warn", "medium": "b-info", "info": "b-info"}.get(sev, "b-info")
    return f'<span class="badge {cls}">{_esc(sev)}</span>'


def _table(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return '<p class="meta">None detected.</p>'
    th = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = ""
    for r in rows:
        body += "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def build_review_pack(project: OnboardingProject, out_path: Path) -> Path:
    s = project.to_summary_dict()
    counts = s["counts"]
    blocking_qs = [q for q in project.gap_questions if q.severity == "blocking"]

    status = project.review_status
    if status == "blocked":
        status_badge = '<span class="badge b-block">BLOCKED</span>'
    elif status in ("review_required", "requires_review"):
        status_badge = '<span class="badge b-warn">REVIEW REQUIRED</span>'
    elif status.startswith("ready_for"):
        status_badge = f'<span class="badge b-ok">{_esc(status.upper())}</span>'
    else:
        status_badge = f'<span class="badge b-info">{_esc(status)}</span>'

    # Mode policy (PART 3)
    try:
        from .mode_policy import load_mode_policy
        policy = load_mode_policy(project.onboarding_mode)
    except Exception:
        policy = None
    mode_name = project.onboarding_mode
    objective = getattr(policy, "objective", "") if policy else ""
    recommended_outputs = getattr(policy, "recommended_outputs", []) if policy else []
    optional_outputs = getattr(policy, "optional_outputs", []) if policy else []

    # Blocking / high questions specifically for this mode.
    mode_blocking = [q for q in project.gap_questions if q.severity == "blocking"]
    mode_high = [q for q in project.gap_questions if q.severity == "high"]
    block_rows = [
        [_esc(q.question_id), _sev_badge(q.severity), _esc(q.category), _esc(q.question)]
        for q in (mode_blocking + mode_high)
    ]
    mode_readiness_html = f"""
    <p class="meta">Mode: <span class="badge b-info">{_esc(mode_name)}</span>
      &nbsp;·&nbsp; Readiness: {status_badge}</p>
    <p class="meta">{_esc(objective)}</p>
    <div class="split-grid">
      <div><h4 class="chart-title">Outputs in scope</h4>
        <ul>{''.join(f'<li>{_esc(o)}</li>' for o in recommended_outputs) or '<li>—</li>'}</ul></div>
      <div><h4 class="chart-title">Outputs out of scope / optional</h4>
        <ul>{''.join(f'<li>{_esc(o)}</li>' for o in optional_outputs) or '<li>—</li>'}</ul></div>
    </div>
    <h4 class="chart-title">Blocking / high questions for this mode</h4>
    {_table(["ID", "Severity", "Category", "Question"], block_rows)}
    """

    # Field scope (PART 7) — registry category + core_canonical driven.
    _SCOPE_EXPLANATIONS = {
        "mi_only": [
            "Regulatory non-core fields are excluded from mapping, typing and "
            "enum-validation requirements.",
            "Fields marked core_canonical:true remain in scope even if their "
            "category is regulatory.",
            "Analytics fields are mapped where available but missing analytics "
            "fields do not block onboarding.",
            "Regime configuration, classification_year and ESMA/FCA delivery "
            "readiness are skipped in MI-only mode.",
        ],
        "mna_dd": [
            "Full field coverage is attempted, including regulatory fields, for "
            "diligence and data-quality visibility.",
            "Regulatory gaps are visible but not blocking unless they impair "
            "structural viability.",
            "Regime transformation is optional.",
        ],
        "regulatory_mi": [
            "Full field coverage and regime configuration are in scope.",
            "Mandatory regulatory gaps and invalid mandatory regulatory enums "
            "may block handoff.",
        ],
        "warehouse_securitisation": [
            "Core + analytics + warehouse/funding/pipeline/cashflow fields are in scope.",
            "Regulatory fields are excluded unless regulatory reporting is enabled.",
            "Warehouse core terms and the authoritative balance source may block.",
        ],
    }
    scope_points = _SCOPE_EXPLANATIONS.get(mode_name, [])
    fs = project.field_scope_summary or {}
    by_cat = fs.get("mapping_candidates_by_category", {}) or {}
    metric_rows = [
        ["included_fields_count", fs.get("included_fields_count", "—")],
        ["core_canonical_fields_count", fs.get("core_canonical_fields_count", "—")],
        ["analytics_fields_count", fs.get("analytics_fields_count", "—")],
        ["regulatory_fields_count", fs.get("regulatory_fields_count", "—")],
        ["excluded_regulatory_fields_count", fs.get("excluded_regulatory_fields_count", "—")],
        ["blocking_fields_count", fs.get("blocking_fields_count", "—")],
        ["out_of_scope_fields_count", fs.get("out_of_scope_fields_count", len(project.out_of_scope_fields))],
        ["mapping_candidates_by_category",
         ", ".join(f"{k}={v}" for k, v in by_cat.items()) or "—"],
    ]
    oos_rows = [
        [_esc(o.get("source_file", "")), _esc(o.get("source_column", "")),
         _esc(o.get("candidate_field", "")), _esc(o.get("category", "")), _esc(o.get("reason", ""))]
        for o in project.out_of_scope_fields
    ]
    field_scope_html = f"""
    <ul>{''.join(f'<li>{_esc(p)}</li>' for p in scope_points) or '<li>—</li>'}</ul>
    <h4 class="chart-title">Field scope metrics</h4>
    {_table(["Metric", "Value"], [[_esc(a), _esc(b)] for a, b in metric_rows])}
    <h4 class="chart-title">Out-of-scope fields (excluded by mode)</h4>
    {_table(["File", "Column", "Candidate field", "Category", "Reason"], oos_rows)}
    """

    # 1c. Mapping ambiguities resolved by policy (PART 2)
    amb_rows = []
    for a in project.mapping_ambiguities:
        amb_rows.append([
            _esc(a.source_file), _esc(a.source_column),
            f"{_esc(a.selected_canonical_field)}<br><small>{_esc(a.selected_category)}"
            f"{' · core' if a.selected_core_canonical else ''} · {_esc(round(a.selected_confidence, 3))}</small>",
            f"{_esc(a.alternative_canonical_field)}<br><small>{_esc(a.alternative_category)} · "
            f"{_esc(round(a.alternative_confidence, 3))}</small>",
            _esc(round(a.confidence_delta, 3)),
            '<span class="badge b-warn">review</span>',
        ])
    if mode_name == "mi_only":
        amb_explainer = (
            "Regulatory non-core candidates were not selected because this is "
            "MI-only onboarding. Where a regulatory candidate is core_canonical it "
            "may still be selected; otherwise the analytics interpretation is kept "
            "and the regulatory candidate is diverted to out-of-scope."
        )
    else:
        amb_explainer = (
            "The system found multiple plausible meanings for these source columns. "
            "Because this onboarding mode includes regulatory fields, it selected the "
            "regulatory interpretation as the safer default, but marked it for review."
        )
    ambiguities_html = f"""
    <p class="meta">{_esc(amb_explainer)}</p>
    {_table(["File", "Column", "Selected", "Alternative", "Δ conf", "Status"], amb_rows)}
    """

    # 1d. LLM usage / cost (PART 5) — shown when enabled or skipped due to budget.
    llm = project.llm_usage_summary or {}
    llm_enabled = bool(llm.get("llm_enabled"))
    llm_skipped_budget = int(llm.get("skipped_due_to_budget", 0) or 0)
    show_llm = llm_enabled or llm_skipped_budget or llm.get("status")
    llm_html = ""
    if show_llm:
        llm_rows = [
            ["llm_enabled", llm.get("llm_enabled", False)],
            ["status", llm.get("status", "ok")],
            ["provider / model", f"{llm.get('provider')} / {llm.get('model')}"],
            ["calls_completed", llm.get("calls_completed", 0)],
            ["items_sent", llm.get("items_sent", 0)],
            ["prompt_chars_estimated", llm.get("prompt_chars_estimated", 0)],
            ["output_tokens", llm.get("output_tokens_estimated_or_reported", 0)],
            ["skipped_due_to_zero_cost_first", llm.get("skipped_due_to_zero_cost_first", 0)],
            ["skipped_due_to_budget", llm_skipped_budget],
            ["converted_to_gap_questions", llm.get("unresolved_items_converted_to_gap_questions", 0)],
            ["estimated_cost", llm.get("estimated_cost", 0)],
        ]
        llm_html = _table(["Metric", "Value"],
                          [[_esc(a), _esc(b)] for a, b in llm_rows])

    # 1. Executive summary KPIs
    kpis = [
        ("Files", counts["classified_files"]),
        ("Columns profiled", counts["column_profiles"]),
        ("Overlap findings", counts["overlap_findings"]),
        ("Mapping candidates", counts["mapping_candidates"]),
        ("Open questions", counts["gap_questions"]),
    ]
    kpi_html = "".join(
        f'<div class="kpi-card"><div class="kpi-label">{_esc(l)}</div>'
        f'<div class="kpi-value">{_esc(v)}</div></div>'
        for l, v in kpis
    )

    # 2. File inventory
    inv_rows = [
        [_esc(i.file_name), _esc(i.classification), _esc(i.confidence),
         _esc(i.row_count if i.row_count is not None else "—"),
         _esc(i.column_count if i.column_count is not None else "—"),
         _esc(i.detected_reporting_date or "—")]
        for i in project.file_inventory
    ]
    inv_html = _table(
        ["File", "Classification", "Conf", "Rows", "Cols", "Reporting date"], inv_rows
    )

    # 3. Detected reporting periods
    dates = sorted({p.date_max for p in project.field_profiles if p.likely_reporting_date and p.date_max})
    periods_html = (
        '<ul>' + "".join(f"<li>{_esc(d)}</li>" for d in dates) + "</ul>"
        if dates else '<p class="meta">No reporting date columns detected.</p>'
    )

    # 4. Candidate keys
    key_rows = [
        [_esc(k.candidate_key), _esc(k.file_name), _esc(k.source_column),
         _esc(k.uniqueness_ratio), _esc(k.confidence)]
        for k in project.candidate_keys
    ]
    keys_html = _table(["Key concept", "File", "Column", "Uniqueness", "Conf"], key_rows)

    # 5. Overlap / duplicate fields
    ov_rows = [
        [_esc(o.canonical_candidate), f"{_esc(o.source_file_a)}<br><small>{_esc(o.source_column_a)}</small>",
         f"{_esc(o.source_file_b)}<br><small>{_esc(o.source_column_b)}</small>",
         _esc(o.sample_match_rate), _esc(o.recommended_primary_source)]
        for o in project.overlap_analysis
    ]
    ov_html = _table(
        ["Field", "Source A", "Source B", "Match rate", "Recommended primary"], ov_rows
    )

    # 6. Mapping candidates (mapped only, sorted by file)
    mapped = [m for m in project.mapping_candidates if m.candidate_canonical_field]
    map_rows = [
        [_esc(m.source_file), _esc(m.source_column), _esc(m.candidate_canonical_field),
         _esc(m.method), _esc(round(m.confidence, 3)),
         '<span class="badge b-warn">review</span>' if m.requires_review else '<span class="badge b-ok">auto</span>']
        for m in mapped
    ]
    map_html = _table(
        ["File", "Source column", "Canonical field", "Method", "Conf", "Status"], map_rows
    )

    # 7. Config suggestions
    cfg_rows = [
        [_esc(c.field), _esc(c.suggested_value), _esc(c.confidence),
         _esc(c.review_status), _esc(c.evidence)]
        for c in project.config_suggestions
    ]
    cfg_html = _table(["Field", "Suggested", "Conf", "Status", "Evidence"], cfg_rows)

    # 8. Gap questions
    q_rows = [
        [_esc(q.question_id), _sev_badge(q.severity), _esc(q.category),
         f"{_esc(q.question)}<br><small class='meta'>{_esc(q.reason)}</small>",
         _esc(q.default_recommendation)]
        for q in project.gap_questions
    ]
    q_html = _table(["ID", "Severity", "Category", "Question", "Default"], q_rows)

    # 9. Readiness assessment
    if status == "blocked":
        readiness = (
            f'<div class="callout block">{len(blocking_qs)} blocking question(s) must be '
            "answered before any pipeline handoff. The pack is NOT ready to run Gates 1–5.</div>"
        )
    else:
        readiness = (
            '<div class="callout warn">No hard blockers, but mappings and config require '
            "human review before handoff. Draft handoff artefacts have been generated for review only.</div>"
        )

    # 10. Recommended next actions
    actions = [
        "Resolve all blocking gap questions (dates, mandatory config).",
        "Confirm the authoritative source for each overlapping field.",
        "Review low-confidence mapping candidates and add aliases where needed.",
        "Confirm inferred config values (asset class, currency, reporting date, regime).",
        "Approve the ESMA UK geography policy (GBZZZ for ESMA, ITL3 for MI).",
        "Once approved, promote draft handoff configs into the existing Gate 1–5 pipeline.",
    ]
    actions_html = "<ul>" + "".join(f"<li>{_esc(a)}</li>" for a in actions) + "</ul>"

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trakt Onboarding Review Pack — {_esc(project.client_name)}</title>
<style>{_CSS}</style></head>
<body><div class="page">
  <div class="hero">
    <h1>Onboarding Review Pack</h1>
    <div class="sub">Client: {_esc(project.client_name)} &nbsp;·&nbsp; Project: {_esc(project.project_id)}
    &nbsp;·&nbsp; Mode: {_esc(mode_name)} &nbsp;·&nbsp; Status: {status_badge}</div>
  </div>

  <div class="card"><h2>1. Executive summary</h2>
    <div class="kpi-grid">{kpi_html}</div>
    <p class="meta">Input: {_esc(project.input_dir)} &nbsp;·&nbsp; Output: {_esc(project.output_dir)}</p>
    {readiness}
  </div>

  <div class="card"><h2>1a. Onboarding mode &amp; mode-specific readiness</h2>{mode_readiness_html}</div>
  <div class="card"><h2>1b. Field scope for this onboarding mode</h2>{field_scope_html}</div>
  <div class="card"><h2>1c. Mapping ambiguities resolved by policy</h2>{ambiguities_html}</div>
  {f'<div class="card"><h2>1d. LLM mapping review usage &amp; cost</h2>{llm_html}</div>' if show_llm else ''}

  <div class="card"><h2>2. File inventory</h2>{inv_html}</div>
  <div class="card"><h2>3. Detected reporting periods</h2>{periods_html}</div>
  <div class="card"><h2>4. Candidate keys</h2>{keys_html}</div>
  <div class="card"><h2>5. Source overlap / duplicate fields</h2>{ov_html}</div>
  <div class="card"><h2>6. Mapping candidates</h2>{map_html}</div>
  <div class="card"><h2>7. Config suggestions</h2>{cfg_html}</div>
  <div class="card"><h2>8. Gap questions</h2>{q_html}</div>
  <div class="card"><h2>9. Readiness assessment</h2>{readiness}</div>
  <div class="card"><h2>10. Recommended next actions</h2>{actions_html}</div>
  {_APPROVAL_MARKER}
</div></body></html>"""

    out_path = Path(out_path)
    # If approved artefacts already exist in this output dir, fold them in.
    doc = doc.replace(_APPROVAL_MARKER, build_approval_section_html(out_path.parent))
    out_path.write_text(doc, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# PART 7 — approved artefacts section (shown only once answers are ingested)
# ---------------------------------------------------------------------------

# Marker so the approval section can be (re)injected idempotently after ingestion.
_APPROVAL_MARKER = "<!--APPROVAL_SECTION-->"


def _load_yaml(path: Path):
    import yaml
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_json(path: Path):
    import json
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_approval_section_html(project_dir: Path) -> str:
    """Return an HTML card for the approved artefacts, or '' if none exist."""
    project_dir = Path(project_dir)
    approved_project = _load_yaml(project_dir / "10_approved_onboarding_project.yaml")
    approved_config = _load_yaml(project_dir / "11_approved_config.yaml")
    precedence = _load_yaml(project_dir / "13_source_precedence_rules.yaml") or {}
    enum_decisions = _load_yaml(project_dir / "14_enum_review_decisions.yaml") or {}
    report = _load_json(project_dir / "15_answer_ingestion_report.json") or {}

    if not approved_project:
        return ""

    status = approved_project.get("approval_status", "")
    ready = status == "ready_for_handoff"
    badge = (
        '<span class="badge b-ok">READY FOR HANDOFF</span>' if ready
        else f'<span class="badge b-warn">{_esc(status)}</span>'
    )

    answered = approved_project.get("answered_questions", [])
    unresolved = approved_project.get("unresolved_questions", [])
    cfg = approved_config or {}
    geo = (cfg.get("geography_policy", {}) or {}).get("ESMA_Annex2", {}) or {}

    # Source-of-truth rows
    prec_rows = [
        [_esc(field),
         f"{_esc(v.get('primary_source_file',''))}<br><small>{_esc(v.get('primary_source_column',''))}</small>",
         _esc(v.get("reconciliation_status", ""))]
        for field, v in (precedence or {}).items()
    ]
    # Enum decision rows
    enum_rows = []
    for field, vals in (enum_decisions or {}).items():
        for raw, d in vals.items():
            enum_rows.append([_esc(field), _esc(raw), _esc(d.get("decision", ""))])

    return f"""
  <div class="card" id="approval"><h2>11. Approval status (answer ingestion)</h2>
    <p class="meta">Approval: {badge} &nbsp;·&nbsp;
      blocking answered {_esc(report.get('blocking_answered','?'))}/{_esc(report.get('blocking_total','?'))}
      &nbsp;·&nbsp; invalid answers: {_esc(report.get('answers_invalid', 0))}</p>
    <p class="meta">Answered: {_esc(', '.join(answered)) or '—'}</p>
    <p class="meta">Unresolved: {_esc(', '.join(unresolved)) or '—'}</p>
    <table><tbody>
      <tr><th>Approved reporting date</th><td>{_esc(cfg.get('reporting_date',''))}</td></tr>
      <tr><th>Classification year (policy)</th><td>{_esc(cfg.get('classification_year',''))}</td></tr>
      <tr><th>ESMA UK geography mode</th><td>{_esc(geo.get('uk_geography_mode',''))}</td></tr>
    </tbody></table>
    <h4 class="chart-title">Approved source-of-truth</h4>
    {_table(["Field", "Primary source", "Reconciliation"], prec_rows)}
    <h4 class="chart-title">Approved enum decisions</h4>
    {_table(["Field", "Raw value", "Decision"], enum_rows)}
    <div class="callout {'pass' if ready else 'warn'}">
      {"All blocking questions resolved — pack is READY for pipeline handoff (review-only)."
       if ready else
       "Pack is not yet ready for handoff: unresolved or invalid answers remain."}
    </div>
  </div>"""


def refresh_review_pack_approval(project_dir: Path) -> None:
    """Re-inject the approval section into an existing 08 review pack."""
    project_dir = Path(project_dir)
    pack = project_dir / "08_onboarding_review_pack.html"
    if not pack.exists():
        return
    text = pack.read_text(encoding="utf-8")
    section = build_approval_section_html(project_dir)
    if _APPROVAL_MARKER in text:
        text = text.replace(_APPROVAL_MARKER, section)
    else:
        # Remove any previously injected approval card, then re-add before close.
        import re
        text = re.sub(r'\s*<div class="card" id="approval">.*?</div>\s*(?=</div></body>)',
                      "", text, flags=re.DOTALL)
        text = text.replace("</div></body></html>", f"{section}\n</div></body></html>")
    pack.write_text(text, encoding="utf-8")
