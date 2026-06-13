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
    status_badge = {
        "blocked": '<span class="badge b-block">BLOCKED</span>',
        "review_required": '<span class="badge b-warn">REVIEW REQUIRED</span>',
        "draft": '<span class="badge b-info">DRAFT</span>',
    }.get(status, status)

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
    &nbsp;·&nbsp; Status: {status_badge}</div>
  </div>

  <div class="card"><h2>1. Executive summary</h2>
    <div class="kpi-grid">{kpi_html}</div>
    <p class="meta">Input: {_esc(project.input_dir)} &nbsp;·&nbsp; Output: {_esc(project.output_dir)}</p>
    {readiness}
  </div>

  <div class="card"><h2>2. File inventory</h2>{inv_html}</div>
  <div class="card"><h2>3. Detected reporting periods</h2>{periods_html}</div>
  <div class="card"><h2>4. Candidate keys</h2>{keys_html}</div>
  <div class="card"><h2>5. Source overlap / duplicate fields</h2>{ov_html}</div>
  <div class="card"><h2>6. Mapping candidates</h2>{map_html}</div>
  <div class="card"><h2>7. Config suggestions</h2>{cfg_html}</div>
  <div class="card"><h2>8. Gap questions</h2>{q_html}</div>
  <div class="card"><h2>9. Readiness assessment</h2>{readiness}</div>
  <div class="card"><h2>10. Recommended next actions</h2>{actions_html}</div>
</div></body></html>"""

    out_path = Path(out_path)
    out_path.write_text(doc, encoding="utf-8")
    return out_path
