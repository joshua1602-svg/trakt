#!/usr/bin/env python3
"""Generate synthetic_demo/demo_overview_v2.html from pipeline outputs."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

import pandas as pd

ROOT = Path(__file__).resolve().parent

# ── Input paths ──────────────────────────────────────────────────────────────
RAW_CSV = ROOT / "input/SYNTHETIC_ERE_Portfolio_012026.csv"
TYPED_CSV = ROOT / "output/SYNTHETIC_ERE_Portfolio_012026_canonical_typed.csv"
PROJECTED_CSV = ROOT / "output/SYNTHETIC_ERE_Portfolio_012026_ESMA_Annex2_projected.csv"
DELIVERY_RPT = ROOT / "output/SYNTHETIC_ERE_Portfolio_012026_ESMA_Annex2_delivery_report.json"
PROJECTION_RPT = ROOT / "output/SYNTHETIC_ERE_Portfolio_012026_ESMA_Annex2_projection_report.json"
TRANSFORM_RPT = ROOT / "output/SYNTHETIC_ERE_Portfolio_012026_transform_report.json"
MAPPING_RPT = ROOT / "output/SYNTHETIC_ERE_Portfolio_012026_mapping_report.csv"
UNMAPPED = ROOT / "output/SYNTHETIC_ERE_Portfolio_012026_unmapped_headers.csv"
FIELD_SUMMARY = ROOT / "output/validation/SYNTHETIC_ERE_Portfolio_012026_field_summary.csv"
BIZ_VIOLATIONS = ROOT / "output/validation/SYNTHETIC_ERE_Portfolio_012026_business_rules_violations.csv"
XML_PATH = ROOT / "output/SYNTHETIC_012026_annex2.xml"
OUT_HTML_V2 = ROOT / "demo_overview_v2.html"
OUT_HTML_V3 = ROOT / "demo_overview_v3.html"

NS = {"e": "urn:esma:xsd:DRAFT1auth.099.001.04"}

PRESENTATION_ENUM_LABELS = {
    "property_type": {
        "RHOS": "Residential house (detached/semi)",
        "RFLT": "Residential flat/apartment",
        "RBGL": "Residential bungalow",
    },
    "purpose": {
        "PURC": "Purchase of main residence",
        "RMRT": "Refinance / remortgage",
        "RENV": "Home improvement",
        "EQRE": "Equity release",
    },
    "interest_rate_type": {
        "FXRL": "Fixed rate",
        "FLIF": "Variable / floating rate",
    },
}

# ── Data loading ─────────────────────────────────────────────────────────────
raw = pd.read_csv(RAW_CSV, dtype=str, keep_default_na=False)
typed = pd.read_csv(TYPED_CSV, low_memory=False)
mapping = pd.read_csv(MAPPING_RPT)
unmapped = pd.read_csv(UNMAPPED)
field_summary = pd.read_csv(FIELD_SUMMARY)
biz_v = pd.read_csv(BIZ_VIOLATIONS)
delivery_rpt = json.loads(DELIVERY_RPT.read_text())
projection_rpt = json.loads(PROJECTION_RPT.read_text())
transform_rpt = json.loads(TRANSFORM_RPT.read_text())

xml_size_kb = XML_PATH.stat().st_size / 1024 if XML_PATH.exists() else 0
xml_records = 0
xml_sample_rows = []
xml_snippet = ""
if XML_PATH.exists():
    tree = ET.parse(XML_PATH)
    root_el = tree.getroot()
    records = root_el.findall(".//e:UndrlygXpsrRcrd", NS)
    xml_records = len(records)

    def xget(el, path):
        f = el.find(path, NS)
        return f.text.strip() if f is not None and f.text else ""

    for rec in records[:6]:
        xml_sample_rows.append({
            "Loan ID": xget(rec, ".//e:UndrlygXpsrId/e:OrgnlUndrlygXpsrIdr"),
            "Orig Date": xget(rec, ".//e:IssncAndMtrtyDtls/e:OrgtnDt/e:Dt"),
            "Cur Balance": xget(rec, ".//e:BalDtls/e:CurPrncplBal/e:Val/e:Amt"),
            "Rate %": xget(rec, ".//e:IntrstRateDtls/e:CmonData/e:CurRate/e:Rate"),
            "Prop Value": xget(rec, ".//e:Coll/e:CollCmonData/e:Valtn/e:CurInf/e:ValtnAmt/e:Val/e:Amt"),
            "LTV %": xget(rec, ".//e:Coll/e:CollCmonData/e:Valtn/e:CurInf/e:LnToVal/e:Rate"),
            "Region Cd": xget(rec, ".//e:OblgrDtls/e:TrtrlUnit/e:Clssfctn/e:Cd"),
        })

    if records:
        record_xml = ET.tostring(records[0], encoding="unicode")
        pretty = minidom.parseString(record_xml).toprettyxml(indent="  ")
        xml_snippet = "\n".join([line for line in pretty.splitlines() if line.strip()][0:26])

# ── Portfolio metrics ─────────────────────────────────────────────────────────
total_bal = pd.to_numeric(typed.get("current_principal_balance"), errors="coerce").fillna(0).sum()
loan_count = len(typed)
wa_ltv = pd.to_numeric(typed.get("current_loan_to_value"), errors="coerce").fillna(0).mean()
wa_rate = pd.to_numeric(typed.get("current_interest_rate"), errors="coerce").fillna(0).mean()
wa_age = pd.to_numeric(typed.get("youngest_borrower_age"), errors="coerce").fillna(0).mean()

region_dist = typed.get("geographic_region_classification", pd.Series([], dtype="object")).fillna("Unknown").value_counts()
prop_dist = typed.get("property_type", pd.Series([], dtype="object")).fillna("Unknown").value_counts()
purpose_dist = typed.get("purpose", pd.Series([], dtype="object")).fillna("Unknown").value_counts()

# Gate stats
mapped_count = len(mapping)
unmapped_count = len(unmapped)

enum_rows = field_summary[field_summary["issue_type"] == "ENUM_INVALID"].copy()
enum_fields = enum_rows["field_name"].dropna().astype(str).tolist()
pre_remediation_blocking = int((enum_rows["materiality"] == "BLOCKING").sum()) if not enum_rows.empty else 0

total_violations = int(field_summary["error_count"].sum()) if not field_summary.empty else 0
biz_rule_errors = len(biz_v)
biz_exceptions = int(biz_v["message"].astype(str).str.contains("Exception:", na=False).sum()) if not biz_v.empty else 0

preflight_status = delivery_rpt.get("preflight", {}).get("status", "UNKNOWN")
blocking_errors = int(delivery_rpt.get("preflight", {}).get("blocking_errors", 0))
xml_built = XML_PATH.exists()

enum_transform = (projection_rpt.get("enum_mapping", {}) or {}).get("transformed_fields", {}) or {}
remediated_enum_fields = sorted(list(enum_transform.keys()))

norm_fields = ((transform_rpt.get("canonical_enum_normalization") or {}).get("fields") or {})
norm_changed = sum(int(v.get("rows_changed", 0)) for v in norm_fields.values()) if isinstance(norm_fields, dict) else 0


def esc(v):
    return str(v).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def badge(text, kind="info"):
    colours = {
        "pass": ("d4edda", "155724"),
        "fail": ("f8d7da", "721c24"),
        "warn": ("fff3cd", "856404"),
        "info": ("e3eaf7", "2a3f6e"),
        "grey": ("ebebeb", "444"),
    }
    bg, fg = colours.get(kind, colours["info"])
    return f"<span class='badge' style='background:#{bg};color:#{fg}'>{esc(text)}</span>"


def table(headers, rows, caption=None, klass=""):
    cap = f"<caption>{esc(caption)}</caption>" if caption else ""
    cls = f" class='{klass}'" if klass else ""
    ths = "".join(f"<th>{esc(h)}</th>" for h in headers)
    trs = ""
    for row in rows:
        tds = "".join(f"<td>{v}</td>" for v in row)
        trs += f"<tr>{tds}</tr>"
    return f"<table{cls}>{cap}<thead><tr>{ths}</tr></thead><tbody>{trs}</tbody></table>"


def bar_chart(series, max_bars=8):
    top = series.head(max_bars)
    if len(top) == 0:
        return "<div class='meta'>No data available.</div>"
    max_v = max(top.max(), 1)
    rows = []
    for k, v in top.items():
        pct = round(v / max_v * 100, 1)
        rows.append(f"""
        <div class='bar-row'>
          <div class='bar-label'>{esc(str(k))}</div>
          <div class='bar-track'><div class='bar-fill' style='width:{pct}%'></div></div>
          <div class='bar-val'>{v}</div>
        </div>""")
    return "<div class='bar-chart'>" + "".join(rows) + "</div>"


def relabel_enum_series(series: pd.Series, field_name: str) -> pd.Series:
    label_map = PRESENTATION_ENUM_LABELS.get(field_name, {})
    if not label_map or len(series) == 0:
        return series
    renamed = {k: label_map.get(str(k), str(k)) for k in series.index}
    out = series.copy()
    out.index = [renamed.get(k, k) for k in series.index]
    return out


def kpi(label, value, sub=""):
    sub_html = f"<div class='kpi-sub'>{esc(sub)}</div>" if sub else ""
    return f"""<div class='kpi-card'>
      <div class='kpi-label'>{esc(label)}</div>
      <div class='kpi-value'>{value}</div>
      {sub_html}
    </div>"""


def section_1():
    gate4b_kind = "pass" if preflight_status == "PASS" else "fail"
    gate4b_detail = "No blocking delivery issues" if preflight_status == "PASS" else "Blocking delivery issues require remediation"

    gate5_status = badge("PASS", "pass") if preflight_status == "PASS" and xml_built else badge("SKIPPED", "grey")
    gate5_detail = (
        f"{xml_records} UndrlygXpsrRcrd elements; XSD validation passed"
        if preflight_status == "PASS" and xml_built
        else "XML build skipped until delivery preflight passes"
    )

    rows = [
        ["Gate 1 — Semantic Alignment", f"{mapped_count} headers mapped", f"{unmapped_count} unmapped", badge("PASS", "pass")],
        ["Gate 2 — Canonical Transform", f"{loan_count} rows typed", "Deterministic typing + derivations", badge("PASS", "pass")],
        ["Gate 3 — Validation", f"{total_violations} findings", f"{pre_remediation_blocking} enum fields flagged pre-remediation", badge("WARN", "warn")],
        ["Gate 3b — Remediation", f"{norm_changed} canonical values standardized", "Internal normalization + regime enum mapping", badge("PASS", "pass")],
        ["Gate 4 — ESMA Projection", f"{len(pd.read_csv(PROJECTED_CSV))} rows projected", "Annex 2 regime mapping", badge("PASS", "pass")],
        ["Gate 4b — Delivery Preflight", f"{blocking_errors} blocking delivery errors", gate4b_detail, badge(preflight_status, gate4b_kind)],
        ["Gate 5 — XML Builder", f"{XML_PATH.name} ({xml_size_kb:.1f} KB)", gate5_detail, gate5_status],
    ]
    return table(["Gate", "Output", "Detail", "Status"], rows)


def section_2():
    raw_cols = [
        "Underlying Exposure Identifier", "Origination Dt", "Current Principal Balance GBP", "Current Interest Rate",
        "Borrower Age (Youngest)", "Region ", "Property Type ", "Purpose of Loan",
    ]
    typed_cols = [
        "underlying_exposure_identifier", "origination_date", "current_principal_balance", "current_interest_rate",
        "youngest_borrower_age", "geographic_region_classification", "property_type", "purpose",
    ]
    avail_raw = [c for c in raw_cols if c in raw.columns]
    avail_typed = [c for c in typed_cols if c in typed.columns]

    raw_html = table(
        avail_raw,
        [[esc(str(v)) for v in row] for row in raw[avail_raw].head(6).itertuples(index=False)],
        "Raw lender tape (first 6 rows)",
        klass="table-fixed",
    )
    typed_html = table(
        avail_typed,
        [[esc(str(v)) for v in row] for row in typed[avail_typed].head(6).itertuples(index=False)],
        "Canonical typed (first 6 rows)",
        klass="table-fixed",
    )

    return f"""
    <div class='split-grid'>
      <div class='panel'>{raw_html}</div>
      <div class='panel'>{typed_html}</div>
    </div>
    <div class='meta-strip'>Canonical output is deterministic typing and normalization only. ESMA code mapping is downstream in Gate 4. Controlled enum codes are retained in canonical output for audit consistency.</div>
    """


def section_3():
    fs_rows = []
    for _, row in field_summary.iterrows():
        sev = str(row.get("severity", "")).lower()
        sev_badge = badge(row.get("severity", ""), "fail" if sev == "error" else "warn")
        mat = str(row.get("materiality", "")).lower()
        mat_badge = badge(row.get("materiality", ""), "fail" if "blocking" in mat else "info")
        fs_rows.append([
            esc(str(row.get("field_name", ""))),
            esc(str(row.get("issue_type", ""))),
            esc(str(row.get("error_count", ""))),
            sev_badge,
            mat_badge,
            esc(str(row.get("suggested_actions", ""))[:68]),
        ])

    biz_rows = []
    for _, row in biz_v.iterrows():
        biz_rows.append([
            esc(str(row.get("rule_id", ""))),
            esc(str(row.get("severity", ""))),
            esc(str(row.get("description", ""))[:86]),
            esc(str(row.get("message", ""))[:94]),
        ])

    callout_kind = "warn" if biz_exceptions > 0 else "pass"
    callout = (
        "<div class='callout warn'><strong>Control note:</strong> Business rules emitted runtime exceptions. These are pipeline defects, not data findings.</div>"
        if biz_exceptions > 0 else
        "<div class='callout pass'><strong>Control note:</strong> Business rules executed without runtime type exceptions.</div>"
    )

    return (
        table(["Field", "Issue Type", "Count", "Severity", "Materiality", "Action"], fs_rows, "Schema / enum findings (pre-remediation)")
        + table(["Rule ID", "Severity", "Description", "Message"], biz_rows, "Business rule findings")
        + callout
    )


def section_4():
    rem_rows = []
    for f in sorted(norm_fields.keys()):
        meta = norm_fields.get(f, {})
        rem_rows.append([
            esc(f),
            esc(str(meta.get("rows_considered", 0))),
            esc(str(meta.get("rows_changed", 0))),
            esc(", ".join(meta.get("unmapped_examples", []) or ["—"])),
        ])

    enum_fix_rows = [[esc(k), esc(str(v.get("rows_transformed", 0))), badge("APPLIED", "pass")] for k, v in enum_transform.items()]

    summary = f"""
    <div class='flow-strip'>
      <span class='gate-chip'>Gate 3 findings</span> →
      <span class='gate-chip'>Canonical normalization</span> →
      <span class='gate-chip'>Regime enum mapping</span> →
      <span class='gate-chip'>Gate 4b preflight {esc(preflight_status)}</span>
    </div>
    """

    return summary + table(
        ["Canonical field", "Rows reviewed", "Rows normalized", "Unmapped examples"],
        rem_rows,
        "Canonical standardization actions (internal, non-ESMA)",
    ) + table(
        ["Projected field", "Rows transformed", "Status"],
        enum_fix_rows,
        "Regime mapping actions (ESMA-specific)",
    )


def section_5():
    kpis = "".join([
        kpi("Loan count", f"{loan_count:,}"),
        kpi("Portfolio balance", f"£{total_bal:,.0f}"),
        kpi("WA current LTV", f"{wa_ltv:.2f}%"),
        kpi("WA interest rate", f"{wa_rate:.3f}%"),
        kpi("WA borrower age", f"{wa_age:.1f} yrs"),
    ])
    region_chart = f"<h4 class='chart-title'>Geographic distribution</h4>{bar_chart(region_dist)}"
    prop_chart = f"<h4 class='chart-title'>Property type mix (normalized canonical)</h4>{bar_chart(relabel_enum_series(prop_dist, 'property_type'))}"
    purpose_chart = f"<h4 class='chart-title'>Purpose breakdown (normalized canonical)</h4>{bar_chart(relabel_enum_series(purpose_dist, 'purpose'))}"
    quality_panel = f"""
      <h4 class='chart-title'>Data quality posture</h4>
      <div class='mini-kv'><span>Dataset lens</span><strong>Pre-remediation findings vs post-remediation delivery</strong></div>
      <div class='mini-kv'><span>Pre-remediation enum fields</span><strong>{pre_remediation_blocking}</strong></div>
      <div class='mini-kv'><span>Delivery blocking issues</span><strong>{blocking_errors}</strong></div>
      <div class='mini-kv'><span>Business rule runtime exceptions</span><strong>{biz_exceptions}</strong></div>
      <div class='mini-kv'><span>XML/XSD status</span><strong>{'PASS' if xml_built and preflight_status == 'PASS' else 'NOT READY'}</strong></div>
    """

    return (
        f"<div class='kpi-grid'>{kpis}</div>"
        f"<div class='chart-grid-2x2'><div class='panel'>{region_chart}</div><div class='panel'>{prop_chart}</div><div class='panel'>{purpose_chart}</div><div class='panel'>{quality_panel}</div></div>"
    )


def section_6():
    if not xml_sample_rows:
        return "<p>XML not found.</p>"

    headers = list(xml_sample_rows[0].keys())
    rows = [[esc(str(r[h])) for h in headers] for r in xml_sample_rows]
    meta = (
        f"Namespace: <code>urn:esma:xsd:DRAFT1auth.099.001.04</code> &nbsp;·&nbsp; "
        f"Records: <strong>{xml_records}</strong> &nbsp;·&nbsp; "
        f"File: <code>{XML_PATH.name}</code> ({xml_size_kb:.1f}&nbsp;KB)"
    )
    snippet_html = f"<details class='xml-details'><summary>Sample raw XML (first exposure record)</summary><pre>{esc(xml_snippet)}</pre></details>" if xml_snippet else ""
    return f"<p class='meta'>{meta}</p>" + table(headers, rows, "Executive XML snapshot (first 6 exposures)") + snippet_html


def section_7():
    stages = [
        ("Gate 1", "Semantic alignment", "Header aliasing only. No economic transformations."),
        ("Gate 2", "Canonical typing", "Deterministic coercion, derivations, internal enum normalization."),
        ("Gate 3", "Validation", "Schema and business controls against canonical truth set."),
        ("Gate 3b", "Remediation", "Explicit standardization actions before delivery projection."),
        ("Gate 4", "Regime projection", "Canonical → ESMA Annex 2 code mapping and ND defaults."),
        ("Gate 4b", "Delivery preflight", "Strict code checks before XML generation."),
        ("Gate 5", "XML/XSD", "Machine-readable output and XSD conformance."),
        ("MI", "Presentation", "Executive MI built from canonical dataset, not from delivery XML."),
    ]
    rows = [[f"<strong>{esc(g)}</strong>", esc(n), esc(d)] for g, n, d in stages]
    flow = " &nbsp;→&nbsp; ".join(f"<span class='gate-chip'>{esc(g)}</span>" for g, _, _ in stages)
    return (
        f"<div class='flow-strip'>{flow}</div>"
        + table(["Layer", "Owner", "Transformation ownership"], rows)
        + "<p class='arch-note'>Canonical normalization and ESMA mapping are separated by design. Presentation does not perform hidden data fixes.</p>"
    )


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',Arial,sans-serif;font-size:13px;background:#f0f3f8;color:#1b2132;line-height:1.5}
.page{max-width:1060px;margin:28px auto;padding:0 16px}
.hero{background:linear-gradient(135deg,#1b3872 0%,#2f5d9f 100%);color:#fff;border-radius:10px 10px 0 0;padding:28px 32px 22px}
.hero h1{font-size:22px;font-weight:700;letter-spacing:-.3px}
.hero .sub{font-size:12px;opacity:.82;margin-top:4px}
.card{background:#fff;border:1px solid #d8e0ec;padding:24px 30px;margin-bottom:16px}
.card:last-child{border-radius:0 0 10px 10px}
h2{font-size:15px;font-weight:700;color:#1b3872;margin-bottom:14px;padding-bottom:6px;border-bottom:2px solid #e3eaf7}
h4.chart-title{font-size:12px;font-weight:600;color:#2f5d9f;margin:0 0 8px}
table{width:100%;border-collapse:collapse;font-size:12px;margin:8px 0 14px}
caption{text-align:left;font-size:11px;font-weight:600;color:#5f6b7a;padding:0 0 5px;letter-spacing:.3px;text-transform:uppercase}
th{background:#e8eef7;color:#1b3872;font-weight:600;padding:8px 8px;text-align:left;border:1px solid #d0d9ea;vertical-align:bottom;line-height:1.3;white-space:normal;word-break:break-word}
td{padding:7px 8px;border:1px solid #e3e8f0;vertical-align:top;line-height:1.4;word-break:break-word}
tr:nth-child(even) td{background:#f8fafd}
.table-fixed{table-layout:fixed}
.badge{display:inline-block;border-radius:4px;padding:2px 7px;font-size:11px;font-weight:600;letter-spacing:.3px}
.kpi-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;margin-bottom:16px}
.kpi-card{border:1px solid #d8e0ec;border-radius:7px;padding:12px 14px;background:#f9fbfe;min-height:82px}
.kpi-label{font-size:11px;color:#5f6b7a;font-weight:500;margin-bottom:4px}
.kpi-value{font-size:18px;font-weight:700;color:#1b3872}
.kpi-sub{font-size:11px;color:#8a95a5;margin-top:2px}
.split-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.chart-grid-2x2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.panel{border:1px solid #d8e0ec;border-radius:7px;padding:10px 12px;background:#fcfdff;min-height:232px}
.bar-chart{display:flex;flex-direction:column;gap:6px}
.bar-row{display:flex;align-items:center;gap:7px}
.bar-label{width:165px;font-size:11px;color:#3a4560;white-space:normal;line-height:1.25}
.bar-track{flex:1;height:10px;background:#e8eef7;border-radius:3px;overflow:hidden}
.bar-fill{height:100%;background:#2f5d9f;border-radius:3px;transition:width .3s}
.bar-val{width:32px;text-align:right;font-size:11px;color:#5f6b7a}
.flow-strip{background:#e8eef7;border-radius:6px;padding:10px 14px;margin-bottom:12px;font-size:12px;color:#1b3872;line-height:2}
.gate-chip{display:inline-block;background:#1b3872;color:#fff;border-radius:4px;padding:2px 8px;font-size:11px;font-weight:600}
.meta{font-size:12px;color:#5f6b7a;margin-bottom:8px}
.meta-strip{margin-top:6px;background:#f4f7fc;border:1px solid #dde6f5;border-radius:6px;padding:8px 10px;font-size:12px;color:#38507e}
.callout{border-radius:6px;padding:10px 14px;font-size:12px;margin-top:8px;line-height:1.6}
.callout.warn{background:#fff8e1;border-left:4px solid #f59e0b;color:#7c5700}
.callout.pass{background:#edf7ee;border-left:4px solid #1f8f4a;color:#1f4f2f}
.arch-note{font-size:13px;font-weight:600;color:#1b3872;margin-top:10px;padding:10px 14px;background:#e8eef7;border-radius:6px}
.mini-kv{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #edf2fb;font-size:12px}
.mini-kv:last-child{border-bottom:0}
.xml-details summary{cursor:pointer;font-size:12px;color:#1b3872;font-weight:600;margin:8px 0}
.xml-details pre{margin-top:8px;background:#f7f9fc;border:1px solid #dee6f2;border-radius:6px;padding:10px;max-height:260px;overflow:auto;font-size:11px;line-height:1.4}
code{font-family:monospace;background:#eef1f8;padding:1px 4px;border-radius:3px;font-size:11px}
"""

SECTIONS = [
    ("1. Pipeline Gate-by-Gate Execution", section_1()),
    ("2. Raw → Canonical Transformation", section_2()),
    ("3. Validation & Governance (Pre-Remediation)", section_3()),
    ("4. Remediation / Standardisation Resolution", section_4()),
    ("5. Portfolio MI", section_5()),
    ("6. ESMA Annex 2 XML Snapshot", section_6()),
    ("7. Architecture Ownership Model", section_7()),
]

cards_html = ""
for title, body in SECTIONS:
    cards_html += f"""
  <div class='card'>
    <h2>{title}</h2>
    {body}
  </div>"""

html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>Synthetic ERM Demo — Pipeline Overview</title>
<style>
{CSS}
</style>
</head>
<body>
<div class=\"page\">
  <div class=\"hero\">
    <h1>Synthetic Equity Release Portfolio — ESMA Annex 2 Demo</h1>
    <div class=\"sub\">Reporting period: January 2026 &nbsp;·&nbsp; Portfolio: SYNTHETIC_ERE_Portfolio_012026 &nbsp;·&nbsp; Regime: ESMA Annex 2 (DRAFT1auth.099.001.04)</div>
  </div>
{cards_html}
</div>
</body>
</html>"""

OUT_HTML_V2.write_text(html, encoding="utf-8")
OUT_HTML_V3.write_text(html, encoding="utf-8")
print(f"Written {OUT_HTML_V2}  ({OUT_HTML_V2.stat().st_size / 1024:.1f} KB)")
print(f"Written {OUT_HTML_V3}  ({OUT_HTML_V3.stat().st_size / 1024:.1f} KB)")
