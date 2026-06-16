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


def _domain_status_badge(status: str) -> str:
    cls = {"covered": "b-ok", "partially_covered": "b-warn",
           "missing": "b-block", "out_of_scope": "b-info"}.get(status, "b-info")
    return f'<span class="badge {cls}">{_esc(status)}</span>'


# ---------------------------------------------------------------------------
# Target-contract-first gates (28a / 28b / 28c) — primary managed-service flow
# ---------------------------------------------------------------------------

# Coverage status -> badge class.
_COV_BADGE = {
    "source_mapped": "b-ok", "source_mapped_with_alternatives": "b-ok",
    "derived": "b-info", "configured_static": "b-info", "defaulted": "b-info",
    "defaulted_ND": "b-info", "defaulted_value": "b-info", "not_applicable": "b-info",
    "deferred": "b-warn", "pending_regime_rule": "b-warn",
    "needs_confirmation": "b-warn", "optional_for_mi": "b-warn",
    "missing_required": "b-block",
}

# Render order for the coverage matrix: blocking first, then by "settledness".
_COV_ORDER = ["missing_required", "pending_regime_rule", "needs_confirmation",
              "source_mapped_with_alternatives", "deferred", "optional_for_mi",
              "source_mapped", "derived", "configured_static",
              "defaulted", "defaulted_value", "defaulted_ND", "not_applicable"]


def _cov_badge(status: str) -> str:
    return f'<span class="badge {_COV_BADGE.get(status, "b-info")}">{_esc(status)}</span>'


def _as_bool(v) -> bool:
    """Coerce a JSON bool OR a CSV string ('True'/'False'/'') to a real bool."""
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes")


def _candidate_dirs(project_dir: Path, output_root: Path | None) -> list:
    """All plausible run locations for the numbered artefacts (de-duplicated)."""
    dirs = [project_dir, project_dir / "output"]
    if output_root is not None:
        dirs += [output_root, output_root.parent]
    seen, out = set(), []
    for d in dirs:
        d = Path(d)
        if str(d) not in seen:
            seen.add(str(d))
            out.append(d)
    return out


def _find_artifact(project_dir: Path, output_root: Path | None, basename: str):
    for d in _candidate_dirs(project_dir, output_root):
        p = d / basename
        if p.exists():
            return p
    return None


def _load_csv_rows(path: Path) -> list:
    import csv
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _derive_summaries(tf: dict) -> None:
    """Fill missing summaries from rows (e.g. when only the CSV was found)."""
    cov = tf.get("coverage")
    if cov is not None and not cov.get("summary"):
        rows = cov.get("rows", [])
        sc: dict = {}
        for r in rows:
            st = r.get("coverage_status", "")
            sc[st] = sc.get(st, 0) + 1
        cov["summary"] = {
            "target_fields_total": len(rows),
            "coverage_status_counts": sc,
            "source_mapped_fields": sc.get("source_mapped", 0) + sc.get("source_mapped_with_alternatives", 0),
            "derived_config_defaulted_fields": (sc.get("derived", 0) + sc.get("configured_static", 0)
                                                + sc.get("defaulted", 0) + sc.get("defaulted_ND", 0)),
            "missing_required_fields": sc.get("missing_required", 0),
            "needs_confirmation_fields": sc.get("needs_confirmation", 0),
            "not_applicable_fields": sc.get("not_applicable", 0),
            "optional_for_mi_fields": sc.get("optional_for_mi", 0),
        }
    res = tf.get("residual")
    if res is not None and not res.get("summary"):
        rows = res.get("rows", [])
        cc: dict = {}
        for r in rows:
            cl = r.get("residual_class", "")
            cc[cl] = cc.get(cl, 0) + 1
        res["summary"] = {
            "residual_source_columns_total": len(rows),
            "suppressed_from_main_queue": sum(1 for r in rows if _as_bool(r.get("suppressed_from_main_queue"))),
            "operator_visible": sum(1 for r in rows if _as_bool(r.get("operator_visible"))),
            "residual_class_counts": cc,
        }
    dec = tf.get("decision")
    if dec is not None and not dec.get("summary"):
        rows = dec.get("rows", [])
        tc: dict = {}
        for r in rows:
            dt = r.get("decision_type", "")
            tc[dt] = tc.get(dt, 0) + 1
        dec["summary"] = {
            "human_decision_rows_total": len(rows),
            "blocking_decisions": sum(1 for r in rows if _as_bool(r.get("blocking"))),
            "decision_type_counts": tc,
        }


def _load_target_first_artifacts(project_dir: Path, output_root: Path | None = None) -> dict:
    """Load the 28a/28b/28c (+33/29a) artefacts, searching all plausible run dirs.

    Prefers the JSON artefact (rich, native types + summary) and falls back to the
    CSV when only that is present. Searched locations, in order:
        project_dir / filename
        project_dir / output / filename
        output_root / filename
        parent(output_root) / filename
    """
    project_dir = Path(project_dir)
    output_root = Path(output_root) if output_root else None
    specs = {
        "coverage": "28a_target_coverage_matrix",
        "residual": "28b_source_residual_register",
        "decision": "28c_human_decision_queue",
        "queue33": "33_mapping_review_queue",
    }
    tf: dict = {}
    for key, base in specs.items():
        json_p = _find_artifact(project_dir, output_root, base + ".json")
        data = _load_json(json_p) if json_p else None
        if data is None:
            csv_p = _find_artifact(project_dir, output_root, base + ".csv")
            if csv_p:
                rows = _load_csv_rows(csv_p)
                data = {"summary": {}, ("items" if key == "queue33" else "rows"): rows}
        tf[key] = data
    fc = _find_artifact(project_dir, output_root, "29a_column_evidence_file_coverage.json")
    tf["file_coverage"] = _load_json(fc) if fc else None
    dlog = _find_artifact(project_dir, output_root,
                          "35_target_first_decision_application_log.json")
    tf["decision_log"] = _load_json(dlog) if dlog else None
    adv = _find_artifact(project_dir, output_root,
                         "36_target_first_llm_recommendations.json")
    tf["llm_advisor"] = _load_json(adv) if adv else None
    advu = _find_artifact(project_dir, output_root,
                          "36_target_first_llm_usage_summary.json")
    tf["llm_advisor_usage"] = _load_json(advu) if advu else None
    cfgval = _find_artifact(project_dir, output_root,
                            "42_annex2_config_validation.json")
    tf["config_validation"] = _load_json(cfgval) if cfgval else None
    recon = _find_artifact(project_dir, output_root,
                           "43_annex2_field_universe_reconciliation.json")
    tf["field_universe"] = _load_json(recon) if recon else None
    nd = _find_artifact(project_dir, output_root,
                        "44_annex2_nd_eligibility_reconciliation.json")
    tf["nd_eligibility"] = _load_json(nd) if nd else None
    ca = _find_artifact(project_dir, output_root,
                        "45_annex2_config_alignment_review.json")
    tf["config_alignment"] = _load_json(ca) if ca else None
    ec = _find_artifact(project_dir, output_root,
                        "46_annex2_enum_coverage_reconciliation.json")
    tf["enum_coverage"] = _load_json(ec) if ec else None
    sm = _find_artifact(project_dir, output_root,
                        "47_annex2_semantic_mapping_reconciliation.json")
    tf["semantic_mapping"] = _load_json(sm) if sm else None
    mp = _find_artifact(project_dir, output_root,
                        "48_annex2_mapping_correction_proposals.json")
    tf["mapping_proposals"] = _load_json(mp) if mp else None
    _derive_summaries(tf)
    return tf


# Backwards-compatible alias (single-dir lookup) used by older call sites.
def _load_target_first(project_dir: Path) -> dict:
    return _load_target_first_artifacts(project_dir)


def _counts_table(title: str, counts: dict) -> str:
    if not counts:
        return ""
    rows = [[_esc(k), _esc(v)] for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]
    return f'<h4 class="chart-title">{_esc(title)}</h4>' + _table(["Status", "Count"], rows)


def _coverage_full_table(rows: list) -> str:
    """The full one-row-per-target-field matrix (audit/detail)."""
    def sort_key(r):
        st = r.get("coverage_status", "")
        idx = _COV_ORDER.index(st) if st in _COV_ORDER else len(_COV_ORDER)
        return (0 if _as_bool(r.get("blocking")) else 1, idx, r.get("target_field", ""))

    body = []
    for r in sorted(rows, key=sort_key):
        sel_file = r.get("selected_source_file", "")
        sel = r.get("selected_source_column", "") or "—"
        if sel_file:
            sel = f"{sel}<br><small>{_esc(sel_file)}</small>"
        if r.get("selected_source_confidence") not in ("", None):
            sel += f'<br><small>conf {_esc(r.get("selected_source_confidence"))}</small>'
        rule = " · ".join(x for x in [
            r.get("coverage_basis", ""),
            (f"default: {r.get('default_rule')}" if r.get("default_rule") else ""),
            (f"derive: {r.get('derivation_rule')}" if r.get("derivation_rule") else ""),
            (f"config: {r.get('configured_value_source')}" if r.get("configured_value_source") else ""),
        ] if x)
        if r.get("operator_question"):
            rule += f'<br><small>Q: {_esc(r.get("operator_question"))}</small>'
        body.append([
            _esc(r.get("target_field", "")), _esc(r.get("target_domain", "")),
            _esc(r.get("required_status", "")), _cov_badge(r.get("coverage_status", "")),
            sel, _esc(r.get("alternative_source_candidates", "") or "—"),
            rule or "—",
        ])
    return _table(["Target field", "Domain", "Required", "Coverage", "Selected source",
                   "Alternatives", "Coverage basis / rule / question"], body)


def _is_annex2(tf: dict) -> bool:
    return ((tf.get("coverage") or {}).get("target_contract_id", "") == "esma_annex_2")


def _annex2_executive_html(tf: dict) -> str:
    """ESMA Annex 2 executive lines (Gate 1) — target contract + two config layers."""
    if not _is_annex2(tf):
        return ""
    cov = tf.get("coverage") or {}
    cov_sum = cov.get("summary", {}) or {}
    dec_sum = (tf.get("decision") or {}).get("summary", {}) or {}
    cfg = tf.get("config_validation") or {}
    cfg_sum = cfg.get("summary", {}) or {}
    n_block = dec_sum.get("blocking_decisions", 0)
    n_total = dec_sum.get("human_decision_rows_total", 0)
    rows = [
        ["Target contract", _esc(cov.get("target_contract_id", "ESMA_Annex2"))],
        ["Regime config", _esc(cfg.get("regime_config_source", "")
                               or cov.get("target_contract_source", ""))],
        ["Asset config", _esc(cfg.get("asset_config_source", ""))],
        ["Annex 2 fields", _esc(cov_sum.get("target_fields_total", 0))],
        ["Source mapped / derived",
         _esc(cov_sum.get("source_mapped_fields", 0) + cov_sum.get("derived_fields", 0))],
        ["Configured / static / defaulted",
         _esc(cov_sum.get("configured_static_fields", 0)
              + cov_sum.get("defaulted_value_fields", 0)
              + cov_sum.get("defaulted_nd_fields", 0))],
        ["ND / defaulted", _esc(cov_sum.get("defaulted_nd_fields", 0))],
        ["Blocking decisions", _esc(n_block)],
        ["Non-blocking confirmations", _esc(max(0, n_total - n_block))],
        ["Invalid asset defaults (surfaced)",
         _esc(cfg_sum.get("invalid_default_not_allowed", 0))],
    ]
    return ('<h4 class="chart-title">ESMA Annex 2 delivery — target contract &amp; '
            'config layers</h4>'
            '<p class="meta">Annex 2 delivery runs through the target-first operator '
            'workflow with two config layers: the ESMA regime rules and the ERM asset '
            'defaults. Asset defaults are validated against the regime envelope '
            '(see <code>42_annex2_config_validation.csv</code>).</p>'
            + _table(["Item", "Value"], rows)
            + _annex2_universe_html(tf))


def _annex2_universe_html(tf: dict) -> str:
    """Annex 2 field-universe reconciliation (43) — completeness of the universe."""
    fu = tf.get("field_universe")
    if not fu:
        return ""
    s = fu.get("summary", {}) or {}
    missing = int(s.get("missing_from_28a_count", 0))
    pending = int(s.get("missing_from_regime_rules_count", 0))
    rows = [
        ["Authoritative Annex 2 fields (workbook)", _esc(s.get("authoritative_field_count", 0))],
        ["Registry mapped", _esc(s.get("registry_mapped_count", 0))],
        ["Registry gaps", _esc(s.get("registry_gap_count", 0))],
        ["Present in regime rules", _esc(s.get("regime_rule_count", 0))],
        ["Present in 28a coverage", _esc(s.get("coverage_field_count", 0))],
        ["Deferred / pending reconciliation", _esc(s.get("deferred_field_count", 0))],
        ["Pending regime rule (config gap)", _esc(pending)],
        ["Active phantom deferred fields", _esc(s.get("not_in_authoritative_universe_count", 0))],
        ["Missing from 28a", _esc(missing)],
        ["Deliverable (rule + coverage)", _esc(s.get("deliverable_field_count", 0))],
    ]
    if missing:
        note = (f'<div class="callout block">{missing} authoritative Annex 2 code(s) '
                "are MISSING from 28a — the target universe is not fully loaded.</div>")
    elif pending:
        note = (f'<div class="callout warn">{pending} Annex 2 code(s) are in the '
                "authoritative universe but have no full regime rule yet "
                "(<code>pending_regime_rule</code>) — regime config is incomplete "
                "relative to the workbook universe.</div>")
    else:
        note = ('<div class="callout pass">28a covers the full authoritative Annex 2 '
                "universe with full regime rules.</div>")
    warn_html = ""
    for w in (fu.get("warnings") or []):
        warn_html += f'<p class="meta">⚠ {_esc(w)}</p>'
    return ('<h4 class="chart-title">Annex 2 field universe reconciliation</h4>'
            + note + warn_html + _table(["Item", "Value"], rows)
            + _annex2_nd_eligibility_html(tf)
            + _annex2_config_alignment_html(tf)
            + _annex2_enum_coverage_html(tf)
            + _annex2_semantic_mapping_html(tf)
            + _annex2_mapping_proposals_html(tf))


def _annex2_mapping_proposals_html(tf: dict) -> str:
    """Mapping-correction proposals (48): proposed source/ND/mechanics fixes."""
    mp = tf.get("mapping_proposals")
    if not mp:
        return ""
    s = mp.get("summary", {}) or {}
    total = int(s.get("proposal_rows_total", 0))
    rows = [
        ["Proposed corrections", _esc(total)],
        ["Re-point source only", _esc(s.get("re_point_source_only", 0))],
        ["Need rule-mechanics changes", _esc(s.get("needs_rule_mechanics_changes", 0))],
        ["Need mechanics review", _esc(s.get("needs_mechanics_review", 0))],
    ]
    note = ('<div class="callout warn">' + _esc(total) +
            " proposed Annex 2 mapping correction(s) await manual approval "
            "(report-only, nothing applied) — see "
            "<code>48_annex2_mapping_correction_proposals.csv</code>.</div>"
            if total else
            '<div class="callout pass">No mapping corrections proposed.</div>')
    return ('<h4 class="chart-title">Annex 2 mapping-correction proposals</h4>'
            + note + _table(["Item", "Value"], rows))


def _annex2_semantic_mapping_html(tf: dict) -> str:
    """Semantic-mapping reconciliation (47): regime source vs workbook field."""
    sm = tf.get("semantic_mapping")
    if not sm:
        return ""
    s = sm.get("summary", {}) or {}
    mism = int(s.get("semantic_mismatch", 0))
    rows = [
        ["Ruled codes checked", _esc(s.get("semantic_rows_total", 0))],
        ["Source matches workbook field", _esc(s.get("aligned", 0))],
        ["Suspected code↔field mismap (review)", _esc(mism)],
    ]
    if mism:
        note = ('<div class="callout warn">' + _esc(mism) +
                " Annex 2 regime rule(s) map a source field that does not match the "
                "workbook field for that code — manual mapping review required; see "
                "<code>47_annex2_semantic_mapping_reconciliation.csv</code>.</div>")
    else:
        note = ('<div class="callout pass">Every regime rule maps the workbook field '
                "for its code.</div>")
    return ('<h4 class="chart-title">Annex 2 semantic-mapping reconciliation</h4>'
            + note + _table(["Item", "Value"], rows))


def _annex2_enum_coverage_html(tf: dict) -> str:
    """Enum-coverage reconciliation (46): regime enum_map vs workbook codes."""
    ec = tf.get("enum_coverage")
    if not ec:
        return ""
    s = ec.get("summary", {}) or {}
    outside = int(s.get("targets_outside_workbook", 0))
    semantic = int(s.get("semantic_mismatch", 0))
    rows = [
        ["Constrained to workbook codes", _esc(s.get("constrained_within_workbook", 0))],
        ["Unconstrained (no enum_map)", _esc(s.get("unconstrained_no_enum_map", 0))],
        ["Targets outside workbook (risk)", _esc(outside)],
        ["Semantic mismatch (mapping review)", _esc(semantic)],
        ["No regime rule yet", _esc(s.get("no_regime_rule", 0))],
    ]
    if outside or semantic:
        note = ('<div class="callout warn">' + _esc(outside + semantic) +
                " Annex 2 {LIST} field(s) need enum review (targets outside workbook "
                "or source/field mismatch) — see "
                "<code>46_annex2_enum_coverage_reconciliation.csv</code>.</div>")
    else:
        note = ('<div class="callout pass">All constrained enum maps are within the '
                "workbook's allowed codes; no enum values exceed ESMA.</div>")
    return ('<h4 class="chart-title">Annex 2 enum-coverage reconciliation</h4>'
            + note + _table(["Item", "Value"], rows))


def _annex2_config_alignment_html(tf: dict) -> str:
    """Config-alignment review (45): actions taken + manual-review items."""
    ca = tf.get("config_alignment")
    if not ca:
        return ""
    s = ca.get("summary", {}) or {}
    manual = int(s.get("requires_manual_review_count", 0))
    rows = [
        ["Tightened to workbook (compliance fixes)", _esc(s.get("tightened_to_workbook", 0))],
        ["Registry mappings added", _esc(s.get("registry_mapping_added", 0))],
        ["Phantom deferred removed", _esc(s.get("phantom_deferred_removed", 0))],
        ["Left stricter by policy", _esc(s.get("left_stricter_by_policy", 0))],
        ["Divergent (manual review)", _esc(s.get("divergent_requires_review", 0))],
        ["Asset-default conflicts", _esc(s.get("asset_default_conflict", 0))],
        ["Items requiring manual review", _esc(manual)],
    ]
    note = (f'<div class="callout warn">{manual} Annex 2 alignment item(s) require manual '
            "review — see <code>45_annex2_config_alignment_review.csv</code>.</div>"
            if manual else
            '<div class="callout pass">All Annex 2 config-alignment actions resolved; '
            "no items require manual review.</div>")
    return ('<h4 class="chart-title">Annex 2 config-alignment review</h4>'
            + note + _table(["Item", "Value"], rows))


def _annex2_nd_eligibility_html(tf: dict) -> str:
    """ND-eligibility reconciliation (44): regime nd_allowed vs workbook."""
    nd = tf.get("nd_eligibility")
    if not nd:
        return ""
    s = nd.get("summary", {}) or {}
    risk = int(s.get("nd_compliance_risk_count", 0))
    rows = [
        ["Match", _esc(s.get("match", 0))],
        ["Regime stricter than workbook", _esc(s.get("regime_stricter", 0))],
        ["Regime broader than workbook (risk)", _esc(s.get("regime_broader", 0))],
        ["Divergent ND sets", _esc(s.get("divergent", 0))],
    ]
    note = ('<div class="callout warn">' + _esc(risk) +
            " Annex 2 code(s) where the regime ND policy diverges from the "
            "authoritative workbook ND eligibility — see "
            "<code>44_annex2_nd_eligibility_reconciliation.csv</code>.</div>"
            ) if risk else (
            '<div class="callout pass">Regime ND policy is within the workbook ND '
            "eligibility for all compared codes.</div>")
    return ('<h4 class="chart-title">Annex 2 ND-eligibility reconciliation</h4>'
            + note + _table(["Item", "Value"], rows))


def _gate3_summary_html(tf: dict) -> str:
    """Gate 3 — target coverage SUMMARY (counts + grouped table + missing-first).

    The full 72-row matrix is collapsed into a <details> block so the main body
    stays decision-led rather than dominated by the per-field matrix.
    """
    cov = tf.get("coverage")
    if not cov:
        return ('<div class="callout warn">Target coverage matrix '
                '(<code>28a_target_coverage_matrix.csv</code>) not available for this run. '
                'Run with <code>--enable-mapping-review</code>.</div>')
    summary = cov.get("summary", {})
    rows = cov.get("rows", [])
    intro = (
        f'<p class="meta">Target contract: '
        f'<span class="badge b-info">{_esc(cov.get("target_contract_id",""))}</span> '
        f'&nbsp;·&nbsp; {summary.get("target_fields_total", len(rows))} target fields. '
        "Coverage is target-field-led. The full per-field matrix is in the collapsible "
        "block below.</p>")
    counts_html = _counts_table("Coverage status counts",
                                summary.get("coverage_status_counts", {}))

    # Short table grouped by coverage_status x target_domain.
    grouped: dict = {}
    for r in rows:
        key = (r.get("coverage_status", ""), r.get("target_domain", ""))
        grouped[key] = grouped.get(key, 0) + 1

    def gkey(item):
        (st, dom), _c = item
        idx = _COV_ORDER.index(st) if st in _COV_ORDER else len(_COV_ORDER)
        return (idx, dom)

    grp_rows = [[_cov_badge(st), _esc(dom), _esc(c)]
                for (st, dom), c in sorted(grouped.items(), key=gkey)]
    grouped_html = ('<h4 class="chart-title">Coverage by status &amp; domain</h4>'
                    + _table(["Coverage status", "Target domain", "Count"], grp_rows))

    # Annex 2: also break coverage down by ESMA field family (RREL / RREC).
    family_html = ""
    if cov.get("target_contract_id", "") == "esma_annex_2":
        fam: dict = {}
        for r in rows:
            code = str(r.get("esma_code", "") or r.get("target_field", ""))
            family = "RREL" if code.startswith("RREL") else (
                "RREC" if code.startswith("RREC") else "other")
            key = (family, r.get("coverage_status", ""))
            fam[key] = fam.get(key, 0) + 1
        fam_rows = [[_esc(family), _cov_badge(st), _esc(c)]
                    for (family, st), c in sorted(fam.items())]
        family_html = ('<h4 class="chart-title">Annex 2 coverage by field family '
                       '(RREL / RREC)</h4>'
                       + _table(["Field family", "Coverage status", "Count"], fam_rows))

    # Required-fields summary.
    req = [r for r in rows if r.get("required_status") in ("required", "mandatory")]
    req_missing = sum(1 for r in req if r.get("coverage_status") == "missing_required")
    req_confirm = sum(1 for r in req if r.get("coverage_status") == "needs_confirmation")
    req_covered = len(req) - req_missing - req_confirm
    req_html = ('<h4 class="chart-title">Required target fields</h4>'
                + _table(["Required total", "Covered by source/rule", "Needs confirmation",
                          "Missing / blocking"],
                         [[_esc(len(req)), _esc(req_covered), _esc(req_confirm),
                           _esc(req_missing)]]))

    # Missing / blocking first if present.
    miss_rows = [r for r in rows if r.get("coverage_status") == "missing_required"]
    if miss_rows:
        ml = "".join(f"<li><code>{_esc(r.get('target_field',''))}</code> "
                     f"({_esc(r.get('target_domain',''))}) — "
                     f"{_esc(r.get('decision_reason',''))}</li>" for r in miss_rows)
        miss_html = (f'<div class="callout block">{len(miss_rows)} missing / blocking '
                     f"required target field(s) — see Gate 4:<ul>{ml}</ul></div>")
    else:
        miss_html = ('<div class="callout pass">No missing / blocking required target '
                     "fields — all are covered by a source mapping, derivation, default "
                     "or ND rule.</div>")

    full = (f'<details><summary class="meta"><strong>Full target coverage matrix '
            f"(audit/detail)</strong> — {len(rows)} target fields</summary>"
            f"{_coverage_full_table(rows)}</details>")
    return (intro + miss_html + counts_html + grouped_html + family_html
            + req_html + full)


def _decision_application_html(tf: dict) -> str:
    """Minimal applied-decisions summary (35 log) for the operator pack."""
    dlog = tf.get("decision_log")
    if not dlog:
        return ""
    summary = dlog.get("summary", {}) or {}
    rows = [
        ["Decisions supplied", summary.get("decisions_supplied", 0)],
        ["Applied", summary.get("applied", 0)],
        ["Deferred", summary.get("deferred", 0)],
        ["Requires operator review", summary.get("requires_operator_review", 0)],
        ["Invalid / not found", summary.get("invalid", 0)],
    ]
    return ('<h4 class="chart-title">Applied target-first decisions</h4>'
            '<p class="meta">Approved Gate 4 decisions applied this run from '
            f'<code>{_esc(dlog.get("decisions_source",""))}</code>.</p>'
            + _table(["Metric", "Value"], [[_esc(a), _esc(b)] for a, b in rows]))


def _gate5_mi_handoff_status_html(tf: dict, n_block: int, n_decisions: int) -> str:
    """Gate 5 — handoff/readiness status derived from 28c (never legacy BLOCKED).

    Wording reflects the target contract: ESMA Annex 2 delivery for the regulatory
    contract, MI handoff otherwise.
    """
    label = "Annex 2 delivery" if _is_annex2(tf) else "MI handoff"
    if tf.get("decision") is None:
        return (f'<div class="callout warn">{label} status unavailable — the compact '
                "decision queue (28c) was not found for this run.</div>")
    if n_block > 0:
        return (f'<div class="callout block"><strong>{label}: BLOCKED.</strong> '
                f"{n_block} blocking target decision(s) must be resolved (Gate 4).</div>")
    if n_decisions > 0:
        return (f'<div class="callout warn"><strong>{label}: NEEDS CONFIRMATION.</strong> '
                f"No blocking decisions; {n_decisions} non-blocking confirmation(s) remain "
                "(Gate 4). The pack can proceed once confirmations are acknowledged.</div>")
    return (f'<div class="callout pass"><strong>{label}: READY.</strong> No blocking '
            "decisions and no outstanding required confirmations.</div>")


def _gate4_decision_queue_html(tf: dict) -> str:
    """Gate 4 — compact human decision queue (28c) ONLY, grouped by blocking/type."""
    dec = tf.get("decision")
    if not dec:
        return ('<div class="callout warn">Compact human decision queue '
                '(<code>28c_human_decision_queue.csv</code>) not available for this run.</div>')
    summary = dec.get("summary", {})
    rows = dec.get("rows", [])
    blocking = [r for r in rows if _as_bool(r.get("blocking"))]
    nonblocking = [r for r in rows if not _as_bool(r.get("blocking"))]
    n_block = summary.get("blocking_decisions", len(blocking))

    if not rows:
        head = ('<div class="callout pass">No operator decisions required — target '
                "coverage is complete for this pack.</div>")
    elif n_block == 1:
        b = blocking[0]
        head = ('<div class="callout block"><strong>Only ONE blocking decision remains.</strong>'
                f'<br>{_esc(b.get("target_field",""))}: {_esc(b.get("operator_question") or b.get("issue",""))}</div>')
    elif n_block:
        head = (f'<div class="callout block"><strong>{n_block} blocking decision(s)</strong> '
                "must be resolved before MI handoff.</div>")
    else:
        head = ('<div class="callout warn">No blocking decisions; only optional '
                "confirmations remain.</div>")

    intro = ('<p class="meta">This compact queue is the managed-service operator '
             "workflow. It is derived from target coverage gaps/conflicts (Gate 3) and "
             "blocking residuals — NOT from every source column.</p>")
    counts_html = _counts_table("Decision type counts", summary.get("decision_type_counts", {}))

    # Optional target-first LLM ADVISOR overlay (advisory only).
    adv = tf.get("llm_advisor") or {}
    rec_by_id = {r.get("decision_id", ""): r for r in (adv.get("rows", []) or [])}
    advisor_present = bool(rec_by_id)
    advisor_html = ""
    if advisor_present or tf.get("llm_advisor_usage"):
        u = tf.get("llm_advisor_usage") or {}
        asum = adv.get("summary", {}) or {}
        reviewed = u.get("decision_rows_reviewed", asum.get("recommendations_total", 0))
        advised = u.get("decision_rows_advised", asum.get("advised", 0))
        review = asum.get("requires_operator_review", 0)
        advisor_html = (
            '<div class="callout pass"><strong>Target-first LLM advisor:</strong> '
            f"reviewed {len(rec_by_id)} Gate 4 decision(s); {advised} advised; "
            f"{review} require operator review. Advisory only — deterministic 28a/28c "
            "are unchanged; the operator still approves via 34_target_first_decisions.yaml."
            "</div>"
            '<p class="meta">Source-column LLM review and the target-first LLM advisor '
            "are separate layers; a source-column count of 0 does not mean the advisor "
            "did nothing.</p>")

    def block(label, items, badge):
        if not items:
            return ""
        body = []
        hdr = ["ID", "Type / priority", "Target field", "Operator question",
               "Deterministic recommendation", "Options", "Evidence"]
        if advisor_present:
            hdr += ["LLM advisory", "LLM conf.", "LLM rationale / note"]
        for r in items:
            row = [
                _esc(r.get("decision_id", "")),
                f'<span class="badge {badge}">{_esc(r.get("decision_type",""))}</span>'
                f'<br><small>{_esc(r.get("priority",""))}</small>',
                _esc(r.get("target_field", "") or "—"),
                f'<strong>{_esc(r.get("operator_question") or r.get("issue",""))}</strong>',
                _esc(r.get("recommendation", "")),
                _esc(r.get("options", "") or "—"),
                _esc(r.get("evidence_summary", "") or "—"),
            ]
            if advisor_present:
                rec = rec_by_id.get(r.get("decision_id", ""), {})
                src = rec.get("llm_recommended_source_column", "")
                action = rec.get("llm_recommended_action", "") or "—"
                adv_cell = _esc(action) + (f"<br><small>{_esc(src)}</small>" if src else "")
                note = (rec.get("llm_rationale", "") or rec.get("llm_operator_note", "") or "")
                row += [adv_cell, _esc(rec.get("llm_confidence", "")),
                        _esc(note or "—")]
            body.append(row)
        return (f'<h4 class="chart-title">{_esc(label)} ({len(items)})</h4>'
                + _table(hdr, body))

    return (intro + head + advisor_html + counts_html
            + block("Blocking decisions", blocking, "b-block")
            + block("Confirmations / non-blocking", nonblocking, "b-warn"))


def _residual_section_html(tf: dict) -> str:
    """Residual source fields (28b) — explicitly NOT primary approvals."""
    res = tf.get("residual")
    if not res:
        return ('<div class="callout warn">Source residual register '
                '(<code>28b_source_residual_register.csv</code>) not available for this run.</div>')
    summary = res.get("summary", {})
    rows = res.get("rows", [])
    intro = (
        '<div class="callout pass">These source columns are NOT part of the primary '
        "approval workflow. They were not selected as the primary source for any target "
        f"field and are suppressed from the main queue "
        f'({summary.get("suppressed_from_main_queue", 0)} of '
        f'{summary.get("residual_source_columns_total", len(rows))} suppressed).</div>')
    counts_html = _counts_table("Residual class counts", summary.get("residual_class_counts", {}))
    body = [[
        _esc(r.get("source_file", "")), _esc(r.get("source_column", "")),
        _esc(r.get("residual_class", "")),
        _esc(r.get("duplicate_of_target_field", "") or "—"),
        _esc(r.get("residual_reason", "")),
        _esc(r.get("possible_future_use", "") or "—"),
    ] for r in rows]
    detail = _table(["File", "Source column", "Residual class", "Duplicate of target",
                     "Reason", "Possible future use"], body)
    return (intro + counts_html
            + f"<details><summary class='meta'>Show {len(rows)} residual source "
              f"columns (detail)</summary>{detail}</details>")


def _audit_queue_html(tf: dict) -> str:
    """Detailed source-column audit detail (33) — NOT the primary gate."""
    q = tf.get("queue33")
    banner = ('<div class="callout warn"><strong>Source-column audit detail, not the '
              "primary gate.</strong> The legacy 33 review queue lists one row per source "
              "column. It is retained for audit/traceability only; the managed-service "
              "onboarding burden is the compact decision queue (Gate 4), not this list.</div>")
    if not q:
        return banner + '<p class="meta">33_mapping_review_queue.json not available.</p>'
    summary = q.get("summary", {})
    items = q.get("items", [])
    approvals = sum(1 for it in items if it.get("requires_user_approval"))
    rows = [
        ["Columns reviewed (audit)", summary.get("total_columns_reviewed", len(items))],
        ["requires_user_approval rows (audit only)", approvals],
        ["auto-approved", summary.get("auto_approved", 0)],
        ["needs decision (audit grouping)", summary.get("needs_review", 0)],
        ["missing target (audit grouping)", summary.get("missing_target", 0)],
        ["cashflow ledger candidates (audit)", summary.get("cashflow_ledger_candidates", 0)],
        ["ignored / null / empty (audit)", summary.get("ignored_null_empty", 0)],
    ]
    return (banner
            + _table(["Audit metric", "Value"], [[_esc(a), _esc(b)] for a, b in rows]))


def _domain_coverage_html(project: OnboardingProject) -> str:
    rows = []
    for d in project.domain_coverage:
        rows.append([
            _esc(d.domain), _domain_status_badge(d.status),
            _esc(", ".join(d.source_files) or "—"),
            _esc(f"{d.mapped_fields_count}/{d.required_fields_count}"),
            '<span class="badge b-block">blocking</span>' if d.blocking else "—",
            _esc(d.notes),
        ])
    intro = (
        "Onboarding is domain-based, not file-based: a combined master tape can "
        "cover the loan AND collateral domains at once, so a separate collateral "
        "file is not required where collateral/property fields are present."
    )
    return (f'<p class="meta">{intro}</p>'
            + _table(["Domain", "Status", "Source files", "Mapped/Required",
                      "Blocking", "Coverage note"], rows))


def _mapping_trace_html(project: OnboardingProject) -> str:
    s = project.mapping_trace_summary or {}
    if not s:
        return '<p class="meta">Mapping trace not available.</p>'
    alias_files = s.get("alias_files_loaded", []) or []
    rows = [
        ["Mapping trace available", "yes"],
        ["Alias files loaded", "yes — " + ", ".join(alias_files) if alias_files else "no"],
        ["Registry fields loaded", s.get("registry_fields_count", "—")],
        ["Columns mapped by alias", s.get("mapped_by_alias", 0)],
        ["Columns mapped by registry/header scoring", s.get("mapped_by_registry_header", 0)],
        ["— of which via Gate 1 semantic alignment (fuzzy)", s.get("mapped_by_semantic_alignment", 0)],
        ["Columns mapped by value match/context", s.get("mapped_by_value_or_context", 0)],
        ["Columns out of scope (mode)", s.get("out_of_scope", 0)],
        ["Columns requiring user review", s.get("ambiguous_needs_review", 0)],
        ["Columns unmapped", s.get("unmapped", 0)],
        ["Columns sent to LLM", s.get("sent_to_llm", 0)],
    ]
    intro = ("Deterministic-first: Python profiling → field registry → alias libraries "
             "→ scoring → value matching → source precedence. The LLM only reviews "
             "unresolved ambiguity and never writes final mappings. Full detail in "
             "<code>05c_mapping_trace.csv</code> / <code>05d_mapping_explanation.md</code>.")
    return (f'<p class="meta">{intro}</p>'
            + _table(["Metric", "Value"], [[_esc(a), _esc(b)] for a, b in rows]))


def _azure_metadata_html(project: OnboardingProject) -> str:
    rows = [
        ["Client ID", project.client_id or "—"],
        ["Run ID", project.run_id or "—"],
        ["Storage backend", project.storage_backend or "local"],
        ["Input URI", project.input_uri or "— (local)"],
        ["Output URI", project.output_uri or "— (local)"],
    ]
    return _table(["Field", "Value"], [[_esc(a), _esc(b)] for a, b in rows])


def _promotion_section_html(output_root: Path) -> str:
    """Central tape / pipeline / readiness / manifest status (after promote)."""
    import json as _json

    central = output_root / "central"
    manifests = output_root / "manifests"
    lender_summary = central / "18d_central_tape_summary.json"
    pipeline_summary = central / "18a_central_pipeline_summary.json"
    readiness = manifests / "21_pipeline_handoff_readiness.json"

    if not lender_summary.exists() and not readiness.exists():
        return ('<div class="callout warn">Promotion not yet run. Run '
                '<code>cli promote --project-dir &lt;dir&gt; --approved-only --dry-run</code> '
                'to build the central tapes and Azure-ready handoff manifests.</div>')

    def load(p):
        try:
            return _json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}

    ls = load(lender_summary)
    ps = load(pipeline_summary)
    rj = load(readiness)

    central_rows = [
        ["Central lender tape created", "yes" if lender_summary.exists() else "no"],
        ["Loan count", ls.get("loan_count", "—")],
        ["Mapped fields", ls.get("canonical_fields_populated", "—")],
        ["Unresolved gaps", ls.get("gap_count", "—")],
        ["Conflict count", ls.get("conflict_count", "—")],
        ["Lineage path", "output/lineage/18b_central_tape_lineage.csv"],
    ]
    pipeline_rows = [
        ["Central pipeline tape created", "yes" if pipeline_summary.exists() else "no"],
        ["Application count", ps.get("pipeline_count", "—")],
        ["Linked funded loans", ps.get("linked_to_funded_loans", "—")],
        ["Application-only rows", ps.get("application_only_rows", "—")],
    ]
    readiness_rows = [
        ["Ready for MI", rj.get("ready_for_mi_agent", "—")],
        ["Ready for Gate 1 handoff", rj.get("ready_for_gate1_handoff", "—")],
        ["Ready for regulatory projection", rj.get("ready_for_regulatory_projection", "—")],
        ["Ready for warehouse analysis", rj.get("ready_for_warehouse_analysis", "—")],
    ]
    manifest_files = [
        "19_promotion_plan.yaml", "20_pipeline_handoff_manifest.yaml",
        "21_pipeline_handoff_readiness.json", "23_pipeline_trigger.json",
    ]
    manifest_rows = [
        [name, "present" if (manifests / name).exists() else "missing"]
        for name in manifest_files
    ]
    return (
        '<h4 class="chart-title">Central lender tape</h4>'
        + _table(["Item", "Value"], [[_esc(a), _esc(b)] for a, b in central_rows])
        + '<h4 class="chart-title">Central pipeline tape</h4>'
        + _table(["Item", "Value"], [[_esc(a), _esc(b)] for a, b in pipeline_rows])
        + '<h4 class="chart-title">Dry-run readiness</h4>'
        + _table(["Readiness", "Value"], [[_esc(a), _esc(b)] for a, b in readiness_rows])
        + '<h4 class="chart-title">Azure-ready handoff manifests</h4>'
        + _table(["Manifest", "Status"], [[_esc(a), _esc(b)] for a, b in manifest_rows])
    )


_HANDOFF_MARKER = "<!--ONBOARDING_HANDOFF_SECTION-->"

_HANDOFF_WARNING = (
    "The central lender tape is a canonical onboarding handoff artefact. It is "
    "not raw source input and not an XML-ready regulatory delivery tape. "
    "Downstream agents must consume this through the Transformation &amp; "
    "Validation handoff path.")


def _handoff_section_html(output_root: Path) -> str:
    """Onboarding → Transformation &amp; Validation handoff status (24–27)."""
    import json as _json

    handoff = Path(output_root) / "handoff"
    manifest = handoff / "24_onboarding_handoff_manifest.json"
    if not manifest.exists():
        return (f'<div class="callout warn"><strong>{_HANDOFF_WARNING}</strong></div>'
                '<p class="meta">Onboarding handoff package not yet generated for this '
                'run (Annex 2 / regulatory mode only).</p>')

    try:
        m = _json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        m = {}

    def yn(v) -> str:
        return "yes" if v else "no"

    head_rows = [
        ["Handoff type", m.get("handoff_type", "")],
        ["Next agent", m.get("next_agent", "")],
        ["Central tape path", m.get("central_tape_path", "")],
        ["Not raw source", yn(m.get("not_raw_source"))],
        ["Not XML ready", yn(m.get("not_xml_ready"))],
        ["Do not rerun Gate 1 on central tape",
         yn(m.get("do_not_rerun_gate1_on_central_tape"))],
    ]
    readiness_rows = [
        ["Ready for transformation & validation",
         yn(m.get("ready_for_transformation_validation"))],
        ["Ready for projection", yn(m.get("ready_for_projection"))],
        ["Ready for XML delivery", yn(m.get("ready_for_xml_delivery"))],
    ]
    classified_rows = [
        ["Operator decisions pending", m.get("operator_decision_pending_count", 0)],
        ["Blocking decisions", m.get("blocking_decision_count", 0)],
        ["Downstream defaults required", m.get("downstream_default_required_count", 0)],
        ["ND defaults", m.get("defaulted_nd_count", 0)],
        ["Pending regime rules", m.get("pending_regime_rule_count", 0)],
        ["Semantic derivations required", m.get("semantic_derivation_required_count", 0)],
        ["Source absent fields", m.get("source_absent_count", 0)],
    ]
    return (
        f'<div class="callout warn"><strong>{_HANDOFF_WARNING}</strong></div>'
        + _table(["Item", "Value"], [[_esc(a), _esc(b)] for a, b in head_rows])
        + '<h4 class="chart-title">Handoff readiness (separate from XML readiness)</h4>'
        + _table(["Readiness", "Value"], [[_esc(a), _esc(b)] for a, b in readiness_rows])
        + '<h4 class="chart-title">Classified for the next agent</h4>'
        + _table(["Item", "Count"], [[_esc(a), _esc(b)] for a, b in classified_rows])
    )


def refresh_review_pack_handoff(project_dir: Path, output_root: Path | None = None) -> None:
    """Re-inject the Onboarding Handoff section into an existing review pack."""
    import re

    project_dir = Path(project_dir)
    pack = project_dir / "08_onboarding_review_pack.html"
    if not pack.exists():
        return
    if output_root is None:
        output_root = project_dir / "output"
    section = _handoff_section_html(Path(output_root))
    text = pack.read_text(encoding="utf-8")
    text = re.sub(r"<!--HANDOFF_START-->.*?<!--HANDOFF_END-->",
                  f"<!--HANDOFF_START-->{section}<!--HANDOFF_END-->",
                  text, flags=re.DOTALL)
    pack.write_text(text, encoding="utf-8")


def build_review_pack(project: OnboardingProject, out_path: Path,
                      output_root: Path | None = None) -> Path:
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

    # Legacy readiness callout (gap-question based). Retained for the supporting
    # section and as the fallback headline only when no target-first artefacts exist.
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

    # Target-contract-first artefacts (28a/28b/28c) drive the primary headline.
    # Search all plausible run dirs so the pack is correct wherever it is written.
    _eff_root = Path(output_root) if output_root is not None else (Path(out_path).parent / "output")
    tf = _load_target_first_artifacts(Path(out_path).parent, _eff_root)
    tf_present = tf.get("coverage") is not None
    tf_decision_present = tf.get("decision") is not None
    cov_sum = (tf.get("coverage") or {}).get("summary", {}) or {}
    res_sum = (tf.get("residual") or {}).get("summary", {}) or {}
    dec_sum = (tf.get("decision") or {}).get("summary", {}) or {}
    cov_counts = cov_sum.get("coverage_status_counts", {}) or {}
    q33 = tf.get("queue33") or {}
    q33_items = q33.get("items", []) or []
    old33_approvals = sum(1 for it in q33_items if _as_bool(it.get("requires_user_approval")))

    # Headline status: when 28c exists it SUPERSEDES the legacy gap-question
    # blocker count (never let the two contradict each other).
    tf_n_block = dec_sum.get("blocking_decisions",
                            sum(1 for r in (tf.get("decision") or {}).get("rows", [])
                                if _as_bool(r.get("blocking"))))
    tf_n_decisions = dec_sum.get("human_decision_rows_total",
                                len((tf.get("decision") or {}).get("rows", [])))
    tf_n_nonblock = max(0, tf_n_decisions - tf_n_block)
    if tf_decision_present:
        if tf_n_block == 1:
            tf_status_badge = '<span class="badge b-block">1 BLOCKING TARGET DECISION</span>'
            exec_status_html = (
                '<div class="callout block"><strong>Status: BLOCKED.</strong> Only ONE '
                "blocking target decision remains — resolve it in Gate 4 — Compact human "
                "decision queue. Legacy source-column gap questions are superseded by the "
                "target-first queue.</div>")
        elif tf_n_block > 1:
            tf_status_badge = (f'<span class="badge b-block">{tf_n_block} BLOCKING '
                               "TARGET DECISIONS</span>")
            exec_status_html = (
                f'<div class="callout block"><strong>Status: BLOCKED — {tf_n_block} blocking '
                "target decisions</strong> remain — see Gate 4. These supersede the legacy "
                "source-column gap questions.</div>")
        elif tf_n_decisions:
            tf_status_badge = '<span class="badge b-warn">NEEDS CONFIRMATION</span>'
            exec_status_html = (
                '<div class="callout warn"><strong>Status: NEEDS CONFIRMATION.</strong> '
                f"No blocking target decisions. {tf_n_nonblock} non-blocking confirmation"
                f"{'s' if tf_n_nonblock != 1 else ''} remain — see Gate 4.</div>")
        else:
            _fu_sum = (tf.get("field_universe") or {}).get("summary", {}) or {}
            _univ_gap = (int(_fu_sum.get("missing_from_28a_count", 0))
                         + int(_fu_sum.get("missing_from_regime_rules_count", 0)))
            if _is_annex2(tf) and _univ_gap > 0:
                # No 28c decisions, but the Annex 2 universe is not fully configured.
                tf_status_badge = '<span class="badge b-warn">NEEDS CONFIGURATION</span>'
                exec_status_html = (
                    '<div class="callout warn"><strong>Status: NEEDS CONFIGURATION.</strong> '
                    "No outstanding Gate 4 decisions, but the Annex 2 target universe is not "
                    f"fully configured ({_univ_gap} code(s) pending a regime rule / missing "
                    "from 28a) — see the Annex 2 field universe reconciliation.</div>")
            else:
                _ready_label = ("READY FOR ANNEX 2 DELIVERY" if _is_annex2(tf)
                                else "READY FOR MI HANDOFF")
                tf_status_badge = f'<span class="badge b-ok">{_ready_label}</span>'
                exec_status_html = (
                    f'<div class="callout pass"><strong>Status: {_ready_label}.</strong> '
                    "No outstanding target decisions — target coverage is complete.</div>")
        # Override the hero badge so it cannot show a stale legacy blocker count.
        status_badge = tf_status_badge
    else:
        exec_status_html = readiness  # legacy readiness when no target-first artefacts
    fcov = tf.get("file_coverage") or []
    parsed_files = (sum(1 for f in fcov if f.get("column_evidence_rows", 0) > 0)
                    if fcov else counts["classified_files"])

    def _dash(v):  # show — when the target-first artefacts are not present
        return v if (tf.get("coverage") is not None) else "—"

    configured_defaulted = (cov_counts.get("configured_static", 0)
                            + cov_counts.get("defaulted", 0)
                            + cov_counts.get("defaulted_ND", 0))
    mapped_derived_configured = (cov_sum.get("source_mapped_fields", 0)
                                 + cov_counts.get("derived", 0) + configured_defaulted)
    # 1. Executive onboarding summary KPIs — operator-first headline from 28a/28b/28c.
    kpis = [
        ("Input source files", counts["classified_files"]),
        ("Target fields", _dash(cov_sum.get("target_fields_total", 0))),
        ("Mapped / derived / configured", _dash(mapped_derived_configured)),
        ("Non-blocking confirmations", _dash(tf_n_nonblock) if tf_decision_present else "—"),
        ("Blocking target decisions", _dash(tf_n_block) if tf_decision_present else "—"),
        ("Residual source fields suppressed", _dash(res_sum.get("suppressed_from_main_queue", 0))),
        ("Compact decision queue", _dash(dec_sum.get("human_decision_rows_total", 0))),
        ("Old source-column approvals — audit only", _dash(old33_approvals)),
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

    llm_block = (f'<h4 class="chart-title">LLM mapping review usage &amp; cost</h4>{llm_html}'
                 if show_llm else '')

    # When 28c exists, the legacy gap-question / readiness views are clearly
    # marked as superseded so the two blocker counts can never contradict.
    _legacy_note = (
        '<p class="meta"><span class="badge b-info">audit</span> Legacy/supporting gap '
        "questions — superseded by target-first Gate 4 where 28c is available.</p>"
        if tf_decision_present else "")

    # Gate 2 source-pack readiness summary (file-level; never implies the pack is
    # blocked when target-first Gate 4 has no blocking rows).
    gate2_html = (
        f'<p class="meta">{counts["classified_files"]} source file(s) received; '
        f"{parsed_files} parsed / classified.</p>"
        f'<h4 class="chart-title">File inventory</h4>{inv_html}'
        f'<h4 class="chart-title">Data domain coverage</h4>{_domain_coverage_html(project)}')

    # Appendices: every legacy / source-column section, clearly superseded.
    appendix = f"""
  <div class="card"><h2>6. Appendices / audit detail</h2>
    <div class="callout warn"><strong>Legacy / audit detail — superseded by target-first
      Gate 4 where 28c is available.</strong> These source-column diagnostics are NOT the
      managed-service approval burden.</div>
    <h4 class="chart-title">Residual source fields</h4>{_residual_section_html(tf)}
    <h4 class="chart-title">Detailed source-column audit queue (33) — audit only, not the primary gate</h4>{_audit_queue_html(tf)}
    <h4 class="chart-title">Legacy / supporting gap questions</h4>{_legacy_note}{q_html}
    <h4 class="chart-title">Legacy readiness assessment</h4>{readiness}
    <h4 class="chart-title">Onboarding mode &amp; mode-specific readiness (legacy)</h4>{mode_readiness_html}
    <h4 class="chart-title">Field scope for this onboarding mode (legacy)</h4>{field_scope_html}
    <h4 class="chart-title">Recommended next actions (legacy)</h4>{actions_html}
    <h4 class="chart-title">Mapping ambiguities resolved by policy (audit)</h4>{ambiguities_html}
    <h4 class="chart-title">Deterministic mapping trace (audit)</h4>{_mapping_trace_html(project)}
    <h4 class="chart-title">Source overlap / duplicate fields (audit)</h4>{ov_html}
    <h4 class="chart-title">Mapping candidates — source-column detail (audit)</h4>{map_html}
    {llm_block}
    <h4 class="chart-title">Azure-ready run metadata</h4>{_azure_metadata_html(project)}
    <h4 class="chart-title">Detected reporting periods</h4>{periods_html}
    <h4 class="chart-title">Candidate keys</h4>{keys_html}
  </div>"""
    # Neutralise legacy red blocking styling in the appendix when 28c supersedes it.
    if tf_decision_present:
        appendix = appendix.replace("badge b-block", "badge b-info").replace(
            "callout block", "callout warn")

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

  <div class="card"><h2>1. Executive onboarding summary</h2>
    <p class="meta">Executive summary — operator-first. The managed-service workflow is
      the compact human decision queue (Gate 4); the full target matrix and the 33
      source-column queue are audit detail.</p>
    <div class="kpi-grid">{kpi_html}</div>
    <p class="meta">Input: {_esc(project.input_dir)} &nbsp;·&nbsp; Output: {_esc(project.output_dir)}</p>
    {exec_status_html}
    {_annex2_executive_html(tf)}
  </div>

  <div class="card"><h2>2. Gate 4 — Compact human decision queue</h2>
    <p class="meta">What the operator must action — read this before the 72-field matrix.</p>
    {_gate4_decision_queue_html(tf)}
  </div>

  <div class="card"><h2>3. Gate 3 — Target coverage summary</h2>{_gate3_summary_html(tf)}</div>

  <div class="card"><h2>4. Gate 2 — Source pack readiness summary</h2>{gate2_html}</div>

  <div class="card"><h2>5. Gate 5 — {"Annex 2 delivery readiness" if _is_annex2(tf) else "MI handoff readiness"}</h2>
    {_gate5_mi_handoff_status_html(tf, tf_n_block, tf_n_decisions)}
    {_decision_application_html(tf)}
    <h4 class="chart-title">Config suggestions</h4>{cfg_html}
    <h4 class="chart-title">Central tapes &amp; Azure-ready handoff (dry-run)</h4>
    <span id="promotion"></span><!--PROMO_START-->{_PROMOTION_MARKER}<!--PROMO_END-->
  </div>

  <div class="card" id="onboarding-handoff"><h2>5b. Onboarding handoff (→ Transformation &amp; Validation)</h2>
    <p class="meta">Governed canonical onboarding package — the next valid consumer is the
      Transformation &amp; Validation Agent. Downstream agents must not re-run raw Gate 1
      source canonicalisation on the central tape.</p>
    <span id="onboarding-handoff-status"></span><!--HANDOFF_START-->{_HANDOFF_MARKER}<!--HANDOFF_END-->
  </div>
  {appendix}
  {_APPROVAL_MARKER}
</div></body></html>"""

    out_path = Path(out_path)
    if output_root is None:
        output_root = out_path.parent / "output"
    # If approved artefacts already exist in this output dir, fold them in.
    doc = doc.replace(_APPROVAL_MARKER, build_approval_section_html(out_path.parent))
    doc = doc.replace(_PROMOTION_MARKER, _promotion_section_html(Path(output_root)))
    doc = doc.replace(_HANDOFF_MARKER, _handoff_section_html(Path(output_root)))
    out_path.write_text(doc, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# PART 7 — approved artefacts section (shown only once answers are ingested)
# ---------------------------------------------------------------------------

# Marker so the approval section can be (re)injected idempotently after ingestion.
_APPROVAL_MARKER = "<!--APPROVAL_SECTION-->"
# Marker so the promotion / central-tape section can be re-injected after promote.
_PROMOTION_MARKER = "<!--PROMOTION_SECTION-->"


def refresh_review_pack_promotion(project_dir: Path, output_root: Path | None = None) -> None:
    """Re-inject the central-tape / handoff section into an existing review pack."""
    import re

    project_dir = Path(project_dir)
    pack = project_dir / "08_onboarding_review_pack.html"
    if not pack.exists():
        return
    if output_root is None:
        output_root = project_dir / "output"
    section = _promotion_section_html(Path(output_root))
    text = pack.read_text(encoding="utf-8")
    text = re.sub(r"<!--PROMO_START-->.*?<!--PROMO_END-->",
                  f"<!--PROMO_START-->{section}<!--PROMO_END-->",
                  text, flags=re.DOTALL)
    pack.write_text(text, encoding="utf-8")


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
