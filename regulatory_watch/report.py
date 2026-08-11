"""regulatory_watch.report — the Stage 1 outputs.

One machine-readable JSON report (the data contract a Stage 2 OCC Regulatory
Config view will consume, deliberately UI-agnostic) and one concise
human-readable Markdown report.

The JSON shape is stable and additive:

.. code-block:: json

    {
      "regime": "ESMA_Annex2",
      "baseline": {},
      "candidate": {},
      "source_status": "SOURCE_CHANGED",
      "gating_source_status": "SOURCE_CHANGED",
      "corroborating_source_status": "SOURCE_CHECK_FAILED",
      "unverified_sources": [],
      "outcome": "REGULATORY_SPEC_CHANGED",
      "specification_current": "no",
      "regulatory_delta_count": 0,
      "implementation_impact_count": 0,
      "deltas": []
    }

with ``spec_status``, ``impacts``, ``unresolved`` and ``evidence`` alongside.
``generated_at`` is supplied by the caller so a report can be byte-reproduced.

``outcome`` is decided by the GATING sources only; a corroborating source that
could not be verified is always listed in ``unverified_sources`` and shown in
the Markdown, but does not by itself withhold the determination.
``specification_current`` is the unambiguous headline — ``yes`` only when every
gating source was verified and no regulatory delta exists.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import REPORT_CONTRACT_VERSION
from .comparator import spec_status, unresolved_items
from .contracts import (
    GATING,
    ImpactFinding,
    MANUAL_REVIEW_REQUIRED,
    NO_IMPLEMENTATION_CHANGE,
    NormalizedSpec,
    SOURCE_CHECK_FAILED,
    SpecDelta,
    WatchReport,
    overall_status,
    specification_current,
    split_source_status,
)


def _spec_summary(spec: NormalizedSpec) -> Dict[str, Any]:
    return {
        "spec_version": spec.spec_version,
        "parser_version": spec.parser_version,
        "field_count": len(spec.fields),
        "scope": dict(spec.scope),
        "source_digest": spec.source_digest,
        "sources": [
            {
                "artefact_id": s.artefact_id,
                "artefact_sha256": s.sha256,
                "byte_size": s.byte_size,
                "retrieved_at": s.retrieved_at,
                "snapshot_path": s.snapshot_path,
                "source_url": s.source_url,
                "external_version": s.external_version,
                "publication_date": s.publication_date,
                "effective_date": s.effective_date,
                "criticality": s.criticality,
                "source_status": s.source_status,
                "retrieval_status": s.retrieval_status,
                "retrieval_detail": s.retrieval_detail,
            }
            for s in sorted(spec.snapshots, key=lambda s: s.artefact_id)
        ],
        "parse_warnings": list(spec.parse_warnings),
    }


def build_report(baseline: NormalizedSpec, candidate: NormalizedSpec,
                 deltas: Sequence[SpecDelta],
                 impacts: Sequence[ImpactFinding],
                 generated_at: str,
                 evidence: Optional[Dict[str, Any]] = None) -> WatchReport:
    """Assemble the Stage 1 report from already-computed parts."""
    split = split_source_status(baseline.snapshots, candidate.snapshots)
    spec = spec_status(deltas, baseline, candidate)
    outcome = overall_status(split["gating"], spec)
    actionable = [i for i in impacts if i.status != NO_IMPLEMENTATION_CHANGE]

    return WatchReport(
        regime=candidate.regime,
        contract_version=REPORT_CONTRACT_VERSION,
        generated_at=generated_at,
        baseline=_spec_summary(baseline),
        candidate=_spec_summary(candidate),
        source_status=split["combined"],
        gating_source_status=split["gating"],
        corroborating_source_status=split["corroborating"],
        unverified_sources=split["unverified"],
        spec_status=spec,
        outcome=outcome,
        specification_current=specification_current(outcome),
        regulatory_delta_count=len(deltas),
        implementation_impact_count=len(actionable),
        deltas=[d.to_dict() for d in deltas],
        impacts=[i.to_dict() for i in impacts],
        unresolved=unresolved_items(baseline, candidate),
        evidence=dict(evidence or {}),
    )


def to_json(report: WatchReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True,
                      ensure_ascii=False) + "\n"


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #

def _source_table(summary: Dict[str, Any]) -> List[str]:
    lines = ["| artefact | criticality | external version | sha256 | status |",
             "| --- | --- | --- | --- | --- |"]
    for s in summary.get("sources", []):
        digest = s["artefact_sha256"]
        crit = s.get("criticality", GATING)
        lines.append(
            f"| `{s['artefact_id']}` | "
            f"{'**gating**' if crit == GATING else crit} | "
            f"{s['external_version']} | "
            f"`{digest[:16] + '…' if digest else '—'}` | "
            f"{s['retrieval_status']}"
            f"{' — ' + s['retrieval_detail'] if s['retrieval_detail'] else ''} |")
    if len(lines) == 2:
        lines.append("| _none_ | | | | |")
    return lines


def _fmt(value: Any, limit: int = 160) -> str:
    if value is None:
        return "—"
    text = json.dumps(value, ensure_ascii=False, sort_keys=True) \
        if isinstance(value, (dict, list)) else str(value)
    text = text.replace("|", "\\|").replace("\n", " ")
    return text if len(text) <= limit else text[:limit - 1] + "…"


def to_markdown(report: WatchReport) -> str:
    r = report
    out: List[str] = []
    out.append(f"# ESMA Annex 2 Regulatory Watch — {r.regime}")
    out.append("")
    out.append(f"*Generated {r.generated_at} · report contract "
               f"v{r.contract_version} · Stage 1, observational only — no "
               f"active configuration was modified.*")
    out.append("")
    out.append(f"**Outcome: `{r.outcome}`** — "
               f"is the implemented machine-readable Annex 2 specification "
               f"current? **{r.specification_current.upper()}**")
    out.append("")
    out.append(f"- gating sources: `{r.gating_source_status}` · "
               f"corroborating sources: `{r.corroborating_source_status}`")
    out.append(f"- parsed specification: `{r.spec_status}`")
    out.append(f"- {r.regulatory_delta_count} regulatory delta(s), "
               f"{r.implementation_impact_count} implementation impact(s)")
    out.append("")

    # 1 + 2. baseline / candidate
    out.append("## 1. Baseline source / version")
    out.append("")
    out.append(f"- **Spec version:** `{r.baseline.get('spec_version')}`")
    out.append(f"- **Fields parsed:** {r.baseline.get('field_count')}")
    out.append(f"- **Normalizer:** `{r.baseline.get('parser_version')}`")
    out.append(f"- **Scope:** {_fmt(r.baseline.get('scope'), 400)}")
    out.append("")
    out.extend(_source_table(r.baseline))
    out.append("")
    out.append("## 2. Candidate source / version")
    out.append("")
    out.append(f"- **Spec version:** `{r.candidate.get('spec_version')}`")
    out.append(f"- **Fields parsed:** {r.candidate.get('field_count')}")
    out.append(f"- **Normalizer:** `{r.candidate.get('parser_version')}`")
    out.append(f"- **Scope:** {_fmt(r.candidate.get('scope'), 400)}")
    out.append("")
    out.extend(_source_table(r.candidate))
    out.append("")

    # 3. source content change
    out.append("## 3. Did the authoritative source content change?")
    out.append("")
    out.append(f"- **Gating sources** (the specification is derived from "
               f"these): `{r.gating_source_status}`")
    out.append(f"- **Corroborating sources** (tracked, nothing derived from "
               f"them): `{r.corroborating_source_status}`")
    out.append("")
    if r.gating_source_status == SOURCE_CHECK_FAILED:
        out.append("> A **gating** source could not be checked. This is "
                   "**not** evidence that the regime is current, and the "
                   "run cannot resolve to `CURRENT`.")
        out.append("")
    if r.unverified_sources:
        out.append("Unverified sources:")
        out.append("")
        for s in r.unverified_sources:
            out.append(f"- `{s['artefact_id']}` ({s['criticality']}, "
                       f"{s['side']}) — {s['retrieval_status']}"
                       f"{': ' + s['detail'] if s.get('detail') else ''}")
        if r.corroborating_source_status == SOURCE_CHECK_FAILED \
                and r.gating_source_status != SOURCE_CHECK_FAILED:
            out.append("")
            out.append("> A **corroborating** source is unverified. No "
                       "compared attribute is derived from it, so it does not "
                       "change the determination about the machine-readable "
                       "specification — but the obligation it carries has not "
                       "been reviewed by this run.")
        out.append("")
    out.append(f"- baseline source digest: `{r.baseline.get('source_digest') or '—'}`")
    out.append(f"- candidate source digest: `{r.candidate.get('source_digest') or '—'}`")
    out.append("")

    # 4. parsed spec change
    out.append("## 4. Did the parsed Annex 2 specification change?")
    out.append("")
    out.append(f"**`{r.spec_status}`**")
    out.append("")
    if r.gating_source_status == "SOURCE_CHANGED" \
            and r.spec_status == "SPEC_UNCHANGED":
        out.append("> The authoritative bytes differ but no Annex 2 "
                   "requirement moved. A source hash change is not, by itself, "
                   "a regulatory change.")
        out.append("")

    # 5. regulatory deltas
    out.append("## 5. Regulatory deltas")
    out.append("")
    if not r.deltas:
        out.append("_No regulatory delta detected._")
    else:
        out.append("| code | change | severity | confidence | old | new |")
        out.append("| --- | --- | --- | --- | --- | --- |")
        for d in r.deltas:
            out.append(f"| `{d['code']}` | {d['change_type']} | "
                       f"{d['severity']} | {d['confidence']} | "
                       f"{_fmt(d['old_value'])} | {_fmt(d['new_value'])} |")
    out.append("")

    # 6. Trakt impact
    out.append("## 6. Likely Trakt implementation impact")
    out.append("")
    actionable = [i for i in r.impacts
                  if i.get("status") != NO_IMPLEMENTATION_CHANGE]
    if not r.impacts:
        out.append("_No impact assessed (no deltas)._")
    else:
        by_status: Dict[str, int] = {}
        for i in r.impacts:
            by_status[i["status"]] = by_status.get(i["status"], 0) + 1
        out.append("Findings by status: "
                   + ", ".join(f"`{k}` × {v}"
                               for k, v in sorted(by_status.items())))
        out.append("")
        out.append("| code | change | component | status | locations | current |")
        out.append("| --- | --- | --- | --- | --- | --- |")
        for i in (actionable or r.impacts):
            out.append(
                f"| `{i['code']}` | {i['change_type']} | {i['component']} | "
                f"**{i['status']}** | "
                f"{', '.join('`' + l + '`' for l in i['locations']) or '—'} | "
                f"{_fmt(i['current_implementation'], 90)} |")
        if actionable and len(actionable) != len(r.impacts):
            out.append("")
            out.append(f"_{len(r.impacts) - len(actionable)} further finding(s) "
                       f"assessed as `NO_IMPLEMENTATION_CHANGE`; see the JSON "
                       f"report._")
    out.append("")

    # 7. unresolved / review
    out.append("## 7. Unresolved parser / review items")
    out.append("")
    review = [i for i in r.impacts if i.get("status") == MANUAL_REVIEW_REQUIRED]
    if not r.unresolved and not review:
        out.append("_None. Every compared attribute was derived "
                   "deterministically from the authoritative artefacts._")
    else:
        for item in r.unresolved:
            code = item.get("code") or "—"
            attrs = ", ".join(item.get("unresolved_attributes") or []) or "—"
            out.append(f"- **{item['side']}** `{code}` "
                       f"(attributes: {attrs}) — {item['reason']}")
        for item in review:
            out.append(f"- **review** `{item['code']}` / {item['component']} — "
                       f"{item['rationale']}")
    out.append("")

    # 8. evidence / provenance
    out.append("## 8. Evidence and provenance")
    out.append("")
    out.append("Every delta in the JSON report carries `evidence[]` entries "
               "naming the artefact and the exact locator "
               "(`sheet=<name>;row=<n>` for the workbook, "
               "`xsd:simpleType/<name>` for a schema code list).")
    out.append("")
    for key, value in sorted((r.evidence or {}).items()):
        out.append(f"- **{key}:** {_fmt(value, 600)}")
    if not r.evidence:
        out.append("- _no additional evidence recorded for this run_")
    out.append("")
    out.append("---")
    out.append("")
    out.append("*Stage 1 is observational. No Annex 2 configuration, regime "
               "version, projection, enum behaviour, ND behaviour or XML "
               "output was created or modified by this run.*")
    out.append("")
    return "\n".join(out)


def write_reports(report: WatchReport, out_dir: Path,
                  stem: str = "esma_annex2_regulatory_watch") -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(to_json(report), encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}
