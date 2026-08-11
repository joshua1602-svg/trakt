"""regulatory_watch.discovery.report — the Stage 1B outputs.

One machine-readable JSON report (the contract a Stage 2 OCC Regulatory Config
view consumes) and one deliberately short Markdown summary: a weekly control
that takes two minutes to read gets read.

``discovery_date`` is supplied by the caller so a report is byte-reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .contracts import (
    DISCOVERY_FAILED,
    DISCOVERY_OK,
    DiscoveryReport,
    MATCHES_STAGE1A_SOURCE,
    NEW_TECHNICAL_SOURCE_REVIEW_REQUIRED,
    TECHNICAL_SPEC_CHANGE,
)


def to_json(report: DiscoveryReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True,
                      ensure_ascii=False) + "\n"


def _fmt(value: Any, limit: int = 140) -> str:
    if value in (None, "", []):
        return "—"
    text = str(value).replace("|", "\\|").replace("\n", " ")
    return text if len(text) <= limit else text[:limit - 1] + "…"


def to_markdown(report: DiscoveryReport) -> str:
    r = report
    out: List[str] = []
    out.append("# ESMA Annex 2 Regulatory Watch — publication discovery")
    out.append("")
    out.append(f"*Discovery date: {r.discovery_date} · "
               f"{r.discovery_version} · report contract "
               f"v{r.contract_version} · Stage 1B, observational only.*")
    out.append("")
    out.append(f"- **Discovery status:** `{r.discovery_status}`")
    out.append(f"- Sources checked: {r.sources_checked}")
    out.append(f"- Source failures: {r.sources_failed}")
    out.append(f"- New ESMA publications: {r.new_publications}"
               + (f" (plus {r.revised_publications} revised)"
                  if r.revised_publications else ""))
    out.append(f"- Previously seen (skipped): "
               f"{r.previously_seen_publications}")
    out.append(f"- Potentially relevant to Annex 2: {r.potentially_relevant}")
    out.append(f"- Human review required: {r.review_required_count}")
    out.append("")

    if r.discovery_status == DISCOVERY_FAILED:
        out.append("> **A gating discovery source could not be checked.** This "
                   "run is **not** evidence that no relevant ESMA publication "
                   "exists.")
        out.append("")
    elif r.discovery_status != DISCOVERY_OK:
        out.append("> A corroborating discovery source could not be checked, "
                   "so coverage this week is narrower than usual.")
        out.append("")

    failures = [o for o in r.source_outcomes
                if o.get("outcome") not in ("OK", "DISABLED")]
    if failures:
        out.append("## Source failures")
        out.append("")
        for outcome in failures:
            out.append(f"- `{outcome['source_id']}` "
                       f"({outcome.get('criticality')}) — "
                       f"**{outcome['outcome']}**: {_fmt(outcome.get('detail'))}")
        out.append("")

    # -- the developments --------------------------------------------------- #
    out.append("## Potentially relevant to Annex 2")
    out.append("")
    if not r.developments:
        out.append("_None this week._")
    for index, dev in enumerate(r.developments, start=1):
        publication = dev.get("publication", {})
        out.append(f"{index}. **{_fmt(publication.get('title'), 160)}**")
        out.append(f"   Status: `{dev['development_status']}` · "
                   f"type `{dev['publication_type']}` · "
                   f"relevance `{dev['annex2_relevance']}` · "
                   f"confidence {dev['confidence']}")
        out.append(f"   Reason: {_fmt(dev.get('relevance_reason'), 200)}")
        linkage = dev.get("technical_spec_linkage") or {}
        status = linkage.get("status")
        if dev["development_status"] == TECHNICAL_SPEC_CHANGE or \
                status != "NO_TECHNICAL_ARTEFACT":
            out.append(f"   Technical artefacts changed: "
                       f"{'Yes' if status != 'NO_TECHNICAL_ARTEFACT' else 'No'}")
            if linkage.get("regulatory_delta_count") is not None:
                out.append(
                    f"   Stage 1A comparison: "
                    f"{linkage['regulatory_delta_count']} regulatory delta(s); "
                    f"Trakt implementation impacts: "
                    f"{linkage.get('implementation_impact_count')}")
            elif linkage.get("stage1a_verification_required"):
                out.append("   Stage 1A comparison: **not yet run** — "
                           "technical-source verification required")
            for link in linkage.get("artefact_links", []):
                if link.get("status") == NEW_TECHNICAL_SOURCE_REVIEW_REQUIRED:
                    out.append(f"   ⚠ `{NEW_TECHNICAL_SOURCE_REVIEW_REQUIRED}`: "
                               f"{_fmt(link.get('url'), 120)}")
                elif link.get("status") == MATCHES_STAGE1A_SOURCE:
                    out.append(f"   Matches Stage 1A source: "
                               f"`{link.get('stage1a_artefact_id')}`")
        else:
            out.append("   Technical artefacts changed: No")
        for label, key in (("Effective", "effective_date"),
                           ("Implementation", "implementation_date"),
                           ("Consultation closes", "consultation_deadline")):
            if dev.get(key):
                out.append(f"   {label}: {dev[key]}")
        out.append(f"   Human review: "
                   f"{'Yes' if dev['review_required'] else 'No'}")
        out.append(f"   URL: {_fmt(publication.get('url'), 160)}")
        out.append("")

    # -- rejected ----------------------------------------------------------- #
    if r.rejected:
        out.append("## Not relevant")
        out.append("")
        out.append(f"{len(r.rejected)} publication(s) rejected by the "
                   f"deterministic relevance filter. Each retains its reason "
                   f"in the JSON report:")
        out.append("")
        for item in r.rejected[:10]:
            out.append(f"- {_fmt(item.get('title'), 110)} — "
                       f"{_fmt(item.get('reason'), 110)}")
        if len(r.rejected) > 10:
            out.append(f"- _…and {len(r.rejected) - 10} more (see JSON)._")
        out.append("")

    for note in r.notes:
        out.append(f"> {note}")
        out.append("")

    out.append("---")
    out.append("")
    out.append("*Stage 1B detects and classifies ESMA publications. It does "
               "not determine field-level Annex 2 changes — only Stage 1A "
               "does — and it never modifies the active regime.*")
    out.append("")
    return "\n".join(out)


def write_reports(report: DiscoveryReport, out_dir: Path,
                  stem: str = "esma_annex2_publication_discovery",
                  ) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(to_json(report), encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}
