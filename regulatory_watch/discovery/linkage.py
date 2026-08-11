"""regulatory_watch.discovery.linkage — the Stage 1A boundary.

Stage 1B may say *"a technical artefact moved, Stage 1A should run"* and, when
handed a Stage 1A report, may quote that report's identifiers and counts. It
may not compute a field delta, and this module is where that boundary is
enforced structurally:

* it reads Stage 1A's **published JSON report contract** — it never imports the
  comparator, the normalizer or the impact engine, so there is no code path by
  which Stage 1B could produce a delta of its own;
* it reads Stage 1A's **artefact allowlist** only to answer "is this URL one we
  already track?", and it never writes to it. A technical URL Stage 1A does not
  know is flagged ``NEW_TECHNICAL_SOURCE_REVIEW_REQUIRED`` for a human, never
  auto-added.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit

from .contracts import (
    MATCHES_STAGE1A_SOURCE,
    NEW_TECHNICAL_SOURCE_REVIEW_REQUIRED,
    NO_TECHNICAL_ARTEFACT,
    RawPublication,
    TechnicalArtefactLink,
    TechnicalSpecLinkage,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

#: File extensions that mark a link as a technical artefact rather than prose.
TECHNICAL_SUFFIXES = (".xsd", ".xlsx", ".xls", ".xml", ".zip")

#: Path/URL fragments that mark an ESMA technical package page.
TECHNICAL_URL_HINTS = ("xml-schema", "schema-and-instructions",
                       "technical-standards", "reporting-templates",
                       "validation-rules", "taxonomy")


def _normalise(url: str) -> str:
    parts = urlsplit((url or "").strip())
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(),
                       parts.path.rstrip("/"), "", ""))


def looks_technical(url: str) -> bool:
    """Whether a URL points at an Annex 2 technical artefact (or its page)."""
    lowered = (url or "").lower()
    if lowered.endswith(TECHNICAL_SUFFIXES):
        return True
    return any(hint in lowered for hint in TECHNICAL_URL_HINTS)


def stage1a_source_urls(repo_root: Optional[Path] = None) -> Dict[str, List[str]]:
    """``{normalised url: [artefact_id, ...]}`` from Stage 1A's allowlist.

    A list, not a single id: Stage 1A declares the workbook, the schema and the
    reporting instructions against the SAME ESMA package URL, so a match names
    every artefact a reviewer must re-verify, not whichever one happened to be
    declared last.

    Read-only. Import failure is not fatal: linkage then reports every
    technical URL as needing review, which is the conservative answer.
    """
    try:
        from ..manifest import load_manifest
        manifest = load_manifest()
    except Exception:                     # noqa: BLE001 - stay decoupled
        return {}
    out: Dict[str, List[str]] = {}
    for artefact in manifest.artefacts:
        if artefact.source_url:
            out.setdefault(_normalise(artefact.source_url), []).append(
                artefact.artefact_id)
    return {k: sorted(v) for k, v in out.items()}


def assess_links(publication: RawPublication,
                 known_urls: Optional[Dict[str, List[str]]] = None,
                 ) -> TechnicalSpecLinkage:
    """Classify the technical artefacts a publication points at.

    The publication's own URL counts: an ESMA technical-standards *page* is
    itself the artefact carrier, and that is exactly the Stage 1A source URL.
    """
    known = stage1a_source_urls() if known_urls is None else known_urls
    candidates: List[str] = []
    for url in [publication.url, *publication.linked_urls]:
        if url and looks_technical(url) and url not in candidates:
            candidates.append(url)

    links: List[TechnicalArtefactLink] = []
    for url in candidates:
        artefact_ids = known.get(_normalise(url)) or []
        if artefact_ids:
            links.append(TechnicalArtefactLink(
                url=url, status=MATCHES_STAGE1A_SOURCE,
                stage1a_artefact_id=", ".join(artefact_ids),
                detail=(f"matches {len(artefact_ids)} allowlisted Stage 1A "
                        f"technical source(s); run the Stage 1A "
                        f"snapshot/compare to determine the exact field "
                        f"deltas")))
        else:
            links.append(TechnicalArtefactLink(
                url=url, status=NEW_TECHNICAL_SOURCE_REVIEW_REQUIRED,
                detail=("technical artefact URL that Stage 1A does not track; "
                        "a human must decide whether to add it to the Stage 1A "
                        "allowlist — Stage 1B never edits it")))

    if not links:
        return TechnicalSpecLinkage(status=NO_TECHNICAL_ARTEFACT)

    status = (MATCHES_STAGE1A_SOURCE
              if any(l.status == MATCHES_STAGE1A_SOURCE for l in links)
              else NEW_TECHNICAL_SOURCE_REVIEW_REQUIRED)
    return TechnicalSpecLinkage(
        status=status, artefact_links=links,
        stage1a_verification_required=True)


def attach_stage1a_report(linkage: TechnicalSpecLinkage,
                          report_path: Path) -> TechnicalSpecLinkage:
    """Quote a Stage 1A report into a linkage. Reads the contract, nothing else.

    Stage 1B does not run Stage 1A and does not interpret its deltas — it
    records which report was run, over which snapshots, and how many deltas and
    impacts it found, so a reviewer can go straight to it.
    """
    path = Path(report_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        linkage.stage1a_report_path = str(path)
        linkage.stage1a_outcome = f"STAGE1A_REPORT_UNREADABLE: {exc}"
        linkage.stage1a_verification_required = True
        return linkage

    def snapshot_ids(side: str) -> List[str]:
        sources = (payload.get(side) or {}).get("sources") or []
        return [f"{s.get('artefact_id')}@{str(s.get('artefact_sha256') or '')[:16]}"
                for s in sources if s.get("artefact_sha256")]

    linkage.stage1a_report_path = str(path)
    linkage.stage1a_outcome = payload.get("outcome")
    linkage.stage1a_baseline_snapshot_ids = snapshot_ids("baseline")
    linkage.stage1a_candidate_snapshot_ids = snapshot_ids("candidate")
    linkage.regulatory_delta_count = payload.get("regulatory_delta_count")
    linkage.implementation_impact_count = payload.get(
        "implementation_impact_count")
    # A Stage 1A run has happened; verification is no longer outstanding.
    linkage.stage1a_verification_required = False
    return linkage
