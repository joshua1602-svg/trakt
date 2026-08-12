"""regulatory_watch.discovery.pipeline — discover -> dedupe -> classify -> report.

The whole Stage 1B control in one deterministic pass. Given the same payloads,
the same state store and the same timestamp it produces byte-identical output.

Retrieval is injected: the pipeline takes already-fetched payloads (from a
fetcher, or from fixtures), so nothing here needs the network and CI never
does. The state store is written only once, at the end of a successful pass,
so a run that dies midway cannot half-remember a publication.

The one thing this module will not do is decide that a regime changed. When a
publication carries a technical artefact it says ``TECHNICAL_SPEC_CHANGE`` and
points at Stage 1A. Stage 1A does the rest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import AUTHORITY, DISCOVERY_REPORT_CONTRACT_VERSION, DISCOVERY_VERSION, REGIME
from .contracts import (
    DiscoveryReport,
    NOT_RELEVANT,
    PublicationSource,
    RawPublication,
    RegulatoryDevelopment,
    SEEN_NEW,
    SEEN_REVISED,
    SEEN_UNCHANGED,
    SOURCE_OK,
    SOURCE_PARSE_FAILED,
    SOURCE_UNEXPECTED_SCHEMA,
    SourceOutcome,
    TECHNICAL_SPEC_CHANGE,
    overall_discovery_status,
)
from .linkage import assess_links, attach_stage1a_report, stage1a_source_urls
from .parsers import FeedParseError, parse
from .sources import PublicationSourceSet
from .state import SeenStore
from .triage import SemanticClassifier, triage_and_classify


@dataclass
class SourcePayload:
    """A source plus the bytes retrieved for it (or the failure that occurred)."""

    source: PublicationSource
    payload: Optional[bytes]
    outcome: SourceOutcome


@dataclass
class DiscoveryResult:
    report: DiscoveryReport
    developments: List[RegulatoryDevelopment] = field(default_factory=list)
    store: Optional[SeenStore] = None


def _publication_block(publication: RawPublication, fingerprint: str,
                       source_id: str) -> Dict[str, Any]:
    return {
        "title": publication.title,
        "url": publication.url,
        "publication_date": publication.publication_date,
        "source_id": source_id,
        "official_identifier": publication.official_identifier,
        "content_sha256": fingerprint,
        "summary": publication.summary,
        "categories": list(publication.categories),
        "linked_urls": list(publication.linked_urls),
        "raw_locator": publication.raw_locator,
    }


def run_discovery(sources: PublicationSourceSet,
                  payloads: Sequence[SourcePayload],
                  store: SeenStore,
                  discovery_date: str,
                  *,
                  classifier: Optional[SemanticClassifier] = None,
                  stage1a_reports: Optional[Dict[str, Path]] = None,
                  known_technical_urls: Optional[Dict[str, str]] = None,
                  commit_state: bool = True) -> DiscoveryResult:
    """Run one weekly discovery pass.

    ``stage1a_reports`` optionally maps a development_id to a Stage 1A report
    path, so a run that has already executed Stage 1A can quote it. Stage 1B
    never runs Stage 1A itself.
    """
    known = (stage1a_source_urls() if known_technical_urls is None
             else known_technical_urls)

    outcomes: List[SourceOutcome] = []
    publications: List[RawPublication] = []

    # -- 1. discovery ------------------------------------------------------- #
    for entry in payloads:
        outcome = entry.outcome
        if entry.payload is None or not outcome.ok:
            outcomes.append(outcome)
            continue
        try:
            items = parse(entry.payload, entry.source.parser,
                          entry.source.source_id, base_url=entry.source.url)
        except FeedParseError as exc:
            outcome.outcome = SOURCE_UNEXPECTED_SCHEMA
            outcome.detail = str(exc)
            outcomes.append(outcome)
            continue
        except Exception as exc:          # noqa: BLE001 - never crash a sweep
            outcome.outcome = SOURCE_PARSE_FAILED
            outcome.detail = f"{type(exc).__name__}: {exc}"
            outcomes.append(outcome)
            continue
        outcome.outcome = SOURCE_OK
        outcome.item_count = len(items)
        outcomes.append(outcome)
        publications.extend(items)

    # -- 2. dedupe + 3. triage + 4. classify -------------------------------- #
    developments: List[RegulatoryDevelopment] = []
    rejected: List[Dict[str, Any]] = []
    observations: List[Tuple[RawPublication, Any]] = []
    counts = {SEEN_NEW: 0, SEEN_UNCHANGED: 0, SEEN_REVISED: 0}

    by_id: Dict[str, RegulatoryDevelopment] = {}
    seen_this_run: Dict[str, str] = {}          # development_id -> source_id
    cross_source_duplicates = 0

    for publication in publications:
        identity = store.classify(publication, discovery_date)

        # The same publication routinely appears in more than one allowlisted
        # source (the feed AND the technical-standards page carry the auth.099
        # package). Identity is source-independent, so the second sighting is
        # the same development: record the extra source, do not raise it twice.
        if identity.development_id in seen_this_run:
            cross_source_duplicates += 1
            existing = by_id.get(identity.development_id)
            if existing is not None:
                also = existing.publication.setdefault(
                    "also_seen_in_sources", [])
                if publication.source_id not in also:
                    also.append(publication.source_id)
            continue
        seen_this_run[identity.development_id] = publication.source_id

        counts[identity.seen_state] = counts.get(identity.seen_state, 0) + 1
        observations.append((publication, identity))

        # An unchanged publication was already triaged in an earlier run. It is
        # counted and not re-raised — that is the dedupe guarantee.
        if identity.seen_state == SEEN_UNCHANGED:
            continue

        triage, classification = triage_and_classify(publication, classifier)
        if classification is None:
            rejected.append({
                "development_id": identity.development_id,
                "title": publication.title,
                "url": publication.url,
                "source_id": publication.source_id,
                "seen_state": identity.seen_state,
                "rejected_by": "deterministic_relevance_filter",
                "reason": triage.reason,
            })
            continue

        linkage = assess_links(publication, known_urls=known)
        report_path = (stage1a_reports or {}).get(identity.development_id)
        if report_path is not None:
            linkage = attach_stage1a_report(linkage, Path(report_path))
        dates = _dates(publication)

        development = RegulatoryDevelopment(
            development_id=identity.development_id,
            authority=AUTHORITY,
            regime=REGIME,
            publication=_publication_block(publication, identity.fingerprint,
                                           publication.source_id),
            publication_type=classification.publication_type,
            development_status=classification.development_status,
            annex2_relevance=classification.annex2_relevance,
            relevance_reason=classification.relevance_reason,
            confidence=classification.confidence,
            review_required=classification.review_required,
            seen_state=identity.seen_state,
            first_seen=identity.first_seen,
            discovered_at=discovery_date,
            discovery_version=DISCOVERY_VERSION,
            classifier=classification.classifier,
            effective_date=dates["effective_date"],
            implementation_date=dates["implementation_date"],
            consultation_deadline=dates["consultation_deadline"],
            supersedes=None,
            final_status=_final_status(classification.publication_type),
            technical_spec_linkage=linkage.to_dict(),
            triage=triage.to_dict(),
            matched_signals=classification.matched_signals,
        )
        developments.append(development)
        by_id[identity.development_id] = development

    # -- 5. commit state ---------------------------------------------------- #
    # Only after everything above succeeded: a half-run must not teach the
    # store that it has seen a publication it never triaged.
    if commit_state:
        for publication, identity in observations:
            store.record(publication, identity, discovery_date)

    # -- 6. report ---------------------------------------------------------- #
    status = overall_discovery_status(outcomes)
    from .contracts import INFORMATION_ONLY, RELEVANCE_NOT_RELEVANT
    information_only = [d for d in developments
                        if d.development_status in (NOT_RELEVANT,
                                                    INFORMATION_ONLY)
                        or d.annex2_relevance == RELEVANCE_NOT_RELEVANT]
    relevant = [d for d in developments if d not in information_only]
    technical = [d for d in developments
                 if d.development_status == TECHNICAL_SPEC_CHANGE]
    linkages = [
        {"development_id": d.development_id,
         "title": d.publication.get("title"),
         **(d.technical_spec_linkage or {})}
        for d in developments
        if (d.technical_spec_linkage or {}).get("status")
        != "NO_TECHNICAL_ARTEFACT"
    ]

    notes: List[str] = []
    if cross_source_duplicates:
        notes.append(
            f"{cross_source_duplicates} publication sighting(s) were the same "
            f"publication reached through more than one allowlisted source, "
            f"and were merged into a single development record.")
    if status != "DISCOVERY_OK":
        notes.append(
            "Discovery did not complete cleanly. This run cannot be read as "
            "'no new relevant ESMA publication'.")
    if technical:
        notes.append(
            "A technical-spec-change candidate was found. Stage 1B does not "
            "determine field deltas — run Stage 1A against the linked "
            "artefacts.")

    report = DiscoveryReport(
        authority=AUTHORITY,
        regime=REGIME,
        contract_version=DISCOVERY_REPORT_CONTRACT_VERSION,
        discovery_version=DISCOVERY_VERSION,
        discovery_date=discovery_date,
        discovery_status=status,
        sources_checked=len(outcomes),
        sources_failed=sum(1 for o in outcomes if not o.ok
                           and o.outcome != "DISABLED"),
        source_outcomes=[o.to_dict() for o in outcomes],
        publications_seen=len(publications),
        new_publications=counts.get(SEEN_NEW, 0),
        revised_publications=counts.get(SEEN_REVISED, 0),
        previously_seen_publications=counts.get(SEEN_UNCHANGED, 0),
        potentially_relevant=len(relevant),
        not_relevant=len(rejected) + len(information_only),
        review_required_count=sum(1 for d in developments if d.review_required),
        technical_spec_change_candidates=len(technical),
        developments=[d.to_dict() for d in developments],
        rejected=rejected,
        stage1a_linkages=linkages,
        notes=notes,
    )
    return DiscoveryResult(report=report, developments=developments,
                           store=store)


def _dates(publication: RawPublication) -> Dict[str, Optional[str]]:
    from .triage import extract_dates
    return extract_dates(publication)


def _final_status(publication_type: str) -> Optional[str]:
    """Only where the publication type states it. Never guessed."""
    from .contracts import CONSULTATION, FINAL_REPORT, RTS_ITS
    if publication_type == CONSULTATION:
        return "consultation"
    if publication_type in (FINAL_REPORT, RTS_ITS):
        return "final"
    return None
