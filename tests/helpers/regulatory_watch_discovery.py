"""tests.helpers.regulatory_watch_discovery — Stage 1B fixtures and builders.

Deterministic: no randomness, no timestamps, no network. Every helper is a pure
function of its inputs so a scenario replays byte-identically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

from regulatory_watch.discovery.contracts import (
    GATING,
    PublicationSource,
    RawPublication,
    SOURCE_OK,
    SourceOutcome,
)
from regulatory_watch.discovery.pipeline import SourcePayload
from regulatory_watch.discovery.sources import PublicationSourceSet

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = (REPO_ROOT / "tests" / "fixtures" / "regulatory_watch"
               / "publications")

FEED_FIXTURE = FIXTURE_DIR / "esma_publications_feed.xml"
TECHNICAL_PAGE_FIXTURE = (FIXTURE_DIR
                          / "esma_securitisation_technical_standards.html")
LANDING_FIXTURE = FIXTURE_DIR / "esma_securitisation_landing.html"

DISCOVERY_DATE = "2026-02-16T00:00:00Z"

#: The REAL Stage 1A technical-source URL. Using the actual URL is what makes
#: the linkage tests prove something rather than assert against a stub.
STAGE1A_PACKAGE_URL = ("https://www.esma.europa.eu/document/"
                       "securitisation-reporting-technical-standards-"
                       "xml-schema-and-instructions")

#: A REAL historical ESMA publication: config/system/esma_code_order.yaml cites
#: this document as the source of the ESMA field ordering Trakt implements.
REAL_FINAL_REPORT_URL = ("https://www.esma.europa.eu/document/"
                         "final-report-draft-rts-and-its-securitisation-"
                         "reporting")


def publication(title: str, url: str = "https://www.esma.europa.eu/document/x",
                *, source_id: str = "esma_publications_feed",
                summary: str = "", publication_date: Optional[str] = None,
                categories: Optional[Sequence[str]] = None,
                linked_urls: Optional[Sequence[str]] = None,
                official_identifier: Optional[str] = None) -> RawPublication:
    """A publication with only the fields a scenario cares about."""
    return RawPublication(
        source_id=source_id, title=title, url=url,
        official_identifier=official_identifier,
        publication_date=publication_date, summary=summary,
        categories=list(categories or []), linked_urls=list(linked_urls or []),
        raw_locator="test")


def source(source_id: str, *, parser: str = "rss",
           criticality: str = GATING, enabled: bool = True,
           url: Optional[str] = None) -> PublicationSource:
    return PublicationSource(
        source_id=source_id, source_type="rss_feed",
        url=url or f"https://www.esma.europa.eu/{source_id}",
        parser=parser, criticality=criticality, enabled=enabled)


def source_set(sources: Sequence[PublicationSource], *,
               retrieval_enabled: bool = False) -> PublicationSourceSet:
    return PublicationSourceSet(
        manifest_id="test", regime="ESMA_Annex2", authority="ESMA", version=1,
        retrieval_enabled=retrieval_enabled, allowlist_only=True,
        default_frequency="weekly", sources=list(sources), path="<test>")


def payload(src: PublicationSource, data: bytes,
            retrieved_at: str = DISCOVERY_DATE) -> SourcePayload:
    return SourcePayload(src, data, SourceOutcome(
        source_id=src.source_id, criticality=src.criticality,
        outcome=SOURCE_OK, retrieved_at=retrieved_at))


def failed_payload(src: PublicationSource, outcome: str, detail: str = "",
                   retrieved_at: str = DISCOVERY_DATE) -> SourcePayload:
    return SourcePayload(src, None, SourceOutcome(
        source_id=src.source_id, criticality=src.criticality,
        outcome=outcome, detail=detail, retrieved_at=retrieved_at))


def feed_bytes() -> bytes:
    return FEED_FIXTURE.read_bytes()


def rss(*items: str, channel_title: str = "ESMA publications") -> bytes:
    """Build a minimal RSS document from item bodies."""
    body = "\n".join(items)
    return (f'<?xml version="1.0" encoding="utf-8"?>\n'
            f'<rss version="2.0"><channel><title>{channel_title}</title>'
            f'{body}</channel></rss>').encode("utf-8")


def _xml(text: str) -> str:
    """Escape text for XML. Titles legitimately contain '&' (e.g. "Q&A")."""
    from xml.sax.saxutils import escape
    return escape(text)


def rss_item(title: str, link: str, *, pub_date: str = "",
             description: str = "", guid: str = "",
             categories: Sequence[str] = ()) -> str:
    parts = [f"<title>{_xml(title)}</title>", f"<link>{_xml(link)}</link>"]
    if guid:
        parts.append(f"<guid>{_xml(guid)}</guid>")
    if pub_date:
        parts.append(f"<pubDate>{_xml(pub_date)}</pubDate>")
    if description:
        parts.append(f"<description>{_xml(description)}</description>")
    parts.extend(f"<category>{_xml(c)}</category>" for c in categories)
    return "<item>" + "".join(parts) + "</item>"


#: Stage 1A allowlist stand-in for linkage tests that must not depend on the
#: live Stage 1A config staying byte-identical.
KNOWN_TECHNICAL_URLS: Dict[str, List[str]] = {
    STAGE1A_PACKAGE_URL: ["esma_annex2_message_workbook",
                          "esma_annex2_xml_schema"],
}
