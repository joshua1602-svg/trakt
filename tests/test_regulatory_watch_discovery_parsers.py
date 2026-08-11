"""tests/test_regulatory_watch_discovery_parsers.py — discovery and its failures.

Covers the required discovery scenarios: a feed with no new items, one new
item, a malformed feed, an unexpected schema, an unavailable source and a
non-allowlisted source.

The property under test throughout: **a discovery failure is never an empty
item list**. An empty list means "this source published nothing"; a failure
means "we do not know what this source published". Collapsing the two is the
single most dangerous thing a regulatory watch can do.

Run: python -m pytest tests/test_regulatory_watch_discovery_parsers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from regulatory_watch.discovery.contracts import (            # noqa: E402
    CORROBORATING,
    DISCOVERY_FAILED,
    DISCOVERY_OK,
    DISCOVERY_PARTIAL,
    GATING,
    SOURCE_DISABLED,
    SOURCE_FETCH_FAILED,
    SOURCE_NOT_ALLOWLISTED,
    SOURCE_OK,
    SourceOutcome,
    overall_discovery_status,
)
from regulatory_watch.discovery.parsers import (              # noqa: E402
    FeedParseError, parse, parse_atom, parse_date, parse_html_listing,
    parse_rss,
)
from regulatory_watch.discovery.retrieval import fetch_source  # noqa: E402
from tests.helpers import regulatory_watch_discovery as fx     # noqa: E402


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #

def test_a_feed_with_no_items_yields_no_publications():
    """Valid, empty, and NOT a failure — the source genuinely published none."""
    items = parse_rss(fx.rss(), "s")
    assert items == []


def test_one_new_item_is_parsed_with_its_metadata():
    payload = fx.rss(fx.rss_item(
        "Q&A on securitisation disclosure",
        "https://www.esma.europa.eu/document/qa-x",
        pub_date="Wed, 11 Feb 2026 09:00:00 +0000",
        description="Clarifies residential real estate fields.",
        guid="esma-qa-x", categories=("Q&A", "Securitisation")))
    items = parse_rss(payload, "esma_publications_feed")
    assert len(items) == 1
    item = items[0]
    assert item.title == "Q&A on securitisation disclosure"
    assert item.url == "https://www.esma.europa.eu/document/qa-x"
    assert item.official_identifier == "esma-qa-x"
    assert item.publication_date == "2026-02-11"
    assert item.categories == ["Q&A", "Securitisation"]
    assert item.source_id == "esma_publications_feed"
    assert item.raw_locator == "channel/item[0]"


def test_the_committed_feed_fixture_parses():
    items = parse_rss(fx.feed_bytes(), "esma_publications_feed")
    assert len(items) == 6
    assert all(i.title and i.url for i in items)


def test_atom_is_parsed_when_declared():
    payload = (b'<?xml version="1.0"?>'
               b'<feed xmlns="http://www.w3.org/2005/Atom">'
               b'<entry><title>Securitisation disclosure Q&amp;A</title>'
               b'<link href="https://www.esma.europa.eu/document/a"/>'
               b'<id>urn:esma:a</id>'
               b'<updated>2026-02-11T09:00:00Z</updated>'
               b'<summary>Underlying exposures.</summary></entry></feed>')
    items = parse_atom(payload, "s")
    assert len(items) == 1
    assert items[0].publication_date == "2026-02-11"
    assert items[0].official_identifier == "urn:esma:a"


def test_html_listing_ignores_navigation_and_keeps_documents():
    items = parse_html_listing(fx.TECHNICAL_PAGE_FIXTURE.read_bytes(),
                               "esma_securitisation_technical_standards",
                               base_url="https://www.esma.europa.eu/x")
    urls = [i.url for i in items]
    assert len(items) == 2
    assert all("/document/" in u for u in urls)
    assert not any("cookies" in u or "about-esma" in u for u in urls)
    assert items[0].official_identifier == "ESMA-SECURITISATION-TS-1.3.1"
    assert items[0].publication_date == "2021-06-11"


def test_relative_listing_links_are_resolved_against_the_source_origin():
    items = parse_html_listing(fx.LANDING_FIXTURE.read_bytes(),
                               "esma_securitisation_landing",
                               base_url="https://www.esma.europa.eu/topic")
    assert items[0].url.startswith("https://www.esma.europa.eu/document/")


def test_an_item_missing_both_title_and_link_is_skipped_not_fatal():
    payload = fx.rss("<item><description>orphan</description></item>",
                     fx.rss_item("Real", "https://www.esma.europa.eu/document/r"))
    items = parse_rss(payload, "s")
    assert [i.title for i in items] == ["Real"]


# --------------------------------------------------------------------------- #
# Dates are never guessed
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("raw,expected", [
    ("Fri, 11 Jun 2021 00:00:00 +0000", "2021-06-11"),
    ("2021-06-11T09:00:00Z", "2021-06-11"),
    ("2021-06-11", "2021-06-11"),
    ("2021-06-11 00:00:00", "2021-06-11"),
])
def test_recognised_date_formats_normalise(raw, expected):
    assert parse_date(raw) == expected


@pytest.mark.parametrize("raw", ["", None, "next spring", "Q3 2026", "soon"])
def test_an_unrecognised_date_stays_unknown(raw):
    """A fabricated date corrupts identity and dedupe. Unknown is safer."""
    assert parse_date(raw) is None


# --------------------------------------------------------------------------- #
# Failure paths
# --------------------------------------------------------------------------- #

def test_a_malformed_feed_raises_rather_than_returning_nothing():
    with pytest.raises(FeedParseError) as exc:
        parse_rss(b"<rss><channel><item><title>unclosed", "s")
    assert "well-formed" in str(exc.value)


def test_atom_served_where_rss_is_declared_is_an_unexpected_schema():
    """The parser is never sniffed: a format change must be a visible failure."""
    atom = (b'<feed xmlns="http://www.w3.org/2005/Atom"><entry>'
            b'<title>x</title></entry></feed>')
    with pytest.raises(FeedParseError) as exc:
        parse(atom, "rss", "s")
    assert "expected an <rss> root" in str(exc.value)


def test_rss_without_a_channel_is_an_unexpected_schema():
    with pytest.raises(FeedParseError):
        parse_rss(b"<rss version='2.0'></rss>", "s")


def test_html_where_a_feed_is_declared_is_an_unexpected_schema():
    with pytest.raises(FeedParseError):
        parse(b"<html><body>not a feed</body></html>", "rss", "s")


def test_a_listing_payload_with_no_markup_is_refused():
    with pytest.raises(FeedParseError):
        parse_html_listing(b"just some prose", "s")


def test_an_empty_payload_is_refused():
    with pytest.raises(FeedParseError):
        parse(b"", "rss", "s")


def test_an_unregistered_parser_is_refused():
    with pytest.raises(FeedParseError):
        parse(b"<rss/>", "telepathy", "s")


# --------------------------------------------------------------------------- #
# Retrieval failures
# --------------------------------------------------------------------------- #

def test_a_source_that_is_unavailable_is_fetch_failed():
    sources = fx.source_set([fx.source("a")], retrieval_enabled=True)

    def fetcher(url):
        raise ConnectionError("connection refused")

    payload, outcome = fetch_source(sources.sources[0], sources, fx.DISCOVERY_DATE,
                                    fetcher)
    assert payload is None
    assert outcome.outcome == SOURCE_FETCH_FAILED
    assert "ConnectionError" in outcome.detail


def test_retrieval_disabled_is_fetch_failed_not_an_empty_sweep():
    sources = fx.source_set([fx.source("a")], retrieval_enabled=False)
    payload, outcome = fetch_source(sources.sources[0], sources,
                                    fx.DISCOVERY_DATE, lambda url: b"x")
    assert payload is None
    assert outcome.outcome == SOURCE_FETCH_FAILED
    assert "retrieval disabled" in outcome.detail


def test_a_non_allowlisted_source_is_refused_without_fetching():
    sources = fx.source_set([fx.source("a")], retrieval_enabled=True)
    rogue = fx.source("rogue", url="https://www.esma.europa.eu/not-declared")

    def fetcher(url):                       # pragma: no cover - must not run
        raise AssertionError(f"fetcher must not be called for {url}")

    payload, outcome = fetch_source(rogue, sources, fx.DISCOVERY_DATE, fetcher)
    assert payload is None
    assert outcome.outcome == SOURCE_NOT_ALLOWLISTED


def test_a_disabled_source_is_reported_not_omitted():
    sources = fx.source_set([fx.source("a", enabled=False)],
                            retrieval_enabled=True)
    _, outcome = fetch_source(sources.sources[0], sources, fx.DISCOVERY_DATE,
                              lambda url: b"x")
    assert outcome.outcome == SOURCE_DISABLED


def test_an_empty_response_is_fetch_failed():
    sources = fx.source_set([fx.source("a")], retrieval_enabled=True)
    _, outcome = fetch_source(sources.sources[0], sources, fx.DISCOVERY_DATE,
                              lambda url: b"")
    assert outcome.outcome == SOURCE_FETCH_FAILED


# --------------------------------------------------------------------------- #
# Discovery status resolution
# --------------------------------------------------------------------------- #

def _outcome(outcome: str, criticality: str = GATING) -> SourceOutcome:
    return SourceOutcome(source_id="s", criticality=criticality,
                         outcome=outcome)


def test_all_sources_ok_is_discovery_ok():
    assert overall_discovery_status([_outcome(SOURCE_OK),
                                     _outcome(SOURCE_OK)]) == DISCOVERY_OK


def test_a_gating_failure_is_discovery_failed():
    assert overall_discovery_status([
        _outcome(SOURCE_OK), _outcome(SOURCE_FETCH_FAILED, GATING),
    ]) == DISCOVERY_FAILED


def test_a_corroborating_failure_is_partial_not_failed():
    assert overall_discovery_status([
        _outcome(SOURCE_OK, GATING),
        _outcome(SOURCE_FETCH_FAILED, CORROBORATING),
    ]) == DISCOVERY_PARTIAL


def test_no_checkable_sources_at_all_is_discovery_failed():
    assert overall_discovery_status([_outcome(SOURCE_DISABLED)]) == \
        DISCOVERY_FAILED
    assert overall_discovery_status([]) == DISCOVERY_FAILED


def test_a_disabled_source_does_not_by_itself_fail_the_run():
    assert overall_discovery_status([
        _outcome(SOURCE_OK, GATING), _outcome(SOURCE_DISABLED, GATING),
    ]) == DISCOVERY_OK
