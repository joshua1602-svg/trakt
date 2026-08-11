"""regulatory_watch.discovery.parsers — feed/page bytes -> RawPublication[].

Pure functions over bytes. No network, so every test replays from a committed
fixture and CI never needs outbound access.

The parsers are deliberately strict about *structure* and permissive about
*content*:

* structure — a document that is not the declared shape raises
  :class:`FeedParseError`, which the pipeline reports as ``UNEXPECTED_SCHEMA``.
  A wrong guess about what ESMA serves must never degrade into an empty item
  list, because an empty list is indistinguishable from "nothing new".
* content — a single unparseable item is skipped and counted, not fatal. One
  malformed entry in an otherwise valid feed should not blind the whole sweep.

Nothing here interprets a publication. Dates are taken verbatim where the
source states them in a recognisable form and left ``None`` otherwise.
"""

from __future__ import annotations

import re
from datetime import datetime
from html.parser import HTMLParser
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree

from .contracts import PARSER_ATOM, PARSER_HTML_LISTING, PARSER_RSS, RawPublication


class FeedParseError(ValueError):
    """The document is not the declared shape. Reported as UNEXPECTED_SCHEMA."""


_ATOM_NS = "{http://www.w3.org/2005/Atom}"

#: Date formats ESMA feeds plausibly use. Anything else stays None rather than
#: being coerced — a wrong publication date corrupts identity and dedupe.
_DATE_FORMATS = (
    "%a, %d %b %Y %H:%M:%S %z",       # RFC 822 (RSS pubDate)
    "%a, %d %b %Y %H:%M:%S %Z",
    "%Y-%m-%dT%H:%M:%S%z",            # ISO 8601 (Atom updated)
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d",
)


def parse_date(value: Optional[str]) -> Optional[str]:
    """Normalize a published date to an ISO date, or ``None``.

    Never guesses. An unrecognised format yields ``None`` so the record says
    "unknown" instead of carrying a fabricated date into the identity hash.
    """
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    # Bare ISO date embedded in a longer string (e.g. "2021-06-11 00:00:00").
    match = re.match(r"^(\d{4}-\d{2}-\d{2})", text)
    return match.group(1) if match else None


def _text(node: Optional[ElementTree.Element]) -> str:
    if node is None:
        return ""
    return " ".join((node.text or "").split())


def _root(payload: bytes) -> ElementTree.Element:
    try:
        return ElementTree.fromstring(payload)
    except ElementTree.ParseError as exc:
        raise FeedParseError(f"document is not well-formed XML: {exc}")


# --------------------------------------------------------------------------- #
# RSS
# --------------------------------------------------------------------------- #

def parse_rss(payload: bytes, source_id: str) -> List[RawPublication]:
    """Parse an RSS 2.0 document into publications."""
    root = _root(payload)
    if root.tag != "rss":
        raise FeedParseError(
            f"expected an <rss> root, found <{root.tag}>; if the source now "
            f"serves Atom, switch the declared parser rather than guessing")
    channel = root.find("channel")
    if channel is None:
        raise FeedParseError("<rss> document has no <channel>")

    out: List[RawPublication] = []
    for index, item in enumerate(channel.findall("item")):
        title = _text(item.find("title"))
        link = _text(item.find("link"))
        if not title and not link:
            continue                     # unusable entry; skipped, not fatal
        guid_node = item.find("guid")
        guid = _text(guid_node)
        out.append(RawPublication(
            source_id=source_id,
            title=title,
            url=link,
            official_identifier=guid or None,
            publication_date=parse_date(_text(item.find("pubDate"))),
            summary=_text(item.find("description")),
            categories=[_text(c) for c in item.findall("category") if _text(c)],
            linked_urls=_extract_links(_raw_text(item.find("description"))),
            raw_locator=f"channel/item[{index}]",
        ))
    return out


# --------------------------------------------------------------------------- #
# Atom
# --------------------------------------------------------------------------- #

def parse_atom(payload: bytes, source_id: str) -> List[RawPublication]:
    """Parse an Atom 1.0 document into publications."""
    root = _root(payload)
    if root.tag != f"{_ATOM_NS}feed":
        raise FeedParseError(
            f"expected an Atom <feed> root, found <{root.tag}>")

    out: List[RawPublication] = []
    for index, entry in enumerate(root.findall(f"{_ATOM_NS}entry")):
        title = _text(entry.find(f"{_ATOM_NS}title"))
        link_node = entry.find(f"{_ATOM_NS}link")
        link = (link_node.get("href") or "") if link_node is not None else ""
        if not title and not link:
            continue
        summary = (_text(entry.find(f"{_ATOM_NS}summary"))
                   or _text(entry.find(f"{_ATOM_NS}content")))
        published = (_text(entry.find(f"{_ATOM_NS}published"))
                     or _text(entry.find(f"{_ATOM_NS}updated")))
        out.append(RawPublication(
            source_id=source_id,
            title=title,
            url=link,
            official_identifier=_text(entry.find(f"{_ATOM_NS}id")) or None,
            publication_date=parse_date(published),
            summary=summary,
            categories=[c.get("term", "") for c
                        in entry.findall(f"{_ATOM_NS}category")
                        if c.get("term")],
            linked_urls=_extract_links(summary),
            raw_locator=f"feed/entry[{index}]",
        ))
    return out


# --------------------------------------------------------------------------- #
# HTML listing
# --------------------------------------------------------------------------- #

_HREF_RE = re.compile(r"""href\s*=\s*["']([^"']+)["']""", re.I)


def _raw_text(node: Optional[ElementTree.Element]) -> str:
    return (node.text or "") if node is not None else ""


def _extract_links(text: str) -> List[str]:
    """Absolute ESMA links embedded in a description/summary blob."""
    out: List[str] = []
    for href in _HREF_RE.findall(text or ""):
        if href.startswith("http") and href not in out:
            out.append(href)
    return out


class _ListingParser(HTMLParser):
    """Collects ``<a href>`` anchors with their visible text, in order."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.anchors: List[Tuple[str, str, Dict[str, str]]] = []
        self._href: Optional[str] = None
        self._attrs: Dict[str, str] = {}
        self._buffer: List[str] = []
        self.saw_html = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in ("html", "body", "div", "ul", "a"):
            self.saw_html = True
        if tag == "a":
            attributes = {k: (v or "") for k, v in attrs}
            self._href = attributes.get("href")
            self._attrs = attributes
            self._buffer = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href is not None:
            text = " ".join("".join(self._buffer).split())
            self.anchors.append((self._href, text, self._attrs))
            self._href = None
            self._attrs = {}
            self._buffer = []


#: A listing anchor is treated as a publication only when it points at an ESMA
#: document/news path. Everything else on the page (nav, cookie banners,
#: language switchers) is ignored by construction rather than by blocklist.
_PUBLICATION_PATH_RE = re.compile(
    r"^(https://(?:www\.)?esma\.europa\.eu)?/(document|press-news|"
    r"publications?|databases-library)/", re.I)


def parse_html_listing(payload: bytes, source_id: str,
                       base_url: str = "") -> List[RawPublication]:
    """Parse an ESMA publication/landing page into publications.

    Structure-strict: a payload with no HTML structure at all raises. Anchors
    that are not ESMA document links are skipped — a listing page is mostly
    navigation, and treating navigation as publications would flood triage.
    """
    try:
        text = payload.decode("utf-8", errors="replace")
    except Exception as exc:              # noqa: BLE001
        raise FeedParseError(f"listing payload is not decodable: {exc}")
    if "<" not in text:
        raise FeedParseError("listing payload contains no markup")

    parser = _ListingParser()
    try:
        parser.feed(text)
    except Exception as exc:              # noqa: BLE001 - malformed markup
        raise FeedParseError(f"listing markup could not be parsed: {exc}")
    if not parser.saw_html:
        raise FeedParseError("listing payload has no recognisable HTML "
                             "structure")

    origin = ""
    if base_url:
        match = re.match(r"^(https://[^/]+)", base_url)
        origin = match.group(1) if match else ""

    out: List[RawPublication] = []
    seen: set = set()
    for index, (href, label, attrs) in enumerate(parser.anchors):
        if not href or not _PUBLICATION_PATH_RE.match(href):
            continue
        url = href if href.startswith("http") else f"{origin}{href}"
        if url in seen or not label:
            continue
        seen.add(url)
        out.append(RawPublication(
            source_id=source_id,
            title=label,
            url=url,
            official_identifier=attrs.get("data-document-reference") or None,
            publication_date=parse_date(attrs.get("data-publication-date")),
            summary=attrs.get("title", ""),
            categories=[attrs["data-category"]] if attrs.get("data-category")
            else [],
            linked_urls=[],
            raw_locator=f"a[{index}]",
        ))
    return out


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #

_PARSERS = {
    PARSER_RSS: parse_rss,
    PARSER_ATOM: parse_atom,
    PARSER_HTML_LISTING: parse_html_listing,
}


def parse(payload: bytes, parser: str, source_id: str,
          base_url: str = "") -> List[RawPublication]:
    """Parse ``payload`` with the source's DECLARED parser.

    The parser is never sniffed from the content: a source that starts serving
    a different format must be a visible failure and a config decision, not a
    silent adaptation.
    """
    handler = _PARSERS.get(parser)
    if handler is None:
        raise FeedParseError(f"no parser registered for '{parser}'")
    if not payload:
        raise FeedParseError("empty payload")
    if handler is parse_html_listing:
        return parse_html_listing(payload, source_id, base_url=base_url)
    return handler(payload, source_id)
