"""regulatory_watch.discovery.state — publication identity and deduplication.

Answers one question across weekly runs: *have we seen this exact publication
before?*

**Identity** is deterministic, stable and *source-independent*: it is keyed on
the canonical URL, which is the one thing every source agrees on for a web
publication. That matters because the same ESMA publication is routinely
reachable from more than one allowlisted source, and those sources publish
different identifiers for it — keying on the source, or on whichever identifier
a source happened to supply, would raise the same publication once per source
every single week.

**Revision** is a separate axis, tracked PER SOURCE. Two sightings with the
same identity but a different content fingerprint are the SAME publication,
REVISED — the case a corrected or replaced ESMA publication must fall into, and
which must not be confused with either a new publication or an unchanged one.

Fingerprints are kept per source because two sources describe the same
publication differently: the feed carries an editorial summary, the listing
page carries a link label. Comparing across sources would make a publication
flip between REVISED and UNCHANGED depending on which source happened to be
scanned first. A revision therefore means "this source's description of this
publication changed", which is the signal that actually indicates ESMA
re-issued something.

The store is an append-only JSON document. It records only what is needed to
recognise a publication again; it is not an archive of publication content.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlsplit, urlunsplit

from . import DISCOVERY_VERSION
from .contracts import (
    PublicationIdentity,
    RawPublication,
    SEEN_NEW,
    SEEN_REVISED,
    SEEN_UNCHANGED,
)

#: Query parameters ESMA (and most CMSes) append for analytics. They change the
#: URL without changing the publication, so they are stripped before hashing.
_VOLATILE_QUERY_PREFIXES = ("utm_", "fbclid", "gclid", "mc_cid", "mc_eid")


def canonical_url(url: str) -> str:
    """Canonicalise a URL for identity: scheme/host lowercased, tracking gone.

    Deliberately conservative — path case and trailing slash are preserved,
    because on some sites they are significant. The goal is to stop *known*
    volatile decoration from minting a duplicate, not to normalise aggressively
    and risk collapsing two genuinely different documents into one.
    """
    if not url:
        return ""
    parts = urlsplit(url.strip())
    query = "&".join(
        piece for piece in parts.query.split("&")
        if piece and not piece.split("=")[0].lower().startswith(
            _VOLATILE_QUERY_PREFIXES))
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path,
                       query, ""))


def _digest(*parts: str) -> str:
    return hashlib.sha256("␟".join(parts).encode("utf-8")).hexdigest()


def derive_identity(publication: RawPublication) -> tuple:
    """Return ``(development_id, identity_basis)`` for a publication.

    Identity is deliberately **source-independent**. The same ESMA publication
    routinely appears in more than one allowlisted source — the publications
    feed and the technical-standards page both carry the auth.099 package — and
    those sources disagree about identifiers: the feed publishes a `guid` (the
    URL), the page publishes a document reference. Keying on either the source
    or its chosen identifier would raise the same publication twice, once per
    source, every week.

    The canonical URL is therefore the primary key: it is the one thing every
    source agrees on for a web publication. An official identifier is used only
    when there is no URL to key on, and the title/date tuple only when neither
    exists.

    ``development_id`` is prefixed so a record is self-describing in a report
    or a UI without a lookup.
    """
    url = canonical_url(publication.url)
    if url:
        return f"esma-a2-{_digest('url', url)[:20]}", "canonical_url"
    official = (publication.official_identifier or "").strip()
    if official:
        return (f"esma-a2-{_digest('official', official)[:20]}",
                "official_identifier")
    return (f"esma-a2-{_digest('derived', ' '.join(publication.title.split()), publication.publication_date or '')[:20]}",
            "derived")


def content_fingerprint(publication: RawPublication) -> str:
    """Fingerprint of the publication's material content.

    Covers title, canonical URL, publication date and the linked technical
    artefacts — the things whose change means ESMA re-issued or corrected the
    item. The summary is included because a materially reworded abstract is a
    revision worth surfacing for review.
    """
    return _digest(
        " ".join(publication.title.split()),
        canonical_url(publication.url),
        publication.publication_date or "",
        " ".join(publication.summary.split()),
        "|".join(sorted(canonical_url(u) for u in publication.linked_urls)),
    )


@dataclass
class SeenRecord:
    development_id: str
    fingerprint: str                    # most recent, any source
    first_seen: str
    last_seen: str
    title: str = ""
    url: str = ""
    revision_count: int = 0
    #: source_id -> the fingerprint that source last published for this item.
    fingerprints: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = {k: getattr(self, k) for k in self.__dataclass_fields__}
        out["fingerprints"] = dict(sorted(self.fingerprints.items()))
        return out


@dataclass
class SeenStore:
    """Append-only record of publications this watch has already seen."""

    path: Path
    records: Dict[str, SeenRecord] = field(default_factory=dict)
    discovery_version: str = DISCOVERY_VERSION

    # -- load / save -------------------------------------------------------- #
    @classmethod
    def load(cls, path: Path) -> "SeenStore":
        path = Path(path)
        store = cls(path=path)
        if not path.exists():
            return store
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # A corrupt store must not silently become "we have seen nothing",
            # which would re-raise every publication as new. Refuse instead.
            raise StateStoreError(f"publication state store is unreadable: "
                                  f"{path}")
        for entry in data.get("records", []):
            known = set(SeenRecord.__dataclass_fields__)
            record = SeenRecord(**{k: v for k, v in entry.items() if k in known})
            store.records[record.development_id] = record
        store.discovery_version = str(data.get("discovery_version")
                                      or DISCOVERY_VERSION)
        return store

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "discovery_version": self.discovery_version,
            "record_count": len(self.records),
            "records": [self.records[k].to_dict()
                        for k in sorted(self.records)],
        }
        self.path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        return self.path

    # -- identity ----------------------------------------------------------- #
    def classify(self, publication: RawPublication,
                 observed_at: str) -> PublicationIdentity:
        """Identify a publication and say whether it is new/unchanged/revised.

        Read-only: call :meth:`record` to commit. Splitting the two means a run
        that fails midway cannot half-remember a publication.
        """
        development_id, basis = derive_identity(publication)
        fingerprint = content_fingerprint(publication)
        existing = self.records.get(development_id)
        if existing is None:
            return PublicationIdentity(
                development_id=development_id, fingerprint=fingerprint,
                identity_basis=basis, seen_state=SEEN_NEW,
                first_seen=observed_at)

        previous = existing.fingerprints.get(publication.source_id)
        if previous is None:
            # A source we have not seen carry this publication before. The
            # PUBLICATION is not new — we already know it — so this is not a
            # revision either; a new vantage point on a known item.
            state = SEEN_UNCHANGED
        else:
            state = (SEEN_UNCHANGED if previous == fingerprint
                     else SEEN_REVISED)
        return PublicationIdentity(
            development_id=development_id, fingerprint=fingerprint,
            identity_basis=basis, seen_state=state,
            first_seen=existing.first_seen, previous_fingerprint=previous)

    def record(self, publication: RawPublication,
               identity: PublicationIdentity, observed_at: str) -> None:
        """Commit an observation. Idempotent for an unchanged publication."""
        existing = self.records.get(identity.development_id)
        if existing is None:
            self.records[identity.development_id] = SeenRecord(
                development_id=identity.development_id,
                fingerprint=identity.fingerprint,
                first_seen=identity.first_seen or observed_at,
                last_seen=observed_at,
                title=publication.title, url=publication.url,
                fingerprints={publication.source_id: identity.fingerprint})
            return
        existing.last_seen = observed_at
        previous = existing.fingerprints.get(publication.source_id)
        existing.fingerprints[publication.source_id] = identity.fingerprint
        if previous is not None and previous != identity.fingerprint:
            existing.fingerprint = identity.fingerprint
            existing.revision_count += 1
            existing.title = publication.title
            existing.url = publication.url

    def __len__(self) -> int:
        return len(self.records)


class StateStoreError(RuntimeError):
    """The publication state store could not be read. Never downgraded.

    Treating an unreadable store as empty would re-raise every historical
    publication as new, flooding review and destroying the dedupe guarantee.
    """
