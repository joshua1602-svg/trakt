"""regulatory_watch.discovery.retrieval — isolated, optional publication fetch.

Retrieval is kept separate from parsing so the whole of Stage 1B runs offline
from committed fixtures and CI never needs outbound access. Nothing in the test
suite reaches this module's network path.

Two guarantees, both structural:

* **Allowlist only.** A URL that is not the declared ``url`` of an allowlisted
  source is refused with ``NOT_ALLOWLISTED``. There is no crawling, no
  link-following and no redirect to an off-allowlist host.
* **A failed fetch is never "nothing new".** Every failure mode returns a
  :class:`~regulatory_watch.discovery.contracts.SourceOutcome` carrying an
  explicit non-OK outcome. Nothing raises past the caller, and nothing returns
  a silently-empty success.

The default fetcher is ``None``: with no fetcher injected and
``retrieval_enabled: false`` in the allowlist, :func:`fetch_source` reports
``FETCH_FAILED`` rather than pretending the source was checked.
"""

from __future__ import annotations

import hashlib
from typing import Callable, List, Optional, Tuple

from .contracts import (
    PublicationSource,
    SOURCE_DISABLED,
    SOURCE_FETCH_FAILED,
    SOURCE_NOT_ALLOWLISTED,
    SOURCE_OK,
    SourceOutcome,
)
from .sources import PublicationSourceSet

#: A fetcher takes a URL and returns the response bytes. It may raise; the
#: exception is captured and converted into FETCH_FAILED.
Fetcher = Callable[[str], bytes]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fetch_source(source: PublicationSource, sources: PublicationSourceSet,
                 retrieved_at: str, fetcher: Optional[Fetcher] = None,
                 ) -> Tuple[Optional[bytes], SourceOutcome]:
    """Fetch one allowlisted source. Returns ``(payload_or_None, outcome)``.

    Never raises. A disabled source is reported as ``DISABLED`` rather than
    omitted, so the report always accounts for every declared source.
    """
    def outcome(status: str, detail: str = "", payload: bytes = b"") -> SourceOutcome:
        return SourceOutcome(
            source_id=source.source_id, criticality=source.criticality,
            outcome=status, detail=detail, retrieved_at=retrieved_at,
            content_sha256=sha256_bytes(payload) if payload else "")

    if not source.enabled:
        return None, outcome(SOURCE_DISABLED, "source disabled in the allowlist")
    if sources.allowlist_only and not sources.is_allowlisted(source.url):
        return None, outcome(SOURCE_NOT_ALLOWLISTED,
                             f"url not in allowlist: {source.url}")
    if not sources.retrieval_enabled:
        return None, outcome(SOURCE_FETCH_FAILED,
                             "retrieval disabled in the publication allowlist")
    if fetcher is None:
        return None, outcome(SOURCE_FETCH_FAILED, "no fetcher configured")

    try:
        payload = fetcher(source.url)
    except Exception as exc:              # noqa: BLE001 - fail closed
        return None, outcome(SOURCE_FETCH_FAILED,
                             f"fetch failed: {type(exc).__name__}: {exc}")
    if not payload:
        return None, outcome(SOURCE_FETCH_FAILED, "fetch returned no content")
    return payload, outcome(SOURCE_OK, payload=payload)


def fetch_all(sources: PublicationSourceSet, retrieved_at: str,
              fetcher: Optional[Fetcher] = None,
              ) -> List[Tuple[PublicationSource, Optional[bytes], SourceOutcome]]:
    """Fetch every declared source. Never raises; never omits a source."""
    return [(s,) + fetch_source(s, sources, retrieved_at, fetcher)
            for s in sources.sources]
