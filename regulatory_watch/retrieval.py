"""regulatory_watch.retrieval — isolated, optional source retrieval.

Kept deliberately separate from parsing and comparison so the whole watch runs
offline from local fixtures. Tests never reach this module's network path.

Two guarantees:

* **Allowlist only.** A URL that is not the declared ``source_url`` of a
  manifest artefact is refused. There is no crawling and no link-following.
* **A failed retrieval is never "no regulatory change".** Every failure mode
  returns a :class:`~regulatory_watch.contracts.SourceSnapshot` carrying
  ``retrieval_status = SOURCE_CHECK_FAILED``; nothing here raises past the
  caller and nothing here returns a silently-empty success.

The default fetcher is ``None``: with no fetcher injected and
``retrieval_enabled: false`` in the manifest, :func:`retrieve` reports
``SOURCE_CHECK_FAILED`` rather than pretending the source was checked.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from .contracts import SourceArtefact, SourceSnapshot
from .manifest import SourceManifest
from .snapshots import SnapshotStore

#: A fetcher takes a URL and returns ``(bytes, suffix)``. It may raise; the
#: exception is captured and converted into SOURCE_CHECK_FAILED.
Fetcher = Callable[[str], Tuple[bytes, str]]


class RetrievalRefused(RuntimeError):
    """A URL outside the allowlist was requested."""


def retrieve(artefact: SourceArtefact, manifest: SourceManifest,
             store: SnapshotStore, retrieved_at: str, parser_version: str,
             fetcher: Optional[Fetcher] = None) -> SourceSnapshot:
    """Fetch one allowlisted artefact and snapshot it.

    Returns a ``SOURCE_CHECK_FAILED`` snapshot for: retrieval disabled, no
    fetcher configured, missing/blocked URL, or any fetch error.
    """
    def failed(detail: str) -> SourceSnapshot:
        return SnapshotStore.failed_snapshot(artefact, retrieved_at,
                                             parser_version, detail)

    if not manifest.retrieval_enabled:
        return failed("retrieval disabled in the source manifest")
    if fetcher is None:
        return failed("no fetcher configured")
    if not artefact.source_url:
        return failed("artefact declares no source_url")
    if manifest.allowlist_only and artefact.source_url not in manifest.allowed_urls:
        return failed(f"url not in allowlist: {artefact.source_url}")

    try:
        payload, suffix = fetcher(artefact.source_url)
    except Exception as exc:                       # noqa: BLE001 - fail closed
        return failed(f"fetch failed: {type(exc).__name__}: {exc}")

    if not payload:
        return failed("fetch returned no content")

    return store.snapshot_bytes(artefact, payload, retrieved_at,
                                parser_version, suffix=suffix or "")


def retrieve_all(manifest: SourceManifest, store: SnapshotStore,
                 retrieved_at: str, parser_version: str,
                 fetcher: Optional[Fetcher] = None) -> List[SourceSnapshot]:
    """Retrieve every allowlisted artefact. Never raises."""
    return [retrieve(a, manifest, store, retrieved_at, parser_version, fetcher)
            for a in manifest.artefacts]
