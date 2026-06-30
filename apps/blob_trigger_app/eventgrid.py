"""apps.blob_trigger_app.eventgrid — pure Event Grid subject parsing.

Azure delivers blob events to the root Function App via an Event Grid
subscription. The handler needs to know (a) which container the blob landed in
and (b) the blob path within it, then decide whether to accept the event. This
module is the Azure-free core of that decision so it can be unit-tested without
a storage account.

Event Grid blob subject format:
    /blobServices/default/containers/{container}/blobs/{blob_path}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

#: Default container watched, overridable via the TRAKT_BLOB_CONTAINER app
#: setting (production uses ``raw-v2``).
DEFAULT_CONTAINER = "raw"


class EventGridSubjectError(ValueError):
    """Raised when an Event Grid subject can't be parsed (fail closed)."""


@dataclass(frozen=True)
class BlobEventRef:
    container: str
    blob_path: str          # path within the container, no leading container name
    accepted: bool          # container matches the configured one
    reason: str


def parse_blob_subject(subject: str) -> "tuple[str, str]":
    """Return ``(container, blob_path)`` from an Event Grid blob subject.

    Raises :class:`EventGridSubjectError` on an unexpected subject shape.
    """
    if not subject or "/blobs/" not in subject:
        raise EventGridSubjectError(f"unexpected subject (no /blobs/): {subject!r}")
    head, blob_path = subject.split("/blobs/", 1)
    # head: /blobServices/default/containers/{container}
    container = head.rsplit("/", 1)[-1]
    if not container or not blob_path:
        raise EventGridSubjectError(f"could not extract container/blob from {subject!r}")
    return container, blob_path


def classify_blob_event(subject: str, configured_container: str = DEFAULT_CONTAINER) -> BlobEventRef:
    """Parse the subject and decide whether the blob is in the watched container.

    Only blobs in ``configured_container`` are accepted; everything else is
    rejected (this is the fix for the legacy hardcoded ``inbound`` check — the
    accepted container is now configuration, not a constant).
    """
    container, blob_path = parse_blob_subject(subject)
    if container != configured_container:
        return BlobEventRef(
            container=container, blob_path=blob_path, accepted=False,
            reason=f"container {container!r} != configured {configured_container!r}")
    return BlobEventRef(
        container=container, blob_path=blob_path, accepted=True,
        reason=f"container {container!r} accepted")
