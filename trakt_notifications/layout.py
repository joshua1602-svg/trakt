#!/usr/bin/env python3
"""Where notification state lives.

Reuses the storage technology already in the project — the ``blob://`` URI
abstraction in :mod:`apps.blob_trigger_app.storage`, which maps to Azure Blob
in the cloud and to the filesystem locally and in tests. No new datastore, no
Redis, no event bus.

Layout, all under the existing state container::

    trakt-state/notifications/recipients/{tenant}/{recipient_id}.json
    trakt-state/notifications/batches/{tenant}/{batch_id}.json
    trakt-state/notifications/reporting/{tenant}/{reporting_key}.json
    trakt-state/notifications/outbox/{tenant}/{item_id}.json

The tenant segment is part of the path rather than a field inside the document
so a listing can never cross a tenant boundary by accident: a query for one
tenant's outbox physically cannot enumerate another's.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from apps.blob_trigger_app.layout import Layout as _CoreLayout
from apps.blob_trigger_app.storage import BLOB_SCHEME, join_uri

#: Path segment sanitiser. Applied to every caller-influenced segment so a
#: tenant or recipient id can never traverse out of its prefix or inject a
#: container name.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")

ROOT = "notifications"


def safe_segment(value: str, *, fallback: str = "unknown") -> str:
    cleaned = _UNSAFE.sub("_", str(value or "")).strip("._-")
    return cleaned or fallback


@dataclass(frozen=True)
class NotificationLayout:
    """URI construction for the notification state store."""

    state_container: str = "trakt-state"

    @classmethod
    def from_env(cls) -> "NotificationLayout":
        # Deliberately derived from the EXISTING layout so a deployment that
        # renames its state container renames this too, in one place.
        return cls(state_container=_CoreLayout.from_env().state_container)

    def _root(self, *parts: str) -> str:
        return join_uri(f"{BLOB_SCHEME}{self.state_container}", ROOT, *parts)

    # -- recipients -------------------------------------------------------- #
    def recipients_prefix(self, tenant_id: str) -> str:
        return self._root("recipients", safe_segment(tenant_id))

    def recipient_uri(self, tenant_id: str, recipient_id: str) -> str:
        return join_uri(self.recipients_prefix(tenant_id),
                        f"{safe_segment(recipient_id)}.json")

    # -- generated batches (immutable once written) ------------------------ #
    def batches_prefix(self, tenant_id: str) -> str:
        return self._root("batches", safe_segment(tenant_id))

    def batch_uri(self, tenant_id: str, batch_id: str) -> str:
        return join_uri(self.batches_prefix(tenant_id),
                        f"{safe_segment(batch_id)}.json")

    # -- reporting-position index (correction detection) ------------------- #
    def reporting_uri(self, tenant_id: str, reporting_key: str) -> str:
        """Pointer from a reporting position to the batch that last described it.

        This is what makes a corrected upload recognisable as a correction of a
        specific earlier batch rather than as an unrelated new one.
        """
        return self._root("reporting", safe_segment(tenant_id),
                          f"{safe_segment(reporting_key)}.json")

    # -- outbox ------------------------------------------------------------ #
    def outbox_prefix(self, tenant_id: str) -> str:
        return self._root("outbox", safe_segment(tenant_id))

    def outbox_uri(self, tenant_id: str, item_id: str) -> str:
        return join_uri(self.outbox_prefix(tenant_id),
                        f"{safe_segment(item_id)}.json")

    def tenants_prefix(self) -> str:
        """Prefix under which per-tenant outbox folders sit.

        Used only by the operator diagnostic view, which must be able to see
        every tenant's failures; the delivery path always addresses one tenant.
        """
        return self._root("outbox")
