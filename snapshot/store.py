"""snapshot.store — storage-neutral SnapshotStore interface.

Phase 2 snapshot/history layer. Business logic depends on this interface, never
on filesystem (or, later, Azure) specifics. Adapters implement the four storage
primitives (``register_snapshot``, ``list_snapshots``, ``get_snapshot``,
``load_loans``); the temporal *resolvers* are implemented once here in terms of
``list_snapshots`` so every adapter behaves identically.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import pandas as pd

from .model import (
    SnapshotHeader,
    SnapshotNotFoundError,
    parse_date,
)


@dataclass
class RegistrationResult:
    """Outcome of :meth:`SnapshotStore.register_snapshot`.

    ``snapshot_id`` is always present; ``created`` is ``False`` for an
    idempotent re-registration of identical content. ``issues`` carries the
    structured registration issues.
    """

    snapshot_id: str
    created: bool
    idempotent: bool
    header: SnapshotHeader
    issues: List[dict] = field(default_factory=list)


def _sorted_by_reporting_date(headers: List[SnapshotHeader]
                              ) -> List[SnapshotHeader]:
    def key(h: SnapshotHeader) -> Tuple[Any, Any]:
        rd = parse_date(h.reporting_date)
        # None reporting dates sort first; tie-break on upload_timestamp string.
        return (rd or parse_date("0001-01-01"), h.upload_timestamp or "")
    return sorted(headers, key=key)


class SnapshotStore(abc.ABC):
    """Abstract snapshot store. Adapters implement the storage primitives."""

    # -- storage primitives (adapter-specific) ----------------------------- #

    @abc.abstractmethod
    def register_snapshot(self, header: SnapshotHeader,
                          frame: pd.DataFrame) -> RegistrationResult:
        """Append-only, idempotent registration of one snapshot."""

    @abc.abstractmethod
    def list_snapshots(self, client_id: str, route: Optional[str] = None,
                       cadence: Optional[str] = None,
                       since: Any = None, until: Any = None
                       ) -> List[SnapshotHeader]:
        """Return headers for *client_id*, filtered and ordered by reporting
        date ascending."""

    @abc.abstractmethod
    def get_snapshot(self, snapshot_id: str) -> SnapshotHeader:
        """Return the header for *snapshot_id* (raises if unknown)."""

    @abc.abstractmethod
    def load_loans(self, snapshot_id: str) -> pd.DataFrame:
        """Return the loan-level frame for *snapshot_id*."""

    # -- temporal resolvers (shared, in terms of list_snapshots) ----------- #

    def resolve_latest(self, client_id: str,
                       route: Optional[str] = None) -> SnapshotHeader:
        headers = self.list_snapshots(client_id, route=route)
        if not headers:
            raise SnapshotNotFoundError(
                f"no snapshots for client={client_id!r} route={route!r}")
        return _sorted_by_reporting_date(headers)[-1]

    def resolve_as_of(self, client_id: str, reporting_date: Any,
                      route: Optional[str] = None) -> SnapshotHeader:
        target = parse_date(reporting_date)
        if target is None:
            raise SnapshotNotFoundError(f"invalid as-of date {reporting_date!r}")
        headers = [h for h in self.list_snapshots(client_id, route=route)
                   if parse_date(h.reporting_date)
                   and parse_date(h.reporting_date) <= target]
        if not headers:
            raise SnapshotNotFoundError(
                f"no snapshot on/before {target.isoformat()} for "
                f"client={client_id!r} route={route!r}")
        return _sorted_by_reporting_date(headers)[-1]

    def resolve_range(self, client_id: str, start_date: Any, end_date: Any,
                      route: Optional[str] = None) -> List[SnapshotHeader]:
        start = parse_date(start_date)
        end = parse_date(end_date)
        headers = []
        for h in self.list_snapshots(client_id, route=route):
            rd = parse_date(h.reporting_date)
            if rd is None:
                continue
            if (start is None or rd >= start) and (end is None or rd <= end):
                headers.append(h)
        return _sorted_by_reporting_date(headers)

    def resolve_compare(self, client_id: str, baseline_date: Any,
                        current_date: Any, route: Optional[str] = None
                        ) -> Tuple[SnapshotHeader, SnapshotHeader]:
        baseline = self.resolve_as_of(client_id, baseline_date, route=route)
        current = self.resolve_as_of(client_id, current_date, route=route)
        return baseline, current
