"""mi_agent.states.selectors — snapshot selection for state assembly.

Phase 3 MI state assembler. A small selector model that resolves *which*
snapshot(s) a state should be assembled from, delegating to the Phase 2
``SnapshotStore`` resolvers. Phase 3 only needs single-snapshot (point-in-time)
resolution for its core states; ``range`` / ``compare`` are provided to *prepare*
for Phase 4 temporal trends but the trend runtime itself is not built here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from snapshot.model import SnapshotHeader
from snapshot.store import SnapshotStore

LATEST = "latest"
AS_OF = "as_of"
RANGE = "range"
COMPARE = "compare"

_SINGLE_MODES = {LATEST, AS_OF}


@dataclass
class SnapshotSelector:
    """Declarative snapshot selection.

    ``mode`` is one of ``latest`` / ``as_of`` / ``range`` / ``compare``. The
    remaining fields are interpreted per mode.
    """

    client_id: str
    mode: str = LATEST
    route: Optional[str] = None
    reporting_date: Any = None          # as_of
    start_date: Any = None              # range
    end_date: Any = None               # range
    baseline_date: Any = None          # compare
    current_date: Any = None           # compare

    # -- factories --------------------------------------------------------- #

    @classmethod
    def latest(cls, client_id: str, route: Optional[str] = None) -> "SnapshotSelector":
        return cls(client_id=client_id, mode=LATEST, route=route)

    @classmethod
    def as_of(cls, client_id: str, reporting_date: Any,
              route: Optional[str] = None) -> "SnapshotSelector":
        return cls(client_id=client_id, mode=AS_OF, route=route,
                   reporting_date=reporting_date)

    @classmethod
    def range(cls, client_id: str, start_date: Any, end_date: Any,
              route: Optional[str] = None) -> "SnapshotSelector":
        return cls(client_id=client_id, mode=RANGE, route=route,
                   start_date=start_date, end_date=end_date)

    @classmethod
    def compare(cls, client_id: str, baseline_date: Any, current_date: Any,
                route: Optional[str] = None) -> "SnapshotSelector":
        return cls(client_id=client_id, mode=COMPARE, route=route,
                   baseline_date=baseline_date, current_date=current_date)

    # -- resolution -------------------------------------------------------- #

    @property
    def is_single(self) -> bool:
        return self.mode in _SINGLE_MODES

    def resolve_single(self, store: SnapshotStore) -> SnapshotHeader:
        """Resolve to exactly one header (``latest`` / ``as_of`` only)."""
        if self.mode == LATEST:
            return store.resolve_latest(self.client_id, route=self.route)
        if self.mode == AS_OF:
            return store.resolve_as_of(self.client_id, self.reporting_date,
                                       route=self.route)
        raise ValueError(
            f"resolve_single requires a single-snapshot mode, got {self.mode!r}")

    def resolve(self, store: SnapshotStore):
        """Resolve per mode: header | list[header] | (baseline, current)."""
        if self.is_single:
            return self.resolve_single(store)
        if self.mode == RANGE:
            return store.resolve_range(self.client_id, self.start_date,
                                       self.end_date, route=self.route)
        if self.mode == COMPARE:
            return store.resolve_compare(self.client_id, self.baseline_date,
                                         self.current_date, route=self.route)
        raise ValueError(f"unknown selector mode {self.mode!r}")
