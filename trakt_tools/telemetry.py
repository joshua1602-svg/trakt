"""trakt_tools.telemetry — what one governed tool call actually cost.

Five numbers per execution, recorded by the handler and emitted on the result:

    tool               which capability ran
    duration_ms        wall clock inside the governed path
    rows_scanned       rows the governed frame held AFTER the authorisation
                       narrowing and BEFORE the tool's own selection
    rows_returned      rows (or groups) the caller actually received
    cache              hit / miss / disabled, per cached artefact touched

The pair that matters is ``rows_scanned`` against ``rows_returned``. It is the
difference between an agent investigating and an agent enumerating: twenty loans
out of four thousand scanned is a query, and four thousand out of four thousand
is a download wearing a query's clothes. Nothing here throttles that — this is
measurement, and a limit that nobody can see the need for is a limit nobody
tunes correctly.

Why on the result rather than only in a log
-------------------------------------------
A client agent cannot see Trakt's logs, and an agent that cannot see what its
last call cost has no way to choose a cheaper next one. Publishing the counters
is what lets ``rank_loans`` over 500,000 rows visibly beat ``get_loans`` in a
loop, rather than merely being documented as better.

Deliberately NOT here: no timing of individual pandas operations (that is a
profiler's job), no per-row accounting, and no sampling — five integers on a
call that already does real work cost nothing worth measuring.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("trakt_tools.telemetry")

CACHE_HIT = "hit"
CACHE_MISS = "miss"
CACHE_DISABLED = "disabled"


@dataclass
class ToolTelemetry:
    """A mutable counter for one execution.

    Mutable on purpose, and the one mutable thing a handler is given: the
    alternative is threading a return value through every helper, which would put
    measurement in the way of the code being measured.

    One instance per execution, so there is no shared state to race on. The lock
    guards only the case of a handler fanning work out across threads.
    """

    tool: str = ""
    rows_scanned: int = 0
    rows_returned: int = 0
    #: artefact name -> hit / miss / disabled
    cache: Dict[str, str] = field(default_factory=dict)
    #: Free-form counters a specific tool wants to publish (groups_returned,
    #: observations_considered, …). Integers only, so the shape stays predictable.
    counters: Dict[str, int] = field(default_factory=dict)
    _lock: Any = field(default_factory=threading.Lock, repr=False, compare=False)

    def scanned(self, rows: int) -> None:
        """Rows in the governed frame this call worked over."""
        with self._lock:
            self.rows_scanned += max(0, int(rows or 0))

    def returned(self, rows: int) -> None:
        """Rows or groups handed back to the caller."""
        with self._lock:
            self.rows_returned += max(0, int(rows or 0))

    def cached(self, artefact: str, outcome: str) -> None:
        with self._lock:
            self.cache[str(artefact)] = str(outcome)

    def count(self, name: str, value: int = 1) -> None:
        with self._lock:
            self.counters[str(name)] = self.counters.get(str(name), 0) + int(value)

    # -- publication -------------------------------------------------------- #
    @property
    def selectivity(self) -> Optional[float]:
        """Returned over scanned. ``None`` when nothing was scanned.

        The single number worth alerting on: a tool whose selectivity trends to
        1.0 is being used to extract the book rather than to ask about it.
        """
        if not self.rows_scanned:
            return None
        return round(self.rows_returned / self.rows_scanned, 6)

    def to_dict(self, *, duration_ms: Optional[int] = None) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "duration_ms": duration_ms,
            "rows_scanned": self.rows_scanned,
            "rows_returned": self.rows_returned,
            "selectivity": self.selectivity,
            "cache": dict(sorted(self.cache.items())),
            **({"counters": dict(sorted(self.counters.items()))}
               if self.counters else {}),
        }

    def emit(self, *, duration_ms: Optional[int] = None,
             tenant_id: str = "", resource: str = "",
             request_id: str = "") -> Dict[str, Any]:
        """Log the line and return the published block.

        Structured on the ``extra`` so a log pipeline can aggregate it, and
        readable in the message so a developer tailing a terminal can too.
        """
        payload = self.to_dict(duration_ms=duration_ms)
        try:
            logger.info(
                "tool=%s duration_ms=%s rows_scanned=%s rows_returned=%s "
                "selectivity=%s cache=%s",
                payload["tool"], payload["duration_ms"], payload["rows_scanned"],
                payload["rows_returned"], payload["selectivity"], payload["cache"],
                extra={"trakt_telemetry": payload, "tenant_id": tenant_id,
                       "resource": resource, "request_id": request_id})
        except Exception:  # noqa: BLE001 - telemetry must never fail a call
            pass
        return payload


class _Stopwatch:
    """Wall clock for one execution, in the units the envelope publishes."""

    def __init__(self) -> None:
        self._t0 = time.monotonic()

    @property
    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self._t0) * 1000)
