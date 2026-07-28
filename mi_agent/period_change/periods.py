"""mi_agent.period_change.periods — governed two-snapshot period resolution.

Turns "what changed this month", "since the previous reporting date", "between
March and June" or "compare 2025-12-31 with 2026-03-31" into exactly TWO
governed snapshots, or into an explicit failure.

Design rules, all from §5 of the workflow specification:

* the platform's reporting dates are **irregular** — nothing here assumes a
  calendar month end. An EXPLICITLY requested date resolves to the latest
  snapshot at or before it, never forward to a period that had not yet occurred;
  a relative mode (month-on-month, …) whose target is derived arithmetically
  takes the absolute nearest. Either way the adjustment and its size in days are
  recorded, never hidden, and a gap beyond the governed ceiling is refused;
* every resolution records what was asked for, what it resolved to, how, and
  whether either end moved;
* the failure cases are enumerated and explicit. In particular the current
  snapshot is never silently compared with itself.

Pure and deterministic: it reads a list of snapshots and a request, and returns
a :class:`~mi_agent.period_change.models.PeriodResolution`. No I/O.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .models import (
    FAIL_AMBIGUOUS_PERIOD_RANGE,
    FAIL_IDENTICAL_SNAPSHOTS,
    FAIL_INSUFFICIENT_SNAPSHOTS,
    FAIL_PORTFOLIO_ABSENT_AT_PERIOD,
    FAIL_REVERSED_PERIOD_RANGE,
    METHOD_CURRENT_VS_PREVIOUS,
    METHOD_EXPLICIT_DATES,
    METHOD_LATEST_AVAILABLE_PAIR,
    METHOD_MONTH_ON_MONTH,
    METHOD_QUARTER_ON_QUARTER,
    METHOD_YEAR_ON_YEAR,
    METHOD_YEAR_TO_DATE,
    FlowBasis,
    PeriodChangeFailure,
    PeriodResolution,
    PortfolioScopeRef,
    SnapshotFrame,
)

_MONTH_NAMES: Mapping[str, int] = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

_ISO_DAY = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_ISO_MONTH = re.compile(r"^(\d{4})-(\d{2})$")
_DMY = re.compile(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})$")
#: "31 March 2026" / "31st March 2026" / "March 2026" / "March"
_TEXT_DATE = re.compile(
    r"^(?:(\d{1,2})(?:st|nd|rd|th)?\s+)?([a-z]+)(?:\s+(\d{4}))?$")

GRANULARITY_DAY = "day"
GRANULARITY_MONTH = "month"

#: The relative modes this resolver understands, mapped to the month offset the
#: start period sits at behind the end period. ``None`` means the offset is not
#: a fixed month count (previous snapshot, year-to-date).
RELATIVE_MONTH_OFFSET: Mapping[str, Optional[int]] = {
    METHOD_CURRENT_VS_PREVIOUS: None,
    METHOD_MONTH_ON_MONTH: 1,
    METHOD_QUARTER_ON_QUARTER: 3,
    METHOD_YEAR_ON_YEAR: 12,
    METHOD_YEAR_TO_DATE: None,
}


@dataclass(frozen=True)
class PeriodRequest:
    """What the question asked for, before any resolution.

    ``relative_mode`` is one of the ``METHOD_*`` relative constants; when it is
    set, ``requested_start``/``requested_end`` are usually absent. When neither
    is supplied the resolver falls back to the latest available pair, which is
    the only defensible reading of a bare "what changed?".
    """

    requested_start: Optional[str] = None
    requested_end: Optional[str] = None
    relative_mode: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"requested_start": self.requested_start,
                "requested_end": self.requested_end,
                "relative_mode": self.relative_mode}


# --------------------------------------------------------------------------- #
# Date helpers — no third-party date library, deterministic by construction
# --------------------------------------------------------------------------- #
def _last_day(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def _is_month_end(value: date) -> bool:
    return value.day == calendar.monthrange(value.year, value.month)[1]


def shift_months(value: date, months: int) -> date:
    """Shift by whole months, preserving a month-end.

    A reporting date of 31 March shifted back one month must be 28/29 February,
    not 28 March — anything else would resolve a month-on-month request to the
    wrong snapshot on every 31-day month.
    """
    total = (value.year * 12 + (value.month - 1)) - months
    year, month = divmod(total, 12)
    month += 1
    if _is_month_end(value):
        return _last_day(year, month)
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def parse_date(text: Optional[str]) -> Optional[date]:
    """Parse an ISO reporting date. ``None`` when absent or unparseable."""
    if not text:
        return None
    raw = str(text).strip()[:10]
    m = _ISO_DAY.match(raw)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    m = _ISO_MONTH.match(raw)
    if m:
        try:
            return _last_day(int(m.group(1)), int(m.group(2)))
        except ValueError:
            return None
    return None


def parse_period_token(token: Optional[str], *, reference: Optional[date] = None
                       ) -> Tuple[Optional[date], str]:
    """``(target_date, granularity)`` for a requested period token.

    A month without a year resolves inside ``reference``'s year — the year of the
    most recent governed snapshot — because that is the only year a reader can
    mean without saying so. A month resolves to its LAST day, matching how
    reporting dates are expressed.
    """
    if not token:
        return None, GRANULARITY_DAY
    raw = str(token).strip().lower().rstrip(".")
    if not raw:
        return None, GRANULARITY_DAY

    iso = parse_date(raw)
    if iso is not None:
        return iso, (GRANULARITY_MONTH if _ISO_MONTH.match(raw[:7]) and
                     len(raw) == 7 else GRANULARITY_DAY)

    m = _DMY.match(raw)
    if m:
        try:
            return date(int(m.group(3)), int(m.group(2)), int(m.group(1))), GRANULARITY_DAY
        except ValueError:
            return None, GRANULARITY_DAY

    m = _TEXT_DATE.match(raw)
    if m:
        month = _MONTH_NAMES.get(m.group(2))
        if month is None:
            return None, GRANULARITY_DAY
        year = int(m.group(3)) if m.group(3) else (
            reference.year if reference else None)
        if year is None:
            return None, GRANULARITY_DAY
        if m.group(1):
            try:
                return date(year, month, int(m.group(1))), GRANULARITY_DAY
            except ValueError:
                return None, GRANULARITY_DAY
        return _last_day(year, month), GRANULARITY_MONTH

    return None, GRANULARITY_DAY


# --------------------------------------------------------------------------- #
# Snapshot ordering and matching
# --------------------------------------------------------------------------- #
def _dated(snapshots: Sequence[SnapshotFrame]
           ) -> Tuple[Tuple[SnapshotFrame, Optional[date]], ...]:
    return tuple((s, parse_date(s.reporting_date)) for s in snapshots)


def order_snapshots(snapshots: Sequence[SnapshotFrame]
                    ) -> Tuple[SnapshotFrame, ...]:
    """Oldest → newest. Undated snapshots keep their given order, at the end.

    Sorting on the reporting date rather than trusting the caller's order means
    a resolver decision never depends on how a dataset service happened to walk
    a directory.
    """
    dated = [(s, d) for s, d in _dated(snapshots) if d is not None]
    undated = [s for s, d in _dated(snapshots) if d is None]
    dated.sort(key=lambda pair: (pair[1], pair[0].snapshot_id))
    return tuple([s for s, _ in dated] + undated)


def _guard_gap(gap_days: int, target: date, snap_date: date, label: str,
               max_gap_days: Optional[int]) -> None:
    """Refuse a resolution that is further from the request than the ceiling.

    Without this a request for January against a book holding only June 2025 and
    June 2026 silently answers about a snapshot six months away.
    """
    if max_gap_days is None or gap_days <= max_gap_days:
        return
    raise PeriodChangeFailure(
        FAIL_AMBIGUOUS_PERIOD_RANGE,
        f"The requested {label} date {target.isoformat()} is {gap_days} days "
        f"from the nearest usable governed snapshot "
        f"({snap_date.isoformat()}), beyond the {max_gap_days}-day limit for "
        f"this portfolio. The comparison is not performed rather than answered "
        f"about a different period.",
        detail={"requested": target.isoformat(),
                "resolved": snap_date.isoformat(),
                "gap_days": gap_days, "max_snapshot_gap_days": max_gap_days})


def _on_or_before(snapshots: Sequence[SnapshotFrame], target: date, *,
                  granularity: str, label: str,
                  max_gap_days: Optional[int] = None
                  ) -> Tuple[SnapshotFrame, bool, Optional[str], int]:
    """The latest snapshot AT OR BEFORE ``target``: ``(snap, adjusted, note, gap)``.

    The rule for an explicitly requested date: never resolve forward to a period
    that had not yet occurred when the caller's date fell. Absolute-nearest
    matching would step over a period boundary — a request for 15 January lands
    on 31 December because it is one day closer — which answers about a
    different reporting period than the one asked for.
    """
    candidates = [(s, d) for s, d in _dated(snapshots) if d is not None]
    if not candidates:
        raise PeriodChangeFailure(
            FAIL_AMBIGUOUS_PERIOD_RANGE,
            f"No governed snapshot carries a reporting date, so the requested "
            f"{label} period could not be resolved.")

    eligible = [(s, d) for s, d in candidates if d <= target]
    if not eligible:
        earliest = min(d for _s, d in candidates)
        raise PeriodChangeFailure(
            FAIL_AMBIGUOUS_PERIOD_RANGE,
            f"The requested {label} date {target.isoformat()} precedes every "
            f"governed snapshot (the earliest is {earliest.isoformat()}). A "
            f"later snapshot is not substituted for a period that had not yet "
            f"occurred.",
            detail={"requested": target.isoformat(),
                    "earliest_snapshot": earliest.isoformat()})

    snap, snap_date = max(eligible, key=lambda pair: (pair[1], pair[0].snapshot_id))
    gap_days = (target - snap_date).days
    _guard_gap(gap_days, target, snap_date, label, max_gap_days)

    adjusted = snap_date != target
    note = None
    if adjusted:
        in_month = [d for _s, d in candidates
                    if (d.year, d.month) == (target.year, target.month)]
        detail = ("" if not in_month or granularity != GRANULARITY_MONTH else
                  f" ({len(in_month)} snapshot(s) fall inside the requested month)")
        note = (f"Requested {label} period {target.isoformat()} resolved to the "
                f"latest snapshot at or before it, {snap_date.isoformat()} "
                f"({gap_days} days earlier){detail}.")
    return snap, adjusted, note, gap_days


def _nearest(snapshots: Sequence[SnapshotFrame], target: date, *,
             granularity: str, label: str,
             max_gap_days: Optional[int] = None
             ) -> Tuple[SnapshotFrame, bool, Optional[str], int]:
    """The snapshot closest to ``target``: ``(snapshot, adjusted, note, gap)``.

    Used for RELATIVE modes only, whose target date is derived arithmetically:
    on a book whose month end is the 30th, a month-on-month target of the 30th
    must be allowed to match a 31st. Explicit requests use
    :func:`_on_or_before` instead.

    A month-granularity request prefers a snapshot inside that calendar month
    before falling back to the nearest one. Two snapshots equidistant from a
    day-precise request is genuinely ambiguous and fails rather than guessing.
    """
    candidates = [(s, d) for s, d in _dated(snapshots) if d is not None]
    if not candidates:
        raise PeriodChangeFailure(
            FAIL_AMBIGUOUS_PERIOD_RANGE,
            f"No governed snapshot carries a reporting date, so the requested "
            f"{label} period could not be resolved.")

    if granularity == GRANULARITY_MONTH:
        in_month = [(s, d) for s, d in candidates
                    if (d.year, d.month) == (target.year, target.month)]
        if len(in_month) == 1:
            snap, snap_date = in_month[0]
            return snap, snap_date != target, None, abs((snap_date - target).days)
        if len(in_month) > 1:
            # Several snapshots inside the requested month: the month end is the
            # governed reading of "that month".
            snap, snap_date = max(in_month, key=lambda pair: pair[1])
            return snap, snap_date != target, (
                f"{len(in_month)} snapshots fall inside the requested {label} "
                f"month; the latest ({snap_date.isoformat()}) was used."
            ), abs((snap_date - target).days)

    distances = sorted(
        ((abs((d - target).days), d, s) for s, d in candidates),
        key=lambda item: (item[0], item[1], item[2].snapshot_id))
    best_distance = distances[0][0]
    tied = [item for item in distances if item[0] == best_distance]
    if len(tied) > 1 and granularity == GRANULARITY_DAY:
        raise PeriodChangeFailure(
            FAIL_AMBIGUOUS_PERIOD_RANGE,
            f"The requested {label} date {target.isoformat()} is equidistant "
            f"from {len(tied)} governed snapshots "
            f"({', '.join(item[1].isoformat() for item in tied)}); it cannot be "
            f"resolved without guessing which was meant.",
            detail={"requested": target.isoformat(),
                    "equidistant": [item[1].isoformat() for item in tied]})

    gap_days, snap_date, snap = distances[0]
    _guard_gap(gap_days, target, snap_date, label, max_gap_days)
    adjusted = snap_date != target
    note = None
    if adjusted:
        note = (f"Requested {label} period {target.isoformat()} adjusted to the "
                f"nearest available snapshot {snap_date.isoformat()} "
                f"({gap_days} days away).")
    return snap, adjusted, note, gap_days


def _latest_on_or_before(snapshots: Sequence[SnapshotFrame], target: date
                         ) -> Optional[Tuple[SnapshotFrame, date]]:
    eligible = [(s, d) for s, d in _dated(snapshots) if d is not None and d <= target]
    if not eligible:
        return None
    return max(eligible, key=lambda pair: (pair[1], pair[0].snapshot_id))


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #
def _portfolio_presence(snapshot: SnapshotFrame,
                        wanted: Sequence[str]) -> Tuple[str, ...]:
    """Requested portfolios that this snapshot does not contain.

    An empty ``portfolio_ids`` on the snapshot means the dataset carries no
    source-portfolio provenance at all. That is an absence of evidence, not
    evidence of absence, so it is not reported as a missing portfolio — the
    workflow discloses the un-narrowed scope instead.
    """
    if not wanted or not snapshot.portfolio_ids:
        return ()
    present = set(snapshot.portfolio_ids)
    return tuple(p for p in wanted if p not in present)


def reporting_period_days(ordered: Sequence[SnapshotFrame],
                          snapshot: SnapshotFrame) -> Optional[int]:
    """The length of ``snapshot``'s own reporting period, in days.

    Derived from the gap to its immediate predecessor in the governed series —
    the only evidence the platform holds about how long a ``period_flow`` value
    covers. ``None`` when the snapshot is the earliest held, or is undated.
    """
    target = parse_date(snapshot.reporting_date)
    if target is None:
        return None
    earlier = [d for _s, d in _dated(ordered) if d is not None and d < target]
    if not earlier:
        return None
    return (target - max(earlier)).days


def resolve_flow_basis(ordered: Sequence[SnapshotFrame],
                       start: SnapshotFrame, end: SnapshotFrame, *,
                       tolerance_days: int = 5) -> FlowBasis:
    """Whether the two snapshots' reporting periods are of comparable length."""
    start_days = reporting_period_days(ordered, start)
    end_days = reporting_period_days(ordered, end)
    if start_days is None or end_days is None:
        return FlowBasis(start_days, end_days, tolerance_days, None)
    return FlowBasis(start_days, end_days, tolerance_days,
                     abs(start_days - end_days) <= tolerance_days)


def resolve_periods(snapshots: Sequence[SnapshotFrame],
                    request: PeriodRequest, *,
                    scope: Optional[PortfolioScopeRef] = None,
                    max_gap_days: Optional[int] = None,
                    flow_basis_tolerance_days: int = 5,
                    ) -> PeriodResolution:
    """Resolve a period request to two governed snapshots, or fail explicitly."""
    scope = scope or PortfolioScopeRef()
    ordered = order_snapshots(snapshots)
    if len(ordered) < 2:
        raise PeriodChangeFailure(
            FAIL_INSUFFICIENT_SNAPSHOTS,
            detail={"available_snapshots": [s.snapshot_id for s in ordered]})

    available = tuple(s.snapshot_id for s in ordered)
    latest_date = parse_date(ordered[-1].reporting_date)
    notes: List[str] = []
    gaps: Dict[str, Optional[int]] = {"start": None, "end": None}

    start, end, method, start_adjusted, end_adjusted = _resolve_pair(
        ordered, request, latest_date, notes, max_gap_days, gaps)

    start_date = parse_date(start.reporting_date)
    end_date = parse_date(end.reporting_date)
    if start_date and end_date and start_date > end_date:
        raise PeriodChangeFailure(
            FAIL_REVERSED_PERIOD_RANGE,
            detail={"resolved_start": start.reporting_date,
                    "resolved_end": end.reporting_date})
    if start.snapshot_id == end.snapshot_id:
        raise PeriodChangeFailure(
            FAIL_IDENTICAL_SNAPSHOTS,
            detail={"snapshot_id": start.snapshot_id,
                    "resolution_method": method,
                    "available_snapshots": list(available)})

    wanted = tuple(scope.portfolio_ids)
    missing_start = _portfolio_presence(start, wanted)
    missing_end = _portfolio_presence(end, wanted)
    if missing_start or missing_end:
        raise PeriodChangeFailure(
            FAIL_PORTFOLIO_ABSENT_AT_PERIOD,
            detail={"missing_at_start": list(missing_start),
                    "missing_at_end": list(missing_end),
                    "start_snapshot": start.snapshot_id,
                    "end_snapshot": end.snapshot_id})

    flow_basis = resolve_flow_basis(ordered, start, end,
                                    tolerance_days=flow_basis_tolerance_days)
    if flow_basis.comparable is False:
        notes.append(
            f"The two snapshots cover reporting periods of different length "
            f"({flow_basis.start_period_days} and {flow_basis.end_period_days} "
            f"days). Period-flow fields are reported but not compared.")

    return PeriodResolution(
        requested_start=request.requested_start,
        requested_end=request.requested_end,
        resolution_method=method,
        start_snapshot=start, end_snapshot=end,
        start_adjusted=start_adjusted, end_adjusted=end_adjusted,
        adjustment_notes=tuple(notes),
        available_snapshots=available,
        portfolio_scope=scope,
        start_gap_days=gaps["start"], end_gap_days=gaps["end"],
        max_gap_days=max_gap_days, flow_basis=flow_basis)


def _resolve_pair(ordered: Sequence[SnapshotFrame], request: PeriodRequest,
                  latest_date: Optional[date], notes: List[str],
                  max_gap_days: Optional[int],
                  gaps: Dict[str, Optional[int]]
                  ) -> Tuple[SnapshotFrame, SnapshotFrame, str, bool, bool]:
    """The (start, end, method, start_adjusted, end_adjusted) decision."""
    # 1. Explicit dates win: the caller named the periods.
    if request.requested_start or request.requested_end:
        return _resolve_explicit(ordered, request, latest_date, notes,
                                 max_gap_days, gaps)

    mode = request.relative_mode
    # 2. Previous governed snapshot — no calendar arithmetic at all.
    if mode in (None, METHOD_CURRENT_VS_PREVIOUS, METHOD_LATEST_AVAILABLE_PAIR):
        method = mode or METHOD_LATEST_AVAILABLE_PAIR
        return ordered[-2], ordered[-1], method, False, False

    if mode == METHOD_YEAR_TO_DATE:
        return _resolve_ytd(ordered, latest_date, notes)

    offset = RELATIVE_MONTH_OFFSET.get(mode)
    if offset is None:
        raise PeriodChangeFailure(
            FAIL_AMBIGUOUS_PERIOD_RANGE,
            f"Period mode {mode!r} is not a governed period-resolution method.")

    end = ordered[-1]
    if latest_date is None:
        raise PeriodChangeFailure(
            FAIL_AMBIGUOUS_PERIOD_RANGE,
            "The latest governed snapshot carries no reporting date, so a "
            f"{mode.replace('_', '-')} comparison cannot be resolved.")
    target = shift_months(latest_date, offset)
    start, adjusted, note, gap = _nearest(
        ordered[:-1], target, granularity=GRANULARITY_DAY, label="start",
        max_gap_days=max_gap_days)
    gaps["start"] = gap
    if note:
        notes.append(note)
    return start, end, mode, adjusted, False


def _resolve_explicit(ordered: Sequence[SnapshotFrame], request: PeriodRequest,
                      latest_date: Optional[date], notes: List[str],
                      max_gap_days: Optional[int],
                      gaps: Dict[str, Optional[int]]
                      ) -> Tuple[SnapshotFrame, SnapshotFrame, str, bool, bool]:
    start_target, start_grain = parse_period_token(
        request.requested_start, reference=latest_date)
    end_target, end_grain = parse_period_token(
        request.requested_end, reference=latest_date)

    if request.requested_start and start_target is None:
        raise PeriodChangeFailure(
            FAIL_AMBIGUOUS_PERIOD_RANGE,
            f"The requested start period {request.requested_start!r} could not "
            f"be resolved to a date.")
    if request.requested_end and end_target is None:
        raise PeriodChangeFailure(
            FAIL_AMBIGUOUS_PERIOD_RANGE,
            f"The requested end period {request.requested_end!r} could not be "
            f"resolved to a date.")
    if start_target and end_target and start_target > end_target:
        raise PeriodChangeFailure(
            FAIL_REVERSED_PERIOD_RANGE,
            detail={"requested_start": start_target.isoformat(),
                    "requested_end": end_target.isoformat()})

    if end_target is None:
        end, end_adjusted = ordered[-1], False
    else:
        end, end_adjusted, note, gap = _on_or_before(
            ordered, end_target, granularity=end_grain, label="end",
            max_gap_days=max_gap_days)
        gaps["end"] = gap
        if note:
            notes.append(note)

    if start_target is None:
        # Only an end date was named: the governed reading is "versus the
        # snapshot immediately before it".
        earlier = [s for s in ordered
                   if s.snapshot_id != end.snapshot_id
                   and (parse_date(s.reporting_date) or date.min)
                   < (parse_date(end.reporting_date) or date.max)]
        if not earlier:
            raise PeriodChangeFailure(
                FAIL_INSUFFICIENT_SNAPSHOTS,
                "There is no governed snapshot before the requested end period.",
                detail={"end_snapshot": end.snapshot_id})
        return earlier[-1], end, METHOD_EXPLICIT_DATES, False, end_adjusted

    # The start is resolved against the FULL snapshot list, independently of the
    # end. Excluding the end snapshot here would quietly slide a "compare 30 June
    # with 30 June" request onto the previous period instead of reporting that
    # both dates resolve to the same snapshot — which is the failure §5 requires.
    start, start_adjusted, note, gap = _on_or_before(
        ordered, start_target, granularity=start_grain, label="start",
        max_gap_days=max_gap_days)
    gaps["start"] = gap
    if note:
        notes.append(note)
    return start, end, METHOD_EXPLICIT_DATES, start_adjusted, end_adjusted


def _resolve_ytd(ordered: Sequence[SnapshotFrame], latest_date: Optional[date],
                 notes: List[str]
                 ) -> Tuple[SnapshotFrame, SnapshotFrame, str, bool, bool]:
    """Year-to-date: the movement from the position at the year opening."""
    end = ordered[-1]
    if latest_date is None:
        raise PeriodChangeFailure(
            FAIL_AMBIGUOUS_PERIOD_RANGE,
            "The latest governed snapshot carries no reporting date, so a "
            "year-to-date comparison cannot be resolved.")
    opening = date(latest_date.year - 1, 12, 31)
    found = _latest_on_or_before(ordered[:-1], opening)
    if found is not None:
        snap, snap_date = found
        adjusted = snap_date != opening
        if adjusted:
            notes.append(
                f"Year-to-date opening position taken from the latest snapshot "
                f"on or before {opening.isoformat()} ({snap_date.isoformat()}).")
        return snap, end, METHOD_YEAR_TO_DATE, adjusted, False

    # No snapshot before the year opening: use the earliest snapshot in the
    # current year and say so, rather than silently reporting a partial year as
    # a full one.
    in_year = [(s, d) for s, d in _dated(ordered[:-1])
               if d is not None and d.year == latest_date.year]
    if not in_year:
        raise PeriodChangeFailure(
            FAIL_INSUFFICIENT_SNAPSHOTS,
            "A year-to-date comparison needs a snapshot at or before the start "
            "of the year; none is available.",
            detail={"end_snapshot": end.snapshot_id})
    snap, snap_date = min(in_year, key=lambda pair: (pair[1], pair[0].snapshot_id))
    notes.append(
        f"No governed snapshot exists at or before {opening.isoformat()}; the "
        f"year-to-date opening position is the earliest snapshot in "
        f"{latest_date.year} ({snap_date.isoformat()}), so the movement covers "
        f"part of the year only.")
    return snap, end, METHOD_YEAR_TO_DATE, True, False
