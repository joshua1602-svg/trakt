#!/usr/bin/env python3
"""Slice 2: one governed measure, across governed snapshots. Offline only.

WHAT THIS ADDS TO SLICE 1, AND WHAT IT DELIBERATELY DOES NOT.

Slice 1 hands a `GovernedQueryPlan` to `execute_mi_query` against ONE frame the
caller already resolved. Slice 2 changes exactly one thing: WHICH frames, and
how many. The measure, the predicates, the axes, the statistic and the
`MIQuerySpec` binding are the accepted slice 1 ones — `adapter.spec_for_plan` is
called ONCE and the SAME spec runs against every selected snapshot, so a filter
or an axis cannot survive on one period and vanish on another without the
per-snapshot receipt saying so.

It adds NO calculation owner. `execute_mi_query` computes every figure, exactly
as it does today. The only arithmetic in this module is a subtraction and a
division for a period comparison, over two figures the executor produced, and it
is written here rather than delegated because the estate's existing
period-comparison owner (`mi_agent.states.temporal._compare_total`) can express
only `loan_count` and `balance_sum` and could not carry an average LTV.

THREE OWNERSHIP RULES, AND THIS MODULE IS SUBORDINATE TO ALL OF THEM.

    1. The PLAN owns requested semantics. Nothing here reads a question: it is
       not a parameter of any function in this module, the module imports no
       parser, recogniser or router, and it imports no `re` — so it cannot
       pattern-match a sentence even by accident.
    2. `SnapshotStore` / `SnapshotSelector` own snapshot resolution. This module
       chooses no file, no snapshot id and no date. It translates the plan's
       `PeriodBinding` into a `SnapshotSelector`, and the SELECTOR resolves —
       every date it ever puts on a selector was read out of a header the
       catalogue returned, never computed from a calendar.
    3. The RECEIPT owns what actually executed. Reconciliation per snapshot is
       `adapter.reconcile_receipt`, the same structural check the slice 1B
       serving canary uses.

FAIL CLOSED, NEVER NARROW. A period the catalogue cannot honour is
`PERIOD_NOT_AVAILABLE`; a period label this contract does not recognise is
`PERIOD_LABEL_UNRESOLVED`; a cadence the catalogue does not carry is
`UNSUPPORTED_CADENCE`. None of them fall back to the latest snapshot, to a
shorter window, or to a row-level date filter over one frame. There is no code
path in this module that filters rows by a date column, because simulating a
missing snapshot from the rows of a present one is the single failure this
architecture exists to prevent.

Nothing here serves a user. There is no flag, no envelope and no call site in
`mi_service`: slice 2 is an offline capability with an acceptance bank, and
wiring it to a request is a later decision made with evidence in hand.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from mi_agent import plan_runtime_adapter as adapter
from mi_agent.states.selectors import SnapshotSelector
from snapshot.model import SnapshotNotFoundError

# --------------------------------------------------------------------------- #
# the slice 2 eligibility perimeter
# --------------------------------------------------------------------------- #

#: Operations slice 2 serves. Every one of them is a SINGLE measure evaluated
#: per snapshot: `series` and `compare` state the temporal shape themselves,
#: `point_in_time` and `breakdown` take the shape their period form implies.
#: `movement`, `bridge` and `distribution` are absent on purpose — a movement is
#: an attribution of a change to its causes, which is a different question from
#: what this module computes and is owned elsewhere.
SLICE_2_OPERATIONS = frozenset({"point_in_time", "breakdown", "series", "compare"})

#: Period forms slice 2 serves. `current` is deliberately EXCLUDED: a
#: current-period plan is slice 1's, and keeping the two perimeters disjoint
#: means no question can be claimed by both and no slice 1 disposition can
#: change because this module exists.
SLICE_2_PERIOD_FORMS = frozenset({
    "explicit_period", "range", "series", "previous_reporting_period",
    "relative_pair"})

#: The funded book, and only the funded book. Slice 1 never needed this test
#: because its caller resolved the frame by view; slice 2 SELECTS its own
#: frames, so the population base becomes this module's to honour.
SLICE_2_POPULATION_BASES = frozenset({"funded"})

#: Operations whose period form is fixed by the operation itself. A `series`
#: over a period PAIR, or a `compare` over an open span, is a question whose
#: shape and whose window disagree; answering either would mean choosing one and
#: dropping the other.
_SPAN_FORMS = frozenset({"series", "range"})
_PAIR_FORMS = frozenset({"previous_reporting_period", "relative_pair"})
_OPERATION_REQUIRED_FORMS: Mapping[str, frozenset] = {
    "series": _SPAN_FORMS,
    "compare": _PAIR_FORMS,
}

# Ineligibility and unresolvability reasons. Stable strings: the bank groups on
# them and the report counts them.
OPERATION_NOT_TEMPORAL = "OPERATION_NOT_TEMPORAL"
PERIOD_NOT_TEMPORAL = "PERIOD_NOT_TEMPORAL"
POPULATION_NOT_FUNDED = "POPULATION_NOT_FUNDED"
OPERATION_PERIOD_MISMATCH = "OPERATION_PERIOD_MISMATCH"
UNSUPPORTED_CADENCE = "UNSUPPORTED_CADENCE"
PERIOD_NOT_AVAILABLE = "PERIOD_NOT_AVAILABLE"
PERIOD_LABEL_UNRESOLVED = "PERIOD_LABEL_UNRESOLVED"
PERIOD_LABEL_AMBIGUOUS = "PERIOD_LABEL_AMBIGUOUS"
NO_SNAPSHOTS = "NO_SNAPSHOTS"
INSUFFICIENT_SNAPSHOTS = "INSUFFICIENT_SNAPSHOTS"
SNAPSHOT_LOAD_FAILED = "SNAPSHOT_LOAD_FAILED"
EXECUTION_FAILED = "EXECUTION_FAILED"
RECONCILIATION_FAILED = "RECONCILIATION_FAILED"
EMPTY_ACROSS_EVERY_SNAPSHOT = "EMPTY_ACROSS_EVERY_SNAPSHOT"

#: Reasons a human could settle by asking a more precise question. Everything
#: else is a refusal. The split mirrors `interpretation_v2.outcomes`: it is data,
#: not a judgement made per call.
CLARIFIABLE_REASONS = frozenset({
    PERIOD_NOT_AVAILABLE, PERIOD_LABEL_UNRESOLVED, PERIOD_LABEL_AMBIGUOUS,
    INSUFFICIENT_SNAPSHOTS, UNSUPPORTED_CADENCE,
})

#: The column the assembled series is keyed by. The governed header's reporting
#: date, never a row-level date field — naming it once keeps it that way.
REPORTING_DATE = "reporting_date"

# The three response shapes this slice may produce.
SHAPE_SERIES = "time_series"
SHAPE_POINT = "single_period"
SHAPE_COMPARISON = "period_comparison"


def check_temporal_eligibility(plan: Any) -> Tuple[bool, str, str]:
    """`(eligible, reason, detail)` for slice 2. Reads the plan only.

    Reuses the slice 1 capability test and the slice 1 STRUCTURAL test
    byte-for-byte — one measure, at most two bound non-time axes, bound
    predicates one per field, no geography axis, no explicit lens, no comparison
    of populations, no target, one output. Only the temporal perimeter differs,
    and it differs by admitting exactly the forms this module can resolve
    against a governed snapshot catalogue.
    """
    body = adapter._as_mapping(plan)
    ok, reason, detail = adapter.check_capability(plan)
    if not ok:
        return ok, reason, detail

    operation = body.get("operation")
    if operation not in SLICE_2_OPERATIONS:
        return (False, OPERATION_NOT_TEMPORAL,
                f"operation={operation!r} is not a single-measure evaluation "
                f"across snapshots")

    period = body.get("period") or {}
    form = period.get("form")
    if form not in SLICE_2_PERIOD_FORMS:
        return (False, PERIOD_NOT_TEMPORAL,
                f"period.form={form!r} is not a governed temporal span or pair "
                f"this slice resolves")

    required = _OPERATION_REQUIRED_FORMS.get(str(operation))
    if required is not None and form not in required:
        return (False, OPERATION_PERIOD_MISMATCH,
                f"operation={operation!r} needs a period in {sorted(required)}, "
                f"and the plan states {form!r}")

    base = str(((body.get("population") or {}).get("base") or "")).strip().lower()
    if base not in SLICE_2_POPULATION_BASES:
        return (False, POPULATION_NOT_FUNDED,
                f"population.base={base!r}; this slice selects funded-book "
                f"snapshots and no other dataset")

    return adapter.check_structure(plan)


# --------------------------------------------------------------------------- #
# the governed temporal vocabulary: plan label -> governed period anchor
# --------------------------------------------------------------------------- #

#: Labels that mean "every governed reporting period the book carries". A CLOSED
#: set, because the alternative to recognising a span phrase is not guessing at
#: it: it is clarifying. Matched after `_normalise`, so casing, articles and
#: trailing punctuation do not multiply the entries.
WHOLE_SERIES_LABELS = frozenset({
    "each month", "every month", "month by month", "monthly", "each period",
    "every period", "each reporting period", "every reporting period",
    "by month", "by period", "over time", "across time", "month on month",
    "per month", "each of the last months", "the whole period",
})

#: Words that prefix a span label without changing which period it names.
#: Stripped so "since March" and "March" resolve to the same governed anchor.
_ANCHOR_PREFIXES = ("since ", "from ", "starting ", "beginning ", "as at ",
                    "as of ", "in ", "during ", "for ", "the ", "back to ")

_MONTHS: Mapping[str, int] = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12,
    "dec": 12,
}

#: Cadence words that mean the same reporting rhythm. `SnapshotHeader.cadence`
#: and `vocabulary.TIME_GRAINS` spell it differently in places, and a mismatch
#: of spelling is not a mismatch of cadence.
_CADENCE_SYNONYMS: Mapping[str, str] = {
    "monthly": "monthly", "month": "monthly",
    "weekly": "weekly", "week": "weekly",
    "daily": "daily", "day": "daily",
    "quarterly": "quarterly", "quarter": "quarterly",
    "annual": "annual", "annually": "annual", "yearly": "annual",
    "year": "annual",
}


def _normalise(label: Any) -> str:
    """A label reduced to the words that name a period. No regular expressions.

    THE ONE STRING THIS MODULE READS is `period.labels`, and it is governed plan
    content rather than the reader's sentence: the compiler put it there, and
    `interpretation_v2.intent` already refuses a payload whose label carries a
    date or a snapshot id. Nothing else on the plan is read as prose.
    """
    words = str(label or "").strip().lower()
    for character in ".,;:!?'\"()[]":
        words = words.replace(character, " ")
    return " ".join(words.split())


def _cadence(value: Any) -> str:
    return _CADENCE_SYNONYMS.get(_normalise(value), _normalise(value))


@dataclass(frozen=True)
class PeriodAnchor:
    """A month, and optionally a year, that a plan label named.

    Never a date. A month with no year is matched against the months the
    catalogue actually carries, and it is the CATALOGUE that supplies the day.
    """

    month: int
    year: Optional[int] = None
    label: str = ""


def parse_anchor(label: Any) -> Optional[PeriodAnchor]:
    """The governed period a plan label names, or None if it names none.

    Recognises a month name, optionally with a four-digit year, after the
    prefixes a span phrase puts in front of it. Everything else returns None,
    and the caller clarifies rather than choosing a period on the reader's
    behalf.

    A four-digit year alone is NOT an anchor: "2025" names twelve reporting
    periods, and picking one of them would be exactly the substitution this
    module refuses.
    """
    words = _normalise(label)
    if not words:
        return None
    changed = True
    while changed:
        changed = False
        for prefix in _ANCHOR_PREFIXES:
            if words.startswith(prefix):
                words = words[len(prefix):]
                changed = True
    tokens = [token for token in words.split() if token]
    month: Optional[int] = None
    year: Optional[int] = None
    for token in tokens:
        if token in _MONTHS:
            if month is not None and _MONTHS[token] != month:
                return None                      # two months in one label
            month = _MONTHS[token]
            continue
        if len(token) == 4 and token.isdigit():
            candidate = int(token)
            if 1900 <= candidate <= 2999:
                if year is not None and year != candidate:
                    return None
                year = candidate
                continue
        return None                              # a word this contract cannot read
    if month is None:
        return None
    return PeriodAnchor(month=month, year=year, label=_normalise(label))


# --------------------------------------------------------------------------- #
# plan period -> snapshot selector
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TemporalResolution:
    """How a plan's period became a set of governed snapshots, and on what basis."""

    ok: bool
    reason: str = ""
    detail: str = ""
    shape: str = ""
    selector: Optional[SnapshotSelector] = None
    headers: Tuple[Any, ...] = ()
    #: `count` | `anchor` | `whole_series` | `pair` — how the window was chosen.
    basis: str = ""
    #: What the plan asked for, transcribed. Never re-derived from a sentence.
    requested: Mapping[str, Any] = field(default_factory=dict)

    @property
    def clarifiable(self) -> bool:
        return self.reason in CLARIFIABLE_REASONS

    @property
    def snapshot_ids(self) -> Tuple[str, ...]:
        return tuple(str(h.snapshot_id) for h in self.headers)

    @property
    def reporting_dates(self) -> Tuple[str, ...]:
        return tuple(str(h.reporting_date) for h in self.headers)


def requested_temporal_semantics(plan: Any) -> Dict[str, Any]:
    """The plan's temporal request, transcribed. A read, not an interpretation."""
    body = adapter._as_mapping(plan)
    period = body.get("period") or {}
    return {
        "period_form": period.get("form"),
        "period_labels": list(period.get("labels") or ()),
        "period_grain": period.get("grain"),
        "period_periods_back": period.get("periods_back"),
        "period_contract": period.get("contract"),
    }


def _fail(reason: str, detail: str, requested: Mapping[str, Any]
          ) -> TemporalResolution:
    return TemporalResolution(ok=False, reason=reason, detail=detail,
                              requested=dict(requested))


def _catalogue(store: Any, client_id: str, route: Optional[str]) -> List[Any]:
    """Every governed header for this book, ascending. The store's own order."""
    headers = [h for h in store.list_snapshots(client_id, route=route)
               if h.reporting_date]
    return sorted(headers, key=lambda h: (str(h.reporting_date),
                                          str(h.upload_timestamp or "")))


def declared_cadence(headers: Sequence[Any]) -> Optional[str]:
    """The one cadence this catalogue declares, or None if it declares 0 or 2.

    One owner for the question, because two callers now ask it: the guard below,
    which refuses a grain the catalogue cannot honour, and `_resolve_span`, which
    reads a bare grain as a statement of the whole series only when the
    catalogue is actually keeping that rhythm.
    """
    found = {_cadence(h.cadence) for h in headers}
    found.discard("")
    return next(iter(found)) if len(found) == 1 else None


def _cadence_check(period: Mapping[str, Any], headers: Sequence[Any],
                   requested: Mapping[str, Any]) -> Optional[TemporalResolution]:
    """The plan's grain against the cadence the CATALOGUE declares.

    A catalogue that declares no cadence cannot honour a stated grain, and
    answering anyway would be asserting a rhythm nobody recorded. That is
    stricter than the estate has been before, and it is the direction the
    no-silent-substitution rule points: a weekly question against a monthly book
    must not come back monthly with the word "weekly" still in it.
    """
    grain = _cadence(period.get("grain"))
    if not grain:
        return None
    if declared_cadence(headers) == grain:
        return None
    declared = sorted({_cadence(h.cadence) for h in headers} - {""})
    return _fail(UNSUPPORTED_CADENCE,
                 f"the plan states grain={grain!r}; the catalogue declares "
                 f"cadence {declared or ['none']}", requested)


def _match_anchor(anchor: PeriodAnchor, headers: Sequence[Any]) -> List[Any]:
    """Headers whose reporting month IS the anchor's. Never the nearest one."""
    found = []
    for header in headers:
        date = str(header.reporting_date or "")
        parts = date.split("-")
        if len(parts) < 2:
            continue
        try:
            year, month = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if month != anchor.month:
            continue
        if anchor.year is not None and year != anchor.year:
            continue
        found.append(header)
    return found


def _resolve_anchor(labels: Sequence[str], headers: Sequence[Any],
                    requested: Mapping[str, Any]
                    ) -> Tuple[Optional[Any], Optional[TemporalResolution]]:
    """`(header, failure)` for the one governed period the labels name."""
    anchors = [a for a in (parse_anchor(label) for label in labels)
               if a is not None]
    if not anchors:
        return None, _fail(
            PERIOD_LABEL_UNRESOLVED,
            f"no governed period is named by {list(labels)!r}; this contract "
            f"reads a month, optionally with a year, and clarifies otherwise",
            requested)
    distinct = {(a.month, a.year) for a in anchors}
    if len(distinct) > 1:
        return None, _fail(PERIOD_LABEL_AMBIGUOUS,
                           f"{list(labels)!r} names more than one period",
                           requested)
    anchor = anchors[0]
    matches = _match_anchor(anchor, headers)
    if not matches:
        return None, _fail(
            PERIOD_NOT_AVAILABLE,
            f"the catalogue carries no reporting period for {anchor.label!r}; "
            f"it carries {[str(h.reporting_date) for h in headers]}", requested)
    if len(matches) > 1:
        # A bare month against a book with more than one of them. Choosing the
        # most recent would answer a question the reader did not ask in the
        # years they did not name.
        return None, _fail(
            PERIOD_LABEL_AMBIGUOUS,
            f"{anchor.label!r} names a month the catalogue carries "
            f"{len(matches)} times "
            f"({[str(h.reporting_date) for h in matches]}); name the year",
            requested)
    return matches[0], None


def resolve_temporal(plan: Any, store: Any, *, client_id: str,
                     route: Optional[str] = None) -> TemporalResolution:
    """A plan's governed period -> the snapshots that answer it. Deterministic.

    Inputs are the PLAN and the CATALOGUE. There is no third input, and in
    particular there is no question.

    Every date this function puts on a `SnapshotSelector` was read off a header
    the catalogue returned. It never constructs one from a calendar, so a
    selector cannot name a reporting date the book does not have.
    """
    body = adapter._as_mapping(plan)
    period = body.get("period") or {}
    requested = requested_temporal_semantics(plan)
    form = str(period.get("form") or "")
    operation = str(body.get("operation") or "")
    labels = [str(x) for x in (period.get("labels") or ())]
    periods_back = period.get("periods_back")

    try:
        headers = _catalogue(store, client_id, route)
    except Exception as exc:                                         # noqa: BLE001
        return _fail(NO_SNAPSHOTS,
                     f"the catalogue could not be read: "
                     f"{type(exc).__name__}: {exc}"[:200], requested)
    if not headers:
        return _fail(NO_SNAPSHOTS,
                     f"no governed snapshots for client={client_id!r} "
                     f"route={route!r}", requested)

    cadence_failure = _cadence_check(period, headers, requested)
    if cadence_failure is not None:
        return cadence_failure

    if form in _PAIR_FORMS:
        return _resolve_pair(form, operation, periods_back, headers, requested,
                             client_id=client_id, route=route, store=store)
    if form == "explicit_period":
        header, failure = _resolve_anchor(labels, headers, requested)
        if failure is not None:
            return failure
        # `as_of` on the header's OWN reporting date, so the selector resolves
        # the very snapshot the anchor matched rather than a neighbour.
        return TemporalResolution(
            ok=True, shape=SHAPE_POINT, basis="anchor",
            selector=SnapshotSelector.as_of(client_id, header.reporting_date,
                                            route=route),
            headers=(header,), requested=requested)
    if form in _SPAN_FORMS:
        return _resolve_span(form, labels, periods_back, headers, requested,
                             client_id=client_id, route=route, store=store,
                             operation=operation,
                             grain=_cadence(period.get("grain")),
                             catalogue_cadence=declared_cadence(headers))
    return _fail(PERIOD_NOT_TEMPORAL, f"period.form={form!r}", requested)


def _resolve_pair(form: str, operation: str, periods_back: Any,
                  headers: Sequence[Any], requested: Mapping[str, Any], *,
                  client_id: str, route: Optional[str], store: Any
                  ) -> TemporalResolution:
    """The period N back, and — where the question named two — the current one.

    THE TWO PAIR FORMS DO NOT NAME THE SAME THING.
    `previous_reporting_period` names ONE governed period, the one before the
    latest; the vocabulary's own note records that saying "two periods" is the
    job of `relative_pair`. So a measure asked of the previous period is a
    single-period answer about that period, and only a `compare` — or the
    explicitly paired form — puts the current period beside it. Reading the
    singular form as a pair would answer with a change nobody asked for; reading
    the paired form as a point would drop half the question.

    `previous_reporting_period` is N = 1 by definition, so a count is neither
    required nor invented.
    """
    steps = 1
    if form == "relative_pair" and isinstance(periods_back, int) \
            and not isinstance(periods_back, bool) and periods_back >= 1:
        steps = periods_back
    needed = steps + 1
    selector = SnapshotSelector.last_n(client_id, needed, route=route)
    try:
        chosen = selector.resolve(store)
    except SnapshotNotFoundError as exc:
        return _fail(INSUFFICIENT_SNAPSHOTS,
                     f"a period {steps} back needs {needed} governed "
                     f"snapshots: {exc}", requested)
    if form == "previous_reporting_period" and operation != "compare":
        return TemporalResolution(
            ok=True, shape=SHAPE_POINT, basis="previous", selector=selector,
            headers=(chosen[0],), requested=requested)
    return TemporalResolution(
        ok=True, shape=SHAPE_COMPARISON, basis="pair", selector=selector,
        headers=(chosen[0], chosen[-1]), requested=requested)


def _resolve_span(form: str, labels: Sequence[str], periods_back: Any,
                  headers: Sequence[Any], requested: Mapping[str, Any], *,
                  client_id: str, route: Optional[str], store: Any,
                  operation: str, grain: str = "",
                  catalogue_cadence: Optional[str] = None) -> TemporalResolution:
    """A span of governed reporting periods: a count, an anchor, a cadence, or none.

    A STATED COUNT WINS OVER A LABEL, and the labels are then not read at all.
    "the last six months" carries both the count and the words for it, and
    reading the words as a second statement of span would let one paraphrase
    override another for a window they both describe. A span whose label is
    present but unreadable still clarifies; it never widens to the whole book,
    because a widened window is an answer to a question nobody asked.

    A BARE CADENCE IS A STATED SPAN, and that was this module's defect. The live
    run measured Opus emitting, for all four "each month" questions,
    `{form: series, grain: monthly, labels: [], periods_back: null}` — the
    rhythm stated and the window left open, which is what "each month" means.
    `compiler._bind_period` accepts exactly that and emits a plan: it raises
    AMBIGUOUS_PERIOD only when a span has NONE of labels, count or grain. This
    function then refused the plan the compiler had just authorised, so two
    governed layers disagreed about what a stated span is. The disagreement was
    slice 2's, and it is resolved here in the compiler's favour.

    It is not a widening. There is no narrower window being passed over: the
    request names a rhythm and no bound, and "every governed period at that
    rhythm" is the only window that answers it. The guards stay:
    `_cadence_check` has already refused a grain the catalogue cannot honour,
    and this branch independently requires the grain to equal the cadence the
    catalogue DECLARES, so a rhythm the book is not keeping still fails closed.
    """
    if isinstance(periods_back, int) and not isinstance(periods_back, bool) \
            and periods_back >= 1:
        selector = SnapshotSelector.last_n(client_id, periods_back, route=route)
        try:
            chosen = selector.resolve(store)
        except SnapshotNotFoundError as exc:
            return _fail(PERIOD_NOT_AVAILABLE, str(exc), requested)
        return TemporalResolution(ok=True, shape=SHAPE_SERIES, basis="count",
                                  selector=selector, headers=tuple(chosen),
                                  requested=requested)

    normalised = [_normalise(label) for label in labels]
    if any(label in WHOLE_SERIES_LABELS for label in normalised):
        selector = SnapshotSelector.range(client_id, None, None, route=route)
        chosen = selector.resolve(store)
        return TemporalResolution(ok=True, shape=SHAPE_SERIES,
                                  basis="whole_series", selector=selector,
                                  headers=tuple(chosen), requested=requested)

    if labels:
        start, failure = _resolve_anchor(labels, headers, requested)
        if failure is not None:
            return failure
        selector = SnapshotSelector.range(client_id, start.reporting_date, None,
                                          route=route)
        chosen = selector.resolve(store)
        return TemporalResolution(ok=True, shape=SHAPE_SERIES, basis="anchor",
                                  selector=selector, headers=tuple(chosen),
                                  requested=requested)

    # No count, and no label at all. A `series` that nevertheless names the
    # catalogue's OWN cadence has stated its window: every period the book keeps
    # at that rhythm. `range` is deliberately excluded — a range states bounds,
    # and one with neither bound and no label is an incomplete request rather
    # than an open-ended one.
    if form == "series" and grain and grain == catalogue_cadence:
        selector = SnapshotSelector.range(client_id, None, None, route=route)
        chosen = selector.resolve(store)
        return TemporalResolution(ok=True, shape=SHAPE_SERIES,
                                  basis="cadence", selector=selector,
                                  headers=tuple(chosen), requested=requested)

    return _fail(PERIOD_LABEL_UNRESOLVED,
                 "the plan states a span with no period count, no period label "
                 "and no cadence the catalogue keeps; the window it names "
                 "cannot be settled", requested)


# --------------------------------------------------------------------------- #
# execution: the slice 1 owner, once per resolved snapshot
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SnapshotPoint:
    """One snapshot's execution: its identity, its figure, and its receipt."""

    snapshot_id: str
    reporting_date: str
    value: Optional[float] = None
    cells: Tuple[Mapping[str, Any], ...] = ()
    cells_note: Optional[str] = None
    receipt: Mapping[str, Any] = field(default_factory=dict)
    warnings: Tuple[str, ...] = ()
    empty: bool = False


@dataclass(frozen=True)
class TemporalOutcome:
    """One slice 2 attempt, start to finish. Never a user-visible answer."""

    eligible: bool
    reason: str = ""
    detail: str = ""
    plan_id: str = ""
    shape: str = ""
    basis: str = ""
    requested: Mapping[str, Any] = field(default_factory=dict)
    spec: Any = None
    resolution: Optional[TemporalResolution] = None
    points: Tuple[SnapshotPoint, ...] = ()
    comparison: Optional[Mapping[str, Any]] = None
    reconciled: bool = False
    error: str = ""

    @property
    def executed(self) -> bool:
        return bool(self.eligible and not self.error and self.points)

    @property
    def clarifiable(self) -> bool:
        return self.reason in CLARIFIABLE_REASONS

    @property
    def snapshot_ids(self) -> Tuple[str, ...]:
        return tuple(p.snapshot_id for p in self.points)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "eligible": self.eligible, "reason": self.reason,
            "detail": self.detail, "plan_id": self.plan_id,
            "shape": self.shape, "basis": self.basis,
            "requested": dict(self.requested),
            "bound_spec": (self.spec.to_dict() if self.spec is not None else None),
            "resolution": {
                "selector_mode": (self.resolution.selector.mode
                                  if self.resolution and self.resolution.selector
                                  else None),
                "snapshot_ids": list(self.resolution.snapshot_ids)
                                if self.resolution else [],
                "reporting_dates": list(self.resolution.reporting_dates)
                                   if self.resolution else [],
                "n_snapshots": len(self.resolution.headers) if self.resolution else 0,
            },
            "points": [{
                "snapshot_id": p.snapshot_id,
                "reporting_date": p.reporting_date,
                "value": p.value,
                "cells": [dict(c) for c in p.cells],
                "cells_note": p.cells_note,
                "empty": p.empty,
                "receipt": dict(p.receipt),
                "warnings": list(p.warnings),
            } for p in self.points],
            "comparison": dict(self.comparison) if self.comparison else None,
            "reconciled": self.reconciled,
            "error": self.error,
        }


def claims(plan: Any) -> bool:
    """Whether the TEMPORAL runtime owns this plan. One structural read.

    The dispatch decision the serving path makes, and it is not a judgement:
    slice 1's perimeter accepts `period.form == "current"` and nothing else, and
    `SLICE_2_PERIOD_FORMS` excludes `current` and nothing else, so the two are
    disjoint by construction and the PLAN says which runtime owns it. No new
    semantic owner appears at the seam, and no question is read to reach this.

    True does NOT mean eligible. A temporal plan outside the slice 2 contract is
    claimed here and then refused by `check_temporal_eligibility`, which is the
    point: it is refused with a TEMPORAL reason rather than falling through to
    slice 1 and being refused for not being current.
    """
    period = adapter._as_mapping(plan).get("period") or {}
    return period.get("form") in SLICE_2_PERIOD_FORMS


def series_frame(outcome: "TemporalOutcome", spec: Any) -> Any:
    """The whole temporal result as ONE frame: a period per row, plus any axes.

    So the existing response contract can carry a series without a parallel
    envelope. Every figure in it was produced by `execute_mi_query` on its own
    snapshot; this only stacks them, and the reporting date it stacks them by is
    the governed header's own, never a row-level date column.

    The value column is named exactly as the executor names it
    (`adapter.value_column`), so the display hints, the chart factory and the
    renderer read it as they read any grouped result.
    """
    import pandas as pd

    column = adapter.value_column(spec)
    dimensions = list(getattr(spec, "dimensions", None) or ())
    rows: List[Dict[str, Any]] = []
    for point in outcome.points:
        if dimensions:
            for cell in point.cells:
                row = {REPORTING_DATE: point.reporting_date}
                row.update({axis: cell.get(axis) for axis in dimensions})
                row[column] = cell.get("value")
                rows.append(row)
        else:
            rows.append({REPORTING_DATE: point.reporting_date,
                         column: point.value})
    return pd.DataFrame(rows, columns=[REPORTING_DATE, *dimensions, column])


def served_evidence(outcome: "TemporalOutcome") -> Dict[str, Any]:
    """What was REQUESTED and what was EXECUTED, for post-execution governance.

    The same two-sided shape slice 1B puts on `metadata.governedPlan`, widened
    to say it per snapshot. A governance layer reading this can establish the
    requested temporal semantics, the exact snapshots selected, the predicates
    applied on each, the grouping, the measure and the resulting figures —
    without re-reading the question, which is the whole point of carrying it.
    """
    resolution = outcome.resolution
    return {
        "shape": outcome.shape,
        "basis": outcome.basis,
        "selector_mode": (resolution.selector.mode
                          if resolution is not None and resolution.selector
                          else None),
        "snapshot_count": len(outcome.points),
        "snapshots": [{
            "snapshot_id": point.snapshot_id,
            "reporting_date": point.reporting_date,
            "value": point.value,
            "cells": [dict(cell) for cell in point.cells],
            "applied_predicates": point.receipt.get("applied_predicates") or [],
            "group_field_keys": list(point.receipt.get("group_field_keys") or ()),
            "aggregation": point.receipt.get("aggregation"),
            "filtered_row_count": point.receipt.get("filtered_row_count"),
            "empty": point.empty,
        } for point in outcome.points],
        "comparison": dict(outcome.comparison) if outcome.comparison else None,
    }


def _receipt_of(result: Any) -> Dict[str, Any]:
    metadata = dict(getattr(result, "metadata", None) or {})
    return {
        "aggregation": metadata.get("aggregation"),
        "group_field_keys": list(metadata.get("group_field_keys") or ()),
        "applied_predicates": metadata.get("applied_predicates") or [],
        "input_row_count": metadata.get("input_row_count"),
        "filtered_row_count": metadata.get("filtered_row_count"),
        "balance_field_used": metadata.get("balance_field_used"),
        "result_type": getattr(result, "result_type", None),
        "row_count": getattr(result, "row_count", None),
    }


def _change(baseline: Optional[float], current: Optional[float]
            ) -> Dict[str, Any]:
    """Absolute and percentage change, computed here and by nothing else.

    A percentage change off a zero baseline is undefined and is reported as
    None. It is not reported as zero, and it is not reported as infinite: both
    would be this module asserting something the arithmetic does not say.
    """
    if baseline is None or current is None:
        return {"absolute_change": None, "percent_change": None,
                "note": "a period produced no comparable figure"}
    absolute = float(current) - float(baseline)
    percent = (None if float(baseline) == 0.0
               else absolute / float(baseline) * 100.0)
    return {"absolute_change": absolute, "percent_change": percent,
            "note": ("" if percent is not None
                     else "the baseline is zero; percentage change is undefined")}


def _cell_key(cell: Mapping[str, Any], dimensions: Sequence[str]
              ) -> Tuple[str, ...]:
    return tuple(str(cell.get(dimension)) for dimension in dimensions)


def _grouped_comparison(baseline: SnapshotPoint, current: SnapshotPoint,
                        dimensions: Sequence[str]) -> List[Dict[str, Any]]:
    """Per-cell change across two snapshots, over the union of their groups.

    A group present on one side only is carried with the other side None rather
    than zero. "This band did not exist last month" and "this band held nothing
    last month" are different facts, and an outer join that filled zeros would
    report the first as the second.
    """
    left = {_cell_key(c, dimensions): c.get("value") for c in baseline.cells}
    right = {_cell_key(c, dimensions): c.get("value") for c in current.cells}
    rows: List[Dict[str, Any]] = []
    for key in sorted(set(left) | set(right)):
        row: Dict[str, Any] = dict(zip(dimensions, key))
        row["baseline_value"] = left.get(key)
        row["current_value"] = right.get(key)
        row.update(_change(left.get(key), right.get(key)))
        rows.append(row)
    return rows


def execute_temporal_plan(plan: Any, *, store: Any, client_id: str,
                          semantics: Any, route: Optional[str] = None
                          ) -> TemporalOutcome:
    """Validate, resolve, execute per snapshot, and assemble. Raises nothing.

    `store` is the governed snapshot catalogue; `semantics` is the governed
    field registry the caller already loaded — the same two objects slice 1's
    caller supplies, minus the single frame it no longer chooses.
    """
    body = adapter._as_mapping(plan)
    plan_id = str(body.get("plan_id") or "")
    requested = dict(adapter.requested_semantics(plan) if body else {})
    requested.update(requested_temporal_semantics(plan))

    eligible, reason, detail = check_temporal_eligibility(plan)
    if not eligible:
        return TemporalOutcome(eligible=False, reason=reason, detail=detail,
                               plan_id=plan_id, requested=requested)

    resolution = resolve_temporal(plan, store, client_id=client_id, route=route)
    if not resolution.ok:
        return TemporalOutcome(eligible=True, reason=resolution.reason,
                               detail=resolution.detail, plan_id=plan_id,
                               requested=requested, resolution=resolution)

    try:
        spec = adapter.spec_for_plan(plan)
    except Exception as exc:                                         # noqa: BLE001
        return TemporalOutcome(
            eligible=True, plan_id=plan_id, requested=requested,
            resolution=resolution, shape=resolution.shape,
            error=f"bind failed: {type(exc).__name__}: {exc}"[:300])

    from mi_agent.mi_query_executor import execute_mi_query
    from mi_agent.plan_shadow_evidence import cells_of

    dimensions = list(getattr(spec, "dimensions", None) or ())
    column = adapter.value_column(spec)
    points: List[SnapshotPoint] = []

    for header in resolution.headers:
        snapshot_id = str(header.snapshot_id)
        try:
            frame = store.load_loans(snapshot_id)
        except Exception as exc:                                     # noqa: BLE001
            return TemporalOutcome(
                eligible=True, plan_id=plan_id, requested=requested,
                resolution=resolution, shape=resolution.shape, spec=spec,
                reason=SNAPSHOT_LOAD_FAILED,
                detail=f"snapshot {snapshot_id}: "
                       f"{type(exc).__name__}: {exc}"[:250])
        try:
            result = execute_mi_query(spec, frame, semantics)
        except Exception as exc:                                     # noqa: BLE001
            return TemporalOutcome(
                eligible=True, plan_id=plan_id, requested=requested,
                resolution=resolution, shape=resolution.shape, spec=spec,
                reason=EXECUTION_FAILED,
                detail=f"snapshot {snapshot_id}: "
                       f"{type(exc).__name__}: {exc}"[:250])

        # THE SAME PLAN ON EVERY SNAPSHOT, PROVED PER SNAPSHOT. A predicate or an
        # axis that the executor did not apply on ONE period would make the
        # series a comparison of two different questions, so one failure ends
        # the whole attempt rather than dropping a point.
        ok, why_not = adapter.reconcile_receipt(spec, result)
        if not ok:
            return TemporalOutcome(
                eligible=True, plan_id=plan_id, requested=requested,
                resolution=resolution, shape=resolution.shape, spec=spec,
                reason=RECONCILIATION_FAILED,
                detail=f"snapshot {snapshot_id}: {why_not}")

        receipt = _receipt_of(result)
        measured = receipt.get("filtered_row_count")
        empty = isinstance(measured, int) and measured <= 0
        cells: Tuple[Mapping[str, Any], ...] = ()
        note = None
        if dimensions:
            found, note = cells_of(getattr(result, "data", None), dimensions,
                                   column)
            cells = tuple(found)
        points.append(SnapshotPoint(
            snapshot_id=snapshot_id,
            reporting_date=str(header.reporting_date),
            value=adapter._scalar_of(result, spec),
            cells=cells, cells_note=note, receipt=receipt, empty=empty,
            warnings=tuple(str(w)[:200]
                           for w in (getattr(result, "warnings", None) or ()))))

    if points and all(point.empty for point in points):
        # Slice 1B declines to SERVE an empty measured population, and the same
        # ruling holds for a series that is empty in every period. A single
        # empty period inside a populated series is kept: "nothing that month"
        # is the answer to that month, and dropping it would be a silent hole.
        return TemporalOutcome(
            eligible=True, plan_id=plan_id, requested=requested,
            resolution=resolution, shape=resolution.shape, spec=spec,
            points=tuple(points), reason=EMPTY_ACROSS_EVERY_SNAPSHOT,
            detail="the measured population is empty in every governed period")

    comparison = None
    if resolution.shape == SHAPE_COMPARISON and len(points) == 2:
        baseline, current = points
        comparison = {
            "baseline_snapshot_id": baseline.snapshot_id,
            "baseline_reporting_date": baseline.reporting_date,
            "current_snapshot_id": current.snapshot_id,
            "current_reporting_date": current.reporting_date,
            "baseline_value": baseline.value,
            "current_value": current.value,
            **_change(baseline.value, current.value),
        }
        if dimensions:
            comparison["cells"] = _grouped_comparison(baseline, current,
                                                      dimensions)

    return TemporalOutcome(
        eligible=True, plan_id=plan_id, shape=resolution.shape,
        basis=resolution.basis, requested=requested, spec=spec,
        resolution=resolution, points=tuple(points), comparison=comparison,
        reconciled=True)
