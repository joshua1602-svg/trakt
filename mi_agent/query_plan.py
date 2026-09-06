#!/usr/bin/env python3
"""The analytical contracts: one population, several outputs, optional narrowing.

WHAT THIS IS FOR
----------------
Every composition defect this programme has found reduced to the same shape:
two owners disagreeing about one concept. Five readings of "a count was
requested"; two population resolvers, neither a superset of the other; three
homes for "amount means the balance"; a receipt guard inferring whether a filter
ran from whether the row count fell. Each was fixed by naming the owner.

One concept still had no owner at all, and it is the one composition needs:

    WHICH ROWS IS THIS FIGURE ABOUT?

`MIQuerySpec` carries a single `filters` dict for a whole request, so a request
about two populations cannot be represented — only mis-represented. That is not
a parser bug, and no amount of parsing fixes it:

    "How many joint loans are there, what is their balance, and how much of
     that balance has LTV above 40%?"

        measures  [loan_count/count, balance/sum]
        filters   {borrower_type: Joint, current_loan_to_value: >40}

Three figures were asked for and two were produced, both over a population
neither clause described. The third output could not exist, because output
identity was measure-plus-aggregation and it shared both with the second.

THE CONTRACTS
-------------
    Predicate         one canonical narrowing: field, operator, value
    AnalyticalScope   which rows — dataset, lens, period, predicates, axes
    ScopeDelta        a narrowing belonging to ONE output
    PlannedOutput     one requested figure, with its own optional narrowing
    QueryPlan         one shared scope and the outputs asked of it

WHAT THEY DELIBERATELY DO NOT DO
--------------------------------
Nothing here calculates. There is no filtering, no aggregation, no arithmetic
and no second analytics engine — the governed executor owns all of it, and the
compiler seam turns these plans into the execution contracts it already
accepts. A `QueryPlan` is what the request MEANS, between interpretation and
execution, and it is the only thing in this module.

Presentation is not represented either. Chart, table and prose consume governed
results; they may not decide a population, a measure or an aggregation, so they
have no say in this contract.

TWO PROPERTIES THAT ARE LOAD-BEARING
------------------------------------
IMMUTABILITY. Resolving an output must not change the plan it came from. Every
contract here is a frozen dataclass and every collection is a tuple, so a delta
cannot reach the shared scope by aliasing — which is how the live defect leaked
in the first place, through a mutable dict shared by every measure.

STRUCTURAL IDENTITY. `rate > 6` and `rate >= 6` select different rows and are
different scopes; the same predicates in a different order are the same scope.
Neither fact survives rendering to a string, and neither can be inferred from a
row count. `scope_equivalent` compares canonical structure, so "requested Joint,
executed the whole book" and "requested Joint, executed Joint AND LTV > 40" are
both detectable — the two failure modes composition has to be able to see.

A NOTE ON WHAT COMES NEXT. Conversational progression needs one more thing from
this module and nothing else: the ability to take an executed `AnalyticalScope`,
apply a governed mutation, and get the next one. That is why scope is not
composition-specific and does not mention outputs. It is not implemented here.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace
from typing import Any, Optional, Tuple

__all__ = [
    "COUNT", "SUM", "AVERAGE", "WEIGHTED_AVERAGE", "MEDIAN", "MIN", "MAX",
    "OPERATIONS", "Predicate", "AnalyticalScope", "ScopeDelta", "PlannedOutput",
    "QueryPlan", "ScopeConflict", "effective_scope", "scope_equivalent",
]


# --------------------------------------------------------------------------- #
# Operations
# --------------------------------------------------------------------------- #
# The governed statistic vocabulary, named here rather than invented: these are
# the aggregations `mi_query_executor` already dispatches on, so a PlannedOutput
# cannot ask for something the executor has no branch for.
COUNT = "count"
SUM = "sum"
AVERAGE = "avg"
WEIGHTED_AVERAGE = "weighted_avg"
MEDIAN = "median"
MIN = "min"
MAX = "max"

OPERATIONS: Tuple[str, ...] = (COUNT, SUM, AVERAGE, WEIGHTED_AVERAGE,
                               MEDIAN, MIN, MAX)

#: Operators whose bounds can contradict each other on one field. `eq`/`in` are
#: not here: two categorical values on one field are a widening ("Scotland and
#: Wales"), which the executor's `in` already expresses.
_DIRECTIONAL = {"gt": 1, "ge": 1, "lt": -1, "le": -1}


class ScopeConflict(ValueError):
    """A delta that cannot narrow the scope it was applied to.

    Raised rather than resolved. Two bounds pointing opposite ways on one field
    are either a range the reader did not write or a mistake, and guessing which
    is how a request for one population becomes an answer about another.
    """


# --------------------------------------------------------------------------- #
# Predicate
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Predicate:
    """One canonical narrowing: ``field``, ``op``, ``value``.

    The same shape `MIQuerySpec.filters` carries as ``{field: {op, value}}``,
    made comparable. A list value (``in``) is normalised to a tuple so the
    predicate stays hashable and two orderings of one restriction compare equal.
    """

    field: str
    op: str
    value: Any = None

    def __post_init__(self) -> None:
        if isinstance(self.value, (list, set)):
            object.__setattr__(self, "value", tuple(sorted(
                self.value, key=lambda v: (str(type(v)), str(v)))))

    @property
    def key(self) -> Tuple[str, str, Any]:
        """What makes this predicate the predicate it is."""
        return (self.field, str(self.op).lower(), self.value)


# --------------------------------------------------------------------------- #
# AnalyticalScope
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AnalyticalScope:
    """Which rows a calculation is about.

    Deliberately NOT composition-specific: it answers the same question for an
    atomic query, for one output of a composed one, and — later — for the
    population a follow-up turn inherits. One model, so those three cannot drift
    apart the way the estate's population resolvers did.
    """

    dataset: Optional[str] = None
    portfolio_lens: Optional[str] = None
    period: Optional[str] = None
    filters: Tuple[Predicate, ...] = ()
    dimensions: Tuple[str, ...] = ()

    def narrowed_by(self, delta: Optional["ScopeDelta"]) -> "AnalyticalScope":
        """A NEW scope carrying this one's predicates and the delta's.

        Never mutates: the shared scope of a plan is read by every output, and
        an in-place narrowing is exactly how a clause-local bound reached its
        siblings.
        """
        if delta is None or not delta.filters:
            return self
        merged = list(self.filters)
        for predicate in delta.filters:
            _reject_conflict(merged, predicate)
            if predicate not in merged:
                merged.append(predicate)
        return replace(self, filters=tuple(merged))

    @property
    def identity(self) -> Tuple[Any, ...]:
        """Canonical, order-insensitive identity — see `scope_equivalent`."""
        return (self.dataset, self.portfolio_lens, self.period,
                tuple(sorted(p.key for p in self.filters)),
                tuple(sorted(self.dimensions)))


def _reject_conflict(existing, predicate: Predicate) -> None:
    """Raise when ``predicate`` cannot be a NARROWING of ``existing``."""
    for current in existing:
        if current.field != predicate.field:
            continue
        here = _DIRECTIONAL.get(str(predicate.op).lower())
        there = _DIRECTIONAL.get(str(current.op).lower())
        if here and there and here != there:
            raise ScopeConflict(
                f"{predicate.field}: {current.op} {current.value} and "
                f"{predicate.op} {predicate.value} bound the same field in "
                "opposite directions; this is a range or a mistake, and "
                "neither is assumed")
        if not here and not there and current.value != predicate.value:
            raise ScopeConflict(
                f"{predicate.field}: already restricted to {current.value!r}; "
                f"{predicate.value!r} would select different rows rather than "
                "fewer")


def scope_equivalent(one: Optional[AnalyticalScope],
                     other: Optional[AnalyticalScope]) -> bool:
    """Do these two describe the SAME population?

    Structural, never textual and never inferred from a row count. The three
    differences composition must be able to see:

        requested Joint            vs executed the whole book   → False
        requested Joint            vs executed Joint + LTV > 40 → False
        requested rate > 6         vs executed rate >= 6        → False

    and the one it must not mistake for a difference:

        LTV > 40 then rate > 6     vs rate > 6 then LTV > 40    → True
    """
    if one is None or other is None:
        return one is other
    return one.identity == other.identity


# --------------------------------------------------------------------------- #
# ScopeDelta
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScopeDelta:
    """A narrowing belonging to ONE output.

    NARROWING ONLY, and the asymmetry is the point. There is no field for
    removing a shared predicate, because an output that covered MORE than the
    request's own population would be answering a question nobody asked — the
    silent widening this estate refuses everywhere else. A delta can add rows to
    the restriction and never to the answer.
    """

    filters: Tuple[Predicate, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.filters)


# --------------------------------------------------------------------------- #
# PlannedOutput
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PlannedOutput:
    """One requested figure.

    A COUNT IS AN OUTPUT IN ITS OWN RIGHT — `operation=COUNT`, `measure=None`.
    The estate grew five separate readings of "a count was requested" precisely
    because a count had nowhere to live: every branch owned one top-level
    metric, and a count is not a metric.
    """

    output_id: str
    operation: str
    measure: Optional[str] = None
    #: Where the operation's default is not what the reader asked for. None
    #: means "the operation is the aggregation", which is true for all but the
    #: weighted cases.
    weight_field: Optional[str] = None
    local_scope_delta: Optional[ScopeDelta] = None

    def __post_init__(self) -> None:
        if self.operation not in OPERATIONS:
            raise ValueError(
                f"{self.operation!r} is not a governed operation; the executor "
                f"dispatches on {', '.join(OPERATIONS)}")
        if self.operation != COUNT and not self.measure:
            raise ValueError(
                f"{self.operation!r} needs a measure to operate on; only "
                "COUNT measures the rows themselves")


# --------------------------------------------------------------------------- #
# QueryPlan
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class QueryPlan:
    """One shared population, and the outputs asked of it.

    Owns the RELATIONSHIP between a scope and the figures requested over it, and
    nothing else: not the calculation, not the presentation, not the transport.
    """

    shared_scope: AnalyticalScope
    outputs: Tuple[PlannedOutput, ...]
    plan_id: str = field(default_factory=lambda: f"plan-{next(_PLAN_SEQUENCE)}")

    def __post_init__(self) -> None:
        if not self.outputs:
            raise ValueError("a plan with no outputs requests nothing")
        ids = [o.output_id for o in self.outputs]
        if len(set(ids)) != len(ids):
            raise ValueError(
                "output ids must be unique: they are how a result, a receipt "
                "and a later reference all name the same figure")

    def output(self, output_id: str) -> PlannedOutput:
        for candidate in self.outputs:
            if candidate.output_id == output_id:
                return candidate
        raise KeyError(output_id)


_PLAN_SEQUENCE = itertools.count(1)


def effective_scope(plan: QueryPlan, output: PlannedOutput) -> AnalyticalScope:
    """The population THIS output is about.

    The whole structure exists for this one line: the delta is applied to a COPY
    of the shared scope, so it reaches the output that owns it and nothing else.
    """
    return plan.shared_scope.narrowed_by(output.local_scope_delta)
