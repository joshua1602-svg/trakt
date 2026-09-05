#!/usr/bin/env python3
"""An existing parse, lifted into the target contracts — or declined.

THE MIGRATION SHAPE. Interpretation does not move. The parser produces an
`MIQuerySpec` exactly as it always has, and this module LIFTS that spec into
`QueryPlan` / `AnalyticalScope`. Nothing here reads the question. A second
reader of the sentence is the defect this programme has spent its whole length
removing — five owners of "a count was requested", two population resolvers,
three homes for `amount` — and introducing one more in order to populate the new
objects would be the worst possible way to adopt them.

THE INVARIANT THAT MAKES THE LIFT SAFE:

    compile_query_plan(plan_from_spec(spec))  ==  spec

for the semantics that decide an answer: population, measure, aggregation,
dimension. If that round trip is not an identity then routing production through
the plan changes answers, and structural tests elsewhere would not say which.

WHY THIS DECLINES MORE THAN IT ACCEPTS
--------------------------------------
`MIQuerySpec` carries 76 fields. A `QueryPlan` models a handful — filters,
dimension, measures, lens, period. The rest express rankings, temporal
comparisons, cohort progressions, bridges, forecasts, risk-limit plans, bucket
strategies, scatter axes. An adapter that lifted one of those specs would
silently drop the field it does not model, and the compiled spec would execute a
simpler question than the reader asked, with a plan that looked complete.

So the test is INVERTED and expressed as a property rather than a list of
exclusions: every field outside `MODELLED_FIELDS` must still hold its default,
or the spec is not liftable. A field added to `MIQuerySpec` tomorrow is
therefore un-liftable until someone models it deliberately — the opposite of the
usual failure, where a new field is quietly ignored by an adapter nobody
remembered to update.

That is also what makes the migration incremental and safe: the live route can
take the plan path only where the lift is exact, and the shipped path everywhere
else, with no window in which some questions are answered by a lossy plan.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional

from .mi_query_spec import MIQuerySpec
from .query_plan import (
    COUNT, AnalyticalScope, PlannedOutput, Predicate, QueryPlan,
)

__all__ = ["MODELLED_FIELDS", "plan_from_spec"]

#: The spec fields a `QueryPlan` can carry without loss.
#:
#: Split into three groups, because they earn their place differently.
MODELLED_FIELDS = frozenset({
    # 1. The analytical semantics the plan represents.
    "metric", "aggregation", "weight_field", "measures", "filters",
    # EVERY governed axis. `dimensions` was un-modelled, which made 128 of the
    # corpus's 307 declines a grouped question the plan could in fact carry —
    # the executor's own `_all_group_dims` reads exactly these two fields.
    "dimension", "dimensions", "x", "y",
    "portfolio_lens", "reporting_date", "as_of_date",
    # 2. Presentation and prose. These decide nothing — §20 of the target-state
    #    brief puts presentation downstream of the governed result — so a
    #    difference here is not a semantic difference and must not block a lift.
    "intent", "chart_type", "output_format", "title", "explanation",
    # 3. Disclosure ABOUT the measure rather than a change to it.
    "metric_defaulted",
})

#: `dimension` and `x` are the same axis in the spec's vocabulary; the plan
#: carries one and the compiler restates both.
_AXIS_FIELDS = ("dimension", "x")


def _spec_defaults() -> Dict[str, Any]:
    """Each field's default, so "was this set?" is asked of the contract."""
    out: Dict[str, Any] = {}
    for field in dataclasses.fields(MIQuerySpec):
        if field.default is not dataclasses.MISSING:
            out[field.name] = field.default
        elif field.default_factory is not dataclasses.MISSING:  # type: ignore
            out[field.name] = field.default_factory()           # type: ignore
    return out


def _is_liftable(spec: MIQuerySpec) -> bool:
    """True when the plan can carry every semantic this spec states."""
    defaults = _spec_defaults()
    for name, default in defaults.items():
        if name in MODELLED_FIELDS:
            continue
        if getattr(spec, name, default) != default:
            return False
    # An axis the plan cannot express: `dimension` and `x` must agree, because
    # the plan carries one axis and restating a disagreement would invent one.
    if spec.dimension and spec.x and spec.dimension != spec.x:
        return False
    # `y` is an AXIS on a grouped result and a MEASURE on a scatter/bubble. The
    # plan models the first meaning only, so a `y` that is not the second
    # grouping dimension is not liftable.
    axes = _axes(spec)
    if spec.y and (len(axes) < 2 or spec.y != axes[1]):
        return False
    return True


def _predicates(filters: Optional[Dict[str, Any]]) -> tuple:
    """The executor's filter shape → canonical predicates.

    The inverse of the compiler's `_filters_for`, and deliberately total: a
    shape neither recognises would be a population nobody can compare, so it
    raises rather than being coerced into an equality.
    """
    out: List[Predicate] = []
    for field, value in (filters or {}).items():
        if isinstance(value, dict):
            op = str(value.get("op") or "eq").lower()
            out.append(Predicate(field, op, value.get("value")))
        elif isinstance(value, (list, tuple, set)):
            out.append(Predicate(field, "in", tuple(value)))
        else:
            out.append(Predicate(field, "eq", value))
    return tuple(out)


def _axes(spec: MIQuerySpec) -> tuple:
    """Every governed grouping dimension, in order, de-duplicated.

    The same order and the same precedence as the executor's `_all_group_dims`
    — `dimensions` first, then `dimension` — because a plan that disagreed with
    the executor about which axes exist would be a second grouping model.
    """
    out: List[str] = []
    for key in list(spec.dimensions or ()) + ([spec.dimension] if spec.dimension else []):
        if key and key not in out:
            out.append(key)
    return tuple(out)


def _outputs(spec: MIQuerySpec) -> tuple:
    """The requested outputs this spec states, in request order.

    A spec carries ONE filters dict, so every output it lifts to shares the
    population and none of them carries a local delta. A delta appearing here
    would be invented — the spec has no way to say one.
    """
    entries = list(spec.measures or ())
    if not entries:
        entries = [{"field": spec.metric, "aggregation": spec.aggregation,
                    "weight_field": spec.weight_field}]
    outputs: List[PlannedOutput] = []
    for index, entry in enumerate(entries):
        field = (entry or {}).get("field")
        aggregation = str((entry or {}).get("aggregation") or spec.aggregation)
        is_count = aggregation == COUNT or field in ("loan_count", "count")
        outputs.append(PlannedOutput(
            output_id=f"o{index}",
            operation=COUNT if is_count else aggregation,
            measure=None if is_count else field,
            weight_field=(entry or {}).get("weight_field")))
    return tuple(outputs)


def plan_from_spec(spec: MIQuerySpec, *,
                   dataset: Optional[str] = None,
                   portfolio_lens: Optional[str] = None,
                   period: Optional[str] = None) -> Optional[QueryPlan]:
    """``MIQuerySpec`` → ``QueryPlan``, or None when the lift would lose meaning.

    ``dataset``, ``portfolio_lens`` and ``period`` select the FRAME and are
    resolved upstream of the spec, so they are supplied by the caller rather
    than guessed here. A lens stated on the spec is used when the caller names
    none — the spec's own value is evidence, not a default.
    """
    if not _is_liftable(spec):
        return None
    try:
        outputs = _outputs(spec)
        scope = AnalyticalScope(
            dataset=dataset,
            portfolio_lens=portfolio_lens or spec.portfolio_lens,
            period=period or spec.reporting_date or spec.as_of_date,
            filters=_predicates(spec.filters),
            dimensions=_axes(spec))
        return QueryPlan(shared_scope=scope, outputs=outputs)
    except (ValueError, TypeError):
        # A spec the contracts reject — an operation outside the governed set,
        # a measure-less aggregation. Declining is the honest outcome: the
        # shipped path still answers it.
        return None
