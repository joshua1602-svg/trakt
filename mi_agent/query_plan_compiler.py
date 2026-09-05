#!/usr/bin/env python3
"""QueryPlan → the execution contracts the governed executor already accepts.

THE SEAM, AND WHY IT IS ONLY A SEAM. A `QueryPlan` says what a request MEANS.
`mi_query_executor` already knows how to calculate — filtering, grouping, count,
sum, averages, weighted averages, and the P1E multi-measure path that executes
several measures over one population. Nothing here calculates anything. This
module's entire job is to turn one analytical plan into specs that layer already
consumes, so the target-state model sits ABOVE the deterministic engine instead
of duplicating it.

    outputs sharing one effective scope
        → ONE MIQuerySpec carrying a measure set  (the shipped P1E path)

    outputs whose effective scopes differ
        → SEVERAL ordinary specs, one per population

The second case is the one `MIQuerySpec` could never express. It carries a
single `filters` dict, so a request about two populations had to be flattened
into one — which is how

    "How many joint loans are there, what is their balance, and how much of
     that balance has LTV above 40%?"

came to compute its count and its balance over joint loans above 40% LTV, a
population neither clause described, and lost its third output entirely because
output identity was measure-plus-aggregation and it shared both with the second.
Compiling to several ordinary specs makes that third clause a real output, and
each child execution takes exactly the path an equivalent atomic question takes.

THE OPTIMISATION MAY NOT CHANGE THE SEMANTICS. Folding compatible outputs into
one execution is an efficiency; the correctness of the population is not. Two
outputs share a spec only when their effective scopes are STRUCTURALLY
equivalent — `scope_equivalent`, on canonical predicates — never because they
look similar or happen to name the same measure.

BACKWARD COMPATIBILITY IS THE POINT. A single-output plan compiles to the spec
the atomic path already produces: same metric, same aggregation, same filters,
`measures` left empty so `normalise_measures` folds it exactly as it does today.
A compiler that quietly produced a different spec for simple questions would
move every atomic answer in the estate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .mi_query_spec import MIQuerySpec
from .query_plan import (
    COUNT, AnalyticalScope, PlannedOutput, Predicate, QueryPlan,
    effective_scope,
)

__all__ = ["CompiledExecution", "compile_query_plan"]

#: The executor's own name for "count the rows". A count is an output in its own
#: right; in the measure-set contract it is spelled as this field.
_ROW_COUNT_FIELD = "loan_count"


@dataclass(frozen=True)
class CompiledExecution:
    """One governed execution, and the outputs it answers.

    ``effective_scope`` travels with it rather than being re-derived from the
    spec: the spec's `filters` is the executor's shape, and the scope is the
    canonical one that provenance and reconciliation compare against. Deriving
    the second from the first would put a parser between a request and the
    record of what it asked for.
    """

    spec: MIQuerySpec
    output_ids: Tuple[str, ...]
    effective_scope: AnalyticalScope


def _filters_for(scope: AnalyticalScope) -> Dict[str, Any]:
    """Canonical predicates → the executor's ``{field: value | {op, value}}``.

    The executor's shape is not a second model of a population; it is the wire
    format of this one. A categorical equality is a bare value because that is
    what `_apply_filters` has always accepted, and anything with a direction
    keeps its operator.
    """
    out: Dict[str, Any] = {}
    for predicate in scope.filters:
        op = str(predicate.op or "").lower()
        if op in ("eq", "", "equals"):
            out[predicate.field] = _executor_value(predicate.value)
        else:
            out[predicate.field] = {"op": op,
                                    "value": _executor_value(predicate.value)}
    return out


def _executor_value(value: Any) -> Any:
    """A predicate's value in the shape the executor's filters carry.

    `Predicate` normalises a multi-valued bound to a TUPLE so a scope stays
    hashable and two orderings of one restriction compare equal. The executor's
    `filters` have always held LISTS — `{"op": "between", "value": [40, 60]}` —
    and the round trip is only an identity if the shape comes back too.

    Found by running the whole 882-question corpus through
    `plan_from_spec` → `compile_query_plan`: seven questions, all of them a
    `between`, came back structurally right and shaped wrong. Semantically it
    changes nothing — the executor accepts either — but the round-trip identity
    is the entire argument for routing production through the plan, and an
    invariant with seven exceptions is not an invariant.
    """
    return list(value) if isinstance(value, tuple) else value


def _measure_entry(output: PlannedOutput) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "field": _ROW_COUNT_FIELD if output.operation == COUNT else output.measure,
        "aggregation": output.operation,
    }
    if output.weight_field:
        entry["weight_field"] = output.weight_field
    return entry


def _spec_for(scope: AnalyticalScope,
              outputs: List[PlannedOutput]) -> MIQuerySpec:
    """The execution contract for these outputs over this population."""
    dimension = scope.dimensions[0] if scope.dimensions else None
    grouped = bool(dimension)
    filters = _filters_for(scope)

    if len(outputs) == 1:
        # THE ATOMIC SHAPE, UNCHANGED. `measures` stays empty so the spec is
        # byte-for-byte the shape the single-measure paths have always emitted
        # and `normalise_measures` folds it the same way.
        only = outputs[0]
        return MIQuerySpec(
            intent="chart" if grouped else "summary",
            chart_type="bar" if grouped else "none",
            metric=None if only.operation == COUNT else only.measure,
            aggregation=only.operation,
            weight_field=only.weight_field,
            dimension=dimension, x=dimension,
            filters=filters,
            output_format="chart" if grouped else "text",
            explanation="One governed output over one analytical population.")

    return MIQuerySpec(
        intent="chart" if grouped else "summary",
        chart_type="bar" if grouped else "none",
        measures=[_measure_entry(o) for o in outputs],
        metric=(None if outputs[0].operation == COUNT else outputs[0].measure),
        aggregation=outputs[0].operation,
        dimension=dimension, x=dimension,
        filters=filters,
        output_format="chart_and_table" if grouped else "table",
        explanation="Several governed outputs over one analytical population.")


def compile_query_plan(plan: QueryPlan) -> Tuple[CompiledExecution, ...]:
    """``QueryPlan`` → the executions that answer it, in request order.

    Outputs are grouped by the STRUCTURAL identity of their effective scope, so
    two outputs share an execution when and only when they are about the same
    rows. Every requested output appears in exactly one execution — the
    reconciliation layer depends on that, and so does any honest claim that a
    composed answer is complete.
    """
    order: List[Any] = []
    groups: Dict[Any, List[PlannedOutput]] = {}
    scopes: Dict[Any, AnalyticalScope] = {}

    for output in plan.outputs:
        scope = effective_scope(plan, output)
        identity = scope.identity
        if identity not in groups:
            groups[identity] = []
            scopes[identity] = scope
            order.append(identity)
        groups[identity].append(output)

    return tuple(
        CompiledExecution(spec=_spec_for(scopes[identity], groups[identity]),
                          output_ids=tuple(o.output_id
                                           for o in groups[identity]),
                          effective_scope=scopes[identity])
        for identity in order)
