#!/usr/bin/env python3
"""What each requested figure was, what ran, and whether they agree.

TWO WAYS A COMPOSED ANSWER GOES WRONG.

MISSING — three outputs requested, two executed, returned as a success. The
machinery to catch this existed and could not see the requests: its trigger read
the question with its own vocabulary, which had no word for a count and none for
"amount", so for the commonest composed shape it never fired.

MISCOPED — every output produced, every number arithmetically correct, one of
them about a different population than the one asked for. This is the worse
failure because the figure is real, and it is invisible from anywhere except a
structural comparison of the two populations:

    requested Joint   executed the whole book
    requested Joint   executed Joint AND LTV > 40
    requested rate>6  executed rate>=6

None of those survives rendering to a string, and none can be inferred from a
row count — that heuristic is what declared a correct filter lost whenever every
row happened to satisfy it.

OUTPUT IDENTITY IS measure + aggregation + EFFECTIVE SCOPE. Without the third
term the second and third clauses of "how many joint loans, what is their
balance, and how much of that balance has LTV above 40%" are the same output,
which is exactly why one of them disappeared.

THIS IS NOT AN ORACLE. It does not know whether a number is right. It compares
two records that already exist: what the plan asked for, and what each execution
reports having done. And it REFERENCES the governed receipt rather than copying
it — a second metadata model would be one more owner of what happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Tuple

from .query_plan import AnalyticalScope, QueryPlan, effective_scope, scope_equivalent

__all__ = ["OutputResult", "PlanReconciliation", "MultiResultEnvelope",
           "reconcile_plan"]


@dataclass(frozen=True)
class OutputResult:
    """One governed figure, with the population it was asked for and the one it ran over.

    BOTH SCOPES ARE KEPT, deliberately. Holding only the executed one makes a
    widened answer indistinguishable from a correct one; holding only the
    requested one makes the record a restatement of the question rather than
    evidence about the answer. Reconciliation needs the pair.
    """

    output_id: str
    requested_scope: AnalyticalScope
    executed_scope: AnalyticalScope
    #: A REFERENCE to the governed execution/receipt — never a copy of it.
    execution_ref: Any = None
    value: Any = None


@dataclass(frozen=True)
class PlanReconciliation:
    """Which requested outputs are accounted for, and how the rest failed."""

    complete: bool
    missing: Tuple[str, ...] = ()
    miscoped: Tuple[str, ...] = ()
    unrequested: Tuple[str, ...] = ()
    duplicated: Tuple[str, ...] = ()

    def reason(self) -> str:
        """A sentence naming what stopped this being a complete answer."""
        parts = []
        if self.missing:
            parts.append("not executed: " + ", ".join(self.missing))
        if self.miscoped:
            parts.append("executed over a different population than requested: "
                         + ", ".join(self.miscoped))
        if self.duplicated:
            parts.append("more than one result: " + ", ".join(self.duplicated))
        if self.unrequested:
            parts.append("returned but not requested: "
                         + ", ".join(self.unrequested))
        return "; ".join(parts)


def reconcile_plan(plan: QueryPlan,
                   results: Iterable[OutputResult]) -> PlanReconciliation:
    """Compare what the plan asked for against what the executions report.

    Fail-closed by construction: `complete` is true only when every requested
    output has exactly one result AND that result ran over a population
    structurally equivalent to the one requested for it. Anything else is named,
    never averaged away.
    """
    by_id: dict = {}
    duplicated: list = []
    unrequested: list = []
    requested_ids = [o.output_id for o in plan.outputs]

    for result in results:
        if result.output_id not in requested_ids:
            unrequested.append(result.output_id)
            continue
        if result.output_id in by_id:
            duplicated.append(result.output_id)
            continue
        by_id[result.output_id] = result

    missing: list = []
    miscoped: list = []
    for output in plan.outputs:
        result = by_id.get(output.output_id)
        if result is None:
            missing.append(output.output_id)
            continue
        wanted = effective_scope(plan, output)
        if not scope_equivalent(wanted, result.executed_scope):
            miscoped.append(output.output_id)

    return PlanReconciliation(
        complete=not (missing or miscoped or unrequested or duplicated),
        missing=tuple(missing), miscoped=tuple(miscoped),
        unrequested=tuple(dict.fromkeys(unrequested)),
        duplicated=tuple(dict.fromkeys(duplicated)))


@dataclass(frozen=True)
class MultiResultEnvelope:
    """Several governed outputs, and whether together they answer the request."""

    plan_id: str
    shared_scope: AnalyticalScope
    outputs: Tuple[OutputResult, ...]
    completeness: PlanReconciliation

    @classmethod
    def build(cls, plan: QueryPlan,
              results: Iterable[OutputResult]) -> "MultiResultEnvelope":
        ordered = list(results)
        index = {r.output_id: r for r in ordered}
        return cls(
            plan_id=plan.plan_id,
            shared_scope=plan.shared_scope,
            outputs=tuple(index[o.output_id] for o in plan.outputs
                          if o.output_id in index),
            completeness=reconcile_plan(plan, ordered))
