#!/usr/bin/env python3
"""Run a plan through the existing executor, and read back what it actually did.

WHAT THIS ADDS, AND WHAT IT REFUSES TO ADD. It runs each compiled execution
through `execute_mi_query` — the same call an equivalent atomic question makes —
and assembles the per-output record. There is no arithmetic here, no filtering,
no aggregation, and nothing that combines one output's figure with another's.
A composed answer is several ordinary governed executions, not a calculation
over their results.

THE EXECUTED SCOPE IS READ FROM EVIDENCE, NOT ASSUMED
-----------------------------------------------------
The point of reconciliation is to catch an answer computed over the wrong
population, so the executed scope may not be a restatement of what we asked
for. It is built from two things the executor publishes:

    metadata["applied_filter_fields"]   the fields a predicate ACTUALLY ran
                                        against in this book — the executor's
                                        own words, and explicitly distinct from
                                        `reconciliation.filters`, which merely
                                        echoes the spec
    metadata["group_field_keys"]        the axes it actually grouped on

A predicate the executor did not record applying is therefore ABSENT from the
executed scope, and `scope_equivalent` then reports the output as miscoped. That
is how "requested Joint, executed the whole book" is detected — structurally,
from the execution record, with no reference to any figure.

Row counts are not consulted anywhere in this module. Inferring that a filter
ran from the population having shrunk is the heuristic that declared a correct
filter lost whenever every row satisfied it, and it is exactly as wrong in the
other direction: a filter that removes rows for an unrelated reason would look
applied.

WHY THE RECEIPT DID NOT NEED EXTENDING. §13 of the target-state brief allows a
small general extension to receipts if per-output effective scope cannot be
recovered. It can: `applied_filter_fields` already names the fields, and the
executed SPEC carries their operators and values, so the executed scope is the
intersection of a contract we hold and evidence the executor published. Nothing
new is invented and no second truth metadata is created.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .mi_query_executor import execute_mi_query
from .query_plan import AnalyticalScope, Predicate, QueryPlan
from .query_plan_compiler import CompiledExecution, compile_query_plan
from .query_plan_result import MultiResultEnvelope, OutputResult

__all__ = ["execute_query_plan"]


def _executed_scope(execution: CompiledExecution, result: Any) -> AnalyticalScope:
    """The population this execution actually ran over, from its own record."""
    meta = getattr(result, "metadata", None) or {}
    applied = set(meta.get("applied_filter_fields") or ())
    # Only the predicates the executor recorded running. A requested predicate
    # it did not apply is absent here, which is what makes the mismatch visible.
    survived = tuple(p for p in execution.effective_scope.filters
                     if p.field in applied)
    grouped = tuple(meta.get("group_field_keys")
                    or ([execution.spec.dimension] if execution.spec.dimension
                        else ()))
    return AnalyticalScope(
        dataset=execution.effective_scope.dataset,
        portfolio_lens=execution.effective_scope.portfolio_lens,
        period=execution.effective_scope.period,
        filters=survived,
        dimensions=tuple(g for g in grouped if g))


def _values_by_output(execution: CompiledExecution, result: Any) -> Dict[str, Any]:
    """Each output's figure, read from the executed frame — never recomputed.

    A grouped result is a table and has no single figure per output, so the
    value is left None and the result frame stands as the answer. Nothing here
    aggregates: if the number cannot be read from what the executor returned, it
    is not manufactured.
    """
    out: Dict[str, Any] = {}
    data = getattr(result, "data", None)
    if data is None or not len(data) or getattr(result, "result_type", "") != "summary":
        return out
    row = data.iloc[0]
    meta = getattr(result, "metadata", None) or {}
    executed = meta.get("measures_executed") or []
    if executed and len(executed) == len(execution.output_ids):
        for output_id, measure in zip(execution.output_ids, executed):
            column = (measure or {}).get("column")
            if column in data.columns:
                out[output_id] = row[column]
        return out
    if len(execution.output_ids) == 1:
        # A single-output summary still carries `loan_count` alongside the
        # measure — the executor has always published both — so the column is
        # chosen by what the output ASKED for rather than by being the only one.
        only = execution.spec
        column = ("loan_count" if only.aggregation == "count"
                  else f"{only.metric}_{only.aggregation}")
        if column in data.columns:
            out[execution.output_ids[0]] = row[column]
        elif only.metric in data.columns:
            out[execution.output_ids[0]] = row[only.metric]
    return out


def execute_query_plan(plan: QueryPlan, data, semantics, **executor_kwargs
                       ) -> MultiResultEnvelope:
    """``QueryPlan`` → one governed execution per distinct population → envelope.

    Each child execution takes the same path an equivalent atomic question
    takes. The envelope's completeness is decided by `reconcile_plan`, which
    compares the population each output ASKED for against the one its execution
    reports having used.
    """
    results: List[OutputResult] = []
    for execution in compile_query_plan(plan):
        result = execute_mi_query(execution.spec, data, semantics,
                                  **executor_kwargs)
        executed = _executed_scope(execution, result)
        values = _values_by_output(execution, result)
        for output_id in execution.output_ids:
            results.append(OutputResult(
                output_id=output_id,
                requested_scope=execution.effective_scope,
                executed_scope=executed,
                execution_ref=result,
                value=values.get(output_id)))
    return MultiResultEnvelope.build(plan, results)
