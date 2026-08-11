"""trakt_tools.handlers.movement — ``covenant_drillthrough`` and ``period_change``.

    capability: loan:read  (drillthrough names loans)
                risk:read  (period_change is aggregate)

Both wrap engines Trakt already has, and neither computes anything:

``covenant_drillthrough`` calls
``mi_agent_api.concentration_tests_api.compute_drillthrough`` — the same function
the React Risk Limits workspace calls. That engine reuses the *same evaluator and
mask* that produced the covenant numerator, which is why the returned population
reconciles exactly: its row count equals ``loans_in_numerator`` and its basis
balance equals ``numerator_value``. Rebuilding the population here from the test's
parameters would produce a second answer that agreed most of the time, which is
worse than one that disagrees loudly.

``period_change`` calls ``mi_agent_api.period_change_route.analyse_period_change``,
whose result is already a complete auditable envelope with its own period
resolution, dataset provenance, warnings and limitations. The tool carries it
through and adds the resource scope.

Why an agent needs these
------------------------
``evaluate_covenants`` says a test is breached. The next question is always
*which loans* — and without a drillthrough tool an agent answers it by guessing
the population from the test's display name and rebuilding it with ``get_loans``
and a filter. That guess is unreconciled and usually subtly wrong, because a
covenant's numerator is defined by a registered metric rather than by a field
comparison anyone can infer.

Likewise, "has this got worse?" is the second question about any figure, and an
agent that cannot ask it will compare a number it has now against a number it
remembers — across snapshots it never verified were comparable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from trakt_core.errors import ErrorCode, TraktError

from ..schema import object_schema
from ..spec import ToolInvocation
from .loans import _jsonable, _scope_block

#: Rows one drillthrough may return. The engine's own default; restated so the
#: published contract states it rather than inheriting it silently.
MAX_DRILLTHROUGH_ROWS = 500

_FUNDED = "funded"

_RESOURCE_PROPERTY = {
    "type": "string",
    "description": ("The resource the test was evaluated against, as "
                    "'{tenant}/{kind}/{resource_id}'."),
    "pattern": r"^[^/]+/[^/]+/[^/]+$",
}


def _refuse_unpartitionable(inv: ToolInvocation, what: str) -> Any:
    """The two narrowings the funded engines cannot honour.

    Identical to the checks ``evaluate_covenants`` makes, and made again here
    rather than assumed: a tool that skips them because a sibling tool performs
    them is one refactor away from being a way around them.
    """
    resolved = inv.authorised.resource
    if resolved.population and resolved.population != _FUNDED:
        raise TraktError(
            ErrorCode.RESOURCE_NOT_PARTITIONABLE,
            f"{what} is evaluated on the funded book. This resource is pinned to "
            "a different population, so it cannot be answered without reporting "
            "on data it does not name.",
            request_id=inv.request_id,
            details={"resource": resolved.ref.key,
                     "population": resolved.population})
    if resolved.spv_id:
        raise TraktError(
            ErrorCode.RESOURCE_NOT_PARTITIONABLE,
            f"This resource is defined by an SPV boundary, which {what} cannot "
            "currently apply. Answering it would report on the enclosing book "
            "rather than the SPV.",
            request_id=inv.request_id, details={"resource": resolved.ref.key})
    scope = resolved.to_portfolio_scope()
    if scope is None and not resolved.whole_tenant_book:
        raise TraktError(
            ErrorCode.RESOURCE_NOT_PARTITIONABLE,
            "This resource declares no book narrowing and is not registered as "
            "the tenant's whole book, so the population it means is undefined.",
            request_id=inv.request_id, details={"resource": resolved.ref.key})
    return scope


# =========================================================================== #
# covenant_drillthrough
# =========================================================================== #
DRILLTHROUGH_INPUT = object_schema(
    description=("The loans contributing to ONE covenant or concentration test's "
                 "numerator. Use this after evaluate_covenants tells you a test "
                 "is breached or close to its limit."),
    properties={
        "resource": _RESOURCE_PROPERTY,
        "test_id": {
            "type": "string",
            "description": ("The test to drill into, as returned in "
                            "evaluate_covenants' tests[].test_id."),
        },
        "as_of_run_id": {
            "type": ["string", "null"],
            "description": ("Optional governed run. Omit for the current approved "
                            "reporting position. Use the SAME value you passed to "
                            "evaluate_covenants, or the two will not reconcile."),
            "default": None,
        },
        "limit": {
            "type": "integer", "minimum": 1, "maximum": MAX_DRILLTHROUGH_ROWS,
            "description": f"Rows to return, at most {MAX_DRILLTHROUGH_ROWS}.",
            "default": 100,
        },
    },
    required=["resource", "test_id"],
)

DRILLTHROUGH_OUTPUT = object_schema(
    description="The contributing population, and whether it reconciles.",
    properties={
        "resource": {"type": "string"},
        "available": {"type": "boolean"},
        "reason": {"type": ["string", "null"]},
        "test_id": {"type": ["string", "null"]},
        "metric_id": {"type": ["string", "null"]},
        "display_name": {"type": ["string", "null"]},
        "columns": {"type": "array", "items": {"type": "string"}},
        "loans": {"type": "array", "items": {"type": "object"}},
        "row_count": {"type": "integer"},
        "returned_count": {"type": "integer"},
        "truncated": {"type": "boolean"},
        "loans_in_numerator": {"type": ["integer", "null"]},
        "numerator_value": {"type": ["number", "null"]},
        "denominator_value": {"type": ["number", "null"]},
        "denominator_basis": {"type": ["string", "null"]},
        "reconciles": {
            "type": "boolean",
            "description": ("True when the returned population's size equals the "
                            "test's loans_in_numerator. False means the "
                            "drill-through and the evaluated figure disagree, and "
                            "the figure should not be attributed to these loans."),
        },
        "reporting_date": {"type": ["string", "null"]},
        "configuration_version": {"type": ["string", "null"]},
        "scope": {"type": "object"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    required=["resource", "available", "loans", "row_count", "returned_count",
              "truncated", "reconciles", "scope"],
)


def covenant_drillthrough(args: Dict[str, Any],
                          inv: ToolInvocation) -> Dict[str, Any]:
    """Which loans make up a covenant test's numerator. Wraps the existing engine."""
    from mi_agent_api import concentration_tests_api as conc_mod

    resolved = inv.authorised.resource
    scope = _refuse_unpartitionable(inv, "Concentration drill-through")
    deps = inv.dependencies
    limit = min(int(args.get("limit") or 100), MAX_DRILLTHROUGH_ROWS)

    driller = (getattr(deps, "drillthrough_evaluator", None)
               or conc_mod.compute_drillthrough)
    out = driller(getattr(deps, "output_root", None), inv.tenant_id,
                  args.get("as_of_run_id"), str(args.get("test_id") or ""),
                  scope=scope, max_rows=limit)

    rows = [{k: _jsonable(v) for k, v in row.items()}
            for row in (out.get("rows") or [])]
    warnings: List[str] = []
    if not out.get("available"):
        # "No population" and "no result" both produce an empty list.
        warnings.append(
            "No contributing population is available for this test"
            + (f": {out.get('reason')}" if out.get("reason") else ".")
            + " This is an absence of evidence, not an empty breach.")
    elif not out.get("reconciles", True):
        warnings.append(
            "This population does NOT reconcile to the evaluated test: its size "
            f"({out.get('rowCount')}) differs from the test's numerator loan "
            f"count ({out.get('loansInNumerator')}). Do not attribute the "
            "reported figure to these loans.")
    if out.get("truncated"):
        warnings.append(
            f"{out.get('rowCount')} loans contribute; the first {len(rows)} are "
            "returned. The totals reported here are for the FULL population, "
            "not for the rows shown.")

    inv.telemetry.returned(len(rows))
    inv.telemetry.count("contributing_loans", int(out.get("rowCount") or 0))
    return {
        "resource": resolved.ref.key,
        "available": bool(out.get("available")),
        "reason": out.get("reason"),
        "test_id": out.get("testId") or args.get("test_id"),
        "metric_id": out.get("metricId"),
        "display_name": out.get("displayName"),
        "columns": list(out.get("columns") or ()),
        "loans": rows,
        "row_count": int(out.get("rowCount") or 0),
        "returned_count": len(rows),
        "truncated": bool(out.get("truncated")),
        "loans_in_numerator": out.get("loansInNumerator"),
        "numerator_value": out.get("numeratorValue"),
        "denominator_value": out.get("denominatorValue"),
        "denominator_basis": out.get("denominatorBasis"),
        "reconciles": bool(out.get("reconciles")),
        "reporting_date": out.get("reportingDate"),
        "configuration_version": out.get("configurationVersion"),
        "scope": _scope_block(inv),
        "warnings": warnings,
    }


# =========================================================================== #
# period_change
# =========================================================================== #
PERIOD_CHANGE_INPUT = object_schema(
    description=("What changed between two governed reporting periods, and why. "
                 "Trakt resolves which periods are comparable — do not compare "
                 "figures you gathered from separate calls yourself."),
    properties={
        "resource": _RESOURCE_PROPERTY,
        "fields": {
            "type": ["array", "null"], "items": {"type": "string"},
            "description": ("Canonical fields to analyse. Omit for the portfolio "
                            "overview, which selects the fields that moved."),
            "default": None,
        },
        "as_of_run_id": {
            "type": ["string", "null"],
            "description": "Optional end run. Omit for the current position.",
            "default": None,
        },
        "include_bridge": {
            "type": "boolean",
            "description": ("Include the balance bridge — how the opening balance "
                            "became the closing one (redemptions, additions, "
                            "amortisation)."),
            "default": True,
        },
    },
    required=["resource"],
)

PERIOD_CHANGE_OUTPUT = object_schema(
    description="The governed period-change envelope, carried through verbatim.",
    properties={
        "resource": {"type": "string"},
        "available": {"type": "boolean"},
        "reason": {"type": ["string", "null"]},
        "period_resolution": {
            "type": ["object", "null"],
            "description": ("Which two periods were compared and why they are "
                            "comparable. Read this before the numbers."),
        },
        "summary": {"type": ["object", "null"]},
        "metric_changes": {"type": "array", "items": {"type": "object"}},
        "distribution_changes": {"type": "array", "items": {"type": "object"}},
        "balance_bridge": {"type": ["object", "null"]},
        "dataset_provenance": {"type": "array", "items": {"type": "object"}},
        "limitations": {
            "type": "array", "items": {"type": "string"},
            "description": ("What this comparison cannot support. Not optional "
                            "reading: a period comparison across a changed "
                            "population is a different question."),
        },
        "workflow_id": {"type": ["string", "null"]},
        "scope": {"type": "object"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    required=["resource", "available", "metric_changes", "limitations", "scope"],
)


def period_change(args: Dict[str, Any], inv: ToolInvocation) -> Dict[str, Any]:
    """Movement between two governed periods. Wraps the period-change workflow.

    Trakt resolves which periods are comparable. That is the part an agent must
    not do for itself: comparing a figure from one call against a figure it
    remembers from another silently assumes the two snapshots covered the same
    population, and a portfolio that acquired a book between them did not.
    """
    from mi_agent.period_change.models import PeriodChangeFailure
    from mi_agent_api import period_change_route as pc_mod

    resolved = inv.authorised.resource
    scope = _refuse_unpartitionable(inv, "Period-change analysis")
    deps = inv.dependencies

    analyse = (getattr(deps, "period_change_analyser", None)
               or pc_mod.analyse_period_change)
    try:
        outcome = analyse(
            client_id=inv.tenant_id,
            output_root=getattr(deps, "output_root", None),
            requested_fields=tuple(str(f) for f in (args.get("fields") or ())),
            to_run_id=args.get("as_of_run_id"),
            scope=scope,
            tenant_id=inv.tenant_id,
            authorised_portfolio_ids=tuple(resolved.source_portfolio_ids),
            include_bridge=bool(args.get("include_bridge", True)))
    except PeriodChangeFailure as failure:
        # A refusal the workflow OWNS — no comparable prior period, an
        # incomparable population — is a real answer, not a tool fault. It is
        # returned rather than raised so the agent reads the reason.
        return {
            "resource": resolved.ref.key, "available": False,
            "reason": str(failure), "period_resolution": None, "summary": None,
            "metric_changes": [], "distribution_changes": [],
            "balance_bridge": None, "dataset_provenance": [],
            "limitations": [], "workflow_id": None,
            "scope": _scope_block(inv),
            "warnings": [f"No period comparison is available: {failure}. This is "
                         "an absence of evidence, not an absence of movement."],
        }

    envelope = outcome.to_dict() if hasattr(outcome, "to_dict") else dict(outcome)
    warnings = list(envelope.get("warnings") or ())
    limitations = list(envelope.get("limitations") or ())
    if limitations:
        warnings.append(
            "This comparison carries stated limitations; read 'limitations' "
            "before quoting any movement.")

    inv.telemetry.returned(len(envelope.get("metric_changes") or ()))
    return {
        "resource": resolved.ref.key,
        "available": True,
        "reason": None,
        "period_resolution": envelope.get("period_resolution"),
        "summary": envelope.get("summary"),
        "metric_changes": list(envelope.get("metric_changes") or ()),
        "distribution_changes": list(envelope.get("distribution_changes") or ()),
        "balance_bridge": envelope.get("balance_bridge"),
        "dataset_provenance": list(envelope.get("dataset_provenance") or ()),
        "limitations": limitations,
        "workflow_id": envelope.get("workflow_id"),
        "scope": _scope_block(inv),
        "warnings": warnings,
    }
