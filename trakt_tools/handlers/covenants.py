"""trakt_tools.handlers.covenants — the ``evaluate_covenants`` tool.

    capability: risk:read
    handler:    mi_agent_api.concentration_tests_api.compute_concentration_tests

This module computes **nothing**. It resolves the authorised resource into the
narrowing the existing evaluator already accepts, calls that evaluator — the
same function ``GET /mi/concentration-tests`` calls — and re-keys the result into
the tool's published output schema.

Why re-key at all: the existing envelope is camelCase and shaped for the React
Risk Limits workspace, and it carries UI-only blocks (proposal counts, legacy
envelope details, chart-oriented fields). The tool contract is snake_case, typed,
and closed. Every value is carried through **verbatim** — no rounding, no unit
conversion, no recomputation, no defaulting of a missing number to zero.
``tests/test_agent_tool_covenants.py::test_projection_carries_values_verbatim``
pins that.

Two refusals this handler owns, both because the underlying evaluator cannot
honour the constraint rather than because policy forbids it:

* **SPV narrowing.** ``ResolvedResource.predicate()`` may carry ``spv_id``, but
  the funded concentration path narrows through ``PortfolioScope``, which knows
  only about books. Serving an SPV-scoped grant here would silently return the
  enclosing book.
* **A non-funded population.** Concentration tests are evaluated on the funded
  book. A resource pinned to ``pipeline`` or ``forecast`` is refused rather than
  answered from a population it does not name.

And one fail-closed check: when the resource narrows by book, the governed tape
must actually carry ``source_portfolio_id``. ``apply_scope`` returns the frame
unchanged when it does not — fine for a UI lens, a fail-open for an entitlement.

Known cost of that check: it resolves the funded frame, so a book-scoped call
resolves it twice. It runs ONLY when the resource actually narrows by book (a
whole-book resource needs no attribution), and it is a security control, so
correctness comes before performance here. The fix is to thread the resolved
frames through to the evaluator — ``risk_limits.compute_risk_limits`` already
accepts ``prepared_funded`` for exactly this reason — which is a change to the
evaluator's signature and is deliberately not part of this sprint.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from trakt_core.errors import ErrorCode, TraktError

from ..schema import object_schema
from ..spec import ToolInvocation

#: Population a concentration test is defined on. A resource pinned to anything
#: else is refused rather than answered from the funded book it did not name.
_FUNDED = "funded"

#: Keys copied straight from one evaluated test. Left of the arrow is the tool's
#: published name; right is the existing envelope's. Nothing is derived.
_TEST_FIELDS = (
    ("test_id", "testId"),
    ("metric_id", "metricId"),
    ("display_name", "displayName"),
    ("category", "category"),
    ("current_value", "currentValue"),
    ("prior_value", "priorValue"),
    ("prior_reporting_date", "priorReportingDate"),
    ("prior_available", "priorAvailable"),
    ("absolute_change", "absoluteChange"),
    ("percentage_point_change", "percentagePointChange"),
    ("threshold", "threshold"),
    ("operator", "operator"),
    ("warning_fraction", "warningFraction"),
    ("unit", "unit"),
    ("utilisation", "utilization"),
    ("headroom", "headroom"),
    ("breach_amount", "breachAmount"),
    ("status", "status"),
    ("prior_status", "priorStatus"),
    ("status_transition", "statusTransition"),
    ("deteriorated", "deteriorated"),
    ("data_status", "dataStatus"),
    ("missing_fields", "missingFields"),
    ("numerator_value", "numeratorValue"),
    ("denominator_value", "denominatorValue"),
    ("denominator_basis", "denominatorBasis"),
    ("loans_in_numerator", "loansInNumerator"),
    ("total_loans", "totalLoans"),
    ("severity", "severity"),
    ("effective_date", "effectiveDate"),
    ("expiry_date", "expiryDate"),
    ("notes", "notes"),
    ("resolved_columns", "resolvedColumns"),
    ("parameters", "parameters"),
    ("legacy", "legacy"),
)

_SUMMARY_FIELDS = (
    ("overall_status", "overallStatus"),
    ("active_tests", "activeTests"),
    ("breaches", "breaches"),
    ("warnings", "warnings"),
    ("passes", "passes"),
    ("unavailable", "unavailable"),
    ("deteriorations", "deteriorations"),
    ("pending_effective_date", "pendingEffectiveDate"),
    ("expired", "expired"),
    ("reporting_date", "reportingDate"),
    ("prior_reporting_date", "priorReportingDate"),
    ("prior_available", "priorAvailable"),
    ("closest_to_limit", "closestToLimit"),
)


# --------------------------------------------------------------------------- #
# Published contract
# --------------------------------------------------------------------------- #
INPUT_SCHEMA = object_schema(
    description=(
        "Evaluate the operator-approved concentration and covenant tests for one "
        "authorised resource."),
    properties={
        "resource": {
            "type": "string",
            "description": (
                "The resource to evaluate, as '{tenant}/{kind}/{resource_id}' — "
                "for example 'ERE/source_portfolio/direct_001'. Call GET "
                "/v1/agent/tools to see the resources you may name."),
            "pattern": r"^[^/]+/[^/]+/[^/]+$",
        },
        "as_of_run_id": {
            "type": ["string", "null"],
            "description": (
                "Optional governed run to evaluate as at. Omit for the current "
                "approved reporting position."),
            "default": None,
        },
        "category": {
            "type": ["string", "null"],
            "description": (
                "Optional category filter, e.g. 'geographic'. Omit for every "
                "active test."),
            "default": None,
        },
    },
    required=["resource"],
)

OUTPUT_SCHEMA = object_schema(
    description=(
        "The evaluated tests, a summary, and the configuration provenance that "
        "shows which approved definitions produced them."),
    properties={
        "resource": {"type": "string"},
        "available": {"type": "boolean"},
        "source": {
            "type": "string",
            "description": (
                "approved_configuration | legacy_extracted | none. Only "
                "'approved_configuration' is operator-approved; 'legacy_extracted' "
                "is shown for continuity and is explicitly NOT approved."),
        },
        "approval_status": {"type": ["string", "null"]},
        "reporting_date": {"type": ["string", "null"]},
        "prior_reporting_date": {"type": ["string", "null"]},
        "tests": {"type": "array", "items": {"type": "object"}},
        "summary": {"type": "object"},
        "lineage": {
            "type": "object",
            "description": (
                "Which approved configuration version, hash, approver and library "
                "version produced these numbers."),
        },
        "scope": {
            "type": "object",
            "description": "The book narrowing this authorisation applied.",
        },
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    required=["resource", "available", "source", "tests", "summary", "lineage",
              "scope"],
)


# --------------------------------------------------------------------------- #
# Projection (re-keying only — never a calculation)
# --------------------------------------------------------------------------- #
def _project_test(test: Dict[str, Any]) -> Dict[str, Any]:
    out = {ours: test.get(theirs) for ours, theirs in _TEST_FIELDS}
    # Configuration provenance for this single test, carried through as-is.
    provenance = test.get("provenance") or {}
    out["provenance"] = {
        "source_reference": provenance.get("sourceReference"),
        "source_text": provenance.get("sourceText"),
        "locator": provenance.get("locator"),
        "approved_by": provenance.get("approvedBy"),
        "approved_at": provenance.get("approvedAt"),
        "proposal_id": provenance.get("proposalId"),
    }
    definition = test.get("definition") or {}
    out["definition"] = {
        "numerator": definition.get("numerator"),
        "aggregation": definition.get("aggregation"),
        "description": definition.get("description"),
        "definition_notes": definition.get("definitionNotes"),
    }
    out["configuration_version"] = test.get("configurationVersion")
    out["evaluated_at"] = test.get("evaluatedAt")
    return out


def _project_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {ours: summary.get(theirs) for ours, theirs in _SUMMARY_FIELDS}


def _project_lineage(envelope: Dict[str, Any]) -> Dict[str, Any]:
    lineage = envelope.get("lineage") or {}
    return {
        "source": lineage.get("source") or envelope.get("source"),
        "configuration_version": lineage.get("configurationVersion"),
        "configuration_hash": lineage.get("configurationHash"),
        "activated_by": lineage.get("activatedBy"),
        "activated_at": lineage.get("activatedAt"),
        "library_version": lineage.get("libraryVersion"),
        "data_source": lineage.get("dataSource"),
        "reporting_date": lineage.get("reportingDate"),
        "prior_reporting_date": lineage.get("priorReportingDate"),
        "note": lineage.get("note"),
    }


# --------------------------------------------------------------------------- #
# The handler
# --------------------------------------------------------------------------- #
def evaluate_covenants(args: Dict[str, Any], inv: ToolInvocation) -> Dict[str, Any]:
    """Evaluate approved concentration/covenant tests for the authorised resource."""
    from mi_agent_api import concentration_tests_api as conc_mod

    resolved = inv.authorised.resource
    warnings: List[str] = []

    # -- constraints this evaluator cannot honour ------------------------- #
    if resolved.population and resolved.population != _FUNDED:
        raise TraktError(
            ErrorCode.RESOURCE_NOT_PARTITIONABLE,
            "Concentration tests are evaluated on the funded book. This "
            "resource is pinned to a different population, so it cannot be "
            "answered without reporting on data it does not name.",
            request_id=inv.request_id,
            details={"resource": resolved.ref.key,
                     "population": resolved.population})
    if resolved.spv_id:
        raise TraktError(
            ErrorCode.RESOURCE_NOT_PARTITIONABLE,
            "This resource is defined by an SPV boundary, which the funded "
            "concentration evaluation cannot currently apply. Answering it would "
            "report on the enclosing book rather than the SPV.",
            request_id=inv.request_id,
            details={"resource": resolved.ref.key})

    # -- the book narrowing this authorisation permits -------------------- #
    scope = resolved.to_portfolio_scope()
    deps = inv.dependencies
    output_root = getattr(deps, "output_root", None)
    client_id = inv.tenant_id

    if scope is not None:
        # Fail closed: a scope that cannot be applied is not a narrower answer,
        # it is the whole book with a narrower label on it.
        attribution = conc_mod.funded_attribution_status(
            output_root, client_id, args.get("as_of_run_id"))
        if attribution.get("available") and not attribution.get("attributed"):
            raise TraktError(
                ErrorCode.RESOURCE_NOT_PARTITIONABLE,
                attribution.get("reason")
                or "This resource cannot be separated from the rest of the book.",
                request_id=inv.request_id,
                details={"resource": resolved.ref.key,
                         "required_fields": sorted(resolved.required_fields)})
    elif not resolved.whole_tenant_book:
        # Neither a book narrowing nor an explicit whole-book declaration. The
        # resource model forbids this shape at config load; refusing here too
        # means a catalogue that changed under a live context cannot widen a call.
        raise TraktError(
            ErrorCode.RESOURCE_NOT_PARTITIONABLE,
            "This resource declares no book narrowing and is not registered as "
            "the tenant's whole book, so the population it means is undefined.",
            request_id=inv.request_id,
            details={"resource": resolved.ref.key})

    # -- the existing evaluation, unchanged -------------------------------- #
    evaluate = getattr(deps, "covenant_evaluator", None) or \
        conc_mod.compute_concentration_tests
    envelope = evaluate(output_root, client_id, args.get("as_of_run_id"),
                        scope=scope)

    tests = [_project_test(t) for t in (envelope.get("tests") or [])]

    category = args.get("category")
    if category:
        wanted = str(category).strip().lower()
        matched = [t for t in tests if str(t.get("category") or "").lower() == wanted]
        if not matched and tests:
            # Never silently return everything when a filter matched nothing:
            # an agent would read the full book as the filtered answer.
            warnings.append(
                f"No active test carries the category {category!r}. "
                f"Available categories: "
                f"{sorted({str(t.get('category')) for t in tests if t.get('category')})}.")
        tests = matched

    if envelope.get("source") == conc_mod.SOURCE_LEGACY:
        warnings.append(
            "These limits come from the extracted Schedule 8 monitor and are NOT "
            "operator-approved concentration tests. Treat them as indicative.")

    if not envelope.get("available"):
        # An autonomous caller reads `warnings` before it reads a nested field.
        # "Nothing is configured" and "everything passed" both produce an empty
        # test list, and an agent must not report the first as the second.
        note = (envelope.get("lineage") or {}).get("note")
        warnings.append(
            "No concentration or covenant result is available for this resource"
            + (f": {note}" if note else ".")
            + " This is an absence of evidence, not a clean result.")

    return {
        "resource": resolved.ref.key,
        "available": bool(envelope.get("available")),
        "source": envelope.get("source") or conc_mod.SOURCE_NONE,
        "approval_status": envelope.get("approvalStatus"),
        "reporting_date": envelope.get("reportingDate"),
        "prior_reporting_date": envelope.get("priorReportingDate"),
        "tests": tests,
        "summary": _project_summary(envelope.get("summary") or {}),
        "lineage": _project_lineage(envelope),
        "scope": {
            "resource": resolved.ref.key,
            "display_label": resolved.display_label,
            "source_portfolio_ids": list(resolved.source_portfolio_ids),
            "population": resolved.population,
            "whole_tenant_book": resolved.whole_tenant_book,
            "attribution": resolved.attribution,
        },
        "warnings": warnings,
    }
