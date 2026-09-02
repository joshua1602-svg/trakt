"""trakt_tools.handlers.portfolio_review — the primitives a period review needs.

    capability: risk:read  (every tool here is an aggregate)

WHAT THESE ARE FOR
------------------
The existing tool surface answers questions about a funded book at a point in
time, and ``period_change`` answers "has it moved". A review of a REPORTING
PERIOD needs two things that surface cannot reach:

* **the pipeline.** Not one tool touched it, and ``covenants`` refuses a
  pipeline-pinned resource outright, so an agent asked what happened this week
  had nothing to call;
* **why the funded book moved.** ``period_change`` says the balance rose £72m.
  Whether that was lending, redemptions or a book arriving is a different fact,
  and the one a reader acts on.

FIVE TOOLS, NOT ONE PER QUESTION
--------------------------------
Each is a reusable governed primitive an agent composes, not a canned answer:
position, movement, conversion, composition, forward risk. "Which product drove
the pipeline" and "did London grow" are the same ``pipeline_movement`` call read
two ways, and a tool for each would be a checklist wearing a tool surface.

NOTHING HERE COMPUTES
---------------------
Every handler resolves a governed service and re-keys its output. The services
are the ones the dashboard, the weekly brief and the notification resolver
already call — ``pipeline_contract``, ``movement_detail``, ``evolution``,
``funded_composition``, ``concentration_tests_api`` — so an agent and the
workspace cannot be given different numbers for the same week.

BOUNDED, AND NO LOAN ROWS
-------------------------
Contributor lists are capped and say when they were capped. Nothing here returns
a loan identifier or a per-case figure: a pipeline movement is reported as
dimension aggregates, which is the same discipline the governed insight contract
holds itself to.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from trakt_core.errors import ErrorCode, TraktError

from ..schema import object_schema
from ..spec import ToolInvocation
from .loans import _scope_block

#: Contributors returned per dimension. The agent drills with a second call
#: rather than being handed a league table it did not ask for.
MAX_CONTRIBUTORS = 5

#: Populations these tools may be pinned to.
_PIPELINE = "pipeline"
_FUNDED = "funded"

_RESOURCE_PROPERTY = {
    "type": "string",
    "description": ("The portfolio to review, as '{tenant}/{kind}/{resource_id}'."),
    "pattern": r"^[^/]+/[^/]+/[^/]+$",
}


# --------------------------------------------------------------------------- #
# Scope guards
# --------------------------------------------------------------------------- #
def _refuse_spv(inv: ToolInvocation, what: str) -> None:
    """An SPV boundary the weekly and funded engines cannot apply.

    Checked here rather than assumed from a sibling tool: a guard skipped
    because another tool performs it is one refactor away from being a way
    around it.
    """
    resolved = inv.authorised.resource
    if resolved.spv_id:
        raise TraktError(
            ErrorCode.RESOURCE_NOT_PARTITIONABLE,
            f"This resource is defined by an SPV boundary, which {what} cannot "
            "currently apply. Answering it would report on the enclosing book "
            "rather than the SPV.",
            request_id=inv.request_id, details={"resource": resolved.ref.key})


def _pipeline_scope(inv: ToolInvocation, what: str) -> Optional[str]:
    """The governed scope for a PIPELINE question.

    Unlike the funded tools, a resource pinned to ``pipeline`` is the natural
    case here rather than a refusal — but one pinned to ``funded`` is refused,
    for the same reason and in the same direction: answering it would report on
    a population the resource does not name.
    """
    resolved = inv.authorised.resource
    _refuse_spv(inv, what)
    if resolved.population and resolved.population != _PIPELINE:
        raise TraktError(
            ErrorCode.RESOURCE_NOT_PARTITIONABLE,
            f"{what} is evaluated on the weekly pipeline. This resource is "
            f"pinned to the {resolved.population!r} population, so it cannot be "
            "answered without reporting on data it does not name.",
            request_id=inv.request_id,
            details={"resource": resolved.ref.key,
                     "population": resolved.population})
    return getattr(resolved, "portfolio_context", None) or None


def _funded_scope(inv: ToolInvocation, what: str):
    resolved = inv.authorised.resource
    _refuse_spv(inv, what)
    if resolved.population and resolved.population != _FUNDED:
        raise TraktError(
            ErrorCode.RESOURCE_NOT_PARTITIONABLE,
            f"{what} is evaluated on the funded book. This resource is pinned "
            f"to the {resolved.population!r} population, so it cannot be "
            "answered without reporting on data it does not name.",
            request_id=inv.request_id,
            details={"resource": resolved.ref.key,
                     "population": resolved.population})
    scope = resolved.to_portfolio_scope()
    if scope is None and not resolved.whole_tenant_book:
        raise TraktError(
            ErrorCode.RESOURCE_NOT_PARTITIONABLE,
            "This resource declares no book narrowing and is not registered as "
            "the tenant's whole book, so the population it means is undefined.",
            request_id=inv.request_id, details={"resource": resolved.ref.key})
    return scope


def _pipeline_root(inv: ToolInvocation) -> str:
    root = getattr(inv.dependencies, "pipeline_root", None)
    if not root:
        raise TraktError(
            ErrorCode.DATA_SOURCE_UNAVAILABLE,
            "No governed weekly pipeline root is configured for this "
            "deployment, so no pipeline question can be answered. This is an "
            "absence of data, not an absence of pipeline.",
            request_id=inv.request_id)
    return str(root)


def _output_root(inv: ToolInvocation) -> Any:
    return getattr(inv.dependencies, "output_root", None)


def _unavailable(inv: ToolInvocation, reason: str, **extra: Any) -> Dict[str, Any]:
    """A governed non-answer. Returned, never raised.

    A refusal an agent can read is a finding — "this book has no comparable
    prior week" is worth reporting — and an exception it has to catch and
    interpret is not.
    """
    return {"resource": inv.authorised.resource.ref.key, "available": False,
            "reason": reason, "scope": _scope_block(inv),
            "warnings": [f"No answer is available: {reason}. This is an absence "
                         "of evidence, not an absence of movement."],
            **extra}


def _capped(rows: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    return list(rows or ())[:MAX_CONTRIBUTORS]


# =========================================================================== #
# 1. pipeline_position
# =========================================================================== #
POSITION_INPUT = object_schema(
    description=("The current governed weekly pipeline position: case count, "
                 "value, weighted expected funding, and the breakdown by stage."),
    properties={
        "resource": _RESOURCE_PROPERTY,
        "as_of": {"type": "string",
                  "description": ("Weekly extract date (YYYY-MM-DD). Omit for "
                                  "the latest governed extract.")},
    },
    required=["resource"],
)

POSITION_OUTPUT = object_schema(
    description="The governed weekly pipeline position.",
    properties={
        "resource": {"type": "string"},
        "available": {"type": "boolean"},
        "reason": {"type": ["string", "null"]},
        "as_of_date": {"type": ["string", "null"]},
        "position": {"type": "object",
                     "description": ("Case count, pipeline amount and weighted "
                                     "expected funded amount.")},
        "stages": {"type": "array", "items": {"type": "object"},
                   "description": "Case count and amount per governed stage."},
        "expected_completion": {"type": "array", "items": {"type": "object"}},
        "data_quality": {"type": "object"},
        "scope": {"type": "object"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    required=["resource", "available", "scope"],
)


def pipeline_position(args: Dict[str, Any], inv: ToolInvocation) -> Dict[str, Any]:
    """Where the pipeline stands. Wraps ``compute_pipeline_snapshot``."""
    from mi_agent_api import pipeline_contract as pipeline_mod

    scope = _pipeline_scope(inv, "A pipeline position")
    root = _pipeline_root(inv)

    inventory = pipeline_mod.weekly_extract_inventory(root, inv.tenant_id)
    extracts = inventory.get("extracts") or []
    if not extracts:
        return _unavailable(inv, "no governed weekly pipeline extract is "
                                 "available for this portfolio")

    as_of = args.get("as_of")
    chosen = (next((e for e in extracts
                    if e.get("pipeline_extract_date") == as_of), None)
              if as_of else extracts[-1])
    if chosen is None:
        return _unavailable(
            inv, f"no governed weekly pipeline extract matches {as_of!r}")

    frame, prep = pipeline_mod.load_prepared_pipeline(chosen)
    snapshot = pipeline_mod.compute_pipeline_snapshot(
        frame, prep, client_id=inv.tenant_id, scope=scope)
    inv.telemetry.scanned(len(frame))

    return {
        "resource": inv.authorised.resource.ref.key,
        "available": True,
        "reason": None,
        "as_of_date": chosen.get("pipeline_extract_date"),
        "position": {
            "case_count": snapshot.get("rowCount"),
            "pipeline_amount": snapshot.get("pipelineAmount"),
            "weighted_expected_funded_amount":
                snapshot.get("weightedExpectedFundedAmount"),
        },
        "stages": list(snapshot.get("stageBreakdown") or ()),
        "expected_completion": list(snapshot.get("expectedCompletion") or ()),
        "data_quality": snapshot.get("dataQuality") or {},
        "scope": _scope_block(inv),
        "warnings": list(snapshot.get("warnings") or ()),
    }


# =========================================================================== #
# 2. pipeline_movement
# =========================================================================== #
MOVEMENT_INPUT = object_schema(
    description=("Week-on-week pipeline movement with its governed attribution: "
                 "which brokers, regions and PRODUCTS moved the number, and the "
                 "new / removed / progressed / repriced decomposition behind it."),
    properties={
        "resource": _RESOURCE_PROPERTY,
        "as_of": {"type": "string",
                  "description": ("Weekly extract date (YYYY-MM-DD). Omit for "
                                  "the latest governed extract.")},
        "measure": {
            "type": "string", "enum": ["pipeline", "completions"],
            "default": "pipeline",
            "description": ("'pipeline' is open pipeline exposure; "
                            "'completions' is cases reaching COMPLETED stage — "
                            "a pipeline-stage measure, NOT funded balance."),
        },
    },
    required=["resource"],
)

MOVEMENT_OUTPUT = object_schema(
    description="Governed weekly pipeline movement and attribution.",
    properties={
        "resource": {"type": "string"},
        "available": {"type": "boolean"},
        "reason": {"type": ["string", "null"]},
        "as_of_date": {"type": ["string", "null"]},
        "comparison_date": {"type": ["string", "null"]},
        "headline": {"type": "object"},
        "counts": {"type": "object"},
        "contributors": {
            "type": "object",
            "description": ("Dimension aggregates per governed dimension "
                            "(brokers, regions, products). Each dimension is a "
                            "separate decomposition of the SAME movement and "
                            "sums to it on its own — they are not additive with "
                            "one another."),
        },
        "components": {"type": "object"},
        "methodology": {"type": "object"},
        "scope": {"type": "object"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    required=["resource", "available", "scope"],
)


def pipeline_movement(args: Dict[str, Any], inv: ToolInvocation) -> Dict[str, Any]:
    """What moved the pipeline. Wraps ``resolve_movement_detail``."""
    from mi_agent_api import movement_detail as md

    scope = _pipeline_scope(inv, "A pipeline movement")
    root = _pipeline_root(inv)
    detail_type = (md.DETAIL_COMPLETIONS
                   if str(args.get("measure") or "pipeline") == "completions"
                   else md.DETAIL_PIPELINE)

    payload = md.resolve_movement_detail(
        root, inv.tenant_id, detail_type, as_of=args.get("as_of"), scope=scope,
        top_n=MAX_CONTRIBUTORS)

    if not payload.get("available"):
        return _unavailable(
            inv, payload.get("reason") or "no comparable prior week is available",
            as_of_date=payload.get("as_of_date"),
            comparison_date=payload.get("comparison_date"))

    contributors = {k: _capped(v)
                    for k, v in (payload.get("contributors") or {}).items()}
    inv.telemetry.returned(sum(len(v) for v in contributors.values()))
    return {
        "resource": inv.authorised.resource.ref.key,
        "available": True,
        "reason": None,
        "as_of_date": payload.get("as_of_date"),
        "comparison_date": payload.get("comparison_date"),
        "headline": payload.get("headline_metric") or {},
        "counts": payload.get("counts") or {},
        "contributors": contributors,
        "components": payload.get("components") or {},
        "methodology": payload.get("methodology") or {},
        "scope": _scope_block(inv),
        "warnings": [],
    }


# =========================================================================== #
# 3. pipeline_conversion
# =========================================================================== #
CONVERSION_INPUT = object_schema(
    description=("How the pipeline converts: the stage funnel, the governed "
                 "conversion rate over its observation window, and the weekly "
                 "completion flow."),
    properties={"resource": _RESOURCE_PROPERTY},
    required=["resource"],
)

CONVERSION_OUTPUT = object_schema(
    description="Governed origination funnel and conversion.",
    properties={
        "resource": {"type": "string"},
        "available": {"type": "boolean"},
        "reason": {"type": ["string", "null"]},
        "summary": {"type": "object",
                    "description": "Per-stage levels, flow and conversion."},
        "sufficient": {
            "type": ["boolean", "null"],
            "description": ("False when the observation window is too short for "
                            "the rate to be published. A rate marked "
                            "insufficient must not be quoted as the book's "
                            "conversion."),
        },
        "weeks_in_window": {"type": ["integer", "null"]},
        "lag_weeks": {"type": ["integer", "null"]},
        "scope": {"type": "object"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    required=["resource", "available", "scope"],
)


def pipeline_conversion(args: Dict[str, Any],
                        inv: ToolInvocation) -> Dict[str, Any]:
    """Funnel and conversion. Wraps ``pipeline_funnel_evolution``.

    ``lag_weeks`` is passed exactly as the dashboard and the weekly brief pass
    it. Omitting it computes the rate UNLAGGED — a different, larger number than
    every other surface publishes for the same week.
    """
    from mi_agent_api import datasets as datasets_mod
    from mi_agent_api import evolution as evolution_mod

    _pipeline_scope(inv, "A conversion analysis")
    root = _pipeline_root(inv)

    history = datasets_mod._pipeline_history(inv.tenant_id)
    funnel = evolution_mod.pipeline_funnel_evolution(
        root, inv.tenant_id, None,
        lag_weeks=datasets_mod._kfi_lag_weeks_from_model(history),
        historical_model=history)

    summary = (funnel or {}).get("summary") or {}
    if not summary:
        return _unavailable(inv, "no governed origination funnel is available "
                                 "for this portfolio")

    completed = summary.get("COMPLETED") or {}
    conversion = completed.get("conversion") or {}
    return {
        "resource": inv.authorised.resource.ref.key,
        "available": True,
        "reason": None,
        "summary": summary,
        "sufficient": conversion.get("sufficient"),
        "weeks_in_window": conversion.get("weeksInWindow"),
        "lag_weeks": datasets_mod._kfi_lag_weeks_from_model(history),
        "scope": _scope_block(inv),
        "warnings": ([] if conversion.get("sufficient") is not False else
                     ["The governed observation window is too short to publish "
                      "a conversion rate; do not quote one."]),
    }


# =========================================================================== #
# 4. funded_composition
# =========================================================================== #
COMPOSITION_INPUT = object_schema(
    description=("WHY the funded book moved: new lending, redemptions and "
                 "exits, existing-book movement, and any source portfolio "
                 "ADDED or DISPOSED of this period. Use this whenever a funded "
                 "movement needs explaining — the headline alone cannot "
                 "distinguish organic growth from a book arriving."),
    properties={
        "resource": _RESOURCE_PROPERTY,
        "as_of_run_id": {"type": "string",
                         "description": "Governed reporting run. Omit for the latest."},
        "span_periods": {"type": "integer", "minimum": 1, "default": 1,
                         "description": ("How many governed reporting periods "
                                         "back to compare. 1 is month-on-month.")},
        "underlying_only": {
            "type": "boolean", "default": False,
            "description": ("True decomposes the EXISTING book only, excluding "
                            "portfolios added this period — so an acquisition "
                            "cannot hide a movement in the incumbent book."),
        },
    },
    required=["resource"],
)

COMPOSITION_OUTPUT = object_schema(
    description="The governed funded movement decomposition.",
    properties={
        "resource": {"type": "string"},
        "available": {"type": "boolean"},
        "reason": {"type": ["string", "null"]},
        "current_reporting_date": {"type": ["string", "null"]},
        "prior_reporting_date": {"type": ["string", "null"]},
        "opening_balance": {"type": ["number", "null"]},
        "closing_balance": {"type": ["number", "null"]},
        "movement": {"type": ["number", "null"]},
        "components": {
            "type": "object",
            "description": ("portfolio_additions, portfolio_disposals, "
                            "organic_new_lending, exits, existing_book_movement. "
                            "A null component was not derivable and is named in "
                            "'unavailable'; it is not zero."),
        },
        "portfolio_additions": {
            "type": "array", "items": {"type": "object"},
            "description": ("Source portfolios present now and absent prior. "
                            "'portfolio_type' is acquired / direct / "
                            "unclassified, resolved from governed identity — "
                            "NEVER from the size of the movement. An "
                            "unclassified addition is a new source portfolio "
                            "and must not be described as an acquisition."),
        },
        "portfolio_disposals": {"type": "array", "items": {"type": "object"}},
        "dominant_addition": {
            "type": ["object", "null"],
            "description": ("The largest addition, with its governed shares. "
                            "Null when nothing was added."),
        },
        "addition_share_of_movement": {
            "type": ["number", "null"],
            "description": ("The dominant addition's share of the movement, as "
                            "a fraction. Governed: do NOT divide the addition "
                            "by the movement yourself. Null where the movement "
                            "is zero or negative, or the addition exceeds it — "
                            "in which case use "
                            "'addition_share_of_closing_balance' instead, "
                            "because a share of a smaller movement would read "
                            "as over 100%."),
        },
        "addition_share_of_closing_balance": {
            "type": ["number", "null"],
            "description": ("The dominant addition's share of the closing "
                            "balance, as a fraction. Governed."),
        },
        "continuing_portfolio_ids": {"type": "array", "items": {"type": "string"}},
        "counts": {"type": "object"},
        "reconciliation": {
            "type": "object",
            "description": ("Components sum to the movement by construction. "
                            "Check 'reconciles' before quoting a component."),
        },
        "unavailable": {"type": "object"},
        "scope": {"type": "object"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    required=["resource", "available", "scope"],
)


def funded_composition(args: Dict[str, Any],
                       inv: ToolInvocation) -> Dict[str, Any]:
    """Why the funded book moved. Wraps ``composition_movement``."""
    from mi_agent_api import funded_composition as comp

    scope = _funded_scope(inv, "A funded composition analysis")
    output_root = _output_root(inv)
    if not output_root:
        return _unavailable(inv, "no governed funded output root is configured "
                                 "for this deployment")

    span = max(1, int(args.get("span_periods") or 1))
    payload = comp.composition_movement(
        output_root, inv.tenant_id, to_run_id=args.get("as_of_run_id"),
        span_periods=span, scope=scope)

    if payload.get("available") and args.get("underlying_only"):
        filters = comp.underlying_lens_filters(payload)
        if filters is None:
            return _unavailable(
                inv, "no source portfolio was added this period, so the "
                     "underlying book is the whole book — ask for the whole "
                     "book instead of an underlying view of it")
        payload = comp.composition_movement(
            output_root, inv.tenant_id, to_run_id=args.get("as_of_run_id"),
            span_periods=span, scope=scope, lens_filters=filters,
            lens_label="Underlying")

    if not payload.get("available"):
        return _unavailable(inv, payload.get("reason")
                            or "the funded movement could not be decomposed")

    reconciliation = payload.get("reconciliation") or {}
    #: The shares ``dominant_addition`` already computes. Returned because the
    #: first real-model red-team showed an agent needing exactly these two
    #: figures, finding no governed way to obtain them, and dividing the numbers
    #: itself — publishing "93% of the period's balance growth" from a division
    #: Trakt never performed. Withholding a number the deterministic layer has
    #: already calculated correctly does not stop it being stated; it only
    #: decides who calculates it.
    lead = comp.dominant_addition(payload) or {}
    warnings: List[str] = []
    if not reconciliation.get("reconciles", True):
        warnings.append(
            "The components do not sum to the movement; do not attribute the "
            "movement to them until the residual is explained.")
    for reason in (payload.get("unavailable") or {}).values():
        warnings.append(str(reason))

    return {
        "resource": inv.authorised.resource.ref.key,
        "available": True,
        "reason": None,
        "lens": payload.get("lens"),
        "current_reporting_date": payload.get("currentReportingDate"),
        "prior_reporting_date": payload.get("priorReportingDate"),
        "opening_balance": payload.get("opening_balance"),
        "closing_balance": payload.get("closing_balance"),
        "movement": payload.get("movement"),
        "components": payload.get("components") or {},
        "portfolio_additions": payload.get("portfolio_additions") or [],
        "portfolio_disposals": payload.get("portfolio_disposals") or [],
        "dominant_addition": lead or None,
        "addition_share_of_movement": lead.get("share_of_movement"),
        "addition_share_of_closing_balance": lead.get(
            "share_of_closing_balance"),
        "continuing_portfolio_ids": payload.get("continuing_portfolio_ids") or [],
        "counts": payload.get("counts") or {},
        "reconciliation": reconciliation,
        "unavailable": payload.get("unavailable") or {},
        "scope": _scope_block(inv),
        "warnings": warnings,
    }


# =========================================================================== #
# 5. forward_concentration
# =========================================================================== #
FORWARD_INPUT = object_schema(
    description=("Concentration across three clearly separated portfolio "
                 "states: funded (contractual), expected_forecast (funded plus "
                 "pipeline weighted by governed completion probability) and "
                 "full_pipeline (funded plus 100% of active pipeline — a stress "
                 "maximum, never a prediction)."),
    properties={
        "resource": _RESOURCE_PROPERTY,
        "as_of_run_id": {"type": "string",
                         "description": "Governed reporting run. Omit for the latest."},
    },
    required=["resource"],
)

FORWARD_OUTPUT = object_schema(
    description="Governed three-state concentration evaluation.",
    properties={
        "resource": {"type": "string"},
        "available": {"type": "boolean"},
        "reason": {"type": ["string", "null"]},
        "source": {
            "type": ["string", "null"],
            "description": ("'approved_configuration' or 'legacy_extracted'. A "
                            "legacy result is NOT operator-approved and must be "
                            "described as indicative."),
        },
        "reporting_date": {"type": ["string", "null"]},
        "tests": {"type": "array", "items": {"type": "object"}},
        "states": {"type": "object"},
        "emerging_risks": {
            "type": "array", "items": {"type": "object"},
            "description": ("Governed findings in the governed rank order: "
                            "current breach, expected breach, low expected "
                            "headroom, deterioration, stress-only, limitation. "
                            "The order is the engine's; do not re-rank it."),
        },
        "lineage": {"type": "object"},
        "scope": {"type": "object"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    required=["resource", "available", "scope"],
)


def forward_concentration(args: Dict[str, Any],
                          inv: ToolInvocation) -> Dict[str, Any]:
    """Where the pipeline takes the limits. Wraps ``compute_concentration_tests``.

    ``evaluate_covenants`` already publishes the funded verdict per test. This
    publishes what that verdict becomes once the pipeline lands, which is the
    question a weekly review exists to ask and the one the funded projection
    deliberately does not answer.
    """
    from mi_agent_api import concentration_tests_api as conc_mod

    scope = _funded_scope(inv, "A forward concentration analysis")
    output_root = _output_root(inv)
    if not output_root:
        return _unavailable(inv, "no governed funded output root is configured "
                                 "for this deployment")

    payload = conc_mod.compute_concentration_tests(
        output_root, inv.tenant_id, args.get("as_of_run_id"), scope=scope)

    if not payload.get("available"):
        return _unavailable(inv, payload.get("reason")
                            or "no governed concentration evaluation is available")

    warnings: List[str] = []
    if payload.get("source") == "legacy_extracted":
        warnings.append(
            "These limits were extracted rather than operator-approved. Report "
            "them as indicative and never as an approved covenant position.")
    states = payload.get("states") or {}
    if not states.get("available"):
        warnings.append(
            "Forward states are unavailable, so only the funded position is "
            "evaluated here. That is not evidence the pipeline is immaterial.")

    risks = list(payload.get("emergingRisks") or ())
    inv.telemetry.returned(len(risks))
    return {
        "resource": inv.authorised.resource.ref.key,
        "available": True,
        "reason": None,
        "source": payload.get("source"),
        "reporting_date": payload.get("reportingDate"),
        "tests": list(payload.get("tests") or ()),
        "states": states,
        "emerging_risks": risks,
        "lineage": payload.get("lineage") or {},
        "scope": _scope_block(inv),
        "warnings": warnings,
    }
