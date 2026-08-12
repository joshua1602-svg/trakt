"""trakt_tools.handlers.capability — what can Trakt tell you about this book?

One tool, ``portfolio_capabilities``. It answers the question an autonomous
consumer should ask *first*, and today has no way to ask: **what analytical
capabilities are available for this portfolio, and where they are not, why?**

Without it an agent has to discover Trakt's knowledge by calling tools blindly
and interpreting failures — which is both expensive and unreliable, because a
tool that returns nothing looks the same whether the metric is inapplicable,
the data is missing, or the call was wrong.

Cheap on purpose
----------------
Discovery resolves availability from column presence, snapshot count and two
enum mixes. It does **not** compute the metrics it describes. *Can calculate*
is not *calculate now*: surveying a portfolio must not cost a cash-flow
enumeration over every loan, or agents will stop surveying.

No second catalogue
-------------------
Everything published here comes from ``config/system/mi_capability_registry.yaml``
via ``trakt_core.capability``. This module resolves the governed frame and
translates; it holds no capability list of its own.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..schema import object_schema
from ..spec import ToolInvocation
from .analysis import _apply_filters
from .loans import _scope_block, resolve_governed_frame

_RESOURCE_PROPERTY = {
    "type": "string",
    "description": "The portfolio resource, as '{tenant}/{kind}/{resource_id}'.",
    "pattern": r"^[^/]+/[^/]+/[^/]+$",
}

CAPABILITIES_INPUT = object_schema(
    description=(
        "The analytical capabilities Trakt can produce for this portfolio, "
        "and the reason for each one it cannot. Resolved from the governed "
        "data without computing any of the metrics."),
    properties={
        "resource": _RESOURCE_PROPERTY,
        "categories": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": ("Restrict to named categories — exposure, "
                            "collateral, delinquency, prepayment, loss, "
                            "pricing, cashflow, history, data_quality."),
            "default": None,
        },
        "status": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": ("Restrict to named availability states. Asking "
                            "for AVAILABLE alone gives the working set; "
                            "asking for everything else gives the reasons."),
            "default": None,
        },
        "include_definition": {
            "type": "boolean",
            "description": ("Include each capability's methodology, "
                            "calculation source, unit and basis alongside its "
                            "status."),
            "default": False,
        },
        "filters": {"type": ["object", "null"], "default": None},
    },
    required=["resource"],
)

CAPABILITIES_OUTPUT = object_schema(
    description="The capability catalogue, resolved for one portfolio.",
    properties={
        "resource": {"type": "string"},
        "registry_version": {"type": "integer"},
        "metrics": {"type": "array", "items": {"type": "object"}},
        "summary": {"type": "object"},
        "portfolio_shape": {"type": "object"},
        "scope": {"type": "object"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    required=["resource", "metrics", "summary", "scope"],
)


def portfolio_capabilities(args: Dict[str, Any],
                           inv: ToolInvocation) -> Dict[str, Any]:
    """Wraps ``trakt_core.capability`` — shared, not agent-specific."""
    from trakt_core.capability import (
        AVAILABLE,
        METHODOLOGY_NOT_APPROVED,
        MODEL_REQUIRED,
        NOT_APPLICABLE,
        describe_portfolio,
        load_registry,
        resolve_all,
        summarise,
    )

    df = resolve_governed_frame(inv)
    inv.telemetry.scanned(len(df))
    df = _apply_filters(df, args.get("filters"), inv)

    history_periods = _history_periods(inv)
    shape = describe_portfolio(df, history_periods=history_periods)
    registry = load_registry()

    results = resolve_all(shape, registry=registry,
                          categories=args.get("categories") or None)
    wanted_status = set(args.get("status") or ())
    if wanted_status:
        results = [r for r in results if r.status in wanted_status]

    include_definition = bool(args.get("include_definition"))
    metrics: List[Dict[str, Any]] = []
    for result in results:
        entry = result.to_dict()
        if include_definition:
            capability = registry.get(result.metric)
            if capability is not None:
                entry["definition"] = capability.to_dict()
        metrics.append(entry)

    summary = summarise(results)
    inv.telemetry.returned(len(metrics))

    warnings: List[str] = [
        "Availability is resolved WITHOUT computing any metric, so an "
        "AVAILABLE status means the inputs and methodology conditions are "
        "met — not that a value has been calculated. Request the metric to "
        "obtain one.",
    ]
    if summary.get(NOT_APPLICABLE):
        warnings.append(
            f"{summary[NOT_APPLICABLE]} capabilities are NOT_APPLICABLE. That "
            "is a property of this portfolio's economics, not a data gap: no "
            "field could be supplied to change it, and it should not be "
            "reported as a deficiency.")
    if summary.get(METHODOLOGY_NOT_APPROVED):
        warnings.append(
            f"{summary[METHODOLOGY_NOT_APPROVED]} capabilities have the data "
            "but no settled definition. Do not substitute a plausible formula "
            "— report them as outstanding methodology decisions.")
    if summary.get(MODEL_REQUIRED):
        warnings.append(
            f"{summary[MODEL_REQUIRED]} capabilities need behavioural "
            "modelling Trakt does not perform. The deterministic alternatives "
            "named in each explanation are the supported answer.")

    return {
        "resource": inv.authorised.resource.ref.key,
        "registry_version": registry.version,
        "metrics": metrics,
        "summary": summary,
        "portfolio_shape": {
            "rows": shape.row_count,
            "history_periods": shape.history_periods,
            "amortisation_mix": {k: round(v, 6)
                                 for k, v in shape.amortisation_mix.items()},
            "rate_type_mix": {k: round(v, 6)
                              for k, v in shape.rate_type_mix.items()},
        },
        "scope": _scope_block(inv),
        "warnings": warnings,
    }


def _history_periods(inv: ToolInvocation) -> int:
    """How many governed snapshots this resource has, where the deployment can
    say. Falls back to one — the conservative answer, which reports a
    period-over-period metric as needing more history rather than claiming it
    is available and failing later."""
    try:
        from .history import _resolve_history

        return max(1, len(_resolve_history(inv)))
    except Exception:
        return 1
