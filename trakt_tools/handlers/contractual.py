"""trakt_tools.handlers.contractual — contractual cash-flow analytics.

**One tool, both metrics.** Contractual WAL and contractual YTM come off the
same enumerated schedule, so splitting them into two tools would enumerate the
book twice and let the two answers drift apart on the same loan. A caller asks
for whichever it wants; the schedule is built once either way.

No calculation lives here. Everything comes from
:mod:`analytics_lib.contractual`, which is the same module a React route or the
MI query layer would call. This handler resolves the governed frame, applies
the caller's filters, and translates.

What this tool is for, and what it refuses
------------------------------------------
The status on every result is the point of the tool, not a footnote:

    SUPPORTED            the contract determines the answer
    ASSUMPTION_REQUIRED  a floating or resetting rate makes it unknowable
    NOT_APPLICABLE       economically undefined for this product (ERM)
    FIELD_GAP            a contractual input is missing from the tape

An agent that receives ``NOT_APPLICABLE`` with a reason can report *why* there
is no WAL. An agent that receives ``null`` will either drop the metric silently
or retry forever, and on an equity-release book the honest answer is precisely
that a contractual WAL does not exist.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..schema import object_schema
from ..spec import ToolInvocation
from .analysis import _apply_filters, _balance_column, _population_block
from .loans import _scope_block, resolve_governed_frame

_RESOURCE_PROPERTY = {
    "type": "string",
    "description": "The portfolio resource, as '{tenant}/{kind}/{resource_id}'.",
    "pattern": r"^[^/]+/[^/]+/[^/]+$",
}

CONTRACTUAL_INPUT = object_schema(
    description=(
        "Contractual weighted average life and yield, derived only from terms "
        "the contract fixes. Nothing here forecasts: floating-rate exposures "
        "report ASSUMPTION_REQUIRED and lifetime mortgages report "
        "NOT_APPLICABLE rather than producing a number from a legal maturity."),
    properties={
        "resource": _RESOURCE_PROPERTY,
        "metrics": {
            "type": "array",
            "items": {"type": "string",
                      "enum": ["contractual_wal", "contractual_ytm_periodic"]},
            "description": ("Which metrics to return. Both are built from one "
                            "schedule pass, so asking for both costs no more "
                            "than asking for one."),
            "default": ["contractual_wal"],
        },
        "price_percent_of_par": {
            "type": ["number", "null"],
            "description": ("Override the price for yield, as a percentage of "
                            "par on the RREL34 scale (100 = par). Omitted, "
                            "each loan's own purchase_price is used and a loan "
                            "without one reports FIELD_GAP rather than "
                            "assuming par."),
            "default": None,
        },
        "explain_loan_id": {
            "type": ["string", "null"],
            "description": ("Return the full cash-flow schedule and "
                            "methodology explanation for one named exposure."),
            "default": None,
        },
        "filters": {"type": ["object", "null"], "default": None},
    },
    required=["resource"],
)

CONTRACTUAL_OUTPUT = object_schema(
    description="Contractual analytics, with a status on every metric.",
    properties={
        "resource": {"type": "string"},
        "as_of": {"type": ["string", "null"]},
        "contractual_wal": {"type": ["object", "null"]},
        "contractual_ytm_periodic": {"type": ["object", "null"]},
        "coverage": {"type": "object"},
        "explanation": {"type": ["object", "null"]},
        "population": {"type": "object"},
        "scope": {"type": "object"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    required=["resource", "coverage", "scope"],
)


def contractual_analytics(args: Dict[str, Any],
                          inv: ToolInvocation) -> Dict[str, Any]:
    """Wraps ``analytics_lib.contractual`` — shared, not agent-specific."""
    from analytics_lib.contractual import (
        ASSUMPTION_REQUIRED,
        NOT_APPLICABLE,
        SUPPORTED,
        contractual_ytm,
        contractual_wal,
        portfolio_wal,
        schedule_frame,
    )

    df = resolve_governed_frame(inv)
    inv.telemetry.scanned(len(df))
    df = _apply_filters(df, args.get("filters"), inv)

    wanted = list(args.get("metrics") or ["contractual_wal"])
    schedules = schedule_frame(df)
    inv.telemetry.count("schedules_built", len(schedules))

    coverage: Dict[str, int] = {}
    for schedule in schedules:
        coverage[schedule.status] = coverage.get(schedule.status, 0) + 1

    result: Dict[str, Any] = {
        "resource": inv.authorised.resource.ref.key,
        "as_of": next((s.as_of.isoformat() for s in schedules if s.as_of), None),
        "coverage": coverage,
        "population": _population_block(df, _balance_column(df)),
        "scope": _scope_block(inv),
    }

    if "contractual_wal" in wanted:
        result["contractual_wal"] = portfolio_wal(schedules).to_dict()

    if "contractual_ytm_periodic" in wanted:
        price = args.get("price_percent_of_par")
        yields = [contractual_ytm(s, price_percent_of_par=price)
                  for s in schedules if s.status == SUPPORTED
                  and s.interest_deterministic]
        solved = [y for y in yields if y.status == SUPPORTED]
        if solved:
            # Balance-weighted, because a yield is a rate and averaging rates
            # by loan count would let a £50k loan move the book as much as a
            # £5m one. The weights are the opening balances the schedules
            # were built from.
            weights = [s.opening_balance for s, y in
                       zip([s for s in schedules if s.status == SUPPORTED
                            and s.interest_deterministic], yields)
                       if y.status == SUPPORTED]
            total = sum(weights) or 1.0
            weighted = sum(y.ytm_pct * w for y, w in zip(solved, weights))
            result["contractual_ytm_periodic"] = {
                "metric": "contractual_ytm_periodic",
                "status": SUPPORTED,
                "methodology_version": solved[0].method,
                "weighted_average_ytm_pct": round(weighted / total, 6),
                "weighting": "opening contractual balance",
                "loans_included": len(solved),
                "loans_excluded": len(schedules) - len(solved),
                "day_count": solved[0].explanation.get("day_count"),
            }
        else:
            result["contractual_ytm_periodic"] = {
                "metric": "contractual_ytm_periodic",
                "status": (ASSUMPTION_REQUIRED if coverage.get(SUPPORTED)
                           else NOT_APPLICABLE),
                "reason": ("no exposure in this population has a contractually "
                           "determined interest series and a stated price. A "
                           "rate path is not assumed and a missing price is "
                           "not treated as par."),
            }

    explain_id = args.get("explain_loan_id")
    if explain_id:
        match = next((s for s in schedules if s.loan_id == str(explain_id)),
                     None)
        if match is None:
            result["explanation"] = {
                "loan_id": explain_id,
                "reason": "no exposure with that identifier in this population",
            }
        else:
            block = match.to_dict()
            block["contractual_wal"] = contractual_wal(match).to_dict()
            block["contractual_ytm_periodic"] = contractual_ytm(
                match, price_percent_of_par=args.get("price_percent_of_par"),
                loan=None).to_dict()
            result["explanation"] = block

    warnings: List[str] = [
        "CONTRACTUAL, not observed. These are the cash flows the contract "
        "fixes, not what borrowers are expected to do. Compare with the "
        "OBSERVED prepayment and loss metrics before drawing a conclusion.",
    ]
    if coverage.get(NOT_APPLICABLE):
        warnings.append(
            f"{coverage[NOT_APPLICABLE]} exposure(s) report NOT_APPLICABLE. "
            "For a lifetime mortgage that is the correct answer, not a gap: "
            "repayment is contingent on a borrower event, so no contractual "
            "WAL exists. Do not substitute the legal maturity.")
    if coverage.get(ASSUMPTION_REQUIRED):
        warnings.append(
            f"{coverage[ASSUMPTION_REQUIRED]} exposure(s) require a rate "
            "assumption Trakt does not hold, so they are excluded rather than "
            "projected.")
    result["warnings"] = warnings

    inv.telemetry.returned(coverage.get(SUPPORTED, 0))
    return result
