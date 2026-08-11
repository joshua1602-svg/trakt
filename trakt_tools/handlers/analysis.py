"""trakt_tools.handlers.analysis — aggregate tools, so an agent stops pulling rows.

    capability: risk:read  (aggregates)  /  loan:read  (anything naming loans)

The problem these exist to solve
--------------------------------
An agent given only ``get_loans`` will answer "what is the weighted-average LTV
of the London exposure?" by retrieving the London loans and averaging them
itself. That is wrong three ways, in increasing order of seriousness: it is slow,
it is expensive, and **the answer is the agent's rather than Trakt's** — a
weighted average computed in a model's context window has no governed definition,
no provenance and no reproducibility.

So every question an agent can reasonably ask about a *population* has a tool
that answers it as an aggregate. ``rank_loans`` is the deliberate bridge: it is
how an agent finds the twenty loans worth looking at without reading the other
499,980, and it returns identifiers and the ranked metric rather than rows, so
the natural next step is one bounded ``get_loans`` call.

Every calculation here is an EXISTING implementation
----------------------------------------------------
``analytics_lib.stratify`` and ``analytics_lib.concentration`` are pure,
route-agnostic and already used by the MI workflows. These handlers resolve the
governed frame, choose columns and shape the response; they compute nothing. If
one of them needed a new calculation, the calculation would belong in
``analytics_lib`` where MI and the agent both read it — not here.

Bounded by construction
-----------------------
Every tool caps what it returns: a stratification is capped at
``MAX_GROUPS`` categories, a ranking at ``MAX_RANK``. A truncated result SAYS it
was truncated, because a silently shortened league table reads as the whole one.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from trakt_core.errors import ErrorCode, TraktError

from ..schema import object_schema
from ..spec import ToolInvocation
from .loans import (
    LOAN_ID_FIELD,
    _jsonable,
    _scope_block,
    resolve_governed_frame,
)

#: The balance a share is measured against, in preference order. The canonical
#: tape carries both spellings depending on asset class.
BALANCE_FIELDS = ("current_principal_balance", "current_outstanding_balance")

#: Ceilings. A stratification with 4,000 groups is a row dump with a GROUP BY in
#: front of it, and a league table longer than this is not being read.
MAX_GROUPS = 200
MAX_RANK = 200

#: Weighted averages a summary reports when the tape carries them. Weighted by
#: balance, which is the only weighting a credit audience means by default.
SUMMARY_WEIGHTED = ("current_loan_to_value", "indexed_loan_to_value",
                    "current_interest_rate")


# --------------------------------------------------------------------------- #
# Shared pieces
# --------------------------------------------------------------------------- #
_RESOURCE_PROPERTY = {
    "type": "string",
    "description": ("The portfolio resource to analyse, as "
                    "'{tenant}/{kind}/{resource_id}'."),
    "pattern": r"^[^/]+/[^/]+/[^/]+$",
}

_FILTERS_PROPERTY = {
    "type": ["object", "null"],
    "description": ("Optional equality filters as {canonical_field: value} or "
                    "{canonical_field: [values]}. Applied before aggregation."),
    "default": None,
}


def _balance_column(df) -> Optional[str]:
    for candidate in BALANCE_FIELDS:
        if candidate in df.columns:
            return candidate
    return None


def _require_column(df, column: str, inv: ToolInvocation, role: str) -> str:
    if column not in df.columns:
        raise TraktError(
            ErrorCode.TOOL_INPUT_INVALID,
            f"This tape does not carry {column!r}, so it cannot be used as the "
            f"{role}.",
            request_id=inv.request_id,
            details={"requested": column,
                     "available": sorted(str(c) for c in df.columns)[:200]})
    return column


def _apply_filters(df, filters: Optional[Dict[str, Any]], inv: ToolInvocation):
    """Equality filters, refusing an unknown column rather than ignoring it.

    Silently dropping a filter would answer a NARROWER question with a WIDER
    population and label it correctly — the most dangerous shape of wrong.
    """
    if not filters:
        return df
    import pandas as pd

    mask = pd.Series(True, index=df.index)
    for column, want in filters.items():
        if column not in df.columns:
            raise TraktError(
                ErrorCode.TOOL_INPUT_INVALID,
                f"Cannot filter on {column!r}: this tape does not carry it. The "
                "filter was not silently ignored, because that would answer a "
                "narrower question with a wider population.",
                request_id=inv.request_id, details={"filter": column})
        series = df[column]
        if isinstance(want, (list, tuple, set)):
            mask &= series.astype(str).isin([str(v) for v in want])
        else:
            mask &= series.astype(str) == str(want)
    return df[mask]


def _table_records(table, limit: int) -> List[Dict[str, Any]]:
    head = table.head(limit)
    return [{k: _jsonable(v) for k, v in record.items()}
            for record in head.to_dict(orient="records")]


def _population_block(df, balance_column: Optional[str]) -> Dict[str, Any]:
    total = None
    if balance_column:
        import pandas as pd
        total = float(pd.to_numeric(df[balance_column], errors="coerce").sum())
    return {"loan_count": int(len(df)),
            "balance_column": balance_column,
            "total_balance": total}


# =========================================================================== #
# portfolio_summary
# =========================================================================== #
PORTFOLIO_SUMMARY_INPUT = object_schema(
    description=("The headline shape of a whole portfolio in ONE call: size, "
                 "balance, weighted averages, arrears and stage mix."),
    properties={
        "resource": _RESOURCE_PROPERTY,
        "filters": _FILTERS_PROPERTY,
    },
    required=["resource"],
)

PORTFOLIO_SUMMARY_OUTPUT = object_schema(
    description="Portfolio-level aggregates, all computed by Trakt.",
    properties={
        "resource": {"type": "string"},
        "population": {"type": "object"},
        "weighted_averages": {"type": "object"},
        "arrears": {"type": "object"},
        "composition": {"type": "object"},
        "fields_unavailable": {"type": "array", "items": {"type": "string"}},
        "scope": {"type": "object"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    required=["resource", "population", "weighted_averages", "scope"],
)


def portfolio_summary(args: Dict[str, Any], inv: ToolInvocation) -> Dict[str, Any]:
    """The first call an agent should make about a book it has not seen."""
    import pandas as pd

    df = resolve_governed_frame(inv)
    inv.telemetry.scanned(len(df))
    df = _apply_filters(df, args.get("filters"), inv)

    balance_column = _balance_column(df)
    warnings: List[str] = []
    unavailable: List[str] = []
    if balance_column is None:
        warnings.append(
            "This tape carries no balance column, so shares and weighted "
            "averages are unavailable and only counts are reported.")

    weights = (pd.to_numeric(df[balance_column], errors="coerce")
               if balance_column else None)
    weighted: Dict[str, Any] = {}
    for metric in SUMMARY_WEIGHTED:
        if metric not in df.columns:
            unavailable.append(metric)
            continue
        values = pd.to_numeric(df[metric], errors="coerce")
        if weights is not None:
            paired = pd.DataFrame({"v": values, "w": weights}).dropna()
            total_weight = float(paired["w"].sum())
            weighted[metric] = (float((paired["v"] * paired["w"]).sum()) / total_weight
                                if total_weight else None)
        else:
            weighted[metric] = (float(values.mean())
                                if values.notna().any() else None)
        weighted[f"{metric}_simple_average"] = (
            float(values.mean()) if values.notna().any() else None)

    arrears: Dict[str, Any] = {}
    if "arrears_balance" in df.columns:
        arrears_values = pd.to_numeric(df["arrears_balance"], errors="coerce")
        in_arrears = arrears_values.fillna(0) > 0
        arrears["loans_in_arrears"] = int(in_arrears.sum())
        arrears["arrears_balance"] = float(arrears_values.sum())
        if balance_column and weights is not None:
            total = float(weights.sum())
            arrears["balance_in_arrears"] = float(weights[in_arrears].sum())
            arrears["balance_in_arrears_share"] = (
                arrears["balance_in_arrears"] / total * 100 if total else None)
    else:
        unavailable.append("arrears_balance")
    if "number_of_days_in_arrears" in df.columns:
        days = pd.to_numeric(df["number_of_days_in_arrears"], errors="coerce")
        arrears["loans_over_90_days"] = int((days.fillna(0) > 90).sum())

    # Composition over the dimensions a credit reader always asks for, when the
    # tape carries them. Each is the SAME stratification the stratify tool runs.
    from analytics_lib.stratify import stratify as stratify_fn

    composition: Dict[str, Any] = {}
    for dimension in ("account_status", "ifrs9_stage",
                      "geographic_region_collateral", "collateral_type"):
        if dimension not in df.columns:
            unavailable.append(dimension)
            continue
        table = stratify_fn(df, dimension, balance_column,
                            loan_id_col=(LOAN_ID_FIELD
                                         if LOAN_ID_FIELD in df.columns else None))
        composition[dimension] = _table_records(table, MAX_GROUPS)

    inv.telemetry.returned(sum(len(v) for v in composition.values()))
    inv.telemetry.count("groups_returned",
                        sum(len(v) for v in composition.values()))
    return {
        "resource": inv.authorised.resource.ref.key,
        "population": _population_block(df, balance_column),
        "weighted_averages": weighted,
        "arrears": arrears,
        "composition": composition,
        "fields_unavailable": sorted(set(unavailable)),
        "scope": _scope_block(inv),
        "warnings": warnings,
    }


# =========================================================================== #
# stratify
# =========================================================================== #
STRATIFY_INPUT = object_schema(
    description=("Break a portfolio down by ONE dimension and return balance and "
                 "count per category. Use this instead of retrieving loans and "
                 "grouping them yourself."),
    properties={
        "resource": _RESOURCE_PROPERTY,
        "dimension": {
            "type": "string",
            "description": ("The canonical field to group by, e.g. "
                            "'geographic_region_collateral', 'account_status'."),
        },
        "weighted_metrics": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": ("Canonical fields to report as balance-weighted "
                            "averages per category, e.g. "
                            "['current_loan_to_value']."),
            "default": None,
        },
        "filters": _FILTERS_PROPERTY,
        "limit": {
            "type": "integer", "minimum": 1, "maximum": MAX_GROUPS,
            "description": f"Maximum categories to return (at most {MAX_GROUPS}).",
            "default": 50,
        },
    },
    required=["resource", "dimension"],
)

STRATIFY_OUTPUT = object_schema(
    description="One row per category, ordered by balance descending.",
    properties={
        "resource": {"type": "string"},
        "dimension": {"type": "string"},
        "groups": {"type": "array", "items": {"type": "object"}},
        "group_count": {"type": "integer"},
        "truncated": {
            "type": "boolean",
            "description": ("True when more categories exist than were returned. "
                            "A silently shortened breakdown reads as the whole "
                            "one."),
        },
        "population": {"type": "object"},
        "scope": {"type": "object"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    required=["resource", "dimension", "groups", "group_count", "truncated",
              "population", "scope"],
)


def stratify(args: Dict[str, Any], inv: ToolInvocation) -> Dict[str, Any]:
    """Group by one dimension. Wraps ``analytics_lib.stratify.stratify``."""
    from analytics_lib.stratify import stratify as stratify_fn

    df = resolve_governed_frame(inv)
    inv.telemetry.scanned(len(df))
    df = _apply_filters(df, args.get("filters"), inv)

    dimension = _require_column(df, str(args.get("dimension") or ""), inv,
                                "dimension")
    balance_column = _balance_column(df)
    warnings: List[str] = []

    requested_metrics = [str(m) for m in (args.get("weighted_metrics") or ())]
    metrics = [m for m in requested_metrics if m in df.columns]
    if len(metrics) != len(requested_metrics):
        missing = sorted(set(requested_metrics) - set(metrics))
        warnings.append(f"This tape does not carry: {', '.join(missing)}.")

    table = stratify_fn(df, dimension, balance_column,
                        loan_id_col=(LOAN_ID_FIELD if LOAN_ID_FIELD in df.columns
                                     else None),
                        weighted_metrics=metrics or None)

    limit = min(int(args.get("limit") or 50), MAX_GROUPS)
    groups = _table_records(table, limit)
    truncated = len(table) > len(groups)
    if truncated:
        warnings.append(
            f"{len(table)} categories exist; the {len(groups)} largest by "
            "balance are returned. Raise 'limit' or narrow with 'filters'.")

    inv.telemetry.returned(len(groups))
    return {
        "resource": inv.authorised.resource.ref.key,
        "dimension": dimension,
        "groups": groups,
        "group_count": int(len(table)),
        "truncated": truncated,
        "population": _population_block(df, balance_column),
        "scope": _scope_block(inv),
        "warnings": warnings,
    }


# =========================================================================== #
# concentration
# =========================================================================== #
CONCENTRATION_INPUT = object_schema(
    description=("How concentrated a portfolio is in one dimension: the top-N "
                 "groups' combined share of balance and count."),
    properties={
        "resource": _RESOURCE_PROPERTY,
        "dimension": {
            "type": "string",
            "description": ("What to measure concentration in, e.g. "
                            "'borrower_identifier', "
                            "'geographic_region_collateral'."),
        },
        "top_n": {"type": "integer", "minimum": 1, "maximum": MAX_GROUPS,
                  "description": "How many leading groups to combine.",
                  "default": 10},
        "filters": _FILTERS_PROPERTY,
    },
    required=["resource", "dimension"],
)

CONCENTRATION_OUTPUT = object_schema(
    description="Top-N concentration, with the contributing groups.",
    properties={
        "resource": {"type": "string"},
        "dimension": {"type": "string"},
        "top_n": {"type": "integer"},
        "group_count": {"type": "integer"},
        "balance_concentration_pct": {"type": ["number", "null"]},
        "count_concentration_pct": {"type": ["number", "null"]},
        "groups": {"type": "array", "items": {"type": "object"}},
        "population": {"type": "object"},
        "limit_evaluation": {
            "type": ["object", "null"],
            "description": ("Absent by design. Whether a share BREACHES a limit "
                            "is an operator-approved covenant test — call "
                            "evaluate_covenants, which carries the approved "
                            "threshold, approver and configuration version."),
        },
        "scope": {"type": "object"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    required=["resource", "dimension", "top_n", "group_count", "groups",
              "population", "scope"],
)


def concentration(args: Dict[str, Any], inv: ToolInvocation) -> Dict[str, Any]:
    """Top-N concentration. Wraps ``analytics_lib.concentration``.

    Deliberately reports the SHARE and not a pass/fail. Whether 55% in one region
    breaches anything depends on an operator-approved threshold, and that lives in
    the covenant configuration with its approver and version —
    ``evaluate_covenants`` is the tool that carries it. Answering "is this a
    breach?" here would be a second, unapproved opinion.
    """
    from analytics_lib.concentration import top_n_concentration

    df = resolve_governed_frame(inv)
    inv.telemetry.scanned(len(df))
    df = _apply_filters(df, args.get("filters"), inv)

    dimension = _require_column(df, str(args.get("dimension") or ""), inv,
                                "dimension")
    balance_column = _balance_column(df)
    if balance_column is None:
        raise TraktError(
            ErrorCode.CALCULATION_FAILED,
            "This tape carries no balance column, so a balance share cannot be "
            "computed.", request_id=inv.request_id)

    top_n = min(int(args.get("top_n") or 10), MAX_GROUPS)
    outcome = top_n_concentration(
        df, dimension, balance_column, n=top_n,
        loan_id_col=LOAN_ID_FIELD if LOAN_ID_FIELD in df.columns else None)

    groups = _table_records(outcome["groups"], top_n)
    inv.telemetry.returned(len(groups))
    return {
        "resource": inv.authorised.resource.ref.key,
        "dimension": dimension,
        "top_n": top_n,
        "group_count": int(outcome["n_groups"]),
        # analytics_lib returns fractions; the published contract is percentage
        # points, matching every other percentage Trakt reports.
        "balance_concentration_pct": round(
            float(outcome["balance_concentration"]) * 100, 6),
        "count_concentration_pct": round(
            float(outcome["count_concentration"]) * 100, 6),
        "groups": groups,
        "population": _population_block(df, balance_column),
        "limit_evaluation": None,
        "scope": _scope_block(inv),
        "warnings": [
            "This is the measured share, not a limit test. For whether it "
            "breaches an operator-approved threshold, call evaluate_covenants."],
    }


# =========================================================================== #
# rank_loans
# =========================================================================== #
RANK_LOANS_INPUT = object_schema(
    description=("Find the N loans at the top or bottom of a metric WITHOUT "
                 "retrieving the population. Returns identifiers and the ranked "
                 "value; follow up with one get_loans call for the ones that "
                 "matter."),
    properties={
        "resource": _RESOURCE_PROPERTY,
        "metric": {"type": "string",
                   "description": ("The canonical field to rank by, e.g. "
                                   "'current_loan_to_value'.")},
        "direction": {"type": "string", "enum": ["highest", "lowest"],
                      "description": "Which end of the distribution.",
                      "default": "highest"},
        "limit": {"type": "integer", "minimum": 1, "maximum": MAX_RANK,
                  "description": f"How many loans to return (at most {MAX_RANK}).",
                  "default": 20},
        "filters": _FILTERS_PROPERTY,
        "include_fields": {
            "type": ["array", "null"], "items": {"type": "string"},
            "description": ("Extra canonical fields to carry on each ranked "
                            "loan. Keep it short — this is a shortlist, not a "
                            "retrieval."),
            "default": None,
        },
    },
    required=["resource", "metric"],
)

RANK_LOANS_OUTPUT = object_schema(
    description="A bounded, ordered shortlist.",
    properties={
        "resource": {"type": "string"},
        "metric": {"type": "string"},
        "direction": {"type": "string"},
        "loans": {"type": "array", "items": {"type": "object"}},
        "returned_count": {"type": "integer"},
        "population": {"type": "object"},
        "metric_coverage": {
            "type": "object",
            "description": ("How many loans carry the metric at all. Ranking a "
                            "field that is 60% null ranks the 40% that is "
                            "populated, and that has to be visible."),
        },
        "scope": {"type": "object"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    required=["resource", "metric", "direction", "loans", "returned_count",
              "population", "metric_coverage", "scope"],
)


def rank_loans(args: Dict[str, Any], inv: ToolInvocation) -> Dict[str, Any]:
    """The bridge from population to investigation.

    This is how an agent finds the twenty loans worth looking at without reading
    the other 499,980. It returns identifiers and the ranked metric — not rows —
    so the natural next step is one bounded ``get_loans`` call.
    """
    import pandas as pd

    df = resolve_governed_frame(inv)
    inv.telemetry.scanned(len(df))
    df = _apply_filters(df, args.get("filters"), inv)

    metric = _require_column(df, str(args.get("metric") or ""), inv, "metric")
    direction = str(args.get("direction") or "highest")
    limit = min(int(args.get("limit") or 20), MAX_RANK)

    values = pd.to_numeric(df[metric], errors="coerce")
    populated = int(values.notna().sum())
    warnings: List[str] = []
    coverage = {"loans_with_metric": populated,
                "loans_without_metric": int(len(df) - populated),
                "coverage_pct": (round(populated / len(df) * 100, 4)
                                 if len(df) else None)}
    if populated < len(df):
        # Ranking a 60%-populated field ranks the 40%, and an agent that cannot
        # see that will report "the highest LTV in the book" about a subset.
        warnings.append(
            f"{len(df) - populated} of {len(df)} loans carry no {metric!r}; the "
            "ranking covers only those that do.")

    extra = [f for f in (args.get("include_fields") or ()) if f in df.columns]
    columns = [c for c in ([LOAN_ID_FIELD, metric] + extra) if c in df.columns]
    ordered = (df.assign(_rank_value=values)
                 .dropna(subset=["_rank_value"])
                 .sort_values(by=["_rank_value", LOAN_ID_FIELD]
                              if LOAN_ID_FIELD in df.columns else ["_rank_value"],
                              ascending=[direction == "lowest", True]
                              if LOAN_ID_FIELD in df.columns
                              else [direction == "lowest"],
                              kind="mergesort")
                 .head(limit))

    loans = [{k: _jsonable(v) for k, v in record.items()}
             for record in ordered[columns].to_dict(orient="records")]
    inv.telemetry.returned(len(loans))

    return {
        "resource": inv.authorised.resource.ref.key,
        "metric": metric,
        "direction": direction,
        "loans": loans,
        "returned_count": len(loans),
        "population": _population_block(df, _balance_column(df)),
        "metric_coverage": coverage,
        "scope": _scope_block(inv),
        "warnings": warnings,
    }


# =========================================================================== #
# data_completeness
# =========================================================================== #
DATA_COMPLETENESS_INPUT = object_schema(
    description=("How populated the canonical fields are across a portfolio. The "
                 "first diligence question about any tape: what is actually "
                 "there."),
    properties={
        "resource": _RESOURCE_PROPERTY,
        "fields": {
            "type": ["array", "null"], "items": {"type": "string"},
            "description": ("Canonical fields to report on. Omit for every field "
                            "the tape carries."),
            "default": None,
        },
        "max_completeness_pct": {
            "type": ["number", "null"],
            "description": ("Report only fields populated at or below this "
                            "percentage — the gaps, rather than the whole tape."),
            "default": None,
        },
    },
    required=["resource"],
)

DATA_COMPLETENESS_OUTPUT = object_schema(
    description="Per-field population counts across the governed population.",
    properties={
        "resource": {"type": "string"},
        "fields": {"type": "array", "items": {"type": "object"}},
        "field_count": {"type": "integer"},
        "fully_populated_count": {"type": "integer"},
        "population": {"type": "object"},
        "scope": {"type": "object"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    required=["resource", "fields", "field_count", "population", "scope"],
)


def data_completeness(args: Dict[str, Any], inv: ToolInvocation) -> Dict[str, Any]:
    """Per-field population. One pass over the frame, not one pass per field."""
    df = resolve_governed_frame(inv)
    inv.telemetry.scanned(len(df))

    requested = [str(f) for f in (args.get("fields") or ())]
    columns = ([c for c in requested if c in df.columns] if requested
               else [str(c) for c in df.columns])
    warnings: List[str] = []
    if requested:
        missing = sorted(set(requested) - set(columns))
        if missing:
            warnings.append(f"This tape does not carry: {', '.join(missing)}.")

    total = int(len(df))
    # One vectorised count over the whole projection, rather than a loop of
    # per-column passes over the frame.
    populated = df[columns].notna().sum() if columns else None

    ceiling = args.get("max_completeness_pct")
    rows: List[Dict[str, Any]] = []
    fully = 0
    for column in columns:
        count = int(populated[column]) if populated is not None else 0
        pct = round(count / total * 100, 4) if total else None
        if count == total and total:
            fully += 1
        if ceiling is not None and pct is not None and pct > float(ceiling):
            continue
        rows.append({"canonical_field": column,
                     "populated": count,
                     "missing": total - count,
                     "completeness_pct": pct})
    rows.sort(key=lambda r: (r["completeness_pct"] if r["completeness_pct"]
                             is not None else -1.0, r["canonical_field"]))

    inv.telemetry.returned(len(rows))
    return {
        "resource": inv.authorised.resource.ref.key,
        "fields": rows,
        "field_count": len(rows),
        "fully_populated_count": fully,
        "population": _population_block(df, _balance_column(df)),
        "scope": _scope_block(inv),
        "warnings": warnings,
    }


# =========================================================================== #
# list_validation_exceptions
# =========================================================================== #
LIST_VALIDATION_EXCEPTIONS_INPUT = object_schema(
    description=("Which canonical fields failed validation in this snapshot, "
                 "with the rules that failed and how many errors each raised."),
    properties={
        "resource": _RESOURCE_PROPERTY,
        "status": {
            "type": ["array", "null"],
            "items": {"type": "string",
                      "enum": ["failed", "warning", "passed", "not_validated"]},
            "description": ("Which validation outcomes to return. Omit for "
                            "failures and warnings only."),
            "default": None,
        },
    },
    required=["resource"],
)

LIST_VALIDATION_EXCEPTIONS_OUTPUT = object_schema(
    description="Field-level validation outcomes for this snapshot.",
    properties={
        "resource": {"type": "string"},
        "exceptions": {"type": "array", "items": {"type": "object"}},
        "exception_count": {"type": "integer"},
        "total_error_count": {"type": "integer"},
        "lineage_available": {
            "type": "boolean",
            "description": ("False when this snapshot carries no lineage index. "
                            "An empty list then means 'unknown', NOT 'clean'."),
        },
        "grain": {
            "type": "string",
            "description": ("Always 'canonical_field'. Validation is a property "
                            "of a field within a snapshot, not of a cell."),
        },
        "scoped_to": {
            "type": "string",
            "description": ("Always 'snapshot'. The error counts cover the WHOLE "
                            "snapshot, not just the loans this resource narrows "
                            "to — the lineage index has no row dimension to "
                            "narrow by. Do not attribute these counts to this "
                            "resource's population."),
        },
        "snapshot": {"type": "object"},
        "scope": {"type": "object"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    required=["resource", "exceptions", "exception_count", "lineage_available",
              "grain", "snapshot", "scope"],
)

#: What counts as an exception when the caller does not say.
DEFAULT_EXCEPTION_STATUSES = ("failed", "warning")


def list_validation_exceptions(args: Dict[str, Any],
                               inv: ToolInvocation) -> Dict[str, Any]:
    """The validation findings for this snapshot, from the lineage index.

    Field-level, because that is the grain at which validation is recorded: a
    rule runs over a column of a snapshot and produces a status and an error
    count. There is no per-cell validation state, and inventing one here would
    mean re-running validation — a second implementation of the thing
    ``engine/gate_3_validation`` already owns.

    No index is reported as ``lineage_available: false`` and NOT as a clean
    tape. "I don't know" and "nothing wrong" are different answers, and a
    diligence agent must not read the first as the second.

    **Scope.** The index describes a SNAPSHOT's fields and has no row dimension,
    so the counts cannot be narrowed to a resource's population. Two consequences,
    both handled explicitly rather than left implicit: an SPV-scoped grant is
    refused, because handing whole-snapshot facts to a caller entitled to one SPV
    contradicts the grant; and a book-scoped caller is TOLD the counts are
    snapshot-wide, so it does not attribute them to its own book.
    """
    from .provenance import _index_for, _snapshot_block

    resolved = inv.authorised.resource
    if resolved.spv_id:
        raise TraktError(
            ErrorCode.RESOURCE_NOT_PARTITIONABLE,
            "This resource is defined by an SPV boundary. Validation outcomes are "
            "recorded per canonical field of a whole snapshot and carry no row "
            "dimension, so they cannot be narrowed to an SPV — returning them "
            "would report on the enclosing book.",
            request_id=inv.request_id, details={"resource": resolved.ref.key})

    index = _index_for(inv)
    inv.telemetry.cached("lineage_index", "hit" if index is not None else "miss")
    snapshot = _snapshot_block(inv)
    warnings: List[str] = []

    wanted = tuple(str(s) for s in (args.get("status")
                                    or DEFAULT_EXCEPTION_STATUSES))
    if index is None:
        warnings.append(
            "This snapshot carries no lineage index, so validation outcomes are "
            "unknown. An empty list here does NOT mean the tape is clean.")
        return {
            "resource": resolved.ref.key,
            "exceptions": [], "exception_count": 0, "total_error_count": 0,
            "lineage_available": False, "grain": "canonical_field",
            "scoped_to": "snapshot",
            "snapshot": snapshot, "scope": _scope_block(inv),
            "warnings": warnings,
        }

    exceptions: List[Dict[str, Any]] = []
    total_errors = 0
    for canonical_field, entry in sorted(index.entries.items()):
        status = getattr(entry, "validation_status", None) or "not_validated"
        if status not in wanted:
            continue
        errors = int(getattr(entry, "validation_error_count", 0) or 0)
        total_errors += errors
        exceptions.append({
            "canonical_field": canonical_field,
            "validation_status": status,
            "rules_applied": list(getattr(entry, "validation_rules", ()) or ()),
            "error_count": errors,
            "origin": getattr(entry, "origin", None),
            "source_field": getattr(entry, "source_field", None),
            "derivation_rule": getattr(entry, "derivation_rule", None),
        })
    # Worst first: most errors, then alphabetically for a stable order.
    exceptions.sort(key=lambda e: (-e["error_count"], e["canonical_field"]))

    if resolved.source_portfolio_ids and exceptions:
        warnings.append(
            "These error counts cover the WHOLE snapshot, not just this "
            "resource's book: validation is recorded per canonical field and "
            "carries no row dimension to narrow by.")

    inv.telemetry.returned(len(exceptions))
    return {
        "resource": resolved.ref.key,
        "exceptions": exceptions,
        "exception_count": len(exceptions),
        "total_error_count": total_errors,
        "lineage_available": True,
        "grain": "canonical_field",
        "scoped_to": "snapshot",
        "snapshot": snapshot,
        "scope": _scope_block(inv),
        "warnings": warnings,
    }
