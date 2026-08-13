"""trakt_tools.handlers.history — observed behaviour, through governed tools.

Four tools, deliberately few. The brief's instruction was to prefer reusable
primitives over one tool per metric, and these are the four *shapes* of
historical question rather than the twenty metrics they can answer:

    portfolio_history    how has any measure moved across N periods?
    prepayment_analysis  what is the observed prepayment / redemption rate?
    loss_analysis        what losses and recoveries have actually occurred?
    cohort_comparison    how does this cohort compare with the rest of the book?

Transitions are NOT a fifth tool here: ``mi_agent.risk_monitor.migration`` already
computes them for the Risk Monitor, and ``transition_analysis`` below wraps that
existing implementation rather than restating it.

Every calculation is shared
---------------------------
``analytics_lib.history`` holds the three genuinely new calculations, beside
``stratify`` and ``concentration``, so React and MI Query can call them too.
Nothing in this module computes; it resolves the governed snapshots, chooses
measures from the existing metric library, and shapes the response.

Bounded, and one pass per snapshot
----------------------------------
A twelve-period, eight-measure request costs twelve frame walks, not
ninety-six: ``portfolio_series`` evaluates every measure in one pass per
snapshot. No tool here returns a historical tape, a per-loan series, or one row
per loan per period — the whole point is that a trend is cheap enough to ask for
routinely.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional

from trakt_core.errors import ErrorCode, TraktError

from ..schema import object_schema
from ..spec import ToolInvocation
from .analysis import _apply_filters, _balance_column, _population_block
from .loans import LOAN_ID_FIELD, _scope_block

_RESOURCE_PROPERTY = {
    "type": "string",
    "description": ("The portfolio resource, as '{tenant}/{kind}/{resource_id}'."),
    "pattern": r"^[^/]+/[^/]+/[^/]+$",
}

#: Ceiling on periods in one request. Above this the caller is asking for a data
#: extract rather than a trend, and should narrow the window.
MAX_PERIODS = 60

#: Measures the history tools can evaluate, each a deterministic function of one
#: snapshot. Named rather than free-form so an agent cannot ask for an arbitrary
#: expression — there is no formula language here, exactly as the concentration
#: test library refuses one.
BUILTIN_MEASURES = (
    "total_balance", "loan_count", "wa_ltv",
    "arrears_30_plus_pct", "arrears_60_plus_pct", "arrears_90_plus_pct",
    "defaulted_balance_pct", "high_ltv_share_pct",
    "largest_region_share_pct", "stale_valuation_pct",
)


def _measure_functions(names) -> Dict[str, Callable]:
    """Resolve measure names to deterministic functions over one snapshot.

    Each is the same arithmetic the current-state tools use — balance-weighted
    shares over the governed balance field — so a point on a trend line and the
    same figure read from ``portfolio_summary`` agree.
    """
    import pandas as pd

    def _balance(df):
        column = _balance_column(df)
        if column is None:
            return None
        return pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    def _share_where(df, mask) -> Optional[float]:
        balances = _balance(df)
        if balances is None:
            return None
        total = float(balances.sum())
        return round(float(balances[mask].sum()) / total * 100, 6) if total else None

    def _dpd(df, minimum: int) -> Optional[float]:
        """Arrears share, delegated to the GOVERNED library metric.

        Sprint 2.5C: this used to apply its own ``>= minimum`` mask while the
        concentration library applied ``> min_days``, so the same nominal "30+
        DPD" reported 50% here and 25% there on a book with loans at exactly 30
        days. Two definitions of one metric is the failure this whole sprint
        exists to remove, so the measure now CALLS the library rather than
        agreeing with it by hand — agreement by hand is what drifts.
        """
        from mi_agent.concentration_tests.library import load_library
        from mi_agent.concentration_tests.metrics import evaluate_metric

        library = load_library()
        metric = library.get("perf_arrears_share")
        if metric is None or not len(df):
            return None
        computation = evaluate_metric(df, library, metric,
                                      {"min_days": minimum})
        return computation.value

    catalogue: Dict[str, Callable] = {
        "total_balance": lambda df: (float(_balance(df).sum())
                                     if _balance(df) is not None else None),
        "loan_count": lambda df: float(len(df)),
        "wa_ltv": lambda df: _weighted(df, "current_loan_to_value"),
        "arrears_30_plus_pct": lambda df: _dpd(df, 30),
        "arrears_60_plus_pct": lambda df: _dpd(df, 60),
        "arrears_90_plus_pct": lambda df: _dpd(df, 90),
        "defaulted_balance_pct": lambda df: (
            _share_where(df, df["account_status"].astype(str).str.lower()
                         .str.contains("default", na=False))
            if "account_status" in df.columns else None),
        "high_ltv_share_pct": lambda df: (
            _share_where(df, pd.to_numeric(df["current_loan_to_value"],
                                           errors="coerce").fillna(-1) > 80)
            if "current_loan_to_value" in df.columns else None),
        "largest_region_share_pct": lambda df: _largest_region(df),
        "stale_valuation_pct": lambda df: _stale(df),
    }

    def _weighted(df, column: str) -> Optional[float]:
        if column not in df.columns:
            return None
        balances = _balance(df)
        if balances is None:
            return None
        values = pd.to_numeric(df[column], errors="coerce")
        paired = pd.DataFrame({"v": values, "w": balances}).dropna()
        weight = float(paired["w"].sum())
        return (round(float((paired["v"] * paired["w"]).sum()) / weight, 6)
                if weight else None)

    def _largest_region(df) -> Optional[float]:
        column = next((c for c in ("geographic_region_collateral",
                                   "geographic_region_obligor")
                       if c in df.columns), None)
        balances = _balance(df)
        if column is None or balances is None:
            return None
        total = float(balances.sum())
        if not total:
            return None
        grouped = balances.groupby(df[column]).sum()
        return round(float(grouped.max()) / total * 100, 6) if len(grouped) else None

    def _stale(df) -> Optional[float]:
        # Delegates to the shared profile so "stale" means one thing across
        # current-state and historical views.
        from analytics_lib.valuation_age import valuation_age_profile

        outcome = valuation_age_profile(df)
        return outcome.get("stale_balance_pct") if outcome.get("available") else None

    wanted = [str(n) for n in (names or BUILTIN_MEASURES)]
    unknown = [n for n in wanted if n not in catalogue]
    if unknown:
        raise TraktError(
            ErrorCode.TOOL_INPUT_INVALID,
            f"Unknown measure(s): {', '.join(sorted(unknown))}. There is no "
            "expression language here; ask for a named measure.",
            details={"available": sorted(catalogue)})
    return {name: catalogue[name] for name in wanted}


def _resolve_history(inv: ToolInvocation) -> Dict[str, Any]:
    """The governed snapshot history for this resource.

    Production discovers snapshots the same way the MI period-change workflow
    does, so a trend an agent reads and one the workspace shows come from one
    set of snapshots rather than two discoveries that could disagree.
    """
    resolver = getattr(inv.dependencies, "loan_history_resolver", None)
    if resolver is None:
        raise TraktError(
            ErrorCode.DATA_SOURCE_UNAVAILABLE,
            "No governed snapshot history is available for this deployment, so "
            "historical behaviour cannot be observed. This is an absence of "
            "history, not a finding of stability.",
            request_id=inv.request_id)
    snapshots = resolver()
    if not snapshots:
        raise TraktError(
            ErrorCode.DATA_SOURCE_UNAVAILABLE,
            "The governed snapshot history is empty.",
            request_id=inv.request_id)
    return snapshots


def _window(snapshots: Mapping[str, Any], args: Dict[str, Any],
            inv: ToolInvocation) -> List[str]:
    """The ordered periods this request covers, bounded."""
    order = sorted(snapshots)
    start, end = args.get("from_period"), args.get("to_period")
    if start:
        order = [p for p in order if p >= str(start)]
    if end:
        order = [p for p in order if p <= str(end)]
    months = args.get("months")
    if months:
        order = order[-int(months):]
    if len(order) > MAX_PERIODS:
        raise TraktError(
            ErrorCode.TOOL_INPUT_INVALID,
            f"{len(order)} periods exceed the {MAX_PERIODS} ceiling for one "
            "call. Narrow the window — a longer span is a data extract rather "
            "than a trend.", request_id=inv.request_id)
    if not order:
        raise TraktError(
            ErrorCode.TOOL_INPUT_INVALID,
            "No governed snapshot falls inside the requested window.",
            request_id=inv.request_id,
            details={"available_from": min(snapshots), "to": max(snapshots)})
    return order


# =========================================================================== #
# portfolio_history
# =========================================================================== #
HISTORY_INPUT = object_schema(
    description=("How named measures have moved across governed snapshots. One "
                 "call covers every measure and every period."),
    properties={
        "resource": _RESOURCE_PROPERTY,
        "measures": {
            "type": ["array", "null"], "items": {"type": "string"},
            "description": ("Named measures. Omit for the standard set. There is "
                            "no expression language — ask for a named measure."),
            "default": None,
        },
        "months": {"type": ["integer", "null"], "minimum": 1, "maximum": MAX_PERIODS,
                   "description": "Most recent N periods.", "default": None},
        "from_period": {"type": ["string", "null"], "default": None},
        "to_period": {"type": ["string", "null"], "default": None},
        "filters": {"type": ["object", "null"],
                    "description": "Applied to every snapshot before measuring.",
                    "default": None},
    },
    required=["resource"],
)

HISTORY_OUTPUT = object_schema(
    description="One series per measure, across the requested periods.",
    properties={
        "resource": {"type": "string"},
        "method": {"type": "string"},
        "periods": {"type": "array", "items": {"type": "string"}},
        "period_count": {"type": "integer"},
        "loan_count": {"type": "array", "items": {"type": "integer"}},
        "total_balance": {"type": "array", "items": {"type": "number"}},
        "series": {"type": "object"},
        "available_history": {"type": "object"},
        "scope": {"type": "object"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    required=["resource", "periods", "series", "scope"],
)


def portfolio_history(args: Dict[str, Any], inv: ToolInvocation) -> Dict[str, Any]:
    """Named measures across governed snapshots. One pass per snapshot."""
    from analytics_lib.history import portfolio_series

    snapshots = _resolve_history(inv)
    periods = _window(snapshots, args, inv)
    measures = _measure_functions(args.get("measures"))

    filters = args.get("filters")
    frames = {p: _apply_filters(snapshots[p], filters, inv) for p in periods}
    inv.telemetry.scanned(sum(len(f) for f in frames.values()))

    outcome = portfolio_series(frames, measures, periods=periods)
    inv.telemetry.returned(len(periods) * len(measures))
    inv.telemetry.count("snapshots_read", len(periods))

    return {
        "resource": inv.authorised.resource.ref.key,
        **outcome,
        "available_history": {"from": min(snapshots), "to": max(snapshots),
                              "snapshot_count": len(snapshots)},
        "scope": _scope_block(inv),
        "warnings": ([] if len(periods) > 1 else
                     ["Only one period is in the window, so no movement can be "
                      "observed. That is an absence of history, not stability."]),
    }


# =========================================================================== #
# prepayment_analysis
# =========================================================================== #
PREPAYMENT_INPUT = object_schema(
    description=("Observed prepayment and redemption, from the canonical "
                 "unscheduled-principal field. Never inferred from balance "
                 "movement."),
    properties={
        "resource": _RESOURCE_PROPERTY,
        "months": {"type": ["integer", "null"], "minimum": 2, "maximum": MAX_PERIODS,
                   "description": "Most recent N periods.", "default": None},
        "from_period": {"type": ["string", "null"], "default": None},
        "to_period": {"type": ["string", "null"], "default": None},
        "include_redemptions": {
            "type": "boolean",
            "description": "Count full redemptions alongside partial prepayment.",
            "default": True},
        "annualise": {"type": "boolean",
                      "description": "Also report the annualised (CPR) form.",
                      "default": True},
    },
    required=["resource"],
)

PREPAYMENT_OUTPUT = object_schema(
    description="The observed rate, its inputs, and its methodology.",
    properties={
        "resource": {"type": "string"},
        "method": {"type": "string"},
        "available": {"type": "boolean"},
        "reason": {"type": ["string", "null"]},
        "smm_pct": {"type": ["number", "null"]},
        "cpr_pct": {"type": ["number", "null"]},
        "periods_observed": {"type": "integer"},
        "unscheduled_principal": {"type": "number"},
        "redemptions": {"type": "number"},
        "per_period": {"type": "array", "items": {"type": "object"}},
        "notes": {"type": "array", "items": {"type": "string"}},
        "scope": {"type": "object"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    required=["resource", "method", "available", "scope"],
)


def prepayment_analysis(args: Dict[str, Any],
                        inv: ToolInvocation) -> Dict[str, Any]:
    """Observed prepayment. Wraps ``analytics_lib.history.prepayment_rate``."""
    from analytics_lib.history import prepayment_rate

    snapshots = _resolve_history(inv)
    periods = _window(snapshots, args, inv)
    inv.telemetry.scanned(sum(len(snapshots[p]) for p in periods))

    result = prepayment_rate(
        snapshots, periods=periods,
        include_redemptions=bool(args.get("include_redemptions", True)),
        annualise=bool(args.get("annualise", True)))

    inv.telemetry.returned(len(result.per_period))
    warnings: List[str] = []
    if not result.available:
        warnings.append(
            f"No prepayment rate could be measured: {result.reason} A rate is "
            "deliberately NOT inferred from the change in total balance.")
    return {
        "resource": inv.authorised.resource.ref.key,
        **result.to_dict(),
        "window": {"from": periods[0], "to": periods[-1]},
        "scope": _scope_block(inv),
        "warnings": warnings,
    }


# =========================================================================== #
# loss_analysis
# =========================================================================== #
LOSS_INPUT = object_schema(
    description=("Observed losses and recoveries, reported on every defensible "
                 "denominator rather than one chosen for you."),
    properties={
        "resource": _RESOURCE_PROPERTY,
        "months": {"type": ["integer", "null"], "minimum": 1, "maximum": MAX_PERIODS,
                   "default": None},
        "from_period": {"type": ["string", "null"], "default": None},
        "to_period": {"type": ["string", "null"], "default": None},
    },
    required=["resource"],
)

LOSS_OUTPUT = object_schema(
    description="Losses, recoveries and the rates they support.",
    properties={
        "resource": {"type": "string"},
        "available": {"type": "boolean"},
        "reason": {"type": ["string", "null"]},
        "method": {"type": ["string", "null"]},
        "window": {"type": "object"},
        "cumulative_losses": {"type": ["number", "null"]},
        "cumulative_recoveries": {"type": ["number", "null"]},
        "loss_rate_on_original_pct": {"type": ["number", "null"]},
        "loss_rate_on_opening_pct": {"type": ["number", "null"]},
        "loss_rate_on_current_pct": {"type": ["number", "null"]},
        "recovery_rate_on_defaulted_pct": {"type": ["number", "null"]},
        "recoveries_against_residual_loss_pct": {"type": ["number", "null"]},
        "loss_severity": {"type": "object"},
        "denominators": {"type": "object"},
        "notes": {"type": "array", "items": {"type": "string"}},
        "scope": {"type": "object"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    required=["resource", "available", "scope"],
)


DEFAULT_CURE_OUTPUT = object_schema(
    description="Observed default and cure rates across governed snapshots.",
    properties={
        "resource": {"type": "string"},
        "available": {"type": "boolean"},
        "reason": {"type": ["string", "null"]},
        "method": {"type": ["string", "null"]},
        "authority": {"type": ["string", "null"]},
        "window": {"type": "object"},
        "annualised_cdr_pct": {"type": ["number", "null"]},
        "mean_periodic_default_rate_pct": {"type": ["number", "null"]},
        "default_stock_pct": {"type": ["number", "null"]},
        "mean_cure_rate_pct": {"type": ["number", "null"]},
        "per_period": {"type": "array", "items": {"type": "object"}},
        "notes": {"type": "array", "items": {"type": "string"}},
        "scope": {"type": "object"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    required=["resource", "available", "scope"],
)


def default_analysis(args: Dict[str, Any], inv: ToolInvocation) -> Dict[str, Any]:
    """Observed default rate. Wraps ``analytics_lib.history.default_rate``."""
    from analytics_lib.history import default_rate

    snapshots = _resolve_history(inv)
    periods = _window(snapshots, args, inv)
    inv.telemetry.scanned(len(snapshots[periods[-1]]))
    outcome = default_rate(snapshots, periods=periods)
    inv.telemetry.returned(len(outcome.get("per_period") or ()))
    return {
        "resource": inv.authorised.resource.ref.key,
        **outcome,
        "reason": outcome.get("reason"),
        "scope": _scope_block(inv),
        "warnings": [
            "OBSERVED default, not a probability of default. This is what "
            "happened, not what is expected to happen.",
            "The default RATE is a flow and the default STOCK is a level. A "
            "book can carry a large stock and have a zero rate. Quote the one "
            "you mean.",
        ],
    }


def cure_analysis(args: Dict[str, Any], inv: ToolInvocation) -> Dict[str, Any]:
    """Observed cure rate. Wraps ``analytics_lib.history.cure_rate``."""
    from analytics_lib.history import cure_rate

    snapshots = _resolve_history(inv)
    periods = _window(snapshots, args, inv)
    inv.telemetry.scanned(len(snapshots[periods[-1]]))
    outcome = cure_rate(snapshots, periods=periods,
                        minimum_dpd=int(args.get("minimum_dpd") or 1))
    inv.telemetry.returned(len(outcome.get("per_period") or ()))
    return {
        "resource": inv.authorised.resource.ref.key,
        **outcome,
        "reason": outcome.get("reason"),
        "scope": _scope_block(inv),
        "warnings": [
            "A cure is a return to CURRENT. Improvement to a better but still "
            "delinquent bucket is reported as improved_not_cured_balance and "
            "is NOT in the numerator — that movement is a roll, which "
            "transition_analysis measures.",
            "Unlike the default rate, this convention is Trakt's rather than "
            "a regulator's: ESMA defines no loan-level cure concept.",
        ],
    }


def loss_analysis(args: Dict[str, Any], inv: ToolInvocation) -> Dict[str, Any]:
    """Observed loss and recovery. Wraps ``analytics_lib.history``."""
    from analytics_lib.history import loss_and_recovery

    snapshots = _resolve_history(inv)
    periods = _window(snapshots, args, inv)
    inv.telemetry.scanned(len(snapshots[periods[-1]]))

    outcome = loss_and_recovery(snapshots, periods=periods)
    inv.telemetry.returned(1)
    warnings: List[str] = []
    if not outcome.get("available"):
        warnings.append(f"Losses could not be measured: {outcome.get('reason')}")
    else:
        warnings.append(
            "Several loss denominators are reported because each answers a "
            "different question. Quote the one you mean, and say which.")
    return {
        "resource": inv.authorised.resource.ref.key,
        **outcome,
        "reason": outcome.get("reason"),
        "scope": _scope_block(inv),
        "warnings": warnings,
    }


# =========================================================================== #
# cohort_comparison
# =========================================================================== #
COHORT_INPUT = object_schema(
    description=("Compare a cohort against the REST of the portfolio on the "
                 "same measures. Optionally across periods."),
    properties={
        "resource": _RESOURCE_PROPERTY,
        "cohort_field": {
            "type": "string",
            "description": ("The canonical field defining the cohort, e.g. "
                            "'portfolio_cohort' for vintage or "
                            "'geographic_region_collateral' for region."),
        },
        "cohort_value": {
            "type": ["string", "null"],
            "description": ("The value identifying the cohort. Omit with "
                            "'above'/'below' to define a numeric cohort."),
            "default": None,
        },
        "above": {"type": ["number", "null"],
                  "description": "Numeric cohort: values strictly above this.",
                  "default": None},
        "below": {"type": ["number", "null"],
                  "description": "Numeric cohort: values strictly below this.",
                  "default": None},
        "measures": {"type": ["array", "null"], "items": {"type": "string"},
                     "default": None},
        "period": {"type": ["string", "null"],
                   "description": "Which snapshot. Omit for the most recent.",
                   "default": None},
    },
    required=["resource", "cohort_field"],
)

COHORT_OUTPUT = object_schema(
    description="The cohort and the remainder, on the same measures.",
    properties={
        "resource": {"type": "string"},
        "period": {"type": "string"},
        "cohort_label": {"type": "string"},
        "cohort": {"type": "object"},
        "remainder": {"type": "object"},
        "measures": {"type": "object"},
        "note": {"type": "string"},
        "scope": {"type": "object"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    required=["resource", "period", "cohort", "remainder", "measures", "scope"],
)


def cohort_comparison(args: Dict[str, Any],
                      inv: ToolInvocation) -> Dict[str, Any]:
    """A cohort against the rest of the book.

    The comparison is the finding. "The high-LTV cohort has 4% arrears" is a
    number; "4.0% against 1.2% for the rest" is something a reviewer can act on.
    """
    import pandas as pd

    from analytics_lib.history import compare_cohorts

    snapshots = _resolve_history(inv)
    period = str(args.get("period") or max(snapshots))
    frame = snapshots.get(period)
    if frame is None:
        raise TraktError(ErrorCode.TOOL_INPUT_INVALID,
                         f"No governed snapshot for period {period!r}.",
                         request_id=inv.request_id,
                         details={"available": sorted(snapshots)[-12:]})

    field = str(args.get("cohort_field") or "")
    if field not in frame.columns:
        raise TraktError(
            ErrorCode.TOOL_INPUT_INVALID,
            f"This tape does not carry {field!r}, so that cohort cannot be "
            "defined.", request_id=inv.request_id)

    value, above, below = args.get("cohort_value"), args.get("above"), args.get("below")
    if value is not None:
        mask = frame[field].astype(str) == str(value)
        label = f"{field} = {value}"
    elif above is not None:
        mask = pd.to_numeric(frame[field], errors="coerce").fillna(-1e18) > float(above)
        label = f"{field} > {above}"
    elif below is not None:
        mask = pd.to_numeric(frame[field], errors="coerce").fillna(1e18) < float(below)
        label = f"{field} < {below}"
    else:
        raise TraktError(
            ErrorCode.TOOL_INPUT_INVALID,
            "Define the cohort with cohort_value, above, or below.",
            request_id=inv.request_id)

    inv.telemetry.scanned(len(frame))
    measures = _measure_functions(args.get("measures"))
    outcome = compare_cohorts(frame, mask, measures, cohort_label=label,
                              remainder_label="rest of portfolio")
    inv.telemetry.returned(len(measures) * 2)

    warnings: List[str] = []
    if outcome["cohort"]["loan_count"] == 0:
        warnings.append(
            f"No loan matches {label}, so the comparison is empty. That is not "
            "evidence the cohort performs well.")
    return {
        "resource": inv.authorised.resource.ref.key,
        "period": period,
        **outcome,
        "scope": _scope_block(inv),
        "warnings": warnings,
    }


# =========================================================================== #
# transition_analysis — wraps the EXISTING risk-monitor implementation
# =========================================================================== #
TRANSITION_INPUT = object_schema(
    description=("How loans moved between states across two governed snapshots: "
                 "DPD deterioration, cures, entries and exits."),
    properties={
        "resource": _RESOURCE_PROPERTY,
        "dimension": {
            "type": "string",
            "description": ("The state to track, e.g. 'account_status' or "
                            "'ifrs9_stage'."),
            "default": "account_status",
        },
        "from_period": {"type": ["string", "null"],
                        "description": "Baseline snapshot. Omit for the earliest.",
                        "default": None},
        "to_period": {"type": ["string", "null"],
                      "description": "Current snapshot. Omit for the latest.",
                      "default": None},
    },
    required=["resource"],
)

TRANSITION_OUTPUT = object_schema(
    description="The aggregate transition matrix between two snapshots.",
    properties={
        "resource": {"type": "string"},
        "dimension": {"type": "string"},
        "from_period": {"type": "string"},
        "to_period": {"type": "string"},
        "transitions": {"type": "array", "items": {"type": "object"}},
        "issues": {"type": "array", "items": {"type": "object"}},
        "calculation_source": {"type": "string"},
        "scope": {"type": "object"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    required=["resource", "dimension", "transitions", "scope"],
)


def transition_analysis(args: Dict[str, Any],
                        inv: ToolInvocation) -> Dict[str, Any]:
    """State transitions between two snapshots.

    Wraps ``mi_agent.risk_monitor.migration.migration_matrix``, which the Risk
    Monitor already uses. Nothing is reimplemented: a transition an agent reads
    and one the workspace shows are the same computation.
    """
    from mi_agent.risk_monitor.migration import migration_matrix

    snapshots = _resolve_history(inv)
    order = sorted(snapshots)
    baseline_period = str(args.get("from_period") or order[0])
    current_period = str(args.get("to_period") or order[-1])
    for period in (baseline_period, current_period):
        if period not in snapshots:
            raise TraktError(ErrorCode.TOOL_INPUT_INVALID,
                             f"No governed snapshot for period {period!r}.",
                             request_id=inv.request_id)

    baseline, current = snapshots[baseline_period], snapshots[current_period]
    inv.telemetry.scanned(len(baseline) + len(current))

    dimension = str(args.get("dimension") or "account_status")
    # The risk monitor defaults to `loan_id`; the canonical identifier is
    # `loan_identifier`, so it is named explicitly rather than relying on a
    # default that silently yields an empty matrix.
    result = migration_matrix(baseline, current, dimension,
                              key=LOAN_ID_FIELD)

    frame = getattr(result, "frame", None)
    transitions = ([{k: (None if isinstance(v, float) and v != v else v)
                     for k, v in row.items()}
                    for row in frame.to_dict(orient="records")]
                   if frame is not None and len(frame) else [])
    inv.telemetry.returned(len(transitions))

    return {
        "resource": inv.authorised.resource.ref.key,
        "dimension": dimension,
        "from_period": baseline_period,
        "to_period": current_period,
        "transitions": transitions,
        "issues": list(getattr(result, "issues", ()) or ()),
        "calculation_source": "mi_agent.risk_monitor.migration.migration_matrix",
        "scope": _scope_block(inv),
        "warnings": [
            "Transitions need a stable loan identifier in both snapshots. A "
            "loan absent from one is reported as an entry or an exit, not as a "
            "state change."],
    }
