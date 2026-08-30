"""mi_agent_pptx.composition — which slides this portfolio actually justifies.

The deck used to render a fixed sequence and substitute a branded placeholder
wherever a payload was missing, so a book with no pipeline produced pages of
"no data". An investor pack must instead contain only slides that answer a
question, and must say — once, in the appendix — what it left out and why.

Two independent gates decide inclusion, and a slide must pass both:

  1. ``when:``      a declarative condition in the deck config, evaluated against
                    governed FACTS about the portfolio (does an acquired book
                    exist, is there a forecast, is there >1 reporting period);
  2. ``will_render`` a data guard that inspects the actual payload the handler
                    would draw from, so a slide can never reach the deck only to
                    discover it has nothing to show.

Everything dropped becomes a :class:`SlideOmission` with a reason. Silence is
not an option: a reader must be able to tell "this book has no pipeline" from
"the pipeline section failed".

The ``when:`` evaluator is a restricted AST walk over a fixed fact namespace —
names, booleans, comparisons and boolean operators only. It is deliberately not
``eval``: a deck config must never be able to execute code.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

#: Reasons are investor-facing (they render in the appendix), so they read as
#: statements of fact about the book, not as internal error strings.
REASON_CONDITION = "condition"
REASON_NO_DATA = "no data"
#: A slide dropped because ANOTHER slide in this deck answers the same question
#: better for this book. This is not an absent capability, and the ledger must
#: not describe it as one: a pack that renders concentration headroom and then
#: states "no governed risk-limit artefact" contradicts itself on the page.
REASON_SUPERSEDED = "superseded"


@dataclass(frozen=True)
class SlideOmission:
    """One slide the deck deliberately does not contain, and why."""

    slide_id: str
    title: str
    reason: str
    category: str = REASON_CONDITION

    def to_dict(self) -> Dict[str, Any]:
        return {"slide_id": self.slide_id, "title": self.title,
                "reason": self.reason, "category": self.category}


# --------------------------------------------------------------------------- #
# Restricted expression evaluation.
# --------------------------------------------------------------------------- #

_ALLOWED_NODES = (
    ast.Expression, ast.BoolOp, ast.UnaryOp, ast.Compare, ast.Name, ast.Load,
    ast.And, ast.Or, ast.Not, ast.Constant,
    ast.Eq, ast.NotEq, ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.In, ast.NotIn,
    ast.Tuple, ast.List,
)


class ConditionError(ValueError):
    """A ``when:`` expression that is not safe or not understood."""


def evaluate_condition(expression: str, facts: Mapping[str, Any]) -> bool:
    """Evaluate a ``when:`` expression against the governed facts.

    Supports names, ``and`` / ``or`` / ``not``, comparisons and ``in``. Anything
    else — calls, attributes, subscripts, arithmetic — is rejected rather than
    executed, so a deck config can never become an execution vector.
    """
    text = (expression or "").strip()
    if not text:
        return True
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ConditionError(f"could not parse condition {text!r}: {exc}") from exc

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ConditionError(
                f"condition {text!r} uses an unsupported construct "
                f"({type(node).__name__})")
        if isinstance(node, ast.Name) and node.id not in facts:
            raise ConditionError(f"condition {text!r} refers to unknown fact "
                                 f"{node.id!r}")
    return bool(eval(compile(tree, "<when>", "eval"),  # noqa: S307 - AST-restricted
                     {"__builtins__": {}}, dict(facts)))


# --------------------------------------------------------------------------- #
# Facts.
# --------------------------------------------------------------------------- #

def _periods(payload: Any, minimum: int = 2) -> bool:
    if not isinstance(payload, Mapping):
        return False
    if payload.get("singlePeriod"):
        return False
    return len(payload.get("periods") or ()) >= minimum


def _cohort_progression_ready(data: Any) -> bool:
    """True when the governed static pool has something to season.

    Asks the cohort adapter rather than re-deriving sufficiency here, so the
    slide's guard and the slide's own emptiness check can never disagree.
    """
    payload = getattr(data, "cohort_series", {}) or {}
    if not payload.get("available"):
        return False
    try:
        from . import cohorts as _co
        return _co.progression_is_meaningful(
            [_co.adapt_progression(p, v)
             for v, p in (payload.get("series") or {}).items()])
    except Exception:  # noqa: BLE001 — a guard must never break composition
        return False


def build_facts(data: Any) -> Dict[str, Any]:
    """The governed facts a ``when:`` expression may refer to.

    Derived entirely from resolved payloads and the portfolio context — never
    from filenames, and never by recomputing a metric.
    """
    ctx = getattr(data, "portfolio", None)
    funded = getattr(data, "funded", {}) or {}
    forecast = getattr(data, "forecast", {}) or {}
    extrap = getattr(data, "extrapolation", {}) or {}
    risk = getattr(data, "risk", {}) or {}

    projection = bool(
        (extrap.get("completionRunRateForecast") or {}).get("available")
        or (extrap.get("kfiConversionForecast") or {}).get("available"))
    bridge = (forecast.get("forecastBridge") or {})

    facts: Dict[str, Any] = {
        # -- scope / composition -------------------------------------------
        "scope": (ctx.report_scope if ctx else "total"),
        "is_total": bool(ctx.is_total) if ctx else True,
        "portfolio_count": int(ctx.portfolio_count) if ctx else 0,
        "type_count": len(ctx.type_slices) if ctx else 0,
        "has_direct": bool(ctx and ctx.has_direct),
        "has_acquired": bool(ctx and ctx.has_acquired),
        "is_mixed": bool(ctx and ctx.is_mixed),
        "mixed_reporting_dates": bool(ctx and ctx.has_mixed_reporting_dates),
        # -- funded ---------------------------------------------------------
        "has_funded": bool(funded.get("kpis")),
        "has_stratifications": bool(funded.get("stratifications")),
        "has_movement": bool(ctx and any(s.has_movement for s in ctx.type_slices)),
        # Governed attribution across at least one dimension.
        "has_attribution": any(getattr(b, "available", False)
                               for b in (getattr(data, "movement", {}) or {}).values()),
        "has_funded_history": _periods(getattr(data, "funded_evolution", {})),
        "has_geo": bool((getattr(data, "geo", {}) or {}).get("areas")),
        "has_cohorts": bool((getattr(data, "cohorts", {}) or {}).get("cohorts")),
        # Seasoning exists only when a cohort holds loans in TWO OR MORE
        # reporting periods. One period is a formation snapshot; drawing a line
        # through it would claim a trend the data does not contain.
        "has_cohort_progression": _cohort_progression_ready(data),
        "has_multidim": bool(getattr(data, "multidim", {}) or {}),
        # The reconciled economic bridge (opening + new - exited + continuing).
        # Only true when the identity actually closed for this book: an
        # unreconciled bridge is not shown, it is omitted with its reason.
        "has_balance_movement": bool((getattr(data, "balance_movement", {}) or {}
                                      ).get("available")),
        # Exit reasons carried by evidence on the tape. Where this is false the
        # bridge still renders, with exits in one bar rather than split.
        "has_exit_reasons": bool((getattr(data, "balance_movement", {}) or {}
                                  ).get("exitsClassified")
                                 and (getattr(data, "balance_movement", {}) or {}
                                      ).get("exitsReconcile")),
        # A per-book forward view only means something when a book-level
        # projection actually resolved.
        "has_portfolio_projections": bool((getattr(data, "portfolio_projections", {}) or {}
                                           ).get("portfolios")),
        # -- pipeline --------------------------------------------------------
        "has_pipeline": bool(getattr(data, "pipeline", {}) or {}),
        "has_pipeline_history": _periods(getattr(data, "pipeline_evolution", {})),
        "has_funnel": bool((getattr(data, "funnel", {}) or {}).get("series")
                           or (getattr(data, "pipeline", {}) or {}).get("stageBreakdown")),
        # -- forecast / risk --------------------------------------------------
        # A forecast exists only when a pipeline actually contributes to it.
        "has_forecast": bool(bridge.get("weightedExpectedFundedAmount")),
        "has_forecast_projection": projection,
        "has_forecast_history": _periods(getattr(data, "forecast_evolution", {})),
        "has_risk": bool(risk.get("tests")),
        # -- concentration ----------------------------------------------------
        "has_concentration": bool((getattr(data, "concentration", {}) or {}).get("tests")),
        "has_concentration_forward": bool(
            ((getattr(data, "concentration", {}) or {}).get("states") or {}).get("available")),
    }

    # -- QUANTITATIVE facts -------------------------------------------------
    # The booleans above answer "does this exist?". A conditional pack also
    # needs "how much of it is there?", because the difference between a new
    # book and a seasoned one is not that one has no history — it is that one
    # has too little history for a trend to mean anything. These three are read
    # off already-resolved payloads; nothing is computed for them.
    funded_periods = len((getattr(data, "funded_evolution", {}) or {}).get("periods") or ())
    pipeline_periods = len((getattr(data, "pipeline_evolution", {}) or {}).get("periods") or ())
    forecast_periods = len((getattr(data, "forecast_evolution", {}) or {}).get("periods") or ())
    cohort_count = len((getattr(data, "cohorts", {}) or {}).get("cohorts") or ())
    constituent_books = len({
        str(row.get("key")) for row in
        (((getattr(data, "funded_evolution", {}) or {}).get("breakdowns") or {}
          ).get("portfolio") or ())
        if row.get("key") is not None})

    funded_balance = _kpi_raw(funded, "balance")
    pipeline_amount = _num((getattr(data, "pipeline", {}) or {}).get("pipelineAmount"))
    denominator = (funded_balance or 0.0) + (pipeline_amount or 0.0)

    facts.update({
        #: Reporting periods of funded history actually resolved.
        "funded_periods": funded_periods,
        #: Weekly pipeline extracts actually resolved.
        "pipeline_periods": pipeline_periods,
        #: Funded runs carrying a forecast. A forecast-vs-actual comparison needs
        #: THREE: two to produce a prior forecast and an actual to test it
        #: against, plus one more before the comparison is a track record rather
        #: than a single data point.
        "forecast_periods": forecast_periods,
        #: Origination vintages the governed cohort table found.
        "cohort_count": cohort_count,
        #: Distinct constituent books present in the governed period x book
        #: funded history. This is what separates "one book" from "a portfolio
        #: of books": a stack, a per-book forward view and a book-level driver
        #: sentence all need more than one, and none of them is worth a page
        #: when there is only one.
        "constituent_books": constituent_books,
        #: Funded balance, from the governed KPI (never recomputed).
        "funded_balance": float(funded_balance or 0.0),
        #: Pipeline balance, from the governed pipeline snapshot.
        "pipeline_amount": float(pipeline_amount or 0.0),
        #: Pipeline as a share of the book it would join. This is what makes a
        #: book "growing": not that a pipeline exists, but that it is large
        #: enough relative to the funded book for the origination story to be
        #: the story. A fraction 0-1.
        "pipeline_share": (round(float(pipeline_amount or 0.0) / denominator, 4)
                           if denominator else 0.0),
    })
    return facts


def _num(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _kpi_raw(funded: Mapping[str, Any], kpi_id: str) -> Optional[float]:
    """The raw value behind a governed KPI tile, or ``None``."""
    for kpi in (funded or {}).get("kpis") or ():
        if isinstance(kpi, Mapping) and kpi.get("id") == kpi_id:
            return _num(kpi.get("raw"))
    return None


# --------------------------------------------------------------------------- #
# Data guards — "would this handler actually draw something?"
# --------------------------------------------------------------------------- #

def _strat_guard(spec: Mapping[str, Any], data: Any) -> Optional[str]:
    strats = (getattr(data, "funded", {}) or {}).get("stratifications") or []
    keys = spec.get("keys")
    if keys:
        strats = [s for s in strats if s.get("key") in keys]
    if not any(s.get("bars") for s in strats):
        wanted = ", ".join(keys) if keys else "the requested dimensions"
        return f"the funded tape carries no {wanted} stratification"
    return None


def _geo_guard(spec: Mapping[str, Any], data: Any) -> Optional[str]:
    """The AREA-LEVEL map, and an honest reason when it is absent.

    Geography is two different things on this tape. The map needs area-level
    exposure (ITL3); the region stratification needs only a region field, and a
    book routinely has the second without the first. Saying "no geographic
    exposure resolved" while a regional bar list renders four pages earlier
    reads, correctly, as a contradiction — so the reason names WHICH geography
    is missing and, when the coarser cut did render, points at it.
    """
    if (getattr(data, "geo", {}) or {}).get("areas"):
        return None
    strats = (getattr(data, "funded", {}) or {}).get("stratifications") or []
    regional = any(st.get("bars") for st in strats
                   if str(st.get("key") or "") in ("region", "geographic_region_obligor"))
    if regional:
        return ("no area-level (ITL3) exposure on this tape; regional "
                "distribution is reported on the funded stratifications")
    return "no geographic exposure resolved for this book"


def _evolution_guard(attr: str, label: str, minimum: int = 2
                     ) -> Callable[[Mapping[str, Any], Any], Optional[str]]:
    def guard(spec: Mapping[str, Any], data: Any) -> Optional[str]:
        payload = getattr(data, attr, {}) or {}
        periods = len(payload.get("periods") or ())
        if not _periods(payload, minimum):
            return (f"{label} needs at least {minimum} reporting periods; "
                    f"{periods} available")
        return None
    return guard


#: ``slide type -> guard``. A guard returns ``None`` to include, or the investor
#: -facing reason the slide is being dropped. Types absent here always render
#: (cover, methodology, appendix, and the context-driven slides).
_GUARDS: Dict[str, Callable[[Mapping[str, Any], Any], Optional[str]]] = {
    "kpi_summary": lambda s, d: (None if (getattr(d, "funded", {}) or {}).get("kpis")
                                 else "no funded book resolved for this run"),
    "strat_barlists": _strat_guard,
    # At least one paired-dimension panel must actually have data — an "available"
    # multidim payload whose panels are all empty would render an empty slide.
    "multidim": lambda s, d: (
        None if any(((getattr(d, "multidim", {}) or {}).get(k) or {}).get(f)
                    for k, f in (("ltv_age", "points"),
                                 ("ltv_borrower_type", "matrix"),
                                 ("ltv_region", "matrix")))
        else "paired-dimension analysis is not available for this book"),
    "geo": _geo_guard,
    "funded_evolution": _evolution_guard("funded_evolution", "funded evolution"),
    "cohorts": lambda s, d: (None if (getattr(d, "cohorts", {}) or {}).get("cohorts")
                             else "no origination vintage data on the funded tape"),
    "cohort_progression": lambda s, d: (
        None if _cohort_progression_ready(d)
        else "static-pool seasoning needs a cohort with loans in at least two "
             "reporting periods"),
    "pipeline_summary": lambda s, d: (None if (getattr(d, "pipeline", {}) or {})
                                      else "no governed pipeline source for this book"),
    "pipeline_evolution": _evolution_guard("pipeline_evolution", "pipeline evolution"),
    "funnel": lambda s, d: (None if ((getattr(d, "funnel", {}) or {}).get("summary")
                                     or (getattr(d, "pipeline", {}) or {}).get("stageBreakdown"))
                            else "no governed pipeline source for this book"),
    "origination_flow": lambda s, d: (None if (getattr(d, "funnel", {}) or {}).get("series")
                                      else "weekly origination flow needs at least 2 pipeline extracts"),
    # A bridge with no weighted pipeline is Funded → Funded: it restates the
    # balance rather than bridging to anything, so it is not a forecast.
    "forecast_bridge": lambda s, d: (
        None if ((getattr(d, "forecast", {}) or {}).get("forecastBridge") or {}
                 ).get("weightedExpectedFundedAmount")
        else "a forecast bridge needs a pipeline contributing expected completions"),
    "forecast_projection": lambda s, d: (
        None if ((getattr(d, "extrapolation", {}) or {}).get("completionRunRateForecast") or {}
                 ).get("available")
        or ((getattr(d, "extrapolation", {}) or {}).get("kfiConversionForecast") or {}
            ).get("available")
        else "insufficient run-rate history for a scale-up projection"),
    "forecast_evolution": _evolution_guard("forecast_evolution", "forecast evolution"),
    "risk": lambda s, d: (None if (getattr(d, "risk", {}) or {}).get("tests")
                          else "no governed risk-limit artefact for this run"),
    "movement_drivers": lambda s, d: (
        None if (any(getattr(b, "available", False)
                     for b in (getattr(d, "movement", {}) or {}).values())
                 or (getattr(d, "portfolio", None) is not None
                     and any(sl.has_movement for sl in d.portfolio.type_slices)))
        else "no prior reporting period to attribute movement against"),
    # The watch list always renders: "no material items" is itself a finding,
    # and a reader must be able to tell it from a check that never ran.
    "watchlist": lambda s, d: (None if getattr(d, "watchlist", None) is not None
                               else "the watch-list evaluation did not run"),
    "concentration": lambda s, d: (
        None if (getattr(d, "concentration", {}) or {}).get("tests")
        else "no governed concentration tests are configured for this portfolio"),
    "portfolio_composition": lambda s, d: (
        None if getattr(d, "portfolio", None) is not None
        else "no governed portfolio context resolved"),
    "portfolio_comparison": lambda s, d: (
        None if (getattr(d, "portfolio", None) is not None
                 and len(d.portfolio.type_slices) > 1)
        else "only one portfolio type is in scope"),
    "balance_movement": lambda s, d: (
        None if (getattr(d, "balance_movement", {}) or {}).get("available")
        else str((getattr(d, "balance_movement", {}) or {}).get("reason")
                 or "the funded balance bridge did not reconcile for this period")),
    "funded_stock": _evolution_guard("funded_evolution", "funded stock over time"),
    "portfolio_projections": lambda s, d: (
        None if (getattr(d, "portfolio_projections", {}) or {}).get("portfolios")
        else "no constituent-book projection resolved for this scope"),
    "exec_insights": lambda s, d: (
        None if (getattr(d, "insights", {}) or {}).get("insights")
        else "no governed observations cleared the materiality thresholds"),
}


def will_render(spec: Mapping[str, Any], data: Any) -> Optional[str]:
    """``None`` when the slide has real content, else the reason it does not."""
    guard = _GUARDS.get(str(spec.get("type") or ""))
    if guard is None:
        return None
    try:
        return guard(spec, data)
    except Exception:  # noqa: BLE001 — a guard must never break composition
        return None


# --------------------------------------------------------------------------- #
# Selection.
# --------------------------------------------------------------------------- #

def select_slides(slides: Sequence[Mapping[str, Any]], data: Any,
                  facts: Optional[Mapping[str, Any]] = None
                  ) -> Tuple[List[Dict[str, Any]], List[SlideOmission]]:
    """Choose the slides this portfolio justifies, with reasons for the rest.

    A slide with neither a ``when:`` nor a guard is always included, so an
    existing config keeps its existing deck.
    """
    known = dict(facts if facts is not None else build_facts(data))
    kept: List[Dict[str, Any]] = []
    omitted: List[SlideOmission] = []

    for spec in slides:
        spec = dict(spec)
        sid = str(spec.get("id") or spec.get("type") or "")
        title = str(spec.get("title") or sid)

        condition = spec.get("when")
        if condition:
            try:
                included = evaluate_condition(str(condition), known)
            except ConditionError:
                # A malformed condition must not silently drop investor content.
                included = True
            if not included:
                omitted.append(_omission(spec, sid, title,
                                         _explain(str(condition), known),
                                         REASON_CONDITION, kept))
                continue

        reason = will_render(spec, data)
        if reason:
            omitted.append(_omission(spec, sid, title, reason,
                                     REASON_NO_DATA, kept))
            continue

        kept.append(spec)

    return kept, omitted


def _omission(spec: Mapping[str, Any], sid: str, title: str, reason: str,
              category: str, kept: Sequence[Mapping[str, Any]]) -> SlideOmission:
    """Record a dropped slide, preferring "covered elsewhere" to "unavailable".

    A slide config may name the slide that SUPERSEDES it. When that slide is in
    the deck, the honest reason this one is absent is that the reader already
    has the answer — not that the capability is missing. Saying the latter while
    the superseding slide renders two pages earlier is the contradiction this
    exists to prevent.
    """
    replacement = str(spec.get("superseded_by") or "")
    if replacement:
        by = next((k for k in kept if str(k.get("id")) == replacement), None)
        if by is not None:
            return SlideOmission(
                sid, title,
                f"covered by {by.get('title') or replacement}",
                REASON_SUPERSEDED)
    return SlideOmission(sid, title, reason, category)


#: Investor-facing wording for the conditions the config actually uses.
_CONDITION_WORDING: Dict[str, str] = {
    "has_acquired": "no acquired portfolio is in scope",
    "has_direct": "no direct origination book is in scope",
    "is_mixed": "only one portfolio type is in scope",
    "type_count > 1": "only one portfolio type is in scope",
    "portfolio_count > 1": "only one portfolio is in scope",
    "has_pipeline": "no governed pipeline source for this book",
    "has_forecast": "no forecast is available for this book",
    "has_risk": "no governed risk-limit artefact for this run",
    "has_funded_history": "fewer than two reporting periods are available",
    "has_movement": "no prior reporting period to compare against",
    "has_concentration": "no governed concentration tests are configured for this portfolio",
    "has_attribution": "no prior reporting period to attribute movement against",
    "has_pipeline_history": "fewer than two weekly pipeline extracts are available",
    "has_geo": "no area-level (ITL3) exposure resolved for this book",
    "has_balance_movement": "the funded balance bridge did not reconcile for this period",
    "has_portfolio_projections": "no constituent-book projection resolved for this scope",
    "constituent_books > 1": "only one constituent book is in scope",
}


def _explain(condition: str, facts: Mapping[str, Any]) -> str:
    """A readable reason for a condition that excluded a slide."""
    text = condition.strip()
    if text in _CONDITION_WORDING:
        return _CONDITION_WORDING[text]
    # Name the first governed fact in the expression that is falsy — that is the
    # one an investor would want explained.
    for name, wording in _CONDITION_WORDING.items():
        if " " not in name and name in text and not facts.get(name):
            return wording
    return f"the reporting condition '{text}' was not met"
