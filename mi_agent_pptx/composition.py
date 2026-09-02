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
        # -- pipeline --------------------------------------------------------
        "has_pipeline": bool(getattr(data, "pipeline", {}) or {}),
        "has_pipeline_history": _periods(getattr(data, "pipeline_evolution", {})),
        # The engine's OWN typed availability decides this. The deck never
        # re-derives whether a transition answer exists — a second opinion here
        # could contradict the payload it is about to render.
        "has_stage_transitions": bool(
            (getattr(data, "stage_transitions", {}) or {}).get("available")),
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
    return facts


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
    "geo": lambda s, d: (None if (getattr(d, "geo", {}) or {}).get("areas")
                         else "no geographic exposure resolved for this book"),
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
    # The engine's typed unavailability IS the omission reason — a duplicate
    # identifier or a single snapshot is disclosed in the reader's words rather
    # than being restated as a generic "no data".
    "stage_transitions": lambda s, d: (
        None if (getattr(d, "stage_transitions", {}) or {}).get("available")
        else ((getattr(d, "stage_transitions", {}) or {}).get("reason")
              or "no governed pair of weekly pipeline extracts to compare")),
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
                omitted.append(SlideOmission(sid, title, _explain(str(condition), known),
                                             REASON_CONDITION))
                continue

        reason = will_render(spec, data)
        if reason:
            omitted.append(SlideOmission(sid, title, reason, REASON_NO_DATA))
            continue

        kept.append(spec)

    return kept, omitted


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
