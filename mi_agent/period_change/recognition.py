"""mi_agent.period_change.recognition — is this a period-change question?

§4. Recognition is deterministic and reads the EXISTING structured parse: the
:class:`~mi_agent.mi_query_spec.MIQuerySpec` the single parse already produced
(``temporal_mode``, ``metric``, ``dimension``, ``compare_periods``,
``start_date``/``end_date``). There is no second parser here, and nothing in this
module produces a spec.

Two responsibilities:

* **match or decline.** Matching on the literal words "period change" would miss
  almost every real question, so a controlled vocabulary of change language is
  used instead. Just as important, the module DECLINES questions that belong to
  another capability — a forecast, a trend series, a static cross-portfolio
  comparison, a single-date summary, a raw tape request, a transaction
  reconciliation, and the single-metric named-month comparison that the existing
  ``temporal_compare`` route owns;
* **interpret.** It derives the workflow mode (requested-metric / concept /
  portfolio-overview) and the period request, both from the structured parse plus
  a controlled vocabulary drawn from the registry's own concept taxonomy.

Period-token extraction follows the pattern the existing ``temporal_compare``
module already uses (month names and ISO dates in a bounded scan), and only
where the structured spec did not already carry the periods.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..business_semantics import (
    WORKFLOW_PERIOD_CHANGE,
    BusinessSemanticsRegistry,
)
from .models import (
    METHOD_CURRENT_VS_PREVIOUS,
    METHOD_MONTH_ON_MONTH,
    METHOD_QUARTER_ON_QUARTER,
    METHOD_YEAR_ON_YEAR,
    METHOD_YEAR_TO_DATE,
    MODE_CONCEPT,
    MODE_PORTFOLIO_OVERVIEW,
    MODE_REQUESTED_METRIC,
)
from .periods import PeriodRequest

# --------------------------------------------------------------------------- #
# Controlled vocabulary
# --------------------------------------------------------------------------- #
#: Explicit change language. A question carrying any of these is asking what
#: moved, whatever nouns surround it.
CHANGE_MARKERS: Tuple[str, ...] = (
    "changed", "change in", "changes in", "what changed", "has changed",
    "have changed", "movement", "movements", "moved", "what moved",
    "increase", "increased", "decrease", "decreased",
    "improved", "deteriorated", "improvement", "deterioration",
    "grown", "shrunk", "rose", "fell", "risen", "fallen",
    "drivers of", "driver of", "what drove", "main drivers",
    "portfolio evolution", "evolution of the portfolio",
    "composition change", "composition changed", "shifted", "shift in",
    # The same verb family, in the tenses a ranked question actually uses.
    # "grown", "rose" and "fell" were already here; "grew", "growth",
    # "declined" and "dropped" are the missing conjugations of the identical
    # intent, and without them "which region grew the most last month?" was not
    # recognised as change language at all.
    "grew", "grow", "growth", "shrank",
    "decline", "declined", "declining", "dropped", "drop in",
    "rising", "falling", "gained", "lost balance",
)

#: Period tokens that are themselves change language: nobody writes
#: "quarter-on-quarter" about a single date. "prior period" alone is NOT here —
#: "show me the prior month balance" is an as-at question, not a change question.
COMPARISON_PERIOD_MARKERS: Tuple[str, ...] = (
    "month on month", "month-on-month", " mom ", "quarter on quarter",
    "quarter-on-quarter", " qoq ", "year on year", "year-on-year", " yoy ",
    "year to date", "year-to-date", " ytd ",
    "current versus previous", "current vs previous",
    "versus the previous", "versus the prior", "vs the previous", "vs the prior",
    "with the previous", "with the prior",
    "since the previous", "since the prior", "since the last",
)

#: Relative period modes, longest/most specific first so "year-on-year" is not
#: shadowed by a looser marker.
RELATIVE_MODE_MARKERS: Tuple[Tuple[str, str], ...] = (
    ("month on month", METHOD_MONTH_ON_MONTH),
    ("month-on-month", METHOD_MONTH_ON_MONTH),
    (" mom ", METHOD_MONTH_ON_MONTH),
    ("quarter on quarter", METHOD_QUARTER_ON_QUARTER),
    ("quarter-on-quarter", METHOD_QUARTER_ON_QUARTER),
    (" qoq ", METHOD_QUARTER_ON_QUARTER),
    ("year on year", METHOD_YEAR_ON_YEAR),
    ("year-on-year", METHOD_YEAR_ON_YEAR),
    (" yoy ", METHOD_YEAR_ON_YEAR),
    ("year to date", METHOD_YEAR_TO_DATE),
    ("year-to-date", METHOD_YEAR_TO_DATE),
    (" ytd ", METHOD_YEAR_TO_DATE),
    ("this year so far", METHOD_YEAR_TO_DATE),
    ("previous reporting date", METHOD_CURRENT_VS_PREVIOUS),
    ("prior reporting date", METHOD_CURRENT_VS_PREVIOUS),
    ("previous period", METHOD_CURRENT_VS_PREVIOUS),
    ("prior period", METHOD_CURRENT_VS_PREVIOUS),
    ("previous snapshot", METHOD_CURRENT_VS_PREVIOUS),
    ("prior snapshot", METHOD_CURRENT_VS_PREVIOUS),
    ("previous month", METHOD_MONTH_ON_MONTH),
    ("prior month", METHOD_MONTH_ON_MONTH),
    ("last month", METHOD_MONTH_ON_MONTH),
    ("this month", METHOD_CURRENT_VS_PREVIOUS),
    ("this quarter", METHOD_QUARTER_ON_QUARTER),
)

# -- decline vocabularies ---------------------------------------------------- #
FORECAST_MARKERS: Tuple[str, ...] = (
    "forecast", "forecasting", "predict", "prediction", "projected",
    "projection", "next month", "next quarter", "next year", "outlook",
    "expected to reach", "will reach", "run rate", "run-rate", "scenario",
    "what if", "extrapolat",
)

TREND_MARKERS: Tuple[str, ...] = (
    "over time", "by month", "by week", "by quarter", "monthly", "weekly",
    "trend", "time series", "each month", "every month", "month by month",
    "evolution by", "series",
)

RAW_RECORD_MARKERS: Tuple[str, ...] = (
    "loan level", "loan-level", "line by line", "record level", "record-level",
    "raw tape", "the tape", "loan tape", "list of loans", "list the loans",
    "export the loans", "individual loans", "each loan", "per loan",
    "underlying exposures list",
)

RECONCILIATION_MARKERS: Tuple[str, ...] = (
    "transaction", "transactions", "ledger", "payment file", "remittance",
    "cash ledger", "reconcile the cash", "cash reconciliation",
)

CROSS_PORTFOLIO_MARKERS: Tuple[str, ...] = (
    "across portfolios", "between portfolios", "by portfolio", "per portfolio",
    "which portfolio", "portfolio a", "portfolio b", "compare portfolios",
    "compare the portfolios", "against the other portfolio",
)

#: A metric focus phrase that the ``temporal_compare`` route already answers
#: as a single-metric comparison when explicit calendar periods are named.
COUNT_FOCUS_MARKERS: Tuple[str, ...] = (
    "count", "number of loans", "how many loans",
)

#: Language that makes a question a portfolio-wide overview even when the parse
#: happened to resolve a single metric.
OVERVIEW_MARKERS: Tuple[str, ...] = (
    "the portfolio", "the book", "portfolio composition", "composition",
    "overall", "across the board", "risk indicators", "key metrics",
    "what changed", "what has changed", "what moved", "what improved",
    "improved and deteriorated", "everything",
)

#: Language that asks for the balance decomposition specifically (§14).
BRIDGE_MARKERS: Tuple[str, ...] = (
    "drivers of", "driver of", "what drove", "main drivers", "decomposition",
    "bridge", "breakdown of the movement", "attribution",
)

#: Language that asks for composition/distribution specifically (§13).
COMPOSITION_MARKERS: Tuple[str, ...] = (
    "composition", "mix", "product mix", "distribution", "make-up", "makeup",
    "concentration",
)

#: Natural phrasings → the registry's own ``analytical_concept`` taxonomy values.
#: A controlled vocabulary over governed taxonomy terms, not a field parser: it
#: never names a canonical field, only a concept the registry already defines.
CONCEPT_TERMS: Tuple[Tuple[str, str], ...] = (
    ("credit quality", "credit_quality"),
    ("credit risk", "credit_quality"),
    ("asset quality", "credit_quality"),
    ("arrears", "payment_performance"),
    ("delinquenc", "payment_performance"),
    ("payment performance", "payment_performance"),
    ("payment behaviour", "payment_performance"),
    ("collections", "cashflow"),
    ("cashflow", "cashflow"),
    ("cash flow", "cashflow"),
    ("exposure", "exposure"),
    ("balances", "exposure"),
    ("leverage", "leverage"),
    ("gearing", "leverage"),
    ("ltv", "leverage"),
    ("loan to value", "leverage"),
    ("collateral", "collateral"),
    ("valuation", "valuation"),
    ("property value", "valuation"),
    ("pricing", "pricing"),
    ("interest rate", "pricing"),
    ("margin", "pricing"),
    ("maturity", "maturity"),
    ("term", "maturity"),
    ("losses", "loss"),
    ("loss", "loss"),
    ("recoveries", "loss"),
    ("coverage", "coverage"),
    ("liquidity", "liquidity"),
    ("product mix", "product_mix"),
    ("geography", "geography"),
    ("geographic", "geography"),
    ("region", "geography"),
    ("origination", "origination"),
    ("new lending", "origination"),
)

_MONTH_WORDS = (
    "january|february|march|april|may|june|july|august|september|october|"
    "november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec")
#: A period token: an ISO date, an ISO month, or a month name with optional
#: day/year. Mirrors the bounded scan ``temporal_compare`` already performs.
_PERIOD_TOKEN = (
    r"(\d{4}-\d{2}-\d{2}|\d{4}-\d{2}|"
    r"(?:\d{1,2}(?:st|nd|rd|th)?\s+)?(?:" + _MONTH_WORDS + r")(?:\s+\d{4})?)")
_BETWEEN_RE = re.compile(
    rf"\bbetween\s+{_PERIOD_TOKEN}\s+and\s+{_PERIOD_TOKEN}", re.IGNORECASE)
_FROM_TO_RE = re.compile(
    rf"\bfrom\s+{_PERIOD_TOKEN}\s+(?:to|until|through)\s+{_PERIOD_TOKEN}",
    re.IGNORECASE)
_VERSUS_RE = re.compile(
    rf"{_PERIOD_TOKEN}\s+(?:versus|vs\.?|against|compared (?:to|with)|with)\s+"
    rf"{_PERIOD_TOKEN}", re.IGNORECASE)
_SINCE_RE = re.compile(rf"\bsince\s+{_PERIOD_TOKEN}", re.IGNORECASE)
_CALENDAR_TOKEN_RE = re.compile(_PERIOD_TOKEN, re.IGNORECASE)

# Decline reason codes — stable, and reported so a routing decision is auditable.
DECLINE_NO_CHANGE_LANGUAGE = "no_change_language"
DECLINE_FORECAST = "forecast_question"
DECLINE_TREND_SERIES = "trend_series_question"
DECLINE_RAW_RECORDS = "raw_record_request"
DECLINE_TRANSACTION_RECONCILIATION = "transaction_reconciliation_request"
DECLINE_CROSS_PORTFOLIO = "static_cross_portfolio_comparison"
DECLINE_NOT_FUNDED_VIEW = "not_the_funded_book_view"
DECLINE_INCUMBENT_SINGLE_METRIC_COMPARE = "single_metric_named_period_comparison"


@dataclass(frozen=True)
class PeriodChangeIntent:
    """The recogniser's verdict and, when matched, its interpretation."""

    matched: bool
    reason: str = ""
    mode: str = MODE_PORTFOLIO_OVERVIEW
    period_request: PeriodRequest = dc_field(default_factory=PeriodRequest)
    requested_fields: Tuple[str, ...] = ()
    requested_concepts: Tuple[str, ...] = ()
    include_bridge: bool = False
    composition_focus: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matched": self.matched,
            "reason": self.reason,
            "mode": self.mode,
            "period_request": self.period_request.to_dict(),
            "requested_fields": list(self.requested_fields),
            "requested_concepts": list(self.requested_concepts),
            "include_balance_bridge": self.include_bridge,
            "composition_focus": self.composition_focus,
        }

    @classmethod
    def no(cls, reason: str) -> "PeriodChangeIntent":
        return cls(matched=False, reason=reason)


def _padded(question: str) -> str:
    return f" {str(question or '').strip().lower()} "


def _any(text: str, markers: Sequence[str]) -> bool:
    return any(m in text for m in markers)


def has_change_language(question: str) -> bool:
    """True when the question is asking what moved, not what the position is.

    Three ways a question qualifies: explicit change words, a period token that
    only ever appears in a change question (``quarter-on-quarter``), or an
    explicit two-period construction ("between March and June", "compare
    2025-12-31 with 2026-03-31"). The last one matters because naming two dates
    IS the question — no verb is needed — but it is deliberately the weakest
    signal, and a two-period question with a single metric focus is deferred to
    the incumbent ``temporal_compare`` route below.
    """
    text = _padded(question)
    if _any(text, CHANGE_MARKERS) or _any(text, COMPARISON_PERIOD_MARKERS):
        return True
    return any(pattern.search(question or "")
               for pattern in (_BETWEEN_RE, _FROM_TO_RE, _VERSUS_RE))


def _explicit_periods(question: str, spec: Any) -> Tuple[Optional[str], Optional[str]]:
    """``(start_token, end_token)`` from the structured parse, then the question.

    The structured spec is consulted FIRST — ``compare_periods``,
    ``start_date``/``end_date`` and ``baseline_date``/``current_date`` are period
    facts the single parse already established. The bounded question scan runs
    only when the spec carried nothing, and recognises the same shapes the
    existing temporal-compare route does.
    """
    periods = [p for p in (getattr(spec, "compare_periods", None) or []) if p]
    explicit = [p for p in periods if _CALENDAR_TOKEN_RE.fullmatch(str(p).strip())]
    if len(explicit) >= 2:
        return str(explicit[0]), str(explicit[1])

    start = (getattr(spec, "start_date", None)
             or getattr(spec, "baseline_date", None))
    end = getattr(spec, "end_date", None) or getattr(spec, "current_date", None)
    if start or end:
        return (str(start) if start else None), (str(end) if end else None)

    for pattern in (_BETWEEN_RE, _FROM_TO_RE, _VERSUS_RE):
        match = pattern.search(question or "")
        if match:
            return match.group(1), match.group(2)
    match = _SINCE_RE.search(question or "")
    if match:
        return match.group(1), None
    return None, None


def _relative_mode(question: str) -> Optional[str]:
    text = _padded(question)
    for marker, mode in RELATIVE_MODE_MARKERS:
        if marker in text:
            return mode
    return None


def _requested_concepts(question: str,
                        registry: Optional[BusinessSemanticsRegistry]
                        ) -> Tuple[str, ...]:
    text = _padded(question)
    known = set(registry.concepts(tag=WORKFLOW_PERIOD_CHANGE)) if registry else None
    found: List[str] = []
    for phrase, concept in CONCEPT_TERMS:
        if phrase in text and concept not in found:
            if known is None or concept in known:
                found.append(concept)
    return tuple(found)


def _is_incumbent_single_metric_compare(question: str, spec: Any) -> bool:
    """True for the question shape the existing ``temporal_compare`` route owns.

    That route answers "compare October and November funded balance": two named
    calendar periods and ONE metric focus. Period Change Analysis defers to it
    rather than changing an answer users already receive. A named-period question
    with no single metric focus ("what moved between March and June?") is a
    portfolio-wide change question and is not deferred.
    """
    if getattr(spec, "temporal_mode", None) != "compare":
        return False
    periods = [str(p).strip() for p in (getattr(spec, "compare_periods", None) or [])]
    named = [p for p in periods if _CALENDAR_TOKEN_RE.fullmatch(p)]
    if len(named) < 2:
        return False
    text = _padded(question)
    return bool(getattr(spec, "metric", None)) or _any(text, COUNT_FOCUS_MARKERS)


def recognise(question: str, *, spec: Any = None, view: str = "funded",
              registry: Optional[BusinessSemanticsRegistry] = None,
              semantics_context: Optional[Mapping[str, Any]] = None
              ) -> PeriodChangeIntent:
    """Decide whether this is a Period Change Analysis question, and interpret it.

    ``semantics_context`` is the additive Business Semantics slot on
    ``ParsedQuestion``; when a future resolver populates it with governed field
    or concept hints they are honoured here in preference to the controlled
    vocabulary, without any signature change.
    """
    text = _padded(question)

    # The declines run FIRST and are reported by their own reason: a forecast is
    # declined because it is a forecast, not because it happened to omit a change
    # verb. A caller inspecting the reason learns which capability owns the
    # question, which is what makes a routing decision auditable.
    if (view or "funded").strip().lower() not in ("funded", "", "mi"):
        return PeriodChangeIntent.no(DECLINE_NOT_FUNDED_VIEW)
    if _any(text, FORECAST_MARKERS) or getattr(spec, "forecast_mode", None):
        return PeriodChangeIntent.no(DECLINE_FORECAST)
    if _any(text, TREND_MARKERS) or getattr(spec, "temporal_mode", None) == "trend":
        return PeriodChangeIntent.no(DECLINE_TREND_SERIES)
    if _any(text, RAW_RECORD_MARKERS):
        return PeriodChangeIntent.no(DECLINE_RAW_RECORDS)
    if _any(text, RECONCILIATION_MARKERS):
        return PeriodChangeIntent.no(DECLINE_TRANSACTION_RECONCILIATION)
    if _any(text, CROSS_PORTFOLIO_MARKERS) or _is_cross_portfolio(question):
        return PeriodChangeIntent.no(DECLINE_CROSS_PORTFOLIO)
    if _is_incumbent_single_metric_compare(question, spec):
        return PeriodChangeIntent.no(DECLINE_INCUMBENT_SINGLE_METRIC_COMPARE)
    if not has_change_language(question):
        return PeriodChangeIntent.no(DECLINE_NO_CHANGE_LANGUAGE)

    start_token, end_token = _explicit_periods(question, spec)
    relative = _relative_mode(question)
    period_request = PeriodRequest(
        requested_start=start_token, requested_end=end_token,
        relative_mode=None if (start_token or end_token) else relative)

    context = dict(semantics_context or {})
    concepts = tuple(context.get("period_change_concepts") or ()) \
        or _requested_concepts(question, registry)
    requested_fields = tuple(context.get("period_change_fields") or ())
    if not requested_fields:
        requested_fields = tuple(
            f for f in (getattr(spec, "metric", None),) if f)

    composition = _any(text, COMPOSITION_MARKERS)
    bridge = _any(text, BRIDGE_MARKERS)
    overview_language = _any(text, OVERVIEW_MARKERS)

    mode = _mode(requested_fields, concepts, overview_language, composition)
    if mode != MODE_REQUESTED_METRIC:
        requested_fields = ()
    if mode != MODE_CONCEPT:
        concepts = () if mode == MODE_REQUESTED_METRIC else concepts

    return PeriodChangeIntent(
        matched=True, reason="period_change_language",
        mode=mode, period_request=period_request,
        requested_fields=requested_fields,
        requested_concepts=concepts if mode == MODE_CONCEPT else (),
        include_bridge=bridge, composition_focus=composition)


def _mode(requested_fields: Sequence[str], concepts: Sequence[str],
          overview_language: bool, composition: bool) -> str:
    """The workflow mode. Explicit field beats concept; both beat the overview.

    "What are the main drivers of the balance movement?" resolves a metric and no
    overview noun, so it is requested-metric mode on the balance — which is
    exactly what the question asks. "What changed in the portfolio this month?"
    resolves no metric and carries overview language, so it is an overview.
    """
    if requested_fields and not overview_language:
        return MODE_REQUESTED_METRIC
    if concepts and not composition:
        return MODE_CONCEPT
    return MODE_PORTFOLIO_OVERVIEW


def _is_cross_portfolio(question: str) -> bool:
    """A static side-by-side comparison of two portfolios, not two periods.

    Reuses the governed lens layer rather than re-detecting cohorts here, so a
    question the portfolio lens reads as "direct vs acquired" is never answered
    as a period change.
    """
    try:
        from ..portfolio_lens import resolve_comparison_lenses
        return len(resolve_comparison_lenses(question)) >= 2
    except Exception:  # noqa: BLE001 - recognition must never raise
        return False
