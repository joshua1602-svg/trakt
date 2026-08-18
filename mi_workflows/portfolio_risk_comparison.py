"""mi_workflows/portfolio_risk_comparison — governed portfolio comparison.

Deterministic comparison of two governed portfolio scopes at one reporting
date. The second governed analytical workflow, and the first consumer of the
Business Semantics Registry: every field compared, every aggregation, weight,
share basis, directionality and comparability decision comes from the
registry — this module implements the registry's declarations through the
shared calculation engine (:mod:`mi_workflows.engine`), it never invents a
metric, a mapping, a threshold or a score.

What this workflow is NOT: period change analysis, concentration analysis,
covenant monitoring, eligibility testing, forecasting or risk scoring. It
reports observed characteristics side by side; it never determines whether a
portfolio is acceptable, compliant, within appetite, safer or preferable.

Execution shape (mirrors the architecture in
docs/mi_query_agent_architecture.md §11):

    ParsedQuestion ─► recognition (pure predicates below)
                  ─► scope resolution (portfolio_lens comparison lenses ─►
                     trakt_core.portfolio.resolve_scope ─► one explicit id
                     list per side; mi_agent.portfolio_scope.apply_scope)
                  ─► Business Semantics Registry field selection
                  ─► shared engine calculations
                  ─► governed result contract (+ evidence + audit)

Pure by construction: pandas + the governed registries only. No web
framework, no LLM, no I/O beyond the registry loader the adapter hands in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from mi_agent import portfolio_lens as _lens_mod
from mi_agent.portfolio_scope import apply_scope, registry_for_frame
from trakt_core.portfolio import PortfolioRegistry, PortfolioScope, resolve_scope

from . import engine
from .semantics import (
    COMPARABLE,
    NOT_COMPARABLE,
    REQUIRES_SCALE_ALIGNMENT,
    TAG_PORTFOLIO_COMPARISON,
    WITHIN_ASSET_CLASS_ONLY,
    BusinessSemanticsRegistry,
    SemanticEntry,
)

WORKFLOW_ID = "portfolio_risk_comparison"

#: The dataset this workflow compares. Comparison always runs over the funded
#: canonical tape; pipeline/forecast datasets are other workflows' business.
DATASET = "funded"

#: Canonical reporting-date column on the governed tape.
REPORTING_DATE_FIELD = "reporting_date"

#: Asset-class column, when the tape declares one. Absent means "not declared"
#: — asset-specific fields are then excluded rather than guessed at.
ASSET_CLASS_FIELD = "asset_class"

#: Concepts a portfolio overview covers, in fixed report order (§9 of the
#: workflow brief). Order is part of the contract: same question, same output.
OVERVIEW_CONCEPTS: Tuple[str, ...] = (
    "exposure", "leverage", "payment_performance", "credit_quality",
    "collateral", "valuation", "pricing", "maturity", "product_mix",
    "geography",
)

#: Deterministic caps keeping an overview bounded. Selection inside a concept
#: is by (confidence desc, field name asc); everything not selected is recorded
#: in the audit as considered-but-not-selected, never silently dropped.
OVERVIEW_MEASURES_PER_CONCEPT = 3
OVERVIEW_DIMENSIONS_PER_CONCEPT = 2
_CONFIDENCE_ORDER = {"high": 0, "medium": 1, "low": 2}

# --------------------------------------------------------------------------- #
# Recognition — pure predicates over the question text and the single parse.
# The chat recogniser wraps these; no recogniser logic is duplicated there.
# --------------------------------------------------------------------------- #
#: Comparison language this workflow answers.
_COMPARISON_MARKERS = (
    "compare", "comparison", " versus ", " vs ", " vs. ", "difference between",
    "differences between", "side by side", "side-by-side",
)

#: Nouns that make a comparison a PORTFOLIO comparison.
_PORTFOLIO_NOUNS = (
    "portfolio", "book", "spv", "warehouse pool", "warehouse pools", "pool",
    "originator", "cohort",
)

#: "Which portfolio has higher LTV?" — a comparative question over portfolio
#: scopes with no explicit compare-verb.
_WHICH_COMPARATIVE_RE = re.compile(
    r"\b(?:which|what)\b[^?]{0,60}?\b(?:portfolio|book|spv|pool|cohort)s?\b"
    r"[^?]{0,80}?\b(?:higher|lower|stronger|weaker|better|worse|more|less|"
    r"larger|smaller|bigger)\b")

#: Questions owned by other workflows. Checked FIRST: this workflow must not
#: steal from more specific routes, whatever else the question contains.
_PERIOD_MARKERS = (
    "january", "february", "march", "april", " may ", "june", "july", "august",
    "september", "october", "november", "december",
    "last month", "prior month", "previous month", "last period", "prior period",
    "previous period", "last week", "prior week", "previous week", "last run",
    "prior run", "previous run", "last year", "prior year", "previous year",
    "month on month", "month-on-month", "year on year", "year-on-year",
    "quarter on quarter", "quarter-on-quarter", "mom", " yoy", " qoq",
    "over time", "trend", "since ", "what has changed", "what changed",
    "what's changed", "whats changed",
)
_CONCENTRATION_MARKERS = ("concentration", "concentrated", "herfindahl", "hhi",
                          "top 5", "top 10", "top ten", "top five", "largest",
                          "biggest")
_FORECAST_MARKERS = ("forecast", "project", "projection", "extrapolat",
                     "run rate", "run-rate", "will reach", "when will",
                     "expected to")
_COVENANT_MARKERS = ("covenant", "headroom", "breach", "appetite", "limit",
                     "trigger", "complian")
_ELIGIBILITY_MARKERS = ("eligib",)
_EXPORT_MARKERS = ("export", "download", "loan-level", "loan level",
                   "list of loans", "loan list", "csv", "excel", "raw data")
_RECONCILIATION_MARKERS = ("reconcil",)

_REJECTIONS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("period-change questions belong to the period movement / temporal "
     "comparison routes", _PERIOD_MARKERS),
    ("concentration questions belong to the concentration analysis "
     "capability", _CONCENTRATION_MARKERS),
    ("forecast questions belong to the forecast routes", _FORECAST_MARKERS),
    ("covenant questions belong to covenant monitoring", _COVENANT_MARKERS),
    ("eligibility questions belong to eligibility testing", _ELIGIBILITY_MARKERS),
    ("raw loan exports are not an analytical comparison", _EXPORT_MARKERS),
    ("transaction reconciliation is not an analytical comparison",
     _RECONCILIATION_MARKERS),
)


def rejection_reason(question: str, spec: Any = None) -> Optional[str]:
    """Why this workflow refuses the question, or ``None`` if it does not.

    A question owned by a more specific workflow is rejected regardless of any
    comparison language it also contains.
    """
    q = f" {str(question or '').strip().lower()} "
    if spec is not None:
        if getattr(spec, "temporal_mode", None) in ("compare", "trend"):
            return ("the parse resolved a cross-period intent, owned by the "
                    "temporal comparison / evolution routes")
        if getattr(spec, "forecast_mode", None):
            return "the parse resolved a forecast intent, owned by the forecast routes"
    for reason, markers in _REJECTIONS:
        if any(m in q for m in markers):
            return reason
    return None


def is_portfolio_comparison_question(question: str, spec: Any = None) -> Tuple[bool, str]:
    """``(matched, reason)`` — pure recognition for the chat recogniser.

    Matches multi-portfolio comparison language; defers everything owned by a
    more specific workflow.
    """
    q = f" {str(question or '').strip().lower()} "
    rejection = rejection_reason(question, spec)
    if rejection is not None:
        return False, rejection

    # Two governed scopes named explicitly is the strongest signal.
    if len(_lens_mod.resolve_comparison_lenses(question)) >= 2:
        return True, "two or more governed portfolio scopes named"

    has_marker = any(m in q for m in _COMPARISON_MARKERS)
    has_noun = any(n in q for n in _PORTFOLIO_NOUNS)
    if has_marker and has_noun:
        return True, "comparison language over portfolio scopes"
    if _WHICH_COMPARATIVE_RE.search(q):
        return True, "comparative question across portfolios"
    return False, "no multi-portfolio comparison language"


# --------------------------------------------------------------------------- #
# Scope resolution
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScopeResolution:
    """Two resolved sides, or the controlled reason there are not two."""

    scopes: Tuple[PortfolioScope, ...] = ()
    failure: Optional[str] = None


def resolve_comparison_scopes(question: str,
                              registry: PortfolioRegistry) -> ScopeResolution:
    """Resolve the two governed sides a comparison question names.

    Explicitly named scopes (cohort ids, originated-vs-acquired language) are
    resolved through the SAME governed registry resolution every other channel
    uses. A generic "compare the portfolios" resolves to the two governed type
    groups when both exist, or to the only two portfolios in scope. Anything
    that cannot be resolved to exactly two non-overlapping governed scopes is a
    controlled failure that says what IS available — never a guess.
    """
    if not len(registry):
        return ScopeResolution(failure=(
            "no governed portfolio registry is available for this deployment — "
            "the tape carries no source-portfolio provenance, so there are no "
            "distinct portfolio scopes to compare"))

    lenses = _lens_mod.resolve_comparison_lenses(question)
    if len(lenses) > 2:
        named = ", ".join(l.label for l in lenses)
        return ScopeResolution(failure=(
            f"more than two portfolio scopes were named ({named}); this "
            "comparison is pairwise — name exactly two"))
    if len(lenses) == 2:
        scopes: List[PortfolioScope] = []
        for lens in lenses:
            scope = resolve_scope(registry, _lens_mod.context_id(lens))
            if scope.fell_back_to_total:
                available = ", ".join(registry.ids()) or "none"
                return ScopeResolution(failure=(
                    f"'{lens.label}' is not a governed portfolio scope in this "
                    f"deployment; governed portfolios: {available}"))
            scopes.append(scope)
        if set(scopes[0].portfolio_ids) & set(scopes[1].portfolio_ids):
            return ScopeResolution(failure=(
                f"the scopes '{scopes[0].label}' and '{scopes[1].label}' "
                "overlap; comparison requires two distinct portfolio scopes"))
        return ScopeResolution(scopes=tuple(scopes))

    # Generic multi-portfolio comparison: "compare the portfolios / SPVs".
    # ``registry.types()`` is already deterministic (canonical order, then any
    # other types sorted) — preserve it so "compare the portfolios" always
    # reads originated-vs-acquired the same way round.
    types = [t for t in registry.types() if registry.of_type(t)]
    if len(types) >= 2:
        scopes = tuple(resolve_scope(registry, t) for t in types[:2])
        return ScopeResolution(scopes=scopes)
    ids = registry.ids()
    if len(ids) == 2:
        return ScopeResolution(scopes=tuple(resolve_scope(registry, pid)
                                            for pid in ids))
    if len(ids) < 2:
        return ScopeResolution(failure=(
            "only one governed portfolio is in scope, so there is nothing to "
            "compare it with"))
    available = ", ".join(ids)
    return ScopeResolution(failure=(
        f"several governed portfolios are in scope ({available}); name the two "
        "to compare, e.g. 'compare "
        f"{ids[0]} with {ids[1]}'"))


# --------------------------------------------------------------------------- #
# Reporting date / currency / asset class resolution per side
# --------------------------------------------------------------------------- #
def _distinct(series: pd.Series) -> Tuple[str, ...]:
    values = series.astype("string").str.strip()
    return tuple(sorted(v for v in values.dropna().unique() if v))


def resolve_common_reporting_date(frame_a: pd.DataFrame, frame_b: pd.DataFrame,
                                  as_of: Optional[str]
                                  ) -> Tuple[Optional[str], Optional[str], List[str]]:
    """``(reporting_date, failure, warnings)`` for the two sides.

    Portfolios are only ever compared at ONE reporting date. When the tape
    carries reporting-date provenance, both sides must resolve to the same
    single date; otherwise the comparison is refused. A tape without the
    column is a single governed snapshot by construction — the caller's
    ``as_of`` describes it, and that is disclosed rather than asserted.
    """
    warnings: List[str] = []
    has_a = REPORTING_DATE_FIELD in frame_a.columns
    has_b = REPORTING_DATE_FIELD in frame_b.columns
    if not has_a and not has_b:
        warnings.append(
            "the tape carries no reporting_date column; both scopes come from "
            "the same governed snapshot"
            + (f" (as of {as_of})" if as_of else ""))
        return as_of, None, warnings
    dates_a = _distinct(frame_a[REPORTING_DATE_FIELD]) if has_a else ()
    dates_b = _distinct(frame_b[REPORTING_DATE_FIELD]) if has_b else ()
    if len(dates_a) > 1 or len(dates_b) > 1:
        side = "first" if len(dates_a) > 1 else "second"
        many = dates_a if len(dates_a) > 1 else dates_b
        return None, (
            f"the {side} portfolio scope spans multiple reporting dates "
            f"({', '.join(many)}); a comparison must be at one reporting date"), warnings
    date_a = dates_a[0] if dates_a else None
    date_b = dates_b[0] if dates_b else None
    if date_a and date_b and date_a != date_b:
        return None, (
            f"the portfolios are at different reporting dates ({date_a} vs "
            f"{date_b}); comparison across reporting dates is not performed"), warnings
    common = date_a or date_b or as_of
    return common, None, warnings


def _asset_classes(frame: pd.DataFrame,
                   scope: Optional[PortfolioScope] = None) -> Tuple[str, ...]:
    """The governed asset classes of a side of the comparison.

    GOVERNED PORTFOLIO METADATA FIRST. The asset class is a fact about the book,
    established once at onboarding and carried on the portfolio registry; the
    tape may or may not stamp a column of the same name. Reading only the column
    is what left this workflow with an unknown asset class on every deployment
    whose extract does not carry one — which silently excluded every
    asset-specific metric from every comparison.
    """
    if scope is not None and getattr(scope, "asset_classes", ()):
        return tuple(str(a).lower() for a in scope.asset_classes)
    if ASSET_CLASS_FIELD not in frame.columns:
        return ()
    return tuple(v.lower() for v in _distinct(frame[ASSET_CLASS_FIELD]))


def _shared_asset_class(classes_a: Tuple[str, ...],
                        classes_b: Tuple[str, ...]) -> Optional[str]:
    """The single governed asset class both sides declare, else ``None``."""
    if len(classes_a) == 1 and classes_a == classes_b:
        return classes_a[0]
    return None


# --------------------------------------------------------------------------- #
# Field selection — the Business Semantics Registry decides
# --------------------------------------------------------------------------- #
#: Question-language → analytical concept, first match wins (checked in this
#: order). Only concepts in the registry taxonomy appear here.
_CONCEPT_LANGUAGE: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("payment_performance", ("payment performance", "arrears", "delinquen",
                             "past due", "days past due")),
    ("credit_quality", ("credit quality", "credit risk", "rating", "default",
                        "impair")),
    ("geography", ("geograph", "region", "regional")),
    ("product_mix", ("product mix", "product composition", "products",
                     "product type")),
    ("pricing", ("pricing", "margin", "interest rate", "coupon", "yield")),
    ("leverage", ("leverage", "ltv", "loan to value", "loan-to-value")),
    ("collateral", ("collateral", "security", "valuation")),
    ("maturity", ("maturity", "term", "duration", "seasoning")),
    ("exposure", ("exposure", "balance", "size")),
    ("loss", ("loss", "losses", "recover")),
    ("cashflow", ("cashflow", "cash flow", "prepayment")),
    ("occupancy", ("occupancy",)),
)


def requested_concept(question: str) -> Optional[str]:
    q = str(question or "").lower()
    for concept, markers in _CONCEPT_LANGUAGE:
        if any(m in q for m in markers):
            return concept
    return None


@dataclass(frozen=True)
class FieldDecision:
    """One considered field and what was decided about it."""

    entry: SemanticEntry
    selected: bool
    reason: str

    def to_audit(self) -> Dict[str, Any]:
        return {"field": self.entry.source_field,
                "comparability": self.entry.portfolio_comparability,
                "selected": self.selected, "reason": self.reason}


def _comparability_decision(entry: SemanticEntry, shared_asset: Optional[str],
                            columns: Sequence[str]) -> FieldDecision:
    """Apply the registry's comparability and applicability rules to one field."""
    if not entry.tagged(TAG_PORTFOLIO_COMPARISON):
        return FieldDecision(entry, False,
                             "not governed for portfolio comparison")
    if entry.analytical_role not in ("measure", "dimension"):
        return FieldDecision(entry, False,
                             f"analytical role '{entry.analytical_role}' is not "
                             "reportable as a standalone comparison")
    if entry.portfolio_comparability == NOT_COMPARABLE:
        return FieldDecision(entry, False,
                             "registry declares this field not comparable "
                             "across portfolios")
    if entry.portfolio_comparability == REQUIRES_SCALE_ALIGNMENT:
        return FieldDecision(entry, False,
                             "originator/servicer-specific scale — values are "
                             "not aligned across portfolios, so no comparison "
                             "is made (no heuristic mapping is created)")
    if entry.portfolio_comparability == WITHIN_ASSET_CLASS_ONLY and shared_asset is None:
        return FieldDecision(entry, False,
                             "comparable within one asset class only, and the "
                             "portfolios do not declare a single shared "
                             "governed asset class")
    if not entry.is_cross_asset():
        if shared_asset is None:
            return FieldDecision(entry, False,
                                 "asset-specific field and the portfolios do "
                                 "not declare a single shared governed asset "
                                 "class")
        if shared_asset not in entry.asset_applicability:
            return FieldDecision(entry, False,
                                 f"not applicable to asset class '{shared_asset}'")
    if entry.source_field not in columns:
        return FieldDecision(entry, False, "field not present on the tape")
    return FieldDecision(entry, True, "governed for portfolio comparison")


def _bounded(decisions: List[FieldDecision], per_concept: int,
             concepts: Sequence[str]) -> List[FieldDecision]:
    """Deterministically cap selected fields per concept for an overview."""
    out: List[FieldDecision] = []
    for concept in concepts:
        candidates = sorted(
            (d for d in decisions if d.selected
             and d.entry.analytical_concept == concept),
            key=lambda d: (_CONFIDENCE_ORDER.get(d.entry.confidence, 9),
                           d.entry.source_field))
        out.extend(candidates[:per_concept])
        for dropped in candidates[per_concept:]:
            out.append(FieldDecision(dropped.entry, False,
                                     "considered but not selected (overview "
                                     "bounded per concept)"))
    return out


# --------------------------------------------------------------------------- #
# The workflow
# --------------------------------------------------------------------------- #
def _controlled_failure(reason: str, *, question: str,
                        bsr: Optional[BusinessSemanticsRegistry],
                        as_of: Optional[str]) -> Dict[str, Any]:
    return {
        "workflow_id": WORKFLOW_ID,
        "available": False,
        "reason": reason,
        "comparison_scope": None,
        "reporting_date": as_of,
        "portfolio_results": [],
        "metric_comparisons": [],
        "distribution_comparisons": [],
        "warnings": [reason],
        "limitations": [],
        "summary": [],
        "evidence": [],
        "audit": {
            "workflow_id": WORKFLOW_ID,
            "calculation_version": engine.CALCULATION_VERSION,
            "bsr_version": bsr.version if bsr else None,
            "bsr_schema_version": bsr.schema_version if bsr else None,
            "question": question,
            "outcome": "controlled_failure",
            "reason": reason,
        },
    }


def _side_result(key: str, scope: PortfolioScope, frame: pd.DataFrame,
                 currencies: Tuple[str, ...],
                 asset_classes: Tuple[str, ...]) -> Dict[str, Any]:
    return {
        "key": key,
        "context_id": scope.context_id,
        "label": scope.label,
        "portfolio_ids": list(scope.portfolio_ids),
        "row_count": int(len(frame)),
        "currencies": list(currencies),
        "asset_classes": list(asset_classes),
    }


def run_portfolio_risk_comparison(
        df: pd.DataFrame, *,
        question: str,
        bsr: BusinessSemanticsRegistry,
        mi_semantics: Optional[Mapping[str, Any]] = None,
        registry: Optional[PortfolioRegistry] = None,
        client_id: Optional[str] = None,
        as_of: Optional[str] = None,
        spec: Any = None) -> Dict[str, Any]:
    """Run the workflow over one governed frame. Returns the result contract.

    Never raises for an analytical reason: every non-answer is a controlled
    result with ``available: False`` and an explanation.
    """
    reg = registry if registry is not None else registry_for_frame(df, client_id=client_id)

    resolution = resolve_comparison_scopes(question, reg)
    if resolution.failure:
        return _controlled_failure(resolution.failure, question=question,
                                   bsr=bsr, as_of=as_of)
    scope_a, scope_b = resolution.scopes
    frame_a = apply_scope(df, scope_a)
    frame_b = apply_scope(df, scope_b)
    for scope, frame in ((scope_a, frame_a), (scope_b, frame_b)):
        if frame is None or not len(frame):
            return _controlled_failure(
                f"no rows in scope for '{scope.label}' on the governed dataset",
                question=question, bsr=bsr, as_of=as_of)

    reporting_date, failure, warnings = resolve_common_reporting_date(
        frame_a, frame_b, as_of)
    if failure:
        return _controlled_failure(failure, question=question, bsr=bsr,
                                   as_of=as_of)

    limitations: List[str] = []

    # ---- currency: the mixed-currency guard, no FX ----------------------- #
    currencies_a = engine.currency_profile(frame_a)
    currencies_b = engine.currency_profile(frame_b)
    monetary_ok, currency_reason = engine.mixed_currency_guard(currencies_a,
                                                               currencies_b)
    if not monetary_ok:
        limitations.append(currency_reason)
    elif not currencies_a and not currencies_b:
        warnings.append("the tape declares no currency column; the platform's "
                        "single-currency book assumption applies")

    # ---- asset class ------------------------------------------------------ #
    classes_a = _asset_classes(frame_a, scope_a)
    classes_b = _asset_classes(frame_b, scope_b)
    shared_asset = _shared_asset_class(classes_a, classes_b)
    if classes_a and classes_b and set(classes_a) != set(classes_b):
        limitations.append(
            f"the portfolios hold different asset classes "
            f"({', '.join(classes_a)} vs {', '.join(classes_b)}); "
            "asset-specific metrics are excluded, cross-asset metrics are "
            "compared")
    elif not classes_a and not classes_b:
        limitations.append(
            "neither book declares a governed asset class (no portfolio "
            "metadata and no asset_class column); asset-specific metrics are "
            "excluded and only cross-asset metrics are compared")

    # ---- field selection: registry-governed ------------------------------- #
    columns = tuple(df.columns)
    # P1E: a comparison may be asked for SEVERAL governed measures at once
    # ("compare the books on balance, count, LTV and borrower age"). The
    # requested set is one list; a single requested metric is a one-element case
    # of it, so there is one code path rather than two.
    requested_fields: List[str] = []
    #: Requested measures this comparison cannot express. Carried separately so
    #: they are DISCLOSED rather than dropped: a loan count asked for alongside
    #: three comparable measures used to vanish, and an answer naming three of
    #: four reads as complete.
    uncomparable: List[Dict[str, Any]] = []
    if spec is not None:
        for measure in (getattr(spec, "measures", None) or []):
            name = str((measure or {}).get("field") or "").strip()
            if not name:
                continue
            if name in ("loan_count", "count"):
                # DISCLOSED, not compared. A loan count is not a Business
                # Semantics Registry measure — it carries no directionality or
                # comparability declaration — so it is reported separately from
                # ``requested_metrics``, which is the set of BSR measures this
                # workflow undertook to compare. It must still reach the reader:
                # dropping it here is what let a requested figure vanish.
                if not any(u["field"] == name for u in uncomparable):
                    uncomparable.append({
                        "field": name, "display_name": "Loan count",
                        "compared": False,
                        "reason": ("a loan count is not a Business Semantics "
                                   "Registry measure for portfolio comparison")})
                continue
            if name not in requested_fields:
                requested_fields.append(name)
        single = getattr(spec, "metric", None)
        if not requested_fields and single:
            requested_fields.append(single)
    metric_field = requested_fields[0] if requested_fields else None
    concept = requested_concept(question)

    decisions: List[FieldDecision] = []
    mode: str
    #: What became of a metric the caller explicitly asked for. Reported on the
    #: result so a presenter can never mistake "nothing was compared" for "the
    #: portfolios do not differ" — see ``requested_metric`` in the return value.
    requested_metric: Optional[Dict[str, Any]] = None
    requested_metrics: List[Dict[str, Any]] = []
    if requested_fields:
        mode = "requested_metric"
        for field_name in requested_fields:
            entry = bsr.get(field_name)
            if entry is None:
                limitations.append(
                    f"'{field_name}' is not governed by the Business Semantics "
                    "Registry for analytical comparison, so it is not compared")
                requested_metrics.append({
                    "field": field_name, "display_name": field_name,
                    "compared": False,
                    "reason": ("not governed by the Business Semantics Registry "
                               "for analytical comparison")})
                continue
            decision = _comparability_decision(entry, shared_asset, columns)
            decisions.append(decision)
            if not decision.selected:
                # The governed rules declined this field. That decision is sound
                # — but it must not vanish. Previously it stayed inside
                # `decisions`, `metric_comparisons` came back empty, and the
                # caller rendered the empty list as "no differences observed".
                limitations.append(
                    f"'{field_name}' was requested but not compared: "
                    f"{decision.reason}")
            requested_metrics.append({
                "field": field_name,
                "display_name": entry.display_name,
                "compared": bool(decision.selected),
                "reason": decision.reason,
            })
        for entry_note in uncomparable:
            limitations.append(
                f"'{entry_note['field']}' was requested but not compared: "
                f"{entry_note['reason']}")
        requested_metric = requested_metrics[0] if requested_metrics else None
    elif concept is not None:
        mode = "requested_concept"
        decisions = [_comparability_decision(e, shared_asset, columns)
                     for e in bsr.for_workflow(TAG_PORTFOLIO_COMPARISON)
                     if e.analytical_concept == concept]
        if not decisions:
            limitations.append(
                f"no fields are governed for portfolio comparison under the "
                f"concept '{concept}'")
    else:
        mode = "overview"
        decisions = [_comparability_decision(e, shared_asset, columns)
                     for e in bsr.for_workflow(TAG_PORTFOLIO_COMPARISON)
                     if e.analytical_concept in OVERVIEW_CONCEPTS]
        measures = [d for d in decisions if d.entry.analytical_role == "measure"]
        dimensions = [d for d in decisions if d.entry.analytical_role == "dimension"]
        decisions = (_bounded(measures, OVERVIEW_MEASURES_PER_CONCEPT,
                              OVERVIEW_CONCEPTS)
                     + [d for d in measures if not d.selected]
                     + _bounded(dimensions, OVERVIEW_DIMENSIONS_PER_CONCEPT,
                                OVERVIEW_CONCEPTS)
                     + [d for d in dimensions if not d.selected])

    # ---- calculations: the shared engine only ----------------------------- #
    metric_comparisons: List[Dict[str, Any]] = []
    distribution_comparisons: List[Dict[str, Any]] = []
    evidence: List[Dict[str, Any]] = []
    suppressed_monetary: List[str] = []

    selected = [d for d in decisions if d.selected]
    for decision in selected:
        entry = decision.entry
        if entry.analytical_role == "dimension":
            comparison = _compare_distribution(entry, frame_a, frame_b,
                                               currencies_a, currencies_b,
                                               limitations)
            if comparison is not None:
                distribution_comparisons.append(comparison)
                evidence.append(_distribution_evidence(entry, comparison))
            continue
        unit = engine.unit_for_field(entry.source_field, mi_semantics)
        if engine.is_monetary(unit) and not monetary_ok:
            suppressed_monetary.append(entry.source_field)
            for requested in requested_metrics:
                if requested["field"] == entry.source_field:
                    requested["compared"] = False
                    requested["reason"] = (
                        "suppressed by the currency guard: the portfolios do "
                        "not share a single governed currency")
            continue
        comparison = _compare_metric(entry, frame_a, frame_b, unit)
        metric_comparisons.append(comparison)
        evidence.append(_metric_evidence(entry, comparison))

    if suppressed_monetary:
        limitations.append(
            "monetary metrics suppressed by the currency guard: "
            + ", ".join(sorted(suppressed_monetary)))

    summary = _summary(scope_a, scope_b, metric_comparisons, mode)

    audit = {
        "workflow_id": WORKFLOW_ID,
        "calculation_version": engine.CALCULATION_VERSION,
        "bsr_version": bsr.version,
        "bsr_schema_version": bsr.schema_version,
        "reporting_date": reporting_date,
        "dataset": DATASET,
        "question": question,
        "mode": mode,
        "selected_portfolios": {
            "portfolio_a": scope_a.to_dict(),
            "portfolio_b": scope_b.to_dict(),
        },
        "currencies": {"portfolio_a": list(currencies_a),
                       "portfolio_b": list(currencies_b),
                       "monetary_comparison_allowed": monetary_ok},
        "asset_classes": {"portfolio_a": list(classes_a),
                          "portfolio_b": list(classes_b),
                          "shared_asset_class": shared_asset},
        "fields_considered": sorted({d.entry.source_field for d in decisions}),
        "fields_selected": sorted({d.entry.source_field for d in decisions
                                   if d.selected}
                                  - set(suppressed_monetary)),
        "fields_excluded": sorted(
            [d.to_audit() for d in decisions if not d.selected]
            + [{"field": f, "comparability": COMPARABLE, "selected": False,
                "reason": "monetary comparison suppressed by the currency guard"}
               for f in suppressed_monetary],
            key=lambda item: item["field"]),
        "comparability_decisions": sorted((d.to_audit() for d in decisions),
                                          key=lambda item: item["field"]),
        "aggregations": {c["field"]: c["aggregation"] for c in metric_comparisons},
        "weights": {c["field"]: c["weight_basis"] for c in metric_comparisons
                    if c["weight_basis"]},
        "denominators": {c["field"]: {"portfolio_a": c["portfolio_a"]["denominator"],
                                      "portfolio_b": c["portfolio_b"]["denominator"]}
                         for c in metric_comparisons},
        "outcome": "success",
    }

    return {
        "workflow_id": WORKFLOW_ID,
        "available": True,
        "reason": None,
        "comparison_scope": {
            "mode": mode,
            "dataset": DATASET,
            "portfolio_a": scope_a.to_dict(),
            "portfolio_b": scope_b.to_dict(),
        },
        "reporting_date": reporting_date,
        "portfolio_results": [
            _side_result("portfolio_a", scope_a, frame_a, currencies_a, classes_a),
            _side_result("portfolio_b", scope_b, frame_b, currencies_b, classes_b),
        ],
        "metric_comparisons": metric_comparisons,
        "distribution_comparisons": distribution_comparisons,
        # None unless the caller named a metric. ``compared`` is the invariant a
        # presenter must honour: a requested metric that was not compared may
        # never be answered as though it had been.
        "requested_metric": requested_metric,
        # P1E: the full requested SET. Every entry carries its own outcome, so a
        # four-measure comparison is accounted for measure by measure.
        "requested_metrics": requested_metrics,
        #: Requested measures this workflow does not express at all. Separate
        #: from ``requested_metrics`` — those are BSR measures it undertook to
        #: compare — but reported so the answer can name them.
        "uncomparable_measures": uncomparable,
        "warnings": warnings,
        "limitations": limitations,
        "summary": summary,
        "evidence": evidence,
        "audit": audit,
    }


def _compare_metric(entry: SemanticEntry, frame_a: pd.DataFrame,
                    frame_b: pd.DataFrame, unit: str) -> Dict[str, Any]:
    """One governed metric comparison, computed by the shared engine."""
    agg_a = engine.aggregate(frame_a, entry.source_field,
                             entry.default_aggregation,
                             weight_field=entry.weight_field,
                             share_basis=entry.share_basis, unit=unit)
    agg_b = engine.aggregate(frame_b, entry.source_field,
                             entry.default_aggregation,
                             weight_field=entry.weight_field,
                             share_basis=entry.share_basis, unit=unit)
    return {
        "field": entry.source_field,
        "display_name": entry.display_name,
        "concept": entry.analytical_concept,
        "aggregation": entry.default_aggregation,
        "weight_basis": agg_a.weight_basis,
        "unit": agg_a.unit,
        "portfolio_a": agg_a.to_dict(),
        "portfolio_b": agg_b.to_dict(),
        "difference": engine.compare_values(agg_a.value, agg_b.value),
        "directionality": {
            "declared": entry.directionality,
            **engine.directionality_verdict(entry.directionality,
                                            agg_a.value, agg_b.value),
        },
        "comparability": entry.portfolio_comparability,
        "confidence": entry.confidence,
    }


def _compare_distribution(entry: SemanticEntry, frame_a: pd.DataFrame,
                          frame_b: pd.DataFrame,
                          currencies_a: Tuple[str, ...],
                          currencies_b: Tuple[str, ...],
                          limitations: List[str]) -> Optional[Dict[str, Any]]:
    """One governed distribution comparison (count / exposure / unknown share).

    Exposure shares are within-portfolio ratios, so they stay valid across
    portfolios in different currencies; they are suppressed only for a scope
    that is internally currency-mixed (a ratio over mixed currencies would mix
    units).
    """
    exposure_field = "current_outstanding_balance"
    dist_a = engine.distribution(frame_a, entry.source_field,
                                 exposure_field=None if len(currencies_a) > 1
                                 else exposure_field)
    dist_b = engine.distribution(frame_b, entry.source_field,
                                 exposure_field=None if len(currencies_b) > 1
                                 else exposure_field)
    if dist_a is None or dist_b is None:
        return None
    for side, dist, profile in (("portfolio_a", dist_a, currencies_a),
                                ("portfolio_b", dist_b, currencies_b)):
        if len(profile) > 1:
            note = (f"exposure shares suppressed for {side}: scope holds mixed "
                    f"currencies ({', '.join(profile)})")
            if note not in limitations:
                limitations.append(note)

    categories = sorted({b.category for b in dist_a.buckets}
                        | {b.category for b in dist_b.buckets})
    rows: List[Dict[str, Any]] = []
    for category in categories:
        bucket_a = dist_a.bucket(category)
        bucket_b = dist_b.bucket(category)
        share_a = bucket_a.count_share if bucket_a else 0.0
        share_b = bucket_b.count_share if bucket_b else 0.0
        exp_a = bucket_a.exposure_share if bucket_a else (
            0.0 if dist_a.exposure_field else None)
        exp_b = bucket_b.exposure_share if bucket_b else (
            0.0 if dist_b.exposure_field else None)
        present = [side for side, bucket in (("portfolio_a", bucket_a),
                                             ("portfolio_b", bucket_b))
                   if bucket is not None]
        rows.append({
            "category": category,
            "count_share_a": share_a, "count_share_b": share_b,
            "count_share_difference": (share_a - share_b
                                       if share_a is not None
                                       and share_b is not None else None),
            "exposure_share_a": exp_a, "exposure_share_b": exp_b,
            "exposure_share_difference": (exp_a - exp_b
                                          if exp_a is not None
                                          and exp_b is not None else None),
            "present_in": present,
        })
    return {
        "field": entry.source_field,
        "display_name": entry.display_name,
        "concept": entry.analytical_concept,
        "comparability": entry.portfolio_comparability,
        "confidence": entry.confidence,
        "exposure_basis": exposure_field,
        "population": {"portfolio_a": dist_a.population,
                       "portfolio_b": dist_b.population},
        "unknown_share": {"portfolio_a": dist_a.unknown_share,
                          "portfolio_b": dist_b.unknown_share},
        "categories": rows,
    }


def _metric_evidence(entry: SemanticEntry, comparison: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "kind": "metric_population",
        "field": entry.source_field,
        "aggregation": comparison["aggregation"],
        "weight_basis": comparison["weight_basis"],
        "portfolio_a": {k: comparison["portfolio_a"][k]
                        for k in ("population", "valid_population",
                                  "excluded_population", "denominator")},
        "portfolio_b": {k: comparison["portfolio_b"][k]
                        for k in ("population", "valid_population",
                                  "excluded_population", "denominator")},
        "source": f"{DATASET} canonical tape",
    }


def _distribution_evidence(entry: SemanticEntry,
                           comparison: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "kind": "distribution_population",
        "field": entry.source_field,
        "exposure_basis": comparison["exposure_basis"],
        "portfolio_a": {"population": comparison["population"]["portfolio_a"],
                        "unknown_share": comparison["unknown_share"]["portfolio_a"]},
        "portfolio_b": {"population": comparison["population"]["portfolio_b"],
                        "unknown_share": comparison["unknown_share"]["portfolio_b"]},
        "source": f"{DATASET} canonical tape",
    }


# --------------------------------------------------------------------------- #
# Summary — deterministic, observational, never a judgement
# --------------------------------------------------------------------------- #
def _summary(scope_a: PortfolioScope, scope_b: PortfolioScope,
             metric_comparisons: List[Dict[str, Any]],
             mode: str) -> List[str]:
    """Observational summary lines.

    Each line states an observed relation ("higher", "lower") — never a
    judgement ("safer", "higher quality", "preferable", "within appetite").
    In an overview, at most one line per concept (the first selected metric,
    which the deterministic selection ordered by confidence), plus an
    indicator tally that is explicitly not a score.
    """
    lines: List[str] = []
    seen_concepts: set = set()
    favour_a = favour_b = 0
    for comparison in metric_comparisons:
        verdict = comparison["directionality"]["interpretation"]
        if verdict == engine.FAVOURS_A:
            favour_a += 1
        elif verdict == engine.FAVOURS_B:
            favour_b += 1
        higher = comparison["directionality"]["higher"]
        if higher is None:
            continue
        if mode == "overview":
            if comparison["concept"] in seen_concepts:
                continue
            seen_concepts.add(comparison["concept"])
        high_label = scope_a.label if higher == "portfolio_a" else scope_b.label
        low_label = scope_b.label if higher == "portfolio_a" else scope_a.label
        lines.append(f"{high_label} has higher observed "
                     f"{comparison['display_name']} than {low_label}.")
    directional = favour_a + favour_b
    if directional >= 2:
        lines.append(
            f"Across {directional} governed directional indicators, "
            f"{scope_a.label} is more favourable on {favour_a} and "
            f"{scope_b.label} on {favour_b}. This is an observation per "
            "indicator, not an overall assessment of either portfolio.")
    return lines
