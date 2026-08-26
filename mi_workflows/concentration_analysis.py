"""mi_workflows/concentration_analysis — governed concentration measurement.

Deterministic measurement of how portfolio exposure is distributed across
governed dimensions, at one reporting date, over one governed portfolio scope.
The third governed analytical workflow and the second consumer of the Business
Semantics Registry: every dimension analysed comes from the registry's
declarations (``analytical_role = dimension`` carrying the governed
``concentration`` category), every number comes from the shared calculation
engine (:mod:`mi_workflows.engine`), and every exclusion is recorded with its
reason. This module never invents a dimension, a basis, a threshold or an
index.

The workflow MEASURES concentration; it never decides whether concentration is
acceptable. What this workflow is NOT: portfolio risk comparison, period
change analysis, covenant monitoring, eligibility testing, risk appetite
assessment, or forecasting. No HHI or other concentration index, no limits, no
materiality thresholds — those belong to future workflows that will consume
this one's measurements.

Execution shape (mirrors the architecture in
docs/mi_query_agent_architecture.md §11):

    ParsedQuestion ─► recognition (pure predicates below)
                  ─► scope resolution (portfolio lens ─►
                     trakt_core.portfolio.resolve_scope ─► one explicit id
                     list; mi_agent.portfolio_scope.apply_scope)
                  ─► Business Semantics Registry dimension selection
                  ─► shared engine calculations (ranked distributions)
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
from .portfolio_risk_comparison import (
    _COVENANT_MARKERS,
    _ELIGIBILITY_MARKERS,
    _EXPORT_MARKERS,
    _FORECAST_MARKERS,
    _PERIOD_MARKERS,
    _RECONCILIATION_MARKERS,
)
from .semantics import (
    COMPARABLE,
    BusinessSemanticsRegistry,
    SemanticEntry,
)

WORKFLOW_ID = "concentration_analysis"

#: The dataset this workflow measures. Concentration always runs over the
#: funded canonical tape; pipeline/forecast datasets are other workflows'.
DATASET = "funded"

#: Canonical reporting-date column on the governed tape.
REPORTING_DATE_FIELD = "reporting_date"

#: Asset-class column, when the tape declares one. Absent means "not declared"
#: — asset-specific dimensions are then excluded rather than guessed at.
ASSET_CLASS_FIELD = "asset_class"

#: The single governed exposure basis: current outstanding balance. Loans
#: (count) is the only other basis. No arbitrary bases exist.
EXPOSURE_FIELD = "current_outstanding_balance"

#: The BSR category that declares a dimension relevant to concentration
#: analysis (metadata.taxonomy.categories). Governance, not name inference.
CONCENTRATION_CATEGORY = "concentration"

#: Governed identifier fields for single-name concentration, per named kind,
#: most canonical first. Only these canonical-registry identifiers are used —
#: no heuristic grouping and no connected-counterparty logic.
SINGLE_NAME_IDENTIFIERS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("loan", ("loan_identifier",)),
    ("borrower", ("borrower_identifier", "borrower_1_id")),
    ("obligor", ("original_obligor_identifier", "new_obligor_identifier")),
)

#: Deterministic bound for an overview: at most this many dimensions per
#: analytical concept, selected by (confidence desc, field name asc);
#: everything considered-but-not-selected is recorded in the audit.
OVERVIEW_DIMENSIONS_PER_CONCEPT = 1
_CONFIDENCE_ORDER = {"high": 0, "medium": 1, "low": 2}

#: Names listed for a single-name ranking when no N was requested. Everything
#: beyond the listing is disclosed as an explicit remainder, never dropped.
DEFAULT_SINGLE_NAME_LISTING = 10

#: Default N for "top N account for X%" summary lines when none was requested.
DEFAULT_TOP_N = 5

# --------------------------------------------------------------------------- #
# Recognition — pure predicates over the question text and the single parse.
# The chat recogniser wraps these; no recogniser logic is duplicated there.
# --------------------------------------------------------------------------- #
#: Concentration language this workflow answers.
_CONCENTRATION_TERMS = ("concentration", "concentrations", "concentrated",
                        "diversif", "single name", "single-name")

#: "<family> mix" — a composition question over a governed dimension family.
_MIX_RE = re.compile(
    r"\b(?:product|products|geographic|geography|regional|region|broker|"
    r"brokers|originator|origination|channel|industry|sector|collateral|"
    r"currency|customer|tenure|occupancy|rating|portfolio)\s+mix\b")

#: "exposure by X" / "distribution by X" — a share-of-book composition ask.
#: Deliberately narrow: "balance by broker" and every other "<metric> by X"
#: stays with the point-in-time stratification path.
_EXPOSURE_BY_RE = re.compile(r"\bexposures?\s+by\s+\w")
_DISTRIBUTION_BY_RE = re.compile(r"\bdistribution\s+(?:of\s+exposures?\s+)?by\s+\w")

#: "largest exposures" / "top 10 loans" / "biggest borrowers" — single-name or
#: ranked concentration language.
#: A superlative may be separated from its noun by a qualifier: "largest
#: SINGLE-LOAN exposure", "biggest INDIVIDUAL loan". Without this group the
#: exact B21 wording matched neither pattern below — the question was not
#: recognised as a concentration question at all and fell through to the
#: point-in-time path, which cannot express a largest-single-name share.
#: Defined once so the two patterns can never disagree about what a
#: superlative-over-names looks like.
_SUPERLATIVE = r"(?:largest|top|biggest|highest)"
_NAME_QUALIFIER = (r"(?:\d+\s+)?"
                   r"(?:single[- ]?(?:loan|name|borrower|obligor)s?|"
                   r"individual|single)?\s*")

_TOP_EXPOSURE_RE = re.compile(
    r"\b" + _SUPERLATIVE + r"\s+" + _NAME_QUALIFIER
    + r"(?:exposures?|loans?|borrowers?|obligors?|concentrations?)\b")

#: Questions owned by other routes. Checked FIRST: this workflow must not
#: steal from more specific owners, whatever else the question contains.
_LIMIT_MARKERS = _COVENANT_MARKERS + ("threshold", "schedule 8", "within our",
                                      "within the limits", "excess", "rag ")
_COMPARISON_MARKERS = ("compare", "comparison", " versus ", " vs ", " vs. ",
                       "difference between", "differences between",
                       "side by side", "side-by-side")
_INDEX_MARKERS = ("hhi", "herfindahl", "concentration index",
                  "concentration indices", "gini")

#: The ITL3 geographic exposure capability's territory: a location word plus a
#: superlative / where-is / exposure marker (mirrors
#: ``chat_routing._is_geo_exposure``, which owns those questions today).
_GEO_TERMS = ("geograph", "region", "area", "postcode", "post code", "itl3",
              "county", "city", "town")
_GEO_OWNED_MARKERS = ("concentrat", "largest", "biggest", "most exposed",
                      "top ", "hotspot", "where is", "where are", "where's",
                      "which region", "which area", "exposure")

#: An explicitly grouped ranking — "top 5 brokers BY balance" — is an ordinary
#: ranked stratification owned by the point-in-time executor (it honours the
#: metric, the top-N and the lens). Same boundary the geo route declares.
_GROUPED_RANKING_RE = re.compile(
    r"\b(?:top|bottom|largest|biggest|smallest|lowest|highest)\b[^?]*?\bby\b")

#: "exposure by LTV band" and friends are banded stratifications of a numeric
#: axis, not categorical concentration; they stay with the executor. The
#: lookaheads keep governed categorical dimensions ("interest rate type") in.
_BANDED_AXIS_RE = re.compile(
    r"\bby\s+(?:ltv|loan[\s-]to[\s-]value|interest(?:\s+rate)?(?!\s+(?:type|index))|"
    r"rate(?!\s+type)|age|term|income|balance|valuation|value)\b")

_REJECTIONS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("period-change questions belong to the period movement / temporal "
     "comparison routes", _PERIOD_MARKERS),
    ("limit / covenant / appetite questions belong to the risk-limit "
     "monitor", _LIMIT_MARKERS),
    ("portfolio comparison questions belong to the portfolio risk "
     "comparison workflow", _COMPARISON_MARKERS),
    ("forecast questions belong to the forecast routes", _FORECAST_MARKERS),
    ("eligibility questions belong to eligibility testing", _ELIGIBILITY_MARKERS),
    ("raw loan exports are not a concentration analysis", _EXPORT_MARKERS),
    ("transaction reconciliation is not a concentration analysis",
     _RECONCILIATION_MARKERS),
    ("concentration indices are not implemented — this workflow measures "
     "concentration, it does not score it", _INDEX_MARKERS),
)


def rejection_reason(question: str, spec: Any = None) -> Optional[str]:
    """Why this workflow refuses the question, or ``None`` if it does not.

    A question owned by a more specific route is rejected regardless of any
    concentration language it also contains.
    """
    q = f" {str(question or '').strip().lower()} "
    if spec is not None:
        if getattr(spec, "temporal_mode", None) in ("compare", "trend"):
            return ("the parse resolved a cross-period intent, owned by the "
                    "temporal comparison / evolution routes")
        if getattr(spec, "forecast_mode", None):
            return "the parse resolved a forecast intent, owned by the forecast routes"
        if getattr(spec, "risk_limit_query", None):
            return ("the parse resolved a risk-limit intent, owned by the "
                    "risk-limit monitor")
        if getattr(spec, "bridge_query", None):
            return "the parse resolved an attribution bridge, owned by the bridge route"
    for reason, markers in _REJECTIONS:
        if any(m in q for m in markers):
            return reason
    if len(_lens_mod.resolve_comparison_lenses(question)) >= 2:
        return ("two portfolio scopes were named — a cross-portfolio question "
                "belongs to the portfolio risk comparison workflow")
    if any(t in q for t in _GEO_TERMS) and any(m in q for m in _GEO_OWNED_MARKERS):
        return ("location-plus-superlative questions are owned by the ITL3 "
                "geographic exposure capability")
    if _GROUPED_RANKING_RE.search(q):
        return ("an explicitly grouped ranking is owned by the point-in-time "
                "ranked stratification path")
    if _BANDED_AXIS_RE.search(q):
        return ("a numeric-band stratification is owned by the point-in-time "
                "executor")
    return None


def is_concentration_question(question: str, spec: Any = None) -> Tuple[bool, str]:
    """``(matched, reason)`` — pure recognition for the chat recogniser.

    Matches concentration / composition / diversification language over one
    portfolio scope; defers everything owned by a more specific route.
    """
    q = f" {str(question or '').strip().lower()} "
    rejection = rejection_reason(question, spec)
    if rejection is not None:
        return False, rejection
    if any(t in q for t in _CONCENTRATION_TERMS):
        return True, "concentration / diversification language"
    if _MIX_RE.search(q):
        return True, "composition (mix) language over a governed dimension family"
    if _EXPOSURE_BY_RE.search(q) or _DISTRIBUTION_BY_RE.search(q):
        return True, "exposure / distribution by a dimension"
    if _TOP_EXPOSURE_RE.search(q):
        return True, "largest / top exposure language"
    return False, "no concentration language"


# --------------------------------------------------------------------------- #
# Dimension selection — the Business Semantics Registry decides
# --------------------------------------------------------------------------- #
#: Question-language → analytical concept, first match wins (checked in this
#: order). Only concepts in the registry taxonomy appear here. Used ONLY when
#: the single parse did not itself resolve a governed dimension.
_CONCEPT_LANGUAGE: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("geography", ("geograph", "region", "regional", "jurisdiction",
                   "postcode", "post code", "country", "location")),
    ("product_mix", ("product",)),
    ("origination", ("broker", "intermediary", "introducer", "originator",
                     "origination", "channel", "source portfolio")),
    ("collateral", ("collateral", "property type", "security type",
                    "occupancy", "tenure")),
    ("operational_performance", ("servicer",)),
    ("pricing", ("rate type", "interest rate type", "index",)),
    ("exposure", ("currency", "currencies", "ead")),
    ("loss", ("lgd",)),
    ("credit_quality", ("industry", "sector", "customer", "rating", "grade",
                        "segment", "ifrs", "pd ")),
)


def requested_concept(question: str) -> Optional[str]:
    q = str(question or "").lower()
    for concept, markers in _CONCEPT_LANGUAGE:
        if any(m in q for m in markers):
            return concept
    return None


#: Single-name language → the identifier kind it names. Checked in order;
#: bare "largest/top exposures" defaults to loan-level exposure.
_SINGLE_NAME_LANGUAGE: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("borrower", ("borrower",)),
    ("obligor", ("obligor",)),
    ("loan", ("loan", "single name", "single-name", "exposure")),
)

#: A superlative over NAMES ("largest exposures", "top 10 borrowers") — note
#: "largest concentrations" is deliberately absent: that is an overview of the
#: governed dimensions, not a single-name ranking.
_SINGLE_NAME_RE = re.compile(
    r"\b" + _SUPERLATIVE + r"\s+" + _NAME_QUALIFIER
    + r"(?:exposures?|loans?|borrowers?|obligors?)\b")


def requested_single_name_kind(question: str) -> Optional[str]:
    """The single-name kind a question asks for, or ``None``.

    Only superlative / single-name framings qualify ("largest exposures",
    "top 10 borrowers", "single name concentration") — "broker concentration"
    is a dimension question, not a single-name one.
    """
    q = f" {str(question or '').strip().lower()} "
    if not (_SINGLE_NAME_RE.search(q) or "single name" in q
            or "single-name" in q):
        return None
    for kind, markers in _SINGLE_NAME_LANGUAGE:
        if any(m in q for m in markers):
            return kind
    return "loan"


@dataclass(frozen=True)
class DimensionDecision:
    """One considered dimension and what was decided about it."""

    entry: SemanticEntry
    selected: bool
    reason: str

    def to_audit(self) -> Dict[str, Any]:
        return {"field": self.entry.source_field,
                "concept": self.entry.analytical_concept,
                "selected": self.selected, "reason": self.reason}


def _dimension_decision(entry: SemanticEntry, *, declared_asset: Optional[str],
                        multi_portfolio_scope: bool,
                        columns: Sequence[str]) -> DimensionDecision:
    """Apply the registry's governance rules to one candidate dimension.

    Only ``analytical_role = dimension`` entries carrying the governed
    ``concentration`` category qualify; asset applicability and (for a scope
    spanning several portfolios) portfolio comparability come straight from
    the registry. Nothing is inferred from a field's name.
    """
    if entry.analytical_role != "dimension":
        return DimensionDecision(entry, False,
                                 f"analytical role '{entry.analytical_role}' is "
                                 "not a dimension")
    if CONCENTRATION_CATEGORY not in entry.categories:
        return DimensionDecision(entry, False,
                                 "not governed as a concentration dimension "
                                 "(no 'concentration' category in the registry)")
    if not entry.is_cross_asset():
        if declared_asset is None:
            return DimensionDecision(entry, False,
                                     "asset-specific dimension and the scope "
                                     "does not declare a single governed asset "
                                     "class")
        if declared_asset not in entry.asset_applicability:
            return DimensionDecision(entry, False,
                                     f"not applicable to asset class "
                                     f"'{declared_asset}'")
    if multi_portfolio_scope and entry.portfolio_comparability != COMPARABLE:
        return DimensionDecision(entry, False,
                                 "originator-specific vocabulary "
                                 f"({entry.portfolio_comparability}) and the "
                                 "scope spans several portfolios — categories "
                                 "would mix vocabularies (no heuristic mapping "
                                 "is created)")
    if entry.source_field not in columns:
        return DimensionDecision(entry, False, "field not present on the tape")
    return DimensionDecision(entry, True, "governed concentration dimension")


def _concentration_entries(bsr: BusinessSemanticsRegistry) -> Tuple[SemanticEntry, ...]:
    """Every registry entry that could be a concentration dimension, in
    deterministic field order."""
    return tuple(bsr.entries[name] for name in sorted(bsr.entries)
                 if bsr.entries[name].analytical_role == "dimension"
                 and CONCENTRATION_CATEGORY in bsr.entries[name].categories)


def _bounded(decisions: List[DimensionDecision],
             per_concept: int) -> List[DimensionDecision]:
    """Deterministically cap selected dimensions per concept for an overview."""
    out: List[DimensionDecision] = []
    concepts = sorted({d.entry.analytical_concept for d in decisions if d.selected})
    for concept in concepts:
        candidates = sorted(
            (d for d in decisions if d.selected
             and d.entry.analytical_concept == concept),
            key=lambda d: (_CONFIDENCE_ORDER.get(d.entry.confidence, 9),
                           d.entry.source_field))
        out.extend(candidates[:per_concept])
        for dropped in candidates[per_concept:]:
            out.append(DimensionDecision(dropped.entry, False,
                                         "considered but not selected "
                                         "(overview bounded per concept)"))
    out.extend(d for d in decisions if not d.selected)
    return out


# --------------------------------------------------------------------------- #
# Scope / reporting date helpers
# --------------------------------------------------------------------------- #
def _distinct(series: pd.Series) -> Tuple[str, ...]:
    values = series.astype("string").str.strip()
    return tuple(sorted(v for v in values.dropna().unique() if v))


def resolve_single_reporting_date(frame: pd.DataFrame, as_of: Optional[str]
                                  ) -> Tuple[Optional[str], Optional[str], List[str]]:
    """``(reporting_date, failure, warnings)`` for the scoped frame.

    Concentration is measured at ONE reporting date; a scope spanning several
    dates would double-count loans and is refused. A tape without the column
    is a single governed snapshot by construction — the caller's ``as_of``
    describes it, and that is disclosed rather than asserted.
    """
    warnings: List[str] = []
    if REPORTING_DATE_FIELD not in frame.columns:
        warnings.append(
            "the tape carries no reporting_date column; the scope is one "
            "governed snapshot"
            + (f" (as of {as_of})" if as_of else ""))
        return as_of, None, warnings
    dates = _distinct(frame[REPORTING_DATE_FIELD])
    if len(dates) > 1:
        return None, (
            f"the scope spans multiple reporting dates ({', '.join(dates)}); "
            "concentration is measured at one reporting date"), warnings
    return (dates[0] if dates else as_of), None, warnings


def _declared_asset_class(frame: pd.DataFrame) -> Optional[str]:
    if ASSET_CLASS_FIELD not in frame.columns:
        return None
    classes = tuple(v.lower() for v in _distinct(frame[ASSET_CLASS_FIELD]))
    return classes[0] if len(classes) == 1 else None


def _portfolio_count(frame: pd.DataFrame) -> int:
    """Distinct governed portfolios actually present in the scoped frame."""
    if _lens_mod.SOURCE_ID_FIELD not in frame.columns:
        return 1
    return max(1, len(_distinct(frame[_lens_mod.SOURCE_ID_FIELD])))


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
        "portfolio_scope": None,
        "reporting_date": as_of,
        "concentration_basis": None,
        "dimension_results": [],
        "single_name_results": [],
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


def _money(value: Optional[float]) -> str:
    """A monetary amount in the request's resolved currency.

    Imported lazily: this workflow is importable without the API layer, and a
    formatting helper must never become a hard dependency of the calculation.
    """
    if value is None:
        return "n/a"
    try:
        from mi_agent_api import currency as currency_mod

        return currency_mod.format_money(float(value), suffixes=("bn", "m", "k"))
    except Exception:  # noqa: BLE001 - presentation must not break a figure
        return f"{float(value):,.2f}"


def _share_pct(value: Optional[float]) -> str:
    """A share as a percentage, at enough precision to stay truthful.

    One decimal place renders the largest single loan in an 11,035-loan book as
    "0.0% of exposure" — a real, material concentration figure displayed as
    zero. A share that is small is not a share that is absent, and rounding it
    away is the kind of quiet misstatement a reader cannot detect.

    So the precision follows the magnitude: percentages at or above 1% keep the
    familiar single decimal, and smaller ones widen until two significant
    figures survive. Only a genuine zero prints as "0.0%".
    """
    if value is None:
        return "n/a"
    pct = float(value) * 100.0
    if pct == 0:
        return "0.0%"
    magnitude = abs(pct)
    if magnitude >= 1:
        return f"{pct:.1f}%"
    if magnitude >= 0.1:
        return f"{pct:.2f}%"
    if magnitude >= 0.01:
        return f"{pct:.3f}%"
    return f"{pct:.2g}%"


@dataclass(frozen=True)
class ConcentrationReading:
    """What THIS WORKFLOW reads from a question, read once and carried.

    Both fields name framings the governed contract does not carry a concept
    for — an analytical concept in this workflow's own vocabulary, and whether
    the question asks about single names rather than a dimension. They are
    genuinely specialist, so they stay this module's to read; what changes is
    WHEN. `read_question` is called by the RECOGNISER, before the route claims
    the question, and the result travels to the handler. Nothing downstream of
    the claim reads the sentence.
    """

    concept: Optional[str] = None
    single_name_kind: Optional[str] = None


def read_question(question: str) -> ConcentrationReading:
    """This workflow's pre-claim reading of one question."""
    return ConcentrationReading(concept=requested_concept(question),
                                single_name_kind=requested_single_name_kind(question))


def run_concentration_analysis(
        df: pd.DataFrame, *,
        question: str,
        bsr: BusinessSemanticsRegistry,
        mi_semantics: Optional[Mapping[str, Any]] = None,
        registry: Optional[PortfolioRegistry] = None,
        client_id: Optional[str] = None,
        as_of: Optional[str] = None,
        spec: Any = None,
        parse_meta: Optional[Mapping[str, Any]] = None,
        context_id: Optional[str] = None,
        reading: Optional["ConcentrationReading"] = None) -> Dict[str, Any]:
    """Run the workflow over one governed frame. Returns the result contract.

    ``context_id`` is the pre-resolved workspace scope. ``reading`` is this
    question's PRE-CLAIM reading — the concept and the single-name kind, read
    once by the recogniser before the route was claimed and carried in.

    THE QUESTION IS NO LONGER INTERPRETED HERE. It used to be: this function
    called `requested_concept(question)` and `requested_single_name_kind(
    question)` after the route had already claimed the question, and resolved a
    portfolio lens from the wording when no context arrived. Three semantic
    decisions taken downstream of interpretation, in a workflow with eleven
    vocabularies of its own. The deterministic concentration calculation below
    is unchanged and stays here — it is genuinely specialist. Reading the
    sentence is not.

    Never raises for an analytical reason: every non-answer is a controlled
    result with ``available: False`` and an explanation.
    """
    if reading is None:
        # NO READING, NO ANSWER FROM THIS WORKFLOW. Falling back to
        # `read_question(question)` here would leave the interpreter reachable
        # exactly when the caller forgot to supply one — one owner, or none,
        # the rule Conversion 1 set for the population.
        return _controlled_failure(
            "the question was not read before this workflow ran",
            question=question, bsr=bsr, as_of=as_of)
    reg = registry if registry is not None else registry_for_frame(df, client_id=client_id)

    # ---- scope: ONE governed portfolio scope ------------------------------ #
    # THE SCOPE ARRIVES RESOLVED. The fallback that read it from the wording is
    # gone: a second population owner reachable exactly when the first one
    # failed is the worst moment for two owners to disagree.
    requested_context = context_id
    scope = resolve_scope(reg, requested_context)
    if scope.fell_back_to_total and scope.requested_context_id:
        available = ", ".join(reg.ids()) or "none"
        return _controlled_failure(
            f"'{scope.requested_context_id}' is not a governed portfolio scope "
            f"in this deployment; governed portfolios: {available}",
            question=question, bsr=bsr, as_of=as_of)
    frame = apply_scope(df, scope)
    if frame is None or not len(frame):
        return _controlled_failure(
            f"no rows in scope for '{scope.label}' on the governed dataset",
            question=question, bsr=bsr, as_of=as_of)

    reporting_date, failure, warnings = resolve_single_reporting_date(frame, as_of)
    if failure:
        return _controlled_failure(failure, question=question, bsr=bsr,
                                   as_of=as_of)

    limitations: List[str] = []

    # ---- concentration basis: the mixed-currency guard, no FX ------------- #
    currencies = engine.currency_profile(frame)
    exposure_ok, currency_reason = engine.mixed_currency_guard(currencies,
                                                               currencies)
    if not exposure_ok:
        limitations.append(
            f"{currency_reason}; exposure concentration suppressed, count "
            "concentration continues")
    elif not currencies:
        warnings.append("the tape declares no currency column; the platform's "
                        "single-currency book assumption applies")
    if EXPOSURE_FIELD not in frame.columns:
        if exposure_ok:
            limitations.append(
                f"the tape carries no {EXPOSURE_FIELD} column; count "
                "concentration only")
        exposure_ok = False
    basis = engine.BASIS_EXPOSURE if exposure_ok else engine.BASIS_COUNT
    exposure_field = EXPOSURE_FIELD if exposure_ok else None

    # ---- dimension selection: registry-governed --------------------------- #
    declared_asset = _declared_asset_class(frame)
    multi_portfolio = _portfolio_count(frame) > 1
    columns = tuple(df.columns)
    top_n = getattr(spec, "top_n", None) if spec is not None else None

    def _decide(entry: SemanticEntry) -> DimensionDecision:
        return _dimension_decision(entry, declared_asset=declared_asset,
                                   multi_portfolio_scope=multi_portfolio,
                                   columns=columns)

    single_name_kind = reading.single_name_kind

    #: The parse's dimension counts only when the user actually named it —
    #: the deterministic parser substitutes a default dimension for generic
    #: concentration questions, and a default must not masquerade as a request.
    parsed_dimension = getattr(spec, "dimension", None) if spec is not None else None
    explicit_dimension = bool((parse_meta or {}).get("explicit_dimension_requested")) \
        and not (parse_meta or {}).get("dimension_substituted")
    if parse_meta is None:
        explicit_dimension = parsed_dimension is not None

    decisions: List[DimensionDecision] = []
    mode: str
    if single_name_kind is not None:
        mode = "single_name"
    elif explicit_dimension and parsed_dimension and bsr.get(parsed_dimension) is not None:
        mode = "requested_dimension"
        requested_entry = bsr.get(parsed_dimension)
        decision = _decide(requested_entry)
        decisions = [decision]
        if not decision.selected:
            # The named dimension is governed but unusable here: fall back to
            # the other governed dimensions of the SAME concept, with the
            # exclusion recorded — never a silent substitution.
            limitations.append(
                f"'{parsed_dimension}' cannot be analysed here: "
                f"{decision.reason}")
            decisions.extend(_decide(e) for e in _concentration_entries(bsr)
                             if e.analytical_concept == requested_entry.analytical_concept
                             and e.source_field != parsed_dimension)
    else:
        concept = reading.concept
        if explicit_dimension and parsed_dimension and bsr.get(parsed_dimension) is None:
            limitations.append(
                f"'{parsed_dimension}' is not governed by the Business "
                "Semantics Registry as a concentration dimension")
        if concept is not None:
            mode = "requested_concept"
            decisions = [_decide(e) for e in _concentration_entries(bsr)
                         if e.analytical_concept == concept]
            if not decisions:
                limitations.append(
                    f"no dimensions are governed for concentration under the "
                    f"concept '{concept}'")
        else:
            mode = "overview"
            decisions = _bounded([_decide(e) for e in _concentration_entries(bsr)],
                                 OVERVIEW_DIMENSIONS_PER_CONCEPT)

    # ---- calculations: the shared engine only ----------------------------- #
    dimension_results: List[Dict[str, Any]] = []
    evidence: List[Dict[str, Any]] = []
    selected = [d for d in decisions if d.selected]
    for decision in selected:
        result = _dimension_result(decision.entry, frame, basis=basis,
                                   exposure_field=exposure_field,
                                   top_n=top_n, limitations=limitations)
        if result is not None:
            dimension_results.append(result)
            evidence.append(_dimension_evidence(decision.entry, result))

    single_name_results: List[Dict[str, Any]] = []
    identifiers_considered: List[Dict[str, Any]] = []
    if single_name_kind is not None:
        result, considered = _single_name_result(
            single_name_kind, frame, basis=basis,
            exposure_field=exposure_field, top_n=top_n,
            limitations=limitations)
        identifiers_considered = considered
        if result is not None:
            single_name_results.append(result)
            evidence.append(_single_name_evidence(result))

    if not dimension_results and not single_name_results:
        detail = "; ".join(limitations) if limitations else \
            "no governed concentration dimensions are available on this tape"
        failure = _controlled_failure(
            f"concentration cannot be measured for this scope: {detail}",
            question=question, bsr=bsr, as_of=as_of)
        failure["audit"]["dimensions_considered"] = sorted(
            {d.entry.source_field for d in decisions})
        failure["audit"]["dimensions_excluded"] = sorted(
            (d.to_audit() for d in decisions if not d.selected),
            key=lambda item: item["field"])
        return failure

    summary = _summary(scope, dimension_results, single_name_results,
                       basis=basis, top_n=top_n)

    audit = {
        "workflow_id": WORKFLOW_ID,
        "calculation_version": engine.CALCULATION_VERSION,
        "bsr_version": bsr.version,
        "bsr_schema_version": bsr.schema_version,
        "reporting_date": reporting_date,
        "dataset": DATASET,
        "question": question,
        "mode": mode,
        "portfolio_scope": scope.to_dict(),
        "concentration_basis": basis,
        "top_n": top_n,
        "currencies": list(currencies),
        "exposure_concentration_allowed": basis == engine.BASIS_EXPOSURE,
        "dimensions_considered": sorted({d.entry.source_field for d in decisions}),
        "dimensions_selected": [r["field"] for r in dimension_results],
        "dimensions_excluded": sorted(
            (d.to_audit() for d in decisions if not d.selected),
            key=lambda item: item["field"]),
        "single_name_identifiers": identifiers_considered,
        "denominators": {r["field"]: {"count": r["population"],
                                      "exposure": r["total_exposure"]}
                         for r in dimension_results + single_name_results},
        "outcome": "success",
    }

    return {
        "workflow_id": WORKFLOW_ID,
        "available": True,
        "reason": None,
        "portfolio_scope": {
            "mode": mode,
            "dataset": DATASET,
            **scope.to_dict(),
            "row_count": int(len(frame)),
            "currencies": list(currencies),
        },
        "reporting_date": reporting_date,
        "concentration_basis": basis,
        "dimension_results": dimension_results,
        "single_name_results": single_name_results,
        "warnings": warnings,
        "limitations": limitations,
        "summary": summary,
        "evidence": evidence,
        "audit": audit,
    }


# --------------------------------------------------------------------------- #
# Per-dimension / single-name results — shared engine only
# --------------------------------------------------------------------------- #
def _ranked(frame: pd.DataFrame, field: str, *, basis: str,
            exposure_field: Optional[str],
            limitations: List[str], label: str) -> Optional[engine.RankedDistribution]:
    """One governed ranking, falling back to the count basis with disclosure
    when the exposure basis is requested but carries no positive exposure."""
    ranked = engine.ranked_distribution(frame, field,
                                        exposure_field=exposure_field,
                                        basis=basis)
    if ranked is None and basis == engine.BASIS_EXPOSURE:
        note = (f"no positive exposure to rank {label} by; count "
                "concentration reported instead")
        if note not in limitations:
            limitations.append(note)
        ranked = engine.ranked_distribution(frame, field, exposure_field=None,
                                            basis=engine.BASIS_COUNT)
    return ranked


def _top_shares(ranked: engine.RankedDistribution,
                top_n: Optional[int]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "top_5": ranked.top_share(5),
        "top_10": ranked.top_share(10),
    }
    if top_n:
        out[f"top_{int(top_n)}"] = ranked.top_share(int(top_n))
    return out


def _dimension_result(entry: SemanticEntry, frame: pd.DataFrame, *, basis: str,
                      exposure_field: Optional[str], top_n: Optional[int],
                      limitations: List[str]) -> Optional[Dict[str, Any]]:
    ranked = _ranked(frame, entry.source_field, basis=basis,
                     exposure_field=exposure_field, limitations=limitations,
                     label=f"'{entry.display_name}'")
    if ranked is None:
        return None
    return {
        "field": entry.source_field,
        "display_name": entry.display_name,
        "concept": entry.analytical_concept,
        "confidence": entry.confidence,
        "basis": ranked.basis,
        "population": ranked.population,
        "total_exposure": ranked.total_exposure,
        "category_count": len(ranked.buckets),
        "categories": [b.to_dict() for b in ranked.buckets],
        "top_shares": _top_shares(ranked, top_n),
        "unknown": {
            "count": ranked.unknown_count,
            "count_share": ranked.unknown_count_share,
            "exposure": ranked.unknown_exposure,
            "exposure_share": ranked.unknown_exposure_share,
        },
    }


def _single_name_result(kind: str, frame: pd.DataFrame, *, basis: str,
                        exposure_field: Optional[str], top_n: Optional[int],
                        limitations: List[str]
                        ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """``(result, identifiers_considered)`` for one single-name kind.

    Uses only the governed identifier fields declared for the kind — the first
    present on the tape wins, in declared precedence order. No heuristic
    grouping, no connected-counterparty logic.
    """
    considered: List[Dict[str, Any]] = []
    candidates: Tuple[str, ...] = ()
    for name, fields in SINGLE_NAME_IDENTIFIERS:
        if name == kind:
            candidates = fields
            break
    identifier: Optional[str] = None
    for field in candidates:
        present = field in frame.columns
        considered.append({"kind": kind, "field": field, "present": present,
                           "selected": present and identifier is None})
        if present and identifier is None:
            identifier = field
    if identifier is None:
        limitations.append(
            f"no governed {kind} identifier is present on the tape "
            f"(looked for: {', '.join(candidates)}); single-name "
            "concentration is not measured")
        return None, considered

    ranked = _ranked(frame, identifier, basis=basis,
                     exposure_field=exposure_field, limitations=limitations,
                     label=f"{kind} identifiers")
    if ranked is None:
        return None, considered

    listing = int(top_n) if top_n else DEFAULT_SINGLE_NAME_LISTING
    listed = ranked.buckets[:listing]
    remainder = ranked.buckets[listing:]
    remainder_share = None
    if listed and listed[-1].cumulative_share is not None:
        remainder_share = max(0.0, (1.0
                                    - listed[-1].cumulative_share
                                    - ((ranked.unknown_exposure_share
                                        if ranked.basis == engine.BASIS_EXPOSURE
                                        else ranked.unknown_count_share) or 0.0)))
    return {
        "kind": kind,
        "field": identifier,
        "basis": ranked.basis,
        "population": ranked.population,
        "total_exposure": ranked.total_exposure,
        "distinct_names": len(ranked.buckets),
        "listed": len(listed),
        "categories": [b.to_dict() for b in listed],
        "remainder": {"names": len(remainder),
                      "share": remainder_share},
        "top_shares": _top_shares(ranked, top_n),
        "unknown": {
            "count": ranked.unknown_count,
            "count_share": ranked.unknown_count_share,
            "exposure": ranked.unknown_exposure,
            "exposure_share": ranked.unknown_exposure_share,
        },
    }, considered


def _dimension_evidence(entry: SemanticEntry,
                        result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "kind": "concentration_population",
        "field": entry.source_field,
        "basis": result["basis"],
        "population": result["population"],
        "total_exposure": result["total_exposure"],
        "category_count": result["category_count"],
        "unknown_count": result["unknown"]["count"],
        "source": f"{DATASET} canonical tape",
    }


def _single_name_evidence(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "kind": "single_name_population",
        "field": result["field"],
        "basis": result["basis"],
        "population": result["population"],
        "total_exposure": result["total_exposure"],
        "distinct_names": result["distinct_names"],
        "listed": result["listed"],
        "source": f"{DATASET} canonical tape",
    }


# --------------------------------------------------------------------------- #
# Summary — deterministic, observational, never a judgement
# --------------------------------------------------------------------------- #
def _basis_noun(basis: str) -> str:
    return "exposure" if basis == engine.BASIS_EXPOSURE else "loans"


def _summary(scope: PortfolioScope, dimension_results: List[Dict[str, Any]],
             single_name_results: List[Dict[str, Any]], *, basis: str,
             top_n: Optional[int]) -> List[str]:
    """Observational summary lines.

    Each line states an observed share ("largest", "account for") — never a
    judgement ("excessive", "breaches", "insufficiently diversified"). Those
    belong to future covenant / appetite workflows. A broad overview keeps one
    line per dimension so the summary stays readable; the full detail is in
    ``dimension_results``.
    """
    lines: List[str] = []
    verbose = len(dimension_results) <= 3
    for result in dimension_results:
        cats = result["categories"]
        noun = _basis_noun(result["basis"])
        if cats:
            top = cats[0]
            share = (top["exposure_share"]
                     if result["basis"] == engine.BASIS_EXPOSURE
                     else top["count_share"])
            lines.append(f"Largest {result['display_name'].lower()} "
                         f"{'exposure' if noun == 'exposure' else 'category'} is "
                         f"{top['category']} ({_share_pct(share)} of {noun}).")
        if not verbose:
            continue
        n = int(top_n) if top_n else DEFAULT_TOP_N
        top_share = result["top_shares"].get(f"top_{n}")
        if len(cats) > n and top_share is not None:
            lines.append(f"Top {n} {result['display_name'].lower()} categories "
                         f"account for {_share_pct(top_share)} of {noun}.")
        unknown_share = (result["unknown"]["exposure_share"]
                         if result["basis"] == engine.BASIS_EXPOSURE
                         else result["unknown"]["count_share"])
        if result["unknown"]["count"]:
            lines.append(f"Unknown {result['display_name'].lower()} values "
                         f"represent {_share_pct(unknown_share)} of {noun}.")
    for result in single_name_results:
        cats = result["categories"]
        if not cats:
            continue
        noun = _basis_noun(result["basis"])
        top = cats[0]
        share = (top["exposure_share"] if result["basis"] == engine.BASIS_EXPOSURE
                 else top["count_share"])
        if result["kind"] == "loan" and result["basis"] == engine.BASIS_EXPOSURE:
            # The AMOUNT is the answer to "what is the largest single-loan
            # exposure"; the share answers "what share of the book is it". The
            # loan's identifier answers neither, and printing it puts
            # loan-level data in a portfolio answer — the table artifact
            # carries the drill-down for anyone who needs it.
            lines.append(
                f"The largest single-loan exposure is "
                f"{_money(top['exposure'])}, representing {_share_pct(share)} "
                f"of {noun}.")
        else:
            lines.append(f"Largest single {result['kind']} exposure is "
                         f"{top['category']} ({_share_pct(share)} of {noun}).")
        n = int(top_n) if top_n else DEFAULT_TOP_N
        top_share = result["top_shares"].get(f"top_{n}")
        if result["distinct_names"] > n and top_share is not None:
            lines.append(f"Top {n} {result['kind']}s account for "
                         f"{_share_pct(top_share)} of {noun}.")
    return lines
