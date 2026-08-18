#!/usr/bin/env python3
"""mi_agent/execution_receipt.py — P0 launch hardening.

Two jobs, both deterministic and both derived from the EXECUTED state:

1. **The execution receipt.** A compact, machine-derived statement of what was
   actually calculated — measure, aggregation, filters that really narrowed the
   frame, groupings that really grouped, population, period, scenario. Its
   purpose is to make a semantic scope error visible to a reader who cannot see
   the spec.

2. **The semantic-completeness guard.** The repository already fails closed on
   dimensions and filters that were attached to a *spec* and then lost in
   execution (:mod:`mi_agent.mi_query_contract`). That contract cannot see intent
   the parser dropped BEFORE the spec was built — which is precisely how
   "average LTV in London" came to be answered with the whole book. This module
   re-derives the material facets from the QUESTION and reconciles them against
   execution, so a facet can only end in one of:

       APPLIED · UNAVAILABLE · UNSUPPORTED · REJECTED · LOST

   ``LOST`` is the fail-closed state: the user asked for something material, it
   did not reach execution, and no reason is known. The caller must refuse
   rather than present a broader calculation as though it were the narrow one.

Design constraints for launch:

* **Conservative detection.** A false positive turns a good answer into a
  refusal, so every detector requires an explicit textual marker and, for
  geography, a value that genuinely exists in the governed dimension.
* **Evidence, not prose.** A facet is APPLIED only on structural evidence
  (execution narrowed the frame, the group column is in the result, the route
  declared the period). Answer text is never evidence.
* **Pure.** No I/O, no LLM, no pandas mutation — trivially unit-testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
KIND_GEOGRAPHIC_SCOPE = "geographic_scope"
KIND_THRESHOLD = "threshold"
KIND_GROUPING = "grouping_dimension"
KIND_STRESS = "stress_scenario"
KIND_COMPARISON_PERIOD = "comparison_period"
KIND_RANKING = "ranking"
#: A forward-looking question ("what will the balance be at year end").
KIND_PROJECTION = "projection"
#: A comparison between two named cohorts ("the direct book vs the acquired book").
KIND_COHORT_COMPARISON = "cohort_comparison"
#: More measures named than a single-metric spec can carry.
KIND_MULTI_MEASURE = "multi_measure"
#: "What PROPORTION of the book …". A share is a governed aggregation needing
#: TWO populations — the filtered numerator and the whole-book denominator — so
#: the absolute figures for the numerator alone answer a different question.
#: Without this facet a share request that lost its aggregation came back as a
#: confident pair of absolute numbers with no proportion anywhere in it.
KIND_SHARE = "share"
#: A slot in the question's measure list that named NO governed measure. The
#: parser understood the other slots, so answering them and saying nothing about
#: this one would be a silent 3-of-4 — the one outcome the product must never
#: produce. Structural, not lexical: the guard does not need to know what the
#: unrecognised words mean, only that a coordinated slot resolved to nothing,
#: which is why it also covers vocabulary the parser has not learnt yet.
KIND_UNRESOLVED_MEASURE = "unresolved_measure"
#: One measure asked for RELATIVE TO another ("bigger loans relative to their
#: property value") — a relationship, which a single aggregate cannot express.
KIND_RELATIONSHIP = "relationship"
#: A group's CONTRIBUTION to a portfolio weighted aggregate ("which region
#: contributes most to the weighted average LTV"). Ranking groups by their own
#: value answers a different question — on the demonstration book the two
#: rankings are near-inverted — so a contribution question that reaches a plain
#: per-group ranking must refuse rather than present it.
KIND_CONTRIBUTION = "aggregate_contribution"

#: A requested facet reached execution and demonstrably shaped the result.
APPLIED = "applied"
#: The dataset does not carry the field the facet needs. Disclosable.
UNAVAILABLE = "unavailable"
#: No governed capability expresses this facet. Disclosable.
UNSUPPORTED = "unsupported"
#: Execution considered and explicitly declined it, with a reason. Disclosable.
REJECTED = "rejected"
#: Requested, material, and absent from execution with no reason — FAIL CLOSED.
LOST = "lost"

#: Statuses that permit an answer to stand, provided they are disclosed.
DISCLOSABLE = (UNAVAILABLE, UNSUPPORTED, REJECTED)


@dataclass
class RequestedFacet:
    """One material thing the user asked for, and what became of it."""

    kind: str
    #: Short human label used in the receipt ("London", "youngest borrower age > 85").
    label: str
    #: The semantic field key this facet bears on, where one is known.
    field_key: Optional[str] = None
    #: Other keys that legitimately satisfy this facet. A generic term resolves
    #: to different concrete fields depending on what the dataset carries
    #: ("region" -> the readable geography, or the NUTS3 code field), and every
    #: such resolution is the SAME request being honoured — not a substitution.
    alt_keys: Tuple[str, ...] = ()
    #: For a multi-measure facet: the measure CONCEPTS the question named, so
    #: reconciliation can check the whole set rather than a single label.
    concepts: Tuple[str, ...] = ()
    status: str = LOST
    reason: str = ""

    def satisfied_by(self) -> Tuple[str, ...]:
        """Every field key that counts as honouring this facet."""
        keys = [k for k in (self.field_key,) + tuple(self.alt_keys) if k]
        return tuple(dict.fromkeys(keys))

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "label": self.label, "field": self.field_key,
                "status": self.status, "reason": self.reason}

    def disclosure(self) -> str:
        """The user-facing line for a facet that did not apply."""
        if self.reason:
            return f"{self.label} — {self.reason}"
        return self.label


# --------------------------------------------------------------------------- #
# Detection — from the QUESTION, before any parsing decision
# --------------------------------------------------------------------------- #
#: Words that look like place names but are structural English. A dimension value
#: matching one of these is never treated as a requested geographic scope.
_GEO_STOPWORDS = {
    "total", "other", "unknown", "none", "all", "active", "current", "book",
    "loan", "loans", "value", "type", "types", "region", "regions", "missing",
}

#: Comparison markers that, together with a number, mean the user stated a
#: threshold. Deliberately explicit — a bare number is never a threshold.
_THRESHOLD_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"\b(?:over|above|more than|greater than|exceeding|in excess of)\s+£?\s*(\d[\d,\.]*)\s*(%|percent)?", "over"),
    (r"\b(?:under|below|less than|fewer than|beneath|younger than)\s+£?\s*(\d[\d,\.]*)\s*(%|percent)?", "under"),
    (r"\bolder than\s+£?\s*(\d[\d,\.]*)\s*(%|percent)?", "over"),
    (r"\b(?:at least|no less than|minimum of)\s+£?\s*(\d[\d,\.]*)\s*(%|percent)?", "at least"),
    (r"\b(?:at most|no more than|up to|maximum of|capped at)\s+£?\s*(\d[\d,\.]*)\s*(%|percent)?", "at most"),
    (r"\bbetween\s+£?\s*(\d[\d,\.]*)\s*%?\s+and\s+£?\s*\d[\d,\.]*", "between"),
    (r"(\d[\d,\.]*)\s*(?:\+|\s+or (?:above|over|older|more|greater))\b", "or above"),
    (r"[<>]=?\s*£?\s*(\d[\d,\.]*)\s*(%|percent)?", "comparison"),
)

#: "a 75% LTV cap", "eligible for a 75% LTV ..." — a percentage bound stated
#: directly against a percent measure, without a comparator word.
_PCT_BOUND_RE = re.compile(
    r"(\d[\d\.]*)\s*%\s*(?:current\s+)?(ltv|loan[- ]to[- ]value)\b", re.I)

#: Hypothetical / scenario markers. Require a directional verb so a forecast
#: ("if origination continues") is not mistaken for a stress.
_STRESS_PATTERNS: Tuple[str, ...] = (
    r"\bif\b[^.?]{0,60}\b(fell|falls|fall|drop|drops|dropped|decline|declines|declined|"
    r"rose|rise|rises|increase[sd]?|crash(?:ed|es)?)\b",
    r"\bwhat would\b[^.?]{0,60}\b(fall|drop|decline|rise|increase|shock|stress)\b",
    r"\b(\d[\d\.]*)\s*%\s*(?:house price|property|hpi|property value|valuation)\b",
    r"\b(?:house price|property|hpi)\s+(?:fall|drop|decline|shock|stress|crash)\b",
    r"\bstress(?:ed|ing)?\b(?!\s*test\s*(?:id|name))",
    r"\bdownside scenario\b|\bscenario analysis\b",
)

#: Explicit ranking intent: an interrogative naming a dimension plus a superlative.
_RANKING_RE = re.compile(
    r"\b(which|what)\b[^?]{0,80}?\b(has|have|is|are|was|were)?\s*"
    r"(?:the\s+)?(most|highest|largest|biggest|greatest|fastest|top|worst|lowest|smallest)\b",
    re.I)
_TOP_N_RE = re.compile(r"\btop\s+(\d+)\b", re.I)

#: Explicit period-comparison markers.
_COMPARISON_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"\bsince inception\b", "since inception"),
    (r"\blast quarter\b|\bprevious quarter\b|\bprior quarter\b", "last quarter"),
    (r"\bthis quarter\b", "this quarter"),
    (r"\blast month\b|\bprevious month\b|\bprior month\b", "last month"),
    (r"\blast year\b|\bprevious year\b|\bprior year\b", "last year"),
    (r"\b(?:versus|vs\.?|compared with|compared to|against)\s+(?:the\s+)?"
     r"(?:prior|previous|last)\b", "versus the prior period"),
    (r"\byear[- ]on[- ]year\b|\bquarter[- ]on[- ]quarter\b|\bmonth[- ]on[- ]month\b",
     "period on period"),
    (r"\bhas\s+(?:it\s+)?(?:changed|grown|moved|shifted)\b|\bhow has\b.{0,40}\bchanged\b",
     "change over time"),
    (r"\bgrow(?:n|th|ing)?\b|\bshift(?:ed|s)?\b|\bmovement\b", "change over time"),
    # "converging" / "diverging" are comparisons ACROSS TIME, not point-in-time
    # differences: answering one from a single snapshot answers a different
    # question.
    (r"\bconverg(?:e|es|ed|ing|ence)\b|\bdiverg(?:e|es|ed|ing|ence)\b",
     "convergence over time"),
    (r"\brelative to (?:the\s+)?(?:last|prior|previous)\b", "versus the prior period"),
    (r"\bover the (?:last|past)\s+(?:\w+\s+)?(?:month|quarter|year|months|quarters|years)\b",
     "over the stated period"),
)

#: Routes whose whole purpose IS a period comparison. A comparison facet answered
#: by one of these is applied; anything else has not compared periods.
TEMPORAL_ROUTES = frozenset({
    "period_change_analysis", "temporal_compare", "evolution", "evolution_funnel",
    "evolution_pipeline_stage", "funded_bridge", "cohort_progression",
    "cohort_conversion", "forecast_extrapolation", "scenario",
    # ``period_movement`` reads the current AND prior governed reporting periods
    # and reports the delta between them (mi_agent.movement.period_movement).
    # Its absence here refused "what changed since last month?" as a
    # point-in-time answer — a false refusal of a genuinely two-period
    # capability, which disables working governed analytics rather than
    # preventing a substitution.
    "period_movement",
})

#: Routes that genuinely rank a dimension.
RANKING_ROUTES = frozenset({
    "concentration_analysis", "geo_exposure", "risk_limits", "funded_bridge",
})


def geographic_values(frame, semantics: dict, *, max_cardinality: int = 60
                      ) -> Dict[str, str]:
    """``{lowercased value: semantic field key}`` for READABLE geography columns.

    Only low-cardinality geographic dimensions are scanned: those are where the
    names a person actually types ("London", "South East", "Wales") live. A
    high-cardinality code column (ITL3, postcode) is skipped so a stray token can
    never be mistaken for a requested scope.
    """
    out: Dict[str, str] = {}
    if frame is None or not hasattr(frame, "columns"):
        return out
    fields = semantics.get("fields", {}) if isinstance(semantics, dict) else {}
    for key, entry in fields.items():
        if (entry or {}).get("role") != "dimension":
            continue
        blob = f"{key} {(entry or {}).get('canonical_field', '')}".lower()
        if not ("geograph" in blob or "region" in blob or "collateral_geo" in blob):
            continue
        column = (entry or {}).get("canonical_field", key)
        if column not in frame.columns:
            continue
        try:
            values = frame[column].dropna().unique()
        except Exception:  # noqa: BLE001 - profiling must never break a query
            continue
        if len(values) > max_cardinality:
            continue
        for value in values:
            token = str(value).strip()
            if len(token) < 4 or token.lower() in _GEO_STOPWORDS:
                continue
            out.setdefault(token.lower(), key)
    return out


def _detect_geographic_scope(q: str, geo_values: Dict[str, str]) -> List[RequestedFacet]:
    found: List[RequestedFacet] = []
    seen: Set[str] = set()
    # Longest first, so "South East" wins over any shorter contained value.
    for value in sorted(geo_values, key=len, reverse=True):
        if value in seen:
            continue
        if re.search(r"\b" + re.escape(value) + r"\b", q):
            if any(value in other for other in seen):
                continue
            seen.add(value)
            found.append(RequestedFacet(
                kind=KIND_GEOGRAPHIC_SCOPE, label=value.title(),
                field_key=geo_values[value]))
    return found


def _detect_thresholds(q: str) -> List[RequestedFacet]:
    found: List[RequestedFacet] = []
    for pattern, word in _THRESHOLD_PATTERNS:
        for match in re.finditer(pattern, q, re.I):
            number = match.group(1)
            span = q[max(0, match.start() - 42):match.end() + 18]
            subject = _threshold_subject(span)
            label = (f"{subject} {word} {number}" if subject
                     else f"{word} {number}").strip()
            found.append(RequestedFacet(kind=KIND_THRESHOLD, label=label))
    numbers_seen = {re.sub(r"[^\d.]", "", f.label.split()[-1]) for f in found}
    for match in _PCT_BOUND_RE.finditer(q):
        # "above 60% LTV" already produced a comparator threshold for 60; the
        # percent-bound form exists for "a 75% LTV cap", where no comparator was
        # written. Counting both would demand two filters for one predicate.
        if match.group(1) in numbers_seen:
            continue
        found.append(RequestedFacet(
            kind=KIND_THRESHOLD, label=f"LTV bound of {match.group(1)}%",
            field_key="current_loan_to_value"))
    # De-duplicate by label, preserving order.
    unique: List[RequestedFacet] = []
    labels: Set[str] = set()
    for facet in found:
        if facet.label.lower() not in labels:
            labels.add(facet.label.lower())
            unique.append(facet)
    return unique


_THRESHOLD_SUBJECTS: Tuple[Tuple[str, str], ...] = (
    (r"\bltv\b|\bloan[- ]to[- ]value\b", "LTV"),
    (r"\bage[ds]?\b|\bborrower[s]? (?:aged|age)\b|\bborrowers\b|\byears? old\b", "borrower age"),
    (r"\bbalance\b|\bexposure\b|\bloan size\b|\bticket\b", "balance"),
    (r"\brate\b|\bcoupon\b|\binterest\b", "interest rate"),
    (r"\bvaluation\b|\bproperty value\b|\bcollateral\b", "valuation"),
)


def _threshold_subject(span: str) -> str:
    for pattern, name in _THRESHOLD_SUBJECTS:
        if re.search(pattern, span, re.I):
            return name
    return ""


def _detect_stress(q: str) -> List[RequestedFacet]:
    for pattern in _STRESS_PATTERNS:
        match = re.search(pattern, q, re.I)
        if match:
            phrase = match.group(0).strip()
            return [RequestedFacet(
                kind=KIND_STRESS,
                label=f"stress/scenario condition ({phrase})")]
    return []


def _detect_comparison_period(q: str) -> List[RequestedFacet]:
    for pattern, label in _COMPARISON_PATTERNS:
        if re.search(pattern, q, re.I):
            return [RequestedFacet(kind=KIND_COMPARISON_PERIOD,
                                   label=f"comparison period ({label})")]
    return []


#: P1D — a question about a group's CONTRIBUTION to a portfolio weighted
#: aggregate. Both halves are required: contribution language AND a weighted
#: aggregate as its object. "Which region has the highest LTV?" has neither and
#: is untouched; "which region contributes most to the balance?" has the first
#: but not the second, and a contribution to a plain sum is just its share.
_CONTRIBUTION_RE = re.compile(
    r"\b(?:contributes?|contributing|contributed)\s+(?:the\s+)?most\b|"
    r"\b(?:biggest|largest|greatest|main|primary|top)\s+contributors?\b|"
    r"\bcontributions?\s+to\b|"
    r"\bdriv(?:es|ing|en)\s+(?:the\s+)?most\s+of\b|"
    r"\b(?:accounts?|accounting)\s+for\s+(?:the\s+)?most\s+of\b", re.I)
_WEIGHTED_OBJECT_RE = re.compile(
    r"\bweighted[\s-]?(?:average|avg|mean)\b|\bwa\s+(?:ltv|rate|yield)\b|"
    r"\bportfolio\s+(?:ltv|loan[\s-]?to[\s-]?value|interest\s+rate)\b|"
    r"\baverage\s+(?:ltv|loan[\s-]?to[\s-]?value|interest\s+rate)\b", re.I)


def _detect_contribution(q: str) -> List[RequestedFacet]:
    if not (_CONTRIBUTION_RE.search(q) and _WEIGHTED_OBJECT_RE.search(q)):
        return []
    return [RequestedFacet(
        kind=KIND_CONTRIBUTION,
        label="contribution to the portfolio weighted average")]


def _detect_ranking(q: str, requested_dimensions: Sequence[Tuple[str, str, Tuple[str, ...]]]
                    ) -> List[RequestedFacet]:
    """A ranking facet needs BOTH a superlative interrogative and a dimension to
    rank. "Which region has grown the most" qualifies; "what is the average LTV"
    does not."""
    top_n = _TOP_N_RE.search(q)
    if not (_RANKING_RE.search(q) or top_n):
        return []
    if not requested_dimensions:
        return []
    key, term, alts = requested_dimensions[0]
    label = f"ranking by {term}"
    if top_n:
        label = f"top {top_n.group(1)} by {term}"
    return [RequestedFacet(kind=KIND_RANKING, label=label, field_key=key,
                           alt_keys=alts)]


#: Measure concepts a question can name, longest phrase first. Mirrors the
#: parser's own metric vocabulary so the guard and the parser agree on what
#: counts as "naming a measure".
#: Phrases containing a measure word that do NOT name that measure. Matched
#: first so their span is consumed: "run rate" is a velocity, not the interest
#: rate, and treating it as one would refuse a correct run-rate answer.
_MEASURE_SKIP = "__skip__"
_MEASURE_TERMS: Tuple[Tuple[str, str], ...] = (
    # Dimension names that CONTAIN a measure word. Consumed first so a
    # cross-tab ("balance by ltv band") is not read as two measures, which
    # would refuse a working grouped answer.
    ("ltv band", _MEASURE_SKIP), ("ltv bucket", _MEASURE_SKIP),
    ("ltv banding", _MEASURE_SKIP), ("ltv range", _MEASURE_SKIP),
    ("age band", _MEASURE_SKIP), ("age bucket", _MEASURE_SKIP),
    ("balance band", _MEASURE_SKIP), ("balance bucket", _MEASURE_SKIP),
    ("ticket band", _MEASURE_SKIP), ("ticket bucket", _MEASURE_SKIP),
    ("rate band", _MEASURE_SKIP), ("rate bucket", _MEASURE_SKIP),
    ("run rate", _MEASURE_SKIP), ("run-rate", _MEASURE_SKIP),
    ("current rate", _MEASURE_SKIP), ("growth rate", _MEASURE_SKIP),
    ("conversion rate", _MEASURE_SKIP), ("completion rate", _MEASURE_SKIP),
    ("weighted average ltv", "ltv"), ("loan to value", "ltv"), ("ltv", "ltv"),
    ("collateral value", "valuation"), ("property value", "valuation"),
    ("valuation", "valuation"),
    ("outstanding balance", "balance"), ("balance", "balance"),
    ("exposure", "balance"),
    ("arrears", "arrears"),
    ("interest rate", "rate"), ("coupon", "rate"),
    ("borrower age", "age"),
)

#: Canonical-key fragments -> the concept that key measures. Used to name the
#: concept the executor ACTUALLY calculated.
_CONCEPT_BY_KEY_FRAGMENT: Tuple[Tuple[str, str], ...] = (
    ("loan_to_value", "ltv"),
    ("arrears", "arrears"),
    ("interest_rate", "rate"),
    ("valuation", "valuation"),
    ("borrower_age", "age"),
    ("age", "age"),
    ("balance", "balance"),
    ("principal", "balance"),
)

#: One measure expressed RELATIVE TO another. The temporal sense of "relative
#: to" ("relative to last month") is excluded — that is a period comparison.
_RELATIONSHIP_RE = re.compile(
    r"\brelative to\b(?!\s+(?:the\s+)?(?:last|prior|previous))|"
    r"\bcompared (?:to|with) their\b|\bas a (?:multiple|proportion) of\b", re.I)

#: Forward-looking phrasing. Answering one of these from a point-in-time
#: snapshot returns today's number for a question about the future.
_PROJECTION_RE = re.compile(
    r"\bwhat will\b|\bwhen will\b|\bwill (?:be|reach|hit|exceed)\b|"
    r"\bby year[- ]end\b|\bat year[- ]end\b|\bproject(?:ed|ion|ions)\b|"
    r"\bforecast\b|\bextrapolat", re.I)

#: A comparison between two named cohorts rather than across time.
_COHORT_COMPARISON_RE = re.compile(
    r"\b(?:direct|acquired|new|back)\s+book\b[^?]{0,60}\b(?:vs\.?|versus|compared with|"
    r"compared to|against|better or worse than)\b|"
    r"\b(?:vs\.?|versus|compared with|compared to|better or worse than)\b[^?]{0,60}"
    r"\b(?:direct|acquired|back)\s+book\b|"
    r"\bhow does the\b[^?]{0,40}\bcompare with\b", re.I)

#: Grouping keys that split the platform into its constituent books. Grouping by
#: one of these IS a cohort comparison, even on the point-in-time path.
_COHORT_DIMENSION_KEYS = ("source_portfolio_type", "source_portfolio_id",
                          "source_portfolio_label", "portfolio_cohort")

# --------------------------------------------------------------------------- #
# Cohort SEMANTIC IDENTITY
# --------------------------------------------------------------------------- #
# A cohort question names a CONCEPT, not merely "two books". "Direct versus
# acquired" asks how the loans were SOURCED; "new origination versus the back
# book" asks how long they have been on the book. Both split the portfolio in
# two, and until this table existed the guard could not tell them apart: any
# grouping by a sourcing key marked any cohort facet applied, so a vintage
# question answered by sourcing channel read as correct.
#
# The concept must survive the same chain every other facet does — requested
# concept, resolved governed field, executed grouping, receipt — and a concept
# whose fields this dataset does not carry must refuse by name rather than be
# satisfied by a different split.
#
# Concept language is curated here rather than read from field synonyms because
# the registry describes FIELDS, and "back book" is not a synonym of any field:
# it is a way of asking about seasoning, which vintage_year expresses.
_COHORT_CONCEPTS: Tuple[Tuple[str, str, Tuple[str, ...], Tuple[str, ...]], ...] = (
    (
        "sourcing",
        "how the loans were sourced",
        (r"\bdirect\b", r"\bacquired\b", r"\bpurchased\b", r"\borganic\b",
         r"\bsourcing\b", r"\bsource portfolio\b", r"\boriginated in[- ]house\b"),
        ("source_portfolio_type", "source_portfolio_id",
         "source_portfolio_label", "portfolio_cohort"),
    ),
    (
        "vintage",
        "how long the loans have been on the book",
        (r"\bnew origination(?:s)?\b", r"\bback book\b", r"\bnew lending\b",
         r"\bnewly originated\b", r"\brecent origination(?:s)?\b",
         r"\bseasoned\b", r"\bseasoning\b", r"\bvintage(?:s)?\b",
         r"\bfront book\b", r"\bnew business\b"),
        # GROUPABLE vintage dimensions only. A raw origination_date is on this
        # tape, but a date is not a cohort: splitting a book by it needs a
        # derived year or seasoning band. P1J-1 governs exactly those —
        # vintage_year, seasoning_bucket (analytical bands) and
        # seasoning_segment (the binary front/back split) — so on a book that
        # carries an origination date the concept now RESOLVES rather than
        # refusing. Where a book carries no origination date the guard still
        # refuses by name, which is why the list stays explicit.
        ("vintage_year", "origination_year", "vintage", "vintage_bucket",
         "seasoning_bucket", "seasoning_segment", "origination_vintage"),
    ),
)


#: Comparison framing. A cohort CONCEPT alone is not a comparison — "the
#: acquired book's LTV" asks about one cohort — so the facet is raised only
#: when the question also sets two things against each other.
_COHORT_COMPARISON_FRAMING_RE = re.compile(
    r"\bvs\.?\b|\bversus\b|\bcompared? (?:with|to|against)\b|\bcompare\b|"
    r"\bbetter or worse than\b|\bagainst\b|\bconverging with\b|"
    r"\b(?:higher|lower|better|worse|bigger|smaller) than\b|"
    r"\bdifference between\b|\bhow does\b[^?]{0,60}\bcompare\b", re.I)


def cohort_concepts_named(question: str) -> List[Tuple[str, str, Tuple[str, ...]]]:
    """``[(concept, description, governed_field_keys)]`` the question names.

    Order follows the table, not the sentence: a question naming both is
    genuinely ambiguous and is reported as both, so neither can be quietly
    dropped in favour of whichever executed.
    """
    q = f" {str(question or '').strip().lower()} "
    out: List[Tuple[str, str, Tuple[str, ...]]] = []
    for concept, description, patterns, fields in _COHORT_CONCEPTS:
        if any(re.search(p, q) for p in patterns):
            out.append((concept, description, fields))
    return out


def _cohort_fields_available(fields: Sequence[str],
                             columns: Optional[Iterable[str]],
                             semantics: dict) -> List[str]:
    """The concept's governed fields this dataset actually carries."""
    available = {str(c) for c in (columns or ())}
    if not available:
        return list(fields)
    registry = semantics.get("fields", {}) if isinstance(semantics, dict) else {}
    present = []
    for key in fields:
        canonical = (registry.get(key, {}) or {}).get("canonical_field", key)
        if key in available or canonical in available:
            present.append(key)
    return present

#: Routes that genuinely compare two cohorts.
COHORT_ROUTES = frozenset({"portfolio_risk_comparison", "cohort_progression",
                           "cohort_conversion"})
#: Routes that genuinely project forward.
PROJECTION_ROUTES = frozenset({"forecast_extrapolation", "scenario"})
#: Routes that genuinely apply a stated scenario / uplift.
SCENARIO_ROUTES = frozenset({"scenario"})


#: Filler that can sit between a measure word and its comparator
#: ("LTV of more than 40%", "balance is above £100k").
_FILTER_FILLER = r"(?:of|is|are|that is|which is|at|with|having)?"
#: A comparator immediately AFTER the measure word.
_FILTER_AFTER_RE = re.compile(
    r"^\s*" + _FILTER_FILLER + r"\s*(?:[<>]=?|=)|"
    r"^\s*" + _FILTER_FILLER + r"\s*\b(?:above|below|over|under|more than|"
    r"less than|greater than|at least|at most|between|exceeding|"
    r"in excess of)\b|"
    r"^\s*\d[\d,.]*\s*(?:%|\+)", re.I)
#: A comparator + number immediately BEFORE the measure word ("above 50% LTV").
_FILTER_BEFORE_RE = re.compile(
    r"\b(?:above|below|over|under|more than|less than|greater than|at least|"
    r"at most|exceeding|in excess of|[<>]=?)\s*£?\s*\d[\d,.]*\s*%?\s*$", re.I)


def _is_filter_subject(q: str, start: int, end: int) -> bool:
    """True when this measure word is the subject of a predicate, not a measure.

    "balance by region where LTV above 50%" measures balance and FILTERS on LTV.
    Counting the filter subject as a second requested measure would refuse a
    perfectly good filtered breakdown, so both sides are checked.
    """
    if _FILTER_AFTER_RE.search(q[end:end + 32]):
        return True
    return bool(_FILTER_BEFORE_RE.search(q[max(0, start - 32):start]))


def named_measure_concepts(question: str) -> List[str]:
    """Distinct measure concepts the question names, in order of appearance."""
    q = (question or "").lower()
    found: List[Tuple[int, str]] = []
    seen: Set[str] = set()
    consumed: List[Tuple[int, int]] = []
    for phrase, concept in _MEASURE_TERMS:
        for match in re.finditer(r"\b" + re.escape(phrase) + r"\b", q):
            # A longer phrase already claimed this span ("loan to value" wins
            # over a bare "value" inside it).
            if any(s <= match.start() < e for s, e in consumed):
                continue
            consumed.append((match.start(), match.end()))
            # "balance BY ltv" names one measure and one grouping, not two
            # measures: a measure word introduced by a grouping preposition is
            # the axis, not the thing being measured.
            preceding = q[max(0, match.start() - 8):match.start()]
            if re.search(r"\b(?:by|per|across|split by|grouped by)\s+$", preceding):
                continue
            if _is_filter_subject(q, match.start(), match.end()):
                continue
            if concept != _MEASURE_SKIP and concept not in seen:
                seen.add(concept)
                found.append((match.start(), concept))
            break
    return [c for _, c in sorted(found)]


def executed_measure_concept(metric_key: Optional[str]) -> Optional[str]:
    """The concept the executed metric measures, from its canonical key."""
    if not metric_key:
        return None
    key = str(metric_key).lower()
    for fragment, concept in _CONCEPT_BY_KEY_FRAGMENT:
        if fragment in key:
            return concept
    return None


def executed_measure_concepts(query_result: Any) -> Set[str]:
    """The measure CONCEPTS execution actually returned.

    Read from the executor's declared measure set (P1E) and, failing that, from
    the executed metric — evidence, never the question. Returns an empty set
    when execution declared no measure set, which keeps the pre-P1E branches
    below in charge for a single-measure result.
    """
    metadata = getattr(query_result, "metadata", None) or {}
    executed = metadata.get("measures_executed") or []
    concepts: Set[str] = set()
    for measure in executed:
        key = (measure or {}).get("canonical_field") or (measure or {}).get("field")
        if key in ("loan_count", "count"):
            concepts.add("count")
            continue
        concept = executed_measure_concept(key)
        if concept:
            concepts.add(concept)
    return concepts


def requested_dimension_terms(question: str, semantics: dict,
                              available_columns: Optional[Iterable[str]] = None
                              ) -> List[Tuple[str, str, Tuple[str, ...]]]:
    """``[(field_key, matched_term, alt_keys)]`` the user explicitly named.

    Resolution runs TWICE. Once without dataset-availability filtering, so a
    dimension the dataset lacks is still recorded as requested and can therefore
    be disclosed rather than silently vanishing — reproducing the original defect
    is the thing this exists to prevent. And once WITH the real columns, because
    a generic term ("region") resolves to different concrete fields depending on
    what the dataset carries, and both resolutions honour the same request.
    ``alt_keys`` carries the second resolution so that difference is never
    mistaken for a substitution.

    A plural form is retried in the singular ("which vintages" -> ``vintage``)
    because the term map is keyed on singulars.
    """
    from .llm_query_parser import _explicit_dimensions  # local: avoids a cycle

    q = (question or "").lower()
    keys, terms, _ = _explicit_dimensions(q, semantics, available_columns=None)
    by_term: Dict[str, List[str]] = {}
    for key, term in zip(keys, terms):
        by_term.setdefault(term, []).append(key)
    if available_columns is not None:
        a_keys, a_terms, _ = _explicit_dimensions(
            q, semantics, available_columns=set(available_columns))
        for key, term in zip(a_keys, a_terms):
            by_term.setdefault(term, []).append(key)
    out: List[Tuple[str, str, Tuple[str, ...]]] = []
    for key, term in zip(keys, terms):
        alts = tuple(dict.fromkeys(k for k in by_term.get(term, []) if k != key))
        out.append((key, term, alts))
    seen = {k for k, _, _ in out}
    # Retry in the singular, accepting a term only when its plural genuinely
    # appeared, so this can never invent a dimension the user did not name.
    singular = re.sub(r"\b(\w{4,})s\b", r"\1", q)
    if singular != q:
        s_keys, s_terms, _ = _explicit_dimensions(singular, semantics,
                                                  available_columns=None)
        for key, term in zip(s_keys, s_terms):
            if key in seen:
                continue
            if re.search(r"\b" + re.escape(term) + r"s\b", q):
                out.append((key, term, ()))
                seen.add(key)
    return out


def _asks_for_a_share(q: str) -> bool:
    """True when the question asks for a PROPORTION rather than an amount.

    Uses the parser's own share detector so the guard and the parser can never
    disagree about which questions are share questions.
    """
    try:
        from .llm_query_parser import _SHARE_RE  # local: avoids a cycle
        return bool(_SHARE_RE.search(q or ""))
    except Exception:  # noqa: BLE001 - the guard must never break an answer
        return False


def _unresolved_measure_slots(q: str, semantics: dict, frame) -> Tuple[str, ...]:
    """Measure slots the parser did not understand. Never raises."""
    try:
        from .llm_query_parser import unresolved_measure_slots  # local: cycle
        columns = (list(frame.columns)
                   if frame is not None and hasattr(frame, "columns") else None)
        return unresolved_measure_slots(q, semantics, columns)
    except Exception:  # noqa: BLE001 - the guard must never break an answer
        return ()


def detect_requested_facets(question: str, semantics: dict, *, frame=None,
                            requested_dimensions: Optional[Sequence[Tuple[str, str]]] = None
                            ) -> List[RequestedFacet]:
    """Every material facet stated in ``question``, before any parsing decision.

    ``requested_dimensions`` is ``[(field_key, matched_term)]`` as recognised
    WITHOUT dataset-availability filtering, so a dimension the dataset lacks is
    still recorded as requested (and can therefore be disclosed rather than
    silently vanishing).
    """
    q = (question or "").lower()
    if not q.strip():
        return []
    facets: List[RequestedFacet] = []
    facets.extend(_detect_stress(q))
    facets.extend(_detect_thresholds(q))
    facets.extend(_detect_geographic_scope(q, geographic_values(frame, semantics)))
    facets.extend(_detect_comparison_period(q))
    facets.extend(_detect_ranking(q, list(requested_dimensions or [])))
    facets.extend(_detect_contribution(q))
    if _RELATIONSHIP_RE.search(q):
        facets.append(RequestedFacet(
            kind=KIND_RELATIONSHIP,
            label="one measure relative to another"))
    if _PROJECTION_RE.search(q):
        facets.append(RequestedFacet(
            kind=KIND_PROJECTION, label="a forward projection"))
    named = cohort_concepts_named(q)
    # The facet is raised either by the original two-books phrasing, or by a
    # cohort CONCEPT set against something else. The second arm matters: "how
    # does new lending compare with the seasoned book" names a cohort and a
    # comparison but matches none of the original patterns, so no facet was
    # raised and the identity check below never ran.
    if _COHORT_COMPARISON_RE.search(q) or (
            named and _COHORT_COMPARISON_FRAMING_RE.search(q)):
        # WHICH cohort, not merely that one was asked for. Without the concept
        # the guard could only check that the book had been split in two, and a
        # vintage question answered by sourcing channel satisfied it.
        if named:
            for concept, description, fields in named:
                facets.append(RequestedFacet(
                    kind=KIND_COHORT_COMPARISON,
                    label=f"a comparison by {description}",
                    concepts=(concept,), alt_keys=tuple(fields)))
        else:
            facets.append(RequestedFacet(
                kind=KIND_COHORT_COMPARISON,
                label="a comparison between two books"))
    # A share is the requested ANSWER only when the question is not also a
    # ranking. "Which region increased its share the most" asks WHICH ONE — the
    # share is the metric being ranked, and the ranking facet already guards it.
    # "What proportion of the book is above 60% LTV" asks for the proportion
    # itself, and nothing else was watching for it.
    if _asks_for_a_share(q) and not any(f.kind == KIND_RANKING for f in facets):
        facets.append(RequestedFacet(
            kind=KIND_SHARE, label="a proportion of the book"))
    concepts = named_measure_concepts(question)
    if len(concepts) > 1:
        facets.append(RequestedFacet(
            kind=KIND_MULTI_MEASURE,
            label="more than one measure (" + _join(concepts) + ")",
            concepts=tuple(concepts)))
    for slot in _unresolved_measure_slots(q, semantics, frame):
        # LOST at construction: the parser never resolved these words, so no
        # execution could have honoured them and there is no evidence to weigh.
        facets.append(RequestedFacet(
            kind=KIND_UNRESOLVED_MEASURE, label=slot, status=LOST,
            reason=("this was asked for alongside measures that were "
                    "calculated, but it is not a governed measure in this "
                    "dataset, so it was not calculated")))
    for key, term, alts in (requested_dimensions or []):
        facets.append(RequestedFacet(kind=KIND_GROUPING, label=term,
                                     field_key=key, alt_keys=alts))
    return facets


# --------------------------------------------------------------------------- #
# The receipt
# --------------------------------------------------------------------------- #
_AGGREGATION_LABELS = {
    "weighted_avg": "Weighted-average",
    "avg": "Average",
    "median": "Median",
    "sum": "Total",
    "balance_sum": "Total",
    "count": "Count of",
    "count_distinct": "Distinct count of",
    "distribution": "Distribution of",
    "loan_level": "Loan-level",
    "share": "Share of",
    "contribution": "Contribution to portfolio weighted-average",
}


@dataclass
class ExecutionReceipt:
    """What was actually calculated. Every field is read from execution."""

    measure: Optional[str] = None
    aggregation: Optional[str] = None
    filters: List[str] = field(default_factory=list)
    dimensions: List[str] = field(default_factory=list)
    population: Optional[int] = None
    #: The whole-book population, stated alongside a share so the DENOMINATOR is
    #: auditable from the receipt alone rather than implied.
    population_total: Optional[int] = None
    group_count: Optional[int] = None
    population_label: str = "loans"
    #: True when execution demonstrably narrowed the frame.
    narrowed: bool = False
    period: Optional[str] = None
    comparison_period: Optional[str] = None
    scenario: Optional[str] = None
    #: How a ranked answer was ordered — the basis, the direction and any Top N.
    #: Stated because "which region grew the most" has more than one defensible
    #: reading (money added, growth rate, share gained) and the reader is
    #: entitled to know which one produced the order they are looking at.
    ranking: Optional[str] = None
    parser_confidence: Optional[str] = None
    facets: List[RequestedFacet] = field(default_factory=list)
    #: True for a routed governed capability, whose scope is defined by the
    #: capability rather than by a filter on a frame. Suppresses the
    #: "entire funded portfolio" default, which would misdescribe (say) a
    #: two-book comparison as a whole-book aggregate.
    routed: bool = False

    # -- derived views ---------------------------------------------------- #
    def not_applied(self) -> List[RequestedFacet]:
        return [f for f in self.facets if f.status in DISCLOSABLE]

    def lost(self) -> List[RequestedFacet]:
        return [f for f in self.facets if f.status == LOST]

    def render(self) -> str:
        """The compact one-line receipt, e.g.::

            Calculated: Weighted-average current LTV · London · 1,380 loans ·
            as at 30 June 2026.
        """
        parts: List[str] = []
        head = " ".join(p for p in (_AGGREGATION_LABELS.get(self.aggregation or "", ""),
                                    self.measure or "") if p).strip()
        if head:
            parts.append(head[0].upper() + head[1:])
        if self.filters:
            parts.extend(self.filters)
        elif self.measure and not self.dimensions and not self.routed:
            # State the population explicitly so an unfiltered answer can never
            # be mistaken for a filtered one.
            parts.append("entire funded portfolio")
        if self.dimensions:
            parts.append(("ranked by " if self.ranking else "grouped by ")
                         + _join(self.dimensions))
        if self.ranking:
            parts.append(self.ranking)
        if self.scenario:
            parts.append(self.scenario)
        if self.comparison_period:
            parts.append(self.comparison_period)
        if self.group_count is not None:
            parts.append(f"{self.group_count:,} groups")
        if self.population is not None:
            if self.aggregation == "share" and self.population_total:
                parts.append(f"{self.population:,} qualifying "
                             f"{self.population_label} of "
                             f"{self.population_total:,}")
            else:
                parts.append(f"{self.population:,} {self.population_label}")
        if self.period:
            parts.append(f"as at {self.period}")
        if not parts:
            return ""
        line = "Calculated: " + " · ".join(parts) + "."
        # Surface low parser confidence only when the question actually carried a
        # material facet — i.e. when there was something scope-related to get
        # wrong. The confidence heuristic scores plain KPI questions ("what is
        # the funded balance?") as low, and printing a caveat on answers that are
        # unambiguously right would train readers to ignore it, which is worse
        # than not printing it. The raw value stays in the structured summary.
        if (self.parser_confidence and self.parser_confidence != "high"
                and self.facets):
            line += (f" Interpretation confidence: {self.parser_confidence} — "
                     "check the scope above matches your question.")
        return line

    def render_not_applied(self) -> Optional[str]:
        rows = self.not_applied()
        if not rows:
            return None
        return "Not applied: " + "; ".join(r.disclosure() for r in rows) + "."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "measure": self.measure,
            "aggregation": self.aggregation,
            "filtersApplied": list(self.filters),
            "dimensionsApplied": list(self.dimensions),
            "population": self.population,
            "populationTotal": self.population_total,
            "populationLabel": self.population_label,
            "groupCount": self.group_count,
            "narrowed": self.narrowed,
            "period": self.period,
            "comparisonPeriod": self.comparison_period,
            "scenario": self.scenario,
            "ranking": self.ranking,
            "parserConfidence": self.parser_confidence,
            "facets": [f.to_dict() for f in self.facets],
            "notApplied": [f.disclosure() for f in self.not_applied()],
            "receipt": self.render(),
        }


def _join(items: Sequence[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _business_name(key: Optional[str], semantics: dict) -> Optional[str]:
    if not key:
        return None
    entry = (semantics.get("fields", {}) if isinstance(semantics, dict) else {}).get(key) or {}
    return entry.get("business_name") or entry.get("display_name") or key.replace("_", " ")


def describe_filter(field_key: str, condition: Any, semantics: dict) -> str:
    """A human phrase for one APPLIED filter, e.g. ``youngest borrower age > 85``."""
    name = _business_name(field_key, semantics) or field_key
    if isinstance(condition, dict):
        op = str(condition.get("op", "eq")).strip().lower()
        value = condition.get("value", condition.get("min", condition.get("max")))
        symbols = {"gt": ">", "greater_than": ">", "ge": ">=", "gte": ">=",
                   "lt": "<", "less_than": "<", "le": "<=", "lte": "<=",
                   "eq": "=", "equals": "=", "ne": "≠", "not_equals": "≠"}
        if op in ("between",) and isinstance(value, (list, tuple)) and len(value) == 2:
            return f"{name} between {_num(value[0])} and {_num(value[1])}"
        if op in ("in", "one_of") and isinstance(value, (list, tuple, set)):
            return f"{name} in {', '.join(str(v) for v in value)}"
        return f"{name} {symbols.get(op, op)} {_num(value)}"
    if isinstance(condition, (list, tuple, set)):
        return f"{name} in {', '.join(str(v) for v in condition)}"
    return f"{condition}" if _looks_like_a_place(field_key) else f"{name} = {condition}"


def _num(value: Any) -> str:
    """Render a threshold as a person would write it ("85", not "85.0")."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _looks_like_a_place(field_key: str) -> bool:
    blob = (field_key or "").lower()
    return "geograph" in blob or "region" in blob or "collateral_geo" in blob


# --------------------------------------------------------------------------- #
# Reconciliation — requested facets vs the EXECUTED state
# --------------------------------------------------------------------------- #
#: Facets that change WHICH ROWS the number is computed over, or that ARE the
#: subject of the question. If one of these does not reach execution the answer
#: would be a confident number for a different question, so disclosure is not
#: enough — the caller must refuse.
NUMBER_OR_SUBJECT_FACETS = frozenset({
    KIND_GEOGRAPHIC_SCOPE, KIND_THRESHOLD, KIND_STRESS,
    KIND_COMPARISON_PERIOD, KIND_RANKING, KIND_PROJECTION,
    KIND_COHORT_COMPARISON, KIND_MULTI_MEASURE, KIND_RELATIONSHIP,
    KIND_CONTRIBUTION, KIND_UNRESOLVED_MEASURE, KIND_SHARE,
})
#: Facets that change the SHAPE of a still-valid answer. A partial answer is
#: acceptable provided the unhonoured facet is named.
SHAPE_FACETS = frozenset({KIND_GROUPING})

#: Verdicts from :func:`assess`.
VERDICT_OK = "ok"
VERDICT_PARTIAL = "partial"
VERDICT_REFUSE = "refuse"


def _applied_filter_phrases(spec, semantics: dict, narrowed: bool) -> List[str]:
    """Human phrases for the filters that actually ran, in spec order."""
    filters = getattr(spec, "filters", None) or {}
    if not isinstance(filters, dict) or not narrowed:
        return []
    return [describe_filter(k, v, semantics) for k, v in filters.items()]


def _filter_values(spec) -> List[str]:
    """Lowercased scalar values of every applied filter, for scope matching."""
    out: List[str] = []
    for value in (getattr(spec, "filters", None) or {}).values():
        if isinstance(value, str):
            out.append(value.strip().lower())
        elif isinstance(value, (list, tuple, set)):
            out.extend(str(v).strip().lower() for v in value)
    return out


def _comparison_ops_applied(spec) -> int:
    n = 0
    for value in (getattr(spec, "filters", None) or {}).values():
        if isinstance(value, dict):
            op = str(value.get("op", "")).strip().lower()
            if op not in ("", "eq", "equals", "in", "one_of"):
                n += 1
    return n


def reconcile_facets(facets: Sequence[RequestedFacet], *, spec, query_result,
                     semantics: dict, available_columns: Optional[Iterable[str]] = None,
                     route: Optional[str] = None,
                     scenario_applied: bool = False) -> List[RequestedFacet]:
    """Stamp a status on every requested facet from EXECUTION EVIDENCE.

    Evidence, never prose: a filter counts as applied only when execution
    actually narrowed the frame; a grouping counts as applied only when its
    column is in the executor's group keys or the result columns.
    """
    meta = getattr(query_result, "metadata", None) or {}
    recon = meta.get("reconciliation") or {}
    total = recon.get("total_records")
    after = recon.get("records_after_filters")
    narrowed = bool(total is not None and after is not None and after < total)

    group_keys = set(meta.get("group_field_keys") or [])
    rejected = {r.get("dimension"): (r.get("reason") or "not applied")
                for r in (meta.get("rejected_dimensions") or [])}
    result_cols: Set[str] = set()
    data = getattr(query_result, "data", None)
    if data is not None and hasattr(data, "columns"):
        result_cols = {str(c) for c in data.columns}
    columns = set(available_columns or ())
    fields = semantics.get("fields", {}) if isinstance(semantics, dict) else {}
    values = _filter_values(spec)
    comparison_ops = _comparison_ops_applied(spec)
    thresholds_seen = 0

    for facet in facets:
        if facet.kind == KIND_GEOGRAPHIC_SCOPE:
            if narrowed and any(facet.label.lower() in v or v in facet.label.lower()
                                for v in values):
                facet.status, facet.reason = APPLIED, ""
            elif facet.field_key and columns and \
                    (fields.get(facet.field_key, {}) or {}).get(
                        "canonical_field", facet.field_key) not in columns:
                facet.status = UNAVAILABLE
                facet.reason = "no geographic field in this dataset carries that value"
            else:
                facet.status = LOST
                facet.reason = "the geographic scope was not applied to the calculation"

        elif facet.kind == KIND_THRESHOLD:
            thresholds_seen += 1
            if narrowed and comparison_ops >= thresholds_seen:
                facet.status, facet.reason = APPLIED, ""
            else:
                facet.status = LOST
                facet.reason = "the threshold was not applied to the calculation"

        elif facet.kind == KIND_STRESS:
            if scenario_applied:
                facet.status, facet.reason = APPLIED, ""
            else:
                facet.status = UNSUPPORTED
                facet.reason = ("no governed stress or scenario calculation was run, "
                                "so this figure is unstressed")

        elif facet.kind == KIND_COMPARISON_PERIOD:
            if route in TEMPORAL_ROUTES:
                facet.status, facet.reason = APPLIED, ""
            else:
                facet.status = LOST
                facet.reason = ("the answer is a single point in time; no period "
                                "comparison was calculated")

        elif facet.kind == KIND_PROJECTION:
            facet.status = LOST
            facet.reason = ("this is a point-in-time calculation; no forward "
                            "projection was run")

        elif facet.kind == KIND_COHORT_COMPARISON:
            expected = tuple(facet.alt_keys or ())
            if expected:
                # SEMANTIC IDENTITY: the executed grouping must express the
                # concept the question named. Splitting the book by some other
                # cohort answers a different question with the right arithmetic.
                if any(k in group_keys for k in expected):
                    facet.status, facet.reason = APPLIED, ""
                elif not _cohort_fields_available(expected, columns, semantics):
                    # Name the CONCEPT, not the six field spellings that could
                    # have expressed it — a reader needs to know what the book
                    # cannot answer, not the vocabulary it was searched for.
                    facet.status = UNAVAILABLE
                    facet.reason = (
                        "this dataset carries no governed dimension for "
                        f"{facet.label[len('a comparison by '):]}, so the two "
                        "cohorts cannot be identified")
                elif any(k in group_keys for k in _COHORT_DIMENSION_KEYS):
                    facet.status = LOST
                    facet.reason = (
                        "the book was split by "
                        + _join([_business_name(k, semantics) or k
                                 for k in group_keys
                                 if k in _COHORT_DIMENSION_KEYS])
                        + f", which is not {facet.label[len('a comparison by '):]}")
                else:
                    facet.status = LOST
                    facet.reason = ("the two cohorts were not compared; this "
                                    "figure covers them together")
            elif any(k in group_keys for k in _COHORT_DIMENSION_KEYS):
                facet.status, facet.reason = APPLIED, ""
            else:
                facet.status = LOST
                facet.reason = ("the two books were not compared; this figure "
                                "covers them together")

        elif facet.kind == KIND_MULTI_MEASURE:
            # P1E: the requested measure SET is reconciled against the measures
            # execution actually returned. Every concept the question named must
            # be accounted for — applied, or explicitly named as unavailable.
            executed = executed_measure_concepts(query_result)
            requested = set(facet.concepts or ())
            if requested and executed:
                missing = sorted(requested - executed)
                if not missing:
                    facet.status, facet.reason = APPLIED, ""
                elif set(missing) < requested:
                    # Some ran and some did not. Disclosable ONLY because the
                    # executor names what it could not calculate; a silent 3-of-4
                    # is what this branch exists to prevent.
                    facet.status = UNAVAILABLE
                    facet.reason = ("not available in this dataset: "
                                    + _join(missing))
                else:
                    facet.status = UNSUPPORTED
                    facet.reason = ("none of the requested measures could be "
                                    "calculated")
            # A scatter/bubble expresses several measures at once (x, y, size),
            # so a multi-measure request IS honoured there.
            elif getattr(query_result, "result_type", None) == "loan_level":
                facet.status, facet.reason = APPLIED, ""
            else:
                facet.status = UNSUPPORTED
                facet.reason = ("only one measure can be calculated per question, "
                                "so the others were not returned")

        elif facet.kind == KIND_SHARE:
            # APPLIED only when the governed share aggregation actually ran.
            # Absolute figures for the filtered population are a DIFFERENT
            # answer: "£1.96bn, 11,007 loans" is not a proportion, and reading
            # one out of it is arithmetic the user would have to do themselves
            # against a denominator the answer never states.
            if getattr(spec, "aggregation", None) == "share" or meta.get("share_basis"):
                facet.status, facet.reason = APPLIED, ""
            else:
                facet.status = LOST
                facet.reason = ("an absolute figure was calculated for the "
                                "filtered population, not its proportion of "
                                "the book")

        elif facet.kind == KIND_CONTRIBUTION:
            # APPLIED only when the governed contribution aggregation actually
            # ran. Anything else — most importantly a plain per-group weighted
            # average, which is what this question used to be answered with —
            # is LOST, and a lost contribution facet refuses.
            if getattr(spec, "aggregation", None) == "contribution":
                facet.status, facet.reason = APPLIED, ""
            else:
                facet.status = LOST
                facet.reason = ("each group's own value was calculated, not its "
                                "contribution to the portfolio figure; a small "
                                "group with a high value contributes little, so "
                                "these rank differently")

        elif facet.kind == KIND_RELATIONSHIP:
            # Only a loan-level result (scatter/bubble) actually relates two
            # measures; a single aggregate cannot.
            if getattr(query_result, "result_type", None) == "loan_level":
                facet.status, facet.reason = APPLIED, ""
            else:
                facet.status = UNSUPPORTED
                facet.reason = ("a single aggregate was calculated, which cannot "
                                "express one measure relative to another")

        elif facet.kind in (KIND_GROUPING, KIND_RANKING):
            candidates = facet.satisfied_by()
            canonicals = [(fields.get(k, {}) or {}).get("canonical_field", k)
                          for k in candidates]
            key = facet.field_key
            canonical = canonicals[0] if canonicals else None
            if any(k in group_keys for k in candidates) or \
                    any(c in result_cols for c in canonicals):
                facet.status, facet.reason = APPLIED, ""
            elif key and route in RANKING_ROUTES and facet.kind == KIND_RANKING:
                facet.status, facet.reason = APPLIED, ""
            elif key in rejected:
                facet.status, facet.reason = REJECTED, rejected[key]
            elif canonical and columns and canonical not in columns:
                facet.status = UNAVAILABLE
                facet.reason = "field is unavailable in this dataset"
            else:
                facet.status = LOST
                facet.reason = "the requested breakdown was not applied"
    return facets


def detect_substitution(facets: Sequence[RequestedFacet], *, spec, query_result,
                        semantics: dict) -> Optional[str]:
    """A grouping the user did NOT ask for, standing in for one they did.

    Returns a description of the substitution, or None. This is the
    ``amortisation_type`` failure: "balance by LTV by borrower type" answered as
    a single bar on an unrelated dimension.
    """
    requested = {k for f in facets
                 if f.kind in (KIND_GROUPING, KIND_RANKING)
                 for k in f.satisfied_by()}
    if not requested:
        return None
    unmet = [f for f in facets
             if f.kind in (KIND_GROUPING, KIND_RANKING) and f.status != APPLIED]
    if not unmet:
        return None
    meta = getattr(query_result, "metadata", None) or {}
    executed = [k for k in (meta.get("group_field_keys") or [])]
    extra = [k for k in executed if k not in requested]
    if not extra:
        return None
    names = _join([_business_name(k, semantics) or k for k in extra])
    asked = _join([f.label for f in unmet])
    return (f"the breakdown was changed to {names}, which you did not ask for, "
            f"in place of {asked}")


#: How one executed measure reads in the receipt.
_MEASURE_AGG_WORDS = {
    "sum": "", "balance_sum": "", "count": "", "count_distinct": "Distinct ",
    "avg": "Average ", "median": "Median ", "weighted_avg": "Weighted-average ",
}


def _measure_set_phrase(executed: Sequence[Dict[str, Any]]) -> Optional[str]:
    """"Balance · Loans · Weighted-average Current LTV" — the measures that ran.

    Derived from the executor's declared measure metadata, never from the
    question, which is what lets the receipt expose a measure that went missing.
    Returns None for a single-measure result so today's receipts are unchanged.
    """
    if not executed or len(executed) < 2:
        return None
    parts: List[str] = []
    for measure in executed:
        label = (measure or {}).get("label") or (measure or {}).get("field") or ""
        lead = _MEASURE_AGG_WORDS.get(str((measure or {}).get("aggregation") or ""), "")
        phrase = f"{lead}{label}".strip()
        if phrase and phrase not in parts:
            parts.append(phrase)
    return " · ".join(parts) or None


def build_receipt(*, spec, query_result, semantics: dict, facets: Sequence[RequestedFacet],
                  parser_confidence: Optional[str] = None,
                  period: Optional[str] = None,
                  comparison_period: Optional[str] = None,
                  scenario: Optional[str] = None) -> ExecutionReceipt:
    """The receipt for one executed point-in-time query."""
    meta = getattr(query_result, "metadata", None) or {}
    recon = meta.get("reconciliation") or {}
    total = recon.get("total_records")
    after = recon.get("records_after_filters")
    narrowed = bool(total is not None and after is not None and after < total)
    population = recon.get("records_included")
    if population is None:
        population = after if after is not None else total

    dimensions = [_business_name(k, semantics) or k
                  for k in (meta.get("group_field_keys") or [])]
    group_count = None
    if getattr(query_result, "result_type", None) == "table":
        group_count = getattr(query_result, "row_count", None)

    executed = meta.get("measures_executed") or []
    return ExecutionReceipt(
        measure=(_measure_set_phrase(executed)
                 or _business_name(getattr(spec, "metric", None), semantics)),
        # A measure set carries its own per-measure aggregation, so a single
        # sentence-leading aggregation label would misdescribe it.
        aggregation=(None if executed
                     else getattr(spec, "aggregation", None)),
        filters=_applied_filter_phrases(spec, semantics, narrowed),
        dimensions=dimensions,
        population=int(population) if population is not None else None,
        population_total=int(total) if total is not None else None,
        group_count=group_count,
        narrowed=narrowed,
        period=period,
        comparison_period=comparison_period,
        scenario=scenario,
        parser_confidence=parser_confidence,
        facets=list(facets),
    )


def assess(receipt: ExecutionReceipt, *, substitution: Optional[str] = None
           ) -> Tuple[str, Optional[str]]:
    """``(verdict, refusal_or_disclosure_message)``.

    * ``VERDICT_REFUSE`` — a facet that changes the number, or that IS the
      subject of the question, did not reach execution. Answering would present
      a confident figure for a different question.
    * ``VERDICT_PARTIAL`` — the answer stands, but a requested facet must be
      named as not applied.
    * ``VERDICT_OK`` — everything material was applied.
    """
    blocking = [f for f in receipt.facets
                if f.kind in NUMBER_OR_SUBJECT_FACETS and f.status != APPLIED]
    if blocking:
        detail = "; ".join(f.disclosure() for f in blocking)
        return VERDICT_REFUSE, (
            f"I understood that you asked for {_join([f.label for f in blocking])}, "
            f"but that could not be applied to the calculation ({detail}). "
            "I have not substituted a broader figure.")

    if substitution:
        return VERDICT_REFUSE, (
            "I could not answer this as asked: " + substitution +
            ". I have not returned the substituted breakdown.")

    partial = [f for f in receipt.facets
               if f.kind in SHAPE_FACETS and f.status != APPLIED]
    if partial:
        return VERDICT_PARTIAL, (
            "Not applied: " + "; ".join(f.disclosure() for f in partial) + ".")
    return VERDICT_OK, None


# --------------------------------------------------------------------------- #
# Routed governed capabilities
# --------------------------------------------------------------------------- #
# A routed answer (risk limits, geographic exposure, period change, bridge,
# forecast) never touches the point-in-time executor, so there is no
# MIQueryResult to reconcile against. What a route DOES declare is its identity
# and its own scope, which is enough to decide whether a material facet was
# honoured.

#: Routes whose answer ENUMERATES every category (a limit schedule, a regional
#: ranking). A geographic scope these could not narrow to is a disclosure, not a
#: refusal: the requested category is present in the listing and no single number
#: is being passed off as the narrow one.
LISTING_ROUTES = frozenset({
    "risk_limits", "geo_exposure", "concentration_analysis",
})

#: Routes whose answer STATES a proportion of the book in its own terms — the
#: concentration answer reads "£83.4m (4.2% of the book)". A share facet those
#: satisfy must not be reported as lost merely because no ``share`` aggregation
#: appears in a spec the route never used.
SHARE_BEARING_ROUTES = frozenset({
    "concentration_analysis", "geo_exposure", "risk_limits",
})

#: Human labels for the governed capability that answered.
_ROUTE_LABELS = {
    "risk_limits": "Concentration limits vs the governing document",
    "geo_exposure": "Geographic exposure",
    "concentration_analysis": "Exposure concentration",
    "period_change_analysis": "Governed period change",
    "period_movement": "Month-on-month movement",
    "temporal_compare": "Governed period comparison",
    "evolution": "Metric evolution",
    "funded_bridge": "Funded balance bridge",
    "forecast_extrapolation": "Run-rate extrapolation",
    "portfolio_risk_comparison": "Portfolio comparison",
    "scenario": "Scenario projection",
    "cohort_progression": "Cohort progression",
    "cohort_conversion": "Cohort conversion",
}


def ranking_evidence(envelope: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The ranking a route DECLARES it performed, as structured evidence.

    Returns ``{}`` unless the route stated that a ranking was applied and named
    the canonical field it ranked. A declaration without a field is not evidence
    — it cannot be checked against what the question asked for.
    """
    if not isinstance(envelope, dict):
        return {}
    declared = (envelope.get("metadata") or {}).get("rankedMovement")
    if not isinstance(declared, dict) or not declared.get("applied"):
        return {}
    if not declared.get("canonicalField"):
        return {}
    return declared


def comparison_evidence(envelope: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """What a portfolio comparison DECLARES it actually compared.

    Returns ``{}`` unless the route stated the measures it compared. A route
    that merely ran is not evidence that the requested measure was compared —
    which is the whole defect this exists to catch.
    """
    if not isinstance(envelope, dict):
        return {}
    declared = (envelope.get("metadata") or {}).get("portfolioComparison")
    return declared if isinstance(declared, dict) else {}


_PROPORTION_IN_ANSWER_RE = re.compile(
    r"\d[\d,.]*\s*%|\bper cent\b|\bpercent\b", re.I)


def concentration_evidence(envelope: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The SINGLE-NAME analysis a route declares it performed.

    Returns ``{}`` unless the route states the grain it ranked, both sides of
    the share and the population. A declaration missing any of those is not
    evidence: it cannot show that the numerator is one loan and the denominator
    is the same governed measure over the selected population.
    """
    if not isinstance(envelope, dict):
        return {}
    block = (envelope.get("metadata") or {}).get("concentration")
    if not isinstance(block, dict):
        return {}
    required = ("grainField", "topExposure", "totalExposure", "population")
    if any(block.get(k) is None for k in required):
        return {}
    return block


def _single_loan_share_proven(evidence: Dict[str, Any]) -> bool:
    """The grain invariant, as a predicate.

    Numerator is ONE name at the declared grain; denominator is the same
    governed exposure basis over the whole selected population; and the share
    the route reports is the one those two produce. A largest-loan answer whose
    denominator came from somewhere else would fail here rather than read as a
    correct percentage.
    """
    if not evidence or evidence.get("kind") != "loan":
        return False
    if evidence.get("basis") != "exposure":
        return False
    top, total = evidence.get("topExposure"), evidence.get("totalExposure")
    share = evidence.get("topShare")
    if not total or share is None:
        return False
    try:
        return abs(float(share) - (float(top) / float(total))) < 1e-12
    except Exception:  # noqa: BLE001 - unusable evidence is not proof
        return False


def _states_a_proportion(envelope: Optional[Dict[str, Any]]) -> bool:
    """True when the routed answer itself reports a percentage."""
    if not isinstance(envelope, dict):
        return False
    text = " ".join(str(envelope.get(k) or "")
                    for k in ("answer", "summary", "headline"))
    return bool(_PROPORTION_IN_ANSWER_RE.search(text))


def reconcile_routed_facets(facets: Sequence[RequestedFacet], *, route: Optional[str],
                            semantics: dict,
                            available_columns: Optional[Iterable[str]] = None,
                            envelope: Optional[Dict[str, Any]] = None
                            ) -> List[RequestedFacet]:
    """Stamp facet statuses for an answer produced by a routed capability.

    ``envelope`` is the routed answer itself. It is read for EVIDENCE only — a
    route that claims to have ranked a dimension must name the canonical field
    it ranked, and that field must be one the question's ranking facet resolves
    to. A claim that does not match the question is not accepted.
    """
    columns = set(available_columns or ())
    ranked = ranking_evidence(envelope)
    compared = comparison_evidence(envelope)
    fields = semantics.get("fields", {}) if isinstance(semantics, dict) else {}
    listing = route in LISTING_ROUTES

    def _canonical(key: Optional[str]) -> Optional[str]:
        if not key:
            return None
        return (fields.get(key, {}) or {}).get("canonical_field", key)

    for facet in facets:
        if facet.kind == KIND_SHARE:
            # Execution-proven first: a route that DECLARES a single-name share,
            # with both sides of it and the grain it used, has answered the
            # facet whatever its prose says. Falling back to the answer text is
            # weaker evidence, kept for routes that state a proportion without
            # declaring one.
            if _single_loan_share_proven(concentration_evidence(envelope)):
                facet.status, facet.reason = APPLIED, ""
            elif route in SHARE_BEARING_ROUTES and _states_a_proportion(envelope):
                facet.status, facet.reason = APPLIED, ""
            else:
                facet.status = LOST
                facet.reason = ("this answer does not state what proportion of "
                                "the book the figure represents")

        elif facet.kind == KIND_GEOGRAPHIC_SCOPE:
            if listing:
                facet.status = UNSUPPORTED
                facet.reason = ("this answer covers every region rather than "
                                f"narrowing to {facet.label}")
            else:
                facet.status = LOST
                facet.reason = "the geographic scope was not applied to the calculation"

        elif facet.kind == KIND_THRESHOLD:
            facet.status = LOST
            facet.reason = ("this governed capability does not apply a value "
                            "threshold, so the figure is not restricted to it")

        elif facet.kind == KIND_STRESS:
            if route in SCENARIO_ROUTES:
                facet.status, facet.reason = APPLIED, ""
            else:
                facet.status = UNSUPPORTED
                facet.reason = ("no governed stress or scenario calculation was "
                                "run, so this figure is unstressed")

        elif facet.kind == KIND_COMPARISON_PERIOD:
            if route in TEMPORAL_ROUTES:
                facet.status, facet.reason = APPLIED, ""
            else:
                facet.status = LOST
                facet.reason = ("the answer is a single point in time; no period "
                                "comparison was calculated")

        elif facet.kind == KIND_PROJECTION:
            if route in PROJECTION_ROUTES:
                facet.status, facet.reason = APPLIED, ""
            else:
                facet.status = LOST
                facet.reason = ("this is a point-in-time answer; no forward "
                                "projection was run")

        elif facet.kind == KIND_COHORT_COMPARISON:
            wanted = tuple(facet.concepts or ())
            executed_concept = (compared.get("cohortConcept")
                                if compared else None)
            if wanted and executed_concept and executed_concept not in wanted:
                # SEMANTIC IDENTITY on the routed path. The comparison route
                # splits the book by how loans were SOURCED; a question about
                # how long they have been on the book is a different cohort,
                # and two correctly-compared portfolios do not answer it.
                facet.status = LOST
                facet.reason = (
                    f"the books were compared by {executed_concept}, which is "
                    f"not {facet.label[len('a comparison by '):]}")
            elif wanted and not executed_concept and route in COHORT_ROUTES:
                # A cohort route that does not say which cohort it compared is
                # not evidence that it compared the requested one.
                facet.status = LOST
                facet.reason = ("the capability did not state which cohorts it "
                                "compared, so the requested one is unproven")
            elif compared:
                # The route declared what it compared. A comparison that
                # measured NOTHING, or that dropped the measure the question
                # named, has not compared the two books on that question — even
                # though the route ran end to end.
                if compared.get("requestedMetric") and not compared.get(
                        "requestedMetricCompared"):
                    facet.status = LOST
                    facet.reason = ("the measure the question named was not "
                                    "among the governed indicators compared")
                elif not (compared.get("measuresCompared")
                          or compared.get("dimensionsCompared")):
                    facet.status = LOST
                    facet.reason = ("no governed indicator was compared, so "
                                    "nothing was measured about the difference")
                else:
                    facet.status, facet.reason = APPLIED, ""
            elif route in COHORT_ROUTES:
                facet.status, facet.reason = APPLIED, ""
            else:
                facet.status = LOST
                facet.reason = "the two books were not compared separately"

        elif facet.kind == KIND_CONTRIBUTION:
            # No routed capability decomposes a weighted aggregate across
            # groups. Saying so is what stops a concentration or geography
            # answer standing in for a contribution.
            facet.status = LOST
            facet.reason = ("this governed capability does not decompose a "
                            "weighted average across groups")

        elif facet.kind == KIND_MULTI_MEASURE and compared.get("measuresCompared"):
            # P1E: a governed comparison may carry several measures. Reconciled
            # against what the route declares it compared, measure by measure.
            not_compared = compared.get("requestedMetricsNotCompared") or []
            if not not_compared:
                facet.status, facet.reason = APPLIED, ""
            else:
                facet.status = UNAVAILABLE
                facet.reason = ("not compared for these books: "
                                + _join([str(f) for f in not_compared]))

        elif facet.kind in (KIND_MULTI_MEASURE, KIND_RELATIONSHIP):
            facet.status = UNSUPPORTED
            facet.reason = ("this governed capability returns a single measure, "
                            "so the full question was not expressed")

        elif facet.kind in (KIND_GROUPING, KIND_RANKING):
            canonicals = [_canonical(k) for k in facet.satisfied_by()]
            canonical = canonicals[0] if canonicals else None
            if ranked and _canonical(ranked.get("canonicalField")) in canonicals:
                # The route ranked the dimension this facet asked for, and said
                # so with the field it used. Proven, not assumed.
                facet.status, facet.reason = APPLIED, ""
            elif ranked and facet.kind == KIND_RANKING:
                facet.status = LOST
                facet.reason = (f"the answer ranks "
                                f"{ranked.get('displayName') or 'another dimension'}, "
                                f"not {facet.label}")
            elif listing and facet.kind == KIND_RANKING:
                # The listing engine produces the ranking itself; which column
                # the frame carries is not what decides whether it answered.
                facet.status, facet.reason = APPLIED, ""
            elif canonicals and columns and not any(c in columns for c in canonicals):
                facet.status = UNAVAILABLE
                facet.reason = "field is unavailable in this dataset"
            elif facet.kind == KIND_RANKING and not listing:
                facet.status = LOST
                facet.reason = ("this answer does not rank that dimension")
            else:
                # The route grouped by something it declared; without a result
                # frame we cannot disprove it, and refusing on an unprovable
                # facet would disable working governed analytics.
                facet.status, facet.reason = APPLIED, ""
    return facets


#: Routes whose measure is fixed by the capability itself. Asking one of these a
#: question about a DIFFERENT measure gets an answer about its own measure —
#: which is the routed form of a silent substitution.
ROUTE_FIXED_MEASURE = {
    "funded_bridge": "balance",
    "geo_exposure": "balance",
    "concentration_analysis": "balance",
    "forecast_extrapolation": "balance",
}


def comparison_measure_concepts(compared: Dict[str, Any]) -> Set[str]:
    """The measure CONCEPTS a routed comparison declares it compared.

    Read from the route's own ``measuresCompared`` field list, so the answer is
    checked against what executed rather than against the spec it was handed.
    """
    concepts: Set[str] = set()
    for key in (compared or {}).get("measuresCompared") or []:
        if key in ("loan_count", "count", "loan_identifier"):
            concepts.add("count")
            continue
        concept = executed_measure_concept(key)
        if concept:
            concepts.add(concept)
    return concepts


def detect_measure_substitution(question: str, *, route: Optional[str] = None,
                                metric_key: Optional[str] = None,
                                executed_concepts: Optional[Set[str]] = None
                                ) -> Optional[str]:
    """The measure the answer reports is not one the question named.

    Only fires when the question EXPLICITLY names at least one measure and the
    executed measure is known and is none of them — so a question that names no
    measure, or a capability whose measure cannot be determined, is never
    refused on this basis.

    ``executed_concepts`` is the measure SET that actually ran. It matters
    because a spec can carry a set with no singular ``metric``: passing only
    ``metric_key`` then yields None, the check is skipped, and "compare the two
    books on borrower age" answered with balance and loan count reads as
    correct. That is the defect this parameter closes — the question's measure
    is reconciled against everything that executed, not against one slot.
    """
    named = named_measure_concepts(question)
    if not named:
        return None
    # The route's own fixed measure wins over the spec's: a capability that
    # always reports exposure concentration reports exposure concentration even
    # when the spec it was handed named arrears.
    fixed = ROUTE_FIXED_MEASURE.get(route or "")
    if fixed:
        executed_set = {fixed}
    else:
        executed_set = set(executed_concepts or ())
        single = executed_measure_concept(metric_key)
        if single:
            executed_set.add(single)
    if not executed_set or executed_set & set(named):
        return None
    return (f"the answer reports {_join(sorted(executed_set))}, but the "
            f"question asked about {_join(named)}")


#: Granularity a route reports at, when it is fixed and may differ from the one
#: the question named. ``geo_exposure`` always answers at ITL3 area level, so a
#: question about postcodes gets area-level numbers — useful, but only if the
#: substitution is stated.
_ROUTE_GRANULARITY = {"geo_exposure": ("postcode", "ITL3 area")}


def granularity_disclosure(question: str, route: Optional[str]
                           ) -> Optional[RequestedFacet]:
    """A facet for a granularity the route could not honour, or None."""
    pair = _ROUTE_GRANULARITY.get(route or "")
    if not pair:
        return None
    asked, reported = pair
    if not re.search(r"\b" + asked + r"s?\b", question or "", re.I):
        return None
    return RequestedFacet(
        kind=KIND_GROUPING, label=asked, status=UNAVAILABLE,
        reason=f"this answer is reported at {reported} level, not by {asked}")


#: How a comparison aggregation reads in the receipt.
_COMPARISON_AGG_WORDS = {
    "average": "Average", "avg": "Average", "mean": "Average",
    "weighted_average": "Weighted-average", "weighted_avg": "Weighted-average",
    "sum": "Total", "median": "Median", "share": "Share of",
}


def _comparison_measure(compared: Dict[str, Any]) -> Optional[str]:
    """"Average Youngest Borrower Age" — the measure the comparison EXECUTED.

    Derived from the route's executed-comparison metadata, never from the
    question's wording: a receipt that echoed the question could not expose the
    case where the named measure was never compared.
    """
    if not compared:
        return None
    labels = [l for l in (compared.get("measureLabels") or []) if l]
    if not labels:
        return None
    aggs = [a for a in (compared.get("aggregations") or []) if a]
    # Every compared measure is named. "and 2 further indicators" hid exactly
    # the thing a multi-measure receipt exists to show.
    parts: List[str] = []
    for index, label in enumerate(labels):
        agg = aggs[index] if index < len(aggs) else ""
        lead = _COMPARISON_AGG_WORDS.get(str(agg).lower(), "")
        phrase = f"{lead} {label}".strip() if lead else label
        if phrase not in parts:
            parts.append(phrase)
    return " · ".join(parts)


def _comparison_populations(compared: Dict[str, Any]) -> List[str]:
    """"Direct vs Acquired" — the two populations that were compared."""
    if not compared:
        return []
    a, b = compared.get("portfolioA"), compared.get("portfolioB")
    return [f"{a} vs {b}"] if a and b else []


def build_routed_receipt(*, route: Optional[str], envelope: Dict[str, Any],
                         facets: Sequence[RequestedFacet]) -> ExecutionReceipt:
    """A receipt for a routed answer, built from what the route itself declares."""
    metadata = envelope.get("metadata") or {}
    spec = envelope.get("spec") if isinstance(envelope.get("spec"), dict) else {}
    ranked = ranking_evidence(envelope)
    compared = comparison_evidence(envelope)
    concentrated = concentration_evidence(envelope)
    return ExecutionReceipt(
        measure=(_comparison_measure(compared)
                 or _single_name_measure(concentrated)
                 or _ROUTE_LABELS.get(route or "", "Governed analysis")),
        aggregation=None,
        dimensions=([ranked["displayName"]] if ranked.get("displayName") else []),
        filters=_comparison_populations(compared),
        population=concentrated.get("population"),
        period=(metadata.get("asOfDate") or envelope.get("asOf")
                or concentrated.get("reportingDate")),
        comparison_period=_ranked_period(ranked),
        scenario=None,
        ranking=_ranking_phrase(ranked),
        parser_confidence=(spec or {}).get("parser_confidence"),
        facets=list(facets),
        routed=True,
    )


def _single_name_measure(evidence: Dict[str, Any]) -> Optional[str]:
    """"Largest single-loan current exposure · share of total current exposure".

    Names the calculation that ran rather than the capability that ran it:
    "Exposure concentration" is true of a regional breakdown too, and tells a
    reader nothing about which number they are looking at.
    """
    if not evidence:
        return None
    kind = str(evidence.get("kind") or "name")
    measure = f"Largest single-{kind} current exposure"
    if _single_loan_share_proven(evidence) or evidence.get("topShare") is not None:
        measure += " · share of total current exposure"
    return measure


#: How a ranking direction reads in the receipt.
_RANK_DIRECTION_WORDS = {
    "increase": "largest increases first",
    "decrease": "largest decreases first",
    "movement": "largest movements first, in either direction",
}


def _ranking_phrase(ranked: Dict[str, Any]) -> Optional[str]:
    """"ranked on absolute balance movement, largest increases first, top 3"."""
    if not ranked:
        return None
    parts = [str(ranked.get("basisLabel") or ranked.get("basis") or "").strip()]
    direction = _RANK_DIRECTION_WORDS.get(str(ranked.get("direction") or ""))
    if direction:
        parts.append(direction)
    top_n = ranked.get("topN")
    if top_n:
        parts.append(f"top {int(top_n)} of "
                     f"{int(ranked.get('categoriesAnalysed') or 0):,}")
    return ", ".join(p for p in parts if p) or None


def _ranked_period(ranked: Dict[str, Any]) -> Optional[str]:
    """The two dates the ranking actually spanned, stated as the receipt's
    comparison period so a ranked answer can never imply a span it did not use."""
    if not ranked:
        return None
    opening, closing = ranked.get("openingPeriod"), ranked.get("closingPeriod")
    if not opening or not closing:
        return None
    return f"{opening} → {closing}"


# --------------------------------------------------------------------------- #
# Period grain
# --------------------------------------------------------------------------- #
# A temporal route genuinely compares two periods — but not necessarily the two
# the question named. "Since inception" answered as May-to-June is the failure
# mode P0 lists as "requested period silently changed": the route states the
# dates it used, which is honest about what ran, yet never says the requested
# span was not honoured.

#: Minimum span, in whole months, that a requested comparison implies. A label
#: with no entry places no span requirement.
_REQUIRED_SPAN_MONTHS = {
    "since inception": 2,
    "last quarter": 3,
    "this quarter": 3,
    "last year": 12,
}

_PERIOD_LABELS = ("opening period", "closing period")


def declared_period_span(envelope: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """``(opening, closing)`` as the route itself declared them, or None."""
    found: Dict[str, str] = {}
    for artifact in envelope.get("artifacts") or []:
        for item in (artifact.get("items") or artifact.get("kpis") or []):
            label = str((item or {}).get("label", "")).strip().lower()
            if label in _PERIOD_LABELS and item.get("value"):
                found[label] = str(item["value"])
    if len(found) == 2:
        return found["opening period"], found["closing period"]
    return None


def _months_between(opening: str, closing: str) -> Optional[int]:
    """Whole months between two human dates ("31 May 2026" / "30 June 2026")."""
    from datetime import datetime

    parsed = []
    for value in (opening, closing):
        for fmt in ("%d %B %Y", "%d %b %Y", "%Y-%m-%d", "%d/%m/%Y"):
            try:
                parsed.append(datetime.strptime(value.strip(), fmt))
                break
            except ValueError:
                continue
        else:
            return None
    start, end = parsed
    return (end.year - start.year) * 12 + (end.month - start.month)


def check_period_grain(facets: Sequence[RequestedFacet],
                       envelope: Dict[str, Any]) -> List[RequestedFacet]:
    """Downgrade a comparison facet the route answered over too short a span.

    Only acts when the route DECLARED both periods and the requested label
    carries a span requirement, so a route that states nothing is never refused
    on an unverifiable basis.
    """
    span = declared_period_span(envelope)
    if not span:
        return list(facets)
    months = _months_between(*span)
    if months is None:
        return list(facets)
    for facet in facets:
        if facet.kind != KIND_COMPARISON_PERIOD or facet.status != APPLIED:
            continue
        for label, required in _REQUIRED_SPAN_MONTHS.items():
            if label in facet.label.lower() and months < required:
                facet.status = UNSUPPORTED
                facet.reason = (
                    f"the comparison ran over {months} month(s) "
                    f"({span[0]} to {span[1]}), which does not cover "
                    f"'{label}'")
                break
    return list(facets)


# --------------------------------------------------------------------------- #
# Superlative without a ranking
# --------------------------------------------------------------------------- #
#: "What is the LARGEST single-loan exposure" asks for one extreme row. A spec
#: that carries no ranking, no top-N and no grouping answers it with the WHOLE
#: population — a plausible number for a different question. The deterministic
#: parser ranks and truncates; a model-produced ``loan_level`` spec need not, so
#: this is checked from execution rather than trusted from the parse.
_SUPERLATIVE_RE = re.compile(
    r"\b(?:largest|biggest|highest|greatest|smallest|lowest|maximum|minimum|"
    r"max|min)\b", re.I)


def detect_unranked_superlative(question: str, *, spec, query_result) -> Optional[str]:
    """A superlative question answered over the whole, unranked population."""
    if not _SUPERLATIVE_RE.search(question or ""):
        return None
    if getattr(spec, "top_n", None) or getattr(spec, "ranking_mode", None):
        return None
    meta = getattr(query_result, "metadata", None) or {}
    if meta.get("group_field_keys"):
        return None                      # a grouped ranking IS a ranking
    recon = meta.get("reconciliation") or {}
    total = recon.get("total_records")
    included = recon.get("records_after_filters")
    if total is None or included is None or included < total:
        return None                      # narrowed: not a whole-book answer
    # A LOAN-LEVEL result whose rows are fewer than the population has been
    # ranked and truncated to the extreme rows — that IS the answer. A summary
    # row is not: "1 group" there means one aggregate OVER the whole book, which
    # is the failure this guard exists to catch.
    row_count = getattr(query_result, "row_count", None)
    if (getattr(query_result, "result_type", None) == "loan_level"
            and row_count is not None and total and row_count < total):
        return None
    return ("the question asks for a single extreme value, but the calculation "
            "covered the whole book without ranking it")
