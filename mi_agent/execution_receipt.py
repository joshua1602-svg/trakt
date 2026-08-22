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
from typing import (
    Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple,
)

from question_interpretation import lexical as _lexical

from . import statistic as _statistic

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
KIND_GEOGRAPHIC_SCOPE = "geographic_scope"
#: A row selector the SENTENCE marked, on any dimension, that no source resolved
#: into a predicate. Distinct from KIND_GEOGRAPHIC_SCOPE, which is the same idea
#: restricted to geography and carrying geographic prose, and from
#: KIND_POPULATION, which carries a RESOLVED predicate and its value.
KIND_LOST_NARROWING = "lost_narrowing"
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

#: A material ROW POPULATION the question asked the calculation to run over —
#: "the back book", "borrowers over 85", "loans below 75% LTV". Distinct from
#: KIND_GEOGRAPHIC_SCOPE (one named place) and from the portfolio SCOPE, which
#: travels on the lens rather than in spec.filters.
#:
#: This facet exists because P1K proved the guard was structurally blind to a
#: dropped population: twelve of thirteen specialist routes ignored
#: spec.filters, so a back-book question was answered across the whole book with
#: ok=True and a spec that still claimed the filter. Presence of the filter on
#: the spec is NOT evidence it ran; only execution evidence is.
KIND_POPULATION = "row_population"

#: P1M. The STATISTIC the question asked for — the median, the average, the
#: weighted average. Distinct from the MEASURE (which quantity) and from the
#: POPULATION (which rows): a question can name the right measure over the right
#: rows and still be answered with the wrong statistic.
#:
#: This facet exists because the statistic was the one axis of a question that
#: nothing reconciled. "What is the median LTV?" returned 43.1562 — the weighted
#: average — against a true median of 39.6757, with ok=True and no warning,
#: because the registry does not permit a median for LTV and the request was
#: quietly rounded to the statistic that was permitted. Presence of an
#: aggregation on the spec is NOT evidence that the requested statistic ran.
KIND_STATISTIC = "requested_statistic"

#: A requested reporting GRAIN — the level an answer is reported at, rather than
#: a field it is grouped by. "By postcode" against a route that reports ITL3
#: areas, "by week" against month-end snapshots.
#:
#: Distinct from KIND_GROUPING, which carries a registry `field_key`. A grain is
#: not a field: there is no `postcode` column to be applied or unavailable, and
#: no `week` column either. Filing one as a grouping meant the receipt described
#: a reporting level as a breakdown, and — because the status was written at
#: DETECTION — no reconciler ever adjudicated it, so a grain the route DID
#: honour raised nothing at all and nothing downstream could act on it.
KIND_GRANULARITY = "granularity"

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
    #: Character offsets of the wording in the ORIGINAL question, where the
    #: detector matched on the question itself.
    #:
    #: Recorded because the facet layer and the parser each read one HALF of a
    #: filter clause — the facet supplies the wording and no field, the parser
    #: supplies the field and bound and no wording — and on 76 corpus questions
    #: there is nothing linking the two. Value matching cannot link them: the
    #: label is built from the raw digit group, so "£250k" renders as 250 while
    #: the parser holds 250000.0. Offsets can, and the detectors already
    #: computed them and threw them away.
    #:
    #: None wherever the facet was not produced by matching the question text.
    #: ADDITIVE and read by nothing today.
    span: Optional[Tuple[int, int]] = None

    def satisfied_by(self) -> Tuple[str, ...]:
        """Every field key that counts as honouring this facet."""
        keys = [k for k in (self.field_key,) + tuple(self.alt_keys) if k]
        return tuple(dict.fromkeys(keys))

    def to_dict(self) -> Dict[str, Any]:
        # `span` is deliberately NOT serialised. It exists for the in-process
        # filter join, and the executionSummary carrying facets is user-visible
        # payload: adding a key to it moved 32 of 340 answers on the answer-text
        # diff while the calibration bank passed all 260. Stage 2's acceptance
        # is byte-identical answers, so the span stays in memory.
        return {"kind": self.kind, "label": self.label, "field": self.field_key,
                "status": self.status, "reason": self.reason}

    def disclosure(self, semantics: Optional[dict] = None) -> str:
        """The user-facing line for a facet that did not apply.

        Names the FIELD as well as the wording where one is known. "joint
        borrower — field is unavailable in this dataset" tells a reader that
        something was refused; "joint borrower (Borrower Type) — …" tells them
        which field their book would need to carry to get an answer. A refusal
        that cannot be acted on is only half a refusal.
        """
        label = self.label
        if self.field_key and semantics:
            name = _business_name(self.field_key, semantics)
            if name and name.strip().lower() != str(label).strip().lower():
                label = f"{label} ({name})"
        if self.reason:
            return f"{label} — {self.reason}"
        return label


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
#: Item 1 — THE PHRASES COME FROM `question_interpretation.lexical`, the same
#: owner the parser reads. This list used to name its own words and they agreed
#: on 16 of 30; where both were blind the narrowing vanished and nothing was
#: raised for the guard to honour.
#:
#: DETECTION STAYS HERE, and that is the point of the design rather than an
#: oversight. This reads the SENTENCE, not the parser's output — so a threshold
#: the parser fails to apply is still raised, execution cannot honour it, and
#: the guard refuses instead of answering the whole book. Deriving this facet
#: from the applied filters would delete exactly that protection.
def _threshold_pattern(op: str) -> str:
    return (r"\b(?<!\bno )(?<!\bnot )(?<!\bor )(?:"
            + _lexical.comparator_alternation((op,))
            + r")\s+£?\s*(\d[\d,\.]*)\s*(%|percent)?")


_THRESHOLD_PATTERNS: Tuple[Tuple[str, str], ...] = (
    # The word is the OPERATOR rendered for a reader, from the one mapping in
    # the owning module — not a second list of phrases mapping to words.
    (_threshold_pattern("gt"), _lexical.COMPARATOR_WORD["gt"]),
    (_threshold_pattern("lt"), _lexical.COMPARATOR_WORD["lt"]),
    (_threshold_pattern("ge"), _lexical.COMPARATOR_WORD["ge"]),
    (_threshold_pattern("le"), _lexical.COMPARATOR_WORD["le"]),
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


def book_columns(frame) -> Set[str]:
    """The columns of the BOOK this frame reports on. THE schema answer, once.

    D6 (B14). A view is chosen by a substring test on the question — the word
    "forecast" anywhere selects the forecast view — before the question is parsed
    and before any route claims it. That view is a DERIVED frame of twelve
    columns where the book carries seventy-six, and every availability check read
    whichever frame had been loaded. So "what is the forecast run rate for the
    front book" answered:

        "front book — field is unavailable in this dataset"

    True of the projection and false of the book, which carries
    `seasoning_segment` for all 11,035 loans. Its sibling, asked about REGION —
    one of the twelve the projection kept — already said the true thing: the
    breakdown was not applied. Both refuse; only one was honest about why.

    A frame that IS the book (the funded view) carries no stamp and returns its
    own columns, so nothing changes there.

    WHAT THIS IS NOT. It answers a SCHEMA question — does the book report this
    field — and never a VALUE question. `geographic_values` and `dimension_values`
    must keep reading the LOADED frame, because a value can only be recognised in
    rows that exist. Confusing the two is how this change would silently become
    B19 and take the wrong-number class with it.
    """
    if frame is None:
        return set()
    stamped = None
    try:
        stamped = (getattr(frame, "attrs", None) or {}).get("book_columns")
    except Exception:  # noqa: BLE001 - metadata must never break a query
        stamped = None
    if stamped:
        return {str(c) for c in stamped}
    # `or ()` would be a bug here: a pandas Index has no truth value, and the
    # guard's blanket except would have swallowed the ValueError along with the
    # whole receipt.
    own = getattr(frame, "columns", None)
    return {str(c) for c in own} if own is not None else set()


def dimension_values(frame, semantics: dict, *, max_cardinality: int = 60
                     ) -> Dict[str, str]:
    """``{lowercased value: semantic field key}`` for EVERY low-cardinality dimension.

    `geographic_values` above, without its geography restriction — and the
    restriction is kept there rather than removed, because its consumers are
    geography-specific and its narrowness is what lets it be read with no other
    guard. Generalising it in place would have raised a scope facet on 101 of 697
    corpus questions.

    The reason is one collision, and it is why every consumer of THIS map must
    check the sentence first: **`Broker` is a value of `origination_channel` and
    also the ordinary word for the dimension**, so "balance by broker" — a plain
    breakdown, 78 times over in the corpora — reads as a narrowing to
    `origination_channel = Broker` unless something notices the `by`.
    Geography values (London, South East, Wales) do not collide with dimension
    words. General dimension values do.

    Built from the LOADED BOOK, so a value this tape does not carry cannot be
    recognised — the profiled-allowlist discipline, rather than a denylist that
    has to be completed.

    **A token that names a DIFFERENT field is never accepted as a value.** That
    is the collision, stated as a rule instead of patched per word: `Broker` is a
    value of `origination_channel` and a synonym of `broker_channel`, so
    "balance by broker" would read as a narrowing. The registry already declares
    every field's names and synonyms, so the exclusion is read from it rather
    than maintained as a list here.

    A different DIMENSION, and both qualifiers are load-bearing. `Redeemed` is a
    value of `account_status` and also a synonym of `loan_redemption_flag` — a
    FLAG, not a dimension, so it is no collision: a flag name is not a word
    anyone breaks a book down by. And a token naming the very field it is a value
    of is no collision either ("front book" names `seasoning_segment` and is one
    of its values), which is why that case is left to the seasoning owner through
    `taken` rather than dropped here.

    `_GEO_STOPWORDS` is deliberately NOT reused. It is geography-shaped — it
    contains "active", because a stray "active" must never bind a region — and
    `Active` is a legitimate `account_status` value. Importing one map's
    assumptions into another is how the geography restriction would have leaked.
    """
    out: Dict[str, str] = {}
    if frame is None or not hasattr(frame, "columns"):
        return out
    fields = semantics.get("fields", {}) if isinstance(semantics, dict) else {}
    named_by: Dict[str, Set[str]] = {}
    for key, entry in fields.items():
        entry = entry or {}
        if entry.get("role") != "dimension":
            continue
        names = [str(key), str(entry.get("business_name") or ""),
                 str(entry.get("display_name") or ""),
                 str(entry.get("canonical_field") or "")]
        names.extend(str(n) for n in (entry.get("synonyms") or ()))
        for name in names:
            token = name.strip().lower()
            if token:
                named_by.setdefault(token, set()).add(key)
    for key, entry in fields.items():
        if (entry or {}).get("role") != "dimension":
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
            token = str(value).strip().lower()
            if len(token) < 4 or (named_by.get(token, set()) - {key}):
                continue
            out.setdefault(token, key)
    return out


#: Words that say the question COMPARES two things rather than narrowing to one.
#:
#: Earned by the families each one would otherwise have broken. Without this
#: guard the detector below fires on 13 questions instead of 5, and the extra
#: eight are Q7.3, Q7.4 and Q8.x on both books — the seasoning and provenance
#: comparison families, which is 32c263a's family.
#:
#: A count of named values cannot replace it: "the front book" versus "our
#: seasoned lending" names its second side with a synonym the frame does not
#: carry, and "direct" versus "acquired" splits across two fields so each sees
#: exactly one value. The sentence says "compare"; that is the reliable signal.
_COMPARISON_MARKERS = re.compile(
    r"\b(?:compare[ds]?|comparing|comparison|versus|vs\.?|against|"
    r"relative to|differ(?:s|ent|ently|ence)?|between|both sides)\b", re.I)


def _detect_lost_narrowing(q: str, values: Dict[str, str],
                           taken: Optional[Iterable[str]] = None
                           ) -> List[RequestedFacet]:
    """A row selector the SENTENCE marked, naming a governed value of this book.

    Raised so that a narrowing which never reached execution is RECORDED rather
    than silently answered over the whole book. "What is the balance where
    account status is active?" returned Total Balance grouped by Account Status,
    2 groups, 11,035 loans — the whole book, with the breakdown certified.

    Four conditions, and each one is load-bearing:

      1. the token names a value the LOADED BOOK carries (`dimension_values`);
      2. `lexical.selector_mark` says the sentence used it to select — without
         this, "balance by broker" reads as a narrowing to Broker;
      3. the question contains no comparison marker — without this the seasoning
         and provenance comparison families refuse;
      4. exactly one value of that field is marked, so "London and the South
         East" is not read as a narrowing to one of them.

    Fields another owner has already taken are skipped: `taken` carries the
    seasoning owner's predicate fields and the parser's resolved filters, because
    a narrowing somebody resolved is not lost.

    Deliberately NOT reused for geography: `_detect_geographic_scope` owns that,
    reads a narrower map, and its facet carries prose about geographic scope.
    """
    from question_interpretation.lexical import selector_mark
    from .portfolio_lens import mask_scope_phrases

    if not values or _COMPARISON_MARKERS.search(q):
        return []
    # THE SCOPE OWNER CLAIMS ITS SPANS FIRST.
    #
    # "the direct book" names the POPULATION BEING REPORTED ON, and
    # `portfolio_lens` owns that vocabulary: "It is not a row predicate, not a
    # grouping axis, and not a place. The cure is to CLAIM THE SPAN FIRST."
    # `Direct` is also a value of `origination_channel`, so without this the
    # narrowing owner reads "the balance of the direct book" as a lost row
    # filter and refuses a governed scope question — eleven tests across P1I,
    # P1J-1, P1L and P1N said so.
    #
    # Masking preserves offsets, so every span below stays valid. This is the
    # same call the filter and dimension parsers already make; a sixth reader of
    # the scope vocabulary would have been the defect this programme removes.
    q = mask_scope_phrases(q)
    skip = {str(k) for k in (taken or ())}
    marked: Dict[str, List[Tuple[str, Tuple[int, int]]]] = {}
    seen_span: List[Tuple[int, int]] = []
    for value in sorted(values, key=len, reverse=True):
        field = values[value]
        if field in skip:
            continue
        for match in re.finditer(r"\b" + re.escape(value) + r"\b", q):
            span = (match.start(), match.end())
            # Longest first, so a value contained in a longer one already matched
            # does not match again ("east" inside "south east").
            if any(s <= span[0] and span[1] <= e for s, e in seen_span):
                continue
            seen_span.append(span)
            if selector_mark(q, span[0], span[1]):
                marked.setdefault(field, []).append((value, span))
    found: List[RequestedFacet] = []
    for field, hits in marked.items():
        if len({v for v, _s in hits}) != 1:
            continue                      # two values of one field: a comparison
        value, span = hits[0]
        found.append(RequestedFacet(
            kind=KIND_LOST_NARROWING, label=value.title(), field_key=field,
            concepts=(value,), span=span))
    return found


def _detect_geographic_scope(q: str, geo_values: Dict[str, str]) -> List[RequestedFacet]:
    found: List[RequestedFacet] = []
    seen: Set[str] = set()
    # Longest first, so "South East" wins over any shorter contained value.
    for value in sorted(geo_values, key=len, reverse=True):
        if value in seen:
            continue
        match = re.search(r"\b" + re.escape(value) + r"\b", q)
        if match:
            if any(value in other for other in seen):
                continue
            seen.add(value)
            found.append(RequestedFacet(
                kind=KIND_GEOGRAPHIC_SCOPE, label=value.title(),
                field_key=geo_values[value],
                span=(match.start(), match.end())))
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
            found.append(RequestedFacet(kind=KIND_THRESHOLD, label=label,
                                        span=(match.start(), match.end())))
    numbers_seen = {re.sub(r"[^\d.]", "", f.label.split()[-1]) for f in found}
    for match in _PCT_BOUND_RE.finditer(q):
        # "above 60% LTV" already produced a comparator threshold for 60; the
        # percent-bound form exists for "a 75% LTV cap", where no comparator was
        # written. Counting both would demand two filters for one predicate.
        if match.group(1) in numbers_seen:
            continue
        found.append(RequestedFacet(
            kind=KIND_THRESHOLD, label=f"LTV bound of {match.group(1)}%",
            field_key="current_loan_to_value",
            span=(match.start(), match.end())))
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


def _is_filter_subject(q: str, start: int, end: int) -> bool:
    """True when this measure word is the subject of a predicate, not a measure.

    "balance by region where LTV above 50%" measures balance and FILTERS on LTV.
    Counting the filter subject as a second requested measure would refuse a
    perfectly good filtered breakdown, so both sides are checked.

    Stage 3, conversion 2: the comparator vocabulary and the two window regexes
    that used to sit here now live in ``question_interpretation.lexical``, which
    owns them. They are kept DISTINCT from that module's condition openers
    rather than merged: this test wants comparators, so it carries "between",
    "exceeding", "in excess of" and the operators and does not carry "where",
    "with" or "for". The inventory's finding was that the three subject-side
    splits were separately DECLARED, not that they should be identical.

    Proved byte-identical at 400,097 positions — every offset pair across the
    corpus, not only the ones the receipt layer happens to probe.
    """
    from question_interpretation.lexical import is_filter_subject as _owner
    return _owner(q, start, end)


def named_measure_concepts(question: str) -> List[str]:
    """Distinct measure concepts the question names, in order of appearance."""
    # P1N: a weighting phrase names the STATISTIC, not a second measure.
    q = _statistic.mask_statistic_phrases((question or "").lower())
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
    from .seasoning import (SEASONING_SEGMENT_FIELD,
                            resolve_population_predicate)

    q = (question or "").lower()
    # THE SEASONING ROLE IS NOT DECIDED HERE.
    #
    # This reader used to raise `seasoning_segment` as a dimension whenever the
    # question contained seasoning wording, from the QUESTION TEXT, consulting
    # neither the spec nor the decision anyone else had taken. Two defects came
    # out of that, and they were the same defect:
    #
    #   B15  the parser resolved "back book" to a population and stripped the
    #        field from the spec's dimension slots — correctly — and this raised
    #        a grouping for it anyway, because it never looks at the spec. One
    #        receipt then carried the field as both an applied population and a
    #        lost grouping.
    #   B13  "new lending" named a population no reader on this path knew about,
    #        so this raised a grouping and the answer covered all 11,035 loans.
    #
    # The owner is `seasoning.resolve_population_predicate`. Where it has taken
    # the phrase as a population, no seasoning dimension is raised here at all.
    # Where it has not — a comparison naming both sides, or a book that cannot
    # express the predicate — the dimension stands, because that is exactly when
    # the grouping is what makes the question answerable.
    _population = resolve_population_predicate(question, available_columns)
    _suppress = set(_population or ())
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
        if key in _suppress or (key == SEASONING_SEGMENT_FIELD and _population):
            continue
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
            if key in seen or key in _suppress:
                continue
            if key == SEASONING_SEGMENT_FIELD and _population:
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
                            requested_dimensions: Optional[Sequence[Tuple[str, str]]] = None,
                            resolved_filters: Optional[Iterable[str]] = None
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

    # THE SEASONING POPULATION, RAISED FROM THE OWNER'S DECISION.
    #
    # `requested_dimension_terms` no longer raises a seasoning DIMENSION where
    # the owner has taken the phrase as a population. Suppressing it there and
    # raising nothing here would answer correctly and record nothing: "balance
    # of the front book" narrowed to 1,177 loans with an empty receipt, which
    # trades a wrong claim for no claim.
    #
    # Consuming the decision means raising the RIGHT facet, not none. The
    # population is stated as the governed predicate it is, so the receipt names
    # `months_on_book le 1` for "new lending" rather than a segment the question
    # never mentioned.
    seasoning_population: List[RequestedFacet] = []
    try:
        from .population import Predicate
        from .seasoning import resolve_population_predicate
        # D6: the BOOK's columns, not the loaded frame's. The seasoning owner is
        # answering a SCHEMA question — can this book express `seasoning_segment`
        # — and `requested_dimension_terms` above asks it the same question with
        # the book's columns. Reading the frame here made the two disagree on a
        # projected view: one suppressed the grouping because the owner had taken
        # the phrase, the other raised no population because the projection lacks
        # the column, and the field left the receipt entirely. A narrowed question
        # then returned the whole-book run rate with nothing recorded, which is
        # worse than the false message D6 set out to fix.
        _columns = sorted(book_columns(frame)) if frame is not None else None
        _predicate = resolve_population_predicate(question, _columns)
    except Exception:                                          # pragma: no cover
        _predicate = None
    for _field, _condition in (_predicate or {}).items():
        if isinstance(_condition, Mapping):
            _p = Predicate(_field, str(_condition.get("op") or "eq"),
                           _condition.get("value"))
        else:
            _p = Predicate(_field, "eq", _condition)
        seasoning_population.append(RequestedFacet(
            kind=KIND_POPULATION, label="the population %s" % _p.describe(),
            field_key=_field))
    facets: List[RequestedFacet] = []
    facets.extend(_detect_stress(q))
    facets.extend(_detect_thresholds(q))
    facets.extend(_detect_geographic_scope(q, geographic_values(frame, semantics)))
    # B16a. The same idea on every OTHER dimension, and it needs the sentence's
    # mark where geography does not: geography values do not collide with
    # dimension words and general dimension values do — `Broker` is a value of
    # `origination_channel` and also the word for the dimension.
    #
    # Fields the seasoning owner has taken are skipped, because a narrowing
    # somebody RESOLVED is not lost, and raising both would put the same decision
    # on the receipt twice.
    #
    # `resolved_filters` is the PARSER'S answer, consumed rather than competed
    # with: a field it put in `spec.filters` was resolved into a predicate, so
    # the narrowing is not lost and the role owner reclassifies it into a
    # population downstream. Without this the two owners claimed the same field
    # and the population facet vanished — the D2 carriage tests caught it.
    # B22. A governed scope the question RULED OUT, which no owner can express.
    #
    # "What is the balance excluding the acquired book?" answered over the
    # acquired book — the reader ruled out a cohort and received only that
    # cohort. The lens now DECLINES rather than selecting the opposite, because
    # inferring "therefore the direct book" is a guess about scope and a guess
    # about scope is a substitution.
    #
    # Declining alone would silently widen the answer to include the excluded
    # cohort, which is the fail-open direction honour-or-clarify forbids. So the
    # exclusion is recorded as the narrowing it is, in B16a's kind, unchanged.
    #
    # The PHRASE and the fact both come from `portfolio_lens`, which owns the
    # scope vocabulary. This raises the facet; it decides nothing.
    try:
        from .portfolio_lens import disclaimed_scope_phrase as _disclaimed
        _ruled_out = _disclaimed(question)
    except Exception:                                          # pragma: no cover
        _ruled_out = None
    if _ruled_out:
        facets.append(RequestedFacet(
            kind=KIND_LOST_NARROWING, label=_ruled_out,
            reason=("this scope was ruled out of the question and no governed "
                    "capability can express an exclusion, so the figure covers "
                    "it rather than leaving it out")))
    facets.extend(_detect_lost_narrowing(
        q, dimension_values(frame, semantics),
        taken=set(_predicate or ()) | set(resolved_filters or ()) | {
            f.field_key for f in _detect_geographic_scope(
                q, geographic_values(frame, semantics)) if f.field_key}))
    facets.extend(_detect_comparison_period(q))
    facets.extend(_detect_ranking(q, list(requested_dimensions or [])))
    facets.extend(_detect_contribution(q))
    if _RELATIONSHIP_RE.search(q):
        facets.append(RequestedFacet(
            kind=KIND_RELATIONSHIP,
            label="one measure relative to another"))
    # B21 — THE FOURTH READER. A forecast word the sentence RULED OUT was still
    # raising a requested-projection facet, so "the balance by vintage, ignoring
    # the forecast" computed the right number over the right 11,035 loans and
    # was then refused for not having applied the projection it declined. The
    # honour-or-clarify guard was working exactly as designed on a request that
    # had never been made.
    from .portfolio_lens import is_disclaimed_span as _disclaimed_span
    _projection = _PROJECTION_RE.search(q)
    if _projection and not _disclaimed_span(q, _projection.start()):
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
    requested_statistic = _statistic.statistic_named(
        question, grouped=bool(requested_dimensions))
    if requested_statistic:
        facets.append(RequestedFacet(
            kind=KIND_STATISTIC,
            label=f"the {_statistic.label(requested_statistic)}",
            concepts=(requested_statistic,)))
    for slot in _unresolved_measure_slots(q, semantics, frame):
        # LOST at construction: the parser never resolved these words, so no
        # execution could have honoured them and there is no evidence to weigh.
        facets.append(RequestedFacet(
            kind=KIND_UNRESOLVED_MEASURE, label=slot, status=LOST,
            reason=("this was asked for alongside measures that were "
                    "calculated, but it is not a governed measure in this "
                    "dataset, so it was not calculated")))
    # A field the SELECTOR owner has claimed is not also raised as a breakdown.
    #
    # B15's shape, which `7c46f81` closed for the seasoning vocabulary and which
    # a second selector owner would have reopened: "balance where account status
    # is active" raised `lost_narrowing account_status LOST` and
    # `grouping_dimension account_status APPLIED` on one receipt — the same field
    # asserted both ways, one of them wrong whichever way the reader reads it.
    #
    # This CONSUMES the narrowing owner's answer; it takes no decision of its
    # own. Same treatment `requested_dimension_terms` already gives a phrase the
    # seasoning owner took.
    _narrowed_fields = {f.field_key for f in facets
                        if f.kind == KIND_LOST_NARROWING and f.field_key}
    for key, term, alts in (requested_dimensions or []):
        if key in _narrowed_fields or any(a in _narrowed_fields for a in alts):
            continue
        facets.append(RequestedFacet(kind=KIND_GROUPING, label=term,
                                     field_key=key, alt_keys=alts))
    return seasoning_population + facets


# --------------------------------------------------------------------------- #
# The receipt
# --------------------------------------------------------------------------- #
_AGGREGATION_LABELS = {
    "weighted_avg": "Weighted-average",
    "avg": "Average",
    "median": "Median",
    "min": "Minimum",
    "max": "Maximum",
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
        # A count with no measure is a ROW count. "Count of" expects a measure
        # to follow it, so a metric-less count rendered as a dangling "Count of
        # ·" — the receipt named an aggregation and then named nothing.
        if (self.aggregation or "") == "count" and not (self.measure or ""):
            head = f"Count of {self.population_label}"
        else:
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
        # P1M: the statistic facet is raised by the word "average", so it is
        # present on almost every measure question. Counting it here would print
        # the caveat on exactly the plain KPI answers this test exists to keep it
        # off, so it does not qualify as a scope-related facet.
        scope_facets = [f for f in self.facets if f.kind != KIND_STATISTIC]
        if (self.parser_confidence and self.parser_confidence != "high"
                and scope_facets):
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
    # A grain the answer could not express is a SUBSTITUTION: a different level
    # was measured from the one asked for, and presented as the answer. Tranche D
    # settled that for periods and this programme settled it for populations;
    # disclosure is not honouring. A coverage LIMIT is a different thing and is
    # not this kind — see `assess`.
    KIND_GRANULARITY,
    # Dropping the population changes WHICH ROWS were counted, so it changes
    # every number in the answer. It can never be a partial disclosure.
    KIND_POPULATION,
    # A substituted statistic IS the number. There is no version of "here is the
    # weighted average, you asked for the median" that is a partial answer.
    KIND_STATISTIC,
    # A narrowing the sentence asked for and execution did not apply changes
    # WHICH ROWS were counted, exactly as a population does. Answering over the
    # whole book and disclosing it underneath is the substitution this contract
    # exists to prevent: "balance where account status is active" returned
    # 11,035 loans.
    KIND_LOST_NARROWING,
})
#: Facets that change the SHAPE of a still-valid answer. A partial answer is
#: acceptable provided the unhonoured facet is named.
SHAPE_FACETS = frozenset({KIND_GROUPING})

#: Verdicts from :func:`assess`.
VERDICT_OK = "ok"
VERDICT_PARTIAL = "partial"
VERDICT_REFUSE = "refuse"
#: A question back to the reader, not an answer and not a refusal. Reached only
#: Returned when the question named a dimension no source gave a role to.
VERDICT_CLARIFY = "clarify"

# --------------------------------------------------------------------------- #
# The unresolved-role default — settled: CLARIFY
# --------------------------------------------------------------------------- #
# `_split_named_dimension_roles` gives a named dimension the role the sentence
# gave it. Where NO source supplies a role — the field is neither in
# `spec.filters` nor on an axis — the question is genuinely ambiguous: "balance
# by borrower type" could mean split the book by it or narrow the book to one
# value of it, and nothing in the sentence or the spec settles which.
#
# Three answers were measured side by side on both surfaces in 8a084c9;
# `docs/mi_stage4_unresolved_role_variants.md` holds the numbers. Summarised:
#
#   grouping    leave it. On the corpus this REFUSES — `_blocks` returns True
#               for any unapplied grouping, so the disclosure path everyone
#               assumed was here is unreachable.
#   population  fall to the blocking side, as 32c263a did. Moved NO verdict on
#               either surface: it changed three refusals' wording and nothing
#               else, and the wording got worse — a field key in user-facing
#               prose, and the loss of "field is unavailable in this dataset",
#               the only part that said WHY.
#   clarify     ask which was meant. Converts those three refusals into three
#               questions and changes nothing else.
#
# CLARIFY, and the measured asymmetry is why: a refusal and a clarification both
# decline to answer, but a clarification hands the reader the next move and a
# refusal does not. The switch that made the three measurable is gone along with
# the two variants that lost, so the side-by-side is not re-runnable — the
# recorded numbers stand as the record.
#
# What clarify does NOT touch, measured rather than assumed: a field with a
# RESOLVED filter role. Those go down the population branch under every variant.
# The refusals e35a01b introduced were of exactly that kind, which is why the
# population hole had to be closed first (b79f400) rather than masked by this.

#: A dimension the question named and no source gave a role to, under the
#: "clarify" variant. Distinct from KIND_GROUPING because it must not be
#: disclosed and must not refuse: it asks.
KIND_UNRESOLVED_ROLE = "unresolved_role"

# --------------------------------------------------------------------------- #
# The closed class: every kind a facet can be MOVED into must have a receiver
# --------------------------------------------------------------------------- #
# Two regressions came from the same precondition, though not from the same
# cause. A facet was RECLASSIFIED after detection into a kind, and on the path
# where that happened nothing could confirm the request had been honoured.
#
#   32c263a  the facet reached a branch that READ IT WRONGLY — a field name
#            compared against governed predicate text. A comparison defect,
#            fixed by comparing governed definitions.
#   e35a01b  the facet reached NO BRANCH AT ALL. `reconcile_facets` had no
#            KIND_POPULATION case, so the facet kept the default LOST status
#            whatever execution had done, and a lost population blocks.
#
# The consequence was the same both times — a correct classification producing
# a refusal — and the cause was not. A better comparison could not have
# prevented an absent branch. What generalises is the precondition, and it is
# narrow enough to close by assertion rather than by vigilance:
#
#     any move into a kind whose target reconciler has no branch to receive it.
#
# This registry names every such target. `test_reclassification_targets.py`
# discovers the moves the code actually performs, requires each to be
# registered, and requires each registered target to be dispatched by the
# reconciler on the path that can produce it — or to carry a reason here for
# why dispatch is not needed. Adding a new reclassification without a receiver
# fails that test rather than a user's question.
#
#: kind -> "" when the reconciler on its producing path MUST dispatch it, or a
#: stated reason why no branch is required.
RECLASSIFICATION_TARGETS: Dict[str, str] = {
    KIND_POPULATION: "",
    KIND_UNRESOLVED_ROLE: (
        "`assess` decides this kind from its KIND and its detection status alone "
        "— a clarification when nothing else blocks, and `_blocks` may claim it "
        "first where the wording names a governed seasoning population, in which "
        "case the refusal wins and says why. Neither reader needs a reconciler to "
        "stamp it, because there is no execution evidence to weigh: the facet "
        "records that the question did not settle a role, which no route can "
        "confirm or deny."),
}


# --------------------------------------------------------------------------- #
# D7 (B12) — THE ONE OWNER OF "was the requested grouping actually applied"
# --------------------------------------------------------------------------- #
# Two owners with two bars. The point-in-time reader required the EXECUTOR to
# name the field it grouped by. The routed reader had a three-rung ladder whose
# last rung read:
#
#     "where the answer is cut by SOME axis this cannot identify, the facet
#      stays applied, and which axis it was remains unproven."
#
# That is the exact inverse of the bar `reconcile_population` holds on the same
# receipt — a facet is APPLIED only when execution reports having applied that
# field — and measured across 593 corpus questions it was not a residue at all.
# EVERY routed grouping claim that stood, stood on it: nineteen of them, because
# routes label their columns for a reader (`area`, `code`, `category`) and never
# by canonical field, so the name-match rung above it fired zero times.
#
# Eleven of the nineteen were true and eight were false, in two different ways:
#
#   concentration_analysis  a breakdown existed and it was the WRONG one. "Show
#                           NNEG exposure by borrower age bucket" returns seven
#                           concentration tables, none of them by age bucket, and
#                           the receipt certified the age-bucket breakdown.
#   risk_limits             there was no dimension axis at all — a limit-test
#                           table with columns test/actual/limit/headroom.
#
# The rung is REMOVED rather than narrowed. A rung that stamps APPLIED on what it
# cannot disprove does not become sound by disproving more; the inversion is the
# defect.
#
# What replaces it is what was already there and unread: `concentration_analysis`
# publishes `workflow.dimension_results` carrying the exact canonical field keys
# it grouped by. The reader ignored it and guessed from a display column called
# `category`.


#: Routes whose grouping axis is fixed by the CAPABILITY rather than chosen per
#: answer, and which publish no per-answer declaration of it. Every entry is a
#: statement about what the route always does, and
#: `test_declared_axes_match_what_the_route_publishes` checks each against a real
#: envelope rather than leaving it as an assertion in a comment.
#:
#: A route absent from here and publishing no declaration proves nothing, and its
#: grouping facets are LOST. That is the intended default.
ROUTE_DECLARED_AXES: Dict[str, Tuple[str, ...]] = {
    # Publishes "Funded exposure by ITL3 area": a geography breakdown, at ITL3
    # grain. Both keys, because the ITL3 areas ARE the geography axis and the
    # LEVEL is a separate decision that `granularity_facets` owns — rt_007 and
    # rt_008 pin exactly that pair.
    "geo_exposure": ("collateral_geography",
                     "geographic_region_collateral_itl3"),
}


def declared_group_fields(envelope: Optional[Dict[str, Any]],
                          route: Optional[str] = None) -> Set[str]:
    """The field keys the ANSWER DECLARES it was cut by. Never inferred.

    Read in this order, and every source is a declaration the route made about
    itself rather than a reading of what it displayed:

      1. the concentration workflow's own `dimension_results` — already on the
         envelope before this existed, and never read;
      2. `metadata.groupedBy`, for routes that state their axis directly;
      3. `rankedMovement.canonicalField`, the declaration this receipt has always
         trusted for rankings;
      4. `ROUTE_DECLARED_AXES`, for capabilities whose axis is fixed.

    An empty set means the answer proves no breakdown. That is not the same as
    proving there was none, and the difference is exactly what the removed rung
    used to elide: a route that declares nothing gets no certification, the way
    a route that reports no narrowing gets no population.
    """
    out: Set[str] = set()
    if not isinstance(envelope, dict):
        return out
    workflow = envelope.get("workflow")
    if isinstance(workflow, dict):
        for entry in (workflow.get("dimension_results") or ()):
            if isinstance(entry, Mapping) and entry.get("field"):
                out.add(str(entry["field"]))
        for key in (workflow.get("dimensions_selected") or ()):
            if key:
                out.add(str(key))
    meta = envelope.get("metadata") or {}
    if isinstance(meta, dict):
        for key in (meta.get("groupedBy") or ()):
            if key:
                out.add(str(key))
    ranked = ranking_evidence(envelope)
    if ranked.get("canonicalField"):
        out.add(str(ranked["canonicalField"]))
    out.update(_two_or_more_populations(analytical_evidence(envelope)))
    out.update(ROUTE_DECLARED_AXES.get(route or "", ()))
    # And the answer's own axis keys, which are a declaration in a weak channel:
    # a result column named by REGISTRY FIELD is execution naming the field, and
    # it is the same evidence the point-in-time path accepts through
    # `result_cols`. Keeping it is what makes the two paths' evidence alike
    # rather than merely similarly strict.
    #
    # Safe to add raw, because `grouping_proven` intersects against the keys the
    # facet resolves to: `area`, `code` and `category` match no registry field
    # and prove nothing, which is precisely why this rung fired zero times across
    # 593 corpus questions and every live certification stood on the rung below
    # it instead.
    out.update(str(k) for k in answer_axis_keys(envelope))
    return out


def _two_or_more_populations(plan: Optional[Mapping[str, Any]]) -> Set[str]:
    """Fields an analytical plan cut the answer by, from the populations it declares.

    A plan that measured TWO OR MORE governed populations of the same field has
    cut the answer by that field. "How does the front book compare with our
    older lending" declares

        narrowedTo: [seasoning_segment = Front Book (1,177 rows),
                     seasoning_segment = Back Book (9,858 rows)]

    and that comparison IS the breakdown, with two groups.

    Two or more, never one, and the threshold is the whole rule: narrowing to a
    SINGLE population is a filter — it is what `row_population` records — and
    counting it here would certify a breakdown of an answer reporting one group.
    That is the axis-or-filter distinction D2 owns, honoured at the evidence
    boundary rather than re-decided.
    """
    per_field: Dict[str, Set[str]] = {}
    for entry in ((plan or {}).get("narrowedTo") or ()):
        if isinstance(entry, Mapping) and entry.get("field"):
            per_field.setdefault(str(entry["field"]), set()).add(
                str(entry.get("value")))
    return {field for field, values in per_field.items() if len(values) > 1}


def grouping_proven(facet: RequestedFacet, declared: Optional[Iterable[str]],
                    fields: Optional[Mapping[str, Any]] = None) -> bool:
    """True only where execution NAMED a field this facet resolves to.

    THE decision, for both paths. The two callers differ in what they supply as
    `declared` — the executor's `group_field_keys` on one, the route's
    declaration on the other — and in what they do afterwards with a facet this
    returns False for, because whether the BOOK could express the field is D6's
    decision and is deliberately not taken here.

    Every key the facet resolves to is considered, and every key's canonical
    field: a generic term resolves to different concrete fields depending on the
    book, and execution may legitimately name any of them.
    """
    declared_keys = {str(k) for k in (declared or ()) if k}
    if not declared_keys:
        return False
    registry = fields or {}
    candidates = list(facet.satisfied_by())
    canonicals = [(registry.get(k, {}) or {}).get("canonical_field", k)
                  for k in candidates]
    return bool(declared_keys & set(candidates)) or \
        bool(declared_keys & {c for c in canonicals if c})


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


def executed_statistics(query_result: Any) -> List[str]:
    """The statistics execution reports having run.

    Read from the executor's declared measure set where there is one, and
    otherwise from the aggregation the executor published for the branch it
    dispatched on. Never from the question, and never from the field's default —
    those are the two things the request has to be checked AGAINST.
    """
    meta = getattr(query_result, "metadata", None) or {}
    executed = [str((m or {}).get("aggregation") or "")
                for m in (meta.get("measures_executed") or [])]
    executed = [e for e in executed if e]
    if executed:
        return executed
    published = meta.get("aggregation")
    return [str(published)] if published else []


# --------------------------------------------------------------------------- #
# D2 — THE ONE OWNER OF "axis or filter"
# --------------------------------------------------------------------------- #
# Four readers had an opinion about whether a named dimension is a breakdown or
# a selector, and the census (9ee0e5b) found them disagreeing on 37 of 693
# questions:
#
#   1  `_deterministic_parse`            writes spec.filters / spec.dimensions
#   2  `_split_named_dimension_roles`    reads reader 1's slots — point-in-time only
#   3  `reconcile_routed_facets`         reads the ANSWER's axes — routed only
#   4  `requested_dimension_terms`       raises EVERY named dimension as a
#                                        grouping, deciding nothing, and so
#                                        asserting "axis" by default
#
# Reader 4 is the one the census missed, and on the routed path it is the
# operative decision: reader 2 never runs there, so every dimension the question
# names is asserted to be a breakdown and reader 3 stamps that assertion.
#
# The sources are ordered here, once, and the first that answers wins. What is
# NEW is source 3: `lexical.grouping_cut` and `lexical.is_filter_subject` are the
# declared lexical owners of the sentence's own role markers — "balance BY
# region" marks an axis, "loans FOR joint borrowers" marks a selector — and until
# now the role decision consulted neither. A parser that failed to fill a slot
# made the question look ambiguous when the sentence was not.
#
# Reader 3 is deliberately NOT a source. Execution evidence settles STATUS, never
# ROLE. Letting the answer's axes decide what the question meant is how a false
# APPLIED becomes a role: `risk_limits` publishes no dimension axis at all and its
# receipts still certify breakdowns by region and by account status. That is D7's
# defect (B12) and is fixed there, not laundered into a role here.
#
#: A dimension the sentence or the spec used to SELECT rows.
ROLE_FILTER = "filter"
#: A dimension the sentence or the spec used to SPLIT the answer.
ROLE_AXIS = "axis"
#: No source settled it. The point-in-time path asks; see KIND_UNRESOLVED_ROLE.
ROLE_UNRESOLVED = "unresolved"


def _named_term_span(question: Optional[str],
                     term: Optional[str]) -> Optional[Tuple[int, int]]:
    """Where the question named this dimension, or None.

    Recovered rather than carried. `requested_dimension_terms` returns the
    matched TERM but no offsets, so every dimension facet reaches here with
    ``span=None`` — checked, not assumed, across all 693 corpus questions. The
    term was matched against the lowercased question, so it is found the same
    way, and the offsets are over the ORIGINAL text because that is what the
    lexical owners read.
    """
    text = (question or "")
    needle = str(term or "").strip().lower()
    if not text or not needle:
        return None
    start = text.lower().find(needle)
    if start < 0:
        return None
    return (start, start + len(needle))


def dimension_role(facet: RequestedFacet, *, question: Optional[str],
                   filters: Mapping[str, Any], axes: Set[str],
                   columns: Optional[Set[str]] = None,
                   fields: Optional[Mapping[str, Any]] = None
                   ) -> Tuple[str, Optional[str]]:
    """``(role, filter_key)`` for one named dimension. THE decision, in one place.

    Sources in order; the first that answers wins.

      1. reader 1 put a candidate key in ``spec.filters``          -> FILTER
      2. reader 1 put a candidate key on an axis                   -> AXIS
      3. the SENTENCE marked it, via the lexical owners            -> AXIS
      4. the BOOK cannot express any candidate key                 -> AXIS
      5. otherwise                                                 -> UNRESOLVED

    EVERY key that legitimately satisfies the facet is considered, not just the
    one it is filed under. A generic term resolves to different concrete fields
    depending on what the book carries — "region" is `collateral_geography` on a
    real tape and `geographic_region_obligor` on the generated fixture — and
    reader 1 may have slotted EITHER.

    Source 3 answers AXIS only, and the asymmetry is deliberate rather than an
    omission. A grouping facet reclassified to a filter must carry a resolved
    PREDICATE — `_analytical_population_satisfies` splits the label on the field
    name to recover the value, and a population facet with no value accepts any
    predicate naming that field (B5). Reader 1 is the only source of a resolved
    value, so where the sentence marks a selector reader 1 did not resolve, there
    is a narrowing that was requested and lost and no way to name it. Reporting
    that honestly needs a facet kind this commit does not introduce; until then it
    falls to UNRESOLVED, which is where it already fell.

    Source 4 is reader 2's rule, unchanged: a field the BOOK cannot express has
    no role worth settling. Whether the reader meant to split by broker channel
    or narrow to one, a tape with no broker column can do neither, and "did you
    mean a breakdown or a filter?" invites an answer that cannot be honoured
    either way. Left an axis, so the KIND_GROUPING branch stamps it UNAVAILABLE
    and the reader is told the thing they can act on.
    """
    candidates = list(facet.satisfied_by())
    filter_key = next((k for k in candidates if k in (filters or {})), None)
    if filter_key is not None:
        return ROLE_FILTER, filter_key
    if any(k in (axes or ()) for k in candidates):
        return ROLE_AXIS, None

    span = _named_term_span(question, facet.label)
    if span is not None:
        cut = _lexical.grouping_cut(question or "")
        if cut is not None and span[0] >= cut:
            return ROLE_AXIS, None

    if columns:
        registry = fields or {}
        expressible = any(
            (registry.get(k, {}) or {}).get("canonical_field", k) in columns
            for k in candidates)
        if not expressible:
            return ROLE_AXIS, None
    return ROLE_UNRESOLVED, None


def _split_named_dimension_roles(facets: Sequence[RequestedFacet], spec,
                                 semantics: Optional[dict] = None,
                                 available_columns: Optional[Iterable[str]] = None,
                                 question: Optional[str] = None,
                                 settle_unresolved: bool = True
                                 ) -> List[RequestedFacet]:
    """Rewrite each named-dimension facet into the role its OWNER assigned.

    ``KIND_GROUPING`` has meant "a dimension the question named", with no
    grouping-versus-filter distinction. "balance by region for joint borrowers"
    raises it for the axis and for the filter alike, and the reader is told
    borrower type was a breakdown when it was a selector.

    **This reader does not decide the role.** ``dimension_role`` does, for both
    paths, and this consumes its answer. Only a POSITIVELY identified filter
    moves; an axis is left exactly as it is.

    ``settle_unresolved`` is the only difference between the two callers, and it
    is about the CONSEQUENCE of the decision rather than the decision. The
    point-in-time path turns an unresolved role into a question back; the routed
    path leaves the facet alone until D7 repairs the evidence its status is
    stamped from.

    Where this differs from 32c263a, deliberately
    ---------------------------------------------
    That commit assigned GROUPING only where the words justified it and let
    everything else fall to POPULATION — "the side that blocks" — and the
    over-assignment cost 160 runs. Here a dimension no source assigned a role to
    stays exactly as it is. 32c263a fell to the blocking side when unsure; this
    stays put when unsure.

    Labels are rebuilt in the population form
    -----------------------------------------
    A population facet's label MUST contain its field name. The literal arm of
    ``_analytical_population_satisfies`` derives the value it wants by splitting
    the label on the field name, and an empty value there accepts any predicate
    naming that field — including the wrong population (B5). Reclassifying
    without relabelling is what would make that latent defect live, so the label
    is rebuilt through the same ``Predicate.describe`` the population facets
    already use.
    """
    from .population import Predicate

    if isinstance(spec, dict):
        filters = dict(spec.get("filters") or {})
        axes = list(spec.get("dimensions") or [])
        if spec.get("dimension"):
            axes.append(spec.get("dimension"))
    else:
        filters = dict(getattr(spec, "filters", None) or {})
        axes = list(getattr(spec, "dimensions", None) or [])
        if getattr(spec, "dimension", None):
            axes.append(getattr(spec, "dimension"))
    axes = {str(a) for a in axes if a}

    columns = set(available_columns or ())
    fields = (semantics or {}).get("fields", {}) if isinstance(semantics, dict) else {}

    out: List[RequestedFacet] = []
    for facet in facets:
        field = getattr(facet, "field_key", None)
        if facet.kind != KIND_GROUPING or not field:
            out.append(facet)
            continue
        role, filter_key = dimension_role(
            facet, question=question, filters=filters, axes=axes,
            columns=columns, fields=fields)
        if role == ROLE_AXIS:
            out.append(facet)
            continue
        if role == ROLE_UNRESOLVED:
            if not settle_unresolved:
                # The ROUTED path. The decision is the same one — this reader
                # consumes it rather than taking its own — but what a routed
                # answer OWES an unresolved role is not settled here, because
                # settling it needs evidence D7 (B12) is about to repair.
                #
                # Measured before choosing: on the 15 corpus questions that
                # route and diverge, asking instead of answering would turn TEN
                # correct answers into clarifications. That is 32c263a's shape —
                # a role default falling to the side that declines to answer —
                # and the reason this path leaves the facet exactly as it is.
                out.append(facet)
                continue
            # No source supplied a role, and this path can act on that. Rather
            # than answer over a set the question may not have asked for, or
            # refuse a question nobody has been asked to restate, it becomes a
            # question back. `assess` reads this kind directly and never
            # consults its status, which is why it needs no reconciler branch —
            # recorded as an exemption in RECLASSIFICATION_TARGETS.
            out.append(RequestedFacet(
                kind=KIND_UNRESOLVED_ROLE,
                label=facet.label, field_key=field, alt_keys=facet.alt_keys,
                concepts=facet.concepts, status=facet.status,
                reason=facet.reason, span=facet.span))
            continue
        condition = filters[filter_key]
        if isinstance(condition, Mapping):
            predicate = Predicate(filter_key, str(condition.get("op") or "eq"),
                                  condition.get("value"))
        else:
            predicate = Predicate(filter_key, "eq", condition)
        out.append(RequestedFacet(
            kind=KIND_POPULATION,
            label="the population %s" % predicate.describe(),
            field_key=filter_key, alt_keys=facet.alt_keys,
            concepts=facet.concepts, status=facet.status, reason=facet.reason,
            span=facet.span))
    return out


def reconcile_facets(facets: Sequence[RequestedFacet], *, spec, query_result,
                     semantics: dict, available_columns: Optional[Iterable[str]] = None,
                     route: Optional[str] = None,
                     scenario_applied: bool = False,
                     question: Optional[str] = None) -> List[RequestedFacet]:
    """Stamp a status on every requested facet from EXECUTION EVIDENCE.

    Evidence, never prose: a filter counts as applied only when execution
    actually narrowed the frame; a grouping counts as applied only when its
    column is in the executor's group keys or the result columns.
    """
    facets = _split_named_dimension_roles(facets, spec, semantics,
                                          available_columns, question=question)
    # THE DEDUPE RUNS AFTER THE SPLIT, and that ordering is the whole point.
    #
    # `7c46f81`'s ten-minute defect, arriving on the path D8 added a raiser to:
    # "balance by region" with a drill to South East had the drill ledger raise a
    # population AND the role owner reclassify the named `region` grouping into
    # the same one — identical labels, two facets, one stamped and one lost, and
    # the answer refusing itself.
    #
    # Deduping where the drill is raised would have missed it, because the
    # split's copy does not exist yet at that point. Here every raiser has run.
    _seen_claims: Set[Tuple[str, Optional[str], str]] = set()
    _deduped: List[RequestedFacet] = []
    for _facet in facets:
        _claim = (_facet.kind, _facet.field_key, _facet.label)
        if _facet.kind == KIND_POPULATION and _claim in _seen_claims:
            continue
        _seen_claims.add(_claim)
        _deduped.append(_facet)
    facets = _deduped

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

    ran = executed_statistics(query_result)
    for facet in facets:
        if facet.kind == KIND_STATISTIC:
            requested = (facet.concepts or (None,))[0]
            # P1N. A superlative over a GROUPED result is a ranking, not a
            # statistic on the measure: "largest balance by LTV" ranks the
            # bands, it does not ask for the biggest single loan. Read from
            # execution rather than from the question, because the text-side
            # test misses phrasings where the grouping is recognised only by the
            # parser ("largest balance BY LTV" names no dimension term).
            if requested in ("min", "max") and group_keys:
                facet.status, facet.reason = APPLIED, ""
                continue
            # An analytic mode (contribution, share) is not a statistic that can
            # stand in for another, and it carries its own governed guard. An
            # empty ``ran`` means the route published no statistic evidence at
            # all — this facet then has nothing to reconcile and must not refuse
            # on an absence the route never claimed to report.
            if not ran or any(r in _statistic.ANALYTIC_MODES for r in ran):
                facet.status, facet.reason = APPLIED, ""
            elif _statistic.satisfied_by_any(requested, ran):
                facet.status, facet.reason = APPLIED, ""
            else:
                # REJECTED, not LOST: the statistic was considered and a
                # different one ran, which is a reason the reader can be given.
                # It still refuses — a statistic is never a partial disclosure.
                facet.status = REJECTED
                facet.reason = (
                    f"{facet.label} was requested; the calculation ran "
                    + _join([_statistic.label(r) for r in ran] or ["nothing"])
                    + " instead, and no substitute has been presented as the answer")
            continue
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

        elif facet.kind == KIND_LOST_NARROWING:
            # B16a. Proven the way the geographic scope above is proven, and from
            # the same ledger: the executor records the FIELD KEYS it narrowed
            # on, so a narrowing that ran is recorded as honoured rather than
            # refused. LOST at construction — the first design — would have
            # refused every question the parser DOES resolve, which is 32c263a's
            # shape.
            if population_applied(
                    facet, applied_fields=meta.get("applied_filter_fields"),
                    fields=fields):
                facet.status, facet.reason = APPLIED, ""
            elif narrowed and any(str(facet.label).lower() in v
                                  or v in str(facet.label).lower()
                                  for v in values):
                facet.status, facet.reason = APPLIED, ""
            else:
                facet.status = LOST
                facet.reason = (
                    "this narrowing was not applied, so the figure covers the "
                    "whole book rather than only %s" % facet.label)

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

        elif facet.kind == KIND_GRANULARITY:
            # Stamped from what the route REPORTS, not from what was asked.
            # `concepts` carries (asked, reported); a grain the answer expresses
            # is APPLIED, and one it cannot is UNSUPPORTED with the level it did
            # use — which blocks, because a different level measured and
            # presented as the answer is a substitution, not a shortfall.
            asked, reported = (tuple(facet.concepts or ()) + ("", ""))[:2]
            if asked and reported and str(asked).lower() == str(reported).lower():
                facet.status, facet.reason = APPLIED, ""
            else:
                facet.status = UNSUPPORTED
                facet.reason = ("this answer is reported at %s level, not by %s"
                                % (reported or "another", asked or facet.label))

        elif facet.kind == KIND_POPULATION:
            # THE GAP THAT COST TWO REGRESSIONS.
            #
            # Until this branch existed, `reconcile_facets` had nothing to say
            # about a population at all. A KIND_POPULATION facet arriving on the
            # point-in-time path fell through every branch, kept the default
            # status — LOST — and a lost population blocks. Both the 160-run
            # regression and the "newly originated loans" refusal are that same
            # shape: a facet correctly reclassified into a kind the answering
            # route could not confirm it had applied.
            #
            # The bar is `reconcile_population`'s, deliberately: APPLIED only
            # when the EXECUTOR reports having run a predicate against that
            # field. Spec presence is not evidence — a filter the executor
            # dropped is in `spec.filters` exactly as one it ran, and accepting
            # spec presence is how the P1K silent errors passed.
            candidates = facet.satisfied_by()
            canonicals = [(fields.get(k, {}) or {}).get("canonical_field", k)
                          for k in candidates]
            if population_applied(
                    facet, applied_fields=meta.get("applied_filter_fields"),
                    fields=fields):
                facet.status, facet.reason = APPLIED, ""
            elif canonicals and columns and \
                    not any(c in columns for c in canonicals):
                # The BOOK cannot express this predicate, so it narrowed nothing
                # and no figure is being passed off as the narrow one — the same
                # reading `reconcile_population` gives an absent column.
                facet.status = UNAVAILABLE
                facet.reason = "field is unavailable in this dataset"
            else:
                facet.status = LOST
                facet.reason = "the population was not applied to the calculation"

        elif facet.kind in (KIND_GROUPING, KIND_RANKING):
            candidates = facet.satisfied_by()
            canonicals = [(fields.get(k, {}) or {}).get("canonical_field", k)
                          for k in candidates]
            key = facet.field_key
            canonical = canonicals[0] if canonicals else None
            # D7: the same owner both paths consult. This path's evidence is the
            # executor's DECLARED group keys, plus the canonical field appearing
            # as a result column — the bar this path has always held, now stated
            # once instead of twice.
            if grouping_proven(facet, set(group_keys) | set(result_cols), fields):
                facet.status, facet.reason = APPLIED, ""
            elif any(k in (getattr(spec, "filters", None) or {})
                     for k in candidates if k):
                # The concept was honoured as a POPULATION rather than an axis.
                # "The back book" names the population being reported on, so it
                # runs as a filter; read only as a grouping it looked lost, and
                # the answer was disclosed as a partial when it was in fact
                # exactly what was asked for.
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
    "min": "Minimum ", "max": "Maximum ",
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


def _is_seasoning_population(facet) -> bool:
    """True for a seasoning facet, whose loss can never be a mere partial.

    "The back book" NARROWS the population, so dropping it changes the number:
    a question about the seasoned book was answered with a count of the whole
    direct book, disclosed only as a not-applied breakdown. "Front book vs back
    book" is the other case — dropping it loses the comparison that WAS the
    question. Either way the figure that comes back answers something else, so
    seasoning is treated as a subject facet rather than a shape one.
    """
    from . import seasoning as _seasoning

    keys = {facet.field_key} | set(facet.satisfied_by() or ())
    return _seasoning.SEASONING_SEGMENT_FIELD in {k for k in keys if k}


def assess(receipt: ExecutionReceipt, *, substitution: Optional[str] = None,
           semantics: Optional[dict] = None) -> Tuple[str, Optional[str]]:
    """``(verdict, refusal_or_disclosure_message)``.

    * ``VERDICT_REFUSE`` — a facet that changes the number, or that IS the
      subject of the question, did not reach execution. Answering would present
      a confident figure for a different question.
    * ``VERDICT_PARTIAL`` — the answer stands, but a requested facet must be
      named as not applied.
    * ``VERDICT_OK`` — everything material was applied.
    """
    def _blocks(facet) -> bool:
        if facet.status == APPLIED:
            return False
        if facet.kind in (KIND_POPULATION, KIND_GROUPING, KIND_RANKING):
            # HONOUR-OR-CLARIFY, applied to POPULATIONS as well as periods.
            #
            # The rule used to read: a population the route TRIED and could not
            # express is "honest incapacity — it narrowed nothing, so no figure
            # is being passed off as the narrow one", and was therefore
            # disclosed rather than refused.
            #
            # That reasoning does not survive contact with what the reader
            # receives. "How many joint borrowers are there" returned a KPI of
            # 11,035 loans and £1.96bn — the whole book — with a note attached
            # saying the borrower field was unavailable. The note is a warning;
            # the number is the answer, and it answers a different question.
            # Disclosure is not honouring. That was settled for PERIODS in
            # Tranche D ("this year" answered over one month is a clarification,
            # not a narrower answer with a note); the same reasoning applies
            # here, and this is the more dangerous half, because a wrong
            # population produces a plausible number that describes something
            # else entirely.
            #
            # So a requested population, grouping or ranking that did not reach
            # execution BLOCKS, whatever the reason it failed to. The user's
            # question named it; an answer over the broader set is not that
            # question's answer.
            #
            # The risk the old rule guarded — refusing sound questions over
            # predicates the PARSER invented — is real, and is handled where it
            # belongs: facets are derived from the question's own wording, and
            # the ungrounded-facet guard below drops any that the question does
            # not support.
            return True
        return facet.kind in NUMBER_OR_SUBJECT_FACETS or _is_seasoning_population(facet)

    blocking = [f for f in receipt.facets if _blocks(f)]
    if blocking:
        detail = "; ".join(f.disclosure(semantics) for f in blocking)
        return VERDICT_REFUSE, (
            f"I understood that you asked for {_join([f.label for f in blocking])}, "
            f"but that could not be applied to the calculation ({detail}). "
            "I have not substituted a broader figure.")

    # A dimension no source gave a role to. Asked only once nothing else has
    # already settled the outcome.
    #
    # A clarification is worth asking when answering it changes what the reader
    # can get. Where the run would refuse anyway — a contribution the route did
    # not compute, a statistic it substituted, a population it could not apply —
    # the refusal names the actual obstacle and the question does not: "which
    # region contributes most to the weighted average LTV?" answered by a plain
    # weighted-average spec must say so, not ask whether region was meant as a
    # breakdown or a filter. Answering that question would change nothing.
    #
    # So this runs AFTER the blocking test, not before it. The earlier ordering
    # pre-empted six refusals with a question, including one carrying the P0
    # contribution message.
    unresolved = [f for f in receipt.facets
                  if f.kind == KIND_UNRESOLVED_ROLE and f.status != APPLIED]
    if unresolved:
        return VERDICT_CLARIFY, (
            "I could not tell how you meant "
            f"{_join([f.label for f in unresolved])}. Did you want the book "
            "split by it, or narrowed to one value of it? I have not answered "
            "over the whole book in the meantime.")

    if substitution:
        return VERDICT_REFUSE, (
            "I could not answer this as asked: " + substitution +
            ". I have not returned the substituted breakdown.")

    partial = [f for f in receipt.facets
               if f.kind in SHAPE_FACETS and f.status != APPLIED]
    if partial:
        return VERDICT_PARTIAL, (
            "Not applied: " + "; ".join(f.disclosure(semantics) for f in partial) + ".")
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
    "analytical_composition": "Composed governed capabilities",
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


def _analytical_narrowed_to(evidence: Mapping[str, Any],
                            facet: "RequestedFacet") -> bool:
    """Did the analytical plan narrow to the value this facet names?

    Matched on the VALUE the plan declares it filtered on, because the field a
    facet resolves to and the field a plan narrowed on can be two names for the
    same geography (obligor vs collateral). A row count is required: a predicate
    that selected nothing has not scoped the answer to anything.
    """
    wanted = str(facet.label or "").strip().lower()
    if not wanted:
        return False
    for entry in evidence.get("narrowedTo") or ():
        if not isinstance(entry, Mapping) or not entry.get("rows"):
            continue
        if str(entry.get("value") or "").strip().lower() == wanted:
            return True
    return False


def analytical_populations(envelope: Optional[Dict[str, Any]]) -> List[str]:
    """Every row population an ANALYTICAL answer declares it computed over.

    Read from the predicates the findings carry, which is the plan's own report
    of what it narrowed to — the same standard ``_analytical_narrowed_to``
    applies to geography, extended to any governed population.
    """
    if not isinstance(envelope, dict):
        return []
    findings = (envelope.get("metadata") or {}).get("analyticalFindings")
    if not isinstance(findings, list):
        return []
    out: List[str] = []
    for finding in findings:
        if not isinstance(finding, Mapping):
            continue
        for side in ("population", "comparand"):
            block = finding.get(side)
            if isinstance(block, Mapping) and block.get("predicate"):
                out.append(str(block["predicate"]))
    return out


def _governed_population_predicates(facet: "RequestedFacet") -> List[str]:
    """The declared-predicate forms of any GOVERNED lending window this facet names.

    Why this exists
    ---------------
    A population can be governed under one name and executed under a different
    field. "New lending" is defined by configuration as a months-on-book bound —
    ``months_on_book le 1`` — while the term itself resolves to
    ``seasoning_segment``. Comparing the facet's FIELD NAME against the
    predicate text therefore rejects a population the plan genuinely applied,
    which is how a reclassification of these terms cost 160 runs.

    "Front book" survived that only by coincidence: its governed predicate
    happens to name the same field the term resolves to. Resolving the window
    and comparing ITS predicate treats both alike, so neither passes by
    accident.

    Keyed on the facet's WORDING, not on its field key, so acceptance cannot
    depend on the two names coinciding.
    """
    from .population import Predicate
    from .seasoning import lending_windows_named, load_seasoning_config

    text = " ".join(str(part) for part in (facet.label, facet.field_key) if part)
    keys = lending_windows_named(text)
    if not keys:
        return []
    try:
        config = load_seasoning_config()
    except Exception:                                       # pragma: no cover
        return []

    out: List[str] = []
    for key in keys:
        window = config.lending_window(key)
        if window is None:
            continue
        for field, condition in (window.predicate() or {}).items():
            if isinstance(condition, Mapping):
                out.append(Predicate(field, str(condition.get("op") or "eq"),
                                     condition.get("value")).describe())
            else:
                out.append(Predicate(field, "eq", condition).describe())
    return out


def _analytical_population_satisfies(envelope: Optional[Dict[str, Any]],
                                     facet: "RequestedFacet") -> bool:
    """Is this facet's population the one the analytical plan already resolved?

    The Q3 case: a question about the offer pipeline sometimes parsed with an
    explicit ``pipeline_stage = Offer`` filter and sometimes without. The route
    resolves OFFER from the INTENT either way and declares it —
    ``population.predicate = "pipeline_stage = OFFER"``. With the filter present
    the facet was stamped LOST, because the funded-frame narrowing layer had
    nothing to report, and the same question answered on one run and refused on
    the next.

    A filter naming the population the plan already resolved is a NO-OP, not a
    loss. It is accepted only on the plan's own declaration, and only when the
    declared predicate names BOTH the same field and the value the facet asked
    for — so a plan that narrowed to KFI cannot satisfy a request for OFFER.
    """
    declared = [str(p).strip().lower() for p in analytical_populations(envelope)]

    # Governed comparison FIRST. A window the configuration defines is the same
    # population however it is named, so this is the check that treats "new
    # lending" and "front book" alike rather than letting one pass on a name
    # coincidence the other does not have.
    governed = _governed_population_predicates(facet)
    if governed and any(g.strip().lower() in declared for g in governed):
        return True

    field = str(facet.field_key or "").strip().lower()
    if not field:
        return False
    wanted = str(facet.label or "").lower()
    # The value the facet asked for is whatever follows the field in its label
    # ("the population pipeline_stage = Offer").
    value = wanted.split(field, 1)[-1] if field in wanted else ""
    value = re.sub(r"^[\s=:]+", "", value).strip().strip("'\"[]")
    for predicate in analytical_populations(envelope):
        text = predicate.lower()
        if field not in text:
            continue
        if not value or value in text:
            return True
    return False


def analytical_evidence(envelope: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """What the ANALYTICAL CAPABILITY LAYER declares it actually computed.

    Returns ``{}`` for every other route, so nothing about their reconciliation
    changes. Where the block IS present it is stronger evidence than route
    membership: a composite plan compares periods, projects forward or splits a
    book by cohort only for the questions that asked it to, so route identity
    could not tell those apart — and a route set that claimed all three for
    every analytical answer would weaken the guard rather than satisfy it.

    Only a block naming the plan and its capabilities is accepted. A claim
    without them cannot be checked against what the question asked for, which is
    the standard the other readers here already hold routes to.
    """
    if not isinstance(envelope, dict):
        return {}
    block = (envelope.get("metadata") or {}).get("analyticalComposition")
    if not isinstance(block, dict):
        return {}
    if not block.get("intent") or not block.get("capabilities"):
        return {}
    return block


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


def population_facets(spec: Optional[Dict[str, Any]],
                      semantics: Optional[dict] = None) -> List[RequestedFacet]:
    """Raise a facet for every material row population the spec carries.

    Raised from the SPEC rather than from the question text because the spec is
    where the governed resolution already lives: P1I-A, P1J-1 and the filter
    parsers have all had their say by then, so this reads one settled answer
    instead of re-deriving the population a fourteenth time.
    """
    from .population import material_predicates

    out: List[RequestedFacet] = []
    for predicate in material_predicates((spec or {}).get("filters"), semantics):
        facet = RequestedFacet(kind=KIND_POPULATION,
                               label=f"the population {predicate.describe()}",
                               field_key=predicate.field)
        out.append(facet)
    return out


# --------------------------------------------------------------------------- #
# D8 — THE ONE OWNER OF "was this narrowing actually applied"
# --------------------------------------------------------------------------- #
# Three evidence sources, each read in a different place, each scoped to a path
# the others do not run on — so they could not contradict one another, by
# construction of the paths rather than by design. That is what the census meant
# by AGREE-BY-MAINTENANCE, and the maintenance had already lapsed: B16a needed
# the same three sources for a new kind and got them by COPYING the branches,
# which is the standing rule about a consolidation creating a new reader,
# arriving from inside this programme's own work.
#
# One function reads all three, in a fixed order, for every kind that asks.


def population_applied(facet: RequestedFacet, *,
                       applied_fields: Optional[Iterable[str]] = None,
                       ledger: Optional[Mapping[str, Any]] = None,
                       envelope: Optional[Dict[str, Any]] = None,
                       analytical: Optional[Mapping[str, Any]] = None,
                       fields: Optional[Mapping[str, Any]] = None) -> bool:
    """True only where EXECUTION reports having narrowed on this facet's field.

    The three sources, in order, and every one of them is execution declaring
    what it did rather than the receipt inferring it:

      1. ``applied_fields`` — the point-in-time executor's `applied_filter_fields`;
      2. ``ledger`` — a route's ``metadata.populationApplied``, the seam through
         which the governed population is resolved once and handed to the route;
      3. the analytical plan's declaration, through whichever of its two
         accessors fits the facet's LABEL — and the caller chooses by passing
         ``envelope`` or ``analytical``, because the two are not
         interchangeable:

           ``envelope``   -> ``_analytical_population_satisfies``, which reads a
                             population label ("the population X = South East")
                             and requires the plan to have narrowed on BOTH the
                             field and the value.
           ``analytical`` -> ``_analytical_narrowed_to``, which reads a bare
                             VALUE label ("South East") and matches on the value
                             alone, because the field a facet resolves to and the
                             field a plan narrowed on can be two names for the
                             same geography.

         Passing a population facet through the second silently fails to match.
         Stated here because it did, in this commit, in a test.

    Order matters only for cost: any one of them is sufficient and none
    outranks another, because all three are the same claim made by whichever
    layer ran.

    The bar is `reconcile_population`'s and is deliberately unchanged: a route
    that reports nothing leaves the facet unproven. Spec presence, route identity
    and answer wording are not evidence — that is how the P1K silent errors
    passed.
    """
    keys = {str(k) for k in (facet.satisfied_by() or ())}
    if getattr(facet, "field_key", None):
        keys.add(str(facet.field_key))
    registry = fields or {}
    keys |= {str((registry.get(k, {}) or {}).get("canonical_field", k))
             for k in list(keys)}
    if keys & {str(f) for f in (applied_fields or ())}:
        return True
    declared = {str(a).split(" ")[0] for a in ((ledger or {}).get("applied") or ())}
    if keys & declared:
        return True
    if envelope is not None and _analytical_population_satisfies(envelope, facet):
        return True
    if analytical and _analytical_narrowed_to(analytical, facet):
        return True
    return False


def population_ledger(envelope: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The narrowing ledger a routed envelope carries, or ``{}``."""
    if not isinstance(envelope, dict):
        return {}
    ledger = (envelope.get("metadata") or {}).get("populationApplied")
    return ledger if isinstance(ledger, dict) else {}


def drill_population_facets(extra_filters: Optional[Mapping[str, Any]],
                            semantics: Optional[dict] = None
                            ) -> List[RequestedFacet]:
    """Populations the CALLER supplied, rather than the sentence.

    D8. A drill-through — a UI drilling from a breakdown into one of its groups —
    arrives as `MiQueryRequest.filters` and is merged onto the spec. It is a
    narrowing the reader requested; it is simply not one they typed, so no reader
    of the QUESTION can raise it.

    On the routed path `population_facets(spec)` picks it up, because that ledger
    reads the whole spec. On the point-in-time path nothing did, and the
    consequence was a narrowing that RAN and went unrecorded:

        "What is the balance of the front book?" + {collateral_geography: South East}
          -> "Seasoning Segment = Front Book · South East · 278 loans"   (from 1,177)
          -> receipt: one population facet, for the seasoning segment only.

    The same drill on "balance by region" WAS recorded, because region is named
    and the role owner could reclassify the grouping. So whether a narrowing
    reached the receipt depended on whether the question happened to mention the
    field — not a property of the narrowing at all.

    Why not raise the whole spec here. Measured: raising every `spec.filters`
    entry on the point-in-time path would add a population facet to **81 corpus
    questions**, all of them numeric bounds — "LTV above 50%" — which
    `KIND_THRESHOLD` already represents. Two facets for one predicate is the
    duplicate-claim defect, not a fix for the missing one. The caller's filters
    are precisely the narrowings no other reader can see.
    """
    from .population import material_predicates

    out: List[RequestedFacet] = []
    for predicate in material_predicates(dict(extra_filters or {}), semantics):
        out.append(RequestedFacet(
            kind=KIND_POPULATION,
            label=f"the population {predicate.describe()}",
            field_key=predicate.field))
    return out


def reconcile_population(facets: Sequence[RequestedFacet],
                         evidence: Optional[Dict[str, Any]],
                         dataset_columns: Optional[Iterable[str]] = None) -> None:
    """Stamp population facets from EXECUTION EVIDENCE, in place.

    The bar is deliberately high and deliberately dumb: a facet is APPLIED only
    when the route reports having applied that field. A route that reports
    nothing leaves every population facet LOST, and because the kind is
    number-changing the answer then refuses rather than presenting a whole-book
    figure. Spec presence, route identity and receipt wording are not evidence —
    that was exactly how the P1K silent errors passed.
    """
    applied = {str(a).split(" ")[0] for a in ((evidence or {}).get("applied") or [])}
    unavailable = {str(u).split(" ")[0]
                   for u in ((evidence or {}).get("unavailable") or [])}
    for facet in facets:
        if facet.kind != KIND_POPULATION:
            continue
        expressible = (dataset_columns is None
                       or facet.field_key in set(dataset_columns))
        if population_applied(facet, ledger={"applied": sorted(applied)}):
            facet.status, facet.reason = APPLIED, ""
        elif not expressible:
            # The BOOK cannot express this predicate, so it never narrowed
            # anything and no figure is being passed off as the narrow one —
            # the same reasoning P1I-A used for an absent column, and the same
            # treatment the point-in-time executor already gives an unavailable
            # filter. Disclosed, not refused; refusing here would reject
            # questions over a bogus predicate the parser invented rather than
            # over a population the user could actually have meant.
            facet.status = UNSUPPORTED
            facet.reason = "this book does not carry that field"
        elif facet.field_key in unavailable:
            facet.status = UNAVAILABLE
            facet.reason = ("this analytical route cannot restrict its "
                            "calculation to that population")
        else:
            facet.status = LOST
            facet.reason = ("this analytical route calculated across the whole "
                            "book; it did not narrow to the requested population")


#: Keys in a routed answer's frame that are MEASURES or the time axis, never a
#: categorical breakdown. Everything else is an axis the answer is cut by.
_NON_AXIS_KEYS = frozenset({
    "period", "month", "date", "reporting_date", "reportingdate",
    "value", "values", "balance", "amount", "count", "loans", "share",
    "exposure", "rank", "base", "upside", "downside", "threshold",
    "movement", "delta", "start_value", "end_value", "relative_change",
})
_NON_AXIS_SUFFIXES = ("_sum", "_pct", "_share", "_value", "_count", "_balance")


def answer_axis_keys(envelope: Optional[Dict[str, Any]]) -> Set[str]:
    """Every categorical axis the routed answer's own frame is cut by.

    Read from the artifact rows the route publishes — the same standard
    `declared_series_periods` uses, and the same standard `reconcile_facets`
    applies to the point-in-time result frame via `result_cols`. A routed answer
    was the one place no result frame was consulted, on the belief that there
    was none to consult. There is.
    """
    out: Set[str] = set()
    if not isinstance(envelope, dict):
        return out
    for artifact in envelope.get("artifacts") or []:
        for row in (artifact.get("rows") or [])[:5]:
            if not isinstance(row, Mapping):
                continue
            for key in row:
                name = str(key).strip().lower()
                if name in _NON_AXIS_KEYS or name.endswith(_NON_AXIS_SUFFIXES):
                    continue
                out.add(name)
    return out


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
    analytical = analytical_evidence(envelope)
    fields = semantics.get("fields", {}) if isinstance(semantics, dict) else {}
    listing = route in LISTING_ROUTES
    declared_axes = declared_group_fields(envelope, route)

    def _canonical(key: Optional[str]) -> Optional[str]:
        if not key:
            return None
        return (fields.get(key, {}) or {}).get("canonical_field", key)

    for facet in facets:
        if facet.kind == KIND_STATISTIC:
            # P1M. Specialist routes publish no statistic evidence — a portfolio
            # comparison declares the measures it compared, not the aggregation
            # each ran under. Refusing on that absence would reject every routed
            # answer to a question containing the word "average".
            #
            # This is safe because the guarantee for these paths is upstream: an
            # ungoverned statistic is refused at the parse boundary and never
            # reaches a route at all. The facet's job on the point-in-time path
            # is to catch a statistic lost INSIDE a measure set, which routes do
            # not build.
            facet.status, facet.reason = APPLIED, ""
            continue
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

        elif facet.kind == KIND_GRANULARITY:
            # Stamped from what the route REPORTS, not from what was asked.
            # `concepts` carries (asked, reported); a grain the answer expresses
            # is APPLIED, and one it cannot is UNSUPPORTED with the level it did
            # use — which blocks, because a different level measured and
            # presented as the answer is a substitution, not a shortfall.
            asked, reported = (tuple(facet.concepts or ()) + ("", ""))[:2]
            if asked and reported and str(asked).lower() == str(reported).lower():
                facet.status, facet.reason = APPLIED, ""
            else:
                facet.status = UNSUPPORTED
                facet.reason = ("this answer is reported at %s level, not by %s"
                                % (reported or "another", asked or facet.label))

        elif facet.kind == KIND_POPULATION:
            # EXTEND, rather than constrain: a filter consistent with the
            # population the plan resolved from the intent is a no-op. Anything
            # else keeps the status the population ledger already stamped.
            if population_applied(facet, ledger=population_ledger(envelope),
                                  envelope=envelope if analytical else None):
                facet.status, facet.reason = APPLIED, ""

        elif facet.kind == KIND_LOST_NARROWING:
            # The routed twin, proven from the plan's own declaration — the same
            # evidence the geographic branch below uses. This is what makes a
            # COMPARISON safe with no lexical rule at all: a plan that narrowed
            # to both sides declares both, and both facets are stamped applied.
            if population_applied(facet, ledger=population_ledger(envelope),
                                  analytical=analytical):
                facet.status, facet.reason = APPLIED, ""
            elif listing:
                facet.status = UNSUPPORTED
                facet.reason = ("this answer covers the whole book rather than "
                                f"narrowing to {facet.label}")
            else:
                facet.status = LOST
                facet.reason = (
                    "this narrowing was not applied, so the figure covers the "
                    "whole book rather than only %s" % facet.label)

        elif facet.kind == KIND_GEOGRAPHIC_SCOPE:
            if analytical:
                # A composite plan may answer a two-place question by measuring
                # BOTH places as governed populations. Proven from the row
                # predicates it declares it narrowed to, with their row counts —
                # a claim without a narrowed population is still lost.
                if _analytical_narrowed_to(analytical, facet):
                    facet.status, facet.reason = APPLIED, ""
                else:
                    facet.status = LOST
                    facet.reason = ("this analytical plan did not narrow to "
                                    f"{facet.label}")
            elif listing:
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
            if analytical:
                # Proven per answer: the layer names the two reporting dates it
                # compared, so a plan that ran point-in-time capabilities only
                # is still reported as losing the facet.
                if len(analytical.get("periodsCompared") or ()) >= 2:
                    facet.status, facet.reason = APPLIED, ""
                else:
                    facet.status = LOST
                    facet.reason = ("this analytical plan measured one point in "
                                    "time; no period comparison was calculated")
            elif route in TEMPORAL_ROUTES:
                facet.status, facet.reason = APPLIED, ""
            else:
                facet.status = LOST
                facet.reason = ("the answer is a single point in time; no period "
                                "comparison was calculated")

        elif facet.kind == KIND_PROJECTION:
            if analytical:
                if analytical.get("projected"):
                    facet.status, facet.reason = APPLIED, ""
                else:
                    facet.status = LOST
                    facet.reason = ("this analytical plan ran no forward "
                                    "projection")
            elif route in PROJECTION_ROUTES:
                facet.status, facet.reason = APPLIED, ""
            else:
                facet.status = LOST
                facet.reason = ("this is a point-in-time answer; no forward "
                                "projection was run")

        elif facet.kind == KIND_COHORT_COMPARISON:
            wanted = tuple(facet.concepts or ())
            executed_concept = ((analytical.get("cohortConcept") if analytical
                                 else None)
                                or (compared.get("cohortConcept")
                                    if compared else None))
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
            elif analytical:
                # The layer states the populations it compared and the measures
                # it compared them on. Both are required: two populations with
                # nothing measured about the difference is not a comparison.
                if (len(analytical.get("populationsCompared") or ()) >= 2
                        and analytical.get("measuresCompared")):
                    facet.status, facet.reason = APPLIED, ""
                else:
                    facet.status = LOST
                    facet.reason = ("this analytical plan did not compare two "
                                    "governed populations on a measure")
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

        elif facet.kind in (KIND_MULTI_MEASURE, KIND_RELATIONSHIP) and analytical:
            # A composite plan CAN express one measure relative to another: it
            # measures both sides and states the difference between them. Proven
            # from the comparison it declares, never from route identity.
            if (analytical.get("measuresCompared")
                    and len(analytical.get("populationsCompared") or ()) >= 2):
                facet.status, facet.reason = APPLIED, ""
            else:
                facet.status = UNSUPPORTED
                facet.reason = ("this analytical plan produced a single measure, "
                                "so the full question was not expressed")

        elif facet.kind in (KIND_MULTI_MEASURE, KIND_RELATIONSHIP):
            facet.status = UNSUPPORTED
            facet.reason = ("this governed capability returns a single measure, "
                            "so the full question was not expressed")

        elif facet.kind in (KIND_GROUPING, KIND_RANKING):
            canonicals = [_canonical(k) for k in facet.satisfied_by()]
            canonical = canonicals[0] if canonicals else None
            # D7: the same owner, this path's evidence being what the ROUTE
            # declared it grouped or ranked by. A route that declares nothing
            # proves nothing — the bar `reconcile_population` holds two branches
            # above, now held here too.
            if grouping_proven(facet, declared_axes, fields):
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
                # THE FALSE APPLIED, both halves of it, now closed.
                #
                # The first half read: "the route grouped by something it
                # declared; without a result frame we cannot disprove it." It
                # stamped APPLIED on anything it could not DISPROVE, and the
                # premise was false — `evolution` publishes a frame whose rows
                # carry `period` and `value` and nothing else, so "balance by
                # month BY REGION" returned the whole-book series with the
                # receipt vouching for a regional breakdown never computed.
                #
                # The second half survived that fix as a stated residue: where
                # the answer was cut by SOME axis the reader could not identify,
                # the facet stayed applied. Measured, it was not a residue —
                # NINETEEN of nineteen live certifications stood on it, because
                # routes label columns for a reader (`area`, `code`, `category`)
                # and never by canonical field, so the name-match rung above it
                # fired zero times across 593 questions. Eight of the nineteen
                # were false: four certified a breakdown by a dimension the
                # route had not used, and four certified one over an answer with
                # no dimension axis at all.
                #
                # Both are gone. Evidence is now a DECLARATION and nothing else
                # — see `declared_group_fields`, which reads the
                # `dimension_results` the concentration workflow was already
                # publishing and nobody read. What is left here is the honest
                # negative, worded for BOTH readings of the facet: the kind is
                # GROUPING but the term is often a population — "new lending",
                # "the front book" — so "not broken down by new lending" would
                # assert a breakdown claim the reader never made.
                facet.status = LOST
                facet.reason = (
                    "this answer covers the whole population; it is neither "
                    f"narrowed to nor broken down by {facet.label}")
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

# --------------------------------------------------------------------------- #
# The TIME axis, as an axis in its own right
# --------------------------------------------------------------------------- #
#: The grain each series-publishing route reports at. Every one of these reads
#: the governed month-end funded snapshots, so every one publishes MONTHS, and
#: a question asking for weeks or days receives months.
#:
#: Declared here rather than inferred from the answer, for the same reason
#: `_ROUTE_GRANULARITY` is: a route's reporting grain is a property of the
#: capability, and reading it back out of the prose would be the receipt
#: checking the question against itself.
_ROUTE_TIME_GRAIN = {
    "evolution": "month",
    "evolution_funnel": "month",
    "evolution_pipeline_stage": "month",
    "period_movement": "month",
    "period_change_analysis": "month",
    "temporal_compare": "month",
    "cohort_progression": "month",
    "cohort_conversion": "month",
    "forecast_extrapolation": "month",
    "funded_bridge": "month",
}


def route_time_grain(route: Optional[str]) -> Optional[str]:
    """The grain the route's series is published at, or None if it publishes none."""
    return _ROUTE_TIME_GRAIN.get(route or "")


def time_axis_disclosure(unit: Optional[str], route: Optional[str]
                         ) -> Optional[RequestedFacet]:
    """A facet for the TIME grain a question named, or None.

    Why this is not a `KIND_GROUPING`
    ---------------------------------
    A time grain is not a registry field. There is no `week` column to be
    applied or unavailable, and `satisfied_by()` has nothing to resolve, so
    every reader that works by matching a field key against group keys, result
    columns or spec filters is answering a question about the wrong object. That
    is why the facet layer raised nothing for a time axis in any of the 24
    time-series probes the inventory ran: it had no way to say what it had seen.

    It is a `KIND_GRANULARITY` — the level an answer is reported at — carrying
    `(asked, reported)` on `concepts`, so the same reconciler branch that
    adjudicates a spatial grain adjudicates a temporal one. The two are the same
    kind of claim about an answer, and one implementation is the point of the
    programme.

    Returns None when the route publishes no series, so a point-in-time KPI is
    never told it failed to honour a grain nobody could have expected it to.
    """
    if not unit:
        return None
    grain = route_time_grain(route)
    if not grain:
        return None
    return RequestedFacet(kind=KIND_GRANULARITY, label=unit,
                          concepts=(unit, grain))


def granularity_facets(question: str, route: Optional[str]
                       ) -> List[RequestedFacet]:
    """Every reporting GRAIN this question names, spatial and temporal.

    THE ONE PLACE THE GRAIN IS READ.

    The inventory found eleven distinct entry points reading the raw question,
    and the subject-side clause split implemented three times from vocabularies
    that agreed by maintenance rather than construction. The remedy is not a
    better copy: `requested_unit` is called here, once, and every route inherits
    the result through the facet it produces. Duplicating the call into
    `evolution` — the obvious short path to making time-series questions work —
    would be a twelfth reader and would defeat the premise.

    `period_request.requested_unit` delegates to the single lexical owner, so
    the vocabulary has one definition too.
    """
    from . import period_request as _period_request

    out: List[RequestedFacet] = []
    spatial = granularity_disclosure(question, route)
    if spatial is not None:
        out.append(spatial)
    temporal = time_axis_disclosure(_period_request.requested_unit(question),
                                    route)
    if temporal is not None:
        out.append(temporal)
    return out


def granularity_disclosure(question: str, route: Optional[str]
                           ) -> Optional[RequestedFacet]:
    """A facet for the reporting GRAIN this question names, or None.

    Raised whether or not the route can honour it — the change that makes the
    facet layer able to record a request that SUCCEEDED, not only one that
    failed.

    What this used to do, and why it was not enough
    ----------------------------------------------
    It returned a facet only on a MISMATCH, filed as ``KIND_GROUPING``, with
    ``status=UNAVAILABLE`` written at detection. Three consequences, all of
    which had to go together:

    * a reporting level was described to the reader as a breakdown;
    * the status was decided before execution, so no reconciler adjudicated it
      and no evidence was ever weighed;
    * a grain the route DID honour raised nothing, so nothing downstream could
      act on a correct request. A rule can only be enforced on a request that
      is represented.

    The status is now left at the default and stamped from what the route
    reports, in both reconcilers. The comparison the reconciler needs travels on
    ``concepts`` as ``(asked, reported)``.
    """
    pair = _ROUTE_GRANULARITY.get(route or "")
    if not pair:
        return None
    asked, reported = pair
    for grain in (asked, reported):
        if re.search(r"\b" + re.escape(grain) + r"s?\b", question or "", re.I):
            return RequestedFacet(kind=KIND_GRANULARITY, label=grain,
                                  concepts=(grain, reported))
    return None


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


def declared_series_periods(envelope: Dict[str, Any]) -> Optional[int]:
    """How many reporting periods the route's OWN series contains.

    Read from the periods the artifacts carry, which is the route reporting what
    it delivered — never from the prose, which would be the receipt checking the
    answer's sentence against the question's.
    """
    if not isinstance(envelope, dict):
        return None
    for artifact in envelope.get("artifacts") or []:
        periods = {str(row.get("period"))
                   for row in (artifact.get("rows") or [])
                   if isinstance(row, Mapping) and row.get("period")}
        if periods:
            return len(periods)
    return None


def check_window_coverage(facets: Sequence[RequestedFacet],
                          envelope: Dict[str, Any], question: str,
                          route: Optional[str]) -> List[RequestedFacet]:
    """Downgrade a period facet the route answered over a SHORTER WINDOW than asked.

    The coverage limit, kept distinct from the grain substitution
    ---------------------------------------------------------------
    These two look alike and are not the same defect. A grain substitution
    measured a DIFFERENT LEVEL — weeks asked, months delivered. A coverage limit
    measured the right level over FEWER PERIODS than the question named, because
    the book carries no more: "over the last 12 months" against three governed
    snapshots.

    They are owed different sentences, which is why they are read, measured and
    reported apart. `period_request.clarification` names the window and what
    would be needed; `granularity_clarification` names the level. Telling a
    reader their weekly request was answered monthly does not tell them their
    twelve-month window was three, and vice versa.

    They are NOT owed different verdicts, and that correction is worth stating
    because the Stage 5 pre-registration predicted otherwise. A coverage limit
    was expected to be a disclosure. This tree had already settled the opposite
    for every other route that checks one — `period_movement` returns
    ``ok=False`` with the clarification, and this function's older half stamps
    UNSUPPORTED — and the reasoning is `clarification`'s own: *offering the
    narrower window as the answer is the substitution this guard exists to
    prevent*. A shorter window IS a different window. So: apart in cause, apart
    in message, together in verdict.

    Acts only on the route's own declaration of the periods it delivered, so a
    route that reports nothing is never refused on an unverifiable basis.
    """
    span = _period_request_module().requested_span(question)
    if span is None or not route_time_grain(route):
        return list(facets)
    have = declared_series_periods(envelope)
    if have is None or have >= span.periods:
        return list(facets)
    for facet in facets:
        if facet.kind != KIND_COMPARISON_PERIOD or facet.status != APPLIED:
            continue
        facet.status = UNSUPPORTED
        facet.reason = (
            f"you asked about {span.label}, which spans {span.periods} "
            f"reporting period(s); this answer covers {have}, which is all this "
            f"book carries")
        return list(facets)
    # No period facet to downgrade — the question named a window and nothing in
    # the receipt represents it. Raise one, or the shortfall goes unsaid.
    return list(facets) + [RequestedFacet(
        kind=KIND_COMPARISON_PERIOD, label=span.label, status=UNSUPPORTED,
        reason=(f"you asked about {span.label}, which spans {span.periods} "
                f"reporting period(s); this answer covers {have}, which is all "
                f"this book carries"))]


def _period_request_module():
    from . import period_request
    return period_request


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
    # P1N. A governed min/max IS the single extreme value. Before P1N no
    # aggregation could express one, so any superlative answered by a summary was
    # necessarily the whole-book figure for a different question; now it is the
    # answer, and this guard would refuse the very capability that closes the gap.
    if str(getattr(spec, "aggregation", "") or "") in ("min", "max"):
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
