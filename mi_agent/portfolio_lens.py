"""mi_agent.portfolio_lens — source-portfolio lens resolver for the MI Agent.

Maps natural-language portfolio scope ("the acquired book", "direct only",
"acquired_001") onto deterministic filters over the source-provenance fields
stamped at onboarding (``source_portfolio_type`` / ``source_portfolio_id`` —
see engine/provenance.py). The MI Agent answers through three lenses:

  * total     — all rows (direct + acquired), no source filter;
  * direct    — source_portfolio_type == direct;
  * acquired  — source_portfolio_type == acquired;
  * cohort    — source_portfolio_id == <exact id> (e.g. acquired_001).

This module is pure: it resolves a lens from text and applies it to an
:class:`~mi_agent.mi_query_spec.MIQuerySpec` (merging filters + recording the
lens label). It never touches data, so it is trivially unit-testable and works
regardless of which MI entrypoint builds the spec.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SOURCE_TYPE_FIELD = "source_portfolio_type"
SOURCE_ID_FIELD = "source_portfolio_id"

LENS_TOTAL = "total"
LENS_DIRECT = "direct"
LENS_ACQUIRED = "acquired"
LENS_COHORT = "cohort"

# An exact source-cohort id, e.g. direct_001 / acquired_002. Used for
# NATURAL-LANGUAGE detection, where a narrow pattern is what keeps an unrelated
# snake_case token in a question from being read as a portfolio id.
_COHORT_ID_RE = re.compile(r"\b((?:direct|acquired)_\d+)\b", re.IGNORECASE)

# Any source_portfolio_id the provenance contract allows — a lowercase slug (see
# engine/provenance.py::_ID_RE). Used ONLY for an EXPLICIT selection (the UI
# dropdown / API field), where the caller has named a portfolio outright and
# there is nothing to disambiguate. Without this, a managed-service client whose
# portfolio ids do not use the direct_/acquired_ prefix convention (e.g.
# alp_origination) could be listed by /mi/source-portfolios yet not be selectable
# in the chat, which is the inconsistency this pattern removes.
_SELECTABLE_COHORT_ID_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")

# Phrase → lens. Order matters only within a family; matching is keyword-based.
# "origination" and "originated" are deliberately ABSENT, for the same reason the
# bare total terms below are: they match the DIMENSION they name, not a book.
# "show the portfolio by origination channel", "by origination date",
# "originations by region" all had a Direct scope filter applied to a question
# that had asked for a breakdown, so the answer covered only the direct books
# while still describing itself as a share "of the funded book" — the silent
# scope mutation this module exists to prevent. On a three-book platform it
# understated the funded book by a third.
#
# The QUALIFIED forms are kept, because they name the book rather than the
# dimension: "directly originated", "new origination", "newly originated".
# "new lending" and "current book" are also absent, for the same reason: both
# name a MEASURE, not a book. "show new lending by region" asks for originations
# by region; "what is the current book balance?" asks for an aggregation. Each
# was narrowing the answer to the direct books while presenting itself as the
# whole. Asserted by
# mi_agent/tests/test_mi_query_capability_matrix.py::test_measure_vocabulary_is_not_read_as_a_portfolio_scope.
# PROVENANCE ONLY. Every term here must name WHERE A LOAN CAME FROM.
#
# "new origination" and "newly originated" were removed from _DIRECT_TERMS, and
# "back book", "backbook" and "legacy book" from _ACQUIRED_TERMS (P1J-1). Those
# are VINTAGE / SEASONING concepts — they name WHEN a loan was written — and the
# two axes are independent: a directly-originated loan written five years ago is
# DIRECT by provenance and BACK BOOK by seasoning, and an acquired loan may be
# newly originated. Resolving "the back book" to *acquired* silently excluded
# every seasoned direct loan and included every recent acquired one — the same
# class of silent scope mutation as P1I, on a different axis.
#
# Seasoning now has its own governed model (mi_agent/seasoning.py) and its own
# vocabulary, which must not overlap with this one. Asserted by
# tests/test_p1j1_vintage_seasoning.py.
_DIRECT_TERMS = (
    "direct", "directly originated", "organic", "own book", "in-house",
)

_ACQUIRED_TERMS = (
    "acquired", "acquisition", "purchased book",
    "purchased", "bought book", "inorganic", "m&a",
)
# Phrases that name the CONSOLIDATED SCOPE. Every entry must be
# portfolio-qualified. A bare "total" / "overall" / "combined" is deliberately
# absent: "what is the total current balance?" asks for an aggregation, not for
# the Total book, and treating it as a scope silently widened a Direct or
# Acquired workspace selection back to the whole platform — a silent scope
# mutation, and exactly the class of defect the governed context exists to stop.
#
# This family is the EXPLICIT-WIDENING family: naming it overrides a narrower UI
# selection (see resolve_lens_with_default via mentions_portfolio) because the
# caller has asked, in words, for the whole book. That is the opposite of the
# "current / selected book" family (names_selected_scope), which DEFERS to the
# selection.
#
# "sponsored book / portfolio / platform / aum" is a governed Trakt scope
# phrase: it means the sponsor's (client's) FULL AuM across every directly
# originated and acquired portfolio — equivalent to the entire book, NOT the
# direct book and NOT the spv-sponsored cohort. It is therefore part of this
# explicit-widening family. Left out of it, "the sponsored book" was read on the
# LLM path as source_portfolio_type = direct, returning the direct book's number
# for a question about the whole book. (The sponsored SPV cohort is still
# addressable by its portfolio id; the PHRASE "the sponsored book" is the
# whole-client scope.)
_TOTAL_TERMS = (
    "total portfolio", "total book", "total platform", "whole book",
    "whole portfolio", "all loans", "all portfolios", "entire portfolio",
    "entire book", "combined book", "combined portfolio", "consolidated",
    "across the book", "across all portfolios", "across all books",
    "sponsored book", "sponsored portfolio", "sponsored platform",
    "sponsored aum", "sponsored loan book", "sponsor book", "sponsor portfolio",
    "sponsor aum",
)

_COMPARISON_TERMS = (" vs ", " vs. ", " versus ", "compare", "comparison",
                     "side by side", "side-by-side", "against")


# --------------------------------------------------------------------------- #
# GOVERNED SCOPE PHRASES (P1I-A)
# --------------------------------------------------------------------------- #
# A phrase that names the POPULATION BEING REPORTED ON is a scope reference. It
# is not a row predicate, not a grouping axis, and not a place.
#
# The distinction was being lost three different ways, each producing a
# confident wrong answer or a spurious refusal:
#
#   "the entire portfolio"  -> collateral_geography = "Entire"
#   "the current portfolio" -> collateral_geography = "Current"
#   "the acquired portfolio"-> grouped by acquired_portfolio_id
#   "the funded portfolio"  -> filter funded_status = "Funded"   (LLM path)
#
# All four are the same defect: a scope phrase consumed by a resolver that is
# looking for something else. The cure is to CLAIM THE SPAN FIRST, so the
# invalid filter or dimension is never created — never to create one and delete
# it afterwards, which would silently broaden the population.
#
# Only QUALIFIED phrases count. A bare "current" or "entire" is ordinary
# English; it becomes a scope reference when it qualifies a book noun, which is
# what keeps "current LTV" and "current reporting date" out of this vocabulary.
_SCOPE_NOUNS = ("book", "books", "portfolio", "portfolios", "platform",
                "loan book", "aum")

#: Qualifiers that, in front of a book noun, name a governed scope.
#: ``direct``/``acquired`` are deliberately absent here — they are resolved by
#: the lens families above, which already carry their full synonym sets.
_SCOPE_QUALIFIERS = (
    "funded", "unfunded", "whole", "entire", "total", "current", "overall",
    "consolidated", "combined", "selected", "active", "direct", "acquired",
    "purchased", "originated", "sponsored",
)

#: "the funded book", "of the entire portfolio", "in the current portfolio".
#: The optional article and preposition are matched so the whole clause is
#: claimed, not just the two content words.
_SCOPE_PHRASE_RE = re.compile(
    r"\b(?:(?:for|in|of|across|within|on)\s+)?(?:the\s+|our\s+|my\s+|this\s+)?"
    r"(?:" + "|".join(re.escape(q) for q in _SCOPE_QUALIFIERS) + r")\s+"
    r"(?:" + "|".join(re.escape(n) for n in _SCOPE_NOUNS) + r")\b",
    re.IGNORECASE)

#: Phrases naming the CURRENTLY SELECTED portfolio context rather than a fixed
#: book. These defer to the caller's selection; with nothing selected they mean
#: the whole platform, which is what the workspace is showing.
_SELECTED_SCOPE_RE = re.compile(
    r"\b(?:the\s+|this\s+|my\s+|our\s+)?(?:current|selected|active)\s+"
    r"(?:" + "|".join(re.escape(n) for n in _SCOPE_NOUNS) + r")\b",
    re.IGNORECASE)


#: The lens vocabulary's own nouns. `_SCOPE_NOUNS` is narrower — no "lending",
#: no "loans" — and "organic lending" and "directly originated loans" are both
#: governed provenance language.
#:
#: This list was wrong twice before it was right, and both corrections came from
#: tests written years before this change: `_SCOPE_NOUNS` alone dropped "organic
#: lending", and adding "lending" alone still dropped "directly originated
#: loans" and "the purchased back book". **The vocabulary is the fragile part of
#: this fix**, and the only thing that found its edges was coverage someone else
#: had recorded.
_LENS_NOUNS = _SCOPE_NOUNS + ("lending", "originations", "origination",
                              "loans", "loan")

#: Every provenance word, as a QUALIFIER of one of those nouns. Built from the
#: lens families rather than from `_SCOPE_QUALIFIERS`, which is narrower: it has
#: no `bought`, `acquisition` or `organic`, and three pre-existing cases in
#: `test_p1j1_vintage_seasoning` proved it — "the bought book", "the acquisition
#: book" and "organic lending" are all governed provenance phrases that the
#: scope vocabulary does not contain.
_LENS_QUALIFIERS = tuple(dict.fromkeys(
    [t for term in _DIRECT_TERMS + _ACQUIRED_TERMS
     for t in (term, term.split()[0])]))


def _qualified_span_re(qualifiers, nouns):
    """The qualified-mention test, ONCE, over whichever vocabulary is passed.

    B22. One helper, two callers: the scope resolver and the lens resolver ask
    the same question — *is this word qualifying a book noun, or is it ordinary
    English?* — of different vocabularies. Duplicating the test would create a
    second owner of the decision B22 exists to consolidate; hard-coding one
    vocabulary would have silently dropped three governed provenance phrases.
    """
    return re.compile(
        r"\b(?:(?:for|in|of|across|within|on)\s+)?(?:the\s+|our\s+|my\s+|this\s+)?"
        r"(?:" + "|".join(re.escape(q) for q in
                          sorted(qualifiers, key=len, reverse=True)) + r")\s+"
        # One optional adjective between the qualifier and the noun, because
        # "the purchased BACK book" is governed provenance language. Bounded to
        # a single short word and a NOUN is still required, so "purchased at
        # auction" — qualifier, two words, no noun — does not match.
        r"(?:\w{1,8}\s+)?"
        r"(?:" + "|".join(re.escape(n) for n in
                          sorted(nouns, key=len, reverse=True)) + r")\b",
        re.IGNORECASE)


_LENS_PHRASE_RE = _qualified_span_re(_LENS_QUALIFIERS, _LENS_NOUNS)


def lens_phrase_spans(text: Optional[str]):
    """``((start, end), ...)`` for every governed PROVENANCE phrase in ``text``.

    The lens half of the qualified-mention test. `resolve_lens` consults this;
    nothing else decides whether a provenance word is naming a book.
    """
    if not text:
        return ()
    return tuple((m.start(), m.end()) for m in _LENS_PHRASE_RE.finditer(str(text)))


def scope_phrase_spans(text: Optional[str]):
    """``((start, end), ...)`` for every governed scope phrase in ``text``.

    THE single source of truth for what counts as portfolio-scope language.
    Resolvers that are looking for places, predicates or grouping axes mask
    these spans first, so a scope phrase cannot be consumed as something else.
    """
    if not text:
        return ()
    return tuple((m.start(), m.end()) for m in _SCOPE_PHRASE_RE.finditer(str(text)))


def names_selected_scope(text: Optional[str]) -> bool:
    """True when the question says "the current/selected portfolio"."""
    return bool(text) and bool(_SELECTED_SCOPE_RE.search(str(text)))


def names_total_scope(text: Optional[str]) -> bool:
    """True when the text explicitly names the consolidated FULL-AuM scope.

    The ``_TOTAL_TERMS`` family — "the whole book", "the entire portfolio", "the
    sponsored book". Distinct from :func:`names_selected_scope`: this is an
    EXPLICIT WIDENING to the client's full AuM, so it overrides a narrower UI
    selection rather than deferring to it, and a narrower source-type predicate
    emitted alongside it is a scope misread, not a filter to keep.
    """
    if not text:
        return False
    low = " " + str(text).strip().lower() + " "
    return _contains_any(low, _TOTAL_TERMS)


def mask_scope_phrases(text: Optional[str]) -> str:
    """``text`` with governed scope phrases blanked, preserving offsets.

    Blanking rather than deleting keeps every other offset valid, so a filter or
    dimension detector reading the remainder sees the sentence it expects — the
    same discipline the measure-set parser already uses.
    """
    if not text:
        return text or ""
    out = list(str(text))
    for start, end in scope_phrase_spans(text):
        for i in range(start, end):
            out[i] = " "
    return "".join(out)


@dataclass
class PortfolioLens:
    """A resolved portfolio lens: a label + the filters that realise it."""

    name: str
    label: str
    filters: Dict[str, Any] = field(default_factory=dict)
    cohort_id: Optional[str] = None
    #: Several books selected explicitly. ``cohort_id`` stays populated with the
    #: first for backward compatibility; consumers that understand a selection
    #: read this. Empty for every single-scope lens.
    cohort_ids: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "name": self.name,
            "label": self.label,
            "filters": dict(self.filters),
            "cohort_id": self.cohort_id,
        }
        if self.cohort_ids:
            out["cohort_ids"] = list(self.cohort_ids)
        return out


def total_lens() -> PortfolioLens:
    return PortfolioLens(LENS_TOTAL, "Total", {})


def _cohort_lens(cohort_id: str) -> PortfolioLens:
    cid = cohort_id.strip().lower()
    return PortfolioLens(LENS_COHORT, cid, {SOURCE_ID_FIELD: cid}, cohort_id=cid)


def _selection_lens(ids: Sequence[str]) -> PortfolioLens:
    """Several books chosen explicitly — exactly those, never their type."""
    ids = tuple(str(i).strip().lower() for i in ids if str(i).strip())
    return PortfolioLens(LENS_COHORT, " + ".join(ids),
                         {}, cohort_id=ids[0] if ids else None,
                         cohort_ids=ids)


def _type_lens(ptype: str) -> PortfolioLens:
    label = "Direct" if ptype == LENS_DIRECT else "Acquired"
    return PortfolioLens(ptype, label, {SOURCE_TYPE_FIELD: ptype})


def _contains_any(text: str, terms) -> bool:
    return any(t in text for t in terms)


#: Constructions that DISCLAIM a scope rather than selecting it.
#:
#: B22. "What is the balance excluding the acquired book?" answered over the
#: acquired book — the reader ruled out a cohort and received only that cohort.
#: The mention is QUALIFIED, so the scope-phrase test that settles the other
#: cases does not touch this one.
#:
#: A disclaiming mention DECLINES; it does not select the opposite. "Excluding
#: the acquired book" states what is not wanted, not what is, and inferring
#: "therefore the direct book" is a guess about scope — which this programme
#: treats as a substitution.
_DISCLAIMERS = (
    "excluding", "exclude", "excluded", "ignoring", "ignore", "other than",
    "apart from", "aside from", "setting aside", "net of", "without",
    "not including", "leaving out", "outside of", "before any", "except",
)

#: The disclaiming window, defined ONCE. Every reader that asks "did the
#: sentence rule this out?" measures the same distance and stops at the same
#: sentence boundary, so a term ruled out for one reader is ruled out for all.
_DISCLAIMER_ALT = "|".join(re.escape(d) for d in
                           sorted(_DISCLAIMERS, key=len, reverse=True))
_DISCLAIMER_GAP = r"[^.;?!]{0,24}?"

#: The same window, read BACKWARDS from a span the caller already located.
_DISCLAIMER_BEFORE_RE = re.compile(
    r"\b(?:" + _DISCLAIMER_ALT + r")\b" + _DISCLAIMER_GAP + r"$", re.IGNORECASE)


def is_disclaimed_span(text: Optional[str], start: int) -> bool:
    """True when the sentence RULES OUT the term beginning at ``start``.

    B21. THE primitive, for a reader that has already located its own term —
    a regex match, a vocabulary hit, a substring position. It exists because
    "does this question ask for a forecast?" turned out to have FOUR
    independent readers, each with its own vocabulary and its own way of
    locating a hit, and only the window and the boundary rule are common to
    them. Sharing the window is what makes "ignoring the forecast" mean the
    same thing to the frame resolver, the dataset resolver, the intent
    classifier and the facet raiser.
    """
    if not text or start <= 0:
        return False
    return bool(_DISCLAIMER_BEFORE_RE.search(str(text)[:start]))


def _disclaimed_span_re(target: str) -> "re.Pattern":
    """The disclaiming test, ONCE, over whichever target pattern is passed.

    B21. Parameterised by vocabulary the same way `_qualified_span_re` is, and
    for the same reason: hard-coding one vocabulary into the qualified-mention
    test is exactly what dropped five governed phrases in B22, and this test has
    the same two-caller shape. The scope resolver asks *is this book scope ruled
    out?*; `resolve_active_view` asks *is this view word ruled out?* — one
    question, two vocabularies.

    ``target`` is captured as group 1 so a caller can locate the ruled-out term
    itself, not merely the phrase containing it. ``group(0)`` still spans the
    disclaimer and the target together, which is the wording a receipt quotes.
    """
    return re.compile(r"\b(?:" + _DISCLAIMER_ALT + r")\b" + _DISCLAIMER_GAP
                      + r"(" + target + r")", re.IGNORECASE)


_DISCLAIMED_SCOPE_RE = _disclaimed_span_re(
    r"(?:" + "|".join(re.escape(q) for q in _SCOPE_QUALIFIERS) + r")\s+"
    r"(?:" + "|".join(re.escape(n) for n in _SCOPE_NOUNS) + r")\b")


def disclaimed_scope_phrase(text: Optional[str]) -> Optional[str]:
    """The governed scope the text RULES OUT, or ``None``.

    THE single source of truth, so a reader that wants to RECORD the declined
    narrowing gets both the fact and its wording from one place rather than
    re-deriving either. `resolve_lens` uses it to decline; nothing else decides
    it.
    """
    if not text:
        return None
    match = _DISCLAIMED_SCOPE_RE.search(str(text))
    return match.group(0).strip() if match else None


def disclaims_scope(text: Optional[str]) -> bool:
    """True when the text RULES OUT a governed scope rather than selecting it."""
    return disclaimed_scope_phrase(text) is not None


def undisclaimed_mention(text: Optional[str], term: Optional[str]) -> bool:
    """True when ``text`` names ``term`` at least once WITHOUT ruling it out.

    B21. THE test for "does this question ask for X", for any reader whose X is
    a bare word rather than a qualified phrase. A word every one of whose
    occurrences is disclaimed does not select: *"the balance by vintage,
    ignoring the forecast"* is not a forecast question, and the clause saying so
    is precisely what used to make it one. One undisclaimed occurrence is
    enough, because *"the forecast excluding pipeline"* IS a forecast question —
    a disclaiming construction declines; it does not select the opposite.

    Substring semantics are deliberate, and preserved exactly where nothing is
    disclaimed: the callers this replaces tested ``term in question``, so
    "forecasts" and "forecasting" counted, and they still do. The ONLY behaviour
    that changes is a mention the sentence rules out.

    Why not `_qualified_span_re` — measured, and it does not transfer. B22's
    doctrine is that a provenance word must QUALIFY a book noun, because
    `acquired` is ordinary English about lending. `forecast` and `pipeline` are
    NOUNS naming the subject: "How much pipeline is overdue?", "What is the
    forecast?", "Which broker has the largest pipeline?". Requiring them to
    qualify something would reject the corpus's own pipeline family. Only the
    disclaiming half of the shape is shared.
    """
    if not text or not term:
        return False
    haystack = str(text)
    low = haystack.lower()
    needle = str(term).lower()
    if needle not in low:
        return False
    at = low.find(needle)
    while at != -1:
        if not is_disclaimed_span(haystack, at):
            return True
        at = low.find(needle, at + 1)
    return False


def lens_from_term(term: Optional[str]) -> PortfolioLens:
    """Resolve a lens from a TERM already known to name a book.

    The caller has established that — a registry key, a spec field, a planner
    branch that has already matched "direct"/"acquired". No qualification is
    required and none is checked.

    Separated from `resolve_lens` in B22 because that function now requires a
    QUALIFIED mention, and a caller holding a bare book name would otherwise be
    told it holds none. `populations.resolve_lens` had already worked around the
    absence of this entry point by building `f"the {spec.lens_term} book"` by
    hand — the distinction was real before it was named.
    """
    if not term:
        return total_lens()
    return _lens_from_text(" " + str(term).strip().lower() + " ")


def resolve_lens(text: Optional[str]) -> PortfolioLens:
    """Resolve a single portfolio lens from a QUESTION. Defaults to *total*.

    Precedence: explicit cohort id > a QUALIFIED acquired/direct mention > total.
    An explicit "total/whole book" phrase forces *total* even if other words
    appear.

    B22 — the mention must QUALIFY A BOOK NOUN. `_DIRECT_TERMS` and
    `_ACQUIRED_TERMS` are ordinary English about lending: `purchased`,
    `acquired`, `direct`, `organic`. Read as bare substrings they answered "the
    balance for loans purchased at auction" over the acquired cohort — 3,909 of
    11,035 loans, a complete and correctly formatted answer over 35% of the book
    for a question about how a property was bought.

    The test for that is `scope_phrase_spans`, in this module, over a
    `_SCOPE_QUALIFIERS` list that already contains `direct`, `acquired`,
    `purchased` and `funded`. It states the doctrine — "Only QUALIFIED phrases
    count. A bare 'current' or 'entire' is ordinary English" — and was called by
    the filter and dimension parsers to protect them FROM this vocabulary, and
    never by the decision that owns it. One helper, two callers; duplicating the
    test would create a second owner of the decision this fix consolidates.

    A term caller wants `lens_from_term`.
    """
    if not text:
        return total_lens()
    if disclaims_scope(text):
        # DECLINES, and does not select the opposite. What the answer then owes
        # the reader — a whole-book figure is not it — is recorded by the facet
        # layer through `disclaims_scope`, not decided here.
        return total_lens()
    low = " " + str(text).strip().lower() + " "
    if not lens_phrase_spans(text) and not _COHORT_ID_RE.search(low):
        return total_lens()

    return _lens_from_text(low)


def _lens_from_text(low: str) -> PortfolioLens:
    """The resolution itself, shared by both entry points.

    Neither qualification nor disclaiming is checked here: `resolve_lens` has
    done that for a question, and `lens_from_term` has established it is holding
    a book name.
    """
    # Exact cohort id always wins (most specific).
    m = _COHORT_ID_RE.search(low)
    if m:
        return _cohort_lens(m.group(1))

    has_direct = _contains_any(low, _DIRECT_TERMS)
    has_acquired = _contains_any(low, _ACQUIRED_TERMS)

    # If both families are mentioned (e.g. "direct vs acquired") this is a
    # comparison, not a single lens — fall back to total for the single-lens
    # view; callers wanting both should use resolve_comparison_lenses().
    if has_direct and has_acquired:
        return total_lens()
    if has_acquired:
        return _type_lens(LENS_ACQUIRED)
    if has_direct:
        return _type_lens(LENS_DIRECT)
    # Explicit total phrasing or nothing recognised → whole book.
    return total_lens()


def is_comparison(text: Optional[str]) -> bool:
    if not text:
        return False
    return _contains_any(" " + str(text).lower() + " ", _COMPARISON_TERMS)


def resolve_comparison_lenses(text: Optional[str]) -> List[PortfolioLens]:
    """Resolve a side-by-side comparison into 2+ lenses, else ``[]``.

    Handles the common securitisation cuts:
      * direct vs acquired                  → [Direct, Acquired]
      * direct_001 vs acquired_001          → [direct_001, acquired_001]
      * acquired_001 vs acquired_002        → [acquired_001, acquired_002]
    """
    if not text:
        return []
    low = " " + str(text).strip().lower() + " "

    ids = [c.lower() for c in _COHORT_ID_RE.findall(low)]
    # De-duplicate while preserving order.
    seen: set = set()
    uniq_ids = [c for c in ids if not (c in seen or seen.add(c))]
    if len(uniq_ids) >= 2:
        return [_cohort_lens(c) for c in uniq_ids]

    if not is_comparison(low):
        return []

    has_direct = _contains_any(low, _DIRECT_TERMS)
    has_acquired = _contains_any(low, _ACQUIRED_TERMS)
    if has_direct and has_acquired:
        return [_type_lens(LENS_DIRECT), _type_lens(LENS_ACQUIRED)]
    return []


def lens_title_suffix(lens: PortfolioLens) -> str:
    """Human-readable suffix for a chart/table/card title, e.g. ' — Direct'."""
    return f" — {lens.label}"


def apply_lens(spec, lens: PortfolioLens):
    """Merge a lens onto an MIQuerySpec (in place) and return the spec.

    The lens filters are merged into ``spec.filters`` (lens wins on conflict),
    the resolved lens is recorded on ``spec.portfolio_lens`` for output
    metadata, and a title suffix is appended when the spec has no explicit one.
    """
    if lens is None:
        return spec
    merged = dict(getattr(spec, "filters", {}) or {})
    merged.update(lens.filters)
    spec.filters = merged
    # Record lens metadata (the field is added to MIQuerySpec).
    try:
        spec.portfolio_lens = lens.to_dict()
    except Exception:  # pragma: no cover - spec without the attribute
        pass
    if lens.name != LENS_TOTAL and getattr(spec, "title", None):
        if lens.label not in str(spec.title):
            spec.title = f"{spec.title}{lens_title_suffix(lens)}"
    return spec


def context_id(lens: PortfolioLens) -> str:
    """The governed portfolio-context id a resolved lens names.

    The lens layer's job is *natural-language recognition*; deciding what the
    named scope actually contains belongs to the governed registry
    (:mod:`trakt_core.portfolio`). This is the handover point: a cohort lens
    yields its portfolio id, a type lens its type, and total yields ``total``.
    """
    if lens is None:
        return LENS_TOTAL
    # A multi-selection hands the registry the whole id list, so the governed
    # scope is exactly the books chosen. Handing over one of them, or their
    # type, would answer for a population the caller did not select.
    if lens.cohort_ids and len(lens.cohort_ids) > 1:
        return list(lens.cohort_ids)
    if lens.name == LENS_COHORT and lens.cohort_id:
        return lens.cohort_id
    return lens.name


def apply_scope(spec, lens: PortfolioLens, scope) -> Any:
    """Merge a RESOLVED governed scope onto a spec (in place) and return it.

    Same contract as :func:`apply_lens` — the label and title suffix are the
    lens's — except the filter comes from the resolved scope, so a group narrows
    to the explicit portfolio ids the registry currently holds rather than to a
    type string. That is what makes the hierarchy dynamic: a newly onboarded
    ``direct_002`` joins the Direct answer without any code change here.
    """
    if scope is None:
        return apply_lens(spec, lens)
    merged = dict(getattr(spec, "filters", {}) or {})
    merged.update(scope.filters)
    spec.filters = merged
    label = scope.label if lens is None else lens.label
    try:
        spec.portfolio_lens = {
            **(lens.to_dict() if lens is not None else total_lens().to_dict()),
            "filters": dict(scope.filters),
            "context_id": scope.context_id,
            "context_kind": scope.context_kind,
            "portfolio_ids": list(scope.portfolio_ids),
        }
    except Exception:  # pragma: no cover - spec without the attribute
        pass
    if not scope.is_total and getattr(spec, "title", None):
        if label not in str(spec.title):
            spec.title = f"{spec.title} — {label}"
    return spec


def resolve_and_apply(spec, text: Optional[str]):
    """Convenience: resolve the lens from text and apply it to the spec."""
    return apply_lens(spec, resolve_lens(text))


def mentions_portfolio(text: Optional[str]) -> bool:
    """True if the text refers to a source-portfolio scope (any lens family)."""
    if not text:
        return False
    low = " " + str(text).strip().lower() + " "
    if _COHORT_ID_RE.search(low):
        return True
    return _contains_any(low, _DIRECT_TERMS + _ACQUIRED_TERMS + _TOTAL_TERMS)


def lens_from_selection(value: Any) -> PortfolioLens:
    """Build a lens from an explicit UI/API selection.

    Accepts ``None`` / ``"total"`` → total; ``"direct"`` / ``"acquired"`` →
    the type lens; an exact cohort id (``direct_001`` / ``acquired_002``) → the
    cohort lens. A dict like ``{"id": "acquired_001"}`` is also accepted.
    Unrecognised selections fall back to *total* (never an error).
    """
    if isinstance(value, Mapping):
        value = value.get("id") or value.get("lens") or value.get("value")
    if isinstance(value, (list, tuple, set, frozenset)):
        picked = [str(v).strip() for v in value if str(v).strip()]
        if not picked:
            return total_lens()
        if len(picked) > 1:
            return _selection_lens(picked)
        value = picked[0]
    if value is None:
        return total_lens()
    sel = str(value).strip().lower()
    if sel in ("", "total", "all", "whole_book", "whole-book"):
        return total_lens()
    if sel == LENS_DIRECT:
        return _type_lens(LENS_DIRECT)
    if sel == LENS_ACQUIRED:
        return _type_lens(LENS_ACQUIRED)
    if _COHORT_ID_RE.fullmatch(sel) or _COHORT_ID_RE.search(" " + sel + " "):
        return _cohort_lens(sel)
    # An explicit selection may be any provenance-valid source_portfolio_id.
    if _SELECTABLE_COHORT_ID_RE.fullmatch(sel):
        return _cohort_lens(sel)
    return total_lens()


def resolve_lens_with_default(
    text: Optional[str], default: Optional[PortfolioLens] = None
) -> PortfolioLens:
    """Resolve the effective lens: a portfolio scope named in ``text`` wins
    (natural-language override); otherwise the ``default`` (e.g. the dropdown
    selection); otherwise *total*."""
    if mentions_portfolio(text):
        return resolve_lens(text)
    return default or total_lens()


# NOTE: ``is_acquired_only()`` used to live here and decided, from an
# ``acquired_`` id prefix, that a scope had no funding pipeline. Business
# applicability is now resolved from governed portfolio metadata by
# ``trakt_core.portfolio.resolve_capabilities`` — which is what lets an acquired
# vehicle that DOES originate be configured rather than coded around, and stops
# a portfolio's NAME deciding what analysis it may support.


def available_lenses(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Build the selectable lens list from distinct provenance records.

    ``records`` are distinct ``{source_portfolio_id, source_portfolio_type,
    source_portfolio_label}`` rows from the active central canonical. Returns
    Total, the Direct / Acquired type lenses that are actually present, and one
    cohort lens per ``source_portfolio_id``. Each entry carries ``funded_only``
    so the UI can hide Pipeline / Forecast for acquired-only scopes.
    """
    out: List[Dict[str, Any]] = [{
        "id": LENS_TOTAL, "kind": LENS_TOTAL, "label": "Total",
        "filters": {}, "funded_only": False,
    }]
    types = {str(r.get("source_portfolio_type", "")).strip().lower()
             for r in records if r.get("source_portfolio_type")}
    if LENS_DIRECT in types:
        out.append({"id": LENS_DIRECT, "kind": "type", "label": "Direct",
                    "filters": {SOURCE_TYPE_FIELD: LENS_DIRECT}, "funded_only": False})
    if LENS_ACQUIRED in types:
        out.append({"id": LENS_ACQUIRED, "kind": "type", "label": "Acquired",
                    "filters": {SOURCE_TYPE_FIELD: LENS_ACQUIRED}, "funded_only": True})

    seen: set = set()
    cohorts: List[Dict[str, Any]] = []
    for r in records:
        pid = str(r.get("source_portfolio_id", "")).strip()
        if not pid or pid in seen:
            continue
        seen.add(pid)
        ptype = str(r.get("source_portfolio_type", "")).strip().lower()
        ptype = "" if ptype in ("nan", "none", "nat", "<na>") else ptype
        label = r.get("source_portfolio_label")
        label_str = "" if label is None else str(label).strip()
        # Blank / NaN labels (pandas yields the string "nan") fall back to the id.
        if label_str.lower() in ("", "nan", "none", "nat", "<na>"):
            label_str = pid
        cohorts.append({
            "id": pid, "kind": LENS_COHORT,
            "label": label_str,
            "source_portfolio_type": ptype or None,
            "filters": {SOURCE_ID_FIELD: pid},
            "funded_only": ptype == LENS_ACQUIRED,
        })
    out.extend(sorted(cohorts, key=lambda c: c["id"]))
    return out
