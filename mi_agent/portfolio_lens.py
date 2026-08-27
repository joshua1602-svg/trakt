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
#: PHASE 1E. The question NAMED a portfolio scope and the governed registry does
#: not hold it. Distinct from `total` on purpose: "the whole book" and "I could
#: not resolve what you named" are different answers, and collapsing the second
#: into the first is how a question about one portfolio was answered with every
#: portfolio (docs/mi_phase1c_report.md).
LENS_UNRESOLVED = "unresolved"

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
    # PHASE 1G. "Funded Book" is the business term for the COMPLETE funded
    # population — Direct AND Acquired together, and every other governed funded
    # category. Its absence here was measured in Phase 1F: with the workspace
    # scoped to Acquired, "Summarise the funded book" answered 3,909 of 11,035
    # loans, while "across all portfolios" — the same request in different words
    # — correctly answered for the whole book. Two explicit whole-book phrasings
    # with opposite precedence.
    #
    # QUALIFIED FORMS ONLY, for the reason the module records throughout: bare
    # "funded" names a MEASURE ("funded balance", "funded amount"), and reading
    # a measure as a scope is the silent mutation this vocabulary exists to
    # prevent. Only the noun phrases that name the BOOK are here.
    "funded book", "funded portfolio", "funded loan book",
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


#: THIS module's own fields. A value of one of them naming a book is the scope
#: reading itself, not a competing categorical claim, so it may not mask the
#: phrase this module exists to read: a fixture whose `source_portfolio_label`
#: is literally "Direct Book" turned "show direct book balance" into Total.
_SCOPE_OWNED_FIELDS = (SOURCE_TYPE_FIELD, SOURCE_ID_FIELD,
                       "source_portfolio_label", "portfolio_cohort")


def mask_claimed_value_spans(text: Optional[str], available_values=None) -> str:
    """``text`` with spans a governed CATEGORICAL VALUE has claimed blanked.

    The delegation, once, for every gate in this module. `categorical_spans`
    owns which spans those are and why; this only asks, and names its own fields
    so the rule stays a rule about TWO owners. With no values supplied — the
    default everywhere — the text is returned unchanged, so a caller that cannot
    see the book keeps exactly the reading it had.
    """
    if not text or not available_values:
        return text or ""
    try:
        from .categorical_spans import mask_value_spans
    except Exception:  # noqa: BLE001 - the owner missing must not change a reading
        return str(text)
    return mask_value_spans(text, available_values,
                            exclude_fields=_SCOPE_OWNED_FIELDS)


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


def _unresolved_lens(requested: str) -> PortfolioLens:
    """A named scope the governed registry does not hold.

    Carries the wording so a refusal can quote it back. It deliberately has NO
    filters: a consumer that applies them gets an unnarrowed frame, so the
    failure mode of ignoring this lens is a visible refusal upstream rather than
    a silent whole-book answer.
    """
    return PortfolioLens(LENS_UNRESOLVED, str(requested).strip(), {})


def _named_portfolio_lens(low: str, registry) -> Optional[PortfolioLens]:
    """A GOVERNED portfolio the question names, by display label or by id.

    The registry is the authority React already renders from
    (``portfolio_context.context_index`` -> the workspace selector), so matching
    against it is what makes MI and the UI mean the same thing by a portfolio
    name. There is no alias table here and no vocabulary: add a portfolio to the
    registry and it becomes sayable, rename it and the new name works.

    LONGEST TOKEN FIRST, and checked BEFORE the category branch, because a
    portfolio's label may contain a category word — "ALP **Acquired** Back Book"
    must resolve to that book and not to every acquired book.
    """
    if registry is None:
        return None
    #: (sayable token, governed id, governed display label or None)
    candidates: List[Tuple[str, str, Optional[str]]] = []
    for pid in registry.ids():
        record = registry.get(pid)
        label = getattr(record, "display_label", None) if record else None
        for token in (label, pid):
            token = _clean_token(token)
            if token:
                candidates.append((token, pid, label))
    for token, pid, label in sorted(candidates, key=lambda c: -len(c[0])):
        if not _token_in(low, token):
            continue
        lens = _cohort_lens(pid)
        if not label:
            return lens
        # Answer in the name the CLIENT sees. `_cohort_lens` labels a book with
        # its id, which is the governed identity but not the product one —
        # React renders `display_label`, and an answer that says
        # "(alp_origination)" is naming infrastructure at a reader who selected
        # "ALP Origination Book". The FILTERS are untouched: the label is what
        # the answer says, the id is what it selects on.
        return PortfolioLens(lens.name, str(label), dict(lens.filters),
                             cohort_id=lens.cohort_id,
                             cohort_ids=lens.cohort_ids)
    return None


#: The head noun that makes a capitalised phrase a BOOK NAME rather than an
#: ordinary noun phrase. Matched case-insensitively; it is the capitalisation of
#: the words BEFORE it that carries the "this is a proper name" signal.
_BOOK_NOUN_RE = re.compile(r"\b(book|portfolio)\b", re.IGNORECASE)


def names_a_book_noun(text: Optional[str]) -> bool:
    """True when ``text`` calls something a BOOK or a PORTFOLIO.

    THE HEAD NOUN, exposed. It is what turns a proper name into a book name —
    "the Highgate Mortgages Book" — and a resolver that has failed to resolve a
    phrase needs to know whether the phrase was a book before it refuses in its
    own words: an unheld portfolio has a refusal of its own, from this module,
    and a categorical resolver announcing "no loans match 'highgate mortgages'"
    speaks over it with a worse explanation of the same fact.
    """
    return bool(text) and bool(_BOOK_NOUN_RE.search(str(text)))

#: A token that is capitalised in the ORIGINAL text (so `Highgate`, `ALP`,
#: `NBS`, but not `the`, `of`, `by`).
_PROPER_TOKEN_RE = re.compile(r"^[A-Z][A-Za-z0-9&'\u2019-]*$")

#: THE CAPITAL IS EXPLAINED BY POSITION, so it is not evidence of a name.
#:
#: A run of capitalised tokens is the only signal `_unknown_named_book` has that
#: a book has been NAMED. The first word of a sentence is capitalised by
#: orthographic convention whatever it is, so its capital says nothing — and
#: reading it as part of the name is how "Break Direct portfolio balance down
#: across LTV…" came to refuse with *"'Break Direct portfolio' is not a governed
#: portfolio for this book"*, quoting the reader's own verb back at them as the
#: name of a book they had not named.
#:
#: This is the PROPERTY the guard needed. `_GENERIC_BOOK_WORDS` had been carrying
#: the job as a list — its last block is commented "question scaffolding that can
#: be sentence-initial or capitalised" and holds `show`, `give`, `summarise` — and
#: a list of the verbs someone thought of does not produce silence on the verbs
#: they did not: measured over 1,446 corpus questions, `break`, `plot` and
#: `which` were all missing, and each produced the same fragment. The list is a
#: closed set; sentence position is a property of every sentence.
#:
#: Matches an empty/whitespace prefix (the token opens the text) or one ending in
#: a sentence terminator, with any closing quote or bracket after it.
_SENTENCE_INITIAL_RE = re.compile(r"(?:^|[.!?][\"')\]]*)\s*$")

#: Words that may be capitalised in front of "Book"/"Portfolio" WITHOUT naming a
#: particular book: governed scope vocabulary, seasoning/vintage vocabulary,
#: ordinary lending nouns, and the sentence-initial verbs a question opens with.
#: A capitalised run made only of these is not a proper name — "Summarise the
#: Acquired Back Book" names a category and a seasoning segment, not a portfolio
#: called "Acquired Back".
_GENERIC_BOOK_WORDS = frozenset({
    # governed scope vocabulary
    "direct", "directly", "acquired", "purchased", "originated", "origination",
    "funded", "unfunded", "total", "whole", "entire", "all", "combined",
    "overall", "platform", "group", "consolidated", "aggregate", "source",
    "sub", "book", "books", "portfolio", "portfolios",
    # seasoning / vintage vocabulary — a different axis, owned elsewhere
    "back", "backbook", "front", "legacy", "new", "newly", "current",
    "existing", "historic", "historical", "seasoned", "recent", "live",
    "active", "closed", "open", "run", "off",
    # ordinary lending nouns
    "loan", "loans", "lending", "mortgage", "mortgages", "asset", "assets",
    "equity", "release", "retirement", "interest", "only", "product",
    "products", "main", "master", "primary", "core", "full",
    # question scaffolding that can be sentence-initial or capitalised
    "summarise", "summarize", "summary", "show", "give", "tell", "what",
    "how", "please", "provide", "list", "report", "describe", "explain",
    "the", "this", "that", "a", "an", "our", "my", "its", "of", "for", "in",
    "and", "or", "me", "us", "is", "are", "do", "does",
})


#: A token shaped like a member of a NUMBERED naming family — `spv1`, `spv12`,
#: `fund3`. The alphabetic stem and the digits are captured separately so the
#: stem can be checked against the registry's OWN labels; there is no family
#: vocabulary in this module and none is added when a client onboards a new one.
_FAMILY_TOKEN_RE = re.compile(r"^([a-z][a-z]*)(\d+)$")


def _registry_name_families(registry) -> set:
    """The numbered naming families the registry ITSELF demonstrates.

    A registry holding `spv1` and `spv2` establishes that this client names
    portfolios `spv<n>`. `spv9` is then a member of a family MI can see, and a
    question naming it is a portfolio reference — one this registry does not
    hold, so it must be clarified rather than answered for the whole book.

    Derived per call from the registry's ids and labels, so there is no `spv`
    literal anywhere in this module: a client whose portfolios are `pool1` and
    `pool2` gets `pool7` recognised for exactly the same reason, and a client
    with no numbered family gets no families and no behaviour change at all.
    """
    families = set()
    if registry is None:
        return families
    for pid in registry.ids():
        record = registry.get(pid)
        label = getattr(record, "display_label", None) if record else None
        for token in (pid, label):
            for word in _clean_token(token).split():
                match = _FAMILY_TOKEN_RE.match(word)
                if match:
                    families.add(match.group(1))
    return families


def _unknown_family_member(text: Optional[str], registry) -> Optional[PortfolioLens]:
    """A member of a registry naming family that the registry does not hold.

    PHASE 1G. `_unknown_named_book` below needs a "Book"/"Portfolio" head noun
    to recognise a name, and "Summarise SPV9" has none — measured, it resolved
    to Total and answered for all five portfolios under the name of one that
    does not exist. Naming a portfolio MI cannot find is a question to clarify,
    whether or not the sentence spells out the word "portfolio".

    Fires ONLY on a token whose stem the registry itself uses, so it cannot
    reach ordinary English: `q4`, `top10` and `h1` are inert unless this client
    genuinely has portfolios named `q<n>`, `top<n>` or `h<n>`.
    """
    if registry is None or not text:
        return None
    families = _registry_name_families(registry)
    if not families:
        return None
    held = set()
    for pid in registry.ids():
        record = registry.get(pid)
        held.add(_clean_token(pid))
        held.add(_clean_token(getattr(record, "display_label", None) if record else None))
    for word in re.findall(r"[A-Za-z]+\d+", str(text)):
        token = _clean_token(word)
        match = _FAMILY_TOKEN_RE.match(token)
        if match and match.group(1) in families and token not in held:
            return _unresolved_lens(word)
    return None


def _unknown_named_book(text: Optional[str], registry) -> Optional[PortfolioLens]:
    """A capitalised book NAME the governed registry does not hold.

    PHASE 1E. `_named_portfolio_lens` resolves the names the registry HAS; this
    catches the ones it has not. Measured before it existed: "Summarise the
    Highgate Mortgages Book" — a book this platform has never onboarded —
    returned the whole book's 11,035 loans and 1.96bn with the name it was asked
    for appearing nowhere in the answer. Naming a book MI cannot find is a
    question to clarify, never a licence to answer for every book.

    Deliberately narrow, in two ways that keep it off ordinary prose:

      * it fires only on a run of tokens CAPITALISED IN THE ORIGINAL TEXT
        immediately before "Book"/"Portfolio" — "the portfolio", "the acquired
        book" and "a portfolio summary" carry no such run;
      * the SENTENCE-INITIAL token of that run is discounted, because its
        capital is explained by position rather than by naming anything (see
        `_SENTENCE_INITIAL_RE`); and
      * at least one of the tokens that remain must be outside
        `_GENERIC_BOOK_WORDS`, so "the Acquired Back Book" resolves through the
        category and seasoning vocabulary that owns it rather than being read as
        a proper name.

    KNOWN LIMITS, stated rather than papered over.

    A book named entirely in lower case ("summarise the highgate mortgages
    book") carries no proper-name signal and is not caught here. Requiring
    capitalisation is what stops this refusing ordinary questions; recognising
    lower-case unknown names needs a vocabulary check this layer does not own.

    A ONE-WORD book name opening a sentence ("Highgate Book summary") is
    discounted along with the verbs, because at that position the two are
    genuinely indistinguishable by capitalisation — the only signal here. The
    trade is deliberate and measured: over 1,446 corpus questions the property
    removes three fragments and raises nothing new, and a widening that
    discloses nothing is a worse failure than a name this layer declines to
    recognise.

    A TRAILING SEPARATOR still evades the guard. "Direct-book" yields the token
    `Direct-`, and `direct-` matches no entry in any word list, so the run is
    judged a name whatever the list holds. That is the same shape of defect as
    the one this property fixes — a token the guard cannot match rather than a
    word nobody listed — and it is NOT fixed here: it is coupled to the
    qualifier/noun separator in `_qualified_span_re`, which is unshipped. Four
    questions still refuse with `'Direct- book'`. Measured and recorded in
    `migration_phase0/MI_DIRECT_COLLISION_SCOPE.md`, not left to be rediscovered.

    Returns ``None`` when there is no registry to check against, so a caller
    that supplies none keeps exactly its pre-1E behaviour.
    """
    if registry is None or not text:
        return None
    raw = str(text)
    for match in _BOOK_NOUN_RE.finditer(raw):
        head = raw[:match.start()].rstrip()
        run: List[str] = []
        # What stood before the run's FIRST token, so the sentence-position test
        # below can be asked of the right offset. Re-derived each iteration
        # because the run grows leftwards; the last value is the one that counts.
        before = head
        while len(run) < 5:
            token_match = re.search(r"([A-Za-z0-9&'\u2019-]+)$", head)
            if not token_match:
                break
            token = token_match.group(1)
            if not _PROPER_TOKEN_RE.match(token):
                break
            run.insert(0, token)
            before = head[:token_match.start()]
            head = head[:token_match.start()].rstrip()
        if not run:
            continue
        # THE PROPERTY, not a longer list. A sentence-initial token is
        # capitalised whatever it is, so it carries no proper-name evidence and
        # is not part of the name. Only the FIRST token is discounted: a
        # multi-word name that opens a sentence still has the rest of its run to
        # be judged on, so "Highgate Mortgages Book balance by region" is still a
        # named book.
        judged = run[1:] if _SENTENCE_INITIAL_RE.search(before) else run
        # Nothing left to judge: every capital in the run was explained by
        # position, so no book was named here.
        if not judged:
            continue
        if all(tok.lower() in _GENERIC_BOOK_WORDS for tok in judged):
            continue
        return _unresolved_lens(" ".join(run + [match.group(1)]))
    return None


def _clean_token(token: Optional[str]) -> str:
    """A label/id reduced to the form a sentence would carry it in."""
    if not token:
        return ""
    return re.sub(r"[\s_]+", " ", str(token).strip().lower()).strip()


def _token_in(low: str, token: str) -> bool:
    """Whether ``low`` names ``token``, tolerating `_` vs space in an id.

    ``low`` is already space-padded and lowercased by the caller. Underscores in
    the HAYSTACK are normalised too, so "the nbs_acquired book" matches the id
    `nbs_acquired` and the label "NBS Acquired Book" matches either spelling.
    """
    if not token:
        return False
    haystack = re.sub(r"[\s_]+", " ", low)
    return (" " + token + " ") in haystack


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


def resolve_lens(text: Optional[str], *, registry=None,
                 available_values=None) -> PortfolioLens:
    """Resolve a single portfolio lens from a QUESTION. Defaults to *total*.

    PHASE 1E — ``registry``. When a governed :class:`PortfolioRegistry` is
    supplied, a portfolio NAMED by its display label or its governed id resolves
    to that portfolio, and a cohort-shaped id the registry does not hold becomes
    ``LENS_UNRESOLVED`` instead of silently selecting nothing. Omitted, every
    decision below is exactly what it was — the parameter adds resolution, it
    never removes any.

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

    ``available_values`` — GOVERNED SPAN OWNERSHIP. Handed the book's own
    categorical values, a span already claimed as one of them is blanked before
    the qualified-mention gate reads the sentence, so "how many Gamma Direct
    loans do we have?" no longer ALSO narrows to the direct book. See
    `mi_agent.categorical_spans`, which owns that decision; nothing is decided
    here. Omitted, every reading below is exactly what it was.

    The three NAMED-PORTFOLIO branches above run on the RAW text on purpose: a
    governed portfolio named outright is the most specific thing the sentence
    can say and already wins over everything below, so a categorical value may
    not silence it.

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
    # A governed portfolio named outright wins over everything below: it is the
    # most specific thing the sentence can say, and its label may CONTAIN a
    # category word. Checked before the qualified-noun gate because a governed
    # label need not contain a noun the gate knows ("spv1_sponsored").
    named = _named_portfolio_lens(low, registry)
    if named is not None:
        return named
    # A book NAMED and not held. Checked before the qualified-noun gate for the
    # same reason: an unknown proper name contains no vocabulary the gate knows,
    # so the gate would pass it straight to `total` — the widening this catches.
    unknown = _unknown_named_book(text, registry)
    if unknown is not None:
        return unknown
    # ... and a member of a naming family the registry uses but does not hold,
    # which carries no book noun for the check above to find.
    unknown = _unknown_family_member(text, registry)
    if unknown is not None:
        return unknown
    owned = mask_claimed_value_spans(text, available_values)
    low_owned = " " + str(owned).strip().lower() + " "
    if not lens_phrase_spans(owned) and not _COHORT_ID_RE.search(low_owned):
        return total_lens()

    return _lens_from_text(low_owned, registry=registry)


def _lens_from_text(low: str, *, registry=None) -> PortfolioLens:
    """The resolution itself, shared by both entry points.

    Neither qualification nor disclaiming is checked here: `resolve_lens` has
    done that for a question, and `lens_from_term` has established it is holding
    a book name.
    """
    # Exact cohort id always wins (most specific).
    m = _COHORT_ID_RE.search(low)
    if m:
        requested = m.group(1)
        # PHASE 1E. With a registry to check against, an id it does not hold is
        # UNRESOLVED, not a cohort. Previously this returned a cohort lens whose
        # governed scope then fell back to Total, so "the acquired_001 book"
        # answered for every book under that label.
        if registry is not None and registry.get(requested.lower()) is None:
            return _unresolved_lens(requested)
        return _cohort_lens(requested)

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
    if lens.name == LENS_UNRESOLVED:
        # Hand back the UNRESOLVED wording, never `total`. `resolve_scope` will
        # not find it and will record `fell_back_to_total` with
        # `requested_context_id` set — which is the evidence the facet layer
        # refuses on. Returning `total` here would erase the request.
        return lens.label or LENS_UNRESOLVED
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


def mentions_portfolio(text: Optional[str], *, available_values=None) -> bool:
    """True if the text refers to a source-portfolio scope (any lens family).

    ``available_values`` applies the same GOVERNED SPAN OWNERSHIP rule as
    :func:`resolve_lens`. It matters here independently: this is the PRECEDENCE
    gate, so a broker value read as a scope mention would hand the question to
    `resolve_lens` in place of the caller's selection even when `resolve_lens`
    itself has stopped reading it that way.
    """
    if not text:
        return False
    text = mask_claimed_value_spans(text, available_values)
    if not text.strip():
        return False
    low = " " + str(text).strip().lower() + " "
    if _COHORT_ID_RE.search(low):
        return True
    return _contains_any(low, _DIRECT_TERMS + _ACQUIRED_TERMS + _TOTAL_TERMS)


def lens_from_selection(value: Any, *, registry=None) -> PortfolioLens:
    """Build a lens from an explicit UI/API selection.

    Accepts ``None`` / ``"total"`` → total; ``"direct"`` / ``"acquired"`` →
    the type lens; an exact cohort id (``direct_001`` / ``acquired_002``) → the
    cohort lens. A dict like ``{"id": "acquired_001"}`` is also accepted.
    Unrecognised selections fall back to *total* (never an error).

    PHASE 1G — ``registry``. THE REGISTRY DECIDES WHAT IS SELECTABLE, not a
    naming convention. Measured before this: `_SELECTABLE_COHORT_ID_RE` requires
    an underscore, so a governed portfolio registered as `spv1` fell through
    every branch below and became **Total** — a workspace scoped to SPV1
    answering for the whole book, silently, on every question that did not name
    a scope itself.

    That is the same defect class as Phase 1D's: MI recognising a STORAGE naming
    convention rather than the governed identity. The registry is asked first,
    so an id it holds is selectable whatever it is called, and `spv4` needs a
    registry entry rather than a pattern change.
    """
    if registry is not None:
        picked = value
        if isinstance(picked, Mapping):
            picked = picked.get("id") or picked.get("lens") or picked.get("value")
        if isinstance(picked, str):
            candidate = picked.strip().lower()
            if candidate and registry.get(candidate) is not None:
                return _cohort_lens(candidate)
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
    text: Optional[str], default: Optional[PortfolioLens] = None, *, registry=None,
    available_values=None
) -> PortfolioLens:
    """Resolve the effective lens: a portfolio scope named in ``text`` wins
    (natural-language override); otherwise the ``default`` (e.g. the dropdown
    selection); otherwise *total*.

    ``registry`` is passed through to :func:`resolve_lens` — see PHASE 1E there.
    The PRECEDENCE rule is unchanged: a registry lets the question resolve MORE
    scopes, it does not change which side wins.
    """
    if (mentions_portfolio(text, available_values=available_values)
            or names_governed_portfolio(text, registry)):
        return resolve_lens(text, registry=registry,
                            available_values=available_values)
    return default or total_lens()


def names_governed_portfolio(text: Optional[str], registry=None) -> bool:
    """Whether ``text`` names a governed portfolio by label or id.

    PHASE 1E. ``mentions_portfolio`` tests a fixed QUALIFIER vocabulary, so it
    is blind to "ALP Origination Book" — a governed name it has never heard of.
    Without this, such a question would resolve correctly on its own and then
    lose to a caller-supplied default, because the precedence gate would not
    count it as naming a scope.
    """
    if not text or registry is None:
        return False
    low = " " + str(text).strip().lower() + " "
    if _named_portfolio_lens(low, registry) is not None:
        return True
    if _unknown_family_member(text, registry) is not None:
        return True
    # An UNKNOWN name is still a name. It has to count as naming a scope, or a
    # caller-supplied default would answer in its place — which is the widening
    # by another door.
    return _unknown_named_book(text, registry) is not None


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
