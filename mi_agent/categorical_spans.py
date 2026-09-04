"""THE single owner of what a governed categorical VALUE has claimed.

A question is read by several independent resolvers — the categorical filter
parser, the portfolio-lens resolver, the seasoning segmenter, the place
resolver. Each of them scans the RAW question for its own vocabulary, so the
same characters can be claimed twice, by two resolvers that never meet.

Measured: with a broker called **Gamma Direct** on the book,

    "How many Gamma Direct loans do we have?"

answered **104**. The categorical parser correctly claimed ``Gamma Direct`` as a
value of ``broker_channel``; the portfolio-lens resolver, reading the same raw
string, independently matched ``Direct loans`` against its qualifier/noun
grammar and narrowed the population to the ``direct_001`` book as well. The
broker alone is 147. Nothing was wrong with either resolver in isolation, and
neither could see that the other had already spoken for those characters.

The invariant this module owns:

    Once a contiguous span has been claimed as a governed categorical value,
    the tokens INSIDE that span must not independently create another semantic
    claim, unless the grammar explicitly establishes a second meaning.

"Unless the grammar establishes a second meaning" is why this masks spans
rather than deleting concepts: in

    "How many Gamma Direct loans are in the Direct book?"

the second ``Direct`` sits in its own span, outside the value's, and the
question genuinely carries BOTH claims. Masking preserves offsets, so the lens
resolver still sees ``in the Direct book`` exactly where it is.

Two deliberate limits keep the ownership claim honest:

* Only a value that resolves to EXACTLY ONE governed field is claimed. An
  ambiguous value has not been claimed by anything, so it may not silence
  anything either.
* Only a MULTI-WORD value claims a span. A single-token span has no "inside":
  a book whose ``source_portfolio_type`` carries the value ``direct`` would
  otherwise let that value mask the very phrase — "the direct book" — that the
  lens resolver exists to read. Where the value IS the competing word, the two
  readings are ambiguous rather than colliding, and this module stays silent.
  "Word" is counted on WHITESPACE, never on underscores: ``direct_002`` is one
  token spelled with a separator, and counting it as two let a book value mask
  an explicit cohort id — measured, and caught by
  `test_mi_query_lens_matrix::test_an_exact_cohort_id_in_the_question_wins`.
* ``exclude_fields`` lets a resolver drop ITS OWN fields from the catalogue.
  The rule governs a collision between two DIFFERENT semantic owners; a value
  of ``source_portfolio_label`` naming a book is the scope owner's own reading,
  not a competing claim, so masking it would blind the owner to itself. A
  fixture whose label is literally "Direct Book" proved it.

The value->field resolution itself lives here so there is ONE resolver:
``llm_query_parser._categorical_value_field`` is now a thin alias for
:func:`value_field`, not a second implementation.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional, Tuple

__all__ = ["value_field", "preferred_field", "alias_fields", "value_spans",
           "mask_value_spans"]


def _normalise(text: Any) -> str:
    """Whitespace/underscore-normalised, lower-cased comparison form."""
    return re.sub(r"[\s_]+", " ", str(text or "").strip().lower())


#: Closed-class English function words. A span made of exactly ONE of these is
#: not a reference to a category value, whatever the tape happens to carry.
#:
#: WHY THIS EXISTS. Matching is against the values the BOOK carries, which is
#: the right instinct and the reason nothing here holds a vocabulary of its own.
#: But a real tape carries short codes, and a one-character grade collides with
#: ordinary English. Measured on a live book whose `internal_risk_grade` is
#: A/B/C: the "a" in "Give me A concise overview of the funded portfolio" was
#: claimed as a governed value on that field, the execution never applied it —
#: it is an article, not a filter — and the coverage ledger correctly refused an
#: answer for a concept it could not confirm. 12 of the 166 accepted questions
#: broke that way, every one of them a natural phrasing ("Give me a…",
#: "Show a table of…"). The synthetic book has no single-letter value and showed
#: none of it.
#:
#: The guard is deliberately narrow: ONE token, and only the closed classes —
#: articles, pronouns, copulas/auxiliaries, conjunctions, prepositions,
#: wh-words. A multi-word span is untouched, so a tape value that genuinely is a
#: function word still matches when the reader writes it inside a longer phrase.
#: No content word is listed: "direct", "total", "offer" and their kind are real
#: business values and must keep matching.
_FUNCTION_WORDS = frozenset({
    # articles and determiners
    "a", "an", "the", "this", "that", "these", "those", "any", "some",
    # pronouns
    "i", "me", "my", "we", "us", "our", "you", "your", "it", "its",
    "he", "she", "him", "her", "they", "them", "their",
    # copulas and auxiliaries
    "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "have", "has", "had",
    "will", "would", "can", "could", "shall", "should", "may", "might", "must",
    # conjunctions and negation
    "and", "or", "but", "nor", "not", "no", "if", "than", "as", "so",
    # prepositions
    "of", "in", "on", "at", "to", "for", "by", "with", "from", "into",
    "onto", "off", "out", "up", "down", "about",
    # wh-words and pro-forms
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "there", "here", "then",
})


def _domain_preference(domain: str) -> Tuple[str, ...]:
    """A declared domain's field order, asked of the owner that already walks it.

    NOTHING about any particular domain is written here: this module holds no
    vocabulary of its own — a property `test_governed_span_ownership` asserts
    structurally — and the day it held one would be the day it started
    disagreeing with the owner it copied from.

    A domain whose owner declares no order resolves nothing: its fields may well
    be aliases, but nothing governs which of them a reader's term binds to, and
    choosing anyway is the preference this module refuses to exercise.
    """
    if not domain:
        return ()
    try:
        from .llm_query_parser import domain_field_preference
    except Exception:  # noqa: BLE001 - no owner reachable, no preference
        return ()
    try:
        return tuple(domain_field_preference(domain) or ())
    except Exception:  # noqa: BLE001 - the same
        return ()


def alias_fields(field: Optional[str], semantics: Any = None) -> Tuple[str, ...]:
    """Every OTHER governed field that is a spelling of ``field``'s concept.

    Two fields that declare one ``value_domain`` are aliases of one concept, not
    two concepts — the rule :func:`preferred_field` already binds by. This
    publishes the same fact as a family, so a consumer holding one spelling can
    ask whether a concept is already claimed under another.

    Empty for a field the registry does not carry, for a field that declares no
    domain, and for a domain whose owner declares no preference order — the
    three cases where nothing governs which spellings mean the same thing, and
    guessing is exactly what this module refuses to do.
    """
    entry = _entry(field, semantics)
    domain = (entry or {}).get("value_domain")
    if not domain:
        return ()
    known = {k for k, e in ((semantics or {}).get("fields") or {}).items()
             if (e or {}).get("value_domain") == domain}
    return tuple(f for f in _domain_preference(domain)
                 if f in known and f != field)


def _entry(field: Optional[str], semantics: Any) -> Optional[dict]:
    if not field:
        return None
    return ((semantics or {}).get("fields") or {}).get(str(field))


def preferred_field(fields: Iterable[str],
                    semantics: Any = None) -> Optional[str]:
    """The ONE governed field a claim belongs to, or ``None`` if it is ambiguous.

    ONE claimant is that claimant. SEVERAL claimants are ambiguous — UNLESS
    every one of them declares the same ``value_domain``, because fields drawn
    from one domain are ALIASES of one concept rather than competing concepts.

    Measured, and the reason this exists: `collateral_geography` and
    `geographic_region_obligor` carry the same region spellings on the live
    tape. Two claimants, so every region FILTER resolved to nothing and the
    reader was told ``unknown category: 'london'`` about a region the book
    plainly holds.

    The protection the ambiguity rule was written for is untouched. "lump sum"
    claimed by a product field and a geography field spans two domains — a
    product type bound to geography is exactly the substitution that rule
    stops — so it stays ambiguous and is disclosed. So does a claim by any
    field that declares no domain at all: silence is not agreement.

    Published rather than private because the ambiguity rule has TWO consumers.
    `concept_proposal.vocabulary` decides which values a model may even propose,
    and if it kept counting claimants itself it would withhold as ambiguous the
    very term this function resolves.
    """
    names = list(dict.fromkeys(str(f) for f in (fields or ())))
    if not names:
        return None
    if len(names) == 1:
        return names[0]
    entries = ((semantics or {}).get("fields") or {}) if hasattr(
        semantics, "get") else {}
    if not entries:
        return None
    domains = {str((entries.get(n) or {}).get("value_domain") or "").strip()
               for n in names}
    if len(domains) != 1:
        return None
    domain = next(iter(domains))
    if not domain:
        return None
    claimed = set(names)
    for key in _domain_preference(domain):
        if key in claimed:
            return key
    return None


def value_field(value: str, available_values: Any,
                semantics: Any = None) -> Optional[Tuple[str, str]]:
    """The governed field whose values include ``value`` — ``(field, value)``.

    THE FIELD COMES FROM THE VALUE, not from a fixed choice. The categorical
    filter parser resolved every categorical phrase to the REGION field whatever
    had been named, which is why "what is the balance for lump sum loans?"
    refused citing ``geographic_region_obligor``: a product type was bound to
    geography, matched nothing, and the refusal named a field the reader never
    mentioned.

    Matching is against the values THE BOOK ACTUALLY CARRIES (see
    ``execution_receipt.book_values``), so nothing here holds a vocabulary of
    its own and an asset class the tape does not carry offers no values to
    match.

    A value two governed fields both claim returns ``None``: an ambiguous
    narrowing must be disclosed, never resolved by preference. ``semantics``
    is what lets that rule tell an ambiguity from an ALIAS — see
    :func:`preferred_field`. Without it the strict rule stands, so a caller
    that cannot say which domain a field draws from gets exactly the behaviour
    it had before this parameter existed.
    """
    if not available_values:
        return None
    probe = _normalise(value)
    if not probe:
        return None
    if probe in _FUNCTION_WORDS:
        # ONE owner, so every consumer gets this: the coverage ledger stops
        # inventing a concept nothing can carry, AND `mask_value_spans` stops
        # blanking the word out of the sentence before the scope owner reads it.
        # Fixing it at either call site alone would leave the other corrupting.
        return None
    hits = []
    for field, values in available_values.items():
        for known, spelled in (values.items() if hasattr(values, "items")
                               else ((v, v) for v in values)):
            if _normalise(known) == probe:
                hits.append((field, spelled))
                break
    if not hits:
        return None
    if len(hits) == 1:
        return hits[0]
    chosen = preferred_field([field for field, _ in hits], semantics)
    if chosen is None:
        return None
    # The PREFERRED field's own spelling of the value. The aliases agree about
    # the concept; they need not agree about the capitalisation, and the
    # executor matches case-insensitively either way.
    return next((hit for hit in hits if hit[0] == chosen), None)


def _claimable(available_values: Any, exclude_fields=()) -> Dict[str, str]:
    """``{normalised value: value}`` for every MULTI-WORD unambiguous value.

    Built once per call rather than per candidate span: the book's dimension
    values are already in memory and the alternative is a scan per token.
    """
    if not available_values:
        return {}
    skip = {str(f).strip().lower() for f in (exclude_fields or ())}
    counts: Dict[str, int] = {}
    for field, values in available_values.items():
        if str(field).strip().lower() in skip:
            continue
        seen = set()
        for known in (values.keys() if hasattr(values, "keys") else values):
            # WHITESPACE decides how many words a value is. `_normalise`
            # collapses underscores too, because a book spelling "lump_sum" must
            # match "lump sum" in the question — but an identifier like
            # `direct_002` is ONE word, and treating it as two let it mask the
            # cohort id the reader typed.
            if not re.search(r"\s", str(known or "").strip()):
                continue
            norm = _normalise(known)
            if not norm:
                continue
            seen.add(norm)
        for norm in seen:
            counts[norm] = counts.get(norm, 0) + 1
    return {norm: norm for norm, n in counts.items() if n == 1}


def _span_pattern(claimable: Iterable[str]) -> Optional["re.Pattern"]:
    """One alternation over the claimable values, longest first.

    Longest-first is what makes a value that CONTAINS another value win: with
    both "Gamma Direct" and "Gamma" on the book, the longer claim is the one the
    reader wrote.
    """
    terms = sorted(claimable, key=len, reverse=True)
    if not terms:
        return None
    # The normalised form collapses runs of whitespace and underscores, so the
    # pattern has to match the RAW text the same way: each internal gap becomes
    # "one or more whitespace/underscore characters".
    alts = [r"[\s_]+".join(re.escape(part) for part in term.split(" "))
            for term in terms]
    return re.compile(r"\b(?:" + "|".join(alts) + r")\b", re.IGNORECASE)


def value_spans(text: Optional[str], available_values: Any, exclude_fields=()):
    """``((start, end), ...)`` for every governed categorical VALUE in ``text``.

    THE single source of truth for what a categorical value has claimed.
    Resolvers looking for a scope, a place or a grouping axis mask these spans
    first, so a value's own words cannot be consumed as something else.
    """
    if not text or not available_values:
        return ()
    pattern = _span_pattern(_claimable(available_values, exclude_fields))
    if pattern is None:
        return ()
    return tuple((m.start(), m.end()) for m in pattern.finditer(str(text)))


def mask_value_spans(text: Optional[str], available_values: Any,
                     exclude_fields=()) -> str:
    """``text`` with governed categorical-value spans blanked, offsets preserved.

    Blanking rather than deleting keeps every other offset valid, so a resolver
    reading the remainder sees the sentence it expects — the same discipline
    ``portfolio_lens.mask_scope_phrases`` already uses in the other direction.
    """
    if not text:
        return text or ""
    spans = value_spans(text, available_values, exclude_fields)
    if not spans:
        return str(text)
    out = list(str(text))
    for start, end in spans:
        for i in range(start, end):
            out[i] = " "
    return "".join(out)
