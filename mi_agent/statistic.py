"""mi_agent.statistic — the governed STATISTIC, and whether the one that ran
is the one that was asked for.

A question names three separable things: a MEASURE (LTV), a POPULATION (the back
book) and a STATISTIC (the median). P0 governs the first two. Until P1M nothing
governed the third, and the consequence was a wrong number presented as an
answer::

    "what is the median LTV?"  ->  43.1562   (the weighted average)
                     true median  ->  39.6757

Two independent routes produced it. The deterministic parser never recognised
the word "median" at all, so the field's default statistic was applied and no
record survived that anything else had been asked for. The LLM parser DID emit
``median``; validation correctly refused it; and the repair loop then re-prompted
and accepted a spec in which the statistic had been changed to ``weighted_avg``.
The governance layer detected the violation and the repair negotiated around it.

The invariant this module exists to carry is::

    requested statistic -> governed permission -> executed statistic

A successful answer is only permissible when the executed statistic satisfies the
requested one. Otherwise the answer refuses. A calculation trace that names the
substituted statistic is NOT sufficient: the headline figure must answer the
question that was asked.

Deliberately narrow. P1M governs the statistic families the MI Agent already
needs — sum, count, mean, weighted mean, median, min, max — and adds no new
analytics. A statistic the registry does not permit is refused, never approximated
by a neighbouring one.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional, Sequence

#: The statistic families P1M governs. Anything outside this set is not a
#: governed statistic and is refused rather than approximated.
GOVERNED_STATISTICS = (
    "sum", "count", "count_distinct", "avg", "weighted_avg", "median", "min", "max",
)

#: A request for a plain "average" does not choose BETWEEN the two governed
#: averaging statistics — the field registry does that, and for LTV the house
#: convention is an exposure-weighted average. So "average LTV" is satisfied by
#: ``weighted_avg`` and must not be reported as a substitution.
#:
#: An explicitly WEIGHTED average is a different request: it names the weighting,
#: so only ``weighted_avg`` satisfies it.
#: Aggregations that are ANALYTIC MODES rather than statistics: a contribution
#: decomposes a weighted average across groups, a share is a ratio of two
#: populations, a distribution is a shape. Each has its own governed guard
#: (P1A, P1D), and none of them is a statistic that could stand in for another,
#: so the statistic identity check does not apply to them.
ANALYTIC_MODES = frozenset({"contribution", "share", "distribution", "loan_level",
                            "balance_sum"})

MEAN = "mean"
_MEAN_FAMILY = frozenset({"avg", "weighted_avg"})
_COUNT_FAMILY = frozenset({"count", "count_distinct"})

#: Human labels for refusal text and receipts.
LABELS = {
    "sum": "total", "count": "count", "count_distinct": "distinct count",
    "avg": "average", "weighted_avg": "weighted average", "median": "median",
    "min": "minimum", "max": "maximum", MEAN: "average",
}


def label(statistic: Optional[str]) -> str:
    return LABELS.get(str(statistic or ""), str(statistic or ""))


def satisfies(requested: Optional[str], executed: Optional[str]) -> bool:
    """Does ``executed`` answer a request for ``requested``?

    Identity, with exactly one governed family relaxation: a generic mean request
    is answered by either governed averaging statistic, because which one applies
    is a property of the field and not of the question. Every other pair must
    match exactly — a median is not answered by a weighted average, a maximum is
    not answered by an average, and a total is not answered by a count.
    """
    if not requested:
        return True
    if not executed:
        return False
    req, exe = str(requested), str(executed)
    if req == MEAN:
        return exe in _MEAN_FAMILY
    if req in _COUNT_FAMILY:
        return exe in _COUNT_FAMILY
    return req == exe


def satisfied_by_any(requested: Optional[str],
                     executed: Iterable[Optional[str]]) -> bool:
    """True when at least one executed statistic satisfies the request.

    A multi-measure answer legitimately runs several statistics at once ("balance,
    loan count and weighted-average LTV"). The request is honoured if any of them
    is the statistic that was named; P1E separately guards that no MEASURE was
    dropped, and the two checks compose without either weakening the other.
    """
    executed = list(executed or [])
    if not requested:
        return True
    return any(satisfies(requested, e) for e in executed)


#: Statistic vocabulary, deliberately minimal.
#:
#: Only statistics that a field registry can actually DENY need recognising here,
#: because a denied statistic is what produces the substitution. ``sum`` and
#: ``count`` are permitted almost everywhere and their English is ambiguous —
#: "the total number of loans" is a count, not a sum — so recognising them would
#: buy no safety and risk refusing sound questions.
#:
#: ORDER IS SIGNIFICANT. A superlative and a statistic can appear in the same
#: question — "which region has the **highest average** LTV" — and there the
#: superlative RANKS GROUPS while the statistic is the average. Listing the
#: explicit statistic words first means the average wins that sentence, and
#: "the highest LTV" (no competing statistic) reads as a maximum.
#:
#: A superlative on its own is still not enough: see ``statistic_named``'s
#: ``grouped`` argument, which withholds a min/max reading from a question that
#: names a grouping dimension, because that is a ranking over groups.
#:
#: "Oldest" and "youngest" are deliberately ABSENT. The measure is literally
#: called ``youngest_borrower_age``, so "the youngest borrower age" is the field
#: name rather than a statistic on it, and "oldest borrower" would have to mean
#: the maximum of a field whose own name says youngest. That ambiguity is
#: reported rather than guessed at.
#: "WA" is the estate's own abbreviation for a weighted average — twenty-seven
#: questions in its own corpus use it — and until it was listed here NOTHING
#: owned it. `statistic_named("WA LTV")` returned None, the parser's aggregation
#: reader returned None, and the measure owner matched only "ltv". The weighted
#: average arrived anyway, because it is the REGISTRY DEFAULT for a percent
#: metric: the same answer "LTV" alone produces. Right figure, no reader.
#:
#: That is semantic debt rather than a passing test. With no statistic named,
#: nothing raises a statistic facet, so nothing can reconcile the requested
#: statistic against the executed one — and the day a field's default changes,
#: or the abbreviation is used on a field whose default is a plain mean, the
#: answer moves and no guard notices.
#:
#: WORD BOUNDARIES, NEVER A SUBSTRING. A two-letter alias is exactly the kind
#: that silently eats other words: "Wales", "warehouse", "software" and "await"
#: all contain it. The role is the statistic and only the statistic — the metric
#: still comes from the measure owner, so "WA LTV" is LTV and "WA rate" is the
#: rate.
#: A CONSERVATIVE FALLBACK, used only by a caller that cannot ask the measure
#: owner (see ``is_weight`` below): words that can stand before "weighted"
#: without naming the weight. It is a fallback and not the rule, because a
#: list like this is never finished — measured on the 843-question corpus, it
#: had to be told about "the", then "plus", then "show", each time after a
#: question had already been read wrong. The RULE is that a weighting
#: qualifier names a MEASURE, and only the measure owner knows what does.
_NOT_A_WEIGHT_WORD = (r"the|a|an|and|or|plus|of|to|in|on|at|by|for|with|is|are|"
                      r"was|were|be|been|being|it|we|they|our|its|their|this|"
                      r"that|these|those|not|un|non|equally|equal|simple|"
                      r"heavily|lightly|less|more|most|least|fully|partly|also|"
                      r"show|give|list|display|provide|report|chart|plot|graph|"
                      r"tell|break|split|what|which|how")

_STATISTIC_PHRASES: Sequence[tuple[str, str]] = (
    (r"\bweighted[-\s]+(?:average|avg|mean)\b", "weighted_avg"),
    # P0-E: ANY "<word>-weighted" / "weighted by <word>" is a weighted
    # statistic — "balance-weighted" as much as "exposure-weighted".
    (r"\b(?:(?!(?:" + _NOT_A_WEIGHT_WORD + r")\b)[a-z]{2,})[-\s]+weighted\b(?!\s+by\b)",
     "weighted_avg"),
    (r"\bweighted\s+by\b", "weighted_avg"),
    (r"\bwa\b", "weighted_avg"),
    (r"\bmedian\b", "median"),
    (r"\baverage\b", MEAN),
    (r"\bmean\b", MEAN),
)

#: Superlatives, considered only after the statistic words above and only for a
#: question that is not a grouped ranking. Kept to the smallest commercially
#: natural set — "largest loan" and "highest LTV" are how the questions are
#: actually asked.
_SUPERLATIVE_PHRASES: Sequence[tuple[str, str]] = (
    (r"\b(?:maximum|max|highest|largest|biggest)\b", "max"),
    (r"\b(?:minimum|min|lowest|smallest)\b", "min"),
)

_STATISTIC_RES: tuple = ()
_SUPERLATIVE_RES: tuple = ()


def _statistic_res() -> tuple:
    global _STATISTIC_RES
    if not _STATISTIC_RES:
        _STATISTIC_RES = tuple((re.compile(p, re.I), s)
                               for p, s in _STATISTIC_PHRASES)
    return _STATISTIC_RES


def _superlative_res() -> tuple:
    global _SUPERLATIVE_RES
    if not _SUPERLATIVE_RES:
        _SUPERLATIVE_RES = tuple((re.compile(p, re.I), s)
                                 for p, s in _SUPERLATIVE_PHRASES)
    return _SUPERLATIVE_RES


#: Statistic phrases that also contain MEASURE vocabulary and would otherwise be
#: read as a measure in their own right. "exposure-weighted borrower age" names
#: ONE measure weighted by exposure; without masking, "exposure" resolved to the
#: balance measure, the question became a two-measure request, and the weighted
#: average was refused as a lost statistic. Same discipline P1I-A applies to
#: governed scope phrases and P1J-1 to seasoning phrases.
#: P0-E — THE WEIGHTING QUALIFIER IS THIS OWNER'S. "balance-weighted",
#: "exposure-weighted", "value weighted", "weighted by balance" name HOW a
#: statistic is weighted, never a second measure; the measure set read the
#: "balance" inside "balance-weighted" as a measure and answered a question
#: about the LTV with a balance beside it. Stated as a general rule — any
#: word joined to "weighted" — not as a list of the words a reader has used.
_MASKED_PHRASES = (r"\b(?:(?!(?:" + _NOT_A_WEIGHT_WORD + r")\b)[a-z]{2,})"
                   r"[-\s]+weighted\b(?!\s+by\b)",
                   r"\bweighted[-\s]+(?:average|avg|mean)\b",
                   r"\bweighted\s+by\s+(?:the\s+)?[a-z]+(?:\s+[a-z]+)?\b")

#: "weighted average" / "weighted avg" / "weighted mean" — a weighted statistic
#: whatever the weight is, so this needs no owner to confirm it.
_WEIGHTED_MEAN_RE = re.compile(r"\bweighted[-\s]+(?:average|avg|mean)\b", re.I)

_WEIGHT_WORD_RE = re.compile(
    r"\b((?:(?!(?:" + _NOT_A_WEIGHT_WORD + r")\b)[a-z]{2,})(?:\s+[a-z]+)?)"
    r"[-\s]+weighted\b(?!\s+by\b)"
    r"|\bweighted\s+by\s+(?:the\s+)?([a-z]+(?:\s+[a-z]+)?)\b",
    re.I)
_NOT_A_WEIGHT = frozenset(_NOT_A_WEIGHT_WORD.split("|"))


def weight_word_named(text: Optional[str], is_weight=None) -> Optional[str]:
    """The WORD a weighting qualifier names ("balance" in "balance-weighted",
    "exposure" in "weighted by exposure"), or None. The parser resolves the
    word to a governed weight field with its own measure vocabulary; this
    owner only says which words are the qualifier."""
    spans = _qualifier_spans(text, is_weight)
    return spans[0][2] if spans else None

def mask_weighting_qualifiers(text: Optional[str], is_weight=None) -> str:
    """``text`` with the WEIGHTING QUALIFIERS alone blanked — "balance-weighted",
    "weighted by balance" — offsets preserved. "weighted average" is left, so
    the aggregation intent still reads it. Applied once at the parser's entry
    (P0-E): the "by" of "weighted by balance" is otherwise a grouping marker
    to the dimension reader, and the qualifier's noun a measure to the
    measure reader."""
    if not text:
        return text or ""
    out = list(str(text))
    for start, end, _word in _qualifier_spans(text, is_weight):
        for i in range(start, end):
            out[i] = " "
    return "".join(out)


def mask_statistic_phrases(text: Optional[str], is_weight=None) -> str:
    """``text`` with weighting phrases blanked, preserving offsets.

    Blanking rather than deleting keeps every other span offset valid, which the
    measure/dimension/filter resolvers all depend on.
    """
    if not text:
        return text or ""
    out = list(mask_weighting_qualifiers(text, is_weight))
    for match in _WEIGHTED_MEAN_RE.finditer(str(text)):
        for i in range(match.start(), match.end()):
            out[i] = " "
    return "".join(out)


def _qualifier_spans(text: str, is_weight=None):
    """``((start, end, word), ...)`` for every WEIGHTING QUALIFIER in ``text``.

    ``is_weight`` is the MEASURE OWNER'S test: given the qualifier's word, does
    this book carry a measure by that name? Supplied by the parser, which has
    the semantics; a caller that cannot supply one falls back to the closed
    function-word list above, which is conservative and incomplete by nature.
    """
    out = []
    for m in _WEIGHT_WORD_RE.finditer(str(text or "")):
        word = (m.group(1) or m.group(2) or "").strip().lower()
        if not word:
            continue
        if is_weight is not None:
            # THE LAST WORD IS THE WEIGHT: "loan balance weighted" weights by
            # the balance. Asking about the whole phrase would let a measure
            # word ANYWHERE in it answer for a qualifier it does not name —
            # "balance plus weighted pipeline" is a sum of two things, not a
            # weighting by "balance plus".
            try:
                if not is_weight(word.split()[-1]):
                    continue
            except Exception:  # noqa: BLE001 - an owner that cannot answer declines
                continue
            word = word.split()[-1]
        out.append((m.start(), m.end(), word))
    return tuple(out)


def statistic_named(text: Optional[str], grouped: bool = False,
                    is_weight=None) -> Optional[str]:
    """The statistic a question explicitly asks for, or None.

    First match wins in declaration order, so "weighted average" is read as a
    weighted mean rather than as a bare mean that happens to follow the word
    "weighted", and "the highest AVERAGE LTV" is read as an average.

    ``grouped`` says the question names a grouping dimension. A superlative there
    is a RANKING over groups — "which region has the highest LTV" asks which
    region wins, not for one extreme loan — so no min/max is read from it and the
    existing ranking facet stays in charge. Explicit statistic words are
    unaffected: a grouped question can still ask for an average.
    """
    if not text:
        return None
    if _WEIGHTED_MEAN_RE.search(str(text)) or _qualifier_spans(text, is_weight):
        return "weighted_avg"
    for rx, statistic in _statistic_res():
        if rx.search(str(text)):
            return statistic
    if grouped:
        return None
    for rx, statistic in _superlative_res():
        if rx.search(str(text)):
            return statistic
    return None


def permitted_for(statistic: Optional[str], entry) -> bool:
    """Is ``statistic`` permitted for this registry field entry?

    A generic mean is permitted when the field allows either averaging statistic.
    Counts are permitted everywhere — they count rows, not values.
    """
    allowed = set((entry or {}).get("allowed_aggregations") or ())
    if not statistic:
        return True
    if statistic == MEAN:
        return bool(_MEAN_FAMILY & allowed)
    if statistic in _COUNT_FAMILY:
        return True
    return statistic in allowed


def concrete_for(statistic: Optional[str], entry) -> Optional[str]:
    """The concrete spec aggregation a named statistic resolves to for a field.

    A generic mean resolves to whichever averaging statistic the field governs,
    preferring its own default so the house convention (weighted for LTV) is
    preserved. Returns None when the statistic is not permitted — the caller then
    refuses rather than choosing a neighbour.
    """
    if not statistic:
        return None
    if not permitted_for(statistic, entry):
        return None
    if statistic == MEAN:
        allowed = set((entry or {}).get("allowed_aggregations") or ())
        default = str((entry or {}).get("default_aggregation") or "")
        if default in _MEAN_FAMILY:
            return default
        for candidate in ("weighted_avg", "avg"):
            if candidate in allowed:
                return candidate
        return None
    return statistic
