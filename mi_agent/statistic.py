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
#: Ranking words ("highest", "largest") are deliberately ABSENT: in "which region
#: has the highest average LTV" they name a ranking over groups, not the statistic
#: applied to the measure. The existing extreme-value guard owns min/max, and
#: ``satisfies`` still covers them for any spec that carries one.
_STATISTIC_PHRASES: Sequence[tuple[str, str]] = (
    (r"\bweighted[-\s]+(?:average|avg|mean)\b", "weighted_avg"),
    (r"\bmedian\b", "median"),
    (r"\baverage\b", MEAN),
    (r"\bmean\b", MEAN),
)

_STATISTIC_RES: tuple = ()


def _statistic_res() -> tuple:
    global _STATISTIC_RES
    if not _STATISTIC_RES:
        _STATISTIC_RES = tuple((re.compile(p, re.I), s)
                               for p, s in _STATISTIC_PHRASES)
    return _STATISTIC_RES


def statistic_named(text: Optional[str]) -> Optional[str]:
    """The statistic a question explicitly asks for, or None.

    First match wins in declaration order, so "weighted average" is read as a
    weighted mean rather than as a bare mean that happens to follow the word
    "weighted".
    """
    if not text:
        return None
    for rx, statistic in _statistic_res():
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
