"""Plan period labels -> a governed month. One reader, shared by the runtimes.

Lifted out of `plan_temporal_runtime` unchanged, so the pipeline runtime's dated
periods (P0 Change 3, D7) read "October 2025" by exactly the rule the funded
temporal runtime reads it with — and so that runtime, which must never reach
the funded snapshot catalogue, can read a label without importing the module
that owns that catalogue.

It reads `period.labels`, which is governed plan content rather than the
reader's sentence, and it uses no regular expressions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

#: Words that prefix a span label without changing which period it names.
#: Stripped so "since March" and "March" resolve to the same governed anchor.
_ANCHOR_PREFIXES = ("since ", "from ", "starting ", "beginning ", "as at ",
                    "as of ", "in ", "during ", "for ", "the ", "back to ")

_MONTHS: Mapping[str, int] = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12,
    "dec": 12,
}


def _normalise(label: Any) -> str:
    """A label reduced to the words that name a period. No regular expressions.

    THE ONE STRING THIS MODULE READS is `period.labels`, and it is governed plan
    content rather than the reader's sentence: the compiler put it there, and
    `interpretation_v2.intent` already refuses a payload whose label carries a
    date or a snapshot id. Nothing else on the plan is read as prose.
    """
    words = str(label or "").strip().lower()
    for character in ".,;:!?'\"()[]":
        words = words.replace(character, " ")
    return " ".join(words.split())


@dataclass(frozen=True)
class PeriodAnchor:
    """A month, and optionally a year, that a plan label named.

    Never a date. A month with no year is matched against the months the
    catalogue actually carries, and it is the CATALOGUE that supplies the day.
    """

    month: int
    year: Optional[int] = None
    label: str = ""


def parse_anchor(label: Any) -> Optional[PeriodAnchor]:
    """The governed period a plan label names, or None if it names none.

    Recognises a month name, optionally with a four-digit year, after the
    prefixes a span phrase puts in front of it. Everything else returns None,
    and the caller clarifies rather than choosing a period on the reader's
    behalf.

    A four-digit year alone is NOT an anchor: "2025" names twelve reporting
    periods, and picking one of them would be exactly the substitution this
    module refuses.
    """
    words = _normalise(label)
    if not words:
        return None
    changed = True
    while changed:
        changed = False
        for prefix in _ANCHOR_PREFIXES:
            if words.startswith(prefix):
                words = words[len(prefix):]
                changed = True
    tokens = [token for token in words.split() if token]
    month: Optional[int] = None
    year: Optional[int] = None
    for token in tokens:
        if token in _MONTHS:
            if month is not None and _MONTHS[token] != month:
                return None                      # two months in one label
            month = _MONTHS[token]
            continue
        if len(token) == 4 and token.isdigit():
            candidate = int(token)
            if 1900 <= candidate <= 2999:
                if year is not None and year != candidate:
                    return None
                year = candidate
                continue
        return None                              # a word this contract cannot read
    if month is None:
        return None
    return PeriodAnchor(month=month, year=year, label=_normalise(label))
