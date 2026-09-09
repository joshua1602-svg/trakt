"""THE ONE NORMALISER. Every reader of a lender's sentence sees the same text.

Before this module three readers each had a private rule for the same
punctuation: the parser lower-cased and nothing else, so "loan-to-value" and
"loan to value" were two different questions to it; the borrowing-base route
replaced every hyphen; the analytical-intent boundary kept hyphens as word
characters. A hyphenated question could therefore bind a measure in one reader
and miss it in the next, and the row noun `loan` inside "loan-to-value" was
free for a fourth reader to count.

The rule is small and it is stated once:

  * lower-case;
  * a hyphen standing BETWEEN TWO LETTERS is a space ("loan-to-value",
    "month-on-month", "buy-to-let");
  * nothing else moves. "2025-06", "top-10", "-5%" and a leading or trailing
    hyphen are left alone: a date, a rank and a sign are not compound words.

OFFSET-PRESERVING BY CONSTRUCTION. Every span owner in the estate — measure
hits, value spans, grouping regions, the claimed-span mask — trades in offsets
into the parser's text, so the normalised sentence has exactly the length of
the sentence it came from: one character in, one character out.
"""
from __future__ import annotations

import re

_LETTER_HYPHEN_LETTER = re.compile(r"(?<=[a-z])-(?=[a-z])")


def normalise_question(text: str | None) -> str:
    """The sentence every reader sees. Same length as ``text``."""
    lowered = str(text or "").lower()
    return _LETTER_HYPHEN_LETTER.sub(" ", lowered)


def normalise_term(term: str | None) -> str:
    """The same rule, for a vocabulary entry. A term written "month-on-month"
    must meet a sentence that has already been normalised, so the term is
    normalised by the same owner rather than by a second copy of the rule."""
    return normalise_question(term)


__all__ = ["normalise_question", "normalise_term"]
