"""The period a question STATES, and whether the data reaches back that far.

Why this exists
---------------
"How much has the book grown this year?" was answered over 31 May to 30 June.
The period was disclosed in the answer, so nothing was hidden — but "this year"
is a declared element of the question that was not honoured, and under the
no-silent-substitution rule a declared element that cannot be honoured is a
clarification, not a narrower answer with a note attached.

The rule implemented here is: **honour the stated period where the data covers
it, otherwise clarify**. Written that way it needs no revision when the client's
twelve months of history arrive — the same code answers instead of clarifying,
because the data then covers the span.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SpanRequest:
    """A period span the question named."""

    #: How the question said it, for the clarification text.
    label: str
    #: How many governed reporting periods back the span reaches. A month-on-
    #: month comparison is 1; "this year" is 12.
    periods: int


#: Phrases that NAME a span, longest first so "last twelve months" is not read
#: as "last month". Each maps to the number of monthly reporting periods it
#: spans. Only spans a monthly series can express appear here.
_SPANS = (
    (r"\byear to date\b|\bytd\b|\bthis year\b|\bso far this year\b|"
     r"\bover the year\b|\bin the last year\b|\bover the last year\b", "this year", 12),
    (r"\blast twelve months\b|\blast 12 months\b|\bpast twelve months\b|"
     r"\bpast 12 months\b", "the last 12 months", 12),
    (r"\bthis quarter\b|\bthe quarter\b|\blast quarter\b|\bprior quarter\b|"
     r"\bprevious quarter\b|\blast three months\b|\blast 3 months\b", "the quarter", 3),
    (r"\blast six months\b|\blast 6 months\b|\bhalf year\b", "the last 6 months", 6),
    (r"\bthis month\b|\blast month\b|\bprior month\b|\bprevious month\b|"
     r"\bmonth on month\b|\bmonth-on-month\b", "the month", 1),
)
#: "the last N months", written numerically.
_N_MONTHS_RE = re.compile(r"\b(?:last|past|previous|prior)\s+(\d{1,2})\s+months?\b")


def requested_span(question: str) -> Optional[SpanRequest]:
    """The reporting span the question names, or None when it names none."""
    text = f" {(question or '').lower().strip()} "
    m = _N_MONTHS_RE.search(text)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 60:
            return SpanRequest(f"the last {n} months", n)
    for pattern, label, periods in _SPANS:
        if re.search(pattern, text):
            return SpanRequest(label, periods)
    return None


def clarification(span: SpanRequest, available_periods: int) -> str:
    """Why the stated span cannot be honoured, and what would be needed.

    Never proposes the narrower window as a substitute: offering it as the
    answer is the substitution this guard exists to prevent.
    """
    have = max(available_periods - 1, 0)
    return (f"You asked about {span.label}, which spans {span.periods} reporting "
            f"period(s). This book carries {available_periods} governed reporting "
            f"period(s), so the furthest back I can compare is {have} period(s). "
            f"I have not answered over a shorter window in its place — tell me "
            f"which period you want, or ask again once more history is loaded.")
