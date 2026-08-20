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


#: A question can name a time UNIT without naming a count. "Based on the last
#: few weeks" pins no number, so there is no span to honour or fail — but it
#: does pin a GRANULARITY, and a series that cannot express weeks cannot answer
#: it. The count being vague does not make the unit vague.
_UNIT_PATTERNS = (
    ("week", r"\bweeks?\b|\bweekly\b|\bfortnight\b"),
    ("month", r"\bmonths?\b|\bmonthly\b"),
    ("quarter", r"\bquarters?\b|\bquarterly\b"),
    ("year", r"\byears?\b|\bannual\b|\bannually\b|\bytd\b"),
)
#: Units ordered coarsest-last, so a series' unit can be compared to a request's.
_UNIT_ORDER = {"week": 0, "month": 1, "quarter": 2, "year": 3}


def requested_unit(question: str) -> Optional[str]:
    """The finest time UNIT the question names, or None."""
    text = f" {(question or '').lower().strip()} "
    for unit, pattern in _UNIT_PATTERNS:
        if re.search(pattern, text):
            return unit
    return None


def finer_than(requested: Optional[str], available: str) -> bool:
    """Is the requested unit finer than the one the series can express?"""
    if not requested:
        return False
    return _UNIT_ORDER.get(requested, 99) < _UNIT_ORDER.get(available, 99)


def granularity_clarification(requested: str, available: str, basis: str) -> str:
    """Why a window stated in one unit cannot be answered from a coarser series.

    Names no substitute window. Offering the coarser one as the answer is the
    substitution this exists to prevent — the same rule as the span
    clarification, applied to granularity rather than to reach.
    """
    return (f"You asked about the last few {requested}s. This figure is measured "
            f"from {basis}, so the finest window it can express is one "
            f"{available}. I have not answered over a {available}ly window in "
            f"its place — ask for a {available}ly view, or for this to be "
            f"measured from a series that carries {requested}s.")


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
