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
    """A period span the question named, or that a governed convention settles."""

    #: How the question said it, for the clarification text.
    label: str
    #: How many governed reporting periods back the span reaches. A month-on-
    #: month comparison is 1; "this year" is 12.
    periods: int
    #: True when the question named a VAGUE recency ("recent", "lately", "the
    #: last few months") and a governed convention resolved it, rather than the
    #: question naming a countable span itself.
    governed: bool = False

    def disclosure(self) -> Optional[str]:
        """What the answer must say about how this span was chosen.

        A governed resolution is not silent. The reader asked something
        imprecise and got a precise window; which window, and on whose
        authority, belongs in the answer.
        """
        if not self.governed:
            return None
        return (f"over the last {self.periods} months (governed recent "
                f"window: seasoning.lending_windows.recent_max_months)")


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

#: VAGUE RECENCY. Phrases that mean "not long ago" without naming a count.
#:
#: These used to resolve to nothing, and the routes then fell through to the
#: widest window they had — earliest snapshot to latest. On three snapshots that
#: silently gave two months, which reads as "a few"; on thirteen it gave twelve,
#: which does not. The fallback is the defect, and it is wrong in both
#: directions: a vague span must never take the widest window by default.
#:
#: They are not clarified either. The governed seasoning configuration already
#: defines RECENT lending as ``lending_windows.recent_max_months``; that
#: convention exists precisely to settle this, and declining to apply our own
#: governed configuration would be hard to defend on a phrase this common.
#: Every phrase here resolves to that one window, so near-identical wordings
#: cannot be handled differently from one another.
#:
#: Only MONTH-scale vagueness appears here. "the last few weeks" names a unit
#: the monthly series cannot express and no governed window covers, so it stays
#: with the granularity clarification — clarification is reserved for spans with
#: no governed convention.
_VAGUE_RECENCY_RE = re.compile(
    r"\b(?:recent|recently|lately|of late)\b"
    r"|\brecent months\b"
    r"|\b(?:a |the )?(?:last |past )?few months\b"
    r"|\ba few months ago\b"
    r"|\bcouple of months\b"
    r"|\bthese days\b"
    r"|\bnowadays\b"
)


def governed_recent_months(config=None) -> int:
    """The governed RECENT lending window, in months.

    Read from the seasoning configuration rather than pinned here, for the same
    reason the front/back boundary is: a client re-cuts "recent" by editing
    config, never by changing code.
    """
    if config is None:
        try:
            from .seasoning import load_seasoning_config
            config = load_seasoning_config()
        except Exception:                       # pragma: no cover - config absent
            return 3
    return int(getattr(config, "recent_max_months", 3) or 3)


def requested_span(question: str, config=None) -> Optional[SpanRequest]:
    """The reporting span the question names, or a governed convention settles.

    Returns None only when the question names no period at all. A question that
    names an imprecise recency resolves to the governed recent window with
    ``governed=True``, so the caller can disclose how the window was chosen.
    """
    text = f" {(question or '').lower().strip()} "
    m = _N_MONTHS_RE.search(text)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 60:
            return SpanRequest(f"the last {n} months", n)
    for pattern, label, periods in _SPANS:
        if re.search(pattern, text):
            return SpanRequest(label, periods)
    # Vague recency is checked LAST, so "the last three months" is read as the
    # countable span it is rather than as vague recency.
    if _VAGUE_RECENCY_RE.search(text):
        months = governed_recent_months(config)
        return SpanRequest(f"the last {months} months", months, governed=True)
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
