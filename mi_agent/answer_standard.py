"""THE ANSWER STANDARD: how every governed answer states a figure (P0 design §18).

The funded, pipeline and forecast answer paths used to word the same things
three ways. On the 2026-09-29 spot check the one book's funded balance read
"£87.1MM" in a funded answer and "£87.1m" in a forecast answer; a pipeline
breakdown named ten brokers where a funded one named three; and the forecast
path hard-coded "£" where the platform resolves the client's reporting currency.
A reader comparing two answers could not tell a difference in wording from a
difference in figures.

This module is the one place those words are decided. It is PRESENTATION ONLY:
it formats a figure it is handed and never derives one.

    money(v)             the platform's money formatter
                         (`mi_agent_api.currency.format_money`) with the chat
                         convention its docstring states — lower-case bn / m / k
                         in answers, BN / MM / K on dashboard tiles — in the
                         request's reporting currency.
    plural(n, noun)      "1 month", "12 months" — never "month(s)".
    breakdown_lead(...)  "{Measure} by {axis} — largest: A x, B y, C z, and N
                         more (N groups)": the measure, the grouping, the
                         leaders and how many groups there are. The table
                         carries every group.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, Tuple

__all__ = ["LEAD_GROUPS", "CHAT_SUFFIXES", "money", "plural", "breakdown_lead",
           "ordered"]

#: How many groups a breakdown's sentence names. The table has them all.
LEAD_GROUPS = 3

#: Magnitude suffixes for money in an answer sentence (billions, millions,
#: thousands). Dashboard tiles use ("BN", "MM", "K").
CHAT_SUFFIXES: Tuple[str, str, str] = ("bn", "m", "k")


def money(value: Any) -> str:
    """A reader-facing amount in the request's reporting currency."""
    from mi_agent_api import currency as currency_mod

    if value is None:
        return "n/a"
    return currency_mod.format_money(float(value), suffixes=CHAT_SUFFIXES)


def plural(count: Any, noun: str, many: Optional[str] = None) -> str:
    """`count` and its noun, agreed: "1 case", "1,330 cases"."""
    n = int(count)
    return f"{n:,} {noun if n == 1 else (many or noun + 's')}"


def breakdown_lead(measure: str, axis: str,
                   groups: Sequence[Tuple[str, str]], *, total: int,
                   word: str = "largest", noun: str = "groups") -> str:
    """A breakdown in words, WITHOUT a closing full stop (the caller adds its
    as-at clause first).

    `groups` are `(label, shown value)` pairs ALREADY ORDERED by the caller —
    largest first for a size, time order never (a time axis is a series, not a
    ranking). The first `LEAD_GROUPS` are named; `total` is how many there are.
    """
    named = list(groups)[:LEAD_GROUPS]
    shown = ", ".join(f"{label} {value}" for label, value in named)
    rest = int(total) - len(named)
    more = f", and {rest:,} more" if rest > 0 else ""
    return f"{measure} by {axis} — {word}: {shown}{more} ({int(total):,} {noun})"


def ordered(pairs: Iterable[Tuple[str, Optional[float]]]) -> Sequence[Tuple[str, float]]:
    """`(label, value)` pairs, largest first; a missing value is left out."""
    kept = [(str(label), float(value)) for label, value in pairs
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            and value == value]
    return sorted(kept, key=lambda pair: pair[1], reverse=True)
