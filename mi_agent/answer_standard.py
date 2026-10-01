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
    region_note(...)     which regions, on which location — the property's
                         or the borrower's — and what is in no region.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, Tuple

__all__ = ["LEAD_GROUPS", "CHAT_SUFFIXES", "money", "signed_money", "percent",
           "signed_percent", "plural", "breakdown_lead", "ordered",
           "region_note"]

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


def signed_money(value: Any) -> str:
    """A change in money with its sign: "+£5.4m", "-£1.5m"."""
    from mi_agent_api import currency as currency_mod

    if value is None:
        return "n/a"
    return currency_mod.format_money(float(value), signed=True,
                                     suffixes=CHAT_SUFFIXES)


def signed_percent(value: Any) -> str:
    """A relative change with its sign, one decimal place: "+3.1%"."""
    return "n/a" if value is None else f"{float(value):+.1f}%"


def percent(value: Any, *, fraction: bool = False) -> str:
    """A percentage to one decimal place, as the dashboard shows one. `fraction`
    says the owner published a share (0.7) rather than points (70.0)."""
    if value is None:
        return "n/a"
    points = float(value) * 100.0 if fraction else float(value)
    return f"{points:.1f}%"


def plural(count: Any, noun: str, many: Optional[str] = None) -> str:
    """`count` and its noun, agreed: "1 case", "1,330 cases"."""
    n = int(count)
    return f"{n:,} {noun if n == 1 else (many or noun + 's')}"


def breakdown_lead(measure: str, axis: str,
                   groups: Sequence[Tuple[str, str]], *, total: int,
                   word: str = "largest", noun: str = "groups",
                   lead: Optional[int] = None) -> str:
    """A breakdown in words, WITHOUT a closing full stop (the caller adds its
    as-at clause first).

    `groups` are `(label, shown value)` pairs ALREADY ORDERED by the caller —
    largest first for a size, time order never (a time axis is a series, not a
    ranking). The first `LEAD_GROUPS` are named — or `lead` of them, for a
    ranking that asked for that many; `total` is how many there are.
    """
    named = list(groups)[:(lead or LEAD_GROUPS)]
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


#: How a reader names each geography basis (`mi_agent.mi_geography`). A column
#: that carries no basis is the book's own region field, and is said so.
_BASIS_WORDS = {"collateral": "the property's location",
                "borrower": "the borrower's address"}
_NO_BASIS_WORDS = "the book's own region field"


def region_note(source_field_rows: Optional[dict] = None, *, noun: str = "case",
                unmapped: int = 0, unmapped_amount: float = 0.0,
                unplaced_amount: float = 0.0, unplaced_of: str = "",
                counts: bool = True) -> str:
    """The sentence every reporting-region answer carries, WITHOUT a closing
    full stop: "Regions are the client's reporting regions, by the property's
    location; 2 cases (£150k) whose region has no governed mapping are in no
    region". `counts=False` names two bases without their row counts — for an
    answer over part of a book, where the book's counts are not its own.

    WHICH LOCATION is read from the harmonisation's own record of the column
    each row's region came from (`engine.region_taxonomy.FIELD_SOURCE_FIELD`),
    through `mi_geography`'s basis of that column — never assumed. A book whose
    regions rest on two bases says how many rows rest on each.
    """
    from mi_agent import mi_geography as _geo

    by_basis: dict = {}
    for column, rows in (source_field_rows or {}).items():
        words = _BASIS_WORDS.get(str(_geo.basis_of_field(column) or ""),
                                 _NO_BASIS_WORDS)
        by_basis[words] = by_basis.get(words, 0) + int(rows)
    note = "Regions are the client's reporting regions"
    if len(by_basis) == 1:
        note += f", by {next(iter(by_basis))}"
    elif by_basis:
        ranked = sorted(by_basis.items(), key=lambda kv: -kv[1])
        note += (" — by " + " and ".join(
            f"{words} for {plural(rows, noun)}" for words, rows in ranked)
                 if counts else
                 ", by " + " or ".join(words for words, _ in ranked))
    if unmapped:
        note += (f"; {plural(unmapped, noun)} ({money(unmapped_amount)}) whose "
                 f"region has no governed mapping {'is' if int(unmapped) == 1 else 'are'} "
                 f"in no region")
    if unplaced_amount:
        note += (f"; {money(unplaced_amount)} of {unplaced_of or 'the figure'} "
                 f"has no region and is in none")
    return note
