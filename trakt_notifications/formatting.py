#!/usr/bin/env python3
"""Display formatting — one place, so every card reads the same way.

Deliberately delegates to :mod:`mi_agent_api.insight_generators`, which already
owns how a Trakt money amount and a Trakt percentage are written. A second
formatter here would be a second answer to "is this £503.4m or £503,400,000",
and two channels quoting the same governed number differently is exactly the
class of divergence this project spends effort preventing.

Only the few things a notification needs and an insight headline does not — a
count with a thousands separator, a date written for a human, a signed
comparison against a baseline — are added.
"""

from __future__ import annotations

import datetime as _dt
from typing import Optional

from mi_agent_api.insight_generators import (  # noqa: F401  (re-exported)
    direction,
    money,
    pct,
    signed_money,
    signed_pct,
)

_MONTHS = ("January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December")


def count(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{int(round(float(value))):,}"


def human_date(value: Optional[str]) -> str:
    """``2026-08-07`` → ``7 August``. Falls back to the raw string.

    The year is omitted because a notification is read within days of the date
    it carries; it is restored by :func:`full_date` wherever a card shows two
    dates that could otherwise be mistaken for the same reporting period.
    """
    if not value:
        return "—"
    try:
        d = _dt.date.fromisoformat(str(value)[:10])
    except ValueError:
        return str(value)
    return f"{d.day} {_MONTHS[d.month - 1]}"


def full_date(value: Optional[str]) -> str:
    if not value:
        return "—"
    try:
        d = _dt.date.fromisoformat(str(value)[:10])
    except ValueError:
        return str(value)
    return f"{d.day} {_MONTHS[d.month - 1]} {d.year}"


def against_baseline(difference_pct: Optional[float], *,
                     baseline: str = "the five-week average") -> Optional[str]:
    """``9% above the five-week average`` — or ``None`` when there is nothing
    to say.

    ``None`` rather than "0% above": a caller must be able to drop the clause
    entirely instead of printing a comparison that carries no information.
    """
    if difference_pct is None:
        return None
    value = abs(difference_pct)
    if value < 0.05:
        return f"in line with {baseline}"
    word = "above" if difference_pct > 0 else "below"
    return f"{value:.0f}% {word} {baseline}"


def pp(value: Optional[float], dp: int = 2) -> str:
    """Percentage POINTS, which is not the same statement as a percentage."""
    if value is None:
        return "—"
    return f"{value:+.{dp}f}pp"
