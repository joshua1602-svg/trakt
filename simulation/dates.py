"""simulation.dates — month-end arithmetic for the simulator.

Deliberately tiny and dependency-free. The framework only ever reports at
month-ends, so every date helper here is exact integer arithmetic rather than a
timedelta approximation — a generated history must be byte-identical on any
machine and in any timezone.
"""

from __future__ import annotations

import calendar
from datetime import date
from typing import List


def parse_iso(value: str) -> date:
    return date.fromisoformat(str(value)[:10])


def month_end(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def add_months(anchor: date, months: int) -> date:
    """Shift by whole months, clamping the day to the target month's length."""
    total = anchor.month - 1 + months
    year = anchor.year + total // 12
    month = total % 12 + 1
    return date(year, month, min(anchor.day, calendar.monthrange(year, month)[1]))


def add_month_ends(anchor: date, months: int) -> date:
    """Shift by whole months and land on the target month-end."""
    total = anchor.month - 1 + months
    return month_end(anchor.year + total // 12, total % 12 + 1)


def months_between(start: date, end: date) -> int:
    """Whole months from *start* to *end*; never negative."""
    return max(0, (end.year - start.year) * 12 + (end.month - start.month))


def days_between(start: date, end: date) -> int:
    return (end - start).days


def month_end_series(start_iso: str, months: int) -> List[str]:
    """``months`` consecutive month-end ISO dates beginning at *start_iso*."""
    anchor = parse_iso(start_iso)
    anchor = month_end(anchor.year, anchor.month)
    return [add_month_ends(anchor, i).isoformat() for i in range(months)]


__all__ = ["add_month_ends", "add_months", "days_between", "month_end",
           "month_end_series", "months_between", "parse_iso"]
