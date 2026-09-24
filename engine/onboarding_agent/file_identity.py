"""engine.onboarding_agent.file_identity — the same source file, whatever month.

A lender sends the same extract every month under a name that carries the date:

    PropertyExtract - Omni 2026_09_01.xlsx
    PropertyExtract - Omni 2026_08_01.xlsx

Every answer an operator gives about a FILE — this column of this file is set
aside, this file is authoritative for that field, the source is this column of
this file — was keyed on the exact name. The next month's pack has a different
name, so none of those answers applied, and a backfill of twelve historic
tapes would have asked the same questions twelve more times.

``file_family`` is the name with its date removed. Two files are the same
source when their families match. The exact name still wins where both are
present; the family is the fallback that lets an answer about August's file
find July's.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

_MONTHS = ("jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|"
           "february|march|april|june|july|august|september|october|"
           "november|december")

#: Date tokens a delivery file name carries, most specific first.
_DATE_PATTERNS = (
    # 2026_09_01, 2026-09-01, 2026.09.01, 20260901
    r"(?<!\d)(?:19|20)\d{2}[-_. ]?(?:0[1-9]|1[0-2])[-_. ]?(?:0[1-9]|[12]\d|3[01])(?!\d)",
    # 01_09_2026, 01-09-2026
    r"(?<!\d)(?:0[1-9]|[12]\d|3[01])[-_. ](?:0[1-9]|1[0-2])[-_. ](?:19|20)\d{2}(?!\d)",
    # 2026_09, 2026-09, 202609
    r"(?<!\d)(?:19|20)\d{2}[-_. ]?(?:0[1-9]|1[0-2])(?!\d)",
    # Sep 2026, September_2026, 2026 Sep
    rf"(?<![a-z])(?:{_MONTHS})[-_. ]?(?:19|20)\d{{2}}(?!\d)",
    rf"(?<!\d)(?:19|20)\d{{2}}[-_. ]?(?:{_MONTHS})(?![a-z])",
)

DATE_TOKEN = "{date}"


def file_family(name: Optional[str]) -> str:
    """The file's name without its date, spacing or case.

    ``"PropertyExtract - Omni 2026_09_01.xlsx"`` ->
    ``"propertyextract - omni {date}.xlsx"``. A name with no date is simply
    normalised, so an undated file is its own family.
    """
    text = str(name or "").strip().lower()
    if not text:
        return ""
    for pattern in _DATE_PATTERNS:
        text = re.sub(pattern, DATE_TOKEN, text)
    return " ".join(text.split())


def same_source(a: Optional[str], b: Optional[str]) -> bool:
    """Are these two file names the same source, allowing for the date?"""
    if not a or not b:
        return False
    return a == b or file_family(a) == file_family(b)


def resolve(name: Optional[str], available: Iterable[str]) -> str:
    """The file in ``available`` that ``name`` refers to: itself if present,
    else the one file of the same family. ``""`` when there is none, or when
    more than one file shares the family — guessing between two would be a
    silent decision."""
    names = [str(n) for n in available or [] if n]
    if not name:
        return ""
    if name in names:
        return name
    family = file_family(name)
    hits = [n for n in names if file_family(n) == family]
    return hits[0] if len(hits) == 1 else ""
