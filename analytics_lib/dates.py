"""Shared deterministic date coercion for MI data.

Source MI tapes routinely carry dates as human-typed strings, and — crucially —
a single column often accretes a MIX of conventions as loans are onboarded from
different packs: some rows normalised to ISO ``2025-11-30``, others left as UK
``28/11/2025``. A naive ``pd.to_datetime(col, dayfirst=True)`` infers ONE format
from the first non-null value and applies it column-wide; with
``errors="coerce"`` every row that doesn't match that inferred format is silently
turned into ``NaT``. On a real November book this dropped 59 of 73 loans into an
"Unknown" vintage bucket (the origination-date column was part ISO, part UK).

``coerce_dates`` parses each element individually (``format="mixed"``) with a
``dayfirst`` hint, so an ISO/UK-mixed column parses correctly row by row instead
of collapsing to a single inferred format. This mirrors ``numeric.coerce_numeric``
— one helper so the MI prep layer, the cohort/vintage engine and the evolution
breakdowns all parse dates identically, with no per-call ``to_datetime`` regexes
drifting apart.
"""

from __future__ import annotations

import pandas as pd


def coerce_dates(series: pd.Series, *, dayfirst: bool = True) -> pd.Series:
    """Parse a possibly string-formatted, mixed-convention date column to
    datetime, deterministically and per element.

    * Mixed ISO / UK (``dd/mm/yyyy``) columns parse row by row rather than
      locking onto one format inferred from the first value.
    * ``dayfirst`` disambiguates UK ``dd/mm/yyyy`` (default ``True``).
    * Already-datetime input is returned unchanged (``format="mixed"`` cannot be
      applied to a datetime dtype).
    * Anything still unparseable becomes ``NaT`` — same contract as
      ``pd.to_datetime(errors="coerce")``.

    Defensive: a one-column ``DataFrame`` (which happens when the caller selects a
    column name that is DUPLICATED in the frame) is squeezed to a Series, mirroring
    ``coerce_numeric``.
    """
    if series is None:
        return series
    if isinstance(series, pd.DataFrame):
        if series.shape[1] == 1:
            series = series.iloc[:, 0]
        else:
            name = series.columns[0] if len(series.columns) else "?"
            raise ValueError(
                f"coerce_dates got a multi-column DataFrame (duplicate column '{name}'?)"
            )
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    # Two-pass, so a MIXED column parses each convention correctly:
    #   1. ISO 8601 (``YYYY-MM-DD``) rows — parsed strictly ISO. ``dayfirst`` must
    #      NOT touch these: ``format="mixed", dayfirst=True`` misreads ``2025-02-10``
    #      as day-first (→ October), silently corrupting a clean ISO tape.
    #   2. The remainder (UK ``dd/mm/yyyy`` and friends) — parsed per element with
    #      ``dayfirst`` so ``28/11/2025`` is 28 Nov, not NaT.
    iso = pd.to_datetime(series, errors="coerce", format="ISO8601")
    if iso.notna().all():
        return iso  # fast path: a wholly-ISO column (the common case)
    remainder = series.where(iso.isna())
    uk = pd.to_datetime(remainder, errors="coerce", dayfirst=dayfirst, format="mixed")
    return iso.fillna(uk)
