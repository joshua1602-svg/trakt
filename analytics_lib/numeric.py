"""Shared deterministic numeric coercion for MI data.

Source MI tapes routinely store amounts as human / accounting-formatted strings:
thousands separators (``"111,757.38"``), currency symbols (``"£1,200"``),
accounting negatives (``"(1,234.50)"``) and surrounding whitespace. A naive
``pd.to_numeric(..., errors="coerce")`` turns every one of those into ``NaN``,
which silently zeroes balance sums and empties buckets (the exact failure seen on
real packs where ``current_outstanding_balance`` summed to 0.0).

This mirrors the legacy deterministic parser in the Streamlit ERM app so the MI
prep layer, the bucketing engine and the query executor parse numbers identically
— one helper, no duplicated regexes drifting apart.
"""

from __future__ import annotations

import pandas as pd


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Parse a possibly string-formatted numeric column to float, deterministically.

    Handles commas, currency symbols (£/$/€ etc.), surrounding whitespace,
    accounting negatives ``(1,234.50)`` and blanks; already-numeric input is
    returned unchanged (coerced to numeric dtype). Anything still unparseable
    becomes ``NaN`` — same contract as ``pd.to_numeric(errors="coerce")``.
    """
    if series is None:
        return series
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    s = series.astype("string").str.strip()
    # Accounting negative: "(1,234.50)" -> "-1,234.50".
    s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    # Strip everything that is not a digit, sign, decimal point or comma
    # (drops currency symbols, %, spaces, stray text).
    s = s.str.replace(r"[^\d\-\.\,]", "", regex=True)
    # Thousands separators.
    s = s.str.replace(",", "", regex=False)
    s = s.replace({"": pd.NA, "-": pd.NA})
    return pd.to_numeric(s, errors="coerce")
