"""
file_profiler.py
================

PART 4 — deterministic column profiling for structured files.

For each column we emit a :class:`ColumnProfile` with type inference, null
rates, uniqueness, redacted samples and date/identifier flags.

Redaction note (PART 4 requirement)
-----------------------------------
The Gate 1 redactor (``llm_mapper_agent._redact_sample``) classifies anything
with 6+ digits-and-separators as ``<PHONE>``, which wrongly tags ISO dates
(``2026-01-31``) and long IDs as phone numbers. This module provides an
*improved* redactor, :func:`redact_value`, that:

  * classifies dates as ``<DATE>`` (checked first),
  * classifies long numeric/alphanumeric identifiers as ``<ID>``,
  * only then falls back to ``<PHONE>`` for genuine phone-shaped values.

We intentionally keep the Gate 1 redactor untouched to avoid regressing
existing behaviour, and add dedicated tests for the new one.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import pandas as pd

from engine.gate_1_alignment.semantic_alignment import normalise_name, tokenize
from .onboarding_models import ColumnProfile

# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

# Order matters: dates first, then postcode/email, then IDs, then phones.
_DATE_PATTERNS = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?\b"),   # ISO 2026-01-31
    re.compile(r"\b\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}\b"),              # 31/01/2026
    re.compile(r"\b\d{4}/\d{2}/\d{2}\b"),                              # 2026/01/31
]
_POSTCODE_RE = re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b")
# Long identifier: 8+ digits, or alphanumeric token with 6+ chars containing digits.
_NUMERIC_ID_RE = re.compile(r"\b\d{8,}\b")
_ALNUM_ID_RE = re.compile(r"\b(?=[A-Za-z0-9]*\d)[A-Za-z]+\d[A-Za-z0-9]{4,}\b")
# Phone candidate: 7+ digits using space/dash/paren separators (NOT the dot —
# we never want to treat a decimal financial amount like 148250.55 as a phone).
_PHONE_RE = re.compile(r"\+?\(?\d[\d\s()\-]{5,}\d")
_PHONE_SEPARATORS = " -()"


def _phone_sub(match: "re.Match") -> str:
    s = match.group(0)
    # Only redact if it actually looks like a formatted phone number, i.e. it
    # contains a separator. A run of bare digits is an identifier/amount, not PII.
    return "<PHONE>" if any(c in s for c in _PHONE_SEPARATORS) else s


def redact_value(value: object, max_chars: int = 40) -> str:
    """Redact PII / identifiers from a single sample value.

    Dates are classified as ``<DATE>`` (not ``<PHONE>``), long identifiers as
    ``<ID>``, and plain numeric amounts are left visible. This is the
    onboarding-v2 replacement for the Gate 1 redactor.
    """
    s = str(value)[:max_chars]

    for pat in _DATE_PATTERNS:
        s = pat.sub("<DATE>", s)
    s = _POSTCODE_RE.sub("<UK_POSTCODE>", s)
    s = _EMAIL_RE.sub("<EMAIL>", s)
    s = _NUMERIC_ID_RE.sub("<ID>", s)
    s = _ALNUM_ID_RE.sub("<ID>", s)
    s = _PHONE_RE.sub(_phone_sub, s)
    return s


# ---------------------------------------------------------------------------
# Type inference
# ---------------------------------------------------------------------------

_BOOL_TOKENS = {"y", "n", "yes", "no", "true", "false", "t", "f"}
_ID_NAME_HINTS = ("id", "identifier", "number", "reference", "account", "code")
_DATE_NAME_HINTS = ("date", "dt", "maturity", "origination")


def _looks_like_date_series(non_null: pd.Series) -> bool:
    if non_null.empty:
        return False
    sample = non_null.astype(str).head(50)
    hits = 0
    for v in sample:
        if any(pat.search(v) for pat in _DATE_PATTERNS):
            hits += 1
    return hits >= max(1, int(0.6 * len(sample)))


def _infer_type(col_name: str, series: pd.Series) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return "string"

    name_norm = col_name.lower()
    name_tokens = set(tokenize(col_name))

    # Dates: by name hint or by value shape.
    if any(h in name_norm for h in _DATE_NAME_HINTS) or _looks_like_date_series(non_null):
        if _looks_like_date_series(non_null) or any(h in name_norm for h in _DATE_NAME_HINTS):
            # Only call it a date if values are actually date-shaped.
            if _looks_like_date_series(non_null):
                return "date"

    # Booleans.
    distinct = {str(v).strip().lower() for v in non_null.unique()[:10]}
    if distinct and distinct.issubset(_BOOL_TOKENS):
        return "boolean"

    # Numeric.
    coerced = pd.to_numeric(non_null, errors="coerce")
    numeric_ratio = coerced.notna().mean()
    if numeric_ratio >= 0.95:
        # Identifier-ish numeric (all integers, name hints id) -> identifier
        is_integer = (coerced.dropna() % 1 == 0).all()
        looks_id = any(t in _ID_NAME_HINTS for t in name_tokens) or any(
            h in name_norm for h in _ID_NAME_HINTS
        )
        if is_integer and looks_id:
            return "identifier"
        return "integer" if is_integer else "decimal"

    # Non-numeric identifier (alphanumeric codes with id-ish name).
    if any(h in name_norm for h in _ID_NAME_HINTS):
        return "identifier"

    return "string"


def _is_likely_identifier(col_name: str, series: pd.Series, inferred_type: str) -> bool:
    if inferred_type == "identifier":
        return True
    non_null = series.dropna()
    n = len(non_null)
    if n == 0:
        return False
    # Require an id-ish name AND near-perfect uniqueness — avoids tagging unique
    # numeric amounts (balances, valuations) as identifiers.
    name_norm = col_name.lower()
    name_hint = any(h in name_norm for h in _ID_NAME_HINTS)
    if not name_hint:
        return False
    return non_null.nunique() / n >= 0.98


def _is_likely_reporting_date(col_name: str, series: pd.Series, inferred_type: str) -> bool:
    if inferred_type != "date":
        return False
    name_norm = col_name.lower().replace(" ", "_")
    hints = ("reporting", "report_date", "cut_off", "cutoff", "data_cut", "as_of", "as_at")
    return any(h in name_norm for h in hints)


def profile_column(
    file_path: str,
    file_name: str,
    sheet_name: str,
    col_name: str,
    series: pd.Series,
    max_samples: int = 5,
) -> ColumnProfile:
    non_null = series.dropna()
    n_total = len(series)
    n_null = int(series.isna().sum())
    null_rate = round(n_null / n_total, 3) if n_total else 0.0

    inferred = _infer_type(col_name, series)

    samples = [redact_value(v) for v in non_null.drop_duplicates().head(max_samples).tolist()]

    profile = ColumnProfile(
        file_path=file_path,
        file_name=file_name,
        sheet_name=sheet_name,
        source_column=str(col_name),
        normalized_column_name=normalise_name(col_name),
        inferred_type=inferred,
        non_null_count=int(len(non_null)),
        null_rate=null_rate,
        unique_count=int(non_null.nunique()),
        sample_values_redacted=samples,
    )

    if inferred in ("integer", "decimal"):
        coerced = pd.to_numeric(non_null, errors="coerce").dropna()
        if not coerced.empty:
            profile.min_value = str(coerced.min())
            profile.max_value = str(coerced.max())
    elif inferred == "date":
        parsed = pd.to_datetime(non_null, errors="coerce").dropna()
        if not parsed.empty:
            profile.date_min = parsed.min().strftime("%Y-%m-%d")
            profile.date_max = parsed.max().strftime("%Y-%m-%d")

    profile.likely_identifier = _is_likely_identifier(col_name, series, inferred)
    profile.likely_reporting_date = _is_likely_reporting_date(col_name, series, inferred)
    return profile


def _read_structured(path: Path):
    """Yield (sheet_name, dataframe) for a structured file."""
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        xl = pd.ExcelFile(path)
        for sheet in xl.sheet_names:
            yield str(sheet), xl.parse(sheet)
    else:
        yield "", pd.read_csv(path, low_memory=False)


def profile_file(path: Path) -> List[ColumnProfile]:
    """Profile every column of a structured file."""
    path = Path(path)
    profiles: List[ColumnProfile] = []
    if path.suffix.lower() not in (".csv", ".xlsx", ".xls"):
        return profiles
    try:
        for sheet, df in _read_structured(path):
            for col in df.columns:
                profiles.append(
                    profile_column(str(path), path.name, sheet, str(col), df[col])
                )
    except Exception:
        return profiles
    return profiles
