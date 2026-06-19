"""snapshot.keys — deterministic, storage-neutral key derivation.

Phase 2 snapshot/history layer. Pure functions for the stable identifiers the
snapshot store relies on:

  * ``compute_source_file_id`` — content hash of a raw tape (file or bytes);
  * ``make_snapshot_id``       — deterministic snapshot identity;
  * ``normalise_key_part``     — safe string component for composite keys;
  * ``make_pipeline_opportunity_id`` — deterministic key for *unfunded* pipeline
    opportunities (so the same opportunity is trackable week-to-week);
  * ``select_stable_loan_key`` — pick a real, stable loan id for funded loans.

No imports from the legacy ``analytics/`` Streamlit app and no Azure. The legacy
deterministic opportunity-key *idea* is used only as a design reference; no
legacy code is copied. Funded ``loan_id`` and pipeline ``opportunity_id`` are
kept in distinct namespaces.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

# Default candidate fields for a real, stable funded-loan key (highest first).
DEFAULT_LOAN_KEY_CANDIDATES: Sequence[str] = (
    "loan_id",
    "loan_identifier",
    "loan_policy_number",
    "account_number",
)

# Default candidate fields composing a deterministic pipeline opportunity key.
# (Design reference: the legacy SHA1 opportunity key built from
# KFI/account/broker/product/amount/application-date — re-expressed here.)
DEFAULT_OPPORTUNITY_FIELDS: Sequence[str] = (
    "kfi_number",
    "account_number",
    "broker",
    "broker_channel",
    "product",
    "erm_product_type",
    "loan_amount",
    "amount",
    "application_date",
    "kfi_date",
)

_UNSAFE = re.compile(r"[^a-z0-9._-]+")


def _row_get(row: Any, key: str) -> Any:
    """Read *key* from a dict-like or pandas Series row; ``None`` if absent."""
    if row is None:
        return None
    if hasattr(row, "get"):
        return row.get(key)
    try:
        return row[key]  # pragma: no cover - mapping fallback
    except (KeyError, TypeError, IndexError):
        return None


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    # NaN is the only value not equal to itself.
    if isinstance(value, float) and value != value:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def normalise_key_part(value: Any) -> str:
    """Return a safe, lowercase string component for a composite key.

    Dates/datetimes are rendered ISO; other values are stringified, lowercased,
    whitespace-collapsed, and stripped of characters outside ``[a-z0-9._-]``.
    """
    if _is_empty(value):
        return ""
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    text = str(value).strip().lower()
    text = _UNSAFE.sub("-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text


def compute_source_file_id(path_or_bytes: Any) -> str:
    """Deterministic content hash of a raw tape.

    Accepts a filesystem path (``str``/``Path``) or raw ``bytes``. Returns a
    ``sha256:<hex>`` identifier so identity is storage-portable (not an
    Azure etag).
    """
    if isinstance(path_or_bytes, (bytes, bytearray)):
        data = bytes(path_or_bytes)
    elif isinstance(path_or_bytes, (str, Path)):
        data = Path(path_or_bytes).read_bytes()
    else:
        raise TypeError("compute_source_file_id expects a path or bytes")
    return "sha256:" + hashlib.sha256(data).hexdigest()


def make_snapshot_id(client_id: Any, route: Any, reporting_date: Any,
                     source_file_id: Any) -> str:
    """Deterministic snapshot id from its identifying tuple."""
    parts = [
        normalise_key_part(client_id),
        normalise_key_part(route),
        normalise_key_part(reporting_date),
        normalise_key_part(source_file_id),
    ]
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return "snap_" + digest[:16]


def make_pipeline_opportunity_id(
    row: Any, field_candidates: Optional[Iterable[str]] = None
) -> Optional[str]:
    """Deterministic SHA1-style key for an unfunded pipeline opportunity.

    Built from the available candidate fields (in a fixed order) so equivalent
    rows hash identically across uploads. Returns ``None`` when no candidate
    field is populated (caller should then record a missing-key issue).
    """
    candidates = list(field_candidates or DEFAULT_OPPORTUNITY_FIELDS)
    parts = []
    for field in candidates:
        value = _row_get(row, field)
        if _is_empty(value):
            continue
        parts.append(f"{field}={normalise_key_part(value)}")
    if not parts:
        return None
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return "opp_" + digest[:16]


def select_stable_loan_key(
    row: Any, candidates: Optional[Iterable[str]] = None
) -> Optional[str]:
    """Return the first populated stable loan key for a funded loan.

    Preserves the original id value (only stripped) so real loan identifiers are
    not mangled. Returns ``None`` when none of the candidates are populated.
    """
    for field in (candidates or DEFAULT_LOAN_KEY_CANDIDATES):
        value = _row_get(row, field)
        if not _is_empty(value):
            return str(value).strip()
    return None
