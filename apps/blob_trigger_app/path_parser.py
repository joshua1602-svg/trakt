"""apps.blob_trigger_app.path_parser — parse the raw blob path (fail closed).

Preferred (production) convention — 7 segments after the container:
    {container}/{client_id}/{source_book_type}/{dataset}/{frequency}/{source_portfolio_id}/{period}/{filename}
    e.g. raw-v2/ERE/direct/funded/monthly/direct_001/2025-11-30/LoanExtract.csv
         raw-v2/ERE/acquired/funded/ad_hoc/acquired_001/2025-11-30/LoanExtract.csv

Deprecated compatibility convention — 6 segments (no source_book_type):
    {container}/{client_id}/{dataset}/{frequency}/{source_portfolio_id}/{period}/{filename}
    (book type is derived from the source_portfolio_id prefix).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

VALID_BOOK_TYPES = ("direct", "acquired")
# forecast retained for back-compat; the production structure uses funded/pipeline.
VALID_DATASETS = ("funded", "pipeline", "forecast")
VALID_FREQUENCIES = ("monthly", "weekly", "daily", "adhoc", "ad_hoc")

#: Spellings that mean the same frequency, mapped to the one that is WRITTEN.
#:
#: ``adhoc`` and ``ad_hoc`` were both accepted and are not the same key. The
#: frequency is a path segment, part of the pack key, and part of a snapshot's
#: ``logical_slot`` — and ``snapshot.keys.normalise_key_part`` keeps the
#: underscore, so one stream spelled two ways became two folder trees and two
#: key spaces, with nothing downstream reconciling them. Only ``adhoc`` is in
#: ``snapshot.model.VALID_CADENCES``, so the other spelling also raised a
#: non-fatal ``invalid_cadence`` warning and then settled into the data.
#:
#: Same treatment as ``_normalise_period`` directly below: tolerant about what
#: arrives, strict about what is written.
_FREQUENCY_ALIASES = {"ad_hoc": "adhoc", "ad-hoc": "adhoc"}


def canonical_frequency(value: Optional[str]) -> str:
    """The one spelling of ``value`` that may be written, or "" if blank.

    Blank stays blank: canonicalising must not invent a frequency nobody chose,
    because the caller's own default is the right answer there. Anything that is
    neither blank nor a known frequency raises, rather than being guessed at and
    written into a key.
    """
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"\s+", "_", text)
    text = _FREQUENCY_ALIASES.get(text, text)
    if text not in VALID_FREQUENCIES:
        raise PathParseError(
            f"frequency {value!r} is not one Trakt recognises "
            f"({', '.join(sorted(set(VALID_FREQUENCIES)))})")
    return _FREQUENCY_ALIASES.get(text, text)

# reporting_period: 2026-09-30 (date) or 2026-W39 (ISO week) or 2026-09 (month).
_PERIOD_RE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?|-W\d{2}|-Q[1-4])$")


def _normalise_period(period: str) -> str:
    """Canonicalise a period-folder segment to hyphen form.

    Real uploads use both separators — funded folders arrive as ``2025-10-31`` but
    weekly pipeline folders arrive as ``2025_09_08`` (underscores). Both mean the
    same reporting period, so an underscore variant that resolves to a valid period
    is normalised to hyphens (``2025_09_08`` → ``2025-09-08``, ``2025_W36`` →
    ``2025-W36``). Input blobs are always read from their real path; only the
    parsed ``reporting_period`` (used for pack_key + output layout) is canonicalised,
    so nothing downstream depends on the raw separator. Unrecognised values are
    returned unchanged and fail validation as before (fail closed)."""
    if _PERIOD_RE.match(period):
        return period
    candidate = period.replace("_", "-")
    return candidate if _PERIOD_RE.match(candidate) else period


def canonical_period(value: Optional[str]) -> str:
    """The one spelling of a reporting period that may be written, or "".

    The public counterpart to ``_normalise_period``, and the same contract as
    :func:`canonical_frequency`: blank stays blank for the caller to default,
    a recognised period is returned in hyphen form, and anything else raises
    rather than being written into a path and a pack key.

    ``_normalise_period`` returns an unrecognised value unchanged, which is
    right where a path is being PARSED — the caller validates the whole path a
    moment later and fails closed. It is wrong where a period is being
    ACCEPTED from a person, which is what this is for: there the first and only
    chance to reject "April 2026" is at the door.
    """
    text = str(value or "").strip().upper()
    if not text:
        return ""
    canonical = _normalise_period(text)
    if not _PERIOD_RE.match(canonical):
        raise PathParseError(
            f"reporting period {value!r} is not one Trakt recognises "
            "(2026-04 for a month, 2026-04-30 for a date, 2026-W14 for an "
            "ISO week, 2026-Q2 for a quarter)")
    return canonical


class PathParseError(ValueError):
    """Raised when a blob path does not match the convention (fail closed)."""


@dataclass(frozen=True)
class ParsedPath:
    client_id: str
    dataset: str
    frequency: str
    source_portfolio_id: str
    reporting_period: str
    filename: str
    blob_path: str
    source_book_type: Optional[str] = None
    is_legacy_path: bool = False

    @property
    def source_key(self) -> str:
        """Stable registry key for this source/file-type combination."""
        return f"{self.client_id}/{self.source_portfolio_id}/{self.dataset}/{self.frequency}"


#: Default container the trigger watches (overridable via TRAKT_BLOB_CONTAINER).
DEFAULT_CONTAINER = "raw"


def _derive_book_type(source_portfolio_id: str) -> Optional[str]:
    pid = (source_portfolio_id or "").lower()
    if pid.startswith("direct"):
        return "direct"
    if pid.startswith("acquired"):
        return "acquired"
    return None


def _segments(blob_path: str, container: str) -> List[str]:
    """Return the path segments after the container.

    Works whether or not the path is prefixed with the container name: if the
    configured container appears as a segment, everything after it is returned;
    otherwise the path is assumed to already be the inner segments.
    """
    parts = [p for p in re.split(r"[\\/]+", blob_path.strip()) if p]
    if container in parts:
        return parts[parts.index(container) + 1:]
    return parts


def _validate_common(dataset: str, frequency: str, reporting_period: str, filename: str) -> None:
    if dataset not in VALID_DATASETS:
        raise PathParseError(f"dataset must be one of {VALID_DATASETS}, got {dataset!r}")
    if frequency not in VALID_FREQUENCIES:
        raise PathParseError(f"frequency must be one of {VALID_FREQUENCIES}, got {frequency!r}")
    if not _PERIOD_RE.match(reporting_period):
        raise PathParseError(
            f"reporting_period {reporting_period!r} not a recognised period "
            "(YYYY-MM-DD / YYYY-Www / YYYY-MM / YYYY-Qn)")
    if not filename:
        raise PathParseError("empty filename")


def parse_blob_path(blob_path: str, container: str = DEFAULT_CONTAINER) -> ParsedPath:
    """Parse the raw blob path or raise :class:`PathParseError` (fail closed).

    Accepts the preferred 7-segment structure (with ``source_book_type``) and the
    deprecated 6-segment compatibility structure. ``container`` is the configured
    blob container (default ``raw``; set via ``TRAKT_BLOB_CONTAINER``)."""
    if not blob_path or not str(blob_path).strip():
        raise PathParseError("empty blob path")
    seg = _segments(blob_path, container)

    if len(seg) == 7:
        (client_id, source_book_type, dataset, frequency,
         source_portfolio_id, reporting_period, filename) = seg
        reporting_period = _normalise_period(reporting_period)
        if source_book_type not in VALID_BOOK_TYPES:
            raise PathParseError(
                f"source_book_type must be one of {VALID_BOOK_TYPES}, got "
                f"{source_book_type!r} (path: {client_id}/{source_book_type}/{dataset}/…)")
        _validate_common(dataset, frequency, reporting_period, filename)
        derived = _derive_book_type(source_portfolio_id)
        if derived and derived != source_book_type:
            raise PathParseError(
                f"source_book_type {source_book_type!r} is inconsistent with "
                f"source_portfolio_id {source_portfolio_id!r} (looks {derived!r})")
        return ParsedPath(
            client_id=client_id, dataset=dataset, frequency=frequency,
            source_portfolio_id=source_portfolio_id, reporting_period=reporting_period,
            filename=filename, blob_path=blob_path, source_book_type=source_book_type,
            is_legacy_path=False)

    if len(seg) == 6:
        (client_id, dataset, frequency,
         source_portfolio_id, reporting_period, filename) = seg
        reporting_period = _normalise_period(reporting_period)
        _validate_common(dataset, frequency, reporting_period, filename)
        return ParsedPath(
            client_id=client_id, dataset=dataset, frequency=frequency,
            source_portfolio_id=source_portfolio_id, reporting_period=reporting_period,
            filename=filename, blob_path=blob_path,
            source_book_type=_derive_book_type(source_portfolio_id), is_legacy_path=True)

    raise PathParseError(
        f"expected 7 segments {container}/{{client_id}}/{{source_book_type}}/{{dataset}}/"
        "{frequency}/{source_portfolio_id}/{period}/{filename} (preferred) or 6 "
        f"(deprecated, no source_book_type) after container {container!r}; got {len(seg)}: {seg}")
