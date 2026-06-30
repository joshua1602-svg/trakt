"""apps.blob_trigger_app.path_parser — parse the raw blob path (fail closed).

Convention:
    /raw/{client_id}/{dataset}/{frequency}/{source_portfolio_id}/{reporting_period}/{filename}
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

VALID_DATASETS = ("funded", "pipeline", "forecast")
# "ad_hoc" (and the legacy "adhoc") support one-off acquired-portfolio onboarding
# packs, e.g. {client}/funded/ad_hoc/acquired_001/{reporting_date}/.
VALID_FREQUENCIES = ("monthly", "weekly", "daily", "adhoc", "ad_hoc")

# reporting_period: 2026-09-30 (date) or 2026-W39 (ISO week) or 2026-09 (month).
_PERIOD_RE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?|-W\d{2}|-Q[1-4])$")


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

    @property
    def source_key(self) -> str:
        """Stable registry key for this source/file-type combination."""
        return f"{self.client_id}/{self.source_portfolio_id}/{self.dataset}/{self.frequency}"


#: Default container the trigger watches (overridable via TRAKT_BLOB_CONTAINER).
DEFAULT_CONTAINER = "raw"


def _segments(blob_path: str, container: str) -> List[str]:
    """Return the 6 path segments after the container.

    Works whether or not the path is prefixed with the container name (the Azure
    blob trigger may surface ``blob.name`` with or without it): if the configured
    container appears as a segment, everything after it is returned; otherwise the
    path is assumed to already be the inner segments.
    """
    parts = [p for p in re.split(r"[\\/]+", blob_path.strip()) if p]
    if container in parts:
        return parts[parts.index(container) + 1:]
    return parts


def parse_blob_path(blob_path: str, container: str = DEFAULT_CONTAINER) -> ParsedPath:
    """Parse the raw blob path or raise :class:`PathParseError` (fail closed).

    ``container`` is the configured blob container (default ``raw``; set via the
    ``TRAKT_BLOB_CONTAINER`` app setting, e.g. ``raw-v2``)."""
    if not blob_path or not str(blob_path).strip():
        raise PathParseError("empty blob path")
    seg = _segments(blob_path, container)
    if len(seg) != 6:
        raise PathParseError(
            f"expected {container}/{{client_id}}/{{dataset}}/{{frequency}}/"
            "{source_portfolio_id}/{reporting_period}/{filename} "
            f"(6 segments after the container {container!r}), got {len(seg)}: {seg}")
    client_id, dataset, frequency, source_portfolio_id, reporting_period, filename = seg

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
    # Note: no extension requirement — the completion marker (e.g. _READY) is a
    # legitimate, extensionless filename at this position.

    return ParsedPath(
        client_id=client_id, dataset=dataset, frequency=frequency,
        source_portfolio_id=source_portfolio_id, reporting_period=reporting_period,
        filename=filename, blob_path=blob_path)
