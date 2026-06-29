"""apps.blob_trigger_app.path_parser — parse the raw blob path (fail closed).

Convention:
    /raw/{client_id}/{dataset}/{frequency}/{source_portfolio_id}/{reporting_period}/{filename}
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

VALID_DATASETS = ("funded", "pipeline", "forecast")
VALID_FREQUENCIES = ("monthly", "weekly", "daily", "adhoc")

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


def _segments(blob_path: str) -> List[str]:
    # Tolerate leading container names / slashes; require the 'raw' anchor.
    parts = [p for p in re.split(r"[\\/]+", blob_path.strip()) if p]
    if "raw" not in parts:
        raise PathParseError(
            f"path does not contain a 'raw/' root: {blob_path!r}")
    i = parts.index("raw")
    return parts[i + 1:]


def parse_blob_path(blob_path: str) -> ParsedPath:
    """Parse the raw blob path or raise :class:`PathParseError` (fail closed)."""
    if not blob_path or not str(blob_path).strip():
        raise PathParseError("empty blob path")
    seg = _segments(blob_path)
    if len(seg) != 6:
        raise PathParseError(
            "expected raw/{client_id}/{dataset}/{frequency}/{source_portfolio_id}/"
            f"{{reporting_period}}/{{filename}} (6 segments after 'raw'), got {len(seg)}: {seg}")
    client_id, dataset, frequency, source_portfolio_id, reporting_period, filename = seg

    if dataset not in VALID_DATASETS:
        raise PathParseError(f"dataset must be one of {VALID_DATASETS}, got {dataset!r}")
    if frequency not in VALID_FREQUENCIES:
        raise PathParseError(f"frequency must be one of {VALID_FREQUENCIES}, got {frequency!r}")
    if not _PERIOD_RE.match(reporting_period):
        raise PathParseError(
            f"reporting_period {reporting_period!r} not a recognised period "
            "(YYYY-MM-DD / YYYY-Www / YYYY-MM / YYYY-Qn)")
    if "." not in filename:
        raise PathParseError(f"filename {filename!r} has no extension")

    return ParsedPath(
        client_id=client_id, dataset=dataset, frequency=frequency,
        source_portfolio_id=source_portfolio_id, reporting_period=reporting_period,
        filename=filename, blob_path=blob_path)
