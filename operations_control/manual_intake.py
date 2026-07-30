"""operations_control.manual_intake — the governed destination for a manually
created delivery.

The browser NEVER supplies a storage path. When an operator creates a delivery
by hand, the destination is derived here from the controlled fields already held
on the input batch (client, portfolio, book, dataset, frequency, reporting
period) and then re-parsed with the production path parser
(:mod:`apps.blob_trigger_app.path_parser`) — so a path this module produces is,
by construction, a path the automated intake route would accept. Anything that
does not round-trip is refused (fail closed).

Filenames are sanitised to a leaf name: no directories, no traversal, no control
characters, and never the legacy ``_READY.json`` sentinel (which the Operations
Control Centre replaced with its own readiness assessment and internal run
manifest — see :mod:`operations_control.intake`).
"""

from __future__ import annotations

import os
import re
from typing import Optional

from apps.blob_trigger_app.path_parser import (
    PathParseError,
    VALID_DATASETS,
    VALID_FREQUENCIES,
    parse_blob_path,
)

#: Container manually created deliveries are written to. Same setting the
#: Function App watches, so a manual delivery lands where an automated one does.
RAW_CONTAINER_ENV = "TRAKT_RAW_CONTAINER"
DEFAULT_RAW_CONTAINER = "raw-v2"

#: File types the intake classifier can read (mirrors ``intake.DATA_EXTS``).
ALLOWED_EXTENSIONS = (".csv", ".xlsx", ".xls", ".xlsm")

#: The sentinel the OCC replaced. Never accepted from a manual upload.
LEGACY_SENTINEL = "_READY.json"

#: One path segment: a plain identifier. No separators, no traversal, no spaces.
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

#: A safe leaf filename.
_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.()-]{0,119}$")

_MAX_FILENAME = 120


class ManualIntakeError(ValueError):
    """A manual delivery could not be placed in a governed location."""


def raw_container() -> str:
    return os.environ.get(RAW_CONTAINER_ENV, DEFAULT_RAW_CONTAINER)


def sanitise_filename(name: str) -> str:
    """Reduce an uploaded filename to a safe leaf name, or refuse it.

    Only the leaf is kept — a browser that sends ``../../etc/passwd`` or
    ``C:\\Windows\\x.csv`` contributes its last segment and nothing else.
    """
    raw = (name or "").strip()
    # Keep the last segment only, whichever separator the client used.
    leaf = re.split(r"[\\/]+", raw)[-1].strip()
    if not leaf or leaf in (".", ".."):
        raise ManualIntakeError("A file was sent without a usable name.")
    if leaf == LEGACY_SENTINEL:
        raise ManualIntakeError(
            "Trakt decides when a delivery is complete. Send the data files "
            "only.")
    if any(ord(c) < 32 or ord(c) == 127 for c in leaf):
        raise ManualIntakeError(f"'{raw}' is not a name Trakt can accept.")
    if len(leaf) > _MAX_FILENAME:
        raise ManualIntakeError(f"'{leaf}' has too long a name.")
    if not _FILENAME_RE.match(leaf):
        raise ManualIntakeError(f"'{leaf}' is not a name Trakt can accept.")
    if os.path.splitext(leaf)[1].lower() not in ALLOWED_EXTENSIONS:
        raise ManualIntakeError(
            f"'{leaf}' is not a kind of file Trakt can read. Send a "
            "spreadsheet or a comma-separated file.")
    return leaf


def _segment(value: str, what: str) -> str:
    v = (value or "").strip()
    if not _SEGMENT_RE.match(v):
        raise ManualIntakeError(f"The {what} is not something Trakt can file "
                                "a delivery under.")
    return v


def derive_book_type(source_portfolio_id: str) -> str:
    """The book-type folder for a portfolio.

    Mirrors the production parser's own derivation so the two agree: a portfolio
    whose identifier starts ``acquired`` is an acquired book, everything else is
    filed as direct. The parser refuses any pair that disagrees, so a mismatch
    can never reach storage.
    """
    return "acquired" if (source_portfolio_id or "").lower().startswith(
        "acquired") else "direct"


def derive_raw_prefix(*, client_id: str, portfolio_id: str,
                      reporting_period: str, dataset: str = "funded",
                      frequency: str = "monthly",
                      container: Optional[str] = None) -> str:
    """The governed destination prefix for one manually created delivery.

    Returns ``{container}/{client}/{book}/{dataset}/{frequency}/{portfolio}/
    {period}``. Raises :class:`ManualIntakeError` unless the production path
    parser accepts the result — the parser, not this function, is the authority
    on what a valid delivery location looks like.
    """
    cont = container or raw_container()
    client = _segment(client_id, "client")
    portfolio = _segment(portfolio_id, "portfolio")
    period = _segment(reporting_period, "reporting period")
    ds = (dataset or "funded").strip()
    freq = (frequency or "monthly").strip()
    if ds not in VALID_DATASETS:
        raise ManualIntakeError("That is not a book Trakt can file a delivery "
                                "under.")
    if freq not in VALID_FREQUENCIES:
        raise ManualIntakeError("That is not a delivery frequency Trakt "
                                "recognises.")
    book = derive_book_type(portfolio)
    prefix = f"{cont}/{client}/{book}/{ds}/{freq}/{portfolio}/{period}"
    # Fail closed: prove the automated route would accept this location before
    # a single byte is written to it.
    try:
        parse_blob_path(f"{prefix}/probe.csv", cont)
    except PathParseError as exc:
        raise ManualIntakeError(
            "Trakt could not work out where this delivery belongs. Check the "
            "client, portfolio and reporting period.") from exc
    return prefix


def derive_blob_uri(prefix: str, filename: str) -> str:
    """The storage URI for one file inside a derived prefix."""
    return f"blob://{prefix}/{sanitise_filename(filename)}"
