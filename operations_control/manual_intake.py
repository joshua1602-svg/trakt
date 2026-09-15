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
    VALID_BOOK_TYPES,
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


def derive_book_type(source_portfolio_id: str,
                     declared_type: Optional[str] = None) -> str:
    """The book-type folder for a portfolio: ``direct`` or ``acquired``.

    THE DECLARED ANSWER OUTRANKS THE NAME. Onboarding asks, as a required
    question with its own vocabulary, whether a book was originated by the
    client or acquired from a third party. ``declared_type`` is that answer, and
    where it is supplied it decides — the identifier is then only cross-checked
    against it, never consulted instead of it.

    WHAT THIS REPLACED. The rule used to be, in full: a portfolio whose id
    starts ``acquired`` is acquired, EVERYTHING ELSE IS DIRECT. So
    ``purchased_001`` and ``alp_acquired`` both filed under ``direct/`` — the
    first because it does not start with the word, the second because the word
    is at the end — while the onboarding case said acquired. Nothing objected.
    The delivery simply landed in the wrong book, and every reading of it
    downstream inherited that.

    A guess is the wrong shape of answer here. The two sibling derivations
    agree: ``path_parser._derive_book_type`` and
    ``engine.provenance.derive_portfolio_type`` both return ``None`` for an id
    that gives no clue, the latter saying so outright — "so the caller can fail
    closed asking for an explicit type". This was the only one of the three that
    guessed, and the only one that writes.

    So an unrecognisable id with no declared type is now REFUSED rather than
    filed as direct. That is not a new burden on the ordinary case: a book named
    ``direct_001`` or ``acquired_001`` is unaffected, and one named anything
    else is exactly the case that was being answered wrongly before.
    """
    from engine.provenance import derive_portfolio_type

    from_name = derive_portfolio_type(source_portfolio_id)
    declared = (declared_type or "").strip().lower()

    if declared:
        if declared not in VALID_BOOK_TYPES:
            raise ManualIntakeError(
                "That is not a book type Trakt recognises. A portfolio is "
                "either originated by the client or acquired from a third "
                "party.")
        # A conforming name that contradicts the declaration is not a tie to
        # break — it is two governed answers disagreeing, and picking either
        # silently is how the wrong one ends up in storage.
        if from_name and from_name != declared:
            raise ManualIntakeError(
                f"'{source_portfolio_id}' is named as a {from_name} book but "
                f"is declared {declared}. Correct one of them before sending "
                "a delivery.")
        return declared

    if from_name is None:
        raise ManualIntakeError(
            f"Trakt cannot tell whether '{source_portfolio_id}' is a book you "
            "originated or one you acquired. Name the portfolio so it begins "
            "'direct' or 'acquired', or supply the book type with the "
            "delivery.")
    return from_name


def derive_raw_prefix(*, client_id: str, portfolio_id: str,
                      reporting_period: str, dataset: str = "funded",
                      frequency: str = "monthly",
                      portfolio_type: Optional[str] = None,
                      container: Optional[str] = None) -> str:
    """The governed destination prefix for one manually created delivery.

    Returns ``{container}/{client}/{book}/{dataset}/{frequency}/{portfolio}/
    {period}``. Raises :class:`ManualIntakeError` unless the production path
    parser accepts the result — the parser, not this function, is the authority
    on what a valid delivery location looks like.

    ``portfolio_type`` is the onboarding case's own declaration of whether the
    book was originated or acquired. Supplied, it decides the ``{book}``
    segment; omitted, the identifier has to carry it (see
    :func:`derive_book_type`, which refuses rather than guesses).
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
    book = derive_book_type(portfolio, portfolio_type)
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
