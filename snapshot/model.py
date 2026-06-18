"""snapshot.model — snapshot header, reserved loan columns, issues, validation.

Phase 2 snapshot/history layer. Storage-neutral data model and validation. The
single most important rule enforced here is **date separation**: the operational
``upload_timestamp`` must never silently become the ``reporting_date``, and the
event dates (origination / funding / acquisition / SPV-transfer) are kept
distinct from the snapshot-header dates.

No filesystem, no Azure, no Streamlit, no legacy ``analytics/`` imports.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from . import keys as _keys

# --------------------------------------------------------------------------- #
# Controlled vocabularies (mirror config/routes/*.yaml and the build plan)
# --------------------------------------------------------------------------- #

VALID_ROUTES = {"mi", "mna", "regulatory_annex2", "regulatory_and_mi"}
VALID_CADENCES = {"weekly", "monthly", "adhoc", "backfill"}

SCHEMA_VERSION = "1.0"

# --------------------------------------------------------------------------- #
# Reserved loan-level columns (Phase 0B virtual fields realised per-row)
# --------------------------------------------------------------------------- #

# Always populated on a stored loan frame.
RESERVED_REQUIRED_LOAN_COLUMNS = (
    "snapshot_id",
    "client_id",
    "loan_id",
    "reporting_date",
    "cut_off_date",
    "upload_timestamp",
)

# Populated when the source provides them; absence ⇒ structured issue, not crash.
RESERVED_OPTIONAL_LOAN_COLUMNS = (
    "source_record_id",
    "stable_entity_id",
    "opportunity_id",
    "portfolio_id",
    "spv_id",
    "acquired_portfolio_id",
    "origination_date",
    "funding_date",
    "acquisition_date",
    "spv_transfer_date",
    "pipeline_stage",
    "funded_status",
)

# Segmentation fields whose absence is explicitly surfaced as an issue.
SEGMENTATION_FIELDS = ("portfolio_id", "spv_id", "acquired_portfolio_id")

# Event/header dates kept ISO and never conflated with each other.
HEADER_DATE_FIELDS = ("reporting_date", "cut_off_date")
HEADER_DATETIME_FIELDS = ("upload_timestamp", "created_at")

# --------------------------------------------------------------------------- #
# Issue codes & severities
# --------------------------------------------------------------------------- #

MISSING_REQUIRED_HEADER_FIELD = "missing_required_header_field"
MISSING_REPORTING_DATE = "missing_reporting_date"
MISSING_CLIENT_ID = "missing_client_id"
MISSING_SOURCE_FILE_ID = "missing_source_file_id"
DUPLICATE_SNAPSHOT_SAME_SOURCE = "duplicate_snapshot_same_source"
DUPLICATE_SNAPSHOT_CONFLICTING_SOURCE = "duplicate_snapshot_conflicting_source"
MISSING_STABLE_LOAN_KEY = "missing_stable_loan_key"
MISSING_OPTIONAL_SEGMENTATION_FIELD = "missing_optional_segmentation_field"
CUT_OFF_DATE_DEFAULTED_TO_REPORTING_DATE = "cut_off_date_defaulted_to_reporting_date"
INVALID_DATE = "invalid_date"
INVALID_ROUTE = "invalid_route"
INVALID_CADENCE = "invalid_cadence"

ERROR = "error"
WARNING = "warning"
INFO = "info"


def make_issue(code: str, severity: str, message: str,
               field: Optional[str] = None, **extra: Any) -> Dict[str, Any]:
    issue = {"code": code, "severity": severity, "message": message,
             "field": field}
    issue.update(extra)
    return issue


# --------------------------------------------------------------------------- #
# Exceptions (raised only for *required* failures)
# --------------------------------------------------------------------------- #


class SnapshotError(Exception):
    """Base class for snapshot-layer errors."""


class SnapshotValidationError(SnapshotError):
    """Raised when a required header field is missing/invalid."""

    def __init__(self, message: str, issues: Optional[List[dict]] = None):
        super().__init__(message)
        self.issues = issues or []


class SnapshotConflictError(SnapshotError):
    """Raised when registration would overwrite an existing snapshot with
    different content (same logical slot, different source)."""

    def __init__(self, message: str, issue: Optional[dict] = None):
        super().__init__(message)
        self.issue = issue


class SnapshotNotFoundError(SnapshotError):
    """Raised when a resolve/get finds no matching snapshot."""


# --------------------------------------------------------------------------- #
# Date helpers — ISO in, ISO out, never conflated
# --------------------------------------------------------------------------- #


def _to_iso_date(value: Any) -> Optional[str]:
    """Coerce to an ISO ``YYYY-MM-DD`` string, or ``None`` if unparseable."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.date().isoformat()


def _to_iso_datetime(value: Any) -> Optional[str]:
    """Coerce to an ISO 8601 datetime string, or ``None`` if unparseable."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day).isoformat()
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.isoformat()


def parse_date(value: Any) -> Optional[date]:
    """Parse an ISO date string/date to a ``date`` for comparison."""
    iso = _to_iso_date(value)
    if iso is None:
        return None
    return date.fromisoformat(iso)


# --------------------------------------------------------------------------- #
# SnapshotHeader
# --------------------------------------------------------------------------- #


@dataclass
class SnapshotHeader:
    """One row per upload/cut. Dates are stored ISO and kept separate."""

    client_id: str
    route: str
    reporting_date: Optional[str]
    source_file_id: str
    cadence: Optional[str] = None
    cut_off_date: Optional[str] = None
    upload_timestamp: Optional[str] = None
    source_file_name: Optional[str] = None
    snapshot_id: Optional[str] = None
    row_count: Optional[int] = None
    schema_version: str = SCHEMA_VERSION
    created_at: Optional[str] = None
    content_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalise all dates to ISO on construction (never conflated).
        self.reporting_date = _to_iso_date(self.reporting_date)
        self.cut_off_date = _to_iso_date(self.cut_off_date)
        self.upload_timestamp = _to_iso_datetime(self.upload_timestamp)
        self.created_at = (_to_iso_datetime(self.created_at)
                           or datetime.now(timezone.utc).isoformat())

    # -- serialisation ----------------------------------------------------- #

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SnapshotHeader":
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in fields})

    def ensure_id(self) -> str:
        if not self.snapshot_id:
            self.snapshot_id = _keys.make_snapshot_id(
                self.client_id, self.route, self.reporting_date,
                self.source_file_id)
        return self.snapshot_id

    def logical_slot(self) -> str:
        """Identity of the *logical* reporting slot (independent of source).

        Two registrations sharing this slot but with different ``source_file_id``
        are a conflict (different content for the same period)."""
        return "|".join([
            _keys.normalise_key_part(self.client_id),
            _keys.normalise_key_part(self.route),
            _keys.normalise_key_part(self.reporting_date),
            _keys.normalise_key_part(self.cadence),
        ])


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def validate_header(header: SnapshotHeader,
                    allow_missing_reporting_date: bool = False
                    ) -> List[Dict[str, Any]]:
    """Validate a header. Raises ``SnapshotValidationError`` for *required*
    failures; returns a list of non-fatal issues otherwise."""
    required_issues: List[Dict[str, Any]] = []
    issues: List[Dict[str, Any]] = []

    if not header.client_id:
        required_issues.append(make_issue(
            MISSING_CLIENT_ID, ERROR, "client_id is required", "client_id"))
    if not header.source_file_id:
        required_issues.append(make_issue(
            MISSING_SOURCE_FILE_ID, ERROR, "source_file_id is required",
            "source_file_id"))
    if not header.reporting_date:
        if allow_missing_reporting_date:
            issues.append(make_issue(
                MISSING_REPORTING_DATE, WARNING,
                "reporting_date missing (allowed for test/backfill)",
                "reporting_date"))
        else:
            required_issues.append(make_issue(
                MISSING_REPORTING_DATE, ERROR,
                "reporting_date is required and must not default to "
                "upload_timestamp", "reporting_date"))

    if required_issues:
        raise SnapshotValidationError(
            "snapshot header failed required-field validation",
            issues=required_issues)

    # Non-fatal vocabulary / date-validity checks.
    if header.route not in VALID_ROUTES:
        issues.append(make_issue(
            INVALID_ROUTE, WARNING,
            f"route {header.route!r} not in {sorted(VALID_ROUTES)}", "route"))
    if header.cadence is not None and header.cadence not in VALID_CADENCES:
        issues.append(make_issue(
            INVALID_CADENCE, WARNING,
            f"cadence {header.cadence!r} not in {sorted(VALID_CADENCES)}",
            "cadence"))
    return issues


def normalise_loan_frame(
    frame: pd.DataFrame,
    header: SnapshotHeader,
) -> tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Return ``(frame_out, issues)`` with reserved columns realised.

    * Stamps snapshot-level columns (snapshot_id/client_id and the three header
      dates) onto every row from the header — never inferring reporting_date
      from upload_timestamp.
    * Derives ``stable_entity_id`` / ``opportunity_id`` / ``loan_id`` per row,
      keeping funded ids and pipeline opportunity ids in distinct namespaces.
    * Records issues for rows with no stable key and for absent segmentation
      fields — without crashing.
    """
    out = frame.copy()
    issues: List[Dict[str, Any]] = []
    snapshot_id = header.ensure_id()

    out["snapshot_id"] = snapshot_id
    out["client_id"] = header.client_id
    out["reporting_date"] = header.reporting_date
    out["cut_off_date"] = header.cut_off_date
    out["upload_timestamp"] = header.upload_timestamp

    # Entity keys.
    stable_ids: List[Optional[str]] = []
    opp_ids: List[Optional[str]] = []
    loan_ids: List[Optional[str]] = []
    n_missing_key = 0
    for _, row in out.iterrows():
        stable = _keys.select_stable_loan_key(row)
        if stable:
            stable_ids.append(stable)
            opp_ids.append(None)
            loan_ids.append(stable)
            continue
        opp = _keys.make_pipeline_opportunity_id(row)
        stable_ids.append(None)
        opp_ids.append(opp)
        if opp:
            loan_ids.append("OPP_" + opp)  # distinct namespace from funded ids
        else:
            loan_ids.append(None)
            n_missing_key += 1

    # Preserve an existing explicit loan_id where present; otherwise use derived.
    if "loan_id" in frame.columns:
        existing = frame["loan_id"].tolist()
        out["loan_id"] = [e if not _keys._is_empty(e) else d
                          for e, d in zip(existing, loan_ids)]
    else:
        out["loan_id"] = loan_ids
    out["stable_entity_id"] = stable_ids
    out["opportunity_id"] = opp_ids

    if n_missing_key:
        issues.append(make_issue(
            MISSING_STABLE_LOAN_KEY, WARNING,
            f"{n_missing_key} row(s) have neither a stable loan key nor enough "
            f"fields for a pipeline opportunity id", "loan_id",
            count=n_missing_key))

    for seg in SEGMENTATION_FIELDS:
        if seg not in frame.columns:
            issues.append(make_issue(
                MISSING_OPTIONAL_SEGMENTATION_FIELD, INFO,
                f"optional segmentation field {seg!r} not present", seg))

    header.row_count = int(len(out))
    return out, issues
