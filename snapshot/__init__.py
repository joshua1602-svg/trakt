"""snapshot — local snapshot/history layer for recurring MI (Phase 2).

A storage-neutral snapshot model plus a local filesystem adapter. Persists and
retrieves timestamped portfolio snapshots so future MI states, trends,
migrations and forecast-funded analysis can operate on history.

Deliberate constraints (Phase 2 scope):
  * storage-neutral business logic depends on ``SnapshotStore`` only;
  * local filesystem adapter only (no Azure, no Event Grid / function app);
  * no orchestration, no MI state assembler, no M&A agent, no risk monitor,
    no forecast-funded runtime;
  * not wired into the MI Agent runtime yet;
  * no Streamlit/chart migration, no legacy ``analytics/`` imports, no LLM,
    no Annex 2 / regulatory changes.
"""

from __future__ import annotations

from .keys import (
    compute_source_file_id,
    make_pipeline_opportunity_id,
    make_snapshot_id,
    normalise_key_part,
    select_stable_loan_key,
)
from .model import (
    RESERVED_OPTIONAL_LOAN_COLUMNS,
    RESERVED_REQUIRED_LOAN_COLUMNS,
    SnapshotConflictError,
    SnapshotError,
    SnapshotHeader,
    SnapshotNotFoundError,
    SnapshotValidationError,
)
from .store import RegistrationResult, SnapshotStore

__all__ = [
    "SnapshotHeader",
    "SnapshotStore",
    "RegistrationResult",
    "SnapshotError",
    "SnapshotValidationError",
    "SnapshotConflictError",
    "SnapshotNotFoundError",
    "RESERVED_REQUIRED_LOAN_COLUMNS",
    "RESERVED_OPTIONAL_LOAN_COLUMNS",
    "compute_source_file_id",
    "make_snapshot_id",
    "make_pipeline_opportunity_id",
    "select_stable_loan_key",
    "normalise_key_part",
]
