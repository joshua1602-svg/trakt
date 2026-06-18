"""snapshot.adapters — storage adapters for the SnapshotStore interface.

Phase 2 ships the local filesystem adapter only. Azure Blob / S3 / GCS adapters
are deferred to later phases and will implement the same ``SnapshotStore``
interface behind the same business logic.
"""

from __future__ import annotations

from .local_fs import LocalFsSnapshotStore

__all__ = ["LocalFsSnapshotStore"]
