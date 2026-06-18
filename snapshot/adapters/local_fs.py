"""snapshot.adapters.local_fs — local filesystem SnapshotStore adapter.

Phase 2 snapshot/history layer. The default, dependency-light adapter used for
development and tests. Append-only, idempotent, never silently overwriting.
Loan rows are stored as CSV (no new dependency); a JSON manifest indexes all
snapshots and a per-client/route ``latest`` pointer is maintained.

Layout under ``root``::

    root/
      manifest.json
      latest/<client>/<route>.json            # latest pointer
      <client>/<route>/<snapshot_id>/header.json
      <client>/<route>/<snapshot_id>/loans.csv

No Azure, no Streamlit, no legacy ``analytics/`` imports.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..model import (
    CUT_OFF_DATE_DEFAULTED_TO_REPORTING_DATE,
    DUPLICATE_SNAPSHOT_CONFLICTING_SOURCE,
    DUPLICATE_SNAPSHOT_SAME_SOURCE,
    INFO,
    WARNING,
    SnapshotConflictError,
    SnapshotHeader,
    SnapshotNotFoundError,
    make_issue,
    normalise_loan_frame,
    parse_date,
    validate_header,
)
from ..store import RegistrationResult, SnapshotStore

DEFAULT_ROOT = ".trakt_snapshots"
_MANIFEST = "manifest.json"


class LocalFsSnapshotStore(SnapshotStore):
    """Filesystem-backed snapshot store."""

    def __init__(self, root: Any = DEFAULT_ROOT,
                 default_cut_off_to_reporting: bool = False,
                 allow_missing_reporting_date: bool = False) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.default_cut_off_to_reporting = default_cut_off_to_reporting
        self.allow_missing_reporting_date = allow_missing_reporting_date
        self._manifest_path = self.root / _MANIFEST
        if not self._manifest_path.exists():
            self._write_manifest({})

    # -- manifest helpers -------------------------------------------------- #

    def _read_manifest(self) -> Dict[str, dict]:
        try:
            return json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _write_manifest(self, manifest: Dict[str, dict]) -> None:
        self._manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    def _snapshot_dir(self, header: SnapshotHeader) -> Path:
        from .. import keys as _keys
        return (self.root
                / _keys.normalise_key_part(header.client_id)
                / _keys.normalise_key_part(header.route)
                / header.snapshot_id)

    def _latest_pointer_path(self, client_id: str, route: str) -> Path:
        from .. import keys as _keys
        return (self.root / "latest"
                / _keys.normalise_key_part(client_id)
                / f"{_keys.normalise_key_part(route)}.json")

    @staticmethod
    def _frame_content_hash(frame: pd.DataFrame) -> str:
        # Stable hash of the normalised CSV bytes (column order included).
        csv_bytes = frame.to_csv(index=False).encode("utf-8")
        return "sha256:" + hashlib.sha256(csv_bytes).hexdigest()

    # -- registration ------------------------------------------------------ #

    def register_snapshot(self, header: SnapshotHeader,
                          frame: pd.DataFrame) -> RegistrationResult:
        issues: List[dict] = list(validate_header(
            header, allow_missing_reporting_date=self.allow_missing_reporting_date))

        # Cut-off date defaulting (only when explicitly configured).
        if not header.cut_off_date and header.reporting_date:
            if self.default_cut_off_to_reporting:
                header.cut_off_date = header.reporting_date
                issues.append(make_issue(
                    CUT_OFF_DATE_DEFAULTED_TO_REPORTING_DATE, INFO,
                    "cut_off_date defaulted to reporting_date "
                    "(default_cut_off_to_reporting=True)", "cut_off_date"))

        snapshot_id = header.ensure_id()
        norm_frame, frame_issues = normalise_loan_frame(frame, header)
        issues.extend(frame_issues)
        content_hash = self._frame_content_hash(norm_frame)
        header.content_hash = content_hash

        manifest = self._read_manifest()

        # 1. Exact identity (same snapshot_id) -> idempotent or content clash.
        existing = manifest.get(snapshot_id)
        if existing is not None:
            if (existing.get("content_hash") == content_hash
                    and existing.get("source_file_id") == header.source_file_id):
                issues.append(make_issue(
                    DUPLICATE_SNAPSHOT_SAME_SOURCE, INFO,
                    "identical snapshot already registered; returning existing id",
                    snapshot_id=snapshot_id))
                return RegistrationResult(
                    snapshot_id=snapshot_id, created=False, idempotent=True,
                    header=SnapshotHeader.from_dict(existing), issues=issues)
            issue = make_issue(
                DUPLICATE_SNAPSHOT_CONFLICTING_SOURCE, WARNING,
                "a snapshot with this id exists but content differs; refusing "
                "to overwrite", snapshot_id=snapshot_id)
            issues.append(issue)
            raise SnapshotConflictError(
                f"conflicting content for snapshot {snapshot_id}", issue=issue)

        # 2. Same logical slot, different source -> conflict (no overwrite).
        slot = header.logical_slot()
        for sid, rec in manifest.items():
            if (rec.get("logical_slot") == slot
                    and rec.get("source_file_id") != header.source_file_id):
                issue = make_issue(
                    DUPLICATE_SNAPSHOT_CONFLICTING_SOURCE, WARNING,
                    "a snapshot already exists for this client/route/"
                    "reporting_date/cadence with a different source_file_id",
                    snapshot_id=sid)
                issues.append(issue)
                raise SnapshotConflictError(
                    f"conflicting source for logical slot {slot}", issue=issue)

        # 3. New snapshot -> write append-only.
        snap_dir = self._snapshot_dir(header)
        snap_dir.mkdir(parents=True, exist_ok=True)
        norm_frame.to_csv(snap_dir / "loans.csv", index=False)
        (snap_dir / "header.json").write_text(
            json.dumps(header.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8")

        record = header.to_dict()
        record["logical_slot"] = slot
        manifest[snapshot_id] = record
        self._write_manifest(manifest)
        self._update_latest_pointer(header)

        return RegistrationResult(
            snapshot_id=snapshot_id, created=True, idempotent=False,
            header=header, issues=issues)

    def _update_latest_pointer(self, header: SnapshotHeader) -> None:
        pointer_path = self._latest_pointer_path(header.client_id, header.route)
        pointer_path.parent.mkdir(parents=True, exist_ok=True)
        current_latest = None
        if pointer_path.exists():
            try:
                current_latest = json.loads(
                    pointer_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                current_latest = None
        new_rd = parse_date(header.reporting_date)
        if current_latest:
            cur_rd = parse_date(current_latest.get("reporting_date"))
            if cur_rd and new_rd and new_rd < cur_rd:
                return  # existing pointer is newer; keep it
        pointer_path.write_text(
            json.dumps({"snapshot_id": header.snapshot_id,
                        "reporting_date": header.reporting_date}, indent=2),
            encoding="utf-8")

    # -- reads ------------------------------------------------------------- #

    def list_snapshots(self, client_id: str, route: Optional[str] = None,
                       cadence: Optional[str] = None,
                       since: Any = None, until: Any = None
                       ) -> List[SnapshotHeader]:
        since_d = parse_date(since) if since else None
        until_d = parse_date(until) if until else None
        out: List[SnapshotHeader] = []
        for rec in self._read_manifest().values():
            if rec.get("client_id") != client_id:
                continue
            if route is not None and rec.get("route") != route:
                continue
            if cadence is not None and rec.get("cadence") != cadence:
                continue
            rd = parse_date(rec.get("reporting_date"))
            if since_d and (rd is None or rd < since_d):
                continue
            if until_d and (rd is None or rd > until_d):
                continue
            out.append(SnapshotHeader.from_dict(rec))
        out.sort(key=lambda h: (parse_date(h.reporting_date)
                                or parse_date("0001-01-01"),
                                h.upload_timestamp or ""))
        return out

    def get_snapshot(self, snapshot_id: str) -> SnapshotHeader:
        rec = self._read_manifest().get(snapshot_id)
        if rec is None:
            raise SnapshotNotFoundError(f"unknown snapshot_id {snapshot_id!r}")
        return SnapshotHeader.from_dict(rec)

    def load_loans(self, snapshot_id: str) -> pd.DataFrame:
        header = self.get_snapshot(snapshot_id)
        path = self._snapshot_dir(header) / "loans.csv"
        if not path.exists():
            raise SnapshotNotFoundError(
                f"loan rows missing for snapshot {snapshot_id!r}")
        return pd.read_csv(path)

    def get_latest_pointer(self, client_id: str,
                           route: str) -> Optional[Dict[str, Any]]:
        """Return the persisted latest-pointer record (or ``None``)."""
        path = self._latest_pointer_path(client_id, route)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
