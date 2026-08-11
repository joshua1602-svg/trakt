"""regulatory_watch.snapshots — immutable, content-addressed source evidence.

Retrieval, persistence and comparison are three separate concerns; this module
is only the middle one. It takes bytes that already exist locally and preserves
them (plus a sha256 and the manifest metadata) under an append-only layout::

    <root>/<regime>/<artefact_id>/<sha256[:16]>/
        content<ext>          the immutable bytes
        snapshot.json         the SourceSnapshot sidecar

Re-snapshotting identical bytes is idempotent and never rewrites the stored
copy. Different bytes land in a different directory, so history is preserved by
construction and a snapshot can never be silently mutated under a report that
cites it.

Nothing here touches the network. See :mod:`regulatory_watch.retrieval`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from .contracts import SOURCE_CHECK_FAILED, SourceArtefact, SourceSnapshot

_CHUNK = 1024 * 1024


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Streaming sha256 of a file (the workbook is ~10 MB)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class SnapshotStore:
    """Append-only local snapshot store."""

    root: Path
    regime: str

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    # -- layout ------------------------------------------------------------ #
    def _dir_for(self, artefact_id: str, digest: str) -> Path:
        return self.root / self.regime / artefact_id / digest[:16]

    # -- write ------------------------------------------------------------- #
    def snapshot_local(self, artefact: SourceArtefact, source_file: Path,
                       retrieved_at: str, parser_version: str,
                       ) -> SourceSnapshot:
        """Persist an immutable snapshot of an existing local artefact.

        A missing or unreadable file is reported as ``SOURCE_CHECK_FAILED`` —
        never as an absent-and-therefore-unchanged source.
        """
        source_file = Path(source_file)
        if not source_file.exists() or not source_file.is_file():
            return self._failed(artefact, retrieved_at, parser_version,
                                f"local artefact not found: {source_file}")
        try:
            digest = sha256_file(source_file)
            size = source_file.stat().st_size
            payload = source_file.read_bytes()
        except OSError as exc:
            return self._failed(artefact, retrieved_at, parser_version,
                                f"local artefact unreadable: {exc}")

        target_dir = self._dir_for(artefact.artefact_id, digest)
        content = target_dir / f"content{source_file.suffix}"
        if not content.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
            content.write_bytes(payload)

        snap = SourceSnapshot(
            artefact_id=artefact.artefact_id,
            sha256=digest,
            byte_size=size,
            retrieved_at=retrieved_at,
            snapshot_path=str(content),
            parser_version=parser_version,
            criticality=artefact.criticality,
            source_url=artefact.source_url,
            external_version=artefact.external_version,
            publication_date=artefact.publication_date,
            effective_date=artefact.effective_date,
            source_status=artefact.source_status,
            retrieval_status="OK",
        )
        sidecar = target_dir / "snapshot.json"
        if not sidecar.exists():
            sidecar.write_text(
                json.dumps(snap.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
        return snap

    def snapshot_bytes(self, artefact: SourceArtefact, payload: bytes,
                       retrieved_at: str, parser_version: str,
                       suffix: str = "") -> SourceSnapshot:
        """Persist an immutable snapshot of in-memory bytes (e.g. a fetch)."""
        digest = sha256_bytes(payload)
        target_dir = self._dir_for(artefact.artefact_id, digest)
        content = target_dir / f"content{suffix}"
        if not content.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
            content.write_bytes(payload)
        snap = SourceSnapshot(
            artefact_id=artefact.artefact_id,
            sha256=digest,
            byte_size=len(payload),
            retrieved_at=retrieved_at,
            snapshot_path=str(content),
            parser_version=parser_version,
            criticality=artefact.criticality,
            source_url=artefact.source_url,
            external_version=artefact.external_version,
            publication_date=artefact.publication_date,
            effective_date=artefact.effective_date,
            source_status=artefact.source_status,
            retrieval_status="OK",
        )
        sidecar = target_dir / "snapshot.json"
        if not sidecar.exists():
            sidecar.write_text(
                json.dumps(snap.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
        return snap

    # -- read -------------------------------------------------------------- #
    def list_snapshots(self, artefact_id: str) -> List[SourceSnapshot]:
        base = self.root / self.regime / artefact_id
        if not base.exists():
            return []
        out: List[SourceSnapshot] = []
        for sidecar in sorted(base.glob("*/snapshot.json")):
            try:
                out.append(SourceSnapshot.from_dict(
                    json.loads(sidecar.read_text(encoding="utf-8"))))
            except (OSError, ValueError):
                continue
        return sorted(out, key=lambda s: (s.retrieved_at, s.sha256))

    # -- failure ----------------------------------------------------------- #
    @staticmethod
    def _failed(artefact: SourceArtefact, retrieved_at: str,
                parser_version: str, detail: str) -> SourceSnapshot:
        return SourceSnapshot(
            artefact_id=artefact.artefact_id,
            sha256="",
            byte_size=0,
            retrieved_at=retrieved_at,
            snapshot_path="",
            parser_version=parser_version,
            criticality=artefact.criticality,
            source_url=artefact.source_url,
            external_version=artefact.external_version,
            publication_date=artefact.publication_date,
            effective_date=artefact.effective_date,
            source_status=artefact.source_status,
            retrieval_status=SOURCE_CHECK_FAILED,
            retrieval_detail=detail,
        )

    @classmethod
    def failed_snapshot(cls, artefact: SourceArtefact, retrieved_at: str,
                        parser_version: str, detail: str) -> SourceSnapshot:
        return cls._failed(artefact, retrieved_at, parser_version, detail)


def snapshot_manifest(store: SnapshotStore, artefacts: Iterable[SourceArtefact],
                      repo_root: Path, retrieved_at: str,
                      parser_version: str) -> List[SourceSnapshot]:
    """Snapshot every artefact that declares a vendored local copy.

    Artefacts with no local copy yield an explicit ``SOURCE_CHECK_FAILED``
    snapshot so they are visible in the report as unchecked, not as unchanged.
    """
    out: List[SourceSnapshot] = []
    for artefact in artefacts:
        if not artefact.local_path:
            out.append(SnapshotStore.failed_snapshot(
                artefact, retrieved_at, parser_version,
                "no local copy vendored and retrieval is disabled"))
            continue
        out.append(store.snapshot_local(
            artefact, Path(repo_root) / artefact.local_path,
            retrieved_at, parser_version))
    return out
