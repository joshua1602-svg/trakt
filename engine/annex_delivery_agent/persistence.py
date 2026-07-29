"""
engine.annex_delivery_agent.persistence
=======================================

Operational state for annex delivery runs.

Kept deliberately separate from production/latest canonical data. A delivery run
is *operational* — it has attempts, checkpoints, failures and approvals — and
none of that belongs in the dataset an MI consumer reads. This store therefore
lives under its own root and never writes into a canonical or promoted location.

Two properties matter more than the storage technology:

**Atomic finalisation.** Every write lands on a temporary sibling and is moved
into place with :func:`os.replace`, which is atomic on POSIX and on Windows.
A reader therefore sees either the previous complete file or the new complete
file — never a half-written 200 MB XML that a naive size check would accept.

**Deterministic identity.** The run directory is derived from the governed
inputs (tenant, portfolio, annex, version, reporting date, input hash, config
fingerprints). The same request with the same configuration resolves to the same
directory and can be resumed; a changed rules file resolves elsewhere, so a
stale artefact can never be served as if it were current.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .contracts import ORDERED_STAGES, DeliveryResult, file_sha256

#: Marks a run directory as belonging to this agent. Guards destructive resets.
_MARKER = ".annex_delivery_run"

REQUEST_FILE = "00_request.json"
PREFLIGHT_FILE = "01_preflight.json"
CHECKPOINT_FILE = "02_checkpoint.json"
VALIDATION_FILE = "03_validation.json"
EVIDENCE_FILE = "04_transformation_evidence.json"
RESULT_FILE = "05_result.json"
PUBLICATION_FILE = "06_publication.json"
FAILURE_FILE = "07_failure.json"


# --------------------------------------------------------------------------- #
# Atomic primitives
# --------------------------------------------------------------------------- #


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> Path:
    """Write ``text`` to ``path`` atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".part")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        _quiet_unlink(Path(tmp))
        raise
    return path


def atomic_write_json(path: Path, payload: Any) -> Path:
    return atomic_write_text(path, json.dumps(payload, indent=2, default=str, sort_keys=False))


@dataclass
class StagedFile:
    """A file being produced, finalised atomically only once it is complete.

    Usage::

        with StagedFile(run_dir / "annex2.xml") as staged:
            tree.write(str(staged.temp_path), ...)
        # annex2.xml now exists, complete, or does not exist at all.

    If the body raises, the partial file is removed and the destination is left
    untouched. This is the mechanism behind "never treat a partial XML as valid".
    """

    destination: Path
    temp_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.destination = Path(self.destination)
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.destination.parent),
                                   prefix=f".{self.destination.name}.", suffix=".part")
        os.close(fd)
        self.temp_path = Path(tmp)

    def __enter__(self) -> "StagedFile":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            _quiet_unlink(self.temp_path)
            return False
        self.commit()
        return False

    def commit(self) -> Path:
        os.replace(self.temp_path, self.destination)
        return self.destination

    def abort(self) -> None:  # pragma: no cover - defensive
        _quiet_unlink(self.temp_path)


def _quiet_unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:  # pragma: no cover - best effort cleanup
        pass


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #


@dataclass
class Checkpoint:
    """What has completed for one run, and with what evidence.

    A stage is only replayable if its recorded output still exists *and* still
    hashes to the recorded value. That is what distinguishes "an artefact from a
    completed run" from "an artefact from a run that died mid-write" — the
    second either does not exist (atomic finalisation) or fails the hash.
    """

    run_id: str
    input_fingerprint: str = ""
    stages: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    attempts: int = 0

    def mark(self, stage: str, *, outputs: Optional[Mapping[str, str]] = None,
             detail: Optional[Mapping[str, Any]] = None) -> None:
        record: Dict[str, Any] = {
            "completed_at": time.time(),
            "outputs": {},
            "detail": dict(detail or {}),
        }
        for label, raw in (outputs or {}).items():
            p = Path(raw)
            record["outputs"][label] = {
                "path": str(p),
                "size_bytes": p.stat().st_size if p.exists() else 0,
                "sha256": file_sha256(p) if p.is_file() else "",
            }
        self.stages[stage] = record
        self.updated_at = time.time()

    def completed(self, stage: str) -> bool:
        return stage in self.stages

    def reusable(self, stage: str) -> bool:
        """True when ``stage`` completed and all of its outputs are still intact."""
        record = self.stages.get(stage)
        if record is None:
            return False
        for entry in (record.get("outputs") or {}).values():
            p = Path(entry.get("path", ""))
            if not p.is_file():
                return False
            if int(entry.get("size_bytes") or 0) != p.stat().st_size:
                return False
            digest = entry.get("sha256") or ""
            if digest and file_sha256(p) != digest:
                return False
        return True

    def output(self, stage: str, label: str) -> Optional[Path]:
        record = self.stages.get(stage) or {}
        entry = (record.get("outputs") or {}).get(label)
        return Path(entry["path"]) if entry else None

    def resume_stage(self) -> Optional[str]:
        """The first stage that still needs running, or ``None`` if all are done."""
        for stage in ORDERED_STAGES:
            if not self.reusable(stage):
                return stage
        return None

    def invalidate_from(self, stage: str) -> None:
        """Drop ``stage`` and everything downstream of it."""
        if stage not in ORDERED_STAGES:
            return
        idx = ORDERED_STAGES.index(stage)
        for later in ORDERED_STAGES[idx:]:
            self.stages.pop(later, None)
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "input_fingerprint": self.input_fingerprint,
            "stages": self.stages,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "attempts": self.attempts,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Checkpoint":
        d = dict(data)
        return cls(run_id=str(d.get("run_id", "")),
                   input_fingerprint=str(d.get("input_fingerprint") or ""),
                   stages=dict(d.get("stages") or {}),
                   created_at=float(d.get("created_at") or time.time()),
                   updated_at=float(d.get("updated_at") or time.time()),
                   attempts=int(d.get("attempts") or 0))


# --------------------------------------------------------------------------- #
# The store
# --------------------------------------------------------------------------- #


class DeliveryRunStore:
    """Filesystem-backed operational store for annex delivery runs.

    Layout — tenant first, so a tenant's runs are one prunable subtree and no
    listing operation can span tenants by accident::

        <root>/<tenant>/<annex_id>/<portfolio>/<reporting_date>/<run_id>/
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    # -- addressing --------------------------------------------------------- #
    def tenant_root(self, tenant_id: str) -> Path:
        return self.root / _slug(tenant_id)

    def run_dir(self, *, tenant_id: str, annex_id: str, portfolio_id: Optional[str],
                reporting_date: str, run_id: str) -> Path:
        return (self.tenant_root(tenant_id) / _slug(annex_id)
                / _slug(portfolio_id or "default") / _slug(reporting_date) / _slug(run_id))

    def open_run(self, *, tenant_id: str, annex_id: str, portfolio_id: Optional[str],
                 reporting_date: str, run_id: str) -> Path:
        path = self.run_dir(tenant_id=tenant_id, annex_id=annex_id,
                            portfolio_id=portfolio_id, reporting_date=reporting_date,
                            run_id=run_id)
        path.mkdir(parents=True, exist_ok=True)
        marker = path / _MARKER
        if not marker.exists():
            atomic_write_text(marker, json.dumps({"run_id": run_id, "tenant_id": tenant_id}))
        return path

    # -- documents ---------------------------------------------------------- #
    def write_request(self, run_dir: Path, payload: Mapping[str, Any]) -> Path:
        return atomic_write_json(run_dir / REQUEST_FILE, dict(payload))

    def write_preflight(self, run_dir: Path, payload: Mapping[str, Any]) -> Path:
        return atomic_write_json(run_dir / PREFLIGHT_FILE, dict(payload))

    def write_validation(self, run_dir: Path, payload: Mapping[str, Any]) -> Path:
        return atomic_write_json(run_dir / VALIDATION_FILE, dict(payload))

    def write_evidence(self, run_dir: Path, payload: Mapping[str, Any]) -> Path:
        return atomic_write_json(run_dir / EVIDENCE_FILE, dict(payload))

    def write_failure(self, run_dir: Path, payload: Mapping[str, Any]) -> Path:
        return atomic_write_json(run_dir / FAILURE_FILE, dict(payload))

    def write_publication(self, run_dir: Path, payload: Mapping[str, Any]) -> Path:
        return atomic_write_json(run_dir / PUBLICATION_FILE, dict(payload))

    def read_publication(self, run_dir: Path) -> Optional[Dict[str, Any]]:
        return _read_json(run_dir / PUBLICATION_FILE)

    def write_result(self, run_dir: Path, result: DeliveryResult) -> Path:
        return atomic_write_json(run_dir / RESULT_FILE, result.to_dict())

    def read_result(self, run_dir: Path) -> Optional[DeliveryResult]:
        data = _read_json(run_dir / RESULT_FILE)
        return DeliveryResult.from_dict(data) if data else None

    # -- checkpoints -------------------------------------------------------- #
    def load_checkpoint(self, run_dir: Path, *, run_id: str,
                        input_fingerprint: str) -> Checkpoint:
        data = _read_json(run_dir / CHECKPOINT_FILE)
        if not data:
            return Checkpoint(run_id=run_id, input_fingerprint=input_fingerprint)
        checkpoint = Checkpoint.from_dict(data)
        # A changed fingerprint under the same run id should be impossible (the
        # fingerprint is part of the id) but if it happens, trust nothing.
        if checkpoint.input_fingerprint != input_fingerprint:
            return Checkpoint(run_id=run_id, input_fingerprint=input_fingerprint)
        return checkpoint

    def save_checkpoint(self, run_dir: Path, checkpoint: Checkpoint) -> Path:
        return atomic_write_json(run_dir / CHECKPOINT_FILE, checkpoint.to_dict())

    # -- lifecycle ---------------------------------------------------------- #
    def reset_run(self, run_dir: Path) -> None:
        """Delete a run directory. Refuses anything this agent did not create."""
        run_dir = Path(run_dir)
        if not run_dir.exists():
            return
        if not (run_dir / _MARKER).exists():
            raise ValueError(
                f"Refusing to reset {run_dir}: it is not an annex delivery run directory.")
        shutil.rmtree(run_dir)

    def list_runs(self, tenant_id: str, *, annex_id: Optional[str] = None) -> List[Path]:
        """Every run directory belonging to ``tenant_id``. Never crosses tenants."""
        base = self.tenant_root(tenant_id)
        if annex_id:
            base = base / _slug(annex_id)
        if not base.exists():
            return []
        return sorted(p.parent for p in base.rglob(_MARKER))


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # pragma: no cover - corrupt operational state
        return None


def _slug(value: str) -> str:
    """A filesystem-safe path segment. Never allows traversal."""
    text = str(value or "").strip()
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in text)
    safe = safe.strip("._-") or "unnamed"
    return safe[:128]


__all__ = [
    "DeliveryRunStore", "Checkpoint", "StagedFile",
    "atomic_write_text", "atomic_write_json",
    "REQUEST_FILE", "PREFLIGHT_FILE", "CHECKPOINT_FILE", "VALIDATION_FILE",
    "EVIDENCE_FILE", "RESULT_FILE", "PUBLICATION_FILE", "FAILURE_FILE",
]
