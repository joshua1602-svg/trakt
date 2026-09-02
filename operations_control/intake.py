"""operations_control.intake — OCC-owned input batches and readiness.

Replaces the ``_READY.json`` sentinel. Readiness is determined from uploaded
files, recognised semantic input roles (the EXISTING header-first
``file_roles`` classifier), the workflow's administrator-governed input
requirements, effective-configuration status and unresolved blocking
decisions — then the OCC creates an immutable internal run manifest and
advances the run itself. Users never create a sentinel file.

Batch states: receiving | incomplete | classifying | review_required |
configuration_required | ready | running | completed | failed.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from apps.blob_trigger_app import file_roles as fr
from apps.blob_trigger_app.file_roles import classify_pack

from .contracts import (BATCH_DATASET_DEFAULT, BATCH_FREQUENCY_DEFAULT,
                        new_id, now_iso, stable_hash)
from .stores import OpsStore, _read_json, _write_json


REPO = Path(__file__).resolve().parents[1]
REQUIREMENTS_PATH = REPO / "config/system/workflow_input_requirements.yaml"

B_RECEIVING = "receiving"
B_INCOMPLETE = "incomplete"
B_CLASSIFYING = "classifying"
B_REVIEW_REQUIRED = "review_required"
B_CONFIG_REQUIRED = "configuration_required"
B_READY = "ready"
B_RUNNING = "running"
B_COMPLETED = "completed"
B_FAILED = "failed"

DATA_EXTS = {".csv", ".xlsx", ".xls", ".xlsm"}
LEGACY_SENTINEL = "_READY.json"

#: How many versions of one input pack are probed when asking whether its
#: content has already been seen. Far above any real delivery; a bound only so a
#: corrupt identifier cannot turn the lookup into an unbounded scan.
MAX_PACK_VERSIONS = 50


def load_requirements(path: Path = REQUIREMENTS_PATH) -> Dict[str, Any]:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return doc


@dataclass
class InputFileRecord:
    source_file_id: str
    original_filename: str
    storage_reference: str            # staged path (provenance only)
    sha256: str
    size: int
    received_at: str
    received_by_or_source: str
    #: Governed source location this file arrived at (the raw container path for
    #: both automated arrivals and manual uploads). Provenance only — it is
    #: presented as the delivery's source location, never as a path the browser
    #: may choose.
    source_uri: str = ""
    recognised_file_type: str = ""    # semantic role
    recognised_schema: str = ""       # header-columns digest
    recognition_confidence: Optional[float] = None
    recognition_basis: str = ""       # header_signature | registry_alias | ...
    recognition_status: str = "pending"   # recognised | ambiguous | unsupported | overridden
    duplicate_status: str = "unique"      # unique | duplicate_ignored
    superseded_status: str = "current"    # current | superseded

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class IntakeService:
    """Batch registration + readiness assessment over the OCC store."""

    def __init__(self, store: OpsStore, *, staging_root: Path,
                 requirements_path: Path = REQUIREMENTS_PATH):
        self.store = store
        self.staging_root = Path(staging_root)
        self.requirements_path = requirements_path

    # -- persistence -------------------------------------------------------- #
    def _batch_uri(self, client_id: str, batch_id: str) -> str:
        return (f"blob://{self.store.layout.container}/{client_id}/batches/"
                f"{batch_id}.json")

    def _manifest_uri(self, client_id: str, batch_id: str) -> str:
        return (f"blob://{self.store.layout.container}/{client_id}/batches/"
                f"{batch_id}/run_manifest.json")

    def save_batch(self, batch: Dict[str, Any]) -> None:
        batch["updated_at"] = now_iso()
        _write_json(self.store.storage,
                    self._batch_uri(batch["client_id"], batch["batch_id"]),
                    batch)

    def load_batch(self, client_id: str, batch_id: str) -> Optional[Dict[str, Any]]:
        return _read_json(self.store.storage,
                          self._batch_uri(client_id, batch_id))

    def list_batches(self, client_id: str) -> List[Dict[str, Any]]:
        prefix = f"blob://{self.store.layout.container}/{client_id}/batches"
        out = []
        for uri in self.store.storage.list(prefix):
            if uri.endswith(".json") and "/run_manifest" not in uri \
                    and uri.count("/") == prefix.count("/") + 1:
                doc = _read_json(self.store.storage, uri)
                if doc:
                    out.append(doc)
        return sorted(out, key=lambda b: b.get("created_at", ""), reverse=True)

    # -- batch lifecycle ----------------------------------------------------- #
    def deterministic_batch_id(self, *, client_id: str, portfolio_id: str,
                               reporting_date: str, workflow_type: str,
                               dataset: str = "") -> str:
        """Identity of an input pack.

        ``dataset`` extends the key ONLY when it is something other than the
        funded book. The funded delivery is the pack this system has always
        described, so its identifier stays byte-identical — no existing pack is
        re-keyed and nothing part-way through collection is stranded. A pipeline
        delivery, which is a separate MI view rather than part of the funded pack,
        gets its own identifier from the first upload.
        """
        parts = [client_id, portfolio_id, reporting_date, workflow_type]
        if dataset and dataset != BATCH_DATASET_DEFAULT:
            parts.append(dataset)
        return "batch_" + stable_hash(*parts)

    def create_batch(self, *, tenant_id: str, client_id: str,
                     portfolio_id: str, reporting_date: str,
                     workflow_type: str, created_by: str,
                     auto_start_when_ready: bool = False,
                     batch_id: Optional[str] = None,
                     dataset: str = "",
                     frequency: str = "") -> Dict[str, Any]:
        """Create (or return the existing open) batch for this context."""
        bid = batch_id or self.deterministic_batch_id(
            client_id=client_id, portfolio_id=portfolio_id,
            reporting_date=reporting_date, workflow_type=workflow_type,
            dataset=dataset)
        existing = self.load_batch(client_id, bid)
        if existing is not None and existing.get("status") not in (
                B_RUNNING, B_COMPLETED, B_FAILED):
            return existing
        if existing is not None:
            # Late arrival for a started batch → NEW batch version; the active
            # run's inputs are never mutated.
            version = int(existing.get("version") or 1) + 1
            bid = f"{bid}_v{version}"
            reopened = self.load_batch(client_id, bid)
            if reopened is not None and reopened.get("status") not in (
                    B_RUNNING, B_COMPLETED, B_FAILED):
                return reopened
        req = load_requirements(self.requirements_path)
        wf_req = (req.get("workflows") or {}).get(workflow_type) or {}
        batch = {
            "batch_id": bid,
            "version": int(bid.rsplit("_v", 1)[1]) if "_v" in bid else 1,
            "tenant_id": tenant_id, "client_id": client_id,
            "portfolio_id": portfolio_id, "reporting_date": reporting_date,
            "workflow_type": workflow_type,
            # Recorded so the separation is visible in the OCC rather than implied
            # by the identifier. Defaults to the funded book for packs created
            # without one (manual creation from the OCC).
            "dataset": dataset or BATCH_DATASET_DEFAULT,
            # How often this delivery arrives. Recorded (not part of the pack
            # identity) so the governed source location can be derived without
            # the browser ever supplying one.
            "frequency": frequency or BATCH_FREQUENCY_DEFAULT,
            #: Governed source prefix files for this pack are filed under. Set
            #: on first arrival; shown as the delivery's source location.
            "source_prefix": "",
            "status": B_RECEIVING,
            "status_reason": "Files are still arriving.",
            "files": [],
            "expected_input_roles": {
                "required": list(wf_req.get("required_roles") or []),
                "optional": list(wf_req.get("optional_roles") or []),
            },
            "received_input_roles": [],
            "missing_input_roles": list(wf_req.get("required_roles") or []),
            "ambiguous_files": [],
            "legacy_sentinels_ignored": [],
            "blocking_decisions": [],
            "effective_config_id": "", "effective_config_hash": "",
            "effective_config_status": "",
            "auto_start_when_ready": bool(auto_start_when_ready),
            "workflow_id": "",
            "created_at": now_iso(), "created_by": created_by,
            "updated_at": now_iso(), "started_at": "", "completed_at": "",
        }
        self.save_batch(batch)
        self.store.append_audit(client_id, "input_batch_created",
                                actor=created_by,
                                detail={"batch_id": bid,
                                        "workflow_type": workflow_type,
                                        "reporting_date": reporting_date,
                                        "auto_start": auto_start_when_ready})
        return batch

    def batch_dir(self, batch: Dict[str, Any]) -> Path:
        return (self.staging_root / batch["client_id"] / "batches"
                / batch["batch_id"] / "files")

    # -- pack versions ------------------------------------------------------- #
    @staticmethod
    def base_batch_id(batch_id: str) -> str:
        """The first version of a pack — ``…_v3`` and ``…`` are one pack."""
        return re.sub(r"_v\d+$", "", batch_id or "")

    def pack_family(self, client_id: str, batch_id: str) -> List[Dict[str, Any]]:
        """Every version of one input pack, oldest first."""
        base = self.base_batch_id(batch_id)
        out: List[Dict[str, Any]] = []
        first = self.load_batch(client_id, base)
        if first is not None:
            out.append(first)
        for version in range(2, MAX_PACK_VERSIONS + 1):
            doc = self.load_batch(client_id, f"{base}_v{version}")
            if doc is None:
                break
            out.append(doc)
        return out

    def find_registered(self, client_id: str, batch_id: str, digest: str
                        ) -> Optional[Dict[str, Any]]:
        """The pack version already holding this exact content, if any.

        Identity is the CONTENT, across every version of the pack — not the
        filename, and not one version in isolation. A delivery uploaded by hand
        is filed where an automated one would be, so the same bytes are offered
        to intake twice: once by the operator and once by the storage event that
        their upload raises. The second offer is the same file, not a late one,
        and must not open a successor pack (which would collect a duplicate copy
        of the delivery and start a second run for it).

        This also makes the automated route idempotent against the at-least-once
        redelivery its own event source is entitled to perform.
        """
        for pack in self.pack_family(client_id, batch_id):
            for f in pack.get("files") or []:
                if f.get("sha256") == digest and \
                        f.get("superseded_status") == "current":
                    return {"batch_id": pack["batch_id"], "file": f}
        return None

    # -- file registration --------------------------------------------------- #
    def register_file(self, batch: Dict[str, Any], source_path: Path, *,
                      received_by_or_source: str,
                      original_filename: Optional[str] = None,
                      source_uri: str = "") -> Dict[str, Any]:
        """Register one received file: stage, hash, dedupe. Legacy sentinel
        files are recorded as unsupported legacy artefacts and NEVER trigger
        anything."""
        name = original_filename or Path(source_path).name
        if name == LEGACY_SENTINEL:
            if name not in batch["legacy_sentinels_ignored"]:
                batch["legacy_sentinels_ignored"].append(name)
            self.save_batch(batch)
            self.store.append_audit(
                batch["client_id"], "legacy_sentinel_ignored",
                actor=received_by_or_source,
                detail={"batch_id": batch["batch_id"], "filename": name,
                        "note": "_READY.json is no longer supported and does "
                                "not trigger processing"})
            return batch
        data = Path(source_path).read_bytes()
        digest = "sha256:" + hashlib.sha256(data).hexdigest()

        # Content this pack already holds is not a new file, whichever version
        # of the pack holds it and whichever door offered it. Checked BEFORE the
        # successor branch, so an echo of a file this pack has already processed
        # cannot open a successor pack for itself.
        seen = self.find_registered(batch["client_id"], batch["batch_id"],
                                    digest)
        if seen is not None:
            self.store.append_audit(
                batch["client_id"], "file_registered",
                actor=received_by_or_source,
                detail={"batch_id": batch["batch_id"], "filename": name,
                        "duplicate_of": seen["file"]["source_file_id"],
                        "duplicate_of_batch_id": seen["batch_id"],
                        "duplicate_status": "duplicate_ignored"})
            return batch

        if batch.get("status") in (B_RUNNING, B_COMPLETED):
            # Never mutate a started batch — route into a successor version.
            batch = self.create_batch(
                tenant_id=batch["tenant_id"], client_id=batch["client_id"],
                portfolio_id=batch["portfolio_id"],
                reporting_date=batch["reporting_date"],
                workflow_type=batch["workflow_type"],
                created_by=received_by_or_source,
                auto_start_when_ready=batch.get("auto_start_when_ready", False),
                # Carry the dataset, or a late pipeline file would open its
                # successor in the funded pack's key space — reintroducing the
                # very collision the dataset key exists to prevent.
                dataset=batch.get("dataset", ""),
                frequency=batch.get("frequency", ""))
        dest = self.batch_dir(batch)
        dest.mkdir(parents=True, exist_ok=True)
        staged = dest / name
        replaced = staged.exists()
        staged.write_bytes(data)
        if replaced:
            for f in batch["files"]:
                if f["original_filename"] == name and \
                        f["superseded_status"] == "current":
                    f["superseded_status"] = "superseded"
        rec = InputFileRecord(
            source_file_id=new_id("file"), original_filename=name,
            storage_reference=str(staged), sha256=digest, size=len(data),
            received_at=now_iso(),
            received_by_or_source=received_by_or_source,
            source_uri=source_uri)
        batch["files"].append(rec.to_dict())
        self.save_batch(batch)
        self.store.append_audit(
            batch["client_id"], "file_registered", actor=received_by_or_source,
            detail={"batch_id": batch["batch_id"], "filename": name,
                    "sha256": digest, "size": len(data),
                    "replaced_previous": replaced})
        return batch

    def restage_missing_files(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Put back any registered file whose staged copy has gone.

        The staging root is scratch — ``/tmp`` on the deployed App Service — so a
        worker recycle takes the delivery's files with it while the governed run
        record, its decisions and its rules all survive in durable storage. The
        raw file also survives, at the governed blob location recorded on the
        file when it was registered, so a rerun can put it back rather than
        failing on an empty directory.

        Nothing is re-registered and no new pack version is opened: the same
        bytes are restored to the same path, and the recorded SHA-256 must match
        or the file is left missing. A file with no ``source_uri`` (a hand
        upload from before the location was recorded) cannot be restored.

        Returns ``{"restored": [...], "missing": [...]}``; both empty is the
        normal case where nothing was lost.
        """
        report: Dict[str, Any] = {"restored": [], "missing": []}
        dest = self.batch_dir(batch)
        for f in self._current_files(batch):
            ref = str(f.get("storage_reference") or "")
            if ref and Path(ref).exists():
                continue
            name = str(f.get("original_filename") or "")
            uri = str(f.get("source_uri") or "")
            if not uri:
                report["missing"].append(
                    {"filename": name, "reason": "no governed source location "
                                                 "was recorded for this file"})
                continue
            try:
                dest.mkdir(parents=True, exist_ok=True)
                target = dest / name
                self.store.storage.download_file(uri, target)
                digest = "sha256:" + hashlib.sha256(
                    target.read_bytes()).hexdigest()
                if f.get("sha256") and digest != f["sha256"]:
                    target.unlink(missing_ok=True)
                    report["missing"].append(
                        {"filename": name,
                         "reason": "the file at the governed location is no "
                                   "longer the one that was received"})
                    continue
                f["storage_reference"] = str(target)
                report["restored"].append({"filename": name, "source_uri": uri})
            except Exception as exc:  # noqa: BLE001
                report["missing"].append({"filename": name,
                                          "reason": f"{type(exc).__name__}"})
        if report["restored"]:
            self.save_batch(batch)
            self.store.append_audit(
                batch["client_id"], "delivery_files_restaged", actor="system",
                detail={"batch_id": batch["batch_id"],
                        "restored": [r["filename"] for r in report["restored"]],
                        "note": "staged copies were missing and were restored "
                                "from the governed source location"})
        return report

    # -- recognition --------------------------------------------------------- #
    def _current_files(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [f for f in batch["files"]
                if f["superseded_status"] == "current"
                and f["duplicate_status"] == "unique"]

    def classify(self, batch: Dict[str, Any], *,
                 role_schemas: Optional[Dict[str, List[str]]] = None,
                 aliases: Optional[Dict[str, List[str]]] = None
                 ) -> Dict[str, Any]:
        """Recognise file roles using the EXISTING header-first classifier."""
        files = self._current_files(batch)
        headered = []
        for f in files:
            p = Path(f["storage_reference"])
            if p.suffix.lower() not in DATA_EXTS:
                f["recognition_status"] = "unsupported"
                continue
            cols = self._headers(p)
            headered.append((f, p.name, cols))
        result = classify_pack([(name, cols) for _, name, cols in headered],
                               role_schemas=role_schemas, aliases=aliases)
        diags = {d.get("filename"): d for d in result.diagnostics()}
        for f, name, cols in headered:
            if f["recognition_status"] == "overridden":
                continue    # operator override wins; never re-classified away
            d = diags.get(name) or {}
            f["recognised_file_type"] = str(d.get("assigned_role") or "unknown")
            f["recognition_basis"] = str(d.get("role_basis") or "")
            conf = d.get("confidence")
            f["recognition_confidence"] = (float(conf) if conf is not None
                                           else None)
            f["recognised_schema"] = "cols:" + stable_hash(*map(str, cols))
            # A fallback assignment is not a recognition: the role was invented
            # from the filename at confidence 0 and matches nothing the workflow
            # expects, so the pack waits for an input that will never arrive
            # under that name. Treat it as ambiguous, which raises the file-role
            # decision and gives the operator a governed way to say what the
            # file is — a new lender's tape is called whatever the lender calls
            # it, and waiting silently is not an answer.
            f["recognition_status"] = (
                "ambiguous"
                if (f["recognition_basis"] == fr.BASIS_FALLBACK
                    or (result.ambiguous_role_conflict
                        and f["recognised_file_type"] in result.conflicting_roles))
                else "recognised")
            self.store.append_audit(
                batch["client_id"], "file_classified", actor="system",
                detail={"batch_id": batch["batch_id"], "filename": name,
                        "role": f["recognised_file_type"],
                        "basis": f["recognition_basis"],
                        "confidence": f["recognition_confidence"],
                        "status": f["recognition_status"]})
        self.save_batch(batch)
        return batch

    @staticmethod
    def _headers(p: Path) -> List[str]:
        import pandas as pd
        try:
            if p.suffix.lower() == ".csv":
                return [str(c) for c in pd.read_csv(p, nrows=0).columns]
            return [str(c) for c in pd.read_excel(p, nrows=0).columns]
        except Exception:  # noqa: BLE001
            return []

    def override_classification(self, batch: Dict[str, Any],
                                source_file_id: str, role: str, *,
                                actor: str) -> Dict[str, Any]:
        for f in batch["files"]:
            if f["source_file_id"] == source_file_id:
                f["recognised_file_type"] = role
                f["recognition_status"] = "overridden"
                f["recognition_confidence"] = 1.0
                f["recognition_basis"] = "operator_override"
        self.save_batch(batch)
        self.store.append_audit(
            batch["client_id"], "file_classification_overridden", actor=actor,
            detail={"batch_id": batch["batch_id"],
                    "source_file_id": source_file_id, "role": role})
        return batch

    # -- readiness ----------------------------------------------------------- #
    def assess(self, batch: Dict[str, Any], *,
               config_status: str = "", config_id: str = "",
               config_hash: str = "",
               open_blocking_decisions: Optional[List[str]] = None
               ) -> Dict[str, Any]:
        """Reassess readiness from roles + configuration + blockers. Called
        after every registration/recognition/decision change."""
        req = load_requirements(self.requirements_path)
        min_conf = float(req.get("min_required_role_confidence") or 0.0)
        required = list(batch["expected_input_roles"]["required"])
        files = self._current_files(batch)

        satisfied: Dict[str, Dict[str, Any]] = {}
        ambiguous: List[Dict[str, Any]] = []
        for f in files:
            role = f.get("recognised_file_type") or ""
            status = f.get("recognition_status")
            conf = f.get("recognition_confidence")
            if status == "ambiguous":
                ambiguous.append({"filename": f["original_filename"],
                                  "candidates": [role]})
                continue
            if status in ("recognised", "overridden") and role:
                strong = (status == "overridden" or conf is None
                          or conf >= min_conf
                          or f.get("recognition_basis") == "header_signature")
                if role in required and not strong:
                    ambiguous.append({"filename": f["original_filename"],
                                      "candidates": [role],
                                      "reason": "low_confidence"})
                    continue
                satisfied.setdefault(role, f)
        batch["received_input_roles"] = sorted(satisfied)
        batch["missing_input_roles"] = [r for r in required
                                        if r not in satisfied]
        batch["ambiguous_files"] = ambiguous
        batch["blocking_decisions"] = list(open_blocking_decisions or [])
        if config_status:
            batch["effective_config_status"] = config_status
            batch["effective_config_id"] = config_id
            batch["effective_config_hash"] = config_hash

        if not files:
            status, reason = B_RECEIVING, "Files are still arriving."
        elif ambiguous:
            status = B_REVIEW_REQUIRED
            names = ", ".join(a["filename"] for a in ambiguous[:3])
            reason = (f"{names} could not be identified with confidence. "
                      "Please confirm what each file is.")
        elif batch["missing_input_roles"]:
            status = B_INCOMPLETE
            labels = req.get("role_labels") or {}
            missing = ", ".join(labels.get(r, r.replace("_", " "))
                                for r in batch["missing_input_roles"])
            reason = f"Waiting for required input: {missing}."
        elif batch["blocking_decisions"]:
            status = B_REVIEW_REQUIRED
            reason = "Decisions need your attention before processing."
        elif batch.get("effective_config_status") == "BLOCKED":
            status = B_CONFIG_REQUIRED
            reason = ("The input pack is complete, but required configuration "
                      "is missing.")
        else:
            status, reason = B_READY, "Ready to process."
        batch["status"], batch["status_reason"] = status, reason
        self.save_batch(batch)
        self.store.append_audit(
            batch["client_id"], "readiness_evaluated", actor="system",
            detail={"batch_id": batch["batch_id"], "status": status,
                    "received_roles": batch["received_input_roles"],
                    "missing_roles": batch["missing_input_roles"],
                    "ambiguous": [a["filename"] for a in ambiguous],
                    "config_status": batch.get("effective_config_status", "")})
        if status in (B_INCOMPLETE, B_REVIEW_REQUIRED, B_CONFIG_REQUIRED):
            event = {B_INCOMPLETE: "batch_incomplete",
                     B_REVIEW_REQUIRED: "batch_review_required",
                     B_CONFIG_REQUIRED: "configuration_required"}[status]
            self.store.append_audit(batch["client_id"], event, actor="system",
                                    detail={"batch_id": batch["batch_id"]})
        if status == B_READY:
            self.store.append_audit(batch["client_id"], "batch_ready",
                                    actor="system",
                                    detail={"batch_id": batch["batch_id"]})
        return batch

    # -- internal run manifest ------------------------------------------------ #
    def ensure_manifest(self, batch: Dict[str, Any], *,
                        effective_config: Dict[str, Any],
                        approved_decisions: List[str],
                        expected_outputs: List[str]) -> Dict[str, Any]:
        """Create the immutable internal run manifest (idempotent). This is the
        governed replacement for the operational purpose of _READY.json."""
        uri = self._manifest_uri(batch["client_id"], batch["batch_id"])
        files = [{"source_file_id": f["source_file_id"],
                  "original_filename": f["original_filename"],
                  "sha256": f["sha256"],
                  "role": f["recognised_file_type"],
                  "storage_reference": f["storage_reference"]}
                 for f in self._current_files(batch)]
        idem = stable_hash(batch["batch_id"],
                           *(f["sha256"] for f in files),
                           effective_config.get("content_hash", ""),
                           batch["workflow_type"], batch["reporting_date"])
        existing = _read_json(self.store.storage, uri)
        if existing is not None:
            if existing.get("idempotency_key") != idem:
                raise ValueError(
                    "run manifest already exists for this batch with "
                    "different inputs — a new batch version is required")
            return existing
        manifest = {
            "run_id": "run_" + idem[:12],
            "batch_id": batch["batch_id"],
            "tenant_id": batch["tenant_id"], "client_id": batch["client_id"],
            "portfolio_id": batch["portfolio_id"],
            "reporting_date": batch["reporting_date"],
            "workflow_type": batch["workflow_type"],
            "input_files": files,
            "effective_configuration": {
                "effective_config_id": effective_config.get(
                    "effective_config_id", ""),
                "content_hash": effective_config.get("content_hash", ""),
            },
            "approved_decisions": approved_decisions,
            "expected_outputs": expected_outputs,
            "readiness_evidence": {
                "received_roles": batch["received_input_roles"],
                "expected_roles": batch["expected_input_roles"],
                "assessed_at": now_iso(),
            },
            "idempotency_key": idem,
            "created_at": now_iso(),
        }
        _write_json(self.store.storage, uri, manifest)
        self.store.append_audit(batch["client_id"], "run_manifest_created",
                                actor="system",
                                detail={"batch_id": batch["batch_id"],
                                        "run_id": manifest["run_id"],
                                        "idempotency_key": idem,
                                        "file_count": len(files)})
        return manifest

    def load_manifest(self, client_id: str,
                      batch_id: str) -> Optional[Dict[str, Any]]:
        return _read_json(self.store.storage,
                          self._manifest_uri(client_id, batch_id))
