"""operations_control.stores — persistence for the operations-control container.

All operational governance state (workflow runs, governed results, decisions,
rules, audit, publications) lives in its OWN container, ``operations-control``,
partitioned by client and then by workflow run — never mixed with production
reporting outputs. Storage reuses the existing file/Blob duality
(:mod:`apps.blob_trigger_app.storage`), so local dev is file-backed and Azure is
Blob-backed with no code difference.

Everything except small index/pointer documents is append-only; the audit trail
is hash-chained (same pattern as the remediation ledger) so it is tamper-evident
and every workflow can be reconstructed from the container alone.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import threading
import time

from apps.blob_trigger_app.storage import (
    BlobStorage,
    Storage,
    join_uri,
    open_storage,
)

from .audit_chain import append_to_chain
from .contracts import (
    DEC_OPEN,
    GovernedAgentResult,
    SCHEMA_VERSION,
    WorkflowRun,
    canonical_json,
    new_id,
    now_iso,
    stable_hash,
)

OPS_CONTAINER_ENV = "TRAKT_OPS_CONTAINER"
DEFAULT_OPS_CONTAINER = "operations-control"

GLOBAL_SCOPE_DIR = "_global"


def ops_container() -> str:
    return os.environ.get(OPS_CONTAINER_ENV, DEFAULT_OPS_CONTAINER)


@dataclass(frozen=True)
class OpsLayout:
    """Single path authority for the operations-control container."""

    container: str = DEFAULT_OPS_CONTAINER

    @classmethod
    def from_env(cls) -> "OpsLayout":
        return cls(container=ops_container())

    def _c(self, *parts: str) -> str:
        return join_uri(f"blob://{self.container}", *parts)

    # -- workflow runs ------------------------------------------------------ #
    def workflow_uri(self, client_id: str, workflow_id: str) -> str:
        return self._c(client_id, "workflow-runs", workflow_id, "workflow.json")

    def workflow_prefix(self, client_id: str) -> str:
        return self._c(client_id, "workflow-runs")

    def event_uri(self, client_id: str, workflow_id: str, seq: int, event: str) -> str:
        return self._c(client_id, "workflow-runs", workflow_id, "events",
                       f"{seq:05d}_{event}.json")

    def events_prefix(self, client_id: str, workflow_id: str) -> str:
        return self._c(client_id, "workflow-runs", workflow_id, "events")

    def result_uri(self, client_id: str, workflow_id: str, stage: str) -> str:
        return self._c(client_id, "workflow-runs", workflow_id, "results",
                       f"{stage}.json")

    def result_history_uri(self, client_id: str, workflow_id: str,
                           stage: str, result_id: str) -> str:
        return self._c(client_id, "workflow-runs", workflow_id, "results",
                       "history", f"{stage}_{result_id}.json")

    def lease_uri(self, client_id: str, workflow_id: str) -> str:
        return self._c(client_id, "workflow-runs", workflow_id, "lease.json")

    # -- deliveries --------------------------------------------------------- #
    def delivery_uri(self, client_id: str, delivery_id: str) -> str:
        return self._c(client_id, "deliveries", f"{delivery_id}.json")

    def deliveries_prefix(self, client_id: str) -> str:
        return self._c(client_id, "deliveries")

    # -- decisions ---------------------------------------------------------- #
    def decision_uri(self, client_id: str, decision_id: str) -> str:
        return self._c(client_id, "decisions", f"{decision_id}.json")

    def decisions_prefix(self, client_id: str) -> str:
        return self._c(client_id, "decisions")

    # -- approved onboarding decisions -------------------------------------- #
    def approved_decisions_uri(self, client_id: str, portfolio_id: str,
                               dataset: str, filename: str) -> str:
        """Durable home for the approved mapping contract of ONE source.

        Scoped by client + portfolio + dataset because that is the granularity
        at which a schema is stable: next month's file for the same source
        answers to the same decisions, and no other client's or portfolio's
        source can ever read it.
        """
        return self._c(client_id, "approved-decisions",
                       portfolio_id or "_", dataset or "funded", filename)

    # -- rules -------------------------------------------------------------- #
    def _rules_base(self, client_id: Optional[str]) -> str:
        return self._c(client_id or GLOBAL_SCOPE_DIR, "rules")

    def rule_version_uri(self, client_id: Optional[str], rule_id: str, version: int) -> str:
        return join_uri(self._rules_base(client_id), rule_id, "versions",
                        f"{version:04d}.json")

    def rule_current_uri(self, client_id: Optional[str], rule_id: str) -> str:
        return join_uri(self._rules_base(client_id), rule_id, "current.json")

    def rules_prefix(self, client_id: Optional[str]) -> str:
        return self._rules_base(client_id)

    # -- audit -------------------------------------------------------------- #
    def audit_uri(self, client_id: str, seq: int) -> str:
        return self._c(client_id, "audit", f"{seq:08d}.json")

    def audit_prefix(self, client_id: str) -> str:
        return self._c(client_id, "audit")

    def audit_head_uri(self, client_id: str) -> str:
        return self._c(client_id, "audit", "_head.json")

    def audit_key_uri(self, client_id: str, key: str) -> str:
        """Idempotency markers live OUTSIDE the audit prefix on purpose.

        ``list_audit`` lists that prefix recursively, so a marker stored under it
        would be read back as an audit record.
        """
        safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in key)
        return self._c(client_id, "audit-keys", f"{safe}.json")

    # -- publications / history --------------------------------------------- #
    def publication_uri(self, client_id: str, reporting_period: str) -> str:
        return self._c(client_id, "history", reporting_period, "publication.json")

    def history_prefix(self, client_id: str) -> str:
        return self._c(client_id, "history")

    # -- indexes (small, rebuildable) --------------------------------------- #
    def client_index_uri(self) -> str:
        return self._c("_index", "clients.json")

    def workflow_index_uri(self, client_id: str) -> str:
        return self._c(client_id, "workflow-runs", "index.json")


logger = logging.getLogger("operations_control.stores")


def _read_json(storage: Storage, uri: str) -> Optional[Dict[str, Any]]:
    """Read a JSON document; tolerate a concurrent writer with a short retry
    (filesystem renames are atomic, but a stale ``exists`` + read can still
    race the very first write of a document)."""
    for attempt in range(3):
        if not storage.exists(uri):
            return None
        try:
            return json.loads(storage.read_text(uri))
        except (json.JSONDecodeError, OSError):
            if attempt == 2:
                raise
            time.sleep(0.02 * (attempt + 1))
    return None


def _create_exclusive(storage: Storage, uri: str, text: str) -> bool:
    """Create ``uri`` only if absent. ``True`` when this call created it.

    Delegates to the backend primitive (``O_CREAT | O_EXCL`` on the filesystem,
    a conditional PUT on Blob). The fallback exists only for a substituted
    storage double in a test that predates the primitive; it is *not* safe under
    concurrency, and says so, because a silent unsafe fallback here would
    reintroduce the lost-update defect on whichever backend lacked the method.
    """
    creator = getattr(storage, "create_exclusive", None)
    if creator is not None:
        return bool(creator(uri, text))
    logger.warning("storage backend %s has no create_exclusive; audit appends "
                   "are NOT concurrency-safe on it", type(storage).__name__)
    if storage.exists(uri):
        return False
    storage.write_text(uri, text)
    return True


def _write_json(storage: Storage, uri: str, doc: Dict[str, Any]) -> str:
    """Atomic JSON write. Azure blob uploads are already atomic per blob; the
    filesystem backend gets write-to-temp + rename so a concurrent reader can
    never observe a truncated document."""
    text = json.dumps(doc, indent=2, default=str)
    if isinstance(storage, BlobStorage):
        return storage.write_text(uri, text)
    p = storage._local_path(uri)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)
    return uri


class OpsStore:
    """All operations-control persistence behind one facade.

    Construct via :meth:`from_env` in the API; tests construct with an explicit
    filesystem-backed :class:`Storage` and local root.
    """

    def __init__(self, storage: Storage, layout: Optional[OpsLayout] = None):
        self.storage = storage
        self.layout = layout or OpsLayout.from_env()

    @classmethod
    def from_env(cls) -> "OpsStore":
        return cls(open_storage(), OpsLayout.from_env())

    # ------------------------------------------------------------------ #
    # Client index
    # ------------------------------------------------------------------ #
    def known_clients(self) -> List[str]:
        doc = _read_json(self.storage, self.layout.client_index_uri()) or {}
        return sorted(doc.get("clients") or [])

    def register_client(self, client_id: str) -> None:
        uri = self.layout.client_index_uri()
        doc = _read_json(self.storage, uri) or {"clients": []}
        if client_id not in doc["clients"]:
            doc["clients"].append(client_id)
            _write_json(self.storage, uri, doc)

    # ------------------------------------------------------------------ #
    # Workflow runs
    # ------------------------------------------------------------------ #
    def save_workflow(self, run: WorkflowRun) -> None:
        run.updated_at = now_iso()
        _write_json(self.storage,
                    self.layout.workflow_uri(run.client_id, run.workflow_id),
                    run.to_dict())
        self._update_workflow_index(run)
        self.register_client(run.client_id)

    def load_workflow(self, client_id: str, workflow_id: str) -> Optional[WorkflowRun]:
        doc = _read_json(self.storage,
                         self.layout.workflow_uri(client_id, workflow_id))
        return WorkflowRun.from_dict(doc) if doc else None

    def _update_workflow_index(self, run: WorkflowRun) -> None:
        uri = self.layout.workflow_index_uri(run.client_id)
        doc = _read_json(self.storage, uri) or {"workflows": {}}
        doc["workflows"][run.workflow_id] = {
            "workflow_id": run.workflow_id, "client_id": run.client_id,
            "portfolio_id": run.portfolio_id, "status": run.status,
            "workflow_type": run.workflow_type, "outcome": run.outcome,
            "reporting_period": run.reporting_period,
            "idempotency_key": run.idempotency_key,
            "created_at": run.created_at, "updated_at": run.updated_at,
        }
        _write_json(self.storage, uri, doc)

    def list_workflows(self, client_id: str) -> List[Dict[str, Any]]:
        doc = _read_json(self.storage,
                         self.layout.workflow_index_uri(client_id)) or {}
        rows = list((doc.get("workflows") or {}).values())
        return sorted(rows, key=lambda r: r.get("created_at", ""), reverse=True)

    def find_by_idempotency_key(self, client_id: str,
                                key: str) -> Optional[Dict[str, Any]]:
        from .contracts import RUN_TERMINAL
        for row in self.list_workflows(client_id):
            if row.get("idempotency_key") == key and row.get("status") not in RUN_TERMINAL:
                return row
        return None

    def rebuild_workflow_index(self, client_id: str) -> int:
        """Recovery: rebuild the index by scanning workflow.json documents."""
        n = 0
        prefix = self.layout.workflow_prefix(client_id)
        for uri in self.storage.list(prefix):
            if uri.endswith("/workflow.json"):
                doc = _read_json(self.storage, uri)
                if doc:
                    self._update_workflow_index(WorkflowRun.from_dict(doc))
                    n += 1
        return n

    # -- transition events -------------------------------------------------- #
    def append_event(self, run: WorkflowRun, event: str,
                     detail: Optional[Dict[str, Any]] = None,
                     actor: str = "system") -> int:
        prefix = self.layout.events_prefix(run.client_id, run.workflow_id)
        seq = len([u for u in self.storage.list(prefix)
                   if u.endswith(".json")]) + 1
        _write_json(self.storage,
                    self.layout.event_uri(run.client_id, run.workflow_id, seq, event),
                    {"seq": seq, "event": event, "status": run.status,
                     "actor": actor, "at": now_iso(), "detail": detail or {}})
        return seq

    def list_events(self, client_id: str, workflow_id: str) -> List[Dict[str, Any]]:
        out = []
        for uri in self.storage.list(self.layout.events_prefix(client_id, workflow_id)):
            if not uri.endswith(".json"):
                continue
            doc = _read_json(self.storage, uri)
            if doc:
                out.append(doc)
        return sorted(out, key=lambda e: e.get("seq", 0))

    # -- execution lease ---------------------------------------------------- #
    def write_lease(self, run: WorkflowRun) -> None:
        _write_json(self.storage,
                    self.layout.lease_uri(run.client_id, run.workflow_id),
                    {"owner_pid": os.getpid(), "started_at": now_iso()})

    def read_lease(self, client_id: str, workflow_id: str) -> Optional[Dict[str, Any]]:
        return _read_json(self.storage, self.layout.lease_uri(client_id, workflow_id))

    def clear_lease(self, run: WorkflowRun) -> None:
        uri = self.layout.lease_uri(run.client_id, run.workflow_id)
        if self.storage.exists(uri):
            _write_json(self.storage, uri, {"owner_pid": None, "cleared_at": now_iso()})

    # ------------------------------------------------------------------ #
    # Governed results
    # ------------------------------------------------------------------ #
    def save_result(self, client_id: str, result: GovernedAgentResult) -> None:
        doc = result.to_dict()
        _write_json(self.storage,
                    self.layout.result_uri(client_id, result.run_id, result.stage), doc)
        _write_json(self.storage,
                    self.layout.result_history_uri(client_id, result.run_id,
                                                   result.stage, result.result_id), doc)

    def load_result(self, client_id: str, workflow_id: str,
                    stage: str) -> Optional[GovernedAgentResult]:
        doc = _read_json(self.storage,
                         self.layout.result_uri(client_id, workflow_id, stage))
        return GovernedAgentResult.from_dict(doc) if doc else None

    # ------------------------------------------------------------------ #
    # Deliveries
    # ------------------------------------------------------------------ #
    def save_delivery(self, doc: Dict[str, Any]) -> None:
        _write_json(self.storage,
                    self.layout.delivery_uri(doc["client_id"], doc["delivery_id"]), doc)
        self.register_client(doc["client_id"])

    def load_delivery(self, client_id: str, delivery_id: str) -> Optional[Dict[str, Any]]:
        return _read_json(self.storage,
                          self.layout.delivery_uri(client_id, delivery_id))

    def list_deliveries(self, client_id: str) -> List[Dict[str, Any]]:
        out = []
        for uri in self.storage.list(self.layout.deliveries_prefix(client_id)):
            if not uri.endswith(".json"):
                continue
            doc = _read_json(self.storage, uri)
            if doc:
                out.append(doc)
        return sorted(out, key=lambda d: d.get("registered_at", ""), reverse=True)

    # ------------------------------------------------------------------ #
    # Decisions (review items)
    # ------------------------------------------------------------------ #
    def save_decision(self, client_id: str, doc: Dict[str, Any]) -> None:
        _write_json(self.storage,
                    self.layout.decision_uri(client_id, doc["decision_id"]), doc)

    def load_decision(self, client_id: str, decision_id: str) -> Optional[Dict[str, Any]]:
        return _read_json(self.storage,
                          self.layout.decision_uri(client_id, decision_id))

    def list_decisions(self, client_id: str,
                       status: Optional[str] = None,
                       workflow_id: Optional[str] = None) -> List[Dict[str, Any]]:
        out = []
        for uri in self.storage.list(self.layout.decisions_prefix(client_id)):
            if not uri.endswith(".json"):
                continue
            doc = _read_json(self.storage, uri)
            if not doc:
                continue
            if status and doc.get("status") != status:
                continue
            if workflow_id and doc.get("workflow_id") != workflow_id:
                continue
            out.append(doc)
        return sorted(out, key=lambda d: d.get("created_at", ""))

    def open_decisions(self, client_id: str,
                       workflow_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.list_decisions(client_id, status=DEC_OPEN, workflow_id=workflow_id)

    # ------------------------------------------------------------------ #
    # Audit (hash-chained, per client)
    # ------------------------------------------------------------------ #
    def append_audit(self, client_id: str, event: str, *, actor: str,
                     detail: Optional[Dict[str, Any]] = None,
                     workflow_id: str = "", decision_id: str = "",
                     rule_id: str = "", idempotency_key: str = "") -> Dict[str, Any]:
        """Append one audit event. Safe under concurrent writers.

        Sequence allocation is an atomic exclusive create rather than a
        read-modify-write, so two simultaneous appends can never claim the same
        slot and silently lose one of the records. See
        :mod:`operations_control.audit_chain`.

        ``idempotency_key`` makes a retry return the ORIGINAL record instead of
        appending a second one. Without it every call appends, because two
        separate actions that happen to look alike are two events.
        """
        at = now_iso()

        def _build(seq: int, prev_hash: str) -> Dict[str, Any]:
            return {
                "seq": seq, "event": event, "actor": actor, "at": at,
                "client_id": client_id, "workflow_id": workflow_id,
                "decision_id": decision_id, "rule_id": rule_id,
                "detail": detail or {}, "prev_hash": prev_hash,
                "schema_version": SCHEMA_VERSION,
            }

        def _hash(record: Dict[str, Any]) -> str:
            return stable_hash(canonical_json(
                {k: record[k] for k in
                 ("seq", "event", "actor", "at", "client_id", "workflow_id",
                  "decision_id", "rule_id", "detail", "prev_hash")}))

        record, _seq, _created = append_to_chain(
            read_json=lambda uri: _read_json(self.storage, uri),
            create_exclusive=lambda uri, text: _create_exclusive(
                self.storage, uri, text),
            write_json=lambda uri, doc: _write_json(self.storage, uri, doc),
            head_uri=self.layout.audit_head_uri(client_id),
            record_uri=lambda seq: self.layout.audit_uri(client_id, seq),
            build_record=_build,
            record_hash_of=_hash,
            idempotency_key=idempotency_key,
            key_uri=lambda key: self.layout.audit_key_uri(client_id, key))
        return record

    def list_audit(self, client_id: str) -> List[Dict[str, Any]]:
        out = []
        for uri in self.storage.list(self.layout.audit_prefix(client_id)):
            if uri.endswith("_head.json") or not uri.endswith(".json"):
                continue
            doc = _read_json(self.storage, uri)
            if doc:
                out.append(doc)
        return sorted(out, key=lambda e: e.get("seq", 0))

    def verify_audit_chain(self, client_id: str) -> bool:
        prev = ""
        for rec in self.list_audit(client_id):
            if rec.get("prev_hash") != prev:
                return False
            expected = stable_hash(canonical_json(
                {k: rec.get(k) for k in
                 ("seq", "event", "actor", "at", "client_id", "workflow_id",
                  "decision_id", "rule_id", "detail", "prev_hash")}))
            if rec.get("record_hash") != expected:
                return False
            prev = rec["record_hash"]
        return True

    # ------------------------------------------------------------------ #
    # Publications
    # ------------------------------------------------------------------ #
    def save_publication(self, client_id: str, doc: Dict[str, Any]) -> None:
        _write_json(self.storage,
                    self.layout.publication_uri(client_id, doc["reporting_period"]),
                    doc)

    def load_publication(self, client_id: str,
                         reporting_period: str) -> Optional[Dict[str, Any]]:
        return _read_json(self.storage,
                          self.layout.publication_uri(client_id, reporting_period))

    def list_publications(self, client_id: str) -> List[Dict[str, Any]]:
        out = []
        for uri in self.storage.list(self.layout.history_prefix(client_id)):
            if uri.endswith("/publication.json"):
                doc = _read_json(self.storage, uri)
                if doc:
                    out.append(doc)
        return sorted(out, key=lambda d: d.get("reporting_period", ""), reverse=True)
