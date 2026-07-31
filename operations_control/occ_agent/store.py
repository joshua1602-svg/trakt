"""operations_control.occ_agent.store — synthetic case persistence.

Synthetic cases live in their OWN container, ``operations-control-synthetic``,
never in the live ``operations-control`` container. The separation is structural
rather than conventional: :class:`SyntheticCaseStore` is constructed with its
own :class:`~apps.blob_trigger_app.storage.Storage` root and refuses a container
name that is (or is a prefix of) the live one, so a mis-set environment variable
cannot land synthetic documents among live workflow runs.

Layout, mirroring the live store's shape so a later migration is mechanical::

    blob://operations-control-synthetic/
      {tenant}/cases/{case_id}/case.json
      {tenant}/cases/{case_id}/audit/{00000001}.json
      {tenant}/cases/{case_id}/audit/_head.json
      {tenant}/cases/index.json

Artefact bytes and run working files never go to blob storage at all: they live
under a filesystem sandbox root (``TRAKT_OCC_AGENT_SANDBOX_ROOT``), one directory
per tenant and case, reachable only through
:func:`~operations_control.occ_agent.policy.sandbox_path`.

The audit trail is hash-chained with the same helper the live store uses
(:func:`operations_control.contracts.stable_hash` over canonical JSON), so a
synthetic case's history is tamper-evident in exactly the way a live one's is.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from apps.blob_trigger_app.storage import Storage, join_uri

from ..contracts import canonical_json, new_id, now_iso, stable_hash
from ..engine import OpsError
from ..stores import DEFAULT_OPS_CONTAINER, _read_json, _write_json
from .case import (
    ACTOR_SYSTEM,
    CaseAuditEvent,
    EXEC_DETERMINISTIC,
    SyntheticCase,
)
from .policy import (
    RUNTIME_MODE_SYNTHETIC,
    sandbox_path,
    validate_case_id,
    validate_segment,
)

CONTAINER_ENV = "TRAKT_OCC_AGENT_SYNTHETIC_CONTAINER"
DEFAULT_SYNTHETIC_CONTAINER = "operations-control-synthetic"

SANDBOX_ROOT_ENV = "TRAKT_OCC_AGENT_SANDBOX_ROOT"
DEFAULT_SANDBOX_ROOT = ".occ-agent-synthetic"


class CaseNotFound(OpsError):
    """No such synthetic case for this tenant.

    404 rather than 403, matching :func:`operations_control.api.auth.
    require_client`: another tenant's case must not be revealed to exist.
    """

    def __init__(self) -> None:
        super().__init__("OCC_AGENT_CASE_NOT_FOUND",
                         "That synthetic case could not be found.",
                         http_status=404)


class StoreMisconfigured(OpsError):
    """The synthetic store was pointed at the live container."""

    def __init__(self, detail: str) -> None:
        super().__init__(
            "OCC_AGENT_STORE_MISCONFIGURED",
            "The synthetic case store is not configured safely. Ask your "
            f"administrator. ({detail})", http_status=503)


def synthetic_container() -> str:
    name = (os.environ.get(CONTAINER_ENV) or DEFAULT_SYNTHETIC_CONTAINER).strip()
    _assert_isolated(name)
    return name


def _assert_isolated(container: str) -> None:
    """Refuse any container that could collide with live operational state."""
    if not container:
        raise StoreMisconfigured("no container name")
    if container == DEFAULT_OPS_CONTAINER:
        raise StoreMisconfigured("synthetic cases may not share the live "
                                 "operations container")
    live = os.environ.get("TRAKT_OPS_CONTAINER", DEFAULT_OPS_CONTAINER).strip()
    if live and container == live:
        raise StoreMisconfigured("synthetic cases may not share the live "
                                 "operations container")
    if "/" in container or container.startswith("."):
        raise StoreMisconfigured("container name")


def sandbox_root() -> Path:
    return Path(os.environ.get(SANDBOX_ROOT_ENV) or DEFAULT_SANDBOX_ROOT)


class SyntheticCaseStore:
    """All synthetic-case persistence behind one facade.

    ``storage`` is injected (the API constructs it from the environment; tests
    pass a tmp-dir-backed :class:`Storage`), keeping this class free of
    environment reads at call time and free of any Azure dependency.
    """

    def __init__(self, storage: Storage, *, container: Optional[str] = None,
                 sandbox: Optional[Path] = None):
        self.storage = storage
        self.container = (container or synthetic_container())
        _assert_isolated(self.container)
        self.sandbox = Path(sandbox or sandbox_root())

    # ------------------------------------------------------------------ #
    # URIs and sandbox paths
    # ------------------------------------------------------------------ #
    def _c(self, *parts: str) -> str:
        return join_uri(f"blob://{self.container}", *parts)

    def case_uri(self, tenant: str, case_id: str) -> str:
        return self._c(validate_segment(tenant, "tenant"),
                       "cases", validate_case_id(case_id), "case.json")

    def index_uri(self, tenant: str) -> str:
        return self._c(validate_segment(tenant, "tenant"), "cases",
                       "index.json")

    def audit_uri(self, tenant: str, case_id: str, seq: int) -> str:
        return self._c(validate_segment(tenant, "tenant"), "cases",
                       validate_case_id(case_id), "audit", f"{seq:08d}.json")

    def audit_head_uri(self, tenant: str, case_id: str) -> str:
        return self._c(validate_segment(tenant, "tenant"), "cases",
                       validate_case_id(case_id), "audit", "_head.json")

    def audit_prefix(self, tenant: str, case_id: str) -> str:
        return self._c(validate_segment(tenant, "tenant"), "cases",
                       validate_case_id(case_id), "audit")

    def case_dir(self, tenant: str, case_id: str) -> Path:
        """The filesystem sandbox for one case. Created on demand."""
        p = sandbox_path(self.sandbox,
                         validate_segment(tenant, "tenant"),
                         validate_case_id(case_id))
        p.mkdir(parents=True, exist_ok=True)
        return p

    def artefact_dir(self, tenant: str, case_id: str) -> Path:
        p = sandbox_path(self.case_dir(tenant, case_id), "artefacts")
        p.mkdir(parents=True, exist_ok=True)
        return p

    def run_dir(self, tenant: str, case_id: str) -> Path:
        p = sandbox_path(self.case_dir(tenant, case_id), "run")
        p.mkdir(parents=True, exist_ok=True)
        return p

    def package_dir(self, tenant: str, case_id: str) -> Path:
        p = sandbox_path(self.case_dir(tenant, case_id), "readiness")
        p.mkdir(parents=True, exist_ok=True)
        return p

    def artefact_path(self, tenant: str, case_id: str, filename: str) -> Path:
        """A path for one artefact file. Refuses anything outside the sandbox."""
        return sandbox_path(self.artefact_dir(tenant, case_id), filename)

    # ------------------------------------------------------------------ #
    # Cases
    # ------------------------------------------------------------------ #
    def save(self, case: SyntheticCase) -> SyntheticCase:
        case.validate()
        if case.runtime_mode != RUNTIME_MODE_SYNTHETIC:  # pragma: no cover
            raise StoreMisconfigured("case is not synthetic")
        case.updated_at = now_iso()
        case.case_version += 1
        _write_json(self.storage, self.case_uri(case.tenant, case.case_id),
                    case.to_dict())
        self._index(case)
        return case

    def load(self, tenant: str, case_id: str) -> SyntheticCase:
        doc = _read_json(self.storage, self.case_uri(tenant, case_id))
        if not doc:
            raise CaseNotFound()
        case = SyntheticCase.from_dict(doc)
        # Defence in depth against a document moved between tenant folders.
        if case.tenant != tenant:
            raise CaseNotFound()
        return case

    def exists(self, tenant: str, case_id: str) -> bool:
        return self.storage.exists(self.case_uri(tenant, case_id))

    def _index(self, case: SyntheticCase) -> None:
        uri = self.index_uri(case.tenant)
        doc = _read_json(self.storage, uri) or {"cases": {}}
        doc["cases"][case.case_id] = case.summary_row()
        _write_json(self.storage, uri, doc)

    def list_cases(self, tenant: str, *,
                   state: Optional[str] = None) -> List[Dict[str, Any]]:
        doc = _read_json(self.storage, self.index_uri(tenant)) or {}
        rows = list((doc.get("cases") or {}).values())
        if state:
            rows = [r for r in rows if r.get("state") == state]
        return sorted(rows, key=lambda r: r.get("created_at", ""), reverse=True)

    # ------------------------------------------------------------------ #
    # Audit (hash-chained, per case)
    # ------------------------------------------------------------------ #
    def append_audit(self, tenant: str, case_id: str, *, action: str,
                     actor_type: str = ACTOR_SYSTEM, actor_identity: str = "",
                     prior_state: str = "", resulting_state: str = "",
                     input_reference: str = "", output_reference: str = "",
                     decision_basis: str = "",
                     execution_classification: str = EXEC_DETERMINISTIC,
                     detail: Optional[Dict[str, Any]] = None) -> CaseAuditEvent:
        head = _read_json(self.storage,
                          self.audit_head_uri(tenant, case_id)) or {}
        seq = int(head.get("seq") or 0) + 1
        event = CaseAuditEvent(
            event_id=new_id("cae"), case_id=case_id, at=now_iso(),
            actor_type=actor_type, actor_identity=actor_identity,
            action=action, prior_state=prior_state,
            resulting_state=resulting_state, input_reference=input_reference,
            output_reference=output_reference, decision_basis=decision_basis,
            runtime_mode=RUNTIME_MODE_SYNTHETIC,
            execution_classification=execution_classification,
            detail=detail or {}, prev_hash=head.get("record_hash") or "")
        event.record_hash = self._hash(event)
        _write_json(self.storage, self.audit_uri(tenant, case_id, seq),
                    {"seq": seq, **event.to_dict()})
        _write_json(self.storage, self.audit_head_uri(tenant, case_id),
                    {"seq": seq, "record_hash": event.record_hash})
        return event

    @staticmethod
    def _hash(event: CaseAuditEvent) -> str:
        d = event.to_dict()
        return stable_hash(canonical_json(
            {k: d[k] for k in ("event_id", "case_id", "at", "actor_type",
                               "actor_identity", "action", "prior_state",
                               "resulting_state", "input_reference",
                               "output_reference", "decision_basis",
                               "runtime_mode", "execution_classification",
                               "detail", "prev_hash")}))

    def list_audit(self, tenant: str, case_id: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for uri in self.storage.list(self.audit_prefix(tenant, case_id)):
            if uri.endswith("_head.json") or not uri.endswith(".json"):
                continue
            doc = _read_json(self.storage, uri)
            if doc:
                out.append(doc)
        return sorted(out, key=lambda e: e.get("seq", 0))

    def verify_audit_chain(self, tenant: str, case_id: str) -> bool:
        prev = ""
        for rec in self.list_audit(tenant, case_id):
            if rec.get("prev_hash") != prev:
                return False
            expected = self._hash(CaseAuditEvent.from_dict(rec))
            if rec.get("record_hash") != expected:
                return False
            prev = rec["record_hash"]
        return True

    # ------------------------------------------------------------------ #
    # Sandbox file IO (the only supported way to write case files)
    # ------------------------------------------------------------------ #
    def write_artefact_bytes(self, tenant: str, case_id: str, filename: str,
                             data: bytes) -> Path:
        path = self.artefact_path(tenant, case_id, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def relative(self, tenant: str, case_id: str, path: Path) -> str:
        """A case-relative label for a sandbox path.

        Recorded on the case in place of an absolute path, so a case document
        never carries a machine path and cannot be replayed against a different
        filesystem.
        """
        base = self.case_dir(tenant, case_id)
        try:
            rel = Path(path).resolve().relative_to(base)
        except ValueError:
            return Path(path).name
        return f"synthetic_cases/{case_id}/{rel.as_posix()}"
