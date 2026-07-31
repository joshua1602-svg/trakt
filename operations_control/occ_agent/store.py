"""operations_control.occ_agent.store — synthetic persistence.

The OCC Agent does not own an onboarding case model. It drives the platform's
own :class:`operations_control.onboarding.service.OnboardingService`, and this
module's whole job is to give that service — and the synthetic execution that
follows it — somewhere isolated to live.

Three things:

* :func:`synthetic_ops_store` — an :class:`~operations_control.stores.OpsStore`
  pinned to the synthetic container. Handed to ``OnboardingService``, so every
  case, version and artefact it writes lands in ``operations-control-synthetic``
  and never in the live ``operations-control``. The store refuses a container
  that is (or matches) the live one, so a mis-set environment variable cannot
  put a practice case among real ones.
* :class:`SyntheticRunStore` — the execution record that sits *beside* an
  onboarding case: stage outcomes, mapping report, open decisions, the
  orchestration plan and the readiness verdict. Client Onboarding stops at
  activation and knows nothing about pipeline execution, so this is additive
  rather than a parallel case model. It is keyed by the onboarding case's own
  reference.
* the **sandbox** — a filesystem root, one directory per tenant and case, where
  artefact bytes and run working files live. Reachable only through
  :func:`~operations_control.occ_agent.policy.sandbox_path`, which refuses
  traversal, absolute components and symlink escapes.

The audit trail is hash-chained with the same helper the live store uses, so a
synthetic run's history is tamper-evident in the way a live one's is.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from apps.blob_trigger_app.storage import Storage, join_uri

from ..contracts import canonical_json, new_id, now_iso, stable_hash
from ..engine import OpsError
from ..stores import DEFAULT_OPS_CONTAINER, OpsLayout, OpsStore, _read_json, _write_json
from .policy import (
    RUNTIME_MODE_SYNTHETIC,
    sandbox_path,
    validate_segment,
)
from .run import ACTOR_SYSTEM, EXEC_DETERMINISTIC, RunAuditEvent, SyntheticRun

CONTAINER_ENV = "TRAKT_OCC_AGENT_SYNTHETIC_CONTAINER"
DEFAULT_SYNTHETIC_CONTAINER = "operations-control-synthetic"

SANDBOX_ROOT_ENV = "TRAKT_OCC_AGENT_SANDBOX_ROOT"
DEFAULT_SANDBOX_ROOT = ".occ-agent-synthetic"

#: An onboarding case reference, e.g. ``ONB-2026-0002``. Validated before it is
#: ever used as a path segment.
import re
CASE_REF_RE = re.compile(r"^[A-Z]{2,8}-\d{4}-\d{4,8}$")


class RunNotFound(OpsError):
    """No synthetic run for this case.

    404 rather than 403, matching the live routes: another tenant's case must
    not be revealed to exist.
    """

    def __init__(self) -> None:
        super().__init__("OCC_AGENT_RUN_NOT_FOUND",
                         "That practice case could not be found.",
                         http_status=404)


class StoreMisconfigured(OpsError):
    """The synthetic store was pointed at the live container."""

    def __init__(self, detail: str) -> None:
        super().__init__(
            "OCC_AGENT_STORE_MISCONFIGURED",
            "The practice environment is not configured safely. Ask your "
            f"administrator. ({detail})", http_status=503)


def synthetic_container() -> str:
    name = (os.environ.get(CONTAINER_ENV) or DEFAULT_SYNTHETIC_CONTAINER).strip()
    assert_isolated(name)
    return name


def assert_isolated(container: str) -> None:
    """Refuse any container that could collide with live operational state."""
    if not container:
        raise StoreMisconfigured("no container name")
    if container == DEFAULT_OPS_CONTAINER:
        raise StoreMisconfigured("practice cases may not share the live "
                                 "operations container")
    live = os.environ.get("TRAKT_OPS_CONTAINER", DEFAULT_OPS_CONTAINER).strip()
    if live and container == live:
        raise StoreMisconfigured("practice cases may not share the live "
                                 "operations container")
    if "/" in container or container.startswith("."):
        raise StoreMisconfigured("container name")


def sandbox_root() -> Path:
    return Path(os.environ.get(SANDBOX_ROOT_ENV) or DEFAULT_SANDBOX_ROOT)


def validate_case_ref(case_ref: str) -> str:
    if not CASE_REF_RE.match(str(case_ref or "")):
        from .policy import UnsafePathError
        raise UnsafePathError("case reference")
    return case_ref


def synthetic_ops_store(storage: Storage,
                        container: Optional[str] = None) -> OpsStore:
    """An OpsStore pinned to the synthetic container.

    This is what makes reusing ``OnboardingService`` safe: it is constructed
    with an ``OpsStore``, so pinning the layout here sends every case it writes
    — and every artefact activation would write — into the synthetic container.
    """
    name = container or synthetic_container()
    assert_isolated(name)
    return OpsStore(storage, OpsLayout(container=name))


class SyntheticRunStore:
    """The execution record that sits beside an onboarding case.

    Layout, mirroring the live store's shape so a later migration is
    mechanical::

        blob://operations-control-synthetic/
          {tenant}/agent-runs/{case_ref}/run.json
          {tenant}/agent-runs/{case_ref}/audit/{00000001}.json
          {tenant}/agent-runs/{case_ref}/audit/_head.json
          {tenant}/agent-runs/index.json
    """

    def __init__(self, storage: Storage, *, container: Optional[str] = None,
                 sandbox: Optional[Path] = None):
        self.storage = storage
        self.container = container or synthetic_container()
        assert_isolated(self.container)
        self.sandbox = Path(sandbox or sandbox_root())

    # ------------------------------------------------------------------ #
    # URIs and sandbox paths
    # ------------------------------------------------------------------ #
    def _c(self, *parts: str) -> str:
        return join_uri(f"blob://{self.container}", *parts)

    def run_uri(self, tenant: str, case_ref: str) -> str:
        return self._c(validate_segment(tenant, "tenant"), "agent-runs",
                       validate_case_ref(case_ref), "run.json")

    def index_uri(self, tenant: str) -> str:
        return self._c(validate_segment(tenant, "tenant"), "agent-runs",
                       "index.json")

    def audit_uri(self, tenant: str, case_ref: str, seq: int) -> str:
        return self._c(validate_segment(tenant, "tenant"), "agent-runs",
                       validate_case_ref(case_ref), "audit", f"{seq:08d}.json")

    def audit_head_uri(self, tenant: str, case_ref: str) -> str:
        return self._c(validate_segment(tenant, "tenant"), "agent-runs",
                       validate_case_ref(case_ref), "audit", "_head.json")

    def audit_prefix(self, tenant: str, case_ref: str) -> str:
        return self._c(validate_segment(tenant, "tenant"), "agent-runs",
                       validate_case_ref(case_ref), "audit")

    def case_dir(self, tenant: str, case_ref: str) -> Path:
        p = sandbox_path(self.sandbox, validate_segment(tenant, "tenant"),
                         validate_case_ref(case_ref))
        p.mkdir(parents=True, exist_ok=True)
        return p

    def artefact_dir(self, tenant: str, case_ref: str) -> Path:
        p = sandbox_path(self.case_dir(tenant, case_ref), "artefacts")
        p.mkdir(parents=True, exist_ok=True)
        return p

    def run_dir(self, tenant: str, case_ref: str) -> Path:
        p = sandbox_path(self.case_dir(tenant, case_ref), "run")
        p.mkdir(parents=True, exist_ok=True)
        return p

    def package_dir(self, tenant: str, case_ref: str) -> Path:
        p = sandbox_path(self.case_dir(tenant, case_ref), "readiness")
        p.mkdir(parents=True, exist_ok=True)
        return p

    def artefact_path(self, tenant: str, case_ref: str, filename: str) -> Path:
        return sandbox_path(self.artefact_dir(tenant, case_ref), filename)

    # ------------------------------------------------------------------ #
    # The run record
    # ------------------------------------------------------------------ #
    def save(self, run: SyntheticRun) -> SyntheticRun:
        run.validate()
        if run.runtime_mode != RUNTIME_MODE_SYNTHETIC:  # pragma: no cover
            raise StoreMisconfigured("run is not synthetic")
        run.updated_at = now_iso()
        run.version += 1
        _write_json(self.storage, self.run_uri(run.tenant, run.case_ref),
                    run.to_dict())
        self._index(run)
        return run

    def load(self, tenant: str, case_ref: str) -> SyntheticRun:
        doc = _read_json(self.storage, self.run_uri(tenant, case_ref))
        if not doc:
            raise RunNotFound()
        run = SyntheticRun.from_dict(doc)
        if run.tenant != tenant:      # a document moved between tenant folders
            raise RunNotFound()
        return run

    def exists(self, tenant: str, case_ref: str) -> bool:
        return self.storage.exists(self.run_uri(tenant, case_ref))

    def _index(self, run: SyntheticRun) -> None:
        uri = self.index_uri(run.tenant)
        doc = _read_json(self.storage, uri) or {"runs": {}}
        doc["runs"][run.case_ref] = run.summary_row()
        _write_json(self.storage, uri, doc)

    def list_runs(self, tenant: str) -> List[Dict[str, Any]]:
        doc = _read_json(self.storage, self.index_uri(tenant)) or {}
        rows = list((doc.get("runs") or {}).values())
        return sorted(rows, key=lambda r: r.get("created_at", ""), reverse=True)

    # ------------------------------------------------------------------ #
    # Audit (hash-chained, per case)
    # ------------------------------------------------------------------ #
    def append_audit(self, tenant: str, case_ref: str, *, action: str,
                     actor_type: str = ACTOR_SYSTEM, actor_identity: str = "",
                     prior_state: str = "", resulting_state: str = "",
                     input_reference: str = "", output_reference: str = "",
                     decision_basis: str = "",
                     execution_classification: str = EXEC_DETERMINISTIC,
                     detail: Optional[Dict[str, Any]] = None) -> RunAuditEvent:
        head = _read_json(self.storage,
                          self.audit_head_uri(tenant, case_ref)) or {}
        seq = int(head.get("seq") or 0) + 1
        event = RunAuditEvent(
            event_id=new_id("cae"), case_ref=case_ref, at=now_iso(),
            actor_type=actor_type, actor_identity=actor_identity,
            action=action, prior_state=prior_state,
            resulting_state=resulting_state, input_reference=input_reference,
            output_reference=output_reference, decision_basis=decision_basis,
            runtime_mode=RUNTIME_MODE_SYNTHETIC,
            execution_classification=execution_classification,
            detail=detail or {}, prev_hash=head.get("record_hash") or "")
        event.record_hash = self._hash(event)
        _write_json(self.storage, self.audit_uri(tenant, case_ref, seq),
                    {"seq": seq, **event.to_dict()})
        _write_json(self.storage, self.audit_head_uri(tenant, case_ref),
                    {"seq": seq, "record_hash": event.record_hash})
        return event

    @staticmethod
    def _hash(event: RunAuditEvent) -> str:
        d = event.to_dict()
        return stable_hash(canonical_json(
            {k: d[k] for k in ("event_id", "case_ref", "at", "actor_type",
                               "actor_identity", "action", "prior_state",
                               "resulting_state", "input_reference",
                               "output_reference", "decision_basis",
                               "runtime_mode", "execution_classification",
                               "detail", "prev_hash")}))

    def list_audit(self, tenant: str, case_ref: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for uri in self.storage.list(self.audit_prefix(tenant, case_ref)):
            if uri.endswith("_head.json") or not uri.endswith(".json"):
                continue
            doc = _read_json(self.storage, uri)
            if doc:
                out.append(doc)
        return sorted(out, key=lambda e: e.get("seq", 0))

    def verify_audit_chain(self, tenant: str, case_ref: str) -> bool:
        prev = ""
        for rec in self.list_audit(tenant, case_ref):
            if rec.get("prev_hash") != prev:
                return False
            if rec.get("record_hash") != self._hash(RunAuditEvent.from_dict(rec)):
                return False
            prev = rec["record_hash"]
        return True

    # ------------------------------------------------------------------ #
    # Sandbox file IO (the only supported way to write case files)
    # ------------------------------------------------------------------ #
    def write_artefact_bytes(self, tenant: str, case_ref: str, filename: str,
                             data: bytes) -> Path:
        path = self.artefact_path(tenant, case_ref, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def relative(self, tenant: str, case_ref: str, path: Path) -> str:
        """A case-relative label for a sandbox path.

        Recorded in place of an absolute path, so a run document never carries
        a machine path and cannot be replayed against a different filesystem.
        """
        base = self.case_dir(tenant, case_ref)
        try:
            rel = Path(path).resolve().relative_to(base)
        except ValueError:
            return Path(path).name
        return f"practice_cases/{case_ref}/{rel.as_posix()}"
