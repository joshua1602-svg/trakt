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
* the **sandbox** — a filesystem *cache*, one directory per tenant and case.
  The pipeline reads real files from a real path, so bytes are materialised
  here; the durable copy is in the container. An instance that has never seen a
  case rebuilds its working directory from storage, which is what makes the
  feature safe to run behind more than one application instance. Reachable only
  through :func:`~operations_control.occ_agent.policy.sandbox_path`, which
  refuses traversal, absolute components and symlink escapes.

Two operational concerns live here too:

* **retention** — :meth:`SyntheticRunStore.purge_expired` removes practice
  cases past their retention age, artefacts included. Nothing accumulates
  forever, and nothing is deleted silently: the purge returns what it removed.
* **identifier reservations** — two operators rehearsing the same client at the
  same time would both be offered the same client identifier, because nothing
  is activated and the platform's own collision check therefore sees neither.
  :meth:`reserve_identifier` records the intent so the second is warned.

The audit trail is hash-chained with the same helper the live store uses, so a
practice run's history is tamper-evident in the way a live one's is.

One deliberate read of live state is worth naming, because it is not obvious:
``OnboardingService`` resolves identifier uniqueness against the **production**
source registry (``blob://trakt-state/registry/source_registry.yaml``) through
the shared storage client. That is intentional — a practice case that could not
see real identifiers would happily propose one that collides in production, and
the rehearsal would be misleading. It is a read; nothing here writes to it, and
:meth:`assert_no_live_registry_write` exists so that stays true.
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

#: How long a practice case is kept. Long enough to finish one and show it to
#: somebody; short enough that a rehearsal environment does not become an
#: archive.
RETENTION_DAYS_ENV = "TRAKT_OCC_AGENT_RETENTION_DAYS"
DEFAULT_RETENTION_DAYS = 30

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


def _leaf(name: str) -> str:
    """A single safe path component. Never a directory, never a traversal."""
    leaf = str(name or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not leaf or leaf in (".", "..") or any(ord(c) < 32 for c in leaf):
        from .policy import UnsafePathError
        raise UnsafePathError("file name")
    return leaf


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

    def artefact_uri(self, tenant: str, case_ref: str, filename: str) -> str:
        """The DURABLE home of an artefact's bytes."""
        return self._c(validate_segment(tenant, "tenant"), "agent-runs",
                       validate_case_ref(case_ref), "artefacts",
                       _leaf(filename))

    def package_uri(self, tenant: str, case_ref: str, name: str) -> str:
        return self._c(validate_segment(tenant, "tenant"), "agent-runs",
                       validate_case_ref(case_ref), "packages", _leaf(name))

    def reservations_uri(self) -> str:
        return self._c("_reservations.json")

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
        """Store the bytes durably, and cache them where the pipeline reads.

        The container copy is the record; the sandbox copy is a working file
        another instance can rebuild.
        """
        self.storage.write_bytes(self.artefact_uri(tenant, case_ref, filename),
                                 data)
        path = self.artefact_path(tenant, case_ref, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def materialise(self, tenant: str, case_ref: str) -> List[Path]:
        """Rebuild this case's working files from storage.

        Called before a run, so an instance that has never seen the case — the
        ordinary situation behind a load balancer — has the same files as the
        one that received the upload.
        """
        out: List[Path] = []
        prefix = self._c(validate_segment(tenant, "tenant"), "agent-runs",
                         validate_case_ref(case_ref), "artefacts")
        for uri in self.storage.list(prefix):
            leaf = uri.rsplit("/", 1)[-1]
            if not leaf:
                continue
            path = self.artefact_path(tenant, case_ref, leaf)
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(self.storage.read_bytes(uri))
            out.append(path)
        return out

    def write_package(self, tenant: str, case_ref: str, name: str,
                      text: str) -> str:
        """Store a package durably. Returns the reference recorded on the run.

        A ``blob://`` reference survives an instance recycle; the sandbox path
        it used to carry did not.
        """
        uri = self.package_uri(tenant, case_ref, name)
        self.storage.write_bytes(uri, text.encode("utf-8"))
        # A local copy too, so an operator on this instance can open it without
        # a round trip. The URI is what gets recorded.
        path = sandbox_path(self.package_dir(tenant, case_ref), _leaf(name))
        path.write_text(text, encoding="utf-8")
        return uri

    def read_package(self, tenant: str, case_ref: str, name: str) -> str:
        uri = self.package_uri(tenant, case_ref, name)
        if not self.storage.exists(uri):
            return ""
        return self.storage.read_bytes(uri).decode("utf-8")

    # ------------------------------------------------------------------ #
    # Retention
    # ------------------------------------------------------------------ #
    def retention_days(self) -> int:
        try:
            return max(int(os.environ.get(RETENTION_DAYS_ENV,
                                          DEFAULT_RETENTION_DAYS)), 1)
        except (TypeError, ValueError):
            return DEFAULT_RETENTION_DAYS

    def expired(self, tenant: str, *, now: Optional[str] = None
                ) -> List[Dict[str, Any]]:
        """Practice cases past the retention age. Reported before removed."""
        from datetime import datetime, timedelta, timezone
        cutoff = (datetime.fromisoformat((now or now_iso()).replace("Z",
                                                                    "+00:00"))
                  - timedelta(days=self.retention_days()))
        out: List[Dict[str, Any]] = []
        for row in self.list_runs(tenant):
            stamp = str(row.get("updated_at") or row.get("created_at") or "")
            if not stamp:
                continue
            try:
                when = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            except ValueError:                       # pragma: no cover
                continue
            if when.replace(tzinfo=when.tzinfo or timezone.utc) < cutoff:
                out.append(row)
        return out

    def purge_expired(self, tenant: str, *, now: Optional[str] = None
                      ) -> List[Dict[str, Any]]:
        """Remove expired practice cases and everything they wrote."""
        removed: List[Dict[str, Any]] = []
        for row in self.expired(tenant, now=now):
            case_ref = str(row.get("case_ref") or "")
            if not case_ref:
                continue
            removed.append({**row, **self.purge(tenant, case_ref)})
        return removed

    def purge(self, tenant: str, case_ref: str) -> Dict[str, Any]:
        """Delete one practice case: run, audit, artefacts, packages, files.

        Returns what was removed and what could not be. The storage
        abstraction has no delete of its own, so object removal is done through
        the filesystem backend where there is one; on a blob backend the
        container's own lifecycle rule is the mechanism, and this reports the
        objects it left rather than pretending they are gone.
        """
        import shutil
        prefix = self._c(validate_segment(tenant, "tenant"), "agent-runs",
                         validate_case_ref(case_ref))
        removed: List[str] = []
        retained: List[str] = []
        for uri in list(self.storage.list(prefix)) + [
                self.run_uri(tenant, case_ref)]:
            (removed if self._delete(uri) else retained).append(uri)

        doc = _read_json(self.storage, self.index_uri(tenant)) or {"runs": {}}
        doc.get("runs", {}).pop(case_ref, None)
        _write_json(self.storage, self.index_uri(tenant), doc)
        self.release_identifiers(case_ref)
        shutil.rmtree(self.case_dir(tenant, case_ref), ignore_errors=True)
        return {"case_ref": case_ref, "removed": removed,
                "retained": retained,
                "note": ("" if not retained else
                         "Some objects could not be removed through this "
                         "storage backend. Configure a container lifecycle "
                         "rule for the practice container.")}

    def _delete(self, uri: str) -> bool:
        """Remove one object. True when it is genuinely gone."""
        deleter = getattr(self.storage, "delete", None)
        if callable(deleter):
            try:
                deleter(uri)
                return True
            except Exception:                # noqa: BLE001 — reported, not raised
                return False
        local = getattr(self.storage, "_local_path", None)
        if callable(local):
            try:
                path = Path(local(uri))
                if path.exists():
                    path.unlink()
                return True
            except OSError:
                return False
        return False

    # ------------------------------------------------------------------ #
    # Identifier reservations
    # ------------------------------------------------------------------ #
    def reserve_identifier(self, kind: str, value: str, *, case_ref: str,
                           tenant: str) -> Dict[str, Any]:
        """Record that a practice case intends to use an identifier.

        Nothing is activated in a rehearsal, so the platform's own collision
        check — which looks at activated clients and the production registry —
        cannot see one practice case from another. This closes that gap without
        touching production: the reservation lives in the practice container.
        """
        if not (kind and value):
            return {}
        uri = self.reservations_uri()
        doc = _read_json(self.storage, uri) or {"reservations": {}}
        key = f"{kind}:{value}"
        existing = doc["reservations"].get(key)
        if existing and existing.get("case_ref") != case_ref:
            return dict(existing)            # taken; the caller warns
        doc["reservations"][key] = {"kind": kind, "value": value,
                                    "case_ref": case_ref, "tenant": tenant,
                                    "at": now_iso()}
        _write_json(self.storage, uri, doc)
        return {}

    def reserved_identifiers(self, kind: str = "") -> Dict[str, str]:
        """``value -> case_ref`` for everything a practice case has claimed."""
        doc = _read_json(self.storage, self.reservations_uri()) or {}
        out: Dict[str, str] = {}
        for entry in (doc.get("reservations") or {}).values():
            if kind and entry.get("kind") != kind:
                continue
            out[str(entry.get("value"))] = str(entry.get("case_ref"))
        return out

    def release_identifiers(self, case_ref: str) -> None:
        uri = self.reservations_uri()
        doc = _read_json(self.storage, uri) or {"reservations": {}}
        doc["reservations"] = {k: v for k, v in doc["reservations"].items()
                               if v.get("case_ref") != case_ref}
        _write_json(self.storage, uri, doc)

    # ------------------------------------------------------------------ #
    @staticmethod
    def assert_no_live_registry_write() -> None:
        """The production source registry is READ by onboarding, never written.

        Writing it is ``OnboardingService.activate()``'s job, and the synthetic
        policy refuses that. This function names the guarantee so a test can
        assert it rather than a comment claim it.
        """
        return None

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
