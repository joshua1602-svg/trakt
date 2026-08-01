"""mi_agent.concentration_tests.store — tenant-scoped governed persistence.

Concentration-test governance state lives in the operations-control container
(the same container, storage abstraction and JSON conventions every other OCC
store uses), partitioned by client:

    blob://{container}/{client_id}/concentration-tests/proposals/{proposal_id}.json
    blob://{container}/{client_id}/concentration-tests/proposals/index.json
    blob://{container}/{client_id}/concentration-tests/versions/{0001}.json
    blob://{container}/{client_id}/concentration-tests/current.json

Versions are immutable: activation writes a NEW version document, flips the
previous one to ``superseded`` and rewrites the ``current`` pointer. A repeat
activation with identical content returns the existing version unchanged
(content-hash idempotency — the same guarantee ``OnboardingStore.commit``
gives onboarding configuration). Proposal documents keep their own history in
``updated_at``-stamped rewrites plus the operator notes they accumulate; the
tamper-evident audit trail for decisions is appended by the OCC service through
``OpsStore.append_audit``, not here.

This module is import-safe without Azure: the storage abstraction is imported
lazily and a ``Storage`` instance can be injected (tests use the file backend).
"""

from __future__ import annotations

import datetime as _dt
import os
import re
from typing import Any, Dict, List, Optional

from .models import (
    ActiveConfiguration,
    PROPOSAL_STATUSES,
    TestProposal,
)

DEFAULT_OPS_CONTAINER = "operations-control"
OPS_CONTAINER_ENV = "TRAKT_OPS_CONTAINER"

_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


class StoreUnavailable(RuntimeError):
    """The governed store cannot be reached — callers fail closed."""


def _validate_segment(value: str, what: str) -> str:
    value = str(value or "").strip()
    if not value or not _SAFE_SEGMENT.match(value) or ".." in value:
        raise ValueError(f"{what} is not a valid storage segment: {value!r}")
    return value


def open_default_storage():
    """The shared storage abstraction (blob or file backend, per env)."""
    from apps.blob_trigger_app.storage import open_storage
    return open_storage()


class ConcentrationStore:
    """Persistence for proposals and activated configuration versions."""

    def __init__(self, storage=None, *, container: Optional[str] = None):
        self._storage = storage
        self.container = (container or os.environ.get(OPS_CONTAINER_ENV)
                          or DEFAULT_OPS_CONTAINER)

    # ------------------------------------------------------------------ #
    @property
    def storage(self):
        if self._storage is None:
            self._storage = open_default_storage()
        return self._storage

    def _uri(self, client_id: str, *parts: str) -> str:
        client_id = _validate_segment(client_id, "client_id")
        safe = [str(p) for p in parts]
        return "/".join([f"blob://{self.container}", client_id,
                         "concentration-tests", *safe])

    def _read(self, uri: str) -> Optional[Dict[str, Any]]:
        from operations_control.stores import _read_json
        return _read_json(self.storage, uri)

    def _write(self, uri: str, doc: Dict[str, Any]) -> str:
        from operations_control.stores import _write_json
        return _write_json(self.storage, uri, doc)

    # ------------------------------------------------------------------ #
    # Proposals
    # ------------------------------------------------------------------ #
    def proposal_uri(self, client_id: str, proposal_id: str) -> str:
        proposal_id = _validate_segment(proposal_id, "proposal_id")
        return self._uri(client_id, "proposals", f"{proposal_id}.json")

    def index_uri(self, client_id: str) -> str:
        return self._uri(client_id, "proposals", "index.json")

    def save_proposal(self, client_id: str, proposal: TestProposal
                      ) -> TestProposal:
        if proposal.status not in PROPOSAL_STATUSES:
            raise ValueError(f"unknown proposal status {proposal.status!r}")
        if not proposal.created_at:
            proposal.created_at = now_iso()
        proposal.updated_at = now_iso()
        proposal.client_id = proposal.client_id or client_id
        self._write(self.proposal_uri(client_id, proposal.proposal_id),
                    proposal.to_dict())
        index = self._read(self.index_uri(client_id)) or {"proposal_ids": []}
        if proposal.proposal_id not in index["proposal_ids"]:
            index["proposal_ids"].append(proposal.proposal_id)
            self._write(self.index_uri(client_id), index)
        return proposal

    def get_proposal(self, client_id: str, proposal_id: str
                     ) -> Optional[TestProposal]:
        doc = self._read(self.proposal_uri(client_id, proposal_id))
        return TestProposal.from_dict(doc) if doc else None

    def list_proposals(self, client_id: str) -> List[TestProposal]:
        index = self._read(self.index_uri(client_id)) or {}
        out: List[TestProposal] = []
        for pid in index.get("proposal_ids") or []:
            doc = self._read(self.proposal_uri(client_id, pid))
            if doc:
                out.append(TestProposal.from_dict(doc))
        return out

    # ------------------------------------------------------------------ #
    # Activated configuration versions
    # ------------------------------------------------------------------ #
    def version_uri(self, client_id: str, version: int) -> str:
        return self._uri(client_id, "versions", f"{int(version):04d}.json")

    def current_uri(self, client_id: str) -> str:
        return self._uri(client_id, "current.json")

    def current_configuration(self, client_id: str
                              ) -> Optional[ActiveConfiguration]:
        pointer = self._read(self.current_uri(client_id))
        if not pointer:
            return None
        doc = self._read(self.version_uri(client_id, int(pointer["version"])))
        return ActiveConfiguration.from_dict(doc) if doc else None

    def get_version(self, client_id: str, version: int
                    ) -> Optional[ActiveConfiguration]:
        doc = self._read(self.version_uri(client_id, int(version)))
        return ActiveConfiguration.from_dict(doc) if doc else None

    def list_versions(self, client_id: str) -> List[Dict[str, Any]]:
        """Version headers, oldest first, from the pointer chain."""
        pointer = self._read(self.current_uri(client_id))
        if not pointer:
            return []
        out: List[Dict[str, Any]] = []
        for v in range(1, int(pointer["version"]) + 1):
            doc = self._read(self.version_uri(client_id, v))
            if doc:
                out.append({
                    "version": doc.get("version"),
                    "status": doc.get("status"),
                    "activated_by": doc.get("activated_by"),
                    "activated_at": doc.get("activated_at"),
                    "test_count": len(doc.get("tests") or []),
                    "content_hash": doc.get("content_hash"),
                    "reason": doc.get("reason"),
                })
        return out

    def commit_configuration(self, client_id: str,
                             config: ActiveConfiguration
                             ) -> ActiveConfiguration:
        """Mint the next immutable version; supersede the previous.

        Identical content (by hash) returns the current version unchanged —
        re-activating the same approved set cannot fork history.
        """
        client_id = _validate_segment(client_id, "client_id")
        config.client_id = config.client_id or client_id
        config.content_hash = config.compute_content_hash()

        pointer = self._read(self.current_uri(client_id))
        if pointer:
            current = self._read(self.version_uri(client_id,
                                                  int(pointer["version"])))
            if current and current.get("content_hash") == config.content_hash \
                    and current.get("status") == "active":
                return ActiveConfiguration.from_dict(current)
            next_version = int(pointer["version"]) + 1
            config.based_on_version = int(pointer["version"])
            if current is not None:
                current["status"] = "superseded"
                self._write(self.version_uri(client_id,
                                             int(pointer["version"])), current)
        else:
            next_version = 1
        config.version = next_version
        config.status = "active"
        if not config.activated_at:
            config.activated_at = now_iso()
        self._write(self.version_uri(client_id, next_version), config.to_dict())
        self._write(self.current_uri(client_id),
                    {"version": next_version,
                     "content_hash": config.content_hash,
                     "activated_at": config.activated_at,
                     "schema_version": config.schema_version})
        return config
