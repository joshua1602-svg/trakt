"""operations_control.onboarding.store — versioned standing configuration.

Every approved change writes a NEW version; prior versions are never mutated
and never deleted. Each version carries who made it, when, why, the profile
before, the profile after, a field-level change list and the artefacts that were
generated from it — so "what did this client's configuration look like in March"
is answerable from the container alone.

Storage follows the existing operations-control conventions
(:mod:`operations_control.stores`): one prefix per client, small pointer
documents, everything else append-only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from ..contracts import canonical_json, now_iso, stable_hash
from ..stores import OpsStore, _read_json, _write_json
from .model import (
    PROFILE_ACTIVE,
    PROFILE_DRAFT,
    PROFILE_SUPERSEDED,
    ClientOnboardingProfile,
)


@dataclass
class ProfileVersion:
    """One immutable version of a client's standing configuration."""

    client_id: str
    version: int
    status: str                              # draft | active | superseded
    profile: Dict[str, Any]
    created_by: str = ""
    created_at: str = ""
    approved_by: str = ""
    approved_at: str = ""
    reason: str = ""
    based_on_version: Optional[int] = None
    #: Field-level before/after, computed at approval against the prior version.
    changes: List[Dict[str, Any]] = field(default_factory=list)
    #: What was written: [{"kind", "target", "action", "detail"}]
    artefacts: List[Dict[str, Any]] = field(default_factory=list)
    content_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, doc: Dict[str, Any]) -> "ProfileVersion":
        known = {f for f in cls.__dataclass_fields__}   # type: ignore[attr-defined]
        return cls(**{k: v for k, v in doc.items() if k in known})

    def as_profile(self) -> ClientOnboardingProfile:
        return ClientOnboardingProfile.from_dict(self.profile)


class OnboardingStore:
    """Persistence for onboarding profiles and their generated artefacts."""

    def __init__(self, store: OpsStore):
        self.store = store

    # -- paths -------------------------------------------------------------- #
    def _base(self, client_id: str) -> str:
        return (f"blob://{self.store.layout.container}/{client_id}/onboarding")

    def _version_uri(self, client_id: str, version: int) -> str:
        return f"{self._base(client_id)}/versions/{version:04d}.json"

    def _current_uri(self, client_id: str) -> str:
        return f"{self._base(client_id)}/current.json"

    def _draft_uri(self, client_id: str, draft_id: str) -> str:
        return f"{self._base(client_id)}/drafts/{draft_id}.json"

    def _drafts_prefix(self, client_id: str) -> str:
        return f"{self._base(client_id)}/drafts"

    def _artefact_uri(self, client_id: str, version: int, rel: str) -> str:
        return f"{self._base(client_id)}/artefacts/{version:04d}/{rel}"

    def _artefact_current_uri(self, client_id: str, rel: str) -> str:
        return f"{self._base(client_id)}/artefacts/current/{rel}"

    # -- index -------------------------------------------------------------- #
    def _index_uri(self) -> str:
        return (f"blob://{self.store.layout.container}/_index/"
                "onboarded_clients.json")

    def onboarded_clients(self) -> List[str]:
        doc = _read_json(self.store.storage, self._index_uri()) or {}
        return sorted(doc.get("clients") or [])

    def _register(self, client_id: str) -> None:
        uri = self._index_uri()
        doc = _read_json(self.store.storage, uri) or {"clients": []}
        if client_id not in doc["clients"]:
            doc["clients"].append(client_id)
            _write_json(self.store.storage, uri, doc)

    # -- drafts ------------------------------------------------------------- #
    def save_draft(self, client_id: str, draft: Dict[str, Any]) -> Dict[str, Any]:
        draft["updated_at"] = now_iso()
        _write_json(self.store.storage,
                    self._draft_uri(client_id, draft["draft_id"]), draft)
        return draft

    def load_draft(self, client_id: str,
                   draft_id: str) -> Optional[Dict[str, Any]]:
        return _read_json(self.store.storage,
                          self._draft_uri(client_id, draft_id))

    def list_drafts(self, client_id: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for uri in self.store.storage.list(self._drafts_prefix(client_id)):
            if not uri.endswith(".json"):
                continue
            doc = _read_json(self.store.storage, uri)
            if doc and doc.get("status") == PROFILE_DRAFT:
                out.append(doc)
        return sorted(out, key=lambda d: d.get("updated_at") or "",
                      reverse=True)

    def discard_draft(self, client_id: str, draft_id: str,
                      *, by: str, reason: str = "") -> None:
        doc = self.load_draft(client_id, draft_id)
        if doc is None:
            return
        doc.update(status="discarded", discarded_by=by, discarded_at=now_iso(),
                   discard_reason=reason)
        _write_json(self.store.storage, self._draft_uri(client_id, draft_id),
                    doc)

    # -- versions ----------------------------------------------------------- #
    def current(self, client_id: str) -> Optional[ProfileVersion]:
        ptr = _read_json(self.store.storage, self._current_uri(client_id))
        if not ptr:
            return None
        return self.version(client_id, int(ptr["version"]))

    def version(self, client_id: str,
                version: int) -> Optional[ProfileVersion]:
        doc = _read_json(self.store.storage,
                         self._version_uri(client_id, version))
        return ProfileVersion.from_dict(doc) if doc else None

    def history(self, client_id: str) -> List[ProfileVersion]:
        out: List[ProfileVersion] = []
        for uri in self.store.storage.list(
                f"{self._base(client_id)}/versions"):
            if not uri.endswith(".json"):
                continue
            doc = _read_json(self.store.storage, uri)
            if doc:
                out.append(ProfileVersion.from_dict(doc))
        return sorted(out, key=lambda v: v.version, reverse=True)

    def next_version(self, client_id: str) -> int:
        versions = self.history(client_id)
        return (max(v.version for v in versions) + 1) if versions else 1

    def commit(self, *, client_id: str, profile: ClientOnboardingProfile,
               approved_by: str, reason: str,
               changes: List[Dict[str, Any]],
               artefacts: List[Dict[str, Any]],
               based_on_version: Optional[int] = None) -> ProfileVersion:
        """Write a new immutable version and make it current.

        The previously current version is marked superseded IN PLACE only in its
        status field — its profile content is never rewritten, so a reader can
        still reconstruct exactly what was in force at the time.
        """
        version = self.next_version(client_id)
        doc = profile.to_dict()
        record = ProfileVersion(
            client_id=client_id, version=version, status=PROFILE_ACTIVE,
            profile=doc, created_by=approved_by, created_at=now_iso(),
            approved_by=approved_by, approved_at=now_iso(), reason=reason,
            based_on_version=based_on_version, changes=changes,
            artefacts=artefacts,
            content_hash=stable_hash(canonical_json(doc)))
        previous = self.current(client_id)
        _write_json(self.store.storage,
                    self._version_uri(client_id, version), record.to_dict())
        if previous is not None and previous.version != version:
            previous.status = PROFILE_SUPERSEDED
            _write_json(self.store.storage,
                        self._version_uri(client_id, previous.version),
                        previous.to_dict())
        _write_json(self.store.storage, self._current_uri(client_id),
                    {"version": version, "content_hash": record.content_hash,
                     "approved_by": approved_by,
                     "approved_at": record.approved_at})
        self._register(client_id)
        self.store.register_client(client_id)
        return record

    # -- generated artefacts ------------------------------------------------ #
    def write_artefact(self, client_id: str, version: int, rel: str,
                       text: str) -> str:
        """Persist one generated artefact for a version, and as current.

        ``rel`` is the artefact's repository-relative path (for example
        ``config/client/config_client_ERE.yaml``) so a reader can see at a
        glance which governed file a generated document corresponds to.
        """
        uri = self._artefact_uri(client_id, version, rel)
        self.store.storage.write_text(uri, text)
        self.store.storage.write_text(
            self._artefact_current_uri(client_id, rel), text)
        return uri

    def read_artefact(self, client_id: str, rel: str,
                      version: Optional[int] = None) -> Optional[str]:
        uri = (self._artefact_uri(client_id, version, rel) if version
               else self._artefact_current_uri(client_id, rel))
        if not self.store.storage.exists(uri):
            return None
        return self.store.storage.read_text(uri)

    def artefact_uri(self, client_id: str, rel: str,
                     version: Optional[int] = None) -> str:
        return (self._artefact_uri(client_id, version, rel) if version
                else self._artefact_current_uri(client_id, rel))
