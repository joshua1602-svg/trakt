"""operations_control.onboarding.store — cases, versions and generated artefacts.

Two things are persisted.

**Cases** are the work in progress. A case is opened before anyone knows what
the client will be called, so a case lives under its own reference — not under a
client prefix — and is indexed by client once it names one.

**Versions** are the outcome. Every activation writes a NEW immutable version of
a client's standing configuration; earlier versions are never rewritten, so
"what was in force in March" is answerable from the container alone.

Storage follows the existing operations-control conventions: small pointer
documents, everything else append-only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from ..contracts import canonical_json, now_iso, stable_hash
from ..stores import OpsStore, _read_json, _write_json
from .case import (
    ACTIVATED,
    OnboardingCase,
    TERMINAL,
    WITHDRAWN,
    mint_case_id,
)

VERSION_ACTIVE = "active"
VERSION_SUPERSEDED = "superseded"


@dataclass
class ConfigurationVersion:
    """One immutable version of a client's standing configuration."""

    client_id: str
    version: int
    status: str
    #: The case answers exactly as approved.
    answers: Dict[str, Any] = field(default_factory=dict)
    case_id: str = ""
    case_kind: str = ""
    approved_by: str = ""
    approved_at: str = ""
    activated_by: str = ""
    activated_at: str = ""
    reason: str = ""
    based_on_version: Optional[int] = None
    changes: List[Dict[str, Any]] = field(default_factory=list)
    artefacts: List[Dict[str, Any]] = field(default_factory=list)
    content_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, doc: Dict[str, Any]) -> "ConfigurationVersion":
        known = {f for f in cls.__dataclass_fields__}   # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (doc or {}).items() if k in known})


class OnboardingStore:
    def __init__(self, store: OpsStore):
        self.store = store

    # -- paths -------------------------------------------------------------- #
    def _cases_prefix(self) -> str:
        return f"blob://{self.store.layout.container}/_onboarding/cases"

    def _case_uri(self, case_id: str) -> str:
        return f"{self._cases_prefix()}/{case_id}.json"

    def _sequence_uri(self, year: int) -> str:
        return (f"blob://{self.store.layout.container}/_onboarding/"
                f"sequence-{year}.json")

    def _client_base(self, client_id: str) -> str:
        return f"blob://{self.store.layout.container}/{client_id}/onboarding"

    def _version_uri(self, client_id: str, version: int) -> str:
        return f"{self._client_base(client_id)}/versions/{version:04d}.json"

    def _current_uri(self, client_id: str) -> str:
        return f"{self._client_base(client_id)}/current.json"

    def _artefact_uri(self, client_id: str, version: int, rel: str) -> str:
        return f"{self._client_base(client_id)}/artefacts/{version:04d}/{rel}"

    def _artefact_current_uri(self, client_id: str, rel: str) -> str:
        return f"{self._client_base(client_id)}/artefacts/current/{rel}"

    def _index_uri(self) -> str:
        return (f"blob://{self.store.layout.container}/_onboarding/"
                "clients.json")

    # -- case references ---------------------------------------------------- #
    def next_case_id(self, year: int) -> str:
        """``ONB-2026-0001``. The counter is per year and persisted, so a
        reference is never reused after a restart."""
        uri = self._sequence_uri(year)
        doc = _read_json(self.store.storage, uri) or {"next": 1}
        case_id = mint_case_id(year, int(doc["next"]))
        _write_json(self.store.storage, uri, {"next": int(doc["next"]) + 1})
        return case_id

    # -- cases -------------------------------------------------------------- #
    def save_case(self, case: OnboardingCase) -> OnboardingCase:
        _write_json(self.store.storage, self._case_uri(case.case_id),
                    case.to_dict())
        return case

    def load_case(self, case_id: str) -> Optional[OnboardingCase]:
        doc = _read_json(self.store.storage, self._case_uri(case_id))
        return OnboardingCase.from_dict(doc) if doc else None

    def list_cases(self, *, client_id: Optional[str] = None,
                   include_terminal: bool = True) -> List[OnboardingCase]:
        out: List[OnboardingCase] = []
        for uri in self.store.storage.list(self._cases_prefix()):
            if not uri.endswith(".json"):
                continue
            doc = _read_json(self.store.storage, uri)
            if not doc:
                continue
            case = OnboardingCase.from_dict(doc)
            if client_id is not None and case.client_id != client_id:
                continue
            if not include_terminal and case.status in TERMINAL:
                continue
            out.append(case)
        return sorted(out, key=lambda c: c.case_id, reverse=True)

    # -- client index ------------------------------------------------------- #
    def onboarded_clients(self) -> List[str]:
        doc = _read_json(self.store.storage, self._index_uri()) or {}
        return sorted(doc.get("clients") or [])

    def _register(self, client_id: str) -> None:
        uri = self._index_uri()
        doc = _read_json(self.store.storage, uri) or {"clients": []}
        if client_id not in doc["clients"]:
            doc["clients"].append(client_id)
            _write_json(self.store.storage, uri, doc)

    # -- versions ----------------------------------------------------------- #
    def current(self, client_id: str) -> Optional[ConfigurationVersion]:
        ptr = _read_json(self.store.storage, self._current_uri(client_id))
        if not ptr:
            return None
        return self.version(client_id, int(ptr["version"]))

    def version(self, client_id: str,
                version: int) -> Optional[ConfigurationVersion]:
        doc = _read_json(self.store.storage,
                         self._version_uri(client_id, version))
        return ConfigurationVersion.from_dict(doc) if doc else None

    def history(self, client_id: str) -> List[ConfigurationVersion]:
        out: List[ConfigurationVersion] = []
        for uri in self.store.storage.list(
                f"{self._client_base(client_id)}/versions"):
            if not uri.endswith(".json"):
                continue
            doc = _read_json(self.store.storage, uri)
            if doc:
                out.append(ConfigurationVersion.from_dict(doc))
        return sorted(out, key=lambda v: v.version, reverse=True)

    def next_version(self, client_id: str) -> int:
        versions = self.history(client_id)
        return (max(v.version for v in versions) + 1) if versions else 1

    def commit(self, *, case: OnboardingCase, changes: List[Dict[str, Any]],
               artefacts: List[Dict[str, Any]],
               activated_by: str) -> ConfigurationVersion:
        """Write a new immutable version and make it current.

        Idempotent by content: activating a case whose answers already match the
        current version returns that version rather than minting an identical
        one, so a repeated activation cannot fork a client's history.
        """
        client_id = case.client_id
        content_hash = stable_hash(canonical_json(case.answers))
        previous = self.current(client_id)
        if previous is not None and previous.content_hash == content_hash:
            return previous

        version = self.next_version(client_id)
        record = ConfigurationVersion(
            client_id=client_id, version=version, status=VERSION_ACTIVE,
            answers=case.answers, case_id=case.case_id, case_kind=case.kind,
            approved_by=case.approved_by, approved_at=case.approved_at,
            activated_by=activated_by, activated_at=now_iso(),
            reason=case.approval_reason,
            based_on_version=(previous.version if previous else None),
            changes=changes, artefacts=artefacts, content_hash=content_hash)
        _write_json(self.store.storage,
                    self._version_uri(client_id, version), record.to_dict())
        if previous is not None:
            previous.status = VERSION_SUPERSEDED
            _write_json(self.store.storage,
                        self._version_uri(client_id, previous.version),
                        previous.to_dict())
        _write_json(self.store.storage, self._current_uri(client_id),
                    {"version": version, "content_hash": content_hash,
                     "activated_by": activated_by,
                     "activated_at": record.activated_at})
        self._register(client_id)
        self.store.register_client(client_id)
        return record

    # -- generated artefacts ------------------------------------------------ #
    def write_artefact(self, client_id: str, version: int, rel: str,
                       text: str) -> str:
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
