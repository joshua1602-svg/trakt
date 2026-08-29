"""operations_control.rules — persistent, scoped, versioned operational rules.

Every operator approval becomes a rule. Rules are the system of record in the
operations-control container; the **projector** additionally materialises them
into the artefacts existing agents already read (client mapping memory YAML via
:mod:`engine.onboarding_agent.mapping_memory`), using the existing persistence
functions only — no agent or registry code is touched, and the core field
registry is never written.

Scope precedence (documented, and matching the repository's existing layered
behaviour — run-approved overrides > client memory > registry aliases):

    file  >  portfolio  >  client  >  global

The most specific approved rule wins.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .contracts import SCHEMA_VERSION, new_id, now_iso
from .stores import OpsStore

SCOPE_FILE = "file"
SCOPE_RUN = "run"
SCOPE_PORTFOLIO = "portfolio"
SCOPE_CLIENT = "client"
SCOPE_ASSET = "asset"          # administrator-approved only
SCOPE_GLOBAL = "global"        # administrator-approved only
#: Most specific first — this is the precedence order.
SCOPES = (SCOPE_FILE, SCOPE_RUN, SCOPE_PORTFOLIO, SCOPE_CLIENT,
          SCOPE_ASSET, SCOPE_GLOBAL)
#: Scopes an ordinary operator may not promote decisions to.
ADMIN_ONLY_SCOPES = (SCOPE_ASSET, SCOPE_GLOBAL)

RULE_ACTIVE = "active"
RULE_SUPERSEDED = "superseded"
RULE_RETIRED = "retired"

#: Where projected client-memory YAML lands (filesystem; consumed by the
#: onboarding agent's client-memory application order on future runs).
OPS_MEMORY_ROOT_ENV = "TRAKT_OPS_MEMORY_ROOT"
DEFAULT_OPS_MEMORY_ROOT = ".ops_state/client_memory"


@dataclass
class RuleRecord:
    """One version of one governed rule. Prior versions are immutable."""

    rule_id: str
    version: int
    kind: str                       # contracts.DECISION_KINDS member
    scope: str                      # SCOPES member
    client_id: str = ""             # "" for global/asset scope
    portfolio_id: str = ""          # set for portfolio/file scope
    file_ref: str = ""              # delivery fingerprint for file scope
    run_ref: str = ""               # workflow id for run scope
    status: str = RULE_ACTIVE
    payload: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    # provenance
    suggested_by: str = "operator"  # llm | deterministic | memory | operator
    confidence: Optional[float] = None
    decision_id: str = ""
    workflow_id: str = ""
    approved_by: str = ""
    approved_at: str = field(default_factory=now_iso)
    reason: str = ""
    effective_from: str = field(default_factory=now_iso)
    superseded_by: Optional[int] = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RuleRecord":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def subject_key(self) -> str:
        """What this rule is 'about' — used for precedence resolution and for
        same-scope supersession. Kind-specific."""
        p = self.payload or {}
        if self.kind in ("field_mapping", "alias"):
            return f"{self.kind}:{_norm(p.get('source_column') or p.get('alias', ''))}"
        if self.kind == "enum":
            return f"enum:{p.get('field', '')}:{_norm(p.get('source_value', ''))}"
        if self.kind == "validation_exception":
            return f"exception:{p.get('check', '')}"
        return f"{self.kind}:{_norm(str(p.get('subject', '')))}"


def _norm(s: str) -> str:
    import re
    return re.sub(r"[\s_]+", " ", str(s or "").strip().lower())


class RuleStore:
    """Versioned rule persistence over the OpsStore + scope-precedence lookup."""

    def __init__(self, store: OpsStore):
        self.store = store
        self.layout = store.layout
        self.storage = store.storage

    # -- persistence -------------------------------------------------------- #
    def _client_dir(self, rule: RuleRecord) -> Optional[str]:
        if rule.scope in (SCOPE_GLOBAL, SCOPE_ASSET):
            return None
        return rule.client_id or None

    def approve(self, rule: RuleRecord) -> RuleRecord:
        """Persist a newly-approved rule. If an active rule exists for the same
        (scope, subject), this becomes version n+1 and supersedes it."""
        existing = self._find_same_subject(rule)
        if existing is not None:
            rule.rule_id = existing.rule_id
            rule.version = existing.version + 1
            existing.status = RULE_SUPERSEDED
            existing.superseded_by = rule.version
            self._write_version(existing)
        else:
            rule.rule_id = rule.rule_id or new_id("rule")
            rule.version = rule.version or 1
        rule.status = RULE_ACTIVE
        self._write_version(rule)
        self._write_current(rule)
        return rule

    def retire(self, client_id: Optional[str], rule_id: str, *,
               by: str, reason: str) -> Optional[RuleRecord]:
        cur = self.get(client_id, rule_id)
        if cur is None or cur.status != RULE_ACTIVE:
            return None
        cur.status = RULE_RETIRED
        cur.reason = reason
        cur.approved_by = by
        self._write_version(cur)
        self._write_current(cur)
        return cur

    def _write_version(self, rule: RuleRecord) -> None:
        from .stores import _write_json
        _write_json(self.storage,
                    self.layout.rule_version_uri(self._client_dir(rule),
                                                 rule.rule_id, rule.version),
                    rule.to_dict())

    def _write_current(self, rule: RuleRecord) -> None:
        from .stores import _write_json
        _write_json(self.storage,
                    self.layout.rule_current_uri(self._client_dir(rule),
                                                 rule.rule_id),
                    rule.to_dict())

    # -- queries ------------------------------------------------------------ #
    def get(self, client_id: Optional[str], rule_id: str) -> Optional[RuleRecord]:
        from .stores import _read_json
        doc = _read_json(self.storage,
                         self.layout.rule_current_uri(client_id, rule_id))
        return RuleRecord.from_dict(doc) if doc else None

    def history(self, client_id: Optional[str], rule_id: str) -> List[RuleRecord]:
        from .stores import _read_json
        out = []
        base = self.layout.rules_prefix(client_id)
        for uri in self.storage.list(base):
            if f"/{rule_id}/versions/" in uri and not uri.rpartition("/")[2].startswith("."):
                doc = _read_json(self.storage, uri)
                if doc:
                    out.append(RuleRecord.from_dict(doc))
        return sorted(out, key=lambda r: r.version)

    def list_current(self, client_id: Optional[str]) -> List[RuleRecord]:
        from .stores import _read_json
        out = []
        for uri in self.storage.list(self.layout.rules_prefix(client_id)):
            if uri.endswith("/current.json"):
                doc = _read_json(self.storage, uri)
                if doc:
                    out.append(RuleRecord.from_dict(doc))
        return sorted(out, key=lambda r: (r.kind, r.subject_key()))

    def _find_same_subject(self, rule: RuleRecord) -> Optional[RuleRecord]:
        for cur in self.list_current(self._client_dir(rule)):
            if (cur.status == RULE_ACTIVE and cur.scope == rule.scope
                    and cur.subject_key() == rule.subject_key()
                    and cur.portfolio_id == rule.portfolio_id
                    and cur.file_ref == rule.file_ref
                    and cur.run_ref == rule.run_ref):
                return cur
        return None

    # -- precedence resolution ---------------------------------------------- #
    def applicable(self, *, client_id: str, portfolio_id: str = "",
                   file_ref: str = "", run_ref: str = "") -> List[RuleRecord]:
        """All active rules that apply to this run context, most specific kept
        when the same subject is ruled at several scopes (file > run >
        portfolio > client > asset > global)."""
        candidates: List[RuleRecord] = []
        for r in self.list_current(client_id):
            if r.status != RULE_ACTIVE:
                continue
            if r.scope == SCOPE_FILE and (not file_ref or r.file_ref != file_ref):
                continue
            if r.scope == SCOPE_RUN and (not run_ref or r.run_ref != run_ref):
                continue
            if r.scope == SCOPE_PORTFOLIO and (
                    not portfolio_id or r.portfolio_id != portfolio_id):
                continue
            candidates.append(r)
        candidates.extend(r for r in self.list_current(None)
                          if r.status == RULE_ACTIVE
                          and r.scope in (SCOPE_GLOBAL, SCOPE_ASSET))

        rank = {s: i for i, s in enumerate(SCOPES)}   # file=0 (most specific)
        best: Dict[str, RuleRecord] = {}
        for r in sorted(candidates, key=lambda r: rank.get(r.scope, 99)):
            best.setdefault(r.subject_key(), r)
        return list(best.values())


# --------------------------------------------------------------------------- #
# Projection into agent-readable sinks
# --------------------------------------------------------------------------- #

def memory_root() -> Path:
    return Path(os.environ.get(OPS_MEMORY_ROOT_ENV, DEFAULT_OPS_MEMORY_ROOT))


def project_rules_to_client_memory(rules: List[RuleRecord], client_id: str,
                                   memory_dir: Optional[Path] = None) -> int:
    """Materialise mapping/alias/enum rules as client-memory YAML entries via
    the EXISTING :class:`MappingMemoryStore` — the same artefact the onboarding
    agent's client-memory application step consumes. Idempotent (the store
    de-duplicates on its own key). Returns the number of entries written."""
    from engine.onboarding_agent.mapping_memory import (
        DECISION_ENUM_MAPPING,
        DECISION_MAPPING_OVERRIDE,
        MappingMemoryStore,
        MemoryEntry,
    )

    mdir = memory_dir or (memory_root() / client_id / "client_memory")
    store = MappingMemoryStore(mdir, client_id=client_id)
    n = 0
    for r in rules:
        p = r.payload or {}
        if r.kind in ("field_mapping", "alias"):
            src = p.get("source_column") or p.get("alias") or ""
            tgt = p.get("canonical_field") or ""
            if not src or not tgt:
                continue
            store.save_entry(MemoryEntry(
                client_id=client_id, decision_type=DECISION_MAPPING_OVERRIDE,
                source_column=src, canonical_field=tgt,
                approved_by=r.approved_by, approved_at=r.approved_at,
                confidence=float(r.confidence or 1.0),
                notes=f"operations-control rule {r.rule_id} v{r.version} "
                      f"({r.scope} scope)"))
            n += 1
        elif r.kind == "enum":
            if (p or {}).get("layer") == "regulatory":
                # Belongs to the Annex 2 boundary, not to the canonical.
                continue
            store.save_entry(MemoryEntry(
                client_id=client_id, decision_type=DECISION_ENUM_MAPPING,
                source_column=p.get("field", ""),
                canonical_field=p.get("canonical_value", ""),
                source_value=p.get("source_value", ""),
                approved_by=r.approved_by, approved_at=r.approved_at,
                notes=f"operations-control rule {r.rule_id} v{r.version}"))
            n += 1
    return n
