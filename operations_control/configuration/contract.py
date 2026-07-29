"""EffectiveConfiguration — the immutable per-run configuration contract.

Produced by the OCC resolver BEFORE the Onboarding Agent runs; persisted,
hashed, versioned, tenant-bound and reconstructable. The agent receives this
contract (and a materialised snapshot directory derived from it) instead of
reading mutable repository YAML.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from ..contracts import new_id, now_iso

SCHEMA_VERSION = "1.0.0"

#: The explicit precedence contract, least to most specific. Formalised and
#: tested — never implied by dictionary merge order.
PRECEDENCE = ("system", "regime", "asset", "client", "portfolio",
              "run_decisions")


@dataclass(frozen=True)
class EffectiveConfiguration:
    """Immutable after creation (frozen dataclass; persisted before use)."""

    effective_config_id: str
    client_id: str
    portfolio_id: str
    asset_type: str
    regime_id: str                       # "" for MI-only runs
    regime_version: str
    reporting_date: str
    # Layer pinning: config-package versions + per-file sha256 hashes.
    system_config_version: str
    regime_config_version: str
    asset_config_version: str
    client_config_version: str
    portfolio_config_version: str
    decision_set_version: str
    layer_files: Dict[str, Dict[str, str]]      # layer -> {relpath: sha256}
    # Resolved scalar values with per-key provenance + override records.
    resolved_values: Dict[str, Any]
    value_provenance: Dict[str, str]            # dotted key -> layer
    overrides: List[Dict[str, Any]]             # original/overridden/why/who/when/scope
    # Governed rule sets (from the OCC rule store, already approved + scoped).
    mapping_rules: List[Dict[str, Any]]
    alias_rules: List[Dict[str, Any]]
    enum_rules: List[Dict[str, Any]]
    # Policies (references into the pinned system package).
    validation_policy: Dict[str, Any]
    file_recognition_policy: Dict[str, Any]
    schema_recognition_policy: Dict[str, Any]
    canonical_registry_reference: str
    source_provenance: Dict[str, str]           # layer -> package/source label
    conflicts: List[Dict[str, Any]]
    warnings: List[str]
    blockers: List[str]
    created_at: str
    created_by: str
    content_hash: str = ""
    version: int = 1
    schema_version: str = SCHEMA_VERSION

    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EffectiveConfiguration":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    @staticmethod
    def compute_hash(doc: Dict[str, Any]) -> str:
        """Stable content hash over everything except volatile identity."""
        stable = {k: v for k, v in doc.items()
                  if k not in ("effective_config_id", "created_at",
                               "created_by", "content_hash")}
        blob = json.dumps(stable, sort_keys=True, separators=(",", ":"),
                          default=str)
        return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build(**fields: Any) -> EffectiveConfiguration:
    """Construct with generated id/timestamp and computed content hash."""
    fields.setdefault("effective_config_id", new_id("ecfg"))
    fields.setdefault("created_at", now_iso())
    doc = dict(fields)
    doc["content_hash"] = ""
    fields["content_hash"] = EffectiveConfiguration.compute_hash(doc)
    return EffectiveConfiguration(**fields)
