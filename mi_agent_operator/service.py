"""mi_agent_operator.service — operator approval service (no HTTP, fully testable).

Wraps the existing approval workflow into a small set of operator-facing
operations the console UI calls:

    queue()   → the review queue (pending approvals) as normalised cards
    item(id)  → one item's full detail (why, old→new schema, AI-proposed mapping,
                detected files)
    approve() → accept the (possibly edited) mapping → promote the source ACTIVE
    reject()  → reject with a reason
    edit()    → choose an alternative mapping before approving
    audit()   → the auto-approval governance log (what did NOT need a human)

Every mutating call records the operator identity (``decided_by``) so the audit
trail is complete. Deterministic mapping / registry stays the source of truth.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from apps.blob_trigger_app import approvals as _approvals
from apps.blob_trigger_app import ops as _ops
from apps.blob_trigger_app import run_records as _run_records
from apps.blob_trigger_app.layout import Layout
from apps.blob_trigger_app.persistence import ProductionPersistence
from apps.blob_trigger_app.source_registry import SourceRegistry
from apps.blob_trigger_app.storage import Storage, open_storage

# Friendly labels for the approval kinds the router raises.
_KIND_LABEL = {
    "new_source": "New client / portfolio",
    "schema_drift": "Schema change (review)",
}


def _pack_key(client_id: str, source_portfolio_id: str, dataset: str,
              frequency: str, period: str) -> str:
    """Reconstruct the durable pack_key (same rule as router._pack_key) so an
    approval artifact can be joined to its run record + LLM recommendations."""
    raw = "/".join([client_id or "", source_portfolio_id or "", dataset or "",
                    frequency or "", period or ""])
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class OperatorService:
    storage: Storage
    layout: Layout
    persistence: ProductionPersistence
    registry: SourceRegistry

    @classmethod
    def from_env(cls, *, registry_uri: Optional[str] = None,
                 local_root: Optional[str] = None,
                 storage: Optional[Storage] = None) -> "OperatorService":
        storage = storage or open_storage(local_root=local_root)
        layout = Layout.from_env()
        persistence = ProductionPersistence(storage, layout)
        registry = SourceRegistry(registry_uri or layout.registry_uri, storage=storage)
        return cls(storage=storage, layout=layout, persistence=persistence,
                   registry=registry)

    # -- read ------------------------------------------------------------- #
    def _run_record_for(self, art: Dict[str, Any]) -> Dict[str, Any]:
        pk = _pack_key(art.get("client_id"), art.get("source_portfolio_id"),
                       art.get("dataset"), art.get("frequency"), art.get("period"))
        return self.persistence.load_run_record(pk) or {}

    def _card(self, art: Dict[str, Any]) -> Dict[str, Any]:
        rec = self._run_record_for(art)
        materiality = rec.get("materiality") or {}
        evidence = materiality.get("evidence") or {}
        kind = art.get("kind") or "schema_drift"
        return {
            "approval_id": art.get("approval_id"),
            "kind": kind,
            "kind_label": _KIND_LABEL.get(kind, kind),
            "status": art.get("status"),
            "client_id": art.get("client_id"),
            "source_portfolio_id": art.get("source_portfolio_id"),
            "source_book_type": art.get("source_book_type"),
            "dataset": art.get("dataset"),
            "frequency": art.get("frequency"),
            "period": art.get("period"),
            "new_fingerprint": art.get("schema_fingerprint"),
            "prior_fingerprint": art.get("prior_schema_fingerprint"),
            "detected_files": art.get("detected_files") or [],
            "suggested_mapping_id": art.get("suggested_mapping_id"),
            "suggested_mapping_config_path": art.get("suggested_mapping_config_path"),
            "reason": (rec.get("event_decision") or kind),
            "material": bool(materiality.get("material")) if materiality else None,
            "added_columns": evidence.get("added_columns") or [],
            "removed_columns": evidence.get("removed_columns") or [],
            "created_at": art.get("created_at"),
        }

    def queue(self) -> List[Dict[str, Any]]:
        """The review queue — pending approvals as operator cards, newest first."""
        cards = [self._card(a) for a in _approvals.list_pending(self.storage, self.layout)]
        return sorted(cards, key=lambda c: (c.get("created_at") or ""), reverse=True)

    def item(self, approval_id: str) -> Optional[Dict[str, Any]]:
        """Full detail for one queue item, including the AI-proposed mapping rows
        and the role → header signature the promotion will pin."""
        art = _approvals.show(self.storage, self.layout, approval_id)
        if art is None:
            return None
        card = self._card(art)
        pk = _pack_key(art.get("client_id"), art.get("source_portfolio_id"),
                       art.get("dataset"), art.get("frequency"), art.get("period"))
        rec = self.persistence.load_run_record(pk) or {}
        llm = self.persistence.load_llm_recommendations(pk) or {}
        # Proposed mapping the operator reviews / overrides: prefer the LLM resolver
        # rows; fall back to the deterministic mapping recommendations on the run.
        proposed = llm.get("recommendations") or rec.get("mapping_recommendations") or []
        card.update({
            "pack_key": pk,
            "role_schemas": art.get("role_schemas") or {},
            "source_metadata": art.get("source_metadata") or {},
            "proposed_mapping": proposed,
            "llm_advisory": bool(llm),
            "diagnostics_reason": ((rec.get("run_summary") or {})
                                   .get("central_canonical_unavailable_reason")),
            "next_action": rec.get("next_action") or {},
        })
        return card

    def audit(self, limit: int = 100) -> List[Dict[str, Any]]:
        """The auto-approval governance log — recurring changes that did NOT need a
        human (materiality evidence + old→new fingerprint), newest first."""
        rows: List[Dict[str, Any]] = []
        for rec in _run_records.list_run_records(self.storage, self.layout):
            if not rec.get("auto_approved"):
                continue
            aa = rec.get("auto_approval") or {}
            rows.append({
                "pack_key": rec.get("pack_key"),
                "client_id": rec.get("client_id"),
                "source_portfolio_id": rec.get("source_portfolio_id"),
                "dataset": rec.get("dataset"), "frequency": rec.get("frequency"),
                "period": rec.get("reporting_period"),
                "old_fingerprint": aa.get("old_fingerprint"),
                "new_fingerprint": aa.get("new_fingerprint"),
                "reasons": ((rec.get("materiality") or {}).get("reasons")) or [],
                "governance_artifact_uri": rec.get("governance_artifact_uri"),
                "created_at": rec.get("created_at"),
            })
        return sorted(rows, key=lambda r: (r.get("created_at") or ""), reverse=True)[:limit]

    # -- write ------------------------------------------------------------ #
    def edit(self, approval_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Choose an alternative mapping (edit the suggested mapping id / config /
        metadata / notes) BEFORE approving. Non-editable fields are rejected."""
        return _ops.edit_approval(self.storage, self.layout, approval_id, updates)

    def approve(self, approval_id: str, *, decided_by: str,
                mapping_id: Optional[str] = None,
                mapping_config_path: Optional[str] = None) -> Dict[str, Any]:
        """Accept the (possibly edited) mapping and PROMOTE the source to ACTIVE —
        the genuinely one-click action. Pins the fingerprint + header signatures so
        the next identical upload runs deterministically."""
        art = _approvals.show(self.storage, self.layout, approval_id)
        if art is None:
            raise KeyError(f"no such approval: {approval_id}")
        mid = (mapping_id or art.get("suggested_mapping_id")
               or f"{art.get('source_portfolio_id')}_{art.get('dataset')}_"
                  f"{art.get('frequency')}_operator_v1")
        cfg = mapping_config_path or art.get("suggested_mapping_config_path")
        _approvals.approve(self.storage, self.layout, approval_id, mapping_id=mid,
                           mapping_config_path=cfg, decided_by=decided_by,
                           decided_at=_now())
        rec = _approvals.promote(self.storage, self.layout, self.registry, approval_id)
        return {"status": "promoted", "approval_id": approval_id,
                "registry_key": rec.key, "mapping_id": rec.approved_mapping_id,
                "mapping_version": rec.mapping_version, "decided_by": decided_by}

    def reject(self, approval_id: str, *, reason: str, decided_by: str) -> Dict[str, Any]:
        """Reject the source/change with a reason (nothing is promoted)."""
        art = _approvals.reject(self.storage, self.layout, approval_id,
                                reason=reason, decided_by=decided_by, decided_at=_now())
        return {"status": "rejected", "approval_id": approval_id,
                "reason": art.get("reject_reason"), "decided_by": decided_by}
