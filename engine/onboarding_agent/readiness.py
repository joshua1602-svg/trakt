"""
readiness.py
============

Separate, non-conflated readiness concepts for a promoted onboarding run.

The single ``readiness_status: blocked`` is confusing for an MI-only run: the
central tapes and MI APIs can be fully usable while unresolved *non-blocking*
Gate 4 confirmations keep governance "blocked". This module splits readiness into
three independent concepts so the promotion output can say plainly which is which:

  * ``mi_runtime_readiness``        — can the funded / pipeline MI dashboard + APIs
    run? Ready when the central tapes exist with rows and required MI domains are
    present. NOT affected by pending non-blocking confirmations or value conflicts.
  * ``onboarding_governance_readiness`` — are all Gate 4 decisions / conflicts /
    required-field gaps resolved? Blocked while any remain; ``pending_confirmations``
    when only non-blocking confirmations remain.
  * ``xml_delivery_readiness``      — only ready once the regulatory XML target
    frame exists, which is produced downstream (never at onboarding).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from . import domain_coverage as dc
from . import target_first_decisions as tfd

READY = "ready"
BLOCKED = "blocked"
PENDING_CONFIRMATIONS = "pending_confirmations"
NOT_READY = "not_ready"

_APPROVED_NAME = "34_target_first_decisions_approved.yaml"
_TEMPLATE_NAME = "34_target_first_decisions.yaml"


def _count_gate4(project_dir: str | Path) -> Dict[str, Any]:
    """Count unresolved Gate 4 decisions from the active 34 file (approved file
    preferred over the pending template). A copied pending template is NOT an
    approval — its decisions all count as pending."""
    pdir = Path(project_dir)
    approved_path = pdir / _APPROVED_NAME
    template_path = pdir / _TEMPLATE_NAME
    path = approved_path if approved_path.exists() else template_path
    doc = tfd.load_decisions(path)
    decisions = (doc or {}).get("decisions", []) or []

    def _pending(d: Dict[str, Any]) -> bool:
        return str(d.get("status", "")).strip().lower() != "approved"

    pending = [d for d in decisions if isinstance(d, dict) and _pending(d)]
    pending_blocking = [d for d in pending if bool(d.get("blocking", False))]
    return {
        "decisions_total": len(decisions),
        "pending_total": len(pending),
        "pending_blocking": len(pending_blocking),
        "pending_non_blocking": len(pending) - len(pending_blocking),
        "decisions_source": str(path) if doc is not None else None,
        "real_approval": tfd.is_real_approval(doc),
    }


def compute_readiness_breakdown(
    project_dir: str | Path,
    tape_result: Dict[str, Any],
    coverage: List[Any],
    mode: str,
    regulatory_reporting_enabled: bool,
) -> Dict[str, Any]:
    """Compute the three separated readiness concepts. Deterministic; never raises."""
    lender_created = bool(tape_result.get("central_lender_tape_created"))
    loan_count = int(tape_result.get("loan_count", 0) or 0)
    pipeline_created = bool(tape_result.get("central_pipeline_tape_created"))
    pipeline_count = int(tape_result.get("pipeline_count", 0) or 0)
    conflicts = int(tape_result.get("conflict_count", 0) or 0)
    summary = tape_result.get("lender_summary", {}) or {}
    gaps = int(summary.get("gap_count", tape_result.get("gap_count", 0)) or 0)

    missing_blocking = [dc.DOMAIN_LABELS.get(c.domain, c.domain)
                        for c in coverage if c.blocking and c.status == dc.MISSING]

    # --- MI runtime readiness: can the MI dashboard / APIs run on the tapes? ---
    # Independent of governance: pending non-blocking confirmations and value
    # conflicts (which still have a deterministic selection) do NOT stop MI runtime.
    mi_reasons: List[str] = []
    if not lender_created or loan_count == 0:
        mi_reasons.append("funded central lender tape is empty")
    for d in missing_blocking:
        mi_reasons.append(f"{d} domain missing")
    mi_ready = not mi_reasons

    # --- Governance readiness: are all Gate 4 items resolved? ---
    gate4 = _count_gate4(project_dir)
    gov_reasons: List[str] = []
    if gate4["pending_total"]:
        gov_reasons.append(f"{gate4['pending_total']} Gate 4 decision(s) pending "
                           f"({gate4['pending_blocking']} blocking, "
                           f"{gate4['pending_non_blocking']} non-blocking)")
    if conflicts:
        gov_reasons.append(f"{conflicts} unresolved value conflict(s) in central tape")
    if gaps:
        gov_reasons.append(f"{gaps} unresolved required field(s)")
    gov_ready = not gov_reasons
    if gov_ready:
        gov_status = READY
    elif gate4["pending_blocking"] or conflicts or gaps:
        gov_status = BLOCKED
    else:
        gov_status = PENDING_CONFIRMATIONS  # only non-blocking confirmations remain

    # --- XML delivery readiness: produced downstream, never at onboarding. ---
    xml_reasons = ["regulatory XML target frame is produced by the downstream "
                   "transformation/validation/projection stage, not at onboarding"]
    if not gov_ready:
        xml_reasons.append("onboarding governance approvals are not complete")

    return {
        "mi_runtime_readiness": {
            "status": READY if mi_ready else BLOCKED,
            "ready": mi_ready,
            "reasons": mi_reasons,
            "central_lender_tape_created": lender_created,
            "loan_count": loan_count,
            "central_pipeline_tape_created": pipeline_created,
            "pipeline_count": pipeline_count,
        },
        "onboarding_governance_readiness": {
            "status": gov_status,
            "ready": gov_ready,
            "reasons": gov_reasons,
            "pending_total": gate4["pending_total"],
            "pending_blocking": gate4["pending_blocking"],
            "pending_non_blocking": gate4["pending_non_blocking"],
            "unresolved_conflicts": conflicts,
            "unresolved_required_fields": gaps,
            "decisions_source": gate4["decisions_source"],
            "real_approval": gate4["real_approval"],
        },
        "xml_delivery_readiness": {
            "status": NOT_READY,
            "ready": False,
            "reasons": xml_reasons,
        },
    }


def format_breakdown_lines(breakdown: Dict[str, Any]) -> List[str]:
    """The three plain-language promotion lines."""
    mi = breakdown["mi_runtime_readiness"]
    gov = breakdown["onboarding_governance_readiness"]
    xml = breakdown["xml_delivery_readiness"]
    gov_label = {
        READY: "ready",
        PENDING_CONFIRMATIONS: "pending confirmations (non-blocking)",
        BLOCKED: "blocked",
    }.get(gov["status"], gov["status"])
    return [
        f"MI runtime: {'ready' if mi['ready'] else 'blocked'}"
        + (f" — {'; '.join(mi['reasons'])}" if mi["reasons"] else ""),
        f"Governance approvals: {gov_label}"
        + (f" — {'; '.join(gov['reasons'])}" if gov["reasons"] else ""),
        f"XML delivery: {'ready' if xml['ready'] else 'not ready'}",
    ]
