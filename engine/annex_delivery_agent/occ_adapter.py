"""
engine.annex_delivery_agent.occ_adapter
=======================================

Additive adapter between the Annex Delivery Agent and the Operations Control
Centre (the operator console in :mod:`mi_agent_operator`, backed by the
operational store in :mod:`apps.blob_trigger_app`).

Additive means exactly that: nothing here changes an existing OCC workflow, an
existing approval record or the blob-trigger semantics. It exposes the delivery
stages the console did not previously have — everything after projection —
in the shape the console already renders.

The workflow the operator sees becomes::

    projection → delivery normalisation → XML generation → XSD validation
    → operator review → publication

The wording is the other half of the job. The console must never let "the file
parsed" read as "the submission is correct", so every stage line here states
what was actually established, and the review stage states plainly that values
were supplied on the client's behalf.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from trakt_core.context import ExecutionContext

from .agent import AnnexDeliveryAgent
from .contracts import DeliveryRequest, DeliveryResult, DeliveryStatus

# --------------------------------------------------------------------------- #
# Stage vocabulary shown to the operator
# --------------------------------------------------------------------------- #

STAGE_PROJECTION = "projection"
STAGE_DELIVERY_NORMALISATION = "delivery_normalisation"
STAGE_ARTEFACT = "artefact_generation"
STAGE_SCHEMA_VALIDATION = "schema_validation"
STAGE_OPERATOR_REVIEW = "operator_review"
STAGE_PUBLICATION = "publication"

OCC_STAGES = (STAGE_PROJECTION, STAGE_DELIVERY_NORMALISATION, STAGE_ARTEFACT,
              STAGE_SCHEMA_VALIDATION, STAGE_OPERATOR_REVIEW, STAGE_PUBLICATION)

STATE_PENDING = "pending"
STATE_RUNNING = "running"
STATE_DONE = "done"
STATE_BLOCKED = "blocked"
STATE_ACTION_REQUIRED = "action_required"


@dataclass(frozen=True)
class OperatorStage:
    """One row of the operator's workflow view."""

    key: str
    label: str
    state: str
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "label": self.label, "state": self.state,
                "detail": self.detail}


def operator_view(result: DeliveryResult, *, annex_label: str = "Annex 2") -> Dict[str, Any]:
    """Render a delivery result as the OCC workflow view.

    Pure: takes a result, returns a dictionary. No storage, no HTTP, no
    dependency on the console's own modules — which is what makes it safe to
    call from the FastAPI operator app, a CLI, or a test.
    """
    stages = _stages(result, annex_label)
    evidence = result.transformation_evidence or {}
    return {
        "run_id": result.run_id,
        "annex_id": result.annex_id,
        "annex_version": result.annex_version,
        "annex_label": annex_label,
        "tenant_id": result.tenant_id,
        "portfolio_id": result.portfolio_id,
        "reporting_date": result.reporting_date,
        "status": result.status,
        "headline": _headline(result, annex_label),
        "why_it_matters": result.why_it_matters,
        "stages": [s.to_dict() for s in stages],
        "current_stage": next((s.key for s in stages
                               if s.state in (STATE_RUNNING, STATE_ACTION_REQUIRED,
                                              STATE_BLOCKED)), STAGE_PUBLICATION),
        "metrics": {
            "records": result.record_count,
            "projected_fields": result.projected_field_count,
            "delivered_fields": result.delivered_field_count,
            "artefact_size_bytes": result.artefact.size_bytes if result.artefact else 0,
            "artefact_label": result.artefact.label if result.artefact else None,
            "total_seconds": round(result.total_seconds, 1),
            "peak_rss_mb": result.peak_rss_mb,
        },
        "automatic_values": {
            "fills": evidence.get("automatic_fill_count", 0),
            "coercions": evidence.get("coercion_count", 0),
            "omissions": evidence.get("omission_count", 0),
            "review_required": bool(evidence.get("review_required")),
            "lines": list(evidence.get("operator_warnings") or []),
        },
        "blocking_errors": [dict(e) for e in result.blocking_errors],
        "approval": result.approval.to_dict(),
        "actions": _actions(result),
        "disclaimer": (
            "Schema validation confirms the file is structurally acceptable to the "
            "regulator's parser. It does not confirm that the figures are "
            "economically or regulatorily correct."),
    }


def _headline(result: DeliveryResult, annex_label: str) -> str:
    return {
        DeliveryStatus.RECEIVED: f"{annex_label} delivery requested",
        DeliveryStatus.PREFLIGHT_BLOCKED: f"{annex_label} cannot be prepared",
        DeliveryStatus.READY: f"{annex_label} ready to prepare",
        DeliveryStatus.NORMALISING: f"{annex_label} data being prepared",
        DeliveryStatus.NORMALISATION_FAILED: f"{annex_label} data could not be prepared",
        DeliveryStatus.BUILDING: f"{annex_label} XML being created",
        DeliveryStatus.BUILD_FAILED: f"{annex_label} XML could not be created",
        DeliveryStatus.VALIDATING: f"{annex_label} XML being validated",
        DeliveryStatus.VALIDATION_FAILED: f"{annex_label} XML failed schema validation",
        DeliveryStatus.PREPARED: f"{annex_label} ready for publication",
        DeliveryStatus.PREPARED_WITH_WARNINGS: "Automatic values require review",
        DeliveryStatus.AWAITING_APPROVAL: f"{annex_label} awaiting operator approval",
        DeliveryStatus.APPROVED: f"{annex_label} approved — ready to publish",
        DeliveryStatus.REJECTED: f"{annex_label} rejected by operator",
        DeliveryStatus.PUBLISHED: f"{annex_label} published",
    }.get(result.status, result.status)


def _stages(result: DeliveryResult, annex_label: str) -> List[OperatorStage]:
    status = result.status
    reached = _reached(status)
    evidence = result.transformation_evidence or {}

    projection = OperatorStage(
        STAGE_PROJECTION, "Projection",
        STATE_DONE if reached >= 1 else STATE_PENDING,
        (f"{result.record_count:,} exposure records projected, "
         f"{result.projected_field_count} fields")
        if result.record_count else "Projected data resolved upstream")

    if status == DeliveryStatus.PREFLIGHT_BLOCKED:
        projection = OperatorStage(STAGE_PROJECTION, "Projection", STATE_BLOCKED,
                                   result.summary)

    normalisation = OperatorStage(
        STAGE_DELIVERY_NORMALISATION, f"{annex_label} data prepared",
        STATE_BLOCKED if status == DeliveryStatus.NORMALISATION_FAILED
        else (STATE_DONE if reached >= 2 else STATE_PENDING),
        result.summary if status == DeliveryStatus.NORMALISATION_FAILED
        else "Delivery values normalised against the Annex delivery rules")

    artefact = OperatorStage(
        STAGE_ARTEFACT, f"{annex_label} XML created",
        STATE_BLOCKED if status == DeliveryStatus.BUILD_FAILED
        else (STATE_DONE if reached >= 3 else STATE_PENDING),
        (f"{result.artefact.label} — "
         f"{result.artefact.size_bytes / (1024 * 1024):.1f} MB, "
         f"{result.record_count:,} records")
        if result.artefact else "")

    schema_valid = bool(result.schema_validation and result.schema_validation.valid)
    validation = OperatorStage(
        STAGE_SCHEMA_VALIDATION, "XML passed schema validation",
        STATE_BLOCKED if status == DeliveryStatus.VALIDATION_FAILED
        else (STATE_DONE if (reached >= 4 and schema_valid) else STATE_PENDING),
        _validation_detail(result))

    review_required = bool(evidence.get("review_required"))
    review_state = STATE_PENDING
    if status in (DeliveryStatus.PREPARED, DeliveryStatus.PREPARED_WITH_WARNINGS,
                  DeliveryStatus.AWAITING_APPROVAL):
        review_state = STATE_ACTION_REQUIRED
    elif status in (DeliveryStatus.APPROVED, DeliveryStatus.PUBLISHED):
        review_state = STATE_DONE
    elif status == DeliveryStatus.REJECTED:
        review_state = STATE_BLOCKED

    review = OperatorStage(
        STAGE_OPERATOR_REVIEW,
        "Automatic values require review" if review_required else "Operator review",
        review_state,
        _review_detail(result, evidence))

    publication = OperatorStage(
        STAGE_PUBLICATION, "Ready for publication",
        STATE_DONE if status == DeliveryStatus.PUBLISHED
        else (STATE_ACTION_REQUIRED if status == DeliveryStatus.APPROVED
              else STATE_PENDING),
        "Published" if status == DeliveryStatus.PUBLISHED
        else "Publication requires explicit operator approval")

    return [projection, normalisation, artefact, validation, review, publication]


def _validation_detail(result: DeliveryResult) -> str:
    sv = result.schema_validation
    if sv is None or not sv.executed:
        return ""
    schema = Path(sv.schema_reference).name
    if sv.valid:
        return (f"Validated against {schema}. This confirms the file's structure, "
                f"not the accuracy of its figures.")
    return f"{sv.error_count} schema error(s) against {schema}. The file cannot be submitted."


def _review_detail(result: DeliveryResult, evidence: Mapping[str, Any]) -> str:
    fills = int(evidence.get("automatic_fill_count") or 0)
    coercions = int(evidence.get("coercion_count") or 0)
    omissions = int(evidence.get("omission_count") or 0)
    if not (fills or coercions or omissions):
        return "No automatic values were inserted or adjusted."
    return (f"{fills:,} value(s) supplied automatically, {coercions:,} adjusted to fit "
            f"the required format, {omissions:,} present in your data but not reported. "
            f"Review these before approving.")


def _reached(status: str) -> int:
    order = {
        DeliveryStatus.RECEIVED: 0,
        DeliveryStatus.PREFLIGHT_BLOCKED: 0,
        DeliveryStatus.READY: 1,
        DeliveryStatus.NORMALISING: 1,
        DeliveryStatus.NORMALISATION_FAILED: 1,
        DeliveryStatus.BUILDING: 2,
        DeliveryStatus.BUILD_FAILED: 2,
        DeliveryStatus.VALIDATING: 3,
        DeliveryStatus.VALIDATION_FAILED: 3,
        DeliveryStatus.PREPARED: 4,
        DeliveryStatus.PREPARED_WITH_WARNINGS: 4,
        DeliveryStatus.AWAITING_APPROVAL: 5,
        DeliveryStatus.APPROVED: 5,
        DeliveryStatus.REJECTED: 5,
        DeliveryStatus.PUBLISHED: 6,
    }
    return order.get(status, 0)


def _actions(result: DeliveryResult) -> List[Dict[str, str]]:
    """What the operator may do next. The console renders exactly these."""
    status = result.status
    if status in (DeliveryStatus.PREPARED, DeliveryStatus.PREPARED_WITH_WARNINGS):
        return [{"action": "request_approval", "label": "Send for approval"},
                {"action": "approve", "label": "Approve for publication"},
                {"action": "reject", "label": "Reject"}]
    if status == DeliveryStatus.AWAITING_APPROVAL:
        return [{"action": "approve", "label": "Approve for publication"},
                {"action": "reject", "label": "Reject"}]
    if status == DeliveryStatus.APPROVED:
        return [{"action": "publish", "label": "Publish"}]
    if status in (DeliveryStatus.PREFLIGHT_BLOCKED, DeliveryStatus.NORMALISATION_FAILED,
                  DeliveryStatus.BUILD_FAILED, DeliveryStatus.VALIDATION_FAILED):
        return [{"action": "rerun", "label": "Prepare again"}]
    return []


# --------------------------------------------------------------------------- #
# Service facade for the operator console
# --------------------------------------------------------------------------- #


class AnnexDeliveryOperatorService:
    """The OCC-facing surface. Every call is tenant-bound by the agent.

    A thin facade on purpose: the console should not construct requests,
    resolve run directories or interpret statuses, because each of those is a
    place where a console could accidentally act for the wrong tenant.
    """

    def __init__(self, agent: Optional[AnnexDeliveryAgent] = None) -> None:
        self.agent = agent or AnnexDeliveryAgent()

    def supported_reports(self) -> List[Dict[str, Any]]:
        return self.agent.supported_annexes()

    def prepare(self, context: ExecutionContext, request: DeliveryRequest) -> Dict[str, Any]:
        governed = self.agent.prepare(context, request)
        payload = governed.result or {}
        view = (operator_view(DeliveryResult.from_dict(payload),
                              annex_label=_label(request.annex_id))
                if payload else {"status": "blocked",
                                 "headline": governed.error.message if governed.error else "Blocked"})
        return {"governance": governed.governance_block(), "view": view}

    def request_approval(self, context: ExecutionContext, request: DeliveryRequest,
                         *, run_id: str) -> Dict[str, Any]:
        gate = self.agent.gate_for(context, request, run_id=run_id)
        return operator_view(gate.request_approval(context), annex_label=_label(request.annex_id))

    def approve(self, context: ExecutionContext, request: DeliveryRequest, *,
                run_id: str, note: str = "") -> Dict[str, Any]:
        gate = self.agent.gate_for(context, request, run_id=run_id)
        return operator_view(gate.approve(context, note=note),
                             annex_label=_label(request.annex_id))

    def reject(self, context: ExecutionContext, request: DeliveryRequest, *,
               run_id: str, reason: str) -> Dict[str, Any]:
        gate = self.agent.gate_for(context, request, run_id=run_id)
        return operator_view(gate.reject(context, reason=reason),
                             annex_label=_label(request.annex_id))

    def publish(self, context: ExecutionContext, request: DeliveryRequest,
                *, run_id: str) -> Dict[str, Any]:
        gate = self.agent.gate_for(context, request, run_id=run_id)
        return operator_view(gate.publish(context), annex_label=_label(request.annex_id))


def _label(annex_id: str) -> str:
    text = str(annex_id or "").replace("ESMA_", "").replace("Annex", "Annex ")
    return text.strip() or "Report"


__all__ = [
    "OCC_STAGES", "OperatorStage", "operator_view", "AnnexDeliveryOperatorService",
    "STAGE_PROJECTION", "STAGE_DELIVERY_NORMALISATION", "STAGE_ARTEFACT",
    "STAGE_SCHEMA_VALIDATION", "STAGE_OPERATOR_REVIEW", "STAGE_PUBLICATION",
    "STATE_PENDING", "STATE_RUNNING", "STATE_DONE", "STATE_BLOCKED",
    "STATE_ACTION_REQUIRED",
]
