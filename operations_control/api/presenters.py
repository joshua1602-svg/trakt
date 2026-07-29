"""operations_control.api.presenters — the simplified business view.

The raw Governed Agent Result / workflow documents are never sent to the UI
verbatim: presenters strip provenance paths and internal identifiers and attach
operator labels, so the React application renders plain language only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..contracts import WorkflowRun
from ..language import (
    SCOPE_EXPLANATIONS,
    STAGE_LABELS,
    STATUS_LABELS,
    WORKFLOW_TYPE_LABELS,
)


def present_stage(stage: str, status: str,
                  gar: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out = {
        "stage": stage,
        "label": STAGE_LABELS.get(stage, stage.title()),
        "status": status,
        "status_label": STATUS_LABELS.get(status, status.title()),
    }
    if gar:
        out.update({
            "summary": gar.get("summary", ""),
            "why_it_matters": gar.get("why_it_matters", ""),
            "warnings": gar.get("warnings") or [],
            "blockers": gar.get("blockers") or [],
            "evidence": gar.get("evidence") or [],
            "decision_count": len(gar.get("decisions_required") or []),
        })
    return out


def present_workflow(run: WorkflowRun,
                     results: Dict[str, Dict[str, Any]],
                     open_decisions: int = 0) -> Dict[str, Any]:
    stages = []
    for stage in run.applicable_stages:
        st = run.stage_status(stage)
        stages.append(present_stage(stage, st, results.get(stage)))
    return {
        "workflow_id": run.workflow_id,
        "client_id": run.client_id,
        "portfolio_id": run.portfolio_id,
        "reporting_period": run.reporting_period,
        "outcome": run.outcome,
        "outcome_label": ("MI Reporting + ESMA Annex 2 delivery"
                          if run.outcome == "mi_annex2" else "MI Reporting"),
        "workflow_type": run.workflow_type,
        "workflow_type_label": WORKFLOW_TYPE_LABELS.get(run.workflow_type,
                                                        run.workflow_type),
        "status": run.status,
        "status_sentence": _status_sentence(run, open_decisions),
        "interrupted": run.interrupted,
        "blockers": run.blockers,
        "stages": stages,
        "configuration": ({
            "status": run.effective_config.get("status", ""),
            "package_versions": run.effective_config.get("package_versions", {}),
            "effective_version": run.effective_config.get("version"),
            "content_hash": str(run.effective_config.get("content_hash", ""))[:23],
        } if run.effective_config else None),
        "open_decisions": open_decisions,
        "rerun_count": run.rerun_count,
        "created_at": run.created_at,
        "updated_at": run.updated_at,
    }


def present_workflow_row(row: Dict[str, Any],
                         open_decisions: int = 0) -> Dict[str, Any]:
    return {
        "workflow_id": row.get("workflow_id"),
        "client_id": row.get("client_id"),
        "portfolio_id": row.get("portfolio_id"),
        "reporting_period": row.get("reporting_period"),
        "outcome": row.get("outcome"),
        "workflow_type": row.get("workflow_type"),
        "workflow_type_label": WORKFLOW_TYPE_LABELS.get(
            row.get("workflow_type", ""), row.get("workflow_type", "")),
        "status": row.get("status"),
        "open_decisions": open_decisions,
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def _status_sentence(run: WorkflowRun, open_decisions: int) -> str:
    s = run.status
    if run.interrupted:
        return ("This run was interrupted. Choose 'Run again' to continue "
                "where it left off.")
    if s == "running":
        return "Trakt is working on this. You don't need to do anything yet."
    if s == "needs_review":
        n = open_decisions
        return (f"{n} decision{'s' if n != 1 else ''} need your attention."
                if n else "Some items need your review.")
    if s == "blocked":
        return run.blockers[0] if run.blockers else \
            "Something needs attention before Trakt can continue."
    if s == "awaiting_publication":
        return "The report is ready. It will only be published when you approve it."
    if s == "published":
        return "Published. This is now the latest official version."
    if s == "held":
        return "You held this report back. It has not been published."
    if s == "failed":
        return ("Something did not go as expected on our side. Nothing has "
                "been lost. You can try running it again.")
    if s == "cancelled":
        return "This workflow was cancelled."
    return "Received and ready to start."


def present_decision(doc: Dict[str, Any]) -> Dict[str, Any]:
    rec = doc.get("recommendation") or {}
    provenance_sentence = ""
    if rec:
        src = rec.get("source", "")
        if src == "llm":
            provenance_sentence = ("Suggested by Trakt's assistant and checked "
                                   "for compatibility — please confirm.")
        elif src == "memory":
            provenance_sentence = "Matches a rule you approved previously."
        elif src == "deterministic":
            provenance_sentence = "Suggested automatically."
    return {
        "decision_id": doc.get("decision_id"),
        "workflow_id": doc.get("workflow_id"),
        "client_id": doc.get("client_id"),
        "stage": doc.get("stage"),
        "kind": doc.get("kind"),
        "title": doc.get("title"),
        "question": doc.get("question"),
        "blocking": bool(doc.get("blocking")),
        "status": doc.get("status"),
        "recommendation": ({"value": rec.get("value"),
                            "label": provenance_sentence,
                            "confidence": rec.get("confidence")}
                           if rec else None),
        "options": doc.get("options") or [],
        "evidence": doc.get("evidence") or [],
        "allowed_scopes": doc.get("allowed_scopes") or [],
        "default_scope": doc.get("default_scope", "portfolio"),
        "scope_explanations": SCOPE_EXPLANATIONS,
        "created_at": doc.get("created_at"),
        "resolved_by": doc.get("resolved_by"),
        "resolved_at": doc.get("resolved_at"),
        "resolution_value": doc.get("resolution_value"),
        "resolution_scope": doc.get("resolution_scope"),
    }


def present_rule(doc: Dict[str, Any]) -> Dict[str, Any]:
    p = doc.get("payload") or {}
    return {
        "rule_id": doc.get("rule_id"),
        "version": doc.get("version"),
        "kind": doc.get("kind"),
        "scope": doc.get("scope"),
        "client_id": doc.get("client_id") or "",
        "portfolio_id": doc.get("portfolio_id") or "",
        "status": doc.get("status"),
        "source_term": p.get("source_column") or p.get("source_value")
        or p.get("subject") or "",
        "approved_meaning": p.get("canonical_field") or p.get("canonical_value")
        or p.get("selected_action") or p.get("disposition") or "",
        "description": doc.get("description", ""),
        "approved_by": doc.get("approved_by", ""),
        "approved_at": doc.get("approved_at", ""),
    }


def present_publication(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "publication_id": doc.get("publication_id"),
        "client_id": doc.get("client_id"),
        "workflow_id": doc.get("workflow_id"),
        "workflow_type": doc.get("workflow_type"),
        "workflow_type_label": WORKFLOW_TYPE_LABELS.get(
            doc.get("workflow_type", ""), doc.get("workflow_type", "")),
        "reporting_period": doc.get("reporting_period"),
        "status": doc.get("status"),
        "version": doc.get("version"),
        "rule_version_count": len(doc.get("rule_versions") or {}),
        "rule_versions": doc.get("rule_versions") or {},
        "previous_publication_id": doc.get("previous_publication_id"),
        "approved_by": doc.get("approved_by"),
        "published_at": doc.get("published_at"),
        "prepared_at": doc.get("prepared_at"),
    }


BATCH_STATUS_LABELS = {
    "receiving": "Waiting for files",
    "incomplete": "Waiting for required input",
    "classifying": "Identifying files",
    "review_required": "Review required",
    "configuration_required": "Configuration needed",
    "ready": "Ready to process",
    "running": "Processing",
    "completed": "Completed",
    "failed": "Needs attention",
}


def present_batch(doc: Dict[str, Any],
                  role_labels: Dict[str, str]) -> Dict[str, Any]:
    def label(role: str) -> str:
        return role_labels.get(role, role.replace("_", " ").capitalize())

    files = []
    for f in doc.get("files") or []:
        if f.get("superseded_status") != "current":
            continue
        status = f.get("recognition_status")
        if status in ("recognised", "overridden"):
            sentence = "Received and recognised"
        elif status == "ambiguous":
            sentence = "Needs identification"
        elif status == "unsupported":
            sentence = "Unsupported file"
        else:
            sentence = "Received"
        files.append({
            "source_file_id": f.get("source_file_id"),
            "filename": f.get("original_filename"),
            "role": f.get("recognised_file_type", ""),
            "role_label": label(f.get("recognised_file_type", "")),
            "status": status, "status_sentence": sentence,
            "confidence": f.get("recognition_confidence"),
        })
    roles = []
    expected = doc.get("expected_input_roles") or {}
    received = set(doc.get("received_input_roles") or [])
    for r in expected.get("required") or []:
        roles.append({"role": r, "label": label(r), "required": True,
                      "satisfied": r in received})
    for r in expected.get("optional") or []:
        roles.append({"role": r, "label": label(r), "required": False,
                      "satisfied": r in received})
    return {
        "batch_id": doc.get("batch_id"),
        "client_id": doc.get("client_id"),
        "portfolio_id": doc.get("portfolio_id"),
        "reporting_date": doc.get("reporting_date"),
        "workflow_type": doc.get("workflow_type"),
        "status": doc.get("status"),
        "status_label": BATCH_STATUS_LABELS.get(doc.get("status", ""),
                                                doc.get("status", "")),
        "status_sentence": doc.get("status_reason", ""),
        "auto_start_when_ready": bool(doc.get("auto_start_when_ready")),
        "files": files,
        "input_roles": roles,
        "missing_roles": [label(r) for r in
                          (doc.get("missing_input_roles") or [])],
        "configuration_ready": doc.get("effective_config_status")
        in ("READY", "READY_WITH_WARNINGS"),
        "blocking_decisions": doc.get("blocking_decisions") or [],
        "workflow_id": doc.get("workflow_id", ""),
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
    }


def present_delivery(doc: Dict[str, Any]) -> Dict[str, Any]:
    cls = doc.get("classification") or {}
    return {
        "delivery_id": doc.get("delivery_id"),
        "client_id": doc.get("client_id"),
        "portfolio_id": doc.get("portfolio_id"),
        "reporting_period": doc.get("reporting_period"),
        "dataset": doc.get("dataset"),
        "frequency": doc.get("frequency"),
        "files": [f.get("name", "") for f in (doc.get("files") or [])],
        "classification": cls.get("suggested", ""),
        "classification_label": WORKFLOW_TYPE_LABELS.get(
            cls.get("suggested", ""), ""),
        "classification_sentence": cls.get("sentence", ""),
        "registered_at": doc.get("registered_at"),
    }
