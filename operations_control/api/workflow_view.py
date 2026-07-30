"""operations_control.api.workflow_view — the delivery case file.

A delivery is a governed case, not a form. This module assembles the durable
view of one: the stages it has already been through, the stage it is at, and
what happens next — from the SAME persisted documents the engine already owns
(the workflow run, its governed agent results, the input batch, the decisions
and the publication record). Nothing here decides anything; it reads.

The seven operator steps are a presentation of the engine's own stages, not a
second state machine:

    Files received       <- input batch
    Delivery identified  <- delivery classification on the run
    Configuration selected <- effective configuration pinned to the run
    Data assessed        <- understanding / mapping / validation / assembly
    Issues reviewed      <- decisions raised against the run
    Publication approval <- publication stage + prepared publication
    Published            <- publication record

Every string is operator language; provenance stays in the underlying documents.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..contracts import (
    ANNEX2_STAGES,
    DEC_OPEN,
    KIND_PUBLICATION,
    OUTCOME_MI_ANNEX2,
    PUBLICATION_SCOPE_DEFAULT,
    RUN_AWAITING_PUBLICATION,
    RUN_CANCELLED,
    RUN_FAILED,
    RUN_HELD,
    RUN_PUBLISHED,
    STAGE_ASSEMBLY,
    STAGE_MAPPING,
    STAGE_PUBLICATION,
    STAGE_UNDERSTANDING,
    STAGE_VALIDATION,
    ST_BLOCKED,
    ST_COMPLETED,
    WorkflowRun,
)
from ..language import CLASSIFICATION_SENTENCES, WORKFLOW_TYPE_LABELS

# -- step vocabulary --------------------------------------------------------- #

STEP_FILES = "files_received"
STEP_IDENTIFIED = "delivery_identified"
STEP_CONFIGURATION = "configuration_selected"
STEP_ASSESSED = "data_assessed"
STEP_ISSUES = "issues_reviewed"
STEP_APPROVAL = "publication_approval"
STEP_PUBLISHED = "published"

STEP_ORDER = (STEP_FILES, STEP_IDENTIFIED, STEP_CONFIGURATION, STEP_ASSESSED,
              STEP_ISSUES, STEP_APPROVAL, STEP_PUBLISHED)

STEP_LABELS = {
    STEP_FILES: "Files received",
    STEP_IDENTIFIED: "Delivery identified",
    STEP_CONFIGURATION: "Configuration selected",
    STEP_ASSESSED: "Data assessed",
    STEP_ISSUES: "Issues reviewed",
    STEP_APPROVAL: "Publication approval",
    STEP_PUBLISHED: "Published",
}

COMPLETE = "complete"
CURRENT = "current"
PENDING = "pending"
BLOCKED = "blocked"
NOT_APPLICABLE = "not_applicable"

STEP_STATUS_LABELS = {
    COMPLETE: "Complete",
    CURRENT: "Current",
    PENDING: "Pending",
    BLOCKED: "Blocked",
    NOT_APPLICABLE: "Not applicable",
}

#: Publication approval scopes offered on a delivery. The platform-wide scope is
#: deliberately absent — see ``contracts.PUBLICATION_SCOPES``.
APPROVAL_SCOPES = (
    {"value": "delivery", "label": "No — this delivery only",
     "explanation": "Trakt will ask again for the next delivery."},
    {"value": "portfolio", "label": "Yes — future deliveries for this portfolio",
     "explanation": "Trakt will apply this decision to this portfolio's future "
                    "deliveries."},
    {"value": "client", "label": "Yes — future deliveries for this client",
     "explanation": "Trakt will apply this decision to every portfolio "
                    "belonging to this client."},
)


def _fact(label: str, value: Any) -> Dict[str, str]:
    return {"label": label, "value": "" if value is None else str(value)}


def _kept(facts: List[Optional[Dict[str, str]]]) -> List[Dict[str, str]]:
    return [f for f in facts if f and f["value"]]


def _bytes_label(size: Any) -> str:
    try:
        n = int(size)
    except (TypeError, ValueError):
        return ""
    if n < 1024:
        return f"{n} bytes"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def _short_checksum(sha: str) -> str:
    raw = (sha or "").split(":", 1)[-1]
    return raw[:12] if raw else ""


_MONTHS = ("January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December")


def _period_words(period: str) -> str:
    """``2026-06-30`` -> ``June 2026``. Anything unrecognised is left alone."""
    parts = (period or "").split("-")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        month = int(parts[1])
        if 1 <= month <= 12:
            return f"{_MONTHS[month - 1]} {parts[0]}"
    return period


# -- individual steps -------------------------------------------------------- #

def _files_step(run: WorkflowRun, batch: Optional[Dict[str, Any]],
                role_labels: Dict[str, str]) -> Dict[str, Any]:
    files: List[Dict[str, Any]] = []
    complete = True
    source = ""
    if batch:
        source = batch.get("source_prefix") or ""
        for f in batch.get("files") or []:
            if f.get("superseded_status") != "current":
                continue
            role = f.get("recognised_file_type") or ""
            files.append({
                "filename": f.get("original_filename", ""),
                "size": _bytes_label(f.get("size")),
                "received_at": f.get("received_at", ""),
                "received_from": f.get("received_by_or_source", ""),
                "checksum": _short_checksum(f.get("sha256", "")),
                "role_label": role_labels.get(role,
                                              role.replace("_", " ").capitalize()
                                              if role else ""),
            })
        complete = not (batch.get("missing_input_roles") or [])
    else:
        for f in run.delivery.get("files") or []:
            files.append({"filename": f.get("name", ""), "size": "",
                          "received_at": run.delivery.get("registered_at", ""),
                          "received_from": run.delivery.get("registered_by", ""),
                          "checksum": "", "role_label": ""})

    n = len(files)
    summary = (f"{n} file{'s' if n != 1 else ''} received."
               if n else "No files have been received yet.")
    if n and complete:
        summary += " The pack was complete."
    elif n:
        summary += " The pack was not complete."
    return {
        "key": STEP_FILES,
        "status": COMPLETE if n else PENDING,
        "summary": summary,
        "facts": _kept([
            _fact("Files received", n or ""),
            _fact("Source location", source),
            _fact("Pack complete", "Yes" if complete and n else
                  ("No" if n else "")),
        ]),
        "files": files,
    }


def _identified_step(run: WorkflowRun) -> Dict[str, Any]:
    delivery = run.delivery or {}
    dataset = (delivery.get("dataset") or "funded").lower()
    book = {"funded": "Funded book", "pipeline": "Pipeline",
            "forecast": "Forecast"}.get(dataset, dataset.capitalize())
    type_label = WORKFLOW_TYPE_LABELS.get(run.workflow_type, run.workflow_type)
    if run.classification_overridden:
        basis = (f"An operator chose this instead of Trakt's suggestion "
                 f"({WORKFLOW_TYPE_LABELS.get(run.classification_suggested, run.classification_suggested)}).")
        if run.classification_reason:
            basis += f" Reason given: {run.classification_reason}"
    else:
        basis = CLASSIFICATION_SENTENCES.get(
            run.workflow_type, "Trakt recognised this delivery from the client, "
                               "portfolio and reporting period.")
    return {
        "key": STEP_IDENTIFIED,
        "status": COMPLETE,
        "summary": f"{type_label} for {run.client_id} — {run.portfolio_id}.",
        "facts": _kept([
            _fact("Client", run.client_id),
            _fact("Portfolio", run.portfolio_id),
            _fact("Reporting period", run.reporting_period),
            _fact("Book", book),
            _fact("Delivery type", type_label),
            _fact("How this was decided", basis),
        ]),
    }


def _configuration_step(run: WorkflowRun) -> Dict[str, Any]:
    eff = run.effective_config or {}
    packages = eff.get("package_versions") or {}
    status = str(eff.get("status") or "")
    ready = status.startswith("READY")
    regime = ("ESMA Annex 2" if run.outcome == OUTCOME_MI_ANNEX2
              else "Management information only")
    return {
        "key": STEP_CONFIGURATION,
        "status": COMPLETE if eff else PENDING,
        "summary": ("The configuration for this delivery was selected and "
                    "locked to it." if ready else
                    "Trakt has not settled the configuration for this delivery "
                    "yet."),
        "facts": _kept([
            _fact("Asset package", packages.get("asset") and
                  f"v{packages['asset']}"),
            _fact("Regime", regime),
            _fact("Regime package", packages.get("regime") and
                  f"v{packages['regime']}"),
            _fact("Platform package", packages.get("system") and
                  f"v{packages['system']}"),
            _fact("Configuration version", eff.get("version") and
                  f"v{eff['version']}"),
        ]),
    }


def _assessed_step(run: WorkflowRun,
                   results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    stages = [STAGE_UNDERSTANDING, STAGE_MAPPING, STAGE_VALIDATION,
              STAGE_ASSEMBLY]
    if run.outcome == OUTCOME_MI_ANNEX2:
        stages += list(ANNEX2_STAGES)
    warnings: List[str] = []
    blockers: List[str] = []
    evidence: List[Dict[str, Any]] = []
    summaries: List[str] = []
    statuses = []
    for stage in stages:
        gar = results.get(stage)
        statuses.append(run.stage_status(stage))
        if not gar:
            continue
        if gar.get("summary"):
            summaries.append(gar["summary"])
        for w in gar.get("warnings") or []:
            if w not in warnings:
                warnings.append(w)
        for b in gar.get("blockers") or []:
            if b not in blockers:
                blockers.append(b)
        evidence.extend(gar.get("evidence") or [])

    if ST_BLOCKED in statuses:
        status = BLOCKED
    elif all(s == ST_COMPLETED for s in statuses) and statuses:
        status = COMPLETE
    elif any(s == ST_COMPLETED for s in statuses):
        status = CURRENT
    else:
        status = PENDING
    return {
        "key": STEP_ASSESSED,
        "status": status,
        "summary": (summaries[-1] if summaries else
                    "Trakt has not assessed this data yet."),
        "detail": summaries,
        "warnings": warnings,
        "blockers": blockers,
        "evidence": evidence,
        "facts": _kept([
            _fact("Output eligibility",
                  "Ready for publication" if status == COMPLETE
                  else ("Held by a blocking problem" if status == BLOCKED
                        else "")),
        ]),
    }


def _issues_step(run: WorkflowRun,
                 decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
    resolved, unresolved = [], []
    for d in decisions:
        # The publication question is the approval step's own decision, not an
        # issue raised about the data. It belongs there, once.
        if d.get("kind") == KIND_PUBLICATION:
            continue
        item = {
            "decision_id": d.get("decision_id", ""),
            "title": d.get("title", ""),
            "question": d.get("question", ""),
            "blocking": bool(d.get("blocking")),
            "status": d.get("status", ""),
            "answer": d.get("resolution_value", ""),
            "scope": d.get("resolution_scope", ""),
            "actor": d.get("resolved_by", ""),
            "at": d.get("resolved_at", ""),
            "overridden": (d.get("status") == "approved"
                           and bool(d.get("recommendation"))
                           and d.get("resolution_value") not in
                           ("", (d.get("recommendation") or {}).get("value"))),
        }
        (unresolved if d.get("status") == DEC_OPEN else resolved).append(item)
    if unresolved:
        status = BLOCKED if any(i["blocking"] for i in unresolved) else CURRENT
        n = len(unresolved)
        summary = (f"{n} question{'s' if n != 1 else ''} still need"
                   f"{'' if n != 1 else 's'} an answer.")
    elif resolved:
        status, summary = COMPLETE, (
            f"{len(resolved)} question{'s' if len(resolved) != 1 else ''} "
            "answered. Every answer is recorded against this delivery.")
    else:
        status, summary = NOT_APPLICABLE, "Trakt had no questions about this delivery."
    return {
        "key": STEP_ISSUES,
        "status": status,
        "summary": summary,
        "resolved": resolved,
        "unresolved": unresolved,
        "facts": _kept([
            _fact("Answered", len(resolved) or ""),
            _fact("Still open", len(unresolved) or ""),
            _fact("Answers that differed from Trakt's suggestion",
                  sum(1 for i in resolved if i["overridden"]) or ""),
        ]),
    }


def _approval_step(run: WorkflowRun, results: Dict[str, Dict[str, Any]],
                   files_step: Dict[str, Any], assessed: Dict[str, Any],
                   issues: Dict[str, Any],
                   publication: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    gar = results.get(STAGE_PUBLICATION) or {}
    if run.status == RUN_PUBLISHED:
        status = COMPLETE
    elif run.status == RUN_AWAITING_PUBLICATION:
        status = CURRENT
    elif run.status == RUN_HELD:
        status = BLOCKED
    elif run.status in (RUN_CANCELLED, RUN_FAILED):
        status = NOT_APPLICABLE
    else:
        status = PENDING

    n_files = len(files_step.get("files") or [])
    evidence_lines = [f"{n_files} file{'s' if n_files != 1 else ''} received"]
    if assessed.get("summary") and status in (CURRENT, COMPLETE):
        evidence_lines.append(assessed["summary"])
    n_block = len(assessed.get("blockers") or [])
    evidence_lines.append(f"{n_block} blocking issue{'s' if n_block != 1 else ''}")
    n_warn = len(assessed.get("warnings") or [])
    if n_warn:
        evidence_lines.append(f"{n_warn} warning{'s' if n_warn != 1 else ''} accepted")
    packages = (run.effective_config or {}).get("package_versions") or {}
    if packages.get("asset"):
        evidence_lines.append(f"Asset configuration v{packages['asset']} applied")
    if run.outcome == OUTCOME_MI_ANNEX2:
        evidence_lines.append("ESMA Annex 2 enabled")

    period = _period_words(run.reporting_period) or "this period"
    consequence = (f"This will publish the {period} {run.client_id} "
                   f"{run.portfolio_id} delivery as the latest official "
                   "version.")
    return {
        "key": STEP_APPROVAL,
        "status": status,
        "summary": (gar.get("summary") or
                    ("Published." if status == COMPLETE else
                     "The delivery is ready. It will only be published when "
                     "you approve it." if status == CURRENT else
                     "Nothing is waiting for your approval yet.")),
        "warnings": gar.get("warnings") or [],
        "blockers": gar.get("blockers") or [],
        "evidence": gar.get("evidence") or [],
        "approval": {
            "available": status == CURRENT,
            "headline": "Ready to publish",
            "evidence_lines": evidence_lines,
            "question": "Publish this delivery as the latest official version?",
            "scope_question": "Should Trakt remember this decision for future "
                              "deliveries?",
            "scopes": [dict(s) for s in APPROVAL_SCOPES],
            "default_scope": PUBLICATION_SCOPE_DEFAULT,
            "consequence": consequence,
            "scope_consequences": {
                "delivery": "This decision applies only to this delivery.",
                "portfolio": "This decision will also apply to future "
                             f"deliveries for {run.portfolio_id}.",
                "client": "This decision will also apply to future deliveries "
                          f"for {run.client_id}.",
            },
            "version": (publication or {}).get("version"),
        },
        "facts": _kept([
            _fact("Open questions", len(issues.get("unresolved") or []) or ""),
        ]),
    }


def _published_step(run: WorkflowRun,
                    publication: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if run.status != RUN_PUBLISHED or not publication:
        return {
            "key": STEP_PUBLISHED,
            "status": PENDING,
            "summary": "Nothing has been published for this delivery yet.",
            "facts": [],
            "outputs": [],
        }
    artefacts = publication.get("published_artefacts") or {}
    outputs = []
    if artefacts.get("latest"):
        outputs.append("Management information (latest)")
    if artefacts.get("period"):
        outputs.append(f"Management information ({run.reporting_period})")
    if artefacts.get("regime"):
        outputs.append("ESMA Annex 2 delivery")
    scope = publication.get("approval_scope") or PUBLICATION_SCOPE_DEFAULT
    scope_label = {"delivery": "This delivery only",
                   "portfolio": "Future deliveries for this portfolio",
                   "client": "Future deliveries for this client"}.get(scope, scope)
    return {
        "key": STEP_PUBLISHED,
        "status": COMPLETE,
        "summary": "Published. This is now the latest official version.",
        "facts": _kept([
            _fact("Published", publication.get("published_at")),
            _fact("Published by", publication.get("approved_by")),
            _fact("Version", publication.get("version")),
            _fact("Became the latest version",
                  "Yes" if artefacts.get("latest") else "Yes"),
            _fact("Decision applied to", scope_label),
            _fact("Audit reference", publication.get("publication_id")),
        ]),
        "outputs": outputs,
    }


# -- assembly ---------------------------------------------------------------- #

def build_steps(run: WorkflowRun, results: Dict[str, Dict[str, Any]], *,
                batch: Optional[Dict[str, Any]] = None,
                decisions: Optional[List[Dict[str, Any]]] = None,
                publication: Optional[Dict[str, Any]] = None,
                role_labels: Optional[Dict[str, str]] = None
                ) -> List[Dict[str, Any]]:
    """The seven-step case file for one delivery, in operator order."""
    labels = role_labels or {}
    files = _files_step(run, batch, labels)
    identified = _identified_step(run)
    configuration = _configuration_step(run)
    assessed = _assessed_step(run, results)
    issues = _issues_step(run, decisions or [])
    approval = _approval_step(run, results, files, assessed, issues, publication)
    published = _published_step(run, publication)

    steps = [files, identified, configuration, assessed, issues, approval,
             published]
    if run.status in (RUN_CANCELLED,):
        for step in steps:
            if step["status"] == PENDING:
                step["status"] = NOT_APPLICABLE
    for step in steps:
        step["label"] = STEP_LABELS[step["key"]]
        step["status_label"] = STEP_STATUS_LABELS[step["status"]]
        step.setdefault("facts", [])
        step.setdefault("warnings", [])
        step.setdefault("blockers", [])
    return steps


def current_step(steps: List[Dict[str, Any]]) -> str:
    """The step the delivery is at — where the case file should open.

    A blocked step wins, because that is what is stopping the delivery.
    Otherwise the FURTHEST step still in progress is the one that wants the
    operator; anything earlier is context they can expand if they want it.
    """
    for step in steps:
        if step["status"] == BLOCKED:
            return step["key"]
    in_progress = [s for s in steps if s["status"] == CURRENT]
    if in_progress:
        return in_progress[-1]["key"]
    done = [s for s in steps if s["status"] == COMPLETE]
    return done[-1]["key"] if done else steps[0]["key"]


def step_for_stage(stage: str) -> str:
    """Which case-file step an engine stage (or a decision) belongs to.

    Used to land an existing deep link — a review URL, a stage link — on the
    right step instead of breaking it.
    """
    if stage == STAGE_PUBLICATION:
        return STEP_APPROVAL
    if stage in ("received", ""):
        return STEP_FILES
    return STEP_ASSESSED
