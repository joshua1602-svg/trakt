"""operations_control.concentration — concentration-test governance.

The OCC operating process for concentration tests: extract proposals from a
client's response or supplied documents, hold them for operator review, record
every decision, and mint the immutable activated configuration the funded-book
evaluation service reads.

Boundary rules, same as the rest of the OCC:

* extraction and mapping are deterministic (``mi_agent.concentration_tests``);
  a model may propose through the assist seam, an operator decides;
* nothing activates without an explicit approval decision recorded with the
  operator's identity and timestamp;
* every state change appends to the client's tamper-evident audit chain
  (``OpsStore.append_audit``);
* the tenant is the authenticated principal's binding — this module never
  reads a client identifier out of payload content.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from mi_agent.concentration_tests.library import (
    ConcentrationLibrary,
    load_library,
)
from mi_agent.concentration_tests.matching import ExtractionAssist, build_proposals
from mi_agent.concentration_tests.models import (
    ActiveConfiguration,
    ActiveTest,
    ApprovalRecord,
    ConcernCode,
    PROPOSAL_APPROVED,
    PROPOSAL_CLARIFICATION_REQUESTED,
    PROPOSAL_NOT_APPLICABLE,
    PROPOSAL_PENDING_APPROVAL,
    PROPOSAL_PENDING_CONFIRMATION,
    PROPOSAL_REJECTED,
    PROPOSAL_SUPERSEDED,
    PROPOSAL_UNSUPPORTED,
    SourceEvidence,
    TestProposal,
)
from mi_agent.concentration_tests.store import ConcentrationStore, now_iso

from .stores import OpsStore


class ConcentrationError(ValueError):
    """A governance rule refused the action; the message is operator-facing."""


class ConcentrationGovernanceService:
    """Proposal lifecycle + activation for one operations-control deployment."""

    def __init__(self, ops_store: OpsStore,
                 store: Optional[ConcentrationStore] = None,
                 lib: Optional[ConcentrationLibrary] = None,
                 assist: Optional[ExtractionAssist] = None):
        self.ops = ops_store
        self.store = store or ConcentrationStore(
            ops_store.storage, container=ops_store.layout.container)
        self.lib = lib or load_library()
        #: Optional model seam — proposes candidate extractions only.
        self.assist = assist

    # ------------------------------------------------------------------ #
    # Extraction
    # ------------------------------------------------------------------ #
    def extract(self, client_id: str, *, actor: str, source_reference: str,
                text: str = "", rows: Optional[Sequence[Dict[str, Any]]] = None,
                onboarding_run_id: str = "", portfolio_id: str = ""
                ) -> List[TestProposal]:
        """Extract + map + persist proposals from a client response."""
        if not (text or rows):
            raise ConcentrationError(
                "Nothing to extract: provide the client's response text or a "
                "limits table.")
        proposals = build_proposals(
            lib=self.lib, client_id=client_id, portfolio_id=portfolio_id,
            onboarding_run_id=onboarding_run_id,
            source_reference=source_reference, text=text, rows=rows,
            assist=self.assist)
        for p in proposals:
            self.store.save_proposal(client_id, p)
        self.ops.append_audit(
            client_id, "concentration_tests_extracted", actor=actor,
            detail={"source_reference": source_reference,
                    "proposal_count": len(proposals),
                    "outcomes": {p.proposal_id: p.match_outcome
                                 for p in proposals}})
        return proposals

    # ------------------------------------------------------------------ #
    # Review
    # ------------------------------------------------------------------ #
    def list_proposals(self, client_id: str) -> List[TestProposal]:
        return self.store.list_proposals(client_id)

    def get_proposal(self, client_id: str, proposal_id: str) -> TestProposal:
        proposal = self.store.get_proposal(client_id, proposal_id)
        if proposal is None:
            raise ConcentrationError("That proposal could not be found.")
        return proposal

    def answer_confirmation(self, client_id: str, proposal_id: str, *,
                            actor: str, question: str, answer: str
                            ) -> TestProposal:
        proposal = self.get_proposal(client_id, proposal_id)
        proposal.confirmation_answers.append(
            {"question": question, "answer": answer, "answered_by": actor,
             "answered_at": now_iso()})
        proposal.confirmation_questions = [
            q for q in proposal.confirmation_questions if q != question]
        self._note(proposal, actor, f"Answered: {question} → {answer}")
        self.store.save_proposal(client_id, proposal)
        self.ops.append_audit(
            client_id, "concentration_confirmation_answered", actor=actor,
            detail={"proposal_id": proposal_id, "question": question})
        return proposal

    def update_proposal(self, client_id: str, proposal_id: str, *, actor: str,
                        metric_id: Optional[str] = None,
                        parameters: Optional[Dict[str, Any]] = None,
                        threshold: Optional[float] = None,
                        operator: Optional[str] = None,
                        effective_date: Optional[str] = None,
                        expiry_date: Optional[str] = None,
                        resolved_concerns: Optional[List[str]] = None,
                        note: str = "") -> TestProposal:
        """Operator edit of permitted fields, recorded and validated.

        Edits bump the proposal version; the audit chain records the change so
        an approval always names the exact version it decided on.
        """
        proposal = self.get_proposal(client_id, proposal_id)
        if proposal.status in (PROPOSAL_APPROVED, PROPOSAL_SUPERSEDED):
            raise ConcentrationError(
                "An approved proposal is immutable; supersede it instead.")
        changed: List[str] = []
        if metric_id is not None and metric_id != proposal.proposed_metric_id:
            if self.lib.get(metric_id) is None:
                raise ConcentrationError(f"Unknown metric '{metric_id}'.")
            proposal.proposed_metric_id = metric_id
            changed.append(f"metric → {metric_id}")
        if parameters is not None:
            problems = self.lib.validate_parameters(
                proposal.proposed_metric_id, parameters)
            if problems:
                raise ConcentrationError("; ".join(problems))
            proposal.parameters = dict(parameters)
            changed.append("parameters")
        if threshold is not None:
            proposal.extracted_threshold = float(threshold)
            changed.append(f"threshold → {threshold}")
        if operator is not None:
            metric = self.lib.get(proposal.proposed_metric_id)
            if metric and operator not in metric.supported_operators:
                raise ConcentrationError(
                    f"Operator '{operator}' is not supported by "
                    f"{proposal.proposed_metric_id}.")
            proposal.extracted_operator = operator
            changed.append(f"operator → {operator}")
        if effective_date is not None:
            proposal.effective_date = effective_date
            changed.append(f"effective date → {effective_date}")
        if expiry_date is not None:
            proposal.expiry_date = expiry_date
            changed.append(f"expiry date → {expiry_date}")
        for code in resolved_concerns or []:
            if code in proposal.concern_codes:
                proposal.concern_codes.remove(code)
                changed.append(f"concern resolved: {code}")
        if not changed and not note:
            return proposal
        proposal.version += 1
        if proposal.status == PROPOSAL_PENDING_CONFIRMATION and \
                not proposal.concern_codes and not proposal.confirmation_questions:
            proposal.status = PROPOSAL_PENDING_APPROVAL
        self._note(proposal, actor, note or "; ".join(changed))
        self.store.save_proposal(client_id, proposal)
        self.ops.append_audit(
            client_id, "concentration_proposal_updated", actor=actor,
            detail={"proposal_id": proposal_id, "version": proposal.version,
                    "changes": changed, "note": note})
        return proposal

    # ------------------------------------------------------------------ #
    # Decisions
    # ------------------------------------------------------------------ #
    def approve(self, client_id: str, proposal_id: str, *, actor: str,
                effective_date: str = "", comments: str = "",
                severity: str = "high",
                supersedes_test_id: str = "") -> TestProposal:
        """Approve a proposal. Refused while anything is unresolved."""
        proposal = self.get_proposal(client_id, proposal_id)
        metric = self.lib.get(proposal.proposed_metric_id)
        refusals: List[str] = []
        if metric is None:
            refusals.append("The proposal names no library metric.")
        elif metric.implementation_status == "not_implemented":
            refusals.append(
                f"Metric '{metric.metric_id}' has no deterministic "
                "implementation; mark the proposal unsupported instead.")
        if proposal.status not in (PROPOSAL_PENDING_APPROVAL,
                                   PROPOSAL_PENDING_CONFIRMATION):
            refusals.append(
                f"A proposal in status '{proposal.status}' cannot be approved.")
        if proposal.concern_codes:
            refusals.append(
                "Unresolved concerns remain: "
                + ", ".join(proposal.concern_codes)
                + ". Resolve or explicitly clear them first.")
        if proposal.confirmation_questions:
            refusals.append("Unanswered confirmation questions remain.")
        if proposal.extracted_threshold is None:
            refusals.append("The proposal has no threshold.")
        if proposal.extracted_operator not in ("max", "min"):
            refusals.append("The comparison direction is unresolved.")
        if metric is not None:
            if proposal.extracted_operator in ("max", "min") and \
                    proposal.extracted_operator not in metric.supported_operators:
                refusals.append(
                    f"Operator '{proposal.extracted_operator}' is not "
                    f"supported by {metric.metric_id}.")
            problems = self.lib.validate_parameters(metric.metric_id,
                                                    proposal.parameters)
            refusals.extend(problems)
        if refusals:
            raise ConcentrationError(" ".join(refusals))

        proposal.status = PROPOSAL_APPROVED
        approval = ApprovalRecord(
            decision="approved", operator=actor, decided_at=now_iso(),
            approved_metric_id=proposal.proposed_metric_id,
            approved_parameters=dict(proposal.parameters),
            source_version=proposal.version,
            effective_date=effective_date or proposal.effective_date,
            comments=comments, prior_active_test_id=supersedes_test_id)
        # The approval record travels on the proposal document so the review
        # surface shows it; activation copies it into the immutable version.
        proposal.approval = approval.to_dict()
        proposal.severity = severity or proposal.severity
        proposal.operator_notes.append(
            {"at": approval.decided_at, "actor": actor,
             "note": f"Approved. {comments}".strip()})
        if effective_date:
            proposal.effective_date = effective_date
        self.store.save_proposal(client_id, proposal)
        self.ops.append_audit(
            client_id, "concentration_proposal_approved", actor=actor,
            detail={"proposal_id": proposal_id,
                    "metric_id": proposal.proposed_metric_id,
                    "threshold": proposal.extracted_threshold,
                    "operator": proposal.extracted_operator,
                    "parameters": proposal.parameters,
                    "source_version": proposal.version,
                    "effective_date": approval.effective_date,
                    "comments": comments,
                    "supersedes_test_id": supersedes_test_id})
        return proposal

    def _decide(self, client_id: str, proposal_id: str, *, actor: str,
                status: str, event: str, note: str) -> TestProposal:
        proposal = self.get_proposal(client_id, proposal_id)
        if proposal.status in (PROPOSAL_APPROVED, PROPOSAL_SUPERSEDED):
            raise ConcentrationError(
                "An approved proposal is immutable; supersede it instead.")
        proposal.status = status
        self._note(proposal, actor, note)
        self.store.save_proposal(client_id, proposal)
        self.ops.append_audit(client_id, event, actor=actor,
                              detail={"proposal_id": proposal_id,
                                      "note": note})
        return proposal

    def reject(self, client_id: str, proposal_id: str, *, actor: str,
               reason: str) -> TestProposal:
        return self._decide(client_id, proposal_id, actor=actor,
                            status=PROPOSAL_REJECTED,
                            event="concentration_proposal_rejected",
                            note=f"Rejected: {reason}")

    def mark_unsupported(self, client_id: str, proposal_id: str, *,
                         actor: str, reason: str) -> TestProposal:
        proposal = self._decide(client_id, proposal_id, actor=actor,
                                status=PROPOSAL_UNSUPPORTED,
                                event="concentration_proposal_unsupported",
                                note=f"Unsupported: {reason}")
        proposal.implementation_request = proposal.implementation_request or {
            "requested_capability": proposal.extracted_test_name,
            "source_text": proposal.evidence.source_text,
            "reason": reason,
        }
        self.store.save_proposal(client_id, proposal)
        return proposal

    def mark_not_applicable(self, client_id: str, proposal_id: str, *,
                            actor: str, reason: str) -> TestProposal:
        return self._decide(client_id, proposal_id, actor=actor,
                            status=PROPOSAL_NOT_APPLICABLE,
                            event="concentration_proposal_not_applicable",
                            note=f"Not applicable: {reason}")

    def request_clarification(self, client_id: str, proposal_id: str, *,
                              actor: str, question: str) -> TestProposal:
        proposal = self._decide(client_id, proposal_id, actor=actor,
                                status=PROPOSAL_CLARIFICATION_REQUESTED,
                                event="concentration_clarification_requested",
                                note=f"Client clarification requested: {question}")
        if question not in proposal.confirmation_questions:
            proposal.confirmation_questions.append(question)
        self.store.save_proposal(client_id, proposal)
        return proposal

    def supersede(self, client_id: str, proposal_id: str, *, actor: str,
                  reason: str) -> TestProposal:
        proposal = self.get_proposal(client_id, proposal_id)
        if proposal.status != PROPOSAL_APPROVED:
            raise ConcentrationError("Only an approved proposal can be "
                                     "superseded.")
        proposal.status = PROPOSAL_SUPERSEDED
        self._note(proposal, actor, f"Superseded: {reason}")
        self.store.save_proposal(client_id, proposal)
        self.ops.append_audit(
            client_id, "concentration_proposal_superseded", actor=actor,
            detail={"proposal_id": proposal_id, "reason": reason})
        return proposal

    # ------------------------------------------------------------------ #
    # Activation
    # ------------------------------------------------------------------ #
    def activate(self, client_id: str, *, actor: str, reason: str = ""
                 ) -> ActiveConfiguration:
        """Mint the next immutable configuration version from every currently
        APPROVED proposal. Nothing else is included; a rejected, unsupported
        or still-open proposal cannot leak into the active set."""
        proposals = [p for p in self.store.list_proposals(client_id)
                     if p.status == PROPOSAL_APPROVED]
        if not proposals:
            raise ConcentrationError(
                "No approved proposals — there is nothing to activate.")
        tests: List[ActiveTest] = []
        for p in proposals:
            metric = self.lib.get(p.proposed_metric_id)
            approval = ApprovalRecord.from_dict(p.approval) if p.approval \
                else ApprovalRecord(
                    decision="approved", operator="(recorded in audit chain)",
                    decided_at=p.updated_at,
                    approved_metric_id=p.proposed_metric_id,
                    approved_parameters=dict(p.parameters),
                    source_version=p.version,
                    effective_date=p.effective_date)
            tests.append(ActiveTest(
                proposal_id=p.proposal_id,
                metric_id=p.proposed_metric_id,
                display_name=(f"{metric.display_name}"
                              f"{self._qualifier(p)}" if metric else
                              p.extracted_test_name),
                category=metric.category if metric else p.extracted_category,
                operator=p.extracted_operator,
                threshold=p.extracted_threshold,
                unit=p.extracted_unit or (metric.unit if metric else "percent"),
                parameters=dict(p.parameters),
                effective_date=approval.effective_date or p.effective_date,
                expiry_date=p.expiry_date,
                evidence=SourceEvidence(
                    source_reference=p.evidence.source_reference,
                    source_text=p.evidence.source_text,
                    locator=p.evidence.locator,
                    supplied_at=p.evidence.supplied_at),
                approval=approval,
                severity=p.severity or "high",
                definition_notes=self._definition_notes(p),
            ))
        config = ActiveConfiguration(
            client_id=client_id, tests=tests, activated_by=actor,
            activated_at=now_iso(), reason=reason,
            library_version=self.lib.library_version)
        config = self.store.commit_configuration(client_id, config)
        self.ops.append_audit(
            client_id, "concentration_configuration_activated", actor=actor,
            detail={"version": config.version,
                    "content_hash": config.content_hash,
                    "test_count": len(config.tests),
                    "reason": reason,
                    "proposal_ids": [t.proposal_id for t in config.tests]})
        return config

    # ------------------------------------------------------------------ #
    def review_package(self, client_id: str) -> Dict[str, Any]:
        """Everything an operator needs to review, in one document."""
        proposals = self.store.list_proposals(client_id)
        current = self.store.current_configuration(client_id)
        by_status: Dict[str, int] = {}
        for p in proposals:
            by_status[p.status] = by_status.get(p.status, 0) + 1
        return {
            "clientId": client_id,
            "proposals": [p.to_dict() for p in proposals],
            "countsByStatus": by_status,
            "currentVersion": current.version if current else None,
            "currentTestCount": len(current.tests) if current else 0,
            "versions": self.store.list_versions(client_id),
            "libraryVersion": self.lib.library_version,
        }

    @staticmethod
    def _qualifier(p: TestProposal) -> str:
        params = p.parameters or {}
        for key in ("regions", "countries", "counties", "areas", "values",
                    "years"):
            if params.get(key):
                return " — " + " + ".join(str(v) for v in params[key])
        if params.get("amount") is not None:
            comparison = params.get("comparison", "above")
            return f" — {comparison} {params['amount']:,.0f}"
        if params.get("age") is not None:
            return f" — {params.get('comparison', '')} {params['age']:.0f}".rstrip()
        if params.get("n") is not None:
            return f" — top {params['n']}"
        if params.get("min_loans") is not None:
            return f" — {params['min_loans']}+ loans"
        return ""

    @staticmethod
    def _definition_notes(p: TestProposal) -> str:
        answers = [f"{a.get('question')} {a.get('answer')}"
                   for a in p.confirmation_answers]
        return " ".join(answers)

    @staticmethod
    def _note(proposal: TestProposal, actor: str, note: str) -> None:
        proposal.operator_notes.append(
            {"at": now_iso(), "actor": actor, "note": note})
