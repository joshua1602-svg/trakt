"""operations_control.onboarding.service — the governed onboarding capability.

Three entry points, deliberately distinct:

``start_new_client()``   the product. Opens a blank case with a system-generated
                         reference and no client selected. Nothing is read from
                         any existing configuration.
``start_migration()``    secondary. Opens a case pre-populated from a legacy
                         client's files, for review and adoption.
``start_amendment()``    for an active client. Opens a case pre-populated from
                         its current approved version.

They differ only in where the answers start. The model, the validation, the
information requests, the generation and the approval are identical — a migrated
client and a newly onboarded one are the same client afterwards.

No active configuration is written before ACTIVATION. Approval records the
decision; activation performs it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..contracts import new_id, now_iso
from ..engine import OpsError
from ..stores import OpsStore
from . import artefacts as _artefacts
from . import derivation, inference, migration
from .case import (
    ACTIVATED,
    APPROVED,
    AWAITING_CLIENT,
    CHANGES_REQUIRED,
    DRAFT,
    INFORMATION_REQUESTED,
    IN_REVIEW,
    KIND_AMENDMENT,
    KIND_LABELS,
    KIND_MIGRATION,
    KIND_NEW_CLIENT,
    READY_FOR_APPROVAL,
    REQUEST_ACCEPTED,
    REQUEST_ANSWERED,
    REQUEST_OPEN,
    REQUEST_REJECTED,
    REQUEST_SENT,
    STATUS_LABELS,
    STATUSES,
    WITHDRAWN,
    CaseError,
    InformationRequest,
    OnboardingCase,
    new_entity_id,
    new_request_id,
    source_key,
)
from .catalogue import Catalogue, catalogue, reset_cache  # noqa: F401
from .store import OnboardingStore
from .validation import Validator

#: Wizard steps, in order. Each maps to one catalogue section except review.
STEPS = ("client", "entities", "contacts", "portfolios", "sources",
         "reporting", "regime", "presentation", "review")

STEP_LABELS = {
    "client": "About the client",
    "entities": "Legal and reporting entities",
    "contacts": "Contacts and distribution",
    "portfolios": "Portfolios",
    "sources": "Expected deliveries",
    "reporting": "Reporting requirements",
    "regime": "Regulatory information",
    "presentation": "Report presentation",
    "review": "Review and activate",
}


class OnboardingService:
    def __init__(self, store: OpsStore, *, registry_factory=None,
                 adopt_paths: Optional[Dict[str, Any]] = None,
                 cat: Optional[Catalogue] = None):
        self.store = store
        self.cases = OnboardingStore(store)
        self.catalogue = cat or catalogue()
        self._registry_factory = registry_factory
        self._adopt_paths = adopt_paths or {}

    # ------------------------------------------------------------------ #
    def _registry(self):
        if self._registry_factory is not None:
            return self._registry_factory()
        try:
            from ..classification import open_source_registry
            return open_source_registry(storage=self.store.storage)
        except Exception:      # noqa: BLE001 — onboarding still works read-only
            return None

    def _validator(self) -> Validator:
        """A validator that knows what identifiers are already taken."""
        clients = set(self.store.known_clients())
        portfolios: set = set()
        registry = self._registry()
        if registry is not None:
            for record in registry.records():
                clients.add(record.client_id)
                portfolios.add(f"{record.client_id}/{record.source_portfolio_id}")
        for client_id in self.cases.onboarded_clients():
            clients.add(client_id)
            current = self.cases.current(client_id)
            for p in ((current.answers.get("portfolios") if current else None)
                      or []):
                portfolios.add(f"{client_id}/{p.get('portfolio_id')}")
        return Validator(self.catalogue, existing_clients=clients,
                         existing_portfolios=portfolios)

    def reference(self) -> Dict[str, Any]:
        """Everything the browser needs to render the wizard."""
        return {"catalogue": self.catalogue.to_dict(),
                "steps": [{"key": s, "label": STEP_LABELS[s]} for s in STEPS],
                "statuses": [{"key": s, "label": STATUS_LABELS[s]}
                             for s in STATUSES],
                "kinds": [{"key": k, "label": v} for k, v in KIND_LABELS.items()]}

    # ------------------------------------------------------------------ #
    # Opening a case
    # ------------------------------------------------------------------ #
    def _open(self, *, kind: str, by: str, answers: Optional[Dict[str, Any]] = None,
              client_id: str = "", client_name: str = "",
              provenance: Optional[Dict[str, str]] = None,
              base_documents: Optional[Dict[str, Any]] = None,
              based_on_version: Optional[int] = None) -> OnboardingCase:
        year = datetime.now(timezone.utc).year
        case = OnboardingCase(
            case_id=self.cases.next_case_id(year), kind=kind, status=DRAFT,
            client_id=client_id, client_name=client_name,
            answers=answers or {}, provenance=provenance or {},
            base_documents=base_documents or {},
            based_on_version=based_on_version,
            created_by=by, created_at=now_iso(), updated_by=by,
            updated_at=now_iso())
        case.record("opened", actor=by,
                    detail={"kind": kind,
                            "based_on_version": based_on_version})
        self._sync_derived(case)
        self.cases.save_case(case)
        if client_id:
            self.store.append_audit(client_id, "onboarding_case_opened",
                                    actor=by, detail={"case_id": case.case_id,
                                                      "kind": kind})
        return case

    def start_new_client(self, *, by: str) -> OnboardingCase:
        """The primary journey. Blank: no client selected, nothing read."""
        return self._open(kind=KIND_NEW_CLIENT, by=by, answers={})

    def start_migration(self, *, client_id: str, by: str) -> OnboardingCase:
        """Secondary. Pre-populate the same model from a legacy client's files."""
        if not client_id:
            raise OpsError("OPS_CLIENT_REQUIRED",
                           "Choose which existing client to bring in.", 400)
        adoption = migration.adopt(client_id, catalogue=self.catalogue,
                                   registry=self._registry(),
                                   **self._adopt_paths)
        case = self._open(kind=KIND_MIGRATION, by=by, answers=adoption.answers,
                          client_id=client_id,
                          client_name=str((adoption.answers.get("client") or {})
                                          .get("client_name") or client_id),
                          provenance=adoption.provenance,
                          base_documents=adoption.base_documents)
        case.record("legacy_configuration_read", actor=by,
                    detail={"sources_read": adoption.sources_read,
                            "issues": adoption.issues})
        # Legacy values that fail today's validation are surfaced immediately as
        # migration issues rather than discovered at approval.
        for issue in adoption.issues:
            case.open_questions.append({
                "question_id": new_id("q"), "question": issue["message"],
                "field": issue.get("field", ""), "origin": "migration",
                "raised_at": now_iso(), "raised_by": "Trakt"})
        self.cases.save_case(case)
        return case

    def start_amendment(self, *, client_id: str, by: str) -> OnboardingCase:
        """Change an active client's configuration, starting from what is in
        force. Never a destructive edit of the generated files."""
        current = self.cases.current(client_id)
        if current is None:
            raise OpsError(
                "OPS_CLIENT_NOT_ONBOARDED",
                "This client has no approved configuration to amend.", 409)
        case = self._open(kind=KIND_AMENDMENT, by=by,
                          answers=_deep_copy(current.answers),
                          client_id=client_id,
                          client_name=str((current.answers.get("client") or {})
                                          .get("client_name") or client_id),
                          based_on_version=current.version)
        return case

    # ------------------------------------------------------------------ #
    # Answering
    # ------------------------------------------------------------------ #
    def load_case(self, case_id: str) -> OnboardingCase:
        case = self.cases.load_case(case_id)
        if case is None:
            raise OpsError("OPS_NOT_FOUND", "That could not be found.", 404)
        return case

    def save_step(self, *, case_id: str, step: str, payload: Dict[str, Any],
                  by: str) -> OnboardingCase:
        if step not in STEPS:
            raise OpsError("OPS_BAD_STEP",
                           "That is not a step of this onboarding.", 400)
        case = self.load_case(case_id)
        self._require_editable(case)
        section = self.catalogue.section(step)
        before = _deep_copy(case.answers.get(step))

        if section is not None and section.from_regime:
            # Regime answers are held per product, so a client who later drops a
            # product does not lose the answers if they add it back.
            held = case.block("regime")
            for product, answers in (payload.get("regime") or {}).items():
                held[product] = {**(held.get(product) or {}), **(answers or {})}
        elif section is not None and section.repeatable:
            case.answers[step] = list(payload.get(section.key)
                                      or payload.get("items") or [])
        elif section is not None:
            block = case.block(step)
            block.update({k: v for k, v in (payload or {}).items()})

        # Inference runs FIRST: it may propose the client identifier, and the
        # case has to take its identity from the answers as they finally stand,
        # not as they arrived.
        # Anything the operator just sent is theirs, including a deliberate
        # blank; inference must not overwrite it on this pass.
        self._sync_derived(case, touched={step: set((payload or {}).keys())})
        if step == "client":
            client = case.block("client")
            case.client_id = str(client.get("client_id") or "")
            case.client_name = str(client.get("client_name") or "")
        case.record(f"answered_{step}", actor=by,
                    before={step: before},
                    after={step: _deep_copy(case.answers.get(step))})
        self.cases.save_case(case)
        return case

    def _sync_derived(self, case: OnboardingCase,
                      touched: Optional[Dict[str, set]] = None) -> None:
        """Keep everything Trakt works out for itself in step with the answers.

        Runs on every save, so an answer given late still reaches the values
        that follow from it — naming a jurisdiction fills in the currency and
        time zone, adding the originator entity fills the Annex 2 originator
        block, and choosing a cadence sets the period convention.
        """
        derivation.apply_derived_defaults(case, self.catalogue)
        for entity in case.items("entities"):
            if not entity.get("entity_id"):
                entity["entity_id"] = new_entity_id()
        case.answers["sources"] = derivation.derive_sources(case)
        inference.apply(case, self.catalogue,
                        existing_clients=self._identifiers_in_use(),
                        touched=touched)

    def _identifiers_in_use(self) -> set:
        """Client identifiers already taken, so a proposal never collides."""
        taken = set(self.store.known_clients()) | set(
            self.cases.onboarded_clients())
        registry = self._registry()
        if registry is not None:
            taken |= {r.client_id for r in registry.records()}
        return taken

    def add_pipeline_source(self, *, case_id: str, portfolio_id: str,
                            by: str) -> OnboardingCase:
        """Record that a portfolio also delivers a pipeline book."""
        case = self.load_case(case_id)
        self._require_editable(case)
        if case.portfolio(portfolio_id) is None:
            raise OpsError("OPS_NOT_FOUND", "That could not be found.", 404)
        derivation.add_pipeline_source(case, portfolio_id)
        self._sync_derived(case)
        case.record("pipeline_book_added", actor=by,
                    detail={"portfolio_id": portfolio_id})
        self.cases.save_case(case)
        return case

    def register_sample(self, *, case_id: str, files: List[Dict[str, Any]],
                        by: str) -> OnboardingCase:
        """Record a sample pack the client has supplied.

        ``files`` is ``[{"name": ..., "headers": [...]}]`` — what the existing
        intake already extracts when it reads a pack. Registering one lets Trakt
        answer the file format, the expected file names and often the asset
        class, instead of asking for them.
        """
        case = self.load_case(case_id)
        self._require_editable(case)
        case.answers["sample"] = {"files": files,
                                  "registered_by": by,
                                  "registered_at": now_iso()}
        self._sync_derived(case)
        inferred = inference.infer_from_sample(case.answers["sample"])
        case.record("sample_registered", actor=by,
                    detail={"files": [f.get("name") for f in files],
                            "inferred": sorted(inferred)})
        self.cases.save_case(case)
        return case

    def remove_source(self, *, case_id: str, key: str,
                      by: str) -> OnboardingCase:
        case = self.load_case(case_id)
        self._require_editable(case)
        case.answers["sources"] = [
            s for s in case.items("sources")
            if source_key(str(s.get("portfolio_id") or ""),
                          str(s.get("dataset") or "")) != key]
        case.record("delivery_removed", actor=by, detail={"source": key})
        self.cases.save_case(case)
        return case

    def _require_editable(self, case: OnboardingCase) -> None:
        if case.status in (ACTIVATED, WITHDRAWN, APPROVED):
            raise OpsError(
                "OPS_ONBOARDING_LOCKED",
                f"This onboarding is {STATUS_LABELS[case.status].lower()} and "
                "can no longer be edited.", 409)

    # ------------------------------------------------------------------ #
    # Information requests
    # ------------------------------------------------------------------ #
    def client_checklist(self, case: OnboardingCase) -> List[Dict[str, Any]]:
        """What the CLIENT still has to tell us.

        Derived from the catalogue's conditional requirements and restricted to
        client-supplied fields, so the list is something that can actually be
        sent out rather than a dump of everything unanswered.
        """
        already = {(i.get("section"), i.get("field"), i.get("index"))
                   for r in case.requests() for i in r.items
                   if r.status not in (REQUEST_REJECTED,)}
        return [row for row in self.catalogue.outstanding_for_client(case.answers)
                if (row["section"], row["field"], row["index"]) not in already]

    def create_request(self, *, case_id: str, items: List[Dict[str, Any]],
                       by: str, responsible_party: str = "client",
                       due_date: str = "", note: str = "") -> OnboardingCase:
        case = self.load_case(case_id)
        self._require_editable(case)
        if not items:
            raise OpsError("OPS_NOTHING_TO_REQUEST",
                           "Choose at least one item to ask for.", 400)
        request = InformationRequest(
            request_id=new_request_id(), items=items,
            responsible_party=responsible_party, status=REQUEST_OPEN,
            requested_by=by, requested_at=now_iso(), due_date=due_date,
            review_note=note)
        case.save_request(request)
        case.record("information_requested", actor=by,
                    detail={"request_id": request.request_id,
                            "items": len(items),
                            "responsible_party": responsible_party})
        if case.status == DRAFT:
            case.transition(INFORMATION_REQUESTED, actor=by,
                            reason="Information requested from "
                                   f"{responsible_party}")
        self.cases.save_case(case)
        return case

    def mark_request_sent(self, *, case_id: str, request_id: str,
                          by: str) -> OnboardingCase:
        case = self.load_case(case_id)
        request = self._require_request(case, request_id)
        request.status = REQUEST_SENT
        request.sent_at = now_iso()
        case.save_request(request)
        case.record("information_request_sent", actor=by,
                    detail={"request_id": request_id})
        if case.status == INFORMATION_REQUESTED:
            case.transition(AWAITING_CLIENT, actor=by)
        self.cases.save_case(case)
        return case

    def record_response(self, *, case_id: str, request_id: str, by: str,
                        note: str = "",
                        answers: Optional[Dict[str, Any]] = None,
                        evidence: Optional[List[Dict[str, str]]] = None
                        ) -> OnboardingCase:
        """Record what came back, and apply any values supplied with it."""
        case = self.load_case(case_id)
        request = self._require_request(case, request_id)
        request.status = REQUEST_ANSWERED
        request.response_note = note
        request.responded_at = now_iso()
        request.responded_by = by
        for item in (evidence or []):
            request.evidence.append({**item, "received_at": now_iso(),
                                     "received_by": by})
        case.save_request(request)
        for section, values in (answers or {}).items():
            if isinstance(values, list):
                case.answers[section] = values
            else:
                case.block(section).update(values or {})
        self._sync_derived(case)
        case.record("information_received", actor=by,
                    detail={"request_id": request_id,
                            "evidence": len(request.evidence)})
        if case.status == AWAITING_CLIENT:
            case.transition(IN_REVIEW, actor=by)
        self.cases.save_case(case)
        return case

    def review_response(self, *, case_id: str, request_id: str, accept: bool,
                        by: str, note: str = "") -> OnboardingCase:
        case = self.load_case(case_id)
        request = self._require_request(case, request_id)
        request.status = REQUEST_ACCEPTED if accept else REQUEST_REJECTED
        request.reviewed_by = by
        request.reviewed_at = now_iso()
        request.review_note = note
        case.save_request(request)
        case.record("information_reviewed", actor=by,
                    detail={"request_id": request_id, "accepted": accept,
                            "note": note})
        self.cases.save_case(case)
        return case

    @staticmethod
    def _require_request(case: OnboardingCase,
                         request_id: str) -> InformationRequest:
        request = case.request(request_id)
        if request is None:
            raise OpsError("OPS_NOT_FOUND", "That could not be found.", 404)
        return request

    def add_question(self, *, case_id: str, question: str,
                     by: str) -> OnboardingCase:
        case = self.load_case(case_id)
        self._require_editable(case)
        case.open_questions.append({
            "question_id": new_id("q"), "question": question,
            "origin": "operator", "raised_at": now_iso(), "raised_by": by})
        case.record("question_raised", actor=by, detail={"question": question})
        self.cases.save_case(case)
        return case

    def resolve_question(self, *, case_id: str, question_id: str,
                         resolution: str, by: str) -> OnboardingCase:
        case = self.load_case(case_id)
        for q in case.open_questions:
            if q.get("question_id") == question_id:
                q["resolved_at"] = now_iso()
                q["resolved_by"] = by
                q["resolution"] = resolution
        case.record("question_resolved", actor=by,
                    detail={"question_id": question_id,
                            "resolution": resolution})
        self.cases.save_case(case)
        return case

    # ------------------------------------------------------------------ #
    # Readiness, approval, activation
    # ------------------------------------------------------------------ #
    def readiness(self, case: OnboardingCase) -> Dict[str, Any]:
        validator = self._validator()
        problems = validator.validate(case)
        blocking = [p for p in problems if p.severity == "blocking"]
        by_step: Dict[str, List[Dict[str, Any]]] = {s: [] for s in STEPS}
        for p in problems:
            by_step.setdefault(p.section, []).append(p.to_dict())
        checklist = self.client_checklist(case)
        outstanding = case.outstanding_requests
        return {
            "problems": [p.to_dict() for p in problems],
            "blocking": [p.to_dict() for p in blocking],
            "by_step": by_step,
            "client_checklist": checklist,
            "outstanding_requests": [r.to_dict() for r in outstanding],
            "unresolved_questions": case.unresolved_questions,
            "ready": not blocking and not outstanding,
        }

    def submit_for_approval(self, *, case_id: str, by: str) -> OnboardingCase:
        case = self.load_case(case_id)
        readiness = self.readiness(case)
        if not readiness["ready"]:
            message = (readiness["blocking"][0]["message"]
                       if readiness["blocking"]
                       else "Information is still outstanding from the client.")
            raise OpsError("OPS_ONBOARDING_INCOMPLETE", message, 409)
        if case.status not in (READY_FOR_APPROVAL,):
            case.transition(READY_FOR_APPROVAL, actor=by)
        self.cases.save_case(case)
        return case

    def request_changes(self, *, case_id: str, by: str,
                        reason: str) -> OnboardingCase:
        if not str(reason or "").strip():
            raise OpsError("OPS_REASON_REQUIRED",
                           "Please say what needs to change.", 400)
        case = self.load_case(case_id)
        case.transition(CHANGES_REQUIRED, actor=by, reason=reason)
        self.cases.save_case(case)
        return case

    def preview(self, case: OnboardingCase) -> Dict[str, Any]:
        """Exactly what activation would create or change. Writes nothing."""
        registry = self._registry()
        planned = _artefacts.plan(case, store=self.cases, cat=self.catalogue,
                                  registry=registry)
        current = (self.cases.current(case.client_id) if case.client_id
                   else None)
        changes = diff_answers(current.answers if current else None,
                               case.answers)
        return {
            "case_id": case.case_id,
            "client_id": case.client_id,
            "artefacts": [a.to_dict() for a in planned],
            "changes": changes,
            "current_version": current.version if current else 0,
            "next_version": (self.cases.next_version(case.client_id)
                             if case.client_id else 1),
            "unrepresented": _artefacts.unrepresented(case, self.catalogue),
            "generated_identifiers": self._generated_identifiers(case),
            "defaults_used": self._defaults_used(case),
            **self.readiness(case),
        }

    def _generated_identifiers(self, case: OnboardingCase) -> List[Dict[str, str]]:
        out = [{"label": "Onboarding reference", "value": case.case_id}]
        for entity in case.items("entities"):
            out.append({"label": f"Entity reference — "
                                 f"{entity.get('legal_name', '')}",
                        "value": str(entity.get("entity_id") or "")})
        for source in case.items("sources"):
            if source.get("expected_location"):
                out.append({"label": f"Delivery location — "
                                     f"{source.get('portfolio_id')} "
                                     f"{source.get('dataset')}",
                            "value": str(source["expected_location"])})
        return out

    def _defaults_used(self, case: OnboardingCase) -> List[Dict[str, str]]:
        """Where the value in force is the governed default.

        Reported by comparing the answer to the declaration rather than by
        looking for a gap: defaults are filled in as the case is answered, so by
        the time anyone previews it there is no gap left to find. What the
        approver needs to know is which values nobody chose.
        """
        out: List[Dict[str, str]] = []
        for section in self.catalogue.sections:
            block = case.answers.get(section.key) or {}
            for f in section.fields:
                if f.default is None:
                    continue
                if section.repeatable:
                    holders = [(item, section.label)
                               for item in case.items(section.key)]
                elif section.from_regime:
                    if f.product and f.product not in case.products:
                        continue
                    holders = [(block.get(f.product) or {}, section.label)]
                else:
                    holders = [(block, section.label)]
                for holder, label in holders:
                    if str(holder.get(f.key) or "") == str(f.default):
                        out.append({"label": f.label, "value": str(f.default),
                                    "section": label})
        return out

    def approve(self, *, case_id: str, by: str, reason: str) -> OnboardingCase:
        """Record the decision. Writes NO configuration."""
        if not str(reason or "").strip():
            raise OpsError("OPS_REASON_REQUIRED",
                           "Please say why this is being approved.", 400)
        case = self.load_case(case_id)
        # A finished case is refused for what it IS. Everything else falls
        # through to readiness, which explains what is actually missing —
        # "information is still outstanding" is far more use to an operator
        # than "this cannot be approved from its current status".
        if case.status in (ACTIVATED, WITHDRAWN):
            raise CaseError(
                "OPS_ONBOARDING_BAD_TRANSITION",
                f"This onboarding is {STATUS_LABELS[case.status].lower()} and "
                "can no longer be approved.", 409)
        readiness = self.readiness(case)
        if not readiness["ready"]:
            message = (readiness["blocking"][0]["message"]
                       if readiness["blocking"]
                       else "Information is still outstanding from the client.")
            raise OpsError("OPS_ONBOARDING_INCOMPLETE", message, 409)
        if case.status != READY_FOR_APPROVAL:
            case.transition(READY_FOR_APPROVAL, actor=by)
        case.approved_by = by
        case.approved_at = now_iso()
        case.approval_reason = reason
        case.transition(APPROVED, actor=by, reason=reason)
        self.cases.save_case(case)
        if case.client_id:
            self.store.append_audit(case.client_id, "onboarding_case_approved",
                                    actor=by,
                                    detail={"case_id": case.case_id,
                                            "reason": reason})
        return case

    def activate(self, *, case_id: str, by: str) -> Dict[str, Any]:
        """Write the configuration. The only place active configuration is
        created, and only from an approved case."""
        case = self.load_case(case_id)
        if case.status != APPROVED:
            raise OpsError(
                "OPS_ONBOARDING_NOT_APPROVED",
                "This onboarding has not been approved yet.", 409)
        if not case.client_id:
            raise OpsError("OPS_CLIENT_REQUIRED",
                           "The client needs an identifier before its "
                           "configuration can be created.", 400)

        registry = self._registry()
        planned = _artefacts.plan(case, store=self.cases, cat=self.catalogue,
                                  registry=registry)
        current = self.cases.current(case.client_id)
        changes = diff_answers(current.answers if current else None,
                               case.answers)
        version_number = self.cases.next_version(case.client_id)
        written = _artefacts.apply(case, store=self.cases,
                                   version=version_number, registry=registry,
                                   cat=self.catalogue, artefacts=planned)
        version = self.cases.commit(case=case, changes=changes,
                                    artefacts=written, activated_by=by)
        case.activated_by = by
        case.activated_at = now_iso()
        case.activated_version = version.version
        case.transition(ACTIVATED, actor=by,
                        detail={"version": version.version})
        self.cases.save_case(case)
        self.store.append_audit(
            case.client_id, "onboarding_configuration_activated", actor=by,
            detail={"case_id": case.case_id, "kind": case.kind,
                    "version": version.version, "reason": case.approval_reason,
                    "changes": changes, "artefacts": written,
                    "content_hash": version.content_hash})
        return {"case_id": case.case_id, "client_id": case.client_id,
                "version": version.version, "artefacts": written,
                "changes": changes}

    def withdraw(self, *, case_id: str, by: str, reason: str) -> OnboardingCase:
        """End a case without creating anything. The record is kept."""
        if not str(reason or "").strip():
            raise OpsError("OPS_REASON_REQUIRED",
                           "Please say why this is being withdrawn.", 400)
        case = self.load_case(case_id)
        case.withdrawn_by = by
        case.withdrawn_at = now_iso()
        case.withdrawal_reason = reason
        case.transition(WITHDRAWN, actor=by, reason=reason)
        self.cases.save_case(case)
        if case.client_id:
            self.store.append_audit(case.client_id, "onboarding_case_withdrawn",
                                    actor=by, detail={"case_id": case.case_id,
                                                      "reason": reason})
        return case

    # ------------------------------------------------------------------ #
    # Views
    # ------------------------------------------------------------------ #
    def home(self, *, visible: Optional[List[str]] = None) -> Dict[str, Any]:
        """The queues an operator works from."""
        cases = self.cases.list_cases()
        if visible is not None:
            cases = [c for c in cases
                     if not c.client_id or c.client_id in visible]
        def rows(*statuses: str) -> List[Dict[str, Any]]:
            return [self._case_row(c) for c in cases if c.status in statuses]

        active_clients: List[Dict[str, Any]] = []
        for client_id in self.cases.onboarded_clients():
            if visible is not None and client_id not in visible:
                continue
            current = self.cases.current(client_id)
            if current is None:
                continue
            client = current.answers.get("client") or {}
            active_clients.append({
                "client_id": client_id,
                "display_name": client.get("client_name") or client_id,
                "version": current.version,
                "portfolios": len(current.answers.get("portfolios") or []),
                "products": list((current.answers.get("reporting") or {})
                                 .get("products") or []),
                "activated_at": current.activated_at,
                "activated_by": current.activated_by})

        return {
            "drafts": rows(DRAFT, CHANGES_REQUIRED),
            "awaiting_client": rows(INFORMATION_REQUESTED, AWAITING_CLIENT),
            "in_review": rows(IN_REVIEW, READY_FOR_APPROVAL),
            "approved": rows(APPROVED),
            "active_clients": sorted(active_clients,
                                     key=lambda c: c["display_name"].lower()),
            "legacy_clients": self._legacy_clients(visible),
            "recently_completed": rows(ACTIVATED, WITHDRAWN)[:10],
        }

    def _legacy_clients(self, visible: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Clients Trakt already serves that have never been through
        onboarding. Offered for migration — never as the way to start."""
        onboarded = set(self.cases.onboarded_clients())
        known: List[str] = []
        registry = self._registry()
        if registry is not None:
            for record in registry.records():
                if record.client_id not in known:
                    known.append(record.client_id)
        for client_id in self.store.known_clients():
            if client_id not in known:
                known.append(client_id)
        out = []
        for client_id in sorted(known):
            if client_id in onboarded:
                continue
            if visible is not None and client_id not in visible:
                continue
            out.append({"client_id": client_id, "display_name": client_id})
        return out

    def _case_row(self, case: OnboardingCase) -> Dict[str, Any]:
        return {
            "case_id": case.case_id,
            "kind": case.kind,
            "kind_label": KIND_LABELS.get(case.kind, case.kind),
            "status": case.status,
            "status_label": STATUS_LABELS.get(case.status, case.status),
            "client_id": case.client_id,
            "client_name": case.client_name or case.client_id or "Not yet named",
            "portfolios": len(case.answers.get("portfolios") or []),
            "outstanding_requests": len(case.outstanding_requests),
            "unresolved_questions": len(case.unresolved_questions),
            "updated_at": case.updated_at,
            "updated_by": case.updated_by,
            "created_by": case.created_by,
        }

    def present_case(self, case: OnboardingCase) -> Dict[str, Any]:
        readiness = self.readiness(case)
        return {
            **self._case_row(case),
            "answers": case.answers,
            "provenance": case.provenance,
            "based_on_version": case.based_on_version,
            "information_requests": case.information_requests,
            "open_questions": case.open_questions,
            "events": case.events,
            "product_eligibility": derivation.product_eligibility(
                case, self.catalogue),
            "steps": [{"key": s, "label": STEP_LABELS[s],
                       "problems": len(readiness["by_step"].get(s) or [])}
                      for s in STEPS],
            "approved_by": case.approved_by,
            "approved_at": case.approved_at,
            "approval_reason": case.approval_reason,
            "activated_version": case.activated_version,
            "withdrawal_reason": case.withdrawal_reason,
            **readiness,
        }

    def client_view(self, client_id: str) -> Dict[str, Any]:
        current = self.cases.current(client_id)
        registry = self._registry()
        live = ([r.to_dict() for r in registry.records()
                 if r.client_id == client_id] if registry else [])
        cases = [self._case_row(c)
                 for c in self.cases.list_cases(client_id=client_id)]
        if current is None:
            return {"client_id": client_id, "status": "not_onboarded",
                    "version": 0, "answers": None,
                    "live_source_registrations": live, "history": [],
                    "cases": cases,
                    "summary": "This client has no approved configuration in "
                               "Trakt."}
        return {
            "client_id": client_id, "status": "active",
            "version": current.version,
            "activated_by": current.activated_by,
            "activated_at": current.activated_at,
            "reason": current.reason,
            "answers": current.answers,
            "live_source_registrations": live,
            "artefacts": current.artefacts,
            "cases": cases,
            "history": [self._version_row(v)
                        for v in self.cases.history(client_id)],
        }

    @staticmethod
    def _version_row(v) -> Dict[str, Any]:
        return {"version": v.version, "status": v.status,
                "activated_by": v.activated_by, "activated_at": v.activated_at,
                "approved_by": v.approved_by, "approved_at": v.approved_at,
                "reason": v.reason, "case_id": v.case_id,
                "case_kind": v.case_kind,
                "based_on_version": v.based_on_version,
                "change_count": len(v.changes), "changes": v.changes,
                "artefacts": v.artefacts, "content_hash": v.content_hash}


def _deep_copy(value: Any) -> Any:
    import copy
    return copy.deepcopy(value) if value is not None else {}


# --------------------------------------------------------------------------- #
# Field-level change detection
# --------------------------------------------------------------------------- #

def _flatten(node: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(node, dict):
        for k, v in node.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(node, list):
        if all(not isinstance(v, (dict, list)) for v in node):
            out[prefix] = ", ".join(str(v) for v in node)
        else:
            for i, v in enumerate(node):
                key = str(i)
                if isinstance(v, dict):
                    key = str(v.get("portfolio_id") or v.get("source_key")
                              or v.get("legal_name") or v.get("entity_id") or i)
                out.update(_flatten(v, f"{prefix}[{key}]"))
    else:
        out[prefix] = node
    return out


#: Values that restate another, or that Trakt mints — noise in a change list.
_SUPPRESSED = ("entity_id", "source_key", "expected_location")


def diff_answers(before: Optional[Dict[str, Any]],
                 after: Dict[str, Any]) -> List[Dict[str, Any]]:
    old = _flatten(before or {})
    new = _flatten(after or {})
    changes: List[Dict[str, Any]] = []
    for path in sorted(set(old) | set(new)):
        if any(path.endswith(f".{s}") or path == s for s in _SUPPRESSED):
            continue
        was, now = old.get(path), new.get(path)
        if was == now:
            continue
        if (was is None or str(was) == "") and (now is None or str(now) == ""):
            continue
        changes.append({
            "path": path, "label": _label(path),
            "before": "" if was is None else str(was),
            "after": "" if now is None else str(now),
            "action": ("added" if was in (None, "") else
                       "removed" if now in (None, "") else "changed")})
    return changes


_LABELS = {
    "client.client_name": "Client name",
    "client.client_id": "Client identifier",
    "client.jurisdiction": "Jurisdiction",
    "client.reporting_currency": "Reporting currency",
    "reporting.products": "Reporting products",
}


def _label(path: str) -> str:
    if path in _LABELS:
        return _LABELS[path]
    return path.replace("_", " ").replace(".", " → ")
