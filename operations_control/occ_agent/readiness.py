"""operations_control.occ_agent.readiness — the READY_FOR_EXECUTION decision.

Readiness is **derived**, never declared. :func:`evaluate` walks the criteria one
at a time, each answered from persisted state or from a deterministic control
result, and returns a criterion-by-criterion verdict. The service only permits
the transition to ``READY_FOR_EXECUTION`` when every criterion passes; there is
no code path by which an interpreter, a confidence score or a chat message can
set it.

Half the criteria are not this feature's to judge. Whether the client is
adequately described, whether anything is still outstanding from them, and
whether the configuration would generate cleanly are all questions Client
Onboarding already answers — through ``OnboardingService.readiness()`` and
``OnboardingService.preview()`` — and they are read from there rather than
re-derived. What this module adds is the execution half: the files, the
mappings, the controls, the plan, and the boundary.

One criterion exists only because the case is synthetic: **no configuration was
written.** A practice case reaches readiness having created nothing, and if it
somehow had, readiness fails rather than reporting success.

:func:`build_package` then produces the readiness package, including a
machine-readable execution manifest. The manifest is deterministic: the same run
produces byte-identical content, because nothing volatile (timestamps, absolute
paths, run identifiers) enters the hashed body.

Wording is fixed here too. A ready case says ``READY_FOR_EXECUTION`` and states
plainly what did NOT happen; the words "complete", "live", "published" and
"production successful" are deliberately absent, and a test asserts it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..contracts import canonical_json, stable_hash
from ..onboarding.case import (
    ACTIVATED,
    APPROVED,
    NO_ACTIVE_CONFIGURATION,
    STATUS_LABELS,
    OnboardingCase,
)
from . import states as _states
from .derive import ExecutionFacts, product_label
from .input_roles import artefact_vocabulary
from .policy import RUNTIME_MODE_SYNTHETIC, SyntheticPolicy
from .run import (
    STAGE_DETERMINISTIC_COMPLETED,
    STAGE_HARD_BLOCKED,
    STAGE_SIMULATED,
    SyntheticRun,
)

#: The five sentences the UI must be able to state for a ready case.
NOT_DONE_STATEMENTS = (
    "No live files were written.",
    "No production pipeline was triggered.",
    "No external email was sent.",
    "No client configuration was activated.",
    "Nothing was published.",
)

READY_HEADLINE = "Practice case ready for execution."

#: Words that must never describe a synthetic case's outcome.
FORBIDDEN_OUTCOME_WORDS = ("complete", "live", "published",
                           "production successful")


@dataclass
class Criterion:
    """One readiness criterion and its deterministic verdict."""

    key: str
    label: str
    passed: bool
    detail: str = ""
    #: What the operator must do about it, when it has not passed.
    remedy: str = ""
    #: Which half of the process the criterion belongs to, so the UI can show
    #: an operator whether the work is theirs or the client's.
    stage: str = "execution"        # onboarding | execution | boundary

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "label": self.label, "passed": self.passed,
                "detail": self.detail, "remedy": self.remedy,
                "stage": self.stage}


@dataclass
class ReadinessVerdict:
    criteria: List[Criterion] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return bool(self.criteria) and all(c.passed for c in self.criteria)

    @property
    def outstanding(self) -> List[Criterion]:
        return [c for c in self.criteria if not c.passed]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ready": self.ready,
            "status": (_states.READY_FOR_EXECUTION if self.ready
                       else "NOT_READY"),
            "criteria": [c.to_dict() for c in self.criteria],
            "outstanding": [c.to_dict() for c in self.outstanding],
        }


def evaluate(run: SyntheticRun, case: OnboardingCase, facts: ExecutionFacts,
             policy: SyntheticPolicy, *,
             onboarding: Optional[Dict[str, Any]] = None,
             preview: Optional[Dict[str, Any]] = None) -> ReadinessVerdict:
    """Every criterion, each answered from state or a deterministic control.

    ``onboarding`` is ``OnboardingService.readiness(case)`` and ``preview`` is
    ``OnboardingService.preview(case)``. Both are passed in rather than computed
    here so that readiness cannot disagree with what the Client Onboarding
    screens show the same operator.
    """
    onboarding = onboarding or {}
    preview = preview or {}
    roles = artefact_vocabulary()
    criteria: List[Criterion] = []

    def add(key: str, label: str, passed: bool, detail: str = "",
            remedy: str = "", stage: str = "execution") -> None:
        criteria.append(Criterion(key=key, label=label, passed=bool(passed),
                                  detail=detail, remedy=remedy, stage=stage))

    # -- the onboarding half, as Client Onboarding itself judges it ------- #

    # 1. The onboarding is approved.
    add("onboarding_approved", "Onboarding approved",
        case.status == APPROVED,
        detail=(f"The onboarding is "
                f"{STATUS_LABELS.get(case.status, case.status).lower()}."),
        remedy="Work the onboarding through to approval.",
        stage="onboarding")

    # 2. Nothing is outstanding on the case itself.
    blocking = list(onboarding.get("blocking") or [])
    outstanding_requests = list(onboarding.get("outstanding_requests") or [])
    add("onboarding_clean", "Nothing outstanding on the onboarding",
        not blocking and not outstanding_requests,
        detail=("The onboarding reports no problems and nothing outstanding."
                if not blocking and not outstanding_requests else
                "; ".join(
                    [str(p.get("message") or "") for p in blocking[:3]]
                    + ([f"{len(outstanding_requests)} information request(s) "
                        "are still open."] if outstanding_requests else []))),
        remedy="Clear the onboarding's own problems and information requests.",
        stage="onboarding")

    # 3. The configuration the onboarding would generate is complete.
    #    Read from preview(), which is exactly what activation would create —
    #    and which writes nothing.
    artefacts_planned = list(preview.get("artefacts") or [])
    add("configuration_generates", "Configuration would generate cleanly",
        bool(artefacts_planned) and bool(preview.get("ready")),
        detail=(f"{len(artefacts_planned)} configuration artefact(s) would be "
                "created on activation." if artefacts_planned else
                "No configuration could be generated from this case."),
        remedy="Answer what the onboarding still needs, then preview again.",
        stage="onboarding")

    # -- the execution half ---------------------------------------------- #

    # 4. Required source files received.
    received_roles = {a.artefact_type for a in run.artefacts()}
    required_roles = roles.required_roles(facts.outcome)
    missing_roles = [r for r in required_roles if r not in received_roles]
    add("artefacts_present", "Required source files received",
        bool(required_roles) and not missing_roles,
        detail=("All required files were provided." if not missing_roles
                else "Missing: "
                     + ", ".join(roles.label(r) for r in missing_roles)),
        remedy="Provide the missing file, or generate a practice response.")

    # 5. Mandatory mappings resolved. The mapping REPORT records what the
    #    header mapper did; a mapping the mapper could not settle appears as an
    #    open decision, and that is what has to be clear.
    unresolved = [d for d in run.open_decisions
                  if d.get("status", "open") == "open"
                  and str(d.get("kind", "")).startswith("field_mapping")]
    unmapped = [m for m in run.mapping_report
                if not m.get("canonical_field")
                and m.get("tier") != "operator_approved"]
    add("mappings_resolved", "Field mappings resolved", not unresolved,
        detail=("All mappings the platform needs are resolved."
                + (f" {len(unmapped)} source column(s) were not used."
                   if unmapped else "")
                if not unresolved else
                f"{len(unresolved)} mapping decision(s) are still open."),
        remedy="Answer the open mapping decisions.")

    # 6. Blocking validation exceptions cleared.
    blocking_decisions = run.blocking_decisions()
    add("exceptions_cleared", "Blocking exceptions cleared",
        not blocking_decisions,
        detail=("No blocking exceptions remain." if not blocking_decisions else
                f"{len(blocking_decisions)} blocking decision(s) are open."),
        remedy="Resolve each blocking decision.")

    # 7. Intended Blob paths valid.
    artefacts = run.artefacts()
    without_uri = [a.source_file for a in artefacts if not a.intended_live_uri]
    add("intended_paths_valid", "Intended storage locations valid",
        bool(artefacts) and not without_uri,
        detail=("Every file has a valid intended location."
                if artefacts and not without_uri else
                "No location could be derived for: " + ", ".join(without_uri)
                if without_uri else "No files have been provided."),
        remedy="Confirm the client identifier, the portfolio identifier and "
               "the reporting period so a location can be derived.")

    # 8. Intended pipeline inputs satisfy the existing contracts.
    done = STAGE_DETERMINISTIC_COMPLETED
    add("pipeline_contracts", "Pipeline input contracts satisfied",
        all(run.stage_outcomes.get(s) == done
            for s in ("onboard", "transform", "validate")),
        detail=_stage_detail(run),
        remedy="Run the practice onboarding and clear anything it reports.")

    # 9. Orchestration sequencing valid.
    plan = run.orchestration_plan or {}
    add("orchestration_valid", "Orchestration sequencing valid",
        bool(plan.get("steps")) and bool(plan.get("valid")),
        detail=(f"{len(plan.get('steps') or [])} steps sequenced."
                if plan.get("steps") else "No execution plan has been "
                                          "generated."),
        remedy="Generate the orchestration plan.")

    # 10. Assembler prerequisites satisfied.
    assembler = run.assembler_plan or {}
    add("assembler_prerequisites", "Assembler prerequisites satisfied",
        bool(assembler.get("satisfied")),
        detail=(assembler.get("summary")
                or "Assembler prerequisites have not been checked."),
        remedy="Generate the orchestration plan, which checks them.")

    # 11. Execution readiness approved by a human.
    add("execution_approved", "Readiness approved",
        run.has_approval("execution_readiness"),
        detail=("An operator approved readiness for execution."
                if run.has_approval("execution_readiness")
                else "Readiness has not been approved."),
        remedy="Approve readiness once the plan is right.")

    # -- the boundary ----------------------------------------------------- #

    # 12. No configuration was written. The point of the whole exercise.
    add("no_configuration_written", "No configuration was created",
        case.status in NO_ACTIVE_CONFIGURATION,
        detail=("This practice case created no client configuration."
                if case.status in NO_ACTIVE_CONFIGURATION else
                "A configuration was activated from this case."),
        remedy="This case cannot proceed; report it to your administrator.",
        stage="boundary")

    # 13. No synthetic runtime control breached.
    breached = [e for e in (run.control_results or [])
                if e.get("kind") == "boundary_refusal"]
    add("runtime_controls_intact", "Practice controls intact",
        run.runtime_mode == RUNTIME_MODE_SYNTHETIC
        and policy.runtime_mode == RUNTIME_MODE_SYNTHETIC
        and not any(policy.permits(c) for c in
                    ("external_email", "live_blob_write",
                     "live_pipeline_trigger", "production_config_write",
                     "publish", "activate_configuration")),
        detail=("The practice boundary held for the whole case."
                + (f" {len(breached)} prohibited call(s) were refused."
                   if breached else "")),
        remedy="This case cannot proceed; report it to your administrator.",
        stage="boundary")

    return ReadinessVerdict(criteria=criteria)


def _stage_detail(run: SyntheticRun) -> str:
    if not run.stage_outcomes:
        return "The practice onboarding has not run."
    return "; ".join(f"{stage}: {outcome.replace('_', ' ')}"
                     for stage, outcome in sorted(run.stage_outcomes.items()))


# --------------------------------------------------------------------------- #
# The readiness package
# --------------------------------------------------------------------------- #

def build_package(run: SyntheticRun, case: OnboardingCase,
                  facts: ExecutionFacts, verdict: ReadinessVerdict,
                  audit: List[Dict[str, Any]], policy: SyntheticPolicy, *,
                  onboarding: Optional[Dict[str, Any]] = None,
                  preview: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The readiness package. Deterministic for a given run."""
    onboarding = onboarding or {}
    preview = preview or {}
    roles = artefact_vocabulary()
    artefacts = run.artefacts()

    manifest = _execution_manifest(run, case, facts, verdict)
    return {
        # 1
        "case_summary": {
            "case_ref": run.case_ref,
            "onboarding_status": case.status,
            "onboarding_kind": case.kind,
            "tenant": run.tenant,
            "state": run.state,
            "status": (_states.READY_FOR_EXECUTION if verdict.ready
                       else run.state),
            "runtime_mode": run.runtime_mode,
            "synthetic": True,
            "initiating_user": run.initiating_user,
            "fixture_id": run.fixture_id,
        },
        # 2 — the answers are the case's, not a second copy of them
        "confirmed_answers": case.answers,
        "execution_facts": facts.to_dict(),
        # 3
        "approved_product_scope": [
            {"product_id": p, "label": product_label(p)}
            for p in facts.products],
        # 4 — what activation WOULD create. Nothing was activated.
        "configuration_that_would_be_created": {
            "artefacts": list(preview.get("artefacts") or []),
            "changes": list(preview.get("changes") or []),
            "current_version": preview.get("current_version", 0),
            "next_version": preview.get("next_version", 0),
            "generated_identifiers": list(
                preview.get("generated_identifiers") or []),
            "defaults_used": list(preview.get("defaults_used") or []),
            "unrepresented": list(preview.get("unrepresented") or []),
            "written": False,
            "execution_status": "not_activated",
        },
        # 5
        "artefact_inventory": [
            {"source_file": a.source_file, "artefact_type": a.artefact_type,
             "role_label": roles.label(a.artefact_type) if a.artefact_type
             else "not recognised",
             "synthetic_location": a.synthetic_location,
             "sha256": a.sha256, "size": a.size, "row_count": a.row_count,
             "execution_status": a.execution_status,
             "recognition_basis": a.recognition_basis,
             "recognition_confidence": a.recognition_confidence}
            for a in artefacts],
        # 6
        "intended_live_storage_paths": [
            {"source_file": a.source_file,
             "intended_live_uri": a.intended_live_uri,
             "execution_status": "simulated_only", "written": False}
            for a in artefacts],
        # 7
        "field_mapping_report": run.mapping_report,
        # 8
        "validation_summary": [r for r in run.control_results
                               if r.get("kind") == "validation"],
        # 9
        "approved_exceptions": [d for d in run.open_decisions
                                if d.get("status") in ("approved", "rejected",
                                                       "acknowledged")],
        # 10
        "outstanding_observations": list(run.observations),
        # 11
        "orchestration_execution_plan": run.orchestration_plan,
        # 12
        "assembler_input_plan": run.assembler_plan,
        # 13
        "expected_downstream_outputs": _expected_outputs(run, facts),
        # 14 — the onboarding approval is the CASE's; the execution approval is
        #      the run's. Neither is restated as the other.
        "human_approvals": {
            "onboarding": {"approved_by": case.approved_by,
                           "approved_at": case.approved_at,
                           "reason": case.approval_reason},
            "execution": list(run.approvals),
        },
        # 15
        "audit_trail": audit,
        "onboarding_events": case.events,
        # 16
        "execution_manifest": manifest,

        "readiness": verdict.to_dict(),
        "onboarding_readiness": {
            "problems": list(onboarding.get("problems") or []),
            "blocking": list(onboarding.get("blocking") or []),
            "client_checklist": list(onboarding.get("client_checklist") or []),
            "outstanding_requests": list(
                onboarding.get("outstanding_requests") or []),
        },
        "policy": policy.to_dict(),
        "statement": {
            "headline": READY_HEADLINE if verdict.ready
            else "This practice case is not ready for execution.",
            "not_done": list(NOT_DONE_STATEMENTS),
        },
    }


def _expected_outputs(run: SyntheticRun,
                      facts: ExecutionFacts) -> List[Dict[str, Any]]:
    """What the live run WOULD produce, from the plan — never claimed as made."""
    outputs: List[Dict[str, Any]] = []
    for step in (run.orchestration_plan or {}).get("steps") or []:
        for artefact in step.get("produces") or []:
            outputs.append({"artefact": artefact, "produced_by": step["step"],
                            "execution_status": "not_produced"})
    for product in facts.products:
        outputs.append({"artefact": f"{product_label(product)} output",
                        "produced_by": "downstream product pipeline",
                        "execution_status": "not_produced"})
    return outputs


def _execution_manifest(run: SyntheticRun, case: OnboardingCase,
                        facts: ExecutionFacts,
                        verdict: ReadinessVerdict) -> Dict[str, Any]:
    """The machine-readable manifest. Deterministic and content-hashed.

    Nothing volatile is included — no timestamps, no absolute paths, no run
    identifiers — so re-generating it for an unchanged run yields the same
    ``content_hash``. A test asserts that.
    """
    body = {
        "schema_version": "2.0.0",
        "runtime_mode": RUNTIME_MODE_SYNTHETIC,
        "case_ref": run.case_ref,
        "tenant": run.tenant,
        "client_id": facts.client_id,
        "portfolio_id": facts.portfolio_id,
        "asset_class": facts.asset_class,
        "dataset": run.dataset,
        "reporting_period": run.reporting_period,
        "cadence": facts.cadence,
        "products": sorted(facts.products),
        "outcome": facts.outcome,
        "regime": facts.regime,
        "inputs": sorted(
            ({"source_file": a.source_file,
              "artefact_type": a.artefact_type,
              "sha256": a.sha256,
              "intended_live_uri": a.intended_live_uri,
              "execution_status": "simulated_only"}
             for a in run.artefacts()),
            key=lambda d: d["source_file"]),
        "onboarding_status": case.status,
        "configuration_activated": case.status == ACTIVATED,
        "orchestration_steps": [s.get("step") for s in
                                (run.orchestration_plan or {}).get("steps")
                                or []],
        "assembler_prerequisites": (run.assembler_plan or {}).get(
            "prerequisites", []),
        "readiness_status": (_states.READY_FOR_EXECUTION if verdict.ready
                             else "NOT_READY"),
        "readiness_criteria": [
            {"key": c.key, "passed": c.passed} for c in verdict.criteria],
        "execution_performed": False,
        "live_writes": [],
        "published": False,
        "emails_sent": [],
    }
    body["content_hash"] = stable_hash(canonical_json(body))
    return body


def anything_simulated(run: SyntheticRun) -> bool:
    """True when at least one stage was simulated rather than executed.

    Surfaced so a readiness statement can never let a simulated stage read as a
    completed one.
    """
    return any(o == STAGE_SIMULATED for o in run.stage_outcomes.values())


def anything_blocked(run: SyntheticRun) -> bool:
    return any(o == STAGE_HARD_BLOCKED for o in run.stage_outcomes.values())
