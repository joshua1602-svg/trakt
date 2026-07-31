"""operations_control.occ_agent.readiness — the READY_FOR_EXECUTION decision.

Readiness is **derived**, never declared. :func:`evaluate` walks the criteria in
§16 one at a time, each answered from persisted case state or from a
deterministic control result, and returns a criterion-by-criterion verdict. The
service only permits the transition to ``READY_FOR_EXECUTION`` when every
criterion passes; there is no code path by which an interpreter, a confidence
score or a chat message can set it.

:func:`build_package` then produces the readiness package — the sixteen parts of
§17, including a machine-readable execution manifest. The manifest is
deterministic: the same case produces byte-identical content, because nothing
volatile (timestamps, absolute paths, run identifiers) enters the hashed body.

Wording is fixed here too. A ready case says ``READY_FOR_EXECUTION`` and states
plainly what did NOT happen; the words "complete", "live", "published" and
"production successful" are deliberately absent, and a test asserts it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..contracts import canonical_json, stable_hash
from . import states as _states
from .case import (
    STAGE_DETERMINISTIC_COMPLETED,
    STAGE_HARD_BLOCKED,
    STAGE_SIMULATED,
    SyntheticArtefact,
    SyntheticCase,
)
from .policy import RUNTIME_MODE_SYNTHETIC, SyntheticPolicy
from .vocabulary import artefact_vocabulary, product_vocabulary

#: The four sentences the UI must be able to state for a ready case.
NOT_DONE_STATEMENTS = (
    "No live files were written.",
    "No production pipeline was triggered.",
    "No external email was sent.",
    "Nothing was published.",
)

READY_HEADLINE = "Synthetic case ready for execution."

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

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "label": self.label, "passed": self.passed,
                "detail": self.detail, "remedy": self.remedy}


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


def evaluate(case: SyntheticCase, policy: SyntheticPolicy) -> ReadinessVerdict:
    """The §16 criteria, each answered from state or a control result."""
    req = case.confirmed
    products = product_vocabulary()
    roles = artefact_vocabulary()
    outcome = products.outcome_for(req.required_products)
    criteria: List[Criterion] = []

    def add(key: str, label: str, passed: bool, detail: str = "",
            remedy: str = "") -> None:
        criteria.append(Criterion(key=key, label=label, passed=bool(passed),
                                  detail=detail, remedy=remedy))

    # 1. Required human facts confirmed.
    missing_facts = req.missing_mandatory() if case.confirmed_requirements \
        else ["the interpretation has not been confirmed"]
    add("facts_confirmed", "Required facts confirmed",
        not missing_facts and case.has_approval("requirements"),
        detail=("All required facts are confirmed." if not missing_facts
                else "Missing: " + ", ".join(missing_facts)),
        remedy="Confirm the interpretation, supplying the missing facts.")

    # 2. Required artefacts present.
    received_roles = {SyntheticArtefact.from_dict(a).artefact_type
                      for a in case.received_artefacts}
    required_roles = roles.required_roles(outcome)
    missing_roles = [r for r in required_roles if r not in received_roles]
    add("artefacts_present", "Required source files received",
        not missing_roles,
        detail=("All required files were provided." if not missing_roles
                else "Missing: "
                     + ", ".join(roles.label(r) for r in missing_roles)),
        remedy="Provide the missing file, or select a fixture that includes "
               "it.")

    # 3. Configuration validates.
    config_status = str((case.proposed_configuration or {}).get("status") or "")
    add("configuration_validates", "Configuration validates",
        config_status in ("READY", "READY_WITH_WARNINGS"),
        detail=(f"Configuration resolved: {config_status}." if config_status
                else "No configuration has been drafted."),
        remedy="Draft the configuration and clear any blockers it reports.")

    # 4. Mandatory mappings resolved. The mapping REPORT records what the
    #    header mapper did; a mapping the mapper could not settle appears as an
    #    open decision, and that is what has to be clear.
    unresolved = [d for d in case.open_decisions
                  if d.get("status", "open") == "open"
                  and str(d.get("kind", "")).startswith("field_mapping")]
    unmapped = [m for m in case.mapping_decisions
                if not m.get("canonical_field")
                and m.get("tier") != "operator_approved"]
    add("mappings_resolved", "Field mappings resolved", not unresolved,
        detail=("All mappings the platform needs are resolved."
                + (f" {len(unmapped)} source column(s) were not used."
                   if unmapped else "")
                if not unresolved else
                f"{len(unresolved)} mapping decision(s) are still open."),
        remedy="Answer the open mapping decisions.")

    # 5. Blocking validation exceptions cleared.
    blocking = case.blocking_decisions()
    add("exceptions_cleared", "Blocking exceptions cleared", not blocking,
        detail=("No blocking exceptions remain." if not blocking else
                f"{len(blocking)} blocking decision(s) are open."),
        remedy="Resolve each blocking decision.")

    # 6. Required approvals complete.
    needed = ("requirements", "onboarding_pack", "client_configuration",
              "execution_readiness")
    absent = [s for s in needed if not case.has_approval(s)]
    add("approvals_complete", "Required approvals complete", not absent,
        detail=("All required approvals are recorded." if not absent else
                "Awaiting: " + ", ".join(s.replace("_", " ") for s in absent)),
        remedy="Give the outstanding approvals.")

    # 7. Intended Blob paths valid.
    artefacts = [SyntheticArtefact.from_dict(a) for a in case.received_artefacts]
    without_uri = [a.source_file for a in artefacts if not a.intended_live_uri]
    add("intended_paths_valid", "Intended storage locations valid",
        bool(artefacts) and not without_uri,
        detail=("Every file has a valid intended location."
                if artefacts and not without_uri else
                "No location could be derived for: " + ", ".join(without_uri)
                if without_uri else "No files have been provided."),
        remedy="Confirm the client identifier, portfolio identifier and "
               "reporting date so a location can be derived.")

    # 8. Intended pipeline inputs satisfy the existing contracts.
    onboard_ok = case.stage_outcomes.get("onboard") == \
        STAGE_DETERMINISTIC_COMPLETED
    transform_ok = case.stage_outcomes.get("transform") == \
        STAGE_DETERMINISTIC_COMPLETED
    validate_ok = case.stage_outcomes.get("validate") == \
        STAGE_DETERMINISTIC_COMPLETED
    add("pipeline_contracts", "Pipeline input contracts satisfied",
        onboard_ok and transform_ok and validate_ok,
        detail=_stage_detail(case),
        remedy="Run the synthetic onboarding and clear anything it reports.")

    # 9. Orchestration sequencing valid.
    plan = case.orchestration_plan or {}
    add("orchestration_valid", "Orchestration sequencing valid",
        bool(plan.get("steps")) and bool(plan.get("valid")),
        detail=(f"{len(plan.get('steps') or [])} steps sequenced."
                if plan.get("steps") else "No execution plan has been "
                                          "generated."),
        remedy="Generate the orchestration plan.")

    # 10. Assembler prerequisites satisfied.
    assembler = case.assembler_plan or {}
    add("assembler_prerequisites", "Assembler prerequisites satisfied",
        bool(assembler.get("satisfied")),
        detail=(assembler.get("summary")
                or "Assembler prerequisites have not been checked."),
        remedy="Generate the orchestration plan, which checks them.")

    # 11. Material configuration values confirmed.
    unconfirmed = case.unconfirmed_material_values()
    add("material_values_confirmed", "Material configuration confirmed",
        not unconfirmed,
        detail=("Every material value is confirmed." if not unconfirmed else
                f"{len(unconfirmed)} material or low-confidence value(s) are "
                "unconfirmed."),
        remedy="Confirm or correct each material value.")

    # 12. No synthetic runtime control breached.
    breached = [e for e in (case.control_results or [])
                if e.get("kind") == "boundary_refusal"]
    add("runtime_controls_intact", "Synthetic controls intact",
        case.runtime_mode == RUNTIME_MODE_SYNTHETIC
        and policy.runtime_mode == RUNTIME_MODE_SYNTHETIC
        and not any(policy.permits(c) for c in
                    ("external_email", "live_blob_write",
                     "live_pipeline_trigger", "production_config_write",
                     "publish")),
        detail=("The synthetic boundary held for the whole case."
                + (f" {len(breached)} prohibited call(s) were refused."
                   if breached else "")),
        remedy="This case cannot proceed; report it to your administrator.")

    return ReadinessVerdict(criteria=criteria)


def _stage_detail(case: SyntheticCase) -> str:
    if not case.stage_outcomes:
        return "The synthetic onboarding has not run."
    parts = []
    for stage, outcome in sorted(case.stage_outcomes.items()):
        parts.append(f"{stage}: {outcome.replace('_', ' ')}")
    return "; ".join(parts)


# --------------------------------------------------------------------------- #
# The readiness package
# --------------------------------------------------------------------------- #

def build_package(case: SyntheticCase, verdict: ReadinessVerdict,
                  audit: List[Dict[str, Any]],
                  policy: SyntheticPolicy) -> Dict[str, Any]:
    """The §17 readiness package. Deterministic for a given case."""
    req = case.confirmed
    products = product_vocabulary()
    roles = artefact_vocabulary()
    selected = [p for p in (products.by_id(pid) for pid in req.required_products)
                if p is not None]
    artefacts = [SyntheticArtefact.from_dict(a) for a in case.received_artefacts]

    manifest = _execution_manifest(case, verdict)
    package: Dict[str, Any] = {
        # 1
        "case_summary": {
            "case_id": case.case_id,
            "tenant": case.tenant,
            "state": case.state,
            "status": (_states.READY_FOR_EXECUTION if verdict.ready
                       else case.state),
            "runtime_mode": case.runtime_mode,
            "synthetic": True,
            "initiating_user": case.initiating_user,
            "fixture_id": case.fixture_id,
        },
        # 2
        "confirmed_facts": req.to_dict(),
        # 3
        "approved_product_scope": [
            {"product_id": p.product_id, "label": p.label,
             "capability": p.capability, "regime_id": p.regime_id}
            for p in selected],
        # 4
        "approved_configuration": case.confirmed_configuration
        or (case.proposed_configuration or {}).get("candidate") or {},
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
        "field_mapping_report": case.mapping_decisions,
        # 8
        "validation_summary": [r for r in case.control_results
                               if r.get("kind") == "validation"],
        # 9
        "approved_exceptions": [
            a for a in case.approval_history
            if a.get("subject", "").startswith("exception")
            or a.get("subject") == "decision"],
        # 10
        "outstanding_observations": list(case.observations),
        # 11
        "orchestration_execution_plan": case.orchestration_plan,
        # 12
        "assembler_input_plan": case.assembler_plan,
        # 13
        "expected_downstream_outputs": _expected_outputs(case, selected),
        # 14
        "human_approvals": case.approval_history,
        # 15
        "audit_trail": audit,
        # 16
        "execution_manifest": manifest,

        "readiness": verdict.to_dict(),
        "policy": policy.to_dict(),
        "statement": {
            "headline": READY_HEADLINE if verdict.ready
            else "This synthetic case is not ready for execution.",
            "not_done": list(NOT_DONE_STATEMENTS),
        },
    }
    return package


def _expected_outputs(case: SyntheticCase, selected) -> List[Dict[str, Any]]:
    """What the live run WOULD produce, from the plan — never claimed as made."""
    outputs: List[Dict[str, Any]] = []
    for step in (case.orchestration_plan or {}).get("steps") or []:
        for artefact in step.get("produces") or []:
            outputs.append({"artefact": artefact, "produced_by": step["step"],
                            "execution_status": "not_produced"})
    for product in selected:
        outputs.append({"artefact": f"{product.label} output",
                        "produced_by": "downstream product pipeline",
                        "execution_status": "not_produced"})
    return outputs


def _execution_manifest(case: SyntheticCase,
                        verdict: ReadinessVerdict) -> Dict[str, Any]:
    """The machine-readable manifest. Deterministic and content-hashed.

    Nothing volatile is included — no timestamps, no absolute paths, no run
    identifiers — so re-generating it for an unchanged case yields the same
    ``content_hash``. A test asserts that.
    """
    req = case.confirmed
    artefacts = [SyntheticArtefact.from_dict(a) for a in case.received_artefacts]
    body = {
        "schema_version": "1.0.0",
        "runtime_mode": RUNTIME_MODE_SYNTHETIC,
        "case_id": case.case_id,
        "tenant": case.tenant,
        "client_id": case.client_id,
        "portfolio_id": case.portfolio_id,
        "asset_type": case.asset_type,
        "reporting_date": req.reporting_date,
        "reporting_frequency": req.reporting_frequency,
        "products": sorted(req.required_products),
        "outcome": (case.orchestration_plan or {}).get("outcome", ""),
        "inputs": sorted(
            ({"source_file": a.source_file,
              "artefact_type": a.artefact_type,
              "sha256": a.sha256,
              "intended_live_uri": a.intended_live_uri,
              "execution_status": "simulated_only"}
             for a in artefacts),
            key=lambda d: d["source_file"]),
        "configuration_hash": (case.proposed_configuration or {}).get(
            "content_hash", ""),
        "orchestration_steps": [s.get("step") for s in
                                (case.orchestration_plan or {}).get("steps")
                                or []],
        "assembler_prerequisites": (case.assembler_plan or {}).get(
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


def stage_outcome_summary(case: SyntheticCase) -> Dict[str, int]:
    """How many stages reached each §12 outcome. Used by the UI banner."""
    counts: Dict[str, int] = {}
    for outcome in case.stage_outcomes.values():
        counts[outcome] = counts.get(outcome, 0) + 1
    return counts


def anything_simulated(case: SyntheticCase) -> bool:
    """True when at least one stage was simulated rather than executed.

    Surfaced so a readiness statement can never let a simulated stage read as a
    completed one.
    """
    return any(o == STAGE_SIMULATED for o in case.stage_outcomes.values())


def anything_blocked(case: SyntheticCase) -> bool:
    return any(o == STAGE_HARD_BLOCKED for o in case.stage_outcomes.values())
