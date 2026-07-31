"""operations_control.occ_agent.run — the synthetic execution record.

The OCC Agent does **not** define an onboarding case. The case is
:class:`operations_control.onboarding.case.OnboardingCase` — the platform's own
model, with its own statuses, transitions, answers, information requests and
event history. This module holds only what Client Onboarding has no concept of,
because it stops at activation and never runs a pipeline:

* which pipeline stages ran, and how honestly each one ran;
* the field-mapping report and the decisions the mapper could not settle;
* the artefacts provided as the client's response, and where each would have
  been filed;
* the orchestration and assembler plans;
* the readiness verdict and the readiness package.

One record sits beside one onboarding case, keyed by that case's own reference,
so nothing here duplicates or shadows the case itself. Everything the operator
confirms about the *client* lives on the case; everything about the *run* lives
here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from ..contracts import now_iso
from ..engine import OpsError
from . import states as _states
from .policy import RUNTIME_MODE_SYNTHETIC, UnsafePathError, validate_segment

SCHEMA_VERSION = "2.0.0"

#: Actor types recorded on audit events.
ACTOR_HUMAN = "human"
ACTOR_AGENT = "agent"
ACTOR_SYSTEM = "system"
ACTOR_TYPES = (ACTOR_HUMAN, ACTOR_AGENT, ACTOR_SYSTEM)

#: How an action was executed. Every audit event carries exactly one.
EXEC_DETERMINISTIC = "deterministic"
EXEC_MODEL_PROPOSED = "model_proposed"
EXEC_HUMAN_CONFIRMED = "human_confirmed"
EXEC_SYNTHETICALLY_EXECUTED = "synthetically_executed"
EXEC_SIMULATED = "simulated"
EXEC_BLOCKED = "blocked"
EXECUTION_CLASSIFICATIONS = (
    EXEC_DETERMINISTIC, EXEC_MODEL_PROPOSED, EXEC_HUMAN_CONFIRMED,
    EXEC_SYNTHETICALLY_EXECUTED, EXEC_SIMULATED, EXEC_BLOCKED,
)

#: Per-stage outcome vocabulary. Every stage declares exactly one, so a
#: simulated stage can never be read as a completed one.
STAGE_DETERMINISTIC_COMPLETED = "deterministic_execution_completed"
STAGE_CONTRACT_VALIDATED = "contract_validation_completed"
STAGE_SIMULATED = "execution_simulated"
STAGE_HUMAN_INPUT_REQUIRED = "human_input_required"
STAGE_HARD_BLOCKED = "hard_blocked"
STAGE_OUTCOMES = (STAGE_DETERMINISTIC_COMPLETED, STAGE_CONTRACT_VALIDATED,
                  STAGE_SIMULATED, STAGE_HUMAN_INPUT_REQUIRED,
                  STAGE_HARD_BLOCKED)


class RunSchemaError(OpsError):
    """A run document failed schema validation."""

    def __init__(self, detail: str):
        super().__init__("OCC_AGENT_RUN_INVALID",
                         "That practice case could not be read. "
                         f"({detail})", http_status=400)


@dataclass
class SyntheticArtefact:
    """One artefact provided in place of a client delivery.

    ``intended_live_uri`` is derived with the production path rules and is never
    written to; ``synthetic_location`` is inside the case sandbox.
    """

    artefact_id: str
    source_file: str                       # sanitised leaf name
    artefact_type: str = ""                # semantic input role
    synthetic_location: str = ""           # case-relative
    intended_live_uri: str = ""            # blob:// … never written
    execution_status: str = "simulated_only"
    sha256: str = ""
    size: int = 0
    columns: List[str] = field(default_factory=list)
    row_count: int = 0
    recognition_confidence: Optional[float] = None
    recognition_basis: str = ""
    provided_by: str = ""
    provided_at: str = field(default_factory=now_iso)
    fixture_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SyntheticArtefact":
        known = {k: v for k, v in (d or {}).items()
                 if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class RunAuditEvent:
    """One audit event. Nothing that resembles reasoning is stored."""

    event_id: str
    case_ref: str
    at: str
    actor_type: str
    actor_identity: str
    action: str
    prior_state: str = ""
    resulting_state: str = ""
    input_reference: str = ""
    output_reference: str = ""
    decision_basis: str = ""               # concise rationale only
    runtime_mode: str = RUNTIME_MODE_SYNTHETIC
    execution_classification: str = EXEC_DETERMINISTIC
    detail: Dict[str, Any] = field(default_factory=dict)
    prev_hash: str = ""
    record_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunAuditEvent":
        known = {k: v for k, v in (d or {}).items()
                 if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class Message:
    """One turn of the conversation. Presentation only — never authoritative.

    Everything the agent acts on lives on the onboarding case or in the typed
    fields below; the transcript is a record of how it got there.
    """

    role: str                              # operator | agent
    text: str
    at: str = field(default_factory=now_iso)
    refs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Message":
        known = {k: v for k, v in (d or {}).items()
                 if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class SyntheticRun:
    """The execution record beside one onboarding case."""

    case_ref: str                          # the OnboardingCase's own reference
    tenant: str
    initiating_user: str
    #: Where the EXECUTION half has got to. The onboarding half's status is the
    #: case's own, and is never mirrored here.
    state: str = _states.AWAITING_ONBOARDING
    runtime_mode: str = RUNTIME_MODE_SYNTHETIC
    version: int = 1

    #: What a *delivery* is, which standing configuration rightly does not hold:
    #: which of the client's portfolios this practice run is for, which book,
    #: and for which period.
    portfolio_id: str = ""
    dataset: str = "funded"
    reporting_period: str = ""
    #: The execution facts read off the onboarding case when the run last ran.
    facts: Dict[str, Any] = field(default_factory=dict)

    # The client response.
    received_artefacts: List[Dict[str, Any]] = field(default_factory=list)

    # What the run did.
    stage_outcomes: Dict[str, str] = field(default_factory=dict)
    mapping_report: List[Dict[str, Any]] = field(default_factory=list)
    open_decisions: List[Dict[str, Any]] = field(default_factory=list)
    control_results: List[Dict[str, Any]] = field(default_factory=list)
    planned_pipeline_actions: List[Dict[str, Any]] = field(default_factory=list)

    # Planning and readiness.
    orchestration_plan: Dict[str, Any] = field(default_factory=dict)
    assembler_plan: Dict[str, Any] = field(default_factory=dict)
    readiness: Dict[str, Any] = field(default_factory=dict)
    readiness_status: str = "not_evaluated"
    readiness_package_ref: str = ""

    #: Execution-side approvals only. Every approval about the CLIENT is the
    #: onboarding case's own, and is never copied here.
    approvals: List[Dict[str, Any]] = field(default_factory=list)

    # Operator-facing.
    blockers: List[str] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    fixture_id: str = ""

    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    schema_version: str = SCHEMA_VERSION

    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Belt and braces: a persisted run always says what it is, whatever a
        # future field default does.
        d["synthetic"] = True
        d["runtime_mode"] = RUNTIME_MODE_SYNTHETIC
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SyntheticRun":
        known = {k: v for k, v in (d or {}).items()
                 if k in cls.__dataclass_fields__}
        run = cls(**known)
        run.validate()
        return run

    def validate(self) -> None:
        from .store import validate_case_ref
        try:
            validate_case_ref(self.case_ref)
            validate_segment(self.tenant, "tenant")
        except UnsafePathError as exc:
            raise RunSchemaError(str(exc.message)) from exc
        if self.runtime_mode != RUNTIME_MODE_SYNTHETIC:
            raise RunSchemaError("runtime mode must be synthetic")
        if self.state not in _states.STATE_SPECS:
            raise RunSchemaError(f"unknown state '{self.state}'")
        for stage, outcome in (self.stage_outcomes or {}).items():
            if outcome not in STAGE_OUTCOMES:
                raise RunSchemaError(f"unknown outcome for {stage}")

    # -- helpers -------------------------------------------------------- #
    def artefacts(self) -> List[SyntheticArtefact]:
        return [SyntheticArtefact.from_dict(a) for a in self.received_artefacts]

    def blocking_decisions(self) -> List[Dict[str, Any]]:
        return [d for d in self.open_decisions
                if d.get("blocking") and d.get("status", "open") == "open"]

    def has_approval(self, subject: str) -> bool:
        return any(a.get("subject") == subject and a.get("decision") == "approved"
                   for a in self.approvals)

    def summary_row(self) -> Dict[str, Any]:
        """The list-view projection used by the case navigator."""
        return {
            "case_ref": self.case_ref,
            "tenant": self.tenant,
            "state": self.state,
            "state_label": _states.spec_label(self.state),
            "readiness_status": self.readiness_status,
            "runtime_mode": self.runtime_mode,
            "synthetic": True,
            "open_decisions": len(self.blocking_decisions()),
            "fixture_id": self.fixture_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
