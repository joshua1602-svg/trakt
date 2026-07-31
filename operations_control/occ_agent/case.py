"""operations_control.occ_agent.case — the synthetic onboarding case document.

The case is the system of record for the OCC Agent tab. Chat history is NOT
authoritative: every fact the agent acts on lives in a typed field here, was put
there by a deterministic service call, and carries provenance.

The document follows the repository's existing contract style
(:mod:`operations_control.contracts`): plain dataclasses with ``to_dict`` /
``from_dict``, serialised to JSON, tolerant of unknown keys on read so an older
document still loads after a field is added.

Everything that could be mistaken for live state is explicit:

* ``runtime_mode`` is always ``synthetic`` and is written on creation;
* ``synthetic`` is a hard-coded ``True`` on every serialised document;
* identity (``tenant``, ``client_id``, ``portfolio_id``) is set once and
  :meth:`SyntheticCase.rebind_identity` is the only way to change it — it
  refuses an implicit change, so a natural-language correction cannot silently
  move a case to another client.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from ..contracts import new_id, now_iso
from ..engine import OpsError
from . import states as _states
from .policy import (
    RUNTIME_MODE_SYNTHETIC,
    UnsafePathError,
    validate_case_id,
    validate_segment,
)

SCHEMA_VERSION = "1.0.0"

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

#: Per-stage outcome vocabulary (§12: every stage must distinguish these).
STAGE_DETERMINISTIC_COMPLETED = "deterministic_execution_completed"
STAGE_CONTRACT_VALIDATED = "contract_validation_completed"
STAGE_SIMULATED = "execution_simulated"
STAGE_HUMAN_INPUT_REQUIRED = "human_input_required"
STAGE_HARD_BLOCKED = "hard_blocked"
STAGE_OUTCOMES = (STAGE_DETERMINISTIC_COMPLETED, STAGE_CONTRACT_VALIDATED,
                  STAGE_SIMULATED, STAGE_HUMAN_INPUT_REQUIRED,
                  STAGE_HARD_BLOCKED)

#: Provenance sources for a proposed configuration value (§10).
PROV_HUMAN_INSTRUCTION = "confirmed_human_instruction"
PROV_CLIENT_RESPONSE = "synthetic_client_response"
PROV_SOURCE_ARTEFACT = "source_artefact"
PROV_INHERITED_TEMPLATE = "inherited_template"
PROV_ASSET_CONFIG = "asset_configuration"
PROV_PRODUCT_CONFIG = "product_configuration"
PROV_DEFAULT = "deterministic_default"
PROV_AGENT_INFERENCE = "agent_inference"
PROVENANCE_SOURCES = (PROV_HUMAN_INSTRUCTION, PROV_CLIENT_RESPONSE,
                      PROV_SOURCE_ARTEFACT, PROV_INHERITED_TEMPLATE,
                      PROV_ASSET_CONFIG, PROV_PRODUCT_CONFIG, PROV_DEFAULT,
                      PROV_AGENT_INFERENCE)

#: Confidence at or above which an inferred value may stand without an explicit
#: human confirmation — and ONLY when the value is not material. Mirrors the
#: onboarding agent's own apply threshold (config/asset/product_profiles.yaml).
INFERENCE_APPLY_CONFIDENCE = 0.80


class CaseSchemaError(OpsError):
    """A case document failed schema validation."""

    def __init__(self, detail: str):
        super().__init__("OCC_AGENT_CASE_INVALID",
                         "That synthetic case could not be read. "
                         f"({detail})", http_status=400)


class IdentityChangeRefused(OpsError):
    """A client/portfolio identity change was attempted implicitly."""

    def __init__(self, what: str, old: str, new: str):
        super().__init__(
            "OCC_AGENT_IDENTITY_LOCKED",
            f"The {what} on a synthetic case cannot be changed from '{old}' to "
            f"'{new}' as part of another change. Start a new case, or make the "
            "change explicitly.",
            http_status=409)


def new_case_id() -> str:
    """An immutable, path-safe case identifier."""
    return "CASE-" + new_id("x").split("_")[1][:10].upper()


# --------------------------------------------------------------------------- #
# Sub-documents
# --------------------------------------------------------------------------- #

@dataclass
class ExtractedRequirements:
    """Structured facts read out of a natural-language instruction.

    Every field is optional at extraction time — an instruction that omits the
    reporting date must produce an unresolved question, not a guess.
    """

    client_name: str = ""
    proposed_client_id: str = ""
    portfolio_name: str = ""
    proposed_portfolio_id: str = ""
    asset_type: str = ""
    jurisdiction: str = ""
    required_products: List[str] = field(default_factory=list)
    reporting_frequency: str = ""
    reporting_date: str = ""
    expected_artefacts: List[str] = field(default_factory=list)
    known_counterparties: List[str] = field(default_factory=list)
    target_go_live_date: str = ""
    unresolved_questions: List[str] = field(default_factory=list)
    #: field name -> how it was obtained (see PROVENANCE_SOURCES).
    provenance: Dict[str, str] = field(default_factory=dict)
    #: field name -> 0..1. Only present for inferred values.
    confidence: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExtractedRequirements":
        known = {k: v for k, v in (d or {}).items()
                 if k in cls.__dataclass_fields__}
        return cls(**known)

    def validate(self) -> None:
        """Schema validation. Applied to EVERY set of extracted facts,
        whoever produced them — a deterministic parser or a model."""
        for name in ("client_name", "proposed_client_id", "portfolio_name",
                     "proposed_portfolio_id", "asset_type", "jurisdiction",
                     "reporting_frequency", "reporting_date",
                     "target_go_live_date"):
            v = getattr(self, name)
            if not isinstance(v, str):
                raise CaseSchemaError(f"{name} must be text")
            if len(v) > 200:
                raise CaseSchemaError(f"{name} is too long")
        for name in ("required_products", "expected_artefacts",
                     "known_counterparties", "unresolved_questions"):
            v = getattr(self, name)
            if not isinstance(v, list) or any(not isinstance(x, str) for x in v):
                raise CaseSchemaError(f"{name} must be a list of text")
            if len(v) > 50:
                raise CaseSchemaError(f"{name} has too many entries")
        for name in ("proposed_client_id", "proposed_portfolio_id"):
            v = getattr(self, name)
            if v and not re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$", v):
                raise CaseSchemaError(f"{name} is not a usable identifier")
        for name in ("reporting_date", "target_go_live_date"):
            v = getattr(self, name)
            if v and not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
                raise CaseSchemaError(f"{name} must be a date (YYYY-MM-DD)")
        for k, c in (self.confidence or {}).items():
            if not isinstance(c, (int, float)) or not 0.0 <= float(c) <= 1.0:
                raise CaseSchemaError(f"confidence for {k} must be 0..1")
        for k, p in (self.provenance or {}).items():
            if p not in PROVENANCE_SOURCES:
                raise CaseSchemaError(f"unknown provenance '{p}' for {k}")

    def missing_mandatory(self) -> List[str]:
        """Facts without which no onboarding pack can be built."""
        out = []
        if not self.client_name:
            out.append("client name")
        if not self.asset_type:
            out.append("asset type")
        if not self.required_products:
            out.append("required Trakt products")
        if not self.proposed_portfolio_id and not self.portfolio_name:
            out.append("portfolio name or identifier")
        return out


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
    fixture_id: str = ""                   # set when it came from a fixture

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SyntheticArtefact":
        known = {k: v for k, v in (d or {}).items()
                 if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class ProposedValue:
    """One proposed configuration value with its provenance and materiality."""

    key: str
    value: Any
    source: str                            # PROVENANCE_SOURCES
    evidence: str = ""
    confidence: Optional[float] = None
    downstream_impact: str = ""
    material: bool = False
    requires_human_confirmation: bool = False
    confirmed: bool = False
    confirmed_by: str = ""
    confirmed_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProposedValue":
        known = {k: v for k, v in (d or {}).items()
                 if k in cls.__dataclass_fields__}
        return cls(**known)

    def validate(self) -> None:
        if self.source not in PROVENANCE_SOURCES:
            raise CaseSchemaError(f"unknown provenance '{self.source}'")
        if self.confidence is not None and not 0.0 <= float(self.confidence) <= 1.0:
            raise CaseSchemaError("confidence must be 0..1")

    def needs_confirmation(self) -> bool:
        """A value may stand unconfirmed only when it is neither material nor
        low-confidence. §10: no material or low-confidence inference may
        silently pass."""
        if self.confirmed:
            return False
        if self.material:
            return True
        if self.source == PROV_AGENT_INFERENCE:
            c = self.confidence
            return c is None or float(c) < INFERENCE_APPLY_CONFIDENCE
        return False


@dataclass
class CaseAuditEvent:
    """One audit event. §19 fields, and nothing that resembles reasoning."""

    event_id: str
    case_id: str
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
    def from_dict(cls, d: Dict[str, Any]) -> "CaseAuditEvent":
        known = {k: v for k, v in (d or {}).items()
                 if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class Approval:
    """One recorded human approval."""

    approval_id: str
    subject: str                           # requirements | onboarding_pack | …
    decision: str                          # approved | rejected
    actor: str
    at: str = field(default_factory=now_iso)
    reason: str = ""
    reference: str = ""                    # what exactly was approved (a hash)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Approval":
        known = {k: v for k, v in (d or {}).items()
                 if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class Message:
    """One turn of the conversation. Presentation only — never authoritative."""

    role: str                              # operator | agent
    text: str
    at: str = field(default_factory=now_iso)
    #: Identifiers of the structured artefacts this turn produced, so the UI can
    #: attach a decision card or an interpretation to the right turn.
    refs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Message":
        known = {k: v for k, v in (d or {}).items()
                 if k in cls.__dataclass_fields__}
        return cls(**known)


# --------------------------------------------------------------------------- #
# The case
# --------------------------------------------------------------------------- #

@dataclass
class SyntheticCase:
    """The persistent synthetic onboarding case."""

    case_id: str
    tenant: str
    initiating_user: str
    state: str = _states.CASE_CREATED
    runtime_mode: str = RUNTIME_MODE_SYNTHETIC
    case_version: int = 1

    # Identity. Set from confirmed requirements; never changed implicitly.
    client_id: str = ""
    client_name: str = ""
    portfolio_id: str = ""
    portfolio_name: str = ""
    asset_type: str = ""

    # Requirements.
    extracted_requirements: Dict[str, Any] = field(default_factory=dict)
    confirmed_requirements: Dict[str, Any] = field(default_factory=dict)
    unresolved_questions: List[str] = field(default_factory=list)

    # Onboarding pack.
    onboarding_pack: Dict[str, Any] = field(default_factory=dict)
    pack_issued_synthetically_at: str = ""

    # Artefacts.
    required_artefacts: List[Dict[str, Any]] = field(default_factory=list)
    received_artefacts: List[Dict[str, Any]] = field(default_factory=list)

    # Configuration.
    #: Configuration values the client supplied in answer to the pack, keyed by
    #: the configuration key they satisfy. Kept separate from the requirements
    #: so provenance can distinguish a client answer from an operator statement.
    client_answers: Dict[str, Any] = field(default_factory=dict)
    proposed_configuration: Dict[str, Any] = field(default_factory=dict)
    confirmed_configuration: Dict[str, Any] = field(default_factory=dict)
    configuration_provenance: List[Dict[str, Any]] = field(default_factory=list)

    # Controls and decisions.
    mapping_decisions: List[Dict[str, Any]] = field(default_factory=list)
    open_decisions: List[Dict[str, Any]] = field(default_factory=list)
    control_results: List[Dict[str, Any]] = field(default_factory=list)
    stage_outcomes: Dict[str, str] = field(default_factory=dict)
    blockers: List[str] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)   # non-blocking

    # Planning + readiness.
    planned_pipeline_actions: List[Dict[str, Any]] = field(default_factory=list)
    orchestration_plan: Dict[str, Any] = field(default_factory=dict)
    assembler_plan: Dict[str, Any] = field(default_factory=dict)
    readiness: Dict[str, Any] = field(default_factory=dict)
    readiness_status: str = "not_evaluated"
    readiness_package_ref: str = ""

    # Governance.
    approval_history: List[Dict[str, Any]] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    fixture_id: str = ""

    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    schema_version: str = SCHEMA_VERSION

    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Belt and braces: a persisted case always says what it is, whatever a
        # future field default does.
        d["synthetic"] = True
        d["runtime_mode"] = RUNTIME_MODE_SYNTHETIC
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SyntheticCase":
        known = {k: v for k, v in (d or {}).items()
                 if k in cls.__dataclass_fields__}
        case = cls(**known)
        case.validate()
        return case

    # -- schema --------------------------------------------------------- #
    def validate(self) -> None:
        validate_case_id(self.case_id)
        try:
            validate_segment(self.tenant, "tenant")
        except UnsafePathError as exc:
            raise CaseSchemaError("tenant") from exc
        if self.runtime_mode != RUNTIME_MODE_SYNTHETIC:
            raise CaseSchemaError("runtime mode must be synthetic")
        if self.state not in _states.STATE_SPECS:
            raise CaseSchemaError(f"unknown state '{self.state}'")
        for name in ("client_id", "portfolio_id"):
            v = getattr(self, name)
            if v:
                try:
                    validate_segment(v, name)
                except UnsafePathError as exc:
                    raise CaseSchemaError(name) from exc
        if self.extracted_requirements:
            ExtractedRequirements.from_dict(self.extracted_requirements).validate()
        if self.confirmed_requirements:
            ExtractedRequirements.from_dict(self.confirmed_requirements).validate()

    # -- identity ------------------------------------------------------- #
    def rebind_identity(self, *, client_id: str = "", portfolio_id: str = "",
                        explicit: bool = False) -> None:
        """Set or change client/portfolio identity.

        Setting an empty field is always allowed (first binding). CHANGING an
        already-bound identity requires ``explicit=True``, which only the
        dedicated identity-change service call passes — so a mapping correction
        or a configuration edit can never move a case between clients.
        """
        for attr, value in (("client_id", client_id),
                            ("portfolio_id", portfolio_id)):
            if not value:
                continue
            validate_segment(value, attr)
            current = getattr(self, attr)
            if current and current != value and not explicit:
                raise IdentityChangeRefused(attr.replace("_", " "), current,
                                            value)
            setattr(self, attr, value)

    # -- helpers -------------------------------------------------------- #
    @property
    def extracted(self) -> ExtractedRequirements:
        return ExtractedRequirements.from_dict(self.extracted_requirements)

    @property
    def confirmed(self) -> ExtractedRequirements:
        return ExtractedRequirements.from_dict(self.confirmed_requirements)

    def approvals_for(self, subject: str) -> List[Dict[str, Any]]:
        return [a for a in self.approval_history if a.get("subject") == subject]

    def has_approval(self, subject: str) -> bool:
        return any(a.get("decision") == "approved"
                   for a in self.approvals_for(subject))

    def blocking_decisions(self) -> List[Dict[str, Any]]:
        return [d for d in self.open_decisions
                if d.get("blocking") and d.get("status", "open") == "open"]

    def unconfirmed_material_values(self) -> List[Dict[str, Any]]:
        out = []
        for raw in self.configuration_provenance:
            pv = ProposedValue.from_dict(raw)
            if pv.needs_confirmation():
                out.append(raw)
        return out

    def summary_row(self) -> Dict[str, Any]:
        """The list-view projection used by the case navigator."""
        return {
            "case_id": self.case_id,
            "tenant": self.tenant,
            "client_id": self.client_id,
            "client_name": self.client_name or self.extracted.client_name,
            "portfolio_id": self.portfolio_id,
            "portfolio_name": self.portfolio_name,
            "asset_type": self.asset_type,
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
