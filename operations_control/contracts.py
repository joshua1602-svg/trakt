"""operations_control.contracts — the operational contracts.

The Governed Agent Result is the single contract every stage reports through;
the WorkflowRun document is the persistent system of record for a run. Both are
plain dataclasses serialised to JSON in the operations-control container —
React state is never the system of record.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.0.0"

# --------------------------------------------------------------------------- #
# Vocabularies
# --------------------------------------------------------------------------- #

# Workflow types (delivery classification — operator can override before start).
WF_NEW_CLIENT = "new_client"
WF_NEW_PORTFOLIO = "new_portfolio"
WF_RECURRING = "recurring"
WF_BACKFILL = "backfill"
WORKFLOW_TYPES = (WF_NEW_CLIENT, WF_NEW_PORTFOLIO, WF_RECURRING, WF_BACKFILL)

# Outcomes the operator chooses (never agents).
OUTCOME_MI = "mi"
OUTCOME_MI_ANNEX2 = "mi_annex2"
OUTCOMES = (OUTCOME_MI, OUTCOME_MI_ANNEX2)

# Which book an input pack describes.
#: The dataset a pack describes unless told otherwise. Only a dataset OTHER than
#: this extends the pack identifier, so every funded pack keeps the identity it
#: has always had.
BATCH_DATASET_DEFAULT = "funded"

#: Datasets an operator may choose when opening a pack by hand. Deliberately the
#: production structure only: ``forecast`` is still accepted from a blob path for
#: backwards compatibility, but it is not something to start a new delivery as.
BATCH_DATASETS = (BATCH_DATASET_DEFAULT, "pipeline")

#: How far a publication approval is recorded as reaching. Deliberately does NOT
#: include a platform-wide option: approving one delivery is not an occasion to
#: set policy for every client, and any such mechanism belongs to the governed
#: administrator area, not to a delivery screen.
PUBLICATION_SCOPE_DEFAULT = "delivery"
PUBLICATION_SCOPES = (PUBLICATION_SCOPE_DEFAULT, "portfolio", "client")

#: How often a delivery arrives, when nothing says otherwise. Used only to derive
#: the governed source location — never part of the input pack's identity.
BATCH_FREQUENCY_DEFAULT = "monthly"

#: Datasets that can carry a regime (ESMA Annex 2) delivery. Everything else is
#: an MI view. Declared here so both doors into intake — the blob trigger and the
#: OCC — answer the regime question the same way.
REGIME_CAPABLE_DATASETS = (BATCH_DATASET_DEFAULT,)

# Workflow-run statuses.
RUN_RECEIVED = "received"
RUN_RUNNING = "running"
RUN_NEEDS_REVIEW = "needs_review"
RUN_BLOCKED = "blocked"
RUN_AWAITING_PUBLICATION = "awaiting_publication"
RUN_PUBLISHED = "published"
RUN_HELD = "held"
RUN_CANCELLED = "cancelled"
RUN_FAILED = "failed"
RUN_STATUSES = (RUN_RECEIVED, RUN_RUNNING, RUN_NEEDS_REVIEW, RUN_BLOCKED,
                RUN_AWAITING_PUBLICATION, RUN_PUBLISHED, RUN_HELD,
                RUN_CANCELLED, RUN_FAILED)
RUN_TERMINAL = (RUN_PUBLISHED, RUN_CANCELLED)

#: Legal workflow-run transitions (enforced by the engine, not the UI).
RUN_TRANSITIONS: Dict[str, tuple] = {
    RUN_RECEIVED: (RUN_RUNNING, RUN_CANCELLED),
    RUN_RUNNING: (RUN_NEEDS_REVIEW, RUN_BLOCKED, RUN_AWAITING_PUBLICATION,
                  RUN_FAILED, RUN_CANCELLED),
    # A run held on the REGULATORY branch sits here with its management
    # report prepared, so publishing or holding that report is legal
    # from needs_review — the open questions are about the delivery.
    RUN_NEEDS_REVIEW: (RUN_RUNNING, RUN_CANCELLED, RUN_PUBLISHED,
                       RUN_HELD),
    RUN_BLOCKED: (RUN_RUNNING, RUN_CANCELLED),
    RUN_AWAITING_PUBLICATION: (RUN_PUBLISHED, RUN_HELD, RUN_RUNNING, RUN_CANCELLED),
    RUN_HELD: (RUN_AWAITING_PUBLICATION, RUN_RUNNING, RUN_CANCELLED),
    RUN_FAILED: (RUN_RUNNING, RUN_CANCELLED),
    RUN_PUBLISHED: (),
    RUN_CANCELLED: (),
}

# Stage keys, in operator display order. The regulatory delivery stages
# (regulatory_config .. xml_delivery) apply only to the mi_annex2 outcome and
# run AFTER assembly, mirroring the proven Annex 2 route: assembled canonical
# -> projector -> delivery normaliser -> XML builder + XSD validation.
STAGE_RECEIVED = "received"
STAGE_UNDERSTANDING = "understanding"
STAGE_MAPPING = "mapping"
STAGE_VALIDATION = "validation"
STAGE_ASSEMBLY = "assembly"
STAGE_REG_CONFIG = "regulatory_config"
STAGE_PROJECTION = "projection"
STAGE_DELIVERY_PREP = "delivery_prep"
STAGE_XML = "xml_delivery"
STAGE_PUBLICATION = "publication"
STAGE_ORDER = (STAGE_RECEIVED, STAGE_UNDERSTANDING, STAGE_MAPPING,
               STAGE_VALIDATION, STAGE_ASSEMBLY, STAGE_REG_CONFIG,
               STAGE_PROJECTION, STAGE_DELIVERY_PREP, STAGE_XML,
               STAGE_PUBLICATION)
#: Stages owned by the OCC's governed Annex 2 delivery chain (not by the
#: orchestrator conductor).
ANNEX2_STAGES = (STAGE_REG_CONFIG, STAGE_PROJECTION, STAGE_DELIVERY_PREP,
                 STAGE_XML)

# Stage statuses (the operator vocabulary).
ST_WAITING = "waiting"
ST_RUNNING = "running"
ST_NEEDS_REVIEW = "needs_review"
ST_BLOCKED = "blocked"
ST_READY = "ready"
ST_APPROVED = "approved"
ST_REJECTED = "rejected"
ST_COMPLETED = "completed"
STAGE_STATUSES = (ST_WAITING, ST_RUNNING, ST_NEEDS_REVIEW, ST_BLOCKED,
                  ST_READY, ST_APPROVED, ST_REJECTED, ST_COMPLETED)

# Decision (review item) statuses + kinds.
DEC_OPEN = "open"
DEC_APPROVED = "approved"
DEC_REJECTED = "rejected"
DEC_SUPERSEDED = "superseded"
DECISION_STATUSES = (DEC_OPEN, DEC_APPROVED, DEC_REJECTED, DEC_SUPERSEDED)

KIND_FIELD_MAPPING = "field_mapping"
KIND_ALIAS = "alias"
KIND_ENUM = "enum"
KIND_TRANSFORMATION = "transformation"
KIND_VALIDATION_EXCEPTION = "validation_exception"
KIND_CLIENT_RULE = "client_rule"          # regulatory configuration setting
KIND_FILE_ROLE = "file_role"              # identify an ambiguous input file
KIND_PUBLICATION = "publication"
DECISION_KINDS = (KIND_FIELD_MAPPING, KIND_ALIAS, KIND_ENUM,
                  KIND_TRANSFORMATION, KIND_VALIDATION_EXCEPTION,
                  KIND_CLIENT_RULE, KIND_FILE_ROLE, KIND_PUBLICATION)


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def stable_hash(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Governed Agent Result
# --------------------------------------------------------------------------- #

@dataclass
class DecisionRequired:
    """One decision the operator must make. Self-contained: the question, the
    recommendation (with provenance) and the options are all present, so it can
    be answered without opening any other screen."""

    decision_id: str
    kind: str                                   # DECISION_KINDS
    title: str                                  # plain-language, short
    question: str                               # one sentence (the
                                                # plain-English message)
    blocking: bool = False
    # Structured decision-request fields (agent <-> OCC contract).
    category: str = ""                          # e.g. unmapped_source_field
    severity: str = ""                          # blocking | warning | info
    observed_values: List[str] = field(default_factory=list)
    affected_record_count: int = 0
    source_file: str = ""
    # {source: llm|deterministic|memory|operator, value, confidence, checked: bool}
    recommendation: Dict[str, Any] = field(default_factory=dict)
    options: List[Dict[str, str]] = field(default_factory=list)  # [{value,label}]
    allowed_scopes: List[str] = field(default_factory=lambda: ["file", "portfolio", "client", "global"])
    default_scope: str = "portfolio"
    # Collapsed-by-default supporting detail (humanised before storage).
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    # Internal linkage back to the agent artefact (never rendered).
    subject: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DecisionRequired":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class GovernedAgentResult:
    """The one operational contract every stage reports through.

    ``summary`` / ``why_it_matters`` are business language only; anything
    technical lives under ``evidence`` (pre-humanised) or the ``*_references``
    provenance fields, which the UI never renders raw.
    """

    run_id: str                                  # workflow id
    stage: str                                   # STAGE_ORDER key
    status: str                                  # STAGE_STATUSES
    summary: str
    why_it_matters: str = ""
    decisions_required: List[DecisionRequired] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)       # plain language
    blockers: List[str] = field(default_factory=list)       # plain language
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    input_references: List[str] = field(default_factory=list)   # provenance only
    output_references: List[str] = field(default_factory=list)  # provenance only
    agent_version: str = ""
    configuration_version: str = ""
    started_at: str = ""
    completed_at: str = ""
    result_id: str = field(default_factory=lambda: new_id("gar"))
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["decisions_required"] = [x.to_dict() if isinstance(x, DecisionRequired) else x
                                   for x in self.decisions_required]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GovernedAgentResult":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        known["decisions_required"] = [DecisionRequired.from_dict(x)
                                       for x in (d.get("decisions_required") or [])]
        return cls(**known)


# --------------------------------------------------------------------------- #
# Workflow run document
# --------------------------------------------------------------------------- #

@dataclass
class Delivery:
    """A registered delivery (the pack of files a workflow runs over)."""

    delivery_id: str
    client_id: str
    portfolio_id: str
    input_path: str                              # directory of the pack
    dataset: str = "funded"
    frequency: str = "monthly"
    reporting_period: str = ""
    files: List[Dict[str, Any]] = field(default_factory=list)  # [{name, role?}]
    schema_fingerprint: str = ""
    registered_at: str = field(default_factory=now_iso)
    registered_by: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Delivery":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class WorkflowRun:
    """The persistent workflow-run document (system of record)."""

    workflow_id: str
    client_id: str
    portfolio_id: str
    outcome: str                                 # OUTCOMES
    workflow_type: str                           # WORKFLOW_TYPES (final, after any override)
    delivery: Dict[str, Any]                     # Delivery.to_dict()
    reporting_period: str = ""
    status: str = RUN_RECEIVED
    # Classification governance: what the system suggested, and any override.
    classification_suggested: str = ""
    classification_overridden: bool = False
    classification_overridden_by: str = ""
    classification_reason: str = ""
    # stage_key -> {"status": ..., "result_id": ...}
    stages: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Orchestrator linkage (provenance; never rendered).
    orchestrator_run_id: str = ""
    staging_root: str = ""
    idempotency_key: str = ""
    rerun_count: int = 0
    # Annex 2 delivery-chain artefact references (projected/delivery/XML paths,
    # XSD result, hashes, interventions ref, effective config). Provenance for
    # publication metadata; never rendered raw.
    annex2: Dict[str, Any] = field(default_factory=dict)
    # Immutable effective-configuration pinning for this run:
    # {effective_config_id, version, content_hash, status, package_versions}.
    effective_config: Dict[str, Any] = field(default_factory=dict)
    # Input batch that produced this run (OCC-owned readiness; no sentinel).
    batch_id: str = ""
    # Set when the API restarted while this run was executing; cleared on rerun.
    interrupted: bool = False
    # Digest of the approved-decisions file Gate 1 last consumed. When the
    # operator answers the review queue this changes, which is how a rerun knows
    # Gate 1's previous output is stale and must be produced again.
    onboarding_decisions_digest: str = ""
    blockers: List[str] = field(default_factory=list)   # plain language
    created_at: str = field(default_factory=now_iso)
    created_by: str = ""
    updated_at: str = field(default_factory=now_iso)
    schema_version: str = SCHEMA_VERSION

    def stage_status(self, stage: str) -> str:
        return (self.stages.get(stage) or {}).get("status", ST_WAITING)

    def set_stage(self, stage: str, status: str,
                  result_id: Optional[str] = None) -> None:
        cur = self.stages.setdefault(stage, {})
        cur["status"] = status
        if result_id:
            cur["result_id"] = result_id

    @property
    def applicable_stages(self) -> List[str]:
        if self.outcome == OUTCOME_MI_ANNEX2:
            return list(STAGE_ORDER)
        return [s for s in STAGE_ORDER if s not in ANNEX2_STAGES]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WorkflowRun":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


class IllegalTransition(Exception):
    """A workflow-run status transition the state machine does not allow."""

    def __init__(self, from_status: str, to_status: str):
        self.from_status, self.to_status = from_status, to_status
        super().__init__(f"illegal transition {from_status!r} -> {to_status!r}")


def transition(run: WorkflowRun, to_status: str) -> None:
    """Apply a workflow-run status transition, enforcing legality."""
    if to_status not in RUN_TRANSITIONS.get(run.status, ()):
        raise IllegalTransition(run.status, to_status)
    run.status = to_status
    run.updated_at = now_iso()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
