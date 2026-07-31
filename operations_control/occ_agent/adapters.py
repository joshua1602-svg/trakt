"""operations_control.occ_agent.adapters — one workflow, two kinds of effect.

The operating process is the same whether it is being rehearsed or performed:
the same case, the same catalogue, the same controls, the same review package,
the same human approvals. What differs is what happens to infrastructure at the
end, and that is all these adapters differ in.

:class:`SyntheticExecutionAdapter`
    Rehearsal. Runs the real Onboarding, Orchestration and Assembler agents over
    files in an isolated sandbox, derives the live locations without writing to
    them, and refuses activation.

:class:`LiveExecutionAdapter`
    The real thing, and **not a second pipeline**. It calls the platform's own
    governed path in order:

        ``OnboardingService.activate()``   writes the configuration and
                                           registers the sources
        ``engine.create_batch()``          opens a governed input pack
        ``engine.upload_batch_files()``    places files where the intake expects
        ``engine.start_batch()``           starts the existing Onboarding Agent

    Every one of those already exists and is already governed. This adapter
    sequences them; it does not reimplement any of them.

Live execution is disabled by default (``OCC_AGENT_LIVE_ENABLED``) and, when
enabled, every precondition is checked **in one place** —
:func:`assert_may_activate`. A caller cannot satisfy some of them and proceed:
there is no partial gate, and no second gate anywhere else.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from ..engine import OpsError
from ..onboarding.case import APPROVED, OnboardingCase
from .derive import ExecutionFacts
from .policy import CAP_ACTIVATE_CONFIGURATION, SyntheticPolicy

#: The live feature flag. Absent or anything other than a recognised truthy
#: value means live execution is OFF — the flag fails closed, exactly as the
#: feature flag does.
LIVE_FLAG_ENV = "OCC_AGENT_LIVE_ENABLED"
_TRUTHY = frozenset({"1", "true", "yes", "on", "enabled"})

MODE_SYNTHETIC = "synthetic"
MODE_LIVE = "live"


def live_enabled(env: Optional[Dict[str, str]] = None) -> bool:
    source = env if env is not None else os.environ
    return str(source.get(LIVE_FLAG_ENV, "")).strip().lower() in _TRUTHY


class ActivationRefused(OpsError):
    """A precondition for live activation was not met."""

    def __init__(self, reasons: List[str]):
        self.reasons = list(reasons)
        super().__init__(
            "OCC_AGENT_ACTIVATION_REFUSED",
            "This cannot be activated yet. " + " ".join(reasons),
            http_status=409)


# --------------------------------------------------------------------------- #
# The one gate
# --------------------------------------------------------------------------- #

@dataclass
class ActivationPreconditions:
    """Everything that must be true before anything reaches production.

    Assembled by the service from state it already holds, then checked here.
    Keeping the *checking* in one function is the point: a new caller cannot
    invent a shorter path, and a reader can see the whole gate at once.
    """

    mode: str = MODE_SYNTHETIC
    flag_enabled: bool = False
    case_ref: str = ""
    onboarding_status: str = ""
    configuration_approved: bool = False
    readiness_passed: bool = False
    tenant: str = ""
    client_id: str = ""
    portfolio_id: str = ""
    configuration_valid: bool = False
    configuration_problems: List[str] = field(default_factory=list)
    artefacts_present: int = 0
    required_artefacts_satisfied: bool = False
    approval_audited: bool = False
    already_activated: bool = False
    confirmed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def activation_refusals(pre: ActivationPreconditions) -> List[str]:
    """Every reason this may not activate. All of them, not the first."""
    out: List[str] = []
    if pre.mode != MODE_LIVE:
        out.append("This case is in rehearsal mode.")
    if not pre.flag_enabled:
        out.append("Live execution is not switched on in this environment.")
    if pre.onboarding_status != APPROVED:
        out.append("The onboarding has not been approved.")
    if not pre.configuration_approved:
        out.append("A human has not approved the configuration.")
    if not pre.readiness_passed:
        out.append("Not every readiness criterion has passed.")
    if not (pre.tenant and pre.client_id and pre.portfolio_id):
        out.append("The tenant, client and portfolio are not all identified.")
    if not pre.configuration_valid:
        out.append("The configuration does not validate."
                   + (" " + "; ".join(pre.configuration_problems[:3])
                      if pre.configuration_problems else ""))
    if not (pre.artefacts_present and pre.required_artefacts_satisfied):
        out.append("The required source files are not all present.")
    if not pre.approval_audited:
        out.append("The approval is not recorded in the audit trail.")
    if pre.already_activated:
        out.append("This configuration has already been activated.")
    if not pre.confirmed:
        out.append("The final confirmation has not been given.")
    return out


def assert_may_activate(pre: ActivationPreconditions) -> None:
    """The single gate. Every caller of live activation passes through here."""
    refusals = activation_refusals(pre)
    if refusals:
        raise ActivationRefused(refusals)


# --------------------------------------------------------------------------- #
# What an activation would, or did, do
# --------------------------------------------------------------------------- #

@dataclass
class ActivationIntent:
    """The confirmation an operator is shown immediately before production.

    Deliberately concrete. "Activate this client" is not a decision anyone can
    make; "write this configuration, put these three files here, and start
    ingestion" is.
    """

    client_id: str = ""
    client_name: str = ""
    portfolio_id: str = ""
    dataset: str = ""
    #: What the delivery is FOR, in the engine's own vocabulary. Passed straight
    #: to ``create_batch``, which refuses anything outside ``OUTCOMES``.
    outcome: str = ""
    cadence: str = ""
    reporting_period: str = ""
    files: List[Dict[str, str]] = field(default_factory=list)
    target_locations: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    configuration_artefacts: List[str] = field(default_factory=list)
    statement: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ActivationResult:
    ok: bool = False
    mode: str = MODE_SYNTHETIC
    client_id: str = ""
    version: Optional[int] = None
    artefacts_written: List[Dict[str, Any]] = field(default_factory=list)
    batch_id: str = ""
    workflow_id: str = ""
    files_placed: List[str] = field(default_factory=list)
    message: str = ""
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_intent(case: OnboardingCase, facts: ExecutionFacts, *,
                 reporting_period: str, files: List[Dict[str, str]],
                 configuration_artefacts: List[str]) -> ActivationIntent:
    """What confirming would cause. Built from the case, never from prose."""
    return ActivationIntent(
        client_id=facts.client_id, client_name=facts.client_name,
        portfolio_id=facts.portfolio_id, dataset=facts.dataset,
        outcome=facts.outcome, cadence=facts.cadence,
        reporting_period=reporting_period,
        files=list(files),
        target_locations=sorted({str(f.get("target") or "") for f in files
                                 if f.get("target")}),
        configuration_artefacts=list(configuration_artefacts),
        actions=[
            f"Write {len(configuration_artefacts)} configuration artefact(s) "
            f"for {facts.client_id}, as a new governed version.",
            "Register the expected source deliveries in the production source "
            "registry.",
            f"Place {len(files)} file(s) in the production raw location, "
            "through the platform's own governed intake.",
            "Start the existing Onboarding Agent, which will profile, map, "
            "transform, validate and assemble the delivery.",
        ],
        statement=(
            f"Confirming activates {facts.client_name or facts.client_id}'s "
            "configuration and starts ingestion of the files listed. This is "
            "production. It cannot be undone from this tab."))


# --------------------------------------------------------------------------- #
# The adapters
# --------------------------------------------------------------------------- #

class ExecutionAdapter(Protocol):
    """What differs between rehearsing the process and performing it.

    ``payloads`` maps a file name from :attr:`ActivationIntent.files` to its
    bytes. It is passed separately from the intent so the intent stays a small,
    persistable, human-readable record of *what would happen* — an operator
    confirms a description, not a payload.
    """

    mode: str

    def may_activate(self) -> bool: ...

    def activate(self, *, pre: ActivationPreconditions,
                 intent: ActivationIntent, actor: str,
                 payloads: Optional[Dict[str, bytes]] = None
                 ) -> ActivationResult: ...


class SyntheticExecutionAdapter:
    """Rehearsal. Everything up to the boundary, and nothing across it."""

    mode = MODE_SYNTHETIC

    def __init__(self, policy: SyntheticPolicy):
        self.policy = policy

    def may_activate(self) -> bool:
        return False

    def activate(self, *, pre: ActivationPreconditions,
                 intent: ActivationIntent, actor: str,
                 payloads: Optional[Dict[str, bytes]] = None
                 ) -> ActivationResult:
        """Always refused, and audited. The refusal is the feature."""
        del payloads                        # nothing is ever placed anywhere
        self.policy.require(CAP_ACTIVATE_CONFIGURATION,
                            detail=f"client {intent.client_id}",
                            case_id=pre.case_ref, tenant=pre.tenant,
                            actor=actor)
        raise AssertionError(                       # pragma: no cover
            "the synthetic policy must refuse activation")


class LiveExecutionAdapter:
    """The governed path, sequenced. Not a second pipeline.

    ``onboarding`` is an :class:`~operations_control.onboarding.service.
    OnboardingService`; ``engine`` is the OCC's own engine. Both are injected,
    which is what lets the contract be tested with fakes and no Azure.
    """

    mode = MODE_LIVE

    def __init__(self, onboarding: Any, engine: Any, *,
                 enabled: Optional[bool] = None):
        self.onboarding = onboarding
        self.engine = engine
        self._enabled = live_enabled() if enabled is None else bool(enabled)

    def may_activate(self) -> bool:
        return self._enabled

    def activate(self, *, pre: ActivationPreconditions,
                 intent: ActivationIntent, actor: str,
                 payloads: Optional[Dict[str, bytes]] = None
                 ) -> ActivationResult:
        """Activate, then hand the files to the existing Onboarding Agent.

        The order is the platform's, not this module's: configuration first,
        because the intake resolves against it; then the governed input pack;
        then the files; then the start. A failure at any step returns what had
        already happened, so an operator is never left guessing.
        """
        assert_may_activate(pre)
        payloads = payloads or {}
        missing = [f["name"] for f in intent.files
                   if f["name"] not in payloads]
        if missing:
            raise ActivationRefused(
                [f"The bytes for {', '.join(missing)} are not available."])

        result = ActivationResult(mode=self.mode, client_id=intent.client_id)
        try:
            activation = self.onboarding.activate(case_id=pre.case_ref,
                                                  by=actor)
            result.version = activation.get("version")
            result.artefacts_written = list(activation.get("artefacts") or [])

            # The engine's own signature, in full. Every argument comes from
            # the intent the human confirmed — nothing is defaulted here, and
            # `workflow_type` in particular decides whether the delivery may
            # reach regulatory reporting at all.
            batch = self.engine.create_batch(
                client_id=intent.client_id, portfolio_id=intent.portfolio_id,
                reporting_date=intent.reporting_period,
                workflow_type=intent.outcome,
                dataset=intent.dataset, frequency=intent.cadence,
                created_by=actor)
            result.batch_id = str(batch.get("batch_id") or "")

            self.engine.upload_batch_files(
                client_id=intent.client_id, batch_id=result.batch_id,
                uploads=[(f["name"], payloads[f["name"]])
                         for f in intent.files],
                received_by=actor)
            result.files_placed = [f["name"] for f in intent.files]

            started = self.engine.start_batch(client_id=intent.client_id,
                                              batch_id=result.batch_id,
                                              actor=actor)
            result.workflow_id = str(started.get("workflow_id") or "")
            result.ok = True
            result.message = (
                f"Configuration version {result.version} activated and "
                f"ingestion started as workflow {result.workflow_id}.")
        except Exception as exc:            # noqa: BLE001 — reported, not hidden
            result.ok = False
            result.error = f"{type(exc).__name__}: {exc}"
            result.message = (
                "Activation stopped part-way. What had already happened is "
                "listed above; nothing further was attempted.")
        return result
