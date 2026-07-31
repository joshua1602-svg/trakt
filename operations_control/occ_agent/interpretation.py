"""operations_control.occ_agent.interpretation — natural language in, catalogue answers out.

Two jobs, both ending in a structure the Client Onboarding catalogue validates:

1. :meth:`Interpreter.interpret_instruction` — a sentence becomes an
   :class:`Interpretation`: answers keyed by catalogue section and field, plus
   the delivery facts the catalogue deliberately does not hold, plus **the
   fragments it could not read**.
2. :meth:`Interpreter.interpret_action` — a follow-up becomes a
   :class:`ProposedChange`: one of the lifecycle's named actions and a payload.

Field recognition itself lives in :mod:`.extraction` and is derived from the
catalogue, so a field added to ``field_catalogue.yaml`` becomes answerable
without a change here. What this module adds is the *shape* of an operator's
opening instruction, which is telegraphic in a way no field declaration
describes — "Onboard Northstar Lending. UK equity release." names a client with
a verb and an asset class with an adjective.

Four rules hold whatever produces the structure:

* **The interpreter never decides state.** It proposes an action; controls run;
  :mod:`.states`, or Client Onboarding's own table, decides the move.
* **The interpreter never writes.** Only the service persists, and only through
  ``OnboardingService``.
* **Every field it produces must exist in the catalogue**, so a model swapped in
  behind :class:`Interpreter` cannot invent one.
* **What it could not read is reported.** ``unrecognised`` is part of the
  output, not a silent remainder, and the service refuses to apply a plan
  carrying one without explicit confirmation.

It answers only what the sentence says. The client identifier, the reporting
currency, the period convention and everything else Trakt works out for itself
are left to :mod:`operations_control.onboarding.inference`, which already does
it, records provenance and lets an operator override.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Protocol, Tuple

from ..engine import OpsError
from ..onboarding.case import OnboardingCase
from ..onboarding.catalogue import Catalogue, catalogue
from ..onboarding.service import STEPS
from . import extraction as _extraction
from . import states as _states
from .input_roles import artefact_vocabulary
from .run import SyntheticRun

#: Where an answer came from. The six sources the governed record uses.
PROV_HUMAN = "human_supplied"
PROV_CLIENT = "client_supplied"
PROV_ARTEFACT = "artefact_derived"
PROV_AGENT = "agent_proposed"
PROV_APPROVED = "human_approved"
PROV_INHERITED = "inherited_configuration"

#: Three more, for values nobody typed. A pre-populated field is not one thing:
#: an operator's decision, a value read out of an existing client record, a
#: number Trakt computed and a governed default are four different claims, and
#: an approver reviewing them needs to be able to tell which is which.
PROV_EXISTING_RECORD = "existing_record"
PROV_TRAKT_DERIVED = "trakt_derived"
PROV_INHERITED_DEFAULT = "inherited_default"

PROVENANCE_SOURCES = (PROV_HUMAN, PROV_CLIENT, PROV_ARTEFACT, PROV_AGENT,
                      PROV_APPROVED, PROV_INHERITED, PROV_EXISTING_RECORD,
                      PROV_TRAKT_DERIVED, PROV_INHERITED_DEFAULT)

#: Retained for the older vocabulary used in tests and stored cases.
PROV_HUMAN_INSTRUCTION = PROV_HUMAN
PROV_AGENT_INFERENCE = PROV_AGENT

MAX_VALUE_CHARS = 2000
MAX_ITEMS = 50

#: Sections whose answers apply to every derived delivery rather than to one
#: item. Deliveries are derived from the portfolios, so an operator saying
#: "they deliver monthly" means all of them.
DELIVERY_SECTION = "sources"


class InterpretationError(OpsError):
    """The instruction could not be turned into anything actionable."""

    def __init__(self, detail: str = ""):
        super().__init__(
            "OCC_AGENT_NOT_UNDERSTOOD",
            "Trakt could not tell what to do with that. Try naming the change "
            "directly — for example 'the LEI is 894500...', 'map Current "
            "Balance to current outstanding balance', or 'what is still "
            "needed?'." + (f" ({detail})" if detail else ""),
            http_status=422)


# --------------------------------------------------------------------------- #
# What an instruction produces
# --------------------------------------------------------------------------- #

@dataclass
class Interpretation:
    """Catalogue-shaped answers read out of one instruction.

    ``steps`` is ``{step: payload}`` in the shape ``save_step`` takes, so the
    answers reach the case through the platform's own writer.
    """

    steps: Dict[str, Any] = field(default_factory=dict)
    #: Delivery-specific, so not a catalogue answer: it lives on the run.
    reporting_period: str = ""
    #: Answers for every derived delivery — cadence, channel, format, sender.
    delivery: Dict[str, Any] = field(default_factory=dict)
    #: Semantic input roles the instruction said the client would send.
    expected_artefacts: List[str] = field(default_factory=list)
    #: ``section.field`` -> one of PROVENANCE_SOURCES.
    provenance: Dict[str, str] = field(default_factory=dict)
    #: ``section.field`` -> 0..1, for anything read with less than certainty.
    confidence: Dict[str, float] = field(default_factory=dict)
    #: Clauses nothing could be read from. Reported, never dropped.
    unrecognised: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, doc: Dict[str, Any]) -> "Interpretation":
        known = {f for f in cls.__dataclass_fields__}   # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (doc or {}).items() if k in known})

    @property
    def empty(self) -> bool:
        return not (self.steps or self.reporting_period or self.delivery
                    or self.expected_artefacts)

    @property
    def complete(self) -> bool:
        return not self.unrecognised

    def validate(self, cat: Optional[Catalogue] = None) -> None:
        """Every proposed answer must be a field the catalogue declares."""
        cat = cat or catalogue()
        for step, payload in (self.steps or {}).items():
            if step not in STEPS:
                raise InterpretationError(f"unknown step '{step}'")
            section = cat.section(step)
            if section is None:
                raise InterpretationError(f"unknown step '{step}'")
            if section.repeatable:
                items = payload.get(section.key) or payload.get("items") or []
                if not isinstance(items, list) or len(items) > MAX_ITEMS:
                    raise InterpretationError(f"{step} items")
                for item in items:
                    _check_keys(section, item, step)
            else:
                if not isinstance(payload, dict):
                    raise InterpretationError(f"{step} payload")
                _check_keys(section, payload, step)
        delivery_section = cat.section(DELIVERY_SECTION)
        if self.delivery and delivery_section is not None:
            _check_keys(delivery_section, self.delivery, DELIVERY_SECTION)
        if self.reporting_period and not re.fullmatch(
                r"\d{4}-\d{2}-\d{2}", self.reporting_period):
            raise InterpretationError("reporting period")
        for value in self.confidence.values():
            if not 0.0 <= float(value) <= 1.0:
                raise InterpretationError("confidence must be 0..1")
        for source in self.provenance.values():
            if source not in PROVENANCE_SOURCES:
                raise InterpretationError(f"unknown provenance '{source}'")


def _check_keys(section, item: Any, step: str) -> None:
    if not isinstance(item, dict):
        raise InterpretationError(f"{step} item")
    for key, value in item.items():
        if section.field(key) is None:
            raise InterpretationError(
                f"'{key}' is not a field of {section.label}")
        if isinstance(value, str) and len(value) > MAX_VALUE_CHARS:
            raise InterpretationError(f"value for {key} is too long")
        if isinstance(value, list):
            if len(value) > MAX_ITEMS:
                raise InterpretationError(f"value for {key} has too many items")
            for entry in value:
                if isinstance(entry, str) and len(entry) > MAX_VALUE_CHARS:
                    raise InterpretationError(f"value for {key} is too long")


@dataclass
class ProposedChange:
    """One structured change proposed from a natural-language instruction."""

    action: str                              # a _states ACTION_* value
    payload: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    confidence: float = 1.0
    material: bool = False
    requires_confirmation: bool = False
    #: Short, non-sensitive evidence. Never chain-of-thought.
    basis: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.action not in _ALL_ACTIONS:
            raise InterpretationError(f"unknown action '{self.action}'")
        if not isinstance(self.payload, dict):
            raise InterpretationError("payload must be a mapping")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise InterpretationError("confidence must be 0..1")
        for key, value in self.payload.items():
            if not isinstance(key, str) or len(key) > 64:
                raise InterpretationError("payload key")
            if isinstance(value, str) and len(value) > MAX_VALUE_CHARS:
                raise InterpretationError(f"payload value for {key} is too long")


_ALL_ACTIONS = frozenset(
    _states.ONBOARDING_ACTIONS + _states.EXECUTION_ACTIONS)


class Interpreter(Protocol):
    """The seam a model-backed interpreter would implement."""

    def interpret_instruction(self, text: str) -> Interpretation: ...

    def interpret_action(self, text: str, run: SyntheticRun,
                         case: OnboardingCase) -> ProposedChange: ...


# --------------------------------------------------------------------------- #
# The shape of an opening instruction
# --------------------------------------------------------------------------- #

#: "Onboard Northstar Lending." — the client name is the proper-noun run after
#: the verb. The verb is case-insensitive; the NAME is not, because
#: capitalisation is the evidence that it IS a name. A full stop ends the run.
_NAME_RUN = r"[A-Z][\w&'’-]*(?:\s+[A-Z][\w&'’-]*){0,4}"
_ONBOARD_RE = re.compile(
    r"\b(?i:onboard(?:ing)?|set\s+up|bring\s+on)\s+"
    r"(?i:a\s+new\s+|the\s+|new\s+)?(?i:client\s+|portfolio\s+|lender\s+)?"
    rf"({_NAME_RUN})")
_CLIENT_NAMED_RE = re.compile(
    rf"\b(?i:client)\s+(?i:is\s+|name\s+is\s+|:\s*)?({_NAME_RUN})")

#: Entity roles an instruction can name a counterparty in. The ROLES come from
#: the catalogue's own option list; only the sentence shape is here.
_ROLE_RE = r"\b(?i:{role})\s+(?i:is\s+|are\s+)?({name})"


def _trim_name(candidate: str) -> str:
    """Keep the proper-noun run, dropping a sentence continuation.

    "Northstar Lending It Is A" would otherwise survive capitalised-run
    matching at a sentence boundary.
    """
    stop = {"it", "its", "it's", "they", "the", "this", "we", "please",
            "their", "there", "uk", "us"}
    tokens = [t for t in str(candidate or "").split() if t]
    kept: List[str] = []
    for t in tokens:
        if t.lower() in stop:
            break
        kept.append(t)
    while kept and kept[-1].lower() in stop:
        kept.pop()
    return " ".join(kept).strip(" .,;:")


# --------------------------------------------------------------------------- #
# Deterministic interpreter
# --------------------------------------------------------------------------- #

class DeterministicInterpreter:
    """Rule-based interpretation. Offline, reproducible, no credentials.

    Conservative by design: what it cannot read with confidence is *reported*
    rather than guessed, because a reported gap is a prompt to the human and a
    silent inference is not.
    """

    def __init__(self, *, today: Optional[date] = None,
                 cat: Optional[Catalogue] = None):
        self.today = today
        self.catalogue = cat or catalogue()

    def match_asset(self, text: str) -> Tuple[str, float]:
        """The asset class a phrase names, from the product profiles' own
        vocabulary. ``("", 0.0)`` when the phrase names none."""
        return _extraction.match_asset(text, self.catalogue)

    # -- 1. an instruction ---------------------------------------------- #
    def interpret_instruction(self, text: str) -> Interpretation:
        raw = str(text or "").strip()
        if not raw:
            raise InterpretationError("empty instruction")
        out = Interpretation()

        reading = _extraction.read(raw, self.catalogue)
        out.reporting_period = reading.reporting_period

        # Every catalogue field the text named, grouped by section.
        blocks: Dict[str, Dict[str, Any]] = {}
        for hit in reading.found:
            target = (out.delivery if hit.ref.section == DELIVERY_SECTION
                      else blocks.setdefault(hit.ref.section, {}))
            target[hit.ref.key] = hit.value
            out.provenance[hit.ref.path] = (
                PROV_HUMAN if hit.confidence >= 1.0 else PROV_AGENT)
            if hit.confidence < 1.0:
                out.confidence[hit.ref.path] = hit.confidence

        # The telegraphic shape of an opening instruction.
        rows: Dict[str, List[Dict[str, Any]]] = {}
        consumed = self._read_instruction_shape(raw, blocks, rows, out)

        for section_key, values in blocks.items():
            section = self.catalogue.section(section_key)
            if section is None:              # pragma: no cover — refs are real
                continue
            if section.repeatable:
                rows.setdefault(section_key, []).insert(0, values)
            else:
                out.steps[section_key] = values
        for section_key, items in rows.items():
            out.steps[section_key] = {section_key: items}

        vocabulary = artefact_vocabulary()
        roles = vocabulary.match_all(raw.lower())
        if roles:
            out.expected_artefacts = roles

        # A clause that names the files the client will send IS understood — it
        # is recorded on the run rather than on the case, which is not the same
        # thing as being missed.
        out.unrecognised = [
            clause for clause in reading.unrecognised
            if clause not in consumed
            and not vocabulary.match_all(clause.lower())]
        out.validate(self.catalogue)
        return out

    def _read_instruction_shape(self, raw: str,
                                blocks: Dict[str, Dict[str, Any]],
                                rows: Dict[str, List[Dict[str, Any]]],
                                out: Interpretation) -> set:
        """Name a client with a verb, and a counterparty with its role.

        Returns the clauses this consumed, so they are not reported as
        unrecognised.
        """
        consumed: set = set()
        client = blocks.setdefault("client", {})

        name = ""
        for pattern in (_ONBOARD_RE, _CLIENT_NAMED_RE):
            m = pattern.search(raw)
            if m:
                name = _trim_name(m.group(1))
                if name:
                    consumed.add(_clause_of(raw, m.start()))
                    break
        if name and not client.get("client_name"):
            client["client_name"] = name
            out.provenance["client.client_name"] = PROV_HUMAN

        named = self._entities(raw, name, consumed)
        if named:
            # A cued entity answer ("the LEI is …") describes the entity the
            # sentence is about, which is the first one it named — but only for
            # a field no named entity has already answered. "The servicer is
            # Meridian Servicing" answers Meridian's role, and must not also
            # land on the client's row.
            cued = blocks.pop("entities", None) or {}
            spoken = {key for row in named for key in row}
            for key, value in cued.items():
                if key not in spoken:
                    named[0][key] = value
            rows["entities"] = named
            out.provenance.setdefault("entities.legal_name", PROV_HUMAN)

        if not client:
            blocks.pop("client", None)
        if self.catalogue.section("portfolios") and blocks.get("portfolios") \
                and name and not blocks["portfolios"].get("display_name"):
            blocks["portfolios"]["display_name"] = f"{name} portfolio"
            out.provenance["portfolios.display_name"] = PROV_AGENT
            out.confidence["portfolios.display_name"] = 0.6
        return consumed

    def _entities(self, raw: str, client_name: str,
                  consumed: set) -> List[Dict[str, Any]]:
        """Counterparties named with their role, plus the client as originator.

        The roles are the catalogue's own option list for ``entities.roles``.
        """
        f = self.catalogue.field("entities", "roles")
        roles = [str(o.get("value")) for o in ((f.options if f else None) or [])]
        rows: List[Dict[str, Any]] = []
        seen: set = set()
        for role in roles:
            word = role.replace("_", " ")
            m = re.search(_ROLE_RE.format(role=re.escape(word),
                                          name=_NAME_RUN), raw)
            if not m:
                continue
            legal_name = _trim_name(m.group(1))
            if not legal_name or legal_name.lower() in seen:
                continue
            seen.add(legal_name.lower())
            consumed.add(_clause_of(raw, m.start()))
            rows.append({"legal_name": legal_name, "roles": [role]})
        if client_name and client_name.lower() not in seen \
                and not any("originator" in r["roles"] for r in rows):
            # The client is the originator unless the instruction named someone
            # else. Proposed like any other value, and shown for confirmation.
            rows.insert(0, {"legal_name": client_name,
                            "roles": ["originator"]})
        return rows

    # -- 2. a follow-up ------------------------------------------------- #
    def interpret_action(self, text: str, run: SyntheticRun,
                         case: OnboardingCase) -> ProposedChange:
        raw = str(text or "").strip()
        if not raw:
            raise InterpretationError("empty instruction")
        lower = raw.lower()

        # A question must never be read as an instruction.
        if self._is_question(lower):
            change = ProposedChange(
                action=_states.ACTION_ASK,
                payload={"question": raw[:MAX_VALUE_CHARS]},
                summary="Answer a question about this case.",
                basis="the message is phrased as a question")
            change.validate()
            return change

        mapping = self._mapping_change(raw)
        if mapping is not None:
            mapping.validate()
            return mapping

        for pattern, action, summary, material in _ACTION_RULES:
            if re.search(pattern, lower):
                change = ProposedChange(
                    action=action, summary=summary, material=material,
                    requires_confirmation=material,
                    basis="explicit instruction")
                change.validate()
                return change

        if re.fullmatch(r"(yes|confirm(ed)?|approved?|go ahead|proceed)\.?",
                        lower):
            action = pending_confirmation_action(run, case)
            if action is None:
                raise InterpretationError("nothing is awaiting confirmation")
            change = ProposedChange(
                action=action,
                summary="Confirm what this case is waiting on.",
                basis="bare confirmation resolved against the current state")
            change.validate()
            return change

        answer = self._answer_change(raw, case)
        if answer is not None:
            answer.validate()
            return answer

        raise InterpretationError()

    @staticmethod
    def _is_question(lower: str) -> bool:
        if lower.rstrip().endswith("?"):
            return True
        return bool(re.match(r"^(what|why|which|when|who|how|is|are|does|do|"
                             r"can|could|should)\b", lower))

    def _mapping_change(self, raw: str) -> Optional[ProposedChange]:
        m = re.search(r"\bmap\s+(.+?)\s+to\s+(.+?)(?:[.;]|$)", raw, re.I)
        if not m:
            return None
        source = m.group(1).strip().strip("'\"")
        target = m.group(2).strip().strip("'\"")
        if not source or not target:
            return None
        canonical = re.sub(r"[^a-z0-9]+", "_", target.lower()).strip("_")
        return ProposedChange(
            action=_states.ACTION_RESOLVE_DECISION,
            payload={"kind": "field_mapping", "source_column": source,
                     "canonical_field": canonical, "resolution": "amend"},
            summary=f"Map '{source}' to '{canonical.replace('_', ' ')}'.",
            material=True, requires_confirmation=True,
            basis="explicit mapping instruction")

    def _answer_change(self, raw: str,
                       case: OnboardingCase) -> Optional[ProposedChange]:
        """A sentence that answers the onboarding.

        Read with the same extractor the opening instruction uses, so an answer
        reaches ``save_step`` in the shape the wizard would have written.
        """
        try:
            interpretation = self.interpret_instruction(raw)
        except InterpretationError:
            return None
        if not _says_something(interpretation, case):
            return None
        return ProposedChange(
            action=_states.ACTION_ANSWER,
            payload={"interpretation": interpretation.to_dict()},
            summary="Answer the onboarding.",
            material=True, requires_confirmation=True,
            confidence=(1.0 if interpretation.complete else 0.5),
            basis=("a fact was stated directly" if interpretation.complete
                   else "part of the message could not be read"))


def _says_something(interpretation: Interpretation,
                    case: OnboardingCase) -> bool:
    """Whether a follow-up answers anything beyond naming somebody.

    A bare proper-noun run on a follow-up is almost always a false positive
    ("send it to Northstar"), and the client is already named by then.
    """
    if interpretation.reporting_period or interpretation.delivery:
        return True
    if interpretation.expected_artefacts:
        return True
    for step, payload in (interpretation.steps or {}).items():
        if step == "client":
            if set(payload) - {"client_name"}:
                return True
        elif step == "entities":
            for item in (payload.get("entities") or []):
                if set(item) - {"legal_name", "roles"}:
                    return True
        elif step == "portfolios":
            for item in (payload.get("portfolios") or []):
                if set(item) - {"display_name"}:
                    return True
        elif payload:
            return True
    return False


def _clause_of(text: str, position: int) -> str:
    """The clause containing ``position``, as :mod:`.extraction` would split it."""
    start = max((text.rfind(ch, 0, position) for ch in ".;\n"), default=-1)
    ends = [i for i in (text.find(ch, position) for ch in ".;\n") if i != -1]
    end = min(ends) if ends else len(text)
    return text[start + 1:end].strip()


#: (pattern, action, operator summary, material). Ordered — first match wins.
_ACTION_RULES: Tuple[Tuple[str, str, str, bool], ...] = (
    (r"\bcancel (this )?(case|run)\b", _states.ACTION_CANCEL,
     "Cancel this practice case.", True),
    (r"\bwithdraw\b", _states.ACTION_WITHDRAW,
     "Withdraw the onboarding.", True),
    (r"\b(draft|prepare|generate|build)\b.*\b(pack|questionnaire|email|"
     r"onboarding pack)\b", _states.ACTION_DRAFT_PACK,
     "Draft the onboarding pack and covering email.", False),
    (r"\b(approve|ok)\b.*\b(pack|email|communication)\b",
     _states.ACTION_APPROVE_PACK, "Approve the pack for sending.", True),
    (r"\b(send|issue)\b.*\b(pack|email)\b", _states.ACTION_SEND_PACK,
     "Record the pack as issued to the client.", True),
    (r"\bapprove\b.*\breadiness\b|\bconfirm\b.*\bready for execution\b"
     r"|\bapprove (the )?execution\b",
     _states.ACTION_APPROVE_EXECUTION,
     "Approve readiness for execution.", True),
    # Ordered deliberately: confirming production is matched before approving
    # it, and approving before merely asking for review. A sentence that could
    # be read as more than one of these is read as the LEAST consequential.
    (r"\bconfirm\b.*\bactivat", _states.ACTION_CONFIRM_ACTIVATION,
     "Confirm activation. This is production.", True),
    (r"\b(approve|authorise|authorize)\b.*\bactivat"
     r"|\bapprove\b.*\bconfiguration\b",
     _states.ACTION_APPROVE_ACTIVATION,
     "Approve the configuration for activation. This starts nothing.", True),
    (r"\b(submit|send)\b.*\b(for )?review\b|\bready for review\b",
     _states.ACTION_REQUEST_ACTIVATION,
     "Assemble the review package and submit the case for review.", False),
    (r"\b(activate|go live|start (the )?live)\b",
     _states.ACTION_REQUEST_ACTIVATION,
     "Ask to activate this configuration and start ingestion.", True),
    (r"\b(approve|accept)\b.*\bonboarding\b|\bapprove (the )?case\b",
     _states.ACTION_APPROVE_ONBOARDING, "Approve the onboarding.", True),
    (r"\b(submit|send)\b.*\b(for )?approval\b",
     _states.ACTION_SUBMIT_FOR_APPROVAL,
     "Submit the onboarding for approval.", True),
    (r"\brequest changes?\b|\bsend (it )?back\b",
     _states.ACTION_REQUEST_CHANGES, "Send the onboarding back for changes.",
     True),
    (r"\b(ask|request|chase)\b.*\b(client|information|checklist|outstanding)\b",
     _states.ACTION_REQUEST_INFORMATION,
     "Ask the client for what is still outstanding.", False),
    (r"\b(record|log)\b.*\b(response|reply|answer)\b",
     _states.ACTION_RECORD_RESPONSE, "Record the client's response.", True),
    (r"\b(generate|prepare)\b.*\b(plan|orchestration)\b",
     _states.ACTION_GENERATE_PLAN, "Generate the orchestration plan.", False),
    (r"\b(run|start|re-?run)\b.*\b(onboarding|practice run|synthetic run|"
     r"controls)\b",
     _states.ACTION_RUN_ONBOARDING, "Run the practice onboarding.", False),
    (r"\backnowledge\b", _states.ACTION_ACKNOWLEDGE_EXCEPTION,
     "Acknowledge the non-blocking exception.", False),
)


def pending_confirmation_action(run: SyntheticRun,
                                case: OnboardingCase) -> Optional[str]:
    """What a bare 'yes' means, given where the onboarding and the run are.

    The onboarding half is asked first: until the case is approved the
    execution half has nothing to confirm.
    """
    from ..onboarding.case import (
        APPROVED, CHANGES_REQUIRED, DRAFT, IN_REVIEW, READY_FOR_APPROVAL,
    )
    if case.status in (DRAFT, IN_REVIEW, CHANGES_REQUIRED):
        return _states.ACTION_SUBMIT_FOR_APPROVAL
    if case.status == READY_FOR_APPROVAL:
        return _states.ACTION_APPROVE_ONBOARDING
    if case.status != APPROVED:
        return None
    # ACTIVATION_CONFIRMATION_REQUIRED is deliberately absent. A bare "yes"
    # must never start production: the confirmation has to name what it is
    # confirming, which is what ACTION_CONFIRM_ACTIVATION requires.
    return {
        _states.READY_TO_RUN: _states.ACTION_RUN_ONBOARDING,
        _states.SYNTHETIC_ONBOARDING_PASSED: _states.ACTION_GENERATE_PLAN,
        _states.ORCHESTRATION_PLAN_GENERATED: _states.ACTION_APPROVE_EXECUTION,
        _states.EXECUTION_APPROVAL_REQUIRED: _states.ACTION_APPROVE_EXECUTION,
        _states.READY_FOR_EXECUTION: _states.ACTION_REQUEST_ACTIVATION,
        _states.READY_FOR_REVIEW: _states.ACTION_APPROVE_ACTIVATION,
    }.get(run.state)


def reset_cache() -> None:
    """Test seam: clear the configuration-derived caches."""
    _extraction.reset_cache()
