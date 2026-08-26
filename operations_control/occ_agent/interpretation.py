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
    #: Delivery answers that belong to ONE stream rather than to every
    #: registration: ``{"pipeline": {"cadence": "weekly"}}``. "A weekly
    #: pipeline and monthly MI" is two cadences, and folding them into one
    #: ``delivery.cadence`` gave both registrations whichever was read last.
    #: Anything in ``delivery`` remains the blanket answer for the streams that
    #: did not state their own.
    stream_delivery: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    #: The operational data streams the instruction declared, in the order it
    #: named them — the governed dataset vocabulary ("funded", "pipeline").
    #: A stream is a SEPARATE source registration, never a blanket delivery
    #: answer: "a pipeline and a funded book" is two registrations, and folding
    #: both into one ``delivery.dataset`` would silently drop one of them.
    streams: List[str] = field(default_factory=list)
    #: Semantic input roles the instruction said the client would send.
    expected_artefacts: List[str] = field(default_factory=list)
    #: ``section.field`` -> one of PROVENANCE_SOURCES.
    provenance: Dict[str, str] = field(default_factory=dict)
    #: ``section.field`` -> 0..1, for anything read with less than certainty.
    confidence: Dict[str, float] = field(default_factory=dict)
    #: Clauses nothing could be read from. Reported, never dropped.
    unrecognised: List[str] = field(default_factory=list)
    #: Clauses Trakt DID read and deliberately declined to resolve — two values
    #: of one delivery field with no stream to attach either to. They are also
    #: in ``unrecognised``, which is what puts them in front of the operator;
    #: this list is what stops the turn being discarded as saying nothing, so
    #: they see their own words and the reason rather than a generic "could not
    #: tell what to do with that".
    refused: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, doc: Dict[str, Any]) -> "Interpretation":
        known = {f for f in cls.__dataclass_fields__}   # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (doc or {}).items() if k in known})

    @property
    def empty(self) -> bool:
        return not (self.steps or self.reporting_period or self.delivery
                    or self.streams or self.stream_delivery
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
        if self.stream_delivery:
            if delivery_section is None:
                raise InterpretationError("unknown step 'sources'")
            for stream, payload in self.stream_delivery.items():
                if not isinstance(payload, dict):
                    raise InterpretationError(f"{stream} delivery payload")
                _check_keys(delivery_section, payload, DELIVERY_SECTION)
        if self.streams or self.stream_delivery:
            dataset_field = cat.field(DELIVERY_SECTION, "dataset")
            allowed = {str(o.get("value"))
                       for o in ((dataset_field.options if dataset_field
                                  else None) or [])}
            for stream in list(self.streams) + list(self.stream_delivery):
                if allowed and stream not in allowed:
                    raise InterpretationError(f"unknown stream '{stream}'")
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
#: A continuation token may be numeric — "Capital 123" is one name, and a
#: digit cannot carry capitalisation evidence either way.
_NAME_RUN = r"[A-Z][\w&'’-]*(?:\s+[A-Z0-9][\w&'’-]*){0,4}"
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
            if hit.ref.section == DELIVERY_SECTION \
                    and hit.ref.key == "dataset":
                # A dataset mention declares a STREAM — a separate source
                # registration — never a blanket answer for every delivery.
                # "a pipeline and a funded book" is two registrations, and one
                # ``delivery.dataset`` value would silently drop one of them.
                for value in (hit.value if isinstance(hit.value, list)
                              else [hit.value]):
                    stream = str(value)
                    if stream and stream not in out.streams:
                        out.streams.append(stream)
                out.provenance[hit.ref.path] = PROV_HUMAN
                continue
            target = (out.delivery if hit.ref.section == DELIVERY_SECTION
                      else blocks.setdefault(hit.ref.section, {}))
            target[hit.ref.key] = hit.value
            out.provenance[hit.ref.path] = (
                PROV_HUMAN if hit.confidence >= 1.0 else PROV_AGENT)
            if hit.confidence < 1.0:
                out.confidence[hit.ref.path] = hit.confidence

        # Which value belongs to which stream, for every delivery field the
        # catalogue gives a closed option list. Decided from the sentence's own
        # shape rather than from whichever value the generic reader happened to
        # bind last, so "weekly pipeline, monthly MI" registers a weekly
        # pipeline and a monthly funded book.
        #
        # This was cadence-only, and the field it most needed to cover was
        # file_format: a mixed pack ("the funded files are csv and the pipeline
        # files are xlsx") read as ONE blanket format, silently dropping the
        # first and applying the second to every registration. file_format is
        # inferred and required, and inference abstains on a mixed pack — so
        # the operator's only way to supply it wrote the wrong answer to both.
        ambiguous: List[str] = []
        for key in pairable_fields(self.catalogue):
            paired, blanket, unpaired = stream_values(raw, self.catalogue, key)
            for stream, value in paired.items():
                out.stream_delivery.setdefault(stream, {})[key] = value
                if stream not in out.streams:
                    out.streams.append(stream)
                out.provenance[f"{DELIVERY_SECTION}.{key}"] = PROV_HUMAN
            if unpaired:
                # Two values of one field with nothing to pair them to. The
                # generic reader has already bound whichever it read last;
                # that reading is withdrawn rather than applied.
                out.delivery.pop(key, None)
                out.provenance.pop(f"{DELIVERY_SECTION}.{key}", None)
                ambiguous.extend(unpaired)
            elif paired:
                # The blanket is what is LEFT once each stream has taken its
                # own. Without this the paired value would also be applied to
                # every other registration, the defect in the other direction.
                if blanket:
                    out.delivery[key] = blanket
                else:
                    out.delivery.pop(key, None)

        # The telegraphic shape of an opening instruction.
        rows: Dict[str, List[Dict[str, Any]]] = {}
        consumed = self._read_instruction_shape(raw, blocks, rows, out)

        # A declared stream needs a book to hang off. If the instruction named
        # streams but no portfolio, propose one carrying the client's name —
        # the identifier is minted by inference, and the funded and pipeline
        # registrations are then derived from it separately.
        if out.streams and not blocks.get("portfolios") \
                and not rows.get("portfolios"):
            client_name = str((blocks.get("client") or {})
                              .get("client_name") or "")
            if client_name:
                blocks["portfolios"] = {
                    "display_name": f"{client_name} portfolio"}
                out.provenance["portfolios.display_name"] = PROV_AGENT
                out.confidence["portfolios.display_name"] = 0.6

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
        # A clause naming two values of one delivery field, with no stream to
        # attach either to, is reported rather than resolved. It reaches the
        # operator through the same disclosure as anything else Trakt could not
        # read, and the turn applies nothing until they say which is which.
        for clause in ambiguous:
            if clause not in out.unrecognised:
                out.unrecognised.append(clause)
            if clause not in out.refused:
                out.refused.append(clause)
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
    """Whether a follow-up says anything worth proposing as a change.

    This used to refuse a follow-up whose only content was the client's NAME,
    on the reasoning that a bare proper-noun run is usually a false positive
    ("send it to Northstar") and the client is named by then anyway.

    Both halves were wrong in the case that matters. Correcting the client's
    name is the single most likely correction an operator makes — it is the
    first thing the agent reads and the first thing it can get wrong — and
    refusing it left them with a case named after a verb phrase and no way to
    fix it. "Trakt could not tell what to do with that" was the response to the
    one instruction it most needed to understand.

    The false positive the guard existed for is now handled where it belongs:
    :class:`~operations_control.occ_agent.extraction.Candidate.names` will not
    bind a free-text value on a bare topic cue at all, so "send it to
    Northstar" yields nothing to propose rather than being caught here.
    """
    if interpretation.reporting_period or interpretation.delivery:
        return True
    if interpretation.refused or interpretation.stream_delivery:
        return True
    if interpretation.streams or interpretation.expected_artefacts:
        return True
    for step, payload in (interpretation.steps or {}).items():
        if step == "entities":
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


#: Where one delivery statement ends and the next begins. A comma counts:
#: "weekly pipeline, monthly MI" is two statements, and a reader that ignored
#: the comma would happily pair "pipeline" with "monthly".
#:
#: "and" counts for the same reason. "The funded files are csv and the pipeline
#: files are xlsx" is two statements, and read as one the proximity rule paired
#: csv with PIPELINE — the nearest dataset to it by raw character distance is
#: the one it does not belong to, because "and the pipeline" sits between the
#: value and its own stream.
_SEGMENT_RE = re.compile(r"[,.;\n]+|\band\b")

#: How far apart a cadence and its stream may sit and still be one statement,
#: in characters. "a funded book monthly" is a pairing; two clauses apart is a
#: coincidence.
_PAIR_REACH = 40


def _vocabulary(cat: Catalogue, key: str) -> Dict[str, str]:
    """``{spoken phrase: declared value}`` for one delivery field.

    Both the option's value and its label are accepted, because an operator
    writes "ad hoc" where the catalogue stores ``ad_hoc``. The vocabulary is
    the catalogue's own; nothing is listed here.
    """
    field_ = cat.field(DELIVERY_SECTION, key)
    out: Dict[str, str] = {}
    for option in (field_.options if field_ else None) or []:
        value = str(option.get("value") or "")
        if not value:
            continue
        out[value.replace("_", " ").lower()] = value
        label = str(option.get("label") or "").lower()
        if label:
            out[label] = value
    return out


def _mentions(segment: str, vocabulary: Dict[str, str]
              ) -> List[Tuple[int, int, str]]:
    """Every phrase from ``vocabulary`` in ``segment``, as (start, end, value).

    Longest phrases first, and overlapping matches dropped, so "ad hoc" is one
    mention rather than two.
    """
    out: List[Tuple[int, int, str]] = []
    for phrase in sorted(vocabulary, key=len, reverse=True):
        for match in re.finditer(rf"\b{re.escape(phrase)}\b", segment, re.I):
            if any(match.start() < e and s < match.end() for s, e, _ in out):
                continue
            out.append((match.start(), match.end(), vocabulary[phrase]))
    return sorted(out)


def pairable_fields(cat: Catalogue) -> Tuple[str, ...]:
    """The delivery fields a single stream can claim its own value of.

    Every delivery field the catalogue gives a closed option list, except the
    one that NAMES the stream. Derived rather than listed, so a new
    option-backed delivery field pairs without a change here.
    """
    section = cat.section(DELIVERY_SECTION)
    return tuple(f.key for f in ((section.fields if section else None) or [])
                 if f.key != "dataset" and f.options)


def stream_values(text: str, cat: Catalogue, key: str
                  ) -> Tuple[Dict[str, str], str, List[str]]:
    """Which value of one delivery field belongs to which stream.

    Returns ``({stream: value}, blanket_value, ambiguous_clauses)``.

    The rule is proximity within a single statement: a stream takes the value
    nearest to it, closest pairing first, and each value is claimed once. A
    value no stream claims is the blanket answer for every registration that
    did not state its own.

    So "a weekly pipeline, monthly MI" registers a weekly pipeline and leaves
    monthly to the funded book, and "a funded book monthly and a pipeline
    weekly" reads both — where an order-based rule matched whichever pattern it
    tried first and silently dropped the other pairing.

    ``ambiguous_clauses`` is the refusal. When a sentence names two DIFFERENT
    values of one field and nothing pairs them to a stream, there is no honest
    blanket: taking the first drops the second, and taking the last drops the
    first. Both are a wrong answer written confidently, which is worse than no
    answer, so the clauses are handed back to be reported instead.

    Deliberately conservative in one respect: "MI" is not treated as naming the
    funded book. That would be Trakt inferring a stream from a PRODUCT, and the
    same rule would then attach a value to a book the client never mentioned.
    An unpaired value stays blanket, which is both truthful and correctable.
    """
    values = _vocabulary(cat, key)
    datasets = _vocabulary(cat, "dataset")
    if not values or not datasets:
        return {}, "", []

    paired: Dict[str, str] = {}
    blanket = ""
    ambiguous: List[str] = []
    # Sentence by sentence, then statement by statement within it. The pairing
    # happens per statement; the AMBIGUITY is judged per sentence, because two
    # sentences each naming one format are not in conflict — and because the
    # sentence is what gets quoted back, where a bare statement fragment would
    # be quoted at an operator who never wrote it on its own.
    for sentence in re.split(r"[.;\n]+", str(text or "")):
        sentence = sentence.strip()
        if not sentence:
            continue
        loose: List[str] = []
        for segment in _SEGMENT_RE.split(sentence):
            segment = segment.strip()
            if not segment:
                continue
            value_hits = _mentions(segment, values)
            dataset_hits = _mentions(segment, datasets)
            if not value_hits:
                continue

            # Every possible pairing, closest first. Greedy from there, so the
            # nearest reading wins and nothing is claimed twice.
            options = sorted(
                ((min(abs(d_s - v_e), abs(v_s - d_e)), d_s, v_s, stream, value)
                 for d_s, d_e, stream in dataset_hits
                 for v_s, v_e, value in value_hits),
                key=lambda row: row[:3])
            taken_value: set = set()
            taken_stream: set = set()
            for distance, _d_s, v_s, stream, value in options:
                if distance > _PAIR_REACH:
                    continue
                if stream in taken_stream or v_s in taken_value:
                    continue
                taken_stream.add(stream)
                taken_value.add(v_s)
                paired.setdefault(stream, value)

            loose.extend(value for v_s, _v_e, value in value_hits
                         if v_s not in taken_value)

        distinct = list(dict.fromkeys(loose))
        if len(distinct) > 1:
            ambiguous.append(sentence)      # refused, not guessed
        elif distinct and not blanket:
            blanket = distinct[0]
    if ambiguous:
        return paired, "", ambiguous
    return paired, blanket, []


def stream_cadences(text: str, cat: Catalogue) -> Tuple[Dict[str, str], str]:
    """:func:`stream_values` for cadence, in its original two-part shape."""
    paired, blanket, _ambiguous = stream_values(text, cat, "cadence")
    return paired, blanket


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
