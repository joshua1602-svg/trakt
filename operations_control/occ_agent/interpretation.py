"""operations_control.occ_agent.interpretation — natural language in, typed proposals out.

Two jobs, both of which end in a schema-validated structure:

1. :meth:`Interpreter.interpret_instruction` — an initiating instruction becomes
   :class:`~operations_control.occ_agent.case.ExtractedRequirements`.
2. :meth:`Interpreter.interpret_action` — a follow-up sentence becomes a
   :class:`ProposedChange`: one of the lifecycle's named actions plus a typed
   payload.

Three rules hold whatever produces the structure:

* **The interpreter never decides state.** It proposes an action; the service
  runs the deterministic controls and asks :mod:`.states` whether the resulting
  transition is legal. An action the current state does not allow is refused
  there, not here.
* **The interpreter never writes.** It returns values; only the service
  persists, and only through the store.
* **Every output is schema-validated** before it reaches the case, so a model
  swapped in behind :class:`Interpreter` cannot widen the contract.

The default :class:`DeterministicInterpreter` is rule-based and offline: the
feature must work with no model credentials, and a deterministic interpreter is
what makes the synthetic fixtures reproducible. It is injected, so a
model-backed interpreter can be substituted without touching the service — and
its output still passes through the same validation and the same controls.

Vocabulary is read from the existing configuration (product profiles, the asset
support model, the OCC outcome vocabulary and the input-requirements role
labels) rather than hard-coded, so a new asset or product becomes recognisable
by configuration change.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Protocol, Tuple

from ..engine import OpsError
from . import states as _states
from .case import (
    ExtractedRequirements,
    PROV_AGENT_INFERENCE,
    PROV_HUMAN_INSTRUCTION,
    SyntheticCase,
)
from .vocabulary import (
    artefact_vocabulary,
    asset_vocabulary,
    frequency_vocabulary,
    product_vocabulary,
)


class InterpretationError(OpsError):
    """The instruction could not be turned into anything actionable."""

    def __init__(self, detail: str = ""):
        super().__init__(
            "OCC_AGENT_NOT_UNDERSTOOD",
            "Trakt could not tell what to do with that. Try naming the change "
            "directly — for example 'confirm the interpretation', 'map Current "
            "Balance to current outstanding balance', or 'what is still "
            "needed?'." + (f" ({detail})" if detail else ""),
            http_status=422)


@dataclass
class ProposedChange:
    """One structured change proposed from a natural-language instruction.

    ``requires_confirmation`` is the interpreter's *opening* position; the
    service raises it (never lowers it) once materiality is known.
    """

    action: str                              # a _states ACTION_* value
    payload: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""                        # what the human will see proposed
    confidence: float = 1.0
    material: bool = False
    requires_confirmation: bool = False
    #: Short, non-sensitive evidence for the proposal. Never chain-of-thought.
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
            if isinstance(value, str) and len(value) > 2000:
                raise InterpretationError(f"payload value for {key} is too long")


_ALL_ACTIONS = frozenset({
    _states.ACTION_CONFIRM_REQUIREMENTS, _states.ACTION_CORRECT_REQUIREMENTS,
    _states.ACTION_GENERATE_PACK, _states.ACTION_APPROVE_PACK,
    _states.ACTION_ISSUE_PACK, _states.ACTION_REGISTER_ARTEFACT,
    _states.ACTION_CLASSIFY_ARTEFACTS, _states.ACTION_DRAFT_CONFIG,
    _states.ACTION_APPROVE_CONFIG, _states.ACTION_CORRECT_CONFIG,
    _states.ACTION_RUN_ONBOARDING, _states.ACTION_RESOLVE_DECISION,
    _states.ACTION_ACKNOWLEDGE_EXCEPTION, _states.ACTION_GENERATE_PLAN,
    _states.ACTION_APPROVE_EXECUTION, _states.ACTION_RETURN_TO_STAGE,
    _states.ACTION_CANCEL, _states.ACTION_ASK,
})


class Interpreter(Protocol):
    """The seam a model-backed interpreter would implement."""

    def interpret_instruction(self, text: str) -> ExtractedRequirements: ...

    def interpret_action(self, text: str,
                         case: SyntheticCase) -> ProposedChange: ...


# --------------------------------------------------------------------------- #
# Deterministic interpreter
# --------------------------------------------------------------------------- #

_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_MONTH_NAMES = ("january", "february", "march", "april", "may", "june", "july",
                "august", "september", "october", "november", "december")
_MONTH_YEAR_RE = re.compile(
    r"\b(" + "|".join(_MONTH_NAMES) + r")\s+(\d{4})\b", re.I)

#: "Onboard Northstar Lending." / "onboard client Northstar Lending" — the
#: client name is the proper-noun run that follows the verb. The verb is
#: case-insensitive (scoped inline flag); the NAME is not, because
#: capitalisation is the evidence that it is a name.
#:
#: A full stop ENDS the run rather than continuing it: "Onboard Northstar
#: Lending. UK equity release." must yield "Northstar Lending", not "Northstar
#: Lending. UK". So ``.`` is excluded from the token charset and allowed only as
#: a trailing character, which is then stripped.
_NAME_RUN = r"[A-Z][\w&'’-]*(?:\s+[A-Z][\w&'’-]*){0,4}"
_ONBOARD_RE = re.compile(
    r"\b(?i:onboard(?:ing)?)\s+(?i:a\s+new\s+|the\s+|new\s+)?"
    r"(?i:client\s+|portfolio\s+)?"
    rf"({_NAME_RUN})")

_GO_LIVE_RE = re.compile(
    r"\b(?:go[- ]?live|live)\b[^.]{0,40}?(\d{4}-\d{2}-\d{2})", re.I)
_FIRST_REPORTING_RE = re.compile(
    r"\b(?:first\s+)?report(?:ing)?\s+(?:date|period)\b[^.]{0,40}?"
    r"(\d{4}-\d{2}-\d{2})", re.I)

#: A portfolio identifier stated outright, e.g. "portfolio id direct_001".
_PORTFOLIO_ID_RE = re.compile(
    r"\bportfolio\s+(?:id|identifier|ref(?:erence)?)\s*[:=]?\s*"
    r"([A-Za-z0-9][A-Za-z0-9_.-]{1,63})", re.I)
_CLIENT_ID_RE = re.compile(
    r"\bclient\s+(?:id|identifier|code)\s*[:=]?\s*"
    r"([A-Za-z0-9][A-Za-z0-9_.-]{1,63})", re.I)

#: Jurisdiction tokens. Two-letter ISO codes are too ambiguous in free text to
#: match on their own, so only explicit names/adjectives are recognised.
_JURISDICTIONS: Tuple[Tuple[str, str], ...] = (
    ("united kingdom", "GB"), ("uk ", "GB"), (" uk", "GB"), ("british", "GB"),
    ("england", "GB"), ("scotland", "GB"), ("wales", "GB"),
    ("ireland", "IE"), ("irish", "IE"),
    ("netherlands", "NL"), ("dutch", "NL"),
    ("spain", "ES"), ("spanish", "ES"),
    ("france", "FR"), ("french", "FR"),
    ("germany", "DE"), ("german", "DE"),
)

_LEGAL_SUFFIXES = ("lending", "limited", "ltd", "plc", "capital", "partners",
                   "finance", "financial", "group", "holdings", "bank",
                   "mortgages", "advances", "llp", "sa", "nv", "bv", "gmbh")


def _trim_identifier(value: str) -> str:
    """Drop sentence punctuation an identifier picked up from prose.

    ``.`` and ``-`` are legal INSIDE an identifier, so they cannot simply be
    excluded from the pattern; they are only noise at the very end.
    """
    return str(value or "").strip().rstrip(".,;:-_")


def _slug_id(name: str, *, max_len: int = 32) -> str:
    """A proposed identifier from a display name. Deterministic and path-safe."""
    tokens = [t for t in re.split(r"[^A-Za-z0-9]+", str(name or "")) if t]
    if not tokens:
        return ""
    joined = "_".join(t.lower() for t in tokens)[:max_len].strip("_")
    return joined


class DeterministicInterpreter:
    """Rule-based interpretation. Offline, reproducible, no credentials.

    It is deliberately conservative: anything it cannot read with confidence
    becomes an unresolved question rather than an inferred value, because an
    unresolved question is a prompt to the human and a silent inference is not.
    """

    def __init__(self, *, today: Optional[date] = None):
        # Injected so fixtures and tests are reproducible; ``None`` means the
        # interpreter derives nothing from the current date (it never guesses a
        # reporting date, so this only affects sanity checks).
        self.today = today

    # -- 1. the initiating instruction ---------------------------------- #
    def interpret_instruction(self, text: str) -> ExtractedRequirements:
        raw = str(text or "").strip()
        if not raw:
            raise InterpretationError("empty instruction")
        lower = raw.lower()
        req = ExtractedRequirements()
        prov: Dict[str, str] = {}
        conf: Dict[str, float] = {}

        # Client name.
        name = ""
        m = _ONBOARD_RE.search(raw)
        if m:
            name = self._trim_client_name(m.group(1))
        if not name:
            m2 = re.search(r"\b(?i:client)\s+(?i:is\s+|name\s+is\s+|:\s*)?"
                           rf"({_NAME_RUN})", raw)
            if m2:
                name = self._trim_client_name(m2.group(1))
        if name:
            req.client_name = name
            prov["client_name"] = PROV_HUMAN_INSTRUCTION

        # Identifiers: explicit if stated, otherwise proposed from the name.
        cm = _CLIENT_ID_RE.search(raw)
        if cm:
            req.proposed_client_id = _trim_identifier(cm.group(1))
            prov["proposed_client_id"] = PROV_HUMAN_INSTRUCTION
        elif req.client_name:
            req.proposed_client_id = _slug_id(req.client_name)
            prov["proposed_client_id"] = PROV_AGENT_INFERENCE
            conf["proposed_client_id"] = 0.6

        pm = _PORTFOLIO_ID_RE.search(raw)
        if pm:
            req.proposed_portfolio_id = _trim_identifier(pm.group(1))
            prov["proposed_portfolio_id"] = PROV_HUMAN_INSTRUCTION

        pn = re.search(r"\b(?i:portfolio)\s+(?i:called|named)\s+"
                       rf"({_NAME_RUN})", raw)
        if pn:
            req.portfolio_name = pn.group(1).strip()
            prov["portfolio_name"] = PROV_HUMAN_INSTRUCTION

        # Asset type — matched against the product-profile signal vocabulary.
        asset, asset_conf = asset_vocabulary().match(lower)
        if asset:
            req.asset_type = asset
            prov["asset_type"] = (PROV_HUMAN_INSTRUCTION if asset_conf >= 0.9
                                  else PROV_AGENT_INFERENCE)
            if asset_conf < 0.9:
                conf["asset_type"] = asset_conf

        # Jurisdiction.
        padded = f" {lower} "
        for token, code in _JURISDICTIONS:
            if token in padded:
                req.jurisdiction = code
                prov["jurisdiction"] = PROV_HUMAN_INSTRUCTION
                break

        # Products — matched against the configured product catalogue.
        products = product_vocabulary().match_all(lower)
        if products:
            req.required_products = products
            prov["required_products"] = PROV_HUMAN_INSTRUCTION

        # Frequency.
        freq = frequency_vocabulary().match(lower)
        if freq:
            req.reporting_frequency = freq
            prov["reporting_frequency"] = PROV_HUMAN_INSTRUCTION

        # Expected artefacts — the semantic input roles named in the text.
        artefacts = artefact_vocabulary().match_all(lower)
        if artefacts:
            req.expected_artefacts = artefacts
            prov["expected_artefacts"] = PROV_HUMAN_INSTRUCTION

        # Dates. Only accepted when explicitly qualified: an unqualified date in
        # a sentence about go-live must not become the reporting date.
        gl = _GO_LIVE_RE.search(raw)
        if gl:
            req.target_go_live_date = gl.group(1)
            prov["target_go_live_date"] = PROV_HUMAN_INSTRUCTION
        fr = _FIRST_REPORTING_RE.search(raw)
        if fr:
            req.reporting_date = fr.group(1)
            prov["reporting_date"] = PROV_HUMAN_INSTRUCTION

        # Counterparties.
        for label in ("warehouse", "investor", "funder", "servicer", "trustee"):
            cp = re.search(rf"\b(?i:{label})\s+(?i:is\s+)?({_NAME_RUN})", raw)
            if cp:
                req.known_counterparties.append(
                    f"{label}: {cp.group(1).strip().rstrip('.,;:')}")
        if req.known_counterparties:
            prov["known_counterparties"] = PROV_HUMAN_INSTRUCTION

        req.provenance = prov
        req.confidence = conf
        req.unresolved_questions = self.open_questions(req)
        req.validate()
        return req

    @staticmethod
    def _trim_client_name(candidate: str) -> str:
        """Keep the proper-noun run, dropping a trailing sentence continuation.

        "Northstar Lending It Is A" would otherwise survive capitalised-run
        matching at a sentence boundary, so the run stops at the first token
        that is a common sentence opener rather than part of a name.
        """
        stop = {"it", "its", "it's", "they", "the", "this", "we", "please",
                "their", "there"}
        tokens = [t for t in str(candidate or "").split() if t]
        kept: List[str] = []
        for t in tokens:
            if t.lower() in stop:
                break
            kept.append(t)
        # Trailing bare words that are not part of the entity ("Lending It").
        while kept and kept[-1].lower() in stop:
            kept.pop()
        return " ".join(kept).strip(" .,;:")

    @staticmethod
    def open_questions(req: ExtractedRequirements) -> List[str]:
        """What the agent must still ask. Deterministic, so the same facts
        always produce the same question list."""
        qs: List[str] = []
        if not req.client_name:
            qs.append("What is the client called?")
        if not req.asset_type:
            qs.append("What asset type is this portfolio?")
        if not req.required_products:
            qs.append("Which Trakt products does the client need?")
        if not req.proposed_portfolio_id and not req.portfolio_name:
            qs.append("What is the portfolio identifier?")
        if not req.reporting_date:
            qs.append("What is the first reporting date?")
        if not req.reporting_frequency:
            qs.append("How often will the client deliver?")
        if not req.expected_artefacts:
            qs.append("Which source files will the client send?")
        if not req.jurisdiction:
            qs.append("Which jurisdiction does the portfolio sit in?")
        if not req.target_go_live_date:
            qs.append("What is the target go-live date?")
        return qs

    # -- 2. a follow-up instruction ------------------------------------- #
    def interpret_action(self, text: str,
                         case: SyntheticCase) -> ProposedChange:
        raw = str(text or "").strip()
        if not raw:
            raise InterpretationError("empty instruction")
        lower = raw.lower()

        # Questions first: a question must never be read as an instruction.
        if self._is_question(lower):
            change = ProposedChange(
                action=_states.ACTION_ASK,
                payload={"question": raw[:2000]},
                summary="Answer a question about this case.",
                basis="the message is phrased as a question")
            change.validate()
            return change

        # An explicit mapping instruction: "map X to Y".
        mapping = self._mapping_change(raw)
        if mapping is not None:
            mapping.validate()
            return mapping

        # Returning to an earlier stage.
        back = self._return_change(lower)
        if back is not None:
            back.validate()
            return back

        # Approvals and confirmations, most specific first.
        for pattern, action, summary, material in _APPROVAL_RULES:
            if re.search(pattern, lower):
                change = ProposedChange(
                    action=action, summary=summary, material=material,
                    requires_confirmation=material,
                    basis="explicit instruction")
                change.validate()
                return change

        # A bare "confirm" / "yes" resolves whatever the case is waiting on.
        if re.fullmatch(r"(yes|confirm(ed)?|approved?|go ahead|proceed)\.?",
                        lower):
            action = _pending_confirmation_action(case)
            if action is None:
                raise InterpretationError("nothing is awaiting confirmation")
            change = ProposedChange(
                action=action,
                summary=f"Confirm what this case is waiting on "
                        f"({_states.spec_label(case.state)}).",
                basis="bare confirmation resolved against the current state")
            change.validate()
            return change

        # A correction to the facts: "the reporting date is 2026-06-30".
        fact = self._fact_change(raw, case)
        if fact is not None:
            fact.validate()
            return fact

        # A configuration value the client has supplied.
        config = self._config_change(raw)
        if config is not None:
            config.validate()
            return config

        raise InterpretationError()

    @staticmethod
    def _config_change(raw: str) -> Optional[ProposedChange]:
        """A configuration answer, e.g. "set the originator's LEI to …".

        The key vocabulary is bounded — it is the set of configuration keys the
        platform already asks about — so this cannot set an arbitrary key.
        """
        from .configuration import configuration_key_vocabulary

        vocab = configuration_key_vocabulary()
        m = re.search(
            r"\b(?:set|the)\s+(.+?)\s+(?:to|is|=)\s+(.+?)(?:[.;]|$)", raw, re.I)
        if not m:
            return None
        label, value = m.group(1).strip(), m.group(2).strip().strip("'\"")
        key = vocab.match(label)
        if not key or not value:
            return None
        return ProposedChange(
            action=_states.ACTION_CORRECT_CONFIG,
            payload={"updates": {key: value}},
            summary=f"Set {label.lower()} to '{value}'.",
            material=True, requires_confirmation=True,
            basis="a configuration value was supplied directly")

    @staticmethod
    def _is_question(lower: str) -> bool:
        if lower.rstrip().endswith("?"):
            return True
        return bool(re.match(r"^(what|why|which|when|who|how|is|are|does|do|"
                             r"can|could|should)\b", lower))

    def _mapping_change(self, raw: str) -> Optional[ProposedChange]:
        m = re.search(r"\bmap\s+(.+?)\s+to\s+(.+?)(?:[.;]|$)", raw, re.I)
        if not m:
            # "Confirm Current Balance." inside a mapping decision is handled by
            # the decision-resolution path, not here.
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

    @staticmethod
    def _return_change(lower: str) -> Optional[ProposedChange]:
        if not re.search(r"\b(go back|return|reopen|re-open|back to)\b", lower):
            return None
        for state in _states.RETURNABLE_STATES:
            words = state.lower().replace("_", " ")
            head = " ".join(words.split()[:2])
            if words in lower or head in lower:
                return ProposedChange(
                    action=_states.ACTION_RETURN_TO_STAGE,
                    payload={"target_state": state},
                    summary=f"Return this case to {_states.spec_label(state)}.",
                    material=True, requires_confirmation=True,
                    basis="explicit request to return to an earlier stage")
        # Named the intent but not a recognised stage.
        raise InterpretationError("which stage should the case return to?")

    def _fact_change(self, raw: str,
                     case: SyntheticCase) -> Optional[ProposedChange]:
        """A correction to one confirmed/extracted fact."""
        updates: Dict[str, Any] = {}
        m = _FIRST_REPORTING_RE.search(raw) or re.search(
            r"\breporting date\b[^.]{0,30}?(\d{4}-\d{2}-\d{2})", raw, re.I)
        if m:
            updates["reporting_date"] = m.group(1)
        m = _GO_LIVE_RE.search(raw)
        if m:
            updates["target_go_live_date"] = m.group(1)
        m = _PORTFOLIO_ID_RE.search(raw)
        if m:
            updates["proposed_portfolio_id"] = m.group(1)
        freq = frequency_vocabulary().match(raw.lower())
        if freq and re.search(r"\b(frequency|deliver|delivery|every)\b", raw,
                              re.I):
            updates["reporting_frequency"] = freq
        products = product_vocabulary().match_all(raw.lower())
        if products and re.search(r"\b(product|also need|add|require)\b", raw,
                                  re.I):
            updates["required_products"] = sorted(
                set(case.extracted.required_products) | set(products))
        artefacts = artefact_vocabulary().match_all(raw.lower())
        if artefacts and re.search(r"\b(send|provide|file|files|tape|artefact|"
                                   r"artifact|expect)\b", raw, re.I):
            updates["expected_artefacts"] = sorted(
                set(case.extracted.expected_artefacts) | set(artefacts))
        asset, asset_conf = asset_vocabulary().match(raw.lower())
        if asset and re.search(r"\b(asset|portfolio is|it is a|type)\b", raw,
                               re.I):
            updates["asset_type"] = asset
        if not updates:
            return None
        return ProposedChange(
            action=_states.ACTION_CORRECT_REQUIREMENTS,
            payload={"updates": updates},
            summary="Update " + ", ".join(
                k.replace("_", " ") for k in sorted(updates)) + ".",
            material=True, requires_confirmation=True,
            basis="a fact was stated directly")


#: (pattern, action, operator summary, material). Ordered — first match wins.
_APPROVAL_RULES: Tuple[Tuple[str, str, str, bool], ...] = (
    (r"\bcancel (this )?case\b", _states.ACTION_CANCEL,
     "Cancel this synthetic case.", True),
    (r"\bapprove\b.*\breadiness\b|\bconfirm\b.*\bready for execution\b"
     r"|\bapprove (the )?execution\b",
     _states.ACTION_APPROVE_EXECUTION,
     "Approve readiness for execution.", True),
    (r"\b(approve|accept)\b.*\b(configuration|config)\b",
     _states.ACTION_APPROVE_CONFIG, "Approve the proposed configuration.",
     True),
    (r"\b(approve|accept)\b.*\b(pack|questionnaire|email)\b",
     _states.ACTION_APPROVE_PACK, "Approve the onboarding pack.", True),
    (r"\b(issue|send)\b.*\bpack\b", _states.ACTION_ISSUE_PACK,
     "Record the onboarding pack as synthetically issued.", True),
    (r"\b(generate|draft|regenerate|rebuild)\b.*\b(pack|questionnaire|email)\b",
     _states.ACTION_GENERATE_PACK, "Generate the onboarding pack.", False),
    (r"\b(generate|draft|regenerate)\b.*\b(config|configuration)\b",
     _states.ACTION_DRAFT_CONFIG, "Draft the client configuration.", False),
    (r"\b(generate|prepare)\b.*\b(plan|orchestration)\b",
     _states.ACTION_GENERATE_PLAN, "Generate the orchestration plan.", False),
    (r"\b(run|start|re-?run)\b.*\b(onboarding|synthetic run|controls)\b",
     _states.ACTION_RUN_ONBOARDING, "Run the synthetic onboarding.", False),
    (r"\bclassify\b.*\b(artefact|artifact|file)", _states.ACTION_CLASSIFY_ARTEFACTS,
     "Classify the artefacts received.", False),
    (r"\backnowledge\b", _states.ACTION_ACKNOWLEDGE_EXCEPTION,
     "Acknowledge the non-blocking exception.", False),
    (r"\bconfirm\b.*\b(interpretation|requirements|facts|understanding)\b",
     _states.ACTION_CONFIRM_REQUIREMENTS,
     "Confirm the proposed interpretation.", True),
)


def _pending_confirmation_action(case: SyntheticCase) -> Optional[str]:
    """What a bare 'yes' means, given where the case is."""
    return {
        _states.REQUIREMENTS_EXTRACTED: _states.ACTION_CONFIRM_REQUIREMENTS,
        _states.REQUIREMENTS_CONFIRMATION_REQUIRED:
            _states.ACTION_CONFIRM_REQUIREMENTS,
        _states.ONBOARDING_PACK_GENERATED: _states.ACTION_APPROVE_PACK,
        _states.ONBOARDING_PACK_APPROVAL_REQUIRED: _states.ACTION_APPROVE_PACK,
        _states.CONFIG_DRAFTED: _states.ACTION_APPROVE_CONFIG,
        _states.CONFIG_APPROVAL_REQUIRED: _states.ACTION_APPROVE_CONFIG,
        _states.ORCHESTRATION_PLAN_GENERATED: _states.ACTION_APPROVE_EXECUTION,
        _states.EXECUTION_APPROVAL_REQUIRED: _states.ACTION_APPROVE_EXECUTION,
    }.get(case.state)
