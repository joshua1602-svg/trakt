"""operations_control.occ_agent.interpretation — natural language in, typed answers out.

Two jobs, both of which end in a structure the Client Onboarding catalogue
itself validates:

1. :meth:`Interpreter.interpret_instruction` — an initiating instruction becomes
   an :class:`Interpretation`: one ``save_step`` payload per catalogue step, plus
   the reporting period, which is a property of a *delivery* rather than of a
   client and so belongs to the run.
2. :meth:`Interpreter.interpret_action` — a follow-up sentence becomes a
   :class:`ProposedChange`: one of the lifecycle's named actions plus a typed
   payload.

Four rules hold whatever produces the structure:

* **The interpreter never decides state.** It proposes an action; the service
  runs the deterministic controls and asks :mod:`.states` — or Client
  Onboarding's own transition table — whether the resulting move is legal.
* **The interpreter never writes.** It returns values; only the service
  persists, and only through ``OnboardingService`` or the run store.
* **Every field it produces must exist in the catalogue.** Validation walks the
  proposed answers against ``config/onboarding/field_catalogue.yaml``, so a
  model swapped in behind :class:`Interpreter` cannot invent a field, and cannot
  reach a section the wizard does not have.
* **It answers only what the sentence says.** Everything Trakt works out for
  itself — the client identifier, the portfolio identifier, the reporting
  currency, the asset class implied by a sample pack — is left to
  :mod:`operations_control.onboarding.inference`, which already does it, records
  its provenance, and lets an operator override it.

The default :class:`DeterministicInterpreter` is rule-based and offline: the
feature must work with no model credentials, and a deterministic interpreter is
what makes the practice fixtures reproducible. It is injected, so a model-backed
interpreter can be substituted without touching the service — and its output
still passes through the same validation and the same controls.

Vocabulary is read from the configuration the platform already uses — the
catalogue's own option lists, the regime product declaration, the asset model
and the product profiles' match signals — rather than hard-coded, so a new
asset, product or cadence becomes recognisable by configuration change. Only the
*phrasing* an operator types lives in this module.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import yaml

from ..engine import OpsError
from ..onboarding.case import OnboardingCase
from ..onboarding.catalogue import Catalogue, catalogue
from ..onboarding.service import STEPS
from . import states as _states
from .input_roles import artefact_vocabulary, contains
from .run import SyntheticRun

REPO = Path(__file__).resolve().parents[2]
PRODUCT_PROFILES_PATH = REPO / "config/asset/product_profiles.yaml"

#: Where an answer came from. Recorded alongside the value, in the same shape
#: Client Onboarding's own provenance uses.
PROV_HUMAN_INSTRUCTION = "human_instruction"
PROV_AGENT_INFERENCE = "agent_inference"


class InterpretationError(OpsError):
    """The instruction could not be turned into anything actionable."""

    def __init__(self, detail: str = ""):
        super().__init__(
            "OCC_AGENT_NOT_UNDERSTOOD",
            "Trakt could not tell what to do with that. Try naming the change "
            "directly — for example 'the jurisdiction is the Netherlands', "
            "'map Current Balance to current outstanding balance', or 'what is "
            "still needed?'." + (f" ({detail})" if detail else ""),
            http_status=422)


# --------------------------------------------------------------------------- #
# What an instruction produces
# --------------------------------------------------------------------------- #

#: Bounds on anything the interpreter proposes, so a model cannot post a payload
#: large enough to matter.
MAX_VALUE_CHARS = 2000
MAX_ITEMS = 50


@dataclass
class Interpretation:
    """Catalogue-shaped answers read out of one instruction.

    ``steps`` is ``{step_key: save_step payload}`` — exactly what
    ``OnboardingService.save_step`` takes, so the answers reach the case through
    the platform's own writer, with its own validation, inference and event
    history.
    """

    steps: Dict[str, Any] = field(default_factory=dict)
    #: Delivery-specific, so not a catalogue answer: it lives on the run.
    reporting_period: str = ""
    #: The expected delivery cadence. A catalogue answer, but on the *sources*
    #: section, which Client Onboarding derives from the portfolios — so it is
    #: applied after that derivation rather than sent as a step payload.
    cadence: str = ""
    #: field path -> PROV_*.
    provenance: Dict[str, str] = field(default_factory=dict)
    #: field path -> 0..1, for anything read with less than full confidence.
    confidence: Dict[str, float] = field(default_factory=dict)
    #: Semantic input roles the instruction said the client would send.
    expected_artefacts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def empty(self) -> bool:
        return not (self.steps or self.reporting_period or self.cadence
                    or self.expected_artefacts)

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
        if self.reporting_period and not re.fullmatch(
                r"\d{4}-\d{2}-\d{2}", self.reporting_period):
            raise InterpretationError("reporting period")
        for value in self.confidence.values():
            if not 0.0 <= float(value) <= 1.0:
                raise InterpretationError("confidence must be 0..1")

    def summary(self, cat: Optional[Catalogue] = None) -> List[str]:
        """What the interpretation says, in the catalogue's own labels."""
        cat = cat or catalogue()
        lines: List[str] = []
        for step, payload in (self.steps or {}).items():
            section = cat.section(step)
            if section is None:      # pragma: no cover — validate() refuses it
                continue
            holders = ([payload.get(section.key) or payload.get("items") or []]
                       if section.repeatable else [[payload]])
            for group in holders:
                for item in group:
                    for key, value in (item or {}).items():
                        f = section.field(key)
                        label = f.label if f else key.replace("_", " ")
                        shown = (", ".join(str(v) for v in value)
                                 if isinstance(value, list) else str(value))
                        lines.append(f"{label}: {shown}")
        if self.reporting_period:
            lines.append(f"Reporting period: {self.reporting_period}")
        if self.cadence:
            lines.append(f"Expected cadence: {self.cadence}")
        if self.expected_artefacts:
            vocab = artefact_vocabulary()
            lines.append("Expected files: " + ", ".join(
                vocab.label(r) for r in self.expected_artefacts))
        return lines


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
            if isinstance(value, str) and len(value) > MAX_VALUE_CHARS:
                raise InterpretationError(f"payload value for {key} is too long")


_ALL_ACTIONS = frozenset(
    _states.ONBOARDING_ACTIONS + (
        _states.ACTION_REGISTER_ARTEFACT, _states.ACTION_RUN_ONBOARDING,
        _states.ACTION_RESOLVE_DECISION, _states.ACTION_ACKNOWLEDGE_EXCEPTION,
        _states.ACTION_GENERATE_PLAN, _states.ACTION_APPROVE_EXECUTION,
        _states.ACTION_CANCEL, _states.ACTION_ASK))


class Interpreter(Protocol):
    """The seam a model-backed interpreter would implement."""

    def interpret_instruction(self, text: str) -> Interpretation: ...

    def interpret_action(self, text: str, run: SyntheticRun,
                         case: OnboardingCase) -> ProposedChange: ...


# --------------------------------------------------------------------------- #
# Recognition vocabularies, read from configuration
# --------------------------------------------------------------------------- #

def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


@lru_cache(maxsize=4)
def _asset_tokens(profiles_path: str) -> Tuple[Tuple[str, str, float], ...]:
    """``(token, asset_class, confidence)``, longest token first.

    The asset classes are the catalogue's own option list (which is itself the
    platform's asset model); the recognition synonyms are the ``match`` blocks
    of ``config/asset/product_profiles.yaml`` — the same signals the Onboarding
    Agent recognises a pack by.
    """
    cat = catalogue()
    f = cat.field("portfolios", "asset_class")
    classes = {str(o.get("value")): str(o.get("label") or o.get("value"))
               for o in ((f.options if f else None) or [])}
    tokens: List[Tuple[str, str, float]] = []
    for key, label in classes.items():
        tokens.append((key.replace("_", " "), key, 0.95))
        tokens.append((label.lower(), key, 0.95))

    for _pid, profile in (_load_yaml(Path(profiles_path)).get("profiles")
                          or {}).items():
        match = (profile or {}).get("match") or {}
        owner = next((k for k in classes
                      if k in {str(s).lower()
                               for s in (match.get("asset_class") or [])}), "")
        if not owner:
            continue
        for token in match.get("signal_tokens") or []:
            tokens.append((str(token).lower(), owner, 0.95))
        for group, confidence in (("asset_class", 0.95),
                                  ("product_type", 0.7)):
            for token in match.get(group) or []:
                tokens.append((str(token).replace("_", " ").lower(), owner,
                               confidence))
    strongest: Dict[Tuple[str, str], float] = {}
    for token, asset, confidence in tokens:
        if token:
            strongest[(token, asset)] = max(
                strongest.get((token, asset), 0.0), confidence)
    return tuple(sorted(((t, a, c) for (t, a), c in strongest.items()),
                        key=lambda x: (-len(x[0]), x[0])))


def asset_tokens() -> Tuple[Tuple[str, str, float], ...]:
    return _asset_tokens(str(PRODUCT_PROFILES_PATH))


#: Extra phrasing per product, for the words an operator types. The PRODUCTS
#: themselves come from the regime declaration the catalogue reads.
_PRODUCT_PROSE_TOKENS: Dict[str, Tuple[str, ...]] = {
    "mi": ("management information", "portfolio mi", "monthly mi", "mi pack",
           "mi reporting", "portfolio reporting"),
    "esma_annex2": ("annex 2", "annex2", "esma annex 2", "regulatory reporting",
                    "securitisation reporting", "underlying exposures"),
    "investor_reporting": ("investor reporting", "investor report", "annex 12",
                           "annex12", "noteholder reporting"),
    "static_pools": ("static pool", "static pools", "vintage analysis"),
}


def product_tokens(cat: Optional[Catalogue] = None
                   ) -> List[Tuple[str, str]]:
    """``(token, product_id)``, longest first, from the product declaration."""
    cat = cat or catalogue()
    pairs: List[Tuple[str, str]] = []
    for key, spec in (cat.regime_products or {}).items():
        pairs.append((key.replace("_", " "), key))
        label = str((spec or {}).get("label") or "")
        if label:
            pairs.append((label.lower(), key))
        for token in _PRODUCT_PROSE_TOKENS.get(key, ()):
            pairs.append((token, key))
    return sorted(set(pairs), key=lambda p: (-len(p[0]), p[0]))


def cadence_tokens(cat: Optional[Catalogue] = None) -> List[Tuple[str, str]]:
    """``(token, cadence)`` from the catalogue's own cadence vocabulary."""
    cat = cat or catalogue()
    f = cat.field("sources", "cadence")
    values = [str(o.get("value")) for o in ((f.options if f else None) or [])]
    pairs: List[Tuple[str, str]] = []
    for value in values:
        pairs.append((value.replace("_", " "), value))
        adverb = {"monthly": "each month", "weekly": "each week",
                  "daily": "each day"}.get(value)
        if adverb:
            pairs.append((adverb, value))
    return sorted(set(pairs), key=lambda p: (-len(p[0]), p[0]))


def portfolio_type_tokens(cat: Optional[Catalogue] = None
                          ) -> List[Tuple[str, str]]:
    cat = cat or catalogue()
    f = cat.field("portfolios", "portfolio_type")
    pairs = [(str(o.get("value")), str(o.get("value")))
             for o in ((f.options if f else None) or [])]
    pairs += [("bought", "acquired"), ("purchased", "acquired"),
              ("originated", "direct"), ("own origination", "direct")]
    return sorted(set(pairs), key=lambda p: (-len(p[0]), p[0]))


# --------------------------------------------------------------------------- #
# Deterministic interpreter
# --------------------------------------------------------------------------- #

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

_REPORTING_PERIOD_RE = re.compile(
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
    ("united kingdom", "GB"), ("uk", "GB"), ("british", "GB"),
    ("england", "GB"), ("scotland", "GB"), ("wales", "GB"),
    ("ireland", "IE"), ("irish", "IE"),
    ("netherlands", "NL"), ("dutch", "NL"),
    ("spain", "ES"), ("spanish", "ES"),
    ("france", "FR"), ("french", "FR"),
    ("germany", "DE"), ("german", "DE"),
)

#: Entity roles an instruction can name a counterparty in.
_COUNTERPARTY_ROLES = ("originator", "servicer", "trustee", "sponsor",
                       "seller", "original_lender")


def _trim_identifier(value: str) -> str:
    """Drop sentence punctuation an identifier picked up from prose.

    ``.`` and ``-`` are legal INSIDE an identifier, so they cannot simply be
    excluded from the pattern; they are only noise at the very end.
    """
    return str(value or "").strip().rstrip(".,;:-_")


class DeterministicInterpreter:
    """Rule-based interpretation. Offline, reproducible, no credentials.

    It is deliberately conservative: anything it cannot read with confidence is
    left unanswered, because Client Onboarding's own readiness will then ask for
    it — which is a prompt to the human, where a silent inference is not.
    """

    def __init__(self, *, today: Optional[date] = None,
                 cat: Optional[Catalogue] = None):
        # Injected so fixtures and tests are reproducible; ``None`` means the
        # interpreter derives nothing from the current date (it never guesses a
        # reporting period, so this only affects sanity checks).
        self.today = today
        self.catalogue = cat or catalogue()

    # -- 1. the initiating instruction ---------------------------------- #
    def interpret_instruction(self, text: str) -> Interpretation:
        raw = str(text or "").strip()
        if not raw:
            raise InterpretationError("empty instruction")
        lower = raw.lower()
        out = Interpretation()
        client: Dict[str, Any] = {}
        portfolio: Dict[str, Any] = {}

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
            client["client_name"] = name
            out.provenance["client.client_name"] = PROV_HUMAN_INSTRUCTION

        # Identifiers, only when stated. Trakt proposes them otherwise, with a
        # collision check — see operations_control.onboarding.inference.
        cm = _CLIENT_ID_RE.search(raw)
        if cm:
            client["client_id"] = _trim_identifier(cm.group(1))
            out.provenance["client.client_id"] = PROV_HUMAN_INSTRUCTION

        for token, code in _JURISDICTIONS:
            if contains(lower, token):
                client["jurisdiction"] = code
                out.provenance["client.jurisdiction"] = PROV_HUMAN_INSTRUCTION
                break

        # Portfolio.
        pm = _PORTFOLIO_ID_RE.search(raw)
        if pm:
            portfolio["portfolio_id"] = _trim_identifier(pm.group(1))
            out.provenance["portfolios.portfolio_id"] = PROV_HUMAN_INSTRUCTION
        pn = re.search(r"\b(?i:portfolio)\s+(?i:called|named)\s+"
                       rf"({_NAME_RUN})", raw)
        if pn:
            portfolio["display_name"] = self._trim_client_name(pn.group(1))
            out.provenance["portfolios.display_name"] = PROV_HUMAN_INSTRUCTION
        elif name:
            # A book has to be called something for the operator to work with.
            # Trakt's own inference names it from the client when nothing else
            # does, so this only fills the very first draft.
            portfolio["display_name"] = f"{name} portfolio"
            out.provenance["portfolios.display_name"] = PROV_AGENT_INFERENCE
            out.confidence["portfolios.display_name"] = 0.6

        asset, asset_confidence = self.match_asset(lower)
        if asset:
            portfolio["asset_class"] = asset
            out.provenance["portfolios.asset_class"] = (
                PROV_HUMAN_INSTRUCTION if asset_confidence >= 0.9
                else PROV_AGENT_INFERENCE)
            if asset_confidence < 0.9:
                out.confidence["portfolios.asset_class"] = asset_confidence

        for token, value in portfolio_type_tokens(self.catalogue):
            if contains(lower, token):
                portfolio["portfolio_type"] = value
                out.provenance["portfolios.portfolio_type"] = \
                    PROV_HUMAN_INSTRUCTION
                break

        # Products.
        products = self.match_products(lower)
        if products:
            out.steps["reporting"] = {"products": products}
            out.provenance["reporting.products"] = PROV_HUMAN_INSTRUCTION

        # Entities named as counterparties become entity rows, which is where
        # the catalogue holds them — and what Annex 2 is generated from.
        entities = self._entities(raw, name)
        if entities:
            out.steps["entities"] = {"entities": entities}
            out.provenance["entities"] = PROV_HUMAN_INSTRUCTION

        if client:
            out.steps["client"] = client
        if portfolio:
            out.steps["portfolios"] = {"portfolios": [portfolio]}

        # Delivery-specific facts.
        period = _REPORTING_PERIOD_RE.search(raw)
        if period:
            out.reporting_period = period.group(1)
        out.cadence = self.match_cadence(lower)

        roles = artefact_vocabulary().match_all(lower)
        if roles:
            out.expected_artefacts = roles

        out.validate(self.catalogue)
        return out

    def _entities(self, raw: str, client_name: str) -> List[Dict[str, Any]]:
        """Counterparties named with their role, plus the client as originator.

        The catalogue holds legal entities as a repeatable section with a role
        list; a sentence that says "the servicer is X" is answering that
        section, not creating a free-text note.
        """
        rows: List[Dict[str, Any]] = []
        seen: set = set()
        for role in _COUNTERPARTY_ROLES:
            word = role.replace("_", " ")
            m = re.search(rf"\b(?i:{word})\s+(?i:is\s+)?({_NAME_RUN})", raw)
            if not m:
                continue
            legal_name = self._trim_client_name(m.group(1))
            if not legal_name or legal_name.lower() in seen:
                continue
            seen.add(legal_name.lower())
            rows.append({"legal_name": legal_name, "roles": [role]})
        if client_name and client_name.lower() not in seen:
            # The client is the originator unless the instruction said someone
            # else was. Stated rather than assumed: it is shown for confirmation
            # like every other proposed value.
            if not any("originator" in r["roles"] for r in rows):
                rows.insert(0, {"legal_name": client_name,
                                "roles": ["originator"]})
        return rows

    # -- vocabulary matching -------------------------------------------- #
    def match_asset(self, lower_text: str) -> Tuple[str, float]:
        for token, asset, confidence in asset_tokens():
            if contains(lower_text, token):
                return asset, confidence
        return "", 0.0

    def match_products(self, lower_text: str) -> List[str]:
        hits: List[str] = []
        for token, product in product_tokens(self.catalogue):
            if contains(lower_text, token) and product not in hits:
                hits.append(product)
        return hits

    def match_cadence(self, lower_text: str) -> str:
        for token, value in cadence_tokens(self.catalogue):
            if contains(lower_text, token):
                return value
        return ""

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
        while kept and kept[-1].lower() in stop:
            kept.pop()
        return " ".join(kept).strip(" .,;:")

    # -- 2. a follow-up instruction ------------------------------------- #
    def interpret_action(self, text: str, run: SyntheticRun,
                         case: OnboardingCase) -> ProposedChange:
        raw = str(text or "").strip()
        if not raw:
            raise InterpretationError("empty instruction")
        lower = raw.lower()

        # Questions first: a question must never be read as an instruction.
        if self._is_question(lower):
            change = ProposedChange(
                action=_states.ACTION_ASK,
                payload={"question": raw[:MAX_VALUE_CHARS]},
                summary="Answer a question about this case.",
                basis="the message is phrased as a question")
            change.validate()
            return change

        # An explicit mapping instruction: "map X to Y".
        mapping = self._mapping_change(raw)
        if mapping is not None:
            mapping.validate()
            return mapping

        # Named actions, most specific first.
        for pattern, action, summary, material in _ACTION_RULES:
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
            action = pending_confirmation_action(run, case)
            if action is None:
                raise InterpretationError("nothing is awaiting confirmation")
            change = ProposedChange(
                action=action,
                summary="Confirm what this case is waiting on.",
                basis="bare confirmation resolved against the current state")
            change.validate()
            return change

        # Anything else that answers the onboarding.
        answer = self._answer_change(raw)
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

    def _answer_change(self, raw: str) -> Optional[ProposedChange]:
        """A sentence that answers the onboarding.

        Read with the same extractor the initiating instruction uses, so "the
        jurisdiction is the Netherlands" reaches ``save_step`` in exactly the
        shape the wizard would have written.
        """
        try:
            interpretation = self.interpret_instruction(raw)
        except InterpretationError:
            return None
        if not _says_more_than_a_name(interpretation):
            return None
        return ProposedChange(
            action=_states.ACTION_ANSWER,
            payload={"interpretation": interpretation.to_dict()},
            summary="Answer: " + "; ".join(
                interpretation.summary(self.catalogue)[:4]) + ".",
            material=True, requires_confirmation=True,
            basis="a fact was stated directly")


def _says_more_than_a_name(interpretation: Interpretation) -> bool:
    """Whether a follow-up sentence answers anything beyond naming somebody.

    A bare proper-noun run on a follow-up is almost always a false positive
    ("send it to Northstar"), and the client is already named by then. So a
    follow-up counts as an answer only when it says something the case does not
    already have from its opening instruction.
    """
    if interpretation.reporting_period or interpretation.cadence:
        return True
    for step, payload in (interpretation.steps or {}).items():
        if step == "client":
            if set(payload) - {"client_name"}:
                return True
        elif step == "portfolios":
            for item in (payload.get("portfolios") or []):
                if set(item) - {"display_name"}:
                    return True
        else:
            if payload:
                return True
    return False


#: (pattern, action, operator summary, material). Ordered — first match wins.
_ACTION_RULES: Tuple[Tuple[str, str, str, bool], ...] = (
    (r"\bcancel (this )?(case|run)\b", _states.ACTION_CANCEL,
     "Cancel this practice case.", True),
    (r"\bwithdraw\b", _states.ACTION_WITHDRAW,
     "Withdraw the onboarding.", True),
    (r"\bapprove\b.*\breadiness\b|\bconfirm\b.*\bready for execution\b"
     r"|\bapprove (the )?execution\b",
     _states.ACTION_APPROVE_EXECUTION,
     "Approve readiness for execution.", True),
    (r"\b(approve|accept)\b.*\bonboarding\b|\bapprove (the )?case\b",
     _states.ACTION_APPROVE_ONBOARDING,
     "Approve the onboarding.", True),
    (r"\b(submit|send)\b.*\b(for )?approval\b",
     _states.ACTION_SUBMIT_FOR_APPROVAL,
     "Submit the onboarding for approval.", True),
    (r"\brequest changes?\b|\bsend (it )?back\b",
     _states.ACTION_REQUEST_CHANGES, "Send the onboarding back for changes.",
     True),
    (r"\b(ask|request)\b.*\b(client|information|checklist)\b",
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

    The onboarding half is asked first, because until the case is approved the
    execution half has nothing to confirm.
    """
    from ..onboarding.case import (
        APPROVED, DRAFT, IN_REVIEW, CHANGES_REQUIRED, READY_FOR_APPROVAL,
    )
    if case.status in (DRAFT, IN_REVIEW, CHANGES_REQUIRED):
        return _states.ACTION_SUBMIT_FOR_APPROVAL
    if case.status == READY_FOR_APPROVAL:
        return _states.ACTION_APPROVE_ONBOARDING
    if case.status != APPROVED:
        return None
    return {
        _states.READY_TO_RUN: _states.ACTION_RUN_ONBOARDING,
        _states.SYNTHETIC_ONBOARDING_PASSED: _states.ACTION_GENERATE_PLAN,
        _states.ORCHESTRATION_PLAN_GENERATED: _states.ACTION_APPROVE_EXECUTION,
        _states.EXECUTION_APPROVAL_REQUIRED: _states.ACTION_APPROVE_EXECUTION,
    }.get(run.state)


def reset_cache() -> None:
    """Test seam: clear the configuration-derived token caches."""
    _asset_tokens.cache_clear()
