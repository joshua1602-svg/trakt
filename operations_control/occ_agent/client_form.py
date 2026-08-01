"""operations_control.occ_agent.client_form — the structured client response.

The authoritative interaction model, end to end::

    OCC Agent conversation
      → generated onboarding request
        → secure structured client form          ← this module defines it
          → deterministic persistence into the governed onboarding case
            → OCC Agent follow-up and exception handling
              → human review and approval

**There is no secure external client portal today.** Nothing in Trakt serves a
page to a client, authenticates one, or accepts a submission from outside the
operator's network. This module is the domain and the API contract such a portal
would use, and the OCC's own operator-facing form renders it in the meantime.
Building the portal is an infrastructure and identity task; it is named in the
package README under production enablement and is deliberately not attempted
here. What this module does guarantee is that when the portal arrives it has
nothing to invent: the questions, the keys, the validation and the persistence
already exist and are already governed.

Two properties make the model trustworthy.

**Every answer maps to an authoritative catalogue key.** A response is
``{"contacts.reporting_contact_email": "…"}`` or, for a repeatable section,
``{"portfolios[0].portfolio_type": "acquired"}``. A key the catalogue does not
declare is **rejected**, not stored. There is no free-text lane into the case.

**Nothing an LLM produced is persisted from a structured control.** The agent
generates the form, pre-populates it, explains a question, notices a
contradiction and drafts a follow-up. It does not decide what a structured
answer meant: :func:`apply_response` writes through
``OnboardingService.save_step``, and the value written is the value submitted.

Only :data:`~operations_control.occ_agent.classification.CLIENT` fields reach a
client. Everything else is pre-populated, derived, deferred to the first
delivery, or internal — see :mod:`operations_control.occ_agent.classification`.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..contracts import canonical_json, stable_hash
from ..engine import OpsError
from ..onboarding.case import OnboardingCase
from ..onboarding.catalogue import Catalogue, Field, Section, catalogue
from . import classification as _classification

#: ``section.field`` or ``section[index].field``. The only two shapes a
#: structured answer key may take.
ANSWER_KEY = re.compile(r"^(?P<section>[a-z_]+)"
                        r"(?:\[(?P<index>\d+)\])?"
                        r"\.(?P<field>[a-z0-9_]+)$")

#: The client-facing steps, in the order a client meets them, and which
#: catalogue sections each draws from. Grouping is for a *reader*: the catalogue
#: is organised for generation, and "Report presentation" is not how anyone
#: thinks about being asked for their logo.
STEPS: Tuple[Dict[str, Any], ...] = (
    {"key": "about_you", "label": "About your business",
     "help": "The legal entity behind the portfolio, and who we should talk to.",
     "sections": ("client", "entities", "contacts")},
    {"key": "portfolios", "label": "Your portfolios",
     "help": "One set of answers per book. Everything above is asked once.",
     "sections": ("portfolios",)},
    {"key": "deliveries", "label": "Sending us your data",
     "help": "How each delivery reaches Trakt.",
     "sections": ("sources",)},
    {"key": "reporting", "label": "Your reports",
     "help": "What you receive, and how it should look.",
     "sections": ("reporting", "regime", "presentation")},
    {"key": "covenants", "label": "Your concentration tests",
     "help": "The portfolio-level tests and covenants your funded book is "
             "monitored against. Share whatever you have — a facility "
             "schedule, covenant workbook or a completed limits table.",
     "sections": ("risk_limits",)},
    {"key": "access", "label": "Who needs access",
     "help": "People at your end who need Trakt, or who receive reports.",
     "sections": ("access",)},
    {"key": "meaning", "label": "What your numbers mean",
     "help": "The conventions behind your figures. Trakt works out the FORMAT "
             "of a file itself; what it cannot work out is what you mean by a "
             "balance, a redemption or a valuation.",
     "sections": ("data_semantics",)},
    {"key": "data", "label": "How to read each file",
     "help": "File-specific detail, asked once a file exists. Most of it Trakt "
             "reads for itself.",
     "sections": ("data_definitions",)},
)

_STEP_FOR_SECTION = {s: step["key"] for step in STEPS
                     for s in step["sections"]}


class UnknownAnswerKey(OpsError):
    """A submitted key is not a field the governed catalogue declares."""

    def __init__(self, keys: List[str]):
        self.keys = list(keys)
        super().__init__(
            "OCC_AGENT_UNKNOWN_ANSWER_KEY",
            "These answers do not correspond to anything Trakt asks: "
            + ", ".join(sorted(keys)[:5])
            + ". Nothing was saved.", http_status=400)


class NotAClientQuestion(OpsError):
    """A submitted key is a real field, but not one a client is asked."""

    def __init__(self, keys: List[str]):
        self.keys = list(keys)
        super().__init__(
            "OCC_AGENT_NOT_A_CLIENT_QUESTION",
            "These are not questions Trakt puts to a client: "
            + ", ".join(sorted(keys)[:5])
            + ". Nothing was saved.", http_status=400)


# --------------------------------------------------------------------------- #
# The form
# --------------------------------------------------------------------------- #

@dataclass
class FormField:
    """One question, and the authoritative key its answer is filed under."""

    #: ``section.field`` or ``section[index].field``. The contract.
    key: str
    section: str
    field: str
    label: str
    help: str = ""
    type: str = "text"
    options: List[Dict[str, str]] = field(default_factory=list)
    required: bool = False
    sensitive: bool = False
    evidence_required: bool = False
    max_length: Optional[int] = None
    validation: str = ""
    #: Pre-populated from what Trakt already knows. A client edits or confirms.
    value: Any = None
    #: Which repeatable item this belongs to, and what to call it on screen.
    index: Optional[int] = None
    item: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FormGroup:
    """One catalogue section's questions, within a step."""

    key: str
    label: str
    help: str = ""
    repeatable: bool = False
    index: Optional[int] = None
    item: str = ""
    fields: List[FormField] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {**{k: v for k, v in asdict(self).items() if k != "fields"},
                "fields": [f.to_dict() for f in self.fields],
                "required": self.required_count}

    @property
    def required_count(self) -> int:
        return len([f for f in self.fields if f.required])


@dataclass
class FormStep:
    """One client-facing page."""

    key: str
    label: str
    help: str = ""
    groups: List[FormGroup] = field(default_factory=list)
    #: What has to be true before this step is worth showing. Empty means now.
    unlocked_by: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "label": self.label, "help": self.help,
                "unlocked_by": self.unlocked_by,
                "groups": [g.to_dict() for g in self.groups],
                "questions": self.question_count,
                "required": self.required_count}

    @property
    def question_count(self) -> int:
        return sum(len(g.fields) for g in self.groups)

    @property
    def required_count(self) -> int:
        return sum(g.required_count for g in self.groups)


@dataclass
class ClientForm:
    """Everything, and only what, the client is asked."""

    case_ref: str
    client_name: str = ""
    steps: List[FormStep] = field(default_factory=list)
    #: Steps that exist but are not open yet, and what would open them.
    locked: List[Dict[str, str]] = field(default_factory=list)
    content_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_ref": self.case_ref,
            "client_name": self.client_name,
            "steps": [s.to_dict() for s in self.steps],
            "locked": list(self.locked),
            "questions": self.question_count,
            "required": self.required_count,
            "content_hash": self.content_hash,
        }

    @property
    def question_count(self) -> int:
        return sum(s.question_count for s in self.steps)

    @property
    def required_count(self) -> int:
        return sum(s.required_count for s in self.steps)

    def keys(self) -> List[str]:
        return [f.key for s in self.steps for g in s.groups for f in g.fields]

    def field(self, key: str) -> Optional[FormField]:
        return next((f for s in self.steps for g in s.groups
                     for f in g.fields if f.key == key), None)


# --------------------------------------------------------------------------- #
# Building it
# --------------------------------------------------------------------------- #

def build(case: OnboardingCase, *, cat: Optional[Catalogue] = None
          ) -> ClientForm:
    """The form this client should see now.

    Conditional on everything the catalogue can condition on — asset class,
    selected products, how the book is held, how it is delivered — and
    progressive: a step whose trigger has not happened is reported as locked
    rather than shown empty.
    """
    cat = cat or catalogue()
    answers = case.answers
    client = answers.get("client") or {}
    form = ClientForm(case_ref=case.case_id,
                      client_name=str(client.get("client_name") or ""))

    rows = _classification.classify_all(answers, cat=cat)
    asked = [r for r in rows if r.client_facing]

    for spec in STEPS:
        step = FormStep(key=spec["key"], label=spec["label"],
                        help=spec["help"])
        for section_key in spec["sections"]:
            section = cat.section(section_key)
            if section is None:
                continue
            step.groups.extend(_groups(section, asked, cat))
        if step.question_count:
            form.steps.append(step)

    # A deferred section is not absent, it is not open yet. Say which, and why,
    # so an operator can see the whole shape of what will be asked. Which
    # sections those are is the catalogue's own declaration.
    for section_key, trigger in _classification.deferred_sections(cat).items():
        section = cat.section(section_key)
        if section is None:
            continue
        form.locked.append({
            "step": _STEP_FOR_SECTION.get(section_key, section_key),
            "label": section.label,
            "unlocked_by": f"Asked once {trigger}."})

    form.content_hash = stable_hash(canonical_json(
        {"steps": [s.to_dict() for s in form.steps]}))
    return form


def _groups(section: Section, asked: List[_classification.Classification],
            cat: Catalogue) -> List[FormGroup]:
    rows = [r for r in asked if r.section == section.key]
    if not rows:
        return []
    if not section.repeatable:
        return [FormGroup(key=section.key, label=section.label,
                          help=section.help,
                          fields=[_field(section, r, cat) for r in rows])]
    # One group per item, so a client with three books answers three short
    # groups rather than one long one — and never re-answers a client-level
    # question because of it.
    out: List[FormGroup] = []
    for index in sorted({r.index for r in rows if r.index is not None}):
        item_rows = [r for r in rows if r.index == index]
        label = next((r.item for r in item_rows if r.item), "")
        out.append(FormGroup(
            key=section.key, label=section.label, help=section.help,
            repeatable=True, index=index, item=label,
            fields=[_field(section, r, cat) for r in item_rows]))
    return out


def _field(section: Section, row: _classification.Classification,
           cat: Catalogue) -> FormField:
    f = cat.field(section.key, row.field)
    assert f is not None                      # rows are built from the catalogue
    return FormField(
        key=answer_key(section.key, row.field, row.index),
        section=section.key, field=row.field, label=f.label, help=f.help,
        type=f.type, options=list(f.options or []), required=row.required,
        sensitive=f.sensitive, evidence_required=f.evidence_required,
        max_length=f.max_length, validation=f.validation,
        value=row.value, index=row.index, item=row.item)


def answer_key(section: str, field_key: str,
               index: Optional[int] = None) -> str:
    """The authoritative key one answer is filed under."""
    return (f"{section}[{index}].{field_key}" if index is not None
            else f"{section}.{field_key}")


def parse_key(key: str) -> Tuple[str, Optional[int], str]:
    m = ANSWER_KEY.match(str(key or "").strip())
    if not m:
        raise UnknownAnswerKey([str(key)])
    index = m.group("index")
    return m.group("section"), (int(index) if index is not None else None), \
        m.group("field")


# --------------------------------------------------------------------------- #
# Persisting a response
# --------------------------------------------------------------------------- #

@dataclass
class ResponsePlan:
    """What a submitted response would write. Deterministic; no interpretation.

    ``steps`` is exactly what goes to ``OnboardingService.save_step``. Building
    it is a pure function of the submitted keys, the case's current items and
    the catalogue — an LLM has no part in it, and a test asserts the module
    imports nothing that could give it one.
    """

    steps: Dict[str, Any] = field(default_factory=dict)
    accepted: Dict[str, Any] = field(default_factory=dict)
    #: Answers to questions the client was not asked. Reported, never written.
    ignored: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"steps": self.steps, "accepted": self.accepted,
                "ignored": list(self.ignored), "count": len(self.accepted)}


def plan_response(case: OnboardingCase, response: Dict[str, Any], *,
                  cat: Optional[Catalogue] = None,
                  form: Optional[ClientForm] = None,
                  strict: bool = True) -> ResponsePlan:
    """Turn ``{catalogue key: value}`` into ``save_step`` payloads.

    Every key is checked against the catalogue and against the form the client
    was actually shown. An unknown key is an error rather than a silently
    dropped answer, because a client who typed something into a box is entitled
    to know it did not land.
    """
    cat = cat or catalogue()
    form = form or build(case, cat=cat)
    plan = ResponsePlan()

    unknown: List[str] = []
    not_asked: List[str] = []
    for key in response:
        try:
            section_key, index, field_key = parse_key(key)
        except UnknownAnswerKey:
            unknown.append(str(key))
            continue
        section = cat.section(section_key)
        if section is None or section.field(field_key) is None:
            unknown.append(str(key))
            continue
        if form.field(key) is None:
            not_asked.append(str(key))
    if unknown:
        raise UnknownAnswerKey(unknown)
    if not_asked:
        if strict:
            raise NotAClientQuestion(not_asked)
        plan.ignored = sorted(not_asked)

    for key, value in response.items():
        if key in plan.ignored:
            continue
        section_key, index, field_key = parse_key(key)
        section = cat.section(section_key)
        assert section is not None
        plan.accepted[key] = value
        if index is None:
            plan.steps.setdefault(section_key, {})[field_key] = value
            continue
        rows = plan.steps.get(section_key)
        if not isinstance(rows, dict) or section_key not in rows:
            existing = [dict(item) for item in case.items(section_key)]
            while len(existing) <= index:
                existing.append({})
            plan.steps[section_key] = {section_key: existing}
        plan.steps[section_key][section_key][index][field_key] = value
    return plan


def validate_response(case: OnboardingCase, plan: ResponsePlan, *,
                      cat: Optional[Catalogue] = None) -> List[Dict[str, str]]:
    """Type and option checks, from each field's own declaration.

    Runs before anything is written, so a client sees every problem at once
    rather than one per submission. Client Onboarding validates again on save —
    this does not replace that, it just fails earlier and more legibly.
    """
    cat = cat or catalogue()
    problems: List[Dict[str, str]] = []
    for key, value in plan.accepted.items():
        section_key, _index, field_key = parse_key(key)
        f = cat.field(section_key, field_key)
        if f is None:                         # pragma: no cover — planned above
            continue
        message = _problem(f, value)
        if message:
            problems.append({"key": key, "label": f.label,
                             "message": message})
    return problems


def _problem(f: Field, value: Any) -> str:
    if value in (None, ""):
        return ""
    if f.options and f.type in ("enum", "multi_enum"):
        allowed = {str(o.get("value")) for o in f.options}
        given = value if isinstance(value, list) else [value]
        bad = [str(v) for v in given if str(v) not in allowed]
        if bad:
            return f"{', '.join(bad)} is not one of the options."
    if f.max_length and isinstance(value, str) and len(value) > f.max_length:
        return f"Longer than the {f.max_length} characters allowed."
    return ""
