"""operations_control.occ_agent.pack — the client-facing onboarding pack.

A **projection** of the governed catalogue and the case's current answers.
Nothing here decides what to ask: every question is a field
``config/onboarding/field_catalogue.yaml`` declares, with the help text the
catalogue gives it. A field added there appears in the pack; a field removed
disappears from it. There is no second questionnaire.

Coverage is not the same as a good client experience, though, and this pack used
to prove the first while failing the second: projecting every collected field
produced 58 questions for a straightforward equity-release onboarding, 20 of
which the client could not answer at all. So the pack is now built on two
things:

* :mod:`operations_control.occ_agent.classification` — which of the five
  categories each field is in. Only ``CLIENT`` becomes a question; the rest are
  pre-populated, derived, deferred to the first delivery, or internal.
* :mod:`operations_control.occ_agent.client_form` — the progressive, conditional
  structured form those questions are asked through, keyed by authoritative
  catalogue keys.

What the pack adds around the form is everything else a client needs: what is
already known and worth confirming, the artefact request derived from the
governed input requirements, delivery instructions built with the production
path rules, and a covering email.

One thing the pack states outright, because it is a governed decision rather
than an omission: **field mappings are not asked for.** They are learned from
the first representative delivery and approved through the existing mapping
path. The catalogue records that decision in its own ``not_collected`` list
(``file_role_schemas``: "Learned at mapping approval from a representative
pack"), and :data:`MAPPING_STATEMENT` says so to the client.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from ..contracts import canonical_json, stable_hash
from ..onboarding.case import OnboardingCase
from ..onboarding.catalogue import Catalogue, catalogue
from ..onboarding.derivation import blob_prefix
from . import classification as _classification
from . import client_form as _client_form
from .input_roles import artefact_vocabulary

#: What the pack tells a client about mappings. A governed decision, stated —
#: never a silent omission.
MAPPING_STATEMENT = (
    "Trakt does not ask you to map your fields to ours. Send a representative "
    "file and Trakt will propose the mapping itself; an operator reviews and "
    "approves it during the first ingestion, and it is then fixed for every "
    "later delivery. What Trakt cannot work out on its own is what your "
    "numbers MEAN — Trakt asks about that once it has seen your files, not "
    "before."
)

#: Where an answer stands, from the client's point of view.
ANSWERED = "answered"
OUTSTANDING = "outstanding"
DERIVED = "derived"


@dataclass
class PackQuestion:
    """One catalogue field, as the client sees it."""

    section: str
    field: str
    label: str
    help: str = ""
    status: str = OUTSTANDING
    value: Any = None
    provenance: str = ""
    index: Optional[int] = None
    item: str = ""
    required: bool = False
    evidence_required: bool = False
    sensitive: bool = False
    #: Where the answer ends up, from the field's own declaration.
    writes_to: str = ""
    #: Which client-facing step this question is asked on.
    step: str = ""
    step_label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PackSection:
    key: str
    label: str
    help: str = ""
    repeatable: bool = False
    questions: List[PackQuestion] = field(default_factory=list)
    step: str = ""
    step_label: str = ""
    #: For a repeatable section: which item, and what it is called.
    index: Optional[int] = None
    item: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "label": self.label, "help": self.help,
                "repeatable": self.repeatable, "step": self.step,
                "step_label": self.step_label, "index": self.index,
                "item": self.item,
                "questions": [q.to_dict() for q in self.questions],
                "outstanding": self.outstanding}

    @property
    def outstanding(self) -> int:
        return len([q for q in self.questions if q.status == OUTSTANDING])


@dataclass
class ArtefactRequest:
    """What the client must send, from the governed input requirements."""

    outcome: str = ""
    required: List[Dict[str, str]] = field(default_factory=list)
    optional: List[Dict[str, str]] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DeliveryInstructions:
    """How and where to send it, using the production storage layout."""

    channel: str = ""
    file_format: str = ""
    cadence: str = ""
    locations: List[Dict[str, str]] = field(default_factory=list)
    naming: str = ""
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DraftEmail:
    to: List[str] = field(default_factory=list)
    cc: List[str] = field(default_factory=list)
    subject: str = ""
    body: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OnboardingPack:
    """Everything the client is asked for, in one reviewable document."""

    case_ref: str
    client_name: str = ""
    sections: List[PackSection] = field(default_factory=list)
    #: The progressive structured form, as the client would be served it.
    form: Dict[str, Any] = field(default_factory=dict)
    #: Category 1: already known, and material enough to put to a human.
    confirmations: List[PackQuestion] = field(default_factory=list)
    #: Everything the client is NOT asked, and why. Reported rather than
    #: hidden, so "is that missing?" has an answer.
    not_asked: List[Dict[str, Any]] = field(default_factory=list)
    #: Every field, classified. The operator's view of the whole catalogue.
    classification: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    artefacts: ArtefactRequest = field(default_factory=ArtefactRequest)
    delivery: DeliveryInstructions = field(default_factory=DeliveryInstructions)
    email: DraftEmail = field(default_factory=DraftEmail)
    mapping_statement: str = MAPPING_STATEMENT
    content_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_ref": self.case_ref,
            "client_name": self.client_name,
            "sections": [s.to_dict() for s in self.sections],
            "form": self.form,
            "confirmations": [q.to_dict() for q in self.confirmations],
            "not_asked": list(self.not_asked),
            "classification": list(self.classification),
            "summary": dict(self.summary),
            "artefacts": self.artefacts.to_dict(),
            "delivery": self.delivery.to_dict(),
            "email": self.email.to_dict(),
            "mapping_statement": self.mapping_statement,
            "content_hash": self.content_hash,
            "outstanding": self.outstanding,
            "questions": self.question_count,
        }

    @property
    def outstanding(self) -> int:
        return sum(s.outstanding for s in self.sections)

    @property
    def question_count(self) -> int:
        return sum(len(s.questions) for s in self.sections)

    @property
    def required_count(self) -> int:
        return len([q for s in self.sections for q in s.questions
                    if q.required])

    def document(self) -> str:
        """The pack as a document a human can read and a client can answer."""
        lines = [f"# Onboarding — {self.client_name or self.case_ref}",
                 "", f"Reference: {self.case_ref}", ""]
        lines += [f"There are {self.question_count} question"
                  f"{'s' if self.question_count != 1 else ''} for you, "
                  f"{self.required_count} of them required. Everything else "
                  "Trakt either already holds or works out itself.", ""]

        if self.confirmations:
            lines += ["## Please check these are right", "",
                      "Trakt already holds these. Tell us if any are wrong.",
                      ""]
            for question in self.confirmations:
                where = f" — {question.item}" if question.item else ""
                lines.append(f"- **{question.label}**{where}: "
                             f"{_render(question.value)}")
            lines.append("")

        for step in self._steps():
            lines += [f"## {step['label']}", ""]
            if step["help"]:
                lines += [step["help"], ""]
            for section in step["sections"]:
                if section.repeatable and section.item:
                    lines += [f"### {section.item}", ""]
                elif len(step["sections"]) > 1:
                    lines += [f"### {section.label}", ""]
                for question in section.questions:
                    mark = "[required]" if question.required else "[optional]"
                    lines.append(f"- {mark} **{question.label}**")
                    if question.help:
                        lines.append(f"      {question.help}")
                    if question.status != OUTSTANDING:
                        lines.append(f"      Currently: "
                                     f"{_render(question.value)}")
                lines.append("")

        lines += ["## Files to send", ""]
        for row in self.artefacts.required:
            lines.append(f"- **{row['label']}** — required")
        for row in self.artefacts.optional:
            lines.append(f"- {row['label']} — optional")
        if self.artefacts.note:
            lines += ["", self.artefacts.note]
        lines += ["", "## How to send them", ""]
        for key, value in (("Channel", self.delivery.channel),
                           ("File format", self.delivery.file_format),
                           ("How often", self.delivery.cadence)):
            if value:
                lines.append(f"- {key}: {value}")
        for location in self.delivery.locations:
            lines.append(f"- {location['label']}: `{location['location']}`")
        if self.delivery.naming:
            lines += ["", self.delivery.naming]

        lines += ["", "## What we are NOT asking you for", "",
                  self.mapping_statement, ""]
        deferred = sorted({row["label"] for row in self.not_asked
                           if row["category"] == "first_delivery"})
        if deferred:
            lines += ["Trakt reads these from the first file you send, so "
                      "there is nothing for you to fill in:", ""]
            lines += [f"- {label}" for label in deferred]
            lines.append("")
        return "\n".join(lines)

    def _steps(self) -> List[Dict[str, Any]]:
        """The pack's sections regrouped into the client-facing steps."""
        order: List[str] = []
        for section in self.sections:
            if section.step not in order:
                order.append(section.step)
        out: List[Dict[str, Any]] = []
        for key in order:
            sections = [s for s in self.sections if s.step == key]
            out.append({"key": key,
                        "label": sections[0].step_label or key,
                        "help": _step_help(key),
                        "sections": sections})
        return out


def _step_help(key: str) -> str:
    return next((s["help"] for s in _client_form.STEPS if s["key"] == key), "")


# --------------------------------------------------------------------------- #
# Building it
# --------------------------------------------------------------------------- #

def build(case: OnboardingCase, *, cat: Optional[Catalogue] = None,
          outcome: str = "mi", reporting_period: str = "") -> OnboardingPack:
    """Project the catalogue into what THIS client should actually be asked."""
    cat = cat or catalogue()
    client = case.answers.get("client") or {}
    pack = OnboardingPack(case_ref=case.case_id,
                          client_name=str(client.get("client_name") or ""))

    rows = _classification.classify_all(case.answers, cat=cat,
                                        provenance=case.provenance_class)
    pack.classification = [r.to_dict() for r in rows]
    pack.summary = _classification.summarise(rows)

    form = _client_form.build(case, cat=cat)
    pack.form = form.to_dict()
    pack.sections = _sections_from_form(form, rows, cat)
    pack.confirmations = [
        _question_from(r, cat, status=ANSWERED)
        for r in rows if r.category == _classification.KNOWN and r.confirm]
    pack.not_asked = [
        {"key": f"{r.section}.{r.field}", "label": r.label,
         "category": r.category, "number": r.number, "reason": r.reason}
        for r in rows if r.category in (_classification.DERIVED,
                                        _classification.FIRST_DELIVERY,
                                        _classification.INTERNAL,
                                        _classification.NOT_APPLICABLE)]

    pack.artefacts = _artefacts(outcome)
    pack.delivery = _delivery(case, reporting_period)
    pack.email = _email(case, pack)
    pack.content_hash = stable_hash(canonical_json({
        "form": pack.form,
        "confirmations": [q.to_dict() for q in pack.confirmations],
        "artefacts": pack.artefacts.to_dict(),
        "delivery": pack.delivery.to_dict(),
        "email": pack.email.to_dict(),
    }))
    return pack


def _sections_from_form(form: "_client_form.ClientForm",
                        rows: List[_classification.Classification],
                        cat: Catalogue) -> List[PackSection]:
    """The form, in the pack's own section shape.

    Kept so every existing reader — the review package, the status projection,
    the operator's document — sees one shape whether it came from the form or
    from the catalogue directly.
    """
    by_key = {(r.section, r.field, r.index): r for r in rows}
    out: List[PackSection] = []
    for step in form.steps:
        for group in step.groups:
            questions = []
            for f in group.fields:
                row = by_key.get((f.section, f.field, f.index))
                declared = cat.field(f.section, f.field)
                questions.append(PackQuestion(
                    section=f.section, field=f.field, label=f.label,
                    help=f.help,
                    status=ANSWERED if _present(f.value) else OUTSTANDING,
                    value=f.value,
                    provenance=(row.provenance if row else ""),
                    index=f.index, item=f.item, required=f.required,
                    evidence_required=f.evidence_required,
                    sensitive=f.sensitive,
                    writes_to=(declared.writes_to if declared else ""),
                    step=step.key, step_label=step.label))
            out.append(PackSection(
                key=group.key, label=group.label, help=group.help,
                repeatable=group.repeatable, questions=questions,
                step=step.key, step_label=step.label, item=group.item,
                index=group.index))
    return out


def _question_from(row: _classification.Classification, cat: Catalogue,
                   status: str) -> PackQuestion:
    f = cat.field(row.section, row.field)
    return PackQuestion(
        section=row.section, field=row.field, label=row.label,
        help=(f.help if f else ""), status=status, value=row.value,
        provenance=row.provenance, index=row.index, item=row.item,
        required=row.required,
        evidence_required=bool(f and f.evidence_required),
        sensitive=bool(f and f.sensitive),
        writes_to=(f.writes_to if f else ""))


def _artefacts(outcome: str) -> ArtefactRequest:
    vocab = artefact_vocabulary()
    return ArtefactRequest(
        outcome=outcome,
        required=[{"role": r, "label": vocab.label(r)}
                  for r in vocab.required_roles(outcome)],
        optional=[{"role": r, "label": vocab.label(r)}
                  for r in vocab.optional_roles(outcome)],
        note="One file per book per period. Send the whole file each time — "
             "Trakt takes the latest, and does not merge partial deliveries.")


def _delivery(case: OnboardingCase,
              reporting_period: str) -> DeliveryInstructions:
    sources = case.items("sources")
    first = sources[0] if sources else {}
    client_id = str((case.answers.get("client") or {}).get("client_id") or "")
    locations: List[Dict[str, str]] = []
    for source in sources:
        portfolio = case.portfolio(str(source.get("portfolio_id") or "")) or {}
        location = source.get("expected_location") or blob_prefix(
            client_id=client_id,
            portfolio_type=str(portfolio.get("portfolio_type") or "direct"),
            dataset=str(source.get("dataset") or "funded"),
            cadence=str(source.get("cadence") or ""),
            portfolio_id=str(source.get("portfolio_id") or ""))
        if reporting_period:
            location = str(location).replace("{reporting_period}",
                                             reporting_period)
        locations.append({
            "label": f"{source.get('portfolio_id')} {source.get('dataset')}",
            "location": str(location)})
    return DeliveryInstructions(
        channel=str(first.get("delivery_channel") or ""),
        file_format=str(first.get("file_format") or ""),
        cadence=str(first.get("cadence") or ""),
        locations=locations,
        naming="Name each file so the book and the period are visible; Trakt "
               "recognises the file from its contents, not its name, but a "
               "clear name makes any query faster to answer.",
        note="")


def _email(case: OnboardingCase, pack: OnboardingPack) -> DraftEmail:
    contacts = case.answers.get("contacts") or {}
    to = [str(contacts.get(key)) for key in
          ("reporting_contact_email", "operational_contact_email")
          if contacts.get(key)]
    name = pack.client_name or "there"
    outstanding = pack.outstanding
    body = "\n".join([
        f"Dear {name},",
        "",
        "We are setting your portfolio up on Trakt. Attached is the "
        "onboarding pack: it lists what we still need from you, and the files "
        "to send with it.",
        "",
        (f"There are {outstanding} question{'s' if outstanding != 1 else ''} "
         "outstanding. Everything else we have already, either from you or "
         "because Trakt works it out."
         if outstanding else
         "We have everything we need — please confirm the details in the pack "
         "are right."),
        "",
        "On field mappings: you do not need to map anything to our format. "
        "Send a representative file and we will do that here, and confirm it "
        "with you.",
        "",
        "Kind regards,",
        "Trakt Operations",
    ])
    return DraftEmail(to=to, subject=f"Trakt onboarding — {name} "
                                     f"({pack.case_ref})", body=body)


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return True
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    return bool(str(value).strip())


def _render(value: Any) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value) or "—"
    return str(value)
