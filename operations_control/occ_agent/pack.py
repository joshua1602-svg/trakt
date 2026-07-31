"""operations_control.occ_agent.pack — the client-facing onboarding pack.

A **projection** of the governed catalogue and the case's current answers.
Nothing here decides what to ask: every question is a field
``config/onboarding/field_catalogue.yaml`` declares, in the section the
catalogue puts it in, with the help text the catalogue gives it. A field added
there appears in the pack; a field removed disappears from it. There is no
second questionnaire, and a test asserts every question traces to a catalogue
field.

What the pack adds is the *shape a client can answer*: sections grouped for a
reader rather than a wizard, each question marked answered, outstanding or
worked out by Trakt, the artefact request derived from the governed input
requirements, delivery instructions built with the production path rules, and a
covering email.

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
from ..onboarding.catalogue import Catalogue, Field, Section, catalogue
from ..onboarding.derivation import blob_prefix
from .input_roles import artefact_vocabulary

#: What the pack tells a client about mappings. A governed decision, stated —
#: never a silent omission.
MAPPING_STATEMENT = (
    "Trakt does not ask you to map your fields to ours. Send a representative "
    "file and Trakt will propose the mapping itself; an operator reviews and "
    "approves it during the first ingestion, and it is then fixed for every "
    "later delivery. What Trakt cannot work out on its own is what your "
    "numbers MEAN, which is what the questions under \"How to read the data\" "
    "are for."
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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PackSection:
    key: str
    label: str
    help: str = ""
    repeatable: bool = False
    questions: List[PackQuestion] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "label": self.label, "help": self.help,
                "repeatable": self.repeatable,
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

    def document(self) -> str:
        """The pack as a document a human can read and a client can answer."""
        lines = [f"# Onboarding — {self.client_name or self.case_ref}",
                 "", f"Reference: {self.case_ref}", ""]
        lines += ["Please complete the questions below and return this "
                  "document with the files listed at the end.", ""]
        for section in self.sections:
            if not section.questions:
                continue
            lines += [f"## {section.label}", ""]
            if section.help:
                lines += [section.help, ""]
            for question in section.questions:
                mark = {ANSWERED: "[answered]", DERIVED: "[Trakt supplies]",
                        OUTSTANDING: "[  ]"}[question.status]
                where = f" — {question.item}" if question.item else ""
                lines.append(f"- {mark} **{question.label}**{where}")
                if question.help:
                    lines.append(f"      {question.help}")
                if question.status != OUTSTANDING:
                    lines.append(f"      Currently: {_render(question.value)}")
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
        lines += ["", "## About field mappings", "", self.mapping_statement,
                  ""]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Building it
# --------------------------------------------------------------------------- #

#: Sections a client never sees: Trakt derives every field in them.
def _client_visible(section: Section) -> bool:
    return any(f.collected and not f.answered_by_trakt for f in section.fields)


def build(case: OnboardingCase, *, cat: Optional[Catalogue] = None,
          outcome: str = "mi", reporting_period: str = "") -> OnboardingPack:
    """Project the catalogue and the case's answers into a client pack."""
    cat = cat or catalogue()
    client = case.answers.get("client") or {}
    pack = OnboardingPack(case_ref=case.case_id,
                          client_name=str(client.get("client_name") or ""))

    for section in cat.sections:
        if not _client_visible(section):
            continue
        built = PackSection(key=section.key, label=section.label,
                            help=section.help, repeatable=section.repeatable)
        if section.repeatable:
            items = case.items(section.key) or [{}]
            for index, item in enumerate(items):
                label = str(item.get(section.item_label_field) or "").strip()
                for f in section.fields:
                    q = _question(case, cat, section, f, item, index, label)
                    if q is not None:
                        built.questions.append(q)
        else:
            block = case.answers.get(section.key) or {}
            for f in section.fields:
                if section.from_regime and f.product \
                        and f.product not in case.products:
                    continue
                q = _question(case, cat, section, f, block, None, "")
                if q is not None:
                    built.questions.append(q)
        if built.questions:
            pack.sections.append(built)

    pack.artefacts = _artefacts(outcome)
    pack.delivery = _delivery(case, reporting_period)
    pack.email = _email(case, pack)
    pack.content_hash = stable_hash(canonical_json({
        "sections": [s.to_dict() for s in pack.sections],
        "artefacts": pack.artefacts.to_dict(),
        "delivery": pack.delivery.to_dict(),
        "email": pack.email.to_dict(),
    }))
    return pack


def _question(case: OnboardingCase, cat: Catalogue, section: Section,
              f: Field, holder: Dict[str, Any], index: Optional[int],
              item: str) -> Optional[PackQuestion]:
    if not f.collected:
        return None                    # derived or system: never asked
    value = holder.get(f.key)
    present = _present(value)
    required = cat.is_required(f, case.answers, holder)
    if f.answered_by_trakt:
        status = ANSWERED if present else DERIVED
    else:
        status = ANSWERED if present else OUTSTANDING
    path = f"{section.key}.{f.key}"
    return PackQuestion(
        section=section.key, field=f.key, label=f.label, help=f.help,
        status=status, value=value,
        provenance=str(case.provenance_class.get(path)
                       or case.provenance.get(path) or ""),
        index=index, item=item, required=required,
        evidence_required=f.evidence_required, sensitive=f.sensitive,
        writes_to=f.writes_to)


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
