"""operations_control.occ_agent.classification — who can actually answer this?

Catalogue coverage is not the same thing as a good client experience. Projecting
every collected field into a client pack produced 58 questions for a
straightforward equity-release onboarding, of which 20 were things the client
could not answer at all: identifiers Trakt mints, defaults Trakt applies, and
values the Onboarding Agent reads out of the first file it receives.

So every field is classified into exactly one of five categories, and only one
of them is a client question:

=========================  ===================================================
Category                   What happens to it
=========================  ===================================================
``KNOWN``      (1)         already answered — from the operator's instruction,
                           existing client records, or the selected products.
                           Pre-populated; shown for confirmation when material.
``CLIENT``     (2)         only the client knows it. **This is the pack.**
``DERIVED``    (3)         Trakt works it out. Populated automatically with
                           provenance; surfaced only if confirmation is needed.
``FIRST_DELIVERY`` (4)     learned from the first representative delivery by
                           the existing Onboarding Agent — source schema,
                           canonical mappings, schema fingerprints, data-quality
                           characteristics. **Never in the initial pack.**
``INTERNAL``   (5)         an operator or governance decision. Stays in the OCC
                           workflow; never client-facing.
=========================  ===================================================

The classification is **derived**, not listed. The catalogue's own ``source``
axis already answers most of it — that is what it is for — and this module maps
that axis onto the five categories, then applies two adjustments the axis cannot
express on its own:

* a field that is already answered is ``KNOWN`` whatever its source, because the
  question has been answered and asking it again is noise;
* :data:`DEFERRED_SECTIONS` names sections whose questions only make sense once
  something else exists. Describing what is in a file cannot be asked before the
  client has told Trakt which files there are.

:data:`OVERRIDES` exists for the handful of fields whose ``source`` is right for
generation but wrong for *who to ask*. It is deliberately tiny, and a test
asserts every entry still names a real catalogue field.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from ..onboarding.catalogue import (
    SOURCE_CLIENT,
    SOURCE_DEFAULT,
    SOURCE_DERIVED,
    SOURCE_INFERRED,
    SOURCE_OPERATOR,
    SOURCE_SYSTEM,
    Catalogue,
    Field,
    Section,
    catalogue,
)

# -- the five categories ---------------------------------------------------- #
KNOWN = "known"
CLIENT = "client"
DERIVED = "derived"
FIRST_DELIVERY = "first_delivery"
INTERNAL = "internal"

#: Not one of the five: a field this client will never be asked because the
#: catalogue's own condition excludes it. Reported so the classification is
#: complete, rather than hidden so the counts look tidier.
NOT_APPLICABLE = "not_applicable"

CATEGORIES = (KNOWN, CLIENT, DERIVED, FIRST_DELIVERY, INTERNAL,
              NOT_APPLICABLE)

CATEGORY_LABELS = {
    NOT_APPLICABLE: "Does not apply to this client",
    KNOWN: "Already known",
    CLIENT: "Only the client can answer",
    DERIVED: "Trakt works it out",
    FIRST_DELIVERY: "Learned from the first delivery",
    INTERNAL: "Internal operator decision",
}

#: The brief's five categories, plus a sixth that is not a category at all: a
#: field this client will never be asked because the catalogue's own condition
#: excludes it. Numbered 0 so it sorts ahead of the five and never reads as one.
CATEGORY_NUMBERS = {NOT_APPLICABLE: 0, KNOWN: 1, CLIENT: 2, DERIVED: 3,
                    FIRST_DELIVERY: 4, INTERNAL: 5}

#: The catalogue's ``source`` axis, mapped onto the categories. This is the
#: whole rule for a field nobody has answered yet.
_BY_SOURCE = {
    SOURCE_CLIENT: CLIENT,
    SOURCE_OPERATOR: INTERNAL,
    SOURCE_DEFAULT: DERIVED,
    SOURCE_DERIVED: DERIVED,
    SOURCE_SYSTEM: DERIVED,
    # "inferred" means, in the catalogue's own words, "read from data the client
    # uploads". That is the Onboarding Agent's job, and it happens at the first
    # delivery — so it is category 4, not a question.
    SOURCE_INFERRED: FIRST_DELIVERY,
}

#: Sections whose questions cannot be asked in the initial pack because they are
#: *about* something the client has not told Trakt about yet. Mapped to the
#: answer that has to exist first.
DEFERRED_SECTIONS: Dict[str, str] = {
    # "What does the balance column in this file mean?" needs the file list.
    # The client annotates their data when they describe or send it, not before.
    "data_definitions": "the client has listed the files they will send",
}

#: Fields whose ``source`` is right for *generation* but wrong for *who to ask*.
#: Deliberately tiny; each entry says why.
OVERRIDES: Dict[str, str] = {
    # Trakt mints the client code and the operator owns it. It appears on the
    # pack for information at most, never as a question.
    "client.client_id": INTERNAL,
    # The operator names books; a client's own reference goes in the display
    # name. Asking a client for Trakt's identifier invites a mismatch.
    "portfolios.portfolio_id": INTERNAL,
    # Governance flags an operator sets while working the case.
    "sources.sample_provided": INTERNAL,
    "sources.mapping_complete": INTERNAL,
    # Which of Trakt's books a delivery is: an operational classification.
    "sources.dataset": INTERNAL,
}


@dataclass
class Classification:
    """One field, and what is to be done about it."""

    section: str
    field: str
    label: str
    category: str
    #: Why, in one phrase, for the operator-facing classification view.
    reason: str
    source: str
    required: bool = False
    #: Whether a human should be shown it even though it is answered.
    confirm: bool = False
    value: Any = None
    provenance: str = ""
    index: Optional[int] = None
    item: str = ""

    @property
    def number(self) -> int:
        return CATEGORY_NUMBERS[self.category]

    @property
    def client_facing(self) -> bool:
        return self.category == CLIENT

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "number": self.number,
                "category_label": CATEGORY_LABELS[self.category]}


def classify(section: Section, f: Field, *,
             holder: Optional[Dict[str, Any]] = None,
             answers: Optional[Dict[str, Any]] = None,
             cat: Optional[Catalogue] = None,
             provenance: str = "") -> Classification:
    """The one category this field belongs to, given what is already answered."""
    cat = cat or catalogue()
    holder = holder or {}
    answers = answers or {}
    path = f"{section.key}.{f.key}"
    value = holder.get(f.key)
    required = cat.is_required(f, answers, holder)

    # A question the catalogue says is not worth asking this client is not a
    # question. Checked before everything except "already answered", so a
    # conditional field that HAS been answered still reports as known.
    if not _present(value) and not cat.is_asked(f, answers, holder):
        return Classification(
            section=section.key, field=f.key, label=f.label,
            category=NOT_APPLICABLE, reason=_why_not(f), source=f.source,
            required=False, value=value, provenance=provenance)

    category, reason = _category(section, f, path, value, answers)
    return Classification(
        section=section.key, field=f.key, label=f.label, category=category,
        reason=reason, source=f.source, required=required,
        confirm=_needs_confirmation(category, f, required),
        value=value, provenance=provenance)


def _category(section: Section, f: Field, path: str, value: Any,
              answers: Dict[str, Any]) -> tuple:
    # An internal decision stays internal whether or not it has been made. This
    # comes first: a client identifier Trakt has already minted is still not
    # something to put in front of a client for confirmation.
    override = OVERRIDES.get(path)
    if override:
        return override, "an operator decision, not a client question"

    # An answered question is not a question. A value the operator's instruction
    # supplied is KNOWN whatever its source, and re-asking it is exactly the
    # noise this classification removes.
    if _present(value):
        return KNOWN, f"already answered ({_origin(f)})"

    if section.key in DEFERRED_SECTIONS:
        return FIRST_DELIVERY, f"asked once {DEFERRED_SECTIONS[section.key]}"

    category = _BY_SOURCE.get(f.source, INTERNAL)
    return category, _REASONS[category]


_REASONS = {
    CLIENT: "only the client knows it",
    DERIVED: "Trakt works it out",
    FIRST_DELIVERY: "the Onboarding Agent reads it from the first delivery",
    INTERNAL: "an operator decision, not a client question",
    KNOWN: "already answered",
}


def _why_not(f: Field) -> str:
    return f"does not apply here ({f.asked_when})"


def _origin(f: Field) -> str:
    return {
        SOURCE_CLIENT: "from the client",
        SOURCE_OPERATOR: "set by an operator",
        SOURCE_INFERRED: "read from a delivery",
        SOURCE_DERIVED: "derived by Trakt",
        SOURCE_DEFAULT: "a Trakt default",
        SOURCE_SYSTEM: "minted by Trakt",
    }.get(f.source, f.source)


def _needs_confirmation(category: str, f: Field, required: bool) -> bool:
    """Whether an answered or automatic value should still be put to a human.

    Material means: it was required, or it carries evidence, or it is sensitive.
    An optional default nobody will ever look at is not worth a confirmation,
    and asking for one on every field would make all of them meaningless.
    """
    if category == KNOWN:
        # Only what a client could meaningfully CHECK. An identifier Trakt
        # minted and a link between two records it holds are not things a
        # client has ever seen, so putting them up for confirmation invites a
        # shrug rather than a correction — and makes the ones that do matter
        # easier to skip. Those are confirmed by the operator on the review
        # package instead.
        if f.source in (SOURCE_OPERATOR, SOURCE_SYSTEM, SOURCE_DERIVED):
            return False
        if f.type == "entity_reference":
            return False
        return required or f.evidence_required or f.sensitive
    if category == DERIVED:
        return f.evidence_required or f.sensitive
    return False


# --------------------------------------------------------------------------- #
# Whole-catalogue views
# --------------------------------------------------------------------------- #

def classify_all(answers: Optional[Dict[str, Any]] = None, *,
                 cat: Optional[Catalogue] = None,
                 provenance: Optional[Dict[str, str]] = None
                 ) -> List[Classification]:
    """Every field in the catalogue, classified against these answers.

    Repeatable sections are classified per item, so a case with two portfolios
    reports two portfolio classifications and one set of client-level ones.
    """
    cat = cat or catalogue()
    answers = answers or {}
    provenance = provenance or {}
    products = set((answers.get("reporting") or {}).get("products") or [])
    out: List[Classification] = []
    for section in cat.sections:
        if section.repeatable:
            items = answers.get(section.key) or [{}]
            for index, item in enumerate(items):
                label = str(item.get(section.item_label_field) or "").strip()
                for f in section.fields:
                    if _excluded(section, f, products):
                        continue
                    row = classify(section, f, holder=item, answers=answers,
                                   cat=cat,
                                   provenance=provenance.get(
                                       f"{section.key}[{index}].{f.key}", ""))
                    row.index, row.item = index, label
                    out.append(row)
        else:
            block = answers.get(section.key) or {}
            for f in section.fields:
                if _excluded(section, f, products):
                    continue
                out.append(classify(
                    section, f, holder=block, answers=answers, cat=cat,
                    provenance=provenance.get(f"{section.key}.{f.key}", "")))
    return out


def _excluded(section: Section, f: Field, products: set) -> bool:
    """A regime field for a product this client has not selected is not a field
    for this client at all — it is not asked, derived, deferred or internal."""
    return bool(section.from_regime and f.product and f.product not in products)


def summarise(rows: List[Classification]) -> Dict[str, Any]:
    """Counts per category, for the operator view and for the record."""
    counts = {c: 0 for c in CATEGORIES}
    for row in rows:
        counts[row.category] += 1
    return {
        "total": len(rows),
        "counts": counts,
        "labels": dict(CATEGORY_LABELS),
        "numbers": dict(CATEGORY_NUMBERS),
        "client_facing": counts[CLIENT],
        "confirmations": len([r for r in rows if r.confirm]),
    }


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return True
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    return bool(str(value).strip())
