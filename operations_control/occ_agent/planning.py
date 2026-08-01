"""operations_control.occ_agent.planning — what an instruction would do.

Everything an instruction proposes is worked out here, **without writing
anything**. The service then either shows the plan for confirmation or applies
it. That ordering is the whole point: an operator can never discover after the
fact that a sentence did something they did not see proposed.

Two rules the module exists to enforce.

**Nothing is applied that was not understood.** A plan carries three
populations, and all three are reported: what the agent read (``understood``),
what it could not read (``unrecognised``), and what it read but cannot resolve
without help (``questions``). A plan with either of the last two is refused
unless the human explicitly confirms the disclosed remainder.

**A repeatable item is never silently replaced.** Portfolios, entities and
deliveries are identified by a stable field. The rules are exhaustive and there
is no fallback:

===========================  =========================================
Incoming item                Outcome
===========================  =========================================
identifier matches an item   update that item
identifier differs           add a new item
no identifier, nothing yet   add
no identifier, one existing  *propose* updating it — never assumed
no identifier, several       a question; nothing is applied
===========================  =========================================

The earlier "one book named two ways is one book" shortcut is gone. It renamed
a client's first portfolio when an operator said "they also have direct_102",
and lost the delivery registration with it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from ..onboarding.case import OnboardingCase
from ..onboarding.catalogue import Catalogue, Section

# -- actions ---------------------------------------------------------------- #
ADD = "add"
UPDATE = "update"
PROPOSE_UPDATE = "propose_update"
UNCHANGED = "unchanged"

#: Which field identifies an item within a repeatable catalogue section. Taken
#: from the section's own declaration where it has one, so a new repeatable
#: section does not need a change here.
_IDENTITY_FIELDS = {
    "portfolios": "portfolio_id",
    "entities": "legal_name",
    "sources": "source_key",
}


def identity_field(section: Section) -> str:
    return _IDENTITY_FIELDS.get(section.key, section.repeatable_key or "")


def item_label(section: Section, item: Dict[str, Any], index: int) -> str:
    name = str(item.get(section.item_label_field) or "").strip()
    if name:
        return name
    ident = str(item.get(identity_field(section)) or "").strip()
    return ident or f"item {index + 1}"


# --------------------------------------------------------------------------- #
# One proposed value
# --------------------------------------------------------------------------- #

@dataclass
class ProposedFieldChange:
    """One field this instruction would set, and what it would replace."""

    section: str
    section_label: str
    field: str
    label: str
    before: Any = None
    after: Any = None
    #: ``None`` for a non-repeatable section; the item's position otherwise.
    index: Optional[int] = None
    item: str = ""                      # the item's own label, when repeatable
    action: str = UPDATE                # ADD | UPDATE | UNCHANGED
    provenance: str = ""
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def material(self) -> bool:
        """Whether a human must see this before it is written.

        Overwriting an answer that is already there is material; filling a blank
        is not. So is anything read with less than full confidence.
        """
        if self.action == UNCHANGED:
            return False
        if self.confidence is not None and self.confidence < 1.0:
            return True
        return _present(self.before) and self.before != self.after

    def sentence(self) -> str:
        where = f" ({self.item})" if self.item else ""
        if self.action == ADD or not _present(self.before):
            return f"{self.label}{where}: {_render(self.after)}"
        return (f"{self.label}{where}: {_render(self.before)} → "
                f"{_render(self.after)}")


@dataclass
class ItemQuestion:
    """Something the agent read but cannot place without being told."""

    section: str
    section_label: str
    question: str
    candidates: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# The plan
# --------------------------------------------------------------------------- #

@dataclass
class ApplicationPlan:
    """What one instruction would change. Writes nothing.

    ``steps`` is what would be handed to ``OnboardingService.save_step`` — the
    platform's own writer — once the human is satisfied.
    """

    steps: Dict[str, Any] = field(default_factory=dict)
    understood: List[ProposedFieldChange] = field(default_factory=list)
    questions: List[ItemQuestion] = field(default_factory=list)
    #: Fragments of the instruction the agent could not read at all.
    unrecognised: List[str] = field(default_factory=list)
    #: Delivery facts, which are the run's rather than the case's.
    reporting_period: str = ""
    cadence: str = ""
    #: Operational data streams the instruction declared ("funded",
    #: "pipeline"), each becoming its own source registration on apply.
    streams: List[str] = field(default_factory=list)
    expected_artefacts: List[str] = field(default_factory=list)
    #: Provenance per ``section.field``, carried through to the case.
    provenance: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": self.steps,
            "understood": [c.to_dict() for c in self.understood],
            "questions": [q.to_dict() for q in self.questions],
            "unrecognised": list(self.unrecognised),
            "reporting_period": self.reporting_period,
            "cadence": self.cadence,
            "streams": list(self.streams),
            "expected_artefacts": list(self.expected_artefacts),
            "provenance": dict(self.provenance),
            "changes": self.change_count,
            "material": self.material,
            "complete": self.complete,
            "summary": self.summary(),
        }

    @property
    def change_count(self) -> int:
        return len([c for c in self.understood if c.action != UNCHANGED])

    @property
    def empty(self) -> bool:
        return not (self.change_count or self.reporting_period or self.cadence
                    or self.streams or self.expected_artefacts
                    or self.questions)

    @property
    def complete(self) -> bool:
        """Whether the whole instruction was read."""
        return not self.unrecognised and not self.questions

    @property
    def material(self) -> bool:
        """Whether the human must confirm before anything is written.

        An incomplete instruction is always material, whatever it changes: the
        operator has to see what was dropped before the rest is applied.
        """
        return (not self.complete
                or any(c.material for c in self.understood))

    def summary(self) -> str:
        parts: List[str] = []
        changes = [c for c in self.understood if c.action != UNCHANGED]
        if changes:
            parts.append("; ".join(c.sentence() for c in changes[:6]))
            if len(changes) > 6:
                parts.append(f"and {len(changes) - 6} more")
        if self.streams:
            parts.append("Streams: "
                         + " and ".join(f"a {s} book" for s in self.streams)
                         + ", registered separately")
        if self.reporting_period:
            parts.append(f"Reporting period: {self.reporting_period}")
        if self.cadence:
            parts.append(f"Expected cadence: {self.cadence}")
        return ". ".join(parts) + "." if parts else "Nothing to change."

    def disclosure(self) -> Dict[str, Any]:
        """The four things every turn must report, in operator words."""
        return {
            "understood": [c.sentence() for c in self.understood
                           if c.action != UNCHANGED]
                          + [f"A {s} stream will be registered"
                             for s in self.streams],
            "proposed": self.summary(),
            "questions": [q.question for q in self.questions],
            "unrecognised": list(self.unrecognised),
        }


# --------------------------------------------------------------------------- #
# Building a plan
# --------------------------------------------------------------------------- #

def plan_changes(case: OnboardingCase, cat: Catalogue, *,
                 steps: Dict[str, Any],
                 provenance: Optional[Dict[str, str]] = None,
                 confidence: Optional[Dict[str, float]] = None
                 ) -> ApplicationPlan:
    """Work out what ``steps`` would do to ``case``. Writes nothing."""
    provenance = provenance or {}
    confidence = confidence or {}
    plan = ApplicationPlan(provenance=dict(provenance))

    for step, payload in (steps or {}).items():
        section = cat.section(step)
        if section is None:
            plan.unrecognised.append(f"'{step}' is not part of this onboarding")
            continue
        if section.repeatable:
            _plan_repeatable(plan, case, section, payload, provenance,
                             confidence)
        else:
            _plan_block(plan, case, section, payload, provenance, confidence)
    return plan


def _plan_block(plan: ApplicationPlan, case: OnboardingCase, section: Section,
                payload: Dict[str, Any], provenance: Dict[str, str],
                confidence: Dict[str, float]) -> None:
    block = case.answers.get(section.key) or {}
    proposed: Dict[str, Any] = {}
    for key, value in (payload or {}).items():
        f = section.field(key)
        if f is None:
            plan.unrecognised.append(
                f"'{key}' is not a field of {section.label}")
            continue
        before = block.get(key)
        path = f"{section.key}.{key}"
        change = ProposedFieldChange(
            section=section.key, section_label=section.label, field=key,
            label=f.label, before=before, after=value,
            action=(UNCHANGED if _same(before, value)
                    else UPDATE if _present(before) else ADD),
            provenance=provenance.get(path, ""),
            confidence=confidence.get(path))
        plan.understood.append(change)
        if change.action != UNCHANGED:
            proposed[key] = value
    if proposed:
        plan.steps[section.key] = proposed


def _plan_repeatable(plan: ApplicationPlan, case: OnboardingCase,
                     section: Section, payload: Dict[str, Any],
                     provenance: Dict[str, str],
                     confidence: Dict[str, float]) -> None:
    """The identity rules. Exhaustive, with no fallback."""
    incoming = list((payload or {}).get(section.key)
                    or (payload or {}).get("items") or [])
    existing = [dict(item) for item in case.items(section.key)]
    key = identity_field(section)
    merged = [dict(item) for item in existing]
    touched = False

    for candidate in incoming:
        unknown = [k for k in candidate if section.field(k) is None]
        for k in unknown:
            plan.unrecognised.append(
                f"'{k}' is not a field of {section.label}")
        values = {k: v for k, v in candidate.items() if k not in unknown}
        if not values:
            continue

        ident = str(values.get(key) or "").strip() if key else ""
        index: Optional[int] = None
        action = ADD

        if ident:
            index = next((i for i, m in enumerate(merged)
                          if str(m.get(key) or "").strip() == ident), None)
            # A different explicit identifier ADDS. It never renames.
            action = UPDATE if index is not None else ADD
        elif not merged:
            action = ADD
        elif len(merged) == 1:
            # One unambiguous target: propose, never assume.
            index, action = 0, PROPOSE_UPDATE
        else:
            plan.questions.append(ItemQuestion(
                section=section.key, section_label=section.label,
                question=(f"Which {section.label.lower()} does this apply to? "
                          f"Name it, or give its "
                          f"{_label_for(section, key)}."),
                candidates=[item_label(section, m, i)
                            for i, m in enumerate(merged)]))
            continue

        if action == ADD:
            merged.append(dict(values))
            index = len(merged) - 1
            before_item: Dict[str, Any] = {}
        else:
            before_item = dict(merged[index])           # type: ignore[index]
            merged[index].update(                        # type: ignore[index]
                {k: v for k, v in values.items() if _present(v)})

        label = item_label(section, merged[index], index)  # type: ignore[index]
        for field_key, value in values.items():
            f = section.field(field_key)
            path = f"{section.key}.{field_key}"
            before = before_item.get(field_key)
            change = ProposedFieldChange(
                section=section.key, section_label=section.label,
                field=field_key, label=(f.label if f else field_key),
                before=before, after=value, index=index, item=label,
                action=(ADD if action == ADD
                        else UNCHANGED if _same(before, value)
                        else UPDATE),
                provenance=provenance.get(path, ""),
                confidence=(0.5 if action == PROPOSE_UPDATE
                            else confidence.get(path)))
            plan.understood.append(change)
        touched = True

    if touched:
        plan.steps[section.key] = {section.key: merged}


def _label_for(section: Section, key: str) -> str:
    f = section.field(key) if key else None
    return (f.label if f else "identifier").lower()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return True
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    return bool(str(value).strip())


def _same(before: Any, after: Any) -> bool:
    if isinstance(before, list) and isinstance(after, list):
        return list(before) == list(after)
    return str(before or "") == str(after or "")


def _render(value: Any) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value) or "—"
    return str(value)
