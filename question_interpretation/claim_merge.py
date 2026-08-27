#!/usr/bin/env python3
"""question_interpretation/claim_merge.py — what happens to what the model says.

Stage 2 settled what the model MAY say: registered concepts only, constrained
by kind, zero raw field keys and zero off-tape fields proposable. This settles
what happens to it.

THREE RULES, AND THE THIRD IS THE ONE THAT EARNS ITS PLACE
----------------------------------------------------------
    1. the model may fill an EMPTY slot;
    2. the model may NOT overwrite a filled one;
    3. a disagreement on a filled slot is a FINDING, not a resolution.

Monotonicity alone is not sufficient, which is why rule 3 exists. "The model
may only add" is safe on the 20 type-(c) losses, where the concept never
arrived, and does nothing for Q21B, where the DETERMINISTIC side produced a
WRONG claim rather than a missing one: it bound the measure to
`current_loan_to_value` from the words "50% LTV" on a question asking for
balance growth. A merge that only adds leaves that claim in place and puts a
second one beside it. Reporting the disagreement tells us which side is wrong
before anyone decides who wins.

A GOVERNED DEFAULT IS A FILLED SLOT
-----------------------------------
This is the definition the whole module turns on, and it is not a technicality.

`chat_routing.py:1150–1166` is the guard that makes "Show me the trend." refuse,
and it fires on `subject.provenance == PROV_DEFAULT`. The schema says a default
subject is a real claim carrying a real value — the series still plots the
governed balance; what the provenance adds is that a consumer can TELL, which
nothing could before ("show me the trend" and "show me the balance trend"
produced identical contracts).

If this merge treated a governed default as an empty slot, the model would fill
it, the provenance would stop being `default`, the guard would stop firing, and
the question would answer. That is exactly how the Opus run walked through these
guards: the model supplied the missing element itself, so no default was ever
recorded and no guard ever saw one.

So a default is filled, and the model may not touch it. The measured
consequence is reported rather than designed around: `source_scope`, `dataset`
and `subject` are filled with a default on EVERY question this estate parses,
so this merge cannot reach a loss that lives in one of them.

AMBIGUITY MUST NOT BECOME SILENCE
---------------------------------
`direct` is a value of both `origination_channel` and `source_portfolio_type`,
and Stage 2 rejects it rather than preferring one. If that rejection produced a
silent non-fill it would be indistinguishable from the model proposing nothing
— which is Q20C's shape, where the model dropped `drawdown` entirely, and the
failure this whole split exists to guard against. Every rejected proposal
becomes a finding carrying its reason, so "proposed and refused" and "never
proposed" are different objects, not the same absence.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .schema import (CHOSEN_BY_A_PERSON, PROV_DEFAULT, PROV_EXPLICIT_USER,
                     PROV_MODEL_INFERRED, SCOPE_PROVENANCES)

__all__ = ["SLOTS", "KIND_TO_SLOT", "SlotValue", "MergeFinding", "MergeResult",
           "deterministic_slots", "merge", "merged_contract",
           "FILLED_BY_MODEL", "DECLINED_PERSON", "DECLINED_DEFAULT",
           "DECLINED_UNRECORDED", "AGREED", "AMBIGUOUS", "UNBINDABLE",
           "PROV_MODEL_INFERRED"]

SLOT_SUBJECT = "subject"
SLOT_SOURCE_SCOPE = "source_scope"
SLOT_DATASET = "dataset"
SLOT_DIMENSIONS = "dimensions"
SLOT_ROW_PREDICATES = "row_predicates"

SLOTS: Tuple[str, ...] = (SLOT_SUBJECT, SLOT_SOURCE_SCOPE, SLOT_DATASET,
                          SLOT_DIMENSIONS, SLOT_ROW_PREDICATES)

#: Slots addressed by GOVERNED FIELD rather than by name. A predicate on
#: `erm_product_type` and a predicate on `current_loan_to_value` are two slots,
#: not one slot written twice — which is why adding `drawdown` beside an
#: existing LTV threshold is a FILL and not an overwrite.
KEYED_SLOTS: Tuple[str, ...] = (SLOT_DIMENSIONS, SLOT_ROW_PREDICATES)

#: Which slot each proposal kind addresses. The mapping is total: a proposal
#: whose kind is not here cannot reach any slot.
KIND_TO_SLOT: Dict[str, str] = {
    "threshold": SLOT_ROW_PREDICATES,
    "measure": SLOT_SUBJECT,
    "source_book": SLOT_SOURCE_SCOPE,
    "dataset": SLOT_DATASET,
    "dimension": SLOT_DIMENSIONS,
    "category_value": SLOT_ROW_PREDICATES,
}

FILLED_BY_MODEL = "filled_by_model"
DECLINED_PERSON = "declined_slot_chosen_by_a_person"
DECLINED_DEFAULT = "declined_slot_carries_a_governed_default"
#: A filled slot NOTHING RECORDED THE PROVENANCE OF. Declined like any other
#: filled slot, and named separately because it is a gap rather than a
#: decision: `SubjectClaim.provenance` is None wherever the measure came from
#: the question rather than from the governed default, so a consumer cannot
#: tell "the reader named this measure" from "nobody recorded who did". The
#: decline is safe either way; the label says which of the two we are in.
DECLINED_UNRECORDED = "declined_slot_provenance_was_never_recorded"
AGREED = "agreed_with_the_deterministic_claim"
AMBIGUOUS = "proposal_was_ambiguous"
UNBINDABLE = "proposal_did_not_bind"


@dataclass(frozen=True)
class SlotValue:
    """One addressable slot and what occupies it."""

    slot: str
    key: Optional[str]
    value: Any
    provenance: Optional[str]
    #: THE COMPARISON, where the slot holds a predicate. `gt 50` and `lt 50`
    #: carry the same value and select opposite halves of the book, so a merge
    #: that compared values alone would call them the same claim and report no
    #: disagreement between a threshold and its inverse.
    operator: Optional[str] = None

    @property
    def address(self) -> Tuple[str, Optional[str]]:
        return (self.slot, self.key)

    @property
    def chosen_by_a_person(self) -> bool:
        return self.provenance in CHOSEN_BY_A_PERSON

    def as_dict(self) -> Dict[str, Any]:
        d = {"slot": self.slot, "key": self.key, "value": self.value,
             "provenance": self.provenance}
        if self.operator is not None:
            d["operator"] = self.operator
        return d


@dataclass(frozen=True)
class MergeFinding:
    """One thing that happened to one proposal. Never a silent outcome."""

    outcome: str
    slot: Optional[str]
    key: Optional[str]
    proposed: Any
    deterministic: Any = None
    deterministic_provenance: Optional[str] = None
    detail: str = ""

    @property
    def is_conflict(self) -> bool:
        """A disagreement on a FILLED slot, in either of its two shapes."""
        return self.outcome in (DECLINED_PERSON, DECLINED_DEFAULT,
                                DECLINED_UNRECORDED)

    def as_dict(self) -> Dict[str, Any]:
        return {"outcome": self.outcome, "slot": self.slot, "key": self.key,
                "proposed": self.proposed, "deterministic": self.deterministic,
                "deterministic_provenance": self.deterministic_provenance,
                "detail": self.detail, "is_conflict": self.is_conflict}


@dataclass(frozen=True)
class MergeResult:
    slots: Tuple[SlotValue, ...] = ()
    findings: Tuple[MergeFinding, ...] = ()

    @property
    def filled_by_model(self) -> Tuple[SlotValue, ...]:
        return tuple(s for s in self.slots
                     if s.provenance == PROV_MODEL_INFERRED)

    @property
    def conflicts(self) -> Tuple[MergeFinding, ...]:
        return tuple(f for f in self.findings if f.is_conflict)

    @property
    def ambiguous(self) -> Tuple[MergeFinding, ...]:
        return tuple(f for f in self.findings if f.outcome == AMBIGUOUS)

    def as_dict(self) -> Dict[str, Any]:
        return {"slots": [s.as_dict() for s in self.slots],
                "findings": [f.as_dict() for f in self.findings],
                "filled_by_model": [s.as_dict() for s in self.filled_by_model],
                "conflict_count": len(self.conflicts),
                "ambiguous_count": len(self.ambiguous)}


# --------------------------------------------------------------------------- #
# The deterministic side
# --------------------------------------------------------------------------- #
def deterministic_slots(interpretation: Any) -> Tuple[SlotValue, ...]:
    """The slots the deterministic claim set occupies, and with what authority.

    Read from the interpretation contract, which is where provenance lives. A
    slot ABSENT from this tuple is empty; a slot present is filled, whatever
    filled it.
    """
    out: List[SlotValue] = []

    subject = getattr(interpretation, "subject", None)
    if subject is not None and getattr(subject, "state", None) == "filled":
        out.append(SlotValue(SLOT_SUBJECT, None,
                             getattr(subject, "candidate_concept", None),
                             getattr(subject, "provenance", None)))
    scope = getattr(interpretation, "source_scope", None)
    if scope is not None and getattr(scope, "state", None) == "filled":
        out.append(SlotValue(SLOT_SOURCE_SCOPE, None,
                             getattr(scope, "scope", None),
                             getattr(scope, "provenance", None)))
    dataset = getattr(interpretation, "dataset", None)
    if dataset is not None and getattr(dataset, "state", None) == "filled":
        out.append(SlotValue(SLOT_DATASET, None,
                             getattr(dataset, "dataset", None),
                             getattr(dataset, "provenance", None)))

    # `DimensionClaim` and `RowPredicateClaim` CARRY NO PROVENANCE FIELD, and
    # they do not need one: the deterministic parser never raises either by
    # default. An axis is there because `_explicit_dimensions` found the words,
    # and a predicate is there because a clause resolved. So a filled one is
    # something the reader asked for, and it is recorded as such.
    #
    # If that ever stops being true the error is SAFE in the only direction
    # that matters: a defaulted axis mistaken for an explicit one is declined
    # more firmly, never filled more freely. Asserted by
    # `test_a_filled_axis_or_predicate_is_read_as_the_readers_own`.
    for dim in (getattr(interpretation, "dimensions", None) or ()):
        if getattr(dim, "state", None) != "filled":
            continue
        key = getattr(dim, "field_key", None) or getattr(dim, "candidate_concept", None)
        if key:
            out.append(SlotValue(SLOT_DIMENSIONS, str(key), str(key),
                                 getattr(dim, "provenance", None)
                                 or PROV_EXPLICIT_USER))
    for pred in (getattr(interpretation, "row_predicates", None) or ()):
        if getattr(pred, "state", None) != "filled":
            continue
        key = getattr(pred, "field_key", None) or getattr(pred, "field", None)
        if key:
            out.append(SlotValue(SLOT_ROW_PREDICATES, str(key),
                                 getattr(pred, "value", None),
                                 getattr(pred, "provenance", None)
                                 or PROV_EXPLICIT_USER,
                                 operator=getattr(pred, "operator", None)))
    return tuple(out)


# --------------------------------------------------------------------------- #
# The merge
# --------------------------------------------------------------------------- #
def _address(slot: str, field: Optional[str]) -> Tuple[str, Optional[str]]:
    return (slot, str(field) if slot in KEYED_SLOTS and field else None)


def _same(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is b
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        pass
    return str(a).strip().lower().replace("_", " ") == \
        str(b).strip().lower().replace("_", " ")


def _same_claim(current: "SlotValue", value: Any, operator: Optional[str]) -> bool:
    """The whole claim, not just its value. See `SlotValue.operator`."""
    if not _same(current.value, value):
        return False
    if current.operator is None and operator is None:
        return True
    return str(current.operator or "").lower() == str(operator or "").lower()


def merge(existing: Sequence[SlotValue], bound: Sequence[Any] = (),
          rejected: Sequence[Any] = ()) -> MergeResult:
    """Apply the three rules. Returns the merged slots AND every finding.

    ``bound`` and ``rejected`` are what `concept_proposal.bind` returned. The
    rejected proposals are carried through rather than dropped: a proposal the
    registry refused and a proposal that was never made must not arrive here as
    the same absence.
    """
    slots: Dict[Tuple[str, Optional[str]], SlotValue] = {
        s.address: s for s in existing}
    findings: List[MergeFinding] = []

    for item in bound or ():
        proposal = getattr(item, "proposal", None)
        kind = getattr(proposal, "kind", None)
        slot = KIND_TO_SLOT.get(kind)
        if slot is None:
            findings.append(MergeFinding(
                UNBINDABLE, None, None, getattr(proposal, "term", None),
                detail="no slot is addressed by kind %r" % (kind,)))
            continue
        field = getattr(item, "field", None)
        value = getattr(item, "value", None)
        operator = getattr(item, "operator", None)
        if value is None:
            value = field
        address = _address(slot, field)
        current = slots.get(address)

        if current is None:
            slots[address] = SlotValue(address[0], address[1], value,
                                       PROV_MODEL_INFERRED, operator=operator)
            findings.append(MergeFinding(
                FILLED_BY_MODEL, address[0], address[1], value,
                detail="the slot was empty%s"
                       % (" (%s)" % operator if operator else "")))
            continue
        if _same_claim(current, value, operator):
            findings.append(MergeFinding(
                AGREED, address[0], address[1], value, current.value,
                current.provenance,
                detail="the deterministic claim already says this"))
            continue
        if current.chosen_by_a_person:
            outcome = DECLINED_PERSON
        elif current.provenance == PROV_DEFAULT:
            outcome = DECLINED_DEFAULT
        else:
            outcome = DECLINED_UNRECORDED
        findings.append(MergeFinding(
            outcome, address[0], address[1], value, current.value,
            current.provenance,
            detail=("a filled slot is never overwritten; the disagreement is "
                    "reported and neither side is picked")))

    for item in rejected or ():
        proposal = getattr(item, "proposal", None)
        reason = getattr(item, "reason", "") or ""
        outcome = AMBIGUOUS if "more than one governed field" in reason \
            else UNBINDABLE
        findings.append(MergeFinding(
            outcome, KIND_TO_SLOT.get(getattr(proposal, "kind", None)), None,
            getattr(proposal, "term", None),
            detail="%s: %s" % (reason, getattr(item, "detail", ""))))

    ordered = sorted(slots.values(),
                     key=lambda s: (SLOTS.index(s.slot), s.key or ""))
    return MergeResult(tuple(ordered), tuple(findings))


# --------------------------------------------------------------------------- #
# Feeding Stage 1's completeness check
# --------------------------------------------------------------------------- #
_NUMBER_RE = __import__("re").compile(r"-?\d+(?:\.\d+)?")


def _stated_bounds(concept: Any) -> Tuple[float, ...]:
    """The numbers a stated threshold facet names.

    THE ONLY FACT THAT FACET RELIABLY CARRIES. `execution_receipt._detect_thresholds`
    does not resolve the field — which is why the facet raised for "borrowers
    over 55" is LABELLED "LTV over 55" — so the bound is the only thing both
    sides hold, and matching on it is not a shortcut but the whole of the
    available evidence.
    """
    out = []
    for token in _NUMBER_RE.findall(str(getattr(concept, "value", "") or "")):
        try:
            out.append(float(token))
        except ValueError:
            continue
    return tuple(out)


def merged_contract(contract: Any, result: MergeResult,
                    stated: Sequence[Any] = ()) -> Any:
    """``contract`` with the model's fills added, for the completeness check.

    Stage 1's check must run on the MERGED claim set, not on the deterministic
    one — Q20C is the case it exists for, where the model dropped `drawdown`
    entirely and proposed nothing. The registry cannot rescue a concept that
    never arrives; only the check sees it.

    Additive only, and only for what the model actually filled: a slot the
    merge declined must not appear here, or the check would be told a concept
    survived that did not.
    """
    import dataclasses

    filters = list(getattr(contract, "filters", ()) or ())
    dimensions = list(getattr(contract, "dimensions", ()) or ())
    metric = getattr(contract, "metric", None)
    dataset = getattr(contract, "dataset_reconciled", None)
    scope = getattr(contract, "scope_context", None)

    for slot in result.filled_by_model:
        if slot.slot == SLOT_ROW_PREDICATES and slot.key:
            filters.append(slot.key)
        elif slot.slot == SLOT_DIMENSIONS and slot.key:
            dimensions.append(slot.key)
        elif slot.slot == SLOT_SUBJECT and metric is None:
            metric = slot.value
        elif slot.slot == SLOT_DATASET and dataset is None:
            dataset = slot.value
        elif slot.slot == SLOT_SOURCE_SCOPE and scope in (None, "total"):
            scope = slot.value

    # A THRESHOLD THE MERGE APPLIED, RECORDED WHERE THE CHECK LOOKS.
    #
    # The completeness check decides a stated facet is carried by matching the
    # SERVED facet list on (kind, label), and everything above touches only
    # `filters`. Measured before this existed: a stated threshold stayed LOST
    # after the merge had filled the very predicate that satisfies it, so reach
    # on threshold losses was pinned at zero whatever the model proposed.
    #
    # FAILS CLOSED. A stated threshold is marked applied only where the merge
    # filled a comparator predicate whose bound is one of the numbers that
    # facet names. No match, no mark.
    facets = list(getattr(contract, "facets", ()) or ())
    filled_bounds = {slot.value for slot in result.filled_by_model
                     if slot.slot == SLOT_ROW_PREDICATES and slot.operator}
    if filled_bounds:
        for concept in stated or ():
            if getattr(concept, "kind", None) != "facet:threshold":
                continue
            label = getattr(concept, "value", None)
            if any(_same(bound, wanted) for bound in filled_bounds
                   for wanted in _stated_bounds(concept)):
                facets.append(("threshold", label, "applied"))

    return dataclasses.replace(
        contract, filters=tuple(dict.fromkeys(filters)),
        dimensions=tuple(dict.fromkeys(dimensions)), metric=metric,
        dataset_reconciled=dataset, scope_context=scope,
        facets=tuple(facets))
