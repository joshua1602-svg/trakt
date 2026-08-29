"""mi_agent.population — the governed ROW POPULATION, and proof that it ran.

A question can name a population in two different channels, and until P1L only
one of them survived into a routed answer:

  * the PORTFOLIO SCOPE channel — "the acquired book", "the current portfolio" —
    carried by :mod:`mi_agent.portfolio_lens` and applied by every lens-aware
    route;
  * the ROW PREDICATE channel — "the back book", "borrowers over 85", "loans
    below 75% LTV" — carried in ``spec.filters`` and applied by the point-in-time
    executor.

Twelve of thirteen specialist routes read the first channel and ignored the
second. So "which region grew the most last month in the back book?" computed
regional movement across the WHOLE book, returned ``ok=True``, and shipped a spec
that still claimed the back-book filter. The figure was plausible — the back book
is 89% of this portfolio — and nothing in the answer, the warnings or the receipt
revealed that the population had been dropped.

This module makes that population a first-class thing that must be PROVEN:

    requested predicates  ->  applied to the frame  ->  evidence  ->  P0 facet

The rule it exists to enforce is simple. If a route can honour the population,
it applies it and records what it applied. If it cannot, the answer refuses.
There is no third outcome in which the whole book is quietly substituted.

Scope stays in the scope channel. This module deliberately does NOT treat
``source_portfolio_id`` as a row predicate: P1I-A governs that phrase family as
SCOPE, and collapsing it into a filter here to make one common mechanism would
regress that ruling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import pandas as pd

#: Carried by the portfolio LENS, not by this module. Listing them keeps the two
#: channels from both claiming the same key (see the module docstring).
SCOPE_KEYS = ("source_portfolio_id", "source_portfolio_type")

#: Keys that sit in ``spec.filters`` without defining which rows are included.
#: The governing question from the P1L brief is: "would removing this predicate
#: potentially change which loans enter the calculation?" These do not — they
#: select a reporting basis or a snapshot, which the routes resolve themselves.
#: Anything not listed here is treated as population-defining, so the ledger
#: fails SAFE for a predicate nobody has classified yet.
NON_POPULATION_KEYS = (
    "reporting_date", "data_cut_off_date", "cut_off_date", "snapshot_date",
    "as_of_date", "baseline_date", "current_date", "run_id", "client_id",
)


def _plain(value: Any) -> str:
    """A number a reader would write: 50, not 50.0."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


@dataclass(frozen=True)
class Predicate:
    """One material row-population predicate, as requested."""

    field: str
    op: str
    value: Any

    def describe(self) -> str:
        """The MACHINE form: field key, operator code, value.

        Read by the population ledger, the reachability proof in
        `question_interpretation.b5_reachability` (which depends on this leading
        with the FIELD NAME), and every facet label built from a predicate. It is
        deliberately not prose — see :meth:`spoken` for that.
        """
        if self.op == "eq":
            return f"{self.field} = {self.value}"
        if self.op == "in":
            vals = self.value if isinstance(self.value, (list, tuple, set)) else [self.value]
            return f"{self.field} in {sorted(str(v) for v in vals)}"
        return f"{self.field} {self.op} {self.value}"

    #: Operator codes as a reader says them. `describe` keeps the codes.
    _SPOKEN_OPS = {"eq": "is", "ne": "is not", "in": "is one of",
                   "gt": "over", "ge": "at least", "lt": "under",
                   "le": "at most", "between": "between"}

    def spoken(self, semantics: Optional[Mapping[str, Any]] = None) -> str:
        """The READER'S form: business name, operator in words, value.

        "current_loan_to_value gt 50.0" appeared verbatim in a customer-facing
        refusal. The field key and the operator code are internal identifiers;
        the reader asked about LTV over 50.

        A SECOND rendering, not a replacement: `describe` is still what the
        ledger records and what every facet label is built from, because the B5
        reachability proof holds only while those labels lead with the field
        name. Only prose reads this.
        """
        entry = ((semantics or {}).get("fields") or {}).get(self.field) or {}
        name = (entry.get("business_name") or entry.get("display_name")
                or str(self.field).replace("_", " "))
        word = self._SPOKEN_OPS.get(self.op, self.op)
        if self.op == "in":
            vals = self.value if isinstance(self.value, (list, tuple, set)) else [self.value]
            return f"{name} {word} {', '.join(sorted(str(v) for v in vals))}"
        return f"{name} {word} {_plain(self.value)}"


@dataclass
class PopulationEvidence:
    """What a route ACTUALLY applied. Execution evidence, not intent.

    ``rows_before``/``rows_after`` are the proof: a predicate that is claimed but
    narrowed nothing on a frame that contains the column is reported as applied
    with an unchanged count, which is itself informative. A predicate whose
    column the frame does not carry is ``unavailable`` — never silently skipped.
    """

    applied: List[str] = field(default_factory=list)
    unavailable: List[str] = field(default_factory=list)
    rows_before: Optional[int] = None
    rows_after: Optional[int] = None
    #: The governed failure that stopped execution, when one did. Set together
    #: with a ``None`` frame, so "could not execute" is a state a caller cannot
    #: mistake for "executed and changed nothing".
    blocked_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "applied": list(self.applied), "unavailable": list(self.unavailable),
            "rowsBefore": self.rows_before, "rowsAfter": self.rows_after}
        if self.blocked_reason:
            out["blockedReason"] = self.blocked_reason
        return out

    @property
    def is_usable(self) -> bool:
        """Did every requested predicate actually execute?"""
        return not self.unavailable and self.blocked_reason is None

    @property
    def is_complete(self) -> bool:
        return not self.unavailable


def _canonical(field_key: str, semantics: Optional[Mapping[str, Any]]) -> str:
    fields = (semantics or {}).get("fields") or {}
    entry = fields.get(field_key) or {}
    return str(entry.get("canonical_field") or field_key)


def predicate_of(key: Any, raw: Any) -> Predicate:
    """One spec filter entry, as a governed Predicate.

    THE normaliser for the three shapes ``spec.filters`` uses. Both executors
    go through it, so "what shape did the spec use" stops being a semantic
    input: before this, `_apply_filters` dispatched on the shape and
    `material_predicates` erased it, which is how one predicate came to have
    two meanings.
    """
    name = str(key)
    if isinstance(raw, Mapping) and any(k in raw for k in ("op", "value", "min", "max")):
        # `min`/`max` without an `op` is a shape the executor already accepted;
        # reproducing its unwrapping here is what stops the two from disagreeing
        # about a bound the reader did supply. `material_predicates` used to
        # keep the whole dict as the value, so the receipt claimed a predicate
        # that could only ever match nothing.
        return Predicate(name, str(raw.get("op") or "eq"),
                         raw.get("value", raw.get("min", raw.get("max"))))
    if isinstance(raw, (list, tuple, set)):
        return Predicate(name, "in", list(raw))
    return Predicate(name, "eq", raw)


def material_predicates(filters: Optional[Mapping[str, Any]],
                        semantics: Optional[Mapping[str, Any]] = None,
                        ) -> List[Predicate]:
    """The population-defining predicates in a spec's filters.

    Scope keys are excluded (they travel on the lens) and a small set of
    reporting-basis keys are excluded because they select a snapshot rather than
    a row set. Everything else counts: the ledger must fail safe for a predicate
    that no one has classified, rather than assume it is harmless.
    """
    out: List[Predicate] = []
    for key, raw in (filters or {}).items():
        name = str(key)
        if name in SCOPE_KEYS or name in NON_POPULATION_KEYS:
            continue
        out.append(predicate_of(name, raw))
    return out


def apply_population(frame, predicates: Iterable[Predicate],
                     semantics: Optional[Mapping[str, Any]] = None,
                     ) -> Tuple[Any, PopulationEvidence]:
    """Narrow ``frame`` by every predicate, returning ``(frame, evidence)``.

    Executes through :func:`mi_agent.mi_query_executor.governed_predicate_mask`
    — THE one governed meaning of a predicate — so this and the point-in-time
    executor cannot disagree about what "LTV over 50" selects. It used to reuse
    only the COMPARATOR, which left it with no percent rescaling, no operator
    aliases, no value-domain resolution and no field resolution: five distinct
    ways for a route and the point-in-time path to answer the same question
    differently.

    FAIL CLOSED. A predicate that cannot be executed returns ``(None, evidence)``
    with the reason on the evidence. It does NOT return the frame it failed to
    narrow: a caller that forgets to read the evidence then crashes loudly
    instead of answering a whole-book figure as though it were the narrowed one.
    That was the P1L failure mode, and "the caller is supposed to check" is what
    made it possible.
    """
    evidence = PopulationEvidence()
    predicates = list(predicates or [])
    if frame is None or not predicates:
        return frame, evidence

    from .mi_query_executor import MIQueryExecutionError, governed_predicate_mask

    evidence.rows_before = int(len(frame))
    work = frame
    for predicate in predicates:
        try:
            execution = governed_predicate_mask(work, predicate.field, predicate.op,
                                                predicate.value, semantics or {})
        except MIQueryExecutionError as exc:
            # The same governed failure the point-in-time executor raises, in
            # the shape this path's callers already read.
            evidence.unavailable.append(predicate.describe())
            evidence.blocked_reason = str(exc)
            return None, evidence
        work = work[execution.mask]
        evidence.applied.append(predicate.describe())
    evidence.rows_after = int(len(work))
    return work, evidence


# --------------------------------------------------------------------------- #
# Fabricated populations — the mirror of the P1L loss check
# --------------------------------------------------------------------------- #
#: The governed population CONCEPTS, and the spec keys that execute each of them.
#: A concept is executed if the spec filters on any of its keys; the derivation is
#: authorised, so "back book" may legitimately run as ``months_on_book > cutoff``
#: without the user ever saying "months_on_book".
_CONCEPT_KEYS = {
    "seasoning": ("seasoning_segment", "seasoning_bucket", "months_on_book"),
    "provenance": ("source_portfolio_type",),
}

#: Question vocabulary that REQUESTS each concept, beyond the segment phrases the
#: seasoning module already owns. Taken from the registry synonyms for those
#: fields rather than invented here, so the two stay in step.
_CONCEPT_TERMS = {
    "seasoning": ("months on book", "time on book", "mob", "seasoning", "loan age",
                  "seasoning band", "seasoning bucket", "seasoning cohort",
                  "months on book band", "time on book band"),
    "provenance": ("source portfolio type", "origination type", "book type",
                   "direct or acquired", "portfolio type"),
}


def _requests_concept(concept: str, question: str) -> bool:
    """Does the QUESTION ask for this governed population concept?

    Asked of the governed vocabularies that already own each concept — the
    seasoning segment phrases from :mod:`mi_agent.seasoning`, the provenance
    lens from :mod:`mi_agent.portfolio_lens` — so this adds no new population
    model and cannot drift from them.
    """
    text = (question or "").lower()
    if any(term in text for term in _CONCEPT_TERMS[concept]):
        return True
    if concept == "seasoning":
        from .seasoning import names_lending_window

        # The GOVERNED LENDING WINDOWS, which are a superset of the binary
        # segment phrases: a question naming "our new lending" has requested the
        # seasoning concept just as surely as one naming "the front book", and
        # under the 2026-08 ruling it names a governed window. Consulting the
        # window vocabulary rather than the segment vocabulary keeps this guard
        # in step with the axis instead of with one partition of it.
        return names_lending_window(question)
    from .portfolio_lens import resolve_lens

    return getattr(resolve_lens(question), "name", "") in ("direct", "acquired")


#: A placeholder the model emitted instead of resolving a value —
#: ``origination_date ge LAST_4_WEEKS``. It is not a bound; it is the NAME of
#: one, and nothing downstream can evaluate it. Reported separately from a
#: fabricated bound because the cause is different: a fabricated bound is the
#: model inventing a number, while this is a template token reaching the output
#: unfilled.
_SYMBOLIC_VALUE_RE = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$")
#: A number as a person writes it: thousands separators, decimals, and an
#: optional magnitude suffix. The separated form is matched FIRST, so
#: "£1,500,000" is one number and not "1,500" followed by "000".
_QUESTION_NUMBER_RE = re.compile(
    r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(bn|b|m|k)?", re.I)
_MAGNITUDE = {"bn": 1e9, "b": 1e9, "m": 1e6, "k": 1e3}
#: How close a spec value must be to a stated one to count as the same number.
_VALUE_TOLERANCE = 1e-6


def _question_numbers(question: str) -> set:
    """Every numeric quantity the QUESTION states, as the parser might hold it.

    Three renderings of the same bound must all match, or a legitimate filter is
    rejected for having been normalised:

    * magnitude suffixes — "over £500k" states 500000 as surely as "over 500000"
    * thousands separators — "£250,000" is 250000, and "£1,500,000" is 1500000
    * percentage scale — a question saying "50%" may be held as 50 or as 0.5,
      depending on whether the field stores points or a fraction

    The percentage forms are added unconditionally rather than only for percent
    fields: this set is the ALLOWED list, and a guard that wrongly rejects a
    real filter is worse than one that lets an invented bound through when the
    invented number happens to be a hundredth of a stated one.
    """
    out: set = set()
    for raw, suffix in _QUESTION_NUMBER_RE.findall(question or ""):
        try:
            base = float(raw.replace(",", ""))
        except ValueError:
            continue
        out.add(base)
        out.add(base / 100.0)
        out.add(base * 100.0)
        if suffix:
            scaled = base * _MAGNITUDE[suffix.lower()]
            out.add(scaled)
            out.add(scaled / 100.0)
    return out


def _governed_field(field: str, semantics: Optional[Mapping[str, Any]]) -> bool:
    """Is this a field the governed registry carries?

    Asked of the registry rather than of a list here, so a field added to the
    registry is immediately accepted without a code change.
    """
    if not semantics:
        return True          # nothing to check against; the value test still applies
    fields = semantics.get("fields") or semantics
    return str(field) in fields


def fabricated_bounds(filters: Optional[Mapping[str, Any]], question: str,
                      semantics: Optional[Mapping[str, Any]] = None
                      ) -> List[str]:
    """Filters the spec applies that the question does not support.

    The sibling of :func:`fabricated_concepts`, and the mechanism behind the
    parser's self-disagreement: asked "how does recent lending compare with what
    we were originating earlier in the year?", the model sometimes emitted
    ``origination_date ge 2024-01-01``. The question names no date. Whether the
    model invents one varies run to run, so the same question answered once and
    refused the next time — a shipped surface where asking twice gives two
    different outcomes.

    A filter is rejected when EITHER half fails:

    * the FIELD is not one the governed registry carries, or
    * the VALUE is a scalar bound the question never states.

    Both halves are needed. Checking the value alone lets a fabricated field
    through whenever its invented number happens to appear in the question —
    ``unicorn_score gt 50`` survives "balance where LTV above 50%". Checking the
    field alone lets an invented bound through on a real field, which is the
    common case here.

    Only scalar bounds are considered for the value half. A categorical value
    ("Pipeline", "South East") is a name, not a bound, and is left to the
    concept guard and to the executor's own value matching.

    ``fabricated_concepts`` does not cover this. It reads the same filter slots,
    but only for the two governed population CONCEPTS it knows — seasoning and
    provenance — so a filter on any other field is invisible to it. It is a
    concept guard, not a field guard, and this is the field-and-bound one.
    """
    stated = _question_numbers(question)
    text = (question or "").lower()
    bad: List[str] = []
    for field, raw in (filters or {}).items():
        if not _governed_field(field, semantics):
            bad.append(f"{field} (not a governed field)")
            continue
        value = raw.get("value") if isinstance(raw, Mapping) else raw
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, str):
            if _SYMBOLIC_VALUE_RE.match(value.strip()):
                bad.append(f"{field}={value} (unresolved placeholder)")
                continue
            date = re.match(r"^(\d{4})-\d{2}(?:-\d{2})?$", value.strip())
            if date and date.group(1) not in text and value.strip().lower() not in text:
                bad.append(f"{field}={value}")
            continue
        if isinstance(value, (int, float)):
            if not any(abs(float(value) - n) < _VALUE_TOLERANCE for n in stated):
                bad.append(f"{field}={value}")
    return sorted(bad)


def fabricated_concepts(filters: Optional[Mapping[str, Any]], question: str) -> List[str]:
    """Governed population concepts the SPEC executes that the QUESTION never asked for.

    P1L protects one direction: a population the question requested must reach
    execution or the answer refuses. This is the missing mirror. A model that
    invents a population is not caught by that check, because the predicate really
    was applied, the frame really was narrowed and the evidence really was
    recorded — so the facet is legitimately APPLIED and every guard reports
    success while the answer covers the wrong book.

    The case that forced this: "what is the balance of the sponsored book?" is a
    governed ENTIRE_AUM phrase, and the model emitted ``seasoning_segment = back
    book``. On the demonstration book that returned £1.793bn against a true
    £1.964bn — 8.7% low, entirely plausible, with ok=True and no warning.

    Concepts, never literal words: a question naming "back book" authorises the
    whole seasoning derivation, including a bare ``months_on_book`` predicate.
    """
    present = {str(k) for k in (filters or {})}
    return [concept for concept, keys in _CONCEPT_KEYS.items()
            if (present & set(keys)) and not _requests_concept(concept, question)]
