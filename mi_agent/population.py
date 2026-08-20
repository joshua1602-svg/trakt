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


@dataclass(frozen=True)
class Predicate:
    """One material row-population predicate, as requested."""

    field: str
    op: str
    value: Any

    def describe(self) -> str:
        if self.op == "eq":
            return f"{self.field} = {self.value}"
        if self.op == "in":
            vals = self.value if isinstance(self.value, (list, tuple, set)) else [self.value]
            return f"{self.field} in {sorted(str(v) for v in vals)}"
        return f"{self.field} {self.op} {self.value}"


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

    def to_dict(self) -> Dict[str, Any]:
        return {"applied": list(self.applied), "unavailable": list(self.unavailable),
                "rowsBefore": self.rows_before, "rowsAfter": self.rows_after}

    @property
    def is_complete(self) -> bool:
        return not self.unavailable


def _canonical(field_key: str, semantics: Optional[Mapping[str, Any]]) -> str:
    fields = (semantics or {}).get("fields") or {}
    entry = fields.get(field_key) or {}
    return str(entry.get("canonical_field") or field_key)


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
        if isinstance(raw, Mapping) and "op" in raw:
            out.append(Predicate(name, str(raw.get("op") or "eq"), raw.get("value")))
        elif isinstance(raw, (list, tuple, set)):
            out.append(Predicate(name, "in", list(raw)))
        else:
            out.append(Predicate(name, "eq", raw))
    return out


def apply_population(frame, predicates: Iterable[Predicate],
                     semantics: Optional[Mapping[str, Any]] = None,
                     ) -> Tuple[Any, PopulationEvidence]:
    """Narrow ``frame`` by every predicate, returning ``(frame, evidence)``.

    Reuses the executor's own comparison semantics rather than reimplementing
    them, so a route and the point-in-time path cannot disagree about what
    "age > 85" means. A predicate whose column is absent is recorded as
    UNAVAILABLE and the frame is left alone — the caller then refuses, because a
    population that could not be applied must never be answered as the whole book.
    """
    evidence = PopulationEvidence()
    predicates = list(predicates or [])
    if frame is None or not predicates:
        return frame, evidence

    # NB: `getattr(frame, "columns", []) or []` raises — a pandas Index has no
    # truth value. Build the set directly.
    columns = set(getattr(frame, "columns", []))
    evidence.rows_before = int(len(frame))
    work = frame
    for predicate in predicates:
        column = predicate.field if predicate.field in columns else _canonical(
            predicate.field, semantics)
        if column not in columns:
            evidence.unavailable.append(predicate.describe())
            continue
        try:
            work = work[_mask(work[column], predicate)]
        except Exception:  # noqa: BLE001 - an unappliable predicate is UNAVAILABLE
            evidence.unavailable.append(predicate.describe())
            continue
        evidence.applied.append(predicate.describe())
    evidence.rows_after = int(len(work))
    return work, evidence


def _mask(column: pd.Series, predicate: Predicate) -> pd.Series:
    """Boolean mask for one predicate, delegating numeric/date ops to the
    executor so the two paths cannot drift."""
    op, value = predicate.op, predicate.value
    if op in ("in", "not_in"):
        wanted = {str(v).strip().lower()
                  for v in (value if isinstance(value, (list, tuple, set)) else [value])}
        hit = column.astype("string").str.strip().str.lower().isin(wanted)
        return ~hit if op == "not_in" else hit
    if op in ("eq", "ne"):
        same = (column.astype("string").str.strip().str.lower()
                == str(value).strip().lower())
        return ~same if op == "ne" else same
    from .mi_query_executor import _apply_numeric_op

    return _apply_numeric_op(column, op, value)

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
#: ``origination_date ge LAST_4_WEEKS``. It is not a bound; it is the name of
#: one, and nothing downstream can evaluate it.
_SYMBOLIC_VALUE_RE = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$")
#: A number as a person writes it, with an optional magnitude suffix.
_QUESTION_NUMBER_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(bn|b|m|k)?", re.I)
_MAGNITUDE = {"bn": 1e9, "b": 1e9, "m": 1e6, "k": 1e3}


def _question_numbers(question: str) -> set:
    """Every numeric quantity the QUESTION states, magnitude suffixes expanded.

    "over £500k" states 500000 as surely as "over 500000" does, and a guard that
    only matched digits would call the resulting filter fabricated.
    """
    out: set = set()
    for raw, suffix in _QUESTION_NUMBER_RE.findall(question or ""):
        try:
            base = float(raw.replace(",", ""))
        except ValueError:
            continue
        out.add(base)
        if suffix:
            out.add(base * _MAGNITUDE[suffix.lower()])
    return out


def fabricated_bounds(filters: Optional[Mapping[str, Any]], question: str) -> List[str]:
    """Numeric or date BOUNDS the spec applies that the question never states.

    The sibling of :func:`fabricated_concepts`, and the mechanism behind the
    parser's self-disagreement: asked "how does recent lending compare with what
    we were originating earlier in the year?", the model sometimes emitted
    ``origination_date ge 2024-01-01``. The question names no date. Whether the
    model invents one varies run to run, so the same question answered once and
    refused the next time — a shipped surface where asking twice gives two
    different outcomes.

    Applying such a bound would be worse than refusing: it silently narrows the
    book to a population the user never described. Rejecting it lets the
    question fall back to the deterministic parse, which resolves "recent
    lending" through the GOVERNED window rather than an invented date.

    Only scalar bounds are considered. A categorical value ("Pipeline",
    "South East") is a name, not a bound, and is left to the concept guard and
    to the executor's own value matching.
    """
    stated = _question_numbers(question)
    text = (question or "").lower()
    bad: List[str] = []
    for field, raw in (filters or {}).items():
        value = raw.get("value") if isinstance(raw, Mapping) else raw
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, str):
            if _SYMBOLIC_VALUE_RE.match(value.strip()):
                bad.append(f"{field}={value}")
                continue
            date = re.match(r"^(\d{4})-\d{2}(?:-\d{2})?$", value.strip())
            if date and date.group(1) not in text and value.strip().lower() not in text:
                bad.append(f"{field}={value}")
            continue
        if isinstance(value, (int, float)):
            if not any(abs(float(value) - n) < 1e-9 for n in stated):
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
