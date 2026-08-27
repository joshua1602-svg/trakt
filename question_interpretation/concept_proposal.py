#!/usr/bin/env python3
"""question_interpretation/concept_proposal.py — the model reads, the registry binds.

THE SPLIT, AND WHY IT IS THIS SPLIT
-----------------------------------
Measured over the 75-question acceptance bank: 21 of 25 failures are concepts
plainly present in the sentence that the deterministic grammar cannot reach.
Zero are missing capability or missing data. Every one is computable today.

The Opus run is the natural experiment, because there the model did BOTH halves
— it read the sentence AND chose the governed field. Both of its fixes came
from reading unfamiliar phrasing. Two of its three breaks came from choosing the
field: `lump sum` was bound to `erm_sub_product_type` and `drawdown` to
`account_status`, where the book's own value catalogue claims each of them for
`erm_product_type` and for nothing else.

So: THE MODEL PROPOSES A CONCEPT IN REGISTERED VOCABULARY. IT NEVER NAMES A
GOVERNED FIELD. The registry binds, deterministically, through the owners that
already bind a deterministic claim.

WHAT PROTECTS THE BINDING IS THE KIND
-------------------------------------
A proposal carries a KIND, and each kind has exactly one owner:

    category_value   categorical_spans.value_field      the BOOK's catalogue
    measure          llm_query_parser._detect_metric    the measure grammar
    dimension        llm_query_parser._explicit_dimensions
    source_book      portfolio_lens.lens_from_term
    dataset          workspace.resolve_dataset

A `category_value` proposal is asked of the value catalogue and of nothing
else, so "lump sum" can reach `erm_product_type` and CANNOT reach
`erm_sub_product_type` — not because that field is filtered out, but because a
value proposal never consults the dimension owner at all. This matters more
than it looks: `erm_sub_product_type` IS a registered dimension on this
registry, and "erm sub product type" binds to it today. The registry alone does
not make the Opus break unreachable. The kind does, and dataset availability
closes the rest.

REJECTION, NEVER NEAREST-MATCH
------------------------------
A proposal naming an unregistered concept is REJECTED and reported by name. It
is never mapped to the nearest member — that substitution is invisible
downstream, which is precisely the failure mode this whole programme exists to
remove. A concept two governed fields both claim is likewise rejected, as
ambiguous: `direct` is a value of BOTH `origination_channel` and
`source_portfolio_type`, and resolving that by preference is a coin toss
recorded as a fact. The ambiguity is the registry's to fix.

CASE AND SPACING ARE NORMALISED BEFORE ANYTHING IS ASKED. `_detect_metric`
answers `current_loan_to_value` for "ltv" and `None` for "LTV"; every caller in
the serving path happens to lowercase first, so the trap has never fired there,
and a term-shaped binder walks straight into it.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field as _field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["CONCEPT_KINDS", "ProposedConcept", "BoundConcept", "RejectedConcept",
           "ConceptVocabulary", "vocabulary", "bind", "build_proposal_prompt",
           "parse_proposal_response", "ProposalFormatError",
           "REJECT_UNREGISTERED", "REJECT_AMBIGUOUS", "REJECT_UNAVAILABLE",
           "REJECT_UNKNOWN_KIND"]

KIND_VALUE = "category_value"
KIND_MEASURE = "measure"
KIND_DIMENSION = "dimension"
KIND_BOOK = "source_book"
KIND_DATASET = "dataset"

CONCEPT_KINDS: Tuple[str, ...] = (KIND_VALUE, KIND_MEASURE, KIND_DIMENSION,
                                  KIND_BOOK, KIND_DATASET)

REJECT_UNREGISTERED = "not a registered concept"
REJECT_AMBIGUOUS = "more than one governed field claims this concept"
REJECT_UNAVAILABLE = "this book does not carry the field it names"
REJECT_UNKNOWN_KIND = "not a proposable kind of concept"


class ProposalFormatError(ValueError):
    """The model's output was not a proposal. Never silently coerced."""


def _norm(term: Any) -> str:
    """Comparison form: lower-cased, underscores and runs of space collapsed."""
    return re.sub(r"[\s_]+", " ", str(term or "").strip().lower())


@dataclass(frozen=True)
class ProposedConcept:
    """One concept the model proposes, in registered vocabulary only."""

    kind: str
    term: str
    covers: str = ""

    def as_dict(self) -> Dict[str, str]:
        return {"kind": self.kind, "term": self.term, "covers": self.covers}


@dataclass(frozen=True)
class BoundConcept:
    """A proposal the REGISTRY resolved to a governed field."""

    proposal: ProposedConcept
    field: str
    value: Optional[str]
    owner: str

    def as_dict(self) -> Dict[str, Any]:
        return {**self.proposal.as_dict(), "field": self.field,
                "value": self.value, "bound_by": self.owner}


@dataclass(frozen=True)
class RejectedConcept:
    """A proposal that bound to nothing, and why. Never a nearest match."""

    proposal: ProposedConcept
    reason: str
    detail: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {**self.proposal.as_dict(), "rejected": self.reason,
                "detail": self.detail}


@dataclass(frozen=True)
class ConceptVocabulary:
    """Every concept the model may name, and nothing else.

    ``ambiguous`` is published rather than silently dropped: a concept more
    than one governed field claims is a gap in the registry, and the estate
    should be able to see how many it has without reading this file.
    """

    terms: Dict[str, Tuple[str, ...]] = _field(default_factory=dict)
    ambiguous: Dict[str, Tuple[str, ...]] = _field(default_factory=dict)
    cross_kind: Dict[str, Tuple[str, ...]] = _field(default_factory=dict)
    withheld: Dict[str, Tuple[str, ...]] = _field(default_factory=dict)
    semantics: Any = None
    available_values: Any = None
    available_columns: Any = None

    def offers(self, kind: str, term: str) -> bool:
        return _norm(term) in (self.terms.get(kind) or ())

    def as_dict(self) -> Dict[str, Any]:
        return {"kinds": {k: sorted(v) for k, v in self.terms.items()},
                "ambiguous": {k: sorted(v) for k, v in self.ambiguous.items()},
                "cross_kind": {k: sorted(v) for k, v in self.cross_kind.items()},
                "withheld": {k: sorted(v) for k, v in self.withheld.items()}}

    def size(self) -> int:
        return sum(len(v) for v in self.terms.values())


# --------------------------------------------------------------------------- #
# The vocabulary — assembled from the owners, scoped to the BOOK
# --------------------------------------------------------------------------- #
def vocabulary(semantics: Dict[str, Any], *, available_values: Any = None,
               available_columns: Optional[Iterable[str]] = None
               ) -> ConceptVocabulary:
    """The registered concepts, by kind, for THIS book.

    Scoped to the book on purpose. A registry can declare a field this tape
    does not carry — `erm_sub_product_type` is declared here and is not a
    column — and offering it as proposable would hand the model the exact
    binding the Opus run got wrong.
    """
    from mi_agent import llm_query_parser as LQ

    columns = set(available_columns or ())
    terms: Dict[str, set] = {k: set() for k in CONCEPT_KINDS}
    ambiguous: Dict[str, set] = {k: set() for k in CONCEPT_KINDS}

    # --- values: the BOOK's own catalogue, which is already book-scoped ---- #
    claimed: Dict[str, set] = {}
    for field_key, values in (available_values or {}).items():
        for known in (values.keys() if hasattr(values, "keys") else values):
            claimed.setdefault(_norm(known), set()).add(str(field_key))
    for term, fields in claimed.items():
        (terms if len(fields) == 1 else ambiguous)[KIND_VALUE].add(term)

    # --- dimensions and measures: the registry's maps, then the columns ---- #
    dim_terms = dict(LQ._registry_dimension_terms(semantics))
    dim_terms.update(LQ.EXPLICIT_DIMENSION_TERMS)
    for term, key in dim_terms.items():
        if columns and key not in columns:
            continue
        terms[KIND_DIMENSION].add(_norm(term))
    for term, key in LQ._registry_metric_terms(semantics).items():
        if columns and key not in columns:
            continue
        terms[KIND_MEASURE].add(_norm(term))
    # THE CURATED MEASURE GRAMMAR governs the core measures and the registry
    # maps drop them as over-generic single tokens — `balance` is not in
    # `_registry_metric_terms` at all. Leaving it out would make the commonest
    # measure in the book unproposable.
    for entry in LQ._METRIC_TERMS:
        term = entry[0] if isinstance(entry, (list, tuple)) else entry
        key = LQ._detect_metric(str(term), semantics)[0]
        if key and (not columns or key in columns):
            terms[KIND_MEASURE].add(_norm(term))

    # --- books and datasets: small, governed, and named by their owners ---- #
    terms[KIND_BOOK].update({"direct", "acquired", "total"})
    try:
        from mi_agent_api import workspace as W
        terms[KIND_DATASET].update(_norm(v) for v in getattr(W, "VIEWS", ())
                                   or ("funded", "pipeline"))
    except Exception:  # noqa: BLE001 - no owner reachable, the default pair
        terms[KIND_DATASET].update({"funded", "pipeline"})

    # EVERY OFFERED TERM MUST BIND, AND THAT IS ASSERTED, NOT ASSUMED.
    #
    # The registry's synonym maps and the question-shaped owners do not agree
    # term for term. `interest rate buckets` is in `_registry_dimension_terms`
    # and `_explicit_dimensions` does not recognise it; `portfolio type
    # (source)` is a registered business name whose parentheses the same owner
    # does not match. Offering either would hand the model a term it is invited
    # to use and then rejected for using — and the rejection would read as the
    # model's fault.
    #
    # Found by the census, not by inspection. The candidates are run through the
    # binder here so the class cannot recur: a term the owner will not bind is
    # WITHHELD and recorded, never silently dropped and never offered.
    probe = ConceptVocabulary(
        terms={k: tuple(sorted(v)) for k, v in terms.items()},
        semantics=semantics, available_values=available_values,
        available_columns=columns or None)
    withheld: Dict[str, set] = {}
    for kind in CONCEPT_KINDS:
        for term in sorted(terms[kind]):
            outcome = _bind_one(ProposedConcept(kind, term), probe)
            if not isinstance(outcome, BoundConcept):
                withheld.setdefault(kind, set()).add(term)
    for kind, dropped in withheld.items():
        terms[kind] -= dropped

    # CROSS-KIND COLLISIONS ARE REPORTED, NOT RESOLVED. `broker` is a value of
    # `origination_channel` AND a synonym for the `broker_channel` axis. The
    # KIND separates them, so binding is unambiguous — but a model that
    # proposes the wrong kind gets a wrong binding that looks entirely valid,
    # and the estate should be able to count these without reading this file.
    cross: Dict[str, set] = {}
    for kind, offered in terms.items():
        for other, other_offered in terms.items():
            if other <= kind:
                continue
            for shared in offered & other_offered:
                cross.setdefault(shared, set()).update({kind, other})

    return ConceptVocabulary(
        terms={k: tuple(sorted(v)) for k, v in terms.items()},
        ambiguous={k: tuple(sorted(v)) for k, v in ambiguous.items() if v},
        cross_kind={t: tuple(sorted(k)) for t, k in sorted(cross.items())},
        withheld={k: tuple(sorted(v)) for k, v in sorted(withheld.items())},
        semantics=semantics, available_values=available_values,
        available_columns=columns or None)


# --------------------------------------------------------------------------- #
# Binding — deterministic, by the owner for the proposal's KIND
# --------------------------------------------------------------------------- #
def _bind_one(proposal: ProposedConcept, vocab: ConceptVocabulary):
    from mi_agent import categorical_spans as CS
    from mi_agent import llm_query_parser as LQ

    kind, term = proposal.kind, _norm(proposal.term)
    if kind not in CONCEPT_KINDS:
        return RejectedConcept(proposal, REJECT_UNKNOWN_KIND,
                               "kinds are: " + ", ".join(CONCEPT_KINDS))
    if term in (vocab.ambiguous.get(kind) or ()):
        return RejectedConcept(proposal, REJECT_AMBIGUOUS, term)
    if not vocab.offers(kind, term):
        # NEVER A NEAREST MATCH. The nearest member of a governed vocabulary is
        # a different governed thing, and substituting one for the other is
        # invisible to every consumer downstream.
        return RejectedConcept(proposal, REJECT_UNREGISTERED, term)

    if kind == KIND_VALUE:
        hit = CS.value_field(term, vocab.available_values)
        if hit is None:
            return RejectedConcept(proposal, REJECT_AMBIGUOUS, term)
        return BoundConcept(proposal, hit[0], hit[1],
                            "categorical_spans.value_field")
    if kind == KIND_MEASURE:
        key = LQ._detect_metric(term, vocab.semantics)[0]
        if not key:
            return RejectedConcept(proposal, REJECT_UNREGISTERED, term)
        if vocab.available_columns and key not in vocab.available_columns:
            return RejectedConcept(proposal, REJECT_UNAVAILABLE, key)
        return BoundConcept(proposal, key, None,
                            "llm_query_parser._detect_metric")
    if kind == KIND_DIMENSION:
        keys = LQ._explicit_dimensions(term, vocab.semantics,
                                       available_columns=vocab.available_columns)[0]
        if not keys:
            return RejectedConcept(proposal, REJECT_UNREGISTERED, term)
        if len(set(keys)) > 1:
            return RejectedConcept(proposal, REJECT_AMBIGUOUS, ", ".join(sorted(set(keys))))
        key = keys[0]
        # THE AVAILABILITY FILTER IS ASSERTED HERE, not assumed of the owner.
        # `_explicit_dimensions` returns `erm_sub_product_type` for "erm sub
        # product type" WITH the tape's columns passed — measured — so a
        # binder that trusted it would offer the model the field the Opus run
        # got wrong.
        if vocab.available_columns and key not in vocab.available_columns:
            return RejectedConcept(proposal, REJECT_UNAVAILABLE, key)
        return BoundConcept(proposal, key, None,
                            "llm_query_parser._explicit_dimensions")
    if kind == KIND_BOOK:
        from mi_agent import portfolio_lens as PL
        lens = PL.lens_from_term(term)
        name = getattr(lens, "name", None)
        if not name:
            return RejectedConcept(proposal, REJECT_UNREGISTERED, term)
        return BoundConcept(proposal, "portfolio_lens", name,
                            "portfolio_lens.lens_from_term")
    return BoundConcept(proposal, "dataset", term, "workspace")


def bind(proposals: Sequence[ProposedConcept], vocab: ConceptVocabulary
         ) -> Tuple[List[BoundConcept], List[RejectedConcept]]:
    """``(bound, rejected)`` — deterministic, by the registry, never by the model."""
    bound: List[BoundConcept] = []
    rejected: List[RejectedConcept] = []
    for proposal in proposals or ():
        outcome = _bind_one(proposal, vocab)
        (bound if isinstance(outcome, BoundConcept) else rejected).append(outcome)
    return bound, rejected


# --------------------------------------------------------------------------- #
# The prompt, and reading what comes back
# --------------------------------------------------------------------------- #
_PROMPT_SYSTEM = """\
You map parts of a question about a loan portfolio onto a fixed list of \
concepts. You do not answer the question, choose a database field, write a \
query, or pick an analysis.

Reply with JSON only: {"concepts": [{"kind": ..., "term": ..., "covers": ...}]}

  kind    one of: %(kinds)s
  term    copied EXACTLY from the list below for that kind. Never invent one, \
never adapt one, never abbreviate one.
  covers  the words of the question this concept is for.

If part of the question matches no term in the list, LEAVE IT OUT. Do not offer \
the closest term instead — a near miss is a different thing and will be read as \
though the reader asked for it. An empty list is a valid and useful answer.

THE CONCEPTS AVAILABLE, BY KIND:
%(vocabulary)s"""


def build_proposal_prompt(question: str, vocab: ConceptVocabulary,
                          unresolved: Optional[Sequence[str]] = None
                          ) -> Dict[str, str]:
    """The proposal prompt. The vocabulary is IN it, so nothing else is nameable.

    Listing the vocabulary rather than describing it is the whole mechanism:
    the model is not asked what a phrase means in general, it is asked which of
    these named things the phrase is. A term outside the list is rejected on
    return regardless, but a model that never sees an alternative rarely
    proposes one.
    """
    blocks = []
    for kind in CONCEPT_KINDS:
        offered = vocab.terms.get(kind) or ()
        if offered:
            blocks.append("%s:\n  %s" % (kind, "\n  ".join(offered)))
    system = _PROMPT_SYSTEM % {"kinds": ", ".join(CONCEPT_KINDS),
                               "vocabulary": "\n\n".join(blocks)}
    user = "Question: %s" % (question or "")
    if unresolved:
        user += ("\n\nThese parts were not understood: "
                 + "; ".join(str(u) for u in unresolved))
    return {"system": system, "user": user}


def parse_proposal_response(text: str) -> List[ProposedConcept]:
    """The model's reply as proposals. STRICT — a malformed reply raises.

    Not "best effort": a reply this cannot read is a reply that proposed
    nothing, and inventing a proposal from a fragment of it would put the model
    back in the business of choosing.
    """
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\s*|\s*```$", "", raw, flags=re.I | re.S).strip()
    try:
        payload = json.loads(raw)
    except Exception as exc:  # noqa: BLE001 - reported, never guessed around
        raise ProposalFormatError("proposal was not JSON: %s" % exc) from exc
    if not isinstance(payload, dict) or "concepts" not in payload:
        raise ProposalFormatError("proposal has no 'concepts' key")
    items = payload.get("concepts")
    if not isinstance(items, list):
        raise ProposalFormatError("'concepts' is not a list")
    out: List[ProposedConcept] = []
    for item in items:
        if not isinstance(item, dict):
            raise ProposalFormatError("a concept is not an object: %r" % (item,))
        kind = str(item.get("kind") or "").strip()
        term = str(item.get("term") or "").strip()
        if not kind or not term:
            raise ProposalFormatError("a concept has no kind or no term: %r" % (item,))
        out.append(ProposedConcept(kind=kind, term=term,
                                   covers=str(item.get("covers") or "").strip()))
    return out
