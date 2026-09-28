"""A · The model cannot supply an executable binding — structurally.

This is the safety boundary the whole architecture rests on, so it is asserted
three ways rather than one:

  1. NO SLOT EXISTS. The CandidateIntent dataclass tree is walked and every
     field name is checked against the names a physical binding would need. A
     slot that does not exist cannot be filled, whatever a model returns, and
     this test fails the moment somebody adds one.
  2. AN UNKNOWN KEY IS REFUSED. A payload carrying ``field``, ``column``,
     ``sql`` or ``snapshot`` beside a legitimate slot does not have that key
     ignored — the whole intent is refused.
  3. NAMING IS NOT AUTHORITY. Opus may read a canonical identifier out of the
     governed registry and name it — that is deliberate, and the addendum's
     point. What it cannot do is make one up: an identifier the registry does
     not carry is UNREGISTERED_CONCEPT, an identifier two concepts claim is
     AMBIGUOUS, and one that does not apply to this asset class is
     CONCEPT_UNAVAILABLE. Every binding on a plan is the registry's answer,
     re-derived, not the model's string trusted.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Set, get_args, get_origin

import pytest

from mi_agent.interpretation_v2 import (
    CandidateIntent,
    IntentParseError,
    OUTCOME_PLAN,
    OUTCOME_REFUSE,
    parse_candidate_intent,
)
from mi_agent.interpretation_v2.intent import _PHYSICAL_KEY_NAMES

from .conftest import build_intent, intent_payload

#: Substrings that mark a field name as a physical binding rather than a
#: meaning. Checked against every dataclass field in the intent tree.
FORBIDDEN_FIELD_NAME_PARTS = (
    "field", "column", "canonical", "snapshot", "sql", "pandas", "dataframe",
    "expression", "code", "python", "as_of", "spec", "table", "predicate",
)

#: Names that legitimately contain a forbidden substring. Two only, both about
#: the model's own wording of a period, neither a binding.
ALLOWED_EXCEPTIONS = frozenset()


def _dataclass_tree(root) -> Set[type]:
    seen: Set[type] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if not dataclasses.is_dataclass(node) or node in seen:
            continue
        seen.add(node)
        for field in dataclasses.fields(node):
            for candidate in _annotation_types(field.type):
                if dataclasses.is_dataclass(candidate):
                    stack.append(candidate)
    return seen


def _annotation_types(annotation) -> list:
    if isinstance(annotation, str):
        return []
    found = [annotation]
    for arg in get_args(annotation):
        found.extend(_annotation_types(arg))
    return found


def test_no_intent_slot_can_carry_a_physical_binding():
    # Resolve string annotations (the module uses `from __future__ import
    # annotations`) so the tree walk sees real types.
    import mi_agent.interpretation_v2.intent as intent_module
    import typing

    classes = [obj for obj in vars(intent_module).values()
               if dataclasses.is_dataclass(obj)]
    assert classes, "no intent dataclasses found — the walk would pass vacuously"

    offenders = []
    for cls in classes:
        hints = typing.get_type_hints(cls)
        for name in hints:
            if name in ALLOWED_EXCEPTIONS:
                continue
            lowered = name.lower()
            if any(part in lowered for part in FORBIDDEN_FIELD_NAME_PARTS):
                offenders.append(f"{cls.__name__}.{name}")
    assert offenders == [], (
        "CandidateIntent grew a slot that could carry a physical binding: "
        + ", ".join(offenders))


@pytest.mark.parametrize("key,value", [
    ("field", "current_outstanding_balance"),
    ("column", "current_ltv"),
    ("canonical_field", "current_loan_to_value"),
    ("snapshot", "abc"),
    ("sql", "count"),
    ("dataframe", "loans"),
    ("spec", {}),
])
def test_a_physical_key_beside_a_valid_slot_refuses_the_whole_intent(key, value):
    payload = intent_payload()
    payload[key] = value
    with pytest.raises(IntentParseError) as excinfo:
        parse_candidate_intent(payload)
    assert excinfo.value.code == "MODEL_OUTPUT_CONTAINS_PHYSICAL_BINDING"


def test_a_physical_key_inside_a_measure_refuses_the_whole_intent():
    payload = intent_payload(measures=[{"concept": "balance",
                                        "field": "current_outstanding_balance"}])
    with pytest.raises(IntentParseError) as excinfo:
        parse_candidate_intent(payload)
    assert excinfo.value.code == "MODEL_OUTPUT_CONTAINS_PHYSICAL_BINDING"


def test_every_physical_key_name_is_covered_by_the_guard():
    # The guard is a list, and a list rots. This pins the intent behind it: any
    # name a caller would plausibly use to smuggle a binding is in the set.
    for expected in ("field", "column", "canonical_field", "sql", "snapshot",
                     "pandas", "dataframe", "code", "expression"):
        assert expected in _PHYSICAL_KEY_NAMES


@pytest.mark.parametrize("concept_id,expected_field", [
    ("current_outstanding_balance", "current_outstanding_balance"),
    ("current_loan_to_value", "current_loan_to_value"),
    ("youngest_borrower_age", "youngest_borrower_age"),
    ("erm_product_type", "erm_product_type"),
])
def test_a_named_identifier_is_a_proposal_not_an_authority(
        concept_id, expected_field, compiler):
    """Opus may READ an identifier out of the registry and name it.

    That is the addendum's whole point: an interpreter kept away from the
    governed metadata is an interpreter that clarifies on ordinary questions.
    What naming it does NOT do is shorten the check. The compiler re-derives
    the binding from the same authoritative index, and the plan's field is the
    registry's answer, not the model's string — which is why a term that
    resolves to a DIFFERENT field (an alias) lands on the registry's field and
    not on the word the model typed.
    """
    intent = build_intent(measures=[{"concept": concept_id}])
    result = compiler.compile(intent)
    assert result.outcome == OUTCOME_PLAN, result.codes()
    assert result.plan.outputs[0].measures[0].canonical_field == expected_field


def test_an_alias_binds_to_the_registrys_field_not_to_the_word_typed(compiler):
    intent = build_intent(measures=[{"concept": "balance"}])
    result = compiler.compile(intent)
    assert result.outcome == OUTCOME_PLAN
    measure = result.plan.outputs[0].measures[0]
    assert measure.concept == "current_outstanding_balance"
    assert measure.canonical_field == "current_outstanding_balance"


@pytest.mark.parametrize("invented", [
    "current_outstanding_balance_v2",
    "loans.current_balance",
    "tbl_loans__balance",
    "ebitda",
    "my_custom_column",
    "current_ltv_adjusted",
])
def test_an_unregistered_identifier_still_fails_closed(invented, compiler):
    """Seeing the registry is not the same as being able to invent an entry.

    An identifier that looks like a column but is not IN the registry is
    UNREGISTERED_CONCEPT, never a lucky binding.
    """
    result = compiler.compile(build_intent(measures=[{"concept": invented}]))
    assert result.outcome == OUTCOME_REFUSE
    assert "UNREGISTERED_CONCEPT" in result.codes()
    assert result.plan is None


def test_a_word_two_governed_concepts_claim_is_not_bound_to_either(compiler):
    """"region" is claimed by seven governed fields. Resolving it to one would
    be the compiler deciding what the reader meant."""
    result = compiler.compile(build_intent(measures=[{"concept": "region"}]))
    assert result.plan is None
    assert "AMBIGUOUS_MEASURE" in result.codes()
    reason = next(r for r in result.reasons if r.code == "AMBIGUOUS_MEASURE")
    assert len(reason.spans) > 1, "an ambiguity must name its candidates"


def test_a_concept_outside_this_asset_class_fails_closed(vocabulary, compiler):
    """Existence is not applicability.

    The Business Semantics Registry declares which asset classes a concept
    applies to, and the compiler checks it — a concept that exists but does not
    apply here is CONCEPT_UNAVAILABLE, not a plan.
    """
    from dataclasses import replace as _replace

    from mi_agent.interpretation_v2 import CompilerContext, DeterministicCompiler

    concept = vocabulary.resolve("current_outstanding_balance")
    narrowed = _replace(vocabulary, concepts=dict(
        vocabulary.concepts,
        current_outstanding_balance=_replace(
            concept, asset_applicability=("auto_finance",))))
    result = DeterministicCompiler(CompilerContext(narrowed)).compile(
        build_intent(measures=[{"concept": "current_outstanding_balance"}]))
    assert result.outcome == OUTCOME_REFUSE
    assert "CONCEPT_UNAVAILABLE" in result.codes()


def test_the_orientation_block_is_not_the_registry(vocabulary):
    """The model is oriented, then it RETRIEVES.

    The standing block carries the closed enumerations and counts — not the
    concepts. Dumping 150 of them into every prompt was the previous design's
    other mistake: 24k tokens a question, and still too little about each one.
    """
    payload = vocabulary.orientation_payload()
    assert "concepts" not in payload
    assert payload["concept_counts"]["total"] > 100
    assert "search_concepts" in payload["how_to_find_a_concept"]
