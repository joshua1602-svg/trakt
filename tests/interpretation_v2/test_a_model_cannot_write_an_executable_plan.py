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
  3. A PHYSICAL COLUMN NAME DOES NOT BIND. Feeding the canonical field name
     instead of the business term is UNREGISTERED_CONCEPT, not a lucky match.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Set, get_args, get_origin

import pytest

from mi_agent.interpretation_v2 import (
    CandidateIntent,
    IntentParseError,
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


@pytest.mark.parametrize("physical_name", [
    "current_outstanding_balance",
    "current_loan_to_value",
    "canonical_region_reporting",
    "geographic_region_obligor_itl3",
    "youngest_borrower_age",
    "current_principal_balance",
    "indexed_loan_to_value",
    "original_loan_to_value",
    "current_interest_rate",
    "erm_product_type",
    "broker_channel",
])
def test_a_canonical_column_name_is_not_a_semantic_term(physical_name, compiler):
    """The physical schema is not the model's vocabulary.

    Each of these is a real column in the MI semantics registry whose BUSINESS
    term is something else ("balance", "current_ltv", "region"). Supplying the
    column name is not a shortcut to the field — it is an unregistered concept.
    """
    intent = build_intent(measures=[{"concept": physical_name}])
    result = compiler.compile(intent)
    assert result.outcome == OUTCOME_REFUSE
    assert "UNREGISTERED_CONCEPT" in result.codes()
    assert result.plan is None


def test_the_vocabulary_never_offers_the_model_a_column_name_as_a_term(vocabulary):
    """The model cannot learn the physical schema from what it is offered.

    The check is on the TERMS — the words the model is told to output — not on
    every string in the payload. A synonym list legitimately contains ordinary
    business words ("lien", "charge") that happen to spell some column
    elsewhere; a synonym binds nothing and is there to help the model recognise
    the concept, which is the opposite of leaking a schema.
    """
    from mi_agent.interpretation_v2 import canonical_field_names

    offered = set()
    for rows in vocabulary.prompt_payload()["concepts"].values():
        offered.update(row["term"] for row in rows)

    business_terms = {c.term for c in vocabulary.concepts.values()}
    leaked = sorted(f for f in canonical_field_names()
                    if f in offered and f not in business_terms)
    assert leaked == [], f"canonical fields offered as terms: {leaked}"

    # And every offered term really is a governed concept, not a stray string.
    assert offered <= business_terms
