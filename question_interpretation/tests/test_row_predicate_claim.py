"""The RESOLVED row-predicate channel — what the parser BOUND, on the contract.

Step 1 of the C6 filter binding. The scoping task established that the resolved
`field + operator + value` already exists upstream of every route
(`llm_query_parser._filter_field_of`, normalised by
`population.material_predicates`) and that the only reason a compositional plan
cannot read it is that the projection wrote the field into a provenance STRING
rather than a structure.

These tests are written so that DELETING `projection._row_predicates` fails
them, and so that a projection which merely echoed `FilterClaim` would fail them
too: every assertion names the resolved FIELD KEY, which `FilterClaim` does not
carry and deliberately never will.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from question_interpretation.schema import FILLED, RowPredicateClaim


_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def semantics():
    from mi_agent.mi_query_validator import load_mi_semantics
    return load_mi_semantics(
        _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml")


@pytest.fixture(scope="module")
def interpret(semantics):
    """Through the PRODUCTION assembly path.

    `projection.project` is the read-only Stage 1 harness: it calls
    `_deterministic_parse` directly and therefore never sees
    `resolve_seasoning_role`, which runs inside `parse_with_repair` and is where
    "new lending" becomes `months_on_book <= 1`. Production assembles through
    `from_parts` on the spec `ParsedQuestion.parse` returns. Testing the harness
    path would silently exempt the whole seasoning family.
    """
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    from mi_agent import execution_receipt as R
    from mi_agent.parsed_question import ParsedQuestion
    from question_interpretation import projection as proj

    def _run(question: str):
        spec = ParsedQuestion.parse(question, semantics).spec
        dim_terms = R.requested_dimension_terms(question, semantics, None)
        facets = R.detect_requested_facets(question, semantics, frame=None,
                                           requested_dimensions=dim_terms)
        return proj.from_parts(question, spec=spec, facets=facets,
                               dim_terms=dim_terms, semantics=semantics)

    return _run


def _triples(qi):
    return sorted((p.field_key, p.operator, p.value)
                  for p in qi.row_predicates)


# --------------------------------------------------------------------------- #
# The channel carries the resolved binding
# --------------------------------------------------------------------------- #
def test_a_numeric_threshold_arrives_bound_to_its_governed_field(interpret):
    qi = interpret("Show funded balance evolution by month for loans above 50% LTV.")
    assert _triples(qi) == [("current_loan_to_value", "gt", 50.0)]
    assert all(p.state == FILLED for p in qi.row_predicates)


def test_two_clauses_arrive_as_two_independently_bound_predicates(interpret):
    """Two families in one question, each bound to its OWN field.

    The failure this excludes is the one the scoping instrument hit: reading the
    English again downstream bound "above 50" to whichever field the last match
    left behind, because the measure span had not been masked. The contract
    carries the parser's own binding, so there is nothing left to re-read.
    """
    qi = interpret("Show funded balance for borrowers over 75 with LTV above 50.")
    assert _triples(qi) == [("current_loan_to_value", "gt", 50.0),
                            ("youngest_borrower_age", "gt", 75.0)]


def test_a_categorical_clause_arrives_as_a_value_not_a_threshold(interpret):
    qi = interpret("balance by region for joint borrowers")
    assert [(p.field_key, p.operator) for p in qi.row_predicates] \
        == [("borrower_type", "eq")]
    assert str(qi.row_predicates[0].value).strip().lower() == "joint"


def test_a_derived_population_arrives_as_the_predicate_it_executes(interpret):
    """"New lending" is not a field name, and the channel still carries a field.

    `resolve_seasoning_role` authorises the derivation upstream; the contract
    records the executed predicate, so a plan never has to know the phrase.
    """
    qi = interpret("Has the risk and borrower profile of new business changed recently?")
    assert _triples(qi) == [("months_on_book", "le", 1)]


# --------------------------------------------------------------------------- #
# It fabricates nothing
# --------------------------------------------------------------------------- #
def test_an_unfiltered_question_carries_no_row_predicates(interpret):
    qi = interpret("Show funded balance evolution by month.")
    assert qi.row_predicates == []


def test_a_source_portfolio_scope_never_becomes_a_row_predicate(interpret):
    """The P1I-A ruling: that phrase family is SCOPE and travels on
    `source_scope`. `population.material_predicates` excludes it by name, and
    the contract must not reintroduce it through a second door."""
    qi = interpret("What is the balance of the Northbridge portfolio?")
    assert "source_portfolio_id" not in {p.field_key for p in qi.row_predicates}


# --------------------------------------------------------------------------- #
# It is a THIRD channel, not a rename of either neighbour
# --------------------------------------------------------------------------- #
def test_the_filter_claim_still_says_what_the_question_said(interpret):
    """Additive. `filters` keeps the clause as worded, with no field on it —
    that separation is the reason this claim had to be new."""
    qi = interpret("Show funded balance evolution by month for loans above 50% LTV.")
    assert qi.filters, "the said-channel must still be populated"
    assert not hasattr(qi.filters[0], "field_key")
    assert qi.row_predicates[0].field_key == "current_loan_to_value"


def test_the_binding_survives_serialisation(interpret):
    qi = interpret("Show funded balance evolution by month for loans above 50% LTV.")
    d = qi.as_dict()
    assert d["row_predicates"] == [{
        **{k: v for k, v in d["row_predicates"][0].items()
           if k not in ("field_key", "operator", "value")},
        "field_key": "current_loan_to_value", "operator": "gt", "value": 50.0}]


def test_the_claim_defaults_empty_and_declares_nothing():
    """A bare claim must not assert a binding it does not have."""
    claim = RowPredicateClaim()
    assert (claim.field_key, claim.operator, claim.value) == (None, None, None)
    assert claim.state != FILLED
