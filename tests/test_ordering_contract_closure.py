"""The ordering closure: the contract carries the VALUES, not only the wording.

`OperationClaim.type == RANKING` said a ranking was named. It said nothing about
what to rank, which way, on what basis, or how many — and measured over the 97
corpus questions carrying ranking language, `modifiers` was empty on every one.
All four facts were resolved downstream from raw English.

This is the third application of a closure the contract has already made twice
(`trend_window` -> `window_periods`, `comparison_period` -> `comparison_periods`),
and these tests pin the same three properties each of those closed:

  * the values are carried, not the wording alone;
  * they come from the EXISTING owner, so the contract cannot disagree with it;
  * the wording is kept beside them — the closure is ADDITIVE.

`ordering_of` is the one field that is not merely carriage. It separates a
ranking of a LEVEL from a ranking of a MOVEMENT, which nothing in the contract
could express before, and which is why a question asking which region GREW could
be answered with which region IS.
"""
from __future__ import annotations

import pytest

from question_interpretation.schema import (
    ORDER_BASIS_ABSOLUTE, ORDER_BASIS_COUNT, ORDER_BASIS_SHARE,
    ORDER_DECREASE, ORDER_INCREASE, ORDER_OF_MOVEMENT, DimensionClaim,
    OperationClaim, RANKING)


def _project(question):
    from mi_agent import execution_receipt as R
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent.parsed_question import ParsedQuestion
    from mi_agent_api.data_source import semantics_path
    from question_interpretation import projection as proj

    semantics = load_mi_semantics(semantics_path())
    cols = ["current_outstanding_balance", "current_loan_to_value",
            "broker_channel", "geographic_region_obligor", "loan_identifier",
            "reporting_date"]
    spec = ParsedQuestion.parse(question, semantics).spec
    terms = R.requested_dimension_terms(question, semantics, cols)
    facets = R.detect_requested_facets(question, semantics, frame=None,
                                       requested_dimensions=terms)
    return proj.from_parts(question, spec=spec, facets=facets, dim_terms=terms,
                           semantics=semantics)


# --------------------------------------------------------------------------- #
# The schema itself
# --------------------------------------------------------------------------- #
def test_an_unknown_ordering_value_is_refused():
    """A controlled vocabulary that accepts anything is not one."""
    for kwargs in ({"ordering_direction": "sideways"},
                   {"ordering_basis": "vibes"},
                   {"ordering_of": "neither"}):
        with pytest.raises(ValueError):
            OperationClaim(type=RANKING, **kwargs)
    with pytest.raises(ValueError):
        OperationClaim(type=RANKING, ordering_limit=0)


def test_an_unstated_ordering_subject_is_none_and_not_level():
    """The distinction the field exists for.

    "does not say" must not read as "ranks a level": a consumer that defaults
    the unknown to level reintroduces exactly the substitution this field was
    added to prevent.
    """
    claim = OperationClaim(type=RANKING)
    assert claim.ordering_of is None
    assert claim.orders_a_movement is False
    assert OperationClaim(type=RANKING,
                          ordering_of=ORDER_OF_MOVEMENT).orders_a_movement


def test_a_dimension_claim_carries_every_field_it_could_bind_to():
    claim = DimensionClaim(state="filled", candidate_concept="collateral_geography",
                           alternate_concepts=("geographic_region_obligor",))
    assert claim.candidate_concepts == ("collateral_geography",
                                        "geographic_region_obligor")
    # The primary is never duplicated into the alternates.
    assert DimensionClaim(state="filled", candidate_concept="a",
                          alternate_concepts=("a", "b")).candidate_concepts == ("a", "b")


# --------------------------------------------------------------------------- #
# The projection — values, from the existing owner
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("question,direction,basis,limit", [
    ("Which region has the largest balance?", ORDER_INCREASE, ORDER_BASIS_ABSOLUTE, None),
    ("Which three regions grew the most since last month?", ORDER_INCREASE, ORDER_BASIS_ABSOLUTE, 3),
    ("Which region declined the most since last month?", ORDER_DECREASE, ORDER_BASIS_ABSOLUTE, None),
    ("top 5 brokers by balance", ORDER_INCREASE, ORDER_BASIS_ABSOLUTE, 5),
    ("Which region added the most loans since last month?", ORDER_INCREASE, ORDER_BASIS_COUNT, None),
    ("Which region increased its share the most since last month?", ORDER_INCREASE, ORDER_BASIS_SHARE, None),
])
def test_the_contract_carries_the_ordering_values(question, direction, basis, limit):
    op = _project(question).operation
    assert op.type == RANKING, question
    assert op.ordering_direction == direction, question
    assert op.ordering_basis == basis, question
    assert op.ordering_limit == limit, question


def test_the_contract_agrees_with_the_owner_it_reads():
    """Not a second reading: the same call, so the two cannot drift."""
    from mi_agent.period_change import rank_request as rank

    question = "Which three regions grew the most since last month?"
    op = _project(question).operation
    owner = rank.detect_rank_request(question, "region")
    assert owner is not None
    assert op.ordering_limit == owner.top_n
    assert op.ordering_direction == owner.direction  # both spell increase the same


def test_the_alternates_the_resolver_found_are_carried():
    """D1's missing fact. `requested_dimension_terms` resolves "region" to a
    primary plus alternates; the contract used to drop the alternates, so a book
    carrying one of them was told it carried none."""
    dims = _project("Which region grew the most?").dimensions
    assert dims, "no dimension claim was raised at all"
    claim = dims[0]
    assert claim.candidate_concept == "collateral_geography"
    assert "geographic_region_obligor" in claim.alternate_concepts
    assert "geographic_region_obligor" in claim.candidate_concepts


# --------------------------------------------------------------------------- #
# Level versus movement
# --------------------------------------------------------------------------- #
def test_a_ranking_of_a_movement_says_so():
    qi = _project("Compare October and November: which region grew the most?")
    assert qi.operation.type == RANKING
    assert qi.operation.ordering_of == ORDER_OF_MOVEMENT
    assert qi.operation.orders_a_movement


def test_a_ranking_of_a_level_does_not_claim_to_be_a_movement():
    op = _project("Which region has the largest balance?").operation
    assert op.type == RANKING
    assert op.orders_a_movement is False


def test_the_gap_that_was_pinned_here_is_now_CLOSED():
    """This test used to assert `ordering_of is None`, and said so on purpose.

    "Which region grew the most balance since last month?" is a ranked movement
    to any reader, and the contract could not say so because no owner told it
    the question was temporal — `_compare_recognizer`'s trigger vocabulary did
    not fire on "grew ... since". The assertion pinned the GAP and carried the
    instruction to flip it, with the movement attributed, on the day an owner
    appeared.

    `question_interpretation.lexical.temporal_aspect` is that owner. The
    movement is attributed in the canary bank's authorised_movements ledger as
    M2 and measured in docs/mi_temporal_aspect_owner.md: 113 contracts changed,
    2 plans changed on one route, 4 answers moved and none into a wrong answer.

    The old assertion is NOT deleted quietly — it is inverted, and this
    docstring is the record of what it used to say.
    """
    op = _project("Which region grew the most balance since last month?").operation
    assert op.type == RANKING
    assert op.ordering_of == ORDER_OF_MOVEMENT
    assert op.orders_a_movement
    # And the level sibling must NOT have moved with it.
    level = _project("Which region has the largest balance?").operation
    assert level.orders_a_movement is False
