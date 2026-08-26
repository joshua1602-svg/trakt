"""A governed row predicate must coexist with ranked historical movement.

THE DEFECT THIS PINS. `_compare_recognizer` short-circuited the deterministic
parse: it built its spec directly, never called `_parse_filters`, and ran
`_detect_metric` over the whole question. A comparison carrying a predicate
therefore lost the predicate AND took its field as the measure —

    "Which region added the most balance since last month
     for loans with LTV above 50%?"
        metric  = current_loan_to_value      (should be the balance)
        filters = {}                         (should carry LTV > 50)

The trigger was the TEMPORAL clause, not the ranking: removing "since last
month" fixed it and removing the ranking did not. The field family was
irrelevant — borrower age failed identically.

These tests prove COEXISTENCE, not individual recognition. Each asserts every
channel at once, because the defect was precisely that one channel stole
another's content while both looked individually plausible.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def semantics():
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent_api.data_source import semantics_path
    return load_mi_semantics(semantics_path())


COLUMNS = ["loan_identifier", "current_outstanding_balance",
           "current_loan_to_value", "current_interest_rate",
           "youngest_borrower_age", "broker_channel",
           "geographic_region_obligor", "reporting_date"]


def _contract(question, semantics):
    from mi_agent import execution_receipt as R
    from mi_agent.parsed_question import ParsedQuestion
    from question_interpretation import projection as proj
    spec = ParsedQuestion.parse(question, semantics).spec
    terms = R.requested_dimension_terms(question, semantics, COLUMNS)
    facets = R.detect_requested_facets(question, semantics, frame=None,
                                       requested_dimensions=terms)
    return spec, proj.from_parts(question, spec=spec, facets=facets,
                                 dim_terms=terms, semantics=semantics)


def _preds(qi):
    return {p.field_key: (p.operator, p.value) for p in (qi.row_predicates or ())}


# --------------------------------------------------------------------------- #
# RM-F1 — the defect case, every channel at once
# --------------------------------------------------------------------------- #
def test_rmf1_every_channel_survives_together(semantics):
    q = ("Which region added the most balance since last month "
         "for loans with LTV above 50%?")
    spec, qi = _contract(q, semantics)
    # measure — NOT the filter's field
    assert qi.subject.candidate_concept == "current_outstanding_balance"
    # grouping dimension, with the alternate the book may actually carry
    assert [d.candidate_concept for d in qi.dimensions] == ["collateral_geography"]
    assert "geographic_region_obligor" in qi.dimensions[0].alternate_concepts
    # temporal aspect and the comparison itself
    assert qi.operation.ordering_of == "movement"
    assert list(qi.time.comparison_periods) == ["last month", "latest"]
    # ordering
    assert qi.operation.type == "ranking"
    assert qi.operation.ordering_direction == "increase"
    assert qi.operation.ordering_basis == "absolute"
    # the predicate, which used to vanish
    assert _preds(qi) == {"current_loan_to_value": ("gt", 50.0)}


# --------------------------------------------------------------------------- #
# RM-F2 — a different governed predicate family. The fix must be generic.
# --------------------------------------------------------------------------- #
def test_rmf2_a_different_predicate_family_behaves_identically(semantics):
    q = ("Which region added the most balance since last month "
         "for loans with borrower age above 70?")
    spec, qi = _contract(q, semantics)
    assert qi.subject.candidate_concept == "current_outstanding_balance"
    assert qi.operation.ordering_of == "movement"
    assert _preds(qi) == {"youngest_borrower_age": ("gt", 70.0)}


# --------------------------------------------------------------------------- #
# RM-F3 — the fix must not depend on ranking.
# --------------------------------------------------------------------------- #
def test_rmf3_filtered_unranked_movement(semantics):
    q = "How did balance change since last month for loans with LTV above 50%?"
    spec, qi = _contract(q, semantics)
    assert qi.subject.candidate_concept == "current_outstanding_balance"
    assert list(qi.time.comparison_periods) == ["last month", "latest"]
    assert _preds(qi) == {"current_loan_to_value": ("gt", 50.0)}
    assert qi.operation.type != "ranking"


# --------------------------------------------------------------------------- #
# RM-F4 — unfiltered ranked movement must be untouched.
# --------------------------------------------------------------------------- #
def test_rmf4_unfiltered_ranked_movement_is_unchanged(semantics):
    q = "Which region added the most balance since last month?"
    spec, qi = _contract(q, semantics)
    assert qi.subject.candidate_concept == "current_outstanding_balance"
    assert qi.operation.ordering_of == "movement"
    assert qi.operation.ordering_direction == "increase"
    assert _preds(qi) == {}


# --------------------------------------------------------------------------- #
# NEGATIVE CONTROLS — the fix must not demote every mention to a predicate.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("question", [
    "How did LTV change since last month?",
    "Which region has the highest average LTV?",
])
def test_ltv_stays_the_measure_when_it_is_genuinely_requested(question, semantics):
    spec, qi = _contract(question, semantics)
    assert qi.subject.candidate_concept == "current_loan_to_value", question
    assert _preds(qi) == {}, question


def test_the_load_bearing_bare_threshold_is_preserved(question=None, *,
                                                      semantics=None):
    """"balance ... over 50" binds the threshold to the MEASURE's own field.

    Prior work established this and it is load-bearing; the fix must not move
    it, because the masking only removes clauses `_parse_filters` itself
    resolved.
    """
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent_api.data_source import semantics_path
    sem = semantics or load_mi_semantics(semantics_path())
    spec, qi = _contract("balance by region for loans over 50", sem)
    assert spec.metric == "current_outstanding_balance"
    assert _preds(qi) == {"current_outstanding_balance": ("gt", 50.0)}
