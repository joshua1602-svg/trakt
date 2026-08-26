"""A filter stated FIRST must mean what the same filter stated LAST means.

THE DEFECT THIS PINS. `condition_cut` reports where a condition clause BEGINS,
which is all a caller needs when the condition is stated last: everything
before it is the subject. Stated first it is not enough — the subject is what
FOLLOWS the condition — and `metric_slot` truncated at the start, leaving
nothing, and then handed the whole question to the metric detector.

    "For loans with LTV above 50%, balance by region"
        metric  = current_loan_to_value            (should be the balance)
        filter  = current_outstanding_balance > 50 (should be LTV > 50)

Worse on a second field family, because the clause ran to the end of the
sentence and the bound was read from the swallowed words:

    "For loans with borrower age above 70, balance by region"
        filter  = current_outstanding_balance > 70,000,000,000

Two facts, both measured rather than assumed: the same questions with the
condition stated LAST parsed correctly, and no field name is involved in the
fix — punctuation was simply not a clause boundary, and a clause with no end
cannot be removed from the subject.
"""
from __future__ import annotations

import pytest

COLUMNS = ["loan_identifier", "current_outstanding_balance",
           "current_loan_to_value", "current_interest_rate",
           "youngest_borrower_age", "broker_channel",
           "geographic_region_obligor", "reporting_date"]

#: The channels that must agree between an equivalent leading and trailing form.
CHANNELS = ("measure", "dimensions", "predicates", "comparison", "operation",
            "ordering_of")


@pytest.fixture(scope="module")
def semantics():
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent_api.data_source import semantics_path
    return load_mi_semantics(semantics_path())


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


def _channels(question, semantics):
    spec, qi = _contract(question, semantics)
    return {
        "measure": qi.subject.candidate_concept,
        "dimensions": [d.candidate_concept for d in (qi.dimensions or [])],
        "predicates": {p.field_key: (p.operator, p.value)
                       for p in (qi.row_predicates or ())},
        "comparison": list(qi.time.comparison_periods or ()),
        "operation": qi.operation.type,
        "ordering_of": qi.operation.ordering_of,
    }


# --------------------------------------------------------------------------- #
# Leading / trailing equivalence, across field families and analysis shapes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("leading,trailing", [
    ("For loans with LTV above 50%, balance by region",
     "Balance by region for loans with LTV above 50%."),
    ("For loans with borrower age above 70, balance by region",
     "Balance by region for loans with borrower age above 70."),
    ("For loans with LTV above 50%, how did balance change since last month?",
     "How did balance change since last month for loans with LTV above 50%?"),
    ("For loans with LTV above 50%, which region added the most balance "
     "since last month?",
     "Which region added the most balance since last month for loans with "
     "LTV above 50%?"),
    ("For loans with LTV above 50%, how many loans do we have?",
     "How many loans do we have for loans with LTV above 50%?"),
])
def test_a_leading_filter_means_what_a_trailing_filter_means(leading, trailing,
                                                             semantics):
    lead = _channels(leading, semantics)
    trail = _channels(trailing, semantics)
    for channel in CHANNELS:
        assert lead[channel] == trail[channel], (
            f"{channel}: leading={lead[channel]!r} trailing={trail[channel]!r}")


def test_the_leading_form_binds_the_predicate_to_its_own_field(semantics):
    """The defect's sharpest edge: the bound landed on the wrong field."""
    channels = _channels("For loans with LTV above 50%, balance by region",
                         semantics)
    assert channels["measure"] == "current_outstanding_balance"
    assert channels["predicates"] == {"current_loan_to_value": ("gt", 50.0)}
    assert channels["dimensions"] == ["collateral_geography"]


def test_a_leading_bound_is_not_read_from_the_words_after_it(semantics):
    """"above 70, balance by region" produced a threshold of seventy billion."""
    channels = _channels("For loans with borrower age above 70, "
                         "balance by region", semantics)
    assert channels["predicates"] == {"youngest_borrower_age": ("gt", 70.0)}
    assert channels["measure"] == "current_outstanding_balance"


# --------------------------------------------------------------------------- #
# Controls — the fix must not move anything else
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("question,measure,predicates", [
    # An unfiltered dimension query.
    ("Balance by region.", "current_outstanding_balance", {}),
    # A field genuinely REQUESTED as the measure, not as a filter.
    ("Which region has the highest average LTV?", "current_loan_to_value", {}),
    ("How did LTV change since last month?", "current_loan_to_value", {}),
    # The load-bearing bare threshold: "balance ... over 50" binds to the
    # MEASURE's own field, and prior work established this is deliberate.
    ("balance by region for loans over 50", "current_outstanding_balance",
     {"current_outstanding_balance": ("gt", 50.0)}),
])
def test_unrelated_forms_are_unchanged(question, measure, predicates, semantics):
    channels = _channels(question, semantics)
    assert channels["measure"] == measure, question
    assert channels["predicates"] == predicates, question


def test_a_comma_inside_a_number_is_not_a_clause_boundary():
    """"over 1,500,000" is one number, not two clauses."""
    from question_interpretation.lexical import clause_spans
    text = "loans with balance over 1,500,000"
    spans = clause_spans(text)
    joined = [text[a:b] for a, b in spans]
    assert any("1,500,000" in part for part in joined), joined


def test_a_comma_between_words_is_a_clause_boundary():
    from question_interpretation.lexical import clause_spans
    text = "for loans with ltv above 50%, balance by region"
    parts = [text[a:b].strip() for a, b in clause_spans(text)]
    assert any(part.startswith(",") or part.lstrip(", ").startswith("balance")
               for part in parts), parts


def test_the_condition_span_ends_after_its_bound(semantics):
    """The span used to stop at the connective INSIDE its own opener."""
    from question_interpretation.lexical import condition_span
    text = "for loans with ltv above 50%, balance by region"
    span = condition_span(text)
    assert span is not None
    start, end = span
    clause = text[start:end]
    assert "ltv" in clause and "50" in clause, clause
    assert "balance" not in clause, clause
    assert "region" not in clause, clause
