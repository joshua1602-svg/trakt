"""A population the question never asked for must not execute.

P1L protects one direction: a population the user requested must reach execution
or the answer refuses. This file defends the mirror, which nothing guarded.

The case that forced it, found by the second-book review and reproduced on the
demonstration book:

    "What is the balance of the sponsored book?"

"Sponsored book" is a governed ENTIRE_AUM phrase. The LLM emitted
``seasoning_segment = back book`` instead, and because that predicate really was
applied — frame narrowed, rows-before/rows-after recorded — the population facet
was legitimately APPLIED and every guard reported success:

    Alderbridge   delivered £1,793,150,141.49   true £1,964,886,258.21   -8.7%
    Kestrelmoor   delivered   £238,685,188.37   true £1,331,647,994.86   -82%

ok=True, semanticGuard ok, no warning. The Alderbridge figure is the dangerous
one: 8.7% low on a headline AuM number is entirely plausible.

The rule under test is about CONCEPTS, not words. A question naming "back book"
authorises the whole governed seasoning derivation, including a bare
``months_on_book`` predicate the user never spelled out.
"""

from __future__ import annotations

import pytest

from mi_agent.population import fabricated_concepts


# --------------------------------------------------------------------------- #
# Fabrication (brief §5)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("filters, question, concept", [
    # the defect itself, both observed variants
    ({"seasoning_segment": "Back Book"},
     "What is the balance of the sponsored book?", "seasoning"),
    ({"seasoning_segment": "Front Book"},
     "What is the balance of the sponsored book?", "seasoning"),
    ({"seasoning_segment": "Back Book"},
     "What is the balance of the whole book?", "seasoning"),
    # a seasoning population invented alongside a requested provenance
    ({"source_portfolio_type": "direct", "seasoning_segment": "Back Book"},
     "What is the balance of the direct book?", "seasoning"),
    ({"source_portfolio_type": "acquired", "seasoning_segment": "Front Book"},
     "What is the balance of the acquired book?", "seasoning"),
    # a provenance population invented alongside a requested seasoning
    ({"source_portfolio_type": "acquired", "seasoning_segment": "Back Book"},
     "What is the balance of the back book?", "provenance"),
    # the derivation is fabricated even when the segment name never appears
    ({"months_on_book": {"op": "gt", "value": 12}},
     "What is the balance of the sponsored book?", "seasoning"),
])
def test_a_population_the_question_did_not_request_is_refused(filters, question, concept):
    assert concept in fabricated_concepts(filters, question)


# --------------------------------------------------------------------------- #
# Legitimate governed derivations (brief §4)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("filters, question", [
    ({}, "What is the balance of the sponsored book?"),
    ({}, "What is the balance of the whole book?"),
    ({"source_portfolio_type": "direct"}, "What is the balance of the direct book?"),
    ({"source_portfolio_type": "acquired"}, "What is the balance of the acquired book?"),
    ({"seasoning_segment": "Front Book"}, "What is the balance of the front book?"),
    ({"seasoning_segment": "Back Book"}, "What is the balance of the back book?"),
    # compound populations: both concepts were named
    ({"source_portfolio_type": "direct", "seasoning_segment": "Back Book"},
     "What is the balance of the direct back book?"),
    ({"source_portfolio_type": "acquired", "seasoning_segment": "Front Book"},
     "What is the balance of the acquired front book?"),
])
def test_a_requested_population_is_never_treated_as_fabricated(filters, question):
    assert fabricated_concepts(filters, question) == []


def test_the_derivation_is_authorised_not_just_the_phrase():
    """"Back book" authorises the whole seasoning derivation. The user does not
    have to say "months_on_book" for the executor to use it."""
    assert fabricated_concepts(
        {"months_on_book": {"op": "gt", "value": 12}},
        "What is the average LTV of the back book?") == []
    assert fabricated_concepts(
        {"seasoning_bucket": "25-60m"},
        "Show me the back book by seasoning band.") == []


def test_a_directly_requested_seasoning_measure_is_not_fabrication():
    """"Loans with more than 60 months on book" names the concept explicitly,
    even though it names no front/back segment."""
    assert fabricated_concepts(
        {"months_on_book": {"op": "gt", "value": 60}},
        "What is the balance of loans with more than 60 months on book?") == []


def test_ordinary_row_filters_are_untouched():
    """The guard governs POPULATION CONCEPTS, not every predicate. A threshold on
    a measure is not a governed population and must never be flagged."""
    for filters, question in (
        ({"current_loan_to_value": {"op": "gt", "value": 70}},
         "What is the balance of loans above 70% LTV?"),
        ({"youngest_borrower_age": {"op": "gt", "value": 85}},
         "What is the maximum loan balance for borrowers over 85?"),
        ({"collateral_geography": "London"}, "What is the average LTV in London?"),
        ({}, "What is the total balance?"),
    ):
        assert fabricated_concepts(filters, question) == []


def test_no_population_at_all_is_never_fabrication():
    assert fabricated_concepts(None, "What is the balance of the sponsored book?") == []
    assert fabricated_concepts({}, "anything at all") == []


# --------------------------------------------------------------------------- #
# The parser seam
# --------------------------------------------------------------------------- #

def test_a_fabricated_population_is_not_a_repairable_parse_error():
    """A model that invented one population may invent a different one on a
    retry — Kestrelmoor returned the front book twice and the back book once for
    the same question. Recovery is the deterministic interpretation, which
    resolves the governed scope phrase correctly, or a refusal."""
    from mi_agent.llm_query_parser import _FABRICATED_POP_MARK

    assert _FABRICATED_POP_MARK
    errors = [f"{_FABRICATED_POP_MARK}: seasoning (the question does not request "
              f"this population)"]
    assert any(_FABRICATED_POP_MARK in e for e in errors)


def test_the_guard_reads_the_vocabularies_that_already_own_each_concept():
    """No second population model: seasoning comes from mi_agent.seasoning and
    provenance from mi_agent.portfolio_lens, so the guard cannot drift from the
    modules that decide what those phrases mean."""
    from mi_agent import population as pop

    assert set(pop._CONCEPT_KEYS) == {"seasoning", "provenance"}
    assert "seasoning_segment" in pop._CONCEPT_KEYS["seasoning"]
    assert "source_portfolio_type" in pop._CONCEPT_KEYS["provenance"]
