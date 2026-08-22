"""One owner decides the seasoning role. Every other reader consumes it.

Three defects, one cause. The role of a seasoning phrase — population or axis —
was decided by three readers disagreeing on vocabulary, on when they ran, and on
whether they consulted the book's columns:

  B13  "new lending" is a governed window, not a segment phrase, so the parser's
       resolver declined; the layer that owns lending windows runs only for
       composite plans; the dimension reader fired and "what is the balance of
       new lending?" answered over all 11,035 loans.
  B15  "back book" IS a segment phrase, so the parser resolved it AND stripped
       the spec's dimension slots — but the dimension reader reads the QUESTION
       TEXT, so one receipt carried the field as both an applied population and
       a lost grouping.

The owner takes the lending windows' vocabulary and the parser resolver's
discipline: every parse, columns consulted, one window selects and two compare.
"""
from __future__ import annotations

import pytest

from mi_agent import execution_receipt as R
from mi_agent.seasoning import resolve_population_predicate


# --------------------------------------------------------------------------- #
# The owner
# --------------------------------------------------------------------------- #
def test_a_window_with_no_segment_phrase_still_selects_a_population():
    """B13's core. `_SEGMENT_PHRASES` cannot express this at all."""
    assert resolve_population_predicate("What is the balance of new lending?") == {
        "months_on_book": {"op": "le", "value": 1}}


def test_a_segment_phrase_selects_its_segment():
    assert resolve_population_predicate("balance of the front book") == {
        "seasoning_segment": "Front Book"}


def test_naming_two_windows_is_a_comparison_and_selects_nothing():
    """Narrowing to one side of a comparison answers a different question."""
    assert resolve_population_predicate(
        "How does the front book compare with our older lending?") is None
    assert resolve_population_predicate(
        "front book versus back book") is None


def test_naming_no_window_selects_nothing():
    """Can-fail: an owner that always selected would pass every test above."""
    assert resolve_population_predicate("balance by region") is None
    assert resolve_population_predicate("") is None


def test_a_book_without_the_column_selects_nothing():
    """Reader 1's discipline, kept. A book that cannot express the predicate
    must not acquire one."""
    assert resolve_population_predicate(
        "balance of new lending", columns=["seasoning_segment"]) is None
    assert resolve_population_predicate(
        "balance of the front book", columns=["current_outstanding_balance"]) is None
    assert resolve_population_predicate(
        "balance of the front book", columns=["seasoning_segment"]) == {
            "seasoning_segment": "Front Book"}


# --------------------------------------------------------------------------- #
# Reader 2 consumes rather than derives
# --------------------------------------------------------------------------- #
def _semantics():
    from pathlib import Path

    from mi_agent.mi_query_validator import load_mi_semantics
    return load_mi_semantics(
        Path(__file__).resolve().parents[2] / "mi_agent"
        / "mi_semantics_field_registry.yaml")


@pytest.fixture(scope="module")
def semantics():
    return _semantics()


def test_no_seasoning_dimension_is_raised_where_the_owner_took_a_population(semantics):
    """B15. The receipt must not carry the field as a grouping as well."""
    for question in ("What is the balance of the front book?",
                     "Show the back book balance by month.",
                     "What is the balance of new lending?"):
        terms = R.requested_dimension_terms(question, semantics, None)
        assert not [t for t in terms if t[0] == "seasoning_segment"], question


def test_the_seasoning_dimension_still_stands_for_a_comparison(semantics):
    """Can-fail, and the reason the suppression is conditional.

    When both sides are named the grouping is what makes the question
    answerable, so it must survive. A blanket suppression would pass the test
    above and destroy every seasoning comparison — which is the shape of the
    160-run regression.
    """
    terms = R.requested_dimension_terms(
        "How different is the risk profile of recent originations versus the "
        "back book?", semantics, None)
    assert [t for t in terms if t[0] == "seasoning_segment"]


def test_a_non_seasoning_dimension_is_untouched(semantics):
    terms = R.requested_dimension_terms("balance by region", semantics, None)
    assert [t for t in terms if t[0] == "collateral_geography"]


# --------------------------------------------------------------------------- #
# The decision becomes a facet, rather than vanishing
# --------------------------------------------------------------------------- #
def test_the_population_is_raised_as_a_facet(semantics):
    """Suppressing the grouping and raising nothing would answer correctly and
    record nothing — a wrong claim traded for no claim."""
    facets = R.detect_requested_facets(
        "What is the balance of new lending?", semantics, frame=None,
        requested_dimensions=[])
    populations = [f for f in facets if f.kind == R.KIND_POPULATION]
    assert [f.label for f in populations] == ["the population months_on_book le 1"]


def test_the_facet_names_the_governed_predicate_not_a_segment(semantics):
    """"New lending" is a months-on-book bound. A receipt naming a segment the
    question never mentioned would be describing a different population."""
    facets = R.detect_requested_facets(
        "What is the balance of new lending?", semantics, frame=None,
        requested_dimensions=[])
    labels = " ".join(f.label for f in facets)
    assert "months_on_book" in labels
    assert "seasoning_segment" not in labels


def test_no_population_facet_without_a_window(semantics):
    """Can-fail: raising one unconditionally would pass both tests above."""
    facets = R.detect_requested_facets(
        "balance by region", semantics, frame=None, requested_dimensions=[])
    assert not [f for f in facets if f.kind == R.KIND_POPULATION]
