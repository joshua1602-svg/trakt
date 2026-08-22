"""Item 3 — one owner for "which noun is this threshold on".

TWO OWNERS with near-identical vocabularies and different rules:

  llm_query_parser._filter_field_of     -> a field key, by the subject NEAREST
                                           BEFORE the comparator
  execution_receipt._threshold_subject  -> a display name, by the FIRST entry of
                                           a fixed priority list

They disagreed whenever a measure was named earlier in the sentence than the
threshold's own noun, so "what is the LTV for loans with a balance above
£150,000" disclosed "LTV over 150000" while execution filtered the balance —
the receipt naming a field execution had not touched.

And the same decision, never asked, is why B1 refused: `ticket` names a registry
dimension, `dimension_role` had no source for "this word is the subject of a
threshold", and the role fell to UNRESOLVED over a question whose measure, filter
and 5,857-loan population were all resolved.
"""

from __future__ import annotations

import re

import pytest

from mi_agent import execution_receipt as R
from mi_agent import llm_query_parser as P
from mi_agent.mi_query_validator import load_mi_semantics
from question_interpretation import lexical as L


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics("mi_agent/mi_semantics_field_registry.yaml")


# --------------------------------------------------------------------------- #
# The receipt names the field execution binds
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("question, field", [
    ("what is the ltv for loans with a balance above 150000",
     "current_outstanding_balance"),
    ("what is the ltv for loan tickets above 150000",
     "current_outstanding_balance"),
    ("what is the ltv for borrowers over 75", "youngest_borrower_age"),
    ("what is the balance for loans with an interest rate above 5",
     "current_interest_rate"),
    ("show me loans with ltv above 50", "current_loan_to_value"),
])
def test_the_label_names_the_field_execution_binds(question, field, semantics):
    """Three of eight probed sentences used to disclose a threshold on a field
    execution had not filtered. The receipt and the parser now read one owner
    and one rule."""
    bound = P._parse_filters(question, semantics)
    assert field in bound, "premise: execution binds %s" % field

    facets = R._detect_thresholds(question)
    assert facets, "a threshold was stated"
    kind = L.threshold_subject_kind(question)
    assert facets[0].label.lower().startswith(
        L.THRESHOLD_SUBJECT_WORD[kind].lower())
    assert field in R._SUBJECT_KIND_FIELDS[kind]


def test_a_postfix_subject_binds_tightest():
    """"above 50% LTV" names its subject AFTER the value. The first version of
    the owner implemented only the nearest-BEFORE half of `_filter_field_of`'s
    rule and silently dropped the subject here — the label went from
    "LTV over 50" to "over 50". ONE CORPUS ANSWER MOVED and that is how it was
    found: prefix-only was not the full range of the vocabulary."""
    q = "what percentage of the book is above 50% ltv"
    facets = R._detect_thresholds(q)
    assert facets and facets[0].label == "LTV over 50"


def test_the_rule_is_proximity_not_priority():
    """The correction. An ordered list returned LTV for a balance threshold
    because LTV came first and was named earlier in the sentence."""
    q = "what is the ltv for loans with a balance above 150000"
    assert L.threshold_subject_kind(q, anchor=q.index("above")) == "balance"
    # and the reverse sentence resolves the other way, so it is proximity and
    # not a new fixed order
    q2 = "what is the balance for loans with an ltv above 50"
    assert L.threshold_subject_kind(q2, anchor=q2.index("above")) == "ltv"


def test_neither_consumer_keeps_its_own_subject_vocabulary():
    assert P._FILTER_SUBJECT_PATTERNS is L.THRESHOLD_SUBJECT_PATTERNS
    assert not hasattr(R, "_THRESHOLD_SUBJECTS")


# --------------------------------------------------------------------------- #
# Source 5
# --------------------------------------------------------------------------- #
#: The label is the word the SENTENCE used — "ticket", not the registry's
#: "ticket size". The first version of these tests invented "ticket size", which
#: `_named_term_span` cannot find in the question, so source 5 correctly
#: declined and the test failed while the end-to-end path worked. A unit test
#: whose fixture does not match the data the function receives measures nothing.
def _facet(label, keys):
    f = R.RequestedFacet(kind=R.KIND_GROUPING, label=label)
    f.satisfied_by = lambda: list(keys)   # type: ignore[assignment]
    return f


def test_a_word_naming_a_threshold_subject_is_a_filter():
    """B1's cause. The facet's own keys are the BUCKET; the applied filter is on
    the balance FIELD, so source 1 can never reach it."""
    role, key = R.dimension_role(
        _facet("ticket", ["ticket_bucket"]),
        question="what is the ltv for loan tickets above 150000",
        filters={"current_outstanding_balance": {"op": "gt", "value": 150000}},
        axes=set())
    assert (role, key) == (R.ROLE_FILTER, "current_outstanding_balance")


def test_source_5_cannot_invent_a_filter():
    """THE CAN-FAIL, and the reason source 5 is safe. It reads `filters`; it
    never writes them. With no applied filter of that kind the role is
    unchanged — so this can only EXPLAIN a narrowing execution performed."""
    role, key = R.dimension_role(
        _facet("ticket", ["ticket_bucket"]),
        question="what is the ltv for loan tickets above 150000",
        filters={}, axes=set())
    assert role == R.ROLE_UNRESOLVED and key is None


def test_a_dimension_merely_mentioned_is_not_claimed():
    """THE SECOND CAN-FAIL. `region` is nowhere near the comparator, so the
    owner's nearest-before rule does not name it and source 5 declines."""
    role, key = R.dimension_role(
        _facet("region", ["collateral_geography"]),
        question="what is the ltv by region for loans above 150000",
        filters={"current_outstanding_balance": {"op": "gt", "value": 150000}},
        axes=set())
    assert key != "current_outstanding_balance"


def test_source_5_runs_last():
    """Sources 1-4 keep precedence: a word reader 1 already placed on an axis is
    an axis, even where it also sits beside a comparator."""
    role, _ = R.dimension_role(
        _facet("ticket", ["ticket_bucket"]),
        question="what is the ltv by ticket above 150000",
        filters={"current_outstanding_balance": {"op": "gt", "value": 150000}},
        axes={"ticket_bucket"})
    assert role == R.ROLE_AXIS


# --------------------------------------------------------------------------- #
# The separation item 1 earned is untouched
# --------------------------------------------------------------------------- #
def test_the_detectors_still_stay_separate():
    """Re-asserted here because item 3 touched `_detect_thresholds`. The owner
    is a TEXT rule needing no semantics, no spec and no applied filters, so the
    receipt still raises a threshold the parser misses — which is what makes the
    guard refuse rather than answer over the whole book."""
    import inspect
    assert list(inspect.signature(R._detect_thresholds).parameters) == ["q"]
