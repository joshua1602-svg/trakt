"""The FIELD-and-BOUND guard: a filter the question does not support.

``fabricated_concepts`` guards two governed population concepts. This guards
everything else a filter can invent — a field the registry does not carry, and a
scalar bound the question never states.

The false-rejection half matters as much as the rejection half. A parser
normalises what it reads: "50%" may be held as 50 or as 0.5, "£1.5m" as
1500000, "£250,000" as 250000. A guard that compared the held value against the
question's characters would reject those legitimate filters, and rejecting a
real filter is worse than the defect it was built to catch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from mi_agent.mi_query_validator import load_mi_semantics
from mi_agent.population import fabricated_bounds

SEMANTICS = load_mi_semantics(_REPO / "mi_agent" / "mi_semantics_field_registry.yaml")


# --------------------------------------------------------------------------- #
# Rejected: the question does not support the filter
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("filters,question,expected_fragment", [
    # The model invented a date for a question that names none. This is the
    # shape behind the parser's run-to-run disagreement on Q1.3.
    ({"origination_date": {"op": "ge", "value": "2024-01-01"}},
     "How does recent lending compare with what we were originating earlier in the year?",
     "origination_date"),
    # A template token that reached the output unfilled — not a bound, the NAME
    # of one. Reported with its own reason so it is not read as an invented
    # number: the cause is different and so is the fix.
    ({"origination_date": {"op": "ge", "value": "LAST_4_WEEKS"}},
     "Based on the last few weeks, what level of completions are we achieving?",
     "unresolved placeholder"),
    # A real field, an invented bound: nothing in the question says zero.
    ({"arrears_balance": {"op": "gt", "value": 0}},
     "Which of our limits are currently most at risk?",
     "arrears_balance"),
    # A field the registry does not carry. The VALUE half alone would let this
    # through, because 50 does appear in the question — which is exactly why
    # the guard checks both halves.
    ({"unicorn_score": {"op": "gt", "value": 50}},
     "balance where LTV above 50%",
     "not a governed field"),
])
def test_unsupported_filters_are_rejected(filters, question, expected_fragment):
    rejected = fabricated_bounds(filters, question, SEMANTICS)
    assert rejected, f"{filters} should not survive {question!r}"
    assert any(expected_fragment in r for r in rejected), rejected


# --------------------------------------------------------------------------- #
# Kept: the question DOES support the filter, however the parser normalised it
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("filters,question", [
    # A percentage held in POINTS, and the same bound held as a FRACTION.
    ({"current_loan_to_value": {"op": "gt", "value": 50.0}},
     "balance where LTV above 50%"),
    ({"current_loan_to_value": {"op": "gt", "value": 0.5}},
     "balance where LTV above 50%"),
    # Magnitude suffixes.
    ({"current_outstanding_balance": {"op": "gt", "value": 1_500_000.0}},
     "loans over £1.5m"),
    ({"current_outstanding_balance": {"op": "gt", "value": 500_000.0}},
     "balance for loans over 500k"),
    # Thousands separators, including more than one.
    ({"current_outstanding_balance": {"op": "gt", "value": 250_000.0}},
     "loans above £250,000"),
    ({"current_outstanding_balance": {"op": "gt", "value": 1_500_000.0}},
     "loans above £1,500,000"),
    # A plain integer bound.
    ({"youngest_borrower_age": {"op": "gt", "value": 80}},
     "balance for borrowers over 80"),
    # Categorical values are names, not bounds, and are not this guard's business.
    ({"pipeline_stage": "Offer"},
     "Of the current offer pipeline, how much should convert?"),
    ({"geographic_region_obligor": "South East"},
     "exposure to the South East"),
    # A boolean flag carries no bound to fabricate.
    ({"loan_redemption_flag": True},
     "what completion rate are we running at"),
])
def test_supported_filters_are_kept(filters, question):
    assert fabricated_bounds(filters, question, SEMANTICS) == [], (
        f"{filters} is supported by {question!r} and must not be rejected")


def test_no_filters_is_not_a_rejection():
    assert fabricated_bounds(None, "what is the total balance", SEMANTICS) == []
    assert fabricated_bounds({}, "what is the total balance", SEMANTICS) == []


def test_without_semantics_the_field_half_is_skipped_not_guessed():
    """No registry supplied means the field cannot be checked — so it is not.

    The value half still applies. Silently rejecting every field when the
    registry is absent would turn a missing input into a refusal.
    """
    filters = {"unicorn_score": {"op": "gt", "value": 50}}
    assert fabricated_bounds(filters, "balance where LTV above 50%") == []
    assert fabricated_bounds({"unicorn_score": {"op": "gt", "value": 7}},
                             "balance where LTV above 50%") != []
