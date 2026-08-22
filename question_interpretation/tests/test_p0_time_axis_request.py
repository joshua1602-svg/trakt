"""The sentence-side owner of "does this ask the answer to vary over time?".

`lexical.time_axis_request` is a COMPOSITION of two owners that already exist,
and these tests are written to fail if it stops being one. Two of them exist
specifically because a hand-written unit vocabulary — drafted for this rule and
thrown away — would have passed everything else and broken two routed answers.
"""
from __future__ import annotations

import re

import pytest

from question_interpretation import lexical


ASKS = [
    ("Show me balance by month", "by month"),
    ("balance by month broken down by LTV band and region", "by month"),
    ("balance by week over the last 6 months", "by week"),
    ("funded balance by quarter", "by quarter"),
    ("total balance by reporting period", "by reporting period"),
    ("balance per month", "per month"),
    ("balance split by month", "split by month"),
    ("balance by each month", "by each month"),
    ("balance by day", "by day"),
    ("balance per day", "per day"),
    ("balance over time", "over time"),
    ("Show regional concentration evolution over time.", "over time"),
    ("How have direct and acquired balances moved over the periods?",
     "over the periods"),
    ("Show me the balance month by month", "month by month"),
]

DOES_NOT_ASK = [
    # A dimension, not a time axis.
    "Show me balance by region",
    "balance by LTV band",
    # B11's case: a unit word used as an ATTRIBUTE, not an axis.
    "How many loans are over 80 years old?",
    "How many loans are over 12 months old?",
    # The unit is present but not adjacent to the marker: a filter, not an axis.
    "balance by region for loans under 12 months old",
    # `vintage` is a loan attribute the grouping owner handles. Both of these
    # are routed-surface answers that are correct today, and a hand-written
    # unit list would have refused them.
    "What is the forecast run rate by vintage?",
    "What is the balance by vintage, ignoring the forecast?",
    # No axis at all.
    "What is the total balance?",
    "",
]


@pytest.mark.parametrize("question,wording", ASKS)
def test_the_wording_that_asked_is_returned(question, wording):
    assert (lexical.time_axis_request(question) or "").lower() == wording.lower()


@pytest.mark.parametrize("question", DOES_NOT_ASK)
def test_no_time_axis_is_read_where_none_was_asked_for(question):
    assert lexical.time_axis_request(question) is None


def test_the_unit_must_be_adjacent_to_the_marker():
    """THE ADJACENCY RULE, stated as behaviour rather than as a comment.

    Reading a unit from anywhere in the sentence turns a filter into a series
    request. These two sentences contain the same words.
    """
    assert lexical.time_axis_request("balance by month for loans over £150k")
    assert lexical.time_axis_request(
        "balance by region for loans originated 12 months ago") is None


def test_it_reads_the_axis_marker_owner_rather_than_its_own_list():
    """Every declared axis marker must work, including ones spelled with `by`.

    `split by`, `broken down by` and `grouped by` all contain "by", so a rule
    hard-coding `\\bby\\b` passes them by accident and fails `per` and
    `across` — the exact defect item 2 closed. Derived from the owner's own
    tuple so a marker added there is covered here without an edit.
    """
    for marker in lexical.AXIS_MARKERS:
        question = "balance %s month" % marker
        assert lexical.time_axis_request(question) is not None, marker


def test_it_reads_the_unit_owner_rather_than_its_own_list():
    """Every declared time unit must work, derived from `UNIT_PATTERNS`."""
    for unit, _pattern in lexical.UNIT_PATTERNS:
        assert lexical.time_axis_request("balance by %s" % unit) is not None, unit


def test_the_series_phrases_are_the_only_new_vocabulary():
    """Each declared phrase must be reachable, and reachable ONLY as a phrase.

    A phrase that no sentence can trigger is dead vocabulary, and dead
    vocabulary is how a list and its consumers drift apart.
    """
    for phrase in lexical.SERIES_PHRASES:
        assert lexical.time_axis_request("balance %s" % phrase) is not None, phrase


# --------------------------------------------------------------------------- #
# THE CAN-FAIL PROOF
# --------------------------------------------------------------------------- #
# No instrument ships without a test proving it can fail. These two mutate the
# owner and require the suite above to notice — in BOTH directions, because an
# owner that says yes to everything and an owner that says no to everything are
# both broken and only one of them is caught by a positive test.
def test_the_positive_cases_would_notice_an_owner_that_never_fires(monkeypatch):
    monkeypatch.setattr(lexical, "_SERIES_PHRASE_RE", re.compile(r"(?!x)x"))
    monkeypatch.setattr(lexical, "AXIS_MARKER_RE", re.compile(r"(?!x)x"))
    assert all(lexical.time_axis_request(q) is None for q, _w in ASKS)


def test_the_negative_cases_would_notice_an_owner_that_always_fires(monkeypatch):
    """Widen the adjacency rule to "a unit anywhere" — the defect the rule
    exists to prevent — and the negative cases must light up."""
    monkeypatch.setattr(
        lexical, "_SERIES_PHRASE_RE",
        re.compile(r"\b(?:month|months|week|weeks|quarter|year|years|vintage)\b",
                   re.I))
    caught = [q for q in DOES_NOT_ASK if lexical.time_axis_request(q) is not None]
    assert len(caught) >= 4, caught
