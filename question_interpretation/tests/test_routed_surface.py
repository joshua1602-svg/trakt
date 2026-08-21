"""The routed surface, and the proof that it can fail.

Condition of its existence: it must not become a third frozen artefact with a
blindness of its own. Two of the three surfaces in this programme were clean
throughout defects they could not see, and in both cases the problem was not
that they were wrong — it was that their blindness went unwritten until a defect
walked through it.

So: the bank declares what it can and cannot see, this asserts that it does, and
the checker is proven able to catch every kind of expectation it makes.
"""
from __future__ import annotations

import pytest

from question_interpretation import routed_surface as RS


@pytest.fixture(scope="module")
def doc():
    return RS.bank()


# --------------------------------------------------------------------------- #
# It must be able to fail
# --------------------------------------------------------------------------- #
def test_the_self_test_passes():
    assert RS.self_test() == 0


@pytest.mark.parametrize("case,what", [
    ({"expect_route": "not_a_route"}, "route"),
    ({"expect_ok": False}, "ok"),
    ({"expect_verdict": "refuse"}, "verdict"),
    ({"expect_facets": [["grouping_dimension", "lost"]]}, "facets"),
    ({"expect_answer_contains": "absent phrase"}, "answer"),
])
def test_the_checker_catches_a_wrong_expectation(case, what):
    seen = {"route": "evolution", "ok": True, "verdict": "ok",
            "facets": [("granularity", "applied")], "answer": "a sentence"}
    assert RS.check(case, seen), "a wrong %s was not caught" % what


def test_the_checker_accepts_a_correct_expectation():
    """Otherwise it would 'catch' everything and assert nothing."""
    seen = {"route": "evolution", "ok": True, "verdict": "ok",
            "facets": [("granularity", "applied")], "answer": "a sentence"}
    assert RS.check({"expect_route": "evolution", "expect_verdict": "ok",
                     "expect_facets": [["granularity", "applied"]]}, seen) == []


def test_silence_is_not_an_assertion():
    """A case that omits `expect_facets` asserts nothing about facets.

    Pinning more than a case means to pin is how a surface becomes brittle, then
    gets loosened, then means nothing.
    """
    seen = {"route": "evolution", "ok": True, "verdict": "ok",
            "facets": [("granularity", "applied")], "answer": "x"}
    assert RS.check({}, seen) == []


# --------------------------------------------------------------------------- #
# It must declare its blindness
# --------------------------------------------------------------------------- #
def test_the_bank_states_what_it_cannot_see(doc):
    assert doc["meta"].get("sees")
    assert doc["meta"].get("cannot_see")
    blind = " ".join(doc["meta"]["cannot_see"]).lower()
    # The four that matter, named rather than gestured at.
    for topic in ("numeric", "llm", "kestrelmoor", "text"):
        assert topic in blind, "the bank does not name %r among its limits" % topic


def test_the_count_matches_the_questions(doc):
    assert doc["meta"]["count"] == len(doc["questions"])


def test_every_case_states_at_least_one_expectation(doc):
    """A case with no expectation passes forever and measures nothing."""
    keys = ("expect_route", "expect_ok", "expect_verdict", "expect_facets",
            "expect_answer_contains")
    for case in doc["questions"]:
        assert any(k in case for k in keys), case["id"]


def test_every_expected_failure_names_its_defect(doc):
    """An xfail without a named defect is an excuse, not a record."""
    for case in doc["questions"]:
        if case.get("expected_to_fail"):
            assert len(str(case["expected_to_fail"])) > 60, case["id"]


# --------------------------------------------------------------------------- #
# The xfail accounting
# --------------------------------------------------------------------------- #
def test_a_declared_defect_that_starts_passing_is_reported_not_absorbed():
    rows = [{"expected_to_fail": True, "failures": []},
            {"expected_to_fail": True, "failures": ["still broken"]},
            {"expected_to_fail": False, "failures": []},
            {"expected_to_fail": False, "failures": ["a regression"]}]
    passed, failed, fixed = RS.verdict(rows)
    assert (passed, failed, fixed) == (2, 1, 1)
