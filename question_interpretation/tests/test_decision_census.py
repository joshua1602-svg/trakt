"""The census must be able to report every class it defines, and to be wrong.

Its own biggest risk is not missing a decision — that is stated as a blind spot
— but INFLATING the count by probing two owners of different decisions and
calling the difference a disagreement. It did that three times while being
built, and each time the number came out three to fifty times too large.
"""
from __future__ import annotations

import pytest

from question_interpretation import decision_census as C


def test_the_self_test_passes():
    assert C.self_test() == 0


# --------------------------------------------------------------------------- #
# Every class must be reachable
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("measured,owners,declared,expected", [
    ({"disagreement_count": 2}, ["a", "b"], None, C.DISAGREEING),
    ({"disagreement_count": 0}, ["a", "b"], None, C.BY_MAINTENANCE),
    (None, ["a"], None, C.SINGLE),
    (None, ["a", "b"], None, C.BY_MAINTENANCE),
    (None, ["a", "b"], C.DERIVED, C.DERIVED),
    (None, ["a", "b"], C.BY_CONSTRUCTION, C.BY_CONSTRUCTION),
])
def test_each_class_is_producible(measured, owners, declared, expected):
    decision = {"id": "DX", "owners": owners}
    if declared:
        decision["class"] = declared
    assert C.classify(decision, {"DX": measured} if measured else {}) == expected


# --------------------------------------------------------------------------- #
# The count must not be padded or merged
# --------------------------------------------------------------------------- #
def test_no_decision_is_counted_without_more_than_one_owner():
    """Padding check: nothing single-owner may reach the headline classes."""
    for d in C.DECISIONS:
        cls = d.get("class")
        if cls in (C.BY_MAINTENANCE, C.DISAGREEING):
            assert len(d.get("owners") or []) > 1, d["id"]


def test_every_declared_defect_carries_its_evidence():
    for d in C.DECISIONS:
        if d.get("class") == C.DISAGREEING:
            assert d.get("evidence"), d["id"]
            assert "user_sees" in d, d["id"]


def test_every_removed_probe_says_why():
    """Merging check, inverted. Three probes were removed or narrowed for
    comparing owners of different decisions; each must record that rather than
    quietly vanish, or the census could be tuned to any number."""
    ids = {d["id"] for d in C.DECISIONS}
    assert {"D3", "D9"} <= ids
    for did in ("D3", "D9"):
        note = next(d for d in C.DECISIONS if d["id"] == did).get("note") or ""
        assert len(note) > 80, did


def test_the_decision_list_has_no_duplicate_ids():
    ids = [d["id"] for d in C.DECISIONS]
    assert len(ids) == len(set(ids))


# --------------------------------------------------------------------------- #
# Blind spots
# --------------------------------------------------------------------------- #
def test_the_census_states_its_blind_spots():
    assert len(C.BLIND_SPOTS) >= 4
    blob = " ".join(C.BLIND_SPOTS).lower()
    for topic in ("route", "llm", "book", "authored"):
        assert topic in blob, topic


def test_it_names_itself_as_its_own_biggest_blind_spot():
    """The decision list is authored. A census that did not say so would be
    quoting a lower bound as a total."""
    blob = " ".join(C.BLIND_SPOTS).lower()
    assert "lower bound" in blob
