#!/usr/bin/env python3
"""Stage 4, step 1: the population check compares governed populations.

The 160-run regression, closed at its cause. `_analytical_population_satisfies`
compared the facet's FIELD NAME against the literal text of the predicates the
plan declared. "New lending" is governed as `months_on_book le 1` while the term
resolves to `seasoning_segment`, so a population the plan genuinely applied was
rejected and the answer blocked.

"Front book" survived only by coincidence: its governed predicate happens to
name the same field the term resolves to. Correcting one accidental pass while
preserving another would be no correction at all, so the tests below prove
Q7's acceptance now rests on the governed comparison — it holds when the names
do NOT coincide, and when there is no field key at all.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

warnings.simplefilter("ignore")
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent import execution_receipt as R  # noqa: E402


def _facet(label, field_key="seasoning_segment"):
    return R.RequestedFacet(kind=R.KIND_POPULATION, field_key=field_key,
                            label="the population %s" % label)


def _envelope(*predicates):
    """An envelope declaring exactly these row predicates."""
    return {"metadata": {"analyticalFindings": [
        {"population": {"predicate": "; ".join(predicates)}}]}}


# -- the governed windows resolve to their configured predicates -------------
@pytest.mark.parametrize("label,expected", [
    ("new lending", ["months_on_book le 1"]),
    ("recent lending", ["months_on_book le 3"]),
    ("front book", ["seasoning_segment = Front Book"]),
    ("back book", ["seasoning_segment = Back Book"]),
])
def test_a_governed_window_resolves_to_its_predicate(label, expected):
    assert R._governed_population_predicates(_facet(label)) == expected


def test_an_ungoverned_population_resolves_to_nothing():
    """The guard must not be widened. A phrase the configuration does not
    define is not a governed window and gets no free pass."""
    assert R._governed_population_predicates(_facet("the good book")) == []


# -- the regression, closed --------------------------------------------------
def test_new_lending_is_accepted_against_its_months_on_book_predicate():
    """The 160-run case. The plan applied the population and declared it; the
    check could not see that `months_on_book le 1` IS "new lending"."""
    env = _envelope("months_on_book le 1")
    assert R._analytical_population_satisfies(env, _facet("new lending")) is True


def test_the_old_literal_comparison_would_have_rejected_it():
    """Proof this test is detecting the fix rather than something incidental.

    The literal path required the facet's field name to appear in the predicate
    text. `seasoning_segment` does not appear in `months_on_book le 1`.
    """
    facet = _facet("new lending")
    declared = ["months_on_book le 1"]
    field = str(facet.field_key).lower()
    assert not any(field in p.lower() for p in declared)


# -- Q7 must NOT still be passing on a name coincidence ----------------------
def test_front_book_is_accepted_when_the_names_do_not_coincide():
    """The addition to the brief, and the point of it.

    `field_key` here appears nowhere in the declared predicate, so the literal
    comparison cannot be what accepts this. Only the governed comparison can.
    """
    env = _envelope("seasoning_segment = Front Book")
    facet = _facet("front book", field_key="lending_window")
    assert R._analytical_population_satisfies(env, facet) is True


def test_front_book_is_accepted_with_no_field_key_at_all():
    """The literal path returns False immediately without a field key, so this
    can only be the governed comparison."""
    env = _envelope("seasoning_segment = Front Book")
    assert R._analytical_population_satisfies(env, _facet("front book", None)) is True


def test_that_independence_check_can_fail():
    """A non-coinciding field key with an UNGOVERNED label is still rejected —
    so the two tests above are detecting the governed window, not simply
    accepting anything whose names fail to match."""
    env = _envelope("seasoning_segment = Front Book")
    facet = _facet("the good book", field_key="lending_window")
    assert R._analytical_population_satisfies(env, facet) is False


# -- the guard is not weakened ----------------------------------------------
def test_the_governed_comparison_rejects_a_window_the_plan_did_not_apply():
    """Governed does not mean waved through: the plan must have declared THAT
    population. Asserted on the governed comparison itself, because the literal
    path has a pre-existing permissiveness recorded in the next test."""
    declared = ["seasoning_segment = back book"]
    for label in ("front book", "new lending"):
        governed = [g.lower() for g in R._governed_population_predicates(_facet(label))]
        assert governed, label
        assert not any(g in declared for g in governed), (label, governed)


def test_a_window_the_plan_did_not_apply_is_rejected_on_the_real_label_form():
    """The labels the receipt actually builds embed the field and the value —
    "the population seasoning_segment = Front Book". On that form both paths
    reject a mismatched population."""
    env = _envelope("seasoning_segment = Back Book")
    facet = R.RequestedFacet(
        kind=R.KIND_POPULATION, field_key="seasoning_segment",
        label="the population seasoning_segment = Front Book")
    assert R._analytical_population_satisfies(env, facet) is False


def test_the_literal_path_is_permissive_when_a_label_omits_its_field():
    """PRE-EXISTING, recorded rather than absorbed or silently fixed.

    The literal comparison derives the value it wants by splitting the label on
    the field name. Where the label does not contain the field name the value
    comes out empty, and `if not value or value in text` then accepts ANY
    predicate naming that field — including the wrong population.

    Verified present before this change by stashing it. It is not reachable
    with the labels the receipt builds today, which embed the field, and the
    governed comparison added here is stricter rather than looser. Fixing it
    would change acceptance for populations outside the seasoning family, which
    the pre-registered prediction says to report rather than absorb.
    """
    env = _envelope("seasoning_segment = Back Book")
    facet = _facet("front book")          # label omits the field name
    assert R._analytical_population_satisfies(env, facet) is True


def test_an_empty_envelope_accepts_nothing():
    assert R._analytical_population_satisfies({}, _facet("new lending")) is False
    assert R._analytical_population_satisfies(None, _facet("front book")) is False


def test_the_literal_path_still_works_for_non_seasoning_populations():
    """The Tranche D case the check was built for — `pipeline_stage = Offer` —
    is untouched. This change EXTENDS the check; it does not replace it."""
    env = _envelope("pipeline_stage = OFFER")
    facet = R.RequestedFacet(kind=R.KIND_POPULATION, field_key="pipeline_stage",
                             label="the population pipeline_stage = Offer")
    assert R._analytical_population_satisfies(env, facet) is True


def test_only_the_population_branch_can_be_affected():
    """The pre-registered prediction, asserted structurally as well as measured.

    The governed comparison is reached only from the KIND_POPULATION branch, so
    no other facet kind can move. Measured separately across all 252 calibration
    cases: zero facets of any kind changed status.
    """
    source = (_REPO_ROOT / "mi_agent" / "execution_receipt.py").read_text(encoding="utf-8")
    callers = [line for line in source.splitlines()
               if "_analytical_population_satisfies(" in line and "def " not in line]
    assert len(callers) == 1, callers
    index = source.index(callers[0])
    preceding = source[:index]
    assert preceding.rindex("KIND_POPULATION") > preceding.rindex("elif facet.kind ==") - 1
