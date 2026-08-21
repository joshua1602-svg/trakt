#!/usr/bin/env python3
"""Standing condition 4 for the deterministic robustness arm.

The instrument reports 44 variations, 20 of them the seasoning family that
Stage 4's acceptance condition names. A summary that cannot go down is not a
control, so this proves the instrument detects the exact regression it exists
to catch: the 32c263a failure mode, where governed seasoning populations began
blocking and 160 runs moved from CORRECT to SAFE_REFUSAL.

The frozen scorer is used, not a copy. If `nl_score.grade` ever stops
downgrading that shape, these tests fail rather than quietly passing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVIDENCE = _REPO_ROOT / "due_diligence" / "evidence" / "analytical_intent_v1"
for p in (str(_REPO_ROOT), str(_EVIDENCE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import nl_score  # noqa: E402  (the frozen scorer, imported not copied)

from question_interpretation.run_robustness_deterministic import (  # noqa: E402
    SEASONING_INTENTS, summarise,
)

#: A Q1 run as the release candidate produces it today — the analytical layer
#: claims it and names the intent.
_HEALTHY_Q1 = {
    "ok": True, "route": "analytical_composition",
    "answer": "The profile of new originations shifted over the period. " * 3,
    "intent": "origination_profile_change",
    "capabilities": ["period_movement", "population_profile"],
    "analyticalIntent": "origination_profile_change",
    "spec": {}, "controlledRefusal": False,
    "artifacts": [{"type": "table", "rows": 8}], "warnings": [],
}

#: The 32c263a shape: the seasoning population is classified as unapplied, so a
#: question the route CAN express refuses instead of answering.
_BLOCKED_Q1 = {
    "ok": False, "route": "analytical_composition",
    "answer": ("I could not confirm that the new-lending population was applied "
               "to the calculation, so I have not returned a figure."),
    "intent": None, "capabilities": [], "analyticalIntent": None,
    "spec": {}, "controlledRefusal": True, "artifacts": [], "warnings": [],
}


def test_the_frozen_scorer_grades_a_healthy_seasoning_run_correct():
    outcome, _causes, _note = nl_score.grade("Q1", _HEALTHY_Q1)
    assert outcome == nl_score.CORRECT, outcome


def test_the_frozen_scorer_downgrades_the_32c263a_shape():
    """The regression must not grade as CORRECT. This is the whole control."""
    outcome, _causes, _note = nl_score.grade("Q1", _BLOCKED_Q1)
    assert outcome != nl_score.CORRECT
    assert outcome == nl_score.SAFE_REFUSAL, outcome


def _rows(outcome_for_seasoning: str):
    rows = []
    for intent in ("Q1", "Q7", "Q8"):
        for i in range(4):
            rows.append({"intent": intent, "variation": i + 1,
                         "outcome": outcome_for_seasoning})
    for i in range(4):
        rows.append({"intent": "Q6", "variation": i + 1,
                     "outcome": nl_score.CORRECT})
    return rows


def test_the_summary_reports_the_seasoning_family_by_name():
    """Stage 4's acceptance is 'the seasoning families are unmoved', measured
    explicitly and by name — never as part of an aggregate."""
    s = summarise(_rows(nl_score.CORRECT))
    assert set(s["seasoning_family"]) == set(SEASONING_INTENTS)
    assert s["seasoning_family"]["Q1"] == {nl_score.CORRECT: 4}


def test_the_summary_moves_when_the_seasoning_family_regresses():
    """Proof the report above can fail: feed it the regression and the family
    changes, visibly and by name."""
    healthy = summarise(_rows(nl_score.CORRECT))
    regressed = summarise(_rows(nl_score.SAFE_REFUSAL))

    assert healthy["seasoning_family"] != regressed["seasoning_family"]
    for intent in SEASONING_INTENTS:
        assert regressed["seasoning_family"][intent] == {nl_score.SAFE_REFUSAL: 4}
    assert healthy["by_outcome"][nl_score.CORRECT] == 16
    assert regressed["by_outcome"][nl_score.CORRECT] == 4


def test_an_aggregate_alone_would_hide_it_less_than_a_named_family():
    """Why the family is reported separately.

    The aggregate does move here, but a 12-of-16 drop inside a 44-variation
    total is exactly the kind of movement that gets attributed to noise. The
    named family makes it unambiguous which intents moved."""
    regressed = summarise(_rows(nl_score.SAFE_REFUSAL))
    assert regressed["by_outcome"].get(nl_score.CORRECT, 0) == 4
    assert all(regressed["seasoning_family"][i] == {nl_score.SAFE_REFUSAL: 4}
               for i in SEASONING_INTENTS)


def test_unsafe_outcomes_are_distinguishable_from_safe_refusals():
    """A refusal with no stated reason is a SILENT_SEMANTIC_ERROR, not a safe
    refusal. The programme requires unsafe outcomes to remain zero, so the two
    must never collapse."""
    silent = dict(_BLOCKED_Q1, answer="No.")
    outcome, _causes, _note = nl_score.grade("Q1", silent)
    assert outcome == nl_score.SILENT_SEMANTIC_ERROR, outcome
    assert outcome != nl_score.SAFE_REFUSAL
