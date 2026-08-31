#!/usr/bin/env python3
"""tests/test_concentration_direction_ownership.py — direction respects the limit.

Which way a concentration test moved is not "did the number go up". The governed
operator decides which way is worse: a ``max`` test is a CEILING and rising is
deteriorating; a ``min`` test is a FLOOR and FALLING is deteriorating. A
presentation layer that read direction off the number alone inverted every
minimum-type test in the pack — and the engine has supported minimum tests all
along, so this was a live defect waiting for a client to approve one.

The classification is now ``concentration_tests_api``'s, and travels on the
history payload so the dashboard and the pack cannot disagree.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mi_agent_api.concentration_tests_api import (  # noqa: E402
    DIRECTION_AWAY,
    DIRECTION_TOWARD,
    DIRECTION_UNCHANGED,
    STRESS_EASES,
    STRESS_INERT,
    STRESS_TIGHTENS,
    direction_of_travel,
    stress_effect,
)


# --------------------------------------------------------------------------- #
# A ceiling and a floor are not the same test.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("prior,current,operator,expected", [
    # MAX — a ceiling. Rising is deteriorating.
    (20.0, 26.0, "max", DIRECTION_TOWARD),
    (26.0, 20.0, "max", DIRECTION_AWAY),
    # MIN — a floor. FALLING is deteriorating. This is the case the old
    # presentation-layer rule got exactly backwards.
    (26.0, 20.0, "min", DIRECTION_TOWARD),
    (20.0, 26.0, "min", DIRECTION_AWAY),
])
def test_direction_is_measured_against_the_limit_not_the_number(
        prior, current, operator, expected):
    assert direction_of_travel(prior, current, 30.0, operator) == expected


def test_a_move_inside_the_tolerance_is_not_a_direction():
    """Catches: the book's ordinary noise dressed up as a trend."""
    assert direction_of_travel(20.0, 20.3, 30.0, "max") == DIRECTION_UNCHANGED
    assert direction_of_travel(20.0, 20.7, 30.0, "max") == DIRECTION_TOWARD


def test_no_prior_means_no_direction():
    """A single observation has no direction. Silence beats invention."""
    assert direction_of_travel(None, 20.0, 30.0, "max") is None
    assert direction_of_travel(20.0, None, 30.0, "max") is None


def test_the_tolerance_scales_with_the_limit():
    """A wide limit and a narrow one are held to the same standard."""
    assert direction_of_travel(20.0, 20.5, 100.0, "max") == DIRECTION_UNCHANGED
    assert direction_of_travel(20.0, 20.5, 10.0, "max") == DIRECTION_TOWARD


# --------------------------------------------------------------------------- #
# The stress, which does not always stress.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("current,stressed,operator,expected", [
    (20.0, 27.0, "max", STRESS_TIGHTENS),
    (20.0, 15.0, "max", STRESS_EASES),
    (20.0, 20.05, "max", STRESS_INERT),
    # On a floor test the directions invert with it.
    (20.0, 15.0, "min", STRESS_TIGHTENS),
    (20.0, 26.0, "min", STRESS_EASES),
])
def test_stress_effect_respects_the_limit_direction(
        current, stressed, operator, expected):
    assert stress_effect(current, stressed, 30.0, operator) == expected


# --------------------------------------------------------------------------- #
# Ownership.
# --------------------------------------------------------------------------- #

def test_the_presentation_layer_words_the_finding_and_does_not_classify_it():
    """``travel`` and ``stress_note`` may look up wording. They may not compare."""
    source = (_ROOT / "mi_agent_pptx" / "concentration.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for name in ("travel", "stress_note"):
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == name)
        compares = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Compare)]
        assert not compares, (
            f"{name}() is classifying again at line(s) {compares}; the engine owns "
            "which way is worse")


def test_the_wording_covers_every_engine_code():
    from mi_agent_pptx.concentration import travel, stress_note
    for code in (DIRECTION_TOWARD, DIRECTION_AWAY, DIRECTION_UNCHANGED):
        assert travel({"direction": code}), code
    assert travel({"direction": None}) is None
    for code in (STRESS_EASES, STRESS_INERT):
        assert stress_note({"stress_effect": code}), code
    # A tightening stress is shown as a figure, not explained away.
    assert stress_note({"stress_effect": STRESS_TIGHTENS}) is None
