"""tests/test_comparison_periods_structural.py — the periods, not the wording.

`time.comparison_period` carries what a reader is SHOWN: `", ".join(...)`, a
display join. Splitting that back apart to recover the pair is re-parsing a
serialisation, and it breaks on any period label containing the separator.

This is the closure `window_periods` already made once for `trend_window` —
"the slot carried the WORDING and not the MAGNITUDE, so a consumer that needed
the number had to ask the owner again" — made again for the comparison pair.

Measured before the field existed: 0 of 26 readings of the `temporal_compare`
surface carried the pair structurally. After: 26 of an asserted 26.
"""
from __future__ import annotations

import pytest

from mi_agent_api import analytical_plan as plan
from question_interpretation.projection import project
from question_interpretation.schema import TimeClaim


PAIRS = (
    ("Compare October and November funded balance.", ("October", "November")),
    ("Compare September and October funded balance", ("September", "October")),
    ("Compare October and November loan count.", ("October", "November")),
    ("How did the pipeline amount change from last week?", ("latest", "last week")),
)


@pytest.mark.parametrize("question,expected", PAIRS)
def test_the_contract_carries_the_periods_structurally(question, expected):
    qi = project(question, semantics={})
    assert qi.time.comparison_periods == expected, question


@pytest.mark.parametrize("question,expected", PAIRS)
def test_the_wording_is_kept_beside_the_values(question, expected):
    """Both, not one instead of the other.

    The join is what a reader is shown and other consumers still render it, so
    the structural field is ADDITIVE. A change that replaced the wording would
    be a different change with a different blast radius.
    """
    qi = project(question, semantics={})
    assert qi.time.comparison_period.state == "filled"
    assert qi.time.comparison_period.raw_text == ", ".join(expected)


def test_a_question_naming_no_comparison_carries_an_empty_pair():
    """Empty, not None, and distinguishable from a comparison that failed.

    A consumer must be able to tell "no comparison was named" from "one was
    named and could not be resolved"; the slot's state says which, and the pair
    deliberately does not guess.
    """
    qi = project("What is the funded balance?", semantics={})
    assert qi.time.comparison_periods == ()
    assert qi.time.comparison_period.state == "empty"


def test_the_field_is_normalised_to_strings():
    assert TimeClaim(comparison_periods=[2024, "November"]).comparison_periods \
        == ("2024", "November")


def test_it_serialises_onto_the_contract():
    qi = project("Compare October and November funded balance.", semantics={})
    assert qi.as_dict()["time"]["comparison_periods"] == ["October", "November"]


# --------------------------------------------------------------------------- #
# The planner consumes the structure
# --------------------------------------------------------------------------- #
def test_the_plan_layer_reads_the_pair_and_not_the_question():
    import inspect
    src = inspect.getsource(plan.comparison_periods)
    assert "comparison_periods" in src
    assert "question" not in inspect.signature(plan.comparison_periods).parameters


def test_the_bridge_plan_opens_at_the_first_period_not_the_join():
    """The reason this closure exists, at the one caller that wanted structure.

    A two-period question would have handed `STACK_PERIODS` the string
    "October, November" as its `from` — a start period no tape has. The bridge
    surface never produces one today (all 12 owned cases name at most one
    period, measured before the switch), so this is representation and not
    behaviour; the assertion is here so it stays that way.
    """
    qi = project("Funded balance bridge from October to November", semantics={})
    built = plan.build_funded_bridge_plan(
        qi, dimension_key="region", dimension_label="Region")
    step = next(s for s in built.steps if s.primitive == plan.STACK_PERIODS)
    assert step.inputs["from"] == "October"


def test_the_wording_accessor_still_returns_the_wording():
    """DELIBERATELY UNCHANGED. Making `comparison_period` return the first
    period would change what it MEANS, not how it is represented, on all five
    corpus questions that carry a comparison. The structural read is a separate
    accessor and the one caller that wants structure asks for it by name.
    """
    qi = project("Compare October and November funded balance.", semantics={})
    assert plan.comparison_period(qi) == "October, November"
    assert plan.comparison_periods(qi) == ("October", "November")
