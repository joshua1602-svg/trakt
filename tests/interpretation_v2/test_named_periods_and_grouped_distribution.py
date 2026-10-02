"""Two readings the Claude Opus 5.5 sign-off run exposed (design §40).

1. NAMED PERIODS WRITTEN AS A RELATIVE FORM. "Compare October and November
   pipeline amount" arrived as `relative_pair` with labels ["October",
   "November"]; the runtime resolved the form — the latest month and the one
   before — and answered for August and September, silently. A relative form
   cannot name a month: when every label names one (read by the governed
   label reader the runtimes share), the compiler binds the named months; a
   mix is ambiguous and is asked about.
2. A GROUPED DISTRIBUTION. "How is the balance spread across LTV bands?"
   arrived as `distribution` over the LTV band — one figure per band — and was
   refused; it is a breakdown (normalisation 7).

Both are contract rules, not readings of particular questions: they hold for
any wording and any model. The replay compiles what Claude Opus 5.5 actually
recorded and asserts each lands where the Claude Opus 5 baseline answered.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mi_agent.interpretation_v2.compiler import CompilerContext, DeterministicCompiler
from mi_agent.interpretation_v2.intent import parse_candidate_intent
from mi_agent.period_labels import parse_anchor

_FIXTURE = (Path(__file__).resolve().parents[1]
            / "fixtures/opus55_signoff_readings_20261002.json")


@pytest.fixture(scope="module")
def compiler():
    return DeterministicCompiler(CompilerContext())


def _plan(compiler, payload):
    result = compiler.compile(parse_candidate_intent(payload))
    return result, (result.plan.to_dict() if result.plan else None)


_PIPELINE = {"schema_version": "candidate_intent/1.0", "capability": "pipeline",
             "operation": "compare", "change_form": "level_comparison",
             "population": {"base": "pipeline"},
             "measures": [{"concept": "pipeline_amount"}]}


@pytest.mark.parametrize("label, named", [
    ("October", True), ("October 2025", True), ("Sept", True), ("Nov", True),
    ("last month", False), ("latest", False), ("prior", False),
    ("month-on-month", False), ("8-week", False), ("today", False),
])
def test_the_governed_label_reader_names_months(label, named):
    """The one label reader the runtimes share decides what names a month."""
    assert (parse_anchor(label) is not None) is named


def test_named_months_in_a_relative_form_are_the_named_months(compiler):
    payload = dict(_PIPELINE, time={"form": "relative_pair", "grain": "monthly",
                                    "labels": ["October", "November"]})
    result, plan = _plan(compiler, payload)
    assert plan["period"]["form"] == "explicit_period"
    assert list(plan["period"]["labels"]) == ["October", "November"]
    assert plan["period"]["periods_back"] is None


def test_a_named_month_as_the_previous_period_is_that_month(compiler):
    payload = dict(_PIPELINE, operation="point_in_time", change_form=None,
                   time={"form": "previous_reporting_period", "labels": ["August"]})
    _, plan = _plan(compiler, payload)
    assert plan["period"]["form"] == "explicit_period"
    assert list(plan["period"]["labels"]) == ["August"]


def test_relative_words_stay_relative(compiler):
    payload = dict(_PIPELINE, time={"form": "relative_pair",
                                    "labels": ["latest", "prior"]})
    _, plan = _plan(compiler, payload)
    assert plan["period"]["form"] == "relative_pair"


def test_named_and_relative_together_are_asked_about(compiler):
    payload = dict(_PIPELINE, time={"form": "relative_pair",
                                    "labels": ["latest", "October"]})
    result, plan = _plan(compiler, payload)
    assert plan is None
    assert "AMBIGUOUS_PERIOD" in result.codes()


_FUNDED = {"schema_version": "candidate_intent/1.0",
           "capability": "generic_analysis", "population": {"base": "funded"},
           "measures": [{"concept": "current_outstanding_balance"}],
           "time": {"form": "current"}}


def test_a_grouped_distribution_is_a_breakdown(compiler):
    _, plan = _plan(compiler, dict(_FUNDED, operation="distribution",
                                   dimensions=["ltv_bucket"]))
    assert plan["operation"] == "breakdown"


def test_an_ungrouped_distribution_is_left_as_it_was(compiler):
    result, plan = _plan(compiler, dict(_FUNDED, operation="distribution"))
    assert plan is None or plan["operation"] == "distribution"


# --------------------------------------------------------------------------- #
# the replay of what Claude Opus 5.5 recorded
# --------------------------------------------------------------------------- #

_EXPECTED = {
    "pipeline_evolution_012": ("explicit_period", ["October", "November"]),
    "pipeline_evolution_014": ("explicit_period", ["October", "November"]),
    "hv_pipeline_evolution_011_1": ("explicit_period", ["October", "November"]),
    "hv_pipeline_evolution_012_1": ("explicit_period", ["October", "November"]),
    "hv_pipeline_evolution_014_1": ("explicit_period", ["October", "November"]),
    # Controls: a relative pair in relative words, and an explicit reading.
    "pipeline_evolution_013": ("relative_pair", ["latest", "prior"]),
    "pipeline_evolution_011": ("explicit_period", ["October", "November"]),
}


@pytest.mark.parametrize("qid", sorted(_EXPECTED))
def test_the_recorded_readings_compile_to_the_named_periods(compiler, qid):
    payload = json.loads(_FIXTURE.read_text())["intents"][qid]
    _, plan = _plan(compiler, payload)
    form, labels = _EXPECTED[qid]
    assert plan["period"]["form"] == form
    assert list(plan["period"]["labels"]) == labels


def test_the_recorded_ltv_bands_reading_is_a_breakdown(compiler):
    payload = json.loads(_FIXTURE.read_text())["intents"]["hv_funded_breakdown_1d_014_1"]
    assert payload["operation"] == "distribution"
    _, plan = _plan(compiler, payload)
    assert plan["operation"] == "breakdown"
