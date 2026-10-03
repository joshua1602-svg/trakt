"""D23 (owner decision 2026-09-30): amount or number.

The owner's rule, in their words: "'how much' = amount, 'how many' = count",
and, for the low-risk case where a question says neither, "default to amount".
It is a reading rule, so it lives where the model reads: a governed default in
the standing context it is shown on every question (vocabulary 2.21.0), not a
per-question fix. The words choose the measure; the unstated case is answered
by amount and disclosed, never asked back.

What is checked here is what the rule depends on, not any one question:

  * the model is shown the rule, in the request the interpreter sends;
  * every population the agent answers on carries both sides — an amount and
    a count — so either reading always has a governed concept to bind to;
  * both readings compile to a plan, and a disclosed amount reading is carried
    on the plan as a disclosed note, never as a blocking ambiguity.
"""
from __future__ import annotations

import json

import pytest

from mi_agent.interpretation_v2 import load_governed_vocabulary
from mi_agent.interpretation_v2.compiler import CompilerContext, DeterministicCompiler
from mi_agent.interpretation_v2.intent import parse_candidate_intent
from mi_agent.interpretation_v2.opus_interpreter import build_system_blocks
from mi_agent.interpretation_v2.vocabulary import (
    AMOUNT_OR_NUMBER_RULE, AMOUNT_OR_NUMBER_SLOT, GOVERNED_DEFAULTS,
    VOCABULARY_VERSION)
from mi_agent.tests.test_plan_temporal_runtime import BASE_INTENT
from tests.interpretation_v2.test_specialist_runtime_pipeline import _PIPELINE_INTENT

#: Each population's amount and the count of what it holds. Not exhaustive of
#: the catalogue: the populations the agent answers the size of.
AMOUNT_AND_COUNT = [
    ("funded", "current_outstanding_balance", "loan"),
    ("pipeline", "pipeline_amount", "pipeline_case_count"),
    ("excluded from weighting", "weighting_excluded_amount",
     "weighting_excluded_case_count"),
    ("forecast", "forecast_funded_balance", "forecast_loan_count"),
    ("ineligible", "ineligible_balance", "ineligible_loan_count"),
]

_DISCLOSED = {"slot": "measures", "blocking": False,
              "note": "amount or number not stated: by amount (D23)"}


@pytest.fixture(scope="module")
def vocabulary():
    return load_governed_vocabulary()


def _compile(payload):
    return DeterministicCompiler(CompilerContext()).compile(
        parse_candidate_intent(payload))


def test_the_rule_is_the_owners_words():
    rule = AMOUNT_OR_NUMBER_RULE
    assert rule.startswith("D23")
    assert "'How much' is the AMOUNT" in rule
    assert "'How many'" in rule and "is the NUMBER" in rule
    assert "says neither is answered by AMOUNT" in rule
    assert "non-blocking" in rule and "never a blocking one" in rule
    # A measure the reader names is theirs, whatever words surround it.
    assert "names itself" in rule and "not overridden" in rule
    assert GOVERNED_DEFAULTS[AMOUNT_OR_NUMBER_SLOT] == AMOUNT_OR_NUMBER_RULE
    major, minor, _ = (int(x) for x in VOCABULARY_VERSION.split("."))
    assert (major, minor) >= (2, 21)


def test_the_model_is_shown_the_rule_on_every_question(vocabulary):
    blocks = build_system_blocks(vocabulary)
    governed = next(b["text"] for b in blocks
                    if b["text"].startswith("GOVERNED CONTEXT"))
    payload = json.loads(governed.split("\n", 1)[1])
    assert payload["governed_defaults"][AMOUNT_OR_NUMBER_SLOT] \
        == AMOUNT_OR_NUMBER_RULE
    # In the cached prefix: the same for every question and every client.
    cached_through = next(i for i, b in enumerate(blocks) if "cache_control" in b)
    assert blocks.index(next(b for b in blocks if b["text"] == governed)) \
        <= cached_through


@pytest.mark.parametrize("population, amount, count", AMOUNT_AND_COUNT)
def test_every_population_carries_both_an_amount_and_a_count(
        vocabulary, population, amount, count):
    assert amount in vocabulary.concepts, f"{population}: no amount {amount!r}"
    assert count in vocabulary.concepts, f"{population}: no count {count!r}"
    assert vocabulary.concepts[amount].role == "measure"
    assert vocabulary.concepts[count].role == "measure"


def _pipeline_by_stage(measure):
    return dict(_PIPELINE_INTENT, operation="breakdown",
                measures=[{"concept": measure}], dimensions=["pipeline_stage"])


def _funded_by_borrower_type(measure, statistic):
    return dict(BASE_INTENT, operation="breakdown",
                measures=[{"concept": measure, "statistic": statistic}],
                dimensions=["borrower_type"])


@pytest.mark.parametrize("payload", [
    _pipeline_by_stage("pipeline_amount"),                 # how much / neither
    _pipeline_by_stage("pipeline_case_count"),             # how many
    _funded_by_borrower_type("current_outstanding_balance", "sum"),
    _funded_by_borrower_type("loan", "count"),
])
def test_both_readings_compile_to_a_plan(payload):
    result = _compile(payload)
    assert result.plan is not None, result.codes()


@pytest.mark.parametrize("payload", [
    _pipeline_by_stage("pipeline_amount"),
    _funded_by_borrower_type("current_outstanding_balance", "sum"),
])
def test_the_amount_taken_for_neither_is_disclosed_not_asked(payload):
    result = _compile(dict(payload, ambiguity=[dict(_DISCLOSED)]))
    assert result.plan is not None, result.codes()
    notes = result.plan.to_dict()["provenance"]["notes"]
    assert "disclosed reading [measures]: amount or number not stated: " \
        "by amount (D23)" in notes, notes
