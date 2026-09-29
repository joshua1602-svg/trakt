"""The forecast measures are DEFINED for the model, and nothing else moved.

The owner's production run (2026-09-28) showed the model reading different
questions as one forecast measure because it was shown a name and no
definition. Vocabulary 2.3.0 defines the three. These tests pin that the
definitions reach the model through its lookup tools, that each one still
rules out the readings the production run actually made, and that the change
is confined to those three concepts — the orientation block, the prompt and
the tool schemas are pinned elsewhere (`test_the_interpreter_policy_did_not_move`).
"""

from __future__ import annotations

import pytest

from mi_agent.interpretation_v2 import metadata as metadata_mod
from mi_agent.interpretation_v2.vocabulary import (
    SPECIALIST_MEASURE_DEFINITIONS, SPECIALIST_MEASURES, VOCABULARY_VERSION,
    load_governed_vocabulary)

FORECAST = ("forecast_funded_balance", "forecast_completion_rate",
            "forecast_milestone_date")

#: For each measure, the phrases that rule out a reading the production run
#: made. Weakening a definition until one of these is gone fails here.
RULES_OUT = {
    "forecast_funded_balance": ("ONE figure", "latest weekly extract",
                                "NOT a curve", "month-by-month",
                                "base, downside or upside", "NOT one part"),
    "forecast_completion_rate": ("per MONTH", "NOT a percentage",
                                 "NOT a conversion rate",
                                 "how the forecast is calculated"),
    "forecast_milestone_date": ("ONE", "`target` is `forecast_funded_balance`",
                                "`gte`", "NOT a table of dates"),
}


@pytest.fixture(scope="module")
def vocabulary():
    return load_governed_vocabulary()


@pytest.fixture(scope="module")
def tools(vocabulary):
    return metadata_mod.GovernedMetadataService(vocabulary)


def test_the_version_records_what_the_model_is_shown():
    # 2.3.0 introduced these definitions; later versions keep them.
    major, minor, _ = (int(x) for x in VOCABULARY_VERSION.split("."))
    assert (major, minor) >= (2, 3)


def test_exactly_the_forecast_measures_are_defined():
    assert set(SPECIALIST_MEASURE_DEFINITIONS) == set(FORECAST)
    assert set(FORECAST) == set(SPECIALIST_MEASURES["forecast"])


@pytest.mark.parametrize("concept", FORECAST)
def test_the_definition_reaches_the_model_through_both_lookups(concept, tools):
    shown = tools.call("get_concept_metadata", {"concept_id": concept})
    assert shown["found"] and shown["definition"] == \
        SPECIALIST_MEASURE_DEFINITIONS[concept]
    searched = tools.call("search_concepts", {"query": concept})
    row = next(c for c in searched["concepts"] if c["concept_id"] == concept)
    assert row["definition"] == SPECIALIST_MEASURE_DEFINITIONS[concept]


@pytest.mark.parametrize("concept", FORECAST)
def test_each_definition_rules_out_the_production_misreads(concept):
    definition = SPECIALIST_MEASURE_DEFINITIONS[concept]
    for phrase in RULES_OUT[concept]:
        assert phrase in definition, (concept, phrase)


def test_no_definition_states_a_methodology_the_model_could_decompose():
    """Rule 3 holds: the capability owns the arithmetic."""
    for definition in SPECIALIST_MEASURE_DEFINITIONS.values():
        for leak in ("column", "SELECT", "groupby", "sum(", ".csv",
                     "completion_probability", "weighted_expected_funded_amount"):
            assert leak not in definition


def test_every_other_specialist_measure_keeps_the_generic_description(vocabulary):
    for capability, concepts in SPECIALIST_MEASURES.items():
        if capability == "forecast":
            continue
        for concept_id in concepts:
            description = vocabulary.resolve(concept_id).description
            assert description.startswith(f"Owned by the {capability} capability")
