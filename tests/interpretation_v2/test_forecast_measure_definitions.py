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

from mi_agent import semantic_model
from mi_agent.interpretation_v2 import metadata as metadata_mod
from mi_agent.interpretation_v2.vocabulary import (
    SPECIALIST_MEASURE_DEFINITIONS, SPECIALIST_MEASURES, VOCABULARY_VERSION,
    load_governed_vocabulary)

#: The three 2.3.0 defined; the ones whose readings the production run
#: measured, pinned by RULES_OUT below.
FORECAST_2_3 = ("forecast_funded_balance", "forecast_completion_rate",
                "forecast_milestone_date")
#: Since 2.7.0 every forecast measure is declared, with its definition, in the
#: capability's semantic model (P0 design §16).
FORECAST = tuple(semantic_model.load("forecast").measures)
#: Defined since 2.6.0 (catalogue batch 1): the Pipeline tab's weighted figure,
#: which the model otherwise confused with the pipeline amount.
PIPELINE = ("weighted_expected_funded_amount",)
#: Since 2.10.0 (D2a, §20.2): the stage-movement capability's measured rates,
#: declared with their definitions in its semantic model.
STAGE = tuple(semantic_model.load("pipeline_stage_movement").measures)
#: Since 2.18.0 (§29.2): the stage-movement FLOW figures, which were shown by
#: name only — "how has the pipeline moved" was answered with the flows over
#: every case in the extracts where the live pipeline's own change was asked.
STAGE_FLOWS = ("cases_moved", "amount_moved", "cases_arrived",
               "cases_departed", "cases_stayed", "stayer_amount_change",
               "stage_opening", "stage_closing")

#: For each measure, the phrases that rule out a reading the production run
#: made. Weakening a definition until one of these is gone fails here.
RULES_OUT = {
    # §29.2: a flow between stages is not the pipeline's change in size.
    **{flow: ("between STAGES", "NOT how the pipeline changed")
       for flow in ("cases_moved", "amount_moved", "cases_arrived",
                    "cases_departed", "cases_stayed", "stayer_amount_change")},
    "forecast_funded_balance": ("ONE figure", "latest weekly extract",
                                "NOT a curve", "month-by-month",
                                "base, downside or upside", "NOT one part",
                                # 2026-09-30 full bank: [87] read as the
                                # pipeline's weighted amount (vocabulary 2.13.0).
                                "'the expected funded balance'"),
    "weighted_expected_funded_amount": ("NOT the forecast funded balance",
                                        "'the expected funded BALANCE'"),
    # 2026-09-30 full bank: [102] dropped 'active' and answered the whole
    # extract's exclusion (vocabulary 2.13.0).
    "weighting_excluded_amount": ("WHOLE extract",
                                  "keeps that restriction in the intent"),
    "forecast_completion_rate": ("per MONTH", "NOT a percentage",
                                 "NOT a conversion rate",
                                 "how the forecast is calculated",
                                 # 2026-09-30 full bank: [125, 126] dropped
                                 # their 8- and 12-week window.
                                 "ANOTHER window", "rather than dropping it"),
    "forecast_milestone_date": ("ONE", "`target` is `forecast_funded_balance`",
                                "`gte`", "NOT a table of dates"),
}


@pytest.fixture(scope="module")
def vocabulary():
    return load_governed_vocabulary()


@pytest.fixture(scope="module")
def tools(vocabulary):
    return metadata_mod.GovernedMetadataService(vocabulary)


def test_the_2_3_measures_are_still_in_the_model():
    assert set(FORECAST_2_3) <= set(FORECAST)


def test_the_version_records_what_the_model_is_shown():
    # 2.3.0 introduced these definitions; later versions keep them.
    major, minor, _ = (int(x) for x in VOCABULARY_VERSION.split("."))
    assert (major, minor) >= (2, 3)


def test_exactly_the_forecast_measures_and_the_weighted_pipeline_are_defined():
    assert set(SPECIALIST_MEASURE_DEFINITIONS) == \
        set(FORECAST) | set(PIPELINE) | set(STAGE) | set(STAGE_FLOWS)
    assert set(FORECAST) == set(SPECIALIST_MEASURES["forecast"])


@pytest.mark.parametrize("concept", FORECAST + PIPELINE + STAGE + STAGE_FLOWS)
def test_the_definition_reaches_the_model_through_both_lookups(concept, tools):
    shown = tools.call("get_concept_metadata", {"concept_id": concept})
    assert shown["found"] and shown["definition"] == \
        SPECIALIST_MEASURE_DEFINITIONS[concept]
    searched = tools.call("search_concepts", {"query": concept})
    row = next(c for c in searched["concepts"] if c["concept_id"] == concept)
    assert row["definition"] == SPECIALIST_MEASURE_DEFINITIONS[concept]


@pytest.mark.parametrize("concept", sorted(RULES_OUT))
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
            if concept_id in SPECIALIST_MEASURE_DEFINITIONS:
                continue
            description = vocabulary.resolve(concept_id).description
            assert description.startswith(f"Owned by the {capability} capability")
