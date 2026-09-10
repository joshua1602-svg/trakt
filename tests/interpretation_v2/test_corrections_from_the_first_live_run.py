"""What the first live run over the 135-question bank taught, pinned as tests.

Every case here is a well-formed reading that the compiler REFUSED on the first
measurement, where the refusal was the compiler's fault rather than the model's.
The corrections are recorded as tests so they cannot silently regress, and so a
reader can see exactly what changed between the two runs in
``mi_agent/interpretation_v2/evidence/``.

The common shape of all four: the compiler was demanding that the interpreter
specify something the interpreter is deliberately never shown.
"""

from __future__ import annotations

import pytest

from mi_agent.interpretation_v2 import (
    OUTCOME_PLAN,
    load_governed_vocabulary,
)
from mi_agent.interpretation_v2.vocabulary import GOVERNED_DEFAULTS

from .conftest import build_intent


# --------------------------------------------------------------------------- #
# 1 · a specialist operation owns its own period window
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("capability,operation,measure", [
    ("pipeline_stage_movement", "transition", "cases_moved"),
    ("pipeline_stage_movement", "arrivals", "cases_arrived"),
    ("pipeline_stage_movement", "stayers", "cases_stayed"),
    ("pipeline_stage_movement", "departures", "cases_departed"),
    ("pipeline_stage_movement", "reconciliation", "stage_opening"),
    ("borrowing_base", "bridge", "borrowing_base"),
])
def test_a_capability_owned_movement_needs_no_period_stated(
        capability, operation, measure, compiler):
    """"How many cases moved from KFI to Application?" is a complete question.

    The window a stage transition spans is part of what a stage transition IS.
    Requiring the intent to name two periods refused eleven well-formed pipeline
    questions on the first run.
    """
    intent = build_intent(capability=capability, operation=operation,
                          measures=[{"concept": measure}],
                          population={"base": "pipeline", "lens": "all",
                                      "seasoning": "any"},
                          time={"form": "current"})
    result = compiler.compile(intent)
    assert result.outcome == OUTCOME_PLAN, result.codes()
    assert result.plan.period.owned_by_capability is True


def test_a_GENERIC_movement_still_needs_two_periods(compiler):
    """The exemption is for capability-owned windows only.

    A generic period movement is composed by the compiler from two governed
    snapshots, so which two is a thing the question has to settle.
    """
    intent = build_intent(capability="period_movement", operation="movement",
                          time={"form": "current"})
    result = compiler.compile(intent)
    assert result.outcome != OUTCOME_PLAN
    assert "UNSUPPORTED_COMPOSITION" in result.codes()


# --------------------------------------------------------------------------- #
# 2 · a period pair names its sides through the period contract
# --------------------------------------------------------------------------- #

def test_a_period_pair_needs_no_left_and_right(compiler):
    """"How did the Direct book change last month?" states the pair completely."""
    intent = build_intent(capability="period_movement", operation="movement",
                          time={"form": "relative_pair"},
                          comparison={"kind": "period_pair"})
    result = compiler.compile(intent)
    assert result.outcome == OUTCOME_PLAN, result.codes()
    assert result.plan.comparison_kind == "period_pair"


@pytest.mark.parametrize("kind", ["population_pair", "dimension_pair"])
def test_a_population_or_dimension_pair_still_needs_both_sides(kind, compiler):
    intent = build_intent(operation="compare", comparison={"kind": kind,
                                                           "left": "balance"})
    result = compiler.compile(intent)
    assert result.outcome != OUTCOME_PLAN
    assert "CONFLICTING_CLAIMS" in result.codes()


# --------------------------------------------------------------------------- #
# 3 · share and contribution are governed analytic modes
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("statistic", ["share", "contribution"])
def test_an_analytic_mode_is_permitted_on_an_additive_measure(statistic, compiler):
    """"Which region contributed most to balance growth?" names a mode this
    repository already governs (P1A, P1D) — it is not an aggregation OF the
    balance field, so it is absent from the field's allowed_aggregations."""
    intent = build_intent(capability="period_movement", operation="rank",
                          measures=[{"concept": "balance",
                                     "statistic": statistic}],
                          dimensions=["source_portfolio"],
                          time={"form": "relative_pair"})
    result = compiler.compile(intent)
    assert result.outcome == OUTCOME_PLAN, result.codes()
    assert result.plan.outputs[0].measures[0].statistic == statistic


@pytest.mark.parametrize("statistic", ["share", "contribution"])
def test_an_analytic_mode_is_not_permitted_on_a_non_additive_measure(
        statistic, compiler):
    """A share of an average is not a quantity."""
    vocabulary = load_governed_vocabulary()
    ltv = vocabulary.resolve("current_ltv")
    assert "sum" not in ltv.allowed_statistics, "premise: LTV is not additive"
    assert statistic not in ltv.allowed_statistics

    intent = build_intent(measures=[{"concept": "current_ltv",
                                     "statistic": statistic}])
    result = compiler.compile(intent)
    assert result.outcome != OUTCOME_PLAN
    assert "UNSUPPORTED_STATISTIC" in result.codes()


def test_a_contribution_needs_no_weight_named(compiler):
    intent = build_intent(capability="period_movement", operation="rank",
                          measures=[{"concept": "balance",
                                     "statistic": "contribution"}],
                          dimensions=["source_portfolio"],
                          time={"form": "relative_pair"})
    assert compiler.compile(intent).outcome == OUTCOME_PLAN


# --------------------------------------------------------------------------- #
# 4 · a capability that owns its grouping needs none stated
# --------------------------------------------------------------------------- #

def test_a_concentration_rank_needs_no_dimension_stated(compiler):
    """"Where are our largest concentrations today?" ranks the governed
    concentration tests, and which those are is the capability's to know."""
    intent = build_intent(capability="concentration", operation="rank",
                          measures=[{"concept": "concentration_exposure"}],
                          time={"form": "current"})
    result = compiler.compile(intent)
    assert result.outcome == OUTCOME_PLAN, result.codes()


def test_a_GENERIC_rank_still_needs_something_to_rank(compiler):
    intent = build_intent(operation="rank", measures=[{"concept": "balance"}])
    result = compiler.compile(intent)
    assert result.outcome != OUTCOME_PLAN
    assert "UNSUPPORTED_COMPOSITION" in result.codes()


# --------------------------------------------------------------------------- #
# 5 · the vocabulary declares governed value lists, and their absence
# --------------------------------------------------------------------------- #

def test_a_dimension_with_a_governed_enum_shows_its_business_values():
    from mi_agent.interpretation_v2.metadata import GovernedMetadataService

    vocabulary = load_governed_vocabulary()
    service = GovernedMetadataService(vocabulary)
    result = service.get_allowed_values("collateral_type")
    assert result["has_governed_values"] is True
    assert result["values"]
    # Business spellings, never the ESMA codes behind them.
    assert not any(v.isupper() and len(v) == 4 for v in result["values"])


def test_the_product_and_stage_gap_the_first_runs_closed_over():
    """Runs 1-3's largest finding, and where it actually lived.

    Sixteen of twenty clarifications were the interpreter declining to assert
    "drawdown" or "Offer", because ``fields_registry.allowed_values`` is null
    for product type and pipeline stage. The values were governed all along, in
    two sources the vocabulary was not reading: the asset profile's
    ``match.product_type``, and the estate's one question-side stage
    vocabulary. Reading them is the addendum applied.
    """
    from mi_agent.interpretation_v2.metadata import GovernedMetadataService

    service = GovernedMetadataService(load_governed_vocabulary())

    products = service.get_allowed_values("erm_product_type")
    assert products["has_governed_values"] is True
    assert "drawdown" in products["values"]
    assert "lump_sum" in products["values"]

    stages = service.get_allowed_values("pipeline_stage")
    assert stages["has_governed_values"] is True
    assert {"KFI", "APPLICATION", "OFFER", "COMPLETED"} <= set(stages["values"])


def test_a_dimension_with_no_governed_enum_still_says_so():
    """Absence must be reported, not left as silence.

    Some dimensions genuinely have no governed value list. The tool says so
    explicitly and tells the interpreter what to do about it, because silence
    reads as "any value is fine" — which is how a guess gets asserted.
    """
    from mi_agent.interpretation_v2.metadata import GovernedMetadataService

    vocabulary = load_governed_vocabulary()
    service = GovernedMetadataService(vocabulary)
    ungoverned = [c.concept_id for c in vocabulary.concepts.values()
                  if c.role == "dimension" and not c.values
                  and not c.owning_capability]
    assert ungoverned, "premise: some dimensions carry no governed values"

    result = service.get_allowed_values(ungoverned[0])
    assert result["found"] is True
    assert result["has_governed_values"] is False
    assert "Do NOT assert a filter value" in result["guidance"]


def test_the_governed_defaults_are_declared_to_the_model():
    """An empty slot is only safe if the model is told which slots have
    defaults. On the first run nothing said so, and the interpreter blocked on
    every bare "region" it saw."""
    vocabulary = load_governed_vocabulary()
    declared = vocabulary.orientation_payload()["governed_defaults"]
    assert declared == dict(GOVERNED_DEFAULTS)
    for slot in ("geography.basis", "geography.level", "measures[].statistic",
                 "measures[].weight", "population.base", "time.form"):
        assert slot in declared

    # And what it declares must be true of the compiler.
    assert "nuts3" in declared["geography.basis"], (
        "the declaration must warn that NUTS3 and ITL3 have NO default")


def test_the_declared_geography_default_matches_what_the_compiler_does(compiler):
    intent = build_intent(operation="breakdown",
                          geography={"requested": True, "group_by": True})
    result = compiler.compile(intent)
    assert result.outcome == OUTCOME_PLAN
    assert result.plan.geography.resolved_level == "reporting"
    assert result.plan.geography.canonical_field == "canonical_region_reporting"
    assert result.plan.geography.defaulted is True
