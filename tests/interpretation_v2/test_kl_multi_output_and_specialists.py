"""K, L · Multi-output and specialist capabilities are representable from day one.

K. "How many joint borrower loans are there, what balance do they represent, and
how much of that balance is above 40% LTV?" is ONE governed population with
THREE outputs, the third carrying an output-local predicate. Representing it as
three unrelated questions is how the populations quietly stop matching.

L. "How much room is left to draw against the facility?" names a capability and
an operation. It does NOT decompose into eligible balances, advance rates,
reserves and headroom arithmetic — the borrowing-base capability owns all of
that, and the interpreter is never shown it.
"""

from __future__ import annotations

import json

import pytest

from mi_agent.interpretation_v2 import (
    OUTCOME_PLAN,
    OUTCOME_REFUSE,
    load_governed_vocabulary,
)
from mi_agent.interpretation_v2.vocabulary import SPECIALIST_MEASURES

from .conftest import build_intent


# --------------------------------------------------------------------------- #
# K · multi-output
# --------------------------------------------------------------------------- #

def test_one_population_three_outputs_with_an_output_local_predicate(compiler):
    intent = build_intent(
        operation="point_in_time",
        measures=[],
        filters=[{"concept": "number_of_borrowers", "comparator": "gte",
                  "value": 2}],
        outputs=[
            {"id": "loan_count",
             "measures": [{"concept": "loan", "statistic": "count"}]},
            {"id": "total_balance",
             "measures": [{"concept": "balance", "statistic": "sum"}]},
            {"id": "balance_above_40_ltv",
             "measures": [{"concept": "balance", "statistic": "sum"}],
             "filters": [{"concept": "current_ltv", "comparator": "gt",
                          "value": 40}]},
        ])
    result = compiler.compile(intent)
    assert result.outcome == OUTCOME_PLAN, result.codes()
    plan = result.plan

    # ONE population, shared by all three.
    assert plan.population.base == "funded"
    assert len(plan.filters) == 1
    assert plan.filters[0].canonical_field == "number_of_borrowers"

    # THREE outputs, and only the third narrows further.
    assert [o.id for o in plan.outputs] == ["loan_count", "total_balance",
                                            "balance_above_40_ltv"]
    assert plan.outputs[0].filters == ()
    assert plan.outputs[1].filters == ()
    assert len(plan.outputs[2].filters) == 1
    assert plan.outputs[2].filters[0].canonical_field == "current_loan_to_value"


def test_a_single_figure_question_still_has_exactly_one_output(compiler):
    """The one-question/one-metric limitation is not baked in anywhere: a simple
    question is the general shape with one entry, not a different shape."""
    result = compiler.compile(build_intent(measures=[{"concept": "balance"}]))
    assert result.outcome == OUTCOME_PLAN
    assert len(result.plan.outputs) == 1
    assert result.plan.outputs[0].id == "primary"


def test_duplicate_output_ids_refuse(compiler):
    intent = build_intent(
        measures=[],
        outputs=[{"id": "same", "measures": [{"concept": "balance"}]},
                 {"id": "same", "measures": [{"concept": "loan"}]}])
    result = compiler.compile(intent)
    assert result.outcome == OUTCOME_REFUSE
    assert "CONFLICTING_CLAIMS" in result.codes()


def test_an_unbindable_output_refuses_the_whole_plan(compiler):
    """Two good outputs and one bad one is not two-thirds of an answer."""
    intent = build_intent(
        measures=[],
        outputs=[{"id": "a", "measures": [{"concept": "balance"}]},
                 {"id": "b", "measures": [{"concept": "ebitda"}]}])
    result = compiler.compile(intent)
    assert result.outcome == OUTCOME_REFUSE
    assert result.plan is None


# --------------------------------------------------------------------------- #
# L · specialist capabilities
# --------------------------------------------------------------------------- #

def test_headroom_is_a_capability_and_an_operation_not_an_arithmetic(compiler):
    intent = build_intent(capability="borrowing_base", operation="headroom",
                          measures=[{"concept": "borrowing_base_headroom"}],
                          time={"form": "current"})
    result = compiler.compile(intent)
    assert result.outcome == OUTCOME_PLAN, result.codes()

    measure = result.plan.outputs[0].measures[0]
    assert measure.capability_owner == "borrowing_base"
    assert measure.canonical_field is None, (
        "a specialist measure must not resolve to a column — the methodology "
        "owns it")
    assert measure.statistic == "capability"


def test_the_interpreter_is_never_shown_the_specialist_arithmetic(vocabulary):
    """Opus can NAME the borrowing base. It cannot learn how one is built.

    Checked across the whole retrieval surface, not just the standing prompt:
    the metadata tools are how the model reaches anything now, so an internal
    that leaked through one of them would leak just as surely.
    """
    from mi_agent.interpretation_v2.metadata import GovernedMetadataService

    service = GovernedMetadataService(vocabulary)
    surface = json.dumps([
        vocabulary.orientation_payload(),
        service.search_concepts("borrowing base", limit=40),
        service.get_concept_metadata("borrowing_base_headroom"),
        service.get_capability_metadata("borrowing_base"),
        service.search_capabilities("borrowing", limit=40),
        service.get_asset_metadata(),
    ], default=str)
    for internal in ("advance_rate", "advance rate", "haircut",
                     "eligible_balance_formula", "concentration_limit_deduction",
                     "reserve_amount"):
        assert internal not in surface, (
            f"the specialist's internals leaked to the model: {internal!r}")

    headroom = vocabulary.resolve("borrowing_base_headroom")
    assert headroom is not None
    assert headroom.is_specialist
    assert headroom.allowed_statistics == (), (
        "a specialist measure must offer no statistic to choose from")


def test_a_statistic_cannot_be_imposed_on_a_specialist_measure(compiler):
    intent = build_intent(capability="borrowing_base", operation="headroom",
                          measures=[{"concept": "borrowing_base_headroom",
                                     "statistic": "average"}])
    result = compiler.compile(intent)
    assert result.outcome == OUTCOME_REFUSE
    assert "UNSUPPORTED_STATISTIC" in result.codes()


def test_a_weight_cannot_be_imposed_on_a_specialist_measure(compiler):
    intent = build_intent(capability="borrowing_base", operation="point_in_time",
                          measures=[{"concept": "borrowing_base",
                                     "weight": "balance"}])
    result = compiler.compile(intent)
    assert result.outcome == OUTCOME_REFUSE
    assert "WEIGHT_NOT_PERMITTED" in result.codes()


def test_a_specialist_measure_cannot_share_an_output_with_a_composed_one(compiler):
    """Two owners cannot both produce one figure."""
    intent = build_intent(
        capability="borrowing_base", operation="point_in_time",
        measures=[{"concept": "borrowing_base"},
                  {"concept": "balance", "statistic": "sum"}])
    result = compiler.compile(intent)
    assert result.outcome == OUTCOME_REFUSE
    assert "UNSUPPORTED_COMPOSITION" in result.codes()


@pytest.mark.parametrize("capability,operation,measure", [
    ("borrowing_base", "headroom", "borrowing_base_headroom"),
    ("borrowing_base", "utilisation", "facility_utilisation"),
    ("borrowing_base", "eligibility", "ineligible_balance"),
    ("funded_bridge", "bridge", "funded_balance_movement"),
    ("pipeline", "summary", "pipeline_amount"),
    ("forecast", "forecast_projection", "forecast_funded_balance"),
    ("pipeline_stage_movement", "transition", "cases_moved"),
])
def test_every_shipped_specialist_capability_is_representable(
        capability, operation, measure, compiler):
    time = ({"form": "relative_pair"}
            if operation in ("bridge", "transition") else
            {"form": "forward_looking"} if capability == "forecast" else
            {"form": "current"})
    # A `funded_bridge` bridge is the ATTRIBUTION form, and since the
    # completeness gate a change request must say so. Stated only for the
    # capability that is one of the four forms; every other specialist here is
    # change-shaped in its own right and owns no `change_form`, so asking it for
    # one would be wrong.
    change_form = "attribution" if capability == "funded_bridge" else None
    intent = build_intent(capability=capability, operation=operation,
                          change_form=change_form,
                          measures=[{"concept": measure}], time=time,
                          population={"base": "pipeline", "lens": "all",
                                      "seasoning": "any"}
                          if capability in ("pipeline",
                                            "pipeline_stage_movement")
                          else {"base": "funded", "lens": "all",
                                "seasoning": "any"})
    result = compiler.compile(intent)
    assert result.outcome == OUTCOME_PLAN, (capability, operation,
                                            result.codes())
    assert result.plan.outputs[0].measures[0].capability_owner == capability


#: Capabilities that COMPOSE from governed measures rather than owning a
#: methodology. `generic_analysis` is the composed case by definition, and a
#: period movement is a governed measure over a governed period pair — there is
#: no proprietary arithmetic to hide, so neither needs specialist measures.
COMPOSING_CAPABILITIES = frozenset({"generic_analysis", "period_movement"})


def test_specialist_measures_are_declared_for_every_capability_that_owns_one():
    vocabulary = load_governed_vocabulary()
    specialists = set(vocabulary.capabilities) - COMPOSING_CAPABILITIES
    assert specialists, "no specialist capabilities — this would pass vacuously"
    assert specialists <= set(SPECIALIST_MEASURES), (
        "a capability with no specialist measures would force the interpreter "
        "to compose it from columns: "
        f"{sorted(specialists - set(SPECIALIST_MEASURES))}")
