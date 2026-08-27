#!/usr/bin/env python3
"""The completeness check: does a stated concept survive into the contract?

These tests exercise the COMPARISON, which is the only thing
`question_interpretation.completeness` owns. The readings themselves come from
owners that already ship and have their own tests; what is asserted here is
that a concept present in the sentence and absent from the executed contract is
reported, that a concept present in both is not, and that the three ways a
concept can be present WITHOUT looking present — a scope carried by the lens, a
grouping bound as a filter, a measure supplied by a route — do not read as loss.
"""
from __future__ import annotations

import pytest

from question_interpretation.completeness import (
    ExecutedContract, StatedConcept, from_envelope, unresolved_concepts)


def _c(kind, field, value="", term="", owner="test"):
    return StatedConcept(kind, field, value or field, term, owner)


# --------------------------------------------------------------------------- #
# The defect the check exists to catch
# --------------------------------------------------------------------------- #
def test_a_dropped_categorical_value_is_reported():
    """"How many drawdown loans have LTV above 50%?" applied the threshold,
    dropped `drawdown` and answered over the whole book."""
    stated = [_c("value", "erm_product_type", "drawdown"),
              _c("facet:threshold", "", "LTV over 50")]
    contract = ExecutedContract(
        filters=("current_loan_to_value",),
        facets=(("threshold", "LTV over 50", "applied"),))
    lost = unresolved_concepts(stated, contract)
    assert [c.field for c in lost] == ["erm_product_type"]


def test_a_carried_categorical_value_is_not_reported():
    stated = [_c("value", "erm_product_type", "drawdown")]
    contract = ExecutedContract(filters=("erm_product_type",))
    assert unresolved_concepts(stated, contract) == []


def test_a_dropped_dataset_is_reported():
    """"Summarise the current pipeline" was DECIDED as pipeline and
    RECONCILED against funded. The contradiction is already in the envelope."""
    stated = [_c("dataset", "view", "pipeline")]
    routed = ExecutedContract(dataset_context="pipeline", route="portfolio_summary",
                              dataset_reconciled="funded")
    assert [c.value for c in unresolved_concepts(stated, routed)] == ["pipeline"]


def test_a_dataset_the_answer_reconciled_against_is_not_reported():
    stated = [_c("dataset", "view", "pipeline")]
    served = ExecutedContract(dataset_context="pipeline", dataset_reconciled="pipeline")
    assert unresolved_concepts(stated, served) == []


def test_a_dataset_nothing_reconciled_against_is_reported():
    """A refusal reconciles against nothing, and the decision alone is not
    evidence that anything was read."""
    stated = [_c("dataset", "view", "pipeline")]
    assert len(unresolved_concepts(stated, ExecutedContract(
        dataset_context="pipeline"))) == 1


# --------------------------------------------------------------------------- #
# Present without looking present
# --------------------------------------------------------------------------- #
def test_a_scope_value_carried_by_the_lens_is_not_a_row_filter():
    """The Direct book is a LENS. Requiring it as a row filter made every
    scoped question look incomplete."""
    stated = [_c("value", "source_portfolio_type", "direct")]
    contract = ExecutedContract(scope_context="direct",
                                filters=("source_portfolio_id",))
    assert unresolved_concepts(stated, contract) == []


def test_a_grouping_bound_as_a_filter_is_a_role_change_not_a_loss():
    """"How many owner occupied loans do we have?" makes the axis owner ask for
    `occupancy_type` as a GROUPING; the contract binds it as a filter. The
    concept reached the contract. Conflating the two cost five false
    positives on the composition banks."""
    stated = [_c("facet:grouping_dimension", "occupancy_type", "owner occupied")]
    contract = ExecutedContract(filters=("occupancy_type",))
    assert unresolved_concepts(stated, contract) == []


def test_a_route_that_supplies_its_own_measure_is_not_a_loss():
    stated = [_c("measure", "current_outstanding_balance")]
    bridge = ExecutedContract(metric=None, route="funded_bridge")
    assert unresolved_concepts(stated, bridge) == []


def test_a_measure_bound_to_a_DIFFERENT_field_is_a_loss():
    """Q21B asked for balance growth and bound the measure to
    `current_loan_to_value` from the words "50% LTV"."""
    stated = [_c("measure", "current_outstanding_balance")]
    misbound = ExecutedContract(metric="current_loan_to_value",
                                route="period_change_analysis")
    assert [c.field for c in unresolved_concepts(stated, misbound)] == \
        ["current_outstanding_balance"]


def test_a_field_the_sentence_used_as_a_filter_is_not_a_lost_measure():
    """"balance for loans with borrower age above 75" makes the measure owner
    read `borrower age`; the contract binds it as a filter field."""
    stated = [_c("measure", "youngest_borrower_age")]
    contract = ExecutedContract(metric="current_outstanding_balance",
                                filters=("youngest_borrower_age",))
    assert unresolved_concepts(stated, contract) == []


# --------------------------------------------------------------------------- #
# Silence is a finding, not a pass
# --------------------------------------------------------------------------- #
def test_a_resolved_scope_with_no_recorded_narrowing_is_reported():
    """Q19C published `portfolioScope.context_id = direct` and answered the
    whole book. A scope RESOLVED is not a scope APPLIED."""
    stated = [_c("scope", "portfolio_lens", "direct")]
    declared_only = ExecutedContract(scope_context="direct",
                                     route="period_change_analysis")
    assert len(unresolved_concepts(stated, declared_only)) == 1


def test_an_applied_grouping_axis_is_not_evidence_that_a_scope_was_applied():
    """`applied_fields` carries the field of every applied facet, axes
    included. Counting it as narrowing evidence let a concentration answer's
    `erm_product_type` AXIS vouch for a Direct scope the route never applied."""
    stated = [_c("scope", "portfolio_lens", "direct")]
    axis_only = ExecutedContract(scope_context="direct",
                                 applied_fields=("erm_product_type",),
                                 facets=(("grouping_dimension", "product", "applied"),))
    assert len(unresolved_concepts(stated, axis_only)) == 1


def test_a_scope_with_a_recorded_narrowing_is_not_reported():
    stated = [_c("scope", "portfolio_lens", "direct")]
    applied = ExecutedContract(scope_context="direct", narrowed=True)
    assert unresolved_concepts(stated, applied) == []


# --------------------------------------------------------------------------- #
# The adapter
# --------------------------------------------------------------------------- #
def test_from_envelope_reads_the_estate_s_own_record():
    env = {
        "spec": {"metric": "current_outstanding_balance",
                 "filters": {"erm_product_type": "drawdown"},
                 "dimension": "geographic_region_obligor",
                 "dimensions": []},
        "metadata": {"route": "period_change_analysis",
                     "datasetContext": "funded",
                     "populationApplied": {"applied": ["erm_product_type"],
                                           "rowsBefore": 640, "rowsAfter": 244}},
        "executionSummary": {"narrowed": True, "populationTotal": 640,
                             "facets": [{"kind": "threshold", "label": "LTV over 50",
                                         "field": "current_loan_to_value",
                                         "status": "applied"}]},
        "portfolioScope": {"context_id": "direct"},
    }
    contract = from_envelope(env)
    assert contract.filters == ("erm_product_type",)
    assert contract.dimensions == ("geographic_region_obligor",)
    assert contract.metric == "current_outstanding_balance"
    assert contract.scope_context == "direct"
    assert contract.population_applied is True
    assert contract.narrowing_recorded is True
    assert contract.facet_applied("threshold", "LTV over 50") is True
    assert contract.facet_applied("threshold", "LTV over 90") is False
    assert contract.applied_fields == ("current_loan_to_value",)


def test_an_empty_envelope_carries_nothing_and_reports_everything():
    """A refusal records no execution. Everything stated is unrecorded, and the
    check says so rather than treating an absent record as a pass."""
    stated = [_c("value", "erm_product_type", "drawdown"),
              _c("dimension", "ltv_bucket", "ltv_bucket")]
    assert len(unresolved_concepts(stated, from_envelope({}))) == 2


# --------------------------------------------------------------------------- #
# The readings themselves, against the real owners
# --------------------------------------------------------------------------- #
def test_stated_concepts_delegates_and_never_invents_a_field():
    from migration_phase0.assurance_semantics import load_assurance_semantics
    from question_interpretation.completeness import stated_concepts

    semantics = load_assurance_semantics()
    values = {"erm_product_type": {"drawdown": "drawdown", "lump_sum": "lump_sum"}}
    stated = stated_concepts("How many drawdown loans have LTV above 50%?",
                             semantics, available_values=values)
    kinds = {c.kind for c in stated}
    assert "value" in kinds, stated
    assert any(c.field == "erm_product_type" for c in stated)
    fields = {c.field for c in stated if c.field}
    assert fields <= set(semantics.get("fields") or {}) | {"portfolio_lens", "view"}, \
        "a reading named a field the registry does not carry"


def test_the_target_owner_is_gated_by_the_answer_type_owner():
    """£300,000 in "how many loans are above £300,000" is a THRESHOLD. The
    forecast owner reads it as a milestone and the answer-type owner is what
    stops it being reported as a lost forecast target."""
    from migration_phase0.assurance_semantics import load_assurance_semantics
    from question_interpretation.completeness import stated_concepts

    semantics = load_assurance_semantics()
    threshold = stated_concepts("How many loans are above £300,000?", semantics)
    assert not [c for c in threshold if c.kind == "target"]

    milestone = stated_concepts("At the current trajectory, when do we get to "
                                "£100 million?", semantics)
    assert [c.value for c in milestone if c.kind == "target"] == ["100000000.0"]


def test_the_value_owner_claims_its_spans_before_the_scope_owner():
    """`Gamma Direct` is a broker. Reading `direct` out of it raised a scope
    the reader never named, on a question that answers correctly."""
    from migration_phase0.assurance_semantics import load_assurance_semantics
    from question_interpretation.completeness import stated_concepts

    semantics = load_assurance_semantics()
    values = {"broker_channel": {"gamma direct": "Gamma Direct"}}
    stated = stated_concepts("How many Gamma Direct loans do we have?", semantics,
                             available_values=values)
    assert not [c for c in stated if c.kind == "scope"], stated
