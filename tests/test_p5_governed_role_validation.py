"""tests/test_p5_governed_role_validation.py — a proposed role is advisory.

Two semantic type errors turned five correct answers into refusals when the
constrained concept-merge arm was activated. Both had the same shape: the model
proposed a role, the binder bound it to the one governed owner, and the merge
wrote it into a slot the REQUESTED OPERATION has no use for.

    "When does the funded book reach the £100m milestone?"
        £100m is an aggregate target on the portfolio total. Bound to the one
        governed owner of "balance" it became `current_outstanding_balance >=
        100000000` — a row predicate selecting loans each worth £100m. The
        contract already held the target as `forecast_target_value`.

    "What proportion of the book is in the acquired portfolio?"
        `share` reports one population as a fraction of another. The concept
        belongs to the population; it was written as a grouping axis, which the
        operation never consumes, and the receipt guard correctly refused.

The rule these tests defend:

    a model-proposed role is advisory; the governed contract decides whether
    that role exists for the operation the question already selected.

Both refusals FAIL CLOSED. Neither moves a concept to a different role to make
it execute, and neither touches the downstream guard — a dimension declined
here is a dimension the contract never carries, so the guard that catches a
genuinely dropped axis still catches one.

Run: python -m pytest tests/test_p5_governed_role_validation.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from question_interpretation import claim_merge as CM  # noqa: E402

BALANCE = "current_outstanding_balance"
LTV = "current_loan_to_value"


def _spec(**kw):
    base = dict(aggregation="sum", metric=None, forecast_target_value=None)
    base.update(kw)
    return SimpleNamespace(**base)


def _bound(kind, term, field, value, operator=None):
    """A bound concept, in the shape `concept_proposal.bind` returns."""
    return SimpleNamespace(
        proposal=SimpleNamespace(kind=kind, term=term, covers=term),
        field=field, value=value, operator=operator)


def _outcomes(result):
    return [f.outcome for f in result.findings]


# --------------------------------------------------------------------------- #
# The profile is read off the contract, never off the question
# --------------------------------------------------------------------------- #
def test_a_contract_with_a_milestone_declares_an_aggregate_target():
    p = CM.operation_profile(_spec(forecast_target_value=100000000.0))
    assert p.holds_aggregate_target
    assert (p.aggregate_target_field, p.aggregate_target_value) == (BALANCE, 1e8)


def test_an_ordinary_contract_declares_no_target_and_accepts_an_axis():
    p = CM.operation_profile(_spec())
    assert not p.holds_aggregate_target
    assert p.accepts_grouping_axis


def test_a_share_operation_has_no_grouping_axis_to_give():
    assert not CM.operation_profile(_spec(aggregation="share")).accepts_grouping_axis


def test_no_spec_means_no_opinion():
    p = CM.operation_profile(None)
    assert not p.holds_aggregate_target and p.accepts_grouping_axis


# --------------------------------------------------------------------------- #
# Aggregate target vs row predicate — the four the brief names
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("target", [100000000.0, 250000000.0])
def test_a_milestone_is_the_target_the_contract_holds_not_a_row_predicate(target):
    """'portfolio reaches £100m' / 'funded book reaches £250m' -> aggregate."""
    profile = CM.operation_profile(_spec(forecast_target_value=target))
    result = CM.merge([], [_bound("threshold", "balance", BALANCE, target, "ge")],
                      profile=profile)
    assert _outcomes(result) == [CM.AGREED]
    # THE INVARIANT: nothing is written, so no row predicate can reach the spec.
    assert result.filled_by_model == ()


@pytest.mark.parametrize("field,value,op", [
    (BALANCE, 500000.0, "gt"),          # loans above £500k
    (LTV, 50.0, "gt"),                  # loans with LTV above 50%
])
def test_a_genuine_row_condition_stays_a_row_predicate(field, value, op):
    """No target in the contract, so the ordinary rules decide — and do."""
    result = CM.merge([], [_bound("threshold", "size", field, value, op)],
                      profile=CM.operation_profile(_spec()))
    assert _outcomes(result) == [CM.FILLED_BY_MODEL]
    filled = result.filled_by_model
    assert len(filled) == 1
    assert (filled[0].slot, filled[0].key, filled[0].value) == (
        CM.SLOT_ROW_PREDICATES, field, value)


def test_a_row_condition_on_another_measure_survives_a_milestone_question():
    """The target owns ITS measure, not every measure."""
    profile = CM.operation_profile(_spec(forecast_target_value=1e8))
    result = CM.merge([], [_bound("threshold", "ltv", LTV, 50.0, "gt")],
                      profile=profile)
    assert _outcomes(result) == [CM.FILLED_BY_MODEL]


def test_a_second_number_on_the_targets_measure_fails_closed():
    """Ambiguous between a further target and a row condition. Neither is picked."""
    profile = CM.operation_profile(_spec(forecast_target_value=1e8))
    result = CM.merge([], [_bound("threshold", "balance", BALANCE, 500000.0, "gt")],
                      profile=profile)
    assert _outcomes(result) == [CM.DECLINED_AGGREGATE_TARGET]
    assert result.filled_by_model == ()


# --------------------------------------------------------------------------- #
# Population qualifier vs grouping axis
# --------------------------------------------------------------------------- #
def test_a_share_operation_refuses_a_grouping_axis():
    """CFO63/CFO65: the axis the operation never consumes is never written."""
    profile = CM.operation_profile(_spec(aggregation="share", metric=BALANCE))
    result = CM.merge([], [_bound("dimension", "direct or acquired",
                                  "source_portfolio_type", None)],
                      profile=profile)
    assert _outcomes(result) == [CM.DECLINED_ROLE_NOT_IN_OPERATION]
    assert result.filled_by_model == ()


def test_a_share_operation_still_takes_a_population_qualifier():
    """The concept is not blocked — only the role it cannot have is."""
    profile = CM.operation_profile(_spec(aggregation="share", metric=BALANCE))
    result = CM.merge([], [_bound("category_value", "drawdown",
                                  "erm_product_type", "drawdown")],
                      profile=profile)
    assert _outcomes(result) == [CM.FILLED_BY_MODEL]
    assert result.filled_by_model[0].slot == CM.SLOT_ROW_PREDICATES


@pytest.mark.parametrize("aggregation", ["sum", "avg", "count", "weighted_avg"])
def test_every_other_operation_still_takes_a_grouping_axis(aggregation):
    """'Show balance by product type.' / 'by portfolio.' — dimensions survive."""
    profile = CM.operation_profile(_spec(aggregation=aggregation))
    result = CM.merge([], [_bound("dimension", "product type",
                                  "erm_product_type", None)],
                      profile=profile)
    assert _outcomes(result) == [CM.FILLED_BY_MODEL]
    assert result.filled_by_model[0].slot == CM.SLOT_DIMENSIONS


# --------------------------------------------------------------------------- #
# Nothing else moves
# --------------------------------------------------------------------------- #
def test_rules_one_to_three_are_untouched():
    """A filled slot is still never overwritten, whatever the profile says."""
    existing = [CM.SlotValue(CM.SLOT_DIMENSIONS, "erm_product_type",
                             "erm_product_type", CM.PROV_EXPLICIT_USER)]
    result = CM.merge(existing, [_bound("dimension", "geography",
                                        "erm_product_type", None)],
                      profile=CM.operation_profile(_spec()))
    assert _outcomes(result) == [CM.AGREED]


def test_no_profile_behaves_exactly_as_before():
    """The parameter is optional and its absence changes nothing."""
    bound = [_bound("threshold", "balance", BALANCE, 1e8, "ge"),
             _bound("dimension", "product type", "erm_product_type", None)]
    assert _outcomes(CM.merge([], bound)) == [CM.FILLED_BY_MODEL,
                                              CM.FILLED_BY_MODEL]


def test_a_declined_role_is_a_finding_never_a_silence():
    """Proposed-and-refused must stay distinguishable from never-proposed."""
    profile = CM.operation_profile(_spec(aggregation="share"))
    result = CM.merge([], [_bound("dimension", "x", "source_portfolio_type", None)],
                      profile=profile)
    assert len(result.findings) == 1
    assert result.findings[0].as_dict()["detail"]


# --------------------------------------------------------------------------- #
# A field the reader has already placed
# --------------------------------------------------------------------------- #
def test_a_field_already_narrowed_on_is_not_also_an_axis():
    """'What is the total balance for North loans?'

    The contract narrows on `geographic_region_obligor`. Offering that same
    field as a breakdown axis re-places a concept the reader has placed; the
    route computes a filtered total, never groups, and the guard refuses.
    """
    existing = [CM.SlotValue(CM.SLOT_ROW_PREDICATES, "geographic_region_obligor",
                             "North", CM.PROV_EXPLICIT_USER)]
    result = CM.merge(existing,
                      [_bound("dimension", "region",
                              "geographic_region_obligor", None)],
                      profile=CM.operation_profile(_spec()))
    assert _outcomes(result) == [CM.DECLINED_FIELD_ALREADY_PLACED]
    assert result.filled_by_model == ()


def test_the_mirror_direction_is_left_alone():
    """'Balance by region for London loans.'

    A field held as an AXIS with the value LOST is the recovery this arm exists
    for: the deterministic path refuses saying the scope was not applied, and
    the model supplying `London` is what answers it. Declining this direction
    too would take back seven correct answers to buy two.
    """
    existing = [CM.SlotValue(CM.SLOT_DIMENSIONS, "geographic_region_obligor",
                             "geographic_region_obligor", CM.PROV_EXPLICIT_USER)]
    result = CM.merge(existing,
                      [_bound("category_value", "london",
                              "geographic_region_obligor", "London")],
                      profile=CM.operation_profile(_spec()))
    assert _outcomes(result) == [CM.FILLED_BY_MODEL]
    assert result.filled_by_model[0].slot == CM.SLOT_ROW_PREDICATES


def test_an_axis_on_a_field_nothing_narrows_still_fills():
    """The rule is about a field ALREADY placed, not about axes in general."""
    existing = [CM.SlotValue(CM.SLOT_ROW_PREDICATES, "erm_product_type",
                             "drawdown", CM.PROV_EXPLICIT_USER)]
    result = CM.merge(existing,
                      [_bound("dimension", "geography",
                              "geographic_region_obligor", None)],
                      profile=CM.operation_profile(_spec()))
    assert _outcomes(result) == [CM.FILLED_BY_MODEL]
    assert result.filled_by_model[0].slot == CM.SLOT_DIMENSIONS
