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