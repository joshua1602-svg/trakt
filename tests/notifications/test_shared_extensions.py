"""The two shared-service extensions the notification needed.

Both are additive: React, the investor deck and Copilot get them too. These
tests assert both that they work and that they changed nothing that already
existed, which is the whole basis on which they were made in the shared layer
rather than in the Teams layer.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mi_agent.concentration_tests import forward as forward_mod
from mi_agent_api import evolution as evolution_mod


def _periods(*values):
    return [{"metrics": {"pipeline_amount": amount,
                         "pipeline_case_count": count,
                         "weighted_expected_funded_amount": weighted}}
            for amount, count, weighted in values]


# --------------------------------------------------------------------------- #
# Total-pipeline five-week average
# --------------------------------------------------------------------------- #
def test_five_week_average_uses_the_trailing_window():
    block = evolution_mod.five_week_average(_periods(
        (100.0, 10, 50.0), (200.0, 20, 100.0), (300.0, 30, 150.0),
        (400.0, 40, 200.0), (500.0, 50, 250.0)))
    assert block["available"] is True
    assert block["average"]["pipeline_amount"] == 300.0
    assert block["average"]["pipeline_case_count"] == 30.0
    assert block["weeksObserved"] == 5


def test_only_the_last_five_weeks_count():
    block = evolution_mod.five_week_average(_periods(
        (1.0, 1, 1.0), (100.0, 10, 50.0), (200.0, 20, 100.0),
        (300.0, 30, 150.0), (400.0, 40, 200.0), (500.0, 50, 250.0)))
    assert block["average"]["pipeline_amount"] == 300.0
    assert block["weeksObserved"] == 5


def test_a_short_history_reports_how_many_weeks_it_had():
    block = evolution_mod.five_week_average(_periods(
        (100.0, 10, 50.0), (200.0, 20, 100.0)))
    assert block["average"]["pipeline_amount"] == 150.0
    assert block["weeksObserved"] == 2


def test_no_history_is_unavailable_rather_than_zero():
    """The difference matters: a caller must omit the comparison, not compare
    against a fabricated baseline."""
    block = evolution_mod.five_week_average([])
    assert block["available"] is False
    assert block["average"]["pipeline_amount"] is None
    assert block["differencePct"]["pipeline_amount"] is None


def test_the_difference_is_signed_relative_to_the_average():
    block = evolution_mod.five_week_average(_periods(
        (100.0, 10, 50.0), (100.0, 10, 50.0), (100.0, 10, 50.0),
        (100.0, 10, 50.0), (150.0, 15, 75.0)))
    # current 150 against a trailing mean of 110 → +36.4%
    assert block["current"]["pipeline_amount"] == 150.0
    assert block["average"]["pipeline_amount"] == 110.0
    assert block["differencePct"]["pipeline_amount"] == pytest.approx(36.4)


def test_missing_weeks_are_skipped_not_treated_as_zero():
    block = evolution_mod.five_week_average([
        {"metrics": {"pipeline_amount": 100.0, "pipeline_case_count": 10,
                     "weighted_expected_funded_amount": None}},
        {"metrics": {"pipeline_amount": None, "pipeline_case_count": 20,
                     "weighted_expected_funded_amount": 100.0}},
        {"metrics": {"pipeline_amount": 300.0, "pipeline_case_count": 30,
                     "weighted_expected_funded_amount": 150.0}},
    ])
    assert block["average"]["pipeline_amount"] == 200.0
    assert block["weeksObservedByMetric"]["pipeline_amount"] == 2
    assert block["weeksObservedByMetric"]["pipeline_case_count"] == 3


def test_the_basis_matches_the_funnels_published_convention():
    """A card and the funnel panel must not describe the same window
    differently."""
    block = evolution_mod.five_week_average(_periods((100.0, 10, 50.0)))
    assert block["window"] == evolution_mod._CONVERSION_WINDOW
    assert "trailing mean" in block["basis"]
    assert "5" in block["basis"]


# --------------------------------------------------------------------------- #
# Concentration driver contributors
# --------------------------------------------------------------------------- #
class _Metric:
    metric_id = "share_of_balance"
    parameters: dict = {}


class _Computation:
    def __init__(self, frame, mask, denominator, numerator):
        self.mask = mask
        self.data_status = forward_mod.DATA_OK
        self.denominator_value = denominator
        self.numerator_value = numerator
        self.resolved_columns = {"dimension": "region"}
        self.notes = ""


class _Library:
    def get(self, _metric_id):
        return _Metric()

    def role(self, role):
        class _Spec:
            candidates = {"balance_current": ["balance"],
                          "broker": ["broker"],
                          "region": ["region"]}.get(role, [])
        return _Spec()


class _Test:
    test_id = "region_sw"
    display_name = "South West exposure"
    metric_id = "share_of_balance"
    parameters: dict = {}
    threshold = 25.0
    operator = "max"
    warning_fraction = 0.9


@pytest.fixture
def expected_frame():
    """A three-state frame: two funded rows and four pipeline cases."""
    return pd.DataFrame({
        "balance": [100.0, 100.0, 6_000_000.0, 2_200_000.0, 3_900_000.0,
                    2_000_000.0],
        forward_mod.PROBABILITY_COL: [1.0, 1.0, 0.8, 0.5, 0.6, 0.4],
        "broker": ["Broker A", "Broker B", "Broker A", "Broker A", "Broker B",
                   "Broker B"],
        "region": ["South West", "South East", "South West", "South West",
                   "South West", "South East"],
        forward_mod.STATE_COMPONENT_COL: ["funded", "funded", "pipeline",
                                          "pipeline", "pipeline", "pipeline"],
    })


@pytest.fixture
def patched(monkeypatch, expected_frame):
    def _evaluate(frame, _lib, _metric, _params, external=None):
        mask = frame["region"] == "South West"
        return _Computation(frame, mask, denominator=100_000_000.0,
                            numerator=23_500_000.0)

    monkeypatch.setattr(forward_mod, "evaluate_metric", _evaluate)
    monkeypatch.setattr(forward_mod, "treatment_for",
                        lambda _lib, _test: forward_mod.TREATMENT_EXPOSURE)
    monkeypatch.setattr(forward_mod, "resolve_role_column",
                        lambda frame, _lib, role:
                        {"balance_current": "balance", "broker": "broker",
                         "region": "region"}.get(role))


def test_contributors_carry_no_case_identifiers(patched, expected_frame):
    """The reason this function exists at all: pipeline_drivers answers per
    case, and a notification may not transport loan-level data."""
    result = forward_mod.driver_contributors(
        _Test(), _Library(), expected_frame, {"numeratorValue": 100.0})
    assert result["available"] is True
    assert result["loanLevelExcluded"] is True
    blob = str(result)
    assert "caseId" not in blob
    assert "pipeline_case_identifier" not in blob


def test_contributions_are_balance_times_probability(patched, expected_frame):
    result = forward_mod.driver_contributors(
        _Test(), _Library(), expected_frame, {"numeratorValue": 100.0})
    # 6.0m x 0.8 + 2.2m x 0.5 + 3.9m x 0.6 = 8.24m for Broker A/B in the SW
    assert result["expectedNumeratorMovement"] == pytest.approx(8_240_000.0)


def test_the_broker_region_intersection_is_the_primary_contributor(
        patched, expected_frame):
    result = forward_mod.driver_contributors(
        _Test(), _Library(), expected_frame, {"numeratorValue": 100.0})
    primary = result["primaryContributor"]
    # Broker A in the South West: 6.0m x 0.8 + 2.2m x 0.5 = 5.9m
    assert primary["broker"] == "Broker A"
    assert primary["dimension"] == "South West"
    assert primary["expectedContribution"] == pytest.approx(5_900_000.0)
    assert primary["caseCount"] == 2


def test_the_groupings_reconcile_to_the_total_movement(patched, expected_frame):
    result = forward_mod.driver_contributors(
        _Test(), _Library(), expected_frame, {"numeratorValue": 100.0})
    total = result["expectedNumeratorMovement"]
    for key in ("byBroker", "byDimension", "byBrokerDimension"):
        assert sum(r["expectedContribution"]
                   for r in result[key]) == pytest.approx(total)


def test_ordering_is_deterministic(patched, expected_frame):
    first = forward_mod.driver_contributors(
        _Test(), _Library(), expected_frame, {"numeratorValue": 100.0})
    second = forward_mod.driver_contributors(
        _Test(), _Library(), expected_frame, {"numeratorValue": 100.0})
    assert first == second
    assert [r["name"] for r in first["byBroker"]] == ["Broker A", "Broker B"]


def test_funded_rows_never_contribute_to_the_pipeline_movement(patched,
                                                               expected_frame):
    """The movement is the PIPELINE's contribution; a funded row is already in
    the base position."""
    result = forward_mod.driver_contributors(
        _Test(), _Library(), expected_frame, {"numeratorValue": 100.0})
    assert result["byDimension"][0]["caseCount"] == 3   # three SW pipeline cases


def test_a_non_exposure_weighted_metric_is_declined_not_approximated(
        monkeypatch, expected_frame):
    monkeypatch.setattr(forward_mod, "treatment_for",
                        lambda _lib, _test: forward_mod.TREATMENT_INDICATIVE)
    result = forward_mod.driver_contributors(
        _Test(), _Library(), expected_frame, {"numeratorValue": 100.0})
    assert result["available"] is False
    assert "exposure-weighted" in result["reason"]


def test_an_empty_expected_frame_is_declined(patched):
    result = forward_mod.driver_contributors(
        _Test(), _Library(), pd.DataFrame(), {"numeratorValue": 100.0})
    assert result["available"] is False


def test_unattributed_dimensions_are_labelled_not_dropped(patched,
                                                          expected_frame):
    """Dropping them would make the groupings stop reconciling."""
    frame = expected_frame.copy()
    frame.loc[2, "broker"] = None
    result = forward_mod.driver_contributors(
        _Test(), _Library(), frame, {"numeratorValue": 100.0})
    names = [r["name"] for r in result["byBroker"]]
    assert "Unattributed" in names
    assert sum(r["expectedContribution"] for r in result["byBroker"]) \
        == pytest.approx(result["expectedNumeratorMovement"])


def test_pipeline_drivers_still_answers_per_case(patched, expected_frame):
    """The existing per-case function is untouched — the drill-through drawer
    still needs it."""
    result = forward_mod.pipeline_drivers(
        _Test(), _Library(), expected_frame, {"numeratorValue": 100.0})
    assert result["available"] is True
    assert len(result["drivers"]) == 3
    assert all("caseId" in d for d in result["drivers"])
