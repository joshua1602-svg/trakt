"""Shared fixtures for the notification tests.

Everything runs against the filesystem storage backend under ``tmp_path``, which
is the same :class:`apps.blob_trigger_app.storage.Storage` the runtime uses with
a different root. The tests therefore exercise the real persistence code rather
than a stand-in, and no Azure account is involved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from apps.blob_trigger_app.storage import Storage  # noqa: E402

from trakt_notifications.layout import NotificationLayout  # noqa: E402
from trakt_notifications.recipients import Recipient, RecipientStore  # noqa: E402
from trakt_notifications.sources import GovernedInputs  # noqa: E402

TENANT = "ERE"
MS_TENANT = "11111111-1111-1111-1111-111111111111"


@pytest.fixture
def storage(tmp_path) -> Storage:
    return Storage(local_root=tmp_path)


@pytest.fixture
def layout() -> NotificationLayout:
    return NotificationLayout(state_container="trakt-state")


@pytest.fixture
def enabled_config(tmp_path, monkeypatch) -> Path:
    """A configuration file with delivery switched on.

    Written per test rather than pointing at the repository default, so a test
    can never pass because the shipped configuration happened to allow
    something.
    """
    path = tmp_path / "teams_notifications.yaml"
    path.write_text(
        "teams_notifications:\n"
        "  enabled: true\n"
        "  delivery_scope: personal\n"
        "  update_message_enabled: true\n"
        "  risk_message_enabled: true\n"
        "  recommendation_level: consideration\n"
        "  maximum_risk_items: 3\n"
        "  maximum_update_items: 5\n"
        "  max_attempts: 3\n"
        "  retry_backoff_seconds: [30, 120]\n",
        encoding="utf-8")
    monkeypatch.setenv("TRAKT_TEAMS_NOTIFICATIONS_CONFIG", str(path))
    monkeypatch.delenv("TRAKT_TEAMS_NOTIFICATIONS", raising=False)
    return path


@pytest.fixture
def recipient(storage, layout) -> Recipient:
    """An authorised, addressable pilot recipient."""
    store = RecipientStore(storage, layout)
    captured = store.capture_conversation(
        tenant_id=TENANT, microsoft_tenant_id=MS_TENANT,
        entra_object_id="22222222-2222-2222-2222-222222222222",
        conversation_id="a:1conversation",
        service_url="https://smba.trafficmanager.net/emea/",
        conversation_reference={"conversation": {"id": "a:1conversation"}},
        display_name="Pilot User")
    return store.authorise(TENANT, captured.recipient_id,
                           portfolio_contexts=["total"], actor="operator")


def pipeline_inputs(**overrides) -> GovernedInputs:
    """Governed inputs for a healthy weekly pipeline update.

    Shaped exactly like the real service outputs so the builders are exercised
    against the contracts they consume in production.
    """
    inputs = GovernedInputs(tenant_id=TENANT, portfolio_id=TENANT,
                            portfolio_context="total")
    inputs.pipeline_evolution = {"fiveWeekAverage": {
        "available": True, "window": 5, "weeksObserved": 5,
        "basis": "trailing mean of the last 5 governed weekly extracts",
        "current": {"pipeline_amount": 503_400_000.0,
                    "pipeline_case_count": 812,
                    "weighted_expected_funded_amount": 236_200_000.0},
        "average": {"pipeline_amount": 480_000_000.0,
                    "pipeline_case_count": 745,
                    "weighted_expected_funded_amount": 230_000_000.0},
        "differencePct": {"pipeline_amount": 4.9, "pipeline_case_count": 9.0,
                          "weighted_expected_funded_amount": 2.7}}}
    inputs.funnel = {"summary": {"COMPLETED": {
        "latestFlowValue": 18_700_000.0, "latestFlowCount": 41,
        "fiveWeekAvgFlowValue": 21_744_186.0,
        "conversion": {"sufficient": True, "weeklyRateValue": 4.2,
                       "weeksInWindow": 5}}}}
    inputs.source_dates = {"pipeline_as_of": "2026-08-07",
                           "pipeline_comparison": "2026-07-31"}
    inputs.brief = {"status": "success", "omitted": [], "insights": [{
        "insight_type": "PIPELINE_MOVEMENT", "insight_id": "insight-pm-1",
        "severity": "info",
        "metrics": {"change": 23_400_000.0, "change_pct": 4.9},
        "contributors": {"brokers": [{"name": "Broker A"}],
                         "regions": [{"name": "the South East"}]}}]}
    inputs.concentration = {"available": True, "reportingDate": "2026-07-31",
                            "tests": [], "emergingRisks": [],
                            "states": {"available": True}}
    for key, value in overrides.items():
        setattr(inputs, key, value)
    return inputs


def funded_inputs(**overrides) -> GovernedInputs:
    """Governed inputs for a monthly funded update."""
    inputs = GovernedInputs(tenant_id=TENANT, portfolio_id=TENANT,
                            portfolio_context="total")
    inputs.funded_movement = {
        "available": True,
        "currentReportingDate": "2026-07-31",
        "priorReportingDate": "2026-06-30",
        "current": {"funded_balance": 1_420_000_000.0, "loan_count": 4820,
                    "wa_ltv_points": 31.4},
        "prior": {"funded_balance": 1_395_900_000.0, "loan_count": 4761,
                  "wa_ltv_points": 31.1},
        "delta": {"funded_balance": 24_100_000.0, "loan_count": 59,
                  "wa_ltv_points": 0.3},
        "cohortMovements": [{"id": "direct_001", "label": "Direct",
                             "delta": 18_400_000.0},
                            {"id": "acquired_001", "label": "Acquired",
                             "delta": 5_700_000.0}],
        "primaryRegion": {"region": "South East", "delta": 9_200_000.0},
    }
    inputs.source_dates = {"funded_as_of": "2026-07-31",
                           "funded_comparison": "2026-06-30"}
    inputs.concentration = {"available": True, "reportingDate": "2026-07-31",
                            "tests": [], "emergingRisks": [],
                            "states": {"available": True}}
    for key, value in overrides.items():
        setattr(inputs, key, value)
    return inputs


def concentration_risk(*, expected_breach: bool = True) -> dict:
    """A governed concentration snapshot carrying a South West breach risk."""
    return {
        "available": True,
        "reportingDate": "2026-07-31",
        "states": {"available": True},
        "forecast": {"methodology": "empirical stage completion rates"},
        "tests": [{"testId": "region_sw", "displayName": "South West exposure",
                   "status": "warning", "utilization": 0.94,
                   "expectedBreach": expected_breach}],
        "emergingRisks": [{
            "category": "expected_breach" if expected_breach else "current_breach",
            "rank": 2 if expected_breach else 1,
            "testId": "region_sw",
            "displayName": "South West exposure",
            "statement": ("South West exposure is expected to breach: funded "
                          "23.5% vs 25.0%; expected 25.8% (+0.8pp over), "
                          "crossing around 2026-09."),
            "expectedHeadroom": -0.8}],
    }


def contributors_block() -> dict:
    """Governed contributor aggregates — dimension totals, no case identifiers."""
    return {
        "available": True,
        "testId": "region_sw",
        "displayName": "South West exposure",
        "expectedNumeratorMovement": 14_100_000.0,
        "byBroker": [{"name": "Broker A", "expectedContribution": 8_200_000.0,
                      "shareOfMovement": 0.5816, "caseCount": 31}],
        "byDimension": [{"name": "South West",
                         "expectedContribution": 14_100_000.0,
                         "shareOfMovement": 1.0, "caseCount": 62}],
        "byBrokerDimension": [{"broker": "Broker A", "dimension": "South West",
                               "expectedContribution": 8_200_000.0,
                               "shareOfMovement": 0.5816, "caseCount": 31}],
        "primaryContributor": {"broker": "Broker A", "dimension": "South West",
                               "expectedContribution": 8_200_000.0,
                               "shareOfMovement": 0.5816, "caseCount": 31},
        "reconciles": True,
        "loanLevelExcluded": True,
    }
