"""The borrowing-base service and its route.

Two properties matter here and neither is arithmetic — the arithmetic is tested
in ``tests/borrowing_base`` against hand-worked figures. What is tested here is:

* the Eligibility & Concentrations tab gets the facility position and the
  Schedule 8 results from ONE evaluation of ONE frame, so they cannot disagree;
* nothing on this path ever 500s, and nothing ever presents a missing input as
  a zero.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent_api import borrowing_base_api as bb_mod  # noqa: E402
from mi_agent_api import concentration_tests_api as conc_mod  # noqa: E402
from mi_agent_api.app import app  # noqa: E402

client = TestClient(app)

CLIENT = "bb_client"

FACILITY = {
    "facility_id": "TEST_WAREHOUSE",
    "facility_type": "warehouse",
    "currency": "GBP",
    "commitment": 250_000_000,
    "advance_rate": 1.03,
    "concentration_denominator_floor": 33_000_000,
    "current_drawn_amount": None,
    "environment": "prototype",
    "eligibility": {"rule_version": "t-0",
                    "prototype_assume_financing_portfolio_eligible": True},
    "concentration": {"population": "eligible_mortgage_loans",
                      "borrowing_base_treatment": "monitor_only"},
}


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAKT_STORAGE_BACKEND", "file")
    monkeypatch.setenv("TRAKT_LOCAL_BLOB_ROOT", str(tmp_path / "blob"))
    monkeypatch.setenv("TRAKT_RUNTIME_MODE", "test")
    return tmp_path


@pytest.fixture()
def facility_configured(env, monkeypatch):
    from mi_agent.borrowing_base import config as bb_config
    register = env / "funding_facilities.yaml"
    register.write_text(yaml.safe_dump({
        "schema_version": "1.0.0", "config_version": "test-1",
        "facilities": [{"client_id": CLIENT, **FACILITY}]}), encoding="utf-8")
    monkeypatch.setenv(bb_config.FACILITIES_PATH_ENV, str(register))
    monkeypatch.setenv(bb_config.CLIENT_CONFIG_DIR_ENV, str(env / "no_clients"))
    return register


@pytest.fixture()
def funded_frame() -> pd.DataFrame:
    """£100,000,000 across four loans, so the gross base is £103,000,000."""
    from mi_agent.borrowing_base.eligibility import derive_eligibility
    from mi_agent.borrowing_base.models import FacilityConfiguration

    df = pd.DataFrame({
        "loan_id": ["L1", "L2", "L3", "L4"],
        "current_outstanding_balance": [10_000_000.0, 20_000_000.0,
                                        30_000_000.0, 40_000_000.0],
        "collateral_geography": ["East Of England", "London", "South East",
                                 "East Of England"],
    })
    derive_eligibility(df, FacilityConfiguration(
        client_id=CLIENT, facility_id="TEST_WAREHOUSE",
        environment="prototype",
        prototype_assume_financing_portfolio_eligible=True))
    return df


@pytest.fixture()
def frames(monkeypatch, funded_frame):
    def fake_resolve(output_root, client_id, to_run_id, scope=None):
        return funded_frame, None, "2025-11-30", None, "mi_2025_11"
    monkeypatch.setattr(conc_mod, "_resolve_frames", fake_resolve)
    return funded_frame


class TestTheEnvelope:
    def test_the_concentration_envelope_carries_the_borrowing_base(
            self, facility_configured, frames):
        out = conc_mod.compute_concentration_tests(None, CLIENT, None)
        base = out["borrowingBase"]
        assert base["available"] is True
        assert base["eligibleCurrentBalance"] == 100_000_000.0
        assert base["grossBorrowingBase"] == 103_000_000.0
        assert base["availableBorrowingBase"] == 103_000_000.0

    def test_the_working_frame_never_leaks_into_the_envelope(
            self, facility_configured, frames):
        out = conc_mod.compute_concentration_tests(None, CLIENT, None)
        assert "_fundedFrame" not in out
        assert not any(isinstance(v, pd.DataFrame) for v in out.values())

    def test_the_standalone_route_returns_the_SAME_block(
            self, facility_configured, frames):
        envelope = conc_mod.compute_concentration_tests(None, CLIENT, None)
        standalone = bb_mod.compute_borrowing_base(None, CLIENT, None)
        assert standalone["availableBorrowingBase"] == \
            envelope["borrowingBase"]["availableBorrowingBase"]
        assert standalone["receipt"]["content_hash"] == \
            envelope["borrowingBase"]["receipt"]["content_hash"]

    def test_the_eligible_population_is_disclosed_on_the_envelope(
            self, facility_configured, frames):
        out = conc_mod.compute_concentration_tests(None, CLIENT, None)
        disclosure = out["eligiblePopulation"]
        assert disclosure["basis"] == "governed_eligibility"
        assert disclosure["eligibleLoanCount"] == 4
        assert disclosure["prototypeAssumptionActive"] is True

    def test_a_missing_drawn_amount_reads_NOT_CALCULABLE_not_zero(
            self, facility_configured, frames):
        out = bb_mod.compute_borrowing_base(None, CLIENT, None)
        assert out["currentDrawnAmount"] == "NOT_CALCULABLE"
        assert out["borrowingBaseHeadroom"] == "NOT_CALCULABLE"
        assert out["missingInputs"] == ["current_drawn_amount"]

    def test_the_measures_and_their_definitions_are_served(
            self, facility_configured, frames):
        out = bb_mod.compute_borrowing_base(None, CLIENT, None)
        assert out["measures"]["borrowing_base"] == 103_000_000.0
        assert {d["measure_id"] for d in out["measureDefinitions"]} >= {
            "eligible_balance", "borrowing_base", "facility_utilisation"}


class TestNoFacility:
    def test_a_portfolio_with_no_facility_gets_an_explicit_empty_state(
            self, env, frames, monkeypatch):
        from mi_agent.borrowing_base import config as bb_config
        monkeypatch.setenv(bb_config.FACILITIES_PATH_ENV,
                           str(env / "nothing_here.yaml"))
        monkeypatch.setenv(bb_config.CLIENT_CONFIG_DIR_ENV, str(env / "none"))
        out = bb_mod.compute_borrowing_base(None, "client_with_no_facility", None)
        assert out["available"] is False
        assert "No funding facility is configured" in out["reason"]

    def test_the_concentration_tab_is_unaffected_by_having_no_facility(
            self, env, frames, monkeypatch):
        from mi_agent.borrowing_base import config as bb_config
        monkeypatch.setenv(bb_config.FACILITIES_PATH_ENV,
                           str(env / "nothing_here.yaml"))
        monkeypatch.setenv(bb_config.CLIENT_CONFIG_DIR_ENV, str(env / "none"))
        out = conc_mod.compute_concentration_tests(None, "client_no_facility",
                                                   None)
        assert out["borrowingBase"]["available"] is False
        assert out["eligiblePopulation"]["basis"] == \
            "whole_book_no_facility_configured"
        assert "STAND-IN" in out["eligiblePopulation"]["note"]


class TestTheRoutes:
    def test_the_borrowing_base_route_exists_and_answers(self, facility_configured,
                                                         frames):
        response = client.get(f"/mi/borrowing-base?portfolioId={CLIENT}")
        assert response.status_code == 200
        assert response.json()["availableBorrowingBase"] == 103_000_000.0

    def test_it_never_500s_when_the_service_falls_over(self, monkeypatch):
        def explode(*_a, **_k):
            raise RuntimeError("resolution failed")
        monkeypatch.setattr(conc_mod, "compute_concentration_tests", explode)
        out = bb_mod.compute_borrowing_base(None, CLIENT, None)
        assert out["available"] is False
        assert "could not be resolved" in out["reason"]

    def test_a_broken_borrowing_base_never_takes_the_concentrations_down(
            self, facility_configured, frames, monkeypatch):
        monkeypatch.setattr(bb_mod, "compute_from_frames",
                            lambda *a, **k: (_ for _ in ()).throw(
                                RuntimeError("boom")))
        out = conc_mod.compute_concentration_tests(None, CLIENT, None)
        assert out["borrowingBase"]["available"] is False
        assert "could not be calculated" in out["borrowingBase"]["reason"]
        assert "tests" in out

    def test_the_eligibility_drill_down_lists_loans_with_their_reason(
            self, facility_configured, frames):
        out = bb_mod.compute_eligibility_loans(None, CLIENT, None, "ELIGIBLE")
        assert out["available"] is True
        assert out["rowCount"] == 4
        assert "borrowing_base_eligibility_reason" in out["columns"]

    def test_the_drill_down_refuses_a_status_that_is_not_governed(
            self, facility_configured, frames):
        out = bb_mod.compute_eligibility_loans(None, CLIENT, None, "MAYBE")
        assert out["available"] is False
        assert "not a governed eligibility status" in out["reason"]

    def test_the_drill_down_route_answers(self, facility_configured, frames):
        response = client.get(
            f"/mi/borrowing-base/loans?portfolioId={CLIENT}&status=UNDETERMINED")
        assert response.status_code == 200
        assert response.json()["rowCount"] == 0


class TestThePrototypeAssumptionSurvivesTheWholePath:
    """From configuration, through the canonical frame, to the wire payload.

    The dashboard banner reads `prototypeAssumptionsUsed`. If the assumption is
    active and that list arrives empty, the tab presents an assumed eligible
    collateral balance as a governed one.
    """

    def test_the_envelope_says_the_figures_rest_on_an_assumption(
            self, facility_configured, frames):
        out = bb_mod.compute_borrowing_base(None, CLIENT, None)
        assert out["prototypeAssumptionsUsed"]
        assert "PROTOTYPE ASSUMPTION" in out["prototypeAssumptionsUsed"][0]

    def test_the_concentration_envelope_says_it_too(self, facility_configured,
                                                    frames):
        out = conc_mod.compute_concentration_tests(None, CLIENT, None)
        assert out["borrowingBase"]["prototypeAssumptionsUsed"]
        assert out["eligiblePopulation"]["prototypeAssumptionActive"] is True

    def test_the_receipt_records_it_for_the_auditor(self, facility_configured,
                                                    frames):
        out = bb_mod.compute_borrowing_base(None, CLIENT, None)
        assert out["receipt"]["prototype_assumptions_used"]
