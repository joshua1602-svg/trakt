"""The governed concentration-test service and its surfaces: the REST routes
the React workspace consumes, and the chat route MI Query and Copilot share.

One evaluation service, disclosed sources, honest unavailability: an approved
configuration answers with full provenance; without one the legacy extracted
monitor is presented explicitly as unapproved; and nothing ever reports an
unavailable value as zero or recalculates outside the service.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent_api import concentration_tests_api as conc_mod  # noqa: E402
from mi_agent_api import chat_routing  # noqa: E402
from mi_agent_api.app import app  # noqa: E402

client = TestClient(app)

CLAUSE = ("1.1 The aggregate Current Balance of Loans secured on Properties "
          "located in East Anglia must not exceed 25% of the Portfolio.")


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAKT_STORAGE_BACKEND", "file")
    monkeypatch.setenv("TRAKT_LOCAL_BLOB_ROOT", str(tmp_path / "blob"))
    monkeypatch.setenv("TRAKT_RUNTIME_MODE", "test")
    return tmp_path


@pytest.fixture()
def funded_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "loan_id": ["L1", "L2", "L3", "L4"],
        "current_outstanding_balance": [100_000, 200_000, 300_000, 400_000],
        "collateral_geography": ["East Anglia", "London", "South East",
                                 "East Anglia"],
    })


@pytest.fixture()
def approved_config(env):
    """Seed an activated configuration through the REAL governance service."""
    from apps.blob_trigger_app.storage import open_storage
    from operations_control.concentration import ConcentrationGovernanceService
    from operations_control.stores import OpsLayout, OpsStore
    svc = ConcentrationGovernanceService(
        OpsStore(open_storage(), OpsLayout(container="operations-control")))
    [p] = svc.extract("client_001", actor="Alice",
                      source_reference="covenant.txt", text=CLAUSE)
    svc.approve("client_001", p.proposal_id, actor="Alice",
                effective_date="2025-01-01", comments="Reviewed.")
    return svc.activate("client_001", actor="Alice", reason="initial")


@pytest.fixture()
def frames(monkeypatch, funded_frame):
    prior = funded_frame.copy()
    prior["current_outstanding_balance"] = [100_000, 250_000, 300_000, 250_000]

    def fake_resolve(output_root, client_id, to_run_id, scope=None):
        return funded_frame, prior, "2025-11-30", "2025-10-31", "mi_2025_11"

    monkeypatch.setattr(conc_mod, "_resolve_frames", fake_resolve)
    return funded_frame, prior


class TestComputeService:
    def test_approved_configuration_is_the_source(self, approved_config,
                                                  frames):
        out = conc_mod.compute_concentration_tests(None, "client_001", None)
        assert out["source"] == "approved_configuration"
        assert out["approvalStatus"] == "approved"
        assert out["configurationVersion"] == 1
        [t] = out["tests"]
        # 500k of 1,000k → 50% vs limit 25% → breach; prior 350k/900k = 38.89.
        assert t["currentValue"] == 50.0
        assert t["status"] == "breach"
        assert t["priorValue"] == pytest.approx(38.89, abs=0.01)
        assert t["statusTransition"] == "breach -> breach" or \
            t["priorStatus"] == "breach"
        assert t["provenance"]["approvedBy"] == "Alice"
        assert t["provenance"]["sourceText"].startswith("1.1")
        assert out["summary"]["overallStatus"] == "breach"

    def test_without_a_configuration_the_legacy_source_is_disclosed(
            self, env, frames):
        out = conc_mod.compute_concentration_tests(None, "client_001", None)
        # client_001 has the committed extracted YAML → legacy fallback.
        assert out["source"] == "legacy_extracted"
        assert out["approvalStatus"] == "unapproved_legacy"
        assert "NOT operator-approved" in out["lineage"]["note"]
        assert out["tests"], "legacy limits are presented, clearly marked"
        for t in out["tests"]:
            assert t["legacy"] is True

    def test_a_client_with_nothing_gets_the_explicit_empty_state(self, env,
                                                                 frames):
        out = conc_mod.compute_concentration_tests(None, "client_nothing",
                                                   None)
        assert out["source"] == "none"
        assert out["available"] is False
        assert out["tests"] == []
        assert out["summary"]["overallStatus"] == "unavailable"

    def test_unavailable_is_never_reported_as_zero(self, approved_config,
                                                   monkeypatch):
        def no_frames(output_root, client_id, to_run_id, scope=None):
            return None, None, None, None, None
        monkeypatch.setattr(conc_mod, "_resolve_frames", no_frames)
        out = conc_mod.compute_concentration_tests(None, "client_001", None)
        [t] = out["tests"]
        assert t["currentValue"] is None
        assert t["status"] == "insufficient_data"
        assert out["summary"]["passes"] == 0


class TestRoutes:
    def test_the_endpoint_serves_the_evaluation(self, approved_config, frames):
        r = client.get("/mi/concentration-tests",
                       params={"portfolioId": "client_001"})
        assert r.status_code == 200
        body = r.json()
        assert body["source"] == "approved_configuration"
        assert body["tests"][0]["currentValue"] == 50.0

    def test_drillthrough_reconciles_to_the_numerator(self, approved_config,
                                                      frames):
        snapshot = conc_mod.compute_concentration_tests(None, "client_001",
                                                        None)
        test_id = snapshot["tests"][0]["testId"]
        r = client.get("/mi/concentration-tests/drillthrough",
                       params={"portfolioId": "client_001", "testId": test_id})
        body = r.json()
        assert body["available"] is True
        assert body["reconciles"] is True
        assert body["loansInNumerator"] == 2
        total = sum(row["current_outstanding_balance"] for row in body["rows"])
        assert total == body["numeratorValue"] == 500_000

    def test_drillthrough_refuses_unapproved_sources(self, env, frames):
        r = client.get("/mi/concentration-tests/drillthrough",
                       params={"portfolioId": "client_001", "testId": "x"})
        body = r.json()
        assert body["available"] is False
        assert "approved" in body["reason"]

    def test_history_reports_a_controlled_absence(self, approved_config,
                                                  frames):
        r = client.get("/mi/concentration-tests/history",
                       params={"portfolioId": "client_001"})
        body = r.json()
        assert body["available"] is False
        assert "history" in body["reason"].lower() or body["series"] == []

    def test_the_route_never_500s(self, env, monkeypatch):
        def boom(*a, **kw):
            raise RuntimeError("boom")
        monkeypatch.setattr(conc_mod, "compute_concentration_tests", boom)
        r = client.get("/mi/concentration-tests",
                       params={"portfolioId": "client_001"})
        assert r.status_code == 200
        assert r.json()["available"] is False


@pytest.fixture()
def pipeline_frames(monkeypatch):
    """Governed pipeline + methodology metadata, patched at the resolver seam
    (the same place the real weekly-extract discovery plugs in)."""
    pipeline = pd.DataFrame({
        "pipeline_case_identifier": ["P1", "P2", "P3"],
        "current_outstanding_balance": [640_000, 510_000, 900_000],
        "geographic_region_obligor": ["East Anglia", "East Anglia", "London"],
        "pipeline_stage": ["OFFER", "APPLICATION", "OFFER"],
        "pipeline_status": ["pipeline", "pipeline", "pipeline"],
        "completion_probability": [0.8, 0.5, 0.8],
        "completion_probability_source": ["historical_stage_rate"] * 3,
        "expected_completion_month": ["2026-01", "2026-02", "2026-01"],
    })
    meta = {
        "available": True, "methodology": "completion_trend_model",
        "basis": "historical_observed",
        "observationWindowStart": "2025-10-27",
        "observationWindowEnd": "2025-11-24",
        "weeklyExtractsUsed": 5, "trackedCaseCount": 214,
        "observedCompletionCount": 41, "minObservations": 12,
        "stagesUsingHistoricalRates": ["APPLICATION", "OFFER"],
        "stagesUsingConfigFallback": ["KFI"],
        "stageRates": {"OFFER": {"rate": 0.8, "observed": 41,
                                 "sufficient": True}},
        "excludedStageCounts": {"WITHDRAWN": 7},
        "pointInTimeNote": "Historical expected-state comparisons are "
                           "unavailable until point-in-time snapshots are "
                           "persisted.",
    }
    monkeypatch.setattr(conc_mod, "_resolve_pipeline",
                        lambda client_id, to_run_id, scope=None: (pipeline, meta))
    return pipeline, meta


class TestThreeState:
    def test_states_ride_on_the_same_envelope(self, approved_config, frames,
                                              pipeline_frames):
        out = conc_mod.compute_concentration_tests(None, "client_001", None)
        [t] = out["tests"]
        # Funded: 500k/1,000k = 50% (unchanged). Expected adds EA 512k+255k
        # weighted and 720k London to the denominator.
        assert t["currentValue"] == 50.0
        exp_num = 500_000 + 640_000 * 0.8 + 510_000 * 0.5
        exp_den = 1_000_000 + exp_num - 500_000 + 900_000 * 0.8
        assert t["expected"]["value"] == pytest.approx(
            exp_num / exp_den * 100, abs=0.01)
        assert t["expectedNumerator"] == pytest.approx(exp_num, abs=1)
        assert t["fullPipeline"]["value"] == pytest.approx(
            (500_000 + 1_150_000) / (1_000_000 + 2_050_000) * 100, abs=0.01)
        assert out["forecast"]["observationWindowStart"] == "2025-10-27"
        assert out["summary"]["expectedBreaches"] == 1
        # Funded is already in breach here, so the rank-1 rule claims the
        # test — one risk per test, most severe category wins.
        assert out["emergingRisks"][0]["category"] == "current_breach"

    def test_scoped_context_disables_forward_states_honestly(
            self, approved_config, frames):
        class Scope:
            context_id = "direct"
        _df, meta = conc_mod._resolve_pipeline("client_001", None,
                                               scope=Scope())
        assert _df is None
        assert "total book only" in meta["reason"]

    def test_drivers_route_reconciles(self, approved_config, frames,
                                      pipeline_frames):
        snapshot = conc_mod.compute_concentration_tests(None, "client_001",
                                                        None)
        test_id = snapshot["tests"][0]["testId"]
        out = conc_mod.compute_pipeline_drivers(None, "client_001", None,
                                                test_id)
        assert out["available"] and out["reconciles"]
        ids = [d["caseId"] for d in out["drivers"]]
        assert ids == ["P1", "P2"]  # EA numerator cases only, ranked
        assert out["expectedNumeratorMovement"] == pytest.approx(
            640_000 * 0.8 + 510_000 * 0.5, abs=1)


class _Spec:
    risk_limit_query = True
    risk_limit_category = None


class TestChatDelegation:
    """MI Query and Copilot share this route (`askTraktMi` delegates to the
    same governed service), so these assertions cover both channels."""

    def _ask(self, question):
        return chat_routing._route_risk(
            question, _Spec(), {}, client_id="client_001", run_id=None,
            output_root=None, portfolio_id="client_001", as_of=None)

    def test_limit_questions_answer_from_the_approved_service(
            self, approved_config, frames):
        env_out = self._ask("Are we breaching any concentration limits?")
        assert env_out["metadata"]["configurationVersion"] == 1
        assert "breach" in env_out["answer"].lower()
        assert any("Approved configuration v1" in n["note"]
                   for n in env_out["sourceNotes"])

    def test_intents_route_to_the_governed_capabilities(self, approved_config,
                                                        frames):
        cases = {
            "Which tests are closest to breach?": "list_concentration_warnings",
            "Show regional concentration tests.": "list_concentration_tests",
            "What changed in risk limits this month?":
                "compare_concentration_periods",
            "Which limits cannot currently be calculated?":
                "list_concentration_unavailable",
            "What definition are we using for the East Anglia test?":
                "explain_concentration_test",
            "Which loans contribute to the East Anglia test?":
                "get_concentration_test_drillthrough",
        }
        for question, intent in cases.items():
            out = self._ask(question)
            assert out["metadata"]["concentrationIntent"] == intent, question

    def test_definition_answers_carry_threshold_and_provenance(
            self, approved_config, frames):
        out = self._ask("What definition are we using for the East Anglia "
                        "test?")
        assert "≤ 25.00%" in out["answer"]
        assert "Approved by Alice" in out["answer"]
        assert "1.1 The aggregate Current Balance" in out["answer"]

    def test_drillthrough_answers_with_the_evaluated_population(
            self, approved_config, frames):
        out = self._ask("Which loans contribute to the East Anglia test?")
        assert "2 loan(s)" in out["answer"]
        tables = [a for a in out["artifacts"] if a["type"] == "table"]
        assert len(tables) == 1
        assert len(tables[0]["rows"]) == 2

    def test_without_an_approved_configuration_the_legacy_route_answers(
            self, env, monkeypatch):
        # No approved config for this deployment client → the pre-existing
        # Schedule 8 behaviour is unchanged.
        called = {}
        real = chat_routing.risk_mod.compute_risk_limits

        def spy(*a, **kw):
            called["yes"] = True
            return real(*a, **kw)
        monkeypatch.setattr(chat_routing.risk_mod, "compute_risk_limits", spy)
        out = chat_routing._route_risk(
            "Are we breaching any limits?", _Spec(), {},
            client_id="client_no_config", run_id=None, output_root=None,
            portfolio_id="client_no_config", as_of=None)
        assert called.get("yes") is True
        assert out["metadata"].get("concentrationIntent") is None

    def test_expected_breach_answer(self, approved_config, frames,
                                    pipeline_frames):
        out = self._ask("Are we likely to breach any concentration limits?")
        assert out["metadata"]["concentrationIntent"] == \
            "identify_emerging_concentration_risks"
        assert "expected" in out["answer"].lower()

    def test_full_pipeline_answer_is_labelled_stress(self, approved_config,
                                                     frames, pipeline_frames):
        out = self._ask("What happens if all pipeline loans complete?")
        assert out["metadata"]["concentrationIntent"] == \
            "list_full_pipeline_concentration_breaches"
        assert "maximum-exposure stress" in out["answer"]
        assert "not an expected outcome" in out["answer"]

    def test_methodology_answer_grounds_in_the_engine(self, approved_config,
                                                      frames, pipeline_frames):
        out = self._ask("How reliable is this forecast?")
        assert out["metadata"]["concentrationIntent"] == \
            "get_forecast_methodology"
        assert "2025-10-27" in out["answer"]
        assert "no machine learning" in out["answer"].lower()
        assert "point-in-time" in out["answer"].lower()

    def test_insufficient_history_caveat_travels_as_a_warning(
            self, approved_config, frames, pipeline_frames):
        out = self._ask("Are we breaching any concentration limits?")
        assert any("sufficiency floor" in w for w in out["warnings"])

    def test_driver_answer_names_the_leading_case(self, approved_config,
                                                  frames, pipeline_frames):
        out = self._ask("Which pipeline loans drive the East Anglia increase?")
        assert out["metadata"]["concentrationIntent"] == \
            "get_concentration_pipeline_drivers"
        assert "P1" in out["answer"]
        tables = [a for a in out["artifacts"] if a["type"] == "table"]
        assert tables and tables[0]["rows"][0]["caseId"] == "P1"

    def test_unavailable_forecast_is_explicit_not_zero(self, approved_config,
                                                       frames, monkeypatch):
        monkeypatch.setattr(
            conc_mod, "_resolve_pipeline",
            lambda client_id, to_run_id, scope=None:
                (None, {"available": False,
                        "reason": "No governed pipeline source found."}))
        out = self._ask("What happens if all pipeline loans complete?")
        assert "not available" in out["answer"]
        assert "Funded results are unaffected" in out["answer"]

    def test_the_chat_never_recalculates_from_raw_rows(self, approved_config,
                                                       frames, monkeypatch):
        """The route consumes the evaluation service's envelope verbatim: with
        the evaluator neutered, the chat has no number of its own to offer."""
        def empty(output_root, client_id, run_id, scope=None):
            return {"source": "approved_configuration", "tests": [],
                    "summary": {"activeTests": 0, "passes": 0, "warnings": 0,
                                "breaches": 0, "unavailable": 0,
                                "deteriorations": 0, "closestToLimit": None,
                                "overallStatus": "unavailable",
                                "priorAvailable": False},
                    "configurationVersion": 1, "priorAvailable": False}
        monkeypatch.setattr(chat_routing,
                            "_route_concentration_tests",
                            chat_routing._route_concentration_tests)
        monkeypatch.setattr(
            "mi_agent_api.concentration_tests_api.compute_concentration_tests",
            empty)
        out = self._ask("Are we breaching any concentration limits?")
        assert "0" in out["answer"] or "No concentration" in out["answer"]
        assert not any(a for a in out["artifacts"] if a["type"] == "chart")
