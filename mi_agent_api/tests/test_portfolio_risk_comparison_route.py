"""Portfolio Risk Comparison — recogniser registration and the API adapter.

The pure workflow is covered in ``tests/test_portfolio_risk_comparison_workflow.py``.
Here: how the workflow lands in the platform — the recogniser's declaration
(priority, confidence, BSR terms in ``metadata``), dispatch through
``chat_routing.try_route`` (including that it does not steal questions owned by
more specific routes), the adapter's envelope, controlled degradation when the
BSR or frame is unavailable, and tenancy enforcement ahead of the workflow via
``execute_governed_mi_query``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mi_agent.mi_query_validator import load_mi_semantics
from mi_agent_api import chat_routing
from mi_agent_api.recogniser_registry import REGISTRY
from mi_workflows import portfolio_risk_comparison as prc

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics(
        _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml")


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "source_portfolio_id": ["direct_001"] * 4 + ["acquired_001"] * 4,
        "source_portfolio_type": ["direct"] * 4 + ["acquired"] * 4,
        "reporting_date": ["2026-06-30"] * 8,
        "current_outstanding_balance": [100.0, 200.0, 300.0, 400.0,
                                        150.0, 250.0, 350.0, 450.0],
        "current_loan_to_value": [0.30, 0.40, 0.50, 0.60,
                                  0.55, 0.65, 0.70, 0.75],
        "arrears_balance": [0.0, 0.0, 10.0, 0.0, 5.0, 0.0, 20.0, 15.0],
        "exposure_currency_denomination": ["GBP"] * 8,
        "account_status": ["performing"] * 8,
        "product_type": ["lump"] * 4 + ["draw"] * 4,
    })


def _route(question, semantics, frame_resolver=None, **kw):
    return chat_routing.try_route(
        question, portfolio_id="client_001", view="funded", output_root=None,
        pipeline_root=None, semantics=semantics,
        frame_resolver=frame_resolver, **kw)


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
class TestRegistration:
    def test_registered_with_workflow_confidence_and_bsr_terms(self):
        rec = REGISTRY.get(prc.WORKFLOW_ID)
        assert rec is not None
        assert rec.lens_aware is True
        assert rec.capability is None            # funded-book routes are ungated
        # First recogniser to populate the declarative BSR metadata slot.
        assert rec.metadata["bsr_workflow_tag"] == "portfolio_comparison"
        assert "portfolio_comparability" in rec.metadata["bsr_axes_consumed"]
        assert "directionality" in rec.metadata["bsr_axes_consumed"]

    def test_priority_sits_between_geo_and_period_movement(self):
        names = list(REGISTRY.names())
        assert names.index("geo_exposure") < names.index(prc.WORKFLOW_ID) \
            < names.index("period_movement")

    def test_existing_route_order_is_preserved(self):
        """Adding a capability must not reorder the migrated chain.

        What is asserted is that the twelve MIGRATED routes keep their relative
        order, which is what decides how a question the old chain answered is
        answered now. Routes added since the migration are excluded: the
        workflow layer registers above the chain, and `pipeline_stage_movement`
        registers below all of it — neither sits inside the chain, and each has
        its own test for where it sits (`test_priority_sits_between_geo_and_
        period_movement` here, `test_the_route_registers_after_every_existing_
        recogniser` in `test_stage_movement_query.py`).
        """
        added_since_the_migration = {prc.WORKFLOW_ID, "concentration_analysis",
                                     "analytical_composition",
                                     "pipeline_stage_movement"}
        names = [n for n in REGISTRY.names()
                 if n not in added_since_the_migration]
        assert names == [
            "scenario", "cohort_conversion", "forecast_extrapolation",
            "funded_bridge", "cohort_progression", "geo_exposure",
            "period_movement", "portfolio_summary", "period_change_analysis",
            "temporal_compare", "risk_limits", "evolution"]


# --------------------------------------------------------------------------- #
# Dispatch — matches its questions, steals nothing
# --------------------------------------------------------------------------- #
class TestDispatch:
    def test_portfolio_comparison_routes_to_the_workflow(self, semantics):
        env = _route("Compare direct_001 with acquired_001", semantics,
                     frame_resolver=lambda c, r: _frame())
        assert env is not None
        assert env["metadata"]["route"] == prc.WORKFLOW_ID
        assert env["ok"] is True
        assert env["metadata"]["lensApplied"] is True

    def test_workflow_confidence_outranks_lower_priority_matches(self, semantics):
        from mi_agent.parsed_question import ParsedQuestion

        parsed = ParsedQuestion.parse("Compare geographic composition of the "
                                      "portfolios", semantics)
        request = chat_routing.RouteRequest(
            question=parsed.question, spec=parsed.spec,
            spec_dict=parsed.spec.to_dict(), semantics=semantics, view="funded",
            client_id="client_001", run_id=None, portfolio_id="client_001")
        candidates = [rec.name for rec, _ in REGISTRY.candidates(request)]
        assert candidates and candidates[0] == prc.WORKFLOW_ID

    def test_period_comparisons_are_not_stolen(self, semantics):
        from mi_agent.parsed_question import ParsedQuestion

        question = "Compare October and November funded balance"
        parsed = ParsedQuestion.parse(question, semantics)
        request = chat_routing.RouteRequest(
            question=question, spec=parsed.spec, spec_dict=parsed.spec.to_dict(),
            semantics=semantics, view="funded", client_id="client_001",
            run_id=None, portfolio_id="client_001")
        candidates = [rec.name for rec, _ in REGISTRY.candidates(request)]
        assert prc.WORKFLOW_ID not in candidates
        assert "temporal_compare" in candidates

    def test_movement_questions_are_not_stolen(self, semantics):
        from mi_agent.parsed_question import ParsedQuestion

        question = "What has changed versus the prior month?"
        parsed = ParsedQuestion.parse(question, semantics)
        request = chat_routing.RouteRequest(
            question=question, spec=parsed.spec, spec_dict=parsed.spec.to_dict(),
            semantics=semantics, view="funded", client_id="client_001",
            run_id=None, portfolio_id="client_001")
        candidates = [rec.name for rec, _ in REGISTRY.candidates(request)]
        assert prc.WORKFLOW_ID not in candidates
        assert "period_movement" in candidates

    def test_non_comparison_questions_fall_through(self, semantics):
        env = _route("What is the weighted average LTV?", semantics,
                     frame_resolver=lambda c, r: _frame())
        assert env is None or env["metadata"]["route"] != prc.WORKFLOW_ID


# --------------------------------------------------------------------------- #
# The adapter envelope
# --------------------------------------------------------------------------- #
class TestAdapterEnvelope:
    def test_envelope_carries_workflow_result_tables_and_governance_notes(self, semantics):
        env = _route("Compare direct_001 with acquired_001", semantics,
                     frame_resolver=lambda c, r: _frame())
        assert env["workflow"]["workflow_id"] == prc.WORKFLOW_ID
        assert env["workflow"]["available"] is True
        assert env["workflow"]["audit"]["bsr_version"]
        titles = [a["title"] for a in env["artifacts"]]
        assert any("Portfolio comparison" in t for t in titles)
        assert any("mix" in t for t in titles)
        assert env["reconciliation"]["dataset"] == "funded"
        assert env["reconciliation"]["portfolio_a_rows"] == 4
        assert any("Business Semantics Registry" in n["note"]
                   for n in env["sourceNotes"])

    def test_answer_is_observational(self, semantics):
        env = _route("Which portfolio appears stronger across the indicators?",
                     semantics, frame_resolver=lambda c, r: _frame())
        assert env["metadata"]["route"] == prc.WORKFLOW_ID
        low = env["answer"].lower()
        for forbidden in ("safer", "higher quality", "preferable", "breach",
                          "appetite", "risk score"):
            assert forbidden not in low

    def test_controlled_failure_is_explained_not_500(self, semantics):
        env = _route("Compare direct_001 with acquired_009", semantics,
                     frame_resolver=lambda c, r: _frame())
        assert env["metadata"]["route"] == prc.WORKFLOW_ID
        # CONTROLLED, AND NOT A SUCCESS — see the matching note in
        # test_concentration_analysis_route. Still HTTP 200, still explained,
        # and no longer shaped like a delivered answer.
        assert env["ok"] is False
        assert env["metadata"]["controlledUnsupported"] is True
        assert "acquired_009" in env["answer"]
        assert env["workflow"]["available"] is False

    def test_no_frame_degrades_honestly(self, semantics):
        env = _route("Compare the portfolios", semantics, frame_resolver=None)
        assert env["metadata"]["route"] == prc.WORKFLOW_ID
        assert any("insufficient-data" in w for w in env["warnings"])

    def test_unavailable_bsr_is_a_controlled_outcome(self, semantics, monkeypatch):
        monkeypatch.setattr(chat_routing, "load_business_semantics",
                            lambda: (_ for _ in ()).throw(OSError("gone")))
        env = _route("Compare the portfolios", semantics,
                     frame_resolver=lambda c, r: _frame())
        assert env["metadata"]["route"] == prc.WORKFLOW_ID
        assert "Business Semantics Registry" in env["answer"]
        assert any("business semantics registry unavailable" in w
                   for w in env["warnings"])

    def test_limitations_are_surfaced_as_warnings(self, semantics):
        df = _frame()
        df.loc[df.source_portfolio_id == "acquired_001",
               "exposure_currency_denomination"] = "EUR"
        env = _route("Compare direct_001 with acquired_001", semantics,
                     frame_resolver=lambda c, r: df)
        assert any("limitation:" in w and "FX" in w for w in env["warnings"])


# --------------------------------------------------------------------------- #
# Tenancy — enforced by the governed service before any comparison runs
# --------------------------------------------------------------------------- #
class TestTenancy:
    def test_conflicting_client_id_is_blocked_before_the_workflow(self, monkeypatch):
        from trakt_core.context import ExecutionContext
        from mi_agent_api import data_source
        from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query

        demo = sorted(_REPO_ROOT.glob("synthetic_demo/**/*canonical_typed.csv"))
        assert demo
        monkeypatch.setenv("MI_AGENT_DATA_CSV", str(demo[0]))
        monkeypatch.setenv("MI_AGENT_DATA_CACHE_TTL", "0")
        monkeypatch.setenv("MI_AGENT_LLM_PARSER", "off")
        data_source.reset_cache()
        try:
            result = execute_governed_mi_query(
                MiQueryRequest(question="Compare the portfolios",
                               client_id="some_other_tenant"),
                ExecutionContext.for_internal("ERE"))
        finally:
            data_source.reset_cache()
        assert result.error is not None
        assert result.error.code == "TENANT_MISMATCH"

    def test_single_portfolio_deployment_gets_a_controlled_explanation(self, monkeypatch):
        """The demo tape has no source-portfolio provenance: comparison must be
        refused with a governed explanation, through the full service path."""
        from trakt_core.context import ExecutionContext
        from mi_agent_api import data_source
        from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query

        demo = sorted(_REPO_ROOT.glob("synthetic_demo/**/*canonical_typed.csv"))
        monkeypatch.setenv("MI_AGENT_DATA_CSV", str(demo[0]))
        monkeypatch.setenv("MI_AGENT_DATA_CACHE_TTL", "0")
        monkeypatch.setenv("MI_AGENT_LLM_PARSER", "off")
        data_source.reset_cache()
        try:
            result = execute_governed_mi_query(
                MiQueryRequest(question="Compare the portfolios"),
                ExecutionContext.for_internal("ERE"))
        finally:
            data_source.reset_cache()
        payload = result.result
        assert payload is not None
        assert payload["metadata"]["route"] == prc.WORKFLOW_ID
        assert payload["workflow"]["available"] is False
        assert "provenance" in (payload["workflow"]["reason"] or "")
