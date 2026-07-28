"""Concentration Analysis — recogniser registration and the API adapter.

The pure workflow is covered in ``tests/test_concentration_analysis_workflow.py``.
Here: how the workflow lands in the platform — the recogniser's declaration
(priority, confidence, BSR terms in ``metadata``), dispatch through
``chat_routing.try_route`` (including that it does not steal questions owned
by the ITL3 geographic exposure route, the risk-limit monitor, the ranked
stratification path or the portfolio comparison workflow), the adapter's
envelope, and controlled degradation when the BSR or frame is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mi_agent.mi_query_validator import load_mi_semantics
from mi_agent.parsed_question import ParsedQuestion
from mi_agent_api import chat_routing
from mi_agent_api.recogniser_registry import REGISTRY
from mi_workflows import concentration_analysis as conc
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
        "loan_identifier": [f"L{i}" for i in range(8)],
        "current_outstanding_balance": [100.0, 200.0, 300.0, 400.0,
                                        150.0, 250.0, 350.0, 450.0],
        "exposure_currency_denomination": ["GBP"] * 8,
        "product_type": ["lump", "lump", "draw", "draw",
                         "lump", "draw", "draw", None],
        "origination_channel": ["broker", "direct", "broker", "broker",
                                "direct", "broker", None, "broker"],
        "geographic_region_obligor": ["london", "london", "north", "south",
                                      "london", "north", "south", "london"],
    })


def _route(question, semantics, frame_resolver=None, **kw):
    return chat_routing.try_route(
        question, portfolio_id="client_001", view="funded", output_root=None,
        pipeline_root=None, semantics=semantics,
        frame_resolver=frame_resolver, **kw)


def _candidates(question, semantics):
    parsed = ParsedQuestion.parse(question, semantics)
    request = chat_routing.RouteRequest(
        question=question, spec=parsed.spec, spec_dict=parsed.spec.to_dict(),
        semantics=semantics, view="funded", client_id="client_001",
        run_id=None, portfolio_id="client_001", parse_meta=parsed.meta)
    return [rec.name for rec, _ in REGISTRY.candidates(request)]


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
class TestRegistration:
    def test_registered_with_workflow_confidence_and_bsr_terms(self):
        rec = REGISTRY.get(conc.WORKFLOW_ID)
        assert rec is not None
        assert rec.lens_aware is True
        assert rec.capability is None            # funded-book routes are ungated
        assert rec.metadata["bsr_dimension_category"] == "concentration"
        assert "analytical_role" in rec.metadata["bsr_axes_consumed"]
        assert "portfolio_comparability" in rec.metadata["bsr_axes_consumed"]
        assert rec.metadata["concentration_bases"] == ("exposure", "count")

    def test_priority_sits_between_the_comparison_workflow_and_period_movement(self):
        names = list(REGISTRY.names())
        assert names.index(prc.WORKFLOW_ID) < names.index(conc.WORKFLOW_ID) \
            < names.index("period_movement")


# --------------------------------------------------------------------------- #
# Dispatch — matches its questions, steals nothing
# --------------------------------------------------------------------------- #
class TestDispatch:
    def test_broker_concentration_routes_to_the_workflow(self, semantics):
        env = _route("Show broker concentration", semantics,
                     frame_resolver=lambda c, r: _frame())
        assert env is not None
        assert env["metadata"]["route"] == conc.WORKFLOW_ID
        assert env["ok"] is True
        assert env["metadata"]["lensApplied"] is True

    def test_workflow_confidence_outranks_lower_priority_matches(self, semantics):
        candidates = _candidates("Show product concentration", semantics)
        assert candidates and candidates[0] == conc.WORKFLOW_ID

    def test_itl3_geographic_questions_are_not_stolen(self, semantics):
        for question in ("What is the largest geographic area concentration?",
                         "Where is the book concentrated geographically?",
                         "Show geographic concentration"):
            candidates = _candidates(question, semantics)
            assert conc.WORKFLOW_ID not in candidates, question
            assert "geo_exposure" in candidates, question

    def test_limit_questions_are_not_stolen(self, semantics):
        for question in ("Show geographic concentration limits.",
                         "Which concentration limit is closest to breach?"):
            candidates = _candidates(question, semantics)
            assert conc.WORKFLOW_ID not in candidates, question
            assert "risk_limits" in candidates, question

    def test_portfolio_comparisons_are_not_stolen(self, semantics):
        candidates = _candidates("Compare product mix across the SPVs", semantics)
        assert conc.WORKFLOW_ID not in candidates
        assert prc.WORKFLOW_ID in candidates

    def test_ranked_stratifications_fall_through_to_the_executor(self, semantics):
        for question in ("top 5 brokers by balance", "balance by broker"):
            candidates = _candidates(question, semantics)
            assert conc.WORKFLOW_ID not in candidates, question

    def test_period_questions_are_not_stolen(self, semantics):
        candidates = _candidates(
            "How has broker concentration changed since last month?", semantics)
        assert conc.WORKFLOW_ID not in candidates


# --------------------------------------------------------------------------- #
# The adapter envelope
# --------------------------------------------------------------------------- #
class TestAdapterEnvelope:
    def test_envelope_carries_workflow_result_tables_and_governance_notes(self, semantics):
        env = _route("Show product concentration", semantics,
                     frame_resolver=lambda c, r: _frame())
        assert env["metadata"]["route"] == conc.WORKFLOW_ID
        wf = env["workflow"]
        assert wf["workflow_id"] == conc.WORKFLOW_ID
        assert wf["available"] is True
        assert wf["concentration_basis"] == "exposure"
        assert wf["audit"]["bsr_version"]
        titles = [a["title"] for a in env["artifacts"]]
        assert any("concentration" in t for t in titles)
        assert env["reconciliation"]["dataset"] == "funded"
        assert env["reconciliation"]["rows_in_scope"] == 8
        assert any("Business Semantics Registry" in n["note"]
                   for n in env["sourceNotes"])
        assert "Largest product type exposure" in env["answer"]

    def test_the_unknown_bucket_is_an_explicit_table_row(self, semantics):
        env = _route("Show product concentration", semantics,
                     frame_resolver=lambda c, r: _frame())
        table = next(a for a in env["artifacts"] if a["type"] == "table")
        categories = [r["category"] for r in table["rows"]]
        assert conc.engine.UNKNOWN_CATEGORY in categories

    def test_top_exposures_produce_a_single_name_table(self, semantics):
        env = _route("Show top exposures", semantics,
                     frame_resolver=lambda c, r: _frame())
        assert env["metadata"]["route"] == conc.WORKFLOW_ID
        assert env["workflow"]["single_name_results"]
        titles = [a["title"] for a in env["artifacts"]]
        assert any("Largest loan exposures" in t for t in titles)

    def test_the_question_scope_narrows_the_answer(self, semantics):
        env = _route("Show product concentration for the acquired book",
                     semantics, frame_resolver=lambda c, r: _frame())
        assert env["metadata"]["route"] == conc.WORKFLOW_ID
        assert env["workflow"]["portfolio_scope"]["context_id"] == "acquired"
        assert env["workflow"]["portfolio_scope"]["row_count"] == 4

    def test_answer_is_observational(self, semantics):
        env = _route("How diversified is the portfolio?", semantics,
                     frame_resolver=lambda c, r: _frame())
        assert env["metadata"]["route"] == conc.WORKFLOW_ID
        low = env["answer"].lower()
        for forbidden in ("excessive", "breach", "appetite", "within limits",
                          "acceptable", "insufficient"):
            assert forbidden not in low

    def test_controlled_failure_is_explained_not_500(self, semantics):
        env = _route("Show product concentration for acquired_009", semantics,
                     frame_resolver=lambda c, r: _frame())
        assert env["metadata"]["route"] == conc.WORKFLOW_ID
        assert env["ok"] is True                 # controlled, not an error
        assert "acquired_009" in env["answer"]
        assert env["workflow"]["available"] is False

    def test_no_frame_degrades_honestly(self, semantics):
        env = _route("Show broker concentration", semantics, frame_resolver=None)
        assert env["metadata"]["route"] == conc.WORKFLOW_ID
        assert any("insufficient-data" in w for w in env["warnings"])

    def test_unavailable_bsr_is_a_controlled_outcome(self, semantics, monkeypatch):
        monkeypatch.setattr(chat_routing, "load_business_semantics",
                            lambda: (_ for _ in ()).throw(OSError("gone")))
        env = _route("Show broker concentration", semantics,
                     frame_resolver=lambda c, r: _frame())
        assert env["metadata"]["route"] == conc.WORKFLOW_ID
        assert "Business Semantics Registry" in env["answer"]
        assert any("business semantics registry unavailable" in w
                   for w in env["warnings"])

    def test_limitations_are_surfaced_as_warnings(self, semantics):
        df = _frame()
        df.loc[df.index[:2], "exposure_currency_denomination"] = "USD"
        env = _route("Show product concentration", semantics,
                     frame_resolver=lambda c, r: df)
        assert env["workflow"]["concentration_basis"] == "count"
        assert any("limitation:" in w and "FX" in w for w in env["warnings"])


# --------------------------------------------------------------------------- #
# Through the governed service — tenancy first, workflow second
# --------------------------------------------------------------------------- #
class TestGovernedService:
    def test_the_workflow_answers_through_the_full_service_path(self, monkeypatch):
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
                MiQueryRequest(question="Show top exposures"),
                ExecutionContext.for_internal("ERE"))
        finally:
            data_source.reset_cache()
        payload = result.result
        assert payload is not None
        assert payload["metadata"]["route"] == conc.WORKFLOW_ID
        # The workflow contract rides the governed envelope on every channel;
        # whether this tape supports the measurement, the outcome is controlled.
        assert "workflow" in payload
        assert payload["workflow"]["workflow_id"] == conc.WORKFLOW_ID
