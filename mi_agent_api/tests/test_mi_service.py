#!/usr/bin/env python3
"""mi_agent_api/tests/test_mi_service.py

Unit tests for THE shared governed MI application service
(``mi_agent_api.mi_service.execute_governed_mi_query``) — the single analytical
entrypoint both the React MI Agent and Microsoft 365 Copilot call.

Covered:
  * point-in-time KPI, geography, LTV distribution, top-N and filtered queries
    all answer from the ACTIVE governed dataset with NO run id;
  * a governed-unsupported concept returns a controlled answer, not a guess;
  * an unavailable dataset returns a controlled governed error, never a raw 500;
  * the synthetic demo dataset is refused at the Copilot channel (fail closed);
  * the governance block (selected client / portfolio / run, data-source kind
    and label) is stamped on every envelope, with selectedRun set only where the
    analytical intent genuinely required a run.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest

from mi_agent_api import data_source, mi_service
from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query


def _demo_csv() -> Path:
    hits = sorted(_REPO_ROOT.glob("synthetic_demo/**/*canonical_typed.csv"))
    assert hits, "expected the bundled synthetic demo canonical CSV"
    return hits[0]


@pytest.fixture()
def governed_dataset(monkeypatch):
    """A governed (explicitly configured) live dataset. The demo CSV supplied via
    MI_AGENT_DATA_CSV resolves as kind=explicit_csv, not the synthetic fallback."""
    monkeypatch.setenv("MI_AGENT_DATA_CSV", str(_demo_csv()))
    monkeypatch.setenv("MI_AGENT_DATA_CACHE_TTL", "0")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)   # deterministic parser
    monkeypatch.setenv("MI_AGENT_LLM_PARSER", "off")
    data_source.reset_cache()
    yield
    data_source.reset_cache()


def _ask(question: str, **kw) -> dict:
    return execute_governed_mi_query(MiQueryRequest(question=question, **kw))


def _rows(envelope: dict) -> list:
    for art in envelope.get("artifacts") or []:
        if art.get("type") in ("table", "chart") and art.get("rows"):
            return art["rows"]
    return []


def _dimension(envelope: dict):
    """The governed dimension the executor grouped by (spec key is singular)."""
    spec = envelope.get("spec") or {}
    return spec.get("dimension") or (spec.get("dimensions") or [None])[0] or spec.get("x")


def _kpis(envelope: dict) -> list:
    for art in envelope.get("artifacts") or []:
        if art.get("type") == "kpi":
            return art.get("kpis") or []
    return []


# --------------------------------------------------------------------------- #
# Point-in-time analysis — no run id required
# --------------------------------------------------------------------------- #
def test_point_in_time_kpi_from_the_active_dataset(governed_dataset):
    env = _ask("What is the total current balance?")
    assert env["ok"] is True, env.get("error")
    assert _kpis(env) or _rows(env)
    meta = env["metadata"]
    # A point-in-time question is NOT run-scoped: no run was selected or needed.
    assert meta["selectedRun"] is None
    assert meta["runRequired"] is False
    assert meta["dataSourceKind"] == data_source.KIND_EXPLICIT_CSV


def test_geography_concentration_needs_no_run_id(governed_dataset):
    """The regression this refactor exists for: a point-in-time geographic
    question used to route into a run-specific funded-frame path and fail with
    'no funded frame for the run' whenever no run id was supplied."""
    env = _ask("Where is the book most concentrated by region?")
    assert env["ok"] is True
    assert env["metadata"]["route"] == "geo_exposure"
    assert env["metadata"]["selectedRun"] is None
    assert "can't resolve the funded book" not in env["answer"]
    assert not any("no funded frame" in str(w) for w in env["warnings"])
    assert _rows(env), "expected ranked geographic exposure rows"


def test_balance_by_geography_uses_the_governed_geography_field(governed_dataset):
    env = _ask("Show balance by collateral region.")
    assert env["ok"] is True
    assert "region" in str(_dimension(env)), env["spec"]
    assert _rows(env)


def test_ltv_distribution_needs_no_run_id(governed_dataset):
    env = _ask("Show balance by LTV bucket.")
    assert env["ok"] is True
    assert env["metadata"]["selectedRun"] is None
    # The governed bucket dimension from the semantic registry — not a
    # channel-specific re-bucketing of current_loan_to_value.
    assert _dimension(env) == "ltv_bucket", env["spec"]
    assert _rows(env), "expected LTV distribution rows"


def test_wa_ltv_is_a_point_in_time_metric(governed_dataset):
    env = _ask("What is the weighted average LTV?")
    assert env["ok"] is True
    assert env["metadata"]["selectedRun"] is None
    assert _kpis(env) or _rows(env)


def test_top_n_ranking(governed_dataset):
    env = _ask("Show the top 10 loans by current balance.")
    assert env["ok"] is True
    rows = _rows(env)
    assert 0 < len(rows) <= 10
    assert env["spec"].get("top_n") in (10, None)


def test_filtered_query_carries_its_filters(governed_dataset):
    env = _ask("How many borrowers are aged 75 or above?")
    assert env["ok"] is True
    assert env["spec"].get("filters"), env["spec"]


# --------------------------------------------------------------------------- #
# Governed limits — controlled, never invented
# --------------------------------------------------------------------------- #
def test_unsupported_concept_is_controlled_not_invented(governed_dataset):
    env = _ask("How many loans have negative equity guarantee?")
    # The engine must say what it cannot do, and must not fabricate a figure.
    assert env["metadata"]["controlledUnsupported"] is True
    assert env["ok"] is False
    assert "not available in this dataset" in env["answer"]
    assert env["artifacts"] == []


def test_unavailable_dataset_is_a_controlled_governed_error(monkeypatch):
    monkeypatch.setenv("MI_AGENT_LLM_PARSER", "off")

    def _boom(_view, _portfolio_id):
        raise FileNotFoundError("no governed dataset is configured")

    from mi_agent_api import app as app_mod
    monkeypatch.setattr(app_mod, "_resolve_query_frame", _boom)
    env = _ask("What is the total current balance?")
    assert env["ok"] is False
    assert "no governed dataset is configured" in env["error"]
    assert env["artifacts"] == []
    assert env["validation"]["ok"] is False


def test_frame_error_is_reported_not_downgraded(monkeypatch):
    from mi_agent_api import app as app_mod
    monkeypatch.setattr(app_mod, "_resolve_query_frame",
                        lambda _v, _p: (None, "No governed pipeline data is available."))
    env = _ask("What is the pipeline amount?", dataset_context="pipeline")
    assert env["ok"] is False
    assert "No governed pipeline data" in env["error"]


# --------------------------------------------------------------------------- #
# Governance block
# --------------------------------------------------------------------------- #
def test_client_context_scopes_the_query_without_a_run(governed_dataset):
    env = _ask("What is the total current balance?", client_id="ERE")
    meta = env["metadata"]
    assert meta["selectedClient"] == "ERE"
    assert meta["selectedPortfolio"] == "ERE"
    assert meta["selectedRun"] is None


def test_explicit_portfolio_overrides_the_client_context(governed_dataset):
    env = _ask("What is the total current balance?",
               portfolio_id="ERE/run_2026_01", client_id="OTHER")
    meta = env["metadata"]
    assert meta["selectedClient"] == "ERE"
    assert meta["selectedPortfolio"] == "ERE/run_2026_01"


def test_split_portfolio_forms():
    assert mi_service.split_portfolio(None) == ("client_001", None)
    assert mi_service.split_portfolio("ERE") == ("ERE", None)
    assert mi_service.split_portfolio("ERE/run_1") == ("ERE", "run_1")
    assert mi_service.split_portfolio("ERE/") == ("ERE", None)


def test_only_genuinely_temporal_routes_are_run_scoped():
    assert mi_service._route_requires_run("temporal_compare") is True
    assert mi_service._route_requires_run("evolution") is True
    assert mi_service._route_requires_run("cohort_progression") is True
    assert mi_service._route_requires_run("forecast_extrapolation") is True
    # Geographic exposure is point-in-time — it must NEVER require a run.
    assert mi_service._route_requires_run("geo_exposure") is False
    assert mi_service._route_requires_run(None) is False


# --------------------------------------------------------------------------- #
# Fail-closed: the synthetic demo dataset never reaches Copilot
# --------------------------------------------------------------------------- #
def test_synthetic_dataset_is_refused_by_the_copilot_channel(monkeypatch):
    from fastapi.testclient import TestClient
    from mi_agent_api.app import app

    monkeypatch.setenv("TRAKT_COPILOT_AUTH_MODE", "disabled")
    for var in ("MI_AGENT_DATA_CSV", "MI_AGENT_ANALYTICS_DATASET",
                "MI_AGENT_CENTRAL_TAPE", "MI_AGENT_ONBOARDING_OUTPUT_ROOT",
                "MI_AGENT_PLATFORM_URI", "MI_AGENT_PLATFORM_CANONICAL",
                "MI_AGENT_PLATFORM_DIR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MI_AGENT_DATA_CACHE_TTL", "0")
    data_source.reset_cache()
    try:
        assert data_source.data_source_kind() == data_source.KIND_SYNTHETIC_DEMO
        r = TestClient(app).post("/v1/copilot/mi/query",
                                 json={"question": "What is the total balance?"})
        assert r.status_code == 503
        assert r.json()["ok"] is False
    finally:
        data_source.reset_cache()
