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

from trakt_core.context import ExecutionContext
from mi_agent_api import data_source, mi_service
from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query

#: A trusted context, as an interface adapter would build one. Constructed here
#: WITHOUT importing FastAPI — the point of the capability boundary.
TEST_TENANT = "ERE"


def _context(tenant_id: str = TEST_TENANT, **kw) -> ExecutionContext:
    return ExecutionContext.for_internal(tenant_id, **kw)


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


def _run(question: str, *, context: ExecutionContext | None = None, **kw):
    """Invoke the governed capability and return the full ``GovernedResult``."""
    return execute_governed_mi_query(
        MiQueryRequest(question=question, **kw), context or _context())


def _ask(question: str, **kw) -> dict:
    """The analytical payload — what the channels present. Unchanged in shape."""
    return _run(question, **kw).result


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

    from mi_agent_api import datasets as datasets_mod
    monkeypatch.setattr(datasets_mod, "_resolve_query_frame", _boom)
    env = _ask("What is the total current balance?")
    assert env["ok"] is False
    assert "no governed dataset is configured" in env["error"]
    assert env["artifacts"] == []
    assert env["validation"]["ok"] is False


def test_frame_error_is_reported_not_downgraded(monkeypatch):
    from mi_agent_api import datasets as datasets_mod
    monkeypatch.setattr(datasets_mod, "_resolve_query_frame",
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


def test_explicit_portfolio_narrows_within_the_trusted_tenant(governed_dataset):
    """An explicit portfolio selects a book WITHIN the authenticated tenant."""
    result = _run("What is the total current balance?", portfolio_id="ERE/run_2026_01")
    meta = result.result["metadata"]
    assert meta["selectedClient"] == "ERE"
    assert meta["selectedPortfolio"] == "ERE/run_2026_01"
    assert result.portfolio_id == "ERE/run_2026_01"
    assert result.tenant_id == "ERE"


def test_conflicting_client_id_cannot_redefine_the_tenant(governed_dataset):
    """The defect this boundary exists to close.

    ``client_id`` used to be OR-ed into the effective portfolio and then split
    back out as the client, so a caller-supplied value silently replaced the
    authenticated tenant. It is now a deprecated portfolio fallback only, and a
    value that disagrees with the trusted tenant is refused outright.
    """
    result = _run("What is the total current balance?",
                  portfolio_id="ERE/run_2026_01", client_id="OTHER",
                  context=_context("ERE"))
    assert result.blocked
    assert result.error.code == "TENANT_MISMATCH"
    assert result.tenant_id == "ERE"
    # No analytical answer was produced for the mismatched caller.
    assert result.result["ok"] is False
    assert result.result["artifacts"] == []


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
# Fail-closed: production never answers from the synthetic demo dataset
#
# The refusal now lives in the capability (trakt_core.policy), not in the Copilot
# adapter, so it applies to EVERY channel. These tests set the production runtime
# mode explicitly; the suite default (see the repo-root conftest) is ``test``,
# which is what legitimately allows the fixtures the rest of this file uses.
# --------------------------------------------------------------------------- #
def _force_synthetic(monkeypatch):
    for var in ("MI_AGENT_DATA_CSV", "MI_AGENT_ANALYTICS_DATASET",
                "MI_AGENT_CENTRAL_TAPE", "MI_AGENT_ONBOARDING_OUTPUT_ROOT",
                "MI_AGENT_PLATFORM_URI", "MI_AGENT_PLATFORM_CANONICAL",
                "MI_AGENT_PLATFORM_DIR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MI_AGENT_DATA_CACHE_TTL", "0")
    monkeypatch.setenv("TRAKT_RUNTIME_MODE", "production")
    data_source.reset_cache()


def test_synthetic_dataset_is_refused_by_the_copilot_channel(monkeypatch):
    from fastapi.testclient import TestClient
    from mi_agent_api.app import app

    monkeypatch.setenv("TRAKT_COPILOT_AUTH_MODE", "disabled")
    _force_synthetic(monkeypatch)
    try:
        assert data_source.data_source_kind() == data_source.KIND_SYNTHETIC_DEMO
        r = TestClient(app).post("/v1/copilot/mi/query",
                                 json={"question": "What is the total balance?"})
        assert r.status_code == 503
        body = r.json()
        assert body["ok"] is False
        assert body["errorCode"] == "DATA_SOURCE_NOT_APPROVED"
    finally:
        data_source.reset_cache()


def test_synthetic_dataset_is_refused_by_the_react_channel(monkeypatch):
    """The gap the central policy closes: React had no source-approval check at
    all, so it would answer a production question from the demo dataset."""
    from fastapi.testclient import TestClient
    from mi_agent_api.app import app

    _force_synthetic(monkeypatch)
    try:
        r = TestClient(app).post("/mi/query",
                                 json={"question": "What is the total balance?"})
        assert r.status_code == 503
        body = r.json()
        assert body["ok"] is False
        assert body["governance"]["error"]["code"] == "DATA_SOURCE_NOT_APPROVED"
        assert body["governance"]["policy"]["data_approved"] is False
    finally:
        data_source.reset_cache()


def test_synthetic_dataset_is_refused_for_a_direct_python_caller(monkeypatch):
    """And for an in-process caller — no HTTP layer involved in the refusal."""
    _force_synthetic(monkeypatch)
    try:
        result = _run("What is the total balance?")
        assert result.blocked
        assert result.error.code == "DATA_SOURCE_NOT_APPROVED"
        assert result.policy.data_approved is False
    finally:
        data_source.reset_cache()


def test_explicit_csv_pointing_at_the_demo_pack_is_also_refused(monkeypatch):
    """The hole the old adapter-level check missed.

    ``MI_AGENT_DATA_CSV=<the synthetic demo CSV>`` resolves to display kind
    ``explicit_csv``, so the previous ``kind in {synthetic_demo, unavailable}``
    block did not fire and Copilot answered from demo data. The policy now
    decides on the resolution BASE, so a fixture is a fixture whichever
    environment variable pointed at it.
    """
    monkeypatch.setenv("MI_AGENT_DATA_CSV", str(_demo_csv()))
    monkeypatch.setenv("MI_AGENT_DATA_CACHE_TTL", "0")
    monkeypatch.setenv("TRAKT_RUNTIME_MODE", "production")
    data_source.reset_cache()
    try:
        assert data_source.data_source_kind() == data_source.KIND_EXPLICIT_CSV
        result = _run("What is the total balance?")
        assert result.blocked
        assert result.error.code == "DATA_SOURCE_NOT_APPROVED"
    finally:
        data_source.reset_cache()
