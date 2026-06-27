"""tests/test_chat_routing_e2e.py

End-to-end POST /mi/query routing (MI Agent Chatbot End-to-End Routing v1).

Proves the chat executes the new governed intents — temporal compare, evolution
trend, forecast scale-up, risk-limit oversight — through the internal services and
returns valid structured responses (answer + typed artifacts + reconciliation/
source notes), WITHOUT regressing existing point-in-time questions.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent_api.app import app

_PIPELINE_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "client_001_mi_pack"

client = TestClient(app)


def _write_run(root: Path, run_id: str, reporting_date: str, n: int, scale: float) -> None:
    rng = np.random.default_rng(abs(hash(run_id)) % (2**32))
    regions = rng.choice(["London", "South East", "Scotland", "Wales", "East"], n)
    df = pd.DataFrame({
        "loan_identifier": [f"{run_id}_{i}" for i in range(n)],
        "current_outstanding_balance": (rng.uniform(120_000, 280_000, n) * scale).round(2),
        "current_loan_to_value": rng.uniform(20, 55, n).round(1),
        "current_interest_rate": rng.uniform(3, 8, n).round(2),
        "youngest_borrower_age": rng.integers(62, 88, n),
        "broker_channel": rng.choice(["Alpha", "Beta", "Gamma", "Delta"], n),
        "geographic_region_obligor": regions,
        "reporting_date": [reporting_date] * n,
    })
    d = root / "client_001" / run_id / "output" / "central"
    d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / "18_central_lender_tape.csv", index=False)


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    warnings.simplefilter("ignore")
    monkeypatch.chdir(_REPO_ROOT)  # so config/clients/client_001/... resolves
    root = tmp_path / "onboarding_output"
    _write_run(root, "mi_2025_10", "2025-10-31", 60, 1.0)
    _write_run(root, "mi_2025_11", "2025-11-30", 70, 1.15)
    monkeypatch.setenv("MI_AGENT_ONBOARDING_OUTPUT_ROOT", str(root))
    monkeypatch.setenv("MI_AGENT_PIPELINE_ROOT", str(_PIPELINE_FIXTURE))
    yield


def _ask(question: str, view: str = "funded") -> dict:
    return client.post("/mi/query", json={
        "question": question, "portfolioId": "client_001/mi_2025_11",
        "datasetContext": view, "asOfDate": "2025-11-30",
    }).json()


def _types(resp: dict) -> set:
    return {a.get("type") for a in resp.get("artifacts", [])}


# --------------------------------------------------------------------------- #
# A. Temporal compare
# --------------------------------------------------------------------------- #
def test_compare_funded_balance_e2e():
    r = _ask("Compare October and November funded balance.")
    assert r["ok"] is True
    assert r["metadata"]["route"] == "temporal_compare"
    # Periods resolved from the month names to the governed reporting periods.
    assert "2025-10" in r["answer"] and "2025-11" in r["answer"]
    assert "chart" in _types(r) and "table" in _types(r)
    # The comparison table carries both period values + a delta.
    tbl = next(a for a in r["artifacts"] if a["type"] == "table")
    assert {"period_a", "period_b", "abs_delta", "pct_delta"} <= set(tbl["rows"][0].keys())
    assert r["reconciliation"] is not None


def test_compare_loan_count_uses_count():
    r = _ask("Compare October and November loan count.")
    assert r["ok"] is True and r["metadata"]["route"] == "temporal_compare"
    # Loan count, not balance.
    assert "count" in r["answer"].lower() or "loan" in r["answer"].lower()


def test_compare_single_period_controlled():
    # A pipeline compare where prior pipeline is unavailable in the thin fixture
    # must return a controlled insufficient-data answer, never a fabricated delta.
    r = _ask("Compare latest pipeline with prior pipeline.", view="pipeline")
    assert r["ok"] is True
    # Either a real comparison (if 2 extracts) or a controlled message.
    if not r["artifacts"]:
        assert "can't compare" in r["answer"].lower() or "one reporting period" in r["answer"].lower()


# --------------------------------------------------------------------------- #
# B. Evolution / trend
# --------------------------------------------------------------------------- #
def test_funded_balance_evolution_e2e():
    r = _ask("Show funded balance evolution by month.")
    assert r["ok"] is True and r["metadata"]["route"] == "evolution"
    chart = next(a for a in r["artifacts"] if a["type"] == "chart")
    assert chart["chartType"] == "line"
    assert len(chart["rows"]) == 2  # two governed funded runs
    assert chart["valueFormat"] == "gbp"


def test_loan_count_evolution_uses_count_e2e():
    r = _ask("Show loan count evolution by month.")
    assert r["ok"] is True and r["metadata"]["route"] == "evolution"
    chart = next(a for a in r["artifacts"] if a["type"] == "chart")
    # Count series → number format, not gbp.
    assert chart["valueFormat"] == "number"
    assert any("Loan count" in s["label"] for s in chart["series"])


def test_pipeline_amount_evolution_by_week_e2e():
    r = _ask("Show pipeline amount evolution by week.", view="pipeline")
    assert r["ok"] is True and r["metadata"]["route"] == "evolution"
    chart = next(a for a in r["artifacts"] if a["type"] == "chart")
    assert chart["chartType"] == "line" and chart["valueFormat"] == "gbp"


def test_pipeline_by_stage_over_time_e2e():
    r = _ask("Show pipeline by stage over time.", view="pipeline")
    assert r["ok"] is True
    assert r["metadata"]["route"] in ("evolution_pipeline_stage", "evolution")
    chart = next(a for a in r["artifacts"] if a["type"] == "chart")
    assert chart["chartType"] == "line"
    assert len(chart["series"]) >= 1


def test_kfi_trend_by_week_e2e():
    r = _ask("Show KFI trend by week.", view="pipeline")
    assert r["ok"] is True and r["metadata"]["route"] == "evolution_funnel"
    assert "chart" in _types(r) and "table" in _types(r)
    assert "5-week average" in r["answer"]


# --------------------------------------------------------------------------- #
# C. Forecast scale-up / extrapolation
# --------------------------------------------------------------------------- #
def test_reach_threshold_e2e():
    r = _ask("When do we reach £50m funded balance?")
    assert r["ok"] is True and r["metadata"]["route"] == "forecast_extrapolation"
    assert "£50" in r["answer"] or "50m" in r["answer"].lower()
    # Scenario-band caveat present.
    assert any("scenario band" in w.lower() for w in r["warnings"])


def test_extrapolation_curve_e2e():
    r = _ask("Show the funded balance extrapolation curve.")
    assert r["ok"] is True and r["metadata"]["route"] == "forecast_extrapolation"
    chart = next((a for a in r["artifacts"] if a["type"] == "chart"), None)
    if chart:  # available when there is >=2 funded runs (there are)
        assert {"downside", "base", "upside"} <= {s["key"] for s in chart["series"]}
    # Milestone table present.
    assert any(a["type"] == "table" for a in r["artifacts"])


def test_current_run_rate_e2e():
    r = _ask("What is the current completion run rate?")
    assert r["ok"] is True and r["metadata"]["route"] == "forecast_extrapolation"
    assert "run-rate" in r["answer"].lower() or "run rate" in r["answer"].lower()


# --------------------------------------------------------------------------- #
# D. Risk limits / concentration
# --------------------------------------------------------------------------- #
def test_risk_limit_breaches_e2e():
    r = _ask("Show risk limit breaches.")
    assert r["ok"] is True and r["metadata"]["route"] == "risk_limits"
    assert "passed" in r["answer"] and "breach" in r["answer"]
    # A comprehensive table of all tests is returned.
    tbl = next(a for a in r["artifacts"] if a["type"] == "table")
    cols = {c["key"] for c in tbl["columns"]}
    assert {"test", "actual", "limit", "headroom", "status", "movement", "source"} <= cols


def test_closest_to_breach_e2e():
    r = _ask("Which concentration limit is closest to breach?")
    assert r["ok"] is True and r["metadata"]["route"] == "risk_limits"
    assert "nearest to limit" in r["answer"].lower() or "headroom" in r["answer"].lower()


def test_geographic_concentration_limits_e2e():
    r = _ask("Show geographic concentration limits.")
    assert r["ok"] is True and r["metadata"]["route"] == "risk_limits"
    # Risk artifact (RAG groups) present where computable.
    assert any(a["type"] in ("risk", "table") for a in r["artifacts"])


# --------------------------------------------------------------------------- #
# E. Existing point-in-time behaviour must NOT regress (not routed)
# --------------------------------------------------------------------------- #
def test_point_in_time_balance_by_broker_not_routed():
    r = _ask("balance by broker")
    # Not handled by the new routing — served by the existing MI Agent path.
    # (Point-in-time success is covered by the existing MI suite; the active
    # dataset is an lru_cache shared across the test session, so only the
    # routing decision is asserted here.)
    assert r.get("metadata", {}).get("route") in (None, "")
    assert "ok" in r


def test_point_in_time_interest_rate_kpi_not_routed():
    r = _ask("interest rate")
    assert r.get("metadata", {}).get("route") in (None, "")
    assert "ok" in r


def test_forecast_funded_balance_not_routed_to_extrapolation():
    # The point-in-time bridge question must NOT become a scale-up extrapolation.
    r = _ask("forecast funded balance", view="forecast")
    assert r.get("metadata", {}).get("route") != "forecast_extrapolation"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
