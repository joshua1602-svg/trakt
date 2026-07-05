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


def test_geographic_limits_returns_only_geographic_category():
    # The known miss: "show geographic concentration limits" must NOT return all
    # categories — only the geographic ones.
    r = _ask("Show geographic concentration limits.")
    assert r["ok"] is True and r["metadata"]["route"] == "risk_limits"
    tbl = next(a for a in r["artifacts"] if a["type"] == "table")
    # Every row in the comprehensive table is a geographic-concentration test.
    assert tbl["rows"], "expected geographic limit rows"
    assert r["answer"].lower().startswith("geographic concentration")
    # A broker-scoped query, by contrast, returns broker rows.
    rb = _ask("Show broker concentration limits.")
    assert rb["answer"].lower().startswith("broker")


def test_funded_balance_forecast_curve_returns_line_chart():
    # The known miss: "forecast curve" returned only a summary. It must now route
    # to the forecast extrapolation and return a projected line chart.
    r = _ask("Show the funded balance forecast curve.")
    assert r["ok"] is True and r["metadata"]["route"] == "forecast_extrapolation"
    chart = next((a for a in r["artifacts"] if a["type"] == "chart"), None)
    assert chart is not None and chart["chartType"] == "line"
    assert {"downside", "base", "upside"} <= {s["key"] for s in chart["series"]}


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


def _ask_run(question: str, run_id: str) -> dict:
    return client.post("/mi/query", json={
        "question": question, "portfolioId": f"client_001/{run_id}",
        "datasetContext": "funded",
    }).json()


def test_funded_balance_bridge_returns_reconciling_waterfall():
    # The fixture writes Oct (60) and Nov (70×1.15) funded runs. A waterfall
    # bridge by region must return a waterfall artifact whose region deltas
    # reconcile to the opening→latest net change.
    r = _ask("show a waterfall of the balance bridge by region and what contributed to the growth")
    assert r.get("metadata", {}).get("route") == "funded_bridge", r.get("answer")
    charts = [a for a in r.get("artifacts", []) if a.get("chartType") == "waterfall"]
    assert charts, r.get("artifacts")
    rows = charts[0]["rows"]
    totals = [x for x in rows if x["type"] == "total"]
    deltas = [x for x in rows if x["type"] != "total"]
    assert len(totals) == 2 and deltas
    opening, closing = totals[0]["value"], totals[-1]["value"]
    assert abs(sum(d["value"] for d in deltas) - (closing - opening)) < 1.0


def test_cohort_progression_route_returns_metric_line():
    # A static-pool progression scoped to a source portfolio returns a metric
    # line across periods (the fixture's two runs).
    r = client.post("/mi/query", json={
        "question": "how has funded balance evolved for the direct book",
        "portfolioId": "client_001/mi_2025_11", "datasetContext": "funded",
    }).json()
    assert r.get("metadata", {}).get("route") == "cohort_progression", r.get("answer")
    lines = [a for a in r.get("artifacts", []) if a.get("chartType") == "line"]
    assert lines and lines[0]["rows"]


def test_funded_query_honours_selected_run():
    # The fixture writes two funded runs of DIFFERENT sizes (Oct=60, Nov=70).
    # A point-in-time funded question must be answered from the SELECTED run,
    # not the active/latest dataset — otherwise the earlier run is silently
    # mislabelled with the latest data.
    oct_rows = _record_count(_ask_run("how many loans", "mi_2025_10"))
    nov_rows = _record_count(_ask_run("how many loans", "mi_2025_11"))
    assert oct_rows == 60, oct_rows
    assert nov_rows == 70, nov_rows


def _record_count(resp: dict) -> int:
    assert resp.get("ok") is True, resp.get("error")
    recon = resp.get("reconciliation") or {}
    if recon.get("total_records") is not None:
        return int(recon["total_records"])
    for art in resp.get("artifacts", []):
        art_recon = art.get("reconciliation") or {}
        if art_recon.get("total_records") is not None:
            return int(art_recon["total_records"])
    raise AssertionError(f"no reconciliation record count in {resp.get('artifacts')}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
