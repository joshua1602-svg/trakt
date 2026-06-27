#!/usr/bin/env python3
"""mi_agent_api/tests/test_evolution.py

Funded / pipeline / forecast evolution (time series) over the governed monthly
runs and weekly pipeline extracts. Reuses the existing snapshot + pipeline loaders.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent_api import evolution as evo

_PIPELINE_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "client_001_mi_pack"


def _write_run(root: Path, client_id: str, run_id: str, reporting_date: str,
               n: int, balance_scale: float) -> None:
    rng = np.random.default_rng(abs(hash(run_id)) % (2**32))
    df = pd.DataFrame({
        "loan_identifier": [f"{run_id}_{i}" for i in range(n)],
        "current_outstanding_balance": (rng.uniform(50_000, 300_000, n) * balance_scale).round(2),
        "current_loan_to_value": rng.uniform(20, 70, n).round(1),
        "current_interest_rate": rng.uniform(3, 8, n).round(2),
        "youngest_borrower_age": rng.integers(60, 90, n),
        "broker_channel": rng.choice(["Alpha", "Beta", "Gamma"], n),
        "geographic_region_obligor": rng.choice(["North", "South East"], n),
        "reporting_date": [reporting_date] * n,
    })
    d = root / client_id / run_id / "output" / "central"
    d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / "18_central_lender_tape.csv", index=False)


@pytest.fixture()
def funded_root(tmp_path):
    warnings.simplefilter("ignore")
    root = tmp_path / "onboarding_output"
    _write_run(root, "client_001", "mi_2025_10", "2025-10-31", 40, 1.0)
    _write_run(root, "client_001", "mi_2025_11", "2025-11-30", 45, 1.1)
    return root


# --------------------------------------------------------------------------- #
# Funded evolution
# --------------------------------------------------------------------------- #
def test_funded_evolution_builds_multi_period_series(funded_root):
    out = evo.funded_evolution(funded_root, "client_001", "mi_2025_11")
    assert out["dataset"] == "funded"
    assert out["availableRunIds"] == ["mi_2025_10", "mi_2025_11"]
    assert len(out["periods"]) == 2
    assert out["singlePeriod"] is False
    for p in out["periods"]:
        m = p["metrics"]
        assert m["funded_balance"] > 0
        assert m["loan_count"] > 0
        assert m["wa_ltv"] is not None
        assert m["wa_interest_rate"] is not None
        # Per-period reconciliation + lineage.
        assert p["reconciliation"]["coverage_by_balance_pct"] == 100.0
        assert p["source_file"].endswith("18_central_lender_tape.csv")
    # Balance grows month-on-month (scale 1.0 -> 1.1, more loans).
    bals = [p["metrics"]["funded_balance"] for p in out["periods"]]
    assert bals[1] > bals[0]


def test_funded_evolution_has_breakdowns(funded_root):
    out = evo.funded_evolution(funded_root, "client_001", "mi_2025_11")
    assert "broker" in out["breakdowns"] and out["breakdowns"]["broker"]
    assert "region" in out["breakdowns"]
    # Each breakdown row carries a period + key + value.
    row = out["breakdowns"]["broker"][0]
    assert {"period", "key", "value"} <= set(row)


def test_funded_evolution_to_run_id_truncates(funded_root):
    out = evo.funded_evolution(funded_root, "client_001", "mi_2025_10")
    assert out["availableRunIds"] == ["mi_2025_10"]
    assert out["singlePeriod"] is True


def test_funded_evolution_no_data_is_controlled(tmp_path):
    out = evo.funded_evolution(tmp_path / "empty", "client_001", None)
    assert out["periods"] == []
    assert out["singlePeriod"] is True


# --------------------------------------------------------------------------- #
# Pipeline evolution
# --------------------------------------------------------------------------- #
def test_pipeline_evolution_from_weekly_extracts():
    warnings.simplefilter("ignore")
    out = evo.pipeline_evolution(_PIPELINE_FIXTURE, "client_001", None)
    assert out["dataset"] == "pipeline"
    assert len(out["periods"]) >= 2
    for p in out["periods"]:
        assert p["metrics"]["pipeline_amount"] is not None
        assert p["metrics"]["pipeline_case_count"] >= 0
        assert p["source_file"]
    # by-stage series is populated over time, day-level, with amount AND count.
    assert out["byStage"]
    assert {"period", "week", "stage", "value", "count"} <= set(out["byStage"][0])
    # Day-level period (not a YYYY-MM month) so weekly points are distinguishable.
    import re as _re
    assert _re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(out["byStage"][0]["period"]))
    assert all(isinstance(r["count"], int) for r in out["byStage"])


def test_pipeline_evolution_dedups_extracts():
    out = evo.pipeline_evolution(_PIPELINE_FIXTURE, "client_001", None)
    # Uses unique weekly extracts, not every raw file.
    assert out["uniqueWeeklyExtractsUsed"] is not None


# --------------------------------------------------------------------------- #
# Forecast evolution
# --------------------------------------------------------------------------- #
def test_forecast_evolution_combines_funded_and_pipeline(funded_root):
    out = evo.forecast_evolution(funded_root, _PIPELINE_FIXTURE, "client_001", "mi_2025_11")
    assert out["dataset"] == "forecast"
    assert len(out["periods"]) == 2
    for p in out["periods"]:
        m = p["metrics"]
        assert m["forecast_funded_balance"] >= m["funded_balance"]


# --------------------------------------------------------------------------- #
# API endpoints
# --------------------------------------------------------------------------- #
def test_evolution_endpoints(funded_root, monkeypatch):
    monkeypatch.setenv("MI_AGENT_ONBOARDING_OUTPUT_ROOT", str(funded_root))
    monkeypatch.setenv("MI_AGENT_PIPELINE_ROOT", str(_PIPELINE_FIXTURE))
    from fastapi.testclient import TestClient
    from mi_agent_api.app import app
    client = TestClient(app)

    f = client.get("/mi/evolution/funded", params={"portfolioId": "client_001/mi_2025_11"}).json()
    assert len(f["periods"]) == 2
    assert f["periods"][0]["reconciliation"]["coverage_by_balance_pct"] == 100.0

    p = client.get("/mi/evolution/pipeline", params={"portfolioId": "client_001"}).json()
    assert p["dataset"] == "pipeline"
    assert len(p["periods"]) >= 1

    fc = client.get("/mi/evolution/forecast", params={"portfolioId": "client_001/mi_2025_11"}).json()
    assert fc["dataset"] == "forecast"
    assert len(fc["periods"]) == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
