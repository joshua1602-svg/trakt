"""tests/test_risk_limits.py

Risk-limit / concentration monitor (Part 5). Builds synthetic funded runs (the
same fixture style as test_evolution) and asserts:
  * extracted Schedule 8 limits drive the tests;
  * actual exposure, limit, headroom and status (green/amber/red) are computed;
  * geographic + broker concentration are covered;
  * missing fields -> unavailable (not fabricated);
  * ambiguous limits -> needs_review;
  * movement vs the prior run is reported.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent_api import risk_limits as rl


def _write_run(root: Path, client_id: str, run_id: str, reporting_date: str,
               n: int, london_share: float) -> None:
    rng = np.random.default_rng(abs(hash(run_id)) % (2**32))
    # Force a chunk of balance into London to exercise the 30% limit.
    regions = rng.choice(["London", "South East", "Scotland", "Wales", "East"], n)
    n_london = int(n * london_share)
    regions[:n_london] = "London"
    df = pd.DataFrame({
        "loan_identifier": [f"{run_id}_{i}" for i in range(n)],
        "current_outstanding_balance": rng.uniform(80_000, 250_000, n).round(2),
        "current_loan_to_value": rng.uniform(20, 55, n).round(1),
        "current_interest_rate": rng.uniform(3, 8, n).round(2),
        "youngest_borrower_age": rng.integers(62, 88, n),
        "broker_channel": rng.choice(["Alpha", "Beta", "Gamma", "Delta"], n),
        "geographic_region_obligor": regions,
        "reporting_date": [reporting_date] * n,
    })
    d = root / client_id / run_id / "output" / "central"
    d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / "18_central_lender_tape.csv", index=False)


@pytest.fixture()
def funded_root(tmp_path, monkeypatch):
    warnings.simplefilter("ignore")
    monkeypatch.chdir(_REPO_ROOT)  # so config/clients/client_001/... resolves
    root = tmp_path / "onboarding_output"
    _write_run(root, "client_001", "mi_2025_10", "2025-10-31", 60, 0.25)
    _write_run(root, "client_001", "mi_2025_11", "2025-11-30", 70, 0.45)
    return root


def test_limits_loaded_from_config():
    out = rl.load_extracted_limits("client_001")
    assert out["available"] is True
    assert out["limit_count"] >= 12
    assert "geographic_concentration" in out["categories"]


def test_risk_limits_envelope(funded_root):
    out = rl.compute_risk_limits(funded_root, "client_001", "mi_2025_11")
    assert out["available"] is True
    assert out["fundedDataAvailable"] is True
    assert out["reportingDate"] == "2025-11-30"
    assert out["summary"]["total"] >= 12
    # Geographic + broker categories covered.
    assert "geographic_concentration" in out["testsByCategory"]
    assert "broker_concentration" in out["testsByCategory"]


def test_london_concentration_actual_and_status(funded_root):
    out = rl.compute_risk_limits(funded_root, "client_001", "mi_2025_11")
    geo = out["testsByCategory"]["geographic_concentration"]
    london = next(t for t in geo if (t["region"] or "") == "London")
    assert london["limitValue"] == 30.0
    assert london["actualValue"] is not None
    assert london["status"] in ("green", "amber", "red")
    # headroom = limit - actual
    assert abs(london["headroom"] - (30.0 - london["actualValue"])) < 0.01
    assert london["source"]
    assert london["sourceSnippet"]


def test_movement_vs_prior(funded_root):
    out = rl.compute_risk_limits(funded_root, "client_001", "mi_2025_11")
    geo = out["testsByCategory"]["geographic_concentration"]
    london = next(t for t in geo if (t["region"] or "") == "London")
    # London share rose 25% -> 45% of count between runs, so movement is positive.
    assert london["movementVsPrior"] is not None


def test_top_n_broker_test_present(funded_root):
    out = rl.compute_risk_limits(funded_root, "client_001", "mi_2025_11")
    brokers = out["testsByCategory"]["broker_concentration"]
    assert any("Top" in t["label"] for t in brokers)
    assert any(t["actualValue"] is not None for t in brokers)


def test_missing_field_is_unavailable_not_fabricated(funded_root):
    out = rl.compute_risk_limits(funded_root, "client_001", "mi_2025_11")
    # Interest-rate (variable-rate) and borrower-concentration need fields absent
    # from the funded tape -> unavailable with the field listed.
    ir = out["testsByCategory"].get("interest_rate_limit", [])
    assert ir and ir[0]["status"] == "unavailable"
    assert ir[0]["missingFields"]
    assert ir[0]["actualValue"] is None


def test_ambiguous_limit_needs_review(funded_root):
    out = rl.compute_risk_limits(funded_root, "client_001", "mi_2025_11")
    jb = out["testsByCategory"].get("joint_borrower_limit", [])
    assert jb and jb[0]["status"] == "needs_review"


def test_unavailable_limits_controlled_when_no_config(tmp_path, monkeypatch):
    # Point at a client with no extracted limits and no Schedule 8 doc.
    monkeypatch.chdir(tmp_path)
    out = rl.compute_risk_limits(None, "client_999", None)
    assert out["available"] is False
    assert out["limitsStatus"] == "unavailable"
    assert out["tests"] == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
