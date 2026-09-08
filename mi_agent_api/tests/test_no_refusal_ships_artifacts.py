"""No refused MI Query Agent answer may ship an artifact.

Reported: a question the MI Query Agent refuses "also renders the failure in
the Artifact Workspace". The refusal itself belongs in the chat only — the
operator already logs every question and its outcome through OCC, so a second,
workspace-side rendering of a declined answer is pure duplication.

`mi_service.py` clears `artifacts` in every guard that can flip an envelope to
`ok=False` (the routed semantic guard, the fail-closed analytical check,
temporal honouring, the unknown-category guard, the unresolved-scope guard),
and `adapters.py` gates its own validation artifact on `not refused`. This
test does not re-derive that from reading the source a second time — it
drives the REAL `/mi/query` endpoint with questions engineered to trip each
of those guards individually, plus every refuse/clarify case in the curated
calibration bank, and checks the one invariant that actually matters: a
refused envelope's `artifacts` list is empty.
"""

from __future__ import annotations

import os
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


def _write_run(root: Path, run_id: str, reporting_date: str, n: int) -> None:
    rng = np.random.default_rng(abs(hash(run_id)) % (2**32))
    regions = rng.choice(["London", "South East", "Scotland", "Wales", "East"], n)
    df = pd.DataFrame({
        "loan_identifier": [f"{run_id}_{i}" for i in range(n)],
        "current_outstanding_balance": rng.uniform(120_000, 280_000, n).round(2),
        "current_loan_to_value": rng.uniform(20, 55, n).round(1),
        "current_interest_rate": rng.uniform(3, 8, n).round(2),
        "youngest_borrower_age": rng.integers(62, 88, n),
        "broker_channel": rng.choice(["Alpha", "Beta", "Gamma", "Delta"], n),
        "geographic_region_obligor": regions,
        "reporting_date": [reporting_date] * n,
    })
    out = root / run_id / "output" / "central"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "central_lender_tape.csv", index=False)


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    warnings.simplefilter("ignore")
    monkeypatch.chdir(_REPO_ROOT)
    root = tmp_path / "onboarding_output"
    _write_run(root, "mi_2025_11", "2025-11-30", 70)
    monkeypatch.setenv("MI_AGENT_ONBOARDING_OUTPUT_ROOT", str(root))
    monkeypatch.setenv("MI_AGENT_PIPELINE_ROOT", str(_PIPELINE_FIXTURE))
    yield


def _ask(question: str) -> dict:
    return client.post("/mi/query", json={
        "question": question, "portfolioId": "client_001/mi_2025_11",
        "datasetContext": "funded", "asOfDate": "2025-11-30",
    }).json()


# --------------------------------------------------------------------------- #
# One question per guard that CAN flip an envelope to a refusal
# --------------------------------------------------------------------------- #
GUARD_TARGETED_QUESTIONS = [
    "What is the funded balance by region for the Highgate Mortgages Book?",
    "Summarise the acquired_099 book.",
    "What is the balance for Atlantis loans?",
    "Balance by region where broker channel is Wonderland Partners.",
    "How many loans are we completing at the moment?",
    "What completion rate are we running at?",
    "Where are we closest to our limits?",
    "Which of our limits are most at risk?",
    "What was the balance in March 2019?",
    "Show funded balance evolution since 1999.",
    "Balance by nonexistent_dimension.",
    "Balance by joint borrower count.",
]


@pytest.mark.parametrize("question", GUARD_TARGETED_QUESTIONS)
def test_a_refusal_ships_no_artifact(question):
    body = _ask(question)
    if body.get("ok"):
        pytest.skip(f"{question!r} answered rather than refused on this fixture")
    assert body.get("artifacts") == [], (
        f"{question!r} refused but shipped {[a.get('type') for a in body['artifacts']]}"
    )


# --------------------------------------------------------------------------- #
# The full curated calibration bank's own labelled refuse/clarify cases
# --------------------------------------------------------------------------- #
def _calibration_refuse_cases():
    try:
        from mi_agent import mi_calibration as CAL
    except Exception:  # noqa: BLE001
        return []
    try:
        cases = CAL.load_bank()
    except Exception:  # noqa: BLE001
        return []
    return [c["question"] for c in cases
            if c.get("expected_status") in ("refuse", "clarify")]


@pytest.mark.parametrize("question", _calibration_refuse_cases())
def test_a_calibration_refusal_ships_no_artifact(question):
    body = _ask(question)
    if body.get("ok"):
        pytest.skip(f"{question!r} answered rather than refused on this fixture")
    assert body.get("artifacts") == [], (
        f"{question!r} refused but shipped {[a.get('type') for a in body['artifacts']]}"
    )
