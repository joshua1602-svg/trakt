"""A governed field restricted to one asset class needs the book's asset class.

WHAT THIS PINS, AND WHAT IT DELIBERATELY DOES NOT. `broker_channel` is declared
in the Business Semantics Registry with `asset_applicability: [equity_release]`.
A book whose asset class is unidentified therefore cannot use it as a governed
period-change dimension, and the route says so by name.

That refusal is CORRECT. This module does not weaken applicability to make a
question answer; it proves the three outcomes are the three the policy intends,
and that the missing piece was CARRIAGE — the asset class is decided by
onboarding and supplied through the governed portfolio registry, and the
example registry a client copies did not mention it.

    no asset class          → refused, by name
    asset_class: equity_release   → answered and ranked
    asset_class: residential_mortgage → refused, by name

Point-in-time questions over the same field answer in all three cases, so the
field is not "broken in MI": it is the governed period-change dimension set that
applicability gates.
"""
from __future__ import annotations

import os
import sys

import pytest

QUESTION_RANKED = "Which broker channel added the most balance since last month?"
POINT_IN_TIME = ("Balance by broker channel.",
                 "How many loans by broker channel?",
                 "Balance by broker channel for loans with LTV above 50%.")


def _write_run(root, run_id, reporting_date, rows, scale, portfolio_id):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(sum(ord(c) for c in run_id))
    folder = root / "client_001" / run_id / "output" / "central"
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "loan_identifier": [f"{run_id}_{i}" for i in range(rows)],
        "current_outstanding_balance":
            (rng.uniform(120_000, 280_000, rows) * scale).round(2),
        "current_loan_to_value": rng.uniform(20, 55, rows).round(1),
        "current_interest_rate": rng.uniform(3, 8, rows).round(2),
        "youngest_borrower_age": rng.integers(62, 88, rows),
        "broker_channel": rng.choice(["Alpha", "Beta", "Gamma", "Delta"], rows),
        "geographic_region_obligor":
            rng.choice(["London", "South East", "Scotland"], rows),
        "source_portfolio_id": [portfolio_id] * rows,
        "source_portfolio_type": ["direct"] * rows,
        "reporting_date": [reporting_date] * rows,
    }).to_csv(folder / "18_central_lender_tape.csv", index=False)


def _book(tmp_path, asset_class):
    """A two-snapshot book, optionally with a governed asset class."""
    root = tmp_path / "onboarding_output"
    for run_id, date, rows, scale in (("mi_2026_05", "2026-05-31", 60, 1.0),
                                      ("mi_2026_06", "2026-06-30", 64, 1.12)):
        _write_run(root, run_id, date, rows, scale, "direct_001")
    registry = tmp_path / "portfolio_registry.yaml"
    if asset_class:
        registry.write_text(
            "portfolios:\n  - source_portfolio_id: direct_001\n"
            "    source_portfolio_type: direct\n"
            f"    asset_class: {asset_class}\n", encoding="utf-8")
    return root, (registry if asset_class else None)


def _ask(root, registry, question):
    """A fresh app against this book. MI modules are reloaded so the registry
    overlay is read for THIS case rather than memoised from the last one."""
    os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"] = str(root)
    os.environ["MI_AGENT_AUTH_ENABLED"] = "false"
    if registry is not None:
        os.environ["TRAKT_PORTFOLIO_REGISTRY"] = str(registry)
    else:
        os.environ.pop("TRAKT_PORTFOLIO_REGISTRY", None)
    for name in [n for n in list(sys.modules)
                 if n.startswith(("mi_agent", "mi_agent_api", "trakt_core"))]:
        sys.modules.pop(name, None)
    from fastapi.testclient import TestClient
    from mi_agent_api.app import app
    response = TestClient(app).post("/mi/query", json={
        "question": question, "portfolioId": "client_001/mi_2026_06",
        "asOfDate": "2026-06-30"}).json()
    rows = max([len(a.get("rows") or [])
                for a in (response.get("artifacts") or [])] or [0])
    return response, rows


def test_an_unidentified_asset_class_refuses_by_name(tmp_path):
    root, registry = _book(tmp_path, None)
    response, _rows = _ask(root, registry, QUESTION_RANKED)
    assert response["ok"] is False
    answer = response.get("answer") or ""
    assert "broker channel" in answer
    assert "not a governed period-change dimension" in answer
    assert "have not ranked a different dimension" in answer


def test_the_governed_asset_class_makes_the_field_usable(tmp_path):
    root, registry = _book(tmp_path, "equity_release")
    response, rows = _ask(root, registry, QUESTION_RANKED)
    assert response["ok"] is True, response.get("answer")
    assert rows > 0
    ranked = (response["metadata"].get("rankedMovement") or {})
    assert ranked.get("applied") is True
    assert ranked.get("canonicalField") == "broker_channel"


def test_a_different_asset_class_still_refuses(tmp_path):
    """Applicability is not weakened: the field is equity-release only."""
    root, registry = _book(tmp_path, "residential_mortgage")
    response, _rows = _ask(root, registry, QUESTION_RANKED)
    assert response["ok"] is False
    assert "not a governed period-change dimension" in (response.get("answer") or "")


@pytest.mark.parametrize("question", POINT_IN_TIME)
@pytest.mark.parametrize("asset_class", [None, "equity_release"])
def test_point_in_time_questions_answer_either_way(tmp_path, question,
                                                   asset_class):
    root, registry = _book(tmp_path, asset_class)
    response, rows = _ask(root, registry, question)
    assert response["ok"] is True, response.get("answer")
    assert rows > 0, question


def test_the_example_registry_documents_the_asset_class():
    """A client copies the example. If it omits this, the field is never usable."""
    import yaml
    from pathlib import Path
    doc = yaml.safe_load(
        Path("config/client/portfolio_registry.example.yaml").read_text())
    portfolios = doc.get("portfolios") or []
    assert portfolios, "the example registry declares no portfolios"
    assert any(p.get("asset_class") for p in portfolios), \
        "the example registry does not show asset_class"
