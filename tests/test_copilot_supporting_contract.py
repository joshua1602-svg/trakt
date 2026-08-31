#!/usr/bin/env python3
"""tests/test_copilot_supporting_contract.py — what Trakt hands a language model.

M365 Copilot IS the language model. Trakt does not call one; it serves Copilot a
payload and Copilot phrases it. So the safety question is not "does Trakt let an
LLM calculate" but "does Trakt hand a language model a table with enough context
that it need not, and enough labelling that it cannot do so silently".

Two things were missing, and both are now asserted here:

  * ``totalRows`` reported the rows in hand, not the population. A chart the
    adapter had already capped from thirteen brokers to ten was handed over as
    ``truncated: False, totalRows: 10`` — a complete population, to a reader
    with no way to know otherwise.
  * The additivity and completeness contract React now consumes was not passed
    on at all, so a model holding a column of money-formatted AVERAGES had
    nothing telling it those must not be added.

The route itself is fail-closed: without Entra configuration it answers 503.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

BROKERS = [f"Broker {c}" for c in "ABCDEFGHIJKLM"]     # 13 > the top-10 chart cap
ROUTE = "/v1/copilot/mi/query"
QUESTION = "What is the average balance by broker channel?"


@pytest.fixture()
def book(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd

    root = tmp_path / "runs"
    for run_id, date, n in (("mi_2026_05", "2026-05-31", 120),
                            ("mi_2026_06", "2026-06-30", 130)):
        d = root / "client_001" / run_id / "central"
        d.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(7)
        pd.DataFrame({
            "loan_identifier": [f"{run_id}_{i}" for i in range(n)],
            "unique_identifier": [f"{run_id}_{i}" for i in range(n)],
            "source_portfolio_id": "direct_001", "source_portfolio_type": "direct",
            "current_outstanding_balance": rng.uniform(120_000, 480_000, n).round(2),
            "current_valuation_amount": rng.uniform(500_000, 1_200_000, n).round(2),
            "current_loan_to_value": rng.uniform(20, 70, n).round(1),
            "current_interest_rate": rng.uniform(5, 8, n).round(2),
            "youngest_borrower_age": rng.integers(60, 88, n),
            "broker_channel": [BROKERS[i % len(BROKERS)] for i in range(n)],
            "geographic_region_collateral": "London",
            "origination_date": "2022-05-01", "data_cut_off_date": date,
        }).to_csv(d / "18_central_lender_tape.csv", index=False)
    monkeypatch.setenv("MI_AGENT_ONBOARDING_OUTPUT_ROOT", str(root))
    monkeypatch.setenv("MI_AGENT_AUTH_ENABLED", "false")
    monkeypatch.setenv("TRAKT_RUNTIME_MODE", "test")
    return root


def _client():
    from fastapi.testclient import TestClient
    from mi_agent_api.app import app
    return TestClient(app)


def _ask(book, monkeypatch):
    monkeypatch.setenv("TRAKT_COPILOT_AUTH_MODE", "disabled")
    body = _client().post(ROUTE, json={
        "question": QUESTION, "portfolioId": "client_001/mi_2026_06"}).json()
    assert body.get("ok"), body.get("error")
    return body


# --------------------------------------------------------------------------- #
# The route is fail-closed.
# --------------------------------------------------------------------------- #

def test_the_route_refuses_without_entra_configuration(book, monkeypatch):
    """Catches: a Copilot surface reachable by default.

    ``entra`` is the default mode and an unconfigured deployment must answer
    503, not serve portfolio data.
    """
    monkeypatch.delenv("TRAKT_COPILOT_AUTH_MODE", raising=False)
    for var in ("TRAKT_COPILOT_ENTRA_TENANT_ID", "TRAKT_COPILOT_ENTRA_AUDIENCE"):
        monkeypatch.delenv(var, raising=False)
    res = _client().post(ROUTE, json={"question": QUESTION})
    assert res.status_code == 503, res.text


# --------------------------------------------------------------------------- #
# Trakt still computes the answer.
# --------------------------------------------------------------------------- #

def test_a_deterministic_answer_is_always_supplied(book, monkeypatch):
    """The model is never left to work it out: the governed answer is there."""
    body = _ask(book, monkeypatch)
    answer = body.get("answer") or ""
    assert len(answer) > 40, answer
    assert "Average Balance" in answer
    assert "13 groups" in answer          # the WHOLE population, in the answer


# --------------------------------------------------------------------------- #
# The population is stated honestly.
# --------------------------------------------------------------------------- #

def test_a_capped_chart_is_not_handed_over_as_complete(book, monkeypatch):
    """THE FIX. 13 brokers in, 10 rows out — say so."""
    body = _ask(book, monkeypatch)
    chart = next(s for s in body["supportingValues"] if s["kind"] == "chart")
    assert len(chart["rows"]) == 10
    assert chart["totalRows"] == 13, "the population, not the rows in hand"
    assert chart["truncated"] is True
    assert chart["populationComplete"] is False


def test_the_full_table_is_marked_complete(book, monkeypatch):
    """The table beside it carries every row, and says so — which is what makes
    the capped chart safe to send at all."""
    body = _ask(book, monkeypatch)
    table = next(s for s in body["supportingValues"] if s["kind"] == "table")
    assert len(table["rows"]) == 13
    assert table["truncated"] is False
    assert table["populationComplete"] is True


def test_a_truncated_artefact_carries_a_note_telling_the_model_not_to_recompute(
        book, monkeypatch):
    body = _ask(book, monkeypatch)
    note = body.get("truncationNote")
    notes = note if isinstance(note, str) else " ".join(note or [])
    assert "do not recompute" in notes.lower(), notes
    assert "13 rows" in notes, notes


# --------------------------------------------------------------------------- #
# Additivity travels with the rows.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("kind", ["chart", "table"])
def test_every_supporting_artefact_labels_which_columns_may_be_summed(
        book, monkeypatch, kind):
    body = _ask(book, monkeypatch)
    art = next(s for s in body["supportingValues"] if s["kind"] == kind)
    additive = art["additive"]
    assert additive, f"{kind} carries no additivity contract"
    # Money, and NOT sum-able — the pair that defeats a format-based rule.
    assert additive["current_outstanding_balance_avg"] is False
    assert additive["current_outstanding_balance_total"] is True
    assert additive["broker_channel"] is False


def test_the_agent_is_instructed_not_to_calculate():
    """The prompt-level control, made explicit.

    The old instruction forbade inventing figures from GENERAL KNOWLEDGE. It did
    not forbid arithmetic over the rows Trakt had just supplied, which is the
    only way this surface could produce a number Trakt did not.
    """
    import json
    agent = json.loads((_ROOT / "deploy" / "copilot-agent"
                        / "declarativeAgent.json").read_text(encoding="utf-8"))
    instructions = agent["instructions"].lower()
    assert "never calculate" in instructions
    assert "additive" in instructions
    assert "populationcomplete" in instructions
