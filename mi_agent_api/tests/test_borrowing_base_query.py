"""The borrowing-base MI route: same pipeline as the dashboard, proven.

Four gates, in the order the release runs them:

* THE BANK — `migration_phase0/BORROWING_BASE_MI_BANK.yaml`, executed through
  the live `/mi/query` path against a synthetic three-period book. Route,
  verdict, intent, the measures read, the artifact shape and the governed
  refusal wording are all asserted; nothing is scored on answer rate.
* SAME-PIPELINE PARITY — for one portfolio/run, the React envelope
  (`/mi/concentration-tests` → borrowingBase), the standalone
  `/mi/borrowing-base` block and the MI answer carry IDENTICAL numbers. Zero
  mismatches is the acceptance gate.
* INELIGIBILITY REASON TRUTH — an independent oracle over the two governed
  columns and the balance column reconciles to the table the route returns.
* BRIDGE TRUTH — opening + drivers = closing, and both ends are exactly the
  ordinary pipeline's values for those runs.

Plus the non-regression falsifications: generic "headroom", "utilisation",
"eligible", "facility", "base" and "drawn" wording inside unrelated questions
is NOT claimed by this route.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.borrowing_base.models import (  # noqa: E402
    FIELD_ELIGIBILITY_REASON,
    FIELD_ELIGIBILITY_STATUS,
    INELIGIBLE,
    NOT_CALCULABLE,
)
from mi_agent_api import borrowing_base_query as bbq  # noqa: E402
from mi_agent_api.app import app  # noqa: E402

BANK = _REPO_ROOT / "migration_phase0" / "BORROWING_BASE_MI_BANK.yaml"
CLIENT = "client_001"
RUNS = (("mi_2026_04", "2026-04-30", 60, 1.0), ("mi_2026_05", "2026-05-31", 64, 1.06),
        ("mi_2026_06", "2026-06-30", 68, 1.12))
PORTFOLIO = f"{CLIENT}/mi_2026_06"
AS_OF = "2026-06-30"
BALANCE = "current_outstanding_balance"

client = TestClient(app)


# --------------------------------------------------------------------------- #
# The governed book and register, per facility mode
# --------------------------------------------------------------------------- #
def _facility(mode: str) -> Dict[str, Any]:
    drawn = mode != "nodrawn"
    return {
        "client_id": CLIENT, "facility_id": "TEST_WH_01",
        "facility_label": "Test Warehouse 01", "facility_type": "warehouse",
        "currency": "GBP", "commitment": 12_000_000, "advance_rate": 0.9,
        "concentration_denominator_floor": 5_000_000,
        "current_drawn_amount": 6_000_000 if drawn else None,
        "current_drawn_amount_as_of": AS_OF if drawn else "",
        # The CONTRACTUAL window, as recorded from the agreement. Blank in the
        # `noeffective` mode so the window is unrecorded.
        "effective_date": "" if mode == "noeffective" else "2026-01-31",
        "maturity_date": "",
        # The Trakt record's audit trail. Read for nothing historical.
        "governance": {"approval_status": "approved", "approved_at": "2026-09-01"},
        "environment": "prototype" if mode == "prototype" else "production",
        "eligibility": {"rule_version": "prototype-0", "rules": [],
                        "prototype_assume_financing_portfolio_eligible": True}
        if mode == "prototype" else {"rule_version": "r1", "rules": [
            {"rule_id": "max_current_ltv", "description": "Current LTV must not exceed 45%",
             "field": "current_loan_to_value", "operator": "max", "value": 45,
             "reason_code": "current_ltv_above_facility_limit",
             "reason": "Current LTV exceeds the facility's 45% ceiling"},
            {"rule_id": "min_youngest_age", "description": "Youngest borrower must be at least 66",
             "field": "youngest_borrower_age", "operator": "min", "value": 66,
             "reason_code": "youngest_borrower_below_minimum_age",
             "reason": "Youngest borrower is below the facility minimum age of 66"}]},
        "concentration": {"population": "eligible_mortgage_loans",
                          "borrowing_base_treatment": "monitor_only"},
    }


class Book:
    """A synthetic three-period book with the facility register for one mode."""

    def __init__(self, root: Path, mode: str, monkeypatch) -> None:
        from migration_phase0.compound_canary import _write_run
        self.mode = mode
        self.root = root / mode
        self.output_root = self.root / "onboarding_output"
        for run_id, date, n, scale in RUNS:
            _write_run(self.output_root, run_id, date, n, scale)
        register = self.root / "funding_facilities.yaml"
        if mode == "nofacility":
            register.write_text(yaml.safe_dump({"facilities": []}), encoding="utf-8")
        else:
            register.write_text(yaml.safe_dump({
                "schema_version": "1.0.0", "config_version": f"bank-{mode}",
                "facilities": [_facility(mode)]}), encoding="utf-8")
        for key, value in {
            "MI_AGENT_ONBOARDING_OUTPUT_ROOT": str(self.output_root),
            "MI_AGENT_AUTH_ENABLED": "false",
            "TRAKT_RUNTIME_MODE": "test",
            "TRAKT_STORAGE_BACKEND": "file",
            "TRAKT_LOCAL_BLOB_ROOT": str(self.root / "blob"),
            "MI_AGENT_CLIENT_ID": CLIENT,
            "TRAKT_FUNDING_FACILITIES_PATH": str(register),
            "TRAKT_CLIENT_CONFIG_DIR": str(self.root / "no_clients"),
        }.items():
            monkeypatch.setenv(key, value)

    def ask(self, question: str) -> Dict[str, Any]:
        return client.post("/mi/query", json={
            "question": question, "portfolioId": PORTFOLIO, "asOfDate": AS_OF}).json()

    def envelope(self, run_id: str = "mi_2026_06") -> Dict[str, Any]:
        return client.get("/mi/borrowing-base",
                          params={"portfolioId": f"{CLIENT}/{run_id}"}).json()

    def dashboard(self, run_id: str = "mi_2026_06") -> Dict[str, Any]:
        out = client.get("/mi/concentration-tests",
                         params={"portfolioId": f"{CLIENT}/{run_id}"}).json()
        return out["borrowingBase"]

    def frame(self, run_id: str = "mi_2026_06") -> pd.DataFrame:
        from mi_agent_api.snapshots import load_prepared_run
        tape = (self.output_root / CLIENT / run_id / "output" / "central"
                / "18_central_lender_tape.csv")
        df, _report = load_prepared_run(tape)
        return df


_BOOKS: Dict[str, Book] = {}


@pytest.fixture()
def book(request, tmp_path_factory, monkeypatch) -> Book:
    mode = getattr(request, "param", "approved")
    root = tmp_path_factory.getbasetemp() / "bb_books"
    if mode not in _BOOKS:
        _BOOKS[mode] = Book(root, mode, monkeypatch)
    else:
        Book.__init__(_BOOKS[mode], root, mode, monkeypatch)  # re-point env only
    return _BOOKS[mode]


def _bank() -> List[Dict[str, Any]]:
    return list(yaml.safe_load(BANK.read_text(encoding="utf-8"))["questions"])


def _artifact_shape(resp: Dict[str, Any]) -> List[str]:
    return [f"{a.get('type')}:{a.get('chartType') or ''}".rstrip(":")
            for a in (resp.get("artifacts") or [])]


# --------------------------------------------------------------------------- #
# 1. The bank
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", _bank(), ids=lambda c: c["id"])
def test_the_bank(case, tmp_path_factory, monkeypatch):
    mode = case.get("mode", "approved")
    root = tmp_path_factory.getbasetemp() / "bb_books"
    b = Book(root, mode, monkeypatch)
    resp = b.ask(case["question"])
    meta = resp.get("metadata") or {}
    assert meta.get("route") == case["route"], (
        f"{case['id']}: route {meta.get('route')!r}, expected {case['route']!r}. "
        f"Answer: {resp.get('answer')!r}")
    if case["route"] != "borrowing_base":
        return
    assert bool(resp.get("ok")) is bool(case["ok"]), (
        f"{case['id']}: ok={resp.get('ok')} — {resp.get('answer')!r} "
        f"warnings={resp.get('warnings')}")
    bb = meta.get("borrowingBase") or {}
    reading = bb.get("reading") or {}
    if case.get("intent"):
        assert reading.get("intent") == case["intent"], case["id"]
    if case.get("measures"):
        assert reading.get("measures") == case["measures"], (
            f"{case['id']}: read {reading.get('measures')}")
    if case["ok"]:
        # A DELIVERED answer is never a silent substitution: the receipt guard
        # let it through without rewriting it, and the route is the one that
        # answered.
        guard = resp.get("semanticGuard") or {}
        assert guard.get("verdict") in (None, "ok", "partial"), (
            f"{case['id']}: guard {guard.get('verdict')}: {guard.get('message')}")
        assert not meta.get("semanticGuardRefused")
    else:
        assert resp.get("controlledRefusal") or meta.get("controlledRefusal"), case["id"]
        assert not resp.get("artifacts"), "a refusal ships no artifact"
    if case.get("refusal_contains"):
        assert case["refusal_contains"].lower() in str(resp.get("answer")).lower(), (
            f"{case['id']}: {resp.get('answer')!r}")
    if case.get("artifact"):
        assert case["artifact"] in _artifact_shape(resp), (
            f"{case['id']}: artifacts {_artifact_shape(resp)}")
    if case.get("parity"):
        governed = b.envelope()["measures"]
        values = bb.get("values") or {}
        for measure in case["parity"]:
            assert measure in values, f"{case['id']}: {measure} not delivered"
            assert values[measure] == governed[measure], (
                f"{case['id']}: {measure} MI {values[measure]} != envelope "
                f"{governed[measure]}")
    if case.get("not_calculable"):
        for measure in case["not_calculable"]:
            assert measure in (bb.get("notCalculable") or {}), case["id"]
            assert measure not in (bb.get("values") or {}), case["id"]
            assert "not calculable" in str(resp.get("answer")).lower()
            assert re.search(r"£\s?0\b", str(resp.get("answer"))) is None, (
                "a missing input is never a zero")
    if case.get("bridge_periods"):
        bridge = bb.get("bridge") or {}
        assert bridge["opening"]["label"] == case["bridge_periods"][0]
        assert bridge["closing"]["label"] == case["bridge_periods"][1]


# --------------------------------------------------------------------------- #
# 2. Same-pipeline parity — the hard acceptance gate
# --------------------------------------------------------------------------- #
PARITY_MEASURES = ("borrowing_base", "eligible_balance", "ineligible_balance",
                   "ineligible_loan_count", "facility_drawn",
                   "borrowing_base_headroom", "facility_utilisation",
                   "borrowing_base_utilisation")


def test_dashboard_standalone_and_mi_query_carry_one_number(book: Book):
    dashboard = book.dashboard()
    standalone = book.envelope()
    assert dashboard["measures"] == standalone["measures"]
    assert dashboard["receipt"]["content_hash"] == standalone["receipt"]["content_hash"]
    mismatches = []
    for measure in PARITY_MEASURES:
        resp = book.ask(f"What is the {bbq._label(measure).lower()}?")
        values = (resp["metadata"]["borrowingBase"] or {}).get("values") or {}
        expected = dashboard["measures"][measure]
        if expected == NOT_CALCULABLE:
            if measure in values:
                mismatches.append((measure, "MI produced a number for NOT_CALCULABLE"))
            continue
        if values.get(measure) != expected:
            mismatches.append((measure, values.get(measure), expected))
    assert mismatches == []


@pytest.mark.parametrize("book", ["nodrawn"], indirect=True)
def test_parity_holds_for_the_not_calculable_state_too(book: Book):
    dashboard = book.dashboard()
    for measure in ("facility_drawn", "borrowing_base_headroom",
                    "facility_utilisation", "borrowing_base_utilisation"):
        assert dashboard["measures"][measure] == NOT_CALCULABLE
        resp = book.ask(f"What is the {bbq._label(measure).lower()}?")
        bb = resp["metadata"]["borrowingBase"]
        assert measure in bb["notCalculable"] and measure not in bb["values"]
    resp = book.ask("What is the borrowing base?")
    assert resp["metadata"]["borrowingBase"]["values"]["borrowing_base"] == \
        dashboard["availableBorrowingBase"]


# --------------------------------------------------------------------------- #
# 3. Ineligibility reason truth — an independent oracle
# --------------------------------------------------------------------------- #
def _oracle(df: pd.DataFrame) -> Dict[str, Any]:
    """Only the governed status and reason columns and the balance column."""
    ineligible = df[df[FIELD_ELIGIBILITY_STATUS] == INELIGIBLE]
    rows = ineligible.groupby(FIELD_ELIGIBILITY_REASON)[BALANCE].agg(["count", "sum"])
    fp = df[df[FIELD_ELIGIBILITY_STATUS].notna()]
    return {
        "count": int(len(ineligible)),
        "balance": round(float(ineligible[BALANCE].sum()), 2),
        "count_share": round(len(ineligible) / len(fp) * 100.0, 4),
        "balance_share": round(float(ineligible[BALANCE].sum()) / float(fp[BALANCE].sum()) * 100.0, 4),
        "rows": {str(k): (int(v["count"]), round(float(v["sum"]), 2))
                 for k, v in rows.iterrows()},
    }


def test_the_reason_table_reconciles_to_an_independent_oracle(book: Book):
    truth = _oracle(book.frame())
    assert truth["count"] > 0, "the synthetic book must carry ineligible loans"
    resp = book.ask("Show count, balance and share by ineligibility reason.")
    assert resp["ok"], resp.get("answer")
    reasons = resp["metadata"]["borrowingBase"]["reasons"]
    assert reasons["reconciles"] is True
    assert reasons["ineligibleLoanCount"] == truth["count"]
    assert abs(reasons["ineligibleBalance"] - truth["balance"]) <= 0.02
    assert reasons["ineligibleLoanSharePct"] == truth["count_share"]
    assert reasons["ineligibleBalanceSharePct"] == truth["balance_share"]
    table = {r["reasonCode"]: (r["loanCount"], r["balance"]) for r in reasons["rows"]}
    assert table == truth["rows"]
    assert sum(c for c, _ in table.values()) == truth["count"]
    assert abs(sum(b for _, b in table.values()) - truth["balance"]) <= 0.01 * (len(table) + 1)
    for row in reasons["rows"]:
        assert row["shareOfIneligibleLoansPct"] == round(
            row["loanCount"] / truth["count"] * 100.0, 4)
        assert row["shareOfIneligibleBalancePct"] == round(
            row["balance"] / truth["balance"] * 100.0, 4)
    # The envelope the dashboard renders carries the SAME totals.
    envelope = book.envelope()
    assert envelope["ineligibleLoanCount"] == truth["count"]
    assert abs(envelope["ineligibleCurrentBalance"] - truth["balance"]) <= 0.02
    # And the shipped table is the primary-reason contract.
    art = next(a for a in resp["artifacts"] if a["type"] == "table")
    assert [c["key"] for c in art["columns"]] == [
        "reason", "loans", "balance", "share_loans", "share_balance"]
    assert "primary" in art["description"]


def test_undetermined_loans_never_enter_the_ineligible_population(book: Book):
    df = book.frame()
    undetermined = int((df[FIELD_ELIGIBILITY_STATUS] == "UNDETERMINED").sum())
    resp = book.ask("How many loans are ineligible?")
    count = resp["metadata"]["borrowingBase"]["values"]["ineligible_loan_count"]
    assert count == int((df[FIELD_ELIGIBILITY_STATUS] == INELIGIBLE).sum())
    assert count + undetermined <= len(df)


# --------------------------------------------------------------------------- #
# 4. Bridge truth — both ends are the ordinary pipeline's values
# --------------------------------------------------------------------------- #
def test_the_bridge_reconciles_and_its_ends_are_the_pipelines_own_values(book: Book):
    resp = book.ask("Bridge the borrowing base from April to June.")
    assert resp["ok"], resp.get("answer")
    bridge = resp["metadata"]["borrowingBase"]["bridge"]
    opening = book.envelope("mi_2026_04")
    closing = book.envelope("mi_2026_06")
    assert bridge["opening"]["borrowingBase"] == opening["availableBorrowingBase"]
    assert bridge["closing"]["borrowingBase"] == closing["availableBorrowingBase"]
    assert bridge["opening"]["eligibleBalance"] == opening["eligibleCurrentBalance"]
    assert bridge["closing"]["eligibleBalance"] == closing["eligibleCurrentBalance"]
    drivers = {d["key"]: d["value"] for d in bridge["drivers"]}
    assert [d["key"] for d in bridge["drivers"]] == [
        "eligible_collateral_effect", "advance_rate_effect", "facility_cap_effect"]
    composed = round(bridge["opening"]["borrowingBase"] + sum(drivers.values()), 2)
    assert abs(composed - bridge["closing"]["borrowingBase"]) <= 0.05
    assert bridge["reconciliation"]["holds"]
    # Same advance rate both ends, so the rate effect is exactly zero.
    assert drivers["advance_rate_effect"] == 0.0
    # Independently: (E_c − E_o) × a_o.
    expected_collateral = round((closing["eligibleCurrentBalance"]
                                 - opening["eligibleCurrentBalance"]) * opening["advanceRate"], 2)
    assert drivers["eligible_collateral_effect"] == expected_collateral
    # No period-valid drawing exists for April, so the headroom bridge refuses.
    assert bridge["headroom"]["available"] is False
    assert "period_valid_drawn_amount" in " ".join(bridge["headroom"]["missingInputs"])
    # The waterfall is the renderer's contract.
    art = next(a for a in resp["artifacts"] if a["type"] == "chart")
    assert art["chartType"] == "waterfall" and art["xKey"] == "label"
    assert [r["type"] for r in art["rows"]] == ["total", "delta", "delta", "delta", "total"]


def test_a_trend_is_one_governed_evaluation_per_period(book: Book):
    resp = book.ask("Show the borrowing base over time.")
    assert resp["ok"], resp.get("answer")
    art = next(a for a in resp["artifacts"] if a["type"] == "chart")
    assert art["chartType"] == "line" and art["xKey"] == "period"
    by_period = {r["period"]: r["value"] for r in art["rows"]}
    for run_id, date, _n, _s in RUNS:
        assert by_period[date[:7]] == book.envelope(run_id)["availableBorrowingBase"]


# --------------------------------------------------------------------------- #
# 5. Governance — the amendments, at the route
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("book", ["prototype"], indirect=True)
def test_a_prototype_eligibility_answers_the_current_position_only(book: Book):
    now = book.ask("What is the borrowing base?")
    assert now["ok"] and now["metadata"]["borrowingBase"]["values"]["borrowing_base"] == \
        book.envelope()["availableBorrowingBase"]
    assert any("Prototype assumption" in w for w in now["warnings"])
    for q in ("How has the borrowing base changed?", "Show me the borrowing-base bridge.",
              "How has ineligible balance changed?"):
        resp = book.ask(q)
        assert resp["ok"] is False and resp.get("controlledRefusal")
        assert "not been approved" in resp["answer"] and "prototype assumption" in resp["answer"]
        assert "approved_at" not in resp["answer"]
    trend = book.ask("Show the borrowing base over time.")
    assert trend["ok"] is False and "at least two" in trend["answer"]


@pytest.mark.parametrize("book", ["noeffective"], indirect=True)
def test_approved_terms_with_no_recorded_window_answer_the_current_position_only(book: Book):
    assert book.ask("What is the borrowing base?")["ok"]
    resp = book.ask("How has the borrowing base changed?")
    assert resp["ok"] is False and "no effective date" in resp["answer"]


def test_the_contractual_window_is_the_facility_dates_not_the_approval_timestamp(
        tmp_path_factory, monkeypatch):
    # approved_at (2026-09-01) is AFTER every run; effective_date (2026-01-31)
    # is before them all. April→June bridges because the WINDOW covers it.
    root = tmp_path_factory.getbasetemp() / "bb_books"
    b = Book(root, "approved", monkeypatch)
    resp = b.ask("Bridge the borrowing base from April to June.")
    assert resp["ok"], resp.get("answer")
    assert resp["metadata"]["borrowingBase"]["opening"]["configurationApplicable"] is True
    # Move the effective date AFTER April: April is outside the window and
    # the bridge refuses by naming the date, whatever approved_at says.
    late = Book(root / "late_effective", "approved", monkeypatch)
    facility = _facility("approved")
    facility["effective_date"] = "2026-05-15"
    (late.root / "funding_facilities.yaml").write_text(
        yaml.safe_dump({"facilities": [facility]}), encoding="utf-8")
    resp = late.ask("Bridge the borrowing base from April to June.")
    assert resp["ok"] is False and "predates" in resp["answer"]
    assert "2026-05-15" in resp["answer"]
    # May→June is inside the window and still bridges.
    assert late.ask("Show me the borrowing-base bridge.")["ok"]


def test_the_current_drawing_is_valid_for_its_own_snapshot_only(book: Book):
    change = book.ask("How has headroom changed under the borrowing base?")
    assert change["ok"] is False and "period_valid_drawn_amount" in change["answer"]
    bridge = book.ask("Show me the borrowing-base bridge.")
    assert bridge["metadata"]["borrowingBase"]["closing"]["drawnBasis"] == \
        "operator_supplied_as_of_snapshot_date"
    assert bridge["metadata"]["borrowingBase"]["opening"]["drawnBasis"] == \
        "current_drawn_amount_as_of_not_this_snapshot"


@pytest.mark.parametrize("book", ["prototype"], indirect=True)
def test_prototype_assumptions_are_disclosed_exactly_as_the_envelope_states_them(book: Book):
    resp = book.ask("What is the borrowing base?")
    assert resp["ok"]
    notes = book.envelope()["prototypeAssumptionsUsed"]
    assert notes and all(any(n in w for w in resp["warnings"]) for n in notes)
    reasons = book.ask("Why are loans ineligible?")
    assert reasons["ok"] and "No loans are ineligible" in reasons["answer"]


# --------------------------------------------------------------------------- #
# 6. Non-regression — the route claims nothing generic
# --------------------------------------------------------------------------- #
NOT_CLAIMED = (
    "What is the headroom?",
    "Show risk limit headroom.",
    "Show limit utilisation by category.",
    "What is the headroom on the London concentration limit?",
    "Which concentration limits have the least headroom?",
    "What is utilisation?",
    "How many loans are eligible for the acquired book?",
    "What is the base rate exposure?",
    "Show balance by facility type.",
    "How much pipeline is excluded because of withdrawn status?",
    "What is the drawn balance by region?",
    "What is our commitment to the North?",
    "Show me the funded balance bridge by region.",
    "How has the balance changed since last month?",
    "What is the total balance of the book?",
    "Which loans are ineligible under the concentration tests?",
)


@pytest.mark.parametrize("question", NOT_CLAIMED)
def test_generic_wording_is_not_claimed(question):
    assert bbq.read(question).matched is False, question


def test_the_owner_claims_its_own_nouns():
    """Every noun the parser claims for this owner IS in the owner's vocabulary."""
    from mi_agent.llm_query_parser import _BORROWING_BASE_NOUNS
    source = Path(bbq.__file__).read_text(encoding="utf-8")
    for noun in _BORROWING_BASE_NOUNS:
        assert re.search(r"\b" + re.escape(noun[:-1] if noun.endswith("s") else noun),
                         source), noun


def test_recognition_is_registered_through_the_registry_at_its_declared_arbitration():
    from mi_agent_api.chat_routing import REGISTRY
    rec = REGISTRY.get("borrowing_base")
    assert rec is not None and rec.priority == bbq.ROUTE_PRIORITY
    assert bbq.ROUTE_CONFIDENCE > 0.8, "must outrank the analytical layer on its own vocabulary"


def test_measures_are_read_in_the_questions_own_order():
    reading = bbq.read("What are the borrowing base, headroom and facility utilisation?")
    assert reading.measures == ["borrowing_base", "borrowing_base_headroom",
                                "facility_utilisation"]
    reading = bbq.read("What share of the financing portfolio balance is ineligible?")
    assert reading.measures == ["ineligible_balance_share"]
    reading = bbq.read("What share of loans are ineligible?")
    assert reading.measures == ["ineligible_loan_share"]
    reading = bbq.read("How has the ineligible share changed?")
    assert reading.measures == ["ineligible_loan_share", "ineligible_balance_share"]
