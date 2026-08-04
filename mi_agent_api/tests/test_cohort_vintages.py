"""Genuine cohort/vintage analytics, distinct from portfolio evolution.

The defect these pin: the Cohorts area showed the funded book at each
reporting date — 33 loans, then 73, then 73 — which is portfolio evolution
under a cohort label. ``funded_cohort_progression`` only becomes a static pool
when a vintage is passed; with none selected it returns the whole book per
period, and there was no surface at all on which November reads 40 rather
than 73.

The fixture is the shape described in the brief: 33 October loans, 40 November
loans, 10 December loans, with genuine redemptions out of the October cohort.
So the two views must diverge and stay diverged:

    portfolio evolution   33 -> 73 -> 83      (the book outstanding)
    vintage formation     33,   40,   10      (what entered, once each)

Cohort membership comes from ``origination_date``, the funded book's policy
completion date. Loans are followed between periods by loan identifier — both
are governed fields, and without either the surface reports unavailable rather
than inventing a pool.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mi_agent_api.cohorts import (
    cohort_analysis,
    cohort_entry_map,
    cohort_formation,
    cohort_static_pool,
    loan_id_column,
)

OCT_N, NOV_N, DEC_N = 33, 40, 10
#: October loans that redeem — two in November, three more by December.
REDEEMED_NOV = ["OCT-000", "OCT-001"]
REDEEMED_DEC = REDEEMED_NOV + ["OCT-002", "OCT-003", "OCT-004"]


def _loans(prefix, n, vintage, balance, ltv=0.40, rate=6.0, source="direct"):
    return [{
        "loan_policy_number": f"{prefix}-{i:03d}",
        "origination_date": vintage,
        "current_outstanding_balance": balance,
        "current_loan_to_value": ltv,
        "original_loan_to_value": ltv,
        "current_interest_rate": rate,
        "source_portfolio_type": source,
    } for i in range(n)]


def _frame(rows, reporting_date):
    return {"run_id": reporting_date[:7].replace("-", "_"),
            "reporting_date": reporting_date,
            "df": pd.DataFrame(rows)}


@pytest.fixture()
def frames():
    """Three reporting periods over three origination vintages."""
    oct_loans = _loans("OCT", OCT_N, "2025-10-15", 100_000)
    nov_loans = _loans("NOV", NOV_N, "2025-11-15", 120_000)
    dec_loans = _loans("DEC", DEC_N, "2025-12-15", 90_000)

    p1 = _frame(oct_loans, "2025-10-31")
    # October loans amortise slightly; two redeem out entirely.
    oct_in_nov = [{**r, "current_outstanding_balance": 98_000}
                  for r in oct_loans if r["loan_policy_number"] not in REDEEMED_NOV]
    p2 = _frame(oct_in_nov + nov_loans, "2025-11-30")
    oct_in_dec = [{**r, "current_outstanding_balance": 96_000}
                  for r in oct_loans if r["loan_policy_number"] not in REDEEMED_DEC]
    nov_in_dec = [{**r, "current_outstanding_balance": 118_000} for r in nov_loans]
    p3 = _frame(oct_in_dec + nov_in_dec + dec_loans, "2025-12-31")
    return [p1, p2, p3]


class TestTheTwoViewsDiverge:
    """The whole point: evolution counts the book, formation counts arrivals."""

    def test_portfolio_evolution_is_the_book_outstanding(self, frames):
        assert [len(fr["df"]) for fr in frames] == [33, 71, 78]

    def test_vintage_formation_counts_what_entered(self, frames):
        result = cohort_formation(frames, grain="M")
        counts = {v["vintage"]: v["originalLoanCount"] for v in result["vintages"]}
        assert counts == {"2025-10": OCT_N, "2025-11": NOV_N, "2025-12": DEC_N}

    def test_november_is_forty_not_seventy_three(self, frames):
        """The reported defect, stated as an assertion."""
        result = cohort_formation(frames, grain="M")
        november = next(v for v in result["vintages"] if v["vintage"] == "2025-11")
        assert november["originalLoanCount"] == 40
        assert november["originalLoanCount"] != len(frames[1]["df"])


class TestCohortMembership:
    def test_no_loan_belongs_to_two_cohorts(self, frames):
        entry, _ = cohort_entry_map(frames, "M")
        assert len(entry) == OCT_N + NOV_N + DEC_N
        assert len(set(entry.values())) == 3

    def test_the_november_cohort_excludes_october_loans(self, frames):
        entry, _ = cohort_entry_map(frames, "M")
        november = {k for k, v in entry.items() if v == "2025-11"}
        assert all(k.startswith("NOV-") for k in november)
        assert len(november) == NOV_N

    def test_a_loan_keeps_its_vintage_when_a_later_snapshot_restates_it(self):
        """Late corrections are deterministic: first assignment stands, and the
        restatement is reported. Re-basing would let a static pool grow."""
        a = _frame(_loans("L", 2, "2025-10-15", 100_000), "2025-10-31")
        restated = _loans("L", 2, "2025-11-15", 100_000)
        b = _frame(restated, "2025-11-30")
        entry, corrections = cohort_entry_map([a, b], "M")
        assert set(entry.values()) == {"2025-10"}
        assert [c["restatedTo"] for c in corrections] == ["2025-11", "2025-11"]

    def test_formation_reports_late_corrections(self, frames):
        assert cohort_formation(frames, grain="M")["lateCorrections"] == []


class TestStaticPool:
    def test_the_october_pool_starts_at_its_formation_size(self, frames):
        pool = cohort_static_pool(frames, vintage="2025-10", grain="M")
        assert pool["originalLoanCount"] == OCT_N
        assert pool["periods"][0]["survivingLoanCount"] == OCT_N

    def test_the_pool_declines_only_through_genuine_exits(self, frames):
        pool = cohort_static_pool(frames, vintage="2025-10", grain="M")
        counts = [p["survivingLoanCount"] for p in pool["periods"]]
        assert counts == [33, 31, 28]
        assert counts == sorted(counts, reverse=True), "a static pool never grows"

    def test_exits_are_counted_per_period_and_cumulatively(self, frames):
        pool = cohort_static_pool(frames, vintage="2025-10", grain="M")
        assert [p["exitsInPeriod"] for p in pool["periods"]] == [0, 2, 3]
        assert [p["cumulativeExits"] for p in pool["periods"]] == [0, 2, 5]

    def test_seasoning_is_measured_from_the_vintage(self, frames):
        pool = cohort_static_pool(frames, vintage="2025-10", grain="M")
        assert [p["monthsSinceEntry"] for p in pool["periods"]] == [0, 1, 2]

    def test_retention_is_reported_on_loans_and_balance(self, frames):
        pool = cohort_static_pool(frames, vintage="2025-10", grain="M")
        last = pool["periods"][-1]
        assert last["loanRetention"] == pytest.approx(28 / 33, abs=1e-4)
        # 28 loans at 96,000 against 33 at 100,000 at formation.
        assert last["balanceRetention"] == pytest.approx(
            (28 * 96_000) / (33 * 100_000), abs=1e-4)

    def test_a_later_vintage_starts_when_it_forms_not_before(self, frames):
        pool = cohort_static_pool(frames, vintage="2025-12", grain="M")
        assert [p["period"] for p in pool["periods"]] == ["2025-12"]
        assert pool["originalLoanCount"] == DEC_N

    def test_an_unknown_vintage_is_refused_not_invented(self, frames):
        pool = cohort_static_pool(frames, vintage="2024-01", grain="M")
        assert pool["available"] is False
        assert "originated" in pool["reason"]


class TestScopes:
    def test_scope_is_explicit_across_direct_and_acquired(self):
        direct = _loans("DIR", 5, "2025-10-15", 100_000, source="direct")
        acquired = _loans("ACQ", 7, "2025-10-15", 100_000, source="acquired")
        frames = [_frame(direct + acquired, "2025-10-31")]
        total = cohort_formation(frames, grain="M")["vintages"][0]
        assert total["originalLoanCount"] == 12

        only_direct = [{**frames[0], "df": frames[0]["df"][
            frames[0]["df"]["source_portfolio_type"] == "direct"]}]
        assert cohort_formation(only_direct, grain="M")[
            "vintages"][0]["originalLoanCount"] == 5


class TestCompositionReconcilesToTheCohort:
    def test_composition_of_one_cohort_is_not_the_whole_book(self, frames):
        latest = frames[-1]["df"]
        whole = cohort_analysis(latest, client_id="c", dimension="vintage")
        assert whole["totalLoanCount"] == 78

        entry, _ = cohort_entry_map(frames, "M")
        oct_ids = {k for k, v in entry.items() if v == "2025-10"}
        selected = latest[latest["loan_policy_number"].isin(oct_ids)]
        just_october = cohort_analysis(selected, client_id="c", dimension="ltv")
        assert just_october["totalLoanCount"] == 28
        assert just_october["totalLoanCount"] < whole["totalLoanCount"]


class TestUnitContract:
    def test_cohort_ltv_follows_the_governed_fraction_contract(self, frames):
        pool = cohort_static_pool(frames, vintage="2025-10", grain="M")
        ltv = pool["periods"][0]["waLtv"]
        assert ltv == pytest.approx(0.40), "emitted as a fraction, not 40.0"
        assert round(ltv * 100, 1) == 40.0, "renders as 40.0%"

    def test_formation_ltv_uses_the_same_contract(self, frames):
        october = cohort_formation(frames, grain="M")["vintages"][0]
        assert october["waOriginalLtv"] == pytest.approx(0.40)


class TestStopConditions:
    """When the governed fields are absent the surface says so."""

    def test_no_loan_identifier_means_no_static_pool(self):
        rows = [{"origination_date": "2025-10-15",
                 "current_outstanding_balance": 100_000}]
        frames = [_frame(rows, "2025-10-31")]
        assert loan_id_column(frames[0]["df"]) is None
        result = cohort_formation(frames, grain="M")
        assert result["available"] is False
        assert "loan identifier" in result["reason"]

    def test_no_origination_date_means_no_vintage(self):
        rows = [{"loan_policy_number": "L-1",
                 "current_outstanding_balance": 100_000}]
        frames = [_frame(rows, "2025-10-31")]
        result = cohort_formation(frames, grain="M")
        assert result["available"] is False
        assert "origination_date" in result["reason"]
