"""The deterministic balance bridge — §14.

The bridge reconciles or it does not exist. Every case where loan identity cannot
be established returns an explicit limitation and NO numbers, because an
estimated bridge looks like a reconciliation and is not one.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mi_agent.period_change import models as m
from mi_agent.period_change.bridge import balance_bridge

from .test_period_change_calculations import snap


def book(rows, *, identifier="loan_identifier"):
    """``{loan_id: balance}`` → a funded frame."""
    return pd.DataFrame({identifier: list(rows.keys()),
                         "current_outstanding_balance": list(rows.values())})


def run(start, end):
    return balance_bridge(snap(start, "s0", "2026-03-31"),
                          snap(end, "s1", "2026-06-30"))


# --------------------------------------------------------------------------- #
# The reconciliation
# --------------------------------------------------------------------------- #
class TestReconciliation:
    def test_continuing_new_and_exited_loans_reconcile_exactly(self):
        start = book({"L1": 100.0, "L2": 200.0, "L3": 300.0})
        end = book({"L1": 90.0, "L2": 250.0, "L4": 400.0})
        out = run(start, end)

        assert out.status == m.BRIDGE_STATUS_AVAILABLE
        assert out.opening_balance == 600.0
        assert out.closing_balance == 740.0
        assert out.new_loan_balance == 400.0            # L4
        assert out.exited_loan_balance == 300.0         # L3 at its opening balance
        assert out.continuing_movement == 40.0          # L1 −10, L2 +50
        assert (out.new_loan_count, out.exited_loan_count,
                out.continuing_loan_count) == (1, 1, 2)
        assert out.reconciles is True
        assert out.residual == pytest.approx(0.0)

    def test_the_bridge_identity_holds(self):
        start = book({"A": 1_000.0, "B": 2_500.0, "C": 750.0})
        end = book({"A": 900.0, "C": 800.0, "D": 5_000.0, "E": 12.5})
        out = run(start, end)
        reconstructed = (out.opening_balance + out.new_loan_balance
                         - out.exited_loan_balance + out.continuing_movement)
        assert reconstructed == pytest.approx(out.closing_balance)

    def test_a_book_with_no_churn_shows_only_continuing_movement(self):
        out = run(book({"L1": 100.0}), book({"L1": 120.0}))
        assert (out.new_loan_count, out.exited_loan_count) == (0, 0)
        assert out.continuing_movement == 20.0
        assert out.reconciles is True

    def test_full_redemption_of_the_book(self):
        out = run(book({"L1": 100.0, "L2": 50.0}), book({"L3": 10.0}))
        assert out.exited_loan_balance == 150.0
        assert out.exited_loan_count == 2
        assert out.continuing_loan_count == 0
        assert out.continuing_movement == 0.0
        assert out.reconciles is True

    def test_float_accumulation_stays_inside_the_documented_tolerance(self):
        start = book({f"L{i}": 0.1 for i in range(1000)})
        end = book({f"L{i}": 0.2 for i in range(1000)})
        out = run(start, end)
        assert out.reconciles is True
        assert abs(out.residual) <= m.BRIDGE_ROUNDING_TOLERANCE

    def test_missing_balances_are_treated_as_zero_consistently_on_both_sides(self):
        start = pd.DataFrame({"loan_identifier": ["L1", "L2"],
                              "current_outstanding_balance": [100.0, None]})
        end = pd.DataFrame({"loan_identifier": ["L1", "L2"],
                            "current_outstanding_balance": [110.0, 50.0]})
        out = run(start, end)
        assert out.opening_balance == 100.0
        assert out.closing_balance == 160.0
        assert out.reconciles is True


# --------------------------------------------------------------------------- #
# When the bridge cannot be built — an explicit limitation, never an estimate
# --------------------------------------------------------------------------- #
class TestUnavailable:
    def test_no_loan_identifier_means_no_bridge(self):
        frame = pd.DataFrame({"current_outstanding_balance": [100.0]})
        out = run(frame, frame)
        assert out.status == m.BRIDGE_STATUS_UNAVAILABLE_NO_IDENTIFIER
        assert out.opening_balance is None
        assert "none is estimated" in out.limitation

    def test_no_balance_field_means_no_bridge(self):
        frame = pd.DataFrame({"loan_identifier": ["L1"]})
        out = run(frame, frame)
        assert out.status == m.BRIDGE_STATUS_UNAVAILABLE_NO_BALANCE_FIELD
        assert out.limitation

    def test_duplicate_identifiers_mean_no_bridge(self):
        start = pd.DataFrame({"loan_identifier": ["L1", "L1"],
                              "current_outstanding_balance": [100.0, 100.0]})
        end = book({"L1": 200.0})
        out = run(start, end)
        assert out.status == m.BRIDGE_STATUS_UNAVAILABLE_DUPLICATE_IDENTIFIERS
        assert out.opening_balance is None
        assert "Loan identity cannot be established" in out.limitation

    def test_missing_identifiers_mean_no_bridge(self):
        start = pd.DataFrame({"loan_identifier": ["L1", None],
                              "current_outstanding_balance": [100.0, 50.0]})
        end = book({"L1": 110.0})
        out = run(start, end)
        assert out.status == m.BRIDGE_STATUS_UNAVAILABLE_MISSING_IDENTIFIERS
        assert out.opening_balance is None

    def test_an_alternative_canonical_identifier_is_accepted(self):
        start = book({"L1": 100.0}, identifier="original_loan_identifier")
        end = book({"L1": 150.0}, identifier="original_loan_identifier")
        out = run(start, end)
        assert out.status == m.BRIDGE_STATUS_AVAILABLE
        assert out.identifier_field == "original_loan_identifier"

    def test_the_canonical_identifier_is_preferred_over_an_alternative(self):
        start = pd.DataFrame({"loan_identifier": ["A"],
                              "original_loan_identifier": ["X"],
                              "current_outstanding_balance": [100.0]})
        end = pd.DataFrame({"loan_identifier": ["A"],
                            "original_loan_identifier": ["Y"],
                            "current_outstanding_balance": [110.0]})
        out = run(start, end)
        assert out.identifier_field == "loan_identifier"
        assert out.continuing_loan_count == 1


# --------------------------------------------------------------------------- #
# Discipline
# --------------------------------------------------------------------------- #
def test_the_bridge_is_not_a_cashflow_waterfall():
    """§14: four identity-based movements, and nothing inferred from them."""
    out = run(book({"L1": 100.0}), book({"L1": 120.0}))
    data = out.to_dict()
    assert set(data) == {
        "status", "balance_field", "identifier_field", "identifier_fields",
        "opening_balance", "new_loan_closing_balance",
        "exited_loan_opening_balance", "movement_on_continuing_loans",
        "closing_balance", "new_loan_count", "exited_loan_count",
        "continuing_loan_count", "reconciles", "residual",
        "rounding_tolerance", "limitation", "evidence"}


def test_the_evidence_names_snapshots_not_loans():
    out = run(book({"L1": 100.0}), book({"L1": 120.0}))
    assert [e["snapshot_id"] for e in out.evidence] == ["s0", "s1"]
    assert "L1" not in str(out.evidence)


# --------------------------------------------------------------------------- #
# Loan identity across originators — §14
# --------------------------------------------------------------------------- #
class TestCompositeIdentity:
    def _book(self, rows):
        """``[(portfolio, loan_id, balance)]`` → a consolidated funded frame."""
        return pd.DataFrame({
            "source_portfolio_id": [r[0] for r in rows],
            "loan_identifier": [r[1] for r in rows],
            "current_outstanding_balance": [r[2] for r in rows],
        })

    def test_provenance_makes_the_key_composite(self):
        start = self._book([("alpha", "0001", 100.0)])
        end = self._book([("alpha", "0001", 120.0)])
        out = run(start, end)
        assert out.identifier_fields == ("source_portfolio_id", "loan_identifier")
        assert out.identifier_field == "source_portfolio_id + loan_identifier"
        assert out.continuing_loan_count == 1

    def test_an_identifier_reused_by_two_originators_is_not_one_loan(self):
        """Originator alpha's 0001 exits and beta's 0001 arrives. On a bare
        identifier these read as one continuing loan whose balance moved; on the
        composite key they are correctly an exit and an arrival."""
        start = self._book([("alpha", "0001", 100.0)])
        end = self._book([("beta", "0001", 900.0)])
        out = run(start, end)
        assert out.status == m.BRIDGE_STATUS_AVAILABLE
        assert out.exited_loan_count == 1
        assert out.new_loan_count == 1
        assert out.continuing_loan_count == 0
        assert out.continuing_movement == 0.0
        assert out.exited_loan_balance == 100.0
        assert out.new_loan_balance == 900.0
        assert out.reconciles is True

    def test_the_same_identifier_within_one_originator_is_the_same_loan(self):
        start = self._book([("alpha", "0001", 100.0), ("beta", "0001", 200.0)])
        end = self._book([("alpha", "0001", 150.0), ("beta", "0001", 250.0)])
        out = run(start, end)
        assert out.continuing_loan_count == 2
        assert out.continuing_movement == 100.0
        assert out.new_loan_count == 0 and out.exited_loan_count == 0

    def test_a_collision_within_one_originator_still_fails_closed(self):
        start = self._book([("alpha", "0001", 100.0), ("alpha", "0001", 100.0)])
        end = self._book([("alpha", "0001", 250.0)])
        out = run(start, end)
        assert out.status == m.BRIDGE_STATUS_UNAVAILABLE_DUPLICATE_IDENTIFIERS

    def test_a_blank_key_component_is_a_missing_identifier(self):
        start = self._book([("alpha", "0001", 100.0), ("", "0002", 50.0)])
        end = self._book([("alpha", "0001", 120.0)])
        out = run(start, end)
        assert out.status == m.BRIDGE_STATUS_UNAVAILABLE_MISSING_IDENTIFIERS

    def test_without_provenance_the_key_stays_bare(self):
        out = run(book({"L1": 100.0}), book({"L1": 120.0}))
        assert out.identifier_fields == ("loan_identifier",)


def test_a_mixed_currency_book_gets_no_bridge():
    """A bridge that added GBP to EUR would reconcile arithmetically while
    meaning nothing — the worst possible failure mode for a reconciliation."""
    start = pd.DataFrame({
        "loan_identifier": ["L1", "L2"],
        "current_outstanding_balance": [100.0, 100.0],
        "exposure_currency_denomination": ["GBP", "EUR"]})
    end = pd.DataFrame({
        "loan_identifier": ["L1", "L2"],
        "current_outstanding_balance": [120.0, 130.0],
        "exposure_currency_denomination": ["GBP", "EUR"]})
    out = run(start, end)
    assert out.status == m.BRIDGE_STATUS_UNAVAILABLE_MIXED_CURRENCY
    assert out.opening_balance is None and out.closing_balance is None
    assert out.reconciles is None
    assert "currency" in out.limitation.lower()
