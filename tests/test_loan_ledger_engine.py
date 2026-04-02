from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.gate_1_alignment.loan_ledger_engine import LoanEngineConfig, process_events


class TestLoanLedgerEngineV1(unittest.TestCase):

    def test_process_events_generates_snapshot_and_lockout_flag(self):
        terms = pd.DataFrame([
            {
                "loan_identifier": "L1",
                "origination_date": "2025-01-01",
                "maturity_date": "2026-01-01",
                "original_principal_balance": 1000,
                "current_principal_balance": 1000,
                "current_interest_rate": 0.12,
                "scheduled_interest_payment_frequency": "QUARTERLY",
                "current_valuation_amount": 2000,
                "current_valuation_date": "2025-03-31",
                "account_status": "PERFORMING",
            }
        ])

        payments = pd.DataFrame([
            {"loan_identifier": "L1", "payment_date": "2025-02-15", "payment_amount": 200, "payment_type": "PRINCIPAL"},
            {"loan_identifier": "L1", "payment_date": "2025-03-31", "payment_amount": 30, "payment_type": "INTEREST"},
        ])

        cfg = LoanEngineConfig(
            loan_engine_enabled=True,
            reporting_date=pd.Timestamp("2025-06-30"),
            ledger_db=str(Path(tempfile.gettempdir()) / "loan_engine_test.db"),
        )

        ledger, snap = process_events(terms, payments, cfg)

        self.assertFalse(ledger.empty)
        self.assertEqual(int(snap.loc[0, "current_principal_balance"]), 1000)
        self.assertGreaterEqual(float(snap.loc[0, "arrears_balance"]), 0.0)
        # Lockout breach must emit PAYMENT_WARNING (not STATUS_CHANGE).
        self.assertIn("PAYMENT_WARNING", set(ledger["event_type"]))
        self.assertTrue((ledger["source"].astype(str).str.contains("LOCKOUT_BREACH")).any())


    def test_missed_interest_triggers_status_change(self):
        """A loan with no interest payment should transition to IN_ARREARS via STATUS_CHANGE."""
        terms = pd.DataFrame([
            {
                "loan_identifier": "L2",
                "origination_date": "2025-01-01",
                "maturity_date": "2027-01-01",
                "original_principal_balance": 1000,
                "current_principal_balance": 1000,
                "current_interest_rate": 0.12,
                "scheduled_interest_payment_frequency": "QUARTERLY",
                "current_valuation_amount": 2000,
                "current_valuation_date": "2025-01-01",
                "account_status": "PERFORMING",
            }
        ])
        # No payments at all — interest will accrue and go unpaid.
        payments = pd.DataFrame(columns=["loan_identifier", "payment_date", "payment_amount", "payment_type"])

        cfg = LoanEngineConfig(
            loan_engine_enabled=True,
            reporting_date=pd.Timestamp("2025-06-30"),
            ledger_db=str(Path(tempfile.gettempdir()) / "loan_engine_test_status.db"),
        )

        ledger, snap = process_events(terms, payments, cfg)

        self.assertIn("STATUS_CHANGE", set(ledger["event_type"]))
        # After at least one missed quarterly coupon the loan should leave PERFORMING.
        self.assertNotEqual(snap.loc[0, "account_status"], "PERFORMING")
        self.assertGreater(float(snap.loc[0, "arrears_balance"]), 0.0)

    def test_payment_deduplication_removes_duplicate_rows(self):
        """Duplicate payment rows must be removed before processing."""
        terms = pd.DataFrame([
            {
                "loan_identifier": "L3",
                "origination_date": "2025-01-01",
                "maturity_date": "2027-01-01",
                "original_principal_balance": 1000,
                "current_principal_balance": 1000,
                "current_interest_rate": 0.12,
                "scheduled_interest_payment_frequency": "QUARTERLY",
                "current_valuation_amount": 2000,
                "current_valuation_date": "2025-01-01",
                "account_status": "PERFORMING",
            }
        ])
        # Same payment row duplicated — should be applied only once.
        payment_row = {"loan_identifier": "L3", "payment_date": "2025-04-01", "payment_amount": 35, "payment_type": "INTEREST"}
        payments = pd.DataFrame([payment_row, payment_row])

        cfg = LoanEngineConfig(
            loan_engine_enabled=True,
            reporting_date=pd.Timestamp("2025-06-30"),
            ledger_db=str(Path(tempfile.gettempdir()) / "loan_engine_test_dedup.db"),
        )

        ledger, snap = process_events(terms, payments, cfg)

        # Only one INTEREST_RECEIPT event should exist (not two).
        receipts = ledger[ledger["event_type"] == "INTEREST_RECEIPT"]
        self.assertEqual(len(receipts), 1)


    def test_pik_capitalises_unpaid_coupon_and_stays_performing(self):
        """With pik_enabled, an unpaid coupon capitalises into principal and loan stays PERFORMING."""
        terms = pd.DataFrame([
            {
                "loan_identifier": "L4",
                "origination_date": "2025-01-01",
                "maturity_date": "2027-01-01",
                "original_principal_balance": 1000,
                "current_principal_balance": 1000,
                "current_interest_rate": 0.12,
                "scheduled_interest_payment_frequency": "QUARTERLY",
                "current_valuation_amount": 2000,
                "current_valuation_date": "2025-01-01",
                "account_status": "PERFORMING",
            }
        ])
        # No cash interest payments — coupon should PIK at each quarter.
        payments = pd.DataFrame(columns=["loan_identifier", "payment_date", "payment_amount", "payment_type"])

        cfg = LoanEngineConfig(
            loan_engine_enabled=True,
            pik_enabled=True,
            reporting_date=pd.Timestamp("2025-06-30"),
            ledger_db=str(Path(tempfile.gettempdir()) / "loan_engine_test_pik.db"),
        )

        ledger, snap = process_events(terms, payments, cfg)

        # A PIK_CAPITALISATION event must have been emitted.
        self.assertIn("PIK_CAPITALISATION", set(ledger["event_type"]))

        # No INTEREST_SHORTFALL events — PIK replaces shortfall.
        self.assertNotIn("INTEREST_SHORTFALL", set(ledger["event_type"]))

        # Principal must have grown above original (coupon was capitalised).
        self.assertGreater(float(snap.loc[0, "current_principal_balance"]), 1000.0)

        # Cumulative PIK balance must be positive.
        self.assertGreater(float(snap.loc[0, "cumulative_pik_balance"]), 0.0)

        # Loan must remain PERFORMING / CURR — not in arrears.
        self.assertEqual(snap.loc[0, "account_status"], "CURR")
        self.assertEqual(snap.loc[0, "internal_account_status"], "PERFORMING")


if __name__ == "__main__":
    unittest.main()
