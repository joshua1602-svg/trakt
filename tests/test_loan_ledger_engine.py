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
        self.assertIn("STATUS_CHANGE", set(ledger["event_type"]))
        self.assertTrue((ledger["source"].astype(str).str.contains("LOCKOUT_BREACH")).any())


if __name__ == "__main__":
    unittest.main()
