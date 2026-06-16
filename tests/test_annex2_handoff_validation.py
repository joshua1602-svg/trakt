#!/usr/bin/env python3
"""Tests for the onboarding -> Annex 2 XML handoff validation (artefact 50).

Verifies the handoff contract is diagnosed correctly: asset/regime defaults
resolve the "missing core" fields, canonical alias mismatches are surfaced (not
auto-applied), pending regime rules are reported as a config backlog (not as
"core fields missing"), and the tape is only XML-ready when nothing blocks.
"""
from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from engine.onboarding_agent import annex2_handoff_validation as hv

# Columns observed in the real-client promoted central tape (MI canonical vocab).
_TAPE_COLS = [
    "loan_identifier", "account_status", "accrued_interest_in_period",
    "borrower_1_DOB", "borrower_1_date_of_death", "borrower_1_gender",
    "collateral_geography", "cumulative_accrued_interest", "current_interest_rate",
    "current_outstanding_balance", "current_valuation_amount", "data_cut_off_date",
    "erm_product_type", "loan_sub_type", "original_principal_balance",
    "original_valuation_amount", "pool_identifier", "postcode", "property_type",
    "purpose",
]


class TestAnnex2HandoffValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="annex2_handoff_"))
        tape = cls.tmp / "18_central_lender_tape.csv"
        with tape.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(_TAPE_COLS)
            w.writerow(["L1"] + ["x"] * (len(_TAPE_COLS) - 1))
        cls.rows = hv.build_annex2_xml_handoff_validation(central_tape_path=tape)
        cls.by = {r["esma_code"]: r for r in cls.rows}
        cls.summary = hv.summarise(cls.rows)

    def test_static_fields_resolved_by_asset_default_not_blocking(self):
        # The "missing core fields" the gate-1 error complained about are covered
        # by asset defaults (interest_rate_type=Fixed, amortisation_type=Bullet,
        # exposure_currency_denomination=GBP) — so NOT blocking.
        for canon in ("interest_rate_type", "amortisation_type",
                      "exposure_currency_denomination"):
            r = next(x for x in self.rows if x["annex2_required_field"] == canon)
            self.assertEqual(r["delivery_value_status"], "resolved_by_asset_default")
            self.assertFalse(r["blocking"], canon)

    def test_alias_mismatch_surfaced_not_autoapplied(self):
        # current_principal_balance (RREL30) is NOT in the tape; the related MI
        # column current_outstanding_balance is surfaced as an alias decision.
        r = next(x for x in self.rows
                 if x["annex2_required_field"] == "current_principal_balance")
        self.assertEqual(r["source_resolution"], "canonical_alias_mismatch")
        self.assertEqual(r["matched_promoted_field"], "current_outstanding_balance")
        self.assertIn("map/derive", r["recommended_fix"])

    def test_direct_tape_fields_resolved(self):
        # property_type / purpose are present in the tape under the same canonical.
        for canon in ("property_type", "purpose"):
            r = next(x for x in self.rows if x["annex2_required_field"] == canon)
            self.assertEqual(r["delivery_value_status"], "resolved_from_tape")

    def test_pending_regime_rules_surfaced_as_backlog(self):
        # Codes with no regime rule are reported as pending, NOT as missing tape
        # fields. There should be a non-trivial backlog.
        self.assertGreater(self.summary["pending_regime_rule"], 0)
        pend = [r for r in self.rows
                if r["source_resolution"] == "pending_regime_rule"]
        self.assertTrue(all(r["delivery_value_status"] == "pending_regime_rule"
                            for r in pend))

    def test_not_xml_ready_until_handoff_passes(self):
        # The promoted tape is NOT automatically XML-ready: alias/pending items remain.
        self.assertFalse(self.summary["xml_ready"])
        self.assertEqual(set(self.summary["source_resolution_counts"]) & {"promoted_tape_direct"},
                         {"promoted_tape_direct"})

    def test_artefacts_written(self):
        paths = hv.write(self.tmp, self.rows, self.summary,
                         central_tape_path="t.csv")
        for k in ("csv", "json", "summary_md"):
            self.assertTrue(Path(paths[k]).exists())


if __name__ == "__main__":
    unittest.main()
