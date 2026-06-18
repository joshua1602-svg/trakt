#!/usr/bin/env python3
"""tests/test_legacy_gate5_reconciliation.py

Review/reconciliation of the legacy Gate 5 Annex 2 XML builder against the new
XSD-based field path map.

Covers:
  * the legacy review artefact exists;
  * the comparison CSV exists and has the expected columns;
  * every comparison status is from the allowed vocabulary;
  * the multi-code-cell pollutions (RREC1/RREC2) are flagged, not adopted;
  * RREC fields are NOT flattened in the path map (still nested under Coll);
  * workbook-upgraded fields are inferred_high_confidence, never confirmed;
  * no production XML is generated;
  * production gates remain false/unchanged.
"""

from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_REVIEW = _REPO / "docs" / "legacy_gate5_annex2_xml_builder_review.md"
_CSV = _REPO / "output" / "config_review" / "legacy_gate5_vs_xsd_path_map.csv"
_PATHMAP = _REPO / "config" / "delivery" / "annex2_field_xsd_path_map.yaml"

_STATUSES = {"matches_xsd", "legacy_path_unconfirmed", "legacy_path_conflicts_with_xsd",
             "legacy_flat_but_xsd_nested", "legacy_default_injection_risk",
             "legacy_nd_injection_risk", "not_used_by_legacy_builder",
             "could_upgrade_path_map_after_review", "discard_legacy_assumption"}

_EXPECTED_COLS = ["esma_code", "canonical_field", "legacy_gate5_xml_path_or_tag",
                  "new_xsd_path_map_xml_path", "legacy_assumption", "xsd_evidence",
                  "status", "recommended_action", "risk_level", "notes"]


def _pathmap():
    return yaml.safe_load(_PATHMAP.read_text())["field_xsd_path_map"]


class TestReviewArtefact(unittest.TestCase):
    def test_review_doc_exists(self):
        self.assertTrue(_REVIEW.exists())
        text = _REVIEW.read_text(encoding="utf-8")
        # the review must call out the unsafe silent fill and the wide-row shape.
        self.assertIn("ND5", text)
        self.assertIn("singleton", text.lower())
        self.assertIn("workbook", text.lower())


class TestComparisonCsv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = list(csv.DictReader(open(_CSV, newline="", encoding="utf-8")))
        cls.by_code = {r["esma_code"]: r for r in cls.rows}

    def test_csv_exists_and_columns(self):
        self.assertTrue(_CSV.exists())
        self.assertEqual(list(self.rows[0].keys()), _EXPECTED_COLS)

    def test_all_107_codes(self):
        self.assertEqual(len(self.rows), 107)

    def test_statuses_in_vocabulary(self):
        for r in self.rows:
            self.assertIn(r["status"], _STATUSES, r["esma_code"])

    def test_pollution_flagged_not_adopted(self):
        # RREC1/RREC2 are multi-code-cell pollutions -> flagged as conflicts.
        for code in ("RREC1", "RREC2"):
            self.assertEqual(self.by_code[code]["status"], "legacy_path_conflicts_with_xsd")
            self.assertEqual(self.by_code[code]["risk_level"], "high")

    def test_value_fabrication_flagged(self):
        self.assertEqual(self.by_code["RREL12"]["status"], "legacy_default_injection_risk")

    def test_matches_present(self):
        n = sum(1 for r in self.rows if r["status"] == "matches_xsd")
        self.assertGreater(n, 50)


class TestPathMapSafety(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _pathmap()
        cls.fields = cls.m["fields"]
        cls.by_code = {f["esma_code"]: f for f in cls.fields}

    def test_rrec_never_flattened(self):
        for f in self.fields:
            if f["record_group"] == "RREC":
                self.assertEqual(f["xml_level"], "collateral", f["esma_code"])
                if f["xml_path"]:
                    self.assertIn("/Coll", f["xml_path"], f["esma_code"])

    def test_pollution_codes_not_upgraded(self):
        # RREC1/RREC2 must NOT carry the polluted header/exposure path.
        for code in ("RREC1", "RREC2"):
            f = self.by_code[code]
            self.assertEqual(f["mapping_status"], "unresolved")
            self.assertIsNone(f["xml_path"])

    def test_workbook_upgrades_are_high_not_confirmed(self):
        wb = [f for f in self.fields if f["evidence_source"] == "workbook+xsd_validated"]
        self.assertGreater(len(wb), 50)
        for f in wb:
            self.assertEqual(f["mapping_status"], "inferred_high_confidence", f["esma_code"])
            self.assertTrue(f["blocks_production_xml"], f["esma_code"])

    def test_confirmed_not_from_legacy(self):
        for f in self.fields:
            if f["mapping_status"] == "confirmed":
                self.assertIn(f["evidence_source"], ("sample_xml", "xsd"))

    def test_production_blocking_unchanged(self):
        blocking = sum(1 for f in self.fields if f["blocks_production_xml"])
        self.assertEqual(blocking, 107 - 11)  # only the 11 confirmed don't block


class TestNoProductionEffects(unittest.TestCase):
    def test_gates_remain_false(self):
        gates = _pathmap()["production_guardrails"]["production_gates_remain"]
        self.assertFalse(gates["xml_generation_allowed"])
        self.assertFalse(gates["xml_generated"])
        self.assertFalse(gates["ready_for_xml_delivery"])

    def test_no_xml_artefacts_generated(self):
        self.assertEqual(list((_REPO / "output" / "config_review").glob("*.xml")), [])
        self.assertEqual(list((_REPO / "config" / "delivery").glob("*.xml")), [])


if __name__ == "__main__":
    unittest.main()
