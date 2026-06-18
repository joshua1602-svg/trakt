#!/usr/bin/env python3
"""tests/test_annex2_path_map_promotion.py

Controlled promotion layer for the Annex 2 field-to-XSD path map.

Covers:
  * workbook_xsd_validated paths are NOT treated the same as sample-confirmed;
  * only confirmed_by_xsd_sample paths are path-production-eligible;
  * unresolved / conflict / manual_review_required remain production-blocking;
  * legacy ND/default injection is not imported (no ND5 default values in the map);
  * legacy value fabrication is not imported (no RREL12 -> "2026");
  * RREC paths remain nested under Coll;
  * path readiness and data readiness are SEPARATE (production_ready always false);
  * the promotion checklist exists for all 107 fields with the expected columns;
  * no production XML is generated; production gates remain false.
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

_PATHMAP = _REPO / "config" / "delivery" / "annex2_field_xsd_path_map.yaml"
_CHECKLIST = _REPO / "output" / "config_review" / "annex2_path_map_promotion_checklist.csv"
_POLICY = _REPO / "docs" / "annex2_path_map_promotion_policy.md"

_PROMOTION = {"confirmed_by_xsd_sample", "workbook_xsd_validated",
              "manual_review_required", "unresolved", "conflict"}

_CHECKLIST_COLS = ["esma_code", "canonical_field", "current_mapping_status",
                   "proposed_mapping_status", "xml_path", "xsd_validated",
                   "workbook_evidenced", "sample_evidenced", "manual_review_required",
                   "blocks_production_xml_before_review", "blocks_production_xml_after_review",
                   "promotion_recommendation", "risk_level", "notes"]


def _fields():
    return yaml.safe_load(_PATHMAP.read_text())["field_xsd_path_map"]["fields"]


class TestPromotionLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fields = _fields()
        cls.by_code = {f["esma_code"]: f for f in cls.fields}

    def test_every_field_has_promotion_status(self):
        self.assertEqual(len(self.fields), 107)
        for f in self.fields:
            self.assertIn(f["promotion_status"], _PROMOTION, f["esma_code"])
            for k in ("path_production_eligible", "data_readiness", "production_ready",
                      "builder_eligible_behind_structure_gate"):
                self.assertIn(k, f, f["esma_code"])

    def test_workbook_not_same_as_sample(self):
        wb = [f for f in self.fields if f["promotion_status"] == "workbook_xsd_validated"]
        sample = [f for f in self.fields if f["promotion_status"] == "confirmed_by_xsd_sample"]
        self.assertGreater(len(wb), 50)
        self.assertGreater(len(sample), 0)
        # workbook-validated are NOT path-production-eligible; sample-confirmed are.
        self.assertTrue(all(not f["path_production_eligible"] for f in wb))
        self.assertTrue(all(f["path_production_eligible"] for f in sample))
        # but workbook-validated MAY be used by the builder behind a structure gate.
        self.assertTrue(all(f["builder_eligible_behind_structure_gate"] for f in wb))
        # evidence sources are distinct.
        self.assertTrue(all(f["evidence_source"] == "workbook+xsd_validated" for f in wb))
        self.assertTrue(all(f["evidence_source"] in ("sample_xml", "xsd") for f in sample))

    def test_path_eligible_only_confirmed_sample(self):
        for f in self.fields:
            if f["path_production_eligible"]:
                self.assertEqual(f["promotion_status"], "confirmed_by_xsd_sample", f["esma_code"])

    def test_unresolved_conflict_manual_block_production(self):
        for f in self.fields:
            if f["promotion_status"] in ("unresolved", "conflict", "manual_review_required"):
                self.assertTrue(f["blocks_production_xml"], f["esma_code"])
                self.assertFalse(f["path_production_eligible"], f["esma_code"])

    def test_path_and_data_readiness_separate(self):
        # No field is production_ready, even path-eligible ones, because data
        # readiness is a separate, pending axis.
        self.assertTrue(all(f["production_ready"] is False for f in self.fields))
        self.assertTrue(all(f["data_readiness"] == "pending_delivery_certification"
                            for f in self.fields))
        # at least one field IS path-eligible yet still not production_ready.
        eligible = [f for f in self.fields if f["path_production_eligible"]]
        self.assertGreater(len(eligible), 0)
        self.assertTrue(all(not f["production_ready"] for f in eligible))

    def test_rrec_nested_under_coll(self):
        for f in self.fields:
            if f["record_group"] == "RREC":
                self.assertEqual(f["xml_level"], "collateral", f["esma_code"])
                if f["xml_path"]:
                    self.assertIn("/Coll", f["xml_path"], f["esma_code"])

    def test_pollution_is_conflict_not_promoted(self):
        for code in ("RREC1", "RREC2"):
            self.assertEqual(self.by_code[code]["promotion_status"], "conflict")
            self.assertFalse(self.by_code[code]["path_production_eligible"])

    def test_no_legacy_nd_default_injection(self):
        # the map must not contain hard ND values as data (ND lives only as a
        # value_or_nodata wrapper path, never an injected literal value).
        for f in self.fields:
            for k in ("xml_path", "nd_wrapper_path", "xsd_element"):
                v = f.get(k) or ""
                self.assertNotIn("ND5", v, f["esma_code"])
                self.assertNotIn("ND1", v, f["esma_code"])

    def test_no_legacy_value_fabrication(self):
        # RREL12 must not carry a fabricated "2026" literal anywhere.
        r = self.by_code.get("RREL12", {})
        for v in r.values():
            self.assertNotEqual(str(v), "2026", "RREL12 value fabrication imported")

    def test_value_or_nodata_has_wrapper(self):
        for f in self.fields:
            if f["value_mode"] == "value_or_nodata" and f["xml_path"]:
                self.assertTrue(f["nd_wrapper_path"], f["esma_code"])
                self.assertIn("NoDataOptn", f["nd_wrapper_path"])

    def test_promotion_summary_consistent(self):
        m = yaml.safe_load(_PATHMAP.read_text())["field_xsd_path_map"]
        ps = m["promotion_summary"]
        self.assertEqual(ps["production_ready"], 0)
        self.assertEqual(ps["path_production_eligible"],
                         sum(1 for f in self.fields if f["path_production_eligible"]))
        from collections import Counter
        actual = Counter(f["promotion_status"] for f in self.fields)
        for k, v in ps["by_promotion_status"].items():
            self.assertEqual(v, actual.get(k, 0), k)


class TestPromotionChecklist(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = list(csv.DictReader(open(_CHECKLIST, newline="", encoding="utf-8")))

    def test_checklist_exists_107_with_columns(self):
        self.assertTrue(_CHECKLIST.exists())
        self.assertEqual(len(self.rows), 107)
        self.assertEqual(list(self.rows[0].keys()), _CHECKLIST_COLS)

    def test_before_after_path_blocking(self):
        before = sum(1 for r in self.rows if r["blocks_production_xml_before_review"] == "True")
        after = sum(1 for r in self.rows if r["blocks_production_xml_after_review"] == "True")
        # before: only confirmed don't block (96). after: only manual/unresolved/conflict (7).
        self.assertEqual(before, 96)
        self.assertEqual(after, 7)
        self.assertLess(after, before)

    def test_proposed_status_vocabulary(self):
        for r in self.rows:
            self.assertIn(r["proposed_mapping_status"], _PROMOTION)

    def test_notes_separate_path_from_data(self):
        # every row must make clear production also requires data readiness.
        for r in self.rows:
            self.assertIn("DATA readiness", r["notes"])

    def test_policy_doc_exists(self):
        self.assertTrue(_POLICY.exists())
        text = _POLICY.read_text(encoding="utf-8").lower()
        self.assertIn("keep the map", text)
        self.assertIn("retire the runtime", text)


class TestNoProductionEffects(unittest.TestCase):
    def test_gates_remain_false(self):
        m = yaml.safe_load(_PATHMAP.read_text())["field_xsd_path_map"]
        gates = m["production_guardrails"]["production_gates_remain"]
        self.assertFalse(gates["xml_generation_allowed"])
        self.assertFalse(gates["xml_generated"])
        self.assertFalse(gates["ready_for_xml_delivery"])
        self.assertFalse(m["production_xsd_mapping_configured"])

    def test_no_xml_artefacts(self):
        self.assertEqual(list((_REPO / "output" / "config_review").glob("*.xml")), [])
        self.assertEqual(list((_REPO / "config" / "delivery").glob("*.xml")), [])


if __name__ == "__main__":
    unittest.main()
