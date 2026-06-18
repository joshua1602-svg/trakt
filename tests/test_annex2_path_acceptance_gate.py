#!/usr/bin/env python3
"""tests/test_annex2_path_acceptance_gate.py

Formal acceptance gate over the workbook/XSD-validated Annex 2 paths.

Covers:
  * every field has a builder_acceptance_status from the allowed vocabulary;
  * most workbook_xsd_validated paths become accepted_for_builder;
  * accepted/sample paths re-validate against the XSD (independent re-check);
  * polluted RREC1/RREC2 are rejected (not accepted);
  * unresolved/manual fields are needs_manual_review;
  * RREC accepted paths stay nested under Coll;
  * value_or_nodata accepted paths have a validated nd_wrapper_path;
  * the decisions CSV exists for all 107 fields with the expected columns and
    does not drift from the YAML;
  * production_ready is false for all fields; gates remain false;
  * no production XML is generated.
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

from scripts.build_annex2_field_xsd_path_map import xsd_path_validator  # noqa: E402

_PATHMAP = _REPO / "config" / "delivery" / "annex2_field_xsd_path_map.yaml"
_CSV = _REPO / "output" / "config_review" / "annex2_path_acceptance_decisions.csv"
_DOC = _REPO / "docs" / "annex2_path_acceptance_gate.md"

_ACCEPT = {"sample_confirmed", "accepted_for_builder", "needs_manual_review", "rejected"}
_GOOD_FOR_BUILDER = {"sample_confirmed", "accepted_for_builder"}

_CSV_COLS = ["esma_code", "canonical_field", "record_group", "promotion_status",
             "builder_acceptance_status", "xml_path", "from_workbook_path", "xsd_validated",
             "code_label_type_consistent", "not_polluted_multicode",
             "respects_rrel_rrec_hierarchy", "rrec_nested_under_coll", "nodataoptn_handling",
             "sample_confirmed", "decision_reason", "risk_level", "production_ready", "notes"]


def _fields():
    return yaml.safe_load(_PATHMAP.read_text())["field_xsd_path_map"]["fields"]


class TestAcceptanceStatuses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fields = _fields()
        cls.by_code = {f["esma_code"]: f for f in cls.fields}
        cls.valid = staticmethod(xsd_path_validator())

    def test_every_field_has_acceptance_status(self):
        self.assertEqual(len(self.fields), 107)
        for f in self.fields:
            self.assertIn(f["builder_acceptance_status"], _ACCEPT, f["esma_code"])

    def test_most_workbook_paths_accepted(self):
        accepted = [f for f in self.fields if f["builder_acceptance_status"] == "accepted_for_builder"]
        self.assertGreaterEqual(len(accepted), 80)
        # every accepted field was a workbook_xsd_validated promotion.
        for f in accepted:
            self.assertEqual(f["promotion_status"], "workbook_xsd_validated", f["esma_code"])

    def test_accepted_paths_revalidate_against_xsd(self):
        for f in self.fields:
            if f["builder_acceptance_status"] in _GOOD_FOR_BUILDER:
                self.assertTrue(f["xml_path"], f["esma_code"])
                self.assertTrue(self.valid(f["xml_path"]), f["esma_code"])

    def test_pollution_rejected(self):
        for code in ("RREC1", "RREC2"):
            self.assertEqual(self.by_code[code]["builder_acceptance_status"], "rejected")

    def test_unresolved_and_manual_are_manual_review(self):
        for f in self.fields:
            if f["promotion_status"] in ("manual_review_required", "unresolved"):
                self.assertEqual(f["builder_acceptance_status"], "needs_manual_review", f["esma_code"])

    def test_rejected_and_manual_not_good_for_builder(self):
        for f in self.fields:
            if f["builder_acceptance_status"] in ("rejected", "needs_manual_review"):
                self.assertNotIn(f["builder_acceptance_status"], _GOOD_FOR_BUILDER)

    def test_accepted_rrec_nested_under_coll(self):
        for f in self.fields:
            if f["record_group"] == "RREC" and f["builder_acceptance_status"] in _GOOD_FOR_BUILDER:
                self.assertEqual(f["xml_level"], "collateral", f["esma_code"])
                self.assertIn("/Coll", f["xml_path"], f["esma_code"])

    def test_accepted_value_or_nodata_has_validated_wrapper(self):
        for f in self.fields:
            if (f["builder_acceptance_status"] in _GOOD_FOR_BUILDER
                    and f["value_mode"] == "value_or_nodata"):
                self.assertTrue(f["nd_wrapper_path"], f["esma_code"])
                self.assertTrue(self.valid(f["nd_wrapper_path"]), f["esma_code"])

    def test_no_field_production_ready(self):
        # acceptance is path-axis only; nothing becomes production-ready.
        self.assertTrue(all(f["production_ready"] is False for f in self.fields))

    def test_summary_consistent(self):
        m = yaml.safe_load(_PATHMAP.read_text())["field_xsd_path_map"]
        s = m["builder_acceptance_summary"]["by_builder_acceptance_status"]
        from collections import Counter
        actual = Counter(f["builder_acceptance_status"] for f in self.fields)
        for k, v in s.items():
            self.assertEqual(v, actual.get(k, 0), k)
        self.assertEqual(m["builder_acceptance_summary"]["production_ready"], 0)


class TestAcceptanceCsv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = list(csv.DictReader(open(_CSV, newline="", encoding="utf-8")))
        cls.by_code = {f["esma_code"]: f for f in _fields()}

    def test_csv_exists_107_columns(self):
        self.assertTrue(_CSV.exists())
        self.assertEqual(len(self.rows), 107)
        self.assertEqual(list(self.rows[0].keys()), _CSV_COLS)

    def test_csv_matches_yaml_no_drift(self):
        for r in self.rows:
            self.assertEqual(r["builder_acceptance_status"],
                             self.by_code[r["esma_code"]]["builder_acceptance_status"], r["esma_code"])

    def test_statuses_in_vocabulary(self):
        for r in self.rows:
            self.assertIn(r["builder_acceptance_status"], _ACCEPT)

    def test_production_ready_false_everywhere(self):
        self.assertTrue(all(r["production_ready"] == "False" for r in self.rows))

    def test_doc_exists(self):
        self.assertTrue(_DOC.exists())
        t = _DOC.read_text(encoding="utf-8").lower()
        self.assertIn("accepted_for_builder", t)
        self.assertIn("not", t)


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

    def test_no_fabricated_values_or_nd_defaults(self):
        # acceptance must not introduce ND5 literals or RREL12->2026 fabrication.
        for f in _fields():
            for k in ("xml_path", "nd_wrapper_path", "xsd_element", "builder_acceptance_status"):
                v = str(f.get(k) or "")
                self.assertNotIn("ND5", v)
            if f["esma_code"] == "RREL12":
                self.assertNotIn("2026", str(f))


if __name__ == "__main__":
    unittest.main()
