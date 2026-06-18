#!/usr/bin/env python3
"""tests/test_annex2_field_xsd_path_map.py

Annex 2 field-to-XSD path mapping layer.

Covers:
  * the YAML path map has all 107 Annex 2 fields;
  * every field has esma_code, record_group, xml_level, mapping_status;
  * confirmed mappings carry an xml_path AND documented sample/XSD evidence;
  * unresolved mappings block production XML;
  * RREC fields are NOT treated as flat sibling records (collateral is nested
    under the loan);
  * confirmed collateral mappings are nested under .../PrfrmgLn/Coll;
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

_YAML = _REPO / "config" / "delivery" / "annex2_field_xsd_path_map.yaml"
_CSV = _REPO / "output" / "config_review" / "annex2_field_xsd_path_map.csv"
_UNIVERSE = _REPO / "config" / "regime" / "annex2_field_universe.yaml"

_STATUSES = {"confirmed", "inferred_high_confidence", "inferred_low_confidence",
             "unresolved", "conflict"}
_LEVELS = {"header", "exposure", "collateral", "asset_class_branch", "unknown"}


def _load_map():
    return yaml.safe_load(_YAML.read_text(encoding="utf-8"))["field_xsd_path_map"]


class TestPathMapStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load_map()
        cls.fields = cls.m["fields"]
        cls.by_code = {f["esma_code"]: f for f in cls.fields}

    def test_yaml_exists_and_has_107_fields(self):
        self.assertTrue(_YAML.exists())
        self.assertEqual(len(self.fields), 107)

    def test_matches_universe_codes(self):
        universe = yaml.safe_load(_UNIVERSE.read_text())["fields"]
        self.assertEqual(set(self.by_code), set(universe))

    def test_required_keys_present(self):
        required = {"esma_code", "canonical_field", "record_group", "xml_level",
                    "xml_path", "xsd_element", "xsd_type", "cardinality",
                    "value_mode", "nd_wrapper_path", "mapping_status",
                    "evidence_source", "evidence_note", "blocks_production_xml", "owner"}
        for f in self.fields:
            self.assertTrue(required.issubset(f), f"{f['esma_code']} missing {required - set(f)}")
            self.assertTrue(f["esma_code"])
            self.assertIn(f["record_group"], ("RREL", "RREC"))
            self.assertIn(f["xml_level"], _LEVELS)
            self.assertIn(f["mapping_status"], _STATUSES)

    def test_csv_mirrors_yaml(self):
        self.assertTrue(_CSV.exists())
        with open(_CSV, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 107)
        self.assertEqual({r["esma_code"] for r in rows}, set(self.by_code))

    def test_confirmed_have_path_and_evidence(self):
        confirmed = [f for f in self.fields if f["mapping_status"] == "confirmed"]
        self.assertGreater(len(confirmed), 0)
        for f in confirmed:
            self.assertTrue(f["xml_path"], f"{f['esma_code']} confirmed without xml_path")
            self.assertIn(f["evidence_source"], ("sample_xml", "xsd"),
                          f"{f['esma_code']} confirmed without sample/XSD evidence")
            self.assertFalse(f["blocks_production_xml"])

    def test_unresolved_block_production(self):
        for f in self.fields:
            if f["mapping_status"] == "unresolved":
                self.assertTrue(f["blocks_production_xml"], f["esma_code"])
                self.assertIsNone(f["xml_path"])

    def test_non_confirmed_block_production(self):
        # only confirmed mappings are trusted enough not to block.
        for f in self.fields:
            if f["mapping_status"] != "confirmed":
                self.assertTrue(f["blocks_production_xml"], f["esma_code"])

    def test_rrec_not_flat_siblings(self):
        # every RREC field is collateral-level (nested), never a flat sibling
        # record; none is mapped at header/exposure level by default.
        for f in self.fields:
            if f["record_group"] == "RREC":
                self.assertEqual(f["xml_level"], "collateral", f["esma_code"])
                self.assertEqual(f["cardinality"], "one_per_collateral", f["esma_code"])

    def test_confirmed_collateral_is_nested_under_loan(self):
        for f in self.fields:
            if f["record_group"] == "RREC" and f["xml_path"]:
                self.assertIn("/PrfrmgLn/Coll", f["xml_path"], f["esma_code"])

    def test_rrel1_is_report_level_identifier(self):
        r = self.by_code["RREL1"]
        self.assertEqual(r["xml_level"], "header")
        self.assertEqual(r["cardinality"], "one_per_report")
        self.assertTrue(r["xml_path"].endswith("ScrtstnRpt/ScrtstnIdr"))

    def test_value_or_nodata_has_wrapper_path(self):
        for f in self.fields:
            if f["value_mode"] == "value_or_nodata" and f["xml_path"]:
                self.assertTrue(f["nd_wrapper_path"], f["esma_code"])
                self.assertIn("NoDataOptn", f["nd_wrapper_path"])

    def test_summary_counts_consistent(self):
        s = self.m["summary"]
        self.assertEqual(s["total_fields"], 107)
        from collections import Counter
        actual = Counter(f["mapping_status"] for f in self.fields)
        self.assertEqual(s["confirmed"], actual["confirmed"])
        self.assertEqual(s["unresolved"], actual["unresolved"])
        self.assertEqual(s["conflict"], actual["conflict"])
        self.assertEqual(s["production_blocking_mapping_gaps"],
                         sum(1 for f in self.fields if f["blocks_production_xml"]))


class TestProductionGatesUntouched(unittest.TestCase):
    def test_gates_remain_false(self):
        m = _load_map()
        self.assertFalse(m["production_xsd_mapping_configured"])
        gates = m["production_guardrails"]["production_gates_remain"]
        self.assertFalse(gates["xml_generation_allowed"])
        self.assertFalse(gates["xml_generated"])
        self.assertFalse(gates["ready_for_xml_delivery"])

    def test_no_production_xml_generated(self):
        # the mapping layer must not emit any .xml artefact.
        self.assertEqual(list((_REPO / "config" / "delivery").glob("*.xml")), [])
        self.assertEqual(list((_REPO / "output" / "config_review").glob("*.xml")), [])


class TestGeneratorReproducible(unittest.TestCase):
    def test_regeneration_matches_committed(self):
        from scripts.build_annex2_field_xsd_path_map import build_rows
        rows, _ = build_rows()
        committed = _load_map()["fields"]
        self.assertEqual(len(rows), len(committed))
        self.assertEqual([r["esma_code"] for r in rows],
                         [f["esma_code"] for f in committed])
        # statuses are stable across regeneration.
        self.assertEqual([r["mapping_status"] for r in rows],
                         [f["mapping_status"] for f in committed])


if __name__ == "__main__":
    unittest.main()
