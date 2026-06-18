#!/usr/bin/env python3
"""tests/test_xsd_structured_preview.py

Third non-production mode: xsd_structured_preview — places values inside the
REAL ESMA Annex 2 XSD hierarchy using only builder-accepted field-to-XSD paths.

Covers:
  * disabled by default; disabled mode emits no XML;
  * emits only when explicitly enabled AND readiness allows;
  * production gates remain false; no production XML generated;
  * output goes only under preview/xsd_structured_preview;
  * nested (Document/.../UndrlygXpsrRcrd/.../PrfrmgLn) XML, not flat;
  * RREC/collateral fields nested under Coll;
  * RREC1/RREC2 polluted paths and rejected/needs_manual_review paths are not used;
  * watermark present; XML well-formed;
  * XSD validation result recorded honestly;
  * no valuation/rate/economic fields fabricated;
  * max_records limit respected.
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import tests.test_delivery_xml_preview_readiness as base
from engine.delivery_xml_agent import preview_readiness as pr

NS = "{urn:esma:xsd:DRAFT1auth.099.001.04}"
_PATHMAP = _REPO / "config" / "delivery" / "annex2_field_xsd_path_map.yaml"


def _enable_policy(tmp: Path, *, enabled: bool, max_records: int = 5) -> str:
    data = copy.deepcopy(yaml.safe_load(Path(base.POLICY).read_text()))
    data["preview_modes"]["xsd_structured_preview"]["enabled"] = enabled
    data["preview_modes"]["xsd_structured_preview"]["max_records"] = max_records
    p = tmp / "policy_xsd.yaml"
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    return str(p)


def _run(root, *, enabled, max_records=5):
    out = base._build_delivery(root, base._CLEAN_FRAME_SPEC, base._CLEAN_ISSUES)
    policy = _enable_policy(root, enabled=enabled, max_records=max_records)
    res = pr.evaluate_and_emit(out, policy_path=policy, field_universe_path=base.UNIVERSE)
    return out, res


class TestDisabledByDefault(unittest.TestCase):
    def test_disabled_in_committed_policy(self):
        data = yaml.safe_load(Path(base.POLICY).read_text())
        self.assertFalse(data["preview_modes"]["xsd_structured_preview"]["enabled"])

    def test_disabled_emits_no_xml(self):
        root = Path(tempfile.mkdtemp(prefix="xsd_off_"))
        out, res = _run(root, enabled=False)
        self.assertFalse(res["flags"]["xsd_structured_preview_generated"])
        self.assertFalse((out / "preview" / "xsd_structured_preview").exists())
        # readiness artefacts ARE written even when disabled.
        self.assertTrue((out / "preview" / "78_xsd_structured_preview_readiness.json").exists())
        self.assertTrue((out / "preview" / "79_xsd_structured_preview_readiness.md").exists())


class TestEnabledEmits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="xsd_on_"))
        cls.out, cls.res = _run(cls.root, enabled=True)
        cls.sp = cls.out / "preview" / "xsd_structured_preview"
        cls.verdict = cls.res["xsd_structured_preview_verdict"]
        cls.xml_text = (cls.sp / "105_xsd_structured_preview.xml").read_text()

    def test_allowed_and_generated(self):
        self.assertTrue(self.verdict["allowed"])
        self.assertTrue(self.res["flags"]["xsd_structured_preview_generated"])
        self.assertTrue(self.res["flags"]["ready_for_xsd_structured_preview"])

    def test_all_artefacts_written(self):
        for name in ("100_xsd_structured_preview_frame.csv",
                     "101_xsd_structured_preview_lineage.json",
                     "102_xsd_structured_preview_assumptions.csv",
                     "103_xsd_structured_preview_exclusions.csv",
                     "104_xsd_structured_preview_watermark.txt",
                     "105_xsd_structured_preview.xml",
                     "106_xsd_structured_preview_summary.md",
                     "107_xsd_structured_preview_xsd_validation.json"):
            self.assertTrue((self.sp / name).exists(), name)

    def test_output_only_under_preview_dir(self):
        self.assertEqual(self.sp.parent.name, "preview")
        self.assertEqual(self.sp.parent.parent.name, "delivery_xml")
        # no production XML anywhere in the run output.
        prod = [p for p in (self.out).glob("*.xml")]
        self.assertEqual(prod, [])

    def test_production_gates_false(self):
        manifest = json.loads((self.out / "60_delivery_manifest.json").read_text())
        self.assertFalse(manifest["xml_generation_allowed"])
        self.assertFalse(manifest["ready_for_xml_delivery"])
        self.assertFalse(manifest["xml_generated"])
        pf = self.res["flags"]["production_flags_unchanged"]
        self.assertFalse(any(pf.values()))

    def test_xml_well_formed_and_nested(self):
        root = ET.fromstring(self.xml_text)
        self.assertEqual(root.tag, f"{NS}Document")
        # real nested chain, not the flat TraktNonProductionPreview.
        self.assertNotIn("TraktNonProductionPreview", self.xml_text)
        self.assertIsNotNone(root.find(
            f"{NS}ScrtstnNonAsstBckdComrclPprUndrlygXpsrRpt/{NS}NewCrrctn/{NS}ScrtstnRpt"))
        self.assertTrue(root.iter(f"{NS}UndrlygXpsrRcrd"))

    def test_rrec_nested_under_coll(self):
        # every Coll sits under PrfrmgLn, and RREC fields sit under Coll.
        colls = list(_iter_with_parent(ET.fromstring(self.xml_text)))
        coll_under_prfrmg = [c for (c, parent) in colls
                             if c.tag == f"{NS}Coll" and parent is not None
                             and parent.tag == f"{NS}PrfrmgLn"]
        self.assertTrue(coll_under_prfrmg, "Coll must be nested under PrfrmgLn")
        # PrprtyTp (RREC9) must live under a Coll subtree.
        root = ET.fromstring(self.xml_text)
        for coll in root.iter(f"{NS}Coll"):
            if coll.find(f".//{NS}PrprtyTp") is not None:
                break
        else:
            self.fail("RREC9 PrprtyTp not found nested under Coll")

    def test_rejected_and_manual_paths_not_used(self):
        pm = {f["esma_code"]: f for f in
              yaml.safe_load(_PATHMAP.read_text())["field_xsd_path_map"]["fields"]}
        for code in self.verdict["emit_field_codes"]:
            self.assertIn(pm[code]["builder_acceptance_status"],
                          ("sample_confirmed", "accepted_for_builder"), code)
        # RREC1/RREC2 (rejected pollution) never appear in emitted codes.
        self.assertNotIn("RREC1", self.verdict["emit_field_codes"])
        self.assertNotIn("RREC2", self.verdict["emit_field_codes"])
        self.assertEqual(self.verdict["rejected_or_manual_skipped"], [])

    def test_no_economic_fabrication(self):
        # RREC17 (Original Valuation Amount) is blocked -> excluded, never emitted.
        self.assertIn("RREC17", self.verdict["excluded_codes"])
        self.assertNotIn("RREC17", self.verdict["emit_field_codes"])
        self.assertNotIn("ValtnAmt", self.xml_text)

    def test_watermark_present(self):
        self.assertIn("XSD-STRUCTURED NON-PRODUCTION PREVIEW", self.xml_text)
        self.assertIn("Production XML remains blocked", self.xml_text)
        self.assertTrue((self.sp / "104_xsd_structured_preview_watermark.txt").read_text().strip())

    def test_xsd_validation_recorded_honestly(self):
        v = json.loads((self.sp / "107_xsd_structured_preview_xsd_validation.json").read_text())
        for k in ("xsd_validation_attempted", "xsd_validation_passed", "xsd_path",
                  "validation_errors", "known_limitations"):
            self.assertIn(k, v)
        # we do not claim production validity; if it didn't pass, limitations are listed.
        if not v["xsd_validation_passed"]:
            self.assertTrue(v["known_limitations"])

    def test_placeholders_only_for_identifier_fields(self):
        # RREL2/RREL82 (blocked identifiers) -> placeholders; RREC9/RREL35 real.
        self.assertIn("PREVIEW_ONLY_RREL2", self.xml_text)
        self.assertIn("RREL2", self.verdict["placeholder_codes"])
        self.assertNotIn("RREC9", self.verdict["placeholder_codes"])


class TestMaxRecords(unittest.TestCase):
    def test_max_records_respected(self):
        root = Path(tempfile.mkdtemp(prefix="xsd_max_"))
        out, res = _run(root, enabled=True, max_records=1)
        xml_text = (out / "preview" / "xsd_structured_preview"
                    / "105_xsd_structured_preview.xml").read_text()
        rec = list(ET.fromstring(xml_text).iter(f"{NS}UndrlygXpsrRcrd"))
        # fixture has 2 loans; max_records=1 must cap to a single record.
        self.assertEqual(len(rec), 1)


def _iter_with_parent(root):
    for parent in root.iter():
        for child in list(parent):
            yield child, parent


if __name__ == "__main__":
    unittest.main()
