#!/usr/bin/env python3
"""tests/test_xsd_structured_synthetic.py

Fourth non-production mode: xsd_structured_synthetic_schema_test — an
engineering-only schema test that builds the full mandatory ESMA tree with
type-valid DUMMY values (including economic) to attempt full DRAFT-XSD validation.

Covers:
  * disabled by default; disabled mode emits no XML;
  * emits only when explicitly enabled;
  * uses the real nested ESMA paths (Document/.../UndrlygXpsrRcrd/.../Coll);
  * synthetic economic values allowed ONLY in this mode;
  * xsd_structured_preview still does NOT fabricate economic values;
  * every synthetic value labelled synthetic_schema_test;
  * output watermarked + engineering-only;
  * production gates remain false; no production XML;
  * XSD validation report written honestly (attempted/passed/error_count/...).
"""

from __future__ import annotations

import copy
import csv
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


def _enable(tmp: Path, *, synthetic=False, preview=False):
    data = copy.deepcopy(yaml.safe_load(Path(base.POLICY).read_text()))
    data["preview_modes"]["xsd_structured_synthetic_schema_test"]["enabled"] = synthetic
    data["preview_modes"]["xsd_structured_preview"]["enabled"] = preview
    p = tmp / "policy_synth.yaml"
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    return str(p)


def _run(root, *, synthetic=False, preview=False):
    out = base._build_delivery(root, base._CLEAN_FRAME_SPEC, base._CLEAN_ISSUES)
    policy = _enable(root, synthetic=synthetic, preview=preview)
    res = pr.evaluate_and_emit(out, policy_path=policy, field_universe_path=base.UNIVERSE)
    return out, res


class TestDisabledByDefault(unittest.TestCase):
    def test_disabled_in_committed_policy(self):
        data = yaml.safe_load(Path(base.POLICY).read_text())
        self.assertFalse(
            data["preview_modes"]["xsd_structured_synthetic_schema_test"]["enabled"])

    def test_disabled_emits_no_xml(self):
        root = Path(tempfile.mkdtemp(prefix="synth_off_"))
        out, res = _run(root, synthetic=False)
        self.assertFalse(res["flags"]["xsd_structured_synthetic_generated"])
        self.assertFalse((out / "preview" / "xsd_structured_synthetic_schema_test").exists())


class TestSyntheticEnabled(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="synth_on_"))
        cls.out, cls.res = _run(cls.root, synthetic=True)
        cls.sp = cls.out / "preview" / "xsd_structured_synthetic_schema_test"
        cls.xml_text = (cls.sp / "114_xsd_structured_synthetic.xml").read_text()
        cls.validation = json.loads(
            (cls.sp / "116_xsd_structured_synthetic_xsd_validation.json").read_text())
        cls.catalog = list(csv.DictReader(
            open(cls.sp / "112_xsd_structured_synthetic_values_catalog.csv")))

    def test_generated_and_artefacts(self):
        self.assertTrue(self.res["flags"]["xsd_structured_synthetic_generated"])
        for name in ("110_xsd_structured_synthetic_frame.csv",
                     "111_xsd_structured_synthetic_lineage.json",
                     "112_xsd_structured_synthetic_values_catalog.csv",
                     "113_xsd_structured_synthetic_watermark.txt",
                     "114_xsd_structured_synthetic.xml",
                     "115_xsd_structured_synthetic_summary.md",
                     "116_xsd_structured_synthetic_xsd_validation.json"):
            self.assertTrue((self.sp / name).exists(), name)

    def test_uses_real_nested_esma_paths(self):
        root = ET.fromstring(self.xml_text)
        self.assertEqual(root.tag, f"{NS}Document")
        self.assertIsNotNone(root.find(
            f"{NS}ScrtstnNonAsstBckdComrclPprUndrlygXpsrRpt/{NS}NewCrrctn/{NS}ScrtstnRpt"))
        # collateral nested under the loan.
        ok = False
        for prfrmg in root.iter(f"{NS}PrfrmgLn"):
            if prfrmg.find(f"{NS}Coll") is not None:
                ok = True
        self.assertTrue(ok, "Coll must be nested under PrfrmgLn")

    def test_every_value_labelled_synthetic(self):
        self.assertTrue(self.catalog)
        for r in self.catalog:
            self.assertEqual(r["source"], "synthetic_schema_test", r["xml_path"])
            self.assertEqual(r["source_reason"], "synthetic_schema_test")

    def test_economic_values_present_in_synthetic_only(self):
        # synthetic mode DOES populate valuation amounts (with synthetic Ccy amount).
        self.assertIn("ValtnAmt", self.xml_text)
        self.assertIn("Ccy=", self.xml_text)

    def test_xsd_validation_reported_honestly(self):
        v = self.validation
        for k in ("xsd_validation_attempted", "xsd_validation_passed", "xsd_path",
                  "validation_errors", "error_count", "known_limitations",
                  "records_generated", "fields_generated", "synthetic_values_count"):
            self.assertIn(k, v)
        self.assertTrue(v["xsd_validation_attempted"])
        self.assertEqual(v["error_count"], len(v["validation_errors"]))
        self.assertGreater(v["synthetic_values_count"], 0)
        # if it didn't pass, errors must be present (honest).
        if not v["xsd_validation_passed"]:
            self.assertGreater(v["error_count"], 0)

    def test_watermarked_engineering_only(self):
        self.assertIn("SYNTHETIC SCHEMA TEST", self.xml_text)
        self.assertIn("ENGINEERING ONLY", self.xml_text)
        self.assertIn("synthetic_schema_test", self.xml_text)

    def test_production_gates_false(self):
        manifest = json.loads((self.out / "60_delivery_manifest.json").read_text())
        self.assertFalse(any([manifest["xml_generation_allowed"],
                              manifest["ready_for_xml_delivery"], manifest["xml_generated"]]))
        self.assertFalse(any(self.res["flags"]["production_flags_unchanged"].values()))

    def test_no_production_xml(self):
        self.assertEqual(list(self.out.glob("*.xml")), [])


class TestPreviewStaysHonest(unittest.TestCase):
    """The client-safe preview must NOT fabricate economic values even though the
    synthetic mode does."""

    def test_preview_does_not_fabricate_economic(self):
        root = Path(tempfile.mkdtemp(prefix="synth_hon_"))
        out, res = _run(root, preview=True)  # synthetic disabled
        pv = out / "preview" / "xsd_structured_preview" / "105_xsd_structured_preview.xml"
        xml = pv.read_text()
        self.assertNotIn("ValtnAmt", xml)
        self.assertNotIn("IncmVal", xml)
        # synthetic mode not generated when disabled.
        self.assertFalse(res["flags"]["xsd_structured_synthetic_generated"])
        # preview summary explicitly explains why validation may fail.
        summ = (out / "preview" / "xsd_structured_preview"
                / "106_xsd_structured_preview_summary.md").read_text()
        self.assertIn("economically meaningful mandatory fields", summ)


class TestInspectHelper(unittest.TestCase):
    def test_inspect_synthetic(self):
        root = Path(tempfile.mkdtemp(prefix="synth_insp_"))
        out, _ = _run(root, synthetic=True)
        from scripts.inspect_delivery_xml_readiness import inspect_xsd_structured_synthetic
        p = inspect_xsd_structured_synthetic(out)
        self.assertTrue(p["exists"])
        self.assertTrue(p["xml_generated"])
        self.assertTrue(p["xsd_validation_attempted"])
        self.assertGreater(p["synthetic_values_count"], 0)
        self.assertTrue(p["production_gates_false"])


if __name__ == "__main__":
    unittest.main()
