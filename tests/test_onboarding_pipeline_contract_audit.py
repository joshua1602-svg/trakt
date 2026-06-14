#!/usr/bin/env python3
"""tests/test_onboarding_pipeline_contract_audit.py — PART 13 (1, contract fields)."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engine.onboarding_agent import pipeline_field_contract as pfc

REGISTRY = str(_REPO_ROOT / "config" / "system" / "fields_registry.yaml")


class TestPipelineContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = pfc.build_pipeline_field_contract(REGISTRY)
        cls.out = Path(tempfile.mkdtemp())
        cls.paths = pfc.write_contract_artifacts(cls.rows, cls.out)

    # 1. Contract artefacts are generated.
    def test_artefacts_written(self):
        for k in ("csv", "json", "summary_md"):
            self.assertTrue(Path(self.paths[k]).exists())
        with open(self.paths["csv"], newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertTrue(rows)
        for col in ("field_name", "is_pipeline_field", "is_funded_field",
                    "current_registry_status", "recommended_registry_action"):
            self.assertIn(col, rows[0])

    # The contract surfaces the KFI/application/offer/funded date + stage fields.
    def test_expected_pipeline_fields_present(self):
        names = {r["field_name"] for r in self.rows}
        for f in ("kfi_submitted_date", "application_submitted_date", "offer_date",
                  "date_funds_released", "product", "product_rate", "broker",
                  "status_raw", "loan_amount", "max_facility", "property_region"):
            self.assertIn(f, names)

    # Pipeline-specific fields are flagged missing-from-registry (need a home).
    def test_missing_fields_flagged(self):
        missing = [r for r in self.rows
                   if r["current_registry_status"] == "missing_from_registry"
                   and not r["is_derived"]]
        self.assertTrue(missing)
        self.assertTrue(any(r["recommended_registry_action"]
                            == "add_to_pipeline_registry_extension" for r in missing))

    # The contract matches the LIVE pipeline_prep module (cross-check, not a copy).
    def test_matches_live_pipeline_prep(self):
        try:
            from analytics.pipeline_prep import normalize_pipeline_snapshot
        except Exception:
            self.skipTest("analytics.pipeline_prep not importable in this env")
        df = pd.read_csv(_REPO_ROOT / "tests" / "fixtures" / "kfi_pipeline_headers.csv")
        out = normalize_pipeline_snapshot(df)
        produced = set(out.columns)
        present_headers = set(df.columns)
        # Every contract field whose raw header is in THIS file should be produced
        # by the live pipeline_prep, proving the contract mirrors the real module.
        checked = 0
        for r in self.rows:
            if (r["raw_source_header"] in present_headers and not r["is_derived"]):
                self.assertIn(r["field_name"], produced,
                              f"{r['field_name']} not produced by pipeline_prep")
                checked += 1
        self.assertGreater(checked, 20, "expected many KFI fields cross-checked")


if __name__ == "__main__":
    unittest.main()
