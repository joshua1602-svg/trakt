#!/usr/bin/env python3
"""tests/test_onboarding_date_semantics.py

Asset-agnostic onboarding date semantics: funded-book reporting date vs pipeline
snapshot date.

Proves the design:
  * funded_reporting_date aliases resolve to reporting_date;
  * pipeline_snapshot_date is a separate field (MI registry) from reporting_date;
  * loan/collateral/cashflow on the same funded date passes;
  * loan/collateral/cashflow date mismatch blocks (date_basis_mismatch);
  * a pipeline date different from the funded date is allowed (non-blocking);
  * a pipeline date from a filename maps to pipeline_snapshot_date;
  * a funded loan-tape date from filename/folder maps to reporting_date;
  * role/date folders and an explicit run manifest both work;
  * regulatory_mi behaviour (data_cut_off_date) is unchanged.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml

from engine.onboarding_agent import date_semantics as ds

MI_REGISTRY = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"
REG_REGISTRY = _REPO_ROOT / "config" / "system" / "fields_registry.yaml"
ALIASES_MANDATORY = _REPO_ROOT / "config" / "system" / "aliases_mandatory.yaml"


# --------------------------------------------------------------------------- #
# Registry / alias shape
# --------------------------------------------------------------------------- #
class TestRegistryShape(unittest.TestCase):
    def setUp(self):
        self.mi = yaml.safe_load(MI_REGISTRY.read_text(encoding="utf-8"))["fields"]

    def test_pipeline_snapshot_date_is_separate_mi_field(self):
        self.assertIn("pipeline_snapshot_date", self.mi)
        self.assertIn("reporting_date", self.mi)
        self.assertNotEqual("pipeline_snapshot_date", "reporting_date")
        psd = self.mi["pipeline_snapshot_date"]
        self.assertEqual(psd["role"], "date")
        self.assertTrue(psd.get("virtual"))
        # MI/pipeline date, not regulatory core.
        self.assertIn("pipeline_state", psd.get("source_criteria", []))

    def test_pipeline_snapshot_date_not_in_regulatory_core(self):
        reg = yaml.safe_load(REG_REGISTRY.read_text(encoding="utf-8"))
        reg_fields = reg.get("fields", reg)
        self.assertNotIn("pipeline_snapshot_date", reg_fields)

    def test_reporting_date_carries_funded_aliases(self):
        syns = [s.lower() for s in self.mi["reporting_date"]["synonyms"]]
        for alias in ("funded_reporting_date", "funded book reporting date",
                      "funded as-of date", "loan tape reporting date",
                      "loan extract reporting date", "book date"):
            self.assertIn(alias, syns, alias)

    def test_regulatory_data_cut_off_date_unchanged(self):
        # The regulatory field keeps its own cut-off aliases (not weakened).
        aliases = yaml.safe_load(ALIASES_MANDATORY.read_text(encoding="utf-8"))
        self.assertIn("data_cut_off_date", aliases)
        al = [a.lower() for a in aliases["data_cut_off_date"]["aliases"]]
        self.assertIn("cut-off date", al)
        self.assertIn("data cut off date", al)


# --------------------------------------------------------------------------- #
# Alias resolution
# --------------------------------------------------------------------------- #
class TestAliasResolution(unittest.TestCase):
    def test_funded_aliases_resolve_to_reporting_date(self):
        for alias in ("funded_reporting_date", "funded book reporting date",
                      "loan tape reporting date", "book date"):
            r = ds.resolve_date_field(alias, role="current_loan_report")
            self.assertEqual(r["canonical_field"], ds.REPORTING_DATE, alias)

    def test_pipeline_aliases_resolve_to_pipeline_snapshot_date(self):
        for alias in ("pipeline snapshot date", "pipeline as-of date",
                      "kfi pipeline date", "application pipeline date"):
            r = ds.resolve_date_field(alias, role="pipeline_report")
            self.assertEqual(r["canonical_field"], ds.PIPELINE_SNAPSHOT_DATE, alias)

    def test_generic_date_disambiguated_by_role(self):
        funded = ds.resolve_date_field("as of date", role="collateral_report")
        self.assertEqual(funded["canonical_field"], ds.REPORTING_DATE)
        pipe = ds.resolve_date_field("as of date", role="pipeline_report")
        self.assertEqual(pipe["canonical_field"], ds.PIPELINE_SNAPSHOT_DATE)

    def test_funded_token_in_pipeline_artefact_is_pipeline(self):
        # A pipeline file's "reporting date" column is a pipeline snapshot date.
        r = ds.resolve_date_field("reporting date", role="pipeline_report")
        self.assertEqual(r["canonical_field"], ds.PIPELINE_SNAPSHOT_DATE)


# --------------------------------------------------------------------------- #
# Role basis
# --------------------------------------------------------------------------- #
class TestRoleBasis(unittest.TestCase):
    def test_funded_roles(self):
        for role in ("current_loan_report", "collateral_report", "cashflow_report"):
            self.assertEqual(ds.basis_for_role(role), ds.BASIS_FUNDED, role)
            self.assertEqual(ds.canonical_date_field_for_role(role), ds.REPORTING_DATE)

    def test_pipeline_role(self):
        self.assertEqual(ds.basis_for_role("pipeline_report"), ds.BASIS_PIPELINE)
        self.assertEqual(ds.canonical_date_field_for_role("pipeline_report"),
                         ds.PIPELINE_SNAPSHOT_DATE)


# --------------------------------------------------------------------------- #
# Folder / manifest conventions
# --------------------------------------------------------------------------- #
class TestFolderManifest(unittest.TestCase):
    def test_role_date_folders(self):
        f = ds.parse_role_date_path("input/funded/2025-11-30/loan.csv")
        self.assertEqual((f["basis"], f["date"]), (ds.BASIS_FUNDED, "2025-11-30"))
        p = ds.parse_role_date_path("input/pipeline/2025-12-01/kfi.csv")
        self.assertEqual((p["basis"], p["date"]), (ds.BASIS_PIPELINE, "2025-12-01"))

    def test_explicit_manifest(self):
        man = {"mi_package": {"funded_reporting_date": "2025-11-30",
                              "pipeline_snapshot_date": "2025-12-01"}}
        out = ds.load_run_manifest(man)
        self.assertEqual(out["funded_reporting_date"], "2025-11-30")
        self.assertEqual(out["pipeline_snapshot_date"], "2025-12-01")

    def test_manifest_overrides_per_artefact_not_single_parent(self):
        man = {"funded_reporting_date": "2025-11-30",
               "pipeline_snapshot_date": "2025-12-01"}
        arts = [
            {"file_name": "loan.csv", "role": "current_loan_report"},
            {"file_name": "kfi.csv", "role": "pipeline_report"},
        ]
        dated = {a.role: a for a in ds.assign_artefact_dates(arts, manifest=man)}
        self.assertEqual(dated["current_loan_report"].date, "2025-11-30")
        self.assertEqual(dated["pipeline_report"].date, "2025-12-01")
        self.assertEqual(dated["current_loan_report"].canonical_field, ds.REPORTING_DATE)
        self.assertEqual(dated["pipeline_report"].canonical_field,
                         ds.PIPELINE_SNAPSHOT_DATE)


# --------------------------------------------------------------------------- #
# Inference: filename/folder -> correct basis
# --------------------------------------------------------------------------- #
class TestInferenceMapping(unittest.TestCase):
    def test_pipeline_date_from_filename_maps_to_pipeline_snapshot(self):
        arts = [{"file_name": "origination_pipeline_2025-12-01.csv",
                 "role": "pipeline_report"}]
        ad = ds.assign_artefact_dates(arts)[0]
        self.assertEqual(ad.canonical_field, ds.PIPELINE_SNAPSHOT_DATE)
        self.assertEqual(ad.date, "2025-12-01")
        self.assertEqual(ad.source, "filename")

    def test_funded_tape_date_from_filename_maps_to_reporting_date(self):
        arts = [{"file_name": "loan_tape_2025-11-30.csv",
                 "role": "current_loan_report"}]
        ad = ds.assign_artefact_dates(arts)[0]
        self.assertEqual(ad.canonical_field, ds.REPORTING_DATE)
        self.assertEqual(ad.date, "2025-11-30")

    def test_funded_tape_date_from_folder_maps_to_reporting_date(self):
        arts = [{"file_name": "loan.csv", "role": "current_loan_report",
                 "folder": "input/funded/2025-11-30/loan.csv"}]
        ad = ds.assign_artefact_dates(arts)[0]
        self.assertEqual(ad.canonical_field, ds.REPORTING_DATE)
        self.assertEqual(ad.date, "2025-11-30")
        self.assertEqual(ad.source, "role_date_folder")


# --------------------------------------------------------------------------- #
# Consistency validation — the October/November shape
# --------------------------------------------------------------------------- #
class TestConsistency(unittest.TestCase):
    def _funded_package(self, loan, collat, cash, pipeline=None):
        arts = [
            {"file_name": "loan.csv", "role": "current_loan_report", "detected_date": loan},
            {"file_name": "collateral.csv", "role": "collateral_report", "detected_date": collat},
            {"file_name": "cashflow.csv", "role": "cashflow_report", "detected_date": cash},
        ]
        if pipeline:
            arts.append({"file_name": "pipeline.csv", "role": "pipeline_report",
                         "detected_date": pipeline})
        return ds.assign_artefact_dates(arts)

    def test_same_funded_date_passes(self):
        dates = self._funded_package("2025-11-30", "2025-11-30", "2025-11-30")
        res = ds.validate_date_consistency(dates)
        self.assertFalse(res["blocking"])
        self.assertEqual(res["funded_reporting_date"], "2025-11-30")
        self.assertEqual(res["issues"], [])

    def test_funded_mismatch_blocks(self):
        # collateral 2025-12-01 vs loan/cashflow 2025-11-30 -> blocking mismatch.
        dates = self._funded_package("2025-11-30", "2025-12-01", "2025-11-30")
        res = ds.validate_date_consistency(dates)
        self.assertTrue(res["blocking"])
        codes = [i["code"] for i in res["issues"]]
        self.assertIn(ds.ISSUE_BASIS_MISMATCH, codes)

    def test_funded_mismatch_can_be_explicitly_approved(self):
        dates = self._funded_package("2025-11-30", "2025-12-01", "2025-11-30")
        res = ds.validate_date_consistency(dates, approved_mismatch=True)
        self.assertFalse(res["blocking"])
        mm = [i for i in res["issues"] if i["code"] == ds.ISSUE_BASIS_MISMATCH][0]
        self.assertTrue(mm["approved"])

    def test_pipeline_date_differs_is_allowed(self):
        # funded 2025-11-30, pipeline 2025-12-01 -> valid, non-blocking, recorded.
        dates = self._funded_package("2025-11-30", "2025-11-30", "2025-11-30",
                                     pipeline="2025-12-01")
        res = ds.validate_date_consistency(dates)
        self.assertFalse(res["blocking"])
        self.assertEqual(res["funded_reporting_date"], "2025-11-30")
        self.assertEqual(res["pipeline_snapshot_date"], "2025-12-01")
        diff = [i for i in res["issues"] if i["code"] == ds.ISSUE_PIPELINE_DIFFERENCE]
        self.assertEqual(len(diff), 1)
        self.assertFalse(diff[0]["blocking"])

    def test_pipeline_date_never_forced_onto_funded(self):
        dates = self._funded_package("2025-11-30", "2025-11-30", "2025-11-30",
                                     pipeline="2025-12-01")
        funded = [a for a in dates if a.basis == ds.BASIS_FUNDED]
        for a in funded:
            self.assertEqual(a.date, "2025-11-30")
            self.assertEqual(a.canonical_field, ds.REPORTING_DATE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
