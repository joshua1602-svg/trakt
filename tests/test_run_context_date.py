#!/usr/bin/env python3
"""tests/test_run_context_date.py

Source/config-driven ``data_cut_off_date`` (RREL6) extraction + propagation
across Onboarding handoff -> Transformation -> Validation.

Covers:
  * normalize_to_iso / dates_from_filename deterministic parsing;
  * Onboarding extracts data_cut_off_date from a source column;
  * Onboarding extracts data_cut_off_date from a file name;
  * conflicting candidates are surfaced (not silently resolved);
  * missing data_cut_off_date is surfaced clearly;
  * a CLI override is recorded as cli_override;
  * handoff field contract classifies data_cut_off_date as a context field;
  * Transformation materialises the handoff-derived date into every row;
  * Transformation lineage records the source/context origin;
  * Validation passes data_cut_off_date checks after materialisation;
  * Validation still fails when data_cut_off_date is absent/unresolved;
  * no projection/XML artefacts are created.
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.onboarding_agent import run_context as rc
from engine.onboarding_agent import onboarding_handoff as oh
from engine.transformation_agent import transformation_agent as ta
from engine.validation_agent import validation_agent as va

REGISTRY = str(_REPO_ROOT / "config" / "system" / "fields_registry.yaml")
ASSET = str(_REPO_ROOT / "config" / "asset" / "product_defaults_ERM.yaml")
REGIME = str(_REPO_ROOT / "config" / "regime" / "annex2_delivery_rules.yaml")


# --------------------------------------------------------------------------- #
# Unit: deterministic parsing
# --------------------------------------------------------------------------- #
class TestParsing(unittest.TestCase):
    def test_normalize_iso(self):
        self.assertEqual(rc.normalize_to_iso("2026-01-31"), "2026-01-31")

    def test_normalize_dayfirst(self):
        self.assertEqual(rc.normalize_to_iso("31/01/2026"), "2026-01-31")

    def test_normalize_month_name(self):
        self.assertEqual(rc.normalize_to_iso("January 2026"), "2026-01-31")

    def test_normalize_unparseable(self):
        self.assertIsNone(rc.normalize_to_iso("not a date"))
        self.assertIsNone(rc.normalize_to_iso(""))

    def test_filename_mmyyyy(self):
        self.assertEqual(rc.dates_from_filename("SYNTHETIC_ERE_Portfolio_012026.csv"),
                         ["2026-01-31"])

    def test_filename_iso(self):
        self.assertEqual(rc.dates_from_filename("tape_2026-01-31.csv"), ["2026-01-31"])

    def test_filename_yyyymm(self):
        self.assertEqual(rc.dates_from_filename("export_202601.csv"), ["2026-01-31"])

    def test_filename_none(self):
        self.assertEqual(rc.dates_from_filename("portfolio.csv"), [])

    def test_period_token_yyyy_mm(self):
        self.assertEqual(rc.dates_from_period_token("2025-10"), ["2025-10-31"])
        self.assertEqual(rc.dates_from_period_token("mi_2025_10"), ["2025-10-31"])

    def test_period_token_compact(self):
        self.assertEqual(rc.dates_from_period_token("202511"), ["2025-11-30"])

    def test_period_token_rejects_bare_year_or_bad_month(self):
        self.assertEqual(rc.dates_from_period_token("2025"), [])
        self.assertEqual(rc.dates_from_period_token("2025-13"), [])


class TestPeriodInference(unittest.TestCase):
    def test_from_folder_period(self):
        root = Path(tempfile.mkdtemp(prefix="rc_folder_"))
        central = _mk_project(
            root, inventory_rows=[{"file_name": "portfolio.csv"}],
            central_cols=["loan_identifier"], central_rows=[["LN1"]])
        r = rc.extract_data_cut_off_date(root, central, input_dir="/data/input/2025-10")
        self.assertEqual(r["value"], "2025-10-31")
        self.assertEqual(r["source"], rc.SRC_FOLDER_PERIOD)

    def test_from_run_id(self):
        root = Path(tempfile.mkdtemp(prefix="rc_runid_"))
        central = _mk_project(
            root, inventory_rows=[{"file_name": "portfolio.csv"}],
            central_cols=["loan_identifier"], central_rows=[["LN1"]])
        r = rc.extract_data_cut_off_date(root, central, run_id="mi_2025_10")
        self.assertEqual(r["value"], "2025-10-31")
        self.assertEqual(r["source"], rc.SRC_RUN_ID)

    def test_source_column_beats_folder_and_runid(self):
        root = Path(tempfile.mkdtemp(prefix="rc_prio_"))
        central = _mk_project(
            root, inventory_rows=[{"file_name": "p.csv"}],
            central_cols=["data_cut_off_date"], central_rows=[["2025-09-30"]])
        r = rc.extract_data_cut_off_date(
            root, central, input_dir="/data/input/2025-10", run_id="mi_2025_11")
        self.assertEqual(r["value"], "2025-09-30")
        self.assertEqual(r["source"], rc.SRC_SOURCE_COLUMN)


# --------------------------------------------------------------------------- #
# Unit: extraction tiers + conflict / missing
# --------------------------------------------------------------------------- #
def _mk_project(root: Path, *, inventory_rows, central_cols=None, central_rows=None):
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "01_file_inventory.csv", "w", newline="", encoding="utf-8") as fh:
        cols = ["file_name", "file_path", "detected_reporting_date"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in inventory_rows:
            w.writerow({k: r.get(k, "") for k in cols})
    central = root / "central.csv"
    if central_cols is not None:
        with open(central, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(central_cols)
            for r in (central_rows or []):
                w.writerow(r)
    return central


class TestExtraction(unittest.TestCase):
    def test_from_source_column(self):
        root = Path(tempfile.mkdtemp(prefix="rc_col_"))
        central = _mk_project(
            root, inventory_rows=[{"file_name": "p.csv"}],
            central_cols=["data_cut_off_date"], central_rows=[["31/01/2026"], ["31/01/2026"]])
        r = rc.extract_data_cut_off_date(root, central)
        self.assertEqual(r["value"], "2026-01-31")
        self.assertEqual(r["source"], rc.SRC_SOURCE_COLUMN)
        self.assertFalse(r["conflict"])
        self.assertFalse(r["missing"])

    def test_from_filename(self):
        root = Path(tempfile.mkdtemp(prefix="rc_fn_"))
        central = _mk_project(
            root, inventory_rows=[{"file_name": "ERE_Portfolio_012026.csv"}],
            central_cols=["loan_identifier"], central_rows=[["LN1"]])
        r = rc.extract_data_cut_off_date(root, central)
        self.assertEqual(r["value"], "2026-01-31")
        self.assertEqual(r["source"], rc.SRC_FILENAME)

    def test_conflict_surfaced(self):
        root = Path(tempfile.mkdtemp(prefix="rc_conf_"))
        central = _mk_project(
            root,
            inventory_rows=[{"file_name": "a.csv", "detected_reporting_date": "2026-01-31"},
                            {"file_name": "b.csv", "detected_reporting_date": "2025-12-31"}],
            central_cols=["loan_identifier"], central_rows=[["LN1"]])
        r = rc.extract_data_cut_off_date(root, central)
        self.assertTrue(r["conflict"])
        self.assertEqual(r["value"], "")  # never silently resolved
        self.assertIn("2026-01-31", r["conflict_detail"])

    def test_missing_surfaced(self):
        root = Path(tempfile.mkdtemp(prefix="rc_miss_"))
        central = _mk_project(
            root, inventory_rows=[{"file_name": "portfolio.csv"}],
            central_cols=["loan_identifier"], central_rows=[["LN1"]])
        r = rc.extract_data_cut_off_date(root, central)
        self.assertTrue(r["missing"])
        self.assertEqual(r["value"], "")

    def test_cli_override_recorded(self):
        root = Path(tempfile.mkdtemp(prefix="rc_cli_"))
        central = _mk_project(
            root, inventory_rows=[{"file_name": "a.csv", "detected_reporting_date": "2026-01-31"}],
            central_cols=["loan_identifier"], central_rows=[["LN1"]])
        r = rc.extract_data_cut_off_date(
            root, central, cli_reporting_date="2025-06-30", override_reporting_date=True)
        self.assertEqual(r["value"], "2025-06-30")
        self.assertEqual(r["source"], rc.SRC_CLI_OVERRIDE)

    def test_cli_fallback_only_when_missing(self):
        root = Path(tempfile.mkdtemp(prefix="rc_fb_"))
        central = _mk_project(
            root, inventory_rows=[{"file_name": "portfolio.csv"}],
            central_cols=["loan_identifier"], central_rows=[["LN1"]])
        r = rc.extract_data_cut_off_date(root, central, cli_reporting_date="2025-06-30")
        self.assertEqual(r["value"], "2025-06-30")
        self.assertEqual(r["source"], rc.SRC_CLI_FALLBACK)


# --------------------------------------------------------------------------- #
# Integration: handoff context -> transformation -> validation
# --------------------------------------------------------------------------- #

_TAPE_HEADER = ["unique_identifier", "loan_identifier", "current_principal_balance"]
_TAPE_ROWS = [["LN0001", "LN0001", "1000"], ["LN0002", "LN0002", "2000"]]


def _write_handoff(root: Path, *, with_context_date: bool) -> Path:
    output = root / "output"
    handoff = output / "handoff"
    central = output / "central"
    handoff.mkdir(parents=True, exist_ok=True)
    central.mkdir(parents=True, exist_ok=True)

    # central tape WITHOUT a populated data_cut_off_date column (the real-client case)
    with open(central / "18_central_lender_tape.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(_TAPE_HEADER)
        w.writerows(_TAPE_ROWS)

    contract = [
        dict(target_field="RREL1", esma_code="RREL1", canonical_field="unique_identifier",
             domain="loan", coverage_status="source_mapped", selected_source_file="raw.csv",
             selected_source_column="Id", selected_value_sample="",
             handoff_classification="source_mapped",
             downstream_owner="transformation_validation", notes="", blocking_decision=False),
        dict(target_field="RREL6", esma_code="RREL6", canonical_field="data_cut_off_date",
             domain="loan", coverage_status="source_mapped", selected_source_file="raw.csv",
             selected_source_column="Data Cut-Off Date",
             selected_value_sample=("2026-01-31" if with_context_date else ""),
             handoff_classification=("source_context_mapped" if with_context_date else "source_mapped"),
             downstream_owner="transformation_validation",
             notes="portfolio-level source_column", blocking_decision=False),
    ]
    (handoff / "26_onboarding_handoff_field_contract.json").write_text(
        json.dumps({"target_contract_id": "esma_annex_2", "rows": contract}), encoding="utf-8")
    cols = list(contract[0].keys())
    with open(handoff / "26_onboarding_handoff_field_contract.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in contract:
            w.writerow(r)
    (handoff / "27_onboarding_handoff_lineage.json").write_text(
        json.dumps({"rows": [{"target_field": "RREL1"}]}), encoding="utf-8")

    manifest = {
        "client_id": "client_001", "run_id": "run_ctx",
        "target_contract_id": "esma_annex_2",
        "handoff_type": "canonical_onboarding_package",
        "next_agent": "transformation_validation",
        "not_raw_source": True, "do_not_rerun_gate1_on_central_tape": True,
        "central_tape_path": "central/18_central_lender_tape.csv",
        "blocking_decision_count": 0,
        "asset_config_path": ASSET, "regime_config_path": REGIME, "registry_path": REGISTRY,
        "ready_for_transformation_validation": True,
        "data_cut_off_date": ("2026-01-31" if with_context_date else ""),
        "data_cut_off_date_source": ("source_column" if with_context_date else ""),
        "run_context_fields": ["data_cut_off_date"],
    }
    mpath = handoff / "24_onboarding_handoff_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return mpath


class TestContextMaterialisation(unittest.TestCase):
    def test_materialises_into_every_row(self):
        root = Path(tempfile.mkdtemp(prefix="rc_tx_"))
        mpath = _write_handoff(root, with_context_date=True)
        ta.build_transformation_package(mpath)
        out = root / "output" / "transformation"
        import pandas as pd
        df = pd.read_csv(out / "31_transformed_canonical_tape.csv", dtype=str)
        self.assertEqual(df["data_cut_off_date"].tolist(), ["2026-01-31", "2026-01-31"])
        contract = json.loads((out / "32_transformation_field_contract.json").read_text())["rows"]
        dc = [r for r in contract if r["canonical_field"] == "data_cut_off_date"][0]
        self.assertEqual(dc["transformation_status"], ta.TS_SOURCE_CONTEXT)
        # lineage records the source/context origin
        lin = json.loads((out / "34_transformation_lineage.json").read_text())
        tx = [r for r in lin["transformation_lineage"]
              if r["transformed_field"] == "data_cut_off_date"][0]
        self.assertEqual(tx["transformation_applied"], ta.TS_SOURCE_CONTEXT)
        self.assertTrue(tx["default_source"].startswith("handoff_"))

    def test_validation_passes_after_materialisation(self):
        root = Path(tempfile.mkdtemp(prefix="rc_val_"))
        mpath = _write_handoff(root, with_context_date=True)
        ta.build_transformation_package(mpath)
        tx_manifest = root / "output" / "transformation" / "30_transformation_manifest.json"
        va.build_validation_package(tx_manifest)
        res = json.loads((root / "output" / "validation" /
                          "41_validation_results.json").read_text())["rows"]
        cutoff = [r for r in res if r["validation_rule_id"] == "BR-CUTOFF-DATE"]
        self.assertTrue(cutoff)
        self.assertEqual(cutoff[0]["status"], "pass")
        m = json.loads((root / "output" / "validation" /
                        "40_validation_manifest.json").read_text())
        self.assertEqual(m["blocking_for_validation_count"], 0)
        # no projection / xml artefacts created
        self.assertFalse((root / "output" / "projection").exists())
        self.assertFalse(m["performed_projection"])

    def test_validation_fails_when_absent(self):
        root = Path(tempfile.mkdtemp(prefix="rc_absent_"))
        mpath = _write_handoff(root, with_context_date=False)
        ta.build_transformation_package(mpath)
        tx_manifest = root / "output" / "transformation" / "30_transformation_manifest.json"
        va.build_validation_package(tx_manifest)
        res = json.loads((root / "output" / "validation" /
                          "41_validation_results.json").read_text())["rows"]
        # data_cut_off_date is a mandatory RREL6 field with no ND/default allowed;
        # absent/unresolved -> a blocking validation failure.
        cutoff_fail = [r for r in res
                       if r["canonical_field"] == "data_cut_off_date"
                       and r["status"] == "fail"]
        self.assertTrue(cutoff_fail)
        m = json.loads((root / "output" / "validation" /
                        "40_validation_manifest.json").read_text())
        self.assertGreater(m["blocking_for_validation_count"], 0)
        self.assertFalse(m["ready_for_validation_complete"])


if __name__ == "__main__":
    unittest.main()
