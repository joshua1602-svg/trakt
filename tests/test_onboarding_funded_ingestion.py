#!/usr/bin/env python3
"""tests/test_onboarding_funded_ingestion.py

Real workflow-level regression for the funded-extract ingestion path.

Root cause fixed: file discovery was non-recursive, so files under role/date
subfolders (input/funded/.../, input/pipeline/.../) never entered the
candidate/evidence/mapping pipeline, leaving the funded base-MI fields blocked.

This runs the SAME workflow path the CLI uses (``run_operator_workflow``) on a
fixture that mirrors the real pack — LoanExtract One + PropertyExtract in a
funded subfolder and a Pipeline file in a pipeline subfolder — and asserts the
funded columns flow all the way through to coverage and the central tape, with
the artefact-role guardrail keeping pipeline columns out of the funded fields.
"""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import json

import pandas as pd

from engine.onboarding_agent import workflow as wf
from engine.onboarding_agent import central_tape_builder, storage_paths

PROFILE = "equity_release_lifetime_mortgage"


def _build_pack(root: Path) -> Path:
    inp = root / "input"
    funded = inp / "funded" / "2025-10-31"      # role/date subfolder (recursion)
    pipeline = inp / "pipeline" / "2025-12-01"
    funded.mkdir(parents=True)
    pipeline.mkdir(parents=True)
    pd.DataFrame({
        "Loan ID": ["L1", "L2"], "Loan Interest Rate": [3.10, 3.25],
        "Current Outstanding Balance": [100000, 120000],
        "Policy Completion Date": ["2018-05-01", "2019-06-01"],
        "Latest Property Value": [300000, 320000],
    }).to_csv(funded / "LoanExtract One.csv", index=False)
    pd.DataFrame({
        "Loan ID": ["L1", "L2"], "Interest Rate": [3.10, 3.25],
        "Total OSBalance": [100000, 120000],
        "Date Of Completion": ["2018-05-01", "2019-06-01"],
        "Latest Valuation": [300000, 320000],
    }).to_csv(funded / "PropertyExtract.csv", index=False)
    # Pipeline file: forward-exposure columns that must NOT fill funded fields.
    pd.DataFrame({
        "KFI Number": ["K1"], "product rate": [4.0], "loan amount": [90000],
        "application date": ["2024-01-01"], "date funds released": ["2024-03-01"],
        "property value": [280000],
    }).to_csv(pipeline / "Pipeline.csv", index=False)
    return inp


class TestFundedIngestion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.root = Path(tempfile.mkdtemp(prefix="funded_ingest_"))
        cls.inp = _build_pack(cls.root)
        cls.proj = cls.root / "proj"
        cls.summary = wf.run_operator_workflow(
            input_dir=str(cls.inp), client_name="T", client_id="t",
            run_id="mi_2025_10", mode="mi_only", project_dir=str(cls.proj),
            product_profile=PROFILE)

    def _csv(self, name):
        p = self.proj / name
        return list(csv.DictReader(open(p))) if p.exists() else []

    # -- ingestion: nested files are discovered and profiled ----------------- #
    def test_nested_files_are_profiled(self):
        cols = {r["source_column"] for r in self._csv("02_column_profiles.csv")}
        for c in ("Loan Interest Rate", "Current Outstanding Balance",
                  "Policy Completion Date", "Latest Property Value",
                  "Total OSBalance", "Latest Valuation", "Date Of Completion"):
            self.assertIn(c, cols, f"{c} missing from 02_column_profiles (ingestion)")

    def test_candidates_flow_to_05(self):
        by_field = {}
        for r in self._csv("05_mapping_candidates.csv"):
            by_field.setdefault(r["candidate_canonical_field"], []).append(r["source_column"])
        # In-scope funded fields reach 05 (current_valuation_amount is a regulatory
        # category field excluded from the mi_only funded-tape scope by design — it
        # still maps in the target-first review / 28a below).
        for f in ("current_interest_rate", "current_outstanding_balance",
                  "origination_date"):
            self.assertIn(f, by_field, f"{f} absent from 05_mapping_candidates")

    # -- the 4 base-MI blockers clear without manual approval ----------------- #
    def test_no_gate4_blockers(self):
        self.assertEqual(self.summary["blocking_decisions_count"], 0,
                         f"status={self.summary['status']}")

    def test_28a_selects_funded_sources(self):
        cov = {r["target_field"]: r for r in self._csv("28a_target_coverage_matrix.csv")}
        expect = {
            "current_interest_rate": "Loan Interest Rate",
            "current_outstanding_balance": "Current Outstanding Balance",
            "origination_date": "Policy Completion Date",
        }
        for field, col in expect.items():
            self.assertTrue(cov[field]["coverage_status"].startswith("source_mapped"),
                            f"{field}: {cov[field]['coverage_status']}")
            self.assertEqual(cov[field]["selected_source_column"], col)
        # valuation maps to a funded extract column (either one).
        self.assertTrue(
            cov["current_valuation_amount"]["coverage_status"].startswith("source_mapped"))
        self.assertIn(cov["current_valuation_amount"]["selected_source_column"],
                      ("Latest Property Value", "Latest Valuation"))

    def test_28d_records_principal_proxy_and_reporting_date(self):
        scope = json.loads((self.proj / "28d_product_profile_scope.json").read_text())
        proxied = {c["target_field"] for c in scope.get("proxy_derivations", [])}
        self.assertIn("current_principal_balance", proxied)
        self.assertIn("reporting_date", proxied)

    # -- artefact-role guardrail: pipeline never selected for funded fields --- #
    def test_role_guardrail_keeps_pipeline_out(self):
        cov = {r["target_field"]: r for r in self._csv("28a_target_coverage_matrix.csv")}
        for field in ("current_interest_rate", "origination_date"):
            sel = cov[field]["selected_source_file"]
            self.assertNotIn("Pipeline", sel, f"{field} wrongly sourced from pipeline")
            self.assertEqual(cov[field]["artefact_role_selected"], "funded")

    # -- the central tape carries the funded values (incl. funded balance) ---- #
    def test_central_tape_funded_values_non_null(self):
        rp = storage_paths.resolve_run_paths(
            project_dir=str(self.proj), input_dir=str(self.inp), output_root=None,
            client_id="t", run_id="mi_2025_10", storage_backend="local",
            input_uri="", output_uri="")
        tr = central_tape_builder.build_central_tapes(
            str(self.proj), rp, str(_REPO_ROOT / "config" / "system" / "fields_registry.yaml"),
            mode="mi_only")
        df = pd.read_csv(tr["central_lender_tape_path"])
        # The funded balance and other in-scope funded fields are non-null. (The
        # regulatory-category current_valuation_amount is excluded from the
        # mi_only funded tape by the existing field-scope design; it still clears
        # the Gate-4 blocker via 28a — see test_28a_selects_funded_sources.)
        for f in ("current_interest_rate", "current_outstanding_balance",
                  "origination_date"):
            self.assertIn(f, df.columns, f"{f} absent from central tape")
            self.assertEqual(int(df[f].notna().sum()), len(df), f"{f} has nulls in tape")


class TestPromotionConsumesResolvedMappings(unittest.TestCase):
    """Promotion / central tape must consume the resolved 28a selection (role
    guardrail + profile proxy/inference), not re-resolve from 05 — so funded
    fields are sourced from funded extracts and pipeline columns never populate
    them, and conflicts on resolved fields disappear."""

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.root = Path(tempfile.mkdtemp(prefix="promo_consume_"))
        cls.inp = _build_pack(cls.root)
        # Pipeline file with a conflicting funded-style rate column ("Product
        # Rate") — must NOT win for current_interest_rate.
        pl = cls.inp / "pipeline" / "2025-12-01"
        pd.DataFrame({"KFI Number": ["L1", "L2"], "Product Rate": [9.99, 9.99]}).to_csv(
            pl / "M2L KFI and Pipeline.csv", index=False)
        cls.proj = cls.root / "proj"
        cls.summary = wf.run_operator_workflow(
            input_dir=str(cls.inp), client_name="T", client_id="t",
            run_id="mi_2025_10", mode="mi_only", project_dir=str(cls.proj),
            product_profile=PROFILE)
        rp = storage_paths.resolve_run_paths(
            project_dir=str(cls.proj), input_dir=str(cls.inp), output_root=None,
            client_id="t", run_id="mi_2025_10", storage_backend="local",
            input_uri="", output_uri="")
        cls.tape_result = central_tape_builder.build_central_tapes(
            str(cls.proj), rp,
            str(_REPO_ROOT / "config" / "system" / "fields_registry.yaml"),
            mode="mi_only")
        cls.tape = pd.read_csv(cls.tape_result["central_lender_tape_path"])
        cls.lineage = pd.read_csv(cls.tape_result["central_tape_lineage_path"])

    def test_funded_balance_and_origination_non_null(self):
        for f in ("current_outstanding_balance", "origination_date"):
            self.assertIn(f, self.tape.columns)
            self.assertGreater(int(self.tape[f].notna().sum()), 0, f"{f} all null")

    def test_interest_rate_lineage_is_funded_not_pipeline(self):
        ir = self.lineage[self.lineage["canonical_field"] == "current_interest_rate"]
        self.assertTrue(len(ir) > 0, "no interest-rate lineage")
        files = set(ir["source_file"].astype(str))
        self.assertTrue(any("LoanExtract" in f for f in files), files)
        for f in files:
            self.assertNotIn("Pipeline", f, "interest rate sourced from pipeline")
            self.assertNotIn("KFI", f, "interest rate sourced from pipeline KFI")
        self.assertTrue(all(c == "Loan Interest Rate"
                            for c in ir["source_column"].astype(str)))

    def test_no_conflicts_on_resolved_fields(self):
        # Re-resolution previously produced funded/pipeline conflicts; consuming
        # the resolved selection removes them.
        self.assertEqual(int(self.tape_result.get("conflict_count", 0)), 0)

    def test_principal_and_reporting_materialised(self):
        for f in ("current_principal_balance", "reporting_date"):
            self.assertIn(f, self.tape.columns)
            self.assertGreater(int(self.tape[f].notna().sum()), 0, f"{f} all null")


class TestPromotionMaterialisesMultiSheetNumericIds(unittest.TestCase):
    """Regression for the source_mapped materialisation step: a funded source
    selected from a NON-first sheet of a multi-sheet workbook, with numeric loan
    ids (float vs int across sheets), must still populate row-by-row with lineage.
    """

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.root = Path(tempfile.mkdtemp(prefix="promo_multisheet_"))
        inp = cls.root / "input"
        fb = inp / "funded" / "2025-10-31"
        fb.mkdir(parents=True)
        with pd.ExcelWriter(fb / "M2L Portfolio.xlsx") as xl:
            pd.DataFrame({"info": ["cover sheet — not loan data"]}).to_excel(
                xl, sheet_name="Cover", index=False)
            pd.DataFrame({
                "Loan ID": [76034101, 76034102, 76034103],       # int ids
                "Loan Interest Rate": [3.10, 3.25, 3.40],
                "Current Outstanding Balance": [100000, 120000, 90000],
                "Policy Completion Date": ["2018-05-01", "2019-06-01", "2020-07-01"],
                "Latest Property Value": [300000, 320000, 280000],
            }).to_excel(xl, sheet_name="LoanExtract One", index=False)
            pd.DataFrame({
                "Loan ID": [76034101.0, 76034102.0, 76034103.0],  # float ids
                "Total OSBalance": [100000, 120000, 90000],
            }).to_excel(xl, sheet_name="PropertyExtract", index=False)
        cls.proj = cls.root / "proj"
        cls.summary = wf.run_operator_workflow(
            input_dir=str(inp), client_name="T", client_id="t",
            run_id="mi_2025_10", mode="mi_only", project_dir=str(cls.proj),
            product_profile=PROFILE)
        rp = storage_paths.resolve_run_paths(
            project_dir=str(cls.proj), input_dir=str(inp), output_root=None,
            client_id="t", run_id="mi_2025_10", storage_backend="local",
            input_uri="", output_uri="")
        cls.tr = central_tape_builder.build_central_tapes(
            str(cls.proj), rp,
            str(_REPO_ROOT / "config" / "system" / "fields_registry.yaml"),
            mode="mi_only")
        cls.tape = pd.read_csv(cls.tr["central_lender_tape_path"])
        cls.lineage = pd.read_csv(cls.tr["central_tape_lineage_path"])

    def test_all_loans_present(self):
        self.assertEqual(len(self.tape), 3)

    def test_source_mapped_fields_non_null(self):
        for f in ("current_interest_rate", "current_outstanding_balance",
                  "origination_date"):
            self.assertIn(f, self.tape.columns, f"{f} absent")
            self.assertEqual(int(self.tape[f].notna().sum()), len(self.tape),
                             f"{f} has nulls (materialisation failed)")

    def test_derived_and_inferred_non_null(self):
        for f in ("current_principal_balance", "reporting_date"):
            self.assertIn(f, self.tape.columns)
            self.assertEqual(int(self.tape[f].notna().sum()), len(self.tape), f)

    def test_lineage_exists_for_each_field(self):
        present = set(self.lineage["canonical_field"].astype(str))
        for f in ("current_interest_rate", "current_outstanding_balance",
                  "origination_date", "current_principal_balance", "reporting_date"):
            self.assertIn(f, present, f"no lineage for {f}")

    def test_interest_rate_from_funded_workbook(self):
        ir = self.lineage[self.lineage["canonical_field"] == "current_interest_rate"]
        self.assertTrue(all("M2L Portfolio" in str(f) for f in ir["source_file"]))
        for f in ir["source_file"].astype(str):
            self.assertNotIn("Pipeline", f)


if __name__ == "__main__":
    unittest.main(verbosity=2)
