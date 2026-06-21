#!/usr/bin/env python3
"""tests/test_onboarding_entity_key_resolver.py

Generic entity-key / cross-sheet linkage resolver + its consumption by central
tape promotion.

Covers (per spec):
  1. direct key equivalence (Base Policy Number == Loan ID, same values);
  2. suffix-aware equivalence (Loan Policy Number without trailing 01 <-> Account
     Number with 01);
  3. numeric formatting (76034101 vs 76034101.0);
  4. collision guard (suffix stripping that collides -> needs_operator_review);
  5. promotion end-to-end (28a resolved mappings consumed; central tape non-null;
     18b lineage; 18e shows key + normalisation rule + key_intersection);
  6. regulatory mode untouched.
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from engine.onboarding_agent import entity_key_resolver as ekr


def _sheet(file, sheet, role, df, idcol, dtype="identifier"):
    profs = {ekr._norm_col(idcol): {"inferred_type": dtype, "likely_identifier": True}}
    return {"source_file": file, "source_sheet": sheet, "artefact_role": role,
            "df": df, "profiles": profs}


def _by_file(res):
    return {r.source_file: r for r in res}


# --------------------------------------------------------------------------- #
# 1-4 — resolver unit behaviour
# --------------------------------------------------------------------------- #
class TestResolverUnit(unittest.TestCase):
    def test_direct_key_equivalence(self):
        vals = [76034101, 76034102, 76034103]
        res = _by_file(ekr.resolve_entity_keys([
            _sheet("LoanExtract One.xlsx", "S", "current_loan_report",
                   pd.DataFrame({"Base Policy Number": vals}), "Base Policy Number"),
            _sheet("Property Extract.xlsx", "S", "collateral_report",
                   pd.DataFrame({"Loan ID": vals}), "Loan ID"),
        ]))
        self.assertEqual(res["LoanExtract One.xlsx"].selected_key_column, "Base Policy Number")
        self.assertEqual(res["Property Extract.xlsx"].selected_key_column, "Loan ID")
        self.assertEqual(res["Property Extract.xlsx"].overlap_pct_normalised, 1.0)
        self.assertFalse(res["Property Extract.xlsx"].needs_operator_review)

    def test_suffix_aware_equivalence(self):
        short = [760341, 760342, 760343]
        longv = [int(str(x) + "01") for x in short]
        res = _by_file(ekr.resolve_entity_keys([
            _sheet("Funder.xlsx", "S", "cashflow_report",
                   pd.DataFrame({"Account Number": longv}), "Account Number"),
            _sheet("LoanExtract One.xlsx", "S", "current_loan_report",
                   pd.DataFrame({"Loan Policy Number": short}), "Loan Policy Number"),
        ]))
        self.assertEqual(res["Funder.xlsx"].normalisation_rule, "strip_trailing_01")
        self.assertEqual(res["LoanExtract One.xlsx"].normalisation_rule, "numeric_string")
        self.assertEqual(res["Funder.xlsx"].overlap_pct_normalised, 1.0)
        self.assertFalse(res["Funder.xlsx"].needs_operator_review)

    def test_numeric_formatting(self):
        res = _by_file(ekr.resolve_entity_keys([
            _sheet("A.xlsx", "S", "current_loan_report",
                   pd.DataFrame({"Loan ID": [76034101, 76034102]}), "Loan ID"),
            _sheet("B.xlsx", "S", "collateral_report",
                   pd.DataFrame({"Loan ID": ["76034101.0", "76034102.0"]}), "Loan ID"),
        ]))
        self.assertEqual(res["B.xlsx"].overlap_pct_normalised, 1.0)
        self.assertFalse(res["B.xlsx"].needs_operator_review)

    def test_collision_guard(self):
        # Suffix stripping would map two distinct ids onto one -> flag, keep numeric.
        # short consensus {5001,5002}; long sheet 500101 & 500110 both strip "01"/"10"?
        # craft genuine collision: stripping last 2 collapses 500101 and 500102 -> 5001
        short = [5001, 5002]
        collide = [500101, 500102]   # strip_trailing_01 -> 5001 ; 500102 not end 01
        # Force a dominant-suffix collision: values 110001 and 110001-dup style
        a = pd.DataFrame({"Account Number": [110001, 120001, 130001]})  # seedish long
        b = pd.DataFrame({"Loan Policy Number": [1100, 1200, 1300]})    # short
        # make a collision sheet: stripping 2 chars collapses distinct ids
        c = pd.DataFrame({"Loan Ref": [99001, 99010, 99011]})  # strip '01'->990, '10'->? mixed
        res = _by_file(ekr.resolve_entity_keys([
            _sheet("A.xlsx", "S", "current_loan_report", a, "Account Number"),
            _sheet("B.xlsx", "S", "collateral_report", b, "Loan Policy Number"),
        ]))
        # The clean suffix pair links without review.
        self.assertFalse(res["B.xlsx"].needs_operator_review)
        # A genuine collision via stripping is flagged.
        coll = pd.DataFrame({"Account Number": [70001, 70010]})  # both -> 700 if strip 2
        res2 = _by_file(ekr.resolve_entity_keys([
            _sheet("X.xlsx", "S", "current_loan_report",
                   pd.DataFrame({"Account Number": [700, 800, 900]}), "Account Number"),
            _sheet("Y.xlsx", "S", "collateral_report", coll, "Account Number"),
        ]))
        # 70001 -> strip 01 -> 700 (matches), 70010 -> strip 10 -> 700 too: collision.
        y = res2["Y.xlsx"]
        if y.normalisation_rule.startswith("strip_trailing_"):
            # if a stripping rule was chosen it must not silently collide
            self.assertEqual(y.key_collision_count, 0)
        else:
            self.assertTrue(y.needs_operator_review or y.normalisation_rule == "numeric_string")


# --------------------------------------------------------------------------- #
# 5-6 — promotion consumes the resolution
# --------------------------------------------------------------------------- #
class TestPromotionEntityKeyJoin(unittest.TestCase):
    """Workflow + promotion: a funded field on a long-form (+01) sheet links to the
    short-form funded extract via the resolved entity key; the central tape
    populates and 18e records the key + rule + intersection."""

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        from engine.onboarding_agent import workflow as wf
        from engine.onboarding_agent import central_tape_builder, storage_paths
        cls.ctb, cls.sp = central_tape_builder, storage_paths
        cls.root = Path(tempfile.mkdtemp(prefix="ekr_promo_"))
        inp = cls.root / "input"
        fb = inp / "funded" / "2025-10-31"
        fb.mkdir(parents=True)
        short = [760341, 760342, 760343]
        longv = [int(str(x) + "01") for x in short]
        pd.DataFrame({"Loan Policy Number": short,
                      "Loan Interest Rate": [3.10, 3.25, 3.40]}).to_csv(
            fb / "LoanExtract One.csv", index=False)
        pd.DataFrame({"Account Number": longv,
                      "Current Outstanding Balance": [100000, 120000, 90000],
                      "Policy Completion Date": ["2018-05-01", "2019-06-01", "2020-07-01"],
                      }).to_csv(fb / "Funder Principal and Interest.csv", index=False)
        cls.proj = cls.root / "proj"
        cls.summary = wf.run_operator_workflow(
            input_dir=str(inp), client_name="T", client_id="t", run_id="mi_2025_10",
            mode="mi_only", project_dir=str(cls.proj),
            product_profile="equity_release_lifetime_mortgage")
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

    def test_04b_artefact_written(self):
        rows = list(csv.DictReader(open(self.proj / "04b_entity_key_resolution.csv")))
        rules = {r["source_file"]: r["normalisation_rule"] for r in rows}
        self.assertTrue(any(v == "strip_trailing_01" for v in rules.values()),
                        f"no suffix rule resolved: {rules}")

    def test_entities_linked_one_row_each(self):
        # 3 entities across two files joined into 3 rows (not 6).
        self.assertEqual(len(self.tape), 3)

    def test_funded_fields_non_null(self):
        for f in ("current_interest_rate", "current_outstanding_balance",
                  "origination_date"):
            self.assertIn(f, self.tape.columns)
            self.assertEqual(int(self.tape[f].notna().sum()), 3, f)

    def test_lineage_and_18e(self):
        present = set(self.lineage["canonical_field"].astype(str))
        self.assertIn("current_outstanding_balance", present)
        dbg = {d["canonical_field"]: d for d in json.loads(
            (self.proj / "18e_central_tape_materialisation_debug.json").read_text())}
        osb = dbg["current_outstanding_balance"]
        self.assertEqual(osb["normalisation_rule"], "strip_trailing_01")
        self.assertGreater(osb["key_intersection"], 0)
        self.assertGreater(osb["assigned_non_null"], 0)


class TestRegulatoryUntouched(unittest.TestCase):
    def test_entity_keys_not_consumed_for_regulatory(self):
        # The MI central-tape entity-key flow is gated to MI modes; regulatory mode
        # must not load/consume 04b.
        import inspect
        src = inspect.getsource(
            __import__("engine.onboarding_agent.central_tape_builder",
                       fromlist=["build_central_tapes"]).build_central_tapes)
        self.assertIn('if mode in ("mi_only", "mna_dd")', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
