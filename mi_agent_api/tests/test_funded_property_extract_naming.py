#!/usr/bin/env python3
"""mi_agent_api/tests/test_funded_property_extract_naming.py

Regression for the November valuation/LTV enrichment gap caused by a later-month
collateral extract being delivered under a DIFFERENT filename.

Pack (mirrors the real client_001_mi_pack):
  funding/2025-10/PG_PropertyExtract Internal OMNI_test.csv  (row-2 header, 33 loans)
  funding/2025-11/PropertyExtract - Omni_test.csv            (row-2 header, 73 loans)
  funding/2025-11/LoanExtract One - OMNI_test.csv            (cumulative 33->73)
  pipeline/2025-12-01/M2L KFI and Pipeline 2025_12_01.csv

Before the fix the November extract was discovered + classified collateral, but
its valuation columns never became multi-file enrichment candidates, so only the
single 28a-selected October source enriched (33/73). The fix:
  * the onboarding reader shares redetect_header (row-2 columns map), and
  * the central tape unions same-role period-eligible collateral siblings for a
    forced enrichment field.
so November valuation/LTV is 73/73 and BOTH extracts appear in the diagnostic.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from engine.onboarding_agent.file_classifier import classify_directory

_REGISTRY = str(_REPO_ROOT / "config" / "system" / "fields_registry.yaml")
_OCT_N, _OCT_EACH = 33, 127515.15
_NOV_N, _NOV_EACH = 73, 121958.90
_OCT_PE = "PG_PropertyExtract Internal OMNI_test.csv"
_NOV_PE = "PropertyExtract - Omni_test.csv"


def _pe_text(acct, idxs):
    """A row-2-header PropertyExtract covering the given loan indices."""
    lines = ["PG PropertyExtract Internal OMNI,,,,",
             "Account Number,Latest Property Value,Original Property Value,Broker,Original Loan Amount"]
    for i in idxs:
        bal = _OCT_EACH if i < _OCT_N else _NOV_EACH
        v = bal / (0.29 + (i % 28) * 0.01)
        lines.append(f"{acct[i]},{v:.0f},{v * 1.05:.0f},BrokerX,{100000.0 + i}")
    return "\n".join(lines) + "\n"


def _make_pack(root: Path) -> Path:
    inp = root / "input"
    f10 = inp / "funding" / "2025-10"
    f11 = inp / "funding" / "2025-11"
    pp = inp / "pipeline" / "2025-12-01"
    for d in (f10, f11, pp):
        d.mkdir(parents=True)
    ids = [760000 + i for i in range(_NOV_N)]
    acct = [s * 100 + 1 for s in ids]
    oct_rows, nov_rows = [], []
    for i, lid in enumerate(ids):
        nov_rows.append({"Loan Policy Number": lid, "Month Run": "November",
                         "Loan Interest Rate": 3.10, "Current Outstanding Balance": _NOV_EACH,
                         "Policy Completion Date": "2018-06-01", "Customer 1 DOB": "1950-03-01"})
        if i < _OCT_N:
            oct_rows.append({"Loan Policy Number": lid, "Month Run": "October",
                             "Loan Interest Rate": 3.10, "Current Outstanding Balance": _OCT_EACH,
                             "Policy Completion Date": "2018-06-01", "Customer 1 DOB": "1950-03-01"})
    pd.DataFrame(oct_rows + nov_rows).to_csv(f11 / "LoanExtract One - OMNI_test.csv", index=False)
    # October extract covers the 33 October loans; November extract covers all 73.
    (f10 / _OCT_PE).write_text(_pe_text(acct, range(_OCT_N)), encoding="utf-8")
    (f11 / _NOV_PE).write_text(_pe_text(acct, range(_NOV_N)), encoding="utf-8")
    pd.DataFrame({"application_id": [f"APP{i}" for i in range(20)],
                  "Account Number": [990000 + i for i in range(20)],
                  "product rate": [4.0] * 20}).to_csv(
        pp / "M2L KFI and Pipeline 2025_12_01.csv", index=False)
    return inp


def _promote(root: Path, inp: Path, run_id: str):
    from engine.onboarding_agent import workflow as wf, storage_paths, central_tape_builder
    proj = root / f"proj_{run_id}"
    wf.run_operator_workflow(
        input_dir=str(inp), client_name="C", client_id="client_001",
        run_id=run_id, mode="mi_only", project_dir=str(proj),
        product_profile="equity_release_lifetime_mortgage")
    rp = storage_paths.resolve_run_paths(
        project_dir=str(proj), input_dir=str(inp), output_root=None,
        client_id="client_001", run_id=run_id, storage_backend="local",
        input_uri="", output_uri="")
    res = central_tape_builder.build_central_tapes(str(proj), rp, _REGISTRY, mode="mi_only")
    return Path(res["central_lender_tape_path"]), Path(proj)


class TestNovemberPropertyExtractNaming(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.root = Path(tempfile.mkdtemp(prefix="pe_naming_"))
        cls.inp = _make_pack(cls.root)
        cls.oct_tape, cls.oct_proj = _promote(cls.root, cls.inp, "mi_2025_10")
        cls.nov_tape, cls.nov_proj = _promote(cls.root, cls.inp, "mi_2025_11")

    # Q1/Q2: the differently-named November extract is discovered + classified.
    def test_november_extract_discovered_and_classified_collateral(self):
        by = {it.file_name: it for it in classify_directory(self.inp)}
        self.assertIn(_NOV_PE, by)
        self.assertEqual(by[_NOV_PE].classification, "collateral_report")

    # October snapshot unchanged.
    def test_october_unbroken(self):
        df = pd.read_csv(self.oct_tape)
        self.assertEqual(len(df), 33)
        self.assertEqual(int(df["current_valuation_amount"].notna().sum()), 33)

    # November valuation now 73/73 from the unioned same-role extracts.
    def test_november_valuation_full(self):
        df = pd.read_csv(self.nov_tape)
        self.assertEqual(len(df), 73)
        self.assertEqual(int(df["current_valuation_amount"].notna().sum()), 73)
        self.assertEqual(int(df["original_valuation_amount"].notna().sum()), 73)
        self.assertFalse(any(str(i).startswith("9900") for i in df["loan_identifier"]))

    # Downstream prep: LTV / buckets / age all 73/73 for November.
    def test_november_prep_ltv_and_age(self):
        from mi_agent_api.funded_prep import prepare_funded_mi_dataset
        prep, rep = prepare_funded_mi_dataset(pd.read_csv(self.nov_tape))
        self.assertEqual(int(prep["current_loan_to_value"].notna().sum()), 73)
        self.assertEqual(int(prep["original_loan_to_value"].notna().sum()), 73)
        self.assertEqual(int(prep["youngest_borrower_age"].notna().sum()), 73)
        self.assertIn("ltv_bucket", rep["dimensions_available"])
        self.assertIn("age_bucket", rep["dimensions_available"])

    # The diagnostic now lists BOTH extracts; the November one is eligible 73/73.
    def test_diagnostic_lists_november_extract(self):
        rep = json.loads(next(self.nov_proj.rglob("spine_stage_1_7_report.json")).read_text())
        val = next(f for f in rep["fields"] if f["canonical_field"] == "current_valuation_amount")
        candidate_files = {c["source_file"] for c in val["raw_source_candidates"]}
        self.assertIn(_NOV_PE, candidate_files)
        self.assertEqual(val["promoted_non_null"], 73)
        nov = next((p for p in val["period_eligibility"] if p["source_file"] == _NOV_PE), None)
        self.assertIsNotNone(nov)
        self.assertTrue(nov["period_eligible"])
        self.assertEqual(nov["key_overlap_with_funded_universe"], 73)


if __name__ == "__main__":
    unittest.main(verbosity=2)
