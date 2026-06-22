#!/usr/bin/env python3
"""mi_agent_api/tests/test_funded_period_enrichment.py

Regression cover for the November-LTV spine bug (stages 1-7).

A collateral / PropertyExtract delivered for October is the latest-available
valuation for loans still funded in November, so it must ENRICH the November
funded book — not be dropped as a ``period_mismatch`` under the strict
funded-book cadence. Before the fix:

  * mi_2025_10 -> valuation 33/33, current_loan_to_value 33/33;
  * mi_2025_11 -> valuation 0/73,  current_loan_to_value 0/73   (the bug).

These tests mirror the real pack: period-specific folders
(``funding/2025-10`` + ``funding/2025-11``, ``pipeline/2025-12-01``), a
PropertyExtract whose real header sits in row 2, and an Account-Number key that
is the loan id with a trailing ``01`` (entity-key drift). They assert the funded
universe (33 / 73) is unchanged, no pipeline row is promoted, November valuation
+ LTV now populate, and the run-level spine diagnostic explains the path.
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

from engine.onboarding_agent import source_period_eligibility as spe

_REGISTRY = str(_REPO_ROOT / "config" / "system" / "fields_registry.yaml")
_OCT_N, _OCT_EACH = 33, 127515.15      # -> 4,207,999.95  (~£4.208MM)
_NOV_N, _NOV_EACH = 73, 121958.90      # -> 8,902,999.70  (~£8.903MM)


def _make_period_pack(root: Path, property_extract_name: str) -> Path:
    """A pack that mirrors the real layout: cumulative LoanExtract delivered in
    ``funding/2025-11``; an October-only PropertyExtract (row-2 header) in
    ``funding/2025-10`` covering every loan via an Account-Number = loanid*100+1
    key; a future pipeline file in ``pipeline/2025-12-01``."""
    inp = root / "input"
    fund10 = inp / "funding" / "2025-10"
    fund11 = inp / "funding" / "2025-11"
    pipe = inp / "pipeline" / "2025-12-01"
    for d in (fund10, fund11, pipe):
        d.mkdir(parents=True)

    ids = [760000 + i for i in range(_NOV_N)]
    acct = [s * 100 + 1 for s in ids]   # trailing-01 key variant (entity drift)
    oct_rows, nov_rows = [], []
    for i, lid in enumerate(ids):
        nov_rows.append({"Loan Policy Number": lid, "Month Run": "November",
                         "Loan Interest Rate": 3.10 + (i % 5) * 0.05,
                         "Current Outstanding Balance": _NOV_EACH,
                         "Policy Completion Date": "2025-06-01",
                         "Customer 1 DOB": "1950-03-01"})
        if i < _OCT_N:
            oct_rows.append({"Loan Policy Number": lid, "Month Run": "October",
                             "Loan Interest Rate": 3.10 + (i % 5) * 0.05,
                             "Current Outstanding Balance": _OCT_EACH,
                             "Policy Completion Date": "2025-06-01",
                             "Customer 1 DOB": "1950-03-01"})
    # Cumulative current-book file (row-filtered by Month Run to 33 / 73).
    pd.DataFrame(oct_rows + nov_rows).to_csv(fund11 / "LoanExtract One.csv", index=False)

    # PropertyExtract: title banner on row 1, real header on row 2 (messy extract).
    pe = fund10 / property_extract_name
    lines = ["PG PropertyExtract Internal OMNI,,,,",
             "Account Number,Latest Property Value,Original Property Value,Broker,Original Loan Amount"]
    for i, a in enumerate(acct):
        lines.append(f"{a},{250000.0 + i},{240000.0 + i},BrokerX,{100000.0 + i}")
    pe.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Future pipeline snapshot — must NEVER create funded rows.
    pd.DataFrame({"application_id": [f"APP{i}" for i in range(20)],
                  "Account Number": [990000 + i for i in range(20)],
                  "product rate": [4.0] * 20}).to_csv(
        pipe / "M2L KFI and Pipeline 2025_12_01.csv", index=False)
    return inp


def _promote(root: Path, inp: Path, run_id: str):
    from engine.onboarding_agent import workflow as wf, storage_paths, central_tape_builder
    proj = root / f"proj_{run_id}"
    wf.run_operator_workflow(
        input_dir=str(inp), client_name="Client 001", client_id="client_001",
        run_id=run_id, mode="mi_only", project_dir=str(proj),
        product_profile="equity_release_lifetime_mortgage")
    rp = storage_paths.resolve_run_paths(
        project_dir=str(proj), input_dir=str(inp), output_root=None,
        client_id="client_001", run_id=run_id, storage_backend="local",
        input_uri="", output_uri="")
    res = central_tape_builder.build_central_tapes(str(proj), rp, _REGISTRY, mode="mi_only")
    return Path(res["central_lender_tape_path"]), Path(proj)


class TestPeriodEnrichmentCadence(unittest.TestCase):
    """Unit cover for the enrichment vs funded-book period cadences."""

    def test_enrichment_on_or_before_run_period_is_eligible(self):
        prof = {"present": [], "period": "2025-10", "delivery_date": ""}
        ok, reason = spe._enrichment_period_match(prof, "2025-11", True)
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_enrichment_future_period_excluded(self):
        prof = {"present": [], "period": "2025-12", "delivery_date": ""}
        ok, reason = spe._enrichment_period_match(prof, "2025-11", True)
        self.assertFalse(ok)
        self.assertEqual(reason, "future_period")

    def test_universe_earlier_period_still_strict(self):
        # The funded/current-book cadence is UNCHANGED: an October universe file
        # is not eligible for the November run (it would change the funded count).
        prof = {"present": [], "period": "2025-10", "delivery_date": ""}
        ok, reason = spe._funded_period_match(prof, "2025-11", True)
        self.assertFalse(ok)
        self.assertEqual(reason, "period_mismatch")

    def test_collateral_role_eligible_for_later_run_via_compute(self):
        rec = {"file_name": "PG_PropertyExtract Internal OMNI_test.csv",
               "file_path": "input/funding/2025-10/PG_PropertyExtract Internal OMNI_test.csv",
               "sheet_name": "", "artefact_role": "collateral_report",
               "detected_reporting_date": "", "df": None}
        rows = spe.compute_eligibility([rec], "mi_2025_11")
        tape_row = next(r for r in rows if r.output_domain == "central_lender_tape")
        self.assertTrue(tape_row.is_period_eligible)
        self.assertFalse(tape_row.is_universe_source)  # enrichment never seeds rows


class TestHeaderDriftReadShared(unittest.TestCase):
    """Both real PropertyExtract filenames re-detect their row-2 header through the
    SAME shared loader the source profiler now uses."""

    def _check(self, name: str):
        from engine.onboarding_agent.central_tape_builder import _read_df
        d = Path(tempfile.mkdtemp(prefix="hdr_"))
        f = d / name
        f.write_text(
            "PG PropertyExtract Internal OMNI,,,,\n"
            "Account Number,Latest Property Value,Original Property Value,Broker,Original Loan Amount\n"
            "76034101,500000,450000,BrokerX,100000\n"
            "76034201,300000,280000,BrokerY,90000\n", encoding="utf-8")
        df = _read_df(str(f))
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)
        for col in ("Account Number", "Latest Property Value", "Broker"):
            self.assertIn(col, df.columns)
        self.assertFalse(any(str(c).startswith("Unnamed") for c in df.columns))

    def test_pg_propertyextract_internal_omni(self):
        self._check("PG_PropertyExtract Internal OMNI_test.csv")

    def test_propertyextract_omni(self):
        self._check("PropertyExtract - Omni_test.csv")


class TestNovemberLtvEnrichmentEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.root = Path(tempfile.mkdtemp(prefix="period_enrich_"))
        inp = _make_period_pack(cls.root, "PG_PropertyExtract Internal OMNI_test.csv")
        cls.oct_tape, cls.oct_proj = _promote(cls.root, inp, "mi_2025_10")
        cls.nov_tape, cls.nov_proj = _promote(cls.root, inp, "mi_2025_11")

    # --- universe counts unchanged (acceptance) ---
    def test_october_universe(self):
        df = pd.read_csv(self.oct_tape)
        self.assertEqual(len(df), 33)
        self.assertAlmostEqual(
            pd.to_numeric(df["current_outstanding_balance"]).sum(), 4_208_000, delta=2_000)
        self.assertEqual(int(df["current_valuation_amount"].notna().sum()), 33)

    def test_november_universe_and_valuation_fixed(self):
        df = pd.read_csv(self.nov_tape)
        self.assertEqual(len(df), 73)
        self.assertAlmostEqual(
            pd.to_numeric(df["current_outstanding_balance"]).sum(), 8_903_000, delta=2_000)
        # The regression: November valuation must populate from the October
        # PropertyExtract (latest-available enrichment), not be dropped.
        self.assertEqual(int(df["current_valuation_amount"].notna().sum()), 73)
        self.assertEqual(int(df["original_valuation_amount"].notna().sum()), 73)

    def test_no_pipeline_rows_promoted(self):
        for tape in (self.oct_tape, self.nov_tape):
            ids = set(pd.read_csv(tape)["loan_identifier"].astype(str))
            self.assertFalse(any(i.startswith("9900") for i in ids))

    # --- downstream MI prep: November LTV + ltv_bucket now populate ---
    def test_november_ltv_and_buckets_derivable(self):
        from mi_agent_api.funded_prep import prepare_funded_mi_dataset
        prep, report = prepare_funded_mi_dataset(pd.read_csv(self.nov_tape))
        self.assertEqual(int(prep["current_loan_to_value"].notna().sum()), 73)
        self.assertEqual(int(prep["ltv_bucket"].notna().sum()), 73)
        self.assertIn("ltv_bucket", report["dimensions_available"])
        self.assertIn("original_ltv_bucket", report["dimensions_available"])

    # --- entity-key drift: trailing-01 / float variants join the funded book ---
    def test_entity_key_variants_join(self):
        # Loan ids 760000+i join collateral Account Number = id*100+1 — every
        # funded row resolved a valuation, proving the key variants matched.
        df = pd.read_csv(self.nov_tape)
        self.assertEqual(int(df["current_valuation_amount"].notna().sum()), len(df))

    # --- run-level spine diagnostic (acceptance: explains the path) ---
    def test_spine_diagnostic_written_and_explains(self):
        rep_path = next(self.nov_proj.rglob("spine_stage_1_7_report.json"))
        self.assertTrue(str(rep_path).endswith(
            "mi_2025_11/output/diagnostics/spine_stage_1_7_report.json"))
        rep = json.loads(rep_path.read_text())
        self.assertEqual(rep["universe"]["loan_count"], 73)
        # The October PropertyExtract is no longer excluded for the November run.
        excluded = {e.get("source_file") for e in rep["universe"]["excluded_sources"]}
        self.assertNotIn("PG_PropertyExtract Internal OMNI_test.csv", excluded)
        val = next(f for f in rep["fields"]
                   if f["canonical_field"] == "current_valuation_amount")
        self.assertEqual(val["promoted_non_null"], 73)
        self.assertEqual(val["key_overlap_with_funded_universe"], 73)
        self.assertTrue(any(p["period_eligible"] for p in val["period_eligibility"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
