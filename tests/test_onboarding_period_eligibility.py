#!/usr/bin/env python3
"""tests/test_onboarding_period_eligibility.py

Domain-aware reporting-period eligibility + period-scoped central lender tape
universe (without breaking pipeline reporting).

Unit:
  * run-period inference; period_of_value;
  * output-domain rows (pipeline file excluded from the tape but eligible for
    pipeline_mi / forward_exposure under the snapshot cadence);
  * filename delivery offset; future-period tape exclusion; cumulative
    row-filtering; warehouse_agreement is enrichment (not universe);
  * data_cut_off_date period_label_to_month_end normalisation.

End-to-end (per spec acceptance tests A-F):
  A funded period filtering: mi_2025_10 -> 33 / c.£4.2MM, mi_2025_11 -> 73 / c.£8.9MM;
  B pipeline cadence separation: pipeline files create no tape rows but stay
    available for pipeline_mi;
  C entity-key canonicalisation: short/long-form keys collapse to one funded row;
  D duplicate/stale avoidance: a `_test` duplicate is not the universe source;
  E date cut-off: Month Run October -> 2025-10-31, November -> 2025-11-30, no bare
    month names in promoted data_cut_off_date;
  F regulatory mode universe gate untouched.
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

from engine.onboarding_agent import source_period_eligibility as spe
from engine.onboarding_agent import central_tape_builder as ctb

_REGISTRY = str(_REPO_ROOT / "config" / "system" / "fields_registry.yaml")


def _domain_row(rows, domain):
    return next((r for r in rows if r.output_domain == domain), None)


# --------------------------------------------------------------------------- #
# Unit behaviour
# --------------------------------------------------------------------------- #
class TestPeriodUnit(unittest.TestCase):
    def test_run_period_from_run_id(self):
        self.assertEqual(spe.run_period("mi_2025_10"), ("2025-10", "2025-10-31"))
        self.assertEqual(spe.run_period("mi_2025_11"), ("2025-11", "2025-11-30"))

    def test_period_of_value(self):
        self.assertEqual(spe.period_of_value("October", 2025), "2025-10")
        self.assertEqual(spe.period_of_value("Nov", 2025), "2025-11")
        self.assertEqual(spe.period_of_value("2025-11-30", None), "2025-11")
        self.assertEqual(spe.period_of_value("31/10/2025", None), "2025-10")

    def test_warehouse_agreement_is_enrichment_not_universe(self):
        df = pd.DataFrame({"Account Number": [1, 2]})
        recs = [{"file_name": "funder.csv", "file_path": "/x/funder.csv", "sheet_name": "",
                 "artefact_role": "warehouse_agreement", "detected_reporting_date": "", "df": df}]
        r = _domain_row(spe.compute_eligibility(recs, "mi_2025_10"), "central_lender_tape")
        self.assertTrue(r.is_period_eligible)        # eligible to enrich
        self.assertFalse(r.is_universe_source)       # but not a universe source

    def test_pipeline_excluded_from_tape_but_eligible_for_pipeline_mi(self):
        recs = [{"file_name": "M2L KFI and Pipeline 2025_12_01.csv",
                 "file_path": "/x/M2L KFI and Pipeline 2025_12_01.csv", "sheet_name": "",
                 "artefact_role": "pipeline_report", "detected_reporting_date": "", "df": None}]
        rows = spe.compute_eligibility(recs, "mi_2025_10")
        tape = _domain_row(rows, "central_lender_tape")
        pmi = _domain_row(rows, "pipeline_mi")
        fwd = _domain_row(rows, "forward_exposure")
        self.assertFalse(tape.is_period_eligible)
        self.assertEqual(tape.reason_excluded, "pipeline_role_excluded_from_lender_tape")
        self.assertTrue(pmi.is_period_eligible)            # later delivery still valid
        self.assertEqual(pmi.cadence_rule, "pipeline_snapshot")
        self.assertTrue(fwd.is_period_eligible)

    def test_filename_delivery_offset_pipeline(self):
        cfg = dict(spe._DEFAULTS, filename_delivery_offset_months=-1)
        recs = [{"file_name": "M2L KFI and Pipeline 2025_11_01_113916.xlsx",
                 "file_path": "/x/M2L KFI and Pipeline 2025_11_01_113916.xlsx", "sheet_name": "",
                 "artefact_role": "pipeline_report", "detected_reporting_date": "", "df": None}]
        r = _domain_row(spe.compute_eligibility(recs, "mi_2025_10", config=cfg), "pipeline_mi")
        self.assertEqual(r.inferred_reporting_period, "2025-10")  # delivery -1 month
        self.assertEqual(r.delivery_date, "2025-11-01")           # raw delivery preserved

    def test_future_period_excluded_for_tape(self):
        df = pd.DataFrame({"Loan ID": [1, 2], "Month Run": ["November", "November"]})
        recs = [{"file_name": "nov.csv", "file_path": "/x/nov.csv", "sheet_name": "",
                 "artefact_role": "current_loan_report", "detected_reporting_date": "", "df": df}]
        r = _domain_row(spe.compute_eligibility(recs, "mi_2025_10"), "central_lender_tape")
        self.assertFalse(r.is_period_eligible)
        self.assertEqual(r.reason_excluded, "future_period")

    def test_cumulative_file_row_filterable(self):
        df = pd.DataFrame({"Loan ID": [1, 2, 3],
                           "Month Run": ["October", "November", "November"]})
        recs = [{"file_name": "book.csv", "file_path": "/x/book.csv", "sheet_name": "",
                 "artefact_role": "current_loan_report", "detected_reporting_date": "", "df": df}]
        r = _domain_row(spe.compute_eligibility(recs, "mi_2025_10"), "central_lender_tape")
        self.assertTrue(r.is_period_eligible and r.is_universe_source)
        self.assertEqual(r.source_period_column, "Month Run")
        self.assertEqual(r.source_period_raw_value, "October")

    def test_cutoff_label_to_month_end(self):
        self.assertEqual(ctb._canonicalise_period_cutoff("October", 2025),
                         ("2025-10-31", "period_label_to_month_end", "source_period_column+run_year"))
        self.assertEqual(ctb._canonicalise_period_cutoff("November", 2025)[0], "2025-11-30")
        # explicit date -> date normalised, not forced to month end
        self.assertEqual(ctb._canonicalise_period_cutoff("2025-10-15", 2025),
                         ("2025-10-15", "date_normalised", "source_explicit_date"))

    def test_04c_columns_present(self):
        recs = [{"file_name": "book.csv", "file_path": "/x/book.csv", "sheet_name": "",
                 "artefact_role": "current_loan_report", "detected_reporting_date": "",
                 "df": pd.DataFrame({"Loan ID": [1], "Month Run": ["October"]})}]
        d = spe.compute_eligibility(recs, "mi_2025_10")[0].as_dict()
        for col in ("output_domain", "delivery_date", "cadence_rule", "source_period_column",
                    "source_period_raw_value", "source_period_canonical_value"):
            self.assertIn(col, d)


# --------------------------------------------------------------------------- #
# Promotion helpers
# --------------------------------------------------------------------------- #
def _run_promotion(root: Path, inp: Path, run_id: str, profile="equity_release_lifetime_mortgage"):
    from engine.onboarding_agent import workflow as wf
    from engine.onboarding_agent import storage_paths
    proj = root / f"proj_{run_id}"
    wf.run_operator_workflow(
        input_dir=str(inp), client_name="T", client_id="t", run_id=run_id,
        mode="mi_only", project_dir=str(proj), product_profile=profile)
    rp = storage_paths.resolve_run_paths(
        project_dir=str(proj), input_dir=str(inp), output_root=None,
        client_id="t", run_id=run_id, storage_backend="local", input_uri="", output_uri="")
    tr = ctb.build_central_tapes(str(proj), rp, _REGISTRY, mode="mi_only")
    return proj, tr, pd.read_csv(tr["central_lender_tape_path"], dtype=str)


def _sum_balance(tape) -> float:
    s = tape["current_outstanding_balance"].astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce").dropna().sum()


# A + B + E ----------------------------------------------------------------- #
class TestFundedPeriodAndPipelineCadence(unittest.TestCase):
    OCT_N, NOV_N = 33, 73
    OCT_BAL, NOV_BAL = 4_200_000.0, 8_900_000.0

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.root = Path(tempfile.mkdtemp(prefix="period_promo_"))
        inp = cls.root / "input"
        inp.mkdir(parents=True)
        ids = [760000 + i for i in range(cls.NOV_N)]
        oct_each = round(cls.OCT_BAL / cls.OCT_N, 2)
        nov_each = round(cls.NOV_BAL / cls.NOV_N, 2)
        oct_rows, nov_rows = [], []
        for i, lid in enumerate(ids):
            nov_rows.append({"Loan Policy Number": lid, "Month Run": "November",
                             "Loan Interest Rate": 3.10 + (i % 5) * 0.05,
                             "Current Outstanding Balance": nov_each,
                             "Policy Completion Date": "2025-11-15"})
            if i < cls.OCT_N:
                oct_rows.append({"Loan Policy Number": lid, "Month Run": "October",
                                 "Loan Interest Rate": 3.10 + (i % 5) * 0.05,
                                 "Current Outstanding Balance": oct_each,
                                 "Policy Completion Date": "2025-10-15"})
        pd.DataFrame(oct_rows + nov_rows).to_csv(inp / "LoanExtract One.csv", index=False)
        # Pipeline files delivered AFTER each close (different cadence).
        for tag in ("2025_11_01", "2025_12_01"):
            pd.DataFrame({"application_id": [f"APP{tag}{i}" for i in range(20)],
                          "Account Number": [990000 + i for i in range(20)],
                          "product rate": [4.0] * 20}).to_csv(
                inp / f"M2L KFI and Pipeline {tag}.csv", index=False)
        cls.oct_proj, cls.oct_tr, cls.oct_tape = _run_promotion(cls.root, inp, "mi_2025_10")
        cls.nov_proj, cls.nov_tr, cls.nov_tape = _run_promotion(cls.root, inp, "mi_2025_11")

    def test_A_october_universe(self):
        self.assertEqual(len(self.oct_tape), self.OCT_N)
        self.assertAlmostEqual(_sum_balance(self.oct_tape), self.OCT_BAL, delta=1.0)

    def test_A_november_universe(self):
        self.assertEqual(len(self.nov_tape), self.NOV_N)
        self.assertAlmostEqual(_sum_balance(self.nov_tape), self.NOV_BAL, delta=1.0)

    def test_A_funded_fields_full_universe(self):
        for tape, n in ((self.oct_tape, self.OCT_N), (self.nov_tape, self.NOV_N)):
            for f in ("current_interest_rate", "current_outstanding_balance"):
                self.assertEqual(int(tape[f].notna().sum()), n, f)

    def test_B_pipeline_creates_no_tape_rows(self):
        self.assertNotIn("990000", set(self.oct_tape["loan_identifier"].astype(str)))
        self.assertEqual(len(self.oct_tape), self.OCT_N)

    def test_B_pipeline_available_for_pipeline_mi(self):
        elig = spe.load_eligibility(self.oct_proj, "pipeline_mi")
        eligible_files = {f for (f, _s), r in elig.items() if r.get("is_period_eligible")}
        self.assertTrue(any("M2L" in f for f in eligible_files))
        dbg = json.loads((self.oct_proj / "18f_central_universe_debug.json").read_text())
        self.assertTrue(dbg["pipeline_sources_available_for_pipeline_mi"])

    def test_E_data_cut_off_date_month_end(self):
        self.assertEqual(set(self.oct_tape["data_cut_off_date"]), {"2025-10-31"})
        self.assertEqual(set(self.nov_tape["data_cut_off_date"]), {"2025-11-30"})
        # No bare month names anywhere in the promoted column.
        for tape in (self.oct_tape, self.nov_tape):
            joined = " ".join(tape["data_cut_off_date"].astype(str)).lower()
            self.assertNotIn("october", joined)
            self.assertNotIn("november", joined)

    def test_E_lineage_preserves_raw_and_transform(self):
        lin = pd.read_csv(self.oct_tr["central_tape_lineage_path"], dtype=str)
        cut = lin[lin["canonical_field"] == "data_cut_off_date"]
        self.assertFalse(cut.empty)
        self.assertTrue((cut["source_value"].str.lower() == "october").all())  # raw preserved
        # The period->month-end transform is recorded in the resolution basis.
        self.assertTrue(cut["source_resolution_basis"].str.contains("period_label_to_month_end").all())

    def test_04c_domain_rows(self):
        rows = list(csv.DictReader(open(self.oct_proj / "04c_source_period_eligibility.csv")))
        domains = {r["output_domain"] for r in rows}
        self.assertIn("central_lender_tape", domains)
        self.assertIn("pipeline_mi", domains)
        # the funded extract is a universe source; pipeline files are not
        univ = [r for r in rows if r["is_universe_source"] in ("True", "true")]
        self.assertTrue(any("LoanExtract" in r["source_file"] for r in univ))
        self.assertFalse(any("M2L" in r["source_file"] for r in univ))

    def test_18f_universe_debug_fields(self):
        for proj, n in ((self.oct_proj, self.OCT_N), (self.nov_proj, self.NOV_N)):
            dbg = json.loads((proj / "18f_central_universe_debug.json").read_text())
            self.assertTrue(dbg["period_gate_active"])
            self.assertEqual(dbg["canonical_universe_rows"], n)
            self.assertIn("LoanExtract", dbg["selected_universe_source_file"])
            self.assertTrue(dbg["selected_universe_key_column"])
            self.assertIn("duplicate_raw_keys_collapsed", dbg)


# C — entity-key canonicalisation -------------------------------------------- #
class TestEntityKeyCanonicalisation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.root = Path(tempfile.mkdtemp(prefix="period_ek_"))
        inp = cls.root / "input"
        inp.mkdir(parents=True)
        n = 12
        short = [760000 + i for i in range(n)]                 # 760000..
        long = [s * 100 + 1 for s in short]                    # 76000001.. (stable 01 suffix)
        # Source A: short-form key + the funded balance (current-book, with period).
        pd.DataFrame({"Loan Policy Number": short, "Month Run": ["October"] * n,
                      "Current Outstanding Balance": [100000.0] * n,
                      "Loan Interest Rate": [3.5] * n}).to_csv(
            inp / "LoanExtract One.csv", index=False)
        # Source B: long-form key, enrichment fields (collateral).
        pd.DataFrame({"Account Number": long,
                      "Property Valuation": [250000.0] * n,
                      "Property Post Code": [f"AB{i} 1CD" for i in range(n)]}).to_csv(
            inp / "Collateral Extract.csv", index=False)
        cls.n = n
        cls.proj, cls.tr, cls.tape = _run_promotion(cls.root, inp, "mi_2025_10")

    def test_one_canonical_row_per_entity(self):
        # 12 entities, not 24 short+long duplicates.
        self.assertEqual(len(self.tape), self.n)

    def test_no_longform_duplicate_rows(self):
        ids = set(self.tape["loan_identifier"].astype(str))
        self.assertNotIn("76000001", ids)   # long form collapsed onto short canonical

    def test_raw_source_keys_recorded_in_04b(self):
        rows = list(csv.DictReader(open(self.proj / "04b_entity_key_resolution.csv")))
        cols = {r["selected_key_column"] for r in rows}
        self.assertIn("Loan Policy Number", cols)
        self.assertIn("Account Number", cols)


# D — duplicate / stale source avoidance ------------------------------------- #
class TestStaleDuplicateAvoidance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.root = Path(tempfile.mkdtemp(prefix="period_stale_"))
        inp = cls.root / "input"
        inp.mkdir(parents=True)
        n = 20
        ids = [760000 + i for i in range(n)]
        pd.DataFrame({"Loan Policy Number": ids, "Month Run": ["October"] * n,
                      "Current Outstanding Balance": [100000.0] * n,
                      "Loan Interest Rate": [3.5] * n}).to_csv(
            inp / "LoanExtract One.csv", index=False)
        # Stale `_test` duplicate with DIFFERENT extra ids that must NOT enter the universe.
        stale_ids = [880000 + i for i in range(n)]
        pd.DataFrame({"Loan Policy Number": stale_ids, "Month Run": ["October"] * n,
                      "Current Outstanding Balance": [1.0] * n,
                      "Loan Interest Rate": [9.9] * n}).to_csv(
            inp / "LoanExtract One - OMNI_test.csv", index=False)
        cls.n = n
        cls.proj, cls.tr, cls.tape = _run_promotion(cls.root, inp, "mi_2025_10")

    def test_clean_source_is_universe_not_test_duplicate(self):
        dbg = json.loads((self.proj / "18f_central_universe_debug.json").read_text())
        self.assertNotIn("_test", dbg["selected_universe_source_file"].lower())
        self.assertEqual(len(self.tape), self.n)

    def test_stale_ids_absent(self):
        ids = set(self.tape["loan_identifier"].astype(str))
        self.assertFalse(any(i.startswith("8800") for i in ids))

    def test_stale_recorded_in_excluded(self):
        dbg = json.loads((self.proj / "18f_central_universe_debug.json").read_text())
        excluded = " ".join(json.dumps(e) for e in dbg["excluded_sources"]).lower()
        self.assertIn("_test", excluded)


# Expected-balance reconciliation (generic, config-driven) ------------------- #
class TestExpectedBalanceReconciliation(unittest.TestCase):
    def test_not_configured_when_missing(self):
        r = spe.reconcile_balance(None, 33, 4_200_000.0, 33, True)
        self.assertEqual(r["balance_check_status"], "not_configured")
        self.assertEqual(r["actual_loan_count"], 33)

    def test_pass_within_tolerance(self):
        exp = {"expected_loan_count": 33, "expected_current_outstanding_balance": 4_200_000,
               "tolerance_abs": 50000, "tolerance_pct": 0.02, "severity": "warning"}
        r = spe.reconcile_balance(exp, 33, 4_180_000.0, 33, True)
        self.assertEqual(r["balance_check_status"], "pass")
        self.assertTrue(r["loan_count_match"])
        self.assertEqual(r["expected_loan_count"], 33)

    def test_warning_out_of_tolerance_default_severity(self):
        exp = {"expected_loan_count": 33, "expected_current_outstanding_balance": 4_200_000,
               "tolerance_abs": 1000, "tolerance_pct": 0.0, "severity": "warning"}
        r = spe.reconcile_balance(exp, 30, 3_000_000.0, 30, True)
        self.assertEqual(r["balance_check_status"], "warning")
        self.assertFalse(r["loan_count_match"])
        self.assertIn("loan_count", r["diagnostic"])

    def test_fail_only_when_blocking(self):
        exp = {"expected_loan_count": 33, "expected_current_outstanding_balance": 4_200_000,
               "tolerance_abs": 1000, "tolerance_pct": 0.0, "severity": "blocking"}
        r = spe.reconcile_balance(exp, 30, 3_000_000.0, 30, True)
        self.assertEqual(r["balance_check_status"], "fail")

    def test_missing_balance_reports_diagnostic(self):
        exp = {"expected_loan_count": 33, "expected_current_outstanding_balance": 4_200_000,
               "severity": "warning"}
        r = spe.reconcile_balance(exp, 33, None, 0, False)
        self.assertEqual(r["balance_check_status"], "warning")
        self.assertIn("current_outstanding_balance", r["diagnostic"])

    def test_numeric_normalisation(self):
        self.assertEqual(spe._coerce_float("£4,200,000.00"), 4200000.0)
        self.assertEqual(spe._coerce_float("112,619.77"), 112619.77)
        self.assertIsNone(spe._coerce_float(""))
        self.assertIsNone(spe._coerce_float("n/a"))

    def test_lookup_nested_by_client_then_run(self):
        checks = {"client_001": {"mi_2025_10": {"expected_loan_count": 33}}}
        self.assertEqual(spe.lookup_expected(checks, "client_001", "mi_2025_10")["expected_loan_count"], 33)
        self.assertIsNone(spe.lookup_expected(checks, "client_001", "mi_2099_01"))
        self.assertIsNone(spe.lookup_expected(checks, "other", "mi_2025_10"))

    def test_config_has_client_001_entries_not_in_python(self):
        # The values live in config only — assert the engine reads them from yaml.
        import inspect
        src = inspect.getsource(ctb)
        self.assertNotIn("4200000", src)
        self.assertNotIn("8900000", src)
        checks = spe.load_expected_balance_checks()
        self.assertEqual(
            checks["client_001"]["mi_2025_10"]["expected_current_outstanding_balance"], 4200000)

    def test_18f_includes_reconciliation_block(self):
        # An end-to-end run for a client_id with configured checks records the
        # full reconciliation block in 18f.
        warnings.simplefilter("ignore")
        root = Path(tempfile.mkdtemp(prefix="period_recon_"))
        inp = root / "input"
        inp.mkdir(parents=True)
        n = 33
        pd.DataFrame({"Loan Policy Number": [760000 + i for i in range(n)],
                      "Month Run": ["October"] * n,
                      "Current Outstanding Balance": [round(4_200_000 / n, 2)] * n,
                      "Loan Interest Rate": [3.5] * n}).to_csv(inp / "LoanExtract One.csv", index=False)
        from engine.onboarding_agent import workflow as wf, storage_paths
        proj = root / "proj"
        wf.run_operator_workflow(input_dir=str(inp), client_name="C1", client_id="client_001",
                                 run_id="mi_2025_10", mode="mi_only", project_dir=str(proj),
                                 product_profile="equity_release_lifetime_mortgage")
        rp = storage_paths.resolve_run_paths(
            project_dir=str(proj), input_dir=str(inp), output_root=None,
            client_id="client_001", run_id="mi_2025_10", storage_backend="local",
            input_uri="", output_uri="")
        ctb.build_central_tapes(str(proj), rp, _REGISTRY, mode="mi_only")
        chk = json.loads((proj / "18f_central_universe_debug.json").read_text())["expected_balance_check"]
        for f in ("expected_loan_count", "actual_loan_count", "loan_count_match",
                  "expected_current_outstanding_balance", "actual_current_outstanding_balance",
                  "balance_delta", "balance_delta_pct", "tolerance_abs", "tolerance_pct",
                  "balance_check_status"):
            self.assertIn(f, chk)
        self.assertEqual(chk["expected_loan_count"], 33)
        self.assertEqual(chk["actual_loan_count"], 33)
        self.assertEqual(chk["balance_check_status"], "pass")


# Valuation enrichment + currency default + mapping_method consistency -------- #
class TestFundedTapeEnrichment(unittest.TestCase):
    """Funded current-book (Month Run) + a linked collateral report carrying the
    valuation, keyed long-form (entity-key links short<->long)."""

    N = 14

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.root = Path(tempfile.mkdtemp(prefix="enrich_"))
        inp = cls.root / "input"
        inp.mkdir(parents=True)
        short = [760000 + i for i in range(cls.N)]
        long = [s * 100 + 1 for s in short]
        pd.DataFrame({"Loan Policy Number": short, "Month Run": ["October"] * cls.N,
                      "Current Outstanding Balance": [100000.0] * cls.N,
                      "Loan Interest Rate": [3.5] * cls.N,
                      "Policy Completion Date": ["2025-09-01"] * cls.N}).to_csv(
            inp / "LoanExtract One.csv", index=False)
        pd.DataFrame({"Account Number": long,
                      "Latest Property Value": [250000.0 + i for i in range(cls.N)]}).to_csv(
            inp / "Collateral Extract.csv", index=False)
        cls.proj, cls.tr, cls.tape = _run_promotion(cls.root, inp, "mi_2025_10")
        cls.lin = pd.read_csv(cls.tr["central_tape_lineage_path"], dtype=str)

    # 1 — valuation enrichment from linked collateral source
    def test_valuation_populated_from_collateral(self):
        self.assertIn("current_valuation_amount", self.tape.columns)
        self.assertEqual(int(self.tape["current_valuation_amount"].notna().sum()), self.N)

    def test_valuation_lineage_shows_source_and_join(self):
        v = self.lin[self.lin["canonical_field"] == "current_valuation_amount"].head(1)
        self.assertFalse(v.empty)
        self.assertEqual(v["source_file"].iloc[0], "Collateral Extract.csv")
        self.assertEqual(v["source_column"].iloc[0], "Latest Property Value")
        dbg = json.loads((self.proj / "18f_central_universe_debug.json").read_text())
        diag = {d["canonical_field"]: d for d in dbg["enrichment_field_diagnostics"]}
        self.assertEqual(diag["current_valuation_amount"]["status"], "populated_from_enrichment")
        self.assertTrue(diag["current_valuation_amount"]["entity_key_join_basis"])

    def test_unmapped_enrichment_flagged_not_silently_null(self):
        # original_valuation_amount has no source -> diagnostic, not a silent null.
        dbg = json.loads((self.proj / "18f_central_universe_debug.json").read_text())
        diag = {d["canonical_field"]: d for d in dbg["enrichment_field_diagnostics"]}
        self.assertIn("original_valuation_amount", diag)
        self.assertIn(diag["original_valuation_amount"]["status"],
                      ("no_period_eligible_source", "needs_operator_review"))

    # 2 — configured/default currency
    def test_currency_defaults_to_gbp_all_rows(self):
        self.assertIn("exposure_currency_denomination", self.tape.columns)
        self.assertEqual(set(self.tape["exposure_currency_denomination"].dropna()), {"GBP"})
        self.assertEqual(int(self.tape["exposure_currency_denomination"].notna().sum()), self.N)

    def test_currency_lineage_configured_static(self):
        c = self.lin[self.lin["canonical_field"] == "exposure_currency_denomination"].head(1)
        self.assertFalse(c.empty)
        self.assertEqual(c["mapping_method"].iloc[0], "configured_static")
        self.assertIn("onboarding_agent.yaml", c["source_file"].iloc[0])
        self.assertEqual(str(c["review_required"].iloc[0]).lower(), "false")

    # 3 — mapping_method consistency under target-first
    def test_funded_fields_use_target_first_resolved(self):
        for f in ("current_interest_rate", "current_outstanding_balance",
                  "current_valuation_amount"):
            rows = self.lin[self.lin["canonical_field"] == f]
            self.assertFalse(rows.empty, f)
            self.assertTrue((rows["mapping_method"] == "target_first_resolved").all(), f)

    def test_source_resolution_basis_preserved(self):
        rate = self.lin[self.lin["canonical_field"] == "current_interest_rate"].head(1)
        self.assertEqual(rate["source_resolution_basis"].iloc[0], "alias")


class TestMappingMethodConsistencyAcrossPeriods(unittest.TestCase):
    """Two separate monthly funded extracts: 28a selects one (November); the
    October run uses the period-eligible October file via fallback. Both must
    still report mapping_method = target_first_resolved (issue #3)."""

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.root = Path(tempfile.mkdtemp(prefix="method_"))
        inp = cls.root / "input"
        inp.mkdir(parents=True)
        oct_dir = inp / "2025-10-31"; oct_dir.mkdir()
        nov_dir = inp / "2025-11-30"; nov_dir.mkdir()
        pd.DataFrame({"Loan Policy Number": [760000 + i for i in range(30)],
                      "Current Outstanding Balance": [100000.0] * 30,
                      "Loan Interest Rate": [3.5] * 30,
                      "Policy Completion Date": ["2025-09-01"] * 30}).to_csv(
            oct_dir / "LoanExtract October.csv", index=False)
        pd.DataFrame({"Loan Policy Number": [760000 + i for i in range(50)],
                      "Current Outstanding Balance": [110000.0] * 50,
                      "Loan Interest Rate": [3.6] * 50,
                      "Policy Completion Date": ["2025-09-01"] * 50}).to_csv(
            nov_dir / "LoanExtract November.csv", index=False)
        cls.oct_lin = pd.read_csv(
            _run_promotion(cls.root, inp, "mi_2025_10")[1]["central_tape_lineage_path"], dtype=str)
        cls.nov_lin = pd.read_csv(
            _run_promotion(cls.root, inp, "mi_2025_11")[1]["central_tape_lineage_path"], dtype=str)

    def _method(self, lin, field):
        rows = lin[lin["canonical_field"] == field]
        return set(rows["mapping_method"]) if not rows.empty else set()

    def test_both_months_target_first_resolved(self):
        for field in ("current_interest_rate", "current_outstanding_balance"):
            self.assertEqual(self._method(self.oct_lin, field), {"target_first_resolved"}, field)
            self.assertEqual(self._method(self.nov_lin, field), {"target_first_resolved"}, field)


class TestNoHardcodedCurrencyOrClient(unittest.TestCase):
    def test_no_gbp_or_client_literals_in_python(self):
        import inspect
        from engine.onboarding_agent import central_tape_builder, source_period_eligibility
        for mod in (central_tape_builder, source_period_eligibility):
            src = inspect.getsource(mod)
            self.assertNotIn('"GBP"', src)
            self.assertNotIn("'GBP'", src)
            self.assertNotIn("client_001", src)

    def test_currency_and_enrichment_from_config(self):
        cfg = ctb._load_central_tape_config()
        self.assertIn("current_valuation_amount", cfg.get("mi_enrichment_fields", []))
        self.assertEqual(
            cfg["static_field_defaults"]["exposure_currency_denomination"]["value"], "GBP")

    def test_jurisdiction_inference_fallback(self):
        # When no static default exists, a configured jurisdiction maps to currency.
        specs = ctb._resolve_static_field_specs(
            {"jurisdiction_currency": {"IE": "EUR"}, "default_jurisdiction": "IE",
             "jurisdiction_currency_fields": ["exposure_currency_denomination"]},
            {}, client_id="x")
        self.assertEqual(specs["exposure_currency_denomination"]["value"], "EUR")
        self.assertEqual(specs["exposure_currency_denomination"]["method"], "jurisdiction_inference")


# F — regulatory untouched --------------------------------------------------- #
class TestRegulatoryUntouched(unittest.TestCase):
    def test_period_gate_only_for_mi_modes(self):
        import inspect
        src = inspect.getsource(ctb.build_central_tapes)
        self.assertIn('if mode in ("mi_only", "mna_dd")', src)
        self.assertIn('period_gate: Dict[str, Any] = {}', src)

    def test_load_eligibility_defaults_to_lender_tape_domain(self):
        import inspect
        sig = inspect.signature(spe.load_eligibility)
        self.assertEqual(sig.parameters["output_domain"].default, "central_lender_tape")


if __name__ == "__main__":
    unittest.main(verbosity=2)
