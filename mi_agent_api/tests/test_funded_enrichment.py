#!/usr/bin/env python3
"""mi_agent_api/tests/test_funded_enrichment.py

Raw client fields that exist in the source pack must reach the prepared funded MI
dataset via config-driven enrichment (04b entity-key linkage) + the funded MI prep
layer — not be reported "missing". And LTV must be DERIVED from balance/valuation
when no raw LTV column is supplied (product rule).

Covers the acceptance list:
  * valuation enrichment from a linked collateral source;
  * current LTV derived from balance and valuation;
  * original LTV derived from original balance and original valuation;
  * explicit source LTV preferred over derived LTV;
  * missing valuation -> derivation_inputs_missing (no misleading LTV);
  * one linked enrichment field per source type: loan (age), collateral (valuation),
    channel (origination/broker) -> age_bucket / ltv / region / channel available;
  * /health moves these into dimensionsAvailable.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from mi_agent_api import data_source
from mi_agent_api.funded_prep import missing_dimension_names, prepare_funded_mi_dataset

_REGISTRY = str(_REPO_ROOT / "config" / "system" / "fields_registry.yaml")
_REGIONS = ["UKI", "UKJ", "UKK", "UKD"]


def _promote(root: Path, loan_df: pd.DataFrame, collateral_df: pd.DataFrame | None,
             run_id: str = "mi_2025_10") -> str:
    inp = root / "input"
    inp.mkdir(parents=True, exist_ok=True)
    loan_df.to_csv(inp / "LoanExtract One.csv", index=False)
    if collateral_df is not None:
        collateral_df.to_csv(inp / "Collateral Extract.csv", index=False)
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
    return res["central_lender_tape_path"]


def _rich_loan(n: int = 33, with_source_ltv: bool = False) -> pd.DataFrame:
    ids = [760000 + i for i in range(n)]
    d = {
        "Loan Policy Number": ids, "Month Run": ["October"] * n,
        "Loan Interest Rate": [3.0 + (i % 6) * 0.4 for i in range(n)],
        "Current Outstanding Balance": [100000.0 + i * 2000 for i in range(n)],
        "Original Principal Balance": [90000.0 + i * 1800 for i in range(n)],
        "Policy Completion Date": [f"20{16 + i % 7}-0{1 + i % 9}-15" for i in range(n)],
        "Youngest Age": [60 + i % 25 for i in range(n)],
        "Geo Region": [_REGIONS[i % 4] for i in range(n)],
        "channel": [["Direct", "Broker"][i % 2] for i in range(n)],
        "broker": [["BrokerA", "BrokerB"][i % 2] for i in range(n)],
    }
    if with_source_ltv:
        d["Cur LTV %"] = [40 + (i % 10) for i in range(n)]   # explicit source LTV (percent)
    return pd.DataFrame(d)


def _collateral(n: int = 33, valuation: bool = True) -> pd.DataFrame:
    ids = [760000 + i for i in range(n)]
    d = {"Account Number": [s * 100 + 1 for s in ids],
         "property region": [_REGIONS[i % 4] for i in range(n)]}
    if valuation:
        d["Latest Property Value"] = [240000.0 + i * 3000 for i in range(n)]
        d["initial valuation"] = [230000.0 + i * 2800 for i in range(n)]
    return pd.DataFrame(d)


def _serve(tape: str):
    for k in ("MI_AGENT_CENTRAL_TAPE", "MI_AGENT_DATA_CSV", "MI_AGENT_ANALYTICS_DATASET",
              "MI_AGENT_DISABLE_PREP", "MI_AGENT_CLIENT_ID", "MI_AGENT_RUN_ID"):
        os.environ.pop(k, None)
    os.environ["MI_AGENT_CENTRAL_TAPE"] = str(tape)
    os.environ["MI_AGENT_CLIENT_ID"] = "client_001"
    os.environ["MI_AGENT_RUN_ID"] = "mi_2025_10"
    data_source.reset_cache()


class TestFullEnrichmentReachesDimensions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.root = Path(tempfile.mkdtemp(prefix="enrich_full_"))
        cls.tape = _promote(cls.root, _rich_loan(), _collateral())
        cls.df = pd.read_csv(cls.tape)
        cls.prep, cls.report = prepare_funded_mi_dataset(cls.df)

    @classmethod
    def tearDownClass(cls):
        for k in ("MI_AGENT_CENTRAL_TAPE", "MI_AGENT_CLIENT_ID", "MI_AGENT_RUN_ID"):
            os.environ.pop(k, None)
        data_source.reset_cache()

    def test_enrichment_fields_in_central_tape(self):
        # loan (age/channel/orig principal), collateral (valuation/region) all promoted.
        for f in ("youngest_borrower_age", "geographic_region_obligor", "origination_channel",
                  "broker_channel", "original_principal_balance", "current_valuation_amount",
                  "original_valuation_amount"):
            self.assertIn(f, self.df.columns, f)
            self.assertEqual(int(self.df[f].notna().sum()), len(self.df), f)

    def test_current_ltv_derived_from_balance_and_valuation(self):
        basis = {b["target"]: b for b in self.report["ltv_derivation_basis"]}
        cur = basis["current_loan_to_value"]
        self.assertEqual(cur["method"], "derived_ratio")
        self.assertEqual(cur["numerator"], "current_outstanding_balance")
        self.assertEqual(cur["denominator"], "current_valuation_amount")
        self.assertIn("ltv_bucket", self.prep.columns)
        self.assertTrue((self.prep["current_loan_to_value"].dropna() <= 1.0).all())  # ratio

    def test_original_ltv_derived_from_original_balance_and_valuation(self):
        basis = {b["target"]: b for b in self.report["ltv_derivation_basis"]}
        orig = basis["original_loan_to_value"]
        self.assertEqual(orig["method"], "derived_ratio")
        self.assertEqual(orig["numerator"], "original_principal_balance")
        self.assertIn("original_ltv_bucket", self.prep.columns)

    def test_all_core_dimensions_available(self):
        avail = set(self.report["dimensions_available"])
        for dim in ("age_bucket", "ltv_bucket", "original_ltv_bucket", "interest_rate_bucket",
                    "ticket_bucket", "time_on_book_bucket", "vintage_year",
                    "geographic_region_obligor", "origination_channel"):
            self.assertIn(dim, avail, dim)
        self.assertEqual(self.report["missing_dimensions"], [])

    def test_health_shows_dimensions_available(self):
        from fastapi.testclient import TestClient
        from mi_agent_api.app import app
        _serve(self.tape)
        body = TestClient(app).get("/health").json()
        self.assertTrue(body["preparationApplied"])
        for dim in ("age_bucket", "ltv_bucket", "original_ltv_bucket",
                    "geographic_region_obligor", "origination_channel"):
            self.assertIn(dim, body["dimensionsAvailable"], dim)

    def test_trace_reports_fields_available_end_to_end(self):
        from mi_agent_api import funded_mi_trace as T
        rows = T.trace(self.root / "proj_mi_2025_10", self.tape)
        by = {r["canonical_field"]: r for r in rows}
        self.assertEqual(by["youngest_borrower_age"]["status"], "available")
        self.assertEqual(by["original_principal_balance"]["status"], "available")
        self.assertEqual(by["current_loan_to_value"]["reason"], "dimension_available_via_derivation")
        self.assertTrue(by["broker_channel"]["in_central_tape"])

    def test_query_stratification_by_region_and_age(self):
        from fastapi.testclient import TestClient
        from mi_agent_api.app import app
        _serve(self.tape)
        client = TestClient(app)
        for q in ("current outstanding balance by age bucket",
                  "current outstanding balance by original ltv bucket"):
            body = client.post("/mi/query", json={"question": q}).json()
            self.assertTrue(body["ok"], (q, body.get("validation")))


class TestLtvPreferenceAndMissing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.root = Path(tempfile.mkdtemp(prefix="enrich_ltv_"))

    def test_explicit_source_ltv_preferred_over_derived(self):
        tape = _promote(self.root, _rich_loan(with_source_ltv=True), _collateral(),
                        run_id="mi_2025_10")
        df = pd.read_csv(tape)
        self.assertIn("current_loan_to_value", df.columns)  # source LTV promoted
        _prep, report = prepare_funded_mi_dataset(df)
        cur = {b["target"]: b for b in report["ltv_derivation_basis"]}["current_loan_to_value"]
        self.assertEqual(cur["method"], "source_field")  # preferred over derived_ratio
        self.assertEqual(cur["confidence"], 1.0)

    def test_missing_valuation_reports_derivation_inputs_missing(self):
        # No collateral valuation and no original valuation -> LTV must NOT be faked.
        tape = _promote(self.root, _rich_loan(), _collateral(valuation=False),
                        run_id="mi_2025_11")
        df = pd.read_csv(tape)
        prep, report = prepare_funded_mi_dataset(df)
        self.assertNotIn("ltv_bucket", prep.columns)
        by_dim = {m["dimension"]: m for m in report["missing_dimensions"]}
        self.assertEqual(by_dim["ltv_bucket"]["reason"], "derivation_inputs_missing")
        self.assertEqual(by_dim["original_ltv_bucket"]["reason"], "derivation_inputs_missing")
        from mi_agent_api import funded_mi_trace as T
        rows = T.trace(self.root / "proj_mi_2025_11", tape)
        by = {r["canonical_field"]: r for r in rows}
        self.assertEqual(by["current_loan_to_value"]["reason"], "derivation_inputs_missing")


class TestPrepNumericSafety(unittest.TestCase):
    def test_divide_by_zero_and_non_numeric_safe(self):
        df = pd.DataFrame({
            "loan_identifier": [1, 2, 3, 4],
            "current_outstanding_balance": [100000, 50000, 75000, 60000],
            "current_valuation_amount": [200000, 0, None, "n/a"],  # zero / null / text
        })
        prep, report = prepare_funded_mi_dataset(df)
        cur = {b["target"]: b for b in report["ltv_derivation_basis"]}["current_loan_to_value"]
        self.assertEqual(cur["method"], "derived_ratio")
        ltv = prep["current_loan_to_value"]
        self.assertAlmostEqual(ltv.iloc[0], 0.5)         # 100000/200000
        self.assertTrue(pd.isna(ltv.iloc[1]))            # divide by zero -> NaN, not inf
        self.assertTrue(pd.isna(ltv.iloc[2]) and pd.isna(ltv.iloc[3]))

    def test_percent_source_ltv_normalised_to_ratio(self):
        df = pd.DataFrame({
            "loan_identifier": [1, 2, 3],
            "current_loan_to_value": [45.0, 60.0, 30.0],  # percent inputs
        })
        prep, report = prepare_funded_mi_dataset(df)
        self.assertTrue((prep["current_loan_to_value"] <= 1.0).all())  # -> 0.45, 0.60, 0.30


if __name__ == "__main__":
    unittest.main(verbosity=2)
