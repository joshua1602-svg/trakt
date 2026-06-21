#!/usr/bin/env python3
"""mi_agent_api/tests/test_funded_prep.py

The funded MI data-preparation layer turns the thin promoted central lender tape
into an analytics-ready dataset using the EXISTING canonical bucket engine
(analytics_lib.buckets + config/mi/buckets.yaml) — so Streamlit / React / API are
consistent and no bucketing lives in React.

Covers:
  * prepare_funded_mi_dataset derives LTV / vintage / months-on-book and
    materialises ltv_bucket / interest_rate_bucket / ticket_bucket /
    time_on_book_bucket; reports the dimensions that are unavailable and why;
  * the API prefers the prepared dataset; MI_AGENT_DISABLE_PREP serves the raw
    thin tape; /health reports preparation + dimensions;
  * /mi/query answers a funded stratification (balance by LTV / ticket bucket)
    AND a funded summary from the prepared dataset.
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
from mi_agent_api.funded_prep import prepare_funded_mi_dataset

_REGISTRY = str(_REPO_ROOT / "config" / "system" / "fields_registry.yaml")


def _canonical_funded_df(n: int = 40) -> pd.DataFrame:
    """A central-lender-tape-shaped frame (canonical field names)."""
    return pd.DataFrame({
        "loan_identifier": [760000 + i for i in range(n)],
        "current_outstanding_balance": [100000.0 + i * 1500 for i in range(n)],
        "current_valuation_amount": [250000.0 + i * 2000 for i in range(n)],
        "current_interest_rate": [3.0 + (i % 6) * 0.4 for i in range(n)],
        "current_principal_balance": [100000.0 + i * 1500 for i in range(n)],
        "origination_date": [f"20{15 + i % 8}-0{1 + i % 9}-15" for i in range(n)],
        "reporting_date": ["2025-10-31"] * n,
        "data_cut_off_date": ["2025-10-31"] * n,
        "exposure_currency_denomination": ["GBP"] * n,
    })


# --------------------------------------------------------------------------- #
# Fast unit tests on the prep engine
# --------------------------------------------------------------------------- #
class TestPrepEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.df, cls.report = prepare_funded_mi_dataset(_canonical_funded_df())

    def test_derives_source_fields(self):
        self.assertIn("current_loan_to_value", self.report["derived_fields"])
        self.assertIn("vintage_year", self.report["derived_fields"])
        self.assertIn("months_on_book", self.report["derived_fields"])
        for c in ("current_loan_to_value", "vintage_year", "months_on_book"):
            self.assertIn(c, self.df.columns)

    def test_materialises_expected_buckets(self):
        for dim in ("ltv_bucket", "interest_rate_bucket", "ticket_bucket", "time_on_book_bucket"):
            self.assertIn(dim, self.df.columns, dim)
            self.assertIn(dim, self.report["dimensions_available"], dim)

    def test_reports_missing_dimensions_with_reason(self):
        from mi_agent_api.funded_prep import missing_dimension_names
        names = missing_dimension_names(self.report)
        for dim in ("age_bucket", "geographic_region_obligor", "original_ltv_bucket"):
            self.assertIn(dim, names, dim)
        # original LTV missing -> derivation_inputs_missing (no original valuation).
        by_dim = {m["dimension"]: m for m in self.report["missing_dimensions"]}
        self.assertEqual(by_dim["original_ltv_bucket"]["reason"], "derivation_inputs_missing")

    def test_preparation_does_not_change_row_count_or_balance(self):
        raw = _canonical_funded_df()
        self.assertEqual(len(self.df), len(raw))
        self.assertAlmostEqual(
            pd.to_numeric(self.df["current_outstanding_balance"]).sum(),
            pd.to_numeric(raw["current_outstanding_balance"]).sum(), places=2)

    def test_no_valuation_means_no_ltv_but_still_rate_and_ticket(self):
        thin = _canonical_funded_df().drop(columns=["current_valuation_amount", "origination_date"])
        df2, rep2 = prepare_funded_mi_dataset(thin)
        from mi_agent_api.funded_prep import missing_dimension_names
        self.assertNotIn("ltv_bucket", df2.columns)
        self.assertIn("ltv_bucket", missing_dimension_names(rep2))
        self.assertIn("interest_rate_bucket", df2.columns)
        self.assertIn("ticket_bucket", df2.columns)


# --------------------------------------------------------------------------- #
# End-to-end: real promoted tape -> prepared dataset -> API stratification
# --------------------------------------------------------------------------- #
class TestPreparedDatasetThroughApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.root = Path(tempfile.mkdtemp(prefix="prep_api_"))
        inp = cls.root / "input"
        inp.mkdir(parents=True)
        n = 33
        ids = [760000 + i for i in range(n)]
        pd.DataFrame({
            "Loan Policy Number": ids, "Month Run": ["October"] * n,
            "Loan Interest Rate": [3.0 + (i % 6) * 0.4 for i in range(n)],
            "Current Outstanding Balance": [100000.0 + i * 2000 for i in range(n)],
            "Policy Completion Date": [f"20{16 + i % 7}-0{1 + i % 9}-15" for i in range(n)],
        }).to_csv(inp / "LoanExtract One.csv", index=False)
        pd.DataFrame({"Account Number": [s * 100 + 1 for s in ids],
                      "Latest Property Value": [240000.0 + i * 3000 for i in range(n)]}).to_csv(
            inp / "Collateral Extract.csv", index=False)
        from engine.onboarding_agent import workflow as wf, storage_paths, central_tape_builder
        proj = cls.root / "proj"
        wf.run_operator_workflow(
            input_dir=str(inp), client_name="Client 001", client_id="client_001",
            run_id="mi_2025_10", mode="mi_only", project_dir=str(proj),
            product_profile="equity_release_lifetime_mortgage")
        rp = storage_paths.resolve_run_paths(
            project_dir=str(proj), input_dir=str(inp), output_root=None,
            client_id="client_001", run_id="mi_2025_10", storage_backend="local",
            input_uri="", output_uri="")
        res = central_tape_builder.build_central_tapes(str(proj), rp, _REGISTRY, mode="mi_only")
        cls.tape = res["central_lender_tape_path"]

    def _serve(self, disable_prep=False):
        for k in ("MI_AGENT_CENTRAL_TAPE", "MI_AGENT_DATA_CSV", "MI_AGENT_ANALYTICS_DATASET",
                  "MI_AGENT_DISABLE_PREP", "MI_AGENT_CLIENT_ID", "MI_AGENT_RUN_ID"):
            os.environ.pop(k, None)
        os.environ["MI_AGENT_CENTRAL_TAPE"] = str(self.tape)
        os.environ["MI_AGENT_CLIENT_ID"] = "client_001"
        os.environ["MI_AGENT_RUN_ID"] = "mi_2025_10"
        if disable_prep:
            os.environ["MI_AGENT_DISABLE_PREP"] = "1"
        data_source.reset_cache()

    def tearDown(self):
        for k in ("MI_AGENT_CENTRAL_TAPE", "MI_AGENT_DISABLE_PREP", "MI_AGENT_CLIENT_ID",
                  "MI_AGENT_RUN_ID"):
            os.environ.pop(k, None)
        data_source.reset_cache()

    def test_health_prepared_with_dimensions(self):
        from fastapi.testclient import TestClient
        from mi_agent_api.app import app
        self._serve()
        body = TestClient(app).get("/health").json()
        self.assertEqual(body["dataSourceKind"], data_source.KIND_PREPARED)
        self.assertTrue(body["preparationApplied"])
        self.assertIn("ltv_bucket", body["dimensionsAvailable"])
        self.assertIn("ticket_bucket", body["dimensionsAvailable"])
        self.assertIn("age_bucket", body["missingDimensionNames"])

    def test_disable_prep_serves_raw(self):
        self._serve(disable_prep=True)
        self.assertEqual(data_source.data_source_kind(), data_source.KIND_FUNDED_RAW)
        info = data_source.data_source_info()
        self.assertFalse(info["preparation_applied"])
        self.assertNotIn("ltv_bucket", data_source.get_dataframe().columns)

    def test_query_stratification_by_ltv_bucket(self):
        from fastapi.testclient import TestClient
        from mi_agent_api.app import app
        self._serve()
        body = TestClient(app).post(
            "/mi/query",
            json={"question": "current outstanding balance by ltv bucket",
                  "portfolioId": "client_001/mi_2025_10", "asOfDate": "2025-10-31"}).json()
        self.assertTrue(body["ok"], body.get("validation"))
        types = {a["type"] for a in body["artifacts"]}
        self.assertTrue({"chart", "table"} & types, body)

    def test_query_stratification_by_ticket_bucket(self):
        from fastapi.testclient import TestClient
        from mi_agent_api.app import app
        self._serve()
        body = TestClient(app).post(
            "/mi/query",
            json={"question": "current outstanding balance by ticket bucket",
                  "portfolioId": "client_001/mi_2025_10", "asOfDate": "2025-10-31"}).json()
        self.assertTrue(body["ok"], body.get("validation"))

    def test_raw_mode_stratification_fails_but_summary_works(self):
        # With prep disabled the thin tape can't strat by ltv, but KPIs still work.
        self._serve(disable_prep=True)
        from fastapi.testclient import TestClient
        from mi_agent_api.app import app
        client = TestClient(app)
        strat = client.post("/mi/query", json={
            "question": "current outstanding balance by ltv bucket"}).json()
        self.assertFalse(strat["ok"])
        summary = client.post("/mi/query", json={"question": "portfolio summary"}).json()
        self.assertTrue(summary["ok"], summary.get("validation"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
