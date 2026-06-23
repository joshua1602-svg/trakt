#!/usr/bin/env python3
"""mi_agent_api/tests/test_pipeline_mi.py — Pipeline MI Phase 1.

The pipeline single source of truth: a prepared pipeline MI dataset built from
the committed M2L KFI / pipeline fixture pack, reusing funded MI semantics +
the existing bucket engine, kept SEPARATE from the funded central lender tape,
and exposing API metadata (snapshot/contract) that is forecast-ready.

Covers (Pipeline MI Phase 1, Part 6):
  1. pipeline source discovery;            2. field review / mapping;
  3. prepared pipeline dataset;            4. funded-correlation fields;
  5. separation from the funded tape;      6. API metadata;
  7. forecast-readiness metadata.
"""

from __future__ import annotations

import os
import sys
import unittest
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from mi_agent_api import pipeline_contract as pc
from mi_agent_api.pipeline_prep import (
    diagnostics_by_severity,
    field_correlation_to_funded,
    forecast_readiness,
    load_pipeline_contract,
    prepare_pipeline_mi_dataset,
    resolve_source_columns,
)

_FIXTURE_PACK = _REPO_ROOT / "tests" / "fixtures" / "client_001_mi_pack"
_PIPELINE_ROOT = _FIXTURE_PACK / "pipeline"
_NOV = _PIPELINE_ROOT / "2025-11-01" / "M2L_KFI_and_Pipeline_2025_11_01.csv"
_OCT = _PIPELINE_ROOT / "2025-10-01" / "M2L_KFI_and_Pipeline_2025_10_01.csv"
_REGISTRY = str(_REPO_ROOT / "config" / "system" / "fields_registry.yaml")


def _prep_nov():
    df = pd.read_csv(_NOV)
    return prepare_pipeline_mi_dataset(df, source_file=_NOV.name)


# --------------------------------------------------------------------------- #
# 1. Pipeline source discovery
# --------------------------------------------------------------------------- #
class TestPipelineSourceDiscovery(unittest.TestCase):
    def test_fixture_pack_pipeline_files_exist(self):
        self.assertTrue(_OCT.exists(), _OCT)
        self.assertTrue(_NOV.exists(), _NOV)

    def test_discovers_both_reporting_dates(self):
        sources = pc.discover_pipeline_sources(_FIXTURE_PACK, client_id="client_001")
        dates = [s["reporting_date"] for s in sources]
        self.assertIn("2025-10-01", dates)
        self.assertIn("2025-11-01", dates)
        # ordered oldest -> newest, client inferred, non-empty.
        self.assertEqual(dates, sorted(dates))
        for s in sources:
            self.assertEqual(s["client_id"], "client_001")
            self.assertGreater(s["row_count"], 0)

    def test_resolve_latest_source(self):
        latest = pc.resolve_pipeline_source(_FIXTURE_PACK, "client_001")
        self.assertEqual(latest["reporting_date"], "2025-11-01")


# --------------------------------------------------------------------------- #
# 2. Field review / mapping to existing semantic + funded fields
# --------------------------------------------------------------------------- #
class TestFieldReviewAndMapping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(_NOV)
        cls.mapping, cls.unmatched = resolve_source_columns(cls.df)

    def test_source_headers_map_to_funded_canonical_fields(self):
        m = self.mapping
        self.assertEqual(m.get("current_interest_rate"), "Product Rate")
        self.assertEqual(m.get("current_valuation_amount"), "Estimated Value")
        self.assertEqual(m.get("collateral_geography"), "Property Region")
        self.assertEqual(m.get("broker_channel"), "Broker")
        self.assertEqual(m.get("current_outstanding_balance"), "Loan Amount")
        self.assertEqual(m.get("product_type"), "Product")

    def test_pipeline_specific_headers_mapped(self):
        m = self.mapping
        self.assertEqual(m.get("pipeline_stage"), "Status")
        self.assertEqual(m.get("kfi_date"), "KFI Submitted Date")
        self.assertEqual(m.get("application_date"), "Application Submitted Date")
        self.assertEqual(m.get("offer_date"), "Offer Date")
        self.assertEqual(m.get("pipeline_case_identifier"), "Account Number")

    def test_contract_declares_funded_correlation(self):
        contract = load_pipeline_contract()
        fc = contract["funded_correlated_fields"]
        # economic fields reuse funded canonical names + declare correlation.
        self.assertIn("current_outstanding_balance",
                      fc["current_outstanding_balance"]["funded_correlation"])
        self.assertIn("geographic_region_obligor",
                      fc["collateral_geography"]["funded_correlation"])


# --------------------------------------------------------------------------- #
# 3. Prepared pipeline dataset
# --------------------------------------------------------------------------- #
class TestPreparedPipelineDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.out, cls.report = _prep_nov()

    def test_row_count_non_zero(self):
        self.assertGreater(self.report["row_count"], 0)
        self.assertEqual(len(self.out), 10)

    def test_stage_and_status_present(self):
        self.assertIn("pipeline_stage", self.out.columns)
        self.assertIn("pipeline_status", self.out.columns)
        self.assertIn("pipeline_stage", self.report["dimensions_available"])
        # normalised to the canonical funnel vocabulary.
        self.assertTrue(set(self.out["pipeline_stage"]).issubset(
            {"KFI", "APPLICATION", "OFFER", "COMPLETED", "WITHDRAWN", "UNKNOWN"}))

    def test_economic_amount_present(self):
        self.assertIn("current_outstanding_balance", self.report["metrics_available"])
        self.assertGreater(self.report["total_pipeline_amount"], 0)

    def test_optional_dimensions_present_or_marked(self):
        # broker / region / rate / LTV / valuation are present here.
        for fld in ("broker_channel", "geographic_region_obligor"):
            self.assertIn(fld, self.report["dimensions_available"], fld)
        for fld in ("current_interest_rate", "current_loan_to_value",
                    "current_valuation_amount"):
            self.assertIn(fld, self.report["metrics_available"], fld)

    def test_buckets_reused_from_funded_engine(self):
        for dim in ("ltv_bucket", "age_bucket", "ticket_bucket", "interest_rate_bucket"):
            self.assertIn(dim, self.out.columns, dim)
            self.assertIn(dim, self.report["dimensions_available"], dim)

    def test_pipeline_derived_dimensions(self):
        self.assertIn("expected_completion_month", self.out.columns)
        self.assertIn("pipeline_stage_bucket", self.out.columns)

    def test_amount_when_absent_is_marked_unavailable(self):
        # A source without a loan amount must NOT fabricate one; it is a blocker.
        thin = pd.read_csv(_NOV).drop(columns=["Loan Amount", "Facility"])
        _o, rep = prepare_pipeline_mi_dataset(thin, source_file=_NOV.name)
        self.assertNotIn("current_outstanding_balance", rep["metrics_available"])
        codes = [d["check"] for d in rep["data_quality"]]
        self.assertIn("missing_economic_amount", codes)


# --------------------------------------------------------------------------- #
# 4. Funded-correlation fields
# --------------------------------------------------------------------------- #
class TestFundedCorrelation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.out, cls.report = _prep_nov()
        cls.corr = field_correlation_to_funded(cls.out)

    def test_amount_maps_to_forecast_amount_concept(self):
        fr = forecast_readiness(self.out)
        self.assertEqual(fr["economic_amount_field"], "expected_funded_amount")
        self.assertTrue(self.out["expected_funded_amount"].notna().any())
        self.assertIn("current_outstanding_balance",
                      self.corr["current_outstanding_balance"]["funded_correlation"])

    def test_region_maps_to_funded_geography(self):
        self.assertIn("geographic_region_obligor",
                      self.corr["collateral_geography"]["funded_correlation"])
        self.assertTrue(self.corr["collateral_geography"]["available"])

    def test_broker_maps_to_funded_channel(self):
        self.assertIn("broker_channel", self.corr["broker_channel"]["funded_correlation"])
        self.assertIn("origination_channel", self.corr["broker_channel"]["funded_correlation"])

    def test_ltv_maps_to_funded_ltv(self):
        self.assertIn("current_loan_to_value",
                      self.corr["current_loan_to_value"]["funded_correlation"])
        self.assertTrue(self.corr["current_loan_to_value"]["available"])

    def test_rate_maps_to_funded_rate(self):
        self.assertIn("current_interest_rate",
                      self.corr["current_interest_rate"]["funded_correlation"])

    def test_valuation_maps_to_funded_valuation(self):
        self.assertIn("current_valuation_amount",
                      self.corr["current_valuation_amount"]["funded_correlation"])


# --------------------------------------------------------------------------- #
# 5. Separation from the funded central lender tape
# --------------------------------------------------------------------------- #
class TestSeparationFromFunded(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.out, cls.report = _prep_nov()

    def test_every_row_tagged_pipeline(self):
        self.assertTrue((self.out["record_type"] == "pipeline").all())
        self.assertEqual(self.report["record_type"], "pipeline")

    def test_pipeline_prep_is_independent_of_funded_prep(self):
        # Pipeline preparation does not invoke / mutate the funded tape path.
        from mi_agent_api.funded_prep import prepare_funded_mi_dataset
        funded = pd.DataFrame({
            "loan_identifier": [1, 2, 3],
            "current_outstanding_balance": [100000.0, 120000.0, 90000.0],
        })
        fout, _frep = prepare_funded_mi_dataset(funded)
        # The funded dataset has no pipeline-only columns leaking in.
        for col in ("pipeline_stage", "record_type", "expected_funded_amount"):
            self.assertNotIn(col, fout.columns)
        # And the funded loan identifiers never appear in the pipeline dataset.
        self.assertNotIn("loan_identifier", self.out.columns)


# --------------------------------------------------------------------------- #
# 6. API metadata
# --------------------------------------------------------------------------- #
class TestPipelineApiMetadata(unittest.TestCase):
    def setUp(self):
        os.environ["MI_AGENT_PIPELINE_ROOT"] = str(_FIXTURE_PACK)

    def tearDown(self):
        os.environ.pop("MI_AGENT_PIPELINE_ROOT", None)

    def _client(self):
        from fastapi.testclient import TestClient
        from mi_agent_api.app import app
        return TestClient(app)

    def test_snapshots_endpoint_lists_sources(self):
        body = self._client().get("/mi/pipeline/snapshots",
                                  params={"portfolioId": "client_001"}).json()
        self.assertGreaterEqual(len(body["sources"]), 2)

    def test_snapshot_returns_metrics_dimensions_and_data_quality(self):
        body = self._client().get(
            "/mi/pipeline/snapshot",
            params={"portfolioId": "client_001/pipeline_2025_11"}).json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["recordType"], "pipeline")
        self.assertEqual(body["reportingDate"], "2025-11-01")
        self.assertGreater(body["pipelineRowCount"], 0)
        self.assertGreater(body["pipelineAmount"], 0)
        self.assertIsNotNone(body["weightedExpectedFundedAmount"])
        self.assertTrue(body["availableMetrics"])
        self.assertTrue(body["availableDimensions"])
        self.assertIn("pipeline_stage", body["availableDimensions"])
        self.assertIn("dataQuality", body)
        self.assertTrue(body["stageBreakdown"])
        self.assertTrue(body["expectedCompletionBreakdown"])

    def test_snapshot_exposes_field_correlation_to_funded(self):
        body = self._client().get(
            "/mi/pipeline/snapshot",
            params={"portfolioId": "client_001/pipeline_2025_11"}).json()
        corr = body["fieldCorrelationToFunded"]
        self.assertIn("geographic_region_obligor",
                      corr["collateral_geography"]["funded_correlation"])


# --------------------------------------------------------------------------- #
# 7. Forecast-readiness metadata
# --------------------------------------------------------------------------- #
class TestForecastReadiness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.out, cls.report = _prep_nov()
        cls.fr = forecast_readiness(cls.out)

    def test_identifies_required_forecast_fields(self):
        self.assertEqual(self.fr["economic_amount_field"], "expected_funded_amount")
        self.assertEqual(self.fr["baseline_completion_probability_field"],
                         "completion_probability")
        self.assertIn(self.fr["expected_completion_date_field"],
                      ("expected_completion_date",))
        self.assertTrue(self.fr["fields_available"]["expected_amount"])
        self.assertTrue(self.fr["fields_available"]["completion_probability"])

    def test_documents_forecast_formula(self):
        self.assertIn("current_funded_balance", self.fr["formula"])
        self.assertIn("completion_probability", self.fr["formula"])

    def test_correlation_fields_for_forecast(self):
        corr = self.fr["correlation_fields"]
        for axis in ("amount", "ltv", "valuation", "region",
                     "broker_channel", "rate", "borrower_age", "product"):
            self.assertIn(axis, corr, axis)
        self.assertTrue(all(self.fr["correlation_fields_available"].values()))

    def test_completion_probability_from_config_not_invented(self):
        # Probabilities come from config/client/pipeline_expected_funding.yaml.
        kfi = self.out[self.out["pipeline_stage"] == "KFI"]
        self.assertTrue((kfi["completion_probability"] == 0.20).all())
        self.assertTrue(self.report["weighted_expected_funded_amount"] <
                        self.report["expected_funded_amount"])

    def test_diagnostics_partitioned_by_severity(self):
        groups = diagnostics_by_severity(self.report)
        self.assertEqual(set(groups.keys()), {"blocker", "warning", "info"})
        self.assertFalse(groups["blocker"])  # clean fixture has no blockers


if __name__ == "__main__":
    unittest.main(verbosity=2)
