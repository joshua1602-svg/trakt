#!/usr/bin/env python3
"""mi_agent_api/tests/test_funded_realistic_and_pipeline.py

Reframe check: a field must not be excluded from funded MI merely because the
registry category/layer marks it regulatory/collateral. The active MI target
contract + MI enrichment config + the available source fields decide.

Runs the realistic git-tracked pack (synthetic_onboarding_pack) — the closest
reproducible proxy for the real client_001 pack — and the explicit pipeline-only
enrichment rule.

  * region arrives as collateral_region -> collateral_geography and is recognised
    as the region dimension (group), available even though no obligor-region field;
  * current LTV available (explicit current_ltv + derivable from balance/valuation);
  * original LTV -> derivation_inputs_missing (pack has no original valuation);
  * borrower age -> raw_not_found (pack has none);
  * pipeline-only attribute (broker) enriches matched funded loans WITHOUT creating
    funded rows.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from mi_agent_api import funded_mi_trace as T
from mi_agent_api.funded_prep import prepare_funded_mi_dataset

_REGISTRY = str(_REPO_ROOT / "config" / "system" / "fields_registry.yaml")
_PACK = str(_REPO_ROOT / "synthetic_onboarding_pack")


def _promote(input_dir: str, project_dir: Path, run_id: str):
    from engine.onboarding_agent import workflow as wf, storage_paths, central_tape_builder
    wf.run_operator_workflow(
        input_dir=input_dir, client_name="Client 001", client_id="client_001",
        run_id=run_id, mode="mi_only", project_dir=str(project_dir),
        product_profile="equity_release_lifetime_mortgage")
    rp = storage_paths.resolve_run_paths(
        project_dir=str(project_dir), input_dir=input_dir, output_root=None,
        client_id="client_001", run_id=run_id, storage_backend="local",
        input_uri="", output_uri="")
    res = central_tape_builder.build_central_tapes(str(project_dir), rp, _REGISTRY, mode="mi_only")
    return res["central_lender_tape_path"]


class TestRealisticPackTrace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.root = Path(tempfile.mkdtemp(prefix="realpack_"))
        cls.proj = cls.root / "proj"
        cls.tape = _promote(_PACK, cls.proj, "mi_2026_01")
        cls.df = pd.read_csv(cls.tape)
        cls.prep, cls.report = prepare_funded_mi_dataset(cls.df)
        cls.trace = {r["canonical_field"]: r for r in T.trace(cls.proj, cls.tape)}

    def test_universe_built(self):
        self.assertGreater(len(self.df), 0)

    def test_region_available_via_group_alias(self):
        # collateral_region -> collateral_geography; region dimension available.
        self.assertIn("collateral_geography", self.df.columns)
        self.assertIn("geographic_region_obligor", self.report["dimensions_available"])
        self.assertEqual(self.trace["geographic_region_obligor"]["status"], "available")

    def test_current_ltv_available(self):
        self.assertIn("ltv_bucket", self.report["dimensions_available"])
        self.assertEqual(self.trace["current_loan_to_value"]["status"], "available")

    def test_original_ltv_derivation_inputs_missing(self):
        # The pack has original_principal_balance but NO original valuation.
        by = {m["dimension"]: m for m in self.report["missing_dimensions"]}
        self.assertEqual(by["original_ltv_bucket"]["reason"], "derivation_inputs_missing")
        self.assertEqual(self.trace["original_loan_to_value"]["reason"], "derivation_inputs_missing")

    def test_age_raw_not_found(self):
        self.assertEqual(self.trace["youngest_borrower_age"]["reason"], "raw_not_found")

    def test_trace_report_written(self):
        out = self.root / "trace.md"
        T.trace_to_file(self.proj, self.tape, out, run_id="mi_2026_01", client_id="client_001")
        md = out.read_text()
        self.assertIn("geographic_region_obligor", md)
        self.assertIn("derivation_inputs_missing", md)


class TestRegionAndChannelGrouping(unittest.TestCase):
    def test_collateral_geography_satisfies_region(self):
        df = pd.DataFrame({"loan_identifier": [1, 2, 3],
                           "current_outstanding_balance": [100000, 120000, 90000],
                           "collateral_geography": ["UKI", "UKJ", "UKI"]})
        _prep, rep = prepare_funded_mi_dataset(df)
        self.assertIn("geographic_region_obligor", rep["dimensions_available"])

    def test_broker_channel_satisfies_channel(self):
        df = pd.DataFrame({"loan_identifier": [1, 2],
                           "current_outstanding_balance": [100000, 90000],
                           "broker_channel": ["BrokerA", "BrokerB"]})
        prep, rep = prepare_funded_mi_dataset(df)
        self.assertIn("origination_channel", rep["dimensions_available"])
        self.assertIn("origination_channel", prep.columns)  # aliased from broker_channel


class TestPipelineOnlyEnrichment(unittest.TestCase):
    """Pipeline records must not create funded rows, but a configured pipeline
    attribute (broker) may enrich an existing funded loan when entity-key matched."""

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.root = Path(tempfile.mkdtemp(prefix="pe_"))
        inp = cls.root / "input"
        inp.mkdir(parents=True)
        n = 20
        ids = [760000 + i for i in range(n)]
        pd.DataFrame({"Loan Policy Number": ids, "Month Run": ["October"] * n,
                      "Loan Interest Rate": [3.5] * n,
                      "Current Outstanding Balance": [100000.0] * n,
                      "Policy Completion Date": ["2020-01-01"] * n}).to_csv(
            inp / "LoanExtract One.csv", index=False)
        # Pipeline snapshot delivered post-close, SHARES the funded loan key + broker.
        pd.DataFrame({"application_id": [f"A{i}" for i in range(n)],
                      "Loan Policy Number": ids, "application_stage": ["Completed"] * n,
                      "broker": [["BrokerA", "BrokerB"][i % 2] for i in range(n)],
                      "product_name": ["ERM"] * n}).to_csv(
            inp / "M2L KFI and Pipeline 2025_11_01.csv", index=False)
        cls.tape = _promote(str(inp), cls.root / "proj", "mi_2025_10")
        cls.df = pd.read_csv(cls.tape)

    def test_pipeline_does_not_create_funded_rows(self):
        self.assertEqual(len(self.df), 20)
        ids = set(self.df["loan_identifier"].astype(str))
        self.assertFalse(any(i.startswith("A") for i in ids))  # no application ids

    def test_pipeline_attribute_enriches_funded_rows(self):
        self.assertIn("broker_channel", self.df.columns)
        self.assertEqual(int(self.df["broker_channel"].notna().sum()), 20)


class TestDashboardReviewGenerator(unittest.TestCase):
    """The static review generator captures real /mi/query envelopes per run."""

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.root = Path(tempfile.mkdtemp(prefix="review_"))
        inp = cls.root / "input"
        inp.mkdir(parents=True)
        n = 12
        ids = [760000 + i for i in range(n)]
        pd.DataFrame({"Loan Policy Number": ids, "Month Run": ["October"] * n,
                      "Loan Interest Rate": [3.0 + (i % 5) * 0.5 for i in range(n)],
                      "Current Outstanding Balance": [100000.0 + i * 5000 for i in range(n)],
                      "Geo Region": [["UKI", "UKJ", "UKK"][i % 3] for i in range(n)],
                      "Policy Completion Date": [f"20{16 + i % 6}-0{1 + i % 9}-15" for i in range(n)]}).to_csv(
            inp / "LoanExtract One.csv", index=False)
        pd.DataFrame({"Account Number": [s * 100 + 1 for s in ids],
                      "Latest Property Value": [240000.0 + i * 4000 for i in range(n)]}).to_csv(
            inp / "Collateral Extract.csv", index=False)
        # promote under an output-root the generator can resolve by client/run.
        cls.out_root = cls.root / "out"
        _promote(str(inp), cls.out_root / "mi_2025_10", "mi_2025_10")

    def test_collect_run_and_render(self):
        from mi_agent_api.scripts.generate_dashboard_review import collect_run, render
        run = collect_run(self.out_root, "client_001", "mi_2025_10", _REGISTRY)
        self.assertEqual(run["health"]["dataSourceKind"], "funded_mi_prepared_dataset")
        self.assertTrue(run["health"]["preparationApplied"])
        # every funded MI query was attempted and at least summary + ltv/region succeed.
        by_q = {q["question"]: q for q in run["queries"]}
        self.assertTrue(by_q["portfolio summary"]["ok"])
        self.assertTrue(by_q["current outstanding balance by ltv bucket"]["ok"])
        self.assertTrue(by_q["current outstanding balance by region"]["ok"])
        html = render([run], "test")
        self.assertIn("mi_2025_10", html)
        self.assertIn("funded_mi_prepared_dataset", html)
        # data source honestly labelled
        self.assertIn("REAL prepared funded MI dataset", html)


if __name__ == "__main__":
    unittest.main(verbosity=2)
