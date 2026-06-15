#!/usr/bin/env python3
"""tests/test_onboarding_asset_regime_resolver.py — asset/regime-aware resolver v1.

Context detection (27), required target contract (28_required), LLM resolver
against the contract (31_resolver), and contract coverage in the backstop/queue.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engine.onboarding_agent import onboarding_context as octx
from engine.onboarding_agent import required_target_contract as rtc
from engine.onboarding_agent.llm_assisted_mapping import run_llm_assisted_mapping

KFI = str(_REPO_ROOT / "tests" / "fixtures" / "kfi_pipeline_headers.csv")


class TestContextDetection(unittest.TestCase):
    def test_detect_equity_release_uk(self):
        inv = [{"file_name": "M2L KFI and Pipeline ERE.xlsx", "classification": "",
                "domains_detected": []},
               {"file_name": "Funder Principal And Interest.csv", "classification": "",
                "domains_detected": []}]
        ev = [{"source_column": "Lifetime Mortgage Product",
               "sample_values_distinct_redacted": "Lifetime Mortgage Lump Sum"},
              {"source_column": "Post Code", "sample_values_distinct_redacted": "SW1A 2AA"}]
        ctx = octx.detect_context(inv, ev, mode="mi_only")
        self.assertEqual(ctx["asset_class"], "equity_release_mortgage")
        self.assertEqual(ctx["jurisdiction"], "UK")
        self.assertEqual(ctx["reporting_regime"], "mi_only")
        self.assertIn("cashflow_ledger", ctx["required_domains"])
        self.assertIn("pipeline", ctx["required_domains"])


class TestRequiredContract(unittest.TestCase):
    def test_contract_spans_domains(self):
        ctx = octx.detect_context([], [], mode="mi_only")
        contract = rtc.build_required_contract(ctx)
        domains = {r["domain"] for r in contract}
        # MI is NOT pipeline-only: funded loan, property, cashflow are present.
        for d in ("funded_loan", "collateral_property", "cashflow_ledger", "pipeline"):
            self.assertIn(d, domains)
        # MI-useful fields are in the contract.
        fields = rtc.contract_field_set(contract)
        for f in ("current_valuation_amount", "property_post_code", "redemption_date",
                  "original_underlying_exposure_identifier", "original_valuation_amount"):
            self.assertIn(f, fields)

    def test_artefacts_written(self):
        ctx = octx.detect_context([], [], mode="mi_only")
        out = Path(tempfile.mkdtemp())
        paths = rtc.write_contract_artifacts(rtc.build_required_contract(ctx), out)
        for k in ("csv", "json", "summary_md"):
            self.assertTrue(Path(paths[k]).exists())


class TestResolverEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.out = Path(tempfile.mkdtemp())
        cls.res = run_llm_assisted_mapping(input_file=KFI, output_dir=str(cls.out),
                                           mode="mi_only", client_id="c", run_id="r1")

    def test_artefacts(self):
        for name in ("27_onboarding_context.json", "28_required_target_contract.csv",
                     "31_llm_mapping_resolver.csv", "32_mapping_backstop_validation.csv",
                     "33_mapping_review_queue.csv"):
            self.assertTrue((self.out / name).exists(), name)

    def test_context_and_contract_in_result(self):
        self.assertEqual(self.res["context"]["asset_class"], "equity_release_mortgage")
        self.assertGreater(len(self.res["required_contract"]), 20)
        self.assertIn("mandatory_total", self.res["contract_coverage"])

    def test_resolver_decisions(self):
        decisions = {r["decision"] for r in self.res["resolver"]["resolved"]}
        self.assertIn("map_existing_target", decisions)

    def test_backstop_has_audit_columns(self):
        v = pd.read_csv(self.out / "32_mapping_backstop_validation.csv")
        for c in ("llm_suggested_mapping", "final_mapping", "final_status",
                  "backstop_decision", "backstop_rejection_reason"):
            self.assertIn(c, v.columns)

    def test_llm_resolver_with_fake_llm(self):
        def fake(prompt):
            return json.dumps([{"source_file": Path(KFI).name, "source_column": "Loan Plan",
                                "resolved_target_field": "product_type",
                                "decision": "map_existing_target", "confidence": 0.9,
                                "rationale": "plan == product subtype"}])
        out = Path(tempfile.mkdtemp())
        res = run_llm_assisted_mapping(input_file=KFI, output_dir=str(out), mode="mi_only",
                                       client_id="c", run_id="r1",
                                       enable_llm=True, llm_callable=fake)
        self.assertTrue(res["resolver"]["usage"]["llm_enabled"])
        self.assertGreaterEqual(res["resolver"]["usage"]["calls_completed"], 1)
        # The LLM-resolved row is recorded with llm_used + a batch id.
        rr = {r["source_column"]: r for r in res["resolver"]["resolved"]}
        self.assertTrue(rr["Loan Plan"]["llm_used"])
        self.assertTrue(rr["Loan Plan"]["llm_batch_id"])


class TestLlmContextResolver(unittest.TestCase):
    def _good_ctx(self, prompt):
        return json.dumps({
            "asset_class": "equity_release_mortgage", "jurisdiction": "UK",
            "product_type": "lifetime_mortgage", "reporting_regime": "mi_only",
            "use_cases": ["portfolio_mi", "cashflow_monitoring"],
            "required_domains": ["pipeline", "funded_loan", "collateral_property",
                                 "cashflow_ledger"],
            "suggested_target_contract": "uk_equity_release_mi_v1", "confidence": 0.92,
            "rationale": "KFI/pipeline + cashflow + property indicate UK ERM MI",
            "supporting_evidence": ["KFI and Pipeline"], "open_questions": []})

    def _bad_ctx(self, prompt):
        return json.dumps({"asset_class": "consumer_loan", "jurisdiction": "US",
                           "reporting_regime": "mi_only", "confidence": 0.9,
                           "required_domains": ["funded_loan"]})

    def test_llm_context_accepted_and_artefacts(self):
        out = Path(tempfile.mkdtemp())
        res = run_llm_assisted_mapping(input_file=KFI, output_dir=str(out), mode="mi_only",
                                       client_id="c", run_id="r1",
                                       enable_context_resolver=True,
                                       context_llm_callable=self._good_ctx)
        self.assertTrue((out / "27a_deterministic_context_guess.json").exists())
        self.assertTrue((out / "27b_llm_context_resolution.json").exists())
        self.assertEqual(res["context"]["final_context_source"], "llm")
        self.assertEqual(res["context"]["context_backstop_decision"], "accepted_llm")
        self.assertEqual(res["context_usage"]["calls_completed"], 1)
        # 27b records the LLM's resolution.
        llm = json.loads((out / "27b_llm_context_resolution.json").read_text())
        self.assertEqual(llm["asset_class"], "equity_release_mortgage")

    def test_conflicting_llm_downgraded_to_user_confirmation(self):
        out = Path(tempfile.mkdtemp())
        res = run_llm_assisted_mapping(input_file=KFI, output_dir=str(out), mode="mi_only",
                                       client_id="c", run_id="r1",
                                       enable_context_resolver=True,
                                       context_llm_callable=self._bad_ctx)
        ctx = res["context"]
        # Unsupported asset -> backstop downgrades to deterministic and asks the user.
        self.assertEqual(ctx["context_backstop_decision"], "downgraded_to_deterministic")
        self.assertEqual(ctx["asset_class"], "equity_release_mortgage")
        self.assertTrue(ctx["needs_user_confirmation"])
        self.assertTrue(ctx["open_questions"])

    def test_contract_selected_from_context_not_mode(self):
        from engine.onboarding_agent import onboarding_context as octx
        ctx = octx.detect_context([], [], mode="regulatory_mi")
        ctx["reporting_regime"] = "esma_annex_12"
        self.assertEqual(octx.select_target_contract(ctx),
                         "uk_equity_release_esma_annex12_v1")
        contract = rtc.build_required_contract(
            {**ctx, "selected_target_contract": "uk_equity_release_esma_annex12_v1"})
        fields = rtc.contract_field_set(contract)
        self.assertIn("geographic_region_classification", fields)


class TestWorkbenchContextPanel(unittest.TestCase):
    def test_load_context(self):
        from engine.onboarding_agent import streamlit_onboarding_workbench as wb
        out = Path(tempfile.mkdtemp())
        run_llm_assisted_mapping(input_file=KFI, output_dir=str(out), mode="mi_only",
                                 client_id="c", run_id="r1")
        data = wb.load_onboarding_context(out)
        self.assertTrue(data["final"])
        self.assertIn("final_context_source", data["final"])
        self.assertIn("selected_target_contract", data["final"])


if __name__ == "__main__":
    unittest.main()
