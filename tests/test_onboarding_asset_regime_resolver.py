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


class TestRobustJsonAndAudit(unittest.TestCase):
    def test_extract_json_from_fenced_and_prose(self):
        from engine.onboarding_agent.llm_json import extract_json
        obj, status, _ = extract_json('```json\n{"a": 1}\n```')
        self.assertEqual(obj, {"a": 1})
        self.assertIn(status, ("ok", "ok_extracted"))
        obj2, status2, _ = extract_json('Sure! Here:\n[{"x": 2}]\nHope that helps.')
        self.assertEqual(obj2, [{"x": 2}])
        obj3, status3, err = extract_json("I am not sure, no JSON here.")
        self.assertIsNone(obj3)
        self.assertEqual(status3, "parse_failed")
        self.assertTrue(err)

    def test_27b_fully_auditable_on_fenced_response(self):
        def fake(prompt):
            return ('```json\n{"asset_class":"equity_release_mortgage","jurisdiction":"UK",'
                    '"product_type":"lifetime_mortgage","reporting_regime":"mi_only",'
                    '"use_cases":["portfolio_mi"],"required_domains":["funded_loan"],'
                    '"suggested_target_contract":"uk_equity_release_mi_v1","confidence":0.9,'
                    '"rationale":"r","supporting_evidence":["e"],"open_questions":[]}\n```')
        out = Path(tempfile.mkdtemp())
        res = run_llm_assisted_mapping(input_file=KFI, output_dir=str(out), mode="mi_only",
                                       client_id="c", run_id="r1",
                                       enable_context_resolver=True, context_llm_callable=fake)
        b = json.loads((out / "27b_llm_context_resolution.json").read_text())
        for k in ("llm_enabled", "calls_completed", "model", "estimated_cost_gbp",
                  "asset_class", "jurisdiction", "reporting_regime", "confidence",
                  "rationale", "supporting_evidence", "open_questions",
                  "raw_response_or_parsed_response", "parse_status", "parse_error"):
            self.assertIn(k, b)
        self.assertIn(b["parse_status"], ("ok", "ok_extracted"))
        self.assertEqual(res["context"]["final_context_source"], "llm")

    def test_context_parse_failure_not_labelled_llm(self):
        def prose(prompt):
            return "It looks like UK equity release but I can't be certain."
        out = Path(tempfile.mkdtemp())
        res = run_llm_assisted_mapping(input_file=KFI, output_dir=str(out), mode="mi_only",
                                       client_id="c", run_id="r1",
                                       enable_context_resolver=True, context_llm_callable=prose)
        self.assertEqual(res["context"]["final_context_source"], "deterministic")
        self.assertEqual(res["context"]["context_backstop_decision"],
                         "llm_unavailable_or_parse_failed")
        b = json.loads((out / "27b_llm_context_resolution.json").read_text())
        self.assertEqual(b["parse_status"], "parse_failed")

    def test_field_resolver_reviews_rows_and_surfaces_in_queue(self):
        def fake(prompt):
            return ('```json\n[{"source_column":"B/F Current Balance",'
                    '"resolved_target_field":"cf_bf_current_balance",'
                    '"decision":"propose_new_target_field","confidence":0.8,'
                    '"rationale":"opening ledger balance"}]\n```')
        out = Path(tempfile.mkdtemp())
        inp = Path(tempfile.mkdtemp()) / "Funder PandI.csv"
        pd.DataFrame({"Loan Policy Number": ["L1", "L2"],
                      "B/F Current Balance": [100, 200],
                      "Made Up Field": ["x", "y"]}).to_csv(inp, index=False)
        res = run_llm_assisted_mapping(input_file=str(inp), output_dir=str(out),
                                       mode="mi_only", client_id="c", run_id="r1",
                                       enable_llm=True, llm_callable=fake, only_unresolved=True)
        self.assertEqual(res["resolver"]["usage"]["calls_completed"], 1)
        self.assertGreaterEqual(res["resolver"]["usage"]["rows_llm_reviewed"], 1)
        q = pd.read_csv(out / "33_mapping_review_queue.csv")
        self.assertGreaterEqual(int((q["llm_reviewed"] == True).sum()), 1)  # noqa: E712
        self.assertNotEqual(set(q["llm_decision"]), {"no_llm"})


class TestFieldResolverWiringAndTelemetry(unittest.TestCase):
    def _echo_fake(self, prompt):
        import re
        cols = re.findall(r'"source_column": "([^"]+)"', prompt)
        return json.dumps([{"source_column": c, "resolved_target_field": "cf_x",
                            "decision": "propose_new_target_field", "confidence": 0.8,
                            "rationale": "ledger"} for c in cols])

    def _many_eligible_file(self):
        cols = {f"B/F Field {i} Balance": [1, 2] for i in range(6)}
        cols.update({f"Unknown Metric {i}": ["a", "b"] for i in range(6)})
        p = Path(tempfile.mkdtemp()) / "Funder PandI.csv"
        pd.DataFrame(cols).to_csv(p, index=False)
        return p

    def test_field_llm_triggers_and_separated_usage(self):
        out = Path(tempfile.mkdtemp())
        res = run_llm_assisted_mapping(input_file=str(self._many_eligible_file()),
                                       output_dir=str(out), mode="mi_only", client_id="c",
                                       run_id="r1", enable_llm=True,
                                       llm_callable=self._echo_fake, only_unresolved=True,
                                       max_llm_items=50)
        u = json.loads((out / "31_llm_resolver_usage_summary.json").read_text())
        self.assertTrue(u["llm_enabled"])
        self.assertEqual(u["context_calls_completed"], 0)
        self.assertEqual(u["field_calls_completed"], 1)
        self.assertGreater(u["eligible_field_rows"], 0)
        self.assertGreater(u["field_rows_reviewed"], 0)
        # Reviewed rows surface in the queue with a real LLM decision.
        q = pd.read_csv(out / "33_mapping_review_queue.csv")
        self.assertGreaterEqual(int((q["llm_reviewed"] == True).sum()), 1)  # noqa: E712
        self.assertNotEqual(set(q["llm_decision"]), {"no_llm"})

    def test_cap_reviews_exactly_cap_when_eligible_exceeds(self):
        out = Path(tempfile.mkdtemp())
        res = run_llm_assisted_mapping(input_file=str(self._many_eligible_file()),
                                       output_dir=str(out), mode="mi_only", client_id="c",
                                       run_id="r1", enable_llm=True,
                                       llm_callable=self._echo_fake, only_unresolved=True,
                                       max_llm_items=8)
        u = json.loads((out / "31_llm_resolver_usage_summary.json").read_text())
        self.assertEqual(u["eligible_field_rows"], 12)
        self.assertEqual(u["field_rows_selected_for_llm"], 8)
        self.assertEqual(u["field_rows_reviewed"], 8)
        self.assertEqual(u["field_rows_skipped_due_to_cap"], 4)

    def test_cap_not_success_when_zero_reviewed(self):
        # LLM returns junk (no JSON) -> calls=1 but 0 reviewed; telemetry is honest.
        out = Path(tempfile.mkdtemp())
        res = run_llm_assisted_mapping(input_file=str(self._many_eligible_file()),
                                       output_dir=str(out), mode="mi_only", client_id="c",
                                       run_id="r1", enable_llm=True,
                                       llm_callable=lambda p: "no json here",
                                       only_unresolved=True, max_llm_items=50)
        u = json.loads((out / "31_llm_resolver_usage_summary.json").read_text())
        self.assertEqual(u["field_calls_completed"], 1)
        self.assertEqual(u["field_rows_reviewed"], 0)
        self.assertEqual(u["field_parse_status"], "parse_failed")

    def test_ineligible_rows_skipped_with_reasons(self):
        # An auto-approved pipeline file: high-confidence rows are NOT sent to LLM.
        out = Path(tempfile.mkdtemp())
        res = run_llm_assisted_mapping(input_file=KFI, output_dir=str(out), mode="mi_only",
                                       client_id="c", run_id="r1", enable_llm=True,
                                       llm_callable=self._echo_fake, only_unresolved=True)
        u = json.loads((out / "31_llm_resolver_usage_summary.json").read_text())
        # Auto-approved/ignored rows are recorded as skipped (not silently dropped).
        self.assertIn("field_rows_skipped_reason_counts", u)


class TestCliSummaryMatchesQueue(unittest.TestCase):
    def test_cli_summary_uses_current_group_keys(self):
        # The summary must read the CURRENT queue group keys (regression: the CLI
        # previously looked up the obsolete 'missing_trakt_fields' key -> 0).
        out = Path(tempfile.mkdtemp())
        res = run_llm_assisted_mapping(input_file=KFI, output_dir=str(out), mode="mi_only",
                                       client_id="c", run_id="r1")
        gc = res["review_queue"]["summary"]["group_counts"]
        self.assertNotIn("missing_trakt_fields", gc)
        # Group keys are the current bulk-decision names.
        self.assertTrue(any(k.startswith("auto_approved") or k.startswith("missing_target")
                            or k.startswith("cashflow") for k in gc))


class TestFieldRowIdJoinDiagnostics(unittest.TestCase):
    def _file(self, n_cf=30, n_unknown=30):
        cols = {f"B/F Field {i} Balance": [1, 2] for i in range(n_cf)}
        cols.update({f"Unknown Metric {i}": ["a", "b"] for i in range(n_unknown)})
        p = Path(tempfile.mkdtemp()) / "Funder.csv"
        pd.DataFrame(cols).to_csv(p, index=False)
        return p

    def _run(self, fake, out, max_items=50):
        return run_llm_assisted_mapping(input_file=str(self._file()), output_dir=str(out),
                                        mode="mi_only", client_id="c", run_id="r1",
                                        enable_llm=True, llm_callable=fake,
                                        only_unresolved=True, max_llm_items=max_items)

    def _echo_all(self, prompt):
        import re
        ids = re.findall(r'"row_id": "(row_\d+)"', prompt)
        cols = re.findall(
            r'"row_id": "row_\d+", "source_file": "([^"]*)", "source_column": "([^"]*)"', prompt)
        res = [{"row_id": rid, "source_file": sf, "source_column": sc,
                "decision": "propose_new_target_field", "resolved_target_field": "cf_x",
                "confidence": 0.8, "rationale": "x"} for rid, (sf, sc) in zip(ids, cols)]
        return "```json\n" + json.dumps({"results": res}) + "\n```"

    # 1. 50 selected, response has 50 results -> 50 matched.
    def test_50_results_50_matched(self):
        out = Path(tempfile.mkdtemp())
        self._run(self._echo_all, out)
        u = json.loads((out / "31_llm_resolver_usage_summary.json").read_text())
        self.assertEqual(u["field_rows_selected_for_llm"], 50)
        self.assertEqual(u["field_results_parsed"], 50)
        self.assertEqual(u["field_results_matched"], 50)
        self.assertEqual(u["field_rows_reviewed"], 50)
        self.assertFalse(u["field_incomplete_response"])

    # 2. response has only 1 result -> parsed=1, matched=1, incomplete=true.
    def test_one_result_incomplete(self):
        import re

        def echo_one(prompt):
            rid = re.findall(r'"row_id": "(row_\d+)"', prompt)[0]
            return json.dumps({"results": [{"row_id": rid, "decision": "propose_new_target_field",
                                            "resolved_target_field": "cf_x", "confidence": 0.8,
                                            "rationale": "x"}]})
        out = Path(tempfile.mkdtemp())
        self._run(echo_one, out)
        u = json.loads((out / "31_llm_resolver_usage_summary.json").read_text())
        self.assertEqual(u["field_results_parsed"], 1)
        self.assertEqual(u["field_results_matched"], 1)
        self.assertTrue(u["field_incomplete_response"])

    # 3. 50 results with row_id but SHUFFLED / no source-column echo -> still matched
    #    purely by row_id (no dependence on column-string matching).
    def test_row_id_join_independent_of_column_strings(self):
        import re

        def echo_rowid_only(prompt):
            ids = re.findall(r'"row_id": "(row_\d+)"', prompt)
            res = [{"row_id": rid, "source_column": "DIFFERENT_NAME",
                    "decision": "propose_new_target_field", "resolved_target_field": "cf_x",
                    "confidence": 0.8, "rationale": "x"} for rid in ids]
            return json.dumps({"results": list(reversed(res))})  # shuffled
        out = Path(tempfile.mkdtemp())
        self._run(echo_rowid_only, out)
        u = json.loads((out / "31_llm_resolver_usage_summary.json").read_text())
        self.assertEqual(u["field_results_matched"], 50)
        self.assertEqual(u["field_results_unmatched"], 0)

    # 4. raw response is persisted for inspection.
    def test_raw_response_persisted(self):
        out = Path(tempfile.mkdtemp())
        self._run(self._echo_all, out)
        raw_path = out / "31_llm_field_raw_response.json"
        self.assertTrue(raw_path.exists())
        data = json.loads(raw_path.read_text())
        self.assertIn("raw_response", data)
        self.assertIn("results", data["raw_response"])


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
