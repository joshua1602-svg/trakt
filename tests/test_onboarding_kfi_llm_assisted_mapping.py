#!/usr/bin/env python3
"""tests/test_onboarding_kfi_llm_assisted_mapping.py — PART 13 (23, 26, 27, 29) + KFI results."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engine.onboarding_agent.llm_assisted_mapping import run_llm_assisted_mapping

FIXTURE = str(_REPO_ROOT / "tests" / "fixtures" / "kfi_pipeline_headers.csv")

# Pipeline-only fields that must NEVER appear in the central LENDER (funded) tape.
_PIPELINE_ONLY = {"offer_date", "application_submitted_date", "date_funds_released",
                  "kfi_number", "status_raw", "pipeline_stage", "kfi_submitted_date"}


def _run(mode="regulatory_mi", out=None, enable_llm=False, llm=None):
    out = out or Path(tempfile.mkdtemp())
    warnings.simplefilter("ignore")
    return run_llm_assisted_mapping(input_file=FIXTURE, output_dir=str(out), mode=mode,
                                    client_id="kfi", run_id="r1",
                                    enable_llm=enable_llm, llm_callable=llm), out


class TestKfiResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res, cls.out = _run()
        cls.val = {r["source_column"]: r for r in cls.res["validation"]}

    def test_pipeline_targets_mapped(self):
        cases = {
            "Application Submitted Date": "application_submitted_date",
            "Offer Date": "offer_date", "Date Funds Released": "date_funds_released",
            "Status": "status_raw", "Product Rate": "product_rate",
            "Interest Payment Percentage": "interest_payment_percentage",
            "Max Facility": "max_facility",
        }
        for col, tgt in cases.items():
            self.assertEqual(self.val[col]["proposed_target_field"], tgt, col)

    def test_demographic_fields_registry_target_missing(self):
        for col in ("Gender APP 1", "DOB App 1"):
            self.assertEqual(self.val[col]["validation_status"], "registry_target_missing")

    def test_no_regulatory_pollution(self):
        # No KFI/pipeline column is mapped into a regulatory funded field.
        for col, r in self.val.items():
            self.assertNotIn(r["proposed_target_field"],
                             {"original_principal_balance", "current_principal_balance",
                              "account_status"},
                             f"{col} polluted a regulatory field")


class TestLlmUsageAndControl(unittest.TestCase):
    # 23. LLM usage summary records call count + estimated cost (with fake LLM).
    def test_llm_usage_recorded(self):
        def fake_llm(prompt):
            return json.dumps([{"source_column": "Gender APP 1",
                                "proposed_target_field": "", "confidence": "no_match",
                                "proposed_target_source": "registry_target_missing"}])
        res, out = _run(enable_llm=True, llm=fake_llm)
        usage = res["llm"]["usage"]
        self.assertTrue(usage["llm_enabled"])
        self.assertEqual(usage["calls_completed"], 1)
        self.assertGreater(usage["items_sent"], 0)
        self.assertGreater(usage["estimated_cost_gbp"], 0.0)
        self.assertTrue((out / "31_llm_usage_summary.json").exists())

    # LLM-only proposal for a missing-target column never auto-finalises.
    def test_llm_only_proposal_not_auto_approved(self):
        def fake_llm(prompt):
            return json.dumps([{"source_column": "Gender APP 1",
                                "proposed_target_field": "original_principal_balance",
                                "proposed_target_source": "llm_suggested",
                                "confidence": "high"}])
        res, _ = _run(enable_llm=True, llm=fake_llm)
        val = {r["source_column"]: r for r in res["validation"]}
        # Even a high-confidence LLM proposal to a regulatory field is routed to
        # review (never auto-approved) by the deterministic backstop.
        self.assertNotEqual(val["Gender APP 1"]["validation_status"], "auto_approved_candidate")


class TestFieldScopeModes(unittest.TestCase):
    # 29. Field scope still works across modes.
    def test_field_scope_enforced(self):
        reg_res, _ = _run(mode="regulatory_mi")
        mi_res, _ = _run(mode="mi_only")
        reg_sl = {(r["source_column"], r["candidate_target_field"]): r
                  for r in reg_res["shortlist"]}
        mi_sl = {(r["source_column"], r["candidate_target_field"]): r
                 for r in mi_res["shortlist"]}
        # current_valuation_amount (regulatory non-core) is in scope for regulatory_mi
        # but out of scope for mi_only.
        key = ("Estimated Value", "current_valuation_amount")
        if key in reg_sl:
            self.assertEqual(reg_sl[key]["field_scope_status"], "in_scope")
        if key in mi_sl:
            self.assertEqual(mi_sl[key]["field_scope_status"], "out_of_scope")


class TestTapesNotPolluted(unittest.TestCase):
    # 26/27. Central lender tape is not polluted with pipeline-only fields; the
    #        central pipeline tape carries pipeline fields.
    def test_lender_tape_excludes_pipeline_fields(self):
        from onboarding_domain_fixtures import SCENARIO_A, build_run
        from engine.onboarding_agent import central_tape_builder
        project, pdir, rp = build_run(SCENARIO_A, mode="regulatory_mi", ingest=True)
        res = central_tape_builder.build_central_tapes(
            pdir, rp, str(_REPO_ROOT / "config" / "system" / "fields_registry.yaml"),
            mode="regulatory_mi")
        lender_cols = set(res["lender_summary"]["columns"])
        self.assertFalse(lender_cols & _PIPELINE_ONLY,
                         f"lender tape polluted: {lender_cols & _PIPELINE_ONLY}")
        # Pipeline tape exists and carries pipeline columns.
        self.assertTrue(res["central_pipeline_tape_created"])
        with open(res["central_pipeline_tape_path"], newline="", encoding="utf-8") as fh:
            pipe_cols = set(next(csv.reader(fh)))
        self.assertIn("application_id", pipe_cols)
        self.assertIn("pipeline_stage", pipe_cols)


if __name__ == "__main__":
    unittest.main()
