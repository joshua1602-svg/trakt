#!/usr/bin/env python3
"""tests/test_onboarding_mapping_quality.py — deterministic quality fixes.

Covers: header re-detection, cashflow/ledger grouping, MI-useful in-scope fields,
header-trust type override, LLM visibility columns, and concise grouping.
"""

from __future__ import annotations

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

from engine.gate_1_alignment.semantic_alignment import load_field_registry
from engine.onboarding_agent import mapping_backstop_validator as bv
from engine.onboarding_agent import mapping_candidate_finder as finder
from engine.onboarding_agent import source_table_loader as stl
from engine.onboarding_agent.field_scope import resolve_field_scope
from engine.onboarding_agent.llm_assisted_mapping import run_llm_assisted_mapping
from engine.onboarding_agent.mode_policy import load_mode_policy

REGISTRY = str(_REPO_ROOT / "config" / "system" / "fields_registry.yaml")


def _reg():
    return load_field_registry(Path(REGISTRY)).get("fields", {})


class TestHeaderRedetection(unittest.TestCase):
    def test_redetect_header_row(self):
        df = pd.DataFrame([["EXPORT", None, None],
                           ["Loan Id", "Balance", "Region"],
                           ["L1", 100, "NW"], ["L2", 200, "SE"]])
        # Naive read leaves integer/Unnamed-like columns; rescan finds the header.
        df.columns = ["Unnamed: 0", "Unnamed: 1", "Unnamed: 2"]
        out, hr, failed = stl.redetect_header(df)
        self.assertFalse(failed)
        self.assertEqual(list(out.columns), ["Loan Id", "Balance", "Region"])
        self.assertEqual(len(out), 2)


class TestCashflowAndMiUseful(unittest.TestCase):
    def test_cashflow_ledger_candidate(self):
        ev = {"source_file": "f.csv", "source_column": "B/F Principal Balance",
              "data_type_guess": "amount", "candidate_value_profile_matches": "amount"}
        rows = finder.build_candidate_shortlist([ev], _reg(),
                                                resolve_field_scope(REGISTRY, load_mode_policy("mi_only")))
        self.assertTrue(rows)
        self.assertEqual(rows[0]["candidate_source"], "cashflow_ledger")
        self.assertTrue(rows[0]["candidate_target_field"].startswith("cf_"))

    def test_mi_useful_in_scope_for_mi_only(self):
        ev = {"source_file": "f.csv", "source_column": "Post Code",
              "data_type_guess": "postcode"}
        fs = resolve_field_scope(REGISTRY, load_mode_policy("mi_only"))
        rows = finder.build_candidate_shortlist([ev], _reg(), fs,
                                                extra_in_scope=finder.MI_RELEVANT_FIELDS)
        r = next(r for r in rows if r["candidate_target_field"] == "property_post_code")
        self.assertEqual(r["candidate_source"], "mi_useful")
        self.assertEqual(r["field_scope_status"], "in_scope_mi_useful")


class TestHeaderTrustTypeOverride(unittest.TestCase):
    def test_strong_header_overrides_type(self):
        # An exact pipeline-contract match with an incompatible value profile must
        # route to review, NOT a hard unsafe block.
        rows = bv.validate_mappings([{
            "source_column": "Account Number", "proposed_target_field": "account_number",
            "candidate_source": "pipeline_contract", "confidence": "high",
            "type_compatible": False, "is_pipeline_field": True,
        }], registry_fields=_reg(),
            field_scope=resolve_field_scope(REGISTRY, load_mode_policy("mi_only")))
        self.assertEqual(rows[0]["validation_status"], bv.REVIEW_REQUIRED)
        self.assertNotEqual(rows[0]["validation_status"], bv.UNSAFE)

    def test_weak_match_type_incompatible_still_unsafe(self):
        rows = bv.validate_mappings([{
            "source_column": "X", "proposed_target_field": "current_principal_balance",
            "candidate_source": "semantic_alignment", "confidence": "high",
            "type_compatible": False,
        }], registry_fields=_reg(),
            field_scope=resolve_field_scope(REGISTRY, load_mode_policy("regulatory_mi")))
        self.assertEqual(rows[0]["validation_status"], bv.UNSAFE)


class TestLlmVisibilityAndGrouping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        d = Path(tempfile.mkdtemp())
        cls.cf = d / "Funder PandI.csv"
        pd.DataFrame({"Loan Policy Number": ["L1", "L2"],
                      "B/F Current Balance": [100, 200],
                      "Payment_Allocation - Principal": [10, 20],
                      "Post Code": ["SW1A 2AA", "M1 4WX"]}).to_csv(cls.cf, index=False)
        cls.out = Path(tempfile.mkdtemp())
        cls.res = run_llm_assisted_mapping(input_file=str(cls.cf), output_dir=str(cls.out),
                                           mode="mi_only", client_id="c", run_id="r1")
        cls.queue = pd.read_csv(cls.out / "33_mapping_review_queue.csv")

    def test_llm_columns_present(self):
        for c in ("llm_reviewed", "llm_suggested_mapping", "llm_recommendation",
                  "llm_rationale", "llm_confidence", "llm_decision", "llm_review_status",
                  "llm_batch_id", "backstop_decision", "backstop_rejection_reason"):
            self.assertIn(c, self.queue.columns)

    def test_cashflow_grouped(self):
        groups = set(self.queue["group"])
        self.assertIn("cashflow_ledger_extension_candidates", groups)
        cf = self.queue[self.queue["group"] == "cashflow_ledger_extension_candidates"]
        self.assertTrue(any("Balance" in c or "Allocation" in c for c in cf["source_column"]))

    def test_mi_useful_not_out_of_scope(self):
        pc = self.queue[self.queue["source_column"] == "Post Code"].iloc[0]
        self.assertEqual(pc["suggested_mapping"], "property_post_code")
        self.assertNotEqual(pc["validation_status"], "out_of_scope")

    def test_queue_has_new_columns_for_file_and_domain(self):
        for c in ("source_file", "source_sheet", "domain_guess", "file_domain_guess"):
            self.assertIn(c, self.queue.columns)


if __name__ == "__main__":
    unittest.main()
