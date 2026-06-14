#!/usr/bin/env python3
"""tests/test_onboarding_semantic_alignment_parity.py — PART 10 (1-5, 11, 14, 15).

Proves the parity audit works, the new onboarding path does NOT drop the Gate 1
semantic-alignment tiers, the adapter reuses the existing engine, field scope is
enforced, the LLM stays unused, and the trace records semantic_alignment_used.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TESTS_DIR = Path(__file__).resolve().parent
for _p in (str(_REPO_ROOT), str(_TESTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engine.onboarding_agent import compare_semantic_alignment as csa
from engine.onboarding_agent import semantic_alignment_adapter as adapter
from engine.onboarding_agent.field_scope import resolve_field_scope
from engine.onboarding_agent.mode_policy import load_mode_policy

REGISTRY = str(_REPO_ROOT / "config" / "system" / "fields_registry.yaml")
ALIASES = str(_REPO_ROOT / "config" / "system")
FIXTURE = str(_REPO_ROOT / "tests" / "fixtures" / "kfi_pipeline_headers.csv")


class TestParityAudit(unittest.TestCase):
    # 1. Parity audit produces CSV, JSON and markdown summary.
    def test_artefacts_written(self):
        out = Path(tempfile.mkdtemp())
        res = csa.run_and_write(FIXTURE, REGISTRY, ALIASES, out, mode="regulatory_mi")
        for k in ("csv", "json", "summary_md"):
            self.assertTrue(Path(res["paths"][k]).exists())
        self.assertEqual(Path(res["paths"]["csv"]).name, "27_semantic_alignment_parity.csv")
        data = json.loads(Path(res["paths"]["json"]).read_text())
        self.assertIn("rows", data)
        self.assertIn("summary", data)

    # 2. The same headers run through both old and new paths.
    def test_runs_both_paths(self):
        res = csa.run_parity(FIXTURE, REGISTRY, ALIASES, mode="regulatory_mi")
        rows = res["rows"]
        self.assertEqual(len(rows), 34)
        for r in rows:
            self.assertIn("old_semantic_alignment_candidate", r)
            self.assertIn("new_onboarding_candidate", r)
            self.assertIn("reason_for_difference", r)

    # 3. The parity audit detects the old/new relationship per column. The honest
    #    finding for this file is ZERO true regressions (the new path reuses the
    #    same Gate 1 HeaderMapper), and KFI fields are flagged registry_target_missing.
    def test_detects_no_regression_and_categorises(self):
        res = csa.run_parity(FIXTURE, REGISTRY, ALIASES, mode="regulatory_mi")
        s = res["summary"]
        self.assertEqual(s["old_mapped_new_unmapped"], 0,
                         "premise disproven: no true semantic-tier regression")
        # The audit still actively categorises the gap.
        missing = [r for r in res["rows"]
                   if r["reason_for_difference"] == "both_unmapped_registry_target_missing"]
        self.assertTrue(missing, "KFI/pipeline fields should be flagged registry_target_missing")

    # 3b. The detection MECHANISM works: in mi_only the audit detects columns the
    #     old path maps that the new path diverts out of scope (mode-safe).
    def test_mi_only_field_scope_diversion_detected(self):
        res = csa.run_parity(FIXTURE, REGISTRY, ALIASES, mode="mi_only")
        self.assertGreater(res["summary"]["field_scope_excluded"], 0)
        oos = [r for r in res["rows"] if r["new_field_scope_status"] == "out_of_scope"]
        self.assertTrue(all(r["old_semantic_alignment_candidate"] for r in oos))

    # 5. Loan Amount is not SILENTLY unmapped: the audit flags it as intentionally
    #    ambiguous, and old and new agree (neither maps the documented-ambiguous term).
    def test_loan_amount_flagged_not_silent(self):
        res = csa.run_parity(FIXTURE, REGISTRY, ALIASES, mode="regulatory_mi")
        row = next(r for r in res["rows"] if r["source_column"] == "Loan Amount")
        self.assertFalse(row["old_semantic_alignment_candidate"])
        self.assertFalse(row["new_onboarding_candidate"])
        self.assertEqual(row["reason_for_difference"],
                         "both_unmapped_intentionally_ambiguous")

    # 14. LLM remains unused by default in the parity audit.
    def test_llm_unused(self):
        res = csa.run_parity(FIXTURE, REGISTRY, ALIASES, mode="regulatory_mi")
        self.assertFalse(res["summary"]["llm_used"])


class TestAdapter(unittest.TestCase):
    def _mapper(self):
        mapper, _ = adapter.build_header_mapper(REGISTRY, ALIASES)
        return mapper

    # 4. The adapter reuses the existing engine and surfaces the fuzzy semantic tiers.
    def test_adapter_maps_fuzzy_tier(self):
        mapper = self._mapper()
        res = adapter.align_header(mapper, "Original Principal Balances")
        self.assertEqual(res.candidate, "original_principal_balance")
        self.assertIn(res.method, ("fuzz_ratio_norm", "fuzz_token_set", "token_set"))
        self.assertTrue(res.semantic_alignment_used)

    def test_run_for_headers_returns_candidates(self):
        cands = adapter.run_semantic_alignment_for_headers(
            ["Original Principal Balances", "Maturity Dt", "Totally Unknown Column"],
            REGISTRY, ALIASES, mode="regulatory_mi")
        by_col = {c.source_column: c for c in cands}
        self.assertEqual(by_col["Original Principal Balances"].candidate_canonical_field,
                         "original_principal_balance")
        self.assertEqual(by_col["Totally Unknown Column"].candidate_canonical_field, "")

    # 11. The adapter is field-scope safe (mode-aware): an out-of-scope target is
    #     flagged, never silently promoted.
    def test_adapter_field_scope_safe(self):
        mapper = self._mapper()
        fs = resolve_field_scope(REGISTRY, load_mode_policy("mi_only"))
        # current_valuation_amount is regulatory non-core -> excluded in mi_only.
        res = adapter.align_header(mapper, "Current Valuation Amount", field_scope=fs)
        self.assertEqual(res.candidate, "current_valuation_amount")
        self.assertEqual(res.field_scope_status, "out_of_scope")


class TestTraceRecordsSemanticAlignment(unittest.TestCase):
    # 15. The mapping trace records semantic_alignment_used for fuzzy-tier matches.
    def test_trace_records_semantic_alignment_used(self):
        from engine.onboarding_agent import mapping_trace
        from engine.onboarding_agent.mapping_proposer import propose_mappings
        from engine.onboarding_agent.onboarding_models import FileInventoryItem
        from engine.gate_1_alignment.semantic_alignment import load_field_registry

        headers = ["Original Principal Balances", "Origination Dates", "Maturity Dt"]
        df = pd.DataFrame({h: ["1"] for h in headers})
        tmp = Path(tempfile.mkdtemp()) / "fuzz.csv"
        df.to_csv(tmp, index=False)
        inv = [FileInventoryItem(file_path=str(tmp), file_name="fuzz.csv",
                                 file_type="csv", classification="loan_report")]
        fs = resolve_field_scope(REGISTRY, load_mode_policy("regulatory_mi"))
        cands, oos, amb = propose_mappings(inv, {str(tmp): df}, Path(REGISTRY),
                                           Path(ALIASES), field_scope=fs)
        reg_fields = load_field_registry(Path(REGISTRY)).get("fields", {})
        trace = mapping_trace.build_trace(
            inventory=inv, dataframes={str(tmp): df}, mapping_candidates=cands,
            out_of_scope_fields=oos, mapping_ambiguities=amb, overlap_analysis=[],
            field_scope=fs, registry_fields=reg_fields, aliases_dir=ALIASES,
            llm_suggestions=[], precedence={}, profiles=[])
        sem_rows = [r for r in trace["rows"] if r["semantic_alignment_used"]]
        self.assertTrue(sem_rows, "expected at least one semantic_alignment_used row")
        self.assertGreaterEqual(trace["summary"]["mapped_by_semantic_alignment"], 1)
        for r in sem_rows:
            self.assertIn(r["semantic_alignment_method"],
                          ("token_set", "fuzz_token_set", "fuzz_ratio_norm"))


if __name__ == "__main__":
    unittest.main()
