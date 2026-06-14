#!/usr/bin/env python3
"""tests/test_onboarding_alias_integration.py — PART 9 (1–9)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TESTS_DIR = Path(__file__).resolve().parent
for _p in (str(_REPO_ROOT), str(_TESTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from onboarding_domain_fixtures import ALIASES, REGISTRY, SCENARIO_A, build_run
from engine.gate_1_alignment.semantic_alignment import (
    HeaderMapper,
    load_aliases_from_dir,
    load_field_registry,
    select_registry_fields,
)
from engine.onboarding_agent import mapping_trace


def _mapper():
    registry = load_field_registry(REGISTRY)
    fields = select_registry_fields(registry, "equity_release")
    alias_map = load_aliases_from_dir(ALIASES)
    return HeaderMapper(fields, alias_map)


class TestAliasLoading(unittest.TestCase):
    # 1. Existing alias files are loaded by the Onboarding Agent.
    def test_alias_libraries_loaded(self):
        idx = mapping_trace.AliasIndex.load(ALIASES)
        for name in ("aliases_mandatory.yaml", "aliases_optional.yaml",
                     "aliases_analytics.yaml"):
            self.assertIn(name, idx.files_loaded)
        self.assertGreater(len(idx.by_norm), 100)
        # The shared Gate-1 loader returns the same non-empty map.
        self.assertTrue(load_aliases_from_dir(ALIASES))


class TestTraceProbe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project, cls.pdir, cls.rp = build_run(SCENARIO_A, ingest=False)
        rows = json.loads((cls.pdir / "05c_mapping_trace.json").read_text())["rows"]
        cls.by_col = {r["source_column"]: r for r in rows}

    # 2. Mapping trace records loaded alias files.
    def test_trace_lists_alias_files(self):
        self.assertIn("aliases_mandatory.yaml",
                      self.project.mapping_trace_summary["alias_files_loaded"])

    # 3. original principal -> original_principal_balance (existing/added alias).
    def test_original_principal(self):
        r = self.by_col["original_principal"]
        self.assertEqual(r["selected_candidate"], "original_principal_balance")
        self.assertEqual(r["selection_reason"], "alias_match")
        self.assertTrue(r["alias_hit"])

    # 4. current balance -> current_principal_balance by alias.
    def test_current_balance(self):
        r = self.by_col["current_balance"]
        self.assertEqual(r["selected_candidate"], "current_principal_balance")
        self.assertEqual(r["selection_reason"], "alias_match")
        self.assertTrue(r["alias_hit"])

    # 5. principal outstanding -> current_principal_balance (alias or context).
    def test_principal_outstanding(self):
        r = self.by_col["principal_outstanding"]
        self.assertEqual(r["selected_candidate"], "current_principal_balance")
        self.assertIn(r["selection_reason"], ("alias_match", "domain_context", "value_match"))

    # 6. valuation amount -> current_valuation_amount.
    def test_valuation_amount(self):
        r = self.by_col["valuation_amount"]
        self.assertEqual(r["selected_candidate"], "current_valuation_amount")
        self.assertIn(r["selection_reason"], ("alias_match", "domain_context"))

    # 7. property postcode / property post code -> a postcode canonical field.
    # (The registry has BOTH property_post_code and property_postcode; either is
    # a correct, non-silent deterministic resolution.)
    def test_property_postcode_variants(self):
        mapper = _mapper()
        for header in ("property_post_code", "property postcode", "property post code"):
            canon, method, conf = mapper.map_one(header)
            self.assertIn(canon, ("property_post_code", "property_postcode"),
                          f"{header} -> {canon}")
            self.assertIn(method, ("exact", "normalized", "alias"))

    # 8. If an alias exists but does not fire, fail. current_balance MUST fire.
    def test_alias_must_fire(self):
        mapper = _mapper()
        canon, method, conf = mapper.map_one("current_balance")
        self.assertEqual(canon, "current_principal_balance")
        self.assertEqual(method, "alias")

    # 9. Genuinely absent alias -> unmapped_alias_missing / ambiguous (not silent).
    def test_absent_alias_is_explained(self):
        for col in ("loan_amount", "initial_advance"):
            r = self.by_col[col]
            self.assertIn(r["final_status"], ("unmapped", "ambiguous_needs_review"))
            if r["final_status"] == "unmapped":
                self.assertIn("unmapped_alias_missing", r["unmapped_reason"])
            self.assertFalse(r["alias_hit"])


if __name__ == "__main__":
    unittest.main()
