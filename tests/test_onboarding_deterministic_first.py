#!/usr/bin/env python3
"""tests/test_onboarding_deterministic_first.py — PART 9 (10–15)."""

from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TESTS_DIR = Path(__file__).resolve().parent
for _p in (str(_REPO_ROOT), str(_TESTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from onboarding_domain_fixtures import REGISTRY, SCENARIO_A, build_run
from engine.gate_1_alignment.semantic_alignment import load_field_registry
from engine.onboarding_agent import central_tape_builder
from engine.onboarding_agent.field_scope import resolve_field_scope
from engine.onboarding_agent.llm_mapping_reviewer import (
    _unresolved_candidates,
    build_item,
    validate_suggestion,
)
from engine.onboarding_agent.llm_policy import LLMPolicy
from engine.onboarding_agent.mode_policy import load_mode_policy
from engine.onboarding_agent.onboarding_models import MappingCandidate

REG_FIELDS = load_field_registry(REGISTRY).get("fields", {})


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class TestLLMGating(unittest.TestCase):
    # 10. Alias-resolved mappings are not sent to LLM.
    def test_alias_resolved_not_sent(self):
        policy = LLMPolicy(enabled=True, zero_cost_first=True,
                           deterministic_confidence_above=0.85)
        alias_cand = MappingCandidate(
            source_file="x.csv", source_column="current_balance",
            candidate_canonical_field="current_principal_balance",
            confidence=1.0, method="alias", requires_review=False,
        )
        unmapped_cand = MappingCandidate(
            source_file="x.csv", source_column="mystery", method="unmapped",
            confidence=0.0, requires_review=True,
        )
        unresolved, _ = _unresolved_candidates([alias_cand, unmapped_cand], policy)
        cols = {c.source_column for c in unresolved}
        self.assertNotIn("current_balance", cols)
        self.assertIn("mystery", cols)

    # 11. LLM disabled means zero LLM usage.
    def test_llm_disabled_zero_usage(self):
        project, pdir, _ = build_run(SCENARIO_A, ingest=False)  # LLM off by default
        usage = json.loads((pdir / "22_llm_usage_summary.json").read_text())
        self.assertFalse(usage.get("llm_enabled"))
        self.assertEqual(usage.get("calls_completed", 0), 0)
        self.assertEqual(project.mapping_trace_summary["sent_to_llm"], 0)

    # 12. Out-of-scope MI-only regulatory non-core fields are not sent to LLM.
    def test_mi_only_out_of_scope_not_sent(self):
        scope = resolve_field_scope(REGISTRY, load_mode_policy("mi_only"))
        # property_post_code is regulatory non-core -> excluded in mi_only.
        self.assertTrue(scope.is_excluded("property_post_code"))
        cand = MappingCandidate(
            source_file="x.csv", source_column="property_post_code",
            candidate_canonical_field="property_post_code",
            confidence=0.6, method="token_set", requires_review=True,
        )
        item = build_item(cand, scope, REG_FIELDS, "mi_only",
                          LLMPolicy(enabled=True), {}, False)
        self.assertIsNone(item, "out-of-scope field must not be sent to the LLM")

    # LLM suggestions are rejected if they conflict with field scope.
    def test_suggestion_rejected_if_out_of_scope(self):
        scope = resolve_field_scope(REGISTRY, load_mode_policy("mi_only"))
        item = {"candidate_canonical_fields": [{"field": "property_post_code"}]}
        ok, reason = validate_suggestion(
            {"recommended_canonical_field": "property_post_code", "confidence": 0.99},
            item, scope, "mi_only", REG_FIELDS,
        )
        self.assertFalse(ok)
        self.assertIn("out of scope", reason)


class TestCentralTapeUsesArtefacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project, cls.pdir, cls.rp = build_run(SCENARIO_A, ingest=True)
        cls.res = central_tape_builder.build_central_tapes(
            cls.pdir, cls.rp, str(REGISTRY), mode="regulatory_mi"
        )
        cls.lineage = _read_csv(cls.res["central_tape_lineage_path"])
        cls.candidates = json.loads((cls.pdir / "05_mapping_candidates.json").read_text())

    # 13. Central tape builder uses selected mapping candidates, not remapping.
    def test_no_independent_remapping(self):
        # Every populated (column -> canonical) pair must trace back to a mapping
        # candidate (05) or approved override — never invented by the builder.
        cand_pairs = {(c["source_column"], c["candidate_canonical_field"])
                      for c in self.candidates if c.get("candidate_canonical_field")}
        for r in self.lineage:
            self.assertIn(
                (r["source_column"], r["canonical_field"]), cand_pairs,
                f"lineage pair not from a mapping candidate: {r['source_column']} -> {r['canonical_field']}",
            )

    def test_unmapped_column_never_becomes_a_source(self):
        # loan_amount / initial_advance are unmapped -> must never appear as a
        # lineage source column (the builder did not guess them).
        srcs = {r["source_column"] for r in self.lineage}
        self.assertNotIn("loan_amount", srcs)
        self.assertNotIn("initial_advance", srcs)

    # 14. Value matching evidence is recorded for overlapping mapped fields.
    def test_value_match_evidence_recorded(self):
        bal = [r for r in self.lineage
               if r["canonical_field"] == "current_principal_balance"
               and r["loan_identifier"] == "L0001"]
        self.assertTrue(bal)
        self.assertTrue(bal[0]["validation_sources"])


class TestConflictGating(unittest.TestCase):
    # 15. Conflicting values create gaps unless approved precedence exists.
    def test_conflict_without_precedence(self):
        project, pdir, rp = build_run(SCENARIO_A, ingest=True, drop_precedence=True)
        res = central_tape_builder.build_central_tapes(pdir, rp, str(REGISTRY),
                                                       mode="regulatory_mi")
        gaps = _read_csv(res["central_tape_gaps_path"])
        self.assertTrue([g for g in gaps if g["issue_type"] == "value_conflict"])


if __name__ == "__main__":
    unittest.main()
