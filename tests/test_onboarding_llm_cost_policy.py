#!/usr/bin/env python3
"""
tests/test_onboarding_llm_cost_policy.py

PART 3-8 — the low-cost LLM mapping review policy: off by default, targeted,
bounded, budgeted, suggestion-only, with user-gap fallback. The LLM provider is
always mocked; no network or API key is required.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.gate_1_alignment.semantic_alignment import load_field_registry
from engine.onboarding_agent.field_scope import resolve_field_scope
from engine.onboarding_agent.llm_mapping_reviewer import (
    build_prompt,
    run_llm_mapping_review,
    validate_suggestion,
)
from engine.onboarding_agent.llm_policy import load_llm_policy, resolve_llm_policy
from engine.onboarding_agent.mode_policy import load_mode_policy
from engine.onboarding_agent.onboarding_models import MappingCandidate
from engine.onboarding_agent.onboarding_orchestrator import run_onboarding

REGISTRY_PATH = _REPO_ROOT / "config" / "system" / "fields_registry.yaml"
ALIASES = _REPO_ROOT / "config" / "system"
PACK = _REPO_ROOT / "synthetic_onboarding_pack"
REGISTRY = load_field_registry(REGISTRY_PATH)
REGISTRY_FIELDS = REGISTRY["fields"]


def _scope(mode, **kw):
    return resolve_field_scope(REGISTRY, load_mode_policy(mode), **kw)


def _extract_sent_columns(prompt: dict) -> set:
    """Pull the source columns out of a captured prompt payload."""
    cols = set()
    for line in prompt["user"].splitlines():
        line = line.strip()
        if line.startswith('{"mode"'):
            payload = json.loads(line)
            for it in payload.get("items", []):
                cols.add(it["source_column"])
    return cols


class RecordingLLM:
    """A mock LLM that records prompts and suggests the first candidate field."""

    def __init__(self):
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        suggestions = []
        for line in prompt["user"].splitlines():
            line = line.strip()
            if line.startswith('{"mode"'):
                payload = json.loads(line)
                for it in payload.get("items", []):
                    fields = it["candidate_canonical_fields"]
                    if fields:
                        suggestions.append({
                            "source_file": it["source_file"],
                            "source_column": it["source_column"],
                            "recommended_canonical_field": fields[0]["field"],
                            "confidence": 0.8,
                            "rationale": "mock",
                            "alternatives": [f["field"] for f in fields[1:]],
                            "requires_review": True,
                        })
        return json.dumps({"llm_mapping_suggestions": suggestions})


# An analytics field guaranteed in scope everywhere.
ANALYTICS_FIELD = next(f for f, m in REGISTRY_FIELDS.items()
                       if m.get("category") == "analytics" and not m.get("core_canonical"))
# A regulatory non-core field (excluded in mi_only).
REG_NONCORE = "employment_status"


def _cand(col, field, conf, *, ambiguous=False, review=True, alts=None):
    return MappingCandidate(
        source_file="f.csv", source_file_classification="loan_report",
        source_column=col, candidate_canonical_field=field, confidence=conf,
        method="token_set", sample_values_redacted=["<v1>", "<v2>"],
        requires_review=review,
        ambiguity_rule_applied="regulatory_preference_ambiguity_rule" if ambiguous else "",
        alternative_candidates=(alts or []),
    )


class TestDefaultOff(unittest.TestCase):
    def test_7_llm_off_by_default(self):
        pol = load_llm_policy()
        self.assertFalse(pol.enabled)
        resolved = resolve_llm_policy()
        self.assertFalse(resolved.enabled)

    def test_7b_review_returns_empty_when_off(self):
        pol = load_llm_policy()  # enabled False
        suggestions, usage, gaps = run_llm_mapping_review(
            mapping_candidates=[_cand("a", ANALYTICS_FIELD, 0.5)],
            mapping_ambiguities=[], field_scope=_scope("regulatory_mi"),
            registry_fields=REGISTRY_FIELDS, mode="regulatory_mi", policy=pol,
        )
        self.assertEqual(suggestions, [])
        self.assertEqual(gaps, [])
        self.assertFalse(usage["llm_enabled"])
        self.assertEqual(usage["calls_completed"], 0)

    def test_8_deterministic_run_writes_off_summary(self):
        tmp = Path(tempfile.mkdtemp(prefix="llm_off_"))
        run_onboarding(input_dir=str(PACK), client_name="OFF", output_dir=str(tmp),
                       registry_path=str(REGISTRY_PATH), aliases_dir=str(ALIASES),
                       mode="regulatory_mi")
        summary = json.loads((tmp / "22_llm_usage_summary.json").read_text())
        self.assertFalse(summary["llm_enabled"])
        self.assertEqual(summary["calls_completed"], 0)
        self.assertEqual(summary["estimated_cost"], 0)


class TestTargeting(unittest.TestCase):
    def test_9_only_unresolved_in_scope_sent(self):
        mock = RecordingLLM()
        cands = [
            # Resolved & confident -> skipped (zero-cost-first).
            _cand("resolved_col", ANALYTICS_FIELD, 0.95, review=True),
            # Ambiguous -> sent.
            _cand("amb_col", ANALYTICS_FIELD, 0.80, ambiguous=True,
                  alts=[{"field": "loan_identifier", "category": "analytics",
                         "core_canonical": True, "confidence": 0.78}]),
        ]
        pol = resolve_llm_policy(enable_llm_review=True)
        suggestions, usage, gaps = run_llm_mapping_review(
            mapping_candidates=cands, mapping_ambiguities=[],
            field_scope=_scope("regulatory_mi"), registry_fields=REGISTRY_FIELDS,
            mode="regulatory_mi", policy=pol, llm_callable=mock,
        )
        sent_cols = set()
        for p in mock.prompts:
            sent_cols |= _extract_sent_columns(p)
        self.assertIn("amb_col", sent_cols)
        self.assertNotIn("resolved_col", sent_cols)

    def test_10_zero_cost_first_skips_high_confidence(self):
        mock = RecordingLLM()
        cands = [_cand("resolved_col", ANALYTICS_FIELD, 0.95, review=True)]
        pol = resolve_llm_policy(enable_llm_review=True)
        _, usage, _ = run_llm_mapping_review(
            mapping_candidates=cands, mapping_ambiguities=[],
            field_scope=_scope("regulatory_mi"), registry_fields=REGISTRY_FIELDS,
            mode="regulatory_mi", policy=pol, llm_callable=mock,
        )
        self.assertEqual(usage["skipped_due_to_zero_cost_first"], 1)
        self.assertEqual(usage["calls_completed"], 0)

    def test_11_mi_only_skips_out_of_scope_regulatory(self):
        mock = RecordingLLM()
        # Only candidate is a regulatory non-core field (out of scope in mi_only).
        cands = [_cand("emp_col", REG_NONCORE, 0.80, ambiguous=True)]
        pol = resolve_llm_policy(enable_llm_review=True)
        suggestions, usage, gaps = run_llm_mapping_review(
            mapping_candidates=cands, mapping_ambiguities=[],
            field_scope=_scope("mi_only"), registry_fields=REGISTRY_FIELDS,
            mode="mi_only", policy=pol, llm_callable=mock,
        )
        sent_cols = set()
        for p in mock.prompts:
            sent_cols |= _extract_sent_columns(p)
        self.assertNotIn("emp_col", sent_cols)
        self.assertEqual(usage["items_sent"], 0)

    def test_12_respects_max_calls_and_items(self):
        mock = RecordingLLM()
        cands = [_cand(f"c{i}", ANALYTICS_FIELD, 0.5, review=True) for i in range(20)]
        pol = resolve_llm_policy(enable_llm_review=True, max_calls=2,
                                 max_items_per_call=3)
        _, usage, gaps = run_llm_mapping_review(
            mapping_candidates=cands, mapping_ambiguities=[],
            field_scope=_scope("regulatory_mi"), registry_fields=REGISTRY_FIELDS,
            mode="regulatory_mi", policy=pol, llm_callable=mock,
        )
        # capacity = 2 calls * 3 items = 6 sent; the rest become gap questions.
        self.assertLessEqual(usage["calls_completed"], 2)
        self.assertLessEqual(usage["items_sent"], 6)
        for p in mock.prompts:
            self.assertLessEqual(len(_extract_sent_columns(p)), 3)
        self.assertTrue(gaps)

    def test_13_excess_unresolved_become_gap_questions(self):
        mock = RecordingLLM()
        cands = [_cand(f"c{i}", ANALYTICS_FIELD, 0.5, review=True) for i in range(40)]
        pol = resolve_llm_policy(enable_llm_review=True, max_calls=2,
                                 max_items_per_call=5)
        _, usage, gaps = run_llm_mapping_review(
            mapping_candidates=cands, mapping_ambiguities=[],
            field_scope=_scope("regulatory_mi"), registry_fields=REGISTRY_FIELDS,
            mode="regulatory_mi", policy=pol, llm_callable=mock,
        )
        # 40 items, capacity 10 -> ~30 converted to gap questions.
        self.assertGreaterEqual(len(gaps), 25)
        self.assertTrue(all(g.category == "mapping" for g in gaps))
        self.assertTrue(usage["over_uncertainty_budget"])


class TestValidation(unittest.TestCase):
    def test_14_out_of_scope_recommendation_rejected(self):
        scope = _scope("mi_only")
        item = {
            "source_column": "emp", "source_file": "f.csv",
            "candidate_canonical_fields": [
                {"field": REG_NONCORE, "category": "regulatory", "core_canonical": False},
            ],
        }
        sugg = {"recommended_canonical_field": REG_NONCORE, "confidence": 0.9}
        ok, reason = validate_suggestion(sugg, item, scope, "mi_only", REGISTRY_FIELDS)
        self.assertFalse(ok)

    def test_14b_non_candidate_field_rejected(self):
        scope = _scope("regulatory_mi")
        item = {
            "source_column": "x", "source_file": "f.csv",
            "candidate_canonical_fields": [
                {"field": ANALYTICS_FIELD, "category": "analytics", "core_canonical": False},
            ],
        }
        sugg = {"recommended_canonical_field": "some_field_not_offered", "confidence": 0.9}
        ok, _ = validate_suggestion(sugg, item, scope, "regulatory_mi", REGISTRY_FIELDS)
        self.assertFalse(ok)


class TestUsageAndPrivacy(unittest.TestCase):
    def test_15_usage_summary_written_when_enabled(self):
        tmp = Path(tempfile.mkdtemp(prefix="llm_on_"))
        run_onboarding(
            input_dir=str(PACK), client_name="ON", output_dir=str(tmp),
            registry_path=str(REGISTRY_PATH), aliases_dir=str(ALIASES),
            mode="regulatory_mi", enable_llm_review=True,
            llm_callable=RecordingLLM(),
        )
        summary = json.loads((tmp / "22_llm_usage_summary.json").read_text())
        self.assertTrue(summary["llm_enabled"])
        self.assertIn("calls_completed", summary)
        self.assertIn("estimated_cost", summary)

    def test_16_prompt_excludes_full_registry_and_files(self):
        item = {
            "source_file": "monthly_loan_report.csv", "source_file_type": "csv",
            "source_column": "original principal",
            "normalized_column_name": "original_principal",
            "inferred_type": "decimal",
            "redacted_sample_values": ["<v1>", "<v2>"],
            "candidate_canonical_fields": [
                {"field": "original_principal_balance", "category": "regulatory",
                 "core_canonical": True, "confidence": 0.86, "description": "x"},
                {"field": ANALYTICS_FIELD, "category": "analytics",
                 "core_canonical": False, "confidence": 0.84, "description": "y"},
            ],
            "mode": "regulatory_mi", "field_scope_status": "in_scope",
        }
        prompt = build_prompt([item], "regulatory_mi")
        text = prompt["system"] + prompt["user"]
        # The full registry (471 fields) must NOT be present.
        present = sum(1 for f in REGISTRY_FIELDS if f in text)
        self.assertLessEqual(present, 5)
        # No raw/full file content markers.
        self.assertNotIn("monthly_loan_report.csv\n1,", text)
        # Compact prompt stays well under the per-run char budget.
        self.assertLess(len(text), 5000)


if __name__ == "__main__":
    unittest.main()
