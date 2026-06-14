#!/usr/bin/env python3
"""tests/test_onboarding_llm_mapping_controller.py — PART 13 (7, 8, 9, 24)."""

from __future__ import annotations

import json
import sys
import unittest
import warnings
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engine.gate_1_alignment.semantic_alignment import load_field_registry
from engine.onboarding_agent import column_evidence as ce
from engine.onboarding_agent import mapping_candidate_finder as finder
from engine.onboarding_agent.field_scope import resolve_field_scope
from engine.onboarding_agent.llm_mapping_controller import LLMMappingController, build_evidence_pack
from engine.onboarding_agent.mode_policy import load_mode_policy

REGISTRY = str(_REPO_ROOT / "config" / "system" / "fields_registry.yaml")
FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "kfi_pipeline_headers.csv"


def _setup():
    warnings.simplefilter("ignore")
    df = pd.read_csv(FIXTURE)
    reg = load_field_registry(Path(REGISTRY)).get("fields", {})
    fs = resolve_field_scope(REGISTRY, load_mode_policy("regulatory_mi"))
    ev = ce.build_column_evidence(df, "kfi.csv", registry_fields=reg, field_scope=fs)
    sl = finder.build_candidate_shortlist(ev, reg, fs)
    return df, reg, fs, ev, finder.shortlist_by_column(sl)


class _Recorder:
    """Fake LLM callable that records the prompt and returns structured JSON."""

    def __init__(self, proposals):
        self.prompt = None
        self.proposals = proposals

    def __call__(self, prompt):
        self.prompt = prompt
        return json.dumps(self.proposals)


class TestLLMController(unittest.TestCase):
    # 24. LLM remains off by default (no callable -> not enabled, no proposals).
    def test_llm_off_by_default(self):
        _, reg, fs, ev, sl = _setup()
        ctrl = LLMMappingController(llm_callable=None, registry_fields=reg, field_scope=fs)
        res = ctrl.review(ev, sl)
        self.assertFalse(res["usage"]["llm_enabled"])
        self.assertEqual(res["proposals"], [])

    # 7. LLM receives compact evidence packs only — never the full raw file.
    def test_compact_evidence_only(self):
        df, reg, fs, ev, sl = _setup()
        rec = _Recorder([])
        ctrl = LLMMappingController(llm_callable=rec, registry_fields=reg, field_scope=fs)
        ctrl.review(ev, sl)
        prompt = rec.prompt
        self.assertIn("EVIDENCE_PACKS", prompt)
        # Pack keys are the compact allow-list + shortlist; no raw row dump.
        pack = build_evidence_pack(ev[0], sl.get(ev[0]["source_column"], []))
        allowed = set(__import__("engine.onboarding_agent.llm_mapping_controller",
                                 fromlist=["_PACK_KEYS"])._PACK_KEYS) | {"candidate_shortlist"}
        self.assertTrue(set(pack.keys()).issubset(allowed))
        # Privacy control: sensitive raw values (DOB dates, etc.) are redacted in
        # the evidence pack, never sent verbatim to the LLM.
        self.assertNotIn("1955-03-14", prompt)   # a raw DOB value
        self.assertIn("<DATE>", prompt)          # dates are redacted
        # Distinct samples are capped (compact), not a full-file dump.
        pack_with_samples = build_evidence_pack(
            ev[10], sl.get(ev[10]["source_column"], []))
        self.assertLessEqual(
            len(str(pack_with_samples["sample_values_distinct_redacted"]).split("; ")), 8)

    # 8. LLM reviewer returns structured JSON parsed into the proposal schema.
    def test_structured_json_parsed(self):
        df, reg, fs, ev, sl = _setup()
        rec = _Recorder([{
            "source_column": "Gender APP 1", "proposed_business_meaning": "applicant gender",
            "proposed_target_field": "", "proposed_target_source": "registry_target_missing",
            "confidence": "no_match", "reasoning_summary": "no canonical gender field",
        }])
        ctrl = LLMMappingController(llm_callable=rec, registry_fields=reg, field_scope=fs)
        res = ctrl.review(ev, sl)
        self.assertTrue(res["proposals"])
        p = res["proposals"][0]
        for key in ("source_column", "proposed_target_field", "confidence",
                    "requires_user_approval", "registry_action_recommended"):
            self.assertIn(key, p)
        self.assertEqual(res["usage"]["calls_completed"], 1)

    # 9. The LLM cannot map to a non-existent field unless labelled proposed_new_field.
    def test_cannot_invent_target(self):
        df, reg, fs, ev, sl = _setup()
        rec = _Recorder([{
            "source_column": "Status", "proposed_target_field": "made_up_field_xyz",
            "proposed_target_source": "llm_suggested", "confidence": "high",
        }])
        ctrl = LLMMappingController(llm_callable=rec, registry_fields=reg, field_scope=fs)
        p = ctrl.review(ev, sl)["proposals"][0]
        # The invented field is stripped and flagged registry_target_missing.
        self.assertEqual(p["proposed_target_field"], "")
        self.assertEqual(p["proposed_target_source"], "registry_target_missing")

    # 9b. LLM cannot bypass field scope (out-of-scope target is flagged for approval).
    def test_cannot_bypass_field_scope(self):
        df, reg, _, ev, sl = _setup()
        fs_mi = resolve_field_scope(REGISTRY, load_mode_policy("mi_only"))
        rec = _Recorder([{
            "source_column": "Estimated Value",
            "proposed_target_field": "current_valuation_amount",  # regulatory, OOS in mi_only
            "proposed_target_source": "llm_suggested", "confidence": "high",
        }])
        ctrl = LLMMappingController(llm_callable=rec, registry_fields=reg, field_scope=fs_mi)
        p = ctrl.review(ev, sl)["proposals"][0]
        self.assertEqual(p["field_scope_status"], "out_of_scope")
        self.assertTrue(p["requires_user_approval"])


if __name__ == "__main__":
    unittest.main()
